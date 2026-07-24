"""Cryptographically verify and publish one terminal provider transition.

Preparation receipts and downloaded files are evidence, never authority.  This
module binds the exact prepared state bytes to a GitHub-hosted C0 attestation,
revalidates the protected claimed-state predecessor, and performs the sole
compare-and-swap publication against that predecessor.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from .execution_claim import ProviderPhase
from .github_state_attestation import (
    C0_REF,
    OIDC_ISSUER,
    REPOSITORY,
    GitHubWriteApi,
    LedgerPublicationReceipt,
    parse_sigstore_bundle,
    publish_candidate_ledger_transition,
)
from .suite_attempt import SuiteStateRecord, VerifiedProviderPredecessor

if TYPE_CHECKING:
    from .provider_workflow_orchestration import ProviderWorkflowContext

ProviderTransitionMode = Literal["completion", "failure"]

ONLINE_COMPLETION_PREDICATE_TYPE = (
    "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/online-output-aggregate/v1"
)
LABEL_RELEASE_COMPLETION_PREDICATE_TYPE = (
    "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/label-release/v1"
)
ANALYSIS_COMPLETION_PREDICATE_TYPE = (
    "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/confirmatory-analysis/v1"
)
PROVIDER_FAILURE_PREDICATE_TYPE = (
    "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/provider-failure/v1"
)

_COMPLETION_PREDICATE_TYPES: Mapping[ProviderPhase, str] = {
    "online": ONLINE_COMPLETION_PREDICATE_TYPE,
    "label-release": LABEL_RELEASE_COMPLETION_PREDICATE_TYPE,
    "analysis": ANALYSIS_COMPLETION_PREDICATE_TYPE,
}
_CLAIMED: Mapping[ProviderPhase, tuple[str, int]] = {
    "online": ("RUN_CLAIMED", 1),
    "label-release": ("LABEL_RELEASE_CLAIMED", 3),
    "analysis": ("ANALYSIS_CLAIMED", 5),
}
_COMPLETED: Mapping[ProviderPhase, tuple[str, int]] = {
    "online": ("ONLINE_COMPLETE", 2),
    "label-release": ("LABELS_RELEASED", 4),
    "analysis": ("ANALYSIS_COMPLETE", 6),
}
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
_MAX_GH_OUTPUT_BYTES = 8 * 1024 * 1024
_CAPABILITY = object()


class ProviderTransitionPublicationError(ValueError):
    """Terminal provider evidence or publication authority differs from C0."""


class ProviderTransitionAttestationVerifier(Protocol):
    def verify(
        self,
        *,
        subject_path: Path,
        bundle_path: Path,
        context: ProviderWorkflowContext,
        predicate_type: str,
    ) -> bytes: ...


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ProviderTransitionPublicationError(
            "provider transition evidence is not canonical JSON"
        ) from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _closed_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ProviderTransitionPublicationError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ProviderTransitionPublicationError(f"{label} contains non-finite number {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderTransitionPublicationError(f"cannot decode {label}") from exc
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProviderTransitionPublicationError(f"{label} must contain one JSON object")
    return value


def _secure_file_bytes(path: Path, *, label: str) -> bytes:
    if not path.is_absolute() or path.anchor != "/":
        raise ProviderTransitionPublicationError(f"{label} path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProviderTransitionPublicationError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_EVIDENCE_BYTES
        ):
            raise ProviderTransitionPublicationError(
                f"{label} must be one bounded singly linked regular file"
            )
        chunks: list[bytes] = []
        observed = 0
        while observed <= _MAX_EVIDENCE_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_EVIDENCE_BYTES + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ProviderTransitionPublicationError(f"cannot read {label}") from exc
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if observed > _MAX_EVIDENCE_BYTES or identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ProviderTransitionPublicationError(f"{label} changed while read")
    encoded = b"".join(chunks)
    if len(encoded) != before.st_size:
        raise ProviderTransitionPublicationError(f"{label} changed while read")
    return encoded


def _write_exclusive(path: Path, encoded: bytes, *, label: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise ProviderTransitionPublicationError(f"cannot create {label} once") from exc


def _controlled_output_dir(value: str | Path) -> Path:
    root = Path(value)
    if (
        not root.is_absolute()
        or root.anchor != "/"
        or any(part in {"", ".", ".."} for part in root.parts[1:])
    ):
        raise ProviderTransitionPublicationError(
            "provider transition output directory is not canonical"
        )
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ProviderTransitionPublicationError(
            "provider transition output directory is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or resolved != root
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise ProviderTransitionPublicationError(
            "provider transition output directory is not controlled"
        )
    return root


def _validated_gh_output(encoded: bytes) -> bytes:
    if not encoded or len(encoded) > _MAX_GH_OUTPUT_BYTES:
        raise ProviderTransitionPublicationError(
            "gh attestation verification output is empty or oversized"
        )
    try:
        value = json.loads(encoded.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderTransitionPublicationError(
            "cannot decode gh attestation verification output"
        ) from exc
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], Mapping)
        or not isinstance(value[0].get("verificationResult"), Mapping)
    ):
        raise ProviderTransitionPublicationError("gh must return exactly one verified attestation")
    return encoded


def provider_transition_predicate_type(
    phase: ProviderPhase,
    mode: ProviderTransitionMode,
) -> str:
    if phase not in _CLAIMED:
        raise ProviderTransitionPublicationError("provider transition phase is not registered")
    if mode == "failure":
        return PROVIDER_FAILURE_PREDICATE_TYPE
    if mode != "completion":
        raise ProviderTransitionPublicationError("provider transition mode is not registered")
    return _COMPLETION_PREDICATE_TYPES[phase]


@dataclass(frozen=True)
class GhProviderTransitionAttestationVerifier:
    """Invoke GitHub's Sigstore verifier under the exact C0 hosted identity."""

    executable: str = "gh"
    timeout_seconds: int = 60

    def verify(
        self,
        *,
        subject_path: Path,
        bundle_path: Path,
        context: ProviderWorkflowContext,
        predicate_type: str,
    ) -> bytes:
        from .provider_workflow_orchestration import ProviderWorkflowContext

        if not isinstance(context, ProviderWorkflowContext):
            raise ProviderTransitionPublicationError(
                "transition attestation requires an admitted workflow context"
            )
        if context.job not in {"complete", "fail"}:
            raise ProviderTransitionPublicationError(
                "transition attestation requires a hosted terminal-state job"
            )
        expected_type = provider_transition_predicate_type(
            context.phase,
            "completion" if context.job == "complete" else "failure",
        )
        if predicate_type != expected_type:
            raise ProviderTransitionPublicationError(
                "transition attestation predicate type differs from C0"
            )
        identity = f"https://github.com/{context.workflow_ref}"
        command = [
            self.executable,
            "attestation",
            "verify",
            str(subject_path),
            "--bundle",
            str(bundle_path),
            "--hostname",
            "github.com",
            "--repo",
            REPOSITORY,
            "--cert-identity",
            identity,
            "--cert-oidc-issuer",
            OIDC_ISSUER,
            "--signer-digest",
            context.workflow_sha,
            "--source-digest",
            context.workflow_sha,
            "--source-ref",
            C0_REF,
            "--deny-self-hosted-runners",
            "--predicate-type",
            predicate_type,
            "--format",
            "json",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                env={**os.environ, "GH_PROMPT_DISABLED": "1", "NO_COLOR": "1"},
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProviderTransitionPublicationError(
                "cannot execute provider transition attestation verification"
            ) from exc
        if result.returncode != 0:
            detail = result.stderr[:4096].decode("utf-8", errors="replace").strip()
            raise ProviderTransitionPublicationError(
                f"GitHub rejected the provider transition attestation: {detail}"
            )
        return _validated_gh_output(result.stdout)


@dataclass(frozen=True)
class VerifiedProviderTransitionAttestation:
    phase: ProviderPhase
    mode: ProviderTransitionMode
    suite_attempt_id: str
    target_state: str
    target_sequence: int
    subject_sha256: str
    predicate_sha256: str
    bundle_sha256: str
    predicate_type: str
    signer_identity: str
    signer_digest: str
    rekor_entry_id: str
    rekor_log_index: int
    rekor_integrated_at_utc: str
    gh_verification_sha256: str
    _subject_bytes: bytes
    _predicate_bytes: bytes
    _bundle_bytes: bytes
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _CAPABILITY:
            raise ProviderTransitionPublicationError(
                "verified transition attestation can only come from cryptographic verification"
            )


@dataclass(frozen=True)
class ProviderTransitionPublicationResult:
    phase: ProviderPhase
    mode: ProviderTransitionMode
    state: SuiteStateRecord
    attestation: VerifiedProviderTransitionAttestation
    publication_receipt: LedgerPublicationReceipt
    publication_receipt_path: Path
    published: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.state, SuiteStateRecord)
            or not isinstance(self.attestation, VerifiedProviderTransitionAttestation)
            or not isinstance(self.publication_receipt, LedgerPublicationReceipt)
            or self.attestation.subject_sha256 != self.state.record_sha256
            or self.publication_receipt.state_record_sha256 != self.state.record_sha256
            or self.publication_receipt.state_sequence != self.state.sequence
            or self.publication_receipt.suite_attempt_id != self.state.suite_attempt_id
        ):
            raise ProviderTransitionPublicationError(
                "provider transition publication result is cross-bound poorly"
            )
        path = Path(self.publication_receipt_path)
        if not path.is_absolute():
            raise ProviderTransitionPublicationError(
                "provider transition publication receipt path must be absolute"
            )
        object.__setattr__(self, "publication_receipt_path", path)


def _assert_transition_identity(
    *,
    context: ProviderWorkflowContext,
    phase: ProviderPhase,
    mode: ProviderTransitionMode,
    suite_attempt_id: str,
    predecessor: VerifiedProviderPredecessor,
    target: SuiteStateRecord,
) -> None:
    from .provider_workflow_orchestration import ProviderWorkflowContext

    if not isinstance(context, ProviderWorkflowContext):
        raise ProviderTransitionPublicationError(
            "provider transition requires an admitted workflow context"
        )
    expected_job = "complete" if mode == "completion" else "fail"
    if context.phase != phase or context.job != expected_job:
        raise ProviderTransitionPublicationError(
            "provider transition context differs from its phase or mode"
        )
    if not isinstance(predecessor, VerifiedProviderPredecessor):
        raise ProviderTransitionPublicationError(
            "provider transition requires freshly verified claimed-state authority"
        )
    if not isinstance(target, SuiteStateRecord):
        raise ProviderTransitionPublicationError(
            "provider transition requires one typed candidate state"
        )
    expected_claim = _CLAIMED.get(phase)
    expected_target = (
        _COMPLETED.get(phase)
        if mode == "completion"
        else ("FAILED", expected_claim[1] + 1)
        if expected_claim is not None
        else None
    )
    predecessor_state = predecessor.state
    if (
        expected_claim is None
        or expected_target is None
        or (predecessor_state.state, predecessor_state.sequence) != expected_claim
        or (target.state, target.sequence) != expected_target
        or target.suite_attempt_id != suite_attempt_id
        or predecessor_state.suite_attempt_id != suite_attempt_id
        or target.previous_state_record_sha256 != predecessor_state.record_sha256
        or target.manifest_sha256 != predecessor_state.manifest_sha256
        or target.run_receipt_sha256 != predecessor_state.run_receipt_sha256
        or target.namespace_uri != predecessor_state.namespace_uri
    ):
        raise ProviderTransitionPublicationError(
            "provider transition differs from the claimed-state machine"
        )


def _verify_attestation_snapshot(
    *,
    context: ProviderWorkflowContext,
    phase: ProviderPhase,
    mode: ProviderTransitionMode,
    suite_attempt_id: str,
    target: SuiteStateRecord,
    subject_bytes: bytes,
    predicate_bytes: bytes,
    bundle_bytes: bytes,
    verifier: ProviderTransitionAttestationVerifier,
) -> VerifiedProviderTransitionAttestation:
    expected_subject = target.canonical_bytes() + b"\n"
    if subject_bytes != expected_subject or _sha256(subject_bytes) != target.record_sha256:
        raise ProviderTransitionPublicationError(
            "prepared transition subject differs from the candidate state"
        )
    predicate = _closed_object(predicate_bytes, label="provider transition predicate")
    if predicate_bytes != _canonical_bytes(predicate) + b"\n":
        raise ProviderTransitionPublicationError(
            "provider transition predicate is not canonical JSON plus one LF"
        )
    predicate_type = provider_transition_predicate_type(phase, mode)
    with tempfile.TemporaryDirectory(prefix="fractal-provider-transition-attestation-") as name:
        root = Path(name)
        subject_snapshot = root / "prepared-subject.json"
        bundle_snapshot = root / "transition.sigstore.bundle.json"
        _write_exclusive(
            subject_snapshot,
            subject_bytes,
            label="provider transition subject snapshot",
        )
        _write_exclusive(
            bundle_snapshot,
            bundle_bytes,
            label="provider transition bundle snapshot",
        )
        verified = verifier.verify(
            subject_path=subject_snapshot,
            bundle_path=bundle_snapshot,
            context=context,
            predicate_type=predicate_type,
        )
        _validated_gh_output(verified)
        if (
            _secure_file_bytes(
                subject_snapshot,
                label="provider transition subject snapshot",
            )
            != subject_bytes
            or _secure_file_bytes(
                bundle_snapshot,
                label="provider transition bundle snapshot",
            )
            != bundle_bytes
        ):
            raise ProviderTransitionPublicationError(
                "provider transition verification snapshot changed"
            )
    try:
        observation = parse_sigstore_bundle(bundle_bytes)
    except Exception as exc:
        raise ProviderTransitionPublicationError(
            "provider transition Sigstore bundle is malformed"
        ) from exc
    statement = observation.statement
    expected_keys = {"_type", "predicate", "predicateType", "subject"}
    subjects = statement.get("subject")
    if (
        set(statement) != expected_keys
        or statement.get("_type") != "https://in-toto.io/Statement/v1"
        or statement.get("predicateType") != predicate_type
        or statement.get("predicate") != predicate
        or not isinstance(subjects, list)
        or len(subjects) != 1
        or not isinstance(subjects[0], Mapping)
        or set(subjects[0]) != {"digest", "name"}
        or subjects[0].get("name") != "prepared-subject.json"
        or subjects[0].get("digest") != {"sha256": target.record_sha256}
    ):
        raise ProviderTransitionPublicationError(
            "verified provider transition statement differs from prepared evidence"
        )
    return VerifiedProviderTransitionAttestation(
        phase=phase,
        mode=mode,
        suite_attempt_id=suite_attempt_id,
        target_state=target.state,
        target_sequence=target.sequence,
        subject_sha256=target.record_sha256,
        predicate_sha256=_sha256(predicate_bytes),
        bundle_sha256=_sha256(bundle_bytes),
        predicate_type=predicate_type,
        signer_identity=f"https://github.com/{context.workflow_ref}",
        signer_digest=context.workflow_sha,
        rekor_entry_id=observation.entry_id,
        rekor_log_index=observation.log_index,
        rekor_integrated_at_utc=observation.integrated_at_utc,
        gh_verification_sha256=_sha256(verified),
        _subject_bytes=subject_bytes,
        _predicate_bytes=predicate_bytes,
        _bundle_bytes=bundle_bytes,
        _capability=_CAPABILITY,
    )


def verify_and_publish_provider_transition(
    *,
    context: ProviderWorkflowContext,
    phase: ProviderPhase,
    mode: ProviderTransitionMode,
    suite_attempt_id: str,
    predecessor: VerifiedProviderPredecessor,
    target: SuiteStateRecord,
    subject_path: str | Path,
    predicate_path: str | Path,
    bundle_path: str | Path,
    output_dir: str | Path,
    github_api: GitHubWriteApi,
    verifier: ProviderTransitionAttestationVerifier | None = None,
) -> ProviderTransitionPublicationResult:
    """Verify one hosted attestation and CAS-publish its exact terminal state."""

    _assert_transition_identity(
        context=context,
        phase=phase,
        mode=mode,
        suite_attempt_id=suite_attempt_id,
        predecessor=predecessor,
        target=target,
    )
    subject = Path(subject_path)
    predicate = Path(predicate_path)
    bundle = Path(bundle_path)
    if subject.name != "prepared-subject.json":
        raise ProviderTransitionPublicationError(
            "provider transition subject must use the fixed prepared filename"
        )
    expected_predicate_name = (
        "completion-predicate.json" if mode == "completion" else "failure-predicate.json"
    )
    if predicate.name != expected_predicate_name:
        raise ProviderTransitionPublicationError(
            "provider transition predicate filename differs from its mode"
        )
    root = _controlled_output_dir(output_dir)
    predecessor.assert_current()
    subject_bytes = _secure_file_bytes(subject, label="prepared provider transition subject")
    predicate_bytes = _secure_file_bytes(predicate, label="prepared provider transition predicate")
    bundle_bytes = _secure_file_bytes(bundle, label="provider transition Sigstore bundle")
    active_verifier = GhProviderTransitionAttestationVerifier() if verifier is None else verifier
    observation = _verify_attestation_snapshot(
        context=context,
        phase=phase,
        mode=mode,
        suite_attempt_id=suite_attempt_id,
        target=target,
        subject_bytes=subject_bytes,
        predicate_bytes=predicate_bytes,
        bundle_bytes=bundle_bytes,
        verifier=active_verifier,
    )
    if (
        _secure_file_bytes(subject, label="prepared provider transition subject") != subject_bytes
        or _secure_file_bytes(predicate, label="prepared provider transition predicate")
        != predicate_bytes
        or _secure_file_bytes(bundle, label="provider transition Sigstore bundle") != bundle_bytes
    ):
        raise ProviderTransitionPublicationError(
            "provider transition evidence changed before publication"
        )
    predecessor.assert_current()
    receipt_path = root / "ledger-publication-receipt.json"
    try:
        publication, published = publish_candidate_ledger_transition(
            target=target,
            expected_predecessor_commit=predecessor.ledger_commit,
            receipt_path=receipt_path,
            api=github_api,
        )
    except Exception as exc:
        raise ProviderTransitionPublicationError(
            "provider transition compare-and-swap publication failed"
        ) from exc
    if (
        publication.previous_commit_oid != predecessor.ledger_commit
        or publication.state_record_sha256 != target.record_sha256
        or publication.state_sequence != target.sequence
        or publication.suite_attempt_id != target.suite_attempt_id
        or _secure_file_bytes(
            receipt_path,
            label="provider transition publication receipt",
        )
        != publication.canonical_bytes()
    ):
        raise ProviderTransitionPublicationError(
            "provider transition publication failed exact readback"
        )
    return ProviderTransitionPublicationResult(
        phase=phase,
        mode=mode,
        state=target,
        attestation=observation,
        publication_receipt=publication,
        publication_receipt_path=receipt_path,
        published=published,
    )
