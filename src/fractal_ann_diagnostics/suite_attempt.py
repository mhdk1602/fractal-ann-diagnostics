"""Externally attested one-shot state for the fixed confirmatory suite.

Local exclusive files prevent accidental reruns by one Unix identity.  They do
not prove that another machine, administrator, or repository fork did not run a
second attempt.  This module therefore advances the suite only from canonical
state and evidence files whose signature, GitHub Actions identity, public-log
entry, signed timestamp, and exclusive provider transition have been checked by
an injected verifier.

HTTPS locations in the descriptor are retention and availability endpoints.
Fetching bytes from them is never treated as signature or timestamp evidence.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import unquote, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    DirectoryDigest,
    digest_directory_tree,
    digest_regular_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .audit import GENESIS_RECORD_SHA256
from .confirmatory_analysis import (
    load_action_panel_admission_receipt,
    load_action_panel_artifact,
)
from .execution_claim import (
    OUTPUT_AGGREGATE_DERIVATION,
    AnonymousZenodoAdmission,
    ClaimCorpusBinding,
    CorpusOutputTree,
    ExecutionBeaconVerifier,
    ExecutionClaimContract,
    ExecutionClaimError,
    FailedExecuteJobReceipt,
    LiveExecuteJobReceipt,
    PhaseClaimContract,
    ProviderExecutionIdentity,
    ProviderPhaseFailure,
    RunOutputAggregate,
    VerifiedPhaseClaimCapability,
    VerifiedRunClaimCapability,
    _mint_verified_phase_claim,
    _mint_verified_run_claim,
    derive_phase_runner_label,
    load_provider_phase_plans,
    verify_execution_beacon,
    verify_label_release_beacon,
)
from .label_separation import load_prediction_artifact
from .online_runner import (
    load_cache_preparation_receipt,
    load_execution_order_receipt,
)
from .production_corpus_run import (
    PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
    PRODUCTION_CORPUS_WORKLOAD_ID,
    RUNTIME_ATTESTATION_RECEIPT_FILENAME,
    RUNTIME_INVOCATION_MARKER_FILENAME,
    ProductionCorpusRunError,
    load_production_corpus_command_attempt,
    load_production_corpus_workload_spec,
)
from .runtime_attestation import (
    RuntimeAttestationError,
    load_runtime_attestation_plan,
    load_runtime_attestation_receipt,
    load_runtime_preflight_receipt,
    verify_runtime_attestation_receipt,
)
from .sealed_container_launcher import (
    SealedContainerLauncherError,
    VerifiedProductionRunClosure,
    load_preflight_launch_contract,
    load_registered_plan_instantiation,
    load_runtime_plan_transition,
    load_sealed_launch_contract,
    load_sealed_launch_receipt,
    verify_sealed_launch_evidence,
    verify_sealed_transition,
)
from .sealed_online_execution import (
    load_sealed_online_attempt_receipt,
    load_sealed_online_result_receipt,
    sealed_online_attempt_path,
    sealed_online_result_path,
    verify_sealed_online_outputs,
)
from .study import (
    FIXED_CORPORA,
    VerifiedC1ProtocolRegistration,
    load_sealed_run_receipt,
    load_study_manifest,
    manifest_sha256,
    validate_study_manifest,
)

SUITE_STATE_SCHEMA = "fractal-suite-state-v9"
SUITE_ATTESTATION_DESCRIPTOR_SCHEMA = "fractal-suite-attestation-descriptor-v1"
SUITE_ATTESTATION_EVIDENCE_SCHEMA = "fractal-suite-attestation-evidence-v1"
SUITE_PROVIDER_CLAIMS_SCHEMA = "fractal-suite-provider-claims-v1"
SUITE_OUTPUT_TRANSFER_SCHEMA = "fractal-suite-output-transfer-v2"

SuiteState = Literal[
    "OPENED",
    "RUN_CLAIMED",
    "ONLINE_COMPLETE",
    "LABEL_RELEASE_CLAIMED",
    "LABELS_RELEASED",
    "ANALYSIS_CLAIMED",
    "ANALYSIS_COMPLETE",
    "FAILED",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_WORKFLOW = re.compile(r"^\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml$")
_STATE_CAPABILITY = object()
_PROVIDER_PREDECESSOR_CAPABILITY = object()
_MAX_ATTESTATION_BYTES = 16 * 1024 * 1024
_TRANSFER_COPY_CHUNK_BYTES = 1024 * 1024
_TRANSFER_ROLE_FILE_FIELD = {
    "action-panel": "action_panel_file_sha256",
    "action-panel-admission": "action_panel_admission_file_sha256",
    "audit-chain": "audit_file_sha256",
    "cache-preparation": "cache_preparation_file_sha256",
    "execution-order": "execution_order_file_sha256",
    "predictions": "prediction_file_sha256",
    "production-command-attempt": "production_command_attempt_file_sha256",
    "runtime-attestation-receipt": "runtime_attestation_receipt_file_sha256",
    "runtime-invocation-marker": "runtime_invocation_marker_file_sha256",
    "sealed-online-attempt": "attempt_file_sha256",
    "sealed-online-result": "result_file_sha256",
}
_TRANSFER_CONTROL_FILENAMES = {
    "production-command-attempt": PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
    "runtime-attestation-receipt": RUNTIME_ATTESTATION_RECEIPT_FILENAME,
    "runtime-invocation-marker": RUNTIME_INVOCATION_MARKER_FILENAME,
}


class SuiteAttemptError(ValueError):
    """Raised when the fixed-suite state cannot advance with exact evidence."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SuiteAttemptError("suite evidence must be finite canonical JSON") from exc


def _canonical_utf8_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SuiteAttemptError("audit evidence must be finite canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SuiteAttemptError(f"{name} must be a canonical non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise SuiteAttemptError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SuiteAttemptError(f"{name} cannot contain control characters")
    return value


def _digest(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SuiteAttemptError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise SuiteAttemptError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise SuiteAttemptError(f"{name} must be a positive integer")
    return value


def _timestamp(name: str, value: object) -> datetime:
    text = _text(name, value)
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SuiteAttemptError(f"{name} must use ISO 8601") from exc
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise SuiteAttemptError(f"{name} must use UTC")
    if instant.isoformat() != text:
        raise SuiteAttemptError(f"{name} must use canonical ISO 8601 form")
    return instant


def _https_uri(name: str, value: object) -> str:
    text = _text(name, value)
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path in {"", "/"}
    ):
        raise SuiteAttemptError(f"{name} must be a specific credential-free HTTPS URI")
    return text


def _https_issuer(name: str, value: object) -> str:
    text = _text(name, value)
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise SuiteAttemptError(f"{name} must be an exact credential-free HTTPS issuer")
    return text


def _local_file_uri(name: str, value: object) -> tuple[str, Path]:
    text = _text(name, value)
    parsed = urlsplit(text)
    path = Path(unquote(parsed.path))
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not path.is_absolute()
        or path.anchor != "/"
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or path.as_uri() != text
    ):
        raise SuiteAttemptError(f"{name} must be a canonical absolute local file URI")
    return text, path


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise SuiteAttemptError(f"{label} must be a JSON object with string keys")
    observed = set(value)
    if observed != fields:
        raise SuiteAttemptError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _parse_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SuiteAttemptError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise SuiteAttemptError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuiteAttemptError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SuiteAttemptError(f"{label} must contain one JSON object")
    return value


def _secure_bytes(path: str | Path, *, label: str, max_bytes: int) -> bytes:
    try:
        return read_secure_regular_file(path, max_bytes=max_bytes, label=label)
    except ArtifactIntegrityError as exc:
        raise SuiteAttemptError(f"cannot read {label}: {exc}") from exc


def _regular_file_size(path: Path, *, label: str) -> int:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise SuiteAttemptError(f"cannot stat {label}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise SuiteAttemptError(f"{label} is no longer a regular file")
    return metadata.st_size


def _assert_distinct_regular_files(
    rows: Sequence[tuple[str, Path]],
    *,
    label: str,
) -> None:
    identities: set[tuple[int, int]] = set()
    pathnames: set[Path] = set()
    for role, path in rows:
        if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
            raise SuiteAttemptError(f"{role} path must be canonical absolute")
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise SuiteAttemptError(f"cannot inspect {role}") from exc
        identity = (metadata.st_dev, metadata.st_ino)
        if not stat.S_ISREG(metadata.st_mode) or path in pathnames or identity in identities:
            raise SuiteAttemptError(f"{label} must use pairwise-distinct real regular files")
        pathnames.add(path)
        identities.add(identity)


def _write_once(encoded: bytes, path: Path, *, label: str) -> None:
    try:
        write_exclusive_receipt_bytes(encoded, path)
    except ArtifactIntegrityError as exc:
        raise SuiteAttemptError(f"cannot write {label}: {exc}") from exc


@dataclass(frozen=True)
class SuiteAttestationDescriptor:
    """Closed trust policy for the provider evidence verifier."""

    expected_signer_identity: str
    expected_oidc_issuer: str
    expected_repository: str
    expected_workflow: str
    expected_git_ref: str
    expected_signer_digest: str
    transparency_log_identity: str
    transparency_log_uri: str
    transparency_log_public_key_sha256: str
    timestamp_authority_identity: str
    timestamp_authority_uri: str
    timestamp_authority_public_key_sha256: str
    state_service_identity: str
    state_service_uri: str
    state_key_prefix: str
    schema_version: str = SUITE_ATTESTATION_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "expected_signer_identity",
            "transparency_log_identity",
            "timestamp_authority_identity",
            "state_service_identity",
            "state_key_prefix",
        ):
            _text(name, getattr(self, name))
        _https_issuer("expected_oidc_issuer", self.expected_oidc_issuer)
        for name in (
            "transparency_log_uri",
            "timestamp_authority_uri",
            "state_service_uri",
        ):
            _https_uri(name, getattr(self, name))
        if _REPOSITORY.fullmatch(self.expected_repository) is None:
            raise SuiteAttemptError("expected_repository must use owner/repository syntax")
        if _WORKFLOW.fullmatch(self.expected_workflow) is None:
            raise SuiteAttemptError("expected_workflow must be an exact workflow path")
        if not self.expected_git_ref.startswith("refs/"):
            raise SuiteAttemptError("expected_git_ref must be an exact refs/... value")
        _text("expected_git_ref", self.expected_git_ref)
        if _GIT_COMMIT.fullmatch(self.expected_signer_digest) is None:
            raise SuiteAttemptError("expected_signer_digest must be one full Git commit")
        _digest(
            "transparency_log_public_key_sha256",
            self.transparency_log_public_key_sha256,
        )
        _digest(
            "timestamp_authority_public_key_sha256",
            self.timestamp_authority_public_key_sha256,
        )
        if self.schema_version != SUITE_ATTESTATION_DESCRIPTOR_SCHEMA:
            raise SuiteAttemptError("suite attestation descriptor schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def descriptor_sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> SuiteAttestationDescriptor:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="suite attestation descriptor",
        )
        return cls(**row)


@dataclass(frozen=True)
class SuiteAttestationEvidence:
    """Untrusted statement whose exact claims must be proved from the bundle."""

    suite_attempt_id: str
    state_sequence: int
    state_name: SuiteState
    state_record_sha256: str
    descriptor_sha256: str
    bundle_sha256: str
    bundle_byte_count: int
    signer_identity: str
    oidc_issuer: str
    repository: str
    workflow: str
    git_ref: str
    signer_digest: str
    github_hosted_runner: bool
    transparency_log_identity: str
    transparency_entry_id: str
    transparency_log_index: int
    integrated_at_utc: str
    timestamp_authority_identity: str
    timestamp_token_sha256: str
    signed_at_utc: str
    state_service_identity: str
    state_key: str
    transition_id: str
    previous_transition_id: str | None
    schema_version: str = SUITE_ATTESTATION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "suite_attempt_id",
            "signer_identity",
            "oidc_issuer",
            "repository",
            "workflow",
            "git_ref",
            "transparency_log_identity",
            "transparency_entry_id",
            "timestamp_authority_identity",
            "state_service_identity",
            "state_key",
            "transition_id",
        ):
            _text(name, getattr(self, name))
        if _GIT_COMMIT.fullmatch(self.signer_digest) is None:
            raise SuiteAttemptError("signer_digest must be one full Git commit")
        if type(self.github_hosted_runner) is not bool:
            raise SuiteAttemptError("github_hosted_runner must be boolean")
        if self.state_name not in {
            "OPENED",
            "RUN_CLAIMED",
            "ONLINE_COMPLETE",
            "LABEL_RELEASE_CLAIMED",
            "LABELS_RELEASED",
            "ANALYSIS_CLAIMED",
            "ANALYSIS_COMPLETE",
            "FAILED",
        }:
            raise SuiteAttemptError("state_name is not registered")
        _nonnegative_integer("state_sequence", self.state_sequence)
        _nonnegative_integer("transparency_log_index", self.transparency_log_index)
        _positive_integer("bundle_byte_count", self.bundle_byte_count)
        for name in (
            "state_record_sha256",
            "descriptor_sha256",
            "bundle_sha256",
            "timestamp_token_sha256",
        ):
            _digest(name, getattr(self, name))
        _timestamp("integrated_at_utc", self.integrated_at_utc)
        _timestamp("signed_at_utc", self.signed_at_utc)
        if self.previous_transition_id is not None:
            _text("previous_transition_id", self.previous_transition_id)
        if self.schema_version != SUITE_ATTESTATION_EVIDENCE_SCHEMA:
            raise SuiteAttemptError("suite attestation evidence schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> SuiteAttestationEvidence:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="suite attestation evidence",
        )
        return cls(**row)


@dataclass(frozen=True)
class SuiteProviderClaims:
    """Claims returned after cryptographic provider verification."""

    subject_sha256: str
    bundle_sha256: str
    signer_identity: str
    oidc_issuer: str
    repository: str
    workflow: str
    git_ref: str
    signer_digest: str
    github_hosted_runner: bool
    transparency_log_identity: str
    transparency_entry_id: str
    transparency_log_index: int
    integrated_at_utc: str
    timestamp_authority_identity: str
    timestamp_token_sha256: str
    signed_at_utc: str
    state_service_identity: str
    state_key: str
    transition_id: str
    previous_transition_id: str | None
    signature_verified: bool
    transparency_verified: bool
    timestamp_verified: bool
    exclusive_transition: bool
    schema_version: str = SUITE_PROVIDER_CLAIMS_SCHEMA

    def __post_init__(self) -> None:
        for name in ("subject_sha256", "bundle_sha256", "timestamp_token_sha256"):
            _digest(name, getattr(self, name))
        for name in (
            "signer_identity",
            "oidc_issuer",
            "repository",
            "workflow",
            "git_ref",
            "transparency_log_identity",
            "transparency_entry_id",
            "timestamp_authority_identity",
            "state_service_identity",
            "state_key",
            "transition_id",
        ):
            _text(name, getattr(self, name))
        if _GIT_COMMIT.fullmatch(self.signer_digest) is None:
            raise SuiteAttemptError("signer_digest must be one full Git commit")
        _nonnegative_integer("transparency_log_index", self.transparency_log_index)
        _timestamp("integrated_at_utc", self.integrated_at_utc)
        _timestamp("signed_at_utc", self.signed_at_utc)
        if self.previous_transition_id is not None:
            _text("previous_transition_id", self.previous_transition_id)
        for name in (
            "signature_verified",
            "transparency_verified",
            "timestamp_verified",
            "exclusive_transition",
            "github_hosted_runner",
        ):
            if type(getattr(self, name)) is not bool:
                raise SuiteAttemptError(f"{name} must be boolean")
        if self.schema_version != SUITE_PROVIDER_CLAIMS_SCHEMA:
            raise SuiteAttemptError("suite provider claims schema differs")


class SuiteEvidenceVerifier(Protocol):
    """Verifier for signed identity, transparency, timestamp, and CAS evidence."""

    def verify(
        self,
        *,
        bundle: bytes,
        evidence: SuiteAttestationEvidence,
        descriptor: SuiteAttestationDescriptor,
        state_record_bytes: bytes,
    ) -> SuiteProviderClaims: ...


@dataclass(frozen=True)
class CorpusDigest:
    """One corpus-to-digest binding in fixed UTF-8 corpus order."""

    corpus_id: str
    sha256: str

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise SuiteAttemptError("corpus digest names an unregistered corpus")
        _digest("corpus digest", self.sha256)

    def to_dict(self) -> dict[str, object]:
        return {"corpus_id": self.corpus_id, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> CorpusDigest:
        row = _closed(value, frozenset({"corpus_id", "sha256"}), label="corpus digest")
        return cls(corpus_id=row["corpus_id"], sha256=row["sha256"])


@dataclass(frozen=True)
class CorpusNamespace:
    corpus_id: str
    output_uri: str

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise SuiteAttemptError("corpus namespace names an unregistered corpus")
        text = _text("output_uri", self.output_uri)
        parsed = urlsplit(text)
        if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
            raise SuiteAttemptError("output_uri must be a canonical local file URI")
        path = Path(unquote(parsed.path))
        if not path.is_absolute() or path.as_uri() != text:
            raise SuiteAttemptError("output_uri must use canonical file URI encoding")

    def to_dict(self) -> dict[str, str]:
        return {"corpus_id": self.corpus_id, "output_uri": self.output_uri}

    @classmethod
    def from_dict(cls, value: object) -> CorpusNamespace:
        row = _closed(
            value,
            frozenset({"corpus_id", "output_uri"}),
            label="corpus namespace",
        )
        return cls(corpus_id=row["corpus_id"], output_uri=row["output_uri"])


@dataclass(frozen=True)
class CorpusRuntimePlanBinding:
    """One corpus bound to the typed two-field C1 plan instantiation."""

    corpus_id: str
    plan_sha256: str
    file_sha256: str
    production_run_closure_binding_receipt_sha256: str
    registered_plan_instantiation_receipt_sha256: str
    registered_plan_instantiation_file_sha256: str
    sealed_launch_contract_uri: str
    sealed_launch_contract_sha256: str
    sealed_launch_contract_file_sha256: str

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise SuiteAttemptError("runtime plan binding names an unregistered corpus")
        _local_file_uri("sealed_launch_contract_uri", self.sealed_launch_contract_uri)
        for name in (
            "plan_sha256",
            "file_sha256",
            "production_run_closure_binding_receipt_sha256",
            "registered_plan_instantiation_receipt_sha256",
            "registered_plan_instantiation_file_sha256",
            "sealed_launch_contract_sha256",
            "sealed_launch_contract_file_sha256",
        ):
            _digest(f"runtime plan binding {name}", getattr(self, name))

    def to_dict(self) -> dict[str, str]:
        return {
            "corpus_id": self.corpus_id,
            "file_sha256": self.file_sha256,
            "plan_sha256": self.plan_sha256,
            "production_run_closure_binding_receipt_sha256": (
                self.production_run_closure_binding_receipt_sha256
            ),
            "registered_plan_instantiation_file_sha256": (
                self.registered_plan_instantiation_file_sha256
            ),
            "registered_plan_instantiation_receipt_sha256": (
                self.registered_plan_instantiation_receipt_sha256
            ),
            "sealed_launch_contract_uri": self.sealed_launch_contract_uri,
            "sealed_launch_contract_file_sha256": self.sealed_launch_contract_file_sha256,
            "sealed_launch_contract_sha256": self.sealed_launch_contract_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> CorpusRuntimePlanBinding:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="corpus runtime plan binding",
        )
        return cls(**row)


def _fixed_corpus_rows(
    name: str,
    rows: Sequence[Any],
    *,
    row_type: type,
) -> tuple[Any, ...]:
    try:
        result = tuple(rows)
    except TypeError as exc:
        raise SuiteAttemptError(f"{name} must contain five typed corpus rows") from exc
    if len(result) != len(FIXED_CORPORA) or not all(isinstance(row, row_type) for row in result):
        raise SuiteAttemptError(f"{name} must contain five typed corpus rows")
    expected = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
    observed = tuple(row.corpus_id for row in result)
    if observed != expected:
        raise SuiteAttemptError(f"{name} must contain each fixed corpus once in UTF-8 byte order")
    return result


@dataclass(frozen=True)
class SuiteOpenBindings:
    protocol_registration_receipt_sha256: str
    protocol_registration_receipt_file_sha256: str
    protocol_registry_record_sha256: str
    registered_at_utc: str
    run_receipt_file_sha256: str
    run_started_at_utc: str
    code_commit: str
    runner_image: str
    attestation_descriptor_sha256: str
    production_finalization_receipt_uri: str
    production_finalization_receipt_file_sha256: str
    production_finalization_request_sha256: str
    provisional_closure_tree_sha256: str
    instantiated_closure_tree_sha256: str
    runtime_attestation_plans: tuple[CorpusRuntimePlanBinding, ...]
    execution_artifacts: tuple[CorpusDigest, ...]
    staging_namespaces: tuple[CorpusNamespace, ...]
    output_namespaces: tuple[CorpusNamespace, ...]

    def __post_init__(self) -> None:
        for name in (
            "protocol_registration_receipt_sha256",
            "protocol_registration_receipt_file_sha256",
            "protocol_registry_record_sha256",
            "run_receipt_file_sha256",
            "attestation_descriptor_sha256",
            "production_finalization_receipt_file_sha256",
            "production_finalization_request_sha256",
            "provisional_closure_tree_sha256",
            "instantiated_closure_tree_sha256",
        ):
            _digest(name, getattr(self, name))
        _local_file_uri(
            "production_finalization_receipt_uri",
            self.production_finalization_receipt_uri,
        )
        if self.provisional_closure_tree_sha256 == self.instantiated_closure_tree_sha256:
            raise SuiteAttemptError("OPENED production closure did not transition")
        registered = _timestamp("registered_at_utc", self.registered_at_utc)
        started = _timestamp("run_started_at_utc", self.run_started_at_utc)
        if started < registered:
            raise SuiteAttemptError("sealed run starts before protocol registration")
        _text("code_commit", self.code_commit)
        _text("runner_image", self.runner_image)
        executions = _fixed_corpus_rows(
            "execution_artifacts",
            self.execution_artifacts,
            row_type=CorpusDigest,
        )
        namespaces = _fixed_corpus_rows(
            "output_namespaces",
            self.output_namespaces,
            row_type=CorpusNamespace,
        )
        staging = _fixed_corpus_rows(
            "staging_namespaces",
            self.staging_namespaces,
            row_type=CorpusNamespace,
        )
        runtime_plans = _fixed_corpus_rows(
            "runtime_attestation_plans",
            self.runtime_attestation_plans,
            row_type=CorpusRuntimePlanBinding,
        )
        if len({row.sha256 for row in executions}) != len(FIXED_CORPORA):
            raise SuiteAttemptError("fixed corpora cannot reuse one execution artifact digest")
        if len({row.output_uri for row in namespaces}) != len(FIXED_CORPORA):
            raise SuiteAttemptError("fixed corpora cannot reuse one output namespace")
        if len({row.output_uri for row in staging}) != len(FIXED_CORPORA):
            raise SuiteAttemptError("fixed corpora cannot reuse one staging namespace")
        if {row.output_uri for row in staging} & {row.output_uri for row in namespaces}:
            raise SuiteAttemptError("staging and canonical output namespaces must be disjoint")
        staging_paths = {
            row.corpus_id: Path(unquote(urlsplit(row.output_uri).path)) for row in staging
        }
        canonical_paths = {
            row.corpus_id: Path(unquote(urlsplit(row.output_uri).path)) for row in namespaces
        }
        if (
            len({path.parent for path in staging_paths.values()}) != 1
            or len({path.parent for path in canonical_paths.values()}) != 1
            or any(path.name != corpus_id for corpus_id, path in staging_paths.items())
            or any(path.name != corpus_id for corpus_id, path in canonical_paths.items())
        ):
            raise SuiteAttemptError(
                "staging and canonical namespaces must use one corpus-named root each"
            )
        if len({row.plan_sha256 for row in runtime_plans}) != len(FIXED_CORPORA):
            raise SuiteAttemptError("fixed corpora cannot reuse one runtime plan digest")
        if len({row.file_sha256 for row in runtime_plans}) != len(FIXED_CORPORA):
            raise SuiteAttemptError("fixed corpora cannot reuse one runtime plan file")
        if len({row.sealed_launch_contract_uri for row in runtime_plans}) != len(FIXED_CORPORA):
            raise SuiteAttemptError("fixed corpora cannot reuse one sealed launch contract path")
        object.__setattr__(self, "execution_artifacts", executions)
        object.__setattr__(self, "staging_namespaces", staging)
        object.__setattr__(self, "output_namespaces", namespaces)
        object.__setattr__(self, "runtime_attestation_plans", runtime_plans)

    def to_dict(self) -> dict[str, object]:
        return {
            "attestation_descriptor_sha256": self.attestation_descriptor_sha256,
            "code_commit": self.code_commit,
            "execution_artifacts": [row.to_dict() for row in self.execution_artifacts],
            "instantiated_closure_tree_sha256": self.instantiated_closure_tree_sha256,
            "output_namespaces": [row.to_dict() for row in self.output_namespaces],
            "production_finalization_receipt_file_sha256": (
                self.production_finalization_receipt_file_sha256
            ),
            "production_finalization_receipt_uri": self.production_finalization_receipt_uri,
            "production_finalization_request_sha256": (self.production_finalization_request_sha256),
            "protocol_registration_receipt_file_sha256": (
                self.protocol_registration_receipt_file_sha256
            ),
            "protocol_registration_receipt_sha256": (self.protocol_registration_receipt_sha256),
            "protocol_registry_record_sha256": self.protocol_registry_record_sha256,
            "provisional_closure_tree_sha256": self.provisional_closure_tree_sha256,
            "registered_at_utc": self.registered_at_utc,
            "run_receipt_file_sha256": self.run_receipt_file_sha256,
            "run_started_at_utc": self.run_started_at_utc,
            "runner_image": self.runner_image,
            "runtime_attestation_plans": [row.to_dict() for row in self.runtime_attestation_plans],
            "staging_namespaces": [row.to_dict() for row in self.staging_namespaces],
        }

    @classmethod
    def from_dict(cls, value: object) -> SuiteOpenBindings:
        fields = frozenset(
            {
                "attestation_descriptor_sha256",
                "code_commit",
                "execution_artifacts",
                "instantiated_closure_tree_sha256",
                "output_namespaces",
                "production_finalization_receipt_file_sha256",
                "production_finalization_receipt_uri",
                "production_finalization_request_sha256",
                "protocol_registration_receipt_file_sha256",
                "protocol_registration_receipt_sha256",
                "protocol_registry_record_sha256",
                "provisional_closure_tree_sha256",
                "registered_at_utc",
                "run_receipt_file_sha256",
                "run_started_at_utc",
                "runner_image",
                "runtime_attestation_plans",
                "staging_namespaces",
            }
        )
        row = _closed(value, fields, label="suite OPENED bindings")
        executions = row["execution_artifacts"]
        namespaces = row["output_namespaces"]
        staging = row["staging_namespaces"]
        runtime_plans = row["runtime_attestation_plans"]
        if (
            not isinstance(executions, list)
            or not isinstance(namespaces, list)
            or not isinstance(staging, list)
            or not isinstance(runtime_plans, list)
        ):
            raise SuiteAttemptError("OPENED corpus bindings must be arrays")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "execution_artifacts",
                    "output_namespaces",
                    "runtime_attestation_plans",
                    "staging_namespaces",
                }
            },
            execution_artifacts=tuple(CorpusDigest.from_dict(item) for item in executions),
            staging_namespaces=tuple(CorpusNamespace.from_dict(item) for item in staging),
            output_namespaces=tuple(CorpusNamespace.from_dict(item) for item in namespaces),
            runtime_attestation_plans=tuple(
                CorpusRuntimePlanBinding.from_dict(item) for item in runtime_plans
            ),
        )


@dataclass(frozen=True)
class RunClaimBindings:
    """Provider-selected sole admissible online execution lineage."""

    opened_state_sha256: str
    execution_claim: ExecutionClaimContract
    provider_identity: ProviderExecutionIdentity
    zenodo_admission: AnonymousZenodoAdmission
    c1_manifest_rekor_integrated_at_utc: str
    c1_registry_rekor_integrated_at_utc: str
    workload_inputs_opened_before_claim: bool
    public_benchmark_labels_accessible: bool
    human_outcome_blindness: bool
    independent_organizational_custody: bool

    def __post_init__(self) -> None:
        _digest("opened_state_sha256", self.opened_state_sha256)
        if not isinstance(self.execution_claim, ExecutionClaimContract):
            raise SuiteAttemptError("RUN_CLAIMED execution contract must be typed")
        if not isinstance(self.provider_identity, ProviderExecutionIdentity):
            raise SuiteAttemptError("RUN_CLAIMED provider identity must be typed")
        if not isinstance(self.zenodo_admission, AnonymousZenodoAdmission):
            raise SuiteAttemptError("RUN_CLAIMED Zenodo admission must be typed")
        self.provider_identity.matches_contract(self.execution_claim)
        manifest_time = _timestamp(
            "c1_manifest_rekor_integrated_at_utc",
            self.c1_manifest_rekor_integrated_at_utc,
        )
        registry_time = _timestamp(
            "c1_registry_rekor_integrated_at_utc",
            self.c1_registry_rekor_integrated_at_utc,
        )
        published = _timestamp(
            "Zenodo published_at_utc",
            self.zenodo_admission.published_at_utc,
        )
        if published < max(manifest_time, registry_time):
            raise SuiteAttemptError("public Zenodo publication predates its C1 attestations")
        if self.workload_inputs_opened_before_claim is not False:
            raise SuiteAttemptError("RUN_CLAIMED must precede every workload input access")
        if self.public_benchmark_labels_accessible is not True:
            raise SuiteAttemptError("public benchmark label accessibility must be acknowledged")
        if self.human_outcome_blindness is not False:
            raise SuiteAttemptError("human outcome blindness cannot be claimed")
        if self.independent_organizational_custody is not False:
            raise SuiteAttemptError("independent organizational custody cannot be claimed")

    def to_dict(self) -> dict[str, object]:
        return {
            "c1_manifest_rekor_integrated_at_utc": (self.c1_manifest_rekor_integrated_at_utc),
            "c1_registry_rekor_integrated_at_utc": (self.c1_registry_rekor_integrated_at_utc),
            "execution_claim": self.execution_claim.to_dict(),
            "human_outcome_blindness": self.human_outcome_blindness,
            "independent_organizational_custody": self.independent_organizational_custody,
            "opened_state_sha256": self.opened_state_sha256,
            "provider_identity": self.provider_identity.to_dict(),
            "public_benchmark_labels_accessible": self.public_benchmark_labels_accessible,
            "workload_inputs_opened_before_claim": self.workload_inputs_opened_before_claim,
            "zenodo_admission": self.zenodo_admission.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> RunClaimBindings:
        fields = frozenset(
            {
                "c1_manifest_rekor_integrated_at_utc",
                "c1_registry_rekor_integrated_at_utc",
                "execution_claim",
                "human_outcome_blindness",
                "independent_organizational_custody",
                "opened_state_sha256",
                "provider_identity",
                "public_benchmark_labels_accessible",
                "workload_inputs_opened_before_claim",
                "zenodo_admission",
            }
        )
        row = _closed(value, fields, label="RUN_CLAIMED bindings")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key not in {"execution_claim", "provider_identity", "zenodo_admission"}
            },
            execution_claim=ExecutionClaimContract.from_dict(row["execution_claim"]),
            provider_identity=ProviderExecutionIdentity.from_dict(row["provider_identity"]),
            zenodo_admission=AnonymousZenodoAdmission.from_dict(row["zenodo_admission"]),
        )


@dataclass(frozen=True)
class PhaseClaimBindings:
    """Provider-selected sole admissible label-release or analysis lineage."""

    predecessor_state_sha256: str
    phase_claim: PhaseClaimContract
    provider_identity: ProviderExecutionIdentity
    phase_inputs_opened_before_claim: bool
    public_benchmark_labels_accessible: bool
    human_outcome_blindness: bool
    independent_organizational_custody: bool

    def __post_init__(self) -> None:
        _digest("predecessor_state_sha256", self.predecessor_state_sha256)
        if not isinstance(self.phase_claim, PhaseClaimContract):
            raise SuiteAttemptError("provider phase claim contract must be typed")
        if not isinstance(self.provider_identity, ProviderExecutionIdentity):
            raise SuiteAttemptError("provider phase identity must be typed")
        self.provider_identity.matches_phase_contract(self.phase_claim)
        if self.phase_inputs_opened_before_claim is not False:
            raise SuiteAttemptError("provider phase claim must precede every phase input access")
        if self.public_benchmark_labels_accessible is not True:
            raise SuiteAttemptError("public benchmark label accessibility must be acknowledged")
        if self.human_outcome_blindness is not False:
            raise SuiteAttemptError("human outcome blindness cannot be claimed")
        if self.independent_organizational_custody is not False:
            raise SuiteAttemptError("independent organizational custody cannot be claimed")

    def to_dict(self) -> dict[str, object]:
        return {
            "human_outcome_blindness": self.human_outcome_blindness,
            "independent_organizational_custody": self.independent_organizational_custody,
            "phase_claim": self.phase_claim.to_dict(),
            "phase_inputs_opened_before_claim": self.phase_inputs_opened_before_claim,
            "predecessor_state_sha256": self.predecessor_state_sha256,
            "provider_identity": self.provider_identity.to_dict(),
            "public_benchmark_labels_accessible": self.public_benchmark_labels_accessible,
        }

    @classmethod
    def from_dict(cls, value: object) -> PhaseClaimBindings:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="provider phase claim")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key not in {"phase_claim", "provider_identity"}
            },
            phase_claim=PhaseClaimContract.from_dict(row["phase_claim"]),
            provider_identity=ProviderExecutionIdentity.from_dict(row["provider_identity"]),
        )


@dataclass(frozen=True)
class TransferFileBinding:
    """One copied regular file, addressed relative to a corpus output root."""

    role: str
    relative_path: str
    file_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        if self.role not in _TRANSFER_ROLE_FILE_FIELD:
            raise SuiteAttemptError("transfer file role is not registered")
        path = Path(_text("transfer relative_path", self.relative_path))
        if path.is_absolute() or len(path.parts) != 1 or path.name != self.relative_path:
            raise SuiteAttemptError("transfer file path must be one canonical filename")
        _digest("transfer file SHA-256", self.file_sha256)
        _nonnegative_integer("transfer file byte_count", self.byte_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "file_sha256": self.file_sha256,
            "relative_path": self.relative_path,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, value: object) -> TransferFileBinding:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="transfer file binding",
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CorpusOutputTransfer:
    """Exact source-to-canonical byte transfer for one fixed corpus."""

    corpus_id: str
    staging_output_uri: str
    canonical_output_uri: str
    source_tree_sha256: str
    canonical_tree_sha256: str
    entries: tuple[str, ...]
    files: tuple[TransferFileBinding, ...]

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise SuiteAttemptError("output transfer names an unregistered corpus")
        staging_uri, staging = _local_file_uri(
            "staging_output_uri",
            self.staging_output_uri,
        )
        canonical_uri, canonical = _local_file_uri(
            "canonical_output_uri",
            self.canonical_output_uri,
        )
        if staging == canonical:
            raise SuiteAttemptError("staging and canonical corpus roots must differ")
        object.__setattr__(self, "staging_output_uri", staging_uri)
        object.__setattr__(self, "canonical_output_uri", canonical_uri)
        for name in ("source_tree_sha256", "canonical_tree_sha256"):
            _digest(name, getattr(self, name))
        if self.source_tree_sha256 != self.canonical_tree_sha256:
            raise SuiteAttemptError("source and canonical corpus trees differ")
        entries = tuple(_text("transfer entry", item) for item in self.entries)
        if (
            not entries
            or entries != tuple(sorted(entries, key=lambda item: item.encode("utf-8")))
            or len(entries) != len(set(entries))
            or any(Path(item).name != item or len(Path(item).parts) != 1 for item in entries)
        ):
            raise SuiteAttemptError("corpus transfer entries must be unique sorted filenames")
        files = tuple(self.files)
        if (
            not all(isinstance(item, TransferFileBinding) for item in files)
            or files != tuple(sorted(files, key=lambda item: item.relative_path.encode("utf-8")))
            or tuple(item.relative_path for item in files) != entries
            or {item.role for item in files} != set(_TRANSFER_ROLE_FILE_FIELD)
        ):
            raise SuiteAttemptError(
                "corpus transfer files differ from its exact role and filename inventory"
            )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "files", files)

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_output_uri": self.canonical_output_uri,
            "canonical_tree_sha256": self.canonical_tree_sha256,
            "corpus_id": self.corpus_id,
            "entries": list(self.entries),
            "files": [item.to_dict() for item in self.files],
            "source_tree_sha256": self.source_tree_sha256,
            "staging_output_uri": self.staging_output_uri,
        }

    @classmethod
    def from_dict(cls, value: object) -> CorpusOutputTransfer:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="corpus output transfer",
        )
        entries = row["entries"]
        files = row["files"]
        if type(entries) is not list or type(files) is not list:
            raise SuiteAttemptError("corpus output transfer arrays are malformed")
        return cls(
            **{key: item for key, item in row.items() if key not in {"entries", "files"}},
            entries=tuple(entries),
            files=tuple(TransferFileBinding.from_dict(item) for item in files),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class SuiteOutputTransferReceipt:
    """Persisted proof that all five staging trees moved as one exact set."""

    suite_attempt_id: str
    manifest_sha256: str
    production_finalization_receipt_file_sha256: str
    staging_online_root_uri: str
    canonical_online_root_uri: str
    retained_empty_placeholder_uri: str
    empty_placeholder_tree_sha256: str
    source_online_tree_sha256: str
    canonical_online_tree_sha256: str
    entries: tuple[str, ...]
    corpora: tuple[CorpusOutputTransfer, ...]
    schema_version: str = SUITE_OUTPUT_TRANSFER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SUITE_OUTPUT_TRANSFER_SCHEMA:
            raise SuiteAttemptError("suite output transfer schema differs")
        for name in (
            "suite_attempt_id",
            "manifest_sha256",
            "production_finalization_receipt_file_sha256",
            "source_online_tree_sha256",
            "canonical_online_tree_sha256",
            "empty_placeholder_tree_sha256",
        ):
            _digest(name, getattr(self, name))
        source_uri, source = _local_file_uri(
            "staging_online_root_uri",
            self.staging_online_root_uri,
        )
        canonical_uri, canonical = _local_file_uri(
            "canonical_online_root_uri",
            self.canonical_online_root_uri,
        )
        retained_uri, retained = _local_file_uri(
            "retained_empty_placeholder_uri",
            self.retained_empty_placeholder_uri,
        )
        if source == canonical:
            raise SuiteAttemptError("staging and canonical online roots must differ")
        if retained in {source, canonical}:
            raise SuiteAttemptError("retained placeholder must be outside both output roots")
        object.__setattr__(self, "staging_online_root_uri", source_uri)
        object.__setattr__(self, "canonical_online_root_uri", canonical_uri)
        object.__setattr__(self, "retained_empty_placeholder_uri", retained_uri)
        if self.source_online_tree_sha256 != self.canonical_online_tree_sha256:
            raise SuiteAttemptError("source and canonical online trees differ")
        rows = _fixed_corpus_rows(
            "suite output transfer",
            self.corpora,
            row_type=CorpusOutputTransfer,
        )
        expected_entries = tuple(
            sorted(
                (
                    *(row.corpus_id for row in rows),
                    *(f"{row.corpus_id}/{entry}" for row in rows for entry in row.entries),
                ),
                key=lambda item: item.encode("utf-8"),
            )
        )
        entries = tuple(self.entries)
        if entries != expected_entries:
            raise SuiteAttemptError("suite output transfer inventory differs from its corpora")
        for row in rows:
            if (
                Path(unquote(urlsplit(row.staging_output_uri).path)).parent != source
                or Path(unquote(urlsplit(row.canonical_output_uri).path)).parent != canonical
                or Path(unquote(urlsplit(row.staging_output_uri).path)).name != row.corpus_id
                or Path(unquote(urlsplit(row.canonical_output_uri).path)).name != row.corpus_id
            ):
                raise SuiteAttemptError("corpus transfer root differs from the suite roots")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "corpora", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_online_root_uri": self.canonical_online_root_uri,
            "canonical_online_tree_sha256": self.canonical_online_tree_sha256,
            "corpora": [row.to_dict() for row in self.corpora],
            "entries": list(self.entries),
            "empty_placeholder_tree_sha256": self.empty_placeholder_tree_sha256,
            "manifest_sha256": self.manifest_sha256,
            "production_finalization_receipt_file_sha256": (
                self.production_finalization_receipt_file_sha256
            ),
            "retained_empty_placeholder_uri": self.retained_empty_placeholder_uri,
            "schema_version": self.schema_version,
            "source_online_tree_sha256": self.source_online_tree_sha256,
            "staging_online_root_uri": self.staging_online_root_uri,
            "suite_attempt_id": self.suite_attempt_id,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> SuiteOutputTransferReceipt:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="suite output transfer receipt",
        )
        entries = row["entries"]
        corpora = row["corpora"]
        if type(entries) is not list or type(corpora) is not list:
            raise SuiteAttemptError("suite output transfer arrays are malformed")
        return cls(
            **{key: item for key, item in row.items() if key not in {"entries", "corpora"}},
            entries=tuple(entries),
            corpora=tuple(CorpusOutputTransfer.from_dict(item) for item in corpora),
        )  # type: ignore[arg-type]


def load_suite_output_transfer_receipt(
    path: str | Path,
) -> SuiteOutputTransferReceipt:
    encoded = _secure_bytes(
        path,
        label="suite output transfer receipt",
        max_bytes=_MAX_ATTESTATION_BYTES,
    )
    receipt = SuiteOutputTransferReceipt.from_dict(
        _parse_object(encoded, label="suite output transfer receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise SuiteAttemptError("suite output transfer receipt bytes are not canonical")
    return receipt


def _load_private_suite_output_transfer_receipt(path: Path) -> SuiteOutputTransferReceipt:
    try:
        before = path.lstat()
    except OSError as exc:
        raise SuiteAttemptError(f"cannot inspect suite output transfer receipt: {exc}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
    ):
        raise SuiteAttemptError(
            "suite output transfer receipt must be a private singly linked regular file"
        )
    receipt = load_suite_output_transfer_receipt(path)
    try:
        after = path.lstat()
    except OSError as exc:
        raise SuiteAttemptError(f"cannot re-inspect suite output transfer receipt: {exc}") from exc
    if _stable_file_signature(before) != _stable_file_signature(after):
        raise SuiteAttemptError("suite output transfer receipt changed during admission")
    return receipt


@dataclass(frozen=True)
class OnlineCorpusClosure:
    """Exact persisted pre-label closure for one fixed corpus."""

    corpus_id: str
    staging_output_uri: str
    output_uri: str
    execution_artifact_sha256: str
    runtime_attestation_plan_sha256: str
    runtime_attestation_plan_file_sha256: str
    runtime_attestation_receipt_sha256: str
    runtime_attestation_receipt_file_sha256: str
    runtime_invocation_marker_sha256: str
    runtime_invocation_marker_file_sha256: str
    production_command_attempt_sha256: str
    production_command_attempt_file_sha256: str
    sealed_launch_receipt_uri: str
    sealed_launch_copy_output_uri: str
    sealed_launch_contract_sha256: str
    sealed_launch_receipt_sha256: str
    sealed_launch_receipt_file_sha256: str
    sealed_launch_evidence_inventory_sha256: str
    sealed_launch_output_tree_sha256: str
    attempt_receipt_sha256: str
    attempt_file_sha256: str
    result_receipt_sha256: str
    result_file_sha256: str
    prediction_artifact_sha256: str
    prediction_file_sha256: str
    action_panel_artifact_sha256: str
    action_panel_file_sha256: str
    action_panel_admission_receipt_sha256: str
    action_panel_admission_file_sha256: str
    audit_head_sha256: str
    audit_file_sha256: str
    audit_record_count: int
    cache_preparation_receipt_sha256: str
    cache_preparation_file_sha256: str
    execution_order_receipt_sha256: str
    execution_order_file_sha256: str
    transfer_files: tuple[TransferFileBinding, ...]

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise SuiteAttemptError("online closure names an unregistered corpus")
        CorpusNamespace(self.corpus_id, self.staging_output_uri)
        CorpusNamespace(self.corpus_id, self.output_uri)
        if self.staging_output_uri == self.output_uri:
            raise SuiteAttemptError("online closure staging and canonical roots must differ")
        _local_file_uri("sealed_launch_receipt_uri", self.sealed_launch_receipt_uri)
        _local_file_uri("sealed_launch_copy_output_uri", self.sealed_launch_copy_output_uri)
        if self.sealed_launch_copy_output_uri != self.staging_output_uri:
            raise SuiteAttemptError("sealed launch copy root differs from the staging output root")
        for name in self.__dataclass_fields__:
            if name.endswith("sha256"):
                _digest(name, getattr(self, name))
        if self.runtime_invocation_marker_sha256 != self.runtime_invocation_marker_file_sha256:
            raise SuiteAttemptError(
                "runtime invocation marker semantic and file digests must match"
            )
        _positive_integer("audit_record_count", self.audit_record_count)
        files = tuple(self.transfer_files)
        if (
            len(files) != len(_TRANSFER_ROLE_FILE_FIELD)
            or not all(isinstance(item, TransferFileBinding) for item in files)
            or files != tuple(sorted(files, key=lambda item: item.relative_path.encode("utf-8")))
            or len({item.relative_path for item in files}) != len(files)
            or {item.role for item in files} != set(_TRANSFER_ROLE_FILE_FIELD)
        ):
            raise SuiteAttemptError(
                "online closure transfer files must bind each exact role and filename once"
            )
        by_role = {item.role: item for item in files}
        for role, field_name in _TRANSFER_ROLE_FILE_FIELD.items():
            if by_role[role].file_sha256 != getattr(self, field_name):
                raise SuiteAttemptError(
                    f"online closure {role} transfer digest differs from {field_name}"
                )
        for role, filename in _TRANSFER_CONTROL_FILENAMES.items():
            if by_role[role].relative_path != filename:
                raise SuiteAttemptError(f"online closure {role} filename differs")
        object.__setattr__(self, "transfer_files", files)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "transfer_files"
            },
            "transfer_files": [item.to_dict() for item in self.transfer_files],
        }

    @classmethod
    def from_dict(cls, value: object) -> OnlineCorpusClosure:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="online corpus closure",
        )
        transfer_files = row["transfer_files"]
        if type(transfer_files) is not list:
            raise SuiteAttemptError("online corpus closure transfer_files must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "transfer_files"},
            transfer_files=tuple(TransferFileBinding.from_dict(item) for item in transfer_files),
        )


@dataclass(frozen=True)
class OnlineSuiteClosure:
    """Exact per-corpus runtime and output closure for the five online corpora."""

    corpora: tuple[OnlineCorpusClosure, ...]
    output_transfer_receipt_uri: str
    output_transfer_receipt_sha256: str
    output_transfer_receipt_file_sha256: str
    source_online_tree_sha256: str
    canonical_online_tree_sha256: str
    run_output_aggregate: RunOutputAggregate

    def __post_init__(self) -> None:
        rows = _fixed_corpus_rows(
            "online completion",
            self.corpora,
            row_type=OnlineCorpusClosure,
        )
        _local_file_uri("output_transfer_receipt_uri", self.output_transfer_receipt_uri)
        for name in (
            "output_transfer_receipt_sha256",
            "output_transfer_receipt_file_sha256",
            "source_online_tree_sha256",
            "canonical_online_tree_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.source_online_tree_sha256 != self.canonical_online_tree_sha256:
            raise SuiteAttemptError("ONLINE_COMPLETE source and canonical trees differ")
        if not isinstance(self.run_output_aggregate, RunOutputAggregate):
            raise SuiteAttemptError("ONLINE_COMPLETE output aggregate must be typed")
        aggregate_rows = {row.corpus_id: row for row in self.run_output_aggregate.corpus_trees}
        for row in rows:
            if (
                aggregate_rows[row.corpus_id].output_namespace_uri != row.output_uri
                or aggregate_rows[row.corpus_id].tree_sha256 != row.sealed_launch_output_tree_sha256
            ):
                raise SuiteAttemptError("ONLINE_COMPLETE aggregate changes an output namespace")
        for name in (
            "output_uri",
            "staging_output_uri",
            "execution_artifact_sha256",
            "attempt_receipt_sha256",
            "result_receipt_sha256",
            "prediction_artifact_sha256",
            "action_panel_artifact_sha256",
            "runtime_attestation_plan_sha256",
            "runtime_attestation_plan_file_sha256",
            "runtime_attestation_receipt_sha256",
            "runtime_attestation_receipt_file_sha256",
            "runtime_invocation_marker_sha256",
            "runtime_invocation_marker_file_sha256",
            "production_command_attempt_sha256",
            "production_command_attempt_file_sha256",
            "sealed_launch_receipt_uri",
            "sealed_launch_copy_output_uri",
            "sealed_launch_contract_sha256",
            "sealed_launch_receipt_sha256",
            "sealed_launch_receipt_file_sha256",
            "sealed_launch_evidence_inventory_sha256",
            "sealed_launch_output_tree_sha256",
        ):
            if len({getattr(row, name) for row in rows}) != len(FIXED_CORPORA):
                raise SuiteAttemptError(f"online corpus closures repeat {name}")
        object.__setattr__(self, "corpora", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_online_tree_sha256": self.canonical_online_tree_sha256,
            "corpora": [row.to_dict() for row in self.corpora],
            "output_transfer_receipt_file_sha256": (self.output_transfer_receipt_file_sha256),
            "output_transfer_receipt_sha256": self.output_transfer_receipt_sha256,
            "output_transfer_receipt_uri": self.output_transfer_receipt_uri,
            "source_online_tree_sha256": self.source_online_tree_sha256,
            "run_output_aggregate": self.run_output_aggregate.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> OnlineSuiteClosure:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="online suite closure",
        )
        corpora = row["corpora"]
        if not isinstance(corpora, list):
            raise SuiteAttemptError("online suite closure corpora must be an array")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key not in {"corpora", "run_output_aggregate"}
            },
            corpora=tuple(OnlineCorpusClosure.from_dict(item) for item in corpora),
            run_output_aggregate=RunOutputAggregate.from_dict(row["run_output_aggregate"]),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class LabelCorpusClosure:
    corpus_id: str
    decryption_receipt_uri: str
    decryption_receipt_sha256: str
    decryption_receipt_file_sha256: str
    decryption_receipt_byte_count: int
    plaintext_uri: str
    plaintext_sha256: str
    plaintext_byte_count: int

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise SuiteAttemptError("label closure names an unregistered corpus")
        for name in (
            "decryption_receipt_sha256",
            "decryption_receipt_file_sha256",
            "plaintext_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in ("decryption_receipt_uri", "plaintext_uri"):
            value = _text(name, getattr(self, name))
            parsed = urlsplit(value)
            path = Path(unquote(parsed.path))
            if (
                parsed.scheme != "file"
                or parsed.netloc
                or parsed.query
                or parsed.fragment
                or not path.is_absolute()
                or path.as_uri() != value
            ):
                raise SuiteAttemptError(f"{name} must be a canonical local file URI")
        _positive_integer(
            "decryption_receipt_byte_count",
            self.decryption_receipt_byte_count,
        )
        _positive_integer("plaintext_byte_count", self.plaintext_byte_count)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> LabelCorpusClosure:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="label corpus closure",
        )
        return cls(**row)


@dataclass(frozen=True)
class AnalysisClosure:
    confirmatory_input_artifact_sha256: str
    analysis_execution_receipt_uri: str
    analysis_execution_receipt_sha256: str
    analysis_execution_receipt_file_sha256: str
    analysis_execution_receipt_byte_count: int
    analysis_attempt_receipt_uri: str
    analysis_attempt_receipt_sha256: str
    analysis_attempt_file_sha256: str
    analysis_attempt_byte_count: int
    analysis_result_receipt_uri: str
    analysis_result_receipt_sha256: str
    analysis_result_receipt_file_sha256: str
    analysis_result_receipt_byte_count: int
    final_result_uri: str
    final_result_artifact_sha256: str
    final_result_file_sha256: str
    final_result_byte_count: int

    def __post_init__(self) -> None:
        for name in (
            "confirmatory_input_artifact_sha256",
            "analysis_execution_receipt_sha256",
            "analysis_execution_receipt_file_sha256",
            "analysis_attempt_receipt_sha256",
            "analysis_attempt_file_sha256",
            "analysis_result_receipt_sha256",
            "analysis_result_receipt_file_sha256",
            "final_result_artifact_sha256",
            "final_result_file_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in (
            "analysis_execution_receipt_uri",
            "analysis_attempt_receipt_uri",
            "analysis_result_receipt_uri",
            "final_result_uri",
        ):
            _local_file_uri(name, getattr(self, name))
        for name in (
            "analysis_execution_receipt_byte_count",
            "analysis_attempt_byte_count",
            "analysis_result_receipt_byte_count",
            "final_result_byte_count",
        ):
            _positive_integer(name, getattr(self, name))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> AnalysisClosure:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="analysis closure",
        )
        return cls(**row)


StatePayload = (
    SuiteOpenBindings
    | RunClaimBindings
    | PhaseClaimBindings
    | OnlineSuiteClosure
    | tuple[LabelCorpusClosure, ...]
    | AnalysisClosure
    | ProviderPhaseFailure
)


@dataclass(frozen=True)
class SuiteStateRecord:
    suite_attempt_id: str
    manifest_sha256: str
    run_receipt_sha256: str
    namespace_uri: str
    sequence: int
    state: SuiteState
    previous_state_record_sha256: str | None
    payload: StatePayload
    schema_version: str = SUITE_STATE_SCHEMA

    def __post_init__(self) -> None:
        _digest("suite_attempt_id", self.suite_attempt_id)
        _digest("manifest_sha256", self.manifest_sha256)
        _digest("run_receipt_sha256", self.run_receipt_sha256)
        CorpusNamespace(FIXED_CORPORA[0], self.namespace_uri)
        _nonnegative_integer("sequence", self.sequence)
        if self.previous_state_record_sha256 is not None:
            _digest("previous_state_record_sha256", self.previous_state_record_sha256)
        if self.sequence == 0:
            if self.state != "OPENED" or self.previous_state_record_sha256 is not None:
                raise SuiteAttemptError("sequence zero must be an unlinked OPENED state")
            if not isinstance(self.payload, SuiteOpenBindings):
                raise SuiteAttemptError("OPENED state requires SuiteOpenBindings")
            namespace = Path(unquote(urlsplit(self.namespace_uri).path))
            if any(
                Path(unquote(urlsplit(row.output_uri).path)) != namespace / "online" / row.corpus_id
                for row in self.payload.output_namespaces
            ):
                raise SuiteAttemptError(
                    "OPENED canonical outputs are not below the registered namespace"
                )
        elif self.previous_state_record_sha256 is None:
            raise SuiteAttemptError("non-genesis state requires its predecessor digest")
        if self.state == "ONLINE_COMPLETE":
            if not isinstance(self.payload, OnlineSuiteClosure):
                raise SuiteAttemptError("ONLINE_COMPLETE requires OnlineSuiteClosure")
        elif self.state == "RUN_CLAIMED":
            if not isinstance(self.payload, RunClaimBindings):
                raise SuiteAttemptError("RUN_CLAIMED requires RunClaimBindings")
        elif self.state in {"LABEL_RELEASE_CLAIMED", "ANALYSIS_CLAIMED"}:
            if not isinstance(self.payload, PhaseClaimBindings):
                raise SuiteAttemptError(f"{self.state} requires PhaseClaimBindings")
            expected_phase = {
                "LABEL_RELEASE_CLAIMED": "label-release",
                "ANALYSIS_CLAIMED": "analysis",
            }[self.state]
            if self.payload.phase_claim.phase != expected_phase:
                raise SuiteAttemptError(f"{self.state} phase differs")
        elif self.state == "LABELS_RELEASED":
            rows = _fixed_corpus_rows(
                "label release",
                self.payload,  # type: ignore[arg-type]
                row_type=LabelCorpusClosure,
            )
            if len({row.decryption_receipt_sha256 for row in rows}) != len(FIXED_CORPORA):
                raise SuiteAttemptError("label release repeats a decryption receipt")
            object.__setattr__(self, "payload", rows)
        elif self.state == "ANALYSIS_COMPLETE":
            if not isinstance(self.payload, AnalysisClosure):
                raise SuiteAttemptError("ANALYSIS_COMPLETE requires AnalysisClosure")
        elif self.state == "FAILED":
            if not isinstance(self.payload, ProviderPhaseFailure):
                raise SuiteAttemptError("FAILED requires evidence-backed ProviderPhaseFailure")
        elif self.state != "OPENED":
            raise SuiteAttemptError("suite state is not registered")
        if self.schema_version != SUITE_STATE_SCHEMA:
            raise SuiteAttemptError("suite state schema differs")

    def to_dict(self) -> dict[str, object]:
        payload: object
        if isinstance(self.payload, tuple):
            payload = [row.to_dict() for row in self.payload]
        else:
            payload = self.payload.to_dict()
        return {
            "manifest_sha256": self.manifest_sha256,
            "namespace_uri": self.namespace_uri,
            "payload": payload,
            "previous_state_record_sha256": self.previous_state_record_sha256,
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "state": self.state,
            "suite_attempt_id": self.suite_attempt_id,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def record_sha256(self) -> str:
        return _sha256(self.canonical_bytes() + b"\n")

    @classmethod
    def from_dict(cls, value: object) -> SuiteStateRecord:
        fields = frozenset(
            {
                "manifest_sha256",
                "namespace_uri",
                "payload",
                "previous_state_record_sha256",
                "run_receipt_sha256",
                "schema_version",
                "sequence",
                "state",
                "suite_attempt_id",
            }
        )
        row = _closed(value, fields, label="suite state record")
        state = row["state"]
        raw_payload = row["payload"]
        if state == "OPENED":
            payload: StatePayload = SuiteOpenBindings.from_dict(raw_payload)
        elif state == "RUN_CLAIMED":
            payload = RunClaimBindings.from_dict(raw_payload)
        elif state in {"LABEL_RELEASE_CLAIMED", "ANALYSIS_CLAIMED"}:
            payload = PhaseClaimBindings.from_dict(raw_payload)
        elif state == "ONLINE_COMPLETE":
            payload = OnlineSuiteClosure.from_dict(raw_payload)
        elif state == "LABELS_RELEASED":
            if not isinstance(raw_payload, list):
                raise SuiteAttemptError("LABELS_RELEASED payload must be an array")
            payload = tuple(LabelCorpusClosure.from_dict(item) for item in raw_payload)
        elif state == "ANALYSIS_COMPLETE":
            payload = AnalysisClosure.from_dict(raw_payload)
        elif state == "FAILED":
            payload = ProviderPhaseFailure.from_dict(raw_payload)
        else:
            raise SuiteAttemptError("suite state is not registered")
        return cls(
            suite_attempt_id=row["suite_attempt_id"],
            manifest_sha256=row["manifest_sha256"],
            run_receipt_sha256=row["run_receipt_sha256"],
            namespace_uri=row["namespace_uri"],
            sequence=row["sequence"],
            state=state,
            previous_state_record_sha256=row["previous_state_record_sha256"],
            payload=payload,
            schema_version=row["schema_version"],
        )


def suite_attempt_id(manifest_digest: str) -> str:
    """Derive the sole suite-attempt key from the frozen manifest digest."""

    digest = _digest("manifest_sha256", manifest_digest)
    return _sha256(b"fractal-suite-attempt-v1\0" + digest.encode("ascii"))


def suite_namespace(base_root: str | Path, manifest_digest: str) -> Path:
    root = Path(base_root)
    if not root.is_absolute() or root.anchor != "/":
        raise SuiteAttemptError("suite base root must be an absolute POSIX path")
    if any(part in {"", ".", ".."} for part in root.parts[1:]):
        raise SuiteAttemptError("suite base root cannot contain aliasing components")
    return root / f"suite-attempt-{suite_attempt_id(manifest_digest)}"


def _namespace_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.anchor != "/":
        raise SuiteAttemptError("suite namespace must be an absolute POSIX path")
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise SuiteAttemptError("suite namespace cannot contain aliasing components")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SuiteAttemptError(f"cannot inspect suite namespace: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise SuiteAttemptError("suite namespace must be a real directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise SuiteAttemptError("suite namespace must be owned by the current identity")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise SuiteAttemptError("suite namespace cannot be writable by group or other users")
    return path


def _state_path(namespace: Path, sequence: int) -> Path:
    return namespace / f"{sequence:03d}.state.json"


def _evidence_path(namespace: Path, sequence: int) -> Path:
    return namespace / f"{sequence:03d}.attestation.json"


def _bundle_path(namespace: Path, sequence: int) -> Path:
    return namespace / f"{sequence:03d}.sigstore.bundle.json"


def _descriptor_path(namespace: Path) -> Path:
    return namespace / "attestation-descriptor.json"


def online_output_root(namespace: str | Path, corpus_id: str) -> Path:
    root = _namespace_path(namespace)
    if corpus_id not in FIXED_CORPORA:
        raise SuiteAttemptError("corpus_id is not in the fixed suite")
    return root / "online" / corpus_id


def _load_descriptor(path: Path) -> SuiteAttestationDescriptor:
    encoded = _secure_bytes(
        path,
        label="suite attestation descriptor",
        max_bytes=_MAX_ATTESTATION_BYTES,
    )
    descriptor = SuiteAttestationDescriptor.from_dict(
        _parse_object(encoded, label="suite attestation descriptor")
    )
    if encoded != descriptor.canonical_bytes() + b"\n":
        raise SuiteAttemptError("suite attestation descriptor bytes are not canonical")
    return descriptor


def load_suite_state_record(path: str | Path) -> SuiteStateRecord:
    encoded = _secure_bytes(
        path,
        label="suite state record",
        max_bytes=_MAX_ATTESTATION_BYTES,
    )
    record = SuiteStateRecord.from_dict(_parse_object(encoded, label="suite state record"))
    if encoded != record.canonical_bytes() + b"\n":
        raise SuiteAttemptError("suite state record bytes are not canonical")
    return record


def load_suite_attestation_evidence(path: str | Path) -> SuiteAttestationEvidence:
    encoded = _secure_bytes(
        path,
        label="suite attestation evidence",
        max_bytes=_MAX_ATTESTATION_BYTES,
    )
    evidence = SuiteAttestationEvidence.from_dict(
        _parse_object(encoded, label="suite attestation evidence")
    )
    if encoded != evidence.canonical_bytes() + b"\n":
        raise SuiteAttemptError("suite attestation evidence bytes are not canonical")
    return evidence


def write_suite_attestation_evidence(
    evidence: SuiteAttestationEvidence,
    *,
    namespace: str | Path,
    bundle: bytes,
) -> None:
    """Persist provider output without treating it as verified evidence."""

    root = _namespace_path(namespace)
    if not isinstance(evidence, SuiteAttestationEvidence):
        raise SuiteAttemptError("evidence must be SuiteAttestationEvidence")
    if type(bundle) is not bytes or not bundle or len(bundle) > _MAX_ATTESTATION_BYTES:
        raise SuiteAttemptError("provider bundle must be non-empty bounded bytes")
    if len(bundle) != evidence.bundle_byte_count or _sha256(bundle) != evidence.bundle_sha256:
        raise SuiteAttemptError("provider bundle differs from its untrusted evidence record")
    state = load_suite_state_record(_state_path(root, evidence.state_sequence))
    if state.record_sha256 != evidence.state_record_sha256:
        raise SuiteAttemptError("attestation evidence binds another state record")
    _write_once(bundle, _bundle_path(root, evidence.state_sequence), label="provider bundle")
    _write_once(
        evidence.canonical_bytes() + b"\n",
        _evidence_path(root, evidence.state_sequence),
        label="suite attestation evidence",
    )


def _secure_parent(path: Path) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SuiteAttemptError(f"cannot inspect suite base root: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise SuiteAttemptError("suite base root must be a real directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise SuiteAttemptError("suite base root must be owned by the current identity")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise SuiteAttemptError("suite base root cannot be writable by group or other users")


def _singleton_manifest_artifact_sha256(
    manifest: Mapping[str, Any],
    *,
    role: str,
) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SuiteAttemptError("frozen manifest artifacts are malformed")
    matches = [row for row in artifacts if isinstance(row, Mapping) and row.get("role") == role]
    if len(matches) != 1:
        raise SuiteAttemptError(f"frozen manifest must pin exactly one {role!r} artifact")
    return _digest(f"{role} artifact SHA-256", matches[0].get("sha256"))


def _corpus_manifest_artifact_sha256(
    manifest: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SuiteAttemptError("frozen manifest artifacts are malformed")
    matches = [row for row in artifacts if isinstance(row, Mapping) and row.get("role") == role]
    if len(matches) != len(FIXED_CORPORA):
        raise SuiteAttemptError(f"frozen manifest must pin one {role!r} artifact per fixed corpus")
    result: dict[str, str] = {}
    for row in matches:
        corpus_id = row.get("corpus_id")
        if not isinstance(corpus_id, str) or corpus_id not in FIXED_CORPORA or corpus_id in result:
            raise SuiteAttemptError(
                f"frozen manifest must pin one {role!r} artifact per fixed corpus"
            )
        result[corpus_id] = _digest(f"{role} artifact SHA-256", row.get("sha256"))
    if set(result) != set(FIXED_CORPORA):
        raise SuiteAttemptError(f"frozen manifest must pin one {role!r} artifact per fixed corpus")
    return result


def _registered_execution_artifact_sha256(manifest: Mapping[str, Any]) -> dict[str, str]:
    workloads = manifest.get("production_workloads")
    if not isinstance(workloads, list) or len(workloads) != len(FIXED_CORPORA):
        raise SuiteAttemptError("frozen manifest production workloads are malformed")
    result: dict[str, str] = {}
    for row in workloads:
        if not isinstance(row, Mapping):
            raise SuiteAttemptError("frozen manifest production workloads are malformed")
        corpus_id = row.get("corpus_id")
        spec = row.get("spec")
        if (
            not isinstance(corpus_id, str)
            or corpus_id not in FIXED_CORPORA
            or corpus_id in result
            or not isinstance(spec, Mapping)
        ):
            raise SuiteAttemptError("frozen manifest production workloads are malformed")
        result[corpus_id] = _digest(
            f"{corpus_id} registered execution artifact SHA-256",
            spec.get("online_execution_plan_sha256"),
        )
    if set(result) != set(FIXED_CORPORA):
        raise SuiteAttemptError("frozen manifest production workloads are malformed")
    return {corpus_id: result[corpus_id] for corpus_id in FIXED_CORPORA}


def _admit_registered_execution_artifacts(
    manifest: Mapping[str, Any],
    supplied: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(supplied, Mapping) or set(supplied) != set(FIXED_CORPORA):
        raise SuiteAttemptError("OPENED must bind one execution digest for each fixed corpus")
    registered = _registered_execution_artifact_sha256(manifest)
    if any(
        type(supplied[corpus_id]) is not str or supplied[corpus_id] != registered[corpus_id]
        for corpus_id in FIXED_CORPORA
    ):
        raise SuiteAttemptError("OPENED execution artifacts differ from the frozen C1 workloads")
    return registered


def _fixed_corpus_paths(
    name: str,
    values: Mapping[str, str | Path],
) -> dict[str, Path]:
    if not isinstance(values, Mapping):
        raise SuiteAttemptError(f"{name} must map each fixed corpus to one path")
    if set(values) != set(FIXED_CORPORA) or len(values) != len(FIXED_CORPORA):
        raise SuiteAttemptError(f"{name} must map each fixed corpus to one path")
    try:
        paths = {corpus_id: Path(values[corpus_id]) for corpus_id in FIXED_CORPORA}
    except TypeError as exc:
        raise SuiteAttemptError(f"{name} values must be filesystem paths") from exc
    if not all(path.is_absolute() for path in paths.values()):
        raise SuiteAttemptError(f"{name} paths must be absolute")
    if len(set(paths.values())) != len(FIXED_CORPORA):
        raise SuiteAttemptError(f"{name} cannot reuse one path across fixed corpora")
    return paths


def _fixed_verified_closures(
    values: Mapping[str, VerifiedProductionRunClosure],
) -> dict[str, VerifiedProductionRunClosure]:
    if not isinstance(values, Mapping) or set(values) != set(FIXED_CORPORA):
        raise SuiteAttemptError(
            "verified production closures must contain each fixed corpus exactly once"
        )
    result: dict[str, VerifiedProductionRunClosure] = {}
    for corpus_id in FIXED_CORPORA:
        value = values[corpus_id]
        if not isinstance(value, VerifiedProductionRunClosure):
            raise SuiteAttemptError("production closure authority must be guarded and typed")
        value.assert_current()
        if value.binding.corpus_id != corpus_id:
            raise SuiteAttemptError("production closure authority names another corpus")
        result[corpus_id] = value
    return result


@dataclass(frozen=True)
class _AdmittedProductionFinalization:
    receipt: Any
    receipt_path: Path
    receipt_file_sha256: str
    closures: Mapping[str, VerifiedProductionRunClosure]


def _admit_production_finalization(
    *,
    manifest_digest: str,
    verified_registration: VerifiedC1ProtocolRegistration,
    finalization_receipt_path: str | Path,
    verified_closures: Mapping[str, VerifiedProductionRunClosure],
) -> _AdmittedProductionFinalization:
    # Lazy import avoids the intentional production_controls -> suite_attempt edge.
    from .production_controls import (
        ProductionControlError,
        load_production_control_finalization_receipt,
    )

    path = Path(finalization_receipt_path)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise SuiteAttemptError("production finalization receipt path must be canonical absolute")
    closures = _fixed_verified_closures(verified_closures)
    try:
        receipt = load_production_control_finalization_receipt(path)
        receipt_file_sha256 = digest_regular_file(
            path,
            label="production control finalization receipt",
        )
    except (ArtifactIntegrityError, ProductionControlError) as exc:
        raise SuiteAttemptError(f"cannot admit production finalization: {exc}") from exc
    verified_registration.assert_current()
    expected_attempt_id = suite_attempt_id(manifest_digest)
    canonical = Path(receipt.canonical_suite_namespace)
    staging = Path(receipt.pre_c1_output_staging_root)
    if (
        receipt_file_sha256 != receipt.receipt_sha256
        or receipt.manifest_sha256 != manifest_digest
        or receipt.c0_commit != verified_registration.c0_commit
        or receipt.c1_commit != verified_registration.c1_commit
        or receipt.suite_attempt_id != expected_attempt_id
        or canonical.name != f"suite-attempt-{expected_attempt_id}"
        or canonical != suite_namespace(canonical.parent, manifest_digest)
        or staging.parent != canonical.parent
        or staging == canonical
    ):
        raise SuiteAttemptError(
            "production finalization differs from the verified C1 suite identity"
        )
    rows = {row.corpus_id: row for row in receipt.corpora}
    if set(rows) != set(FIXED_CORPORA):
        raise SuiteAttemptError("production finalization omits a fixed corpus")
    shared: set[tuple[str, str, str]] = set()
    for corpus_id in FIXED_CORPORA:
        capability = closures[corpus_id]
        binding = capability.binding
        if binding != rows[corpus_id].closure_binding:
            raise SuiteAttemptError(
                f"{corpus_id} guarded closure differs from the finalization receipt"
            )
        if (
            binding.manifest_sha256 != manifest_digest
            or binding.provisional_closure_tree_sha256 != receipt.provisional_closure_tree_sha256
            or binding.instantiated_closure_tree_sha256 != receipt.instantiated_closure_tree_sha256
        ):
            raise SuiteAttemptError(
                f"{corpus_id} registered manifest and closure instantiation differ"
            )
        shared.add(
            (
                binding.manifest_sha256,
                binding.provisional_closure_tree_sha256,
                binding.instantiated_closure_tree_sha256,
            )
        )
    if len(shared) != 1:
        raise SuiteAttemptError("five corpus authorities do not share one exact instantiation")
    return _AdmittedProductionFinalization(
        receipt=receipt,
        receipt_path=path,
        receipt_file_sha256=receipt_file_sha256,
        closures=closures,
    )


def _output_transfer_receipt_path(namespace: Path) -> Path:
    return namespace.parent / f"{namespace.name}.output-transfer.json"


def _output_transfer_staging_path(namespace: Path) -> Path:
    return namespace.parent / f".{namespace.name}.online-transfer"


def _output_transfer_lock_path(namespace: Path) -> Path:
    return namespace.parent / f".{namespace.name}.online-transfer.lock"


@contextmanager
def _output_transfer_lock(namespace: Path) -> Iterator[None]:
    """Serialize recovery of the one derived post-scientific transfer."""

    path = _output_transfer_lock_path(namespace)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_descriptor: int | None = None
    descriptor: int | None = None
    try:
        parent_descriptor = _open_transfer_directory(
            path.parent,
            label="suite output-transfer lock parent",
            private=False,
        )
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
    except OSError as exc:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise SuiteAttemptError(f"cannot open suite output-transfer lock: {exc}") from exc
    except BaseException:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise
    assert descriptor is not None and parent_descriptor is not None
    acquired = False
    try:
        try:
            metadata = os.fstat(descriptor)
            path_metadata = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise SuiteAttemptError(f"cannot validate suite output-transfer lock: {exc}") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or _directory_identity(metadata) != _directory_identity(path_metadata)
        ):
            raise SuiteAttemptError(
                "suite output-transfer lock must be a private singly linked regular file"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuiteAttemptError("suite output transfer already has a live worker") from exc
        except OSError as exc:
            raise SuiteAttemptError(f"cannot acquire suite output-transfer lock: {exc}") from exc
        acquired = True
        yield
        try:
            metadata_after = os.fstat(descriptor)
            path_metadata_after = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise SuiteAttemptError(
                f"suite output-transfer lock changed while held: {exc}"
            ) from exc
        if _stable_file_signature(metadata) != _stable_file_signature(
            metadata_after
        ) or _directory_identity(metadata_after) != _directory_identity(path_metadata_after):
            raise SuiteAttemptError("suite output-transfer lock changed while held")
    finally:
        if acquired:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(descriptor)
        finally:
            os.close(parent_descriptor)


def open_suite_attempt(
    manifest: Mapping[str, Any],
    *,
    verified_protocol_registration: VerifiedC1ProtocolRegistration,
    production_finalization_receipt_path: str | Path,
    verified_production_closures: Mapping[str, VerifiedProductionRunClosure],
    run_receipt_path: str | Path,
    preflight_contract_paths: Mapping[str, str | Path],
    runtime_preflight_receipt_paths: Mapping[str, str | Path],
    runtime_plan_transition_paths: Mapping[str, str | Path],
    registered_plan_instantiation_paths: Mapping[str, str | Path],
    sealed_launch_contract_paths: Mapping[str, str | Path],
    execution_artifacts: Mapping[str, str],
    attestation_descriptor: SuiteAttestationDescriptor,
) -> Path:
    """Open the C1-derived namespace after typed two-field plan verification."""

    try:
        validate_study_manifest(manifest, require_frozen=True)
    except ValueError as exc:
        raise SuiteAttemptError(f"invalid frozen study manifest: {exc}") from exc
    digest = manifest_sha256(manifest)
    if not isinstance(verified_protocol_registration, VerifiedC1ProtocolRegistration):
        raise SuiteAttemptError("OPENED requires the guarded public C1 registration")
    if verified_protocol_registration.manifest_sha256 != digest:
        raise SuiteAttemptError("OPENED manifest differs from the verified public C1 bytes")
    registered_execution_artifacts = _admit_registered_execution_artifacts(
        manifest,
        execution_artifacts,
    )
    if not isinstance(attestation_descriptor, SuiteAttestationDescriptor):
        raise SuiteAttemptError("attestation_descriptor must use the closed suite schema")

    admitted = _admit_production_finalization(
        manifest_digest=digest,
        verified_registration=verified_protocol_registration,
        finalization_receipt_path=production_finalization_receipt_path,
        verified_closures=verified_production_closures,
    )
    finalization = admitted.receipt
    finalized_workload_specs = {}
    for corpus_id in FIXED_CORPORA:
        binding = admitted.closures[corpus_id].binding
        workload_path = Path(binding.closure_source) / binding.workload_spec_relative_path
        try:
            workload_spec = load_production_corpus_workload_spec(
                workload_path,
                expected_file_sha256=binding.workload_spec_file_sha256,
            )
        except ProductionCorpusRunError as exc:
            raise SuiteAttemptError(
                f"{corpus_id} finalized workload specification cannot be reopened: {exc}"
            ) from exc
        if (
            workload_spec.corpus_id != corpus_id
            or workload_spec.online_execution_plan_sha256
            != registered_execution_artifacts[corpus_id]
        ):
            raise SuiteAttemptError(
                f"{corpus_id} finalized workload differs from the frozen C1 execution"
            )
        finalized_workload_specs[corpus_id] = workload_spec
    registration_path = verified_protocol_registration.registration_receipt_path
    registry_path = verified_protocol_registration.registration_record_path
    run_path = Path(run_receipt_path)
    preflight_paths = _fixed_corpus_paths("preflight_contract_paths", preflight_contract_paths)
    preflight_receipt_paths = _fixed_corpus_paths(
        "runtime_preflight_receipt_paths",
        runtime_preflight_receipt_paths,
    )
    transition_paths = _fixed_corpus_paths(
        "runtime_plan_transition_paths",
        runtime_plan_transition_paths,
    )
    instantiation_paths = _fixed_corpus_paths(
        "registered_plan_instantiation_paths",
        registered_plan_instantiation_paths,
    )
    sealed_contract_paths = _fixed_corpus_paths(
        "sealed_launch_contract_paths",
        sealed_launch_contract_paths,
    )
    if not all(path.is_absolute() for path in (registration_path, registry_path, run_path)):
        raise SuiteAttemptError("OPENED evidence paths must be absolute")
    registration = verified_protocol_registration.receipt
    registry = verified_protocol_registration.record
    run = load_sealed_run_receipt(run_path)
    if (
        registration.manifest_sha256 != digest
        or registry.manifest_sha256 != digest
        or run.manifest_sha256 != digest
    ):
        raise SuiteAttemptError("OPENED evidence belongs to another frozen manifest")
    if (
        registry.record_sha256 != registration.registry_record_sha256
        or run.protocol_registration_receipt_sha256 != registration.receipt_sha256
        or run.protocol_registration_receipt_uri != registration_path.as_uri()
        or run.protocol_registration_record_uri != registry_path.as_uri()
    ):
        raise SuiteAttemptError("protocol registration and sealed run bindings differ")
    if (
        attestation_descriptor.expected_signer_digest != run.code_commit
        or run.code_commit != finalization.c0_commit
        or run.runner_image != manifest["sealed_execution"]["runner_image"]
    ):
        raise SuiteAttemptError("signing-workflow digest differs from the sealed run identity")
    ordered_corpora = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
    plan_template_pins = _corpus_manifest_artifact_sha256(
        manifest,
        role="runtime-attestation-plan-template",
    )
    runtime_plan_bindings: list[CorpusRuntimePlanBinding] = []
    for corpus_id in ordered_corpora:
        try:
            preflight = load_preflight_launch_contract(preflight_paths[corpus_id])
            preflight_receipt = load_runtime_preflight_receipt(preflight_receipt_paths[corpus_id])
            transition = load_runtime_plan_transition(transition_paths[corpus_id])
            instantiation = load_registered_plan_instantiation(instantiation_paths[corpus_id])
            sealed = load_sealed_launch_contract(sealed_contract_paths[corpus_id])
            plan = verify_sealed_transition(
                sealed,
                preflight,
                preflight_receipt,
                transition,
                instantiation,
                admitted.closures[corpus_id].binding,
            )
        except (RuntimeAttestationError, SealedContainerLauncherError) as exc:
            raise SuiteAttemptError(
                f"{corpus_id} registered two-field plan instantiation failed: {exc}"
            ) from exc
        plan_path = (
            Path(sealed.geometry.control_mount.source)
            / instantiation.instantiated_plan_relative_path
        )
        if plan.manifest_sha256 != digest:
            raise SuiteAttemptError("OPENED runtime plan belongs to another frozen manifest")
        if (
            plan.code_commit != run.code_commit
            or plan.oci_image_digest != run.runner_image
            or plan.runner_identity != run.runner_identity
            or plan.workload_sha256 != finalized_workload_specs[corpus_id].file_sha256
        ):
            raise SuiteAttemptError("runtime plan differs from the sealed run identity")
        if (
            transition.final_plan_template_file_sha256 != plan_template_pins[corpus_id]
            or instantiation.manifest_sha256 != digest
            or instantiation.production_run_closure_binding_receipt_sha256
            != admitted.closures[corpus_id].binding.receipt_sha256
            or sealed.manifest_sha256 != digest
        ):
            raise SuiteAttemptError(
                f"{corpus_id} plan does not instantiate the registered manifest and closure"
            )
        runtime_plan_bindings.append(
            CorpusRuntimePlanBinding(
                corpus_id=corpus_id,
                plan_sha256=plan.plan_sha256,
                file_sha256=digest_regular_file(
                    plan_path,
                    label=f"{corpus_id} runtime attestation plan",
                ),
                production_run_closure_binding_receipt_sha256=(
                    admitted.closures[corpus_id].binding.receipt_sha256
                ),
                registered_plan_instantiation_receipt_sha256=(instantiation.receipt_sha256),
                registered_plan_instantiation_file_sha256=digest_regular_file(
                    instantiation_paths[corpus_id],
                    label=f"{corpus_id} registered plan instantiation",
                ),
                sealed_launch_contract_uri=sealed_contract_paths[corpus_id].as_uri(),
                sealed_launch_contract_sha256=sealed.contract_sha256,
                sealed_launch_contract_file_sha256=digest_regular_file(
                    sealed_contract_paths[corpus_id],
                    label=f"{corpus_id} sealed launch contract",
                ),
            )
        )
    descriptor_file_sha256 = _sha256(attestation_descriptor.canonical_bytes() + b"\n")
    if descriptor_file_sha256 != _singleton_manifest_artifact_sha256(
        manifest,
        role="suite-attestation-descriptor",
    ):
        raise SuiteAttemptError("suite attestation descriptor differs from its C1 artifact pin")

    namespace = Path(finalization.canonical_suite_namespace)
    base = namespace.parent
    _secure_parent(base)
    if namespace != suite_namespace(base, digest):
        raise SuiteAttemptError("canonical suite namespace is not verified-C1-derived")
    transfer_receipt_path = _output_transfer_receipt_path(namespace)
    transfer_staging_path = _output_transfer_staging_path(namespace)
    if os.path.lexists(transfer_receipt_path) or os.path.lexists(transfer_staging_path):
        raise SuiteAttemptError("suite transfer evidence already exists; replay is forbidden")
    staging_online = Path(finalization.pre_c1_output_staging_root) / "online"
    bindings = SuiteOpenBindings(
        protocol_registration_receipt_sha256=registration.receipt_sha256,
        protocol_registration_receipt_file_sha256=digest_regular_file(
            registration_path,
            label="protocol registration receipt",
        ),
        protocol_registry_record_sha256=registry.record_sha256,
        registered_at_utc=registration.registered_at_utc,
        run_receipt_file_sha256=digest_regular_file(run_path, label="sealed run receipt"),
        run_started_at_utc=run.started_at_utc,
        code_commit=run.code_commit,
        runner_image=run.runner_image,
        attestation_descriptor_sha256=attestation_descriptor.descriptor_sha256,
        production_finalization_receipt_uri=admitted.receipt_path.as_uri(),
        production_finalization_receipt_file_sha256=admitted.receipt_file_sha256,
        production_finalization_request_sha256=finalization.finalization_request_sha256,
        provisional_closure_tree_sha256=finalization.provisional_closure_tree_sha256,
        instantiated_closure_tree_sha256=finalization.instantiated_closure_tree_sha256,
        runtime_attestation_plans=tuple(runtime_plan_bindings),
        execution_artifacts=tuple(
            CorpusDigest(corpus_id, registered_execution_artifacts[corpus_id])
            for corpus_id in ordered_corpora
        ),
        staging_namespaces=tuple(
            CorpusNamespace(corpus_id, (staging_online / corpus_id).as_uri())
            for corpus_id in ordered_corpora
        ),
        output_namespaces=tuple(
            CorpusNamespace(corpus_id, (namespace / "online" / corpus_id).as_uri())
            for corpus_id in ordered_corpora
        ),
    )
    record = SuiteStateRecord(
        suite_attempt_id=suite_attempt_id(digest),
        manifest_sha256=digest,
        run_receipt_sha256=run.binding_sha256,
        namespace_uri=namespace.as_uri(),
        sequence=0,
        state="OPENED",
        previous_state_record_sha256=None,
        payload=bindings,
    )
    try:
        os.mkdir(namespace, mode=0o700)
        os.mkdir(namespace / "online", mode=0o700)
    except OSError as exc:
        raise SuiteAttemptError(
            "suite attempt namespace already exists or cannot be created; rerun is forbidden"
        ) from exc

    descriptor_target = _descriptor_path(namespace)
    _write_once(
        attestation_descriptor.canonical_bytes() + b"\n",
        descriptor_target,
        label="suite attestation descriptor",
    )
    _write_once(
        record.canonical_bytes() + b"\n",
        _state_path(namespace, 0),
        label="OPENED suite state",
    )
    return namespace


def _pin_by_role(result: object, role: str) -> object:
    matches = [row for row in result.outputs if row.role == role]
    if len(matches) != 1:
        raise SuiteAttemptError(f"online result does not bind exactly one {role!r} output")
    return matches[0]


def _verify_audit_bytes(
    encoded: bytes,
    *,
    expected_count: int,
    expected_head: str,
) -> tuple[str, ...]:
    if not encoded.endswith(b"\n"):
        raise SuiteAttemptError("audit chain must end with exactly one line terminator")
    lines = encoded[:-1].split(b"\n")
    if len(lines) != expected_count or any(not line for line in lines):
        raise SuiteAttemptError("audit chain record count differs from the result receipt")
    previous = GENESIS_RECORD_SHA256
    observed: list[str] = []
    for sequence, line in enumerate(lines):
        payload = _parse_object(line, label=f"audit record {sequence}")
        if payload.get("sequence") != sequence:
            raise SuiteAttemptError("audit chain sequence is not contiguous")
        if payload.get("previous_record_sha256") != previous:
            raise SuiteAttemptError("audit chain predecessor binding differs")
        record_digest = _digest("audit record_sha256", payload.get("record_sha256"))
        hash_payload = dict(payload)
        del hash_payload["record_sha256"]
        if line != _canonical_utf8_bytes(payload):
            raise SuiteAttemptError("audit chain record bytes are not canonical")
        if _sha256(_canonical_utf8_bytes(hash_payload)) != record_digest:
            raise SuiteAttemptError("audit chain record self-hash differs")
        observed.append(record_digest)
        previous = record_digest
    if previous != expected_head:
        raise SuiteAttemptError("audit chain head differs from the result receipt")
    return tuple(observed)


def _expected_online_names(
    attempt_path: Path,
    result_path: Path,
    result: object,
) -> set[str]:
    return {
        attempt_path.name,
        result_path.name,
        PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
        RUNTIME_ATTESTATION_RECEIPT_FILENAME,
        RUNTIME_INVOCATION_MARKER_FILENAME,
        *(row.filename for row in result.outputs),
    }


def _closure_transfer_binding(
    *,
    role: str,
    path: Path,
    expected_file_sha256: str,
    expected_byte_count: int | None = None,
) -> TransferFileBinding:
    """Freeze one already verified closure file as a transfer triple."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise SuiteAttemptError(f"cannot inspect {role} closure file: {exc}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise SuiteAttemptError(
            f"{role} closure file must be runner-owned, not group-or-other-writable, "
            "and singly linked"
        )
    try:
        observed_sha256 = digest_regular_file(path, label=f"{role} closure file")
        after = path.lstat()
    except (OSError, ArtifactIntegrityError) as exc:
        raise SuiteAttemptError(f"cannot freeze {role} closure file: {exc}") from exc
    if (
        _stable_file_signature(before) != _stable_file_signature(after)
        or observed_sha256 != expected_file_sha256
        or (expected_byte_count is not None and before.st_size != expected_byte_count)
    ):
        raise SuiteAttemptError(f"{role} closure file changed or differs from its typed pin")
    return TransferFileBinding(
        role=role,
        relative_path=path.name,
        file_sha256=observed_sha256,
        byte_count=before.st_size,
    )


def _load_online_corpus_closure(
    root: Path,
    *,
    corpus_id: str,
    opened: SuiteStateRecord,
    runtime_attestation_plan_sha256: str,
    runtime_attestation_plan_file_sha256: str,
    runtime_attestation_receipt_sha256: str,
    runtime_attestation_receipt_file_sha256: str,
    runtime_invocation_marker_sha256: str,
    runtime_invocation_marker_file_sha256: str,
    production_command_attempt_sha256: str,
    production_command_attempt_file_sha256: str,
    sealed_launch_receipt_uri: str,
    sealed_launch_copy_output_uri: str,
    sealed_launch_contract_sha256: str,
    sealed_launch_receipt_sha256: str,
    sealed_launch_receipt_file_sha256: str,
    sealed_launch_evidence_inventory_sha256: str,
    sealed_launch_output_tree_sha256: str,
) -> OnlineCorpusClosure:
    if not isinstance(opened.payload, SuiteOpenBindings):
        raise SuiteAttemptError("suite genesis record is not OPENED")
    staging_by_corpus = {
        row.corpus_id: Path(unquote(urlsplit(row.output_uri).path))
        for row in opened.payload.staging_namespaces
    }
    canonical_by_corpus = {
        row.corpus_id: Path(unquote(urlsplit(row.output_uri).path))
        for row in opened.payload.output_namespaces
    }
    expected_root = staging_by_corpus[corpus_id]
    canonical_root = canonical_by_corpus[corpus_id]
    if root != expected_root:
        raise SuiteAttemptError("online corpus output root is outside its fixed staging namespace")
    _namespace_path(root)
    manifest_digest = opened.manifest_sha256
    attempt_path = sealed_online_attempt_path(root, manifest_digest)
    result_path = sealed_online_result_path(root, manifest_digest)
    attempt = load_sealed_online_attempt_receipt(attempt_path)
    result = load_sealed_online_result_receipt(result_path)
    expected_execution = {row.corpus_id: row.sha256 for row in opened.payload.execution_artifacts}[
        corpus_id
    ]
    for name, observed, expected in (
        ("attempt manifest", attempt.manifest_sha256, manifest_digest),
        ("attempt run", attempt.run_receipt_sha256, opened.run_receipt_sha256),
        ("attempt execution", attempt.execution_artifact_sha256, expected_execution),
        (
            "attempt runtime attestation plan",
            attempt.runtime_attestation_plan_sha256,
            runtime_attestation_plan_sha256,
        ),
        (
            "attempt runtime attestation receipt",
            attempt.runtime_attestation_receipt_sha256,
            runtime_attestation_receipt_sha256,
        ),
        ("attempt output root", attempt.result_directory_uri, root.as_uri()),
        ("result manifest", result.manifest_sha256, manifest_digest),
        ("result run", result.run_receipt_sha256, opened.run_receipt_sha256),
        ("result execution", result.execution_artifact_sha256, expected_execution),
        ("result attempt", result.attempt_receipt_sha256, attempt.receipt_sha256),
    ):
        if observed != expected:
            raise SuiteAttemptError(f"{name} binding differs from OPENED")
    observed_names = {path.name for path in root.iterdir()}
    expected_names = _expected_online_names(attempt_path, result_path, result)
    if observed_names != expected_names:
        raise SuiteAttemptError(
            "online corpus directory must contain exactly the runtime receipt, invocation "
            "marker, production command attempt, sealed attempt, sealed result, prediction, "
            "panel, audit, cache, and order files"
        )
    verify_sealed_online_outputs(result, output_root=root)

    pins = {row.role: row for row in result.outputs}
    prediction_path = root / pins["predictions"].filename
    panel_path = root / pins["action-panel"].filename
    admission_path = root / pins["action-panel-admission"].filename
    audit_path = root / pins["audit-chain"].filename
    cache_path = root / pins["cache-preparation"].filename
    order_path = root / pins["execution-order"].filename
    predictions = load_prediction_artifact(prediction_path)
    panel = load_action_panel_artifact(panel_path)
    admission = load_action_panel_admission_receipt(admission_path)
    cache = load_cache_preparation_receipt(cache_path)
    order = load_execution_order_receipt(order_path)
    for label, value in (
        ("prediction corpus", predictions.corpus),
        ("panel corpus", panel.corpus),
        ("panel admission corpus", admission.corpus),
    ):
        if value != corpus_id:
            raise SuiteAttemptError(f"{label} differs from its fixed corpus directory")
    for label, value in (
        ("prediction manifest", predictions.manifest_sha256),
        ("panel manifest", panel.manifest_sha256),
        ("panel admission manifest", admission.manifest_sha256),
    ):
        if value != manifest_digest:
            raise SuiteAttemptError(f"{label} differs from OPENED")
    for label, value in (
        ("prediction run", predictions.run_receipt_sha256),
        ("panel run", panel.run_receipt_sha256),
        ("panel admission run", admission.run_receipt_sha256),
        ("cache run", cache.run_receipt_sha256),
        ("order run", order.run_receipt_sha256),
    ):
        if value != opened.run_receipt_sha256:
            raise SuiteAttemptError(f"{label} differs from OPENED")
    for label, value in (
        ("prediction execution", predictions.execution_artifact_sha256),
        ("panel execution", panel.execution_artifact_sha256),
        ("panel admission execution", admission.execution_artifact_sha256),
        ("cache execution", cache.execution_artifact_sha256),
        ("order execution", order.execution_artifact_sha256),
    ):
        if value != expected_execution:
            raise SuiteAttemptError(f"{label} differs from OPENED")
    if predictions.stage != "sealed" or panel.stage != "sealed":
        raise SuiteAttemptError("online corpus artifacts must use the sealed stage")
    if (
        admission.action_panel_artifact_sha256 != panel.artifact_sha256
        or admission.audit_head_sha256 != result.audit_head_sha256
        or admission.audit_chain_length != result.audit_record_count
        or order.cache_preparation_receipt_sha256 != cache.receipt_sha256
    ):
        raise SuiteAttemptError("panel, audit, cache, or execution-order links differ")

    audit_bytes = _secure_bytes(
        audit_path,
        label=f"{corpus_id} audit chain",
        max_bytes=pins["audit-chain"].byte_count,
    )
    audit_hashes = _verify_audit_bytes(
        audit_bytes,
        expected_count=result.audit_record_count,
        expected_head=result.audit_head_sha256,
    )
    if audit_hashes != admission.audit_record_sha256s:
        raise SuiteAttemptError("panel admission receipt binds another audit chain")
    semantic_bindings = (
        ("predictions", predictions.artifact_sha256),
        ("action-panel", panel.artifact_sha256),
        ("action-panel-admission", admission.receipt_sha256),
        ("audit-chain", result.audit_head_sha256),
        ("cache-preparation", cache.receipt_sha256),
        ("execution-order", order.receipt_sha256),
    )
    for role, semantic_digest in semantic_bindings:
        if pins[role].semantic_sha256 != semantic_digest:
            raise SuiteAttemptError(f"online result has mismatched {role} semantic digest")
    if {path.name for path in root.iterdir()} != expected_names:
        raise SuiteAttemptError("online corpus directory changed during closure verification")

    attempt_file_sha256 = digest_regular_file(attempt_path, label="online attempt")
    result_file_sha256 = digest_regular_file(result_path, label="online result")
    transfer_files = [
        _closure_transfer_binding(
            role="runtime-attestation-receipt",
            path=root / RUNTIME_ATTESTATION_RECEIPT_FILENAME,
            expected_file_sha256=runtime_attestation_receipt_file_sha256,
        ),
        _closure_transfer_binding(
            role="runtime-invocation-marker",
            path=root / RUNTIME_INVOCATION_MARKER_FILENAME,
            expected_file_sha256=runtime_invocation_marker_file_sha256,
        ),
        _closure_transfer_binding(
            role="production-command-attempt",
            path=root / PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
            expected_file_sha256=production_command_attempt_file_sha256,
        ),
        _closure_transfer_binding(
            role="sealed-online-attempt",
            path=attempt_path,
            expected_file_sha256=attempt_file_sha256,
        ),
        _closure_transfer_binding(
            role="sealed-online-result",
            path=result_path,
            expected_file_sha256=result_file_sha256,
        ),
    ]
    transfer_files.extend(
        _closure_transfer_binding(
            role=pin.role,
            path=root / pin.filename,
            expected_file_sha256=pin.file_sha256,
            expected_byte_count=pin.byte_count,
        )
        for pin in result.outputs
    )
    frozen_transfer_files = tuple(
        sorted(transfer_files, key=lambda item: item.relative_path.encode("utf-8"))
    )
    if {item.relative_path for item in frozen_transfer_files} != expected_names:
        raise SuiteAttemptError("online closure transfer mapping differs from its exact inventory")

    return OnlineCorpusClosure(
        corpus_id=corpus_id,
        staging_output_uri=root.as_uri(),
        output_uri=canonical_root.as_uri(),
        execution_artifact_sha256=expected_execution,
        runtime_attestation_plan_sha256=runtime_attestation_plan_sha256,
        runtime_attestation_plan_file_sha256=runtime_attestation_plan_file_sha256,
        runtime_attestation_receipt_sha256=runtime_attestation_receipt_sha256,
        runtime_attestation_receipt_file_sha256=runtime_attestation_receipt_file_sha256,
        runtime_invocation_marker_sha256=runtime_invocation_marker_sha256,
        runtime_invocation_marker_file_sha256=runtime_invocation_marker_file_sha256,
        production_command_attempt_sha256=production_command_attempt_sha256,
        production_command_attempt_file_sha256=production_command_attempt_file_sha256,
        sealed_launch_receipt_uri=sealed_launch_receipt_uri,
        sealed_launch_copy_output_uri=sealed_launch_copy_output_uri,
        sealed_launch_contract_sha256=sealed_launch_contract_sha256,
        sealed_launch_receipt_sha256=sealed_launch_receipt_sha256,
        sealed_launch_receipt_file_sha256=sealed_launch_receipt_file_sha256,
        sealed_launch_evidence_inventory_sha256=(sealed_launch_evidence_inventory_sha256),
        sealed_launch_output_tree_sha256=sealed_launch_output_tree_sha256,
        attempt_receipt_sha256=attempt.receipt_sha256,
        attempt_file_sha256=attempt_file_sha256,
        result_receipt_sha256=result.receipt_sha256,
        result_file_sha256=result_file_sha256,
        prediction_artifact_sha256=predictions.artifact_sha256,
        prediction_file_sha256=pins["predictions"].file_sha256,
        action_panel_artifact_sha256=panel.artifact_sha256,
        action_panel_file_sha256=pins["action-panel"].file_sha256,
        action_panel_admission_receipt_sha256=admission.receipt_sha256,
        action_panel_admission_file_sha256=pins["action-panel-admission"].file_sha256,
        audit_head_sha256=result.audit_head_sha256,
        audit_file_sha256=pins["audit-chain"].file_sha256,
        audit_record_count=result.audit_record_count,
        cache_preparation_receipt_sha256=cache.receipt_sha256,
        cache_preparation_file_sha256=pins["cache-preparation"].file_sha256,
        execution_order_receipt_sha256=order.receipt_sha256,
        execution_order_file_sha256=pins["execution-order"].file_sha256,
        transfer_files=frozen_transfer_files,
    )


def _stable_file_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_transfer_directory(
    path: Path,
    *,
    label: str,
    private: bool,
) -> int:
    """Open and pin one runner-owned real directory for transfer I/O."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SuiteAttemptError(f"cannot open {label}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        path_metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or not stat.S_ISDIR(path_metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino)
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or (mode != 0o700 if private else bool(mode & 0o022))
        ):
            qualifier = "private " if private else "runner-controlled "
            raise SuiteAttemptError(f"{label} must be a {qualifier}real directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _corpus_transfer_files(
    root: Path,
    snapshot: DirectoryDigest,
    *,
    expected: tuple[TransferFileBinding, ...],
) -> tuple[TransferFileBinding, ...]:
    if snapshot.directory_count != 0 or snapshot.file_count != 11:
        raise SuiteAttemptError("each staged corpus must contain exactly eleven flat regular files")
    expected_by_path = {item.relative_path: item for item in expected}
    if (
        len(expected_by_path) != len(_TRANSFER_ROLE_FILE_FIELD)
        or tuple(expected_by_path) != snapshot.entries
    ):
        raise SuiteAttemptError("staged corpus filenames differ from the verified online closure")
    root_descriptor = _open_transfer_directory(
        root,
        label="staged corpus output root",
        private=False,
    )
    root_identity = _directory_identity(os.fstat(root_descriptor))
    rows: list[TransferFileBinding] = []
    try:
        for relative_path in snapshot.entries:
            if Path(relative_path).name != relative_path:
                raise SuiteAttemptError("staged corpus output cannot contain nested paths")
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    relative_path,
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_descriptor,
                )
                before = os.fstat(descriptor)
                path_before = os.stat(
                    relative_path,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or _directory_identity(before) != _directory_identity(path_before)
                    or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
                    or stat.S_IMODE(before.st_mode) & 0o022
                ):
                    raise SuiteAttemptError(
                        "staged corpus members must be runner-owned, not "
                        "group-or-other-writable, singly linked regular files"
                    )
                digest = hashlib.sha256()
                byte_count = 0
                while True:
                    chunk = os.read(descriptor, _TRANSFER_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
                    byte_count += len(chunk)
                after = os.fstat(descriptor)
                path_after = os.stat(
                    relative_path,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                if (
                    _stable_file_signature(before) != _stable_file_signature(after)
                    or _directory_identity(after) != _directory_identity(path_after)
                    or byte_count != before.st_size
                ):
                    raise SuiteAttemptError("staged corpus member changed during admission")
                rows.append(
                    TransferFileBinding(
                        role=expected_by_path[relative_path].role,
                        relative_path=relative_path,
                        file_sha256=digest.hexdigest(),
                        byte_count=byte_count,
                    )
                )
            except OSError as exc:
                raise SuiteAttemptError(f"cannot admit staged corpus member: {exc}") from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        try:
            root_after = root.lstat()
        except OSError as exc:
            raise SuiteAttemptError(f"cannot re-inspect staged corpus output root: {exc}") from exc
        if root_identity != _directory_identity(root_after):
            raise SuiteAttemptError("staged corpus output root changed during admission")
    finally:
        os.close(root_descriptor)
    return tuple(rows)


def _copy_transfer_file(
    source: Path,
    target: Path,
    expected: TransferFileBinding,
) -> None:
    """Copy or resume one exact file through pinned parent descriptors."""

    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    create_flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    resume_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_parent_descriptor: int | None = None
    target_parent_descriptor: int | None = None
    source_descriptor: int | None = None
    target_descriptor: int | None = None
    try:
        source_parent_descriptor = _open_transfer_directory(
            source.parent,
            label="staged transfer source directory",
            private=False,
        )
        target_parent_descriptor = _open_transfer_directory(
            target.parent,
            label="transfer candidate directory",
            private=True,
        )
        source_descriptor = os.open(
            source.name,
            read_flags,
            dir_fd=source_parent_descriptor,
        )
        before = os.fstat(source_descriptor)
        source_path_before = os.stat(
            source.name,
            dir_fd=source_parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected.byte_count
            or _directory_identity(before) != _directory_identity(source_path_before)
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            raise SuiteAttemptError("staged transfer source is not the admitted regular file")
        try:
            target_descriptor = os.open(
                target.name,
                create_flags,
                0o600,
                dir_fd=target_parent_descriptor,
            )
        except FileExistsError:
            target_descriptor = os.open(
                target.name,
                resume_flags,
                dir_fd=target_parent_descriptor,
            )
        target_before = os.fstat(target_descriptor)
        target_path_before = os.stat(
            target.name,
            dir_fd=target_parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(target_before.st_mode)
            or target_before.st_nlink != 1
            or stat.S_IMODE(target_before.st_mode) != 0o600
            or (hasattr(os, "geteuid") and target_before.st_uid != os.geteuid())
            or target_before.st_size > expected.byte_count
            or _directory_identity(target_before) != _directory_identity(target_path_before)
        ):
            raise SuiteAttemptError(
                "partial transfer target is not a private admissible regular-file prefix"
            )
        retained_prefix = target_before.st_size
        os.lseek(target_descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_descriptor, _TRANSFER_COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            prefix_count = min(len(chunk), max(0, retained_prefix - copied))
            if prefix_count:
                observed = bytearray()
                while len(observed) < prefix_count:
                    part = os.read(target_descriptor, prefix_count - len(observed))
                    if not part:
                        raise SuiteAttemptError(
                            "partial transfer target ended before its admitted size"
                        )
                    observed.extend(part)
                if bytes(observed) != chunk[:prefix_count]:
                    raise SuiteAttemptError("partial transfer target is not an exact source prefix")
            view = memoryview(chunk)[prefix_count:]
            while view:
                written = os.write(target_descriptor, view)
                if written <= 0:  # pragma: no cover - defensive OS boundary
                    raise SuiteAttemptError("output transfer write made no progress")
                view = view[written:]
            copied += len(chunk)
        os.fsync(target_descriptor)
        after = os.fstat(source_descriptor)
        target_metadata = os.fstat(target_descriptor)
        target_path_after = os.stat(
            target.name,
            dir_fd=target_parent_descriptor,
            follow_symlinks=False,
        )
        os.lseek(target_descriptor, 0, os.SEEK_SET)
        target_digest = hashlib.sha256()
        target_byte_count = 0
        while True:
            chunk = os.read(target_descriptor, _TRANSFER_COPY_CHUNK_BYTES)
            if not chunk:
                break
            target_digest.update(chunk)
            target_byte_count += len(chunk)
        target_after_verification = os.fstat(target_descriptor)
        target_path_after_verification = os.stat(
            target.name,
            dir_fd=target_parent_descriptor,
            follow_symlinks=False,
        )
        source_parent_after = source.parent.lstat()
        target_parent_after = target.parent.lstat()
        if (
            _stable_file_signature(before) != _stable_file_signature(after)
            or copied != before.st_size
            or digest.hexdigest() != expected.file_sha256
            or not stat.S_ISREG(target_metadata.st_mode)
            or target_metadata.st_nlink != 1
            or stat.S_IMODE(target_metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and target_metadata.st_uid != os.geteuid())
            or target_metadata.st_size != expected.byte_count
            or target_byte_count != expected.byte_count
            or target_digest.hexdigest() != expected.file_sha256
            or _stable_file_signature(target_metadata)
            != _stable_file_signature(target_after_verification)
            or _directory_identity(target_metadata) != _directory_identity(target_path_after)
            or _directory_identity(target_after_verification)
            != _directory_identity(target_path_after_verification)
            or _directory_identity(os.fstat(source_parent_descriptor))
            != _directory_identity(source_parent_after)
            or _directory_identity(os.fstat(target_parent_descriptor))
            != _directory_identity(target_parent_after)
        ):
            raise SuiteAttemptError("staged output changed during exact transfer")
    except OSError as exc:
        raise SuiteAttemptError(f"cannot copy staged output exclusively: {exc}") from exc
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)
        if target_parent_descriptor is not None:
            os.close(target_parent_descriptor)
        if source_parent_descriptor is not None:
            os.close(source_parent_descriptor)


def _atomic_exchange_directories(first: Path, second: Path) -> None:
    first_parent_descriptor: int | None = None
    second_parent_descriptor: int | None = None
    first_descriptor: int | None = None
    second_descriptor: int | None = None
    try:
        first_parent_descriptor = _open_transfer_directory(
            first.parent,
            label="first exchange parent",
            private=False,
        )
        second_parent_descriptor = _open_transfer_directory(
            second.parent,
            label="second exchange parent",
            private=False,
        )
        first_descriptor = _open_transfer_directory(
            first,
            label="canonical online placeholder",
            private=True,
        )
        second_descriptor = _open_transfer_directory(
            second,
            label="completed transfer candidate",
            private=True,
        )
        first_metadata = os.fstat(first_descriptor)
        second_metadata = os.fstat(second_descriptor)
        if first_metadata.st_dev != second_metadata.st_dev:
            raise SuiteAttemptError(
                "atomic output transfer requires two real directories on one filesystem"
            )

        library = ctypes.CDLL(None, use_errno=True)
        first_encoded = os.fsencode(first.name)
        second_encoded = os.fsencode(second.name)
        ctypes.set_errno(0)
        if sys.platform == "darwin":
            function = getattr(library, "renameatx_np", None)
            if function is None:
                raise SuiteAttemptError("macOS atomic directory exchange is unavailable")
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
            result = function(
                first_parent_descriptor,
                first_encoded,
                second_parent_descriptor,
                second_encoded,
                0x00000002,
            )
        elif sys.platform.startswith("linux"):
            function = getattr(library, "renameat2", None)
            if function is None:
                raise SuiteAttemptError("Linux atomic directory exchange is unavailable")
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
            result = function(
                first_parent_descriptor,
                first_encoded,
                second_parent_descriptor,
                second_encoded,
                0x00000002,
            )
        else:
            raise SuiteAttemptError("platform lacks an admitted atomic directory exchange")
        if result != 0:
            raise SuiteAttemptError(
                f"atomic output transfer failed with errno {ctypes.get_errno()}"
            )

        first_after = os.stat(
            first.name,
            dir_fd=first_parent_descriptor,
            follow_symlinks=False,
        )
        second_after = os.stat(
            second.name,
            dir_fd=second_parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(first_after.st_mode)
            or not stat.S_ISDIR(second_after.st_mode)
            or _directory_identity(first_after) != _directory_identity(second_metadata)
            or _directory_identity(second_after) != _directory_identity(first_metadata)
        ):
            raise SuiteAttemptError("atomic output transfer did not exchange the admitted trees")
        os.fsync(first_parent_descriptor)
        if _directory_identity(os.fstat(first_parent_descriptor)) != _directory_identity(
            os.fstat(second_parent_descriptor)
        ):
            os.fsync(second_parent_descriptor)
    except OSError as exc:
        raise SuiteAttemptError(f"cannot exchange suite output directories: {exc}") from exc
    finally:
        if second_descriptor is not None:
            os.close(second_descriptor)
        if first_descriptor is not None:
            os.close(first_descriptor)
        if second_parent_descriptor is not None:
            os.close(second_parent_descriptor)
        if first_parent_descriptor is not None:
            os.close(first_parent_descriptor)


def _require_tree(
    root: Path,
    *,
    expected_sha256: str,
    expected_entries: tuple[str, ...],
    label: str,
) -> DirectoryDigest:
    try:
        observed = digest_directory_tree(root)
    except ArtifactIntegrityError as exc:
        raise SuiteAttemptError(f"cannot verify {label}: {exc}") from exc
    if observed.sha256 != expected_sha256 or observed.entries != expected_entries:
        raise SuiteAttemptError(f"{label} changed or has the wrong inventory")
    return observed


def _require_private_transfer_directory(path: Path, *, label: str) -> None:
    descriptor = _open_transfer_directory(path, label=label, private=True)
    os.close(descriptor)


def _require_controlled_transfer_directory(path: Path, *, label: str) -> None:
    descriptor = _open_transfer_directory(path, label=label, private=False)
    os.close(descriptor)


def _fsync_transfer_directory(path: Path, *, label: str, private: bool) -> None:
    descriptor = _open_transfer_directory(path, label=label, private=private)
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise SuiteAttemptError(f"cannot durably sync {label}: {exc}") from exc
    finally:
        os.close(descriptor)


def _directory_members(path: Path, *, label: str) -> set[str]:
    try:
        return {child.name for child in path.iterdir()}
    except OSError as exc:
        raise SuiteAttemptError(f"cannot scan {label}: {exc}") from exc


def _resume_transfer_candidate(
    retained: Path,
    *,
    rows: tuple[OnlineCorpusClosure, ...],
    staging_roots: Mapping[str, Path],
    transfer_files: Mapping[str, tuple[TransferFileBinding, ...]],
    expected_snapshot: DirectoryDigest,
) -> None:
    """Complete only missing bytes in an exact crash-retained candidate."""

    if not os.path.lexists(retained):
        try:
            os.mkdir(retained, mode=0o700)
        except OSError as exc:
            raise SuiteAttemptError(f"cannot create suite transfer candidate: {exc}") from exc
    _require_private_transfer_directory(retained, label="suite transfer candidate")
    corpus_names = {row.corpus_id for row in rows}
    observed_corpora = _directory_members(retained, label="suite transfer candidate")
    if not observed_corpora.issubset(corpus_names):
        raise SuiteAttemptError("suite transfer candidate contains an undeclared corpus member")
    for row in rows:
        target_root = retained / row.corpus_id
        if not os.path.lexists(target_root):
            try:
                os.mkdir(target_root, mode=0o700)
            except OSError as exc:
                raise SuiteAttemptError(
                    f"cannot create {row.corpus_id} transfer candidate"
                ) from exc
        _require_private_transfer_directory(
            target_root,
            label=f"{row.corpus_id} transfer candidate",
        )
        expected_names = {item.relative_path for item in transfer_files[row.corpus_id]}
        observed_names = _directory_members(
            target_root,
            label=f"{row.corpus_id} transfer candidate",
        )
        if not observed_names.issubset(expected_names):
            raise SuiteAttemptError(
                f"{row.corpus_id} transfer candidate contains an undeclared file"
            )
        for file_row in transfer_files[row.corpus_id]:
            _copy_transfer_file(
                staging_roots[row.corpus_id] / file_row.relative_path,
                target_root / file_row.relative_path,
                file_row,
            )
        _fsync_transfer_directory(
            target_root,
            label=f"{row.corpus_id} transfer candidate",
            private=True,
        )
    _require_tree(
        retained,
        expected_sha256=expected_snapshot.sha256,
        expected_entries=expected_snapshot.entries,
        label="completed online transfer candidate",
    )
    _fsync_transfer_directory(
        retained,
        label="completed online transfer candidate",
        private=True,
    )
    _fsync_transfer_directory(
        retained.parent,
        label="transfer candidate parent",
        private=False,
    )


def _expected_transfer_receipt(
    *,
    opened: SuiteStateRecord,
    payload: SuiteOpenBindings,
    rows: tuple[OnlineCorpusClosure, ...],
    staging_online: Path,
    canonical_online: Path,
    retained_placeholder: Path,
    empty_placeholder: DirectoryDigest,
    source_online_snapshot: DirectoryDigest,
    source_snapshots: Mapping[str, DirectoryDigest],
    transfer_files: Mapping[str, tuple[TransferFileBinding, ...]],
) -> SuiteOutputTransferReceipt:
    corpus_transfers = tuple(
        CorpusOutputTransfer(
            corpus_id=row.corpus_id,
            staging_output_uri=row.staging_output_uri,
            canonical_output_uri=row.output_uri,
            source_tree_sha256=source_snapshots[row.corpus_id].sha256,
            canonical_tree_sha256=source_snapshots[row.corpus_id].sha256,
            entries=source_snapshots[row.corpus_id].entries,
            files=transfer_files[row.corpus_id],
        )
        for row in rows
    )
    return SuiteOutputTransferReceipt(
        suite_attempt_id=opened.suite_attempt_id,
        manifest_sha256=opened.manifest_sha256,
        production_finalization_receipt_file_sha256=(
            payload.production_finalization_receipt_file_sha256
        ),
        staging_online_root_uri=staging_online.as_uri(),
        canonical_online_root_uri=canonical_online.as_uri(),
        retained_empty_placeholder_uri=retained_placeholder.as_uri(),
        empty_placeholder_tree_sha256=empty_placeholder.sha256,
        source_online_tree_sha256=source_online_snapshot.sha256,
        canonical_online_tree_sha256=source_online_snapshot.sha256,
        entries=source_online_snapshot.entries,
        corpora=corpus_transfers,
    )


def _transfer_staged_online_outputs(
    verified_open: VerifiedSuiteState | VerifiedProviderPredecessor,
    closures: tuple[OnlineCorpusClosure, ...],
) -> SuiteOutputTransferReceipt:
    if isinstance(verified_open, VerifiedProviderPredecessor):
        if verified_open.state.state != "RUN_CLAIMED":
            raise SuiteAttemptError("provider suite output transfer requires verified RUN_CLAIMED")
    elif not isinstance(verified_open, (VerifiedSuiteOpened, VerifiedSuiteRunClaimed)):
        raise SuiteAttemptError("suite output transfer requires verified pre-completion state")
    with _output_transfer_lock(verified_open.namespace):
        return _transfer_staged_online_outputs_locked(verified_open, closures)


def _transfer_staged_online_outputs_locked(
    verified_open: VerifiedSuiteState | VerifiedProviderPredecessor,
    closures: tuple[OnlineCorpusClosure, ...],
) -> SuiteOutputTransferReceipt:
    verified_open.assert_current()
    opened = verified_open.records[0]
    payload = opened.payload
    if not isinstance(payload, SuiteOpenBindings):
        raise SuiteAttemptError("verified suite does not begin with OPENED bindings")
    rows = _fixed_corpus_rows(
        "staged online closures",
        closures,
        row_type=OnlineCorpusClosure,
    )
    admitted_staging = {row.corpus_id: row.output_uri for row in payload.staging_namespaces}
    admitted_canonical = {row.corpus_id: row.output_uri for row in payload.output_namespaces}
    if any(
        row.staging_output_uri != admitted_staging[row.corpus_id]
        or row.output_uri != admitted_canonical[row.corpus_id]
        for row in rows
    ):
        raise SuiteAttemptError("staged closure names the wrong corpus output root")
    staging_roots = {
        row.corpus_id: Path(unquote(urlsplit(row.staging_output_uri).path)) for row in rows
    }
    canonical_roots = {row.corpus_id: Path(unquote(urlsplit(row.output_uri).path)) for row in rows}
    staging_online_roots = {path.parent for path in staging_roots.values()}
    canonical_online_roots = {path.parent for path in canonical_roots.values()}
    if len(staging_online_roots) != 1 or len(canonical_online_roots) != 1:
        raise SuiteAttemptError("five corpus transfers do not share fixed online roots")
    staging_online = next(iter(staging_online_roots))
    canonical_online = next(iter(canonical_online_roots))
    if canonical_online != verified_open.namespace / "online":
        raise SuiteAttemptError("canonical online root differs from verified C1 namespace")
    transfer_path = _output_transfer_receipt_path(verified_open.namespace)
    retained_placeholder = _output_transfer_staging_path(verified_open.namespace)
    _require_controlled_transfer_directory(
        staging_online,
        label="staging online root",
    )
    _require_private_transfer_directory(
        canonical_online,
        label="canonical online root",
    )
    for corpus_id, source_root in staging_roots.items():
        _require_controlled_transfer_directory(
            source_root,
            label=f"{corpus_id} staging output root",
        )
    if os.path.lexists(retained_placeholder):
        _require_private_transfer_directory(
            retained_placeholder,
            label="retained transfer tree",
        )
    source_snapshots: dict[str, DirectoryDigest] = {}
    transfer_files: dict[str, tuple[TransferFileBinding, ...]] = {}
    for row in rows:
        source = staging_roots[row.corpus_id]
        try:
            snapshot = digest_directory_tree(source)
        except ArtifactIntegrityError as exc:
            raise SuiteAttemptError(f"cannot verify {row.corpus_id} staged output: {exc}") from exc
        source_snapshots[row.corpus_id] = snapshot
        observed_transfer_files = _corpus_transfer_files(
            source,
            snapshot,
            expected=row.transfer_files,
        )
        if observed_transfer_files != row.transfer_files:
            raise SuiteAttemptError(
                f"{row.corpus_id} staged bytes differ from its verified online closure"
            )
        transfer_files[row.corpus_id] = observed_transfer_files
    try:
        source_online_snapshot = digest_directory_tree(staging_online)
    except ArtifactIntegrityError as exc:
        raise SuiteAttemptError(f"cannot verify staged suite output: {exc}") from exc
    expected_entries = tuple(
        sorted(
            (
                *(row.corpus_id for row in rows),
                *(
                    f"{row.corpus_id}/{entry}"
                    for row in rows
                    for entry in source_snapshots[row.corpus_id].entries
                ),
            ),
            key=lambda item: item.encode("utf-8"),
        )
    )
    if source_online_snapshot.entries != expected_entries:
        raise SuiteAttemptError("staging online root contains an extra or missing corpus member")
    try:
        canonical_snapshot = digest_directory_tree(canonical_online)
        retained_snapshot = (
            digest_directory_tree(retained_placeholder)
            if os.path.lexists(retained_placeholder)
            else None
        )
    except ArtifactIntegrityError as exc:
        raise SuiteAttemptError(f"cannot classify suite transfer recovery state: {exc}") from exc
    canonical_is_empty = not canonical_snapshot.entries
    canonical_is_complete = (
        canonical_snapshot.sha256 == source_online_snapshot.sha256
        and canonical_snapshot.entries == source_online_snapshot.entries
    )
    if not canonical_is_empty and not canonical_is_complete:
        raise SuiteAttemptError("canonical online tree is neither empty nor the exact transfer")

    if canonical_is_complete:
        if retained_snapshot is None or retained_snapshot.entries:
            raise SuiteAttemptError(
                "post-exchange transfer lacks its exact retained empty placeholder"
            )
        for corpus_id, canonical_root in canonical_roots.items():
            _require_private_transfer_directory(
                canonical_root,
                label=f"{corpus_id} canonical output root",
            )
        empty_placeholder = retained_snapshot
    else:
        empty_placeholder = canonical_snapshot
        if os.path.lexists(transfer_path):
            raise SuiteAttemptError("transfer receipt exists before the atomic exchange")
        _resume_transfer_candidate(
            retained_placeholder,
            rows=rows,
            staging_roots=staging_roots,
            transfer_files=transfer_files,
            expected_snapshot=source_online_snapshot,
        )
        _require_tree(
            staging_online,
            expected_sha256=source_online_snapshot.sha256,
            expected_entries=source_online_snapshot.entries,
            label="staging online source after recoverable copy",
        )
        _atomic_exchange_directories(canonical_online, retained_placeholder)
        _require_tree(
            staging_online,
            expected_sha256=source_online_snapshot.sha256,
            expected_entries=source_online_snapshot.entries,
            label="preserved staging online evidence",
        )
        _require_tree(
            canonical_online,
            expected_sha256=source_online_snapshot.sha256,
            expected_entries=source_online_snapshot.entries,
            label="canonical online transfer",
        )
        _require_tree(
            retained_placeholder,
            expected_sha256=empty_placeholder.sha256,
            expected_entries=(),
            label="retained empty canonical placeholder",
        )

    receipt = _expected_transfer_receipt(
        opened=opened,
        payload=payload,
        rows=rows,
        staging_online=staging_online,
        canonical_online=canonical_online,
        retained_placeholder=retained_placeholder,
        empty_placeholder=empty_placeholder,
        source_online_snapshot=source_online_snapshot,
        source_snapshots=source_snapshots,
        transfer_files=transfer_files,
    )
    if os.path.lexists(transfer_path):
        loaded = _load_private_suite_output_transfer_receipt(transfer_path)
        if loaded != receipt:
            raise SuiteAttemptError("persisted suite output transfer receipt differs")
    else:
        _write_once(
            receipt.canonical_file_bytes(),
            transfer_path,
            label="suite output transfer receipt",
        )
    loaded = _load_private_suite_output_transfer_receipt(transfer_path)
    if loaded != receipt:
        raise SuiteAttemptError("persisted suite output transfer receipt differs")
    _require_tree(
        staging_online,
        expected_sha256=receipt.source_online_tree_sha256,
        expected_entries=receipt.entries,
        label="staging evidence after transfer receipt publication",
    )
    _require_tree(
        canonical_online,
        expected_sha256=receipt.canonical_online_tree_sha256,
        expected_entries=receipt.entries,
        label="canonical output after transfer receipt publication",
    )
    _require_tree(
        retained_placeholder,
        expected_sha256=receipt.empty_placeholder_tree_sha256,
        expected_entries=(),
        label="empty placeholder after transfer receipt publication",
    )
    for corpus_id, source_root in staging_roots.items():
        _require_controlled_transfer_directory(
            source_root,
            label=f"{corpus_id} staging output root after receipt publication",
        )
        _require_private_transfer_directory(
            canonical_roots[corpus_id],
            label=f"{corpus_id} canonical output root after receipt publication",
        )
    _require_private_transfer_directory(
        retained_placeholder,
        label="empty placeholder after receipt publication",
    )
    return receipt


def _revalidate_open_production(
    verified_open: VerifiedSuiteState | VerifiedProviderPredecessor,
    verified_closures: Mapping[str, VerifiedProductionRunClosure],
) -> None:
    from .production_controls import (
        ProductionControlError,
        load_production_control_finalization_receipt,
    )

    verified_open.assert_current()
    opened = verified_open.records[0]
    payload = opened.payload
    if not isinstance(payload, SuiteOpenBindings):
        raise SuiteAttemptError("verified suite does not begin with OPENED bindings")
    closures = _fixed_verified_closures(verified_closures)
    _, receipt_path = _local_file_uri(
        "production_finalization_receipt_uri",
        payload.production_finalization_receipt_uri,
    )
    try:
        receipt = load_production_control_finalization_receipt(receipt_path)
        file_sha256 = digest_regular_file(
            receipt_path,
            label="OPENED production finalization receipt",
        )
    except (ArtifactIntegrityError, ProductionControlError) as exc:
        raise SuiteAttemptError(f"cannot revalidate OPENED finalization: {exc}") from exc
    staging = {
        row.corpus_id: Path(unquote(urlsplit(row.output_uri).path))
        for row in payload.staging_namespaces
    }
    canonical = {
        row.corpus_id: Path(unquote(urlsplit(row.output_uri).path))
        for row in payload.output_namespaces
    }
    if (
        file_sha256 != payload.production_finalization_receipt_file_sha256
        or receipt.receipt_sha256 != file_sha256
        or receipt.finalization_request_sha256 != payload.production_finalization_request_sha256
        or receipt.manifest_sha256 != opened.manifest_sha256
        or receipt.suite_attempt_id != opened.suite_attempt_id
        or Path(receipt.canonical_suite_namespace) != verified_open.namespace
        or Path(receipt.pre_c1_output_staging_root) / "online"
        != next(iter(staging.values())).parent
        or receipt.provisional_closure_tree_sha256 != payload.provisional_closure_tree_sha256
        or receipt.instantiated_closure_tree_sha256 != payload.instantiated_closure_tree_sha256
        or any(
            path != verified_open.namespace / "online" / corpus_id
            for corpus_id, path in canonical.items()
        )
    ):
        raise SuiteAttemptError("OPENED production finalization changed before completion")
    receipt_rows = {row.corpus_id: row for row in receipt.corpora}
    plan_rows = {row.corpus_id: row for row in payload.runtime_attestation_plans}
    for corpus_id in FIXED_CORPORA:
        capability = closures[corpus_id]
        if (
            capability.binding != receipt_rows[corpus_id].closure_binding
            or capability.binding.receipt_sha256
            != plan_rows[corpus_id].production_run_closure_binding_receipt_sha256
        ):
            raise SuiteAttemptError(f"{corpus_id} production authority changed after OPENED")


def _write_transition(
    predecessor: VerifiedSuiteState,
    *,
    state: SuiteState,
    payload: StatePayload,
) -> SuiteStateRecord:
    predecessor.assert_current()
    if predecessor.state.state in {"ANALYSIS_COMPLETE", "FAILED"}:
        raise SuiteAttemptError(f"{predecessor.state.state} is terminal")
    if state == "FAILED" and predecessor.state.state not in {
        "RUN_CLAIMED",
        "LABEL_RELEASE_CLAIMED",
        "ANALYSIS_CLAIMED",
    }:
        raise SuiteAttemptError("FAILED can consume only an already-won provider phase claim")
    expected: dict[str, str] = {
        "OPENED": "RUN_CLAIMED",
        "RUN_CLAIMED": "ONLINE_COMPLETE",
        "ONLINE_COMPLETE": "LABEL_RELEASE_CLAIMED",
        "LABEL_RELEASE_CLAIMED": "LABELS_RELEASED",
        "LABELS_RELEASED": "ANALYSIS_CLAIMED",
        "ANALYSIS_CLAIMED": "ANALYSIS_COMPLETE",
    }
    if state != "FAILED" and expected.get(predecessor.state.state) != state:
        raise SuiteAttemptError("suite transition is not permitted by the canonical state machine")
    sequence = predecessor.state.sequence + 1
    record = SuiteStateRecord(
        suite_attempt_id=predecessor.state.suite_attempt_id,
        manifest_sha256=predecessor.state.manifest_sha256,
        run_receipt_sha256=predecessor.state.run_receipt_sha256,
        namespace_uri=predecessor.state.namespace_uri,
        sequence=sequence,
        state=state,
        previous_state_record_sha256=predecessor.state.record_sha256,
        payload=payload,
    )
    _write_once(
        record.canonical_bytes() + b"\n",
        _state_path(predecessor.namespace, sequence),
        label=f"{state} suite transition",
    )
    return record


def _online_claim_payload(
    opened: SuiteStateRecord,
    *,
    execution_claim: ExecutionClaimContract,
    provider_identity: ProviderExecutionIdentity,
    zenodo_admission: AnonymousZenodoAdmission,
    c1_manifest_rekor_integrated_at_utc: str,
    c1_registry_rekor_integrated_at_utc: str,
) -> RunClaimBindings:
    if not isinstance(opened, SuiteStateRecord) or opened.state != "OPENED":
        raise SuiteAttemptError("RUN_CLAIMED requires verified OPENED state")
    if not isinstance(opened.payload, SuiteOpenBindings):
        raise SuiteAttemptError("RUN_CLAIMED predecessor lacks OPENED bindings")
    if not isinstance(execution_claim, ExecutionClaimContract):
        raise SuiteAttemptError("RUN_CLAIMED execution contract must be typed")
    if not isinstance(provider_identity, ProviderExecutionIdentity):
        raise SuiteAttemptError("RUN_CLAIMED provider identity must be typed")
    if not isinstance(zenodo_admission, AnonymousZenodoAdmission):
        raise SuiteAttemptError("RUN_CLAIMED Zenodo admission must be typed")
    plan_rows = {row.corpus_id: row for row in opened.payload.runtime_attestation_plans}
    namespaces = {row.corpus_id: row for row in opened.payload.output_namespaces}
    staging_namespaces = {row.corpus_id: row for row in opened.payload.staging_namespaces}
    claim_rows = {row.corpus_id: row for row in execution_claim.corpora}
    if (
        execution_claim.manifest_sha256 != opened.manifest_sha256
        or execution_claim.run_receipt_sha256 != opened.run_receipt_sha256
        or execution_claim.run_receipt_file_sha256 != opened.payload.run_receipt_file_sha256
        or set(claim_rows) != set(FIXED_CORPORA)
    ):
        raise SuiteAttemptError("RUN_CLAIMED contract differs from OPENED suite identity")
    for corpus_id in FIXED_CORPORA:
        if (
            claim_rows[corpus_id].canonical_namespace_uri != namespaces[corpus_id].output_uri
            or claim_rows[corpus_id].staging_namespace_uri
            != staging_namespaces[corpus_id].output_uri
            or claim_rows[corpus_id].runtime_plan_sha256 != plan_rows[corpus_id].plan_sha256
            or claim_rows[corpus_id].runtime_plan_file_sha256 != plan_rows[corpus_id].file_sha256
        ):
            raise SuiteAttemptError(
                f"{corpus_id} RUN_CLAIMED namespace or plan differs from OPENED"
            )
    return RunClaimBindings(
        opened_state_sha256=opened.record_sha256,
        execution_claim=execution_claim,
        provider_identity=provider_identity,
        zenodo_admission=zenodo_admission,
        c1_manifest_rekor_integrated_at_utc=c1_manifest_rekor_integrated_at_utc,
        c1_registry_rekor_integrated_at_utc=c1_registry_rekor_integrated_at_utc,
        workload_inputs_opened_before_claim=False,
        public_benchmark_labels_accessible=True,
        human_outcome_blindness=False,
        independent_organizational_custody=False,
    )


def claim_online_suite(
    verified_open: VerifiedSuiteOpened,
    *,
    execution_claim: ExecutionClaimContract,
    provider_identity: ProviderExecutionIdentity,
    zenodo_admission: AnonymousZenodoAdmission,
    c1_manifest_rekor_integrated_at_utc: str,
    c1_registry_rekor_integrated_at_utc: str,
) -> SuiteStateRecord:
    """Materialize the candidate RUN_CLAIMED state before provider CAS."""

    if not isinstance(verified_open, VerifiedSuiteOpened):
        raise SuiteAttemptError("RUN_CLAIMED requires a verified OPENED token")
    verified_open.assert_current()
    payload = _online_claim_payload(
        verified_open.state,
        execution_claim=execution_claim,
        provider_identity=provider_identity,
        zenodo_admission=zenodo_admission,
        c1_manifest_rekor_integrated_at_utc=c1_manifest_rekor_integrated_at_utc,
        c1_registry_rekor_integrated_at_utc=c1_registry_rekor_integrated_at_utc,
    )
    return _write_transition(verified_open, state="RUN_CLAIMED", payload=payload)


def claim_online_provider_candidate(
    predecessor: VerifiedProviderPredecessor,
    *,
    execution_claim: ExecutionClaimContract,
    provider_identity: ProviderExecutionIdentity,
    zenodo_admission: AnonymousZenodoAdmission,
    c1_manifest_rekor_integrated_at_utc: str,
    c1_registry_rekor_integrated_at_utc: str,
) -> SuiteStateRecord:
    """Build RUN_CLAIMED without remapping its provider-registered namespace."""

    if predecessor.state.state != "OPENED":
        raise SuiteAttemptError("provider RUN_CLAIMED requires OPENED predecessor")
    predecessor.assert_current()
    payload = _online_claim_payload(
        predecessor.state,
        execution_claim=execution_claim,
        provider_identity=provider_identity,
        zenodo_admission=zenodo_admission,
        c1_manifest_rekor_integrated_at_utc=c1_manifest_rekor_integrated_at_utc,
        c1_registry_rekor_integrated_at_utc=c1_registry_rekor_integrated_at_utc,
    )
    return _provider_candidate_transition(
        predecessor,
        state="RUN_CLAIMED",
        payload=payload,
    )


def complete_online_suite(
    verified_claimed: VerifiedSuiteRunClaimed | VerifiedProviderPredecessor,
    *,
    run_claim: VerifiedRunClaimCapability,
    verified_production_closures: Mapping[str, VerifiedProductionRunClosure],
    runtime_attestation_plan_paths: Mapping[str, str | Path],
    runtime_attestation_receipt_paths: Mapping[str, str | Path],
    sealed_launch_receipt_paths: Mapping[str, str | Path],
) -> SuiteStateRecord:
    """Transfer the exact five staged outputs, then persist ONLINE_COMPLETE."""

    if isinstance(verified_claimed, VerifiedProviderPredecessor):
        if verified_claimed.state.state != "RUN_CLAIMED":
            raise SuiteAttemptError("provider ONLINE_COMPLETE requires verified RUN_CLAIMED")
    elif not isinstance(verified_claimed, VerifiedSuiteRunClaimed):
        raise SuiteAttemptError("ONLINE_COMPLETE requires verified RUN_CLAIMED")
    if not isinstance(run_claim, VerifiedRunClaimCapability):
        raise SuiteAttemptError("ONLINE_COMPLETE requires the typed run-claim capability")
    run_claim.assert_current()
    if (
        run_claim.claim_state_sha256 != verified_claimed.state.record_sha256
        or run_claim.contract.manifest_sha256 != verified_claimed.state.manifest_sha256
        or run_claim.contract.run_receipt_sha256 != verified_claimed.state.run_receipt_sha256
    ):
        raise SuiteAttemptError("run-claim capability differs from RUN_CLAIMED state")
    _revalidate_open_production(verified_claimed, verified_production_closures)
    plan_paths = _fixed_corpus_paths(
        "runtime_attestation_plan_paths",
        runtime_attestation_plan_paths,
    )
    receipt_paths = _fixed_corpus_paths(
        "runtime_attestation_receipt_paths",
        runtime_attestation_receipt_paths,
    )
    launch_paths = _fixed_corpus_paths(
        "sealed_launch_receipt_paths",
        sealed_launch_receipt_paths,
    )
    opened_payload = verified_claimed.records[0].payload
    if not isinstance(opened_payload, SuiteOpenBindings):
        raise SuiteAttemptError("verified suite does not begin with OPENED bindings")
    corpus_output_roots = {
        row.corpus_id: Path(unquote(urlsplit(row.output_uri).path))
        for row in opened_payload.staging_namespaces
    }
    ordered = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
    opened_plans = {row.corpus_id: row for row in opened_payload.runtime_attestation_plans}
    runtime_evidence: dict[str, tuple[str, str, str, str, str, str, str, str]] = {}
    launch_evidence: dict[str, tuple[str, str, str, str, str, str, str]] = {}
    marker_paths: dict[str, Path] = {}
    for corpus_id in ordered:
        output_root = Path(corpus_output_roots[corpus_id])
        plan_path = plan_paths[corpus_id]
        receipt_path = receipt_paths[corpus_id]
        launch_path = launch_paths[corpus_id]
        if launch_path.name != "sealed-launch-receipt.json":
            raise SuiteAttemptError(
                f"{corpus_id} sealed launch receipt has an unregistered filename"
            )
        opened_plan = opened_plans[corpus_id]
        _, sealed_contract_path = _local_file_uri(
            "sealed_launch_contract_uri",
            opened_plan.sealed_launch_contract_uri,
        )
        try:
            launch_receipt = load_sealed_launch_receipt(launch_path)
            sealed_contract = load_sealed_launch_contract(sealed_contract_path)
            sealed_contract_file_sha256 = digest_regular_file(
                sealed_contract_path,
                label=f"{corpus_id} sealed launch contract",
            )
            verify_sealed_launch_evidence(
                launch_receipt,
                audit_root=launch_path.parent,
                sealed_contract=sealed_contract,
            )
            launch_file_sha256 = digest_regular_file(
                launch_path,
                label=f"{corpus_id} sealed launch receipt",
            )
            copied_tree_sha256 = digest_directory_tree(output_root).sha256
        except (ArtifactIntegrityError, SealedContainerLauncherError) as exc:
            raise SuiteAttemptError(
                f"cannot verify {corpus_id} sealed launch evidence: {exc}"
            ) from exc
        if (
            launch_receipt.outcome != "succeeded"
            or launch_receipt.corpus_id != corpus_id
            or launch_receipt.sealed_launcher_contract_sha256
            != opened_plan.sealed_launch_contract_sha256
            or sealed_contract.contract_sha256 != opened_plan.sealed_launch_contract_sha256
            or sealed_contract_file_sha256 != opened_plan.sealed_launch_contract_file_sha256
            or launch_receipt.registered_plan_instantiation_receipt_sha256
            != opened_plan.registered_plan_instantiation_receipt_sha256
            or launch_receipt.production_run_closure_binding_receipt_sha256
            != opened_plan.production_run_closure_binding_receipt_sha256
            or launch_receipt.copy_output_root != str(output_root)
            or launch_receipt.output_tree_sha256 != copied_tree_sha256
        ):
            raise SuiteAttemptError(
                f"{corpus_id} sealed launch receipt differs from OPENED or staged output"
            )
        launch_evidence[corpus_id] = (
            launch_path.as_uri(),
            output_root.as_uri(),
            launch_receipt.sealed_launcher_contract_sha256,
            launch_receipt.receipt_sha256,
            launch_file_sha256,
            launch_receipt.evidence_inventory_sha256,
            copied_tree_sha256,
        )
        expected_receipt_path = output_root / RUNTIME_ATTESTATION_RECEIPT_FILENAME
        if receipt_path != expected_receipt_path:
            raise SuiteAttemptError(
                f"{corpus_id} runtime attestation receipt is outside its fixed output root"
            )
        plan = load_runtime_attestation_plan(plan_path)
        runtime_receipt = load_runtime_attestation_receipt(receipt_path)
        verify_runtime_attestation_receipt(runtime_receipt, plan)
        plan_file_sha256 = digest_regular_file(
            plan_path,
            label=f"{corpus_id} runtime attestation plan",
        )
        if (
            plan.plan_sha256 != opened_plan.plan_sha256
            or plan_file_sha256 != opened_plan.file_sha256
            or runtime_receipt.manifest_sha256 != verified_claimed.state.manifest_sha256
            or runtime_receipt.code_commit != opened_payload.code_commit
            or runtime_receipt.oci_image_digest != opened_payload.runner_image
        ):
            raise SuiteAttemptError(
                f"{corpus_id} runtime attestation differs from the OPENED suite plan"
            )
        marker_path = Path(runtime_receipt.invocation_marker_path)
        expected_marker_path = output_root / RUNTIME_INVOCATION_MARKER_FILENAME
        if marker_path != expected_marker_path:
            raise SuiteAttemptError(
                f"{corpus_id} runtime invocation marker is outside its fixed output root"
            )
        marker_file_sha256 = digest_regular_file(
            marker_path,
            label=f"{corpus_id} runtime one-shot invocation marker",
        )
        if marker_file_sha256 != runtime_receipt.invocation_marker_sha256:
            raise SuiteAttemptError(f"{corpus_id} runtime one-shot marker differs from its receipt")
        command_path = output_root / PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME
        command_attempt = load_production_corpus_command_attempt(command_path)
        if (
            command_attempt.manifest_sha256 != verified_claimed.state.manifest_sha256
            or command_attempt.runtime_attestation_plan_sha256 != plan.plan_sha256
            or command_attempt.runtime_attestation_receipt_sha256 != runtime_receipt.receipt_sha256
            or command_attempt.config_file_sha256 != plan.workload_sha256
            or command_attempt.workload_id != PRODUCTION_CORPUS_WORKLOAD_ID
        ):
            raise SuiteAttemptError(
                f"{corpus_id} production command attempt differs from its runtime evidence"
            )
        marker_paths[corpus_id] = marker_path
        runtime_evidence[corpus_id] = (
            plan.plan_sha256,
            plan_file_sha256,
            runtime_receipt.receipt_sha256,
            digest_regular_file(
                receipt_path,
                label=f"{corpus_id} runtime attestation receipt",
            ),
            runtime_receipt.invocation_marker_sha256,
            marker_file_sha256,
            command_attempt.receipt_sha256,
            digest_regular_file(
                command_path,
                label=f"{corpus_id} production command attempt",
            ),
        )
    if len(set(marker_paths.values())) != len(FIXED_CORPORA):
        raise SuiteAttemptError(
            "runtime attestation receipts cannot reuse one invocation marker path"
        )
    if (
        len({row[0] for row in launch_evidence.values()}) != len(FIXED_CORPORA)
        or len({row[1] for row in launch_evidence.values()}) != len(FIXED_CORPORA)
        or len({row[2] for row in launch_evidence.values()}) != len(FIXED_CORPORA)
        or len({row[3] for row in launch_evidence.values()}) != len(FIXED_CORPORA)
        or len({row[5] for row in launch_evidence.values()}) != len(FIXED_CORPORA)
        or len({row[6] for row in launch_evidence.values()}) != len(FIXED_CORPORA)
    ):
        raise SuiteAttemptError("fixed corpora cannot reuse sealed launch evidence")
    closures = tuple(
        _load_online_corpus_closure(
            Path(corpus_output_roots[corpus_id]),
            corpus_id=corpus_id,
            opened=verified_claimed.records[0],
            runtime_attestation_plan_sha256=runtime_evidence[corpus_id][0],
            runtime_attestation_plan_file_sha256=runtime_evidence[corpus_id][1],
            runtime_attestation_receipt_sha256=runtime_evidence[corpus_id][2],
            runtime_attestation_receipt_file_sha256=runtime_evidence[corpus_id][3],
            runtime_invocation_marker_sha256=runtime_evidence[corpus_id][4],
            runtime_invocation_marker_file_sha256=runtime_evidence[corpus_id][5],
            production_command_attempt_sha256=runtime_evidence[corpus_id][6],
            production_command_attempt_file_sha256=runtime_evidence[corpus_id][7],
            sealed_launch_receipt_uri=launch_evidence[corpus_id][0],
            sealed_launch_copy_output_uri=launch_evidence[corpus_id][1],
            sealed_launch_contract_sha256=launch_evidence[corpus_id][2],
            sealed_launch_receipt_sha256=launch_evidence[corpus_id][3],
            sealed_launch_receipt_file_sha256=launch_evidence[corpus_id][4],
            sealed_launch_evidence_inventory_sha256=launch_evidence[corpus_id][5],
            sealed_launch_output_tree_sha256=launch_evidence[corpus_id][6],
        )
        for corpus_id in ordered
    )
    transfer = _transfer_staged_online_outputs(verified_claimed, closures)
    transfer_path = _output_transfer_receipt_path(verified_claimed.namespace)
    output_trees = tuple(
        CorpusOutputTree(
            corpus_id=row.corpus_id,
            output_namespace_uri=row.canonical_output_uri,
            tree_sha256=row.canonical_tree_sha256,
        )
        for row in transfer.corpora
    )
    aggregate_payload = {
        "claim_ledger_commit": run_claim.claim_ledger_commit,
        "claim_state_sha256": run_claim.claim_state_sha256,
        "corpus_trees": [row.to_dict() for row in output_trees],
        "derivation": OUTPUT_AGGREGATE_DERIVATION,
        "execute_job_id": run_claim.live_execute_job_receipt.execute_job_id,
        "output_aggregate_identity": run_claim.contract.output_aggregate_identity,
        "provider_identity_sha256": run_claim.provider_identity.identity_sha256,
    }
    output_aggregate = RunOutputAggregate(
        claim_state_sha256=run_claim.claim_state_sha256,
        claim_ledger_commit=run_claim.claim_ledger_commit,
        provider_identity_sha256=run_claim.provider_identity.identity_sha256,
        execute_job_id=run_claim.live_execute_job_receipt.execute_job_id,
        output_aggregate_identity=run_claim.contract.output_aggregate_identity,
        corpus_trees=output_trees,
        aggregate_sha256=_sha256(_canonical_bytes(aggregate_payload)),
    )
    closure = OnlineSuiteClosure(
        corpora=closures,
        output_transfer_receipt_uri=transfer_path.as_uri(),
        output_transfer_receipt_sha256=transfer.receipt_sha256,
        output_transfer_receipt_file_sha256=transfer.file_sha256,
        source_online_tree_sha256=transfer.source_online_tree_sha256,
        canonical_online_tree_sha256=transfer.canonical_online_tree_sha256,
        run_output_aggregate=output_aggregate,
    )
    if isinstance(verified_claimed, VerifiedProviderPredecessor):
        return _provider_candidate_transition(
            verified_claimed,
            state="ONLINE_COMPLETE",
            payload=closure,
        )
    return _write_transition(verified_claimed, state="ONLINE_COMPLETE", payload=closure)


def _root_execution_claim(
    token: VerifiedSuiteState | VerifiedProviderPredecessor,
) -> ExecutionClaimContract:
    matches = [record for record in token.records if record.state == "RUN_CLAIMED"]
    if len(matches) != 1 or not isinstance(matches[0].payload, RunClaimBindings):
        raise SuiteAttemptError("post-online phase lacks the sole RUN_CLAIMED lineage")
    return matches[0].payload.execution_claim


def _claim_provider_phase(
    predecessor: VerifiedSuiteState | VerifiedProviderPredecessor,
    *,
    expected_state: Literal["ONLINE_COMPLETE", "LABELS_RELEASED"],
    phase_contract: PhaseClaimContract,
    provider_identity: ProviderExecutionIdentity,
) -> SuiteStateRecord:
    if predecessor.state.state != expected_state:
        raise SuiteAttemptError(f"provider phase claim requires verified {expected_state}")
    predecessor.assert_current()
    expected_phase = {
        "ONLINE_COMPLETE": "label-release",
        "LABELS_RELEASED": "analysis",
    }[expected_state]
    if phase_contract.phase != expected_phase:
        raise SuiteAttemptError("provider phase contract names another transition")
    root_claim = _root_execution_claim(predecessor)
    provider_identity.matches_phase_contract(phase_contract)
    expected_plan = {
        "label-release": (
            root_claim.label_release_provider_plan_uri,
            root_claim.label_release_provider_plan_sha256,
        ),
        "analysis": (
            root_claim.analysis_provider_plan_uri,
            root_claim.analysis_provider_plan_sha256,
        ),
    }[expected_phase]
    expected_runtime = {
        "label-release": (
            root_claim.release_oci_index_digest,
            root_claim.release_oci_platform_manifest_digest,
            root_claim.release_tle_binary_sha256,
        ),
        "analysis": (
            root_claim.oci_index_digest,
            root_claim.analysis_oci_platform_manifest_digest,
            None,
        ),
    }[expected_phase]
    if (
        phase_contract.predecessor_state_sha256 != predecessor.state.record_sha256
        or phase_contract.predecessor_ledger_commit != predecessor.evidences[-1].transition_id
        or phase_contract.manifest_sha256 != predecessor.state.manifest_sha256
        or phase_contract.run_receipt_sha256 != predecessor.state.run_receipt_sha256
        or phase_contract.c1_commit != root_claim.c1_commit
        or phase_contract.claim_workflow_sha != root_claim.claim_workflow_sha
        or phase_contract.online_execution_claim_contract_sha256 != root_claim.contract_sha256
        or phase_contract.host_tool_contract_sha256 != root_claim.host_tools.contract_sha256
        or (phase_contract.c1_provider_plan_uri, phase_contract.c1_provider_plan_sha256)
        != expected_plan
        or (
            phase_contract.oci_index_digest,
            phase_contract.oci_platform_manifest_digest,
            phase_contract.tle_binary_sha256,
        )
        != expected_runtime
    ):
        raise SuiteAttemptError("provider phase contract changes the claimed C1 lineage")
    if expected_phase == "label-release":
        if (
            phase_contract.label_release_beacon is None
            or phase_contract.label_release_beacon.contract_sha256
            != root_claim.beacon.contract_sha256
        ):
            raise SuiteAttemptError("label-release phase changes the registered beacon")
    else:
        labels = predecessor.state.payload
        if not isinstance(labels, tuple):
            raise SuiteAttemptError("analysis claim predecessor lacks released labels")
        inputs = {row.corpus_id: row for row in phase_contract.corpora}
        online_matches = [
            record for record in predecessor.records if record.state == "ONLINE_COMPLETE"
        ]
        if len(online_matches) != 1 or not isinstance(
            online_matches[0].payload, OnlineSuiteClosure
        ):
            raise SuiteAttemptError("analysis claim lacks the exact online closure")
        online_rows = {row.corpus_id: row for row in online_matches[0].payload.corpora}
        for row in labels:
            if (
                not isinstance(row, LabelCorpusClosure)
                or inputs[row.corpus_id].input_uri != row.plaintext_uri
                or inputs[row.corpus_id].input_sha256 != row.plaintext_sha256
                or inputs[row.corpus_id].supporting_input_uri
                != online_rows[row.corpus_id].output_uri
                or inputs[row.corpus_id].supporting_input_sha256
                != online_rows[row.corpus_id].sealed_launch_output_tree_sha256
            ):
                raise SuiteAttemptError(
                    "analysis claim differs from released labels or online output trees"
                )
    payload = PhaseClaimBindings(
        predecessor_state_sha256=predecessor.state.record_sha256,
        phase_claim=phase_contract,
        provider_identity=provider_identity,
        phase_inputs_opened_before_claim=False,
        public_benchmark_labels_accessible=True,
        human_outcome_blindness=False,
        independent_organizational_custody=False,
    )
    target: SuiteState = {
        "label-release": "LABEL_RELEASE_CLAIMED",
        "analysis": "ANALYSIS_CLAIMED",
    }[expected_phase]  # type: ignore[assignment]
    if isinstance(predecessor, VerifiedProviderPredecessor):
        return _provider_candidate_transition(predecessor, state=target, payload=payload)
    return _write_transition(predecessor, state=target, payload=payload)


def claim_label_release_suite(
    verified_online: VerifiedSuiteOnlineCompletion,
    *,
    phase_contract: PhaseClaimContract,
    provider_identity: ProviderExecutionIdentity,
    manifest: Mapping[str, Any],
    ciphertext_paths: Mapping[str, str | Path],
    encryption_receipt_paths: Mapping[str, str | Path],
) -> SuiteStateRecord:
    """Create the sole candidate label-release claim before custody input access."""

    if not isinstance(verified_online, VerifiedSuiteOnlineCompletion):
        raise SuiteAttemptError("LABEL_RELEASE_CLAIMED requires verified ONLINE_COMPLETE")
    try:
        validate_study_manifest(manifest, require_frozen=True)
    except ValueError as exc:
        raise SuiteAttemptError(f"invalid frozen study manifest: {exc}") from exc
    if manifest_sha256(manifest) != verified_online.state.manifest_sha256:
        raise SuiteAttemptError("label-release claim manifest differs")
    if set(ciphertext_paths) != set(FIXED_CORPORA) or set(encryption_receipt_paths) != set(
        FIXED_CORPORA
    ):
        raise SuiteAttemptError("label-release claim requires exact five custody paths")
    ciphertext_pins = _corpus_manifest_artifact_sha256(
        manifest,
        role="sealed-label-ciphertext",
    )
    receipt_pins = _corpus_manifest_artifact_sha256(
        manifest,
        role="timelock-encryption-receipt",
    )
    claim_rows = {row.corpus_id: row for row in phase_contract.corpora}
    for corpus_id in FIXED_CORPORA:
        ciphertext = Path(ciphertext_paths[corpus_id])
        receipt = Path(encryption_receipt_paths[corpus_id])
        if not ciphertext.is_absolute() or not receipt.is_absolute():
            raise SuiteAttemptError("label-release custody paths must be absolute")
        row = claim_rows[corpus_id]
        if (
            row.input_uri != ciphertext.as_uri()
            or row.input_sha256 != ciphertext_pins[corpus_id]
            or row.supporting_input_uri != receipt.as_uri()
            or row.supporting_input_sha256 != receipt_pins[corpus_id]
        ):
            raise SuiteAttemptError("label-release claim differs from frozen custody records")
    return _claim_provider_phase(
        verified_online,
        expected_state="ONLINE_COMPLETE",
        phase_contract=phase_contract,
        provider_identity=provider_identity,
    )


def claim_analysis_suite(
    verified_labels: VerifiedSuiteLabelsReleased,
    *,
    phase_contract: PhaseClaimContract,
    provider_identity: ProviderExecutionIdentity,
) -> SuiteStateRecord:
    """Create the sole candidate analysis claim before post-label input access."""

    if not isinstance(verified_labels, VerifiedSuiteLabelsReleased):
        raise SuiteAttemptError("ANALYSIS_CLAIMED requires verified LABELS_RELEASED")
    return _claim_provider_phase(
        verified_labels,
        expected_state="LABELS_RELEASED",
        phase_contract=phase_contract,
        provider_identity=provider_identity,
    )


def claim_label_release_provider_candidate(
    predecessor: VerifiedProviderPredecessor,
    *,
    phase_contract: PhaseClaimContract,
    provider_identity: ProviderExecutionIdentity,
    manifest: Mapping[str, Any],
    ciphertext_paths: Mapping[str, str | Path],
    encryption_receipt_paths: Mapping[str, str | Path],
) -> SuiteStateRecord:
    """Build LABEL_RELEASE_CLAIMED from portable provider evidence."""

    if predecessor.state.state != "ONLINE_COMPLETE":
        raise SuiteAttemptError(
            "provider LABEL_RELEASE_CLAIMED requires ONLINE_COMPLETE predecessor"
        )
    try:
        validate_study_manifest(manifest, require_frozen=True)
    except ValueError as exc:
        raise SuiteAttemptError(f"invalid frozen study manifest: {exc}") from exc
    if manifest_sha256(manifest) != predecessor.state.manifest_sha256:
        raise SuiteAttemptError("label-release claim manifest differs")
    if set(ciphertext_paths) != set(FIXED_CORPORA) or set(encryption_receipt_paths) != set(
        FIXED_CORPORA
    ):
        raise SuiteAttemptError("label-release claim requires exact five custody paths")
    ciphertext_pins = _corpus_manifest_artifact_sha256(
        manifest,
        role="sealed-label-ciphertext",
    )
    receipt_pins = _corpus_manifest_artifact_sha256(
        manifest,
        role="timelock-encryption-receipt",
    )
    claim_rows = {row.corpus_id: row for row in phase_contract.corpora}
    if set(claim_rows) != set(FIXED_CORPORA):
        raise SuiteAttemptError("label-release contract omits a fixed corpus")
    for corpus_id in FIXED_CORPORA:
        ciphertext = Path(ciphertext_paths[corpus_id])
        receipt = Path(encryption_receipt_paths[corpus_id])
        if not ciphertext.is_absolute() or not receipt.is_absolute():
            raise SuiteAttemptError("label-release custody paths must be absolute")
        row = claim_rows[corpus_id]
        if (
            row.input_uri != ciphertext.as_uri()
            or row.input_sha256 != ciphertext_pins[corpus_id]
            or row.supporting_input_uri != receipt.as_uri()
            or row.supporting_input_sha256 != receipt_pins[corpus_id]
        ):
            raise SuiteAttemptError("label-release claim differs from frozen custody records")
    return _claim_provider_phase(
        predecessor,
        expected_state="ONLINE_COMPLETE",
        phase_contract=phase_contract,
        provider_identity=provider_identity,
    )


def claim_analysis_provider_candidate(
    predecessor: VerifiedProviderPredecessor,
    *,
    phase_contract: PhaseClaimContract,
    provider_identity: ProviderExecutionIdentity,
) -> SuiteStateRecord:
    """Build ANALYSIS_CLAIMED from portable provider evidence."""

    if predecessor.state.state != "LABELS_RELEASED":
        raise SuiteAttemptError("provider ANALYSIS_CLAIMED requires LABELS_RELEASED predecessor")
    return _claim_provider_phase(
        predecessor,
        expected_state="LABELS_RELEASED",
        phase_contract=phase_contract,
        provider_identity=provider_identity,
    )


def _assert_state_transition(previous: SuiteStateRecord, current: SuiteStateRecord) -> None:
    if (
        current.sequence != previous.sequence + 1
        or current.previous_state_record_sha256 != previous.record_sha256
        or current.suite_attempt_id != previous.suite_attempt_id
        or current.manifest_sha256 != previous.manifest_sha256
        or current.run_receipt_sha256 != previous.run_receipt_sha256
        or current.namespace_uri != previous.namespace_uri
    ):
        raise SuiteAttemptError("suite state chain changes its identity or predecessor")
    expected = {
        "OPENED": "RUN_CLAIMED",
        "RUN_CLAIMED": "ONLINE_COMPLETE",
        "ONLINE_COMPLETE": "LABEL_RELEASE_CLAIMED",
        "LABEL_RELEASE_CLAIMED": "LABELS_RELEASED",
        "LABELS_RELEASED": "ANALYSIS_CLAIMED",
        "ANALYSIS_CLAIMED": "ANALYSIS_COMPLETE",
    }
    if current.state != "FAILED" and expected.get(previous.state) != current.state:
        raise SuiteAttemptError("suite state chain contains a forbidden transition")
    if current.state == "FAILED" and previous.state not in {
        "RUN_CLAIMED",
        "LABEL_RELEASE_CLAIMED",
        "ANALYSIS_CLAIMED",
    }:
        raise SuiteAttemptError("suite FAILED without an already-won provider claim")
    if previous.state == "OPENED" and current.state == "RUN_CLAIMED":
        if not isinstance(previous.payload, SuiteOpenBindings) or not isinstance(
            current.payload, RunClaimBindings
        ):
            raise SuiteAttemptError("suite state chain has malformed RUN_CLAIMED bindings")
        if current.payload.opened_state_sha256 != previous.record_sha256:
            raise SuiteAttemptError("RUN_CLAIMED binds another OPENED state")
        contract = current.payload.execution_claim
        plans = {row.corpus_id: row for row in previous.payload.runtime_attestation_plans}
        namespaces = {row.corpus_id: row for row in previous.payload.output_namespaces}
        staging_namespaces = {row.corpus_id: row for row in previous.payload.staging_namespaces}
        if (
            contract.manifest_sha256 != previous.manifest_sha256
            or contract.run_receipt_sha256 != previous.run_receipt_sha256
            or contract.run_receipt_file_sha256 != previous.payload.run_receipt_file_sha256
        ):
            raise SuiteAttemptError("RUN_CLAIMED changes the OPENED execution identity")
        for row in contract.corpora:
            if (
                row.canonical_namespace_uri != namespaces[row.corpus_id].output_uri
                or row.staging_namespace_uri != staging_namespaces[row.corpus_id].output_uri
                or row.runtime_plan_sha256 != plans[row.corpus_id].plan_sha256
                or row.runtime_plan_file_sha256 != plans[row.corpus_id].file_sha256
            ):
                raise SuiteAttemptError("RUN_CLAIMED changes an OPENED plan or namespace")
    if previous.state == "RUN_CLAIMED" and current.state == "ONLINE_COMPLETE":
        if not isinstance(previous.payload, RunClaimBindings) or not isinstance(
            current.payload, OnlineSuiteClosure
        ):
            raise SuiteAttemptError("suite state chain has malformed online bindings")
        contract = previous.payload.execution_claim
        aggregate = current.payload.run_output_aggregate
        if (
            aggregate.claim_state_sha256 != previous.record_sha256
            or aggregate.provider_identity_sha256
            != previous.payload.provider_identity.identity_sha256
            or aggregate.output_aggregate_identity != contract.output_aggregate_identity
        ):
            raise SuiteAttemptError("ONLINE_COMPLETE aggregate differs from RUN_CLAIMED")
        claim_rows = {row.corpus_id: row for row in contract.corpora}
        for closure in current.payload.corpora:
            claim = claim_rows[closure.corpus_id]
            if (
                closure.output_uri != claim.canonical_namespace_uri
                or closure.staging_output_uri != claim.staging_namespace_uri
                or closure.runtime_attestation_plan_sha256 != claim.runtime_plan_sha256
                or closure.runtime_attestation_plan_file_sha256 != claim.runtime_plan_file_sha256
            ):
                raise SuiteAttemptError("ONLINE_COMPLETE corpus closure differs from RUN_CLAIMED")
        _, transfer_path = _local_file_uri(
            "output_transfer_receipt_uri",
            current.payload.output_transfer_receipt_uri,
        )
        if transfer_path != _output_transfer_receipt_path(
            Path(unquote(urlsplit(previous.namespace_uri).path))
        ):
            raise SuiteAttemptError(
                "ONLINE_COMPLETE transfer receipt is outside the registered suite path"
            )
    if previous.state == "ONLINE_COMPLETE" and current.state == "LABEL_RELEASE_CLAIMED":
        if not isinstance(previous.payload, OnlineSuiteClosure) or not isinstance(
            current.payload, PhaseClaimBindings
        ):
            raise SuiteAttemptError("label-release claim bindings are malformed")
        contract = current.payload.phase_claim
        if (
            contract.phase != "label-release"
            or current.payload.predecessor_state_sha256 != previous.record_sha256
            or contract.predecessor_state_sha256 != previous.record_sha256
            or contract.manifest_sha256 != previous.manifest_sha256
            or contract.run_receipt_sha256 != previous.run_receipt_sha256
        ):
            raise SuiteAttemptError("LABEL_RELEASE_CLAIMED changes its predecessor identity")
    if previous.state == "LABEL_RELEASE_CLAIMED" and current.state == "LABELS_RELEASED":
        if not isinstance(previous.payload, PhaseClaimBindings) or not isinstance(
            current.payload, tuple
        ):
            raise SuiteAttemptError("label release completion bindings are malformed")
        outputs = {row.corpus_id: row.output_uri for row in previous.payload.phase_claim.corpora}
        for row in current.payload:
            if (
                not isinstance(row, LabelCorpusClosure)
                or row.plaintext_uri != outputs[row.corpus_id]
            ):
                raise SuiteAttemptError("LABELS_RELEASED output differs from the winning claim")
    if previous.state == "LABELS_RELEASED" and current.state == "ANALYSIS_CLAIMED":
        if not isinstance(previous.payload, tuple) or not isinstance(
            current.payload, PhaseClaimBindings
        ):
            raise SuiteAttemptError("analysis claim bindings are malformed")
        contract = current.payload.phase_claim
        inputs = {row.corpus_id: row for row in contract.corpora}
        if (
            contract.phase != "analysis"
            or current.payload.predecessor_state_sha256 != previous.record_sha256
            or contract.predecessor_state_sha256 != previous.record_sha256
            or contract.manifest_sha256 != previous.manifest_sha256
            or contract.run_receipt_sha256 != previous.run_receipt_sha256
        ):
            raise SuiteAttemptError("ANALYSIS_CLAIMED changes its predecessor identity")
        for row in previous.payload:
            if (
                not isinstance(row, LabelCorpusClosure)
                or inputs[row.corpus_id].input_uri != row.plaintext_uri
                or inputs[row.corpus_id].input_sha256 != row.plaintext_sha256
            ):
                raise SuiteAttemptError("ANALYSIS_CLAIMED changes released label inputs")
    if previous.state == "ANALYSIS_CLAIMED" and current.state == "ANALYSIS_COMPLETE":
        if not isinstance(previous.payload, PhaseClaimBindings) or not isinstance(
            current.payload, AnalysisClosure
        ):
            raise SuiteAttemptError("analysis completion bindings are malformed")
    if current.state == "FAILED":
        if not isinstance(current.payload, ProviderPhaseFailure):
            raise SuiteAttemptError("FAILED provider evidence is malformed")
        if previous.state == "RUN_CLAIMED" and isinstance(previous.payload, RunClaimBindings):
            expected_failure = (
                "online",
                previous.payload.provider_identity.identity_sha256,
                previous.payload.execution_claim.contract_sha256,
            )
        elif previous.state in {
            "LABEL_RELEASE_CLAIMED",
            "ANALYSIS_CLAIMED",
        } and isinstance(previous.payload, PhaseClaimBindings):
            expected_failure = (
                previous.payload.phase_claim.phase,
                previous.payload.provider_identity.identity_sha256,
                previous.payload.phase_claim.contract_sha256,
            )
        else:  # pragma: no cover - guarded above
            raise SuiteAttemptError("FAILED predecessor has no provider claim")
        if (
            current.payload.phase,
            current.payload.provider_identity_sha256,
            current.payload.phase_input_sha256,
        ) != expected_failure or current.payload.claim_state_sha256 != previous.record_sha256:
            raise SuiteAttemptError("FAILED evidence differs from the winning provider claim")
    if previous.state in {"ANALYSIS_COMPLETE", "FAILED"}:
        raise SuiteAttemptError(f"suite state chain continues after terminal {previous.state}")


def _assert_provider_claims(
    claims: SuiteProviderClaims,
    *,
    evidence: SuiteAttestationEvidence,
    descriptor: SuiteAttestationDescriptor,
    state: SuiteStateRecord,
    bundle: bytes,
    previous_evidence: SuiteAttestationEvidence | None,
) -> None:
    if not isinstance(claims, SuiteProviderClaims):
        raise SuiteAttemptError("provider verifier returned an untyped claim set")
    expected_claims = {
        "subject_sha256": state.record_sha256,
        "bundle_sha256": _sha256(bundle),
        "signer_identity": descriptor.expected_signer_identity,
        "oidc_issuer": descriptor.expected_oidc_issuer,
        "repository": descriptor.expected_repository,
        "workflow": descriptor.expected_workflow,
        "git_ref": descriptor.expected_git_ref,
        "signer_digest": descriptor.expected_signer_digest,
        "github_hosted_runner": True,
        "transparency_log_identity": descriptor.transparency_log_identity,
        "transparency_entry_id": evidence.transparency_entry_id,
        "transparency_log_index": evidence.transparency_log_index,
        "integrated_at_utc": evidence.integrated_at_utc,
        "timestamp_authority_identity": descriptor.timestamp_authority_identity,
        "timestamp_token_sha256": evidence.timestamp_token_sha256,
        "signed_at_utc": evidence.signed_at_utc,
        "state_service_identity": descriptor.state_service_identity,
        "state_key": evidence.state_key,
        "transition_id": evidence.transition_id,
        "previous_transition_id": evidence.previous_transition_id,
    }
    for name, expected in expected_claims.items():
        if getattr(claims, name) != expected:
            raise SuiteAttemptError(f"provider-verified {name} differs from the closed policy")
    evidence_claims = {
        "signer_identity": evidence.signer_identity,
        "oidc_issuer": evidence.oidc_issuer,
        "repository": evidence.repository,
        "workflow": evidence.workflow,
        "git_ref": evidence.git_ref,
        "signer_digest": evidence.signer_digest,
        "github_hosted_runner": evidence.github_hosted_runner,
        "transparency_log_identity": evidence.transparency_log_identity,
        "timestamp_authority_identity": evidence.timestamp_authority_identity,
        "state_service_identity": evidence.state_service_identity,
    }
    for name, observed in evidence_claims.items():
        if getattr(claims, name) != observed:
            raise SuiteAttemptError(f"untrusted evidence misstates provider-verified {name}")
    if not (
        claims.signature_verified
        and claims.transparency_verified
        and claims.timestamp_verified
        and claims.exclusive_transition
    ):
        raise SuiteAttemptError(
            "provider verifier did not prove signature, transparency, timestamp, and "
            "exclusive transition"
        )
    state_key = f"{descriptor.state_key_prefix}/{state.suite_attempt_id}"
    if claims.state_key != state_key:
        raise SuiteAttemptError("provider state key is not the fixed manifest-derived key")
    expected_previous = None if previous_evidence is None else previous_evidence.transition_id
    if claims.previous_transition_id != expected_previous:
        raise SuiteAttemptError("provider transition does not compare-and-swap its predecessor")


def _verify_one_attestation(
    *,
    namespace: Path,
    state: SuiteStateRecord,
    descriptor: SuiteAttestationDescriptor,
    verifier: SuiteEvidenceVerifier,
    previous_evidence: SuiteAttestationEvidence | None,
) -> SuiteAttestationEvidence:
    policy = descriptor
    if state.state == "RUN_CLAIMED":
        if not isinstance(state.payload, RunClaimBindings):
            raise SuiteAttemptError("RUN_CLAIMED attestation payload is malformed")
        contract = state.payload.execution_claim
        policy = replace(
            descriptor,
            expected_signer_identity=f"https://github.com/{contract.claim_workflow_ref}",
            expected_repository=contract.repository,
            expected_workflow=contract.claim_workflow_path,
            expected_git_ref="refs/tags/confirmatory-apparatus-c0",
            expected_signer_digest=contract.claim_workflow_sha,
        )
    elif state.state in {"LABEL_RELEASE_CLAIMED", "ANALYSIS_CLAIMED"}:
        if not isinstance(state.payload, PhaseClaimBindings):
            raise SuiteAttemptError("provider phase attestation payload is malformed")
        contract = state.payload.phase_claim
        policy = replace(
            descriptor,
            expected_signer_identity=f"https://github.com/{contract.claim_workflow_ref}",
            expected_repository=contract.repository,
            expected_workflow=contract.claim_workflow_path,
            expected_git_ref="refs/tags/confirmatory-apparatus-c0",
            expected_signer_digest=contract.claim_workflow_sha,
        )
    evidence = load_suite_attestation_evidence(_evidence_path(namespace, state.sequence))
    bundle = _secure_bytes(
        _bundle_path(namespace, state.sequence),
        label="suite provider bundle",
        max_bytes=_MAX_ATTESTATION_BYTES,
    )
    expected_evidence = {
        "suite_attempt_id": state.suite_attempt_id,
        "state_sequence": state.sequence,
        "state_name": state.state,
        "state_record_sha256": state.record_sha256,
        "descriptor_sha256": descriptor.descriptor_sha256,
        "bundle_sha256": _sha256(bundle),
        "bundle_byte_count": len(bundle),
        "signer_identity": policy.expected_signer_identity,
        "oidc_issuer": policy.expected_oidc_issuer,
        "repository": policy.expected_repository,
        "workflow": policy.expected_workflow,
        "git_ref": policy.expected_git_ref,
        "signer_digest": policy.expected_signer_digest,
        "github_hosted_runner": True,
        "transparency_log_identity": policy.transparency_log_identity,
        "timestamp_authority_identity": policy.timestamp_authority_identity,
        "state_service_identity": policy.state_service_identity,
        "state_key": f"{policy.state_key_prefix}/{state.suite_attempt_id}",
    }
    for name, expected in expected_evidence.items():
        if getattr(evidence, name) != expected:
            raise SuiteAttemptError(f"suite attestation evidence has mismatched {name}")
    state_bytes = _secure_bytes(
        _state_path(namespace, state.sequence),
        label="suite state record",
        max_bytes=_MAX_ATTESTATION_BYTES,
    )
    try:
        claims = verifier.verify(
            bundle=bundle,
            evidence=evidence,
            descriptor=policy,
            state_record_bytes=state_bytes,
        )
    except SuiteAttemptError:
        raise
    except Exception as exc:
        raise SuiteAttemptError("provider evidence verifier rejected the attestation") from exc
    _assert_provider_claims(
        claims,
        evidence=evidence,
        descriptor=policy,
        state=state,
        bundle=bundle,
        previous_evidence=previous_evidence,
    )
    return evidence


def _verify_online_transfer_state(
    namespace: Path,
    opened: SuiteStateRecord,
    online: SuiteStateRecord,
) -> tuple[Path, tuple[tuple[str, str, tuple[str, ...]], ...]]:
    if not isinstance(opened.payload, SuiteOpenBindings) or not isinstance(
        online.payload,
        OnlineSuiteClosure,
    ):
        raise SuiteAttemptError("suite online transfer state is malformed")
    _, transfer_path = _local_file_uri(
        "output_transfer_receipt_uri",
        online.payload.output_transfer_receipt_uri,
    )
    if transfer_path != _output_transfer_receipt_path(namespace):
        raise SuiteAttemptError("suite transfer receipt is not at its derived external path")
    receipt = load_suite_output_transfer_receipt(transfer_path)
    if (
        receipt.suite_attempt_id != opened.suite_attempt_id
        or receipt.manifest_sha256 != opened.manifest_sha256
        or receipt.production_finalization_receipt_file_sha256
        != opened.payload.production_finalization_receipt_file_sha256
        or receipt.receipt_sha256 != online.payload.output_transfer_receipt_sha256
        or receipt.file_sha256 != online.payload.output_transfer_receipt_file_sha256
        or receipt.source_online_tree_sha256 != online.payload.source_online_tree_sha256
        or receipt.canonical_online_tree_sha256 != online.payload.canonical_online_tree_sha256
    ):
        raise SuiteAttemptError("suite output transfer receipt differs from ONLINE_COMPLETE")
    transfer_rows = {row.corpus_id: row for row in receipt.corpora}
    closure_rows = {row.corpus_id: row for row in online.payload.corpora}
    for corpus_id in FIXED_CORPORA:
        transfer = transfer_rows[corpus_id]
        closure = closure_rows[corpus_id]
        if (
            transfer.staging_output_uri != closure.staging_output_uri
            or transfer.canonical_output_uri != closure.output_uri
            or transfer.files != closure.transfer_files
            or transfer.source_tree_sha256 != closure.sealed_launch_output_tree_sha256
        ):
            raise SuiteAttemptError(
                "online closure differs from its typed role, filename, digest, and count transfer"
            )
        roots = (
            (Path(unquote(urlsplit(transfer.staging_output_uri).path)), False),
            (Path(unquote(urlsplit(transfer.canonical_output_uri).path)), True),
        )
        for root, private in roots:
            if private:
                _require_private_transfer_directory(
                    root,
                    label=f"{corpus_id} canonical transferred output",
                )
            else:
                _require_controlled_transfer_directory(
                    root,
                    label=f"{corpus_id} staged transferred output",
                )
            _require_tree(
                root,
                expected_sha256=transfer.source_tree_sha256,
                expected_entries=transfer.entries,
                label=f"{corpus_id} transferred output",
            )
            for file_row in transfer.files:
                path = root / file_row.relative_path
                try:
                    metadata = path.lstat()
                    file_sha256 = digest_regular_file(
                        path,
                        label=f"{corpus_id} transferred file",
                    )
                except (OSError, ArtifactIntegrityError) as exc:
                    raise SuiteAttemptError(f"cannot verify {corpus_id} transferred file") from exc
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_size != file_row.byte_count
                    or file_sha256 != file_row.file_sha256
                    or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
                    or (
                        stat.S_IMODE(metadata.st_mode) != 0o600
                        if private
                        else bool(stat.S_IMODE(metadata.st_mode) & 0o022)
                    )
                ):
                    raise SuiteAttemptError(
                        f"{corpus_id} transferred file differs from its receipt"
                    )
    source = Path(unquote(urlsplit(receipt.staging_online_root_uri).path))
    canonical = Path(unquote(urlsplit(receipt.canonical_online_root_uri).path))
    retained = Path(unquote(urlsplit(receipt.retained_empty_placeholder_uri).path))
    if canonical != namespace / "online" or retained != _output_transfer_staging_path(namespace):
        raise SuiteAttemptError("suite transfer roots are not deterministically derived")
    _require_controlled_transfer_directory(source, label="preserved staging online tree")
    _require_private_transfer_directory(canonical, label="canonical online tree")
    _require_private_transfer_directory(retained, label="retained empty online placeholder")
    _require_tree(
        source,
        expected_sha256=receipt.source_online_tree_sha256,
        expected_entries=receipt.entries,
        label="preserved staging online tree",
    )
    _require_tree(
        canonical,
        expected_sha256=receipt.canonical_online_tree_sha256,
        expected_entries=receipt.entries,
        label="canonical online tree",
    )
    empty = _require_tree(
        retained,
        expected_sha256=receipt.empty_placeholder_tree_sha256,
        expected_entries=(),
        label="retained empty online placeholder",
    )
    return transfer_path, (
        (str(source), receipt.source_online_tree_sha256, receipt.entries),
        (str(canonical), receipt.canonical_online_tree_sha256, receipt.entries),
        (str(retained), empty.sha256, empty.entries),
    )


def _verify_opened_finalization_binding(
    path: Path,
    *,
    opened: SuiteStateRecord,
    namespace: Path,
) -> str:
    if not isinstance(opened.payload, SuiteOpenBindings):
        raise SuiteAttemptError("suite OPENED finalization binding is malformed")
    encoded = _secure_bytes(
        path,
        label="OPENED production finalization receipt",
        max_bytes=_MAX_ATTESTATION_BYTES,
    )
    payload = _parse_object(encoded, label="OPENED production finalization receipt")
    if encoded != _canonical_bytes(payload) + b"\n":
        raise SuiteAttemptError("OPENED production finalization receipt is not canonical")
    required = {
        "canonical_suite_namespace",
        "corpora",
        "finalization_request_sha256",
        "instantiated_closure_entries",
        "instantiated_closure_tree_sha256",
        "manifest_sha256",
        "pre_c1_output_staging_root",
        "provisional_closure_tree_sha256",
        "suite_attempt_id",
    }
    if not required.issubset(payload):
        raise SuiteAttemptError("OPENED production finalization receipt omits suite bindings")
    staging_parent = {
        Path(unquote(urlsplit(row.output_uri).path)).parent
        for row in opened.payload.staging_namespaces
    }
    if len(staging_parent) != 1:
        raise SuiteAttemptError("OPENED staging namespaces do not share one online root")
    staging_root = next(iter(staging_parent)).parent
    if (
        payload["manifest_sha256"] != opened.manifest_sha256
        or payload["suite_attempt_id"] != opened.suite_attempt_id
        or payload["canonical_suite_namespace"] != str(namespace)
        or payload["pre_c1_output_staging_root"] != str(staging_root)
        or payload["finalization_request_sha256"]
        != opened.payload.production_finalization_request_sha256
        or payload["provisional_closure_tree_sha256"]
        != opened.payload.provisional_closure_tree_sha256
        or payload["instantiated_closure_tree_sha256"]
        != opened.payload.instantiated_closure_tree_sha256
    ):
        raise SuiteAttemptError("OPENED state differs from production finalization")
    entries = payload["instantiated_closure_entries"]
    corpora = payload["corpora"]
    if (
        type(entries) is not list
        or not entries
        or not all(type(item) is str and item for item in entries)
        or entries != sorted(entries, key=lambda item: item.encode("utf-8"))
        or len(entries) != len(set(entries))
        or type(corpora) is not list
        or len(corpora) != len(FIXED_CORPORA)
    ):
        raise SuiteAttemptError("production finalization closure inventory is malformed")
    receipt_rows: dict[str, Mapping[str, Any]] = {}
    for row in corpora:
        if not isinstance(row, Mapping):
            raise SuiteAttemptError("production finalization corpus row is malformed")
        corpus_id = row.get("corpus_id")
        binding = row.get("closure_binding")
        if (
            type(corpus_id) is not str
            or corpus_id not in FIXED_CORPORA
            or corpus_id in receipt_rows
            or not isinstance(binding, Mapping)
        ):
            raise SuiteAttemptError("production finalization repeats or omits a corpus")
        if (
            binding.get("corpus_id") != corpus_id
            or binding.get("manifest_sha256") != opened.manifest_sha256
            or binding.get("provisional_closure_tree_sha256")
            != opened.payload.provisional_closure_tree_sha256
            or binding.get("instantiated_closure_tree_sha256")
            != opened.payload.instantiated_closure_tree_sha256
            or binding.get("entries") != entries
        ):
            raise SuiteAttemptError("production finalization corpus closure differs")
        receipt_rows[corpus_id] = binding
    plan_rows = {row.corpus_id: row for row in opened.payload.runtime_attestation_plans}
    if set(receipt_rows) != set(FIXED_CORPORA) or any(
        _sha256(_canonical_bytes(receipt_rows[corpus_id]))
        != plan_rows[corpus_id].production_run_closure_binding_receipt_sha256
        for corpus_id in FIXED_CORPORA
    ):
        raise SuiteAttemptError("OPENED plan instantiations differ from the final shared closure")
    return _sha256(encoded)


@dataclass(frozen=True)
class VerifiedSuiteState:
    """File-backed state token returned only after full provider verification."""

    namespace: Path
    records: tuple[SuiteStateRecord, ...]
    evidences: tuple[SuiteAttestationEvidence, ...]
    descriptor_sha256: str
    _file_sha256s: tuple[tuple[str, str], ...] = field(repr=False, compare=False)
    _capability: object = field(repr=False, compare=False)
    _tree_sha256s: tuple[tuple[str, str, tuple[str, ...]], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._capability is not _STATE_CAPABILITY:
            raise SuiteAttemptError("verified suite state can only come from file verification")
        if not self.records or len(self.records) != len(self.evidences):
            raise SuiteAttemptError("verified suite state has an incomplete attestation chain")
        if self.records[-1].sequence != len(self.records) - 1:
            raise SuiteAttemptError("verified suite state sequence is not contiguous")
        object.__setattr__(self, "namespace", _namespace_path(self.namespace))
        self.assert_current()

    @property
    def state(self) -> SuiteStateRecord:
        return self.records[-1]

    def assert_current(self) -> None:
        for relative, expected_digest in self._file_sha256s:
            candidate = Path(relative)
            target = candidate if candidate.is_absolute() else self.namespace / relative
            observed = digest_regular_file(target, label=f"verified suite file {relative}")
            if observed != expected_digest:
                raise SuiteAttemptError(
                    f"verified suite file {relative} changed after verification"
                )
        for path, expected_digest, expected_entries in self._tree_sha256s:
            _require_tree(
                Path(path),
                expected_sha256=expected_digest,
                expected_entries=expected_entries,
                label=f"verified suite tree {path}",
            )


@dataclass(frozen=True)
class VerifiedProviderPredecessor:
    """Portable state authority minted from stable GitHub and artifact evidence."""

    records: tuple[SuiteStateRecord, ...]
    evidences: tuple[SuiteAttestationEvidence, ...]
    control_inventory_sha256: str
    artifact_receipt_sha256: str
    _fresh_revalidator: Callable[[], None] = field(repr=False, compare=False)
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _PROVIDER_PREDECESSOR_CAPABILITY:
            raise SuiteAttemptError(
                "provider predecessor can only come from provider evidence verification"
            )
        if not self.records or len(self.records) != len(self.evidences):
            raise SuiteAttemptError("provider predecessor has an incomplete evidence chain")
        _digest("control_inventory_sha256", self.control_inventory_sha256)
        _digest("artifact_receipt_sha256", self.artifact_receipt_sha256)
        if not callable(self._fresh_revalidator):
            raise SuiteAttemptError("provider predecessor has no fresh revalidator")
        first = self.records[0]
        if (
            first.sequence != 0
            or first.state != "OPENED"
            or first.previous_state_record_sha256 is not None
            or not isinstance(first.payload, SuiteOpenBindings)
            or first.suite_attempt_id != suite_attempt_id(first.manifest_sha256)
        ):
            raise SuiteAttemptError("provider predecessor lacks manifest-derived OPENED genesis")
        descriptor_digests: set[str] = set()
        for sequence, (record, evidence) in enumerate(
            zip(self.records, self.evidences, strict=True)
        ):
            if sequence and record.state == "FAILED":
                if sequence != len(self.records) - 1:
                    raise SuiteAttemptError("provider predecessor has a post-terminal transition")
            if sequence:
                _assert_state_transition(self.records[sequence - 1], record)
            if (
                record.sequence != sequence
                or record.suite_attempt_id != first.suite_attempt_id
                or record.manifest_sha256 != first.manifest_sha256
                or record.run_receipt_sha256 != first.run_receipt_sha256
                or record.namespace_uri != first.namespace_uri
                or evidence.suite_attempt_id != record.suite_attempt_id
                or evidence.state_sequence != sequence
                or evidence.state_name != record.state
                or evidence.state_record_sha256 != record.record_sha256
            ):
                raise SuiteAttemptError("provider predecessor state and evidence differ")
            descriptor_digests.add(evidence.descriptor_sha256)
        if len(descriptor_digests) != 1 or descriptor_digests != {
            first.payload.attestation_descriptor_sha256
        }:
            raise SuiteAttemptError("provider predecessor changes its descriptor")
        self.assert_current()

    @property
    def state(self) -> SuiteStateRecord:
        return self.records[-1]

    @property
    def namespace(self) -> Path:
        """Return the protected state namespace, never a caller-selected path."""

        _, path = _local_file_uri("provider predecessor namespace_uri", self.state.namespace_uri)
        return _namespace_path(path)

    @property
    def ledger_commit(self) -> str:
        return self.evidences[-1].transition_id

    def assert_current(self) -> None:
        try:
            result = self._fresh_revalidator()
        except SuiteAttemptError:
            raise
        except Exception as exc:
            raise SuiteAttemptError("fresh provider predecessor revalidation failed") from exc
        if result is not None:
            raise SuiteAttemptError("provider predecessor revalidator returned data")


def _mint_verified_provider_predecessor(
    *,
    records: Sequence[SuiteStateRecord],
    evidences: Sequence[SuiteAttestationEvidence],
    control_inventory_sha256: str,
    artifact_receipt_sha256: str,
    fresh_revalidator: Callable[[], None],
) -> VerifiedProviderPredecessor:
    """Internal bridge for the closed provider transport verifier."""

    return VerifiedProviderPredecessor(
        records=tuple(records),
        evidences=tuple(evidences),
        control_inventory_sha256=control_inventory_sha256,
        artifact_receipt_sha256=artifact_receipt_sha256,
        _fresh_revalidator=fresh_revalidator,
        _capability=_PROVIDER_PREDECESSOR_CAPABILITY,
    )


def _provider_candidate_transition(
    predecessor: VerifiedProviderPredecessor,
    *,
    state: SuiteState,
    payload: StatePayload,
) -> SuiteStateRecord:
    if not isinstance(predecessor, VerifiedProviderPredecessor):
        raise SuiteAttemptError("provider candidate requires verified predecessor evidence")
    predecessor.assert_current()
    if predecessor.state.state in {"ANALYSIS_COMPLETE", "FAILED"}:
        raise SuiteAttemptError(f"{predecessor.state.state} is terminal")
    record = SuiteStateRecord(
        suite_attempt_id=predecessor.state.suite_attempt_id,
        manifest_sha256=predecessor.state.manifest_sha256,
        run_receipt_sha256=predecessor.state.run_receipt_sha256,
        namespace_uri=predecessor.state.namespace_uri,
        sequence=predecessor.state.sequence + 1,
        state=state,
        previous_state_record_sha256=predecessor.state.record_sha256,
        payload=payload,
    )
    _assert_state_transition(predecessor.state, record)
    predecessor.assert_current()
    return record


@dataclass(frozen=True)
class VerifiedSuiteOpened(VerifiedSuiteState):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state.state != "OPENED":
            raise SuiteAttemptError("VerifiedSuiteOpened requires OPENED")


def derive_execution_claim_contract(
    *,
    registration: VerifiedC1ProtocolRegistration,
    opened: VerifiedSuiteOpened,
) -> ExecutionClaimContract:
    """Derive the online claim only from verified C1 and verified OPENED state."""

    if not isinstance(registration, VerifiedC1ProtocolRegistration):
        raise SuiteAttemptError("execution claim derivation requires verified C1")
    if not isinstance(opened, VerifiedSuiteOpened):
        raise SuiteAttemptError("execution claim derivation requires verified OPENED state")
    opened.assert_current()
    return derive_execution_claim_contract_from_provider_opened(
        registration=registration,
        opened_state=opened.state,
    )


def derive_execution_claim_contract_from_provider_opened(
    *,
    registration: VerifiedC1ProtocolRegistration,
    opened_state: SuiteStateRecord,
) -> ExecutionClaimContract:
    """Derive C1 claim data after the caller verifies the provider OPENED tip."""

    if not isinstance(registration, VerifiedC1ProtocolRegistration):
        raise SuiteAttemptError("provider execution claim derivation requires verified C1")
    if not isinstance(opened_state, SuiteStateRecord) or opened_state.state != "OPENED":
        raise SuiteAttemptError("provider execution claim derivation requires OPENED")
    registration.assert_current()
    if not isinstance(opened_state.payload, SuiteOpenBindings):
        raise SuiteAttemptError("verified OPENED state has malformed bindings")
    if (
        opened_state.manifest_sha256 != registration.manifest_sha256
        or opened_state.suite_attempt_id != suite_attempt_id(registration.manifest_sha256)
    ):
        raise SuiteAttemptError("verified C1 and OPENED state name different manifests")

    manifest_path = registration.package_root / "study-manifest.json"
    expected_files = dict(registration.package_file_sha256s)
    if expected_files.get("study-manifest.json") != registration.manifest_sha256:
        raise SuiteAttemptError("verified C1 package inventory does not bind its manifest")
    try:
        plans = load_provider_phase_plans(
            manifest_path,
            c1_commit=registration.c1_commit,
        )
        manifest = load_study_manifest(manifest_path)
    except (ExecutionClaimError, ValueError) as exc:
        raise SuiteAttemptError("cannot derive provider plans from verified C1") from exc
    online = plans["online"]
    release = plans["label-release"]
    analysis = plans["analysis"]
    if online.execution_claim_inputs is None:
        raise SuiteAttemptError("verified C1 online plan lacks execution claim inputs")

    payload = opened_state.payload
    runtime_plans = {row.corpus_id: row for row in payload.runtime_attestation_plans}
    staging = {row.corpus_id: row for row in payload.staging_namespaces}
    canonical = {row.corpus_id: row for row in payload.output_namespaces}
    ordered = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
    corpora = tuple(
        ClaimCorpusBinding(
            corpus_id=corpus_id,
            staging_namespace_uri=staging[corpus_id].output_uri,
            canonical_namespace_uri=canonical[corpus_id].output_uri,
            runtime_plan_sha256=runtime_plans[corpus_id].plan_sha256,
            runtime_plan_file_sha256=runtime_plans[corpus_id].file_sha256,
        )
        for corpus_id in ordered
    )
    output_identity = _sha256(
        _canonical_bytes(
            {
                "corpora": [
                    {
                        "canonical_namespace_uri": row.canonical_namespace_uri,
                        "corpus_id": row.corpus_id,
                        "staging_namespace_uri": row.staging_namespace_uri,
                    }
                    for row in corpora
                ],
                "derivation": OUTPUT_AGGREGATE_DERIVATION,
                "manifest_sha256": opened_state.manifest_sha256,
            }
        )
    )
    sealed = manifest.get("sealed_execution")
    if not isinstance(sealed, Mapping) or not isinstance(sealed.get("hardware"), Mapping):
        raise SuiteAttemptError("verified C1 lacks its hardware contract")
    hardware_contract_sha256 = _sha256(_canonical_bytes(sealed["hardware"]))
    if any(
        value is None
        for value in (
            release.tle_binary_sha256,
            release.tle_build_provenance_sha256,
            release.tle_vulnerability_scan_sha256,
            release.tle_interoperability_receipt_sha256,
        )
    ):
        raise SuiteAttemptError("verified C1 release plan lacks its TLE closure")

    inputs = online.execution_claim_inputs
    return ExecutionClaimContract(
        repository=online.repository,
        claim_workflow_path=online.workflow_path,
        claim_workflow_ref=online.workflow_ref,
        claim_workflow_sha=online.workflow_sha,
        run_head_branch=online.run_head_branch,
        claim_job_name=online.claim_job_name,
        execute_job_name=online.execute_job_name,
        unique_runner_label=derive_phase_runner_label(online.claim_nonce, "online"),
        claim_nonce=online.claim_nonce,
        runner_id=online.runner_id,
        runner_name=online.runner_name,
        runner_group_id=online.runner_group_id,
        runner_version=online.runner_version,
        runner_archive_sha256=online.runner_archive_sha256,
        provider_operating_system=online.provider_operating_system,
        provider_architecture=online.provider_architecture,
        host_tools=online.host_tools,
        runtime_probe_receipt_sha256=online.runtime_probe_receipt_sha256,
        design_seed_sha256=inputs.design_seed_sha256,
        registered_online_runtime_budget_seconds=(inputs.registered_online_runtime_budget_seconds),
        maximum_online_runtime_seconds=online.maximum_runtime_seconds,
        c1_commit=registration.c1_commit,
        manifest_sha256=opened_state.manifest_sha256,
        label_release_provider_plan_uri=Path(release.provider_plan_path).as_uri(),
        label_release_provider_plan_sha256=release.plan_sha256,
        analysis_provider_plan_uri=Path(analysis.provider_plan_path).as_uri(),
        analysis_provider_plan_sha256=analysis.plan_sha256,
        run_receipt_sha256=opened_state.run_receipt_sha256,
        run_receipt_file_sha256=payload.run_receipt_file_sha256,
        oci_index_digest=online.oci_index_digest,
        oci_platform_manifest_digest=online.oci_platform_manifest_digest,
        analysis_oci_platform_manifest_digest=analysis.oci_platform_manifest_digest,
        release_oci_index_digest=release.oci_index_digest,
        release_oci_platform_manifest_digest=release.oci_platform_manifest_digest,
        release_tle_binary_sha256=release.tle_binary_sha256,  # type: ignore[arg-type]
        release_tle_build_provenance_sha256=release.tle_build_provenance_sha256,  # type: ignore[arg-type]
        release_tle_vulnerability_scan_sha256=release.tle_vulnerability_scan_sha256,  # type: ignore[arg-type]
        release_tle_interoperability_receipt_sha256=release.tle_interoperability_receipt_sha256,  # type: ignore[arg-type]
        hardware_contract_sha256=hardware_contract_sha256,
        corpora=corpora,
        output_aggregate_identity=output_identity,
        beacon=inputs.beacon,
    )


@dataclass(frozen=True)
class VerifiedSuiteRunClaimed(VerifiedSuiteState):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state.state != "RUN_CLAIMED":
            raise SuiteAttemptError("VerifiedSuiteRunClaimed requires RUN_CLAIMED")
        if not isinstance(self.state.payload, RunClaimBindings):
            raise SuiteAttemptError("verified RUN_CLAIMED payload is malformed")


@dataclass(frozen=True)
class VerifiedSuiteOnlineCompletion(VerifiedSuiteState):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state.state != "ONLINE_COMPLETE":
            raise SuiteAttemptError("VerifiedSuiteOnlineCompletion requires ONLINE_COMPLETE")

    def require_corpus(
        self,
        *,
        manifest_digest: str,
        corpus_id: str,
        online_result_receipt_sha256: str,
    ) -> OnlineCorpusClosure:
        self.assert_current()
        if self.state.manifest_sha256 != manifest_digest:
            raise SuiteAttemptError("verified suite completion belongs to another manifest")
        if not isinstance(self.state.payload, OnlineSuiteClosure):
            raise SuiteAttemptError("verified ONLINE_COMPLETE payload is malformed")
        matches = [row for row in self.state.payload.corpora if row.corpus_id == corpus_id]
        if len(matches) != 1 or not isinstance(matches[0], OnlineCorpusClosure):
            raise SuiteAttemptError("verified suite completion lacks the requested corpus")
        closure = matches[0]
        if closure.result_receipt_sha256 != online_result_receipt_sha256:
            raise SuiteAttemptError("verified suite completion binds another online result")
        return closure


@dataclass(frozen=True)
class VerifiedSuiteLabelReleaseClaimed(VerifiedSuiteState):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state.state != "LABEL_RELEASE_CLAIMED":
            raise SuiteAttemptError(
                "VerifiedSuiteLabelReleaseClaimed requires LABEL_RELEASE_CLAIMED"
            )
        if not isinstance(self.state.payload, PhaseClaimBindings):
            raise SuiteAttemptError("verified label-release claim payload is malformed")


@dataclass(frozen=True)
class VerifiedSuiteLabelsReleased(VerifiedSuiteState):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state.state != "LABELS_RELEASED":
            raise SuiteAttemptError("VerifiedSuiteLabelsReleased requires LABELS_RELEASED")


@dataclass(frozen=True)
class VerifiedSuiteAnalysisClaimed(VerifiedSuiteState):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state.state != "ANALYSIS_CLAIMED":
            raise SuiteAttemptError("VerifiedSuiteAnalysisClaimed requires ANALYSIS_CLAIMED")
        if not isinstance(self.state.payload, PhaseClaimBindings):
            raise SuiteAttemptError("verified analysis claim payload is malformed")


@dataclass(frozen=True)
class VerifiedSuiteAnalysisComplete(VerifiedSuiteState):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state.state != "ANALYSIS_COMPLETE":
            raise SuiteAttemptError("VerifiedSuiteAnalysisComplete requires ANALYSIS_COMPLETE")


@dataclass(frozen=True)
class VerifiedSuiteFailed(VerifiedSuiteState):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state.state != "FAILED":
            raise SuiteAttemptError("VerifiedSuiteFailed requires FAILED")


def _require_claimed_state(
    token: VerifiedSuiteState | VerifiedProviderPredecessor,
    *,
    expected_state: Literal[
        "RUN_CLAIMED",
        "LABEL_RELEASE_CLAIMED",
        "ANALYSIS_CLAIMED",
    ],
    local_type: type[VerifiedSuiteState],
    local_error: str,
) -> None:
    if isinstance(token, VerifiedProviderPredecessor):
        if token.state.state != expected_state:
            raise SuiteAttemptError(f"provider authority requires verified {expected_state}")
    elif not isinstance(token, local_type):
        raise SuiteAttemptError(local_error)
    token.assert_current()


def _claim_tip_revalidator(
    verified_claimed: VerifiedSuiteState | VerifiedProviderPredecessor,
    *,
    expected_state: Literal[
        "RUN_CLAIMED",
        "LABEL_RELEASE_CLAIMED",
        "ANALYSIS_CLAIMED",
    ],
    local_type: type[VerifiedSuiteState],
    fresh_state_revalidator: Callable[[], VerifiedSuiteState | VerifiedProviderPredecessor],
) -> Callable[[], None]:
    evidence = verified_claimed.evidences[-1]
    provider_backed = isinstance(verified_claimed, VerifiedProviderPredecessor)

    def revalidate() -> None:
        fresh = fresh_state_revalidator()
        if provider_backed:
            if (
                not isinstance(fresh, VerifiedProviderPredecessor)
                or fresh.state.state != expected_state
            ):
                raise ExecutionClaimError(f"fresh provider state is not {expected_state}")
        elif not isinstance(fresh, local_type):
            raise ExecutionClaimError(f"fresh provider state is not {expected_state}")
        fresh.assert_current()
        if (
            fresh.state.record_sha256 != verified_claimed.state.record_sha256
            or fresh.evidences[-1].transition_id != evidence.transition_id
            or fresh.evidences[-1].bundle_sha256 != evidence.bundle_sha256
        ):
            raise ExecutionClaimError(f"{expected_state} provider tip changed")
        if provider_backed:
            if not isinstance(fresh, VerifiedProviderPredecessor) or not isinstance(
                verified_claimed, VerifiedProviderPredecessor
            ):
                raise ExecutionClaimError("provider claim authority type changed")
            if (
                fresh.control_inventory_sha256 != verified_claimed.control_inventory_sha256
                or fresh.artifact_receipt_sha256 != verified_claimed.artifact_receipt_sha256
            ):
                raise ExecutionClaimError(f"{expected_state} provider authority changed")

    return revalidate


def admit_run_claim_beacon(
    verified_claimed: VerifiedSuiteRunClaimed | VerifiedProviderPredecessor,
    *,
    beacon_bytes: bytes,
    beacon_verifier: ExecutionBeaconVerifier,
    live_execute_job_receipt: LiveExecuteJobReceipt,
    verified_at_utc: str,
    fresh_state_revalidator: Callable[[], VerifiedSuiteRunClaimed | VerifiedProviderPredecessor],
) -> VerifiedRunClaimCapability:
    """Activate RUN_CLAIMED only after the exact future beacon is published."""

    _require_claimed_state(
        verified_claimed,
        expected_state="RUN_CLAIMED",
        local_type=VerifiedSuiteRunClaimed,
        local_error="beacon admission requires verified RUN_CLAIMED",
    )
    if not callable(fresh_state_revalidator):
        raise SuiteAttemptError("beacon admission requires a fresh provider revalidator")
    payload = verified_claimed.state.payload
    if not isinstance(payload, RunClaimBindings):
        raise SuiteAttemptError("RUN_CLAIMED payload is malformed")
    evidence = verified_claimed.evidences[-1]
    try:
        beacon_receipt = verify_execution_beacon(
            payload.execution_claim.beacon,
            beacon_bytes=beacon_bytes,
            claim_state_sha256=verified_claimed.state.record_sha256,
            claim_ledger_commit=evidence.transition_id,
            provider_identity=payload.provider_identity,
            claim_attested_at_utc=evidence.integrated_at_utc,
            verifier=beacon_verifier,
            verified_at_utc=verified_at_utc,
            design_seed_sha256=payload.execution_claim.design_seed_sha256,
        )
    except ExecutionClaimError as exc:
        raise SuiteAttemptError(f"execution beacon admission failed: {exc}") from exc

    try:
        return _mint_verified_run_claim(
            contract=payload.execution_claim,
            provider_identity=payload.provider_identity,
            claim_state_sha256=verified_claimed.state.record_sha256,
            claim_ledger_commit=evidence.transition_id,
            claim_attested_at_utc=evidence.integrated_at_utc,
            beacon_receipt=beacon_receipt,
            live_execute_job_receipt=live_execute_job_receipt,
            zenodo_admission=payload.zenodo_admission,
            fresh_revalidator=_claim_tip_revalidator(
                verified_claimed,
                expected_state="RUN_CLAIMED",
                local_type=VerifiedSuiteRunClaimed,
                fresh_state_revalidator=fresh_state_revalidator,
            ),
        )
    except ExecutionClaimError as exc:
        raise SuiteAttemptError(f"cannot mint RUN_CLAIMED capability: {exc}") from exc


def _phase_claim_revalidator(
    verified_claimed: VerifiedSuiteState | VerifiedProviderPredecessor,
    *,
    expected_state: Literal["LABEL_RELEASE_CLAIMED", "ANALYSIS_CLAIMED"],
    local_type: type[VerifiedSuiteState],
    fresh_state_revalidator: Callable[[], VerifiedSuiteState | VerifiedProviderPredecessor],
) -> Callable[[], None]:
    return _claim_tip_revalidator(
        verified_claimed,
        expected_state=expected_state,
        local_type=local_type,
        fresh_state_revalidator=fresh_state_revalidator,
    )


def admit_label_release_claim_beacon(
    verified_claimed: VerifiedSuiteLabelReleaseClaimed | VerifiedProviderPredecessor,
    *,
    beacon_bytes: bytes,
    beacon_verifier: ExecutionBeaconVerifier,
    live_execute_job_receipt: LiveExecuteJobReceipt,
    verified_at_utc: str,
    fresh_state_revalidator: Callable[
        [], VerifiedSuiteLabelReleaseClaimed | VerifiedProviderPredecessor
    ],
) -> VerifiedPhaseClaimCapability:
    """Mint label-release authority only after its later registered beacon."""

    _require_claimed_state(
        verified_claimed,
        expected_state="LABEL_RELEASE_CLAIMED",
        local_type=VerifiedSuiteLabelReleaseClaimed,
        local_error="label beacon requires verified LABEL_RELEASE_CLAIMED",
    )
    if not callable(fresh_state_revalidator):
        raise SuiteAttemptError("label beacon requires a fresh provider revalidator")
    payload = verified_claimed.state.payload
    if not isinstance(payload, PhaseClaimBindings):
        raise SuiteAttemptError("LABEL_RELEASE_CLAIMED payload is malformed")
    evidence = verified_claimed.evidences[-1]
    try:
        beacon = verify_label_release_beacon(
            payload.phase_claim,
            beacon_bytes=beacon_bytes,
            phase_claim_state_sha256=verified_claimed.state.record_sha256,
            phase_claim_ledger_commit=evidence.transition_id,
            provider_identity=payload.provider_identity,
            claim_attested_at_utc=evidence.integrated_at_utc,
            live_execute_job_receipt=live_execute_job_receipt,
            verifier=beacon_verifier,
            verified_at_utc=verified_at_utc,
        )
        return _mint_verified_phase_claim(
            contract=payload.phase_claim,
            provider_identity=payload.provider_identity,
            phase_claim_state_sha256=verified_claimed.state.record_sha256,
            phase_claim_ledger_commit=evidence.transition_id,
            claim_attested_at_utc=evidence.integrated_at_utc,
            live_execute_job_receipt=live_execute_job_receipt,
            phase_beacon_receipt=beacon,
            fresh_revalidator=_phase_claim_revalidator(
                verified_claimed,
                expected_state="LABEL_RELEASE_CLAIMED",
                local_type=VerifiedSuiteLabelReleaseClaimed,
                fresh_state_revalidator=fresh_state_revalidator,
            ),
        )
    except ExecutionClaimError as exc:
        raise SuiteAttemptError(f"cannot mint label-release authority: {exc}") from exc


def admit_analysis_claim(
    verified_claimed: VerifiedSuiteAnalysisClaimed | VerifiedProviderPredecessor,
    *,
    live_execute_job_receipt: LiveExecuteJobReceipt,
    fresh_state_revalidator: Callable[
        [], VerifiedSuiteAnalysisClaimed | VerifiedProviderPredecessor
    ],
) -> VerifiedPhaseClaimCapability:
    """Mint analysis authority after the provider claim is freshly revalidated."""

    _require_claimed_state(
        verified_claimed,
        expected_state="ANALYSIS_CLAIMED",
        local_type=VerifiedSuiteAnalysisClaimed,
        local_error="analysis authority requires verified ANALYSIS_CLAIMED",
    )
    if not callable(fresh_state_revalidator):
        raise SuiteAttemptError("analysis authority requires a fresh provider revalidator")
    payload = verified_claimed.state.payload
    if not isinstance(payload, PhaseClaimBindings):
        raise SuiteAttemptError("ANALYSIS_CLAIMED payload is malformed")
    evidence = verified_claimed.evidences[-1]
    try:
        return _mint_verified_phase_claim(
            contract=payload.phase_claim,
            provider_identity=payload.provider_identity,
            phase_claim_state_sha256=verified_claimed.state.record_sha256,
            phase_claim_ledger_commit=evidence.transition_id,
            claim_attested_at_utc=evidence.integrated_at_utc,
            live_execute_job_receipt=live_execute_job_receipt,
            phase_beacon_receipt=None,
            fresh_revalidator=_phase_claim_revalidator(
                verified_claimed,
                expected_state="ANALYSIS_CLAIMED",
                local_type=VerifiedSuiteAnalysisClaimed,
                fresh_state_revalidator=fresh_state_revalidator,
            ),
        )
    except ExecutionClaimError as exc:
        raise SuiteAttemptError(f"cannot mint analysis authority: {exc}") from exc


def verify_suite_state(
    namespace: str | Path,
    *,
    verifier: SuiteEvidenceVerifier,
    expected_state: SuiteState | None = None,
) -> VerifiedSuiteState:
    """Load the canonical chain and verify every provider attestation from bytes."""

    root = _namespace_path(namespace)
    descriptor = _load_descriptor(_descriptor_path(root))
    records: list[SuiteStateRecord] = []
    evidences: list[SuiteAttestationEvidence] = []
    sequence = 0
    while _state_path(root, sequence).exists():
        record = load_suite_state_record(_state_path(root, sequence))
        if record.sequence != sequence:
            raise SuiteAttemptError("suite state filename and sequence differ")
        if record.namespace_uri != root.as_uri():
            raise SuiteAttemptError("suite state record binds another namespace")
        expected_attempt_id = suite_attempt_id(record.manifest_sha256)
        if record.suite_attempt_id != expected_attempt_id:
            raise SuiteAttemptError("suite attempt ID is not manifest-derived")
        if root.name != f"suite-attempt-{expected_attempt_id}":
            raise SuiteAttemptError("suite namespace name is not manifest-derived")
        if records:
            _assert_state_transition(records[-1], record)
        elif not isinstance(record.payload, SuiteOpenBindings):
            raise SuiteAttemptError("suite chain does not begin with OPENED")
        evidence = _verify_one_attestation(
            namespace=root,
            state=record,
            descriptor=descriptor,
            verifier=verifier,
            previous_evidence=None if not evidences else evidences[-1],
        )
        records.append(record)
        evidences.append(evidence)
        sequence += 1
        if record.state == "FAILED":
            break
    if not records:
        raise SuiteAttemptError("suite namespace has no canonical state files")
    observed_state_files = {
        path.name
        for path in root.iterdir()
        if re.fullmatch(r"[0-9]{3}\.state\.json", path.name) is not None
    }
    expected_state_files = {_state_path(root, record.sequence).name for record in records}
    if observed_state_files != expected_state_files:
        raise SuiteAttemptError("suite state files contain a gap or post-terminal transition")
    final = records[-1]
    if expected_state is not None and final.state != expected_state:
        raise SuiteAttemptError(
            f"suite state is {final.state}, expected externally attested {expected_state}"
        )
    if isinstance(records[0].payload, SuiteOpenBindings):
        if records[0].payload.attestation_descriptor_sha256 != descriptor.descriptor_sha256:
            raise SuiteAttemptError("OPENED binds another attestation descriptor")
        floor = max(
            _timestamp("registered_at_utc", records[0].payload.registered_at_utc),
            _timestamp("run_started_at_utc", records[0].payload.run_started_at_utc),
        )
    else:  # pragma: no cover - protected above
        raise SuiteAttemptError("suite chain lacks OPENED bindings")
    previous_integrated: datetime | None = None
    previous_log_index: int | None = None
    transition_ids: set[str] = set()
    log_entry_ids: set[str] = set()
    for evidence in evidences:
        signed = _timestamp("signed_at_utc", evidence.signed_at_utc)
        integrated = _timestamp("integrated_at_utc", evidence.integrated_at_utc)
        if signed < floor or integrated < signed:
            raise SuiteAttemptError("provider evidence is backdated or integrated before signing")
        if previous_integrated is not None and signed < previous_integrated:
            raise SuiteAttemptError("provider transition timestamp predates its predecessor")
        if previous_log_index is not None and evidence.transparency_log_index <= previous_log_index:
            raise SuiteAttemptError("transparency log indices are not strictly increasing")
        if evidence.transition_id in transition_ids:
            raise SuiteAttemptError("provider transition IDs are not unique")
        if evidence.transparency_entry_id in log_entry_ids:
            raise SuiteAttemptError("transparency entry IDs are not unique")
        transition_ids.add(evidence.transition_id)
        log_entry_ids.add(evidence.transparency_entry_id)
        previous_integrated = integrated
        previous_log_index = evidence.transparency_log_index

    claim_records = [record for record in records if record.state == "RUN_CLAIMED"]
    if len(claim_records) > 1:
        raise SuiteAttemptError("suite chain repeats RUN_CLAIMED")
    if claim_records:
        claim = claim_records[0]
        if claim.sequence != 1 or not isinstance(claim.payload, RunClaimBindings):
            raise SuiteAttemptError("RUN_CLAIMED is not the sole successor of OPENED")
        claim_evidence = evidences[claim.sequence]
        signed = _timestamp("RUN_CLAIMED signed_at_utc", claim_evidence.signed_at_utc)
        integrated = _timestamp(
            "RUN_CLAIMED integrated_at_utc",
            claim_evidence.integrated_at_utc,
        )
        prerequisites = (
            _timestamp(
                "C1 manifest Rekor time",
                claim.payload.c1_manifest_rekor_integrated_at_utc,
            ),
            _timestamp(
                "C1 registry Rekor time",
                claim.payload.c1_registry_rekor_integrated_at_utc,
            ),
            _timestamp(
                "Zenodo publication time",
                claim.payload.zenodo_admission.published_at_utc,
            ),
        )
        if signed < max(prerequisites) or integrated < max(prerequisites):
            raise SuiteAttemptError("RUN_CLAIMED predates C1 or public Zenodo admission")
        publication = claim.payload.execution_claim.beacon.execution_publication_time
        if signed >= publication or integrated >= publication:
            raise SuiteAttemptError("RUN_CLAIMED provider evidence is not pre-beacon")
    elif any(record.state != "OPENED" for record in records):
        raise SuiteAttemptError("suite advanced beyond OPENED without RUN_CLAIMED")

    phase_claim_states = {
        "LABEL_RELEASE_CLAIMED": ("ONLINE_COMPLETE", "label-release"),
        "ANALYSIS_CLAIMED": ("LABELS_RELEASED", "analysis"),
    }
    for state_name, (predecessor_name, phase_name) in phase_claim_states.items():
        matches = [record for record in records if record.state == state_name]
        if len(matches) > 1:
            raise SuiteAttemptError(f"suite chain repeats {state_name}")
        progressed = any(
            record.state
            in (
                {"LABELS_RELEASED", "ANALYSIS_CLAIMED", "ANALYSIS_COMPLETE"}
                if state_name == "LABEL_RELEASE_CLAIMED"
                else {"ANALYSIS_COMPLETE"}
            )
            for record in records
        )
        if not matches:
            if progressed:
                raise SuiteAttemptError(f"suite advanced without {state_name}")
            continue
        phase_state = matches[0]
        if not isinstance(phase_state.payload, PhaseClaimBindings):
            raise SuiteAttemptError(f"{state_name} payload is malformed")
        if phase_state.sequence == 0 or records[phase_state.sequence - 1].state != predecessor_name:
            raise SuiteAttemptError(f"{state_name} is not the sole successor of {predecessor_name}")
        predecessor_evidence = evidences[phase_state.sequence - 1]
        phase_evidence = evidences[phase_state.sequence]
        contract = phase_state.payload.phase_claim
        if (
            contract.phase != phase_name
            or contract.predecessor_state_sha256 != records[phase_state.sequence - 1].record_sha256
            or contract.predecessor_ledger_commit != predecessor_evidence.transition_id
            or _GIT_COMMIT.fullmatch(phase_evidence.transition_id) is None
        ):
            raise SuiteAttemptError(f"{state_name} changes its provider predecessor")
        if state_name == "LABEL_RELEASE_CLAIMED":
            assert contract.label_release_beacon is not None
            publication = contract.label_release_beacon.label_release_publication_time
            signed = _timestamp(f"{state_name} signed_at_utc", phase_evidence.signed_at_utc)
            integrated = _timestamp(
                f"{state_name} integrated_at_utc",
                phase_evidence.integrated_at_utc,
            )
            if signed >= publication or integrated >= publication:
                raise SuiteAttemptError("LABEL_RELEASE_CLAIMED provider evidence is not pre-beacon")

    if final.state == "FAILED":
        if not isinstance(final.payload, ProviderPhaseFailure) or final.sequence == 0:
            raise SuiteAttemptError("terminal FAILED lacks typed provider evidence")
        predecessor = records[final.sequence - 1]
        predecessor_evidence = evidences[final.sequence - 1]
        if (
            predecessor.state not in {"RUN_CLAIMED", "LABEL_RELEASE_CLAIMED", "ANALYSIS_CLAIMED"}
            or final.payload.claim_state_sha256 != predecessor.record_sha256
            or final.payload.claim_ledger_commit != predecessor_evidence.transition_id
        ):
            raise SuiteAttemptError("FAILED does not consume the live winning provider claim")

    opened_payload = records[0].payload
    assert isinstance(opened_payload, SuiteOpenBindings)
    _, finalization_path = _local_file_uri(
        "production_finalization_receipt_uri",
        opened_payload.production_finalization_receipt_uri,
    )
    finalization_file_sha256 = _verify_opened_finalization_binding(
        finalization_path,
        opened=records[0],
        namespace=root,
    )
    if finalization_file_sha256 != opened_payload.production_finalization_receipt_file_sha256:
        raise SuiteAttemptError("OPENED production finalization receipt changed")

    external_files: list[tuple[str, str]] = [(str(finalization_path), finalization_file_sha256)]
    label_records = [record for record in records if record.state == "LABELS_RELEASED"]
    if len(label_records) > 1:
        raise SuiteAttemptError("suite chain repeats LABELS_RELEASED")
    if label_records:
        labels = label_records[0].payload
        if not isinstance(labels, tuple):
            raise SuiteAttemptError("LABELS_RELEASED payload is malformed")
        for closure in labels:
            receipt_path = _local_file_uri(
                "decryption_receipt_uri",
                closure.decryption_receipt_uri,
            )[1]
            plaintext_path = _local_file_uri("plaintext_uri", closure.plaintext_uri)[1]
            receipt_digest = digest_regular_file(
                receipt_path,
                label=f"{closure.corpus_id} decryption receipt",
            )
            plaintext_digest = digest_regular_file(
                plaintext_path,
                label=f"{closure.corpus_id} released plaintext",
            )
            if (
                receipt_digest != closure.decryption_receipt_file_sha256
                or _regular_file_size(
                    receipt_path,
                    label=f"{closure.corpus_id} decryption receipt",
                )
                != closure.decryption_receipt_byte_count
                or plaintext_digest != closure.plaintext_sha256
                or _regular_file_size(
                    plaintext_path,
                    label=f"{closure.corpus_id} released plaintext",
                )
                != closure.plaintext_byte_count
            ):
                raise SuiteAttemptError("LABELS_RELEASED files changed after publication")
            external_files.extend(
                (
                    (str(receipt_path), receipt_digest),
                    (str(plaintext_path), plaintext_digest),
                )
            )
    analysis_records = [record for record in records if record.state == "ANALYSIS_COMPLETE"]
    if len(analysis_records) > 1:
        raise SuiteAttemptError("suite chain repeats ANALYSIS_COMPLETE")
    if analysis_records:
        closure = analysis_records[0].payload
        if not isinstance(closure, AnalysisClosure):
            raise SuiteAttemptError("ANALYSIS_COMPLETE payload is malformed")
        analysis_files = (
            (
                "offline analysis execution receipt",
                closure.analysis_execution_receipt_uri,
                closure.analysis_execution_receipt_file_sha256,
                closure.analysis_execution_receipt_byte_count,
            ),
            (
                "analysis attempt receipt",
                closure.analysis_attempt_receipt_uri,
                closure.analysis_attempt_file_sha256,
                closure.analysis_attempt_byte_count,
            ),
            (
                "analysis result receipt",
                closure.analysis_result_receipt_uri,
                closure.analysis_result_receipt_file_sha256,
                closure.analysis_result_receipt_byte_count,
            ),
            (
                "final analysis result",
                closure.final_result_uri,
                closure.final_result_file_sha256,
                closure.final_result_byte_count,
            ),
        )
        for label, uri, expected_digest, expected_size in analysis_files:
            path = _local_file_uri(label, uri)[1]
            observed_digest = digest_regular_file(path, label=label)
            if (
                observed_digest != expected_digest
                or _regular_file_size(path, label=label) != expected_size
            ):
                raise SuiteAttemptError(f"{label} changed after publication")
            external_files.append((str(path), observed_digest))
    if final.state == "FAILED":
        assert isinstance(final.payload, ProviderPhaseFailure)
        _, incident_path = _local_file_uri("incident_uri", final.payload.incident_uri)
        incident_digest = digest_regular_file(incident_path, label="provider failure incident")
        try:
            incident_size = incident_path.stat().st_size
        except OSError as exc:
            raise SuiteAttemptError("cannot stat provider failure incident") from exc
        if (
            incident_digest != final.payload.incident_file_sha256
            or incident_size != final.payload.incident_byte_count
        ):
            raise SuiteAttemptError("provider failure incident bytes differ")
        external_files.append((str(incident_path), incident_digest))
        for partial in final.payload.partial_evidence:
            partial_path = incident_path.parent / partial.relative_path
            partial_digest = digest_regular_file(
                partial_path,
                label=f"provider partial evidence {partial.relative_path}",
            )
            try:
                partial_size = partial_path.stat().st_size
            except OSError as exc:
                raise SuiteAttemptError("cannot stat provider partial evidence") from exc
            if partial_digest != partial.file_sha256 or partial_size != partial.byte_count:
                raise SuiteAttemptError("provider partial evidence bytes differ")
            external_files.append((str(partial_path), partial_digest))
    tree_sha256s: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    online_records = [record for record in records if record.state == "ONLINE_COMPLETE"]
    if len(online_records) > 1:
        raise SuiteAttemptError("suite chain repeats ONLINE_COMPLETE")
    if online_records:
        if not claim_records:
            raise SuiteAttemptError("ONLINE_COMPLETE lacks RUN_CLAIMED")
        opened = records[0]
        claim = claim_records[0]
        assert isinstance(opened.payload, SuiteOpenBindings)
        assert isinstance(claim.payload, RunClaimBindings)
        online_payload = online_records[0].payload
        assert isinstance(online_payload, OnlineSuiteClosure)
        if (
            online_payload.run_output_aggregate.claim_state_sha256 != claim.record_sha256
            or online_payload.run_output_aggregate.claim_ledger_commit
            != evidences[claim.sequence].transition_id
            or online_payload.run_output_aggregate.provider_identity_sha256
            != claim.payload.provider_identity.identity_sha256
        ):
            raise SuiteAttemptError("ONLINE_COMPLETE aggregate changes the provider claim")
        executions = {row.corpus_id: row.sha256 for row in opened.payload.execution_artifacts}
        staging = {row.corpus_id: row.output_uri for row in opened.payload.staging_namespaces}
        plans = {row.corpus_id: row for row in opened.payload.runtime_attestation_plans}
        for closure in online_payload.corpora:
            plan = plans[closure.corpus_id]
            if (
                closure.execution_artifact_sha256 != executions[closure.corpus_id]
                or closure.staging_output_uri != staging[closure.corpus_id]
                or closure.sealed_launch_copy_output_uri != staging[closure.corpus_id]
                or closure.sealed_launch_contract_sha256 != plan.sealed_launch_contract_sha256
            ):
                raise SuiteAttemptError(
                    "ONLINE_COMPLETE corpus closure differs from OPENED after RUN_CLAIMED"
                )
        transfer_path, tree_sha256s = _verify_online_transfer_state(
            root,
            records[0],
            online_records[0],
        )
        transfer_payload = online_records[0].payload
        assert isinstance(transfer_payload, OnlineSuiteClosure)
        external_files.append(
            (str(transfer_path), transfer_payload.output_transfer_receipt_file_sha256)
        )
    else:
        try:
            online_placeholder = digest_directory_tree(root / "online")
        except ArtifactIntegrityError as exc:
            raise SuiteAttemptError(
                f"cannot verify empty canonical online placeholder: {exc}"
            ) from exc
        if online_placeholder.entries:
            raise SuiteAttemptError("canonical online tree contains files before ONLINE_COMPLETE")
        tree_sha256s = ()

    tracked = ["attestation-descriptor.json"]
    for record in records:
        tracked.extend(
            (
                _state_path(root, record.sequence).name,
                _evidence_path(root, record.sequence).name,
                _bundle_path(root, record.sequence).name,
            )
        )
    file_sha256s = tuple(
        (name, digest_regular_file(root / name, label=f"suite verification file {name}"))
        for name in tracked
    ) + tuple(external_files)
    expected_namespace_names = {
        "attestation-descriptor.json",
        "online",
        *(name for name, _ in file_sha256s if not Path(name).is_absolute()),
    }
    try:
        observed_namespace_names = {path.name for path in root.iterdir()}
    except OSError as exc:
        raise SuiteAttemptError("cannot inspect canonical suite namespace") from exc
    if observed_namespace_names != expected_namespace_names:
        raise SuiteAttemptError("canonical suite namespace contains an extra or missing member")
    token_type: type[VerifiedSuiteState]
    token_type = {
        "OPENED": VerifiedSuiteOpened,
        "RUN_CLAIMED": VerifiedSuiteRunClaimed,
        "ONLINE_COMPLETE": VerifiedSuiteOnlineCompletion,
        "LABEL_RELEASE_CLAIMED": VerifiedSuiteLabelReleaseClaimed,
        "LABELS_RELEASED": VerifiedSuiteLabelsReleased,
        "ANALYSIS_CLAIMED": VerifiedSuiteAnalysisClaimed,
        "ANALYSIS_COMPLETE": VerifiedSuiteAnalysisComplete,
        "FAILED": VerifiedSuiteFailed,
    }[final.state]
    return token_type(
        namespace=root,
        records=tuple(records),
        evidences=tuple(evidences),
        descriptor_sha256=descriptor.descriptor_sha256,
        _file_sha256s=file_sha256s,
        _capability=_STATE_CAPABILITY,
        _tree_sha256s=tree_sha256s,
    )


def require_verified_online_completion(
    token: object,
    *,
    manifest_digest: str,
    corpus_id: str,
    online_result_receipt_sha256: str,
) -> OnlineCorpusClosure:
    """Revalidate the all-five online closure at the label-release boundary."""

    if isinstance(token, VerifiedProviderPredecessor):
        if token.state.state != "LABEL_RELEASE_CLAIMED":
            raise SuiteAttemptError(
                "provider label release requires verified LABEL_RELEASE_CLAIMED lineage"
            )
        matches = [record for record in token.records if record.state == "ONLINE_COMPLETE"]
        if len(matches) != 1:
            raise SuiteAttemptError("provider label-release chain lacks one ONLINE_COMPLETE state")
        online_state = matches[0]
        token.assert_current()
        if online_state.manifest_sha256 != manifest_digest:
            raise SuiteAttemptError("verified suite completion belongs to another manifest")
        if not isinstance(online_state.payload, OnlineSuiteClosure):
            raise SuiteAttemptError("verified ONLINE_COMPLETE payload is malformed")
        corpora = [row for row in online_state.payload.corpora if row.corpus_id == corpus_id]
        if len(corpora) != 1 or not isinstance(corpora[0], OnlineCorpusClosure):
            raise SuiteAttemptError("verified suite completion lacks the requested corpus")
        closure = corpora[0]
        if closure.result_receipt_sha256 != online_result_receipt_sha256:
            raise SuiteAttemptError("verified suite completion binds another online result")
        return closure
    if not isinstance(token, VerifiedSuiteOnlineCompletion):
        raise SuiteAttemptError(
            "label release requires externally verified ONLINE_COMPLETE canonical files"
        )
    return token.require_corpus(
        manifest_digest=manifest_digest,
        corpus_id=corpus_id,
        online_result_receipt_sha256=online_result_receipt_sha256,
    )


def require_verified_labels_released(
    token: object,
    *,
    manifest_digest: str,
    run_receipt_sha256: str,
) -> None:
    """Validate the file-backed all-five label state before analysis admission."""

    if isinstance(token, VerifiedProviderPredecessor):
        if token.state.state != "ANALYSIS_CLAIMED":
            raise SuiteAttemptError(
                "provider analysis requires verified ANALYSIS_CLAIMED canonical files"
            )
        matches = [record for record in token.records if record.state == "LABELS_RELEASED"]
        if len(matches) != 1:
            raise SuiteAttemptError("provider analysis chain lacks one LABELS_RELEASED state")
        labels_state = matches[0]
    elif isinstance(token, VerifiedSuiteLabelsReleased):
        labels_state = token.state
    else:
        raise SuiteAttemptError(
            "confirmatory analysis requires externally verified LABELS_RELEASED canonical files"
        )
    token.assert_current()
    if (
        labels_state.manifest_sha256 != manifest_digest
        or labels_state.run_receipt_sha256 != run_receipt_sha256
    ):
        raise SuiteAttemptError("LABELS_RELEASED token belongs to another sealed run")


def complete_label_release(
    verified_claimed: VerifiedSuiteLabelReleaseClaimed | VerifiedProviderPredecessor,
    *,
    phase_claim: VerifiedPhaseClaimCapability,
    phase_claim_factory: Callable[[], VerifiedPhaseClaimCapability] | None = None,
    manifest: Mapping[str, Any],
    decryption_receipt_paths: Mapping[str, str | Path],
    plaintext_paths: Mapping[str, str | Path],
    post_online_completion_aggregate_file_sha256: str,
    label_release_authorities: Mapping[str, object],
) -> SuiteStateRecord:
    """Bind five verified release receipts and exact frozen plaintext files."""

    if isinstance(verified_claimed, VerifiedProviderPredecessor):
        if verified_claimed.state.state != "LABEL_RELEASE_CLAIMED":
            raise SuiteAttemptError(
                "provider LABELS_RELEASED requires verified LABEL_RELEASE_CLAIMED"
            )
    elif not isinstance(verified_claimed, VerifiedSuiteLabelReleaseClaimed):
        raise SuiteAttemptError("LABELS_RELEASED requires verified LABEL_RELEASE_CLAIMED")
    if not isinstance(phase_claim, VerifiedPhaseClaimCapability):
        raise SuiteAttemptError("LABELS_RELEASED requires typed label-release authority")
    if phase_claim_factory is not None and not callable(phase_claim_factory):
        raise SuiteAttemptError("label-release authority factory must be callable")
    phase_claim.assert_current()
    if (
        phase_claim.contract.phase != "label-release"
        or phase_claim.phase_claim_state_sha256 != verified_claimed.state.record_sha256
    ):
        raise SuiteAttemptError("label-release authority differs from the winning claim")
    if set(decryption_receipt_paths) != set(FIXED_CORPORA) or len(decryption_receipt_paths) != len(
        FIXED_CORPORA
    ):
        raise SuiteAttemptError("LABELS_RELEASED requires one receipt for each fixed corpus")
    if set(plaintext_paths) != set(FIXED_CORPORA) or len(plaintext_paths) != len(FIXED_CORPORA):
        raise SuiteAttemptError("LABELS_RELEASED requires one plaintext file for each fixed corpus")
    _digest(
        "post_online_completion_aggregate_file_sha256",
        post_online_completion_aggregate_file_sha256,
    )
    if set(label_release_authorities) != set(FIXED_CORPORA) or len(
        label_release_authorities
    ) != len(FIXED_CORPORA):
        raise SuiteAttemptError(
            "LABELS_RELEASED requires one action authority for each fixed corpus"
        )
    try:
        validate_study_manifest(manifest, require_frozen=True)
    except ValueError as exc:
        raise SuiteAttemptError(f"invalid frozen study manifest: {exc}") from exc
    if manifest_sha256(manifest) != verified_claimed.state.manifest_sha256:
        raise SuiteAttemptError("LABELS_RELEASED manifest differs from its winning claim")
    plaintext_pins = _corpus_manifest_artifact_sha256(
        manifest,
        role="sealed-labels",
    )
    ciphertext_pins = _corpus_manifest_artifact_sha256(
        manifest,
        role="sealed-label-ciphertext",
    )
    encryption_receipt_pins = _corpus_manifest_artifact_sha256(
        manifest,
        role="timelock-encryption-receipt",
    )
    timelock_tool_pin = _singleton_manifest_artifact_sha256(
        manifest,
        role="timelock-tool",
    )
    from .timelock_release import load_timelock_decryption_receipt

    ordered = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
    online_records = [
        record for record in verified_claimed.records if record.state == "ONLINE_COMPLETE"
    ]
    if len(online_records) != 1 or not isinstance(online_records[0].payload, OnlineSuiteClosure):
        raise SuiteAttemptError("verified label claim lacks one ONLINE_COMPLETE payload")
    online_payload = online_records[0].payload
    online_rows = {row.corpus_id: row for row in online_payload.corpora}
    claim_rows = {row.corpus_id: row for row in phase_claim.contract.corpora}
    from .provider_phase_runtime import LabelReleaseOutputAuthority

    action_authorities: dict[str, LabelReleaseOutputAuthority] = {}
    for corpus_id in ordered:
        authority = label_release_authorities[corpus_id]
        if not isinstance(authority, LabelReleaseOutputAuthority):
            raise SuiteAttemptError("LABELS_RELEASED action authority is not typed")
        if (
            authority.corpus_id != corpus_id
            or authority.post_online_completion_aggregate_file_sha256
            != post_online_completion_aggregate_file_sha256
            or authority.label_release_claim_state_sha256 != phase_claim.phase_claim_state_sha256
            or authority.label_release_claim_ledger_commit != phase_claim.phase_claim_ledger_commit
            or authority.label_release_phase_claim_contract_sha256
            != phase_claim.contract.contract_sha256
            or authority.label_release_provider_identity_sha256
            != phase_claim.provider_identity.identity_sha256
        ):
            raise SuiteAttemptError(
                "LABELS_RELEASED action authority differs from the winning claim"
            )
        action_authorities[corpus_id] = authority

    last_job_observation = _timestamp(
        "initial label live-job verification",
        phase_claim.live_execute_job_receipt.verified_at_utc,
    )
    initial_beacon = phase_claim.phase_beacon_receipt
    if initial_beacon is None:
        raise SuiteAttemptError("label-release authority lacks its beacon receipt")
    last_beacon_observation = _timestamp(
        "initial label beacon verification",
        initial_beacon.verified_at_utc,
    )

    def current_phase_claim() -> VerifiedPhaseClaimCapability:
        nonlocal last_beacon_observation, last_job_observation
        current = phase_claim if phase_claim_factory is None else phase_claim_factory()
        if not isinstance(current, VerifiedPhaseClaimCapability):
            raise SuiteAttemptError(
                "label-release authority factory returned an untyped capability"
            )
        current.assert_current()
        current_beacon = current.phase_beacon_receipt
        if (
            current.contract.contract_sha256 != phase_claim.contract.contract_sha256
            or current.provider_identity.identity_sha256
            != phase_claim.provider_identity.identity_sha256
            or current.phase_claim_state_sha256 != phase_claim.phase_claim_state_sha256
            or current.phase_claim_ledger_commit != phase_claim.phase_claim_ledger_commit
            or current.live_execute_job_receipt.job_identity_sha256
            != phase_claim.live_execute_job_receipt.job_identity_sha256
            or current_beacon is None
            or current_beacon.beacon_identity_sha256 != initial_beacon.beacon_identity_sha256
        ):
            raise SuiteAttemptError(
                "renewed label-release authority differs from the winning claim"
            )
        if phase_claim_factory is not None:
            current_job_observation = _timestamp(
                "renewed label live-job verification",
                current.live_execute_job_receipt.verified_at_utc,
            )
            current_beacon_observation = _timestamp(
                "renewed label beacon verification",
                current_beacon.verified_at_utc,
            )
            if (
                current_job_observation <= last_job_observation
                or current_beacon_observation <= last_beacon_observation
            ):
                raise SuiteAttemptError("renewed label-release observations are not newer")
            last_job_observation = current_job_observation
            last_beacon_observation = current_beacon_observation
        return current

    closures: list[LabelCorpusClosure] = []
    receipt_files = {corpus_id: Path(decryption_receipt_paths[corpus_id]) for corpus_id in ordered}
    plaintext_files = {corpus_id: Path(plaintext_paths[corpus_id]) for corpus_id in ordered}
    _assert_distinct_regular_files(
        tuple(
            [(f"{corpus_id} decryption receipt", receipt_files[corpus_id]) for corpus_id in ordered]
            + [
                (f"{corpus_id} released plaintext", plaintext_files[corpus_id])
                for corpus_id in ordered
            ]
        ),
        label="LABELS_RELEASED evidence",
    )
    for corpus_id in ordered:
        path = receipt_files[corpus_id]
        plaintext_path = plaintext_files[corpus_id]
        claim_row = claim_rows[corpus_id]
        if (
            claim_row.input_sha256 != ciphertext_pins[corpus_id]
            or claim_row.output_uri != plaintext_path.as_uri()
        ):
            raise SuiteAttemptError("label input or plaintext output differs from the claim")
        corpus_claim = current_phase_claim()
        corpus_claim.require_input(
            corpus_id=corpus_id,
            input_uri=claim_row.input_uri,
            input_sha256=ciphertext_pins[corpus_id],
            supporting_input_uri=claim_row.supporting_input_uri,
            supporting_input_sha256=claim_row.supporting_input_sha256,
        )
        corpus_beacon = corpus_claim.phase_beacon_receipt
        action_authority = action_authorities[corpus_id]
        if (
            corpus_beacon is None
            or action_authority.label_release_live_execute_job_receipt.job_identity_sha256
            != corpus_claim.live_execute_job_receipt.job_identity_sha256
            or action_authority.label_release_phase_beacon_receipt.beacon_identity_sha256
            != corpus_beacon.beacon_identity_sha256
        ):
            raise SuiteAttemptError("persisted label action evidence differs from fresh authority")
        receipt = load_timelock_decryption_receipt(path)
        try:
            decryption_receipt_byte_count = path.stat().st_size
        except OSError as exc:
            raise SuiteAttemptError(f"cannot stat {corpus_id} timelock decryption receipt") from exc
        if (
            receipt.manifest_sha256 != verified_claimed.state.manifest_sha256
            or receipt.corpus_id != corpus_id
            or receipt.online_execution_result_receipt_sha256
            != online_rows[corpus_id].result_receipt_sha256
            or receipt.plaintext_sha256 != plaintext_pins[corpus_id]
            or receipt.ciphertext_sha256 != ciphertext_pins[corpus_id]
            or receipt.timelock_encryption_receipt_file_sha256 != encryption_receipt_pins[corpus_id]
            or receipt.tle_binary_sha256 != timelock_tool_pin
            or receipt.post_online_completion_aggregate_file_sha256
            != post_online_completion_aggregate_file_sha256
            or receipt.label_release_claim_state_sha256
            != action_authority.label_release_claim_state_sha256
            or receipt.label_release_claim_ledger_commit
            != action_authority.label_release_claim_ledger_commit
            or receipt.label_release_phase_claim_contract_sha256
            != action_authority.label_release_phase_claim_contract_sha256
            or receipt.label_release_phase_beacon_receipt_sha256
            != action_authority.label_release_phase_beacon_receipt_sha256
            or receipt.label_release_live_execute_job_receipt_sha256
            != action_authority.label_release_live_execute_job_receipt_sha256
            or receipt.label_release_provider_identity_sha256
            != action_authority.label_release_provider_identity_sha256
        ):
            raise SuiteAttemptError(
                "decryption receipt differs from ONLINE_COMPLETE or frozen custody pins"
            )
        try:
            plaintext = read_secure_regular_file(
                plaintext_path,
                max_bytes=receipt.plaintext_byte_count,
                label=f"{corpus_id} released plaintext labels",
            )
        except ArtifactIntegrityError as exc:
            raise SuiteAttemptError(
                f"cannot verify {corpus_id} released plaintext labels: {exc}"
            ) from exc
        plaintext_sha256 = _sha256(plaintext)
        if (
            len(plaintext) != receipt.plaintext_byte_count
            or plaintext_sha256 != receipt.plaintext_sha256
        ):
            raise SuiteAttemptError(
                f"{corpus_id} released plaintext differs from its decryption receipt"
            )
        closures.append(
            LabelCorpusClosure(
                corpus_id=corpus_id,
                decryption_receipt_uri=path.as_uri(),
                decryption_receipt_sha256=receipt.receipt_sha256,
                decryption_receipt_file_sha256=digest_regular_file(
                    path,
                    label=f"{corpus_id} timelock decryption receipt",
                ),
                decryption_receipt_byte_count=decryption_receipt_byte_count,
                plaintext_uri=plaintext_path.as_uri(),
                plaintext_sha256=plaintext_sha256,
                plaintext_byte_count=len(plaintext),
            )
        )
    for closure in closures:
        receipt_path = receipt_files[closure.corpus_id]
        plaintext_path = plaintext_files[closure.corpus_id]
        if (
            digest_regular_file(
                receipt_path,
                label=f"{closure.corpus_id} final decryption receipt",
            )
            != closure.decryption_receipt_file_sha256
            or _regular_file_size(
                receipt_path,
                label=f"{closure.corpus_id} final decryption receipt",
            )
            != closure.decryption_receipt_byte_count
        ):
            raise SuiteAttemptError(
                f"{closure.corpus_id} decryption receipt changed before candidate creation"
            )
        try:
            final_plaintext = read_secure_regular_file(
                plaintext_path,
                max_bytes=closure.plaintext_byte_count,
                label=f"{closure.corpus_id} final released plaintext labels",
            )
        except ArtifactIntegrityError as exc:
            raise SuiteAttemptError(
                f"cannot rehash {closure.corpus_id} released plaintext labels: {exc}"
            ) from exc
        if (
            len(final_plaintext) != closure.plaintext_byte_count
            or _sha256(final_plaintext) != closure.plaintext_sha256
        ):
            raise SuiteAttemptError(
                f"{closure.corpus_id} released plaintext changed before candidate creation"
            )
    current_phase_claim().assert_current()
    verified_claimed.assert_current()
    payload = tuple(closures)
    if isinstance(verified_claimed, VerifiedProviderPredecessor):
        return _provider_candidate_transition(
            verified_claimed,
            state="LABELS_RELEASED",
            payload=payload,
        )
    return _write_transition(verified_claimed, state="LABELS_RELEASED", payload=payload)


def complete_confirmatory_analysis(
    verified_claimed: VerifiedSuiteAnalysisClaimed | VerifiedProviderPredecessor,
    *,
    phase_claim: VerifiedPhaseClaimCapability,
    confirmatory_input_artifact_sha256: str,
    execution_receipt_path: str | Path,
    execution_receipt_sha256: str,
    execution_receipt_file_sha256: str,
    attempt_receipt_path: str | Path,
    result_receipt_path: str | Path,
    final_result_path: str | Path,
) -> SuiteStateRecord:
    """Bind the persisted analysis input, attempt, receipt, and final result."""

    if isinstance(verified_claimed, VerifiedProviderPredecessor):
        if verified_claimed.state.state != "ANALYSIS_CLAIMED":
            raise SuiteAttemptError("provider ANALYSIS_COMPLETE requires verified ANALYSIS_CLAIMED")
    elif not isinstance(verified_claimed, VerifiedSuiteAnalysisClaimed):
        raise SuiteAttemptError("ANALYSIS_COMPLETE requires verified ANALYSIS_CLAIMED")
    if not isinstance(phase_claim, VerifiedPhaseClaimCapability):
        raise SuiteAttemptError("ANALYSIS_COMPLETE requires typed analysis authority")
    phase_claim.assert_current()
    if (
        phase_claim.contract.phase != "analysis"
        or phase_claim.phase_claim_state_sha256 != verified_claimed.state.record_sha256
    ):
        raise SuiteAttemptError("analysis authority differs from the winning claim")
    for row in phase_claim.contract.corpora:
        phase_claim.require_input(
            corpus_id=row.corpus_id,
            input_uri=row.input_uri,
            input_sha256=row.input_sha256,
            supporting_input_uri=row.supporting_input_uri,
            supporting_input_sha256=row.supporting_input_sha256,
        )
    from .confirmatory_execution import (
        load_confirmatory_analysis_attempt_receipt,
        load_confirmatory_analysis_result_receipt,
        load_confirmatory_result_artifact_bytes,
    )
    from .offline_analysis_contract import (
        OfflineAnalysisContractError,
        load_offline_analysis_execution_receipt,
    )

    execution_path = Path(execution_receipt_path)
    attempt_path = Path(attempt_receipt_path)
    receipt_path = Path(result_receipt_path)
    result_path = Path(final_result_path)
    if not all(
        path.is_absolute() for path in (execution_path, attempt_path, receipt_path, result_path)
    ):
        raise SuiteAttemptError("confirmatory analysis paths must be absolute")
    _assert_distinct_regular_files(
        (
            ("offline analysis execution receipt", execution_path),
            ("confirmatory analysis attempt receipt", attempt_path),
            ("confirmatory analysis result receipt", receipt_path),
            ("confirmatory final result", result_path),
        ),
        label="ANALYSIS_COMPLETE evidence",
    )
    try:
        execution = load_offline_analysis_execution_receipt(
            execution_path,
            expected_receipt_sha256=execution_receipt_sha256,
            expected_file_sha256=execution_receipt_file_sha256,
        )
    except OfflineAnalysisContractError as exc:
        raise SuiteAttemptError(f"offline analysis execution receipt is invalid: {exc}") from exc
    attempt = load_confirmatory_analysis_attempt_receipt(attempt_path)
    receipt = load_confirmatory_analysis_result_receipt(receipt_path)
    canonical_result = load_confirmatory_result_artifact_bytes(
        result_path,
        result_receipt_path=receipt_path,
        attempt_receipt_path=attempt_path,
    )
    try:
        execution_byte_count = execution_path.stat().st_size
        attempt_byte_count = attempt_path.stat().st_size
        receipt_byte_count = receipt_path.stat().st_size
        result_byte_count = result_path.stat().st_size
    except OSError as exc:
        raise SuiteAttemptError("cannot stat confirmatory analysis evidence") from exc
    input_digest = _digest(
        "confirmatory_input_artifact_sha256",
        confirmatory_input_artifact_sha256,
    )
    if (
        execution.suite_attempt_id != verified_claimed.state.suite_attempt_id
        or execution.manifest_sha256 != verified_claimed.state.manifest_sha256
        or execution.run_receipt_sha256 != verified_claimed.state.run_receipt_sha256
        or execution.provider_state_record_sha256 != verified_claimed.state.record_sha256
        or execution.provider_ledger_commit != verified_claimed.ledger_commit
        or execution.phase_claim_contract_sha256 != phase_claim.contract.contract_sha256
        or execution.phase_claim_state_sha256 != phase_claim.phase_claim_state_sha256
        or execution.phase_claim_ledger_commit != phase_claim.phase_claim_ledger_commit
        or execution.provider_identity_sha256 != phase_claim.provider_identity.identity_sha256
        or execution.attempt_uri != attempt_path.as_uri()
        or execution.attempt_receipt_sha256 != attempt.receipt_sha256
        or execution.attempt_file_sha256
        != digest_regular_file(
            attempt_path,
            label="confirmatory analysis attempt receipt",
        )
        or execution.result_receipt_uri != receipt_path.as_uri()
        or execution.result_receipt_sha256 != receipt.receipt_sha256
        or execution.result_receipt_file_sha256
        != digest_regular_file(
            receipt_path,
            label="confirmatory analysis result receipt",
        )
        or execution.result_uri != result_path.as_uri()
        or execution.result_artifact_sha256 != receipt.result_artifact_sha256
        or execution.result_file_sha256
        != digest_regular_file(
            result_path,
            label="confirmatory final result",
        )
        or attempt.manifest_sha256 != verified_claimed.state.manifest_sha256
        or attempt.run_receipt_sha256 != verified_claimed.state.run_receipt_sha256
        or attempt.confirmatory_input_artifact_sha256 != input_digest
        or receipt.attempt_receipt_sha256 != attempt.receipt_sha256
        or receipt.result_artifact_sha256 != _sha256(canonical_result)
    ):
        raise SuiteAttemptError("confirmatory analysis files differ from the admitted suite")
    closure = AnalysisClosure(
        confirmatory_input_artifact_sha256=input_digest,
        analysis_execution_receipt_uri=execution_path.as_uri(),
        analysis_execution_receipt_sha256=execution.receipt_sha256,
        analysis_execution_receipt_file_sha256=digest_regular_file(
            execution_path,
            label="offline analysis execution receipt",
        ),
        analysis_execution_receipt_byte_count=execution_byte_count,
        analysis_attempt_receipt_uri=attempt_path.as_uri(),
        analysis_attempt_receipt_sha256=attempt.receipt_sha256,
        analysis_attempt_file_sha256=digest_regular_file(
            attempt_path,
            label="confirmatory analysis attempt receipt",
        ),
        analysis_attempt_byte_count=attempt_byte_count,
        analysis_result_receipt_uri=receipt_path.as_uri(),
        analysis_result_receipt_sha256=receipt.receipt_sha256,
        analysis_result_receipt_file_sha256=digest_regular_file(
            receipt_path,
            label="confirmatory analysis result receipt",
        ),
        analysis_result_receipt_byte_count=receipt_byte_count,
        final_result_uri=result_path.as_uri(),
        final_result_artifact_sha256=receipt.result_artifact_sha256,
        final_result_file_sha256=digest_regular_file(
            result_path,
            label="confirmatory final result",
        ),
        final_result_byte_count=result_byte_count,
    )
    if isinstance(verified_claimed, VerifiedProviderPredecessor):
        return _provider_candidate_transition(
            verified_claimed,
            state="ANALYSIS_COMPLETE",
            payload=closure,
        )
    return _write_transition(verified_claimed, state="ANALYSIS_COMPLETE", payload=closure)


def fail_suite_attempt(
    verified_state: VerifiedSuiteState | VerifiedProviderPredecessor,
    *,
    provider_failure: ProviderPhaseFailure,
    failed_execute_job_receipt: FailedExecuteJobReceipt,
) -> SuiteStateRecord:
    """Consume a won provider claim with exact retained failure evidence."""

    if not isinstance(verified_state, (VerifiedSuiteState, VerifiedProviderPredecessor)):
        raise SuiteAttemptError("FAILED requires a verified predecessor state")
    if not isinstance(provider_failure, ProviderPhaseFailure):
        raise SuiteAttemptError("FAILED requires typed provider failure evidence")
    if not isinstance(failed_execute_job_receipt, FailedExecuteJobReceipt):
        raise SuiteAttemptError("FAILED requires live execute-job failure evidence")
    verified_state.assert_current()
    predecessor = verified_state.state
    if predecessor.state in {"ANALYSIS_COMPLETE", "FAILED"}:
        raise SuiteAttemptError(f"{predecessor.state} is terminal")
    if predecessor.state == "RUN_CLAIMED" and isinstance(predecessor.payload, RunClaimBindings):
        phase = "online"
        provider_digest = predecessor.payload.provider_identity.identity_sha256
        phase_input = predecessor.payload.execution_claim.contract_sha256
    elif predecessor.state in {
        "LABEL_RELEASE_CLAIMED",
        "ANALYSIS_CLAIMED",
    } and isinstance(predecessor.payload, PhaseClaimBindings):
        phase = predecessor.payload.phase_claim.phase
        provider_digest = predecessor.payload.provider_identity.identity_sha256
        phase_input = predecessor.payload.phase_claim.contract_sha256
    else:
        raise SuiteAttemptError("FAILED can consume only a live provider phase claim")
    if (
        provider_failure.phase != phase
        or provider_failure.claim_state_sha256 != predecessor.record_sha256
        or provider_failure.claim_ledger_commit != verified_state.evidences[-1].transition_id
        or provider_failure.provider_identity_sha256 != provider_digest
        or provider_failure.phase_input_sha256 != phase_input
        or provider_failure.failed_execute_job_receipt_sha256
        != failed_execute_job_receipt.receipt_sha256
        or provider_failure.execute_job_id != failed_execute_job_receipt.execute_job_id
        or failed_execute_job_receipt.provider_identity_sha256 != provider_digest
    ):
        raise SuiteAttemptError("FAILED evidence differs from the live winning claim")
    _, incident_path = _local_file_uri("incident_uri", provider_failure.incident_uri)
    incident_digest = digest_regular_file(incident_path, label="provider failure incident")
    try:
        incident_size = incident_path.stat().st_size
    except OSError as exc:
        raise SuiteAttemptError("cannot stat provider failure incident") from exc
    if (
        incident_digest != provider_failure.incident_file_sha256
        or incident_size != provider_failure.incident_byte_count
    ):
        raise SuiteAttemptError("provider failure incident bytes differ")
    for partial in provider_failure.partial_evidence:
        partial_path = incident_path.parent / partial.relative_path
        if (
            digest_regular_file(partial_path, label="provider partial failure evidence")
            != partial.file_sha256
            or partial_path.stat().st_size != partial.byte_count
        ):
            raise SuiteAttemptError("provider partial failure evidence bytes differ")
    if isinstance(verified_state, VerifiedProviderPredecessor):
        return _provider_candidate_transition(
            verified_state,
            state="FAILED",
            payload=provider_failure,
        )
    return _write_transition(verified_state, state="FAILED", payload=provider_failure)


def fail_provider_candidate(
    predecessor: VerifiedProviderPredecessor,
    *,
    provider_failure: ProviderPhaseFailure,
    failed_execute_job_receipt: FailedExecuteJobReceipt,
) -> SuiteStateRecord:
    """Build FAILED from stable provider claim evidence without local remapping."""

    if not isinstance(predecessor, VerifiedProviderPredecessor):
        raise SuiteAttemptError("provider FAILED requires verified provider predecessor")
    return fail_suite_attempt(
        predecessor,
        provider_failure=provider_failure,
        failed_execute_job_receipt=failed_execute_job_receipt,
    )
