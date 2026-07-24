"""GitHub-hosted CAS ledger and Sigstore verifier for the confirmatory suite.

The trusted workflow signs only a state record already present at the tip of a
manifest-derived, append-only Git branch.  This module reconstructs that branch
from GitHub's Git database, checks the closed state chain and branch controls,
and then asks ``gh attestation verify`` to validate the exact state bytes under
the C0 workflow identity.  The signed custom predicate is secondary evidence;
the state-record digest remains the attestation subject.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlsplit

from .c0_evidence_release import C0_EVIDENCE_RELEASE_TAG
from .c0_public_verification import (
    C0_PUBLIC_VERIFICATION_SCHEMA,
    C0PublicVerificationError,
    C0PublicVerificationReceipt,
    load_c0_public_verification_receipt,
)
from .c1_manifest_transition import (
    C1_MANIFEST_TRANSITION_RECEIPT_SCHEMA,
    C1ManifestTransitionError,
    loads_c1_manifest_transition_receipt,
    verify_c1_manifest_transition_receipt_bindings,
)
from .study import (
    FIXED_CORPORA,
    ProtocolRegistryRecord,
    load_study_manifest,
    manifest_sha256,
    validate_study_manifest,
)
from .suite_attempt import (
    SuiteAttemptError,
    SuiteAttestationDescriptor,
    SuiteAttestationEvidence,
    SuiteOpenBindings,
    SuiteProviderClaims,
    SuiteStateRecord,
    _assert_state_transition,
    load_suite_state_record,
)
from .suite_attempt import (
    suite_attempt_id as derive_suite_attempt_id,
)

REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
WORKFLOW_PATH = ".github/workflows/confirmatory-state-attestation.yml"
ONLINE_EXECUTION_WORKFLOW_PATH = ".github/workflows/confirmatory-online-execution.yml"
LABEL_RELEASE_WORKFLOW_PATH = ".github/workflows/confirmatory-label-release.yml"
ANALYSIS_WORKFLOW_PATH = ".github/workflows/confirmatory-analysis.yml"
C0_REF = "refs/tags/confirmatory-apparatus-c0"
OIDC_ISSUER = "https://token.actions.githubusercontent.com"
LEDGER_REF_PREFIX = "refs/heads/confirmatory-ledger"
LEDGER_PATH_PREFIX = "suite-attempts"
LEDGER_CONTROL_PREFIX = "suite-controls"
LEDGER_CONTROL_INVENTORY_SCHEMA = "fractal-ledger-control-inventory-v1"
LEDGER_RULESET_NAME = "confirmatory-ledger-append-only-v1"
LEDGER_RULESET_INCLUDE = "refs/heads/confirmatory-ledger/*"
LEDGER_RULE_TYPES = frozenset({"deletion", "non_fast_forward", "required_linear_history"})
LEDGER_PUBLISH_RECEIPT_SCHEMA = "fractal-github-ledger-publication-v1"
GIT_IDENTITY_NAME = "mhdk1602"
GIT_IDENTITY_EMAIL = "mhdk1602@users.noreply.github.com"
STATE_SERVICE_IDENTITY = "github-git-database"
STATE_SERVICE_URI = (
    "https://api.github.com/repos/mhdk1602/fractal-ann-diagnostics/git/refs/heads/"
    "confirmatory-ledger"
)
REKOR_IDENTITY = "sigstore-public-good-rekor"
REKOR_URI = "https://rekor.sigstore.dev/api/v1/log/entries"
PREDICATE_TYPE = (
    "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/confirmatory-state/v1"
)
PREDICATE_SCHEMA = "fractal-github-ledger-attestation-v1"
WORKFLOW_RECEIPT_SCHEMA = "fractal-github-ledger-workflow-receipt-v1"
C1_REF = "refs/tags/confirmatory-freeze-c1"
FREEZE_TAG_RULESET_NAME = "confirmatory-freeze-tags-immutable-v1"
FREEZE_TAG_RULESET_INCLUDES = (C0_REF, C1_REF)
FREEZE_TAG_RULE_TYPES = frozenset({"deletion", "non_fast_forward"})
C1_MANIFEST_PATH = "research/study-manifest.json"
C1_LOCK_PATH = "research/study-manifest.sha256"
C1_TRANSITION_RECEIPT_PATH = "research/manifest-transition-receipt.json"
C1_RESERVATION_PATH = "research/zenodo-reservation.json"
C0_PUBLIC_VERIFICATION_PATH = "c0-public-verification.json"
REGISTRATION_WORKFLOW_PATH = ".github/workflows/confirmatory-registration-attestation.yml"
REGISTRATION_PREDICATE_TYPE = (
    "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/prospective-c1-registration/v2"
)
REGISTRATION_PREDICATE_SCHEMA = "fractal-c1-registration-attestation-v2"
REGISTRATION_RECEIPT_SCHEMA = "fractal-c1-registration-workflow-receipt-v2"
REGISTRY_RECORD_SUBJECT_PATH = "protocol-registry-record.json"
REGISTRY_RECORD_PREDICATE_TYPE = (
    "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/"
    "prospective-c1-registry-record/v2"
)
REGISTRY_RECORD_PREDICATE_SCHEMA = "fractal-c1-registry-record-attestation-v2"
REGISTRY_MATERIALIZATION_SCHEMA = "fractal-c1-registry-record-materialization-v2"
REGISTRY_ATTESTATION_RECEIPT_SCHEMA = "fractal-c1-registry-record-verification-v2"
ZENODO_RECORD_ID = 21361837
ZENODO_RESERVED_DOI = "10.5281/zenodo.21361837"
ZENODO_REGISTRY_IDENTITY = "zenodo-record:21361837;zenodo-doi:10.5281/zenodo.21361837"
ZENODO_REGISTRY_URI = (
    "https://zenodo.org/api/records/21361837/files/protocol-registry-record.json/content"
)
ZENODO_DRAFT_URI = "https://zenodo.org/deposit/21361837"
ZENODO_RESERVATION_CREATED_AT_UTC = "2026-07-14T16:50:55.599204+00:00"
COMMON_CONTROL_LIMITATION = {
    "claim": "github-process-evidence-under-common-administration",
    "independent_organizational_custody": False,
    "same_administrator_controls": [
        "repository",
        "branch-protection",
        "workflow-dispatch",
        "evidence-retention",
        "verifier-policy",
    ],
}

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_IDENTITY_HEADER = re.compile(
    r"^" + re.escape(f"{GIT_IDENTITY_NAME} <{GIT_IDENTITY_EMAIL}>") + r" [0-9]+ [+-][0-9]{4}$"
)
_MAX_GH_OUTPUT_BYTES = 32 * 1024 * 1024
_MAX_REGISTRATION_BYTES = 16 * 1024 * 1024
_MAX_LEDGER_STATES = 8
_MAX_LEDGER_CONTROL_BYTES = 16 * 1024 * 1024


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
        raise SuiteAttemptError("GitHub attestation evidence must be canonical JSON") from exc


def _object(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise SuiteAttemptError(f"{label} must be a JSON object")
    return value


def _array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SuiteAttemptError(f"{label} must be a JSON array")
    return value


def _strict_json(encoded: bytes, *, label: str) -> object:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise SuiteAttemptError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise SuiteAttemptError(f"{label} contains non-finite number {value!r}")

    try:
        return json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuiteAttemptError(f"cannot decode {label}: {exc}") from exc


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SuiteAttemptError(f"{label} must be a canonical non-empty string")
    return value


def _oid(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if _SHA1.fullmatch(text) is None:
        raise SuiteAttemptError(f"{label} must be one lowercase SHA-1 Git object ID")
    return text


def _digest(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if _SHA256.fullmatch(text) is None:
        raise SuiteAttemptError(f"{label} must be one lowercase SHA-256 digest")
    return text


def _integer(value: object, *, label: str) -> int:
    if type(value) is int and value >= 0:
        return value
    if type(value) is str and value.isascii() and value.isdigit():
        return int(value)
    raise SuiteAttemptError(f"{label} must be a non-negative integer")


def _git_blob_oid(encoded: bytes) -> str:
    header = f"blob {len(encoded)}\0".encode("ascii")
    return hashlib.sha1(header + encoded, usedforsecurity=False).hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"})
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "--literal-pathspecs", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            env=environment,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SuiteAttemptError("cannot inspect the C1 Git freeze") from exc
    if result.returncode != 0:
        detail = result.stderr[:4096].decode("utf-8", errors="replace").strip()
        raise SuiteAttemptError(f"C1 Git inspection failed: {detail}")
    try:
        return result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise SuiteAttemptError("C1 Git metadata is not UTF-8") from exc


def _regular_file_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SuiteAttemptError(f"cannot open {label} without following links") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _MAX_REGISTRATION_BYTES
        ):
            raise SuiteAttemptError(f"{label} must be one bounded regular file")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_REGISTRATION_BYTES + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > _MAX_REGISTRATION_BYTES:
                raise SuiteAttemptError(f"{label} exceeds the byte limit")
        after = os.fstat(descriptor)
    except OSError as exc:
        raise SuiteAttemptError(f"cannot read {label}") from exc
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise SuiteAttemptError(f"{label} changed while it was read")
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SuiteAttemptError(f"{label} disappeared after it was read") from exc
    if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
        before.st_dev,
        before.st_ino,
    ):
        raise SuiteAttemptError(f"{label} was replaced while it was read")
    encoded = b"".join(chunks)
    if len(encoded) != before.st_size:
        raise SuiteAttemptError(f"{label} changed while it was read")
    return encoded


def _frozen_manifest_from_bytes(encoded: bytes) -> Mapping[str, Any]:
    """Validate only a private immutable snapshot of admitted manifest bytes."""

    with tempfile.TemporaryDirectory(prefix="fractal-c1-manifest-") as directory:
        snapshot = Path(directory) / Path(C1_MANIFEST_PATH).name
        _write_exclusive(snapshot, encoded)
        manifest = load_study_manifest(snapshot)
        validate_study_manifest(manifest, require_frozen=True)
        if _regular_file_bytes(snapshot, label="C1 manifest snapshot") != encoded:
            raise SuiteAttemptError("C1 manifest snapshot changed during validation")
    return manifest


def _admit_c0_public_verification(
    path: Path,
    *,
    frozen_manifest: Mapping[str, Any],
    frozen_manifest_bytes: bytes,
    c0_commit: str,
) -> tuple[C0PublicVerificationReceipt, bytes]:
    """Admit the durable public C0 receipt against the exact C1 source bytes."""

    try:
        receipt = load_c0_public_verification_receipt(path)
    except C0PublicVerificationError as exc:
        raise SuiteAttemptError(f"C0 public-verification receipt is invalid: {exc}") from exc
    encoded = _regular_file_bytes(path, label="C0 public-verification receipt")
    if encoded != receipt.canonical_file_bytes():
        raise SuiteAttemptError("C0 public-verification receipt changed during C1 admission")
    sealed = _object(
        frozen_manifest.get("sealed_execution"),
        label="frozen C1 sealed_execution",
    )
    binding = _object(
        sealed.get("c0_evidence_release"),
        label="frozen C1 C0 evidence-release binding",
    )
    manifest_file_digest = hashlib.sha256(frozen_manifest_bytes).hexdigest()
    if receipt.binding_source_kind != "frozen-manifest":
        raise SuiteAttemptError(
            "C0 public-verification receipt must use the frozen manifest as its source"
        )
    if receipt.binding_source_file_sha256 != manifest_file_digest:
        raise SuiteAttemptError(
            "C0 public-verification source digest differs from the frozen manifest file"
        )
    if receipt.target_commit != c0_commit:
        raise SuiteAttemptError("C0 public-verification target commit differs from C0")
    if receipt.release_tag != C0_EVIDENCE_RELEASE_TAG:
        raise SuiteAttemptError("C0 public-verification release tag differs from C0")
    if receipt.c0_evidence_release_binding != binding:
        raise SuiteAttemptError(
            "C0 public-verification binding differs from the frozen manifest binding"
        )
    return receipt, encoded


def _verify_c1_attestation_snapshot(
    *,
    subject_name: str,
    subject_bytes: bytes,
    bundle_bytes: bytes,
    c1_commit: str,
    predicate_type: str,
    verifier: C1AttestationVerifier,
) -> bytes:
    """Ask the provider verifier to inspect the exact bytes already admitted."""

    admitted = {
        C1_MANIFEST_PATH: REGISTRATION_PREDICATE_TYPE,
        REGISTRY_RECORD_SUBJECT_PATH: REGISTRY_RECORD_PREDICATE_TYPE,
    }
    if admitted.get(subject_name) != predicate_type:
        raise SuiteAttemptError("C1 attestation snapshot has another subject or predicate type")
    with tempfile.TemporaryDirectory(prefix="fractal-c1-attestation-") as directory:
        root = Path(directory)
        subject_path = root / Path(subject_name).name
        bundle_path = root / f"{Path(subject_name).name}.sigstore.bundle.json"
        _write_exclusive(subject_path, subject_bytes)
        _write_exclusive(bundle_path, bundle_bytes)
        verified = verifier.verify(
            subject_path=subject_path,
            bundle_path=bundle_path,
            c1_commit=c1_commit,
            predicate_type=predicate_type,
        )
        _validated_gh_output(verified)
        if (
            _regular_file_bytes(subject_path, label="C1 attestation subject snapshot")
            != subject_bytes
            or _regular_file_bytes(bundle_path, label="C1 attestation bundle snapshot")
            != bundle_bytes
        ):
            raise SuiteAttemptError("C1 attestation snapshot changed during verification")
    return verified


def _assert_exact_zenodo_url(value: object, *, expected: str, label: str) -> str:
    text = _text(value, label=label)
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise SuiteAttemptError(f"{label} is not a valid URL") from exc
    expected_path = urlsplit(expected).path
    if (
        text != expected
        or parsed.scheme != "https"
        or parsed.netloc != "zenodo.org"
        or parsed.hostname != "zenodo.org"
        or parsed.path != expected_path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
    ):
        raise SuiteAttemptError(f"{label} must name the fixed Zenodo host and record")
    return text


def _load_zenodo_reservation(path: Path) -> tuple[Mapping[str, Any], str]:
    encoded = _regular_file_bytes(path, label="Zenodo reservation")
    reservation = _object(
        _strict_json(encoded, label="Zenodo reservation"),
        label="Zenodo reservation",
    )
    expected = {
        "created_at_utc": ZENODO_RESERVATION_CREATED_AT_UTC,
        "creator": "mhdk1602",
        "deposition_id": ZENODO_RECORD_ID,
        "direct_registry_record_uri": ZENODO_REGISTRY_URI,
        "draft_uri": ZENODO_DRAFT_URI,
        "protocol_version": "0.3.0",
        "reserved_doi": ZENODO_RESERVED_DOI,
        "schema_version": "fractal-zenodo-reservation-v1",
        "state": "unsubmitted",
        "submitted": False,
    }
    if (
        type(reservation.get("deposition_id")) is not int
        or type(reservation.get("submitted")) is not bool
        or reservation != expected
        or encoded != _canonical_bytes(expected) + b"\n"
    ):
        raise SuiteAttemptError("Zenodo reservation differs from the fixed unpublished record")
    _assert_exact_zenodo_url(
        reservation.get("direct_registry_record_uri"),
        expected=ZENODO_REGISTRY_URI,
        label="Zenodo direct registry-record URI",
    )
    _assert_exact_zenodo_url(
        reservation.get("draft_uri"),
        expected=ZENODO_DRAFT_URI,
        label="Zenodo draft URI",
    )
    return reservation, hashlib.sha256(encoded).hexdigest()


def _state_from_bytes(encoded: bytes) -> SuiteStateRecord:
    value = _object(_strict_json(encoded, label="ledger state record"), label="ledger state")
    state = SuiteStateRecord.from_dict(value)
    if encoded != state.canonical_bytes() + b"\n":
        raise SuiteAttemptError("ledger state record is not exact canonical JSON plus one LF")
    return state


@dataclass(frozen=True)
class LedgerProtection:
    branch_protected: bool
    deletion_rule: bool
    non_fast_forward_rule: bool
    required_linear_history: bool
    ruleset_ids: tuple[int, ...]

    @classmethod
    def from_api(cls, branch_value: object, rules_value: object) -> LedgerProtection:
        branch = _object(branch_value, label="GitHub branch response")
        if branch.get("protected") is not True:
            raise SuiteAttemptError("GitHub ledger branch is not protected")
        rules = _array(rules_value, label="GitHub active branch rules")
        rule_types: set[str] = set()
        ruleset_ids: set[int] = set()
        for value in rules:
            rule = _object(value, label="GitHub active branch rule")
            rule_types.add(_text(rule.get("type"), label="GitHub branch rule type"))
            ruleset_id = rule.get("ruleset_id")
            if type(ruleset_id) is not int or ruleset_id <= 0:
                raise SuiteAttemptError("GitHub branch rule lacks a positive ruleset ID")
            ruleset_ids.add(ruleset_id)
        required = {"deletion", "non_fast_forward", "required_linear_history"}
        if not required.issubset(rule_types):
            raise SuiteAttemptError(
                "GitHub ledger rules must prevent deletion and non-fast-forward updates "
                "and require linear history"
            )
        return cls(
            branch_protected=True,
            deletion_rule=True,
            non_fast_forward_rule=True,
            required_linear_history=True,
            ruleset_ids=tuple(sorted(ruleset_ids)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "branch_protected": self.branch_protected,
            "deletion_rule": self.deletion_rule,
            "non_fast_forward_rule": self.non_fast_forward_rule,
            "required_linear_history": self.required_linear_history,
            "ruleset_ids": list(self.ruleset_ids),
        }


@dataclass(frozen=True)
class LedgerTransition:
    commit_oid: str
    previous_commit_oid: str | None
    tree_oid: str
    state_path: str
    state_bytes: bytes
    state: SuiteStateRecord


@dataclass(frozen=True)
class LedgerControlFile:
    """One non-secret genesis control retained in the protected ledger."""

    role: str
    ledger_path: str
    materialization_uri: str
    file_sha256: str
    byte_count: int
    blob_oid: str
    encoded: bytes

    def to_inventory_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "file_sha256": self.file_sha256,
            "ledger_path": self.ledger_path,
            "materialization_uri": self.materialization_uri,
            "role": self.role,
        }


@dataclass(frozen=True)
class LedgerSnapshot:
    repository: str
    state_key: str
    protection: LedgerProtection
    transitions: tuple[LedgerTransition, ...]
    controls: tuple[LedgerControlFile, ...] = ()
    control_inventory_bytes: bytes = b""

    @property
    def tip(self) -> LedgerTransition:
        return self.transitions[-1]


def _control_inventory_path(suite_attempt_id_value: str) -> str:
    return f"{LEDGER_CONTROL_PREFIX}/{suite_attempt_id_value}/inventory.json"


def _canonical_local_file_uri(value: str, *, label: str) -> Path:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise SuiteAttemptError(f"{label} is not one canonical local file URI")
    path = Path(unquote(parsed.path))
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or path.as_uri() != value
    ):
        raise SuiteAttemptError(f"{label} is not one canonical local file URI")
    return path


def _control_rows_from_opened(
    namespace: Path, opened: SuiteStateRecord
) -> tuple[tuple[str, str, Path], ...]:
    if opened.state != "OPENED" or not isinstance(opened.payload, SuiteOpenBindings):
        raise SuiteAttemptError("ledger controls require the typed OPENED state")
    descriptor = namespace / "attestation-descriptor.json"
    finalization = _canonical_local_file_uri(
        opened.payload.production_finalization_receipt_uri,
        label="OPENED production finalization receipt URI",
    )
    rows: list[tuple[str, str, Path]] = [
        ("attestation-descriptor", "attestation-descriptor.json", descriptor),
        (
            "production-finalization-receipt",
            "production-finalization-receipt.json",
            finalization,
        ),
    ]
    for binding in sorted(
        opened.payload.runtime_attestation_plans,
        key=lambda item: item.corpus_id.encode("utf-8"),
    ):
        rows.append(
            (
                f"sealed-launch-contract:{binding.corpus_id}",
                f"sealed-launch-contracts/{binding.corpus_id}.json",
                _canonical_local_file_uri(
                    binding.sealed_launch_contract_uri,
                    label=f"OPENED {binding.corpus_id} sealed-launch contract URI",
                ),
            )
        )
    return tuple(rows)


def _local_ledger_controls(
    namespace: Path, opened: SuiteStateRecord
) -> tuple[LedgerControlFile, ...]:
    attempt_id = opened.suite_attempt_id
    controls: list[LedgerControlFile] = []
    for role, relative, path in _control_rows_from_opened(namespace, opened):
        encoded = _regular_file_bytes(path, label=f"ledger control {role}")
        if not encoded or len(encoded) > _MAX_LEDGER_CONTROL_BYTES:
            raise SuiteAttemptError(f"ledger control {role} is empty or exceeds its bound")
        digest = hashlib.sha256(encoded).hexdigest()
        if role == "attestation-descriptor":
            descriptor = SuiteAttestationDescriptor.from_dict(
                _object(
                    _strict_json(encoded, label="suite attestation descriptor"),
                    label="suite attestation descriptor",
                )
            )
            if (
                encoded != descriptor.canonical_bytes() + b"\n"
                or descriptor.descriptor_sha256 != opened.payload.attestation_descriptor_sha256
            ):
                raise SuiteAttemptError("ledger descriptor differs from OPENED")
        elif role == "production-finalization-receipt":
            if digest != opened.payload.production_finalization_receipt_file_sha256:
                raise SuiteAttemptError("ledger finalization receipt differs from OPENED")
        else:
            corpus_id = role.partition(":")[2]
            matches = [
                row
                for row in opened.payload.runtime_attestation_plans
                if row.corpus_id == corpus_id
            ]
            if len(matches) != 1 or digest != matches[0].sealed_launch_contract_file_sha256:
                raise SuiteAttemptError("ledger sealed-launch contract differs from OPENED")
        ledger_path = f"{LEDGER_CONTROL_PREFIX}/{attempt_id}/{relative}"
        controls.append(
            LedgerControlFile(
                role=role,
                ledger_path=ledger_path,
                materialization_uri=path.as_uri(),
                file_sha256=digest,
                byte_count=len(encoded),
                blob_oid=_git_blob_oid(encoded),
                encoded=encoded,
            )
        )
    return tuple(controls)


def _ledger_control_inventory(
    suite_attempt_id_value: str,
    controls: Sequence[LedgerControlFile],
) -> bytes:
    ordered = tuple(sorted(controls, key=lambda item: item.ledger_path.encode("utf-8")))
    payload = {
        "controls": [item.to_inventory_dict() for item in ordered],
        "schema_version": LEDGER_CONTROL_INVENTORY_SCHEMA,
        "suite_attempt_id": suite_attempt_id_value,
    }
    return _canonical_bytes(payload) + b"\n"


class GitHubApi(Protocol):
    def get(self, endpoint: str) -> object: ...


class GitHubWriteApi(GitHubApi, Protocol):
    def post(self, endpoint: str, payload: Mapping[str, object]) -> object: ...

    def patch(self, endpoint: str, payload: Mapping[str, object]) -> object: ...


@dataclass(frozen=True)
class GhApiClient:
    executable: str = "gh"
    timeout_seconds: int = 30

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Mapping[str, object] | None = None,
    ) -> object:
        command = [
            self.executable,
            "api",
            "--hostname",
            "github.com",
            "--method",
            method,
            endpoint,
        ]
        encoded: bytes | None = None
        if payload is not None:
            command.extend(("--input", "-"))
            encoded = _canonical_bytes(payload)
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                input=encoded,
                env={**os.environ, "GH_PROMPT_DISABLED": "1", "NO_COLOR": "1"},
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SuiteAttemptError("cannot execute GitHub API client") from exc
        if result.returncode != 0:
            detail = result.stderr[:4096].decode("utf-8", errors="replace").strip()
            raise SuiteAttemptError(f"GitHub API rejected ledger verification: {detail}")
        if len(result.stdout) > _MAX_GH_OUTPUT_BYTES:
            raise SuiteAttemptError("GitHub API response exceeds the verifier limit")
        return _strict_json(result.stdout, label=f"GitHub API response for {endpoint}")

    def get(self, endpoint: str) -> object:
        return self._request("GET", endpoint)

    def post(self, endpoint: str, payload: Mapping[str, object]) -> object:
        return self._request("POST", endpoint, payload)

    def patch(self, endpoint: str, payload: Mapping[str, object]) -> object:
        return self._request("PATCH", endpoint, payload)


def _ruleset_id(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise SuiteAttemptError(f"{label} must be a positive integer")
    return value


def _validate_ledger_ruleset(
    value: object,
    *,
    require_bypass_visibility: bool,
) -> int:
    ruleset = _object(value, label="GitHub ledger ruleset")
    ruleset_id = _ruleset_id(ruleset.get("id"), label="GitHub ledger ruleset ID")
    if (
        ruleset.get("name") != LEDGER_RULESET_NAME
        or ruleset.get("target") != "branch"
        or ruleset.get("enforcement") != "active"
    ):
        raise SuiteAttemptError("GitHub ledger ruleset identity or enforcement differs")
    if "bypass_actors" not in ruleset:
        if require_bypass_visibility:
            raise SuiteAttemptError("GitHub ledger ruleset bypass actors are not visible")
    else:
        bypass_actors = _array(
            ruleset.get("bypass_actors"),
            label="GitHub ledger ruleset bypass actors",
        )
        if bypass_actors:
            raise SuiteAttemptError("GitHub ledger ruleset cannot grant a bypass actor")
    conditions = _object(
        ruleset.get("conditions"),
        label="GitHub ledger ruleset conditions",
    )
    if set(conditions) != {"ref_name"}:
        raise SuiteAttemptError("GitHub ledger ruleset must have only a ref-name condition")
    ref_name = _object(
        conditions.get("ref_name"),
        label="GitHub ledger ruleset ref-name condition",
    )
    include = _array(ref_name.get("include"), label="GitHub ledger ruleset includes")
    exclude = _array(ref_name.get("exclude"), label="GitHub ledger ruleset excludes")
    if include != [LEDGER_RULESET_INCLUDE] or exclude:
        raise SuiteAttemptError("GitHub ledger ruleset must cover the sole ledger ref family")
    rules = _array(ruleset.get("rules"), label="GitHub ledger ruleset rules")
    rule_types = {
        _text(
            _object(rule, label="GitHub ledger ruleset rule").get("type"),
            label="GitHub ledger ruleset rule type",
        )
        for rule in rules
    }
    if rule_types != LEDGER_RULE_TYPES or len(rules) != len(LEDGER_RULE_TYPES):
        raise SuiteAttemptError(
            "GitHub ledger ruleset must contain exactly the deletion, non-fast-forward, "
            "and linear-history rules"
        )
    return ruleset_id


def _ledger_ruleset_summaries(api: GitHubApi) -> tuple[Mapping[str, Any], ...]:
    values = _array(
        api.get(f"repos/{REPOSITORY}/rulesets?includes_parents=false&per_page=100"),
        label="GitHub repository rulesets",
    )
    return tuple(_object(value, label="GitHub repository ruleset") for value in values)


def _required_ledger_ruleset(
    api: GitHubApi,
    *,
    applied_ruleset_ids: Sequence[int] | None = None,
    require_bypass_visibility: bool = True,
) -> int:
    matches = [
        row for row in _ledger_ruleset_summaries(api) if row.get("name") == LEDGER_RULESET_NAME
    ]
    if len(matches) != 1:
        raise SuiteAttemptError("GitHub requires exactly one fixed confirmatory ledger ruleset")
    ruleset_id = _ruleset_id(matches[0].get("id"), label="GitHub ledger ruleset ID")
    detail = api.get(f"repos/{REPOSITORY}/rulesets/{ruleset_id}")
    if (
        _validate_ledger_ruleset(
            detail,
            require_bypass_visibility=require_bypass_visibility,
        )
        != ruleset_id
    ):
        raise SuiteAttemptError("GitHub ledger ruleset lookup changed identity")
    if applied_ruleset_ids is not None and ruleset_id not in applied_ruleset_ids:
        raise SuiteAttemptError("fixed GitHub ledger ruleset is not applied to the branch")
    return ruleset_id


def install_ledger_ruleset(*, api: GitHubWriteApi) -> int:
    """Create the sole fixed append-only ruleset, or verify its exact existing form."""

    matches = [
        row for row in _ledger_ruleset_summaries(api) if row.get("name") == LEDGER_RULESET_NAME
    ]
    if matches:
        if len(matches) != 1:
            raise SuiteAttemptError("GitHub has multiple confirmatory ledger rulesets")
        return _required_ledger_ruleset(api)
    payload: dict[str, object] = {
        "bypass_actors": [],
        "conditions": {
            "ref_name": {
                "exclude": [],
                "include": [LEDGER_RULESET_INCLUDE],
            }
        },
        "enforcement": "active",
        "name": LEDGER_RULESET_NAME,
        "rules": [
            {"type": rule_type}
            for rule_type in sorted(LEDGER_RULE_TYPES, key=lambda item: item.encode("utf-8"))
        ],
        "target": "branch",
    }
    created = api.post(f"repos/{REPOSITORY}/rulesets", payload)
    ruleset_id = _validate_ledger_ruleset(created, require_bypass_visibility=True)
    if _required_ledger_ruleset(api) != ruleset_id:
        raise SuiteAttemptError("GitHub returned another ledger ruleset after creation")
    return ruleset_id


def _validate_freeze_tag_ruleset(
    value: object,
    *,
    require_bypass_visibility: bool,
) -> int:
    """Validate the no-bypass ruleset covering only the C0 and C1 freeze tags."""

    ruleset = _object(value, label="GitHub freeze-tag ruleset")
    ruleset_id = _ruleset_id(ruleset.get("id"), label="GitHub freeze-tag ruleset ID")
    if (
        ruleset.get("name") != FREEZE_TAG_RULESET_NAME
        or ruleset.get("target") != "tag"
        or ruleset.get("enforcement") != "active"
    ):
        raise SuiteAttemptError("GitHub freeze-tag ruleset identity or enforcement differs")
    if "bypass_actors" not in ruleset:
        if require_bypass_visibility:
            raise SuiteAttemptError("GitHub freeze-tag ruleset bypass actors are not visible")
    else:
        bypass_actors = _array(
            ruleset.get("bypass_actors"),
            label="GitHub freeze-tag ruleset bypass actors",
        )
        if bypass_actors:
            raise SuiteAttemptError("GitHub freeze-tag ruleset cannot grant a bypass actor")
    conditions = _object(
        ruleset.get("conditions"),
        label="GitHub freeze-tag ruleset conditions",
    )
    if set(conditions) != {"ref_name"}:
        raise SuiteAttemptError("GitHub freeze-tag ruleset must have only a ref-name condition")
    ref_name = _object(
        conditions.get("ref_name"),
        label="GitHub freeze-tag ruleset ref-name condition",
    )
    include = _array(ref_name.get("include"), label="GitHub freeze-tag ruleset includes")
    exclude = _array(ref_name.get("exclude"), label="GitHub freeze-tag ruleset excludes")
    if include != list(FREEZE_TAG_RULESET_INCLUDES) or exclude:
        raise SuiteAttemptError(
            "GitHub freeze-tag ruleset must cover only the exact C0 and C1 refs"
        )
    rules = _array(ruleset.get("rules"), label="GitHub freeze-tag ruleset rules")
    rule_types = {
        _text(
            _object(rule, label="GitHub freeze-tag ruleset rule").get("type"),
            label="GitHub freeze-tag ruleset rule type",
        )
        for rule in rules
    }
    if rule_types != FREEZE_TAG_RULE_TYPES or len(rules) != len(FREEZE_TAG_RULE_TYPES):
        raise SuiteAttemptError(
            "GitHub freeze-tag ruleset must contain exactly the deletion and non-fast-forward rules"
        )
    return ruleset_id


def required_freeze_tag_ruleset(
    api: GitHubApi,
    *,
    require_bypass_visibility: bool = True,
) -> int:
    """Return the exact active freeze-tag ruleset ID or fail closed."""

    matches = [
        row for row in _ledger_ruleset_summaries(api) if row.get("name") == FREEZE_TAG_RULESET_NAME
    ]
    if len(matches) != 1:
        raise SuiteAttemptError("GitHub requires exactly one fixed freeze-tag ruleset")
    ruleset_id = _ruleset_id(matches[0].get("id"), label="GitHub freeze-tag ruleset ID")
    detail = api.get(f"repos/{REPOSITORY}/rulesets/{ruleset_id}")
    if (
        _validate_freeze_tag_ruleset(
            detail,
            require_bypass_visibility=require_bypass_visibility,
        )
        != ruleset_id
    ):
        raise SuiteAttemptError("GitHub freeze-tag ruleset lookup changed identity")
    return ruleset_id


def install_freeze_tag_ruleset(*, api: GitHubWriteApi) -> int:
    """Create or verify the exact, active, no-bypass C0/C1 tag ruleset."""

    matches = [
        row for row in _ledger_ruleset_summaries(api) if row.get("name") == FREEZE_TAG_RULESET_NAME
    ]
    if matches:
        if len(matches) != 1:
            raise SuiteAttemptError("GitHub has multiple fixed freeze-tag rulesets")
        return required_freeze_tag_ruleset(api)
    payload: dict[str, object] = {
        "bypass_actors": [],
        "conditions": {
            "ref_name": {
                "exclude": [],
                "include": list(FREEZE_TAG_RULESET_INCLUDES),
            }
        },
        "enforcement": "active",
        "name": FREEZE_TAG_RULESET_NAME,
        "rules": [
            {"type": rule_type}
            for rule_type in sorted(FREEZE_TAG_RULE_TYPES, key=lambda item: item.encode("utf-8"))
        ],
        "target": "tag",
    }
    created = api.post(f"repos/{REPOSITORY}/rulesets", payload)
    ruleset_id = _validate_freeze_tag_ruleset(created, require_bypass_visibility=True)
    if required_freeze_tag_ruleset(api) != ruleset_id:
        raise SuiteAttemptError("GitHub returned another freeze-tag ruleset after creation")
    return ruleset_id


def _decode_blob(api: GitHubApi, repository: str, oid: str) -> bytes:
    row = _object(
        api.get(f"repos/{repository}/git/blobs/{oid}"),
        label="GitHub Git blob response",
    )
    if row.get("encoding") != "base64":
        raise SuiteAttemptError("GitHub Git blob is not base64 encoded")
    content = _text(row.get("content"), label="GitHub Git blob content")
    try:
        encoded = base64.b64decode(content, validate=True)
    except (ValueError, TypeError) as exc:
        raise SuiteAttemptError("GitHub Git blob contains invalid base64") from exc
    if row.get("size") != len(encoded) or _git_blob_oid(encoded) != oid:
        raise SuiteAttemptError("GitHub Git blob size or object ID differs")
    return encoded


def _tree_blobs(api: GitHubApi, repository: str, tree_oid: str) -> dict[str, str]:
    row = _object(
        api.get(f"repos/{repository}/git/trees/{tree_oid}?recursive=1"),
        label="GitHub Git tree response",
    )
    if row.get("truncated") is not False:
        raise SuiteAttemptError("GitHub ledger tree is truncated")
    entries = _array(row.get("tree"), label="GitHub Git tree entries")
    blobs: dict[str, str] = {}
    trees: set[str] = set()
    for raw in entries:
        entry = _object(raw, label="GitHub Git tree entry")
        path = _text(entry.get("path"), label="GitHub Git tree path")
        kind = entry.get("type")
        if kind == "tree":
            if entry.get("mode") != "040000":
                raise SuiteAttemptError("GitHub ledger directory has an unexpected mode")
            trees.add(path)
            continue
        if kind != "blob" or entry.get("mode") != "100644":
            raise SuiteAttemptError("GitHub ledger contains a non-regular entry")
        if path in blobs:
            raise SuiteAttemptError("GitHub ledger tree repeats a path")
        blobs[path] = _oid(entry.get("sha"), label="GitHub Git blob object ID")
    if not blobs:
        raise SuiteAttemptError("GitHub ledger tree contains no state records")
    attempt_parts: set[str] = set()
    for path in blobs:
        parts = path.split("/")
        if len(parts) < 3 or parts[0] not in {LEDGER_PATH_PREFIX, LEDGER_CONTROL_PREFIX}:
            raise SuiteAttemptError("GitHub ledger contains an unexpected blob path")
        attempt_parts.add(parts[1])
    if len(attempt_parts) != 1:
        raise SuiteAttemptError("GitHub ledger tree mixes suite-attempt directories")
    attempt_id = next(iter(attempt_parts))
    allowed_tree_prefixes = {
        LEDGER_PATH_PREFIX,
        f"{LEDGER_PATH_PREFIX}/{attempt_id}",
        LEDGER_CONTROL_PREFIX,
        f"{LEDGER_CONTROL_PREFIX}/{attempt_id}",
        f"{LEDGER_CONTROL_PREFIX}/{attempt_id}/sealed-launch-contracts",
    }
    if trees - allowed_tree_prefixes:
        raise SuiteAttemptError("GitHub ledger tree contains an unexpected directory")
    return blobs


def _commit_row(api: GitHubApi, repository: str, oid: str) -> Mapping[str, Any]:
    row = _object(
        api.get(f"repos/{repository}/git/commits/{oid}"),
        label="GitHub Git commit response",
    )
    if _oid(row.get("sha"), label="GitHub Git commit object ID") != oid:
        raise SuiteAttemptError("GitHub returned another ledger commit")
    for role in ("author", "committer"):
        identity = _object(row.get(role), label=f"GitHub Git commit {role}")
        if identity.get("name") != GIT_IDENTITY_NAME or identity.get("email") != GIT_IDENTITY_EMAIL:
            raise SuiteAttemptError(f"GitHub ledger commit {role} is not mhdk1602")
        _utc_datetime(identity.get("date"), label=f"GitHub Git commit {role} date")
    return row


def _parent_oids(commit: Mapping[str, Any]) -> tuple[str, ...]:
    parents = _array(commit.get("parents"), label="GitHub Git commit parents")
    return tuple(
        _oid(_object(parent, label="Git parent").get("sha"), label="Git parent object ID")
        for parent in parents
    )


def _provider_ledger_controls(
    api: GitHubApi,
    repository: str,
    suite_attempt_id_value: str,
    blobs: Mapping[str, str],
) -> tuple[tuple[LedgerControlFile, ...], bytes]:
    inventory_path = _control_inventory_path(suite_attempt_id_value)
    inventory_oid = blobs.get(inventory_path)
    if inventory_oid is None:
        raise SuiteAttemptError("GitHub ledger omits its immutable control inventory")
    inventory_bytes = _decode_blob(api, repository, inventory_oid)
    inventory = _object(
        _strict_json(inventory_bytes, label="ledger control inventory"),
        label="ledger control inventory",
    )
    if (
        set(inventory) != {"controls", "schema_version", "suite_attempt_id"}
        or inventory.get("schema_version") != LEDGER_CONTROL_INVENTORY_SCHEMA
        or inventory.get("suite_attempt_id") != suite_attempt_id_value
        or inventory_bytes != _canonical_bytes(inventory) + b"\n"
    ):
        raise SuiteAttemptError("GitHub ledger control inventory is not canonical")
    rows = _array(inventory.get("controls"), label="ledger control inventory rows")
    if len(rows) != 7:
        raise SuiteAttemptError("GitHub ledger control inventory must contain exactly seven files")
    controls: list[LedgerControlFile] = []
    expected_paths = {inventory_path}
    previous_path: bytes | None = None
    roles: set[str] = set()
    for raw in rows:
        row = _object(raw, label="ledger control inventory row")
        if set(row) != {
            "byte_count",
            "file_sha256",
            "ledger_path",
            "materialization_uri",
            "role",
        }:
            raise SuiteAttemptError("ledger control inventory row is not closed")
        role = _text(row.get("role"), label="ledger control role")
        ledger_path = _text(row.get("ledger_path"), label="ledger control path")
        materialization_uri = _text(
            row.get("materialization_uri"),
            label="ledger control materialization URI",
        )
        digest = _digest(row.get("file_sha256"), label="ledger control digest")
        byte_count = _integer(row.get("byte_count"), label="ledger control byte count")
        if byte_count <= 0 or byte_count > _MAX_LEDGER_CONTROL_BYTES:
            raise SuiteAttemptError("ledger control byte count is outside its bound")
        if (
            ledger_path == inventory_path
            or not ledger_path.startswith(f"{LEDGER_CONTROL_PREFIX}/{suite_attempt_id_value}/")
            or ledger_path not in blobs
            or role in roles
        ):
            raise SuiteAttemptError("ledger control path or role is duplicated or foreign")
        encoded_path = ledger_path.encode("utf-8")
        if previous_path is not None and encoded_path <= previous_path:
            raise SuiteAttemptError("ledger control inventory is not byte-sorted")
        previous_path = encoded_path
        roles.add(role)
        encoded = _decode_blob(api, repository, blobs[ledger_path])
        if len(encoded) != byte_count or hashlib.sha256(encoded).hexdigest() != digest:
            raise SuiteAttemptError("ledger control bytes differ from their inventory")
        _canonical_local_file_uri(
            materialization_uri,
            label="ledger control materialization URI",
        )
        controls.append(
            LedgerControlFile(
                role=role,
                ledger_path=ledger_path,
                materialization_uri=materialization_uri,
                file_sha256=digest,
                byte_count=byte_count,
                blob_oid=blobs[ledger_path],
                encoded=encoded,
            )
        )
        expected_paths.add(ledger_path)
    expected_roles = {
        "attestation-descriptor",
        "production-finalization-receipt",
        *(f"sealed-launch-contract:{corpus_id}" for corpus_id in FIXED_CORPORA),
    }
    if roles != expected_roles:
        raise SuiteAttemptError("ledger control inventory changes the fixed role set")
    control_paths = {
        path
        for path in blobs
        if path.startswith(f"{LEDGER_CONTROL_PREFIX}/{suite_attempt_id_value}/")
    }
    if control_paths != expected_paths:
        raise SuiteAttemptError("GitHub ledger has an extra or missing control blob")
    return tuple(controls), inventory_bytes


def _validate_provider_controls_against_opened(
    controls: Sequence[LedgerControlFile],
    opened: SuiteStateRecord,
) -> None:
    if opened.state != "OPENED" or not isinstance(opened.payload, SuiteOpenBindings):
        raise SuiteAttemptError("provider ledger controls lack a typed OPENED binding")
    by_role = {control.role: control for control in controls}
    namespace = _canonical_local_file_uri(
        opened.namespace_uri,
        label="OPENED namespace URI",
    )
    expected: dict[str, tuple[str, str, str]] = {
        "attestation-descriptor": (
            f"{LEDGER_CONTROL_PREFIX}/{opened.suite_attempt_id}/attestation-descriptor.json",
            (namespace / "attestation-descriptor.json").as_uri(),
            "",
        ),
        "production-finalization-receipt": (
            f"{LEDGER_CONTROL_PREFIX}/{opened.suite_attempt_id}/production-finalization-receipt.json",
            opened.payload.production_finalization_receipt_uri,
            opened.payload.production_finalization_receipt_file_sha256,
        ),
    }
    for binding in opened.payload.runtime_attestation_plans:
        expected[f"sealed-launch-contract:{binding.corpus_id}"] = (
            f"{LEDGER_CONTROL_PREFIX}/{opened.suite_attempt_id}/sealed-launch-contracts/"
            f"{binding.corpus_id}.json",
            binding.sealed_launch_contract_uri,
            binding.sealed_launch_contract_file_sha256,
        )
    if set(by_role) != set(expected):
        raise SuiteAttemptError("provider ledger controls change the OPENED role set")
    for role, (ledger_path, materialization_uri, digest) in expected.items():
        control = by_role[role]
        if (
            control.ledger_path != ledger_path
            or control.materialization_uri != materialization_uri
            or (role != "attestation-descriptor" and control.file_sha256 != digest)
        ):
            raise SuiteAttemptError(f"provider ledger control {role} differs from OPENED")
    descriptor_control = by_role["attestation-descriptor"]
    descriptor = SuiteAttestationDescriptor.from_dict(
        _object(
            _strict_json(
                descriptor_control.encoded,
                label="provider ledger attestation descriptor",
            ),
            label="provider ledger attestation descriptor",
        )
    )
    if (
        descriptor_control.encoded != descriptor.canonical_bytes() + b"\n"
        or descriptor.descriptor_sha256 != opened.payload.attestation_descriptor_sha256
    ):
        raise SuiteAttemptError("provider ledger descriptor bytes differ from OPENED")


def load_ledger_snapshot(
    *,
    repository: str,
    suite_attempt_id: str,
    api: GitHubApi,
    require_ruleset_bypass_visibility: bool = True,
) -> LedgerSnapshot:
    """Reconstruct the current protected, single-parent ledger branch."""

    _digest(suite_attempt_id, label="suite attempt ID")
    if repository != REPOSITORY:
        raise SuiteAttemptError("GitHub ledger repository differs from the C0 policy")
    branch = f"confirmatory-ledger/{suite_attempt_id}"
    state_key = f"{LEDGER_REF_PREFIX}/{suite_attempt_id}"
    encoded_branch = quote(branch, safe="")
    ref = _object(
        api.get(f"repos/{repository}/git/ref/heads/{branch}"),
        label="GitHub ledger ref response",
    )
    if ref.get("ref") != state_key:
        raise SuiteAttemptError("GitHub ledger ref differs from the manifest-derived key")
    ref_object = _object(ref.get("object"), label="GitHub ledger ref object")
    if ref_object.get("type") != "commit":
        raise SuiteAttemptError("GitHub ledger ref does not name a commit")
    tip_oid = _oid(ref_object.get("sha"), label="GitHub ledger tip")
    branch_row = _object(
        api.get(f"repos/{repository}/branches/{encoded_branch}"),
        label="GitHub branch response",
    )
    branch_commit = _object(branch_row.get("commit"), label="GitHub branch commit")
    if (
        branch_row.get("name") != branch
        or _oid(branch_commit.get("sha"), label="GitHub branch tip") != tip_oid
    ):
        raise SuiteAttemptError("GitHub branch view differs from the ledger ref")
    protection = LedgerProtection.from_api(
        branch_row,
        api.get(f"repos/{repository}/rules/branches/{encoded_branch}?per_page=100"),
    )
    _required_ledger_ruleset(
        api,
        applied_ruleset_ids=protection.ruleset_ids,
        require_bypass_visibility=require_ruleset_bypass_visibility,
    )

    reverse_commits: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    cursor = tip_oid
    while True:
        if cursor in seen or len(reverse_commits) >= _MAX_LEDGER_STATES:
            raise SuiteAttemptError("GitHub ledger ancestry cycles or exceeds the state machine")
        seen.add(cursor)
        commit = _commit_row(api, repository, cursor)
        reverse_commits.append((cursor, commit))
        parents = _parent_oids(commit)
        if not parents:
            break
        if len(parents) != 1:
            raise SuiteAttemptError("GitHub ledger contains a merge commit")
        cursor = parents[0]

    transitions: list[LedgerTransition] = []
    previous_blobs: dict[str, str] = {}
    previous_state: SuiteStateRecord | None = None
    controls: tuple[LedgerControlFile, ...] = ()
    control_inventory_bytes = b""
    control_blob_oids: dict[str, str] = {}
    for sequence, (commit_oid, commit) in enumerate(reversed(reverse_commits)):
        parents = _parent_oids(commit)
        previous_oid = None if sequence == 0 else transitions[-1].commit_oid
        if parents != (() if previous_oid is None else (previous_oid,)):
            raise SuiteAttemptError(
                "GitHub ledger commit does not compare-and-swap its predecessor"
            )
        tree = _object(commit.get("tree"), label="GitHub ledger commit tree")
        tree_oid = _oid(tree.get("sha"), label="GitHub ledger tree object ID")
        blobs = _tree_blobs(api, repository, tree_oid)
        if sequence == 0:
            controls, control_inventory_bytes = _provider_ledger_controls(
                api,
                repository,
                suite_attempt_id,
                blobs,
            )
            control_blob_oids = {
                _control_inventory_path(suite_attempt_id): blobs[
                    _control_inventory_path(suite_attempt_id)
                ],
                **{control.ledger_path: control.blob_oid for control in controls},
            }
        expected_state_paths = {
            f"{LEDGER_PATH_PREFIX}/{suite_attempt_id}/{index:03d}.state.json"
            for index in range(sequence + 1)
        }
        if set(blobs) != expected_state_paths | set(control_blob_oids):
            raise SuiteAttemptError("GitHub ledger tree is not the exact state and control prefix")
        if any(blobs[path] != oid for path, oid in control_blob_oids.items()):
            raise SuiteAttemptError("GitHub ledger rewrites an immutable control")
        for path, oid in previous_blobs.items():
            if blobs.get(path) != oid:
                raise SuiteAttemptError("GitHub ledger rewrites a preceding state record")
        state_path = f"{LEDGER_PATH_PREFIX}/{suite_attempt_id}/{sequence:03d}.state.json"
        state_bytes = _decode_blob(api, repository, blobs[state_path])
        state = _state_from_bytes(state_bytes)
        if state.suite_attempt_id != suite_attempt_id or state.sequence != sequence:
            raise SuiteAttemptError("GitHub ledger path and state identity differ")
        if sequence == 0:
            namespace = _canonical_local_file_uri(
                state.namespace_uri,
                label="provider OPENED namespace URI",
            )
            if (
                state.state != "OPENED"
                or not isinstance(state.payload, SuiteOpenBindings)
                or state.suite_attempt_id != derive_suite_attempt_id(state.manifest_sha256)
                or namespace.name != f"suite-attempt-{state.suite_attempt_id}"
            ):
                raise SuiteAttemptError(
                    "GitHub ledger genesis is not manifest-derived OPENED state"
                )
            _validate_provider_controls_against_opened(controls, state)
        if previous_state is not None:
            _assert_state_transition(previous_state, state)
        message = _text(commit.get("message"), label="GitHub ledger commit message")
        expected_message = (
            f"confirmatory-state {suite_attempt_id} {sequence:03d} "
            f"{state.state} {state.record_sha256}"
        )
        if message != expected_message:
            raise SuiteAttemptError("GitHub ledger commit message is not canonical")
        transitions.append(
            LedgerTransition(
                commit_oid=commit_oid,
                previous_commit_oid=previous_oid,
                tree_oid=tree_oid,
                state_path=state_path,
                state_bytes=state_bytes,
                state=state,
            )
        )
        previous_blobs = blobs
        previous_state = state
    final_ref = _object(
        api.get(f"repos/{repository}/git/ref/heads/{branch}"),
        label="final GitHub ledger ref response",
    )
    final_object = _object(final_ref.get("object"), label="final GitHub ledger ref object")
    if (
        final_ref.get("ref") != state_key
        or final_object.get("type") != "commit"
        or _oid(final_object.get("sha"), label="final GitHub ledger tip") != tip_oid
    ):
        raise SuiteAttemptError("GitHub ledger changed during reconstruction")
    return LedgerSnapshot(
        repository=repository,
        state_key=state_key,
        protection=protection,
        transitions=tuple(transitions),
        controls=controls,
        control_inventory_bytes=control_inventory_bytes,
    )


@dataclass(frozen=True)
class LedgerPublicationReceipt:
    repository: str
    state_key: str
    ruleset_id: int
    commit_oid: str
    previous_commit_oid: str | None
    tree_oid: str
    blob_oid: str
    state_path: str
    state_record_sha256: str
    state_sequence: int
    suite_attempt_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "blob_oid": self.blob_oid,
            "commit_oid": self.commit_oid,
            "previous_commit_oid": self.previous_commit_oid,
            "repository": self.repository,
            "ruleset_id": self.ruleset_id,
            "schema_version": LEDGER_PUBLISH_RECEIPT_SCHEMA,
            "state_key": self.state_key,
            "state_path": self.state_path,
            "state_record_sha256": self.state_record_sha256,
            "state_sequence": self.state_sequence,
            "suite_attempt_id": self.suite_attempt_id,
            "tree_oid": self.tree_oid,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _local_state_prefix(namespace: Path) -> tuple[SuiteStateRecord, ...]:
    if not namespace.is_absolute() or namespace.anchor != "/":
        raise SuiteAttemptError("suite namespace must be an absolute POSIX path")
    if any(part in {"", ".", ".."} for part in namespace.parts[1:]):
        raise SuiteAttemptError("suite namespace cannot contain aliasing components")
    try:
        metadata = namespace.stat(follow_symlinks=False)
    except OSError as exc:
        raise SuiteAttemptError("cannot inspect the suite namespace") from exc
    if not stat.S_ISDIR(metadata.st_mode) or namespace.is_symlink():
        raise SuiteAttemptError("suite namespace must be one real directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise SuiteAttemptError("suite namespace must be owned by the current identity")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise SuiteAttemptError("suite namespace cannot be writable by another identity")
    names = sorted(
        (
            path.name
            for path in namespace.iterdir()
            if re.fullmatch(r"[0-9]{3}\.state\.json", path.name) is not None
        ),
        key=lambda item: item.encode("utf-8"),
    )
    if not names or len(names) > _MAX_LEDGER_STATES:
        raise SuiteAttemptError("suite namespace has no bounded local state prefix")
    expected_names = [f"{sequence:03d}.state.json" for sequence in range(len(names))]
    if names != expected_names:
        raise SuiteAttemptError("local suite state prefix has a gap")
    states = tuple(load_suite_state_record(namespace / name) for name in names)
    first = states[0]
    if (
        first.sequence != 0
        or first.state != "OPENED"
        or first.previous_state_record_sha256 is not None
        or not isinstance(first.payload, SuiteOpenBindings)
        or first.suite_attempt_id != derive_suite_attempt_id(first.manifest_sha256)
        or first.namespace_uri != namespace.as_uri()
        or namespace.name != f"suite-attempt-{first.suite_attempt_id}"
    ):
        raise SuiteAttemptError("local suite state prefix lacks its exact OPENED identity")
    for sequence, state in enumerate(states):
        if (
            state.sequence != sequence
            or state.suite_attempt_id != first.suite_attempt_id
            or state.namespace_uri != namespace.as_uri()
        ):
            raise SuiteAttemptError("local suite state path and record differ")
        if sequence:
            _assert_state_transition(states[sequence - 1], state)
    return states


def _matching_ledger_ref(
    *,
    api: GitHubApi,
    suite_attempt_id_value: str,
) -> str | None:
    endpoint = (
        f"repos/{REPOSITORY}/git/matching-refs/heads/confirmatory-ledger/{suite_attempt_id_value}"
    )
    rows = _array(api.get(endpoint), label="GitHub matching ledger refs")
    expected_ref = f"{LEDGER_REF_PREFIX}/{suite_attempt_id_value}"
    if not rows:
        return None
    if len(rows) != 1:
        raise SuiteAttemptError("GitHub matching-ref lookup is not a singleton")
    row = _object(rows[0], label="GitHub matching ledger ref")
    if row.get("ref") != expected_ref:
        raise SuiteAttemptError("GitHub matching-ref lookup returned another ledger key")
    ref_object = _object(row.get("object"), label="GitHub matching ledger ref object")
    if ref_object.get("type") != "commit":
        raise SuiteAttemptError("GitHub ledger ref does not name a commit")
    return _oid(ref_object.get("sha"), label="GitHub matching ledger tip")


def _assert_current_ledger_tip(
    *,
    api: GitHubApi,
    suite_attempt_id_value: str,
    expected_commit_oid: str,
) -> None:
    branch = f"confirmatory-ledger/{suite_attempt_id_value}"
    state_key = f"{LEDGER_REF_PREFIX}/{suite_attempt_id_value}"
    row = _object(
        api.get(f"repos/{REPOSITORY}/git/ref/heads/{branch}"),
        label="GitHub ledger authority ref response",
    )
    ref_object = _object(row.get("object"), label="GitHub ledger authority ref object")
    if (
        row.get("ref") != state_key
        or ref_object.get("type") != "commit"
        or _oid(ref_object.get("sha"), label="GitHub ledger authority tip") != expected_commit_oid
    ):
        raise SuiteAttemptError("GitHub ledger changed before authority use")


def _created_oid(value: object, *, label: str) -> str:
    return _oid(_object(value, label=label).get("sha"), label=f"{label} object ID")


def _assert_created_ref(value: object, *, expected_ref: str, expected_oid: str) -> None:
    row = _object(value, label="GitHub created ledger ref")
    if row.get("ref") != expected_ref:
        raise SuiteAttemptError("GitHub created another ledger ref")
    ref_object = _object(row.get("object"), label="GitHub created ledger ref object")
    if (
        ref_object.get("type") != "commit"
        or _oid(ref_object.get("sha"), label="GitHub created ledger tip") != expected_oid
    ):
        raise SuiteAttemptError("GitHub ledger ref response names another commit")


def _publication_receipt(
    *,
    snapshot: LedgerSnapshot,
    ruleset_id: int,
) -> LedgerPublicationReceipt:
    transition = snapshot.tip
    return LedgerPublicationReceipt(
        repository=snapshot.repository,
        state_key=snapshot.state_key,
        ruleset_id=ruleset_id,
        commit_oid=transition.commit_oid,
        previous_commit_oid=transition.previous_commit_oid,
        tree_oid=transition.tree_oid,
        blob_oid=_git_blob_oid(transition.state_bytes),
        state_path=transition.state_path,
        state_record_sha256=transition.state.record_sha256,
        state_sequence=transition.state.sequence,
        suite_attempt_id=transition.state.suite_attempt_id,
    )


def _persist_publication_receipt(path: Path, receipt: LedgerPublicationReceipt) -> None:
    if not path.is_absolute():
        raise SuiteAttemptError("ledger publication receipt path must be absolute")
    encoded = receipt.canonical_bytes()
    if path.exists():
        if _regular_file_bytes(path, label="ledger publication receipt") != encoded:
            raise SuiteAttemptError("existing ledger publication receipt differs")
        return
    _write_exclusive(path, encoded)


def publish_ledger_transition(
    *,
    namespace: Path,
    receipt_path: Path,
    api: GitHubWriteApi,
) -> tuple[LedgerPublicationReceipt, bool]:
    """Publish the latest local state by one create-or-fast-forward CAS operation."""

    states = _local_state_prefix(namespace)
    target = states[-1]
    local_controls = _local_ledger_controls(namespace, states[0])
    local_inventory = _ledger_control_inventory(target.suite_attempt_id, local_controls)
    ruleset_id = _required_ledger_ruleset(api)
    matched_tip = _matching_ledger_ref(
        api=api,
        suite_attempt_id_value=target.suite_attempt_id,
    )
    existing: LedgerSnapshot | None = None
    if matched_tip is not None:
        existing = load_ledger_snapshot(
            repository=REPOSITORY,
            suite_attempt_id=target.suite_attempt_id,
            api=api,
        )
        if existing.tip.commit_oid != matched_tip:
            raise SuiteAttemptError("GitHub ledger changed between ref reads")
        if (
            existing.control_inventory_bytes != local_inventory
            or len(existing.controls) != len(local_controls)
            or any(
                observed.encoded != expected.encoded
                or observed.to_inventory_dict() != expected.to_inventory_dict()
                for observed, expected in zip(
                    existing.controls,
                    sorted(local_controls, key=lambda item: item.ledger_path.encode("utf-8")),
                    strict=True,
                )
            )
        ):
            raise SuiteAttemptError("GitHub ledger controls differ from the local OPENED closure")
        if len(existing.transitions) == len(states):
            if all(
                transition.state_bytes == state.canonical_bytes() + b"\n"
                for transition, state in zip(existing.transitions, states, strict=True)
            ):
                receipt = _publication_receipt(snapshot=existing, ruleset_id=ruleset_id)
                _assert_current_ledger_tip(
                    api=api,
                    suite_attempt_id_value=target.suite_attempt_id,
                    expected_commit_oid=receipt.commit_oid,
                )
                _persist_publication_receipt(receipt_path, receipt)
                return receipt, False
            raise SuiteAttemptError("GitHub ledger and local state prefix differ")
        if len(existing.transitions) != len(states) - 1:
            raise SuiteAttemptError("GitHub ledger is stale or ahead of the local CAS predecessor")
        for transition, state in zip(existing.transitions, states[:-1], strict=True):
            if transition.state_bytes != state.canonical_bytes() + b"\n":
                raise SuiteAttemptError("GitHub ledger predecessor differs from local state")
    elif len(states) != 1:
        raise SuiteAttemptError("GitHub ledger genesis is absent for a later local state")

    state_bytes = target.canonical_bytes() + b"\n"
    state_path = f"{LEDGER_PATH_PREFIX}/{target.suite_attempt_id}/{target.sequence:03d}.state.json"
    blob_response = api.post(
        f"repos/{REPOSITORY}/git/blobs",
        {
            "content": base64.b64encode(state_bytes).decode("ascii"),
            "encoding": "base64",
        },
    )
    blob_oid = _created_oid(blob_response, label="GitHub created ledger blob")
    if blob_oid != _git_blob_oid(state_bytes):
        raise SuiteAttemptError("GitHub created ledger blob has another object ID")
    tree_entries: list[dict[str, object]] = [
        {
            "mode": "100644",
            "path": state_path,
            "sha": blob_oid,
            "type": "blob",
        }
    ]
    if existing is None:
        for control in local_controls:
            created_control_oid = _created_oid(
                api.post(
                    f"repos/{REPOSITORY}/git/blobs",
                    {
                        "content": base64.b64encode(control.encoded).decode("ascii"),
                        "encoding": "base64",
                    },
                ),
                label=f"GitHub created ledger control {control.role}",
            )
            if created_control_oid != control.blob_oid:
                raise SuiteAttemptError("GitHub created another ledger control blob")
            tree_entries.append(
                {
                    "mode": "100644",
                    "path": control.ledger_path,
                    "sha": created_control_oid,
                    "type": "blob",
                }
            )
        inventory_oid = _created_oid(
            api.post(
                f"repos/{REPOSITORY}/git/blobs",
                {
                    "content": base64.b64encode(local_inventory).decode("ascii"),
                    "encoding": "base64",
                },
            ),
            label="GitHub created ledger control inventory",
        )
        if inventory_oid != _git_blob_oid(local_inventory):
            raise SuiteAttemptError("GitHub created another ledger control inventory")
        tree_entries.append(
            {
                "mode": "100644",
                "path": _control_inventory_path(target.suite_attempt_id),
                "sha": inventory_oid,
                "type": "blob",
            }
        )
    tree_payload: dict[str, object] = {"tree": tree_entries}
    if existing is not None:
        tree_payload["base_tree"] = existing.tip.tree_oid
    tree_oid = _created_oid(
        api.post(f"repos/{REPOSITORY}/git/trees", tree_payload),
        label="GitHub created ledger tree",
    )
    identity_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    identity = {
        "date": identity_date,
        "email": GIT_IDENTITY_EMAIL,
        "name": GIT_IDENTITY_NAME,
    }
    previous_oid = None if existing is None else existing.tip.commit_oid
    message = (
        f"confirmatory-state {target.suite_attempt_id} {target.sequence:03d} "
        f"{target.state} {target.record_sha256}"
    )
    commit_oid = _created_oid(
        api.post(
            f"repos/{REPOSITORY}/git/commits",
            {
                "author": identity,
                "committer": identity,
                "message": message,
                "parents": [] if previous_oid is None else [previous_oid],
                "tree": tree_oid,
            },
        ),
        label="GitHub created ledger commit",
    )
    commit = _commit_row(api, REPOSITORY, commit_oid)
    if (
        commit.get("message") != message
        or _parent_oids(commit) != (() if previous_oid is None else (previous_oid,))
        or _oid(
            _object(commit.get("tree"), label="created ledger commit tree").get("sha"),
            label="created ledger commit tree ID",
        )
        != tree_oid
    ):
        raise SuiteAttemptError("created GitHub ledger commit differs before publication")
    state_key = f"{LEDGER_REF_PREFIX}/{target.suite_attempt_id}"
    if previous_oid is None:
        ref_response = api.post(
            f"repos/{REPOSITORY}/git/refs",
            {"ref": state_key, "sha": commit_oid},
        )
    else:
        ref_response = api.patch(
            f"repos/{REPOSITORY}/git/refs/heads/confirmatory-ledger/{target.suite_attempt_id}",
            {"force": False, "sha": commit_oid},
        )
    _assert_created_ref(ref_response, expected_ref=state_key, expected_oid=commit_oid)
    observed = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=target.suite_attempt_id,
        api=api,
    )
    if (
        observed.tip.commit_oid != commit_oid
        or len(observed.transitions) != len(states)
        or any(
            transition.state_bytes != state.canonical_bytes() + b"\n"
            for transition, state in zip(observed.transitions, states, strict=True)
        )
    ):
        raise SuiteAttemptError("published GitHub ledger tip failed exact readback")
    receipt = _publication_receipt(snapshot=observed, ruleset_id=ruleset_id)
    _assert_current_ledger_tip(
        api=api,
        suite_attempt_id_value=target.suite_attempt_id,
        expected_commit_oid=receipt.commit_oid,
    )
    _persist_publication_receipt(receipt_path, receipt)
    return receipt, True


def publish_candidate_ledger_transition(
    *,
    target: SuiteStateRecord,
    expected_predecessor_commit: str,
    receipt_path: Path,
    api: GitHubWriteApi,
) -> tuple[LedgerPublicationReceipt, bool]:
    """Publish one hosted candidate without remapping its self-hosted namespace."""

    if not isinstance(target, SuiteStateRecord) or target.sequence <= 0:
        raise SuiteAttemptError("hosted candidate must be one typed non-genesis state")
    predecessor_oid = _oid(
        expected_predecessor_commit,
        label="expected hosted candidate predecessor",
    )
    ruleset_id = _required_ledger_ruleset(api)
    observed = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=target.suite_attempt_id,
        api=api,
    )
    target_bytes = target.canonical_bytes() + b"\n"
    if (
        len(observed.transitions) == target.sequence + 1
        and observed.tip.state_bytes == target_bytes
        and observed.tip.previous_commit_oid == predecessor_oid
    ):
        receipt = _publication_receipt(snapshot=observed, ruleset_id=ruleset_id)
        _assert_current_ledger_tip(
            api=api,
            suite_attempt_id_value=target.suite_attempt_id,
            expected_commit_oid=receipt.commit_oid,
        )
        _persist_publication_receipt(receipt_path, receipt)
        return receipt, False
    if len(observed.transitions) != target.sequence or observed.tip.commit_oid != predecessor_oid:
        raise SuiteAttemptError("hosted candidate lost its exact provider CAS predecessor")
    _assert_state_transition(observed.tip.state, target)

    state_path = f"{LEDGER_PATH_PREFIX}/{target.suite_attempt_id}/{target.sequence:03d}.state.json"
    blob_oid = _created_oid(
        api.post(
            f"repos/{REPOSITORY}/git/blobs",
            {
                "content": base64.b64encode(target_bytes).decode("ascii"),
                "encoding": "base64",
            },
        ),
        label="GitHub created hosted candidate blob",
    )
    if blob_oid != _git_blob_oid(target_bytes):
        raise SuiteAttemptError("GitHub hosted candidate blob has another object ID")
    tree_oid = _created_oid(
        api.post(
            f"repos/{REPOSITORY}/git/trees",
            {
                "base_tree": observed.tip.tree_oid,
                "tree": [
                    {
                        "mode": "100644",
                        "path": state_path,
                        "sha": blob_oid,
                        "type": "blob",
                    }
                ],
            },
        ),
        label="GitHub created hosted candidate tree",
    )
    identity_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    identity = {
        "date": identity_date,
        "email": GIT_IDENTITY_EMAIL,
        "name": GIT_IDENTITY_NAME,
    }
    message = (
        f"confirmatory-state {target.suite_attempt_id} {target.sequence:03d} "
        f"{target.state} {target.record_sha256}"
    )
    commit_oid = _created_oid(
        api.post(
            f"repos/{REPOSITORY}/git/commits",
            {
                "author": identity,
                "committer": identity,
                "message": message,
                "parents": [predecessor_oid],
                "tree": tree_oid,
            },
        ),
        label="GitHub created hosted candidate commit",
    )
    commit = _commit_row(api, REPOSITORY, commit_oid)
    if (
        commit.get("message") != message
        or _parent_oids(commit) != (predecessor_oid,)
        or _oid(
            _object(commit.get("tree"), label="hosted candidate commit tree").get("sha"),
            label="hosted candidate tree ID",
        )
        != tree_oid
    ):
        raise SuiteAttemptError("created hosted candidate commit differs before publication")
    ref_response = api.patch(
        f"repos/{REPOSITORY}/git/refs/heads/confirmatory-ledger/{target.suite_attempt_id}",
        {"force": False, "sha": commit_oid},
    )
    _assert_created_ref(
        ref_response,
        expected_ref=f"{LEDGER_REF_PREFIX}/{target.suite_attempt_id}",
        expected_oid=commit_oid,
    )
    readback = load_ledger_snapshot(
        repository=REPOSITORY,
        suite_attempt_id=target.suite_attempt_id,
        api=api,
    )
    if (
        readback.tip.commit_oid != commit_oid
        or readback.tip.previous_commit_oid != predecessor_oid
        or readback.tip.state_bytes != target_bytes
        or len(readback.transitions) != target.sequence + 1
    ):
        raise SuiteAttemptError("published hosted candidate failed exact provider readback")
    receipt = _publication_receipt(snapshot=readback, ruleset_id=ruleset_id)
    _assert_current_ledger_tip(
        api=api,
        suite_attempt_id_value=target.suite_attempt_id,
        expected_commit_oid=receipt.commit_oid,
    )
    _persist_publication_receipt(receipt_path, receipt)
    return receipt, True


def ledger_predicate(snapshot: LedgerSnapshot, transition: LedgerTransition) -> dict[str, object]:
    return {
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "ledger": {
            "commit_oid": transition.commit_oid,
            "previous_commit_oid": transition.previous_commit_oid,
            "repository": snapshot.repository,
            "state_key": snapshot.state_key,
            "tree_oid": transition.tree_oid,
        },
        "protection": snapshot.protection.to_dict(),
        "schema_version": PREDICATE_SCHEMA,
        "state": {
            "name": transition.state_path,
            "record_sha256": transition.state.record_sha256,
            "sequence": transition.state.sequence,
            "state": transition.state.state,
            "suite_attempt_id": transition.state.suite_attempt_id,
        },
    }


@dataclass(frozen=True)
class SigstoreObservation:
    statement: Mapping[str, Any]
    log_key_sha256: str
    log_index: int
    entry_id: str
    integrated_at_utc: str
    timestamp_token_sha256: str


def parse_sigstore_bundle(bundle: bytes) -> SigstoreObservation:
    """Extract provider-authenticated values after Sigstore verification succeeds."""

    root = _object(_strict_json(bundle, label="Sigstore bundle"), label="Sigstore bundle")
    if root.get("mediaType") != "application/vnd.dev.sigstore.bundle.v0.3+json":
        raise SuiteAttemptError("Sigstore bundle must use the v0.3 JSON media type")
    material = _object(root.get("verificationMaterial"), label="Sigstore verification material")
    entries = _array(material.get("tlogEntries"), label="Sigstore transparency entries")
    if len(entries) != 1:
        raise SuiteAttemptError("Sigstore bundle must contain exactly one Rekor entry")
    entry = _object(entries[0], label="Sigstore Rekor entry")
    log_id = _object(entry.get("logId"), label="Sigstore Rekor log ID")
    try:
        key_id = base64.b64decode(
            _text(log_id.get("keyId"), label="Sigstore Rekor key ID"), validate=True
        )
        set_bytes = base64.b64decode(
            _text(
                _object(entry.get("inclusionPromise"), label="Rekor inclusion promise").get(
                    "signedEntryTimestamp"
                ),
                label="Rekor signed entry timestamp",
            ),
            validate=True,
        )
        body = base64.b64decode(
            _text(entry.get("canonicalizedBody"), label="Rekor canonical body"),
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise SuiteAttemptError("Sigstore Rekor evidence contains invalid base64") from exc
    if len(key_id) != 32 or not set_bytes or not body:
        raise SuiteAttemptError("Sigstore Rekor key, timestamp, or body is empty or malformed")
    log_index = _integer(entry.get("logIndex"), label="Sigstore Rekor log index")
    integrated_seconds = _integer(
        entry.get("integratedTime"), label="Sigstore Rekor integrated time"
    )
    try:
        integrated = datetime.fromtimestamp(integrated_seconds, timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError) as exc:
        raise SuiteAttemptError("Sigstore Rekor integrated time is outside UTC range") from exc
    envelope = _object(root.get("dsseEnvelope"), label="Sigstore DSSE envelope")
    try:
        payload = base64.b64decode(
            _text(envelope.get("payload"), label="Sigstore DSSE payload"), validate=True
        )
    except (TypeError, ValueError) as exc:
        raise SuiteAttemptError("Sigstore DSSE payload contains invalid base64") from exc
    statement = _object(_strict_json(payload, label="in-toto statement"), label="statement")
    return SigstoreObservation(
        statement=statement,
        log_key_sha256=key_id.hex(),
        log_index=log_index,
        entry_id=f"rekor:{key_id.hex()}:{log_index}",
        integrated_at_utc=integrated,
        timestamp_token_sha256=hashlib.sha256(set_bytes).hexdigest(),
    )


def _verify_statement(
    observation: SigstoreObservation,
    *,
    snapshot: LedgerSnapshot,
    transition: LedgerTransition,
) -> None:
    statement = observation.statement
    if set(statement) != {"_type", "predicate", "predicateType", "subject"}:
        raise SuiteAttemptError("attested in-toto statement fields differ from the closed schema")
    if statement.get("_type") != "https://in-toto.io/Statement/v1":
        raise SuiteAttemptError("attested statement type differs")
    if statement.get("predicateType") != PREDICATE_TYPE:
        raise SuiteAttemptError("attested predicate type differs")
    subjects = _array(statement.get("subject"), label="attested subjects")
    if len(subjects) != 1:
        raise SuiteAttemptError("attestation must bind exactly one state-record subject")
    subject = _object(subjects[0], label="attested state subject")
    if set(subject) != {"digest", "name"} or subject.get("name") != transition.state_path:
        raise SuiteAttemptError("attested subject name differs from the manifest-derived path")
    subject_digest = _object(subject.get("digest"), label="attested subject digest")
    if subject_digest != {"sha256": transition.state.record_sha256}:
        raise SuiteAttemptError("attested subject digest differs from the state record")
    if statement.get("predicate") != ledger_predicate(snapshot, transition):
        raise SuiteAttemptError("attested ledger predicate differs from live GitHub state")


def _validated_gh_output(encoded: bytes) -> bytes:
    if len(encoded) > _MAX_GH_OUTPUT_BYTES:
        raise SuiteAttemptError("gh attestation verification output exceeds the limit")
    verified = _array(
        _strict_json(encoded, label="gh attestation verification output"),
        label="gh attestation verification output",
    )
    if len(verified) != 1:
        raise SuiteAttemptError("gh must verify exactly one attestation")
    result_row = _object(verified[0], label="gh attestation verification result")
    if not isinstance(result_row.get("verificationResult"), Mapping):
        raise SuiteAttemptError("gh output lacks a verified result")
    return encoded


class AttestationVerifier(Protocol):
    def verify(
        self,
        *,
        state_path: Path,
        bundle_path: Path,
        descriptor: SuiteAttestationDescriptor,
    ) -> bytes: ...


@dataclass(frozen=True)
class GhAttestationVerifier:
    executable: str = "gh"
    timeout_seconds: int = 60

    def verify(
        self,
        *,
        state_path: Path,
        bundle_path: Path,
        descriptor: SuiteAttestationDescriptor,
    ) -> bytes:
        command = [
            self.executable,
            "attestation",
            "verify",
            str(state_path),
            "--bundle",
            str(bundle_path),
            "--hostname",
            "github.com",
            "--repo",
            descriptor.expected_repository,
            "--cert-identity",
            descriptor.expected_signer_identity,
            "--cert-oidc-issuer",
            descriptor.expected_oidc_issuer,
            "--signer-digest",
            descriptor.expected_signer_digest,
            "--source-digest",
            descriptor.expected_signer_digest,
            "--source-ref",
            descriptor.expected_git_ref,
            "--deny-self-hosted-runners",
            "--predicate-type",
            PREDICATE_TYPE,
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
            raise SuiteAttemptError("cannot execute gh attestation verify") from exc
        if result.returncode != 0:
            detail = result.stderr[:4096].decode("utf-8", errors="replace").strip()
            raise SuiteAttemptError(f"GitHub rejected the Sigstore attestation: {detail}")
        return _validated_gh_output(result.stdout)


class C1AttestationVerifier(Protocol):
    def verify(
        self,
        *,
        subject_path: Path,
        bundle_path: Path,
        c1_commit: str,
        predicate_type: str,
    ) -> bytes: ...


@dataclass(frozen=True)
class GhC1AttestationVerifier:
    executable: str = "gh"
    timeout_seconds: int = 60

    def verify(
        self,
        *,
        subject_path: Path,
        bundle_path: Path,
        c1_commit: str,
        predicate_type: str,
    ) -> bytes:
        commit = _oid(c1_commit, label="C1 signer digest")
        if predicate_type not in {
            REGISTRATION_PREDICATE_TYPE,
            REGISTRY_RECORD_PREDICATE_TYPE,
        }:
            raise SuiteAttemptError("C1 verifier predicate type is not admitted")
        identity = f"https://github.com/{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}"
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
            commit,
            "--source-digest",
            commit,
            "--source-ref",
            C1_REF,
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
            raise SuiteAttemptError("cannot execute C1 gh attestation verification") from exc
        if result.returncode != 0:
            detail = result.stderr[:4096].decode("utf-8", errors="replace").strip()
            raise SuiteAttemptError(f"GitHub rejected the C1 attestation: {detail}")
        return _validated_gh_output(result.stdout)


def _assert_descriptor(descriptor: SuiteAttestationDescriptor) -> None:
    admitted_workflows = {
        WORKFLOW_PATH,
        ONLINE_EXECUTION_WORKFLOW_PATH,
        LABEL_RELEASE_WORKFLOW_PATH,
        ANALYSIS_WORKFLOW_PATH,
    }
    if descriptor.expected_workflow not in admitted_workflows:
        raise SuiteAttemptError("expected_workflow differs from a fixed C0 provider workflow")
    expected_identity = f"https://github.com/{REPOSITORY}/{descriptor.expected_workflow}@{C0_REF}"
    exact = {
        "expected_signer_identity": expected_identity,
        "expected_oidc_issuer": OIDC_ISSUER,
        "expected_repository": REPOSITORY,
        "expected_git_ref": C0_REF,
        "transparency_log_identity": REKOR_IDENTITY,
        "transparency_log_uri": REKOR_URI,
        "timestamp_authority_identity": REKOR_IDENTITY,
        "timestamp_authority_uri": REKOR_URI,
        "state_service_identity": STATE_SERVICE_IDENTITY,
        "state_service_uri": STATE_SERVICE_URI,
        "state_key_prefix": LEDGER_REF_PREFIX,
    }
    for name, expected in exact.items():
        if getattr(descriptor, name) != expected:
            raise SuiteAttemptError(f"{name} differs from the GitHub C0 policy")


class GitHubSuiteEvidenceVerifier:
    """Production adapter for exact GitHub, Sigstore, and protected-ledger evidence."""

    def __init__(
        self,
        namespace: str | Path,
        *,
        api: GitHubApi | None = None,
        attestation_verifier: AttestationVerifier | None = None,
    ) -> None:
        self.namespace = Path(namespace)
        self.api = GhApiClient() if api is None else api
        self.attestation_verifier = (
            GhAttestationVerifier() if attestation_verifier is None else attestation_verifier
        )
        self._snapshot: LedgerSnapshot | None = None

    def _snapshot_for(self, attempt_id: str) -> LedgerSnapshot:
        if self._snapshot is None:
            snapshot = load_ledger_snapshot(
                repository=REPOSITORY,
                suite_attempt_id=attempt_id,
                api=self.api,
            )
            expected_files = {
                f"{transition.state.sequence:03d}.state.json": transition.state_bytes
                for transition in snapshot.transitions
            }
            observed = {
                path.name
                for path in self.namespace.iterdir()
                if re.fullmatch(r"[0-9]{3}\.state\.json", path.name)
            }
            if observed != set(expected_files):
                raise SuiteAttemptError(
                    "local suite chain is stale or ahead of the GitHub ledger tip"
                )
            for name, encoded in expected_files.items():
                path = self.namespace / name
                metadata = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
                    raise SuiteAttemptError("local suite state is not a regular file")
                if path.read_bytes() != encoded:
                    raise SuiteAttemptError("local suite state differs from the GitHub ledger blob")
            self._snapshot = snapshot
        if self._snapshot.transitions[0].state.suite_attempt_id != attempt_id:
            raise SuiteAttemptError("one verifier instance cannot cross suite attempts")
        return self._snapshot

    def verify(
        self,
        *,
        bundle: bytes,
        evidence: SuiteAttestationEvidence,
        descriptor: SuiteAttestationDescriptor,
        state_record_bytes: bytes,
    ) -> SuiteProviderClaims:
        _assert_descriptor(descriptor)
        state = _state_from_bytes(state_record_bytes)
        snapshot = self._snapshot_for(state.suite_attempt_id)
        if state.sequence >= len(snapshot.transitions):
            raise SuiteAttemptError("state sequence is absent from the GitHub ledger")
        transition = snapshot.transitions[state.sequence]
        if transition.state_bytes != state_record_bytes:
            raise SuiteAttemptError("attested state bytes differ from the GitHub ledger")
        observation = parse_sigstore_bundle(bundle)
        if (
            descriptor.transparency_log_public_key_sha256 != observation.log_key_sha256
            or descriptor.timestamp_authority_public_key_sha256 != observation.log_key_sha256
        ):
            raise SuiteAttemptError("Rekor key ID differs from the descriptor key pin")
        _verify_statement(observation, snapshot=snapshot, transition=transition)
        with tempfile.TemporaryDirectory(prefix="fractal-attestation-") as temporary:
            root = Path(temporary)
            state_path = root / "state.json"
            bundle_path = root / "bundle.json"
            state_path.write_bytes(state_record_bytes)
            bundle_path.write_bytes(bundle)
            os.chmod(state_path, 0o600)
            os.chmod(bundle_path, 0o600)
            self.attestation_verifier.verify(
                state_path=state_path,
                bundle_path=bundle_path,
                descriptor=descriptor,
            )
        return SuiteProviderClaims(
            subject_sha256=transition.state.record_sha256,
            bundle_sha256=hashlib.sha256(bundle).hexdigest(),
            signer_identity=descriptor.expected_signer_identity,
            oidc_issuer=descriptor.expected_oidc_issuer,
            repository=descriptor.expected_repository,
            workflow=descriptor.expected_workflow,
            git_ref=descriptor.expected_git_ref,
            signer_digest=descriptor.expected_signer_digest,
            github_hosted_runner=True,
            transparency_log_identity=REKOR_IDENTITY,
            transparency_entry_id=observation.entry_id,
            transparency_log_index=observation.log_index,
            integrated_at_utc=observation.integrated_at_utc,
            timestamp_authority_identity=REKOR_IDENTITY,
            timestamp_token_sha256=observation.timestamp_token_sha256,
            signed_at_utc=observation.integrated_at_utc,
            state_service_identity=STATE_SERVICE_IDENTITY,
            state_key=snapshot.state_key,
            transition_id=transition.commit_oid,
            previous_transition_id=transition.previous_commit_oid,
            signature_verified=True,
            transparency_verified=True,
            timestamp_verified=True,
            exclusive_transition=True,
        )


def _write_exclusive(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def c1_registration_predicate(
    *,
    c1_commit: str,
    c0_commit: str,
    tag_object_id: str,
    tag_object_type: str,
    manifest_digest: str,
    manifest_file_digest: str,
    lock_file_digest: str,
    transition_receipt_file_digest: str,
    c0_public_verification_file_digest: str,
    c0_public_verification_binding_digest: str,
    candidate_manifest_digest: str,
    candidate_manifest_file_digest: str,
    candidate_assembly_receipt_file_digest: str,
    reservation_file_digest: str,
) -> dict[str, object]:
    """Return the closed predicate for the prospective C1 registration deposit."""

    for name, value in (
        ("c1_commit", c1_commit),
        ("c0_commit", c0_commit),
        ("tag_object_id", tag_object_id),
    ):
        _oid(value, label=name)
    for name, value in (
        ("manifest_digest", manifest_digest),
        ("manifest_file_digest", manifest_file_digest),
        ("lock_file_digest", lock_file_digest),
        ("transition_receipt_file_digest", transition_receipt_file_digest),
        (
            "c0_public_verification_file_digest",
            c0_public_verification_file_digest,
        ),
        (
            "c0_public_verification_binding_digest",
            c0_public_verification_binding_digest,
        ),
        ("candidate_manifest_digest", candidate_manifest_digest),
        ("candidate_manifest_file_digest", candidate_manifest_file_digest),
        (
            "candidate_assembly_receipt_file_digest",
            candidate_assembly_receipt_file_digest,
        ),
        ("reservation_file_digest", reservation_file_digest),
    ):
        _digest(value, label=name)
    if tag_object_type not in {"commit", "tag"}:
        raise SuiteAttemptError("C1 tag object must be a commit or annotated tag")
    return {
        "c0_public_verification": {
            "binding_sha256": c0_public_verification_binding_digest,
            "file_sha256": c0_public_verification_file_digest,
            "path": C0_PUBLIC_VERIFICATION_PATH,
            "release_tag": C0_EVIDENCE_RELEASE_TAG,
            "schema_version": C0_PUBLIC_VERIFICATION_SCHEMA,
            "target_commit": c0_commit,
        },
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "freeze": {
            "c0_commit": c0_commit,
            "c0_ref": C0_REF,
            "c1_commit": c1_commit,
            "c1_ref": C1_REF,
            "tag_object_id": tag_object_id,
            "tag_object_type": tag_object_type,
        },
        "lock": {
            "file_sha256": lock_file_digest,
            "manifest_sha256": manifest_digest,
            "path": C1_LOCK_PATH,
        },
        "manifest": {
            "file_sha256": manifest_file_digest,
            "manifest_sha256": manifest_digest,
            "path": C1_MANIFEST_PATH,
        },
        "manifest_transition": {
            "candidate_manifest_assembly_receipt_file_sha256": (
                candidate_assembly_receipt_file_digest
            ),
            "candidate_manifest_file_sha256": candidate_manifest_file_digest,
            "candidate_manifest_sha256": candidate_manifest_digest,
            "file_sha256": transition_receipt_file_digest,
            "path": C1_TRANSITION_RECEIPT_PATH,
            "schema_version": C1_MANIFEST_TRANSITION_RECEIPT_SCHEMA,
        },
        "registry_reservation": {
            "deposition_id": ZENODO_RECORD_ID,
            "direct_registry_record_uri": ZENODO_REGISTRY_URI,
            "file_sha256": reservation_file_digest,
            "path": C1_RESERVATION_PATH,
            "registry_identity": ZENODO_REGISTRY_IDENTITY,
            "reserved_doi": ZENODO_RESERVED_DOI,
        },
        "schema_version": REGISTRATION_PREDICATE_SCHEMA,
    }


def prepare_c1_registration(
    *,
    repository: str,
    github_ref: str,
    github_sha: str,
    workflow_ref: str,
    workflow_sha: str,
    repository_root: Path,
    c0_public_verification_path: Path,
    output_dir: Path,
) -> Mapping[str, str]:
    """Close and materialize the only C1 manifest admitted for registration."""

    c1_commit = _oid(github_sha, label="github.sha")
    if repository != REPOSITORY or github_ref != C1_REF or workflow_sha != c1_commit:
        raise SuiteAttemptError("registration must run in the fixed repository at C1")
    expected_workflow_ref = f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}"
    if workflow_ref != expected_workflow_ref:
        raise SuiteAttemptError("registration workflow_ref differs from the fixed C1 identity")
    root = repository_root.resolve(strict=True)
    if not repository_root.is_absolute() or repository_root != root:
        raise SuiteAttemptError("C1 repository root must be one absolute real directory")
    if Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve(strict=True) != root:
        raise SuiteAttemptError("C1 repository root differs from the Git worktree root")
    if _git_output(root, "rev-parse", "HEAD") != c1_commit:
        raise SuiteAttemptError("checked-out commit differs from the C1 workflow source")
    if _git_output(root, "rev-parse", f"{C1_REF}^{{commit}}") != c1_commit:
        raise SuiteAttemptError("fixed C1 tag does not resolve to the workflow commit")
    tag_object_id = _oid(
        _git_output(root, "rev-parse", C1_REF),
        label="C1 tag object ID",
    )
    tag_object_type = _git_output(root, "cat-file", "-t", tag_object_id)
    if tag_object_type == "tag":
        tag_record = _git_output(root, "cat-file", "-p", tag_object_id)
        tag_headers = tag_record.partition("\n\n")[0].splitlines()
        taggers = []
        for line in tag_headers:
            if line.startswith("tagger "):
                taggers.append(line.removeprefix("tagger "))
        if len(taggers) != 1 or _GIT_IDENTITY_HEADER.fullmatch(taggers[0]) is None:
            raise SuiteAttemptError("annotated C1 tag must use the fixed mhdk1602 tagger identity")
    c0_commit = _oid(
        _git_output(root, "rev-parse", f"{C0_REF}^{{commit}}"),
        label="C0 commit",
    )
    ancestry = _git_output(root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if ancestry != [c1_commit, c0_commit]:
        raise SuiteAttemptError("C1 must be the direct frozen-transition child of C0")
    identities = _git_output(
        root,
        "show",
        "-s",
        "--format=%an%x00%ae%x00%cn%x00%ce",
        c1_commit,
    ).split("\x00")
    if identities != [
        GIT_IDENTITY_NAME,
        GIT_IDENTITY_EMAIL,
        GIT_IDENTITY_NAME,
        GIT_IDENTITY_EMAIL,
    ]:
        raise SuiteAttemptError("C1 author and committer must both be the fixed mhdk1602 identity")
    commit_message = _git_output(root, "show", "-s", "--format=%B", c1_commit)
    if re.search(r"(?im)^co-authored-by\s*:", commit_message):
        raise SuiteAttemptError("C1 commit must not contain a co-author trailer")
    changed = {
        line
        for line in _git_output(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "HEAD",
        ).splitlines()
        if line
    }
    if changed != {C1_MANIFEST_PATH, C1_LOCK_PATH, C1_TRANSITION_RECEIPT_PATH}:
        raise SuiteAttemptError(
            "C1 commit must change only the fixed manifest, lock, and transition-receipt paths"
        )
    if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise SuiteAttemptError("C1 checkout must contain no tracked or untracked worktree changes")

    manifest_path = root / C1_MANIFEST_PATH
    lock_path = root / C1_LOCK_PATH
    transition_receipt_path = root / C1_TRANSITION_RECEIPT_PATH
    reservation_path = root / C1_RESERVATION_PATH
    manifest_bytes = _regular_file_bytes(manifest_path, label="C1 manifest")
    lock_bytes = _regular_file_bytes(lock_path, label="C1 lock")
    transition_receipt_bytes = _regular_file_bytes(
        transition_receipt_path,
        label="C1 manifest transition receipt",
    )
    reservation_bytes = _regular_file_bytes(reservation_path, label="Zenodo reservation")
    checkout_bytes = {
        C1_MANIFEST_PATH: manifest_bytes,
        C1_LOCK_PATH: lock_bytes,
        C1_TRANSITION_RECEIPT_PATH: transition_receipt_bytes,
        C1_RESERVATION_PATH: reservation_bytes,
    }
    for relative_path, encoded in checkout_bytes.items():
        object_id = _oid(
            _git_output(root, "rev-parse", f"{c1_commit}:{relative_path}"),
            label=f"C1 {relative_path} blob",
        )
        if object_id != _git_blob_oid(encoded):
            raise SuiteAttemptError(
                f"C1 checkout bytes for {relative_path} differ from the committed Git blob"
            )
    manifest = _frozen_manifest_from_bytes(manifest_bytes)
    semantic_digest = manifest_sha256(manifest)
    if lock_bytes != f"{semantic_digest}\n".encode("ascii"):
        raise SuiteAttemptError("C1 lock does not equal the frozen manifest digest plus LF")
    manifest_file_digest = hashlib.sha256(manifest_bytes).hexdigest()
    lock_file_digest = hashlib.sha256(lock_bytes).hexdigest()
    c0_public_verification, c0_public_verification_bytes = _admit_c0_public_verification(
        c0_public_verification_path,
        frozen_manifest=manifest,
        frozen_manifest_bytes=manifest_bytes,
        c0_commit=c0_commit,
    )
    try:
        transition_receipt = loads_c1_manifest_transition_receipt(transition_receipt_bytes)
        verify_c1_manifest_transition_receipt_bindings(
            transition_receipt,
            frozen_manifest=manifest,
            frozen_manifest_bytes=manifest_bytes,
            c0_commit=c0_commit,
        )
    except C1ManifestTransitionError as exc:
        raise SuiteAttemptError(f"C1 transition receipt is invalid: {exc}") from exc
    transition_receipt_file_digest = hashlib.sha256(transition_receipt_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="fractal-c1-reservation-") as directory:
        reservation_snapshot = Path(directory) / Path(C1_RESERVATION_PATH).name
        _write_exclusive(reservation_snapshot, reservation_bytes)
        _, reservation_file_digest = _load_zenodo_reservation(reservation_snapshot)
    if hashlib.sha256(reservation_bytes).hexdigest() != reservation_file_digest:
        raise SuiteAttemptError("Zenodo reservation changed during C1 admission")
    predicate = c1_registration_predicate(
        c1_commit=c1_commit,
        c0_commit=c0_commit,
        tag_object_id=tag_object_id,
        tag_object_type=tag_object_type,
        manifest_digest=semantic_digest,
        manifest_file_digest=manifest_file_digest,
        lock_file_digest=lock_file_digest,
        transition_receipt_file_digest=transition_receipt_file_digest,
        c0_public_verification_file_digest=c0_public_verification.file_sha256,
        c0_public_verification_binding_digest=c0_public_verification.binding_sha256,
        candidate_manifest_digest=transition_receipt.candidate_manifest_sha256,
        candidate_manifest_file_digest=transition_receipt.candidate_manifest_file_sha256,
        candidate_assembly_receipt_file_digest=(
            transition_receipt.candidate_manifest_assembly_receipt_file_sha256
        ),
        reservation_file_digest=reservation_file_digest,
    )
    receipt = {
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "predicate": predicate,
        "predicate_type": REGISTRATION_PREDICATE_TYPE,
        "repository": REPOSITORY,
        "schema_version": REGISTRATION_RECEIPT_SCHEMA,
        "workflow_ref": workflow_ref,
        "workflow_sha": workflow_sha,
    }
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    predicate_path = output_dir / "registration-predicate.json"
    receipt_path = output_dir / "registration-validation.json"
    _write_exclusive(predicate_path, _canonical_bytes(predicate) + b"\n")
    _write_exclusive(receipt_path, _canonical_bytes(receipt) + b"\n")
    for relative_path, encoded in checkout_bytes.items():
        if _regular_file_bytes(root / relative_path, label=f"C1 {relative_path}") != encoded:
            raise SuiteAttemptError(f"C1 checkout path {relative_path} changed during admission")
    if (
        _regular_file_bytes(
            c0_public_verification_path,
            label="C0 public-verification receipt",
        )
        != c0_public_verification_bytes
    ):
        raise SuiteAttemptError("C0 public-verification receipt changed during C1 admission")
    return {
        "c0_commit": c0_commit,
        "c0_public_verification_file_digest": c0_public_verification.file_sha256,
        "c1_commit": c1_commit,
        "lock_file_digest": lock_file_digest,
        "manifest_digest": semantic_digest,
        "manifest_file_digest": manifest_file_digest,
        "predicate_path": str(predicate_path),
        "receipt_path": str(receipt_path),
        "reservation_file_digest": reservation_file_digest,
        "transition_receipt_file_digest": transition_receipt_file_digest,
        "transition_receipt_path": str(transition_receipt_path),
        "tag_object_id": tag_object_id,
        "tag_object_type": tag_object_type,
    }


def _canonical_object_file(path: Path, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    encoded = _regular_file_bytes(path, label=label)
    value = _object(_strict_json(encoded, label=label), label=label)
    if encoded != _canonical_bytes(value) + b"\n":
        raise SuiteAttemptError(f"{label} must be canonical JSON plus one LF")
    return value, encoded


def _load_closed_c1_predicate(path: Path) -> tuple[Mapping[str, Any], str]:
    predicate, _ = _canonical_object_file(path, label="C1 registration predicate")
    c0_public_verification = _object(
        predicate.get("c0_public_verification"),
        label="C1 C0 public-verification predicate",
    )
    freeze = _object(predicate.get("freeze"), label="C1 freeze predicate")
    manifest = _object(predicate.get("manifest"), label="C1 manifest predicate")
    lock = _object(predicate.get("lock"), label="C1 lock predicate")
    transition = _object(
        predicate.get("manifest_transition"),
        label="C1 manifest transition predicate",
    )
    reservation = _object(
        predicate.get("registry_reservation"),
        label="C1 registry reservation predicate",
    )
    c1_commit = _oid(freeze.get("c1_commit"), label="C1 predicate commit")
    expected = c1_registration_predicate(
        c1_commit=c1_commit,
        c0_commit=_oid(freeze.get("c0_commit"), label="C0 predicate commit"),
        tag_object_id=_oid(freeze.get("tag_object_id"), label="C1 predicate tag object"),
        tag_object_type=_text(
            freeze.get("tag_object_type"),
            label="C1 predicate tag-object type",
        ),
        manifest_digest=_digest(
            manifest.get("manifest_sha256"),
            label="C1 predicate manifest digest",
        ),
        manifest_file_digest=_digest(
            manifest.get("file_sha256"),
            label="C1 predicate manifest-file digest",
        ),
        lock_file_digest=_digest(
            lock.get("file_sha256"),
            label="C1 predicate lock-file digest",
        ),
        transition_receipt_file_digest=_digest(
            transition.get("file_sha256"),
            label="C1 predicate transition-receipt digest",
        ),
        c0_public_verification_file_digest=_digest(
            c0_public_verification.get("file_sha256"),
            label="C1 predicate C0 public-verification file digest",
        ),
        c0_public_verification_binding_digest=_digest(
            c0_public_verification.get("binding_sha256"),
            label="C1 predicate C0 public-verification binding digest",
        ),
        candidate_manifest_digest=_digest(
            transition.get("candidate_manifest_sha256"),
            label="C1 predicate candidate-manifest digest",
        ),
        candidate_manifest_file_digest=_digest(
            transition.get("candidate_manifest_file_sha256"),
            label="C1 predicate candidate-manifest-file digest",
        ),
        candidate_assembly_receipt_file_digest=_digest(
            transition.get("candidate_manifest_assembly_receipt_file_sha256"),
            label="C1 predicate candidate-assembly-receipt digest",
        ),
        reservation_file_digest=_digest(
            reservation.get("file_sha256"),
            label="C1 predicate reservation-file digest",
        ),
    )
    if type(reservation.get("deposition_id")) is not int or predicate != expected:
        raise SuiteAttemptError("C1 registration predicate differs from the closed policy")
    return predicate, c1_commit


def _verify_closed_c1_statement(
    observation: SigstoreObservation,
    *,
    predicate_type: str,
    predicate: Mapping[str, Any],
    subject_name: str,
    subject_digest: str,
) -> None:
    statement = observation.statement
    if set(statement) != {"_type", "predicate", "predicateType", "subject"}:
        raise SuiteAttemptError("C1 in-toto statement fields differ from the closed schema")
    if statement.get("_type") != "https://in-toto.io/Statement/v1":
        raise SuiteAttemptError("C1 attestation has another statement type")
    if statement.get("predicateType") != predicate_type or statement.get("predicate") != predicate:
        raise SuiteAttemptError("C1 attestation has another predicate")
    subjects = _array(statement.get("subject"), label="C1 attestation subjects")
    if len(subjects) != 1:
        raise SuiteAttemptError("C1 attestation must contain exactly one subject")
    subject = _object(subjects[0], label="C1 attestation subject")
    if set(subject) != {"digest", "name"} or subject.get("name") != subject_name:
        raise SuiteAttemptError("C1 attestation subject name differs from the fixed path")
    digest = _object(subject.get("digest"), label="C1 attestation subject digest")
    if digest != {"sha256": _digest(subject_digest, label="C1 subject digest")}:
        raise SuiteAttemptError("C1 attestation subject digest differs from the fixed bytes")


def _utc_datetime(value: object, *, label: str) -> datetime:
    text = _text(value, label=label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SuiteAttemptError(f"{label} is not an ISO 8601 timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SuiteAttemptError(f"{label} must use UTC")
    return parsed


def registry_record_predicate(
    *,
    c1_commit: str,
    c0_commit: str,
    manifest_digest: str,
    manifest_file_digest: str,
    registry_record: ProtocolRegistryRecord,
    manifest_bundle_digest: str,
    manifest_observation: SigstoreObservation,
) -> dict[str, object]:
    """Bind the registry record to the verified manifest attestation that timed it."""

    c1 = _oid(c1_commit, label="registry predicate C1 commit")
    c0 = _oid(c0_commit, label="registry predicate C0 commit")
    semantic_digest = _digest(manifest_digest, label="registry predicate manifest digest")
    file_digest = _digest(
        manifest_file_digest,
        label="registry predicate manifest-file digest",
    )
    bundle_digest = _digest(
        manifest_bundle_digest,
        label="registry predicate manifest-bundle digest",
    )
    if not isinstance(registry_record, ProtocolRegistryRecord):
        raise SuiteAttemptError("registry predicate requires a typed protocol registry record")
    if (
        registry_record.manifest_sha256 != semantic_digest
        or registry_record.protocol_version != "0.3.0"
        or registry_record.registry_identity != ZENODO_REGISTRY_IDENTITY
        or registry_record.registry_uri != ZENODO_REGISTRY_URI
        or registry_record.registered_at_utc != manifest_observation.integrated_at_utc
    ):
        raise SuiteAttemptError("protocol registry record differs from the fixed Zenodo identity")
    _assert_exact_zenodo_url(
        registry_record.registry_uri,
        expected=ZENODO_REGISTRY_URI,
        label="protocol registry-record URI",
    )
    return {
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "freeze": {
            "c0_commit": c0,
            "c0_ref": C0_REF,
            "c1_commit": c1,
            "c1_ref": C1_REF,
        },
        "manifest": {
            "file_sha256": file_digest,
            "manifest_sha256": semantic_digest,
            "path": C1_MANIFEST_PATH,
        },
        "manifest_attestation": {
            "bundle_sha256": bundle_digest,
            "integrated_at_utc": manifest_observation.integrated_at_utc,
            "predicate_type": REGISTRATION_PREDICATE_TYPE,
            "rekor_entry_id": manifest_observation.entry_id,
            "rekor_log_index": manifest_observation.log_index,
            "rekor_log_key_sha256": manifest_observation.log_key_sha256,
            "rekor_timestamp_token_sha256": manifest_observation.timestamp_token_sha256,
        },
        "registry_record": {
            "deposition_id": ZENODO_RECORD_ID,
            "direct_uri": ZENODO_REGISTRY_URI,
            "file_sha256": registry_record.record_sha256,
            "path": REGISTRY_RECORD_SUBJECT_PATH,
            "registered_at_utc": registry_record.registered_at_utc,
            "registry_identity": ZENODO_REGISTRY_IDENTITY,
            "reserved_doi": ZENODO_RESERVED_DOI,
        },
        "schema_version": REGISTRY_RECORD_PREDICATE_SCHEMA,
    }


def materialize_protocol_registry_record(
    *,
    manifest_path: Path,
    lock_path: Path,
    reservation_path: Path,
    manifest_bundle_path: Path,
    manifest_predicate_path: Path,
    record_output_path: Path,
    registry_predicate_output_path: Path,
    receipt_output_path: Path,
    verification_output_path: Path,
    verifier: C1AttestationVerifier | None = None,
) -> Mapping[str, str]:
    """Create the fixed Zenodo record using the verified manifest Rekor time."""

    manifest_resolved = manifest_path.resolve(strict=True)
    repository_root = manifest_resolved.parents[1]
    exact_paths = {
        "manifest": (manifest_resolved, repository_root / C1_MANIFEST_PATH),
        "lock": (lock_path.resolve(strict=True), repository_root / C1_LOCK_PATH),
        "reservation": (
            reservation_path.resolve(strict=True),
            repository_root / C1_RESERVATION_PATH,
        ),
        "registry record": (
            record_output_path.resolve(strict=False),
            repository_root / REGISTRY_RECORD_SUBJECT_PATH,
        ),
    }
    for label, (observed, expected) in exact_paths.items():
        if observed != expected:
            raise SuiteAttemptError(f"C1 {label} path differs from the fixed checkout path")

    predicate_bytes = _regular_file_bytes(
        manifest_predicate_path,
        label="C1 registration predicate",
    )
    with tempfile.TemporaryDirectory(prefix="fractal-c1-predicate-") as directory:
        predicate_snapshot = Path(directory) / "registration-predicate.json"
        _write_exclusive(predicate_snapshot, predicate_bytes)
        predicate, c1_commit = _load_closed_c1_predicate(predicate_snapshot)
    freeze = _object(predicate.get("freeze"), label="C1 freeze predicate")
    manifest_row = _object(predicate.get("manifest"), label="C1 manifest predicate")
    lock_row = _object(predicate.get("lock"), label="C1 lock predicate")
    reservation_row = _object(
        predicate.get("registry_reservation"),
        label="C1 reservation predicate",
    )
    manifest_bytes = _regular_file_bytes(manifest_resolved, label="frozen C1 manifest")
    manifest_file_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_file_digest != manifest_row.get("file_sha256"):
        raise SuiteAttemptError("C1 manifest bytes differ from the signed predicate")
    manifest = _frozen_manifest_from_bytes(manifest_bytes)
    semantic_digest = manifest_sha256(manifest)
    if semantic_digest != manifest_row.get("manifest_sha256"):
        raise SuiteAttemptError("C1 manifest meaning differs from the signed predicate")
    lock_bytes = _regular_file_bytes(lock_path, label="C1 manifest lock")
    if lock_bytes != f"{semantic_digest}\n".encode("ascii") or hashlib.sha256(
        lock_bytes
    ).hexdigest() != lock_row.get("file_sha256"):
        raise SuiteAttemptError("C1 lock differs from the signed manifest predicate")
    reservation_bytes = _regular_file_bytes(reservation_path, label="Zenodo reservation")
    with tempfile.TemporaryDirectory(prefix="fractal-c1-reservation-") as directory:
        reservation_snapshot = Path(directory) / Path(C1_RESERVATION_PATH).name
        _write_exclusive(reservation_snapshot, reservation_bytes)
        _, reservation_digest = _load_zenodo_reservation(reservation_snapshot)
    if reservation_digest != reservation_row.get("file_sha256"):
        raise SuiteAttemptError("Zenodo reservation differs from the signed predicate")

    bundle = _regular_file_bytes(manifest_bundle_path, label="manifest Sigstore bundle")
    active_verifier = verifier if verifier is not None else GhC1AttestationVerifier()
    verified = _verify_c1_attestation_snapshot(
        subject_name=C1_MANIFEST_PATH,
        subject_bytes=manifest_bytes,
        bundle_bytes=bundle,
        c1_commit=c1_commit,
        predicate_type=REGISTRATION_PREDICATE_TYPE,
        verifier=active_verifier,
    )
    observation = parse_sigstore_bundle(bundle)
    _verify_closed_c1_statement(
        observation,
        predicate_type=REGISTRATION_PREDICATE_TYPE,
        predicate=predicate,
        subject_name=C1_MANIFEST_PATH,
        subject_digest=manifest_file_digest,
    )
    record = ProtocolRegistryRecord(
        manifest_sha256=semantic_digest,
        protocol_version="0.3.0",
        registered_at_utc=observation.integrated_at_utc,
        registry_identity=ZENODO_REGISTRY_IDENTITY,
        registry_uri=ZENODO_REGISTRY_URI,
    )
    registry_predicate = registry_record_predicate(
        c1_commit=c1_commit,
        c0_commit=_oid(freeze.get("c0_commit"), label="C0 predicate commit"),
        manifest_digest=semantic_digest,
        manifest_file_digest=manifest_file_digest,
        registry_record=record,
        manifest_bundle_digest=hashlib.sha256(bundle).hexdigest(),
        manifest_observation=observation,
    )
    receipt = {
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "manifest_attestation_verification_sha256": hashlib.sha256(verified).hexdigest(),
        "predicate": registry_predicate,
        "predicate_type": REGISTRY_RECORD_PREDICATE_TYPE,
        "registry_record_sha256": record.record_sha256,
        "schema_version": REGISTRY_MATERIALIZATION_SCHEMA,
    }
    current_inputs = (
        (manifest_resolved, manifest_bytes, "frozen C1 manifest"),
        (lock_path, lock_bytes, "C1 manifest lock"),
        (manifest_predicate_path, predicate_bytes, "C1 registration predicate"),
        (manifest_bundle_path, bundle, "manifest Sigstore bundle"),
        (reservation_path, reservation_bytes, "Zenodo reservation"),
    )
    for path, expected, label in current_inputs:
        if _regular_file_bytes(path, label=label) != expected:
            raise SuiteAttemptError(f"{label} changed during registry materialization")
    _write_exclusive(record_output_path, record.canonical_bytes() + b"\n")
    _write_exclusive(
        registry_predicate_output_path,
        _canonical_bytes(registry_predicate) + b"\n",
    )
    _write_exclusive(receipt_output_path, _canonical_bytes(receipt) + b"\n")
    _write_exclusive(verification_output_path, verified)
    return {
        "c1_commit": c1_commit,
        "manifest_rekor_entry_id": observation.entry_id,
        "manifest_rekor_integrated_at_utc": observation.integrated_at_utc,
        "record_path": str(record_output_path),
        "registry_predicate_path": str(registry_predicate_output_path),
        "registry_record_digest": record.record_sha256,
    }


def _load_registry_record_predicate(
    path: Path,
    *,
    record: ProtocolRegistryRecord,
) -> tuple[Mapping[str, Any], str, SigstoreObservation]:
    predicate, _ = _canonical_object_file(path, label="registry-record predicate")
    freeze = _object(predicate.get("freeze"), label="registry-record freeze predicate")
    manifest = _object(predicate.get("manifest"), label="registry-record manifest predicate")
    manifest_attestation = _object(
        predicate.get("manifest_attestation"),
        label="registry-record manifest attestation",
    )
    first_log_key = _digest(
        manifest_attestation.get("rekor_log_key_sha256"),
        label="manifest Rekor log-key digest",
    )
    first_log_index = _integer(
        manifest_attestation.get("rekor_log_index"),
        label="manifest Rekor log index",
    )
    first_observation = SigstoreObservation(
        statement={},
        log_key_sha256=first_log_key,
        log_index=first_log_index,
        entry_id=f"rekor:{first_log_key}:{first_log_index}",
        integrated_at_utc=_text(
            manifest_attestation.get("integrated_at_utc"),
            label="manifest Rekor integrated time",
        ),
        timestamp_token_sha256=_digest(
            manifest_attestation.get("rekor_timestamp_token_sha256"),
            label="manifest Rekor timestamp digest",
        ),
    )
    c1_commit = _oid(
        freeze.get("c1_commit"),
        label="registry predicate C1 commit",
    )
    expected = registry_record_predicate(
        c1_commit=c1_commit,
        c0_commit=_oid(freeze.get("c0_commit"), label="registry predicate C0 commit"),
        manifest_digest=_digest(
            manifest.get("manifest_sha256"),
            label="registry predicate manifest digest",
        ),
        manifest_file_digest=_digest(
            manifest.get("file_sha256"),
            label="registry predicate manifest-file digest",
        ),
        registry_record=record,
        manifest_bundle_digest=_digest(
            manifest_attestation.get("bundle_sha256"),
            label="manifest bundle digest",
        ),
        manifest_observation=first_observation,
    )
    if (
        manifest_attestation.get("rekor_entry_id") != first_observation.entry_id
        or predicate != expected
    ):
        raise SuiteAttemptError("registry-record predicate differs from the closed policy")
    return predicate, c1_commit, first_observation


def verify_c1_registry_record_attestation(
    *,
    record_path: Path,
    predicate_path: Path,
    bundle_path: Path,
    receipt_output_path: Path,
    verification_output_path: Path,
    verifier: C1AttestationVerifier | None = None,
) -> Mapping[str, str]:
    """Verify the separate Rekor attestation over the canonical registry record."""

    if record_path.name != REGISTRY_RECORD_SUBJECT_PATH:
        raise SuiteAttemptError("registry-record subject path differs from the fixed name")
    record_row, record_bytes = _canonical_object_file(
        record_path,
        label="protocol registry record",
    )
    record = ProtocolRegistryRecord.from_dict(record_row)
    if (
        record_bytes != record.canonical_bytes() + b"\n"
        or record.registry_identity != ZENODO_REGISTRY_IDENTITY
        or record.registry_uri != ZENODO_REGISTRY_URI
    ):
        raise SuiteAttemptError("protocol registry record differs from the fixed Zenodo record")
    _assert_exact_zenodo_url(
        record.registry_uri,
        expected=ZENODO_REGISTRY_URI,
        label="protocol registry-record URI",
    )
    predicate_bytes = _regular_file_bytes(predicate_path, label="registry-record predicate")
    with tempfile.TemporaryDirectory(prefix="fractal-c1-registry-predicate-") as directory:
        predicate_snapshot = Path(directory) / "registry-record-predicate.json"
        _write_exclusive(predicate_snapshot, predicate_bytes)
        predicate, c1_commit, first_observation = _load_registry_record_predicate(
            predicate_snapshot,
            record=record,
        )
    bundle = _regular_file_bytes(bundle_path, label="registry-record Sigstore bundle")
    active_verifier = verifier if verifier is not None else GhC1AttestationVerifier()
    verified = _verify_c1_attestation_snapshot(
        subject_name=REGISTRY_RECORD_SUBJECT_PATH,
        subject_bytes=record_bytes,
        bundle_bytes=bundle,
        c1_commit=c1_commit,
        predicate_type=REGISTRY_RECORD_PREDICATE_TYPE,
        verifier=active_verifier,
    )
    observation = parse_sigstore_bundle(bundle)
    _verify_closed_c1_statement(
        observation,
        predicate_type=REGISTRY_RECORD_PREDICATE_TYPE,
        predicate=predicate,
        subject_name=REGISTRY_RECORD_SUBJECT_PATH,
        subject_digest=record.record_sha256,
    )
    if (
        _utc_datetime(
            observation.integrated_at_utc,
            label="registry-record Rekor integrated time",
        )
        < _utc_datetime(
            first_observation.integrated_at_utc,
            label="manifest Rekor integrated time",
        )
        or observation.entry_id == first_observation.entry_id
        or observation.timestamp_token_sha256 == first_observation.timestamp_token_sha256
    ):
        raise SuiteAttemptError("registry-record Rekor entry does not follow the manifest entry")
    receipt = {
        "c1_commit": c1_commit,
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "manifest_rekor_entry_id": first_observation.entry_id,
        "manifest_rekor_integrated_at_utc": first_observation.integrated_at_utc,
        "registry_record_bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "registry_record_rekor_entry_id": observation.entry_id,
        "registry_record_rekor_integrated_at_utc": observation.integrated_at_utc,
        "registry_record_sha256": record.record_sha256,
        "registry_record_verification_sha256": hashlib.sha256(verified).hexdigest(),
        "schema_version": REGISTRY_ATTESTATION_RECEIPT_SCHEMA,
    }
    for path, expected, label in (
        (record_path, record_bytes, "protocol registry record"),
        (predicate_path, predicate_bytes, "registry-record predicate"),
        (bundle_path, bundle, "registry-record Sigstore bundle"),
    ):
        if _regular_file_bytes(path, label=label) != expected:
            raise SuiteAttemptError(f"{label} changed during attestation verification")
    _write_exclusive(receipt_output_path, _canonical_bytes(receipt) + b"\n")
    _write_exclusive(verification_output_path, verified)
    return {
        "c1_commit": c1_commit,
        "registry_record_bundle_digest": hashlib.sha256(bundle).hexdigest(),
        "registry_record_digest": record.record_sha256,
        "registry_record_rekor_entry_id": observation.entry_id,
        "registry_record_rekor_integrated_at_utc": observation.integrated_at_utc,
    }


def _workflow_receipt(
    snapshot: LedgerSnapshot,
    transition: LedgerTransition,
    *,
    workflow_ref: str,
    workflow_sha: str,
) -> dict[str, object]:
    opened = snapshot.transitions[0].state.payload
    if not isinstance(opened, SuiteOpenBindings):
        raise SuiteAttemptError("GitHub ledger does not begin with OPENED bindings")
    return {
        "attestation_descriptor_sha256": opened.attestation_descriptor_sha256,
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "git_ref": C0_REF,
        "ledger_predicate": ledger_predicate(snapshot, transition),
        "previous_transition_id": transition.previous_commit_oid,
        "repository": snapshot.repository,
        "schema_version": WORKFLOW_RECEIPT_SCHEMA,
        "signer_digest": workflow_sha,
        "state_name": transition.state.state,
        "state_path": transition.state_path,
        "state_record_sha256": transition.state.record_sha256,
        "state_sequence": transition.state.sequence,
        "state_service_key": snapshot.state_key,
        "suite_attempt_id": transition.state.suite_attempt_id,
        "transition_id": transition.commit_oid,
        "workflow_ref": workflow_ref,
    }


def validate_workflow_transition(
    *,
    ledger_commit: str,
    repository: str,
    github_ref: str,
    github_sha: str,
    workflow_ref: str,
    workflow_sha: str,
    output_dir: Path,
    api: GitHubApi,
) -> Mapping[str, str]:
    """Validate the sole ledger tip admitted by the trusted C0 workflow."""

    commit_oid = _oid(ledger_commit, label="ledger_commit")
    signer_digest = _oid(workflow_sha, label="github.workflow_sha")
    if repository != REPOSITORY or github_ref != C0_REF:
        raise SuiteAttemptError("state attestation must run in the fixed repository at C0")
    if github_sha != signer_digest:
        raise SuiteAttemptError("workflow source and dispatch source commits differ")
    commit = _commit_row(api, repository, commit_oid)
    tree = _object(commit.get("tree"), label="candidate ledger tree")
    blobs = _tree_blobs(api, repository, _oid(tree.get("sha"), label="candidate tree ID"))
    state_paths = sorted(
        (path for path in blobs if path.startswith(f"{LEDGER_PATH_PREFIX}/")),
        key=lambda item: item.encode("utf-8"),
    )
    if not state_paths:
        raise SuiteAttemptError("candidate ledger tip contains no state records")
    latest_path = state_paths[-1]
    match = re.fullmatch(
        rf"{re.escape(LEDGER_PATH_PREFIX)}/([0-9a-f]{{64}})/(\d{{3}})\.state\.json",
        latest_path,
    )
    if match is None:
        raise SuiteAttemptError("candidate ledger tip lacks a canonical state path")
    attempt_id = match.group(1)
    snapshot = load_ledger_snapshot(
        repository=repository,
        suite_attempt_id=attempt_id,
        api=api,
        require_ruleset_bypass_visibility=False,
    )
    if snapshot.tip.commit_oid != commit_oid:
        raise SuiteAttemptError("caller-selected commit is not the protected ledger tip")
    expected_workflow = {
        "RUN_CLAIMED": ONLINE_EXECUTION_WORKFLOW_PATH,
        "LABEL_RELEASE_CLAIMED": LABEL_RELEASE_WORKFLOW_PATH,
        "ANALYSIS_CLAIMED": ANALYSIS_WORKFLOW_PATH,
    }.get(snapshot.tip.state.state, WORKFLOW_PATH)
    expected_ref = f"{REPOSITORY}/{expected_workflow}@{C0_REF}"
    if workflow_ref != expected_ref:
        raise SuiteAttemptError("GitHub workflow_ref differs from the state-specific C0 identity")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    state_path = output_dir / f"{snapshot.tip.state.sequence:03d}.state.json"
    predicate_path = output_dir / "predicate.json"
    receipt_path = output_dir / "ledger-validation.json"
    _write_exclusive(state_path, snapshot.tip.state_bytes)
    control_root = output_dir / "ledger-controls"
    control_root.mkdir(mode=0o700)
    _write_exclusive(control_root / "inventory.json", snapshot.control_inventory_bytes)
    for control in snapshot.controls:
        relative = Path(control.ledger_path).relative_to(f"{LEDGER_CONTROL_PREFIX}/{attempt_id}")
        target = control_root / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_exclusive(target, control.encoded)
    _write_exclusive(
        predicate_path,
        _canonical_bytes(ledger_predicate(snapshot, snapshot.tip)) + b"\n",
    )
    _write_exclusive(
        receipt_path,
        _canonical_bytes(
            _workflow_receipt(
                snapshot,
                snapshot.tip,
                workflow_ref=workflow_ref,
                workflow_sha=workflow_sha,
            )
        )
        + b"\n",
    )
    return {
        "predicate_path": str(predicate_path),
        "receipt_path": str(receipt_path),
        "state_digest": snapshot.tip.state.record_sha256,
        "state_name": snapshot.tip.state_path,
        "state_path": str(state_path),
        "state_sequence": str(snapshot.tip.state.sequence),
        "suite_attempt_id": snapshot.tip.state.suite_attempt_id,
        "transition_id": snapshot.tip.commit_oid,
        "control_inventory_path": str(control_root / "inventory.json"),
        "control_inventory_sha256": hashlib.sha256(snapshot.control_inventory_bytes).hexdigest(),
    }


def emit_attestation_evidence(
    *,
    bundle_path: Path,
    receipt_path: Path,
    output_path: Path,
) -> SuiteAttestationEvidence:
    bundle = bundle_path.read_bytes()
    receipt = _object(
        _strict_json(receipt_path.read_bytes(), label="workflow ledger receipt"),
        label="workflow ledger receipt",
    )
    if receipt.get("schema_version") != WORKFLOW_RECEIPT_SCHEMA:
        raise SuiteAttemptError("workflow ledger receipt schema differs")
    observation = parse_sigstore_bundle(bundle)
    statement = observation.statement
    if statement.get("predicate") != receipt.get("ledger_predicate"):
        raise SuiteAttemptError("workflow receipt and attested ledger predicate differ")
    workflow_ref = _text(receipt.get("workflow_ref"), label="workflow_ref")
    state_name = _text(receipt.get("state_name"), label="state name")
    expected_workflow = {
        "RUN_CLAIMED": ONLINE_EXECUTION_WORKFLOW_PATH,
        "LABEL_RELEASE_CLAIMED": LABEL_RELEASE_WORKFLOW_PATH,
        "ANALYSIS_CLAIMED": ANALYSIS_WORKFLOW_PATH,
    }.get(state_name, WORKFLOW_PATH)
    expected_ref = f"{REPOSITORY}/{expected_workflow}@{C0_REF}"
    if workflow_ref != expected_ref:
        raise SuiteAttemptError("workflow receipt has another signer identity")
    evidence = SuiteAttestationEvidence(
        suite_attempt_id=_digest(receipt.get("suite_attempt_id"), label="suite attempt ID"),
        state_sequence=_integer(receipt.get("state_sequence"), label="state sequence"),
        state_name=state_name,  # type: ignore[arg-type]
        state_record_sha256=_digest(
            receipt.get("state_record_sha256"), label="state-record digest"
        ),
        descriptor_sha256=_digest(
            receipt.get("attestation_descriptor_sha256"), label="descriptor digest"
        ),
        bundle_sha256=hashlib.sha256(bundle).hexdigest(),
        bundle_byte_count=len(bundle),
        signer_identity=f"https://github.com/{workflow_ref}",
        oidc_issuer=OIDC_ISSUER,
        repository=REPOSITORY,
        workflow=expected_workflow,
        git_ref=C0_REF,
        signer_digest=_oid(receipt.get("signer_digest"), label="signer digest"),
        github_hosted_runner=True,
        transparency_log_identity=REKOR_IDENTITY,
        transparency_entry_id=observation.entry_id,
        transparency_log_index=observation.log_index,
        integrated_at_utc=observation.integrated_at_utc,
        timestamp_authority_identity=REKOR_IDENTITY,
        timestamp_token_sha256=observation.timestamp_token_sha256,
        signed_at_utc=observation.integrated_at_utc,
        state_service_identity=STATE_SERVICE_IDENTITY,
        state_key=_text(receipt.get("state_service_key"), label="state-service key"),
        transition_id=_oid(receipt.get("transition_id"), label="transition ID"),
        previous_transition_id=(
            None
            if receipt.get("previous_transition_id") is None
            else _oid(receipt.get("previous_transition_id"), label="previous transition ID")
        ),
    )
    _write_exclusive(output_path, evidence.canonical_bytes() + b"\n")
    return evidence


def _append_github_outputs(path: Path, outputs: Mapping[str, str]) -> None:
    for name, value in outputs.items():
        if not re.fullmatch(r"[a-z_]+", name) or "\n" in value or "\r" in value:
            raise SuiteAttemptError("workflow output is not safe for GITHUB_OUTPUT")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for name, value in sorted(outputs.items()):
            handle.write(f"{name}={value}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GitHub confirmatory state attestation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "install-ledger-ruleset",
        help="create or verify the fixed append-only ledger ruleset",
    )
    subparsers.add_parser(
        "install-freeze-tag-ruleset",
        help="create or verify the no-bypass C0/C1 freeze-tag ruleset",
    )
    publish = subparsers.add_parser(
        "publish-transition",
        help="publish the latest local suite state through the Git Database CAS",
    )
    publish.add_argument("--namespace", type=Path, required=True)
    publish.add_argument("--receipt", type=Path, required=True)
    validate = subparsers.add_parser("validate-workflow-transition")
    validate.add_argument("--ledger-commit", required=True)
    validate.add_argument("--repository", required=True)
    validate.add_argument("--github-ref", required=True)
    validate.add_argument("--github-sha", required=True)
    validate.add_argument("--workflow-ref", required=True)
    validate.add_argument("--workflow-sha", required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--github-output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-c1-registration")
    prepare.add_argument("--repository", required=True)
    prepare.add_argument("--github-ref", required=True)
    prepare.add_argument("--github-sha", required=True)
    prepare.add_argument("--workflow-ref", required=True)
    prepare.add_argument("--workflow-sha", required=True)
    prepare.add_argument("--repository-root", type=Path, required=True)
    prepare.add_argument("--c0-public-verification", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--github-output", type=Path, required=True)
    materialize = subparsers.add_parser("materialize-c1-registry-record")
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--lock", type=Path, required=True)
    materialize.add_argument("--reservation", type=Path, required=True)
    materialize.add_argument("--manifest-bundle", type=Path, required=True)
    materialize.add_argument("--manifest-predicate", type=Path, required=True)
    materialize.add_argument("--record-output", type=Path, required=True)
    materialize.add_argument("--registry-predicate-output", type=Path, required=True)
    materialize.add_argument("--receipt-output", type=Path, required=True)
    materialize.add_argument("--verification-output", type=Path, required=True)
    materialize.add_argument("--github-output", type=Path, required=True)
    verify_registry = subparsers.add_parser("verify-c1-registry-record")
    verify_registry.add_argument("--record", type=Path, required=True)
    verify_registry.add_argument("--predicate", type=Path, required=True)
    verify_registry.add_argument("--bundle", type=Path, required=True)
    verify_registry.add_argument("--receipt-output", type=Path, required=True)
    verify_registry.add_argument("--verification-output", type=Path, required=True)
    verify_registry.add_argument("--github-output", type=Path, required=True)
    emit = subparsers.add_parser("emit-evidence")
    emit.add_argument("--bundle", type=Path, required=True)
    emit.add_argument("--receipt", type=Path, required=True)
    emit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "install-ledger-ruleset":
        ruleset_id = install_ledger_ruleset(api=GhApiClient())
        print(
            json.dumps(
                {
                    "name": LEDGER_RULESET_NAME,
                    "repository": REPOSITORY,
                    "ruleset_id": ruleset_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    elif args.command == "install-freeze-tag-ruleset":
        ruleset_id = install_freeze_tag_ruleset(api=GhApiClient())
        print(
            json.dumps(
                {
                    "name": FREEZE_TAG_RULESET_NAME,
                    "repository": REPOSITORY,
                    "ruleset_id": ruleset_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    elif args.command == "publish-transition":
        receipt, created = publish_ledger_transition(
            namespace=args.namespace,
            receipt_path=args.receipt,
            api=GhApiClient(),
        )
        print(
            json.dumps(
                {
                    "commit_oid": receipt.commit_oid,
                    "created": created,
                    "receipt_sha256": receipt.receipt_sha256,
                    "state_sequence": receipt.state_sequence,
                    "suite_attempt_id": receipt.suite_attempt_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    elif args.command == "validate-workflow-transition":
        outputs = validate_workflow_transition(
            ledger_commit=args.ledger_commit,
            repository=args.repository,
            github_ref=args.github_ref,
            github_sha=args.github_sha,
            workflow_ref=args.workflow_ref,
            workflow_sha=args.workflow_sha,
            output_dir=args.output_dir,
            api=GhApiClient(),
        )
        _append_github_outputs(args.github_output, outputs)
    elif args.command == "prepare-c1-registration":
        outputs = prepare_c1_registration(
            repository=args.repository,
            github_ref=args.github_ref,
            github_sha=args.github_sha,
            workflow_ref=args.workflow_ref,
            workflow_sha=args.workflow_sha,
            repository_root=args.repository_root,
            c0_public_verification_path=args.c0_public_verification,
            output_dir=args.output_dir,
        )
        _append_github_outputs(args.github_output, outputs)
    elif args.command == "materialize-c1-registry-record":
        outputs = materialize_protocol_registry_record(
            manifest_path=args.manifest,
            lock_path=args.lock,
            reservation_path=args.reservation,
            manifest_bundle_path=args.manifest_bundle,
            manifest_predicate_path=args.manifest_predicate,
            record_output_path=args.record_output,
            registry_predicate_output_path=args.registry_predicate_output,
            receipt_output_path=args.receipt_output,
            verification_output_path=args.verification_output,
        )
        _append_github_outputs(args.github_output, outputs)
    elif args.command == "verify-c1-registry-record":
        outputs = verify_c1_registry_record_attestation(
            record_path=args.record,
            predicate_path=args.predicate,
            bundle_path=args.bundle,
            receipt_output_path=args.receipt_output,
            verification_output_path=args.verification_output,
        )
        _append_github_outputs(args.github_output, outputs)
    elif args.command == "emit-evidence":
        emit_attestation_evidence(
            bundle_path=args.bundle,
            receipt_path=args.receipt,
            output_path=args.output,
        )
    else:  # pragma: no cover
        raise SuiteAttemptError("unknown GitHub state-attestation command")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
