"""Portable contracts for the disconnected confirmatory analysis runtime.

This module deliberately imports no provider transport, GitHub, or subprocess
code.  The host mints an admission after live authority checks; the scientific
container treats the serialized admission as a closed, untrusted object and
revalidates every byte it can consume.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote_to_bytes, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactVerificationReceipt,
    read_secure_regular_file,
)
from .confirmatory_analysis import (
    ActionPanelAdmissionReceipt,
    ActionPanelArtifact,
    ConfirmatoryAnalysisError,
    ConfirmatoryInputArtifact,
)
from .label_separation import (
    OfflineEvaluationArtifact,
    PredictionCompletionReceipt,
    SealedLabelArtifact,
)
from .study import FIXED_CORPORA, SealedRunReceipt

OFFLINE_INPUT_BUNDLE_SCHEMA = "fractal-offline-confirmatory-input-bundle-v1"
OFFLINE_ANALYSIS_EVIDENCE_SCHEMA = "fractal-offline-analysis-evidence-v1"
OFFLINE_ANALYSIS_FILE_SCHEMA = "fractal-offline-analysis-file-v1"
OFFLINE_ANALYSIS_ADMISSION_SCHEMA = "fractal-offline-analysis-admission-v1"
OFFLINE_ANALYSIS_EXECUTION_SCHEMA = "fractal-offline-analysis-execution-v1"

PACKAGE_MOUNT_PATH = "/analysis-input"
RESULTS_MOUNT_PATH = "/analysis-output"
TMPFS_MOUNT_PATH = "/tmp"
RUNTIME_PLATFORM = "linux/amd64"
RUNTIME_MACHINE = "x86_64"
RUNTIME_UID = 65532
RUNTIME_GID = 65532
NETWORK_MODE = "none"
MAX_ADMISSION_BYTES = 16 * 1024 * 1024
MAX_EXECUTION_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_INPUT_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_PACKAGE_FILE_BYTES = 256 * 1024 * 1024

RUNTIME_ENVIRONMENT: tuple[tuple[str, str], ...] = (
    ("HOME", "/home/runner"),
    ("HOSTNAME", "fractal-analysis"),
    ("LANG", "C.UTF-8"),
    ("LC_ALL", "C.UTF-8"),
    ("LD_LIBRARY_PATH", "/opt/native-libs:/usr/local/lib"),
    ("LOGNAME", "runner"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("PATH", "/opt/venv/bin:/usr/local/bin"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
    ("PYTHONHASHSEED", "0"),
    ("PYTHONPATH", "/opt/app/src"),
    ("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt"),
    ("TMPDIR", "/tmp"),
    ("TZ", "UTC"),
    ("USER", "runner"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("XDG_CACHE_HOME", "/tmp/fractal-cache"),
)
RUNTIME_DYNAMIC_ENVIRONMENT_NAMES: tuple[str, ...] = ()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OCI_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_REFERENCE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_CONTAINER_NAME = re.compile(r"^fractal-analysis-[0-9a-f]{64}$")


class OfflineAnalysisContractError(ValueError):
    """Raised when portable offline-analysis evidence is not exact."""


def canonical_bytes(value: object) -> bytes:
    """Serialize finite JSON with one repository-wide byte convention."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise OfflineAnalysisContractError(
            "offline analysis evidence must be finite canonical JSON"
        ) from exc


def sha256_bytes(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise OfflineAnalysisContractError(f"{label} must be a JSON object")
    observed = set(value)
    if observed != fields:
        raise OfflineAnalysisContractError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def decode_canonical_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    """Decode one strict object and reject duplicate or non-finite values."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OfflineAnalysisContractError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise OfflineAnalysisContractError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfflineAnalysisContractError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise OfflineAnalysisContractError(f"{label} must contain one JSON object")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise OfflineAnalysisContractError(f"{name} must be a canonical non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise OfflineAnalysisContractError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise OfflineAnalysisContractError(f"{name} cannot contain control characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise OfflineAnalysisContractError(f"{name} must contain valid Unicode") from exc
    return value


def _sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise OfflineAnalysisContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _oci_sha256(name: str, value: object) -> str:
    if type(value) is not str or _OCI_SHA256.fullmatch(value) is None:
        raise OfflineAnalysisContractError(f"{name} must be an OCI SHA-256 digest")
    return value


def _git_commit(name: str, value: object) -> str:
    if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
        raise OfflineAnalysisContractError(f"{name} must be a lowercase Git commit")
    return value


def _positive(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise OfflineAnalysisContractError(f"{name} must be a positive integer")
    return value


def _nonnegative(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise OfflineAnalysisContractError(f"{name} must be a non-negative integer")
    return value


def _absolute_path(name: str, value: object) -> str:
    text = _text(name, value)
    if "\\" in text or unicodedata.normalize("NFC", text) != text:
        raise OfflineAnalysisContractError(f"{name} is not a canonical POSIX path")
    path = PurePosixPath(text)
    if (
        not path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise OfflineAnalysisContractError(f"{name} must be a canonical absolute POSIX path")
    return text


def _relative_path(name: str, value: object) -> str:
    text = _text(name, value)
    if "\\" in text or unicodedata.normalize("NFC", text) != text:
        raise OfflineAnalysisContractError(f"{name} is not a canonical POSIX path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise OfflineAnalysisContractError(f"{name} must be a canonical relative POSIX path")
    if len(path.parts) != 1:
        raise OfflineAnalysisContractError(f"{name} must be one package-root filename")
    return text


def canonical_file_uri_path(value: object, *, label: str) -> Path:
    """Decode one authority-free canonical local file URI."""

    uri = _text(label, value)
    try:
        parsed = urlsplit(uri)
    except ValueError as exc:
        raise OfflineAnalysisContractError(f"{label} is not a valid URI") from exc
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise OfflineAnalysisContractError(f"{label} must be a canonical local file URI")
    try:
        decoded = unquote_to_bytes(parsed.path).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise OfflineAnalysisContractError(f"{label} path must be valid UTF-8") from exc
    if (
        not decoded.startswith("/")
        or "\\" in decoded
        or unicodedata.normalize("NFC", decoded) != decoded
        or any(part in {"", ".", ".."} for part in decoded.split("/")[1:])
    ):
        raise OfflineAnalysisContractError(f"{label} path is not canonical")
    path = Path(decoded)
    if not path.is_absolute() or path.anchor != "/" or path.as_uri() != uri:
        raise OfflineAnalysisContractError(f"{label} must use canonical file URI encoding")
    return path


@dataclass(frozen=True)
class OfflineConfirmatoryInputBundle:
    """A complete, deserializable snapshot of the typed analysis input."""

    run_receipt: SealedRunReceipt
    frozen_manifest: Mapping[str, Any]
    artifact_verification_receipt: ArtifactVerificationReceipt
    completion_receipts: tuple[PredictionCompletionReceipt, ...]
    offline_evaluations: tuple[OfflineEvaluationArtifact, ...]
    sealed_label_artifacts: tuple[SealedLabelArtifact, ...]
    action_panels: tuple[ActionPanelArtifact, ...]
    action_panel_admission_receipts: tuple[ActionPanelAdmissionReceipt, ...]
    schema_version: str = OFFLINE_INPUT_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != OFFLINE_INPUT_BUNDLE_SCHEMA:
            raise OfflineAnalysisContractError("offline input bundle schema differs")
        try:
            inputs = self.to_confirmatory_input()
        except (ConfirmatoryAnalysisError, ArtifactIntegrityError, ValueError) as exc:
            raise OfflineAnalysisContractError(
                f"offline input bundle is not a valid confirmatory input: {exc}"
            ) from exc
        object.__setattr__(self, "frozen_manifest", inputs.frozen_manifest)
        object.__setattr__(self, "completion_receipts", inputs.completion_receipts)
        object.__setattr__(self, "offline_evaluations", inputs.offline_evaluations)
        object.__setattr__(self, "sealed_label_artifacts", inputs.sealed_label_artifacts)
        object.__setattr__(self, "action_panels", inputs.action_panels)
        object.__setattr__(
            self,
            "action_panel_admission_receipts",
            inputs.action_panel_admission_receipts,
        )

    @classmethod
    def from_confirmatory_input(
        cls,
        inputs: ConfirmatoryInputArtifact,
    ) -> OfflineConfirmatoryInputBundle:
        if not isinstance(inputs, ConfirmatoryInputArtifact):
            raise OfflineAnalysisContractError(
                "offline input bundle requires a ConfirmatoryInputArtifact"
            )
        return cls(
            run_receipt=inputs.run_receipt,
            frozen_manifest=inputs.frozen_manifest,
            artifact_verification_receipt=inputs.artifact_verification_receipt,
            completion_receipts=inputs.completion_receipts,
            offline_evaluations=inputs.offline_evaluations,
            sealed_label_artifacts=inputs.sealed_label_artifacts,
            action_panels=inputs.action_panels,
            action_panel_admission_receipts=inputs.action_panel_admission_receipts,
        )

    def to_confirmatory_input(self) -> ConfirmatoryInputArtifact:
        return ConfirmatoryInputArtifact(
            run_receipt=self.run_receipt,
            frozen_manifest=self.frozen_manifest,
            artifact_verification_receipt=self.artifact_verification_receipt,
            completion_receipts=self.completion_receipts,
            offline_evaluations=self.offline_evaluations,
            sealed_label_artifacts=self.sealed_label_artifacts,
            action_panels=self.action_panels,
            action_panel_admission_receipts=self.action_panel_admission_receipts,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action_panel_admission_receipts": [
                row.to_dict() for row in self.action_panel_admission_receipts
            ],
            "action_panels": [row.to_dict() for row in self.action_panels],
            "artifact_verification_receipt": self.artifact_verification_receipt.to_dict(),
            "completion_receipts": [row.to_dict() for row in self.completion_receipts],
            "frozen_manifest": self.frozen_manifest,
            "offline_evaluations": [row.to_dict() for row in self.offline_evaluations],
            "run_receipt": self.run_receipt.to_dict(),
            "schema_version": self.schema_version,
            "sealed_label_artifacts": [row.to_dict() for row in self.sealed_label_artifacts],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def bundle_sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> OfflineConfirmatoryInputBundle:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="offline input bundle")

        def sequence(name: str) -> Sequence[object]:
            value = row[name]
            if not isinstance(value, list):
                raise OfflineAnalysisContractError(f"{name} must be a JSON array")
            return value

        manifest = row["frozen_manifest"]
        if not isinstance(manifest, Mapping):
            raise OfflineAnalysisContractError("frozen_manifest must be a JSON object")
        return cls(
            run_receipt=SealedRunReceipt.from_dict(row["run_receipt"]),
            frozen_manifest=manifest,
            artifact_verification_receipt=ArtifactVerificationReceipt.from_dict(
                row["artifact_verification_receipt"]
            ),
            completion_receipts=tuple(
                PredictionCompletionReceipt.from_dict(item)
                for item in sequence("completion_receipts")
            ),
            offline_evaluations=tuple(
                OfflineEvaluationArtifact.from_dict(item)
                for item in sequence("offline_evaluations")
            ),
            sealed_label_artifacts=tuple(
                SealedLabelArtifact.from_dict(item) for item in sequence("sealed_label_artifacts")
            ),
            action_panels=tuple(
                ActionPanelArtifact.from_dict(item) for item in sequence("action_panels")
            ),
            action_panel_admission_receipts=tuple(
                ActionPanelAdmissionReceipt.from_dict(item)
                for item in sequence("action_panel_admission_receipts")
            ),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class OfflineAnalysisEvidenceBinding:
    """One source file already admitted by input materialization."""

    role: str
    corpus_id: str | None
    source_uri: str
    semantic_sha256: str
    file_sha256: str
    byte_count: int
    schema_version: str = OFFLINE_ANALYSIS_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        _text("evidence role", self.role)
        if self.corpus_id is not None and self.corpus_id not in FIXED_CORPORA:
            raise OfflineAnalysisContractError("evidence binding names an unknown corpus")
        canonical_file_uri_path(self.source_uri, label="evidence source_uri")
        _sha256("evidence semantic_sha256", self.semantic_sha256)
        _sha256("evidence file_sha256", self.file_sha256)
        _positive("evidence byte_count", self.byte_count)
        if self.schema_version != OFFLINE_ANALYSIS_EVIDENCE_SCHEMA:
            raise OfflineAnalysisContractError("offline evidence binding schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> OfflineAnalysisEvidenceBinding:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="offline evidence binding",
            )
        )


@dataclass(frozen=True)
class OfflineAnalysisFileBinding:
    """One regular package file the runtime may open."""

    role: str
    relative_path: str
    semantic_sha256: str
    file_sha256: str
    byte_count: int
    schema_version: str = OFFLINE_ANALYSIS_FILE_SCHEMA

    def __post_init__(self) -> None:
        _text("package file role", self.role)
        _relative_path("package file relative_path", self.relative_path)
        _sha256("package file semantic_sha256", self.semantic_sha256)
        _sha256("package file file_sha256", self.file_sha256)
        _positive("package file byte_count", self.byte_count)
        if self.schema_version != OFFLINE_ANALYSIS_FILE_SCHEMA:
            raise OfflineAnalysisContractError("offline package file schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> OfflineAnalysisFileBinding:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="offline package file binding",
            )
        )


@dataclass(frozen=True)
class OfflineAnalysisAdmission:
    """Provider-minted, portable authority boundary for one disconnected run."""

    suite_attempt_id: str
    provider_state_record_sha256: str
    provider_ledger_commit: str
    provider_control_inventory_sha256: str
    provider_artifact_receipt_sha256: str
    phase_claim_contract_sha256: str
    phase_claim_state_sha256: str
    phase_claim_ledger_commit: str
    provider_identity_sha256: str
    live_execute_job_receipt_sha256: str
    claim_attested_at_utc: str
    c1_commit: str
    c1_provider_plan_uri: str
    c1_provider_plan_sha256: str
    c1_provider_plan_file_sha256: str
    runtime_image: str
    runtime_platform: str
    runtime_image_role: str
    runtime_index_role: str
    oci_index_digest: str
    oci_platform_manifest_digest: str
    host_tool_contract_sha256: str
    runtime_probe_receipt_sha256: str
    manifest_sha256: str
    run_receipt_sha256: str
    confirmatory_input_artifact_sha256: str
    confirmatory_input_artifact_file_sha256: str
    confirmatory_input_artifact_byte_count: int
    confirmatory_input_receipt_sha256: str
    confirmatory_input_receipt_file_sha256: str
    confirmatory_input_receipt_byte_count: int
    offline_input_bundle_sha256: str
    model_suite_sha256: str
    registered_results_store_uri: str
    host_results_store_path: str
    package_mount_path: str
    results_mount_path: str
    tmpfs_mount_path: str
    network_mode: str
    root_filesystem_read_only: bool
    package_mount_read_only: bool
    results_mount_read_write: bool
    runtime_machine: str
    runtime_uid: int
    runtime_gid: int
    runtime_environment: tuple[tuple[str, str], ...]
    runtime_dynamic_environment_names: tuple[str, ...]
    container_name: str
    registered_attempt_uri: str
    registered_result_receipt_uri: str
    registered_result_uri: str
    container_attempt_path: str
    container_result_receipt_path: str
    container_result_path: str
    expected_attempt_receipt_sha256: str
    evidence: tuple[OfflineAnalysisEvidenceBinding, ...]
    package_files: tuple[OfflineAnalysisFileBinding, ...]
    schema_version: str = OFFLINE_ANALYSIS_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "suite_attempt_id",
            "provider_state_record_sha256",
            "provider_control_inventory_sha256",
            "provider_artifact_receipt_sha256",
            "phase_claim_contract_sha256",
            "phase_claim_state_sha256",
            "provider_identity_sha256",
            "live_execute_job_receipt_sha256",
            "c1_provider_plan_sha256",
            "c1_provider_plan_file_sha256",
            "host_tool_contract_sha256",
            "runtime_probe_receipt_sha256",
            "manifest_sha256",
            "run_receipt_sha256",
            "confirmatory_input_artifact_sha256",
            "confirmatory_input_artifact_file_sha256",
            "confirmatory_input_receipt_sha256",
            "confirmatory_input_receipt_file_sha256",
            "offline_input_bundle_sha256",
            "model_suite_sha256",
            "expected_attempt_receipt_sha256",
        ):
            _sha256(name, getattr(self, name))
        for name in ("provider_ledger_commit", "phase_claim_ledger_commit", "c1_commit"):
            _git_commit(name, getattr(self, name))
        _text("claim_attested_at_utc", self.claim_attested_at_utc)
        plan_path = canonical_file_uri_path(
            self.c1_provider_plan_uri,
            label="c1_provider_plan_uri",
        )
        if (
            type(self.runtime_image) is not str
            or _IMAGE_REFERENCE.fullmatch(self.runtime_image) is None
        ):
            raise OfflineAnalysisContractError("runtime_image must be digest-pinned")
        if (
            self.runtime_platform != RUNTIME_PLATFORM
            or self.runtime_image_role != "scientific"
            or self.runtime_index_role != "main"
        ):
            raise OfflineAnalysisContractError("offline runtime image binding differs")
        _oci_sha256("oci_index_digest", self.oci_index_digest)
        _oci_sha256("oci_platform_manifest_digest", self.oci_platform_manifest_digest)
        if self.runtime_image.rsplit("@", 1)[1] != self.oci_index_digest:
            raise OfflineAnalysisContractError("runtime image and OCI index digest differ")
        if plan_path.name == "":
            raise OfflineAnalysisContractError("C1 provider plan URI has no filename")
        _positive(
            "confirmatory_input_artifact_byte_count",
            self.confirmatory_input_artifact_byte_count,
        )
        _positive(
            "confirmatory_input_receipt_byte_count",
            self.confirmatory_input_receipt_byte_count,
        )
        results_path = canonical_file_uri_path(
            self.registered_results_store_uri,
            label="registered_results_store_uri",
        )
        if str(results_path) != _absolute_path(
            "host_results_store_path",
            self.host_results_store_path,
        ):
            raise OfflineAnalysisContractError(
                "host results-store path differs from its registered URI"
            )
        if (
            self.package_mount_path != PACKAGE_MOUNT_PATH
            or self.results_mount_path != RESULTS_MOUNT_PATH
            or self.tmpfs_mount_path != TMPFS_MOUNT_PATH
            or self.network_mode != NETWORK_MODE
            or self.root_filesystem_read_only is not True
            or self.package_mount_read_only is not True
            or self.results_mount_read_write is not True
            or self.runtime_machine != RUNTIME_MACHINE
            or self.runtime_uid != RUNTIME_UID
            or self.runtime_gid != RUNTIME_GID
        ):
            raise OfflineAnalysisContractError("offline runtime isolation contract differs")
        environment = tuple(self.runtime_environment)
        if environment != RUNTIME_ENVIRONMENT:
            raise OfflineAnalysisContractError("offline runtime environment differs")
        if self.runtime_dynamic_environment_names != RUNTIME_DYNAMIC_ENVIRONMENT_NAMES:
            raise OfflineAnalysisContractError("offline dynamic environment allowlist differs")
        if (
            _CONTAINER_NAME.fullmatch(self.container_name) is None
            or self.container_name != f"fractal-analysis-{self.suite_attempt_id}"
        ):
            raise OfflineAnalysisContractError(
                "offline container name must be suite-attempt-derived"
            )
        filenames = {
            "attempt": f"{self.manifest_sha256}.confirmatory-analysis-attempt.json",
            "result_receipt": (f"{self.manifest_sha256}.confirmatory-result-receipt.json"),
            "result": f"{self.manifest_sha256}.confirmatory-result.json",
        }
        registered = {
            "attempt": self.registered_attempt_uri,
            "result_receipt": self.registered_result_receipt_uri,
            "result": self.registered_result_uri,
        }
        container = {
            "attempt": self.container_attempt_path,
            "result_receipt": self.container_result_receipt_path,
            "result": self.container_result_path,
        }
        for role, filename in filenames.items():
            registered_path = canonical_file_uri_path(
                registered[role],
                label=f"registered_{role}_uri",
            )
            if registered_path != results_path / filename:
                raise OfflineAnalysisContractError(
                    f"registered {role} path is not results-store-derived"
                )
            expected_container = f"{RESULTS_MOUNT_PATH}/{filename}"
            if _absolute_path(f"container_{role}_path", container[role]) != expected_container:
                raise OfflineAnalysisContractError(f"container {role} path is not mount-derived")
        evidence = tuple(self.evidence)
        if not evidence or not all(
            isinstance(row, OfflineAnalysisEvidenceBinding) for row in evidence
        ):
            raise OfflineAnalysisContractError("offline admission lacks typed evidence")
        canonical_evidence = tuple(
            sorted(
                evidence,
                key=lambda row: (
                    (row.corpus_id or "").encode("utf-8"),
                    row.role.encode("utf-8"),
                    row.source_uri.encode("utf-8"),
                ),
            )
        )
        if evidence != canonical_evidence or len({row.source_uri for row in evidence}) != len(
            evidence
        ):
            raise OfflineAnalysisContractError(
                "offline admission evidence is not a unique canonical sequence"
            )
        package_files = tuple(self.package_files)
        if not package_files or not all(
            isinstance(row, OfflineAnalysisFileBinding) for row in package_files
        ):
            raise OfflineAnalysisContractError("offline admission lacks package files")
        canonical_files = tuple(
            sorted(package_files, key=lambda row: row.relative_path.encode("utf-8"))
        )
        required_roles = {
            "confirmatory-input",
            "confirmatory-input-receipt",
            "h1-predictive-model",
            "h2-model-suite",
            "offline-input-bundle",
        }
        if (
            package_files != canonical_files
            or len({row.relative_path for row in package_files}) != len(package_files)
            or len(package_files) != len(required_roles)
            or len({row.role for row in package_files}) != len(package_files)
            or {row.role for row in package_files} != required_roles
        ):
            raise OfflineAnalysisContractError(
                "offline package membership is not the exact canonical closure"
            )
        by_role = {row.role: row for row in package_files}
        expected_package_semantics = {
            "confirmatory-input": self.confirmatory_input_artifact_sha256,
            "confirmatory-input-receipt": self.confirmatory_input_receipt_sha256,
            "h1-predictive-model": by_role["h1-predictive-model"].semantic_sha256,
            "h2-model-suite": self.model_suite_sha256,
            "offline-input-bundle": self.offline_input_bundle_sha256,
        }
        for role, expected in expected_package_semantics.items():
            if by_role[role].semantic_sha256 != expected:
                raise OfflineAnalysisContractError(
                    f"package {role} semantic digest differs from admission"
                )
        if (
            by_role["confirmatory-input"].file_sha256
            != self.confirmatory_input_artifact_file_sha256
            or by_role["confirmatory-input"].byte_count
            != self.confirmatory_input_artifact_byte_count
            or by_role["confirmatory-input-receipt"].file_sha256
            != self.confirmatory_input_receipt_file_sha256
            or by_role["confirmatory-input-receipt"].byte_count
            != self.confirmatory_input_receipt_byte_count
        ):
            raise OfflineAnalysisContractError(
                "materialized input package bytes differ from admission"
            )
        if self.schema_version != OFFLINE_ANALYSIS_ADMISSION_SCHEMA:
            raise OfflineAnalysisContractError("offline analysis admission schema differs")
        object.__setattr__(self, "evidence", canonical_evidence)
        object.__setattr__(self, "package_files", canonical_files)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "evidence",
                    "package_files",
                    "runtime_dynamic_environment_names",
                    "runtime_environment",
                }
            },
            "evidence": [row.to_dict() for row in self.evidence],
            "package_files": [row.to_dict() for row in self.package_files],
            "runtime_dynamic_environment_names": list(self.runtime_dynamic_environment_names),
            "runtime_environment": [
                {"name": name, "value": value} for name, value in self.runtime_environment
            ],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def admission_sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes())

    @property
    def admission_filename(self) -> str:
        return f"{self.manifest_sha256}.offline-analysis-admission.json"

    @classmethod
    def from_dict(cls, value: object) -> OfflineAnalysisAdmission:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="offline analysis admission",
        )
        evidence = row["evidence"]
        package_files = row["package_files"]
        environment = row["runtime_environment"]
        dynamic = row["runtime_dynamic_environment_names"]
        if not isinstance(evidence, list) or not isinstance(package_files, list):
            raise OfflineAnalysisContractError(
                "offline admission evidence and package_files must be arrays"
            )
        if not isinstance(environment, list):
            raise OfflineAnalysisContractError("runtime_environment must be an array")
        env_rows: list[tuple[str, str]] = []
        for item in environment:
            item_row = _closed(
                item,
                frozenset({"name", "value"}),
                label="runtime environment row",
            )
            env_rows.append(
                (
                    _text("runtime environment name", item_row["name"]),
                    _text("runtime environment value", item_row["value"]),
                )
            )
        if not isinstance(dynamic, list) or not all(type(item) is str for item in dynamic):
            raise OfflineAnalysisContractError(
                "runtime_dynamic_environment_names must be a string array"
            )
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "evidence",
                    "package_files",
                    "runtime_dynamic_environment_names",
                    "runtime_environment",
                }
            },
            evidence=tuple(OfflineAnalysisEvidenceBinding.from_dict(item) for item in evidence),
            package_files=tuple(
                OfflineAnalysisFileBinding.from_dict(item) for item in package_files
            ),
            runtime_environment=tuple(env_rows),
            runtime_dynamic_environment_names=tuple(dynamic),
        )


@dataclass(frozen=True)
class OfflineAnalysisExecutionReceipt:
    """Retained host evidence for the exact disconnected container execution."""

    suite_attempt_id: str
    manifest_sha256: str
    run_receipt_sha256: str
    provider_state_record_sha256: str
    provider_ledger_commit: str
    phase_claim_contract_sha256: str
    phase_claim_state_sha256: str
    phase_claim_ledger_commit: str
    provider_identity_sha256: str
    c1_commit: str
    admission_uri: str
    admission_sha256: str
    admission_file_sha256: str
    package_root_uri: str
    package_tree_before_sha256: str
    package_tree_after_sha256: str
    package_entries: tuple[str, ...]
    docker_executable_sha256: str
    docker_pull_argv_sha256: str
    docker_create_argv_sha256: str
    docker_start_argv_sha256: str
    docker_remove_argv_sha256: str
    container_name: str
    runtime_image: str
    runtime_platform: str
    oci_index_digest: str
    oci_platform_manifest_digest: str
    attempt_uri: str
    attempt_receipt_sha256: str
    attempt_file_sha256: str
    result_receipt_uri: str
    result_receipt_sha256: str
    result_receipt_file_sha256: str
    result_uri: str
    result_artifact_sha256: str
    result_file_sha256: str
    results_tree_sha256: str
    results_entries: tuple[str, ...]
    completion_state_record_sha256: str
    completion_ledger_commit: str
    container_absent_after_execution: bool
    schema_version: str = OFFLINE_ANALYSIS_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "suite_attempt_id",
            "manifest_sha256",
            "run_receipt_sha256",
            "provider_state_record_sha256",
            "phase_claim_contract_sha256",
            "phase_claim_state_sha256",
            "provider_identity_sha256",
            "admission_sha256",
            "admission_file_sha256",
            "package_tree_before_sha256",
            "package_tree_after_sha256",
            "docker_executable_sha256",
            "docker_pull_argv_sha256",
            "docker_create_argv_sha256",
            "docker_start_argv_sha256",
            "docker_remove_argv_sha256",
            "attempt_receipt_sha256",
            "attempt_file_sha256",
            "result_receipt_sha256",
            "result_receipt_file_sha256",
            "result_artifact_sha256",
            "result_file_sha256",
            "results_tree_sha256",
            "completion_state_record_sha256",
        ):
            _sha256(name, getattr(self, name))
        for name in (
            "provider_ledger_commit",
            "phase_claim_ledger_commit",
            "c1_commit",
            "completion_ledger_commit",
        ):
            _git_commit(name, getattr(self, name))
        if (
            self.provider_state_record_sha256 != self.phase_claim_state_sha256
            or self.completion_state_record_sha256 != self.provider_state_record_sha256
            or self.completion_ledger_commit != self.provider_ledger_commit
            or self.package_tree_before_sha256 != self.package_tree_after_sha256
        ):
            raise OfflineAnalysisContractError(
                "offline execution authority or package changed across execution"
            )
        admission_path = canonical_file_uri_path(
            self.admission_uri,
            label="offline execution admission_uri",
        )
        package_root = canonical_file_uri_path(
            self.package_root_uri,
            label="offline execution package_root_uri",
        )
        if (
            admission_path.parent != package_root
            or admission_path.name != f"{self.manifest_sha256}.offline-analysis-admission.json"
        ):
            raise OfflineAnalysisContractError("offline execution admission is outside its package")
        if (
            type(self.runtime_image) is not str
            or _IMAGE_REFERENCE.fullmatch(self.runtime_image) is None
            or self.runtime_image.rsplit("@", 1)[1] != self.oci_index_digest
            or self.runtime_platform != RUNTIME_PLATFORM
        ):
            raise OfflineAnalysisContractError("offline execution runtime image binding differs")
        _oci_sha256("oci_index_digest", self.oci_index_digest)
        _oci_sha256(
            "oci_platform_manifest_digest",
            self.oci_platform_manifest_digest,
        )
        if (
            _CONTAINER_NAME.fullmatch(self.container_name) is None
            or self.container_name != f"fractal-analysis-{self.suite_attempt_id}"
        ):
            raise OfflineAnalysisContractError("offline execution container name differs")
        package_entries = tuple(self.package_entries)
        expected_admission = admission_path.name
        if (
            len(package_entries) != 6
            or not all(
                type(value) is str and value and PurePosixPath(value).name == value
                for value in package_entries
            )
            or len(set(package_entries)) != len(package_entries)
            or package_entries
            != tuple(sorted(package_entries, key=lambda value: value.encode("utf-8")))
            or expected_admission not in package_entries
        ):
            raise OfflineAnalysisContractError(
                "offline execution package entries are not the six-file closure"
            )
        attempt_path = canonical_file_uri_path(
            self.attempt_uri,
            label="offline execution attempt_uri",
        )
        receipt_path = canonical_file_uri_path(
            self.result_receipt_uri,
            label="offline execution result_receipt_uri",
        )
        result_path = canonical_file_uri_path(
            self.result_uri,
            label="offline execution result_uri",
        )
        expected_outputs = {
            "attempt": f"{self.manifest_sha256}.confirmatory-analysis-attempt.json",
            "result_receipt": (f"{self.manifest_sha256}.confirmatory-result-receipt.json"),
            "result": f"{self.manifest_sha256}.confirmatory-result.json",
        }
        if (
            attempt_path.parent != result_path.parent
            or receipt_path.parent != result_path.parent
            or attempt_path.name != expected_outputs["attempt"]
            or receipt_path.name != expected_outputs["result_receipt"]
            or result_path.name != expected_outputs["result"]
        ):
            raise OfflineAnalysisContractError(
                "offline execution outputs are outside the registered closure"
            )
        results_entries = tuple(self.results_entries)
        if (
            len(results_entries) != 5
            or not all(
                type(value) is str and value and PurePosixPath(value).name == value
                for value in results_entries
            )
            or len(set(results_entries)) != len(results_entries)
            or results_entries
            != tuple(sorted(results_entries, key=lambda value: value.encode("utf-8")))
            or not set(expected_outputs.values()).issubset(results_entries)
        ):
            raise OfflineAnalysisContractError(
                "offline execution results are not the five-file closure"
            )
        if self.container_absent_after_execution is not True:
            raise OfflineAnalysisContractError("offline execution did not prove container absence")
        if self.schema_version != OFFLINE_ANALYSIS_EXECUTION_SCHEMA:
            raise OfflineAnalysisContractError("offline analysis execution schema differs")
        object.__setattr__(self, "package_entries", package_entries)
        object.__setattr__(self, "results_entries", results_entries)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"package_entries", "results_entries"}
            },
            "package_entries": list(self.package_entries),
            "results_entries": list(self.results_entries),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> OfflineAnalysisExecutionReceipt:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="offline analysis execution receipt",
        )
        package_entries = row["package_entries"]
        results_entries = row["results_entries"]
        if not isinstance(package_entries, list) or not isinstance(results_entries, list):
            raise OfflineAnalysisContractError(
                "offline execution receipt inventories must be arrays"
            )
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key not in {"package_entries", "results_entries"}
            },
            package_entries=tuple(package_entries),
            results_entries=tuple(results_entries),
        )


def _load_canonical(
    path: str | Path,
    *,
    max_bytes: int,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    target = Path(path)
    try:
        encoded = read_secure_regular_file(target, max_bytes=max_bytes, label=label)
    except ArtifactIntegrityError as exc:
        raise OfflineAnalysisContractError(f"cannot read {label}: {exc}") from exc
    payload = decode_canonical_object(encoded, label=label)
    return payload, encoded


def load_offline_input_bundle(path: str | Path) -> OfflineConfirmatoryInputBundle:
    payload, encoded = _load_canonical(
        path,
        max_bytes=MAX_INPUT_BUNDLE_BYTES,
        label="offline confirmatory input bundle",
    )
    bundle = OfflineConfirmatoryInputBundle.from_dict(payload)
    if encoded != bundle.canonical_bytes() + b"\n":
        raise OfflineAnalysisContractError("offline input bundle bytes are not canonical")
    return bundle


def load_offline_analysis_admission(path: str | Path) -> OfflineAnalysisAdmission:
    payload, encoded = _load_canonical(
        path,
        max_bytes=MAX_ADMISSION_BYTES,
        label="offline analysis admission",
    )
    admission = OfflineAnalysisAdmission.from_dict(payload)
    target = Path(path)
    if (
        not target.is_absolute()
        or target.name != admission.admission_filename
        or target.parent != Path(PACKAGE_MOUNT_PATH)
    ):
        raise OfflineAnalysisContractError(
            "offline admission is not at its fixed container package path"
        )
    if encoded != admission.canonical_bytes() + b"\n":
        raise OfflineAnalysisContractError("offline analysis admission bytes are not canonical")
    return admission


def load_offline_analysis_execution_receipt(
    path: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
    expected_file_sha256: str | None = None,
) -> OfflineAnalysisExecutionReceipt:
    """Load one bounded canonical host receipt and verify its retained digests."""

    payload, encoded = _load_canonical(
        path,
        max_bytes=MAX_EXECUTION_RECEIPT_BYTES,
        label="offline analysis execution receipt",
    )
    receipt = OfflineAnalysisExecutionReceipt.from_dict(payload)
    target = Path(path)
    package_root = canonical_file_uri_path(
        receipt.package_root_uri,
        label="offline execution package_root_uri",
    )
    if (
        not target.is_absolute()
        or target.parent != package_root.parent
        or target.name != f"{receipt.manifest_sha256}.offline-analysis-execution-receipt.json"
    ):
        raise OfflineAnalysisContractError(
            "offline execution receipt is outside its admitted host namespace"
        )
    expected_bytes = receipt.canonical_bytes() + b"\n"
    if encoded != expected_bytes:
        raise OfflineAnalysisContractError(
            "offline analysis execution receipt bytes are not canonical"
        )
    if expected_receipt_sha256 is not None and (
        _sha256("expected_receipt_sha256", expected_receipt_sha256) != receipt.receipt_sha256
    ):
        raise OfflineAnalysisContractError("offline execution receipt semantic digest differs")
    observed_file_sha256 = sha256_bytes(encoded)
    if expected_file_sha256 is not None and (
        _sha256("expected_file_sha256", expected_file_sha256) != observed_file_sha256
    ):
        raise OfflineAnalysisContractError("offline execution receipt file digest differs")
    return receipt
