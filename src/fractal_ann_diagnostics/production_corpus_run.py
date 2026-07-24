"""Closed command boundary for the single confirmatory corpus execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .artifact_stage_bundles import IndexStageBundleReceipt, PolicyStageBundleReceipt
from .authorized_index_store import load_authorized_index_store_receipt
from .custody import OnlineCustodyAdmissionReceipt
from .embedding_store import load_embedding_store_receipt
from .execution_claim import RuntimeClaimReceipt
from .policy_intervention import RECEIPT_FILENAME as POLICY_RECEIPT_FILENAME
from .policy_intervention import SCHEDULE_FILENAME as POLICY_SCHEDULE_FILENAME
from .policy_intervention import load_policy_intervention_receipt
from .production_workload_registration import (
    PRODUCTION_WORKLOAD_SPEC_FIELDS,
    PRODUCTION_WORKLOAD_SPEC_SCHEMA,
)
from .runtime_attestation import (
    LinuxRuntimeProbe,
    RuntimeArtifactMount,
    RuntimeAttestationPlan,
    RuntimeAttestationReceipt,
    attest_runtime_once,
    loads_runtime_attestation_plan,
    loads_runtime_attestation_receipt,
    verify_runtime_attestation_receipt,
)
from .scalable_execution import (
    ONLINE_EXECUTION_PLAN_FILENAME,
    ShardedOnlineExecutionPlan,
    loads_sharded_online_execution_plan,
)
from .scalable_partition_audit import load_scalable_partition_audit
from .sealed_online_execution import PersistedSealedOnlineRun, run_sealed_online_once
from .sealed_orchestrator import RequiredArtifactIdBindings
from .study import C0_COMMIT_SENTINEL, FIXED_CORPORA, SealedRunReceipt
from .trial_runtime import (
    QUERY_TRIAL_RECEIPT_FILENAME,
    QueryTrialStoreReceipt,
    RuntimeFeatureBinding,
    TrialRuntimeAdmission,
    TrialRuntimeAdmissionReceipt,
    reconstruct_trial_runtime_admission,
)

PRODUCTION_CORPUS_WORKLOAD_SPEC_SCHEMA = PRODUCTION_WORKLOAD_SPEC_SCHEMA
PRODUCTION_CORPUS_RUN_CONFIG_SCHEMA = "fractal-production-corpus-run-config-v2"
PRODUCTION_CORPUS_COMMAND_ATTEMPT_SCHEMA = "fractal-production-corpus-command-attempt-v2"
PRODUCTION_CORPUS_WORKLOAD_ID = "sealed-online-corpus-workload-spec-v1"
PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME = "production-corpus-workload-spec.json"
PRODUCTION_CORPUS_CONFIG_FILENAME = "corpus-run-config.json"
ONLINE_CUSTODY_ADMISSION_FILENAME = "online-custody-admission.json"
REQUIRED_ARTIFACT_BINDINGS_FILENAME = "required-artifact-bindings.json"
SEALED_RUN_RECEIPT_FILENAME = "sealed-run-receipt.json"
SHARDED_EXECUTION_PLAN_FILENAME = "sharded-online-execution-plan.json"
TRIAL_RUNTIME_RECEIPT_FILENAME = "trial-runtime-admission-receipt.json"
RUNTIME_ATTESTATION_PLAN_FILENAME = "runtime-attestation-plan.json"
RUNTIME_ATTESTATION_RECEIPT_FILENAME = "runtime-attestation-receipt.json"
RUNTIME_INVOCATION_MARKER_FILENAME = "runtime-invocation-marker.json"
PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME = "sealed-corpus-command-attempt.json"

_PYTHON_BINARY = "/opt/venv/bin/python"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OCI_IMAGE = re.compile(r"^[^\s/@]+(?:/[^\s/@]+)+@sha256:[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_WORKLOAD_SPEC_FIELDS = PRODUCTION_WORKLOAD_SPEC_FIELDS
_CONFIG_FIELDS = frozenset(
    {
        "control_root",
        "online_custody_admission_file_sha256",
        "output_root",
        "required_artifact_bindings_file_sha256",
        "runtime_attestation_plan_path",
        "schema_version",
        "sealed_run_receipt_path",
        "sealed_run_receipt_file_sha256",
        "workload_spec_file_sha256",
    }
)
_COMMAND_ATTEMPT_FIELDS = frozenset(
    {
        "config_file_sha256",
        "manifest_sha256",
        "runtime_attestation_plan_sha256",
        "runtime_attestation_receipt_sha256",
        "schema_version",
        "workload_id",
        "workload_spec_file_sha256",
    }
)
_CONTROL_FILENAMES = frozenset(
    {
        PRODUCTION_CORPUS_CONFIG_FILENAME,
        PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME,
        ONLINE_CUSTODY_ADMISSION_FILENAME,
        REQUIRED_ARTIFACT_BINDINGS_FILENAME,
        SHARDED_EXECUTION_PLAN_FILENAME,
        TRIAL_RUNTIME_RECEIPT_FILENAME,
    }
)
_EMPTY_OUTPUT_FILENAMES: frozenset[str] = frozenset()
_ATTESTED_OUTPUT_FILENAMES = frozenset(
    {
        RUNTIME_ATTESTATION_RECEIPT_FILENAME,
        RUNTIME_INVOCATION_MARKER_FILENAME,
    }
)


class ProductionCorpusRunError(RuntimeError):
    """Raised when the closed production invocation differs from frozen state."""


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductionCorpusRunError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_absolute_path(name: str, value: object) -> Path:
    if not isinstance(value, (str, Path)):
        raise ProductionCorpusRunError(f"{name} must be a text path")
    raw = os.fspath(value)
    if (
        not isinstance(raw, str)
        or not raw.startswith("/")
        or "\\" in raw
        or unicodedata.normalize("NFC", raw) != raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
        or PurePosixPath(raw).as_posix() != raw
    ):
        raise ProductionCorpusRunError(f"{name} must be a canonical absolute POSIX path")
    return Path(raw)


def _paths_overlap(first: Path, second: Path) -> bool:
    left = PurePosixPath(str(first)).parts
    right = PurePosixPath(str(second)).parts
    common = min(len(left), len(right))
    return left[:common] == right[:common]


def _closed_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProductionCorpusRunError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise ProductionCorpusRunError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProductionCorpusRunError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ProductionCorpusRunError(f"{label} contains non-finite number {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionCorpusRunError(f"{label} must be canonical UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ProductionCorpusRunError(f"{label} must contain one object")
    return value


def _read_control(path: Path, *, label: str) -> bytes:
    try:
        return read_secure_regular_file(path, max_bytes=_MAX_CONTROL_BYTES, label=label)
    except ArtifactIntegrityError as exc:
        raise ProductionCorpusRunError(f"cannot read {label} safely: {exc}") from exc


def _read_pinned_control(path: Path, expected_sha256: str, *, label: str) -> bytes:
    encoded = _read_control(path, label=label)
    if _sha256(encoded) != _require_sha256(f"{label} file sha256", expected_sha256):
        raise ProductionCorpusRunError(f"{label} differs from its config pin")
    return encoded


def _scan_exact_flat_directory(
    root: Path,
    *,
    expected_names: frozenset[str],
    label: str,
) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise ProductionCorpusRunError(f"cannot open {label}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProductionCorpusRunError(f"{label} must be a real directory")
        with os.scandir(descriptor) as iterator:
            entries = list(iterator)
        names = {entry.name for entry in entries}
        if names != expected_names:
            raise ProductionCorpusRunError(
                f"{label} membership differs; missing={sorted(expected_names - names)}, "
                f"extra={sorted(names - expected_names)}"
            )
        for entry in entries:
            if (
                entry.name in {"", ".", ".."}
                or "/" in entry.name
                or "\\" in entry.name
                or unicodedata.normalize("NFC", entry.name) != entry.name
            ):
                raise ProductionCorpusRunError(f"{label} contains a non-canonical name")
            item = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
                raise ProductionCorpusRunError(
                    f"{label}/{entry.name} must be one singly linked regular file"
                )
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ProductionCorpusWorkloadSpec:
    """C1-frozen, outcome-blind identity for one corpus workload."""

    corpus_id: str
    available_family_count: int
    selected_family_count: int
    factory_config_sha256: str
    factory_suite_receipt_sha256: str
    factory_artifact_tree_sha256: str
    runner_image: str
    runner_platform: str
    runner_identity: str
    code_commit: str
    artifact_root: Path
    artifact_tree_sha256: str
    authorized_index_store_root: Path
    authorized_index_store_tree_sha256: str
    embedding_store_root: Path
    embedding_store_tree_sha256: str
    partition_audit_path: Path
    partition_audit_file_sha256: str
    partition_audit_sha256: str
    policy_intervention_root: Path
    policy_intervention_tree_sha256: str
    pseudonym_key_path: Path
    expected_pseudonym_key_sha256: str
    query_package_root: Path
    query_package_tree_sha256: str
    staged_root: Path
    staged_tree_sha256: str
    expected_authorized_index_store_receipt_sha256: str
    expected_policy_intervention_receipt_sha256: str
    policy_bundle_receipt_sha256: str
    index_bundle_receipt_sha256: str
    policy_bundle_receipt_path: Path
    index_bundle_receipt_path: Path
    query_receipt_sha256: str
    online_execution_plan_sha256: str
    online_execution_tree_sha256: str
    sharded_execution_plan_file_sha256: str
    trial_runtime_admission_receipt_file_sha256: str
    feature_bindings: tuple[RuntimeFeatureBinding, ...]
    schema_version: str = PRODUCTION_CORPUS_WORKLOAD_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCTION_CORPUS_WORKLOAD_SPEC_SCHEMA:
            raise ProductionCorpusRunError("production workload spec schema differs")
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionCorpusRunError("production workload spec names another corpus")
        for name in ("available_family_count", "selected_family_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ProductionCorpusRunError(f"{name} must be a positive integer")
        if self.selected_family_count > self.available_family_count:
            raise ProductionCorpusRunError("selected families exceed the audited denominator")
        if type(self.runner_image) is not str or _OCI_IMAGE.fullmatch(self.runner_image) is None:
            raise ProductionCorpusRunError("runner_image must be digest-qualified")
        if self.runner_platform != "linux/arm64":
            raise ProductionCorpusRunError("runner_platform must equal linux/arm64")
        if (
            type(self.runner_identity) is not str
            or not self.runner_identity
            or self.runner_identity != self.runner_identity.strip()
            or unicodedata.normalize("NFC", self.runner_identity) != self.runner_identity
        ):
            raise ProductionCorpusRunError("runner_identity must be canonical text")
        if type(self.code_commit) is not str or (
            _GIT_COMMIT.fullmatch(self.code_commit) is None
            and self.code_commit != C0_COMMIT_SENTINEL
        ):
            raise ProductionCorpusRunError(
                "code_commit must be one full Git commit or the exact candidate sentinel"
            )
        for name in (
            "artifact_root",
            "authorized_index_store_root",
            "embedding_store_root",
            "index_bundle_receipt_path",
            "partition_audit_path",
            "policy_intervention_root",
            "policy_bundle_receipt_path",
            "pseudonym_key_path",
            "query_package_root",
            "staged_root",
        ):
            object.__setattr__(self, name, _canonical_absolute_path(name, getattr(self, name)))
        for name in (
            "factory_config_sha256",
            "factory_suite_receipt_sha256",
            "factory_artifact_tree_sha256",
            "artifact_tree_sha256",
            "authorized_index_store_tree_sha256",
            "embedding_store_tree_sha256",
            "partition_audit_file_sha256",
            "partition_audit_sha256",
            "policy_intervention_tree_sha256",
            "expected_pseudonym_key_sha256",
            "query_package_tree_sha256",
            "staged_tree_sha256",
            "expected_authorized_index_store_receipt_sha256",
            "expected_policy_intervention_receipt_sha256",
            "policy_bundle_receipt_sha256",
            "index_bundle_receipt_sha256",
            "query_receipt_sha256",
            "online_execution_plan_sha256",
            "online_execution_tree_sha256",
            "sharded_execution_plan_file_sha256",
            "trial_runtime_admission_receipt_file_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        bindings = tuple(self.feature_bindings)
        if not bindings or not all(isinstance(row, RuntimeFeatureBinding) for row in bindings):
            raise ProductionCorpusRunError("feature_bindings must contain typed rows")
        if len({row.block_key for row in bindings}) != len(bindings):
            raise ProductionCorpusRunError("feature_bindings repeat a block key")
        object.__setattr__(self, "feature_bindings", bindings)

    @property
    def schedule_path(self) -> Path:
        return self.policy_intervention_root / POLICY_SCHEDULE_FILENAME

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, Path):
                payload[field] = str(value)
            elif field == "feature_bindings":
                payload[field] = [row.to_dict() for row in self.feature_bindings]
            else:
                payload[field] = value
        return payload

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionCorpusWorkloadSpec:
        row = _closed_mapping(
            value,
            fields=_WORKLOAD_SPEC_FIELDS,
            label="production corpus workload spec",
        )
        features = row["feature_bindings"]
        if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
            raise ProductionCorpusRunError("feature_bindings must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "feature_bindings"},
            feature_bindings=tuple(RuntimeFeatureBinding.from_dict(item) for item in features),
        )


@dataclass(frozen=True)
class ProductionCorpusRunConfig:
    """Post-C1 paths and custody hashes for the fixed workload spec."""

    control_root: Path
    output_root: Path
    sealed_run_receipt_path: Path
    runtime_attestation_plan_path: Path
    workload_spec_file_sha256: str
    online_custody_admission_file_sha256: str
    required_artifact_bindings_file_sha256: str
    sealed_run_receipt_file_sha256: str
    schema_version: str = PRODUCTION_CORPUS_RUN_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCTION_CORPUS_RUN_CONFIG_SCHEMA:
            raise ProductionCorpusRunError("production corpus config schema differs")
        for name in (
            "control_root",
            "output_root",
            "runtime_attestation_plan_path",
            "sealed_run_receipt_path",
        ):
            object.__setattr__(self, name, _canonical_absolute_path(name, getattr(self, name)))
        for name in (
            "workload_spec_file_sha256",
            "online_custody_admission_file_sha256",
            "required_artifact_bindings_file_sha256",
            "sealed_run_receipt_file_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if _paths_overlap(self.control_root, self.output_root):
            raise ProductionCorpusRunError("control_root overlaps output_root")
        if any(
            _paths_overlap(self.sealed_run_receipt_path, root)
            for root in (self.control_root, self.output_root)
        ):
            raise ProductionCorpusRunError("sealed run receipt overlaps a writable root")
        if any(
            _paths_overlap(self.runtime_attestation_plan_path, root)
            for root in (self.control_root, self.output_root)
        ):
            raise ProductionCorpusRunError(
                "runtime attestation plan overlaps a production control root"
            )
        if _paths_overlap(self.runtime_attestation_plan_path, self.sealed_run_receipt_path):
            raise ProductionCorpusRunError("runtime plan overlaps the sealed run receipt")

    @property
    def config_path(self) -> Path:
        return self.control_root / PRODUCTION_CORPUS_CONFIG_FILENAME

    @property
    def workload_spec_path(self) -> Path:
        return self.control_root / PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME

    @property
    def runtime_attestation_receipt_path(self) -> Path:
        return self.output_root / RUNTIME_ATTESTATION_RECEIPT_FILENAME

    @property
    def runtime_invocation_marker_path(self) -> Path:
        return self.output_root / RUNTIME_INVOCATION_MARKER_FILENAME

    def control_path(self, filename: str) -> Path:
        if filename not in _CONTROL_FILENAMES:
            raise ProductionCorpusRunError("control filename is not registered")
        return self.control_root / filename

    def to_dict(self) -> dict[str, object]:
        return {
            "control_root": str(self.control_root),
            "online_custody_admission_file_sha256": self.online_custody_admission_file_sha256,
            "output_root": str(self.output_root),
            "required_artifact_bindings_file_sha256": self.required_artifact_bindings_file_sha256,
            "runtime_attestation_plan_path": str(self.runtime_attestation_plan_path),
            "schema_version": self.schema_version,
            "sealed_run_receipt_file_sha256": self.sealed_run_receipt_file_sha256,
            "sealed_run_receipt_path": str(self.sealed_run_receipt_path),
            "workload_spec_file_sha256": self.workload_spec_file_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionCorpusRunConfig:
        row = _closed_mapping(value, fields=_CONFIG_FIELDS, label="production corpus config")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductionCorpusCommandAttempt:
    """Durable consumption marker for the sole config-bound corpus command."""

    config_file_sha256: str
    manifest_sha256: str
    runtime_attestation_plan_sha256: str
    runtime_attestation_receipt_sha256: str
    workload_spec_file_sha256: str
    workload_id: str
    schema_version: str = PRODUCTION_CORPUS_COMMAND_ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "config_file_sha256",
            "manifest_sha256",
            "runtime_attestation_plan_sha256",
            "runtime_attestation_receipt_sha256",
            "workload_spec_file_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.workload_id != PRODUCTION_CORPUS_WORKLOAD_ID:
            raise ProductionCorpusRunError("production command attempt names another workload")
        if self.schema_version != PRODUCTION_CORPUS_COMMAND_ATTEMPT_SCHEMA:
            raise ProductionCorpusRunError("production command attempt schema differs")

    def to_dict(self) -> dict[str, str]:
        return {
            "config_file_sha256": self.config_file_sha256,
            "manifest_sha256": self.manifest_sha256,
            "runtime_attestation_plan_sha256": self.runtime_attestation_plan_sha256,
            "runtime_attestation_receipt_sha256": (self.runtime_attestation_receipt_sha256),
            "schema_version": self.schema_version,
            "workload_id": self.workload_id,
            "workload_spec_file_sha256": self.workload_spec_file_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionCorpusCommandAttempt:
        row = _closed_mapping(
            value,
            fields=_COMMAND_ATTEMPT_FIELDS,
            label="production command attempt",
        )
        return cls(**row)


@dataclass(frozen=True)
class AdmittedProductionCorpusControls:
    """Typed control closure passed to the sealed production boundary."""

    config: ProductionCorpusRunConfig
    config_file_sha256: str
    workload_spec: ProductionCorpusWorkloadSpec
    workload_spec_file_sha256: str
    admission_receipt: OnlineCustodyAdmissionReceipt
    required_artifacts: RequiredArtifactIdBindings
    run_receipt: SealedRunReceipt
    runtime_admission: TrialRuntimeAdmission
    runtime_attestation_plan: RuntimeAttestationPlan
    runtime_attestation_receipt: RuntimeAttestationReceipt


def load_production_corpus_run_config(
    path: str | Path,
) -> ProductionCorpusRunConfig:
    """Load the self-locating post-C1 config without a caller-supplied digest."""

    return _load_config_snapshot(path)[0]


def _load_config_snapshot(
    path: str | Path,
) -> tuple[ProductionCorpusRunConfig, str, bytes]:
    source = _canonical_absolute_path("config path", path)
    encoded = _read_control(source, label="production corpus config")
    config = ProductionCorpusRunConfig.from_dict(
        _decode_object(encoded, label="production corpus config")
    )
    if encoded != config.canonical_file_bytes():
        raise ProductionCorpusRunError("production corpus config bytes are not canonical")
    if source != config.config_path:
        raise ProductionCorpusRunError(
            "production corpus config is not at its self-declared fixed path"
        )
    return config, _sha256(encoded), encoded


def load_production_corpus_workload_spec(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> ProductionCorpusWorkloadSpec:
    """Load the C1-frozen workload spec through the config's exact file pin."""

    source = _canonical_absolute_path("workload spec path", path)
    encoded = _read_pinned_control(
        source,
        expected_file_sha256,
        label="production corpus workload spec",
    )
    spec = ProductionCorpusWorkloadSpec.from_dict(
        _decode_object(encoded, label="production corpus workload spec")
    )
    if spec.code_commit == C0_COMMIT_SENTINEL:
        raise ProductionCorpusRunError(
            "runtime production workload spec still contains the candidate commit sentinel"
        )
    if encoded != spec.canonical_file_bytes():
        raise ProductionCorpusRunError("production workload spec bytes are not canonical")
    return spec


def load_production_corpus_command_attempt(
    path: str | Path,
) -> ProductionCorpusCommandAttempt:
    """Load the canonical command-consumption marker without following links."""

    source = _canonical_absolute_path("production command attempt path", path)
    encoded = _read_control(source, label="production command attempt")
    attempt = ProductionCorpusCommandAttempt.from_dict(
        _decode_object(encoded, label="production command attempt")
    )
    if encoded != attempt.canonical_file_bytes():
        raise ProductionCorpusRunError("production command attempt bytes are not canonical")
    return attempt


def _parse_online_admission(encoded: bytes) -> OnlineCustodyAdmissionReceipt:
    receipt = OnlineCustodyAdmissionReceipt.from_dict(
        _decode_object(encoded, label="online custody admission")
    )
    if encoded != receipt.canonical_bytes() + b"\n":
        raise ProductionCorpusRunError("online custody admission bytes are not canonical")
    return receipt


def _parse_required_artifacts(encoded: bytes) -> RequiredArtifactIdBindings:
    bindings = RequiredArtifactIdBindings.from_dict(
        _decode_object(encoded, label="required artifact bindings")
    )
    if encoded != bindings.canonical_file_bytes():
        raise ProductionCorpusRunError("required artifact bindings bytes are not canonical")
    return bindings


def _parse_sealed_run(encoded: bytes, *, source: Path) -> SealedRunReceipt:
    receipt = SealedRunReceipt.from_dict(_decode_object(encoded, label="sealed run receipt"))
    if encoded != receipt.canonical_bytes() + b"\n":
        raise ProductionCorpusRunError("sealed run receipt bytes are not canonical")
    if source.as_uri() != receipt.receipt_uri:
        raise ProductionCorpusRunError("sealed run receipt is not at its receipt_uri")
    return receipt


def _parse_trial_runtime_receipt(encoded: bytes) -> TrialRuntimeAdmissionReceipt:
    receipt = TrialRuntimeAdmissionReceipt.from_dict(
        _decode_object(encoded, label="trial runtime admission receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionCorpusRunError("trial runtime admission receipt bytes are not canonical")
    return receipt


def _verify_typed_control_closure(
    *,
    admission: OnlineCustodyAdmissionReceipt,
    required: RequiredArtifactIdBindings,
    run: SealedRunReceipt,
    execution: ShardedOnlineExecutionPlan,
    runtime: TrialRuntimeAdmission,
    attestation_plan: RuntimeAttestationPlan,
    config: ProductionCorpusRunConfig,
    workload_spec: ProductionCorpusWorkloadSpec,
) -> None:
    verification = required.verification_receipt
    if {
        admission.manifest_sha256,
        verification.manifest_sha256,
        run.manifest_sha256,
        attestation_plan.manifest_sha256,
    } != {run.manifest_sha256}:
        raise ProductionCorpusRunError("control files bind different study manifests")
    if (
        admission.run_receipt_sha256 != run.binding_sha256
        or admission.runner_identity != run.runner_identity
        or admission.artifact_verification_receipt_sha256 != verification.receipt_sha256
        or run.verification_receipt_sha256 != verification.receipt_sha256
    ):
        raise ProductionCorpusRunError("custody, run, and artifact verification bindings differ")
    if not required.required_artifact_ids.issubset(admission.verified_artifact_ids):
        raise ProductionCorpusRunError("required artifact IDs exceed the admitted closure")
    verified_by_id = {row.artifact_id: row for row in verification.artifacts}
    execution_row = verified_by_id.get(required.execution_artifact_id)
    if (
        execution_row is None
        or not execution_row.exact
        or required.execution_revision_sha256 != execution.artifact_sha256
    ):
        raise ProductionCorpusRunError(
            "execution plan differs from the manifest logical revision or lacks "
            "an exact outer artifact row"
        )
    if runtime.plan != execution:
        raise ProductionCorpusRunError("runtime admission reconstructed another execution plan")
    if (
        execution.artifact_sha256 != workload_spec.online_execution_plan_sha256
        or runtime.receipt.receipt_sha256
        != workload_spec.trial_runtime_admission_receipt_file_sha256
    ):
        raise ProductionCorpusRunError("runtime controls differ from the frozen workload spec")
    _verify_runtime_invocation_bindings(
        attestation_plan,
        config=config,
        workload_spec=workload_spec,
    )
    if (
        attestation_plan.runner_identity != run.runner_identity
        or attestation_plan.code_commit != run.code_commit
        or attestation_plan.oci_image_digest != run.runner_image
        or workload_spec.runner_identity != run.runner_identity
        or workload_spec.code_commit != run.code_commit
        or workload_spec.runner_image != run.runner_image
    ):
        raise ProductionCorpusRunError("runtime plan differs from the sealed run identity")


def _verify_runtime_invocation_bindings(
    plan: RuntimeAttestationPlan,
    *,
    config: ProductionCorpusRunConfig,
    workload_spec: ProductionCorpusWorkloadSpec,
) -> None:
    """Bind the plan to the C1 workload spec and fixed one-path command."""

    if (
        plan.workload_id != PRODUCTION_CORPUS_WORKLOAD_ID
        or plan.workload_sha256 != workload_spec.file_sha256
        or config.workload_spec_file_sha256 != workload_spec.file_sha256
        or Path(plan.invocation_marker_path) != config.runtime_invocation_marker_path
    ):
        raise ProductionCorpusRunError(
            "runtime plan differs from the frozen workload spec or marker"
        )
    expected_argv = (
        _PYTHON_BINARY,
        "-m",
        "fractal_ann_diagnostics.cli",
        "run-sealed-corpus",
        "--config",
        str(config.config_path),
    )
    if plan.argv != expected_argv:
        raise ProductionCorpusRunError("runtime plan argv differs from the sole production command")

    expected_mounts = (
        (
            "sealed-online-artifact",
            workload_spec.artifact_root,
            "directory",
            workload_spec.artifact_tree_sha256,
        ),
        (
            "authorized-index-store",
            workload_spec.authorized_index_store_root,
            "directory",
            workload_spec.authorized_index_store_tree_sha256,
        ),
        (
            "embedding-store",
            workload_spec.embedding_store_root,
            "directory",
            workload_spec.embedding_store_tree_sha256,
        ),
        (
            "partition-audit",
            workload_spec.partition_audit_path,
            "file",
            workload_spec.partition_audit_file_sha256,
        ),
        (
            "policy-intervention",
            workload_spec.policy_intervention_root,
            "directory",
            workload_spec.policy_intervention_tree_sha256,
        ),
        (
            "pseudonym-key",
            workload_spec.pseudonym_key_path,
            "file",
            workload_spec.expected_pseudonym_key_sha256,
        ),
        (
            "query-package",
            workload_spec.query_package_root,
            "directory",
            workload_spec.query_package_tree_sha256,
        ),
        ("staged-inputs", workload_spec.staged_root, "directory", workload_spec.staged_tree_sha256),
        (
            "policy-stage-bundle",
            workload_spec.policy_bundle_receipt_path,
            "file",
            workload_spec.policy_bundle_receipt_sha256,
        ),
        (
            "index-stage-bundle",
            workload_spec.index_bundle_receipt_path,
            "file",
            workload_spec.index_bundle_receipt_sha256,
        ),
    )
    declared_by_role = {mount.role: mount for mount in plan.mounts}
    if len(declared_by_role) != len(plan.mounts):
        raise ProductionCorpusRunError("runtime plan repeats an artifact mount role")
    for role, root, kind, digest in expected_mounts:
        if declared_by_role.get(role) != RuntimeArtifactMount(
            root=str(root),
            role=role,
            kind=kind,  # type: ignore[arg-type]
            artifact_sha256=digest,
        ):
            raise ProductionCorpusRunError(
                f"runtime plan does not enforce workload source {role!r}"
            )


def _digest_tree(path: Path, *, label: str) -> str:
    try:
        return digest_directory_tree(path).sha256
    except ArtifactIntegrityError as exc:
        raise ProductionCorpusRunError(f"cannot rehash {label}: {exc}") from exc


def _digest_file(path: Path, *, label: str) -> str:
    try:
        return digest_regular_file(path, label=label)
    except ArtifactIntegrityError as exc:
        raise ProductionCorpusRunError(f"cannot rehash {label}: {exc}") from exc


def _verify_workload_spec_sources(spec: ProductionCorpusWorkloadSpec) -> None:
    """Rehash and semantically reopen every source named by the C1 spec."""

    tree_pins = (
        (spec.artifact_root, spec.artifact_tree_sha256, "online execution tree"),
        (
            spec.authorized_index_store_root,
            spec.authorized_index_store_tree_sha256,
            "authorized index tree",
        ),
        (spec.embedding_store_root, spec.embedding_store_tree_sha256, "embedding tree"),
        (
            spec.policy_intervention_root,
            spec.policy_intervention_tree_sha256,
            "policy intervention tree",
        ),
        (spec.query_package_root, spec.query_package_tree_sha256, "query package tree"),
        (spec.staged_root, spec.staged_tree_sha256, "staged input tree"),
    )
    for path, expected, label in tree_pins:
        if _digest_tree(path, label=label) != expected:
            raise ProductionCorpusRunError(f"{label} differs from the C1 workload spec")
    file_pins = (
        (spec.partition_audit_path, spec.partition_audit_file_sha256, "partition audit"),
        (spec.pseudonym_key_path, spec.expected_pseudonym_key_sha256, "pseudonym key"),
        (
            spec.policy_bundle_receipt_path,
            spec.policy_bundle_receipt_sha256,
            "policy stage-bundle receipt",
        ),
        (
            spec.index_bundle_receipt_path,
            spec.index_bundle_receipt_sha256,
            "index stage-bundle receipt",
        ),
    )
    for path, expected, label in file_pins:
        if _digest_file(path, label=label) != expected:
            raise ProductionCorpusRunError(f"{label} differs from the C1 workload spec")
    if spec.artifact_tree_sha256 != spec.online_execution_tree_sha256:
        raise ProductionCorpusRunError("online artifact and factory online-tree pins differ")

    try:
        execution_path = spec.artifact_root / ONLINE_EXECUTION_PLAN_FILENAME
        execution = loads_sharded_online_execution_plan(
            _read_pinned_control(
                execution_path,
                spec.sharded_execution_plan_file_sha256,
                label="online execution plan",
            )
        )
        index = load_authorized_index_store_receipt(spec.authorized_index_store_root)
        embedding = load_embedding_store_receipt(spec.embedding_store_root)
        policy = load_policy_intervention_receipt(
            spec.policy_intervention_root / POLICY_RECEIPT_FILENAME
        )
        partition_audit = load_scalable_partition_audit(spec.partition_audit_path)
        query_bytes = _read_pinned_control(
            spec.query_package_root / QUERY_TRIAL_RECEIPT_FILENAME,
            spec.query_receipt_sha256,
            label="query-trial receipt",
        )
        query = QueryTrialStoreReceipt.from_dict(
            _decode_object(query_bytes, label="query-trial receipt")
        )
        policy_bundle_bytes = _read_pinned_control(
            spec.policy_bundle_receipt_path,
            spec.policy_bundle_receipt_sha256,
            label="policy stage-bundle receipt",
        )
        policy_bundle = PolicyStageBundleReceipt.from_dict(
            _decode_object(policy_bundle_bytes, label="policy stage-bundle receipt")
        )
        index_bundle_bytes = _read_pinned_control(
            spec.index_bundle_receipt_path,
            spec.index_bundle_receipt_sha256,
            label="index stage-bundle receipt",
        )
        index_bundle = IndexStageBundleReceipt.from_dict(
            _decode_object(index_bundle_bytes, label="index stage-bundle receipt")
        )
    except ProductionCorpusRunError:
        raise
    except Exception as exc:
        raise ProductionCorpusRunError(
            f"cannot semantically reopen the C1 workload sources: {exc}"
        ) from exc

    if (
        query_bytes != query.canonical_file_bytes()
        or policy_bundle_bytes != policy_bundle.canonical_file_bytes()
        or index_bundle_bytes != index_bundle.canonical_file_bytes()
    ):
        raise ProductionCorpusRunError("a workload source receipt is not canonical")
    if (
        execution.corpus != spec.corpus_id
        or execution.artifact_sha256 != spec.online_execution_plan_sha256
        or partition_audit.artifact_sha256 != spec.partition_audit_sha256
        or execution.query_partition_audit_sha256 != spec.partition_audit_sha256
        or index.artifact_sha256 != spec.expected_authorized_index_store_receipt_sha256
        or policy.artifact_sha256 != spec.expected_policy_intervention_receipt_sha256
        or index.embedding_receipt_sha256 != embedding.receipt_sha256
        or index.policy_receipt_sha256 != policy.artifact_sha256
        or query.corpus != spec.corpus_id
        or query.embedding_store_receipt_sha256 != embedding.receipt_sha256
        or query.query_partition_audit_sha256 != spec.partition_audit_sha256
        or policy_bundle.corpus_id != spec.corpus_id
        or policy_bundle.receipt_sha256 != spec.policy_bundle_receipt_sha256
        or policy_bundle.stages[-1].receipt_sha256 != policy.artifact_sha256
        or index_bundle.corpus_id != spec.corpus_id
        or index_bundle.receipt_sha256 != spec.index_bundle_receipt_sha256
        or index_bundle.stages[-1].receipt_sha256 != index.artifact_sha256
    ):
        raise ProductionCorpusRunError(
            "typed workload sources differ from the C1 workload specification"
        )


def _attest_production_runtime(
    config: ProductionCorpusRunConfig,
    workload_spec: ProductionCorpusWorkloadSpec,
) -> tuple[
    RuntimeAttestationPlan,
    RuntimeAttestationReceipt,
]:
    """Consume runtime attestation before admitting any subordinate control."""

    _scan_exact_flat_directory(
        config.output_root,
        expected_names=_EMPTY_OUTPUT_FILENAMES,
        label="production output root before runtime attestation",
    )

    plan = loads_runtime_attestation_plan(
        _read_control(
            config.runtime_attestation_plan_path,
            label="runtime attestation plan",
        )
    )
    _verify_runtime_invocation_bindings(
        plan,
        config=config,
        workload_spec=workload_spec,
    )

    created_receipt = attest_runtime_once(
        plan,
        probe=LinuxRuntimeProbe(),
        receipt_target=config.runtime_attestation_receipt_path,
    )
    persisted_receipt = loads_runtime_attestation_receipt(
        _read_control(
            config.runtime_attestation_receipt_path,
            label="runtime attestation receipt",
        )
    )
    verify_runtime_attestation_receipt(persisted_receipt, plan)
    if persisted_receipt != created_receipt:
        raise ProductionCorpusRunError(
            "persisted runtime attestation receipt differs from the same-process result"
        )
    _scan_exact_flat_directory(
        config.output_root,
        expected_names=_ATTESTED_OUTPUT_FILENAMES,
        label="production output root after runtime attestation",
    )
    return plan, persisted_receipt


def load_admitted_production_corpus_controls(
    config_path: str | Path,
) -> AdmittedProductionCorpusControls:
    """Load and cross-bind every pre-existing control without source-package I/O."""

    config, config_digest, config_bytes = _load_config_snapshot(config_path)
    workload_spec = load_production_corpus_workload_spec(
        config.workload_spec_path,
        expected_file_sha256=config.workload_spec_file_sha256,
    )
    return _load_admitted_production_corpus_controls(
        config,
        config_digest=config_digest,
        config_bytes=config_bytes,
        workload_spec=workload_spec,
    )


def _load_admitted_production_corpus_controls(
    config: ProductionCorpusRunConfig,
    *,
    config_digest: str,
    config_bytes: bytes,
    workload_spec: ProductionCorpusWorkloadSpec,
) -> AdmittedProductionCorpusControls:
    _scan_exact_flat_directory(
        config.control_root,
        expected_names=_CONTROL_FILENAMES,
        label="production control root",
    )
    _scan_exact_flat_directory(
        config.output_root,
        expected_names=_ATTESTED_OUTPUT_FILENAMES,
        label="production output root after runtime attestation",
    )

    online_bytes = _read_pinned_control(
        config.control_path(ONLINE_CUSTODY_ADMISSION_FILENAME),
        config.online_custody_admission_file_sha256,
        label="online custody admission",
    )
    binding_bytes = _read_pinned_control(
        config.control_path(REQUIRED_ARTIFACT_BINDINGS_FILENAME),
        config.required_artifact_bindings_file_sha256,
        label="required artifact bindings",
    )
    run_path = config.sealed_run_receipt_path
    run_bytes = _read_pinned_control(
        run_path,
        config.sealed_run_receipt_file_sha256,
        label="sealed run receipt",
    )
    execution_bytes = _read_pinned_control(
        config.control_path(SHARDED_EXECUTION_PLAN_FILENAME),
        workload_spec.sharded_execution_plan_file_sha256,
        label="sharded execution plan",
    )
    runtime_bytes = _read_pinned_control(
        config.control_path(TRIAL_RUNTIME_RECEIPT_FILENAME),
        workload_spec.trial_runtime_admission_receipt_file_sha256,
        label="trial runtime admission receipt",
    )
    attestation_plan_bytes = _read_control(
        config.runtime_attestation_plan_path,
        label="runtime attestation plan",
    )
    attestation_receipt_bytes = _read_control(
        config.runtime_attestation_receipt_path,
        label="runtime attestation receipt",
    )

    admission = _parse_online_admission(online_bytes)
    required = _parse_required_artifacts(binding_bytes)
    run = _parse_sealed_run(run_bytes, source=run_path)
    execution = loads_sharded_online_execution_plan(execution_bytes)
    runtime_receipt = _parse_trial_runtime_receipt(runtime_bytes)
    attestation_plan = loads_runtime_attestation_plan(attestation_plan_bytes)
    attestation_receipt = loads_runtime_attestation_receipt(attestation_receipt_bytes)
    verify_runtime_attestation_receipt(attestation_receipt, attestation_plan)
    runtime_admission = reconstruct_trial_runtime_admission(
        plan=execution,
        receipt=runtime_receipt,
        partition_audit_path=workload_spec.partition_audit_path,
        query_package_root=workload_spec.query_package_root,
        staged_root=workload_spec.staged_root,
        embedding_store_root=workload_spec.embedding_store_root,
        schedule_path=workload_spec.schedule_path,
        feature_bindings=workload_spec.feature_bindings,
    )
    _verify_typed_control_closure(
        admission=admission,
        required=required,
        run=run,
        execution=execution,
        runtime=runtime_admission,
        attestation_plan=attestation_plan,
        config=config,
        workload_spec=workload_spec,
    )
    if _read_control(config.config_path, label="production corpus config") != config_bytes:
        raise ProductionCorpusRunError("production corpus config changed during admission")
    if (
        _read_control(config.workload_spec_path, label="production corpus workload spec")
        != workload_spec.canonical_file_bytes()
    ):
        raise ProductionCorpusRunError("production workload spec changed during admission")
    return AdmittedProductionCorpusControls(
        config=config,
        config_file_sha256=config_digest,
        workload_spec=workload_spec,
        workload_spec_file_sha256=workload_spec.file_sha256,
        admission_receipt=admission,
        required_artifacts=required,
        run_receipt=run,
        runtime_admission=runtime_admission,
        runtime_attestation_plan=attestation_plan,
        runtime_attestation_receipt=attestation_receipt,
    )


def run_sealed_corpus_from_config(
    config_path: str | Path,
    runtime_claim_receipt: RuntimeClaimReceipt,
) -> PersistedSealedOnlineRun:
    """Execute the one admitted corpus with no caller-supplied scientific facts."""

    if not isinstance(runtime_claim_receipt, RuntimeClaimReceipt):
        raise ProductionCorpusRunError(
            "sealed corpus execution requires a typed RUN_CLAIMED runtime receipt"
        )
    config, config_digest, config_bytes = _load_config_snapshot(config_path)
    workload_spec = load_production_corpus_workload_spec(
        config.workload_spec_path,
        expected_file_sha256=config.workload_spec_file_sha256,
    )
    initial_plan, initial_receipt = _attest_production_runtime(
        config,
        workload_spec,
    )
    _verify_workload_spec_sources(workload_spec)
    controls = _load_admitted_production_corpus_controls(
        config,
        config_digest=config_digest,
        config_bytes=config_bytes,
        workload_spec=workload_spec,
    )
    if (
        controls.runtime_attestation_plan != initial_plan
        or controls.runtime_attestation_receipt != initial_receipt
    ):
        raise ProductionCorpusRunError("runtime controls changed after same-process attestation")
    config = controls.config
    command_attempt = ProductionCorpusCommandAttempt(
        config_file_sha256=controls.config_file_sha256,
        manifest_sha256=controls.run_receipt.manifest_sha256,
        runtime_attestation_plan_sha256=(controls.runtime_attestation_plan.plan_sha256),
        runtime_attestation_receipt_sha256=(controls.runtime_attestation_receipt.receipt_sha256),
        workload_spec_file_sha256=controls.workload_spec_file_sha256,
        workload_id=PRODUCTION_CORPUS_WORKLOAD_ID,
    )
    try:
        write_exclusive_receipt_bytes(
            command_attempt.canonical_file_bytes(),
            config.output_root / PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
        )
    except ArtifactIntegrityError as exc:
        raise ProductionCorpusRunError(
            f"cannot consume the sealed corpus command attempt: {exc}"
        ) from exc
    return run_sealed_online_once(
        output_root=config.output_root,
        admission_receipt=controls.admission_receipt,
        required_artifacts=controls.required_artifacts,
        run_receipt=controls.run_receipt,
        runtime_admission=controls.runtime_admission,
        runtime_attestation_plan_path=config.runtime_attestation_plan_path,
        expected_runtime_attestation_plan_sha256=(controls.runtime_attestation_plan.plan_sha256),
        runtime_attestation_receipt_path=config.runtime_attestation_receipt_path,
        expected_runtime_attestation_receipt_sha256=(
            controls.runtime_attestation_receipt.receipt_sha256
        ),
        expected_runtime_receipt_sha256=(controls.runtime_admission.receipt.receipt_sha256),
        artifact_root=workload_spec.artifact_root,
        authorized_index_store_root=workload_spec.authorized_index_store_root,
        expected_authorized_index_store_receipt_sha256=(
            workload_spec.expected_authorized_index_store_receipt_sha256
        ),
        policy_intervention_root=workload_spec.policy_intervention_root,
        expected_policy_intervention_receipt_sha256=(
            workload_spec.expected_policy_intervention_receipt_sha256
        ),
        pseudonym_key_path=workload_spec.pseudonym_key_path,
        expected_pseudonym_key_sha256=workload_spec.expected_pseudonym_key_sha256,
        runtime_claim_receipt=runtime_claim_receipt,
    )
