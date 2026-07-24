"""Typed, resumable development processing after the paired embedding build.

The operator admits one verified five-corpus embedding suite and one exact
partition audit.  It then performs the development-only cohort, policy, index,
paired-execution, freeze, and joint-power stages.  There is no argument for a
sealed label, confirmatory outcome, result tree, plugin, or callback.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .authorized_index_store import (
    AuthorizedIndexStoreError,
    build_authorized_index_store,
    verify_authorized_index_store,
)
from .development_cohort import (
    CALIBRATION_FAMILY_COUNT,
    FIT_FAMILY_COUNT,
    DevelopmentCohortError,
    DevelopmentEmbeddingBinding,
    canonical_development_embedding_bindings_bytes,
    load_development_cohort_selection,
    load_development_embedding_bindings,
    load_development_execution_plan,
    materialize_development_cohort,
    select_development_cohort,
    verify_materialized_development_cohort,
)
from .development_execution import (
    DevelopmentExecutionError,
    DevelopmentExecutionInput,
    DevelopmentPairedExecutionConfig,
    build_development_freeze_config,
    load_development_paired_execution_config,
    run_development_paired_execution,
    verify_development_paired_execution,
    write_bound_development_freeze_config,
)
from .development_freeze import (
    DevelopmentFreezeError,
    canonical_development_freeze_config_bytes,
    compile_development_freeze,
    load_development_freeze_config,
    verify_development_freeze,
)
from .joint_power_design import (
    FIXED_CORPORA,
    JointPowerDesignError,
    canonical_development_panel_bytes,
    canonical_joint_power_config_bytes,
    canonical_joint_power_report_bytes,
    canonical_joint_power_selection_audit_bytes,
    load_development_panel,
    load_joint_power_config,
    load_joint_power_report,
    load_joint_power_selection_audit,
    run_joint_power_design,
    run_joint_power_selection_audit,
    verify_joint_power_selection_audit,
)
from .policy_intervention import (
    PolicyInterventionError,
    verify_policy_intervention_package,
    write_policy_intervention_package,
)
from .production_artifact_factory import (
    ProductionArtifactFactoryError,
    derive_production_policy_config,
    production_authorized_index_components,
)
from .production_embedding_build import (
    ProductionEmbeddingBuildError,
    ProductionEmbeddingConfig,
    ProductionEmbeddingSuiteReceipt,
    admit_frozen_production_embedding_suite,
    load_production_embedding_config,
)
from .scalable_partition_audit import (
    ScalablePartitionAuditError,
    ScalableQueryPartitionAuditReceipt,
    load_scalable_partition_audit,
)

POST_EMBEDDING_CONFIG_SCHEMA = "fractal-post-embedding-development-config-v1"
POST_EMBEDDING_RECEIPT_SCHEMA = "fractal-post-embedding-development-receipt-v1"
POST_EMBEDDING_STRATUM_SCHEMA = "fractal-post-embedding-development-stratum-v1"
POST_EMBEDDING_ARTIFACT_SCHEMA = "fractal-post-embedding-development-artifact-v1"
JOINT_POWER_INVOCATION_SCHEMA = "fractal-joint-power-single-invocation-v1"
POST_EMBEDDING_CLI_RESULT_SCHEMA = "fractal-post-embedding-development-cli-result-v1"

OPERATOR_CONFIG_FILENAME = "operator-config.json"
SELECTION_FILENAME = "selection-receipt.json"
EMBEDDING_BINDINGS_FILENAME = "embedding-bindings.json"
MATERIALIZATION_DIRECTORY = "materialized-development"
POLICY_DIRECTORY = "policy-packages"
INDEX_DIRECTORY = "authorized-index-packages"
EXECUTION_INPUT_FILENAME = "paired-execution-config.json"
EXECUTION_DIRECTORY = "paired-execution"
FREEZE_CONFIG_FILENAME = "development-freeze-config.json"
FREEZE_DIRECTORY = "development-freeze"
JOINT_POWER_INVOCATION_FILENAME = "joint-power-invocation.json"
ANALYSIS_DIRECTORY = "analysis"
JOINT_POWER_DIRECTORY = "joint-power-design"
JOINT_POWER_SELECTION_AUDIT_FILENAME = "selection-audit.json"
RECEIPT_FILENAME = "post-embedding-development-receipt.json"

DEVELOPMENT_STAGES = ("development-fit", "development-calibration")
SOURCE_STAGE = {
    "development-fit": "fit",
    "development-calibration": "calibration",
}
EXPECTED_DEVELOPMENT_FAMILIES = len(FIXED_CORPORA) * (FIT_FAMILY_COUNT + CALIBRATION_FAMILY_COUNT)
EXPECTED_PAIRED_TRIALS = EXPECTED_DEVELOPMENT_FAMILIES * 3
EXPECTED_ACTION_ROWS = EXPECTED_PAIRED_TRIALS * 4

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 256 * 1024 * 1024
_FORBIDDEN_PATH_TOKENS = frozenset(
    {
        "custody",
        "heldout",
        "holdout",
        "label",
        "labels",
        "outcome",
        "outcomes",
        "reserve",
        "reserved",
        "result",
        "results",
        "sealed",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "design_seed_sha256",
        "full_staged_inventory_sha256",
        "full_staged_root",
        "output_root",
        "partition_audit_file_sha256",
        "partition_audit_path",
        "production_embedding_config_path",
        "production_embedding_config_sha256",
        "schema_version",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "byte_count",
        "directory_count",
        "file_count",
        "kind",
        "path",
        "schema_version",
        "sha256",
        "stage_id",
    }
)
_STRATUM_FIELDS = frozenset(
    {
        "authorized_index_config_sha256",
        "authorized_index_receipt_sha256",
        "corpus",
        "development_stage",
        "embedding_receipt_sha256",
        "policy_config_sha256",
        "policy_intervention_receipt_sha256",
        "schema_version",
        "source_stage",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "artifacts",
        "index_config_sha256",
        "config_sha256",
        "development_family_count",
        "embedding_bindings_sha256",
        "embedding_suite_receipt_sha256",
        "execution_config_sha256",
        "execution_receipt_sha256",
        "freeze_config_sha256",
        "freeze_receipt_sha256",
        "freeze_tree_sha256",
        "full_staged_inventory_sha256",
        "joint_power_config_sha256",
        "joint_power_invocation_sha256",
        "joint_power_report_sha256",
        "joint_power_report_tree_sha256",
        "development_materialization_receipt_sha256",
        "paired_action_row_count",
        "paired_trial_count",
        "partition_audit_sha256",
        "partition_audit_file_sha256",
        "design_seed_sha256",
        "schema_version",
        "selected_families_per_corpus",
        "selection_receipt_sha256",
        "strata",
    }
)
_INVOCATION_FIELDS = frozenset(
    {
        "authorized_invocation_count",
        "freeze_tree_sha256",
        "joint_power_config_sha256",
        "panel_sha256s",
        "schema_version",
    }
)
_KNOWN_TOP_LEVEL = frozenset(
    {
        ANALYSIS_DIRECTORY,
        EMBEDDING_BINDINGS_FILENAME,
        EXECUTION_DIRECTORY,
        EXECUTION_INPUT_FILENAME,
        FREEZE_CONFIG_FILENAME,
        FREEZE_DIRECTORY,
        INDEX_DIRECTORY,
        JOINT_POWER_INVOCATION_FILENAME,
        MATERIALIZATION_DIRECTORY,
        OPERATOR_CONFIG_FILENAME,
        POLICY_DIRECTORY,
        RECEIPT_FILENAME,
        SELECTION_FILENAME,
    }
)


class PostEmbeddingDevelopmentError(RuntimeError):
    """Raised when the post-embedding development chain is not exact."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8", errors="strict")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PostEmbeddingDevelopmentError("control data must be finite canonical JSON") from exc


def _decode(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PostEmbeddingDevelopmentError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PostEmbeddingDevelopmentError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostEmbeddingDevelopmentError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PostEmbeddingDevelopmentError(f"{label} must contain one object")
    return value


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PostEmbeddingDevelopmentError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise PostEmbeddingDevelopmentError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PostEmbeddingDevelopmentError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _path_tokens(path: Path) -> set[str]:
    return {
        token for part in path.parts for token in re.split(r"[^a-z0-9]+", part.casefold()) if token
    }


def _canonical_absolute_path(value: object, *, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise PostEmbeddingDevelopmentError(f"{label} must be an absolute canonical POSIX path")
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or str(pure) != value
        or pure == PurePosixPath("/")
        or pure.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in pure.parts[1:])
    ):
        raise PostEmbeddingDevelopmentError(f"{label} must be an absolute canonical POSIX path")
    path = Path(value)
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise PostEmbeddingDevelopmentError(f"cannot resolve {label}: {exc}") from exc
    if resolved != path:
        raise PostEmbeddingDevelopmentError(f"{label} crosses an alias or symbolic link")
    forbidden = sorted(_path_tokens(path).intersection(_FORBIDDEN_PATH_TOKENS))
    if forbidden:
        raise PostEmbeddingDevelopmentError(
            f"{label} contains a forbidden non-development token: {forbidden}"
        )
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_real_directory(path: Path, *, label: str, private: bool = False) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PostEmbeddingDevelopmentError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise PostEmbeddingDevelopmentError(f"{label} must be a real directory")
    if private and (
        (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise PostEmbeddingDevelopmentError(
            f"{label} must be runner-owned and not writable by group or other identities"
        )


def _require_read_only_filesystem(path: Path, *, label: str) -> None:
    try:
        flags = os.statvfs(path).f_flag
    except OSError as exc:
        raise PostEmbeddingDevelopmentError(f"cannot inspect {label} mount: {exc}") from exc
    read_only_flag = getattr(os, "ST_RDONLY", 1)
    if not flags & read_only_flag:
        raise PostEmbeddingDevelopmentError(f"{label} must be mounted read-only")


def _read_control(path: Path, *, label: str) -> bytes:
    try:
        return read_secure_regular_file(path, max_bytes=_MAX_CONTROL_BYTES, label=label)
    except ArtifactIntegrityError as exc:
        raise PostEmbeddingDevelopmentError(f"cannot read {label}: {exc}") from exc


def _write_exclusive(path: Path, encoded: bytes, *, label: str) -> None:
    try:
        write_exclusive_receipt_bytes(encoded, path)
    except ArtifactIntegrityError as exc:
        raise PostEmbeddingDevelopmentError(f"cannot write {label}: {exc}") from exc


@dataclass(frozen=True)
class _FreshJointPowerVerification:
    """Same-process proof that one generated or replayed tree was canonically read back."""

    bundle_root: Path
    freeze_tree_sha256: str
    invocation_bytes: bytes
    power_config: Any
    panels: tuple[Any, ...]
    report: Any
    tree_sha256: str


@dataclass(frozen=True)
class PostEmbeddingDevelopmentConfig:
    """Closed identity for one development-only post-embedding run."""

    production_embedding_config_path: Path
    production_embedding_config_sha256: str
    full_staged_root: Path
    full_staged_inventory_sha256: str
    partition_audit_path: Path
    partition_audit_file_sha256: str
    design_seed_sha256: str
    output_root: Path
    schema_version: str = POST_EMBEDDING_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "production_embedding_config_path",
            "full_staged_root",
            "partition_audit_path",
            "output_root",
        ):
            path = _canonical_absolute_path(str(getattr(self, name)), label=name)
            object.__setattr__(self, name, path)
        for name in (
            "production_embedding_config_sha256",
            "full_staged_inventory_sha256",
            "partition_audit_file_sha256",
            "design_seed_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        paths = (
            self.production_embedding_config_path,
            self.full_staged_root,
            self.partition_audit_path,
            self.output_root,
        )
        for position, left in enumerate(paths):
            for right in paths[position + 1 :]:
                if _paths_overlap(left, right):
                    raise PostEmbeddingDevelopmentError("operator input and output paths overlap")
        if self.schema_version != POST_EMBEDDING_CONFIG_SCHEMA:
            raise PostEmbeddingDevelopmentError("post-embedding config schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "design_seed_sha256": self.design_seed_sha256,
            "full_staged_inventory_sha256": self.full_staged_inventory_sha256,
            "full_staged_root": str(self.full_staged_root),
            "output_root": str(self.output_root),
            "partition_audit_file_sha256": self.partition_audit_file_sha256,
            "partition_audit_path": str(self.partition_audit_path),
            "production_embedding_config_path": str(self.production_embedding_config_path),
            "production_embedding_config_sha256": self.production_embedding_config_sha256,
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> PostEmbeddingDevelopmentConfig:
        row = _closed(value, _CONFIG_FIELDS, label="post-embedding config")
        return cls(
            production_embedding_config_path=Path(row["production_embedding_config_path"]),
            production_embedding_config_sha256=row["production_embedding_config_sha256"],
            full_staged_root=Path(row["full_staged_root"]),
            full_staged_inventory_sha256=row["full_staged_inventory_sha256"],
            partition_audit_path=Path(row["partition_audit_path"]),
            partition_audit_file_sha256=row["partition_audit_file_sha256"],
            design_seed_sha256=row["design_seed_sha256"],
            output_root=Path(row["output_root"]),
            schema_version=row["schema_version"],
        )


def load_post_embedding_development_config(
    path: str | Path,
    *,
    expected_sha256: str,
) -> PostEmbeddingDevelopmentConfig:
    config_path = _canonical_absolute_path(str(path), label="post-embedding config path")
    expected = _require_sha256("expected config SHA-256", expected_sha256)
    encoded = _read_control(config_path, label="post-embedding config")
    if _sha256(encoded) != expected:
        raise PostEmbeddingDevelopmentError("post-embedding config differs from its caller pin")
    config = PostEmbeddingDevelopmentConfig.from_dict(
        _decode(encoded, label="post-embedding config")
    )
    if encoded != config.canonical_file_bytes():
        raise PostEmbeddingDevelopmentError("post-embedding config is not canonical")
    return config


@dataclass(frozen=True)
class _AdmittedUpstream:
    embedding_config: ProductionEmbeddingConfig
    embedding_suite: ProductionEmbeddingSuiteReceipt
    partition_audit: ScalableQueryPartitionAuditReceipt


def _admit_upstream(config: PostEmbeddingDevelopmentConfig) -> _AdmittedUpstream:
    _require_real_directory(config.full_staged_root, label="full staged root")
    try:
        inventory_sha256 = digest_regular_file(
            config.full_staged_root / "inventory.json",
            label="full staged inventory",
        )
        audit_file_sha256 = digest_regular_file(
            config.partition_audit_path,
            label="partition audit",
        )
    except ArtifactIntegrityError as exc:
        raise PostEmbeddingDevelopmentError(f"cannot hash an upstream control: {exc}") from exc
    if inventory_sha256 != config.full_staged_inventory_sha256:
        raise PostEmbeddingDevelopmentError("full staged inventory differs from its config pin")
    if audit_file_sha256 != config.partition_audit_file_sha256:
        raise PostEmbeddingDevelopmentError("partition audit differs from its file pin")
    try:
        embedding_config = load_production_embedding_config(
            config.production_embedding_config_path,
            expected_sha256=config.production_embedding_config_sha256,
        )
        embedding_suite = admit_frozen_production_embedding_suite(embedding_config)
        audit = load_scalable_partition_audit(
            config.partition_audit_path,
            expected_inventory_sha256=config.full_staged_inventory_sha256,
        )
    except (
        ProductionEmbeddingBuildError,
        ScalablePartitionAuditError,
    ) as exc:
        raise PostEmbeddingDevelopmentError(f"upstream admission failed: {exc}") from exc
    if (
        embedding_config.online_inventory_sha256 != config.full_staged_inventory_sha256
        or embedding_suite.production_config_sha256 != config.production_embedding_config_sha256
        or embedding_suite.online_inventory_sha256 != config.full_staged_inventory_sha256
        or audit.artifact_sha256 != config.partition_audit_file_sha256
    ):
        raise PostEmbeddingDevelopmentError(
            "embedding suite, full staging inventory, and partition audit are not one cohort"
        )
    if _paths_overlap(config.output_root, embedding_config.output_root):
        raise PostEmbeddingDevelopmentError("operator output overlaps the embedding suite")
    return _AdmittedUpstream(embedding_config, embedding_suite, audit)


def write_post_embedding_development_config(
    *,
    production_embedding_config_path: str | Path,
    production_embedding_config_sha256: str,
    full_staged_root: str | Path,
    full_staged_inventory_sha256: str,
    partition_audit_path: str | Path,
    partition_audit_file_sha256: str,
    design_seed_sha256: str,
    output_root: str | Path,
    destination: str | Path,
) -> PostEmbeddingDevelopmentConfig:
    """Verify upstream pins and exclusively write one canonical operator config."""

    config = PostEmbeddingDevelopmentConfig(
        production_embedding_config_path=Path(production_embedding_config_path),
        production_embedding_config_sha256=production_embedding_config_sha256,
        full_staged_root=Path(full_staged_root),
        full_staged_inventory_sha256=full_staged_inventory_sha256,
        partition_audit_path=Path(partition_audit_path),
        partition_audit_file_sha256=partition_audit_file_sha256,
        design_seed_sha256=design_seed_sha256,
        output_root=Path(output_root),
    )
    target = _canonical_absolute_path(str(destination), label="config destination")
    if os.path.lexists(config.output_root):
        raise PostEmbeddingDevelopmentError("operator output root already exists")
    if any(
        _paths_overlap(target, source)
        for source in (
            config.full_staged_root,
            config.output_root,
            config.production_embedding_config_path,
            config.partition_audit_path,
        )
    ):
        raise PostEmbeddingDevelopmentError("config destination overlaps an input or output")
    _require_real_directory(target.parent, label="config destination parent", private=True)
    _admit_upstream(config)
    _write_exclusive(target, config.canonical_file_bytes(), label="post-embedding config")
    observed = load_post_embedding_development_config(
        target,
        expected_sha256=config.file_sha256,
    )
    if observed != config:
        raise PostEmbeddingDevelopmentError("written operator config changed")
    return config


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PostEmbeddingDevelopmentError(f"{label} must be a canonical relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PostEmbeddingDevelopmentError(f"{label} must be a canonical relative path")
    return value


@dataclass(frozen=True, order=True)
class PostEmbeddingArtifactPin:
    stage_id: str
    path: str
    kind: Literal["file", "directory"]
    sha256: str
    file_count: int
    directory_count: int
    byte_count: int
    schema_version: str = POST_EMBEDDING_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.stage_id, str) or not self.stage_id:
            raise PostEmbeddingDevelopmentError("artifact stage_id must be non-empty")
        object.__setattr__(self, "path", _relative_path(self.path, label="artifact path"))
        if self.kind not in {"file", "directory"}:
            raise PostEmbeddingDevelopmentError("artifact kind differs")
        _require_sha256("artifact SHA-256", self.sha256)
        for name in ("file_count", "directory_count", "byte_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise PostEmbeddingDevelopmentError(f"artifact {name} must be nonnegative")
        if self.kind == "file" and (self.file_count, self.directory_count) != (1, 0):
            raise PostEmbeddingDevelopmentError("file artifact accounting differs")
        if self.kind == "directory" and self.file_count < 1:
            raise PostEmbeddingDevelopmentError("directory artifact must contain a file")
        if self.schema_version != POST_EMBEDDING_ARTIFACT_SCHEMA:
            raise PostEmbeddingDevelopmentError("artifact pin schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "directory_count": self.directory_count,
            "file_count": self.file_count,
            "kind": self.kind,
            "path": self.path,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "stage_id": self.stage_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> PostEmbeddingArtifactPin:
        row = _closed(value, _ARTIFACT_FIELDS, label="post-embedding artifact pin")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True, order=True)
class PostEmbeddingStratumReceipt:
    corpus: str
    development_stage: str
    source_stage: str
    embedding_receipt_sha256: str
    policy_config_sha256: str
    policy_intervention_receipt_sha256: str
    authorized_index_config_sha256: str
    authorized_index_receipt_sha256: str
    schema_version: str = POST_EMBEDDING_STRATUM_SCHEMA

    def __post_init__(self) -> None:
        if self.corpus not in FIXED_CORPORA:
            raise PostEmbeddingDevelopmentError("stratum corpus differs")
        if SOURCE_STAGE.get(self.development_stage) != self.source_stage:
            raise PostEmbeddingDevelopmentError("stratum development/source stage mapping differs")
        for name in (
            "embedding_receipt_sha256",
            "policy_config_sha256",
            "policy_intervention_receipt_sha256",
            "authorized_index_config_sha256",
            "authorized_index_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.schema_version != POST_EMBEDDING_STRATUM_SCHEMA:
            raise PostEmbeddingDevelopmentError("stratum receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "authorized_index_config_sha256": self.authorized_index_config_sha256,
            "authorized_index_receipt_sha256": self.authorized_index_receipt_sha256,
            "corpus": self.corpus,
            "development_stage": self.development_stage,
            "embedding_receipt_sha256": self.embedding_receipt_sha256,
            "policy_config_sha256": self.policy_config_sha256,
            "policy_intervention_receipt_sha256": (self.policy_intervention_receipt_sha256),
            "schema_version": self.schema_version,
            "source_stage": self.source_stage,
        }

    @classmethod
    def from_dict(cls, value: object) -> PostEmbeddingStratumReceipt:
        row = _closed(value, _STRATUM_FIELDS, label="post-embedding stratum receipt")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PostEmbeddingDevelopmentReceipt:
    config_sha256: str
    full_staged_inventory_sha256: str
    partition_audit_file_sha256: str
    partition_audit_sha256: str
    embedding_suite_receipt_sha256: str
    embedding_bindings_sha256: str
    selection_receipt_sha256: str
    development_materialization_receipt_sha256: str
    design_seed_sha256: str
    index_config_sha256: str
    execution_config_sha256: str
    execution_receipt_sha256: str
    freeze_config_sha256: str
    freeze_receipt_sha256: str
    freeze_tree_sha256: str
    joint_power_invocation_sha256: str
    joint_power_config_sha256: str
    joint_power_report_sha256: str
    joint_power_report_tree_sha256: str
    selected_families_per_corpus: int
    development_family_count: int
    paired_trial_count: int
    paired_action_row_count: int
    strata: tuple[PostEmbeddingStratumReceipt, ...]
    artifacts: tuple[PostEmbeddingArtifactPin, ...]
    schema_version: str = POST_EMBEDDING_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "config_sha256",
            "full_staged_inventory_sha256",
            "partition_audit_file_sha256",
            "partition_audit_sha256",
            "embedding_suite_receipt_sha256",
            "embedding_bindings_sha256",
            "selection_receipt_sha256",
            "development_materialization_receipt_sha256",
            "design_seed_sha256",
            "index_config_sha256",
            "execution_config_sha256",
            "execution_receipt_sha256",
            "freeze_config_sha256",
            "freeze_receipt_sha256",
            "freeze_tree_sha256",
            "joint_power_invocation_sha256",
            "joint_power_config_sha256",
            "joint_power_report_sha256",
            "joint_power_report_tree_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if type(self.selected_families_per_corpus) is not int or (
            self.selected_families_per_corpus < 1
        ):
            raise PostEmbeddingDevelopmentError("joint-power family selection is absent")
        if (
            self.development_family_count != EXPECTED_DEVELOPMENT_FAMILIES
            or self.paired_trial_count != EXPECTED_PAIRED_TRIALS
            or self.paired_action_row_count != EXPECTED_ACTION_ROWS
        ):
            raise PostEmbeddingDevelopmentError("development execution cardinality differs")
        strata = tuple(sorted(self.strata, key=lambda row: (row.development_stage, row.corpus)))
        expected = {(stage, corpus) for stage in DEVELOPMENT_STAGES for corpus in FIXED_CORPORA}
        if (
            len(strata) != len(expected)
            or {(row.development_stage, row.corpus) for row in strata} != expected
        ):
            raise PostEmbeddingDevelopmentError("receipt strata do not cover the fixed ten")
        artifacts = tuple(sorted(self.artifacts, key=lambda row: row.path.encode("utf-8")))
        expected_paths = set(_artifact_contract())
        if (
            len(artifacts) != len(expected_paths)
            or {row.path for row in artifacts} != expected_paths
        ):
            raise PostEmbeddingDevelopmentError("receipt artifact contract differs")
        if self.schema_version != POST_EMBEDDING_RECEIPT_SCHEMA:
            raise PostEmbeddingDevelopmentError("post-embedding receipt schema differs")
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [row.to_dict() for row in self.artifacts],
            "index_config_sha256": self.index_config_sha256,
            "config_sha256": self.config_sha256,
            "development_family_count": self.development_family_count,
            "embedding_bindings_sha256": self.embedding_bindings_sha256,
            "embedding_suite_receipt_sha256": self.embedding_suite_receipt_sha256,
            "execution_config_sha256": self.execution_config_sha256,
            "execution_receipt_sha256": self.execution_receipt_sha256,
            "freeze_config_sha256": self.freeze_config_sha256,
            "freeze_receipt_sha256": self.freeze_receipt_sha256,
            "freeze_tree_sha256": self.freeze_tree_sha256,
            "full_staged_inventory_sha256": self.full_staged_inventory_sha256,
            "joint_power_config_sha256": self.joint_power_config_sha256,
            "joint_power_invocation_sha256": self.joint_power_invocation_sha256,
            "joint_power_report_sha256": self.joint_power_report_sha256,
            "joint_power_report_tree_sha256": self.joint_power_report_tree_sha256,
            "development_materialization_receipt_sha256": (
                self.development_materialization_receipt_sha256
            ),
            "paired_action_row_count": self.paired_action_row_count,
            "paired_trial_count": self.paired_trial_count,
            "partition_audit_sha256": self.partition_audit_sha256,
            "partition_audit_file_sha256": self.partition_audit_file_sha256,
            "design_seed_sha256": self.design_seed_sha256,
            "schema_version": self.schema_version,
            "selected_families_per_corpus": self.selected_families_per_corpus,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "strata": [row.to_dict() for row in self.strata],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> PostEmbeddingDevelopmentReceipt:
        row = _closed(value, _RECEIPT_FIELDS, label="post-embedding receipt")
        strata = row["strata"]
        artifacts = row["artifacts"]
        if not isinstance(strata, list) or not isinstance(artifacts, list):
            raise PostEmbeddingDevelopmentError("receipt arrays differ")
        values = dict(row)
        values["strata"] = tuple(PostEmbeddingStratumReceipt.from_dict(item) for item in strata)
        values["artifacts"] = tuple(PostEmbeddingArtifactPin.from_dict(item) for item in artifacts)
        return cls(**values)  # type: ignore[arg-type]


def _artifact_contract() -> dict[str, tuple[str, Literal["file", "directory"]]]:
    return {
        OPERATOR_CONFIG_FILENAME: ("operator-config", "file"),
        SELECTION_FILENAME: ("cohort-selection", "file"),
        EMBEDDING_BINDINGS_FILENAME: ("embedding-bindings", "file"),
        MATERIALIZATION_DIRECTORY: ("cohort-materialization", "directory"),
        POLICY_DIRECTORY: ("policy-packages", "directory"),
        INDEX_DIRECTORY: ("authorized-index-packages", "directory"),
        EXECUTION_INPUT_FILENAME: ("paired-execution-config", "file"),
        EXECUTION_DIRECTORY: ("paired-execution", "directory"),
        FREEZE_CONFIG_FILENAME: ("development-freeze-config", "file"),
        FREEZE_DIRECTORY: ("development-freeze", "directory"),
        JOINT_POWER_INVOCATION_FILENAME: ("joint-power-invocation", "file"),
        f"{ANALYSIS_DIRECTORY}/{JOINT_POWER_DIRECTORY}": (
            "joint-power-design",
            "directory",
        ),
    }


def _pin_artifact(root: Path, relative: str) -> PostEmbeddingArtifactPin:
    stage_id, kind = _artifact_contract()[relative]
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        if kind == "file":
            digest = digest_regular_file(path, label=stage_id)
            size = path.stat().st_size
            return PostEmbeddingArtifactPin(stage_id, relative, kind, digest, 1, 0, size)
        tree = digest_directory_tree(path)
    except (ArtifactIntegrityError, OSError) as exc:
        raise PostEmbeddingDevelopmentError(f"cannot pin {stage_id}: {exc}") from exc
    return PostEmbeddingArtifactPin(
        stage_id,
        relative,
        kind,
        tree.sha256,
        tree.file_count,
        tree.directory_count,
        tree.byte_count,
    )


def _ensure_output_root(
    config: PostEmbeddingDevelopmentConfig,
    *,
    resume: bool,
) -> None:
    root = config.output_root
    if resume:
        _require_real_directory(root, label="operator output root", private=True)
        encoded = _read_control(root / OPERATOR_CONFIG_FILENAME, label="operator config copy")
        if encoded != config.canonical_file_bytes():
            raise PostEmbeddingDevelopmentError("operator config copy differs")
        return
    if os.path.lexists(root):
        raise PostEmbeddingDevelopmentError("operator output root already exists")
    _require_real_directory(root.parent, label="operator output parent", private=True)
    try:
        root.mkdir(mode=0o700)
    except OSError as exc:
        raise PostEmbeddingDevelopmentError(f"cannot create operator output root: {exc}") from exc
    _write_exclusive(
        root / OPERATOR_CONFIG_FILENAME,
        config.canonical_file_bytes(),
        label="operator config copy",
    )


def _assert_known_tree(root: Path) -> None:
    try:
        tree = digest_directory_tree(root)
    except ArtifactIntegrityError as exc:
        raise PostEmbeddingDevelopmentError(
            f"operator tree is not regular and closed: {exc}"
        ) from exc
    top = {PurePosixPath(path).parts[0] for path in tree.entries}
    unexpected = top - _KNOWN_TOP_LEVEL
    if unexpected:
        raise PostEmbeddingDevelopmentError(
            f"operator output has unexpected top-level entries: {sorted(unexpected)}"
        )


def _ensure_private_directory(path: Path) -> None:
    if path.exists():
        _require_real_directory(path, label=str(path), private=True)
        return
    _ensure_private_directory(path.parent)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        _require_real_directory(path, label=str(path), private=True)


def _exclusive_publish_directory(work: Path, output: Path) -> None:
    """Publish one directory with an operating-system no-replace primitive."""

    if os.path.lexists(output):
        raise PostEmbeddingDevelopmentError(f"publication target already exists: {output}")
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(work)
    destination = os.fsencode(output)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise PostEmbeddingDevelopmentError("exclusive directory rename is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-2, source, -2, destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise PostEmbeddingDevelopmentError("exclusive directory rename is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source, -100, destination, 0x00000001)
    else:
        raise PostEmbeddingDevelopmentError(
            f"exclusive directory rename is unsupported on {sys.platform!r}"
        )
    if result != 0:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise PostEmbeddingDevelopmentError(f"publication target already exists: {output}")
        raise PostEmbeddingDevelopmentError(f"cannot publish directory: {os.strerror(number)}")
    descriptor = os.open(
        output.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _expected_embedding_bindings(
    upstream: _AdmittedUpstream,
) -> tuple[DevelopmentEmbeddingBinding, ...]:
    suite_rows = {row.corpus_id: row for row in upstream.embedding_suite.corpora}
    return tuple(
        DevelopmentEmbeddingBinding(
            corpus=corpus,
            development_stage=stage,
            root=upstream.embedding_config.output_root / corpus,
            receipt_sha256=suite_rows[corpus].embedding_receipt_sha256,
        )
        for stage in DEVELOPMENT_STAGES
        for corpus in FIXED_CORPORA
    )


def _ensure_selection_and_materialization(
    config: PostEmbeddingDevelopmentConfig,
    upstream: _AdmittedUpstream,
    *,
    allow_writes: bool,
) -> tuple[str, str, str]:
    root = config.output_root
    selection_path = root / SELECTION_FILENAME
    bindings_path = root / EMBEDDING_BINDINGS_FILENAME
    materialization_root = root / MATERIALIZATION_DIRECTORY
    if os.path.lexists(selection_path):
        selection = load_development_cohort_selection(
            selection_path,
            expected_inventory_sha256=config.full_staged_inventory_sha256,
        )
        if selection.partition_audit_sha256 != upstream.partition_audit.artifact_sha256:
            raise PostEmbeddingDevelopmentError("selection names another partition audit")
    else:
        if not allow_writes:
            raise PostEmbeddingDevelopmentError("cohort selection is missing")
        if os.path.lexists(bindings_path) or os.path.lexists(materialization_root):
            raise PostEmbeddingDevelopmentError("a later cohort stage exists before selection")
        selection = select_development_cohort(
            config.full_staged_root,
            selection_path,
            staged_inventory_sha256=config.full_staged_inventory_sha256,
            partition_audit_path=config.partition_audit_path,
            partition_audit_sha256=upstream.partition_audit.artifact_sha256,
        )
    expected_bindings = _expected_embedding_bindings(upstream)
    binding_bytes = canonical_development_embedding_bindings_bytes(expected_bindings)
    if os.path.lexists(bindings_path):
        observed_bindings = load_development_embedding_bindings(bindings_path)
        if observed_bindings != tuple(
            sorted(expected_bindings, key=lambda row: (row.development_stage, row.corpus))
        ):
            raise PostEmbeddingDevelopmentError("embedding bindings differ from the suite")
        if _read_control(bindings_path, label="embedding bindings") != binding_bytes:
            raise PostEmbeddingDevelopmentError("embedding binding bytes differ")
    else:
        if not allow_writes:
            raise PostEmbeddingDevelopmentError("embedding bindings are missing")
        if os.path.lexists(materialization_root):
            raise PostEmbeddingDevelopmentError("materialization exists before embedding bindings")
        _write_exclusive(bindings_path, binding_bytes, label="embedding bindings")
    if os.path.lexists(materialization_root):
        materialization = verify_materialized_development_cohort(
            materialization_root,
            verify_label_payloads=True,
        )
    else:
        if not allow_writes:
            raise PostEmbeddingDevelopmentError("development materialization is missing")
        materialization = materialize_development_cohort(
            config.full_staged_root,
            selection_path,
            materialization_root,
            selection_receipt_sha256=selection.artifact_sha256,
            partition_audit_path=config.partition_audit_path,
            embedding_bindings=expected_bindings,
        )
    if (
        materialization.selection_receipt_sha256 != selection.artifact_sha256
        or materialization.partition_audit_sha256 != upstream.partition_audit.artifact_sha256
        or materialization.staged_inventory_sha256 != config.full_staged_inventory_sha256
        or materialization.embedding_bindings
        != tuple(sorted(expected_bindings, key=lambda row: (row.development_stage, row.corpus)))
    ):
        raise PostEmbeddingDevelopmentError("materialization receipt chain differs")
    return (
        selection.artifact_sha256,
        _sha256(binding_bytes),
        materialization.artifact_sha256,
    )


def _stratum_keys() -> tuple[tuple[str, str], ...]:
    return tuple((stage, corpus) for stage in DEVELOPMENT_STAGES for corpus in FIXED_CORPORA)


def _assert_package_prefix(parent: Path, *, label: str) -> None:
    if not os.path.lexists(parent):
        return
    _require_real_directory(parent, label=label, private=True)
    expected = [f"{stage}/{corpus}" for stage, corpus in _stratum_keys()]
    observed: set[str] = set()
    for stage_path in parent.iterdir():
        if not stage_path.is_dir() or stage_path.is_symlink():
            raise PostEmbeddingDevelopmentError(f"{label} contains a non-directory entry")
        for child in stage_path.iterdir():
            if not child.is_dir() or child.is_symlink():
                raise PostEmbeddingDevelopmentError(f"{label} contains a partial package")
            observed.add(f"{stage_path.name}/{child.name}")
    if observed != set(expected[: len(observed)]):
        raise PostEmbeddingDevelopmentError(f"{label} packages are not a canonical prefix")


def _ensure_policy_and_indexes(
    config: PostEmbeddingDevelopmentConfig,
    materialization_receipt_sha256: str,
    *,
    allow_writes: bool,
) -> tuple[tuple[PostEmbeddingStratumReceipt, ...], str]:
    root = config.output_root
    materialization_root = root / MATERIALIZATION_DIRECTORY
    materialization = verify_materialized_development_cohort(
        materialization_root,
        expected_receipt_sha256=materialization_receipt_sha256,
        verify_label_payloads=False,
    )
    bindings = {
        (row.development_stage, row.corpus): row for row in materialization.embedding_bindings
    }
    policy_parent = root / POLICY_DIRECTORY
    index_parent = root / INDEX_DIRECTORY
    _assert_package_prefix(policy_parent, label="policy package tree")
    _assert_package_prefix(index_parent, label="authorized index package tree")
    if not allow_writes and not os.path.lexists(policy_parent):
        raise PostEmbeddingDevelopmentError("policy package tree is missing")
    _ensure_private_directory(policy_parent)
    for stage, corpus in _stratum_keys():
        policy_root = policy_parent / stage / corpus
        plan = load_development_execution_plan(
            materialization_root / stage / corpus / "execution-plan.json"
        )
        policy_config = derive_production_policy_config(
            config.design_seed_sha256,
            corpus,
            SOURCE_STAGE[stage],
        )
        if os.path.lexists(policy_root):
            verification = verify_policy_intervention_package(
                policy_root,
                plan,
                policy_config,
            )
        else:
            if not allow_writes:
                raise PostEmbeddingDevelopmentError(
                    f"policy package is missing for {stage}:{corpus}"
                )
            _ensure_private_directory(policy_root.parent)
            verification = write_policy_intervention_package(
                plan,
                policy_config,
                policy_root,
            )
        if verification.root != policy_root:
            raise PostEmbeddingDevelopmentError("policy verifier returned another root")

    index_config, backend = production_authorized_index_components()
    if not allow_writes and not os.path.lexists(index_parent):
        raise PostEmbeddingDevelopmentError("authorized index package tree is missing")
    _ensure_private_directory(index_parent)
    rows: list[PostEmbeddingStratumReceipt] = []
    for stage, corpus in _stratum_keys():
        policy_root = policy_parent / stage / corpus
        index_root = index_parent / stage / corpus
        plan = load_development_execution_plan(
            materialization_root / stage / corpus / "execution-plan.json"
        )
        policy_config = derive_production_policy_config(
            config.design_seed_sha256,
            corpus,
            SOURCE_STAGE[stage],
        )
        policy = verify_policy_intervention_package(policy_root, plan, policy_config)
        binding = bindings[(stage, corpus)]
        if os.path.lexists(index_root):
            index = verify_authorized_index_store(
                index_root,
                embedding_store_root=binding.root,
                policy_intervention_root=policy_root,
                expected_embedding_receipt_sha256=binding.receipt_sha256,
                expected_policy_receipt_sha256=policy.receipt_sha256,
                backend=backend,
            )
        else:
            if not allow_writes:
                raise PostEmbeddingDevelopmentError(
                    f"authorized index package is missing for {stage}:{corpus}"
                )
            _ensure_private_directory(index_root.parent)
            index = build_authorized_index_store(
                binding.root,
                policy_root,
                index_root,
                expected_embedding_receipt_sha256=binding.receipt_sha256,
                expected_policy_receipt_sha256=policy.receipt_sha256,
                config=index_config,
                backend=backend,
            )
        rows.append(
            PostEmbeddingStratumReceipt(
                corpus=corpus,
                development_stage=stage,
                source_stage=SOURCE_STAGE[stage],
                embedding_receipt_sha256=binding.receipt_sha256,
                policy_config_sha256=policy_config.config_sha256,
                policy_intervention_receipt_sha256=policy.receipt_sha256,
                authorized_index_config_sha256=index_config.config_sha256,
                authorized_index_receipt_sha256=index.receipt_sha256,
            )
        )
    return tuple(rows), index_config.config_sha256


def _expected_execution_config(
    config: PostEmbeddingDevelopmentConfig,
    materialization_receipt_sha256: str,
    strata: Sequence[PostEmbeddingStratumReceipt],
) -> DevelopmentPairedExecutionConfig:
    root = config.output_root
    by_key = {(row.development_stage, row.corpus): row for row in strata}
    return DevelopmentPairedExecutionConfig(
        materialization_root=root / MATERIALIZATION_DIRECTORY,
        materialization_receipt_sha256=materialization_receipt_sha256,
        inputs=tuple(
            DevelopmentExecutionInput(
                corpus=corpus,
                stage=stage,
                policy_intervention_root=root / POLICY_DIRECTORY / stage / corpus,
                policy_intervention_receipt_sha256=(
                    by_key[(stage, corpus)].policy_intervention_receipt_sha256
                ),
                authorized_index_root=root / INDEX_DIRECTORY / stage / corpus,
                authorized_index_receipt_sha256=(
                    by_key[(stage, corpus)].authorized_index_receipt_sha256
                ),
            )
            for stage, corpus in _stratum_keys()
        ),
        output_root=root / EXECUTION_DIRECTORY,
    )


def _ensure_execution(
    config: PostEmbeddingDevelopmentConfig,
    materialization_receipt_sha256: str,
    strata: Sequence[PostEmbeddingStratumReceipt],
    *,
    allow_writes: bool,
):
    root = config.output_root
    expected = _expected_execution_config(config, materialization_receipt_sha256, strata)
    config_path = root / EXECUTION_INPUT_FILENAME
    if os.path.lexists(config_path):
        observed = load_development_paired_execution_config(config_path)
        if (
            observed != expected
            or _read_control(config_path, label="paired execution config")
            != expected.canonical_file_bytes()
        ):
            raise PostEmbeddingDevelopmentError("paired execution config differs")
    else:
        if not allow_writes:
            raise PostEmbeddingDevelopmentError("paired execution config is missing")
        if os.path.lexists(expected.output_root):
            raise PostEmbeddingDevelopmentError("paired execution exists before its config")
        _write_exclusive(
            config_path,
            expected.canonical_file_bytes(),
            label="paired execution config",
        )
    if os.path.lexists(expected.output_root):
        receipt = verify_development_paired_execution(expected.output_root)
    else:
        if not allow_writes:
            raise PostEmbeddingDevelopmentError("paired execution package is missing")
        receipt = run_development_paired_execution(config_path)
    if (
        receipt.config_sha256 != expected.config_sha256
        or receipt.materialization_receipt_sha256 != materialization_receipt_sha256
    ):
        raise PostEmbeddingDevelopmentError("paired execution receipt chain differs")
    family_count = sum(row.selected_family_count for row in receipt.strata)
    trial_count = sum(row.trial_count for row in receipt.strata)
    action_count = sum(
        artifact.record_count
        for row in receipt.strata
        for artifact in row.outputs
        if artifact.role == "paired-actions"
    )
    if (
        family_count != EXPECTED_DEVELOPMENT_FAMILIES
        or trial_count != EXPECTED_PAIRED_TRIALS
        or action_count != EXPECTED_ACTION_ROWS
    ):
        raise PostEmbeddingDevelopmentError("paired execution is not 4,125 trials/16,500 rows")
    return expected, receipt


def _ensure_freeze(
    config: PostEmbeddingDevelopmentConfig,
    execution_receipt_sha256: str,
    *,
    allow_writes: bool,
):
    root = config.output_root
    execution_root = root / EXECUTION_DIRECTORY
    verify_development_paired_execution(
        execution_root,
        expected_receipt_sha256=execution_receipt_sha256,
    )
    config_path = root / FREEZE_CONFIG_FILENAME
    freeze_root = root / FREEZE_DIRECTORY
    expected = build_development_freeze_config(execution_root, output_root=freeze_root)
    if os.path.lexists(config_path):
        observed = load_development_freeze_config(config_path)
        if observed != expected or _read_control(
            config_path, label="development freeze config"
        ) != canonical_development_freeze_config_bytes(expected):
            raise PostEmbeddingDevelopmentError("development freeze config differs")
    else:
        if not allow_writes:
            raise PostEmbeddingDevelopmentError("development freeze config is missing")
        if os.path.lexists(freeze_root):
            raise PostEmbeddingDevelopmentError("development freeze exists before its config")
        written = write_bound_development_freeze_config(
            execution_root,
            config_path,
            output_root=freeze_root,
        )
        if written != expected:
            raise PostEmbeddingDevelopmentError("derived development freeze config changed")
    if os.path.lexists(freeze_root):
        receipt = verify_development_freeze(freeze_root)
    else:
        if not allow_writes:
            raise PostEmbeddingDevelopmentError("development freeze package is missing")
        compile_development_freeze(expected)
        receipt = verify_development_freeze(freeze_root)
    try:
        config_sha = digest_regular_file(config_path, label="development freeze config")
        receipt_sha = digest_regular_file(
            freeze_root / "freeze-receipt.json",
            label="development freeze receipt",
        )
        tree_sha = digest_directory_tree(freeze_root).sha256
    except ArtifactIntegrityError as exc:
        raise PostEmbeddingDevelopmentError(f"cannot pin development freeze: {exc}") from exc
    if not receipt:
        raise PostEmbeddingDevelopmentError("development freeze receipt is empty")
    return config_sha, receipt_sha, tree_sha


def _joint_power_source(freeze_root: Path):
    config_bytes = _read_control(
        freeze_root / "joint-power-config.json",
        label="joint-power config",
    )
    expected_bytes = _read_control(
        freeze_root / "joint-power-expected-panel.json",
        label="expected joint-power panel",
    )
    conservative_bytes = _read_control(
        freeze_root / "joint-power-conservative-panel.json",
        label="conservative joint-power panel",
    )
    try:
        power_config = load_joint_power_config(config_bytes)
        panels = (
            load_development_panel(expected_bytes),
            load_development_panel(conservative_bytes),
        )
    except JointPowerDesignError as exc:
        raise PostEmbeddingDevelopmentError(
            f"development freeze power input is invalid: {exc}"
        ) from exc
    if (
        power_config.test_mode
        or power_config.n_simulations != 5_000
        or power_config.bound_calibration_simulations != 5_000
    ):
        raise PostEmbeddingDevelopmentError(
            "joint-power config must be production mode with exactly 5000+5000 simulations"
        )
    panel_by_id = {panel.scenario_id: panel for panel in panels}
    declared = {row.scenario_id: row.panel_sha256 for row in power_config.effect_scenarios}
    if declared != {key: panel.sha256 for key, panel in panel_by_id.items()}:
        raise PostEmbeddingDevelopmentError("joint-power panels differ from config pins")
    return power_config, tuple(panel_by_id[key] for key in sorted(panel_by_id))


def _invocation_payload(freeze_tree_sha256: str, power_config, panels) -> bytes:
    return _canonical_bytes(
        {
            "authorized_invocation_count": 1,
            "freeze_tree_sha256": freeze_tree_sha256,
            "joint_power_config_sha256": power_config.sha256,
            "panel_sha256s": {panel.scenario_id: panel.sha256 for panel in panels},
            "schema_version": JOINT_POWER_INVOCATION_SCHEMA,
        }
    )


def _verify_joint_power_bundle(
    bundle: Path,
    *,
    freeze_tree_sha256: str,
    invocation_path: Path,
    reproduce_exact: bool = True,
):
    freeze_root = bundle.parent.parent / FREEZE_DIRECTORY
    power_config, source_panels = _joint_power_source(freeze_root)
    expected_invocation = _invocation_payload(freeze_tree_sha256, power_config, source_panels)
    invocation_bytes = _read_control(invocation_path, label="joint-power invocation")
    if invocation_bytes != expected_invocation:
        raise PostEmbeddingDevelopmentError("joint-power invocation marker differs")
    config_bytes = _read_control(bundle / "config.json", label="published joint-power config")
    report_bytes = _read_control(bundle / "report.json", label="published joint-power report")
    audit_bytes = _read_control(
        bundle / JOINT_POWER_SELECTION_AUDIT_FILENAME,
        label="published joint-power selection audit",
    )
    if config_bytes != canonical_joint_power_config_bytes(power_config):
        raise PostEmbeddingDevelopmentError("published joint-power config differs from freeze")
    observed_panels = []
    expected_entries = {
        "config.json",
        "report.json",
        JOINT_POWER_SELECTION_AUDIT_FILENAME,
        "panels",
    }
    for panel in source_panels:
        relative = f"panels/{panel.sha256}.json"
        expected_entries.add(relative)
        encoded = _read_control(bundle / relative, label=f"published panel {panel.scenario_id}")
        if encoded != canonical_development_panel_bytes(panel):
            raise PostEmbeddingDevelopmentError("published joint-power panel differs from freeze")
        observed_panels.append(load_development_panel(encoded))
    try:
        tree = digest_directory_tree(bundle)
        report = load_joint_power_report(report_bytes)
        selection_audit = load_joint_power_selection_audit(audit_bytes)
    except (ArtifactIntegrityError, JointPowerDesignError) as exc:
        raise PostEmbeddingDevelopmentError(
            f"joint-power bundle verification failed: {exc}"
        ) from exc
    if set(tree.entries) != expected_entries:
        raise PostEmbeddingDevelopmentError("joint-power bundle membership differs")
    if (
        report.config_sha256 != power_config.sha256
        or dict(report.panel_sha256s)
        != {panel.scenario_id: panel.sha256 for panel in observed_panels}
        or report.test_mode
        or not report.freeze_ready
        or report.selected_families_per_corpus is None
    ):
        raise PostEmbeddingDevelopmentError("joint-power report is not freeze-ready and bound")
    try:
        recomputed_audit = (
            verify_joint_power_selection_audit(
                power_config,
                tuple(observed_panels),
                selection_audit,
            )
            if reproduce_exact
            else selection_audit
        )
        recomputed = run_joint_power_design(
            power_config,
            tuple(observed_panels),
            selection_audit=recomputed_audit,
        )
    except JointPowerDesignError as exc:
        qualifier = "exact " if reproduce_exact else ""
        raise PostEmbeddingDevelopmentError(
            f"joint-power {qualifier}selection audit does not reproduce: {exc}"
        ) from exc
    if canonical_joint_power_report_bytes(recomputed) != report_bytes:
        raise PostEmbeddingDevelopmentError(
            "joint-power report does not reproduce from the registered config and panels"
        )
    return power_config, tuple(observed_panels), report, tree.sha256


def _reuse_fresh_joint_power_verification(
    verification: _FreshJointPowerVerification,
    *,
    bundle: Path,
    freeze_tree_sha256: str,
    invocation_path: Path,
):
    """Reuse one same-process generation/readback only while every bound byte is unchanged."""

    if verification.bundle_root != bundle or verification.freeze_tree_sha256 != freeze_tree_sha256:
        raise PostEmbeddingDevelopmentError(
            "fresh joint-power verification belongs to another immutable boundary"
        )
    invocation_bytes = _read_control(invocation_path, label="joint-power invocation")
    if invocation_bytes != verification.invocation_bytes:
        raise PostEmbeddingDevelopmentError(
            "joint-power invocation changed after fresh verification"
        )
    expected_invocation = _invocation_payload(
        freeze_tree_sha256,
        verification.power_config,
        verification.panels,
    )
    if invocation_bytes != expected_invocation:
        raise PostEmbeddingDevelopmentError(
            "joint-power invocation no longer matches the verified bundle"
        )
    try:
        tree = digest_directory_tree(bundle)
    except ArtifactIntegrityError as exc:
        raise PostEmbeddingDevelopmentError(
            f"cannot recheck freshly verified joint-power tree: {exc}"
        ) from exc
    if tree.sha256 != verification.tree_sha256:
        raise PostEmbeddingDevelopmentError(
            "joint-power bundle changed after fresh generation or exact verification"
        )
    return (
        verification.power_config,
        verification.panels,
        verification.report,
        verification.tree_sha256,
    )


def _ensure_joint_power(config: PostEmbeddingDevelopmentConfig, freeze_tree_sha256: str):
    root = config.output_root
    freeze_root = root / FREEZE_DIRECTORY
    power_config, panels = _joint_power_source(freeze_root)
    invocation_path = root / JOINT_POWER_INVOCATION_FILENAME
    analysis_parent = root / ANALYSIS_DIRECTORY
    bundle = analysis_parent / JOINT_POWER_DIRECTORY
    if os.path.lexists(bundle):
        if not os.path.lexists(invocation_path):
            raise PostEmbeddingDevelopmentError("joint-power bundle lacks its invocation marker")
        return _verify_joint_power_bundle(
            bundle,
            freeze_tree_sha256=freeze_tree_sha256,
            invocation_path=invocation_path,
            reproduce_exact=False,
        )
    if os.path.lexists(invocation_path):
        raise PostEmbeddingDevelopmentError(
            "joint-power invocation began without a complete package; retry is forbidden"
        )
    if os.path.lexists(analysis_parent):
        _require_real_directory(analysis_parent, label="analysis parent", private=True)
        if any(analysis_parent.iterdir()):
            raise PostEmbeddingDevelopmentError("analysis parent is not empty before publication")
    else:
        _ensure_private_directory(analysis_parent)
    invocation_bytes = _invocation_payload(freeze_tree_sha256, power_config, panels)
    _write_exclusive(invocation_path, invocation_bytes, label="joint-power invocation marker")

    # This is the sole production invocation.  A marker without a final bundle
    # is terminal because a process failure cannot prove whether simulation ran.
    selection_audit = run_joint_power_selection_audit(power_config, panels)
    report = run_joint_power_design(
        power_config,
        panels,
        selection_audit=selection_audit,
    )
    if report.test_mode or not report.freeze_ready or report.selected_families_per_corpus is None:
        raise PostEmbeddingDevelopmentError("joint-power run did not yield a freeze-ready design")
    work = Path(tempfile.mkdtemp(prefix=f".{JOINT_POWER_DIRECTORY}.work-", dir=analysis_parent))
    try:
        (work / "panels").mkdir(mode=0o700)
        _write_exclusive(
            work / "config.json",
            canonical_joint_power_config_bytes(power_config),
            label="joint-power config",
        )
        _write_exclusive(
            work / "report.json",
            canonical_joint_power_report_bytes(report),
            label="joint-power report",
        )
        _write_exclusive(
            work / JOINT_POWER_SELECTION_AUDIT_FILENAME,
            canonical_joint_power_selection_audit_bytes(selection_audit),
            label="joint-power selection audit",
        )
        for panel in panels:
            _write_exclusive(
                work / "panels" / f"{panel.sha256}.json",
                canonical_development_panel_bytes(panel),
                label=f"joint-power panel {panel.scenario_id}",
            )
        _exclusive_publish_directory(work, bundle)
        work = Path()
    finally:
        if work != Path() and work.exists():
            shutil.rmtree(work)
    if _read_control(
        bundle / JOINT_POWER_SELECTION_AUDIT_FILENAME,
        label="published joint-power selection audit",
    ) != canonical_joint_power_selection_audit_bytes(selection_audit) or _read_control(
        bundle / "report.json", label="published joint-power report"
    ) != canonical_joint_power_report_bytes(report):
        raise PostEmbeddingDevelopmentError(
            "published joint-power audit or report differs from the in-memory result"
        )
    return _verify_joint_power_bundle(
        bundle,
        freeze_tree_sha256=freeze_tree_sha256,
        invocation_path=invocation_path,
        reproduce_exact=False,
    )


def _build_receipt(
    config: PostEmbeddingDevelopmentConfig,
    upstream: _AdmittedUpstream,
    *,
    embedding_bindings_sha256: str,
    selection_receipt_sha256: str,
    materialization_receipt_sha256: str,
    strata: Sequence[PostEmbeddingStratumReceipt],
    index_config_sha256: str,
    execution_config,
    execution_receipt,
    freeze_config_sha256: str,
    freeze_receipt_sha256: str,
    freeze_tree_sha256: str,
    power_config,
    power_report,
    joint_tree_sha256: str,
) -> PostEmbeddingDevelopmentReceipt:
    root = config.output_root
    artifacts = tuple(_pin_artifact(root, path) for path in _artifact_contract())
    invocation_sha = next(
        row.sha256 for row in artifacts if row.path == JOINT_POWER_INVOCATION_FILENAME
    )
    family_count = sum(row.selected_family_count for row in execution_receipt.strata)
    trial_count = sum(row.trial_count for row in execution_receipt.strata)
    action_count = sum(
        artifact.record_count
        for row in execution_receipt.strata
        for artifact in row.outputs
        if artifact.role == "paired-actions"
    )
    return PostEmbeddingDevelopmentReceipt(
        config_sha256=config.file_sha256,
        full_staged_inventory_sha256=config.full_staged_inventory_sha256,
        partition_audit_file_sha256=config.partition_audit_file_sha256,
        partition_audit_sha256=upstream.partition_audit.artifact_sha256,
        embedding_suite_receipt_sha256=upstream.embedding_suite.receipt_sha256,
        embedding_bindings_sha256=embedding_bindings_sha256,
        selection_receipt_sha256=selection_receipt_sha256,
        development_materialization_receipt_sha256=materialization_receipt_sha256,
        design_seed_sha256=config.design_seed_sha256,
        index_config_sha256=index_config_sha256,
        execution_config_sha256=execution_config.config_sha256,
        execution_receipt_sha256=execution_receipt.artifact_sha256,
        freeze_config_sha256=freeze_config_sha256,
        freeze_receipt_sha256=freeze_receipt_sha256,
        freeze_tree_sha256=freeze_tree_sha256,
        joint_power_invocation_sha256=invocation_sha,
        joint_power_config_sha256=power_config.sha256,
        joint_power_report_sha256=power_report.sha256,
        joint_power_report_tree_sha256=joint_tree_sha256,
        selected_families_per_corpus=power_report.selected_families_per_corpus,
        development_family_count=family_count,
        paired_trial_count=trial_count,
        paired_action_row_count=action_count,
        strata=tuple(strata),
        artifacts=artifacts,
    )


def load_post_embedding_development_receipt(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> PostEmbeddingDevelopmentReceipt:
    receipt_path = _canonical_absolute_path(str(path), label="post-embedding receipt path")
    encoded = _read_control(receipt_path, label="post-embedding receipt")
    receipt = PostEmbeddingDevelopmentReceipt.from_dict(
        _decode(encoded, label="post-embedding receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise PostEmbeddingDevelopmentError("post-embedding receipt is not canonical")
    if expected_sha256 is not None and receipt.artifact_sha256 != _require_sha256(
        "expected receipt SHA-256", expected_sha256
    ):
        raise PostEmbeddingDevelopmentError("post-embedding receipt differs from its pin")
    return receipt


def _verify_post_embedding_development_config(
    config: PostEmbeddingDevelopmentConfig,
    *,
    expected_receipt_sha256: str | None = None,
    fresh_joint_power: _FreshJointPowerVerification | None = None,
    admitted_upstream: _AdmittedUpstream | None = None,
) -> PostEmbeddingDevelopmentReceipt:
    root = config.output_root
    _require_real_directory(root, label="operator output root", private=True)
    _assert_known_tree(root)
    observed_config = _read_control(root / OPERATOR_CONFIG_FILENAME, label="operator config copy")
    if observed_config != config.canonical_file_bytes():
        raise PostEmbeddingDevelopmentError("operator config copy differs")
    upstream = admitted_upstream if admitted_upstream is not None else _admit_upstream(config)
    selection_sha, bindings_sha, materialization_sha = _ensure_selection_and_materialization(
        config,
        upstream,
        allow_writes=False,
    )
    strata, index_config_sha = _ensure_policy_and_indexes(
        config,
        materialization_sha,
        allow_writes=False,
    )
    execution_config, execution_receipt = _ensure_execution(
        config,
        materialization_sha,
        strata,
        allow_writes=False,
    )
    freeze_config_sha, freeze_receipt_sha, freeze_tree_sha = _ensure_freeze(
        config,
        execution_receipt.artifact_sha256,
        allow_writes=False,
    )
    bundle = root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY
    if not os.path.lexists(bundle) or not os.path.lexists(root / JOINT_POWER_INVOCATION_FILENAME):
        raise PostEmbeddingDevelopmentError("joint-power stage is incomplete")
    if fresh_joint_power is None:
        power_config, _panels, power_report, joint_tree_sha = _verify_joint_power_bundle(
            bundle,
            freeze_tree_sha256=freeze_tree_sha,
            invocation_path=root / JOINT_POWER_INVOCATION_FILENAME,
        )
    else:
        power_config, _panels, power_report, joint_tree_sha = _reuse_fresh_joint_power_verification(
            fresh_joint_power,
            bundle=bundle,
            freeze_tree_sha256=freeze_tree_sha,
            invocation_path=root / JOINT_POWER_INVOCATION_FILENAME,
        )
    reproduced = _build_receipt(
        config,
        upstream,
        embedding_bindings_sha256=bindings_sha,
        selection_receipt_sha256=selection_sha,
        materialization_receipt_sha256=materialization_sha,
        strata=strata,
        index_config_sha256=index_config_sha,
        execution_config=execution_config,
        execution_receipt=execution_receipt,
        freeze_config_sha256=freeze_config_sha,
        freeze_receipt_sha256=freeze_receipt_sha,
        freeze_tree_sha256=freeze_tree_sha,
        power_config=power_config,
        power_report=power_report,
        joint_tree_sha256=joint_tree_sha,
    )
    observed = load_post_embedding_development_receipt(
        root / RECEIPT_FILENAME,
        expected_sha256=expected_receipt_sha256,
    )
    if observed != reproduced:
        raise PostEmbeddingDevelopmentError("post-embedding receipt does not reproduce")
    observed_top = {child.name for child in root.iterdir()}
    if observed_top != _KNOWN_TOP_LEVEL:
        raise PostEmbeddingDevelopmentError("final operator membership differs")
    return observed


def verify_post_embedding_development(
    root: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
) -> PostEmbeddingDevelopmentReceipt:
    """Reverify the terminal package, including a fresh joint-power recomputation."""

    package = _canonical_absolute_path(str(root), label="post-embedding operator root")
    _require_real_directory(package, label="operator output root", private=True)
    encoded = _read_control(package / OPERATOR_CONFIG_FILENAME, label="operator config copy")
    config = PostEmbeddingDevelopmentConfig.from_dict(
        _decode(encoded, label="operator config copy")
    )
    if encoded != config.canonical_file_bytes():
        raise PostEmbeddingDevelopmentError("operator config copy is not canonical")
    if config.output_root != package:
        raise PostEmbeddingDevelopmentError("operator config names another output root")
    return _verify_post_embedding_development_config(
        config,
        expected_receipt_sha256=expected_receipt_sha256,
    )


def admit_frozen_post_embedding_development(
    root: str | Path,
    *,
    expected_receipt_sha256: str,
    production_embedding_config_path: str | Path,
    embedding_config: ProductionEmbeddingConfig,
    embedding_suite: ProductionEmbeddingSuiteReceipt,
    partition_audit_path: str | Path,
    partition_audit: ScalableQueryPartitionAuditReceipt,
) -> PostEmbeddingDevelopmentReceipt:
    """Verify a frozen operator package without reopening the full staged tree.

    The caller supplies the typed embedding suite and label-free partition
    audit admitted in the same downstream operation. The terminal operator
    package is then replayed read-only against those exact objects. Raw staged
    data, including the sealed-label custody subtree, is neither required nor
    opened.
    """

    package = _canonical_absolute_path(str(root), label="frozen post-embedding operator root")
    _require_real_directory(package, label="frozen operator root", private=True)
    _require_read_only_filesystem(package, label="frozen operator root")
    if not isinstance(embedding_config, ProductionEmbeddingConfig):
        raise PostEmbeddingDevelopmentError("frozen embedding config must be typed")
    if not isinstance(embedding_suite, ProductionEmbeddingSuiteReceipt):
        raise PostEmbeddingDevelopmentError("frozen embedding suite must be typed")
    if not isinstance(partition_audit, ScalableQueryPartitionAuditReceipt):
        raise PostEmbeddingDevelopmentError("frozen partition audit must be typed")
    embedding_path = _canonical_absolute_path(
        str(production_embedding_config_path),
        label="frozen production embedding config path",
    )
    audit_path = _canonical_absolute_path(
        str(partition_audit_path),
        label="frozen partition audit path",
    )
    encoded = _read_control(package / OPERATOR_CONFIG_FILENAME, label="operator config copy")
    config = PostEmbeddingDevelopmentConfig.from_dict(
        _decode(encoded, label="operator config copy")
    )
    if encoded != config.canonical_file_bytes():
        raise PostEmbeddingDevelopmentError("operator config copy is not canonical")
    if config.output_root != package:
        raise PostEmbeddingDevelopmentError("operator config names another output root")
    mismatches = []
    expected = {
        "production_embedding_config_path": embedding_path,
        "production_embedding_config_sha256": embedding_config.file_sha256,
        "full_staged_inventory_sha256": embedding_config.online_inventory_sha256,
        "partition_audit_path": audit_path,
        "partition_audit_file_sha256": partition_audit.artifact_sha256,
    }
    mismatches.extend(name for name, value in expected.items() if getattr(config, name) != value)
    if (
        embedding_suite.production_config_sha256 != embedding_config.file_sha256
        or embedding_suite.online_inventory_sha256 != config.full_staged_inventory_sha256
        or partition_audit.staged_inventory_sha256 != config.full_staged_inventory_sha256
    ):
        mismatches.append("upstream cohort")
    if mismatches:
        raise PostEmbeddingDevelopmentError(
            "frozen operator upstream differs at: " + ", ".join(sorted(set(mismatches)))
        )
    return _verify_post_embedding_development_config(
        config,
        expected_receipt_sha256=expected_receipt_sha256,
        admitted_upstream=_AdmittedUpstream(
            embedding_config=embedding_config,
            embedding_suite=embedding_suite,
            partition_audit=partition_audit,
        ),
    )


def _execute_post_embedding_development(
    config: PostEmbeddingDevelopmentConfig,
    *,
    resume: bool,
) -> PostEmbeddingDevelopmentReceipt:
    if not isinstance(config, PostEmbeddingDevelopmentConfig):
        raise PostEmbeddingDevelopmentError("config must be PostEmbeddingDevelopmentConfig")
    _ensure_output_root(config, resume=resume)
    _assert_known_tree(config.output_root)
    if os.path.lexists(config.output_root / RECEIPT_FILENAME):
        if resume:
            return verify_post_embedding_development(config.output_root)
        raise PostEmbeddingDevelopmentError("completed operator output cannot be rerun")
    upstream = _admit_upstream(config)
    selection_sha, bindings_sha, materialization_sha = _ensure_selection_and_materialization(
        config,
        upstream,
        allow_writes=True,
    )
    strata, index_config_sha = _ensure_policy_and_indexes(
        config,
        materialization_sha,
        allow_writes=True,
    )
    execution_config, execution_receipt = _ensure_execution(
        config,
        materialization_sha,
        strata,
        allow_writes=True,
    )
    freeze_config_sha, freeze_receipt_sha, freeze_tree_sha = _ensure_freeze(
        config,
        execution_receipt.artifact_sha256,
        allow_writes=True,
    )
    ensured_power = _ensure_joint_power(
        config,
        freeze_tree_sha,
    )
    joint_bundle = config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY
    joint_invocation = config.output_root / JOINT_POWER_INVOCATION_FILENAME
    if resume:
        power_config, panels, power_report, joint_tree_sha = _verify_joint_power_bundle(
            joint_bundle,
            freeze_tree_sha256=freeze_tree_sha,
            invocation_path=joint_invocation,
        )
    else:
        power_config, panels, power_report, joint_tree_sha = ensured_power
    fresh_joint_power = _FreshJointPowerVerification(
        bundle_root=joint_bundle,
        freeze_tree_sha256=freeze_tree_sha,
        invocation_bytes=_read_control(
            joint_invocation,
            label="joint-power invocation",
        ),
        power_config=power_config,
        panels=panels,
        report=power_report,
        tree_sha256=joint_tree_sha,
    )
    receipt = _build_receipt(
        config,
        upstream,
        embedding_bindings_sha256=bindings_sha,
        selection_receipt_sha256=selection_sha,
        materialization_receipt_sha256=materialization_sha,
        strata=strata,
        index_config_sha256=index_config_sha,
        execution_config=execution_config,
        execution_receipt=execution_receipt,
        freeze_config_sha256=freeze_config_sha,
        freeze_receipt_sha256=freeze_receipt_sha,
        freeze_tree_sha256=freeze_tree_sha,
        power_config=power_config,
        power_report=power_report,
        joint_tree_sha256=joint_tree_sha,
    )
    _write_exclusive(
        config.output_root / RECEIPT_FILENAME,
        receipt.canonical_file_bytes(),
        label="post-embedding receipt",
    )
    return _verify_post_embedding_development_config(
        config,
        expected_receipt_sha256=receipt.artifact_sha256,
        fresh_joint_power=fresh_joint_power,
    )


def run_post_embedding_development(
    config: PostEmbeddingDevelopmentConfig,
) -> PostEmbeddingDevelopmentReceipt:
    """Start a new post-embedding chain; the output root must not exist."""

    return _execute_post_embedding_development(config, resume=False)


def resume_post_embedding_development(
    config: PostEmbeddingDevelopmentConfig,
) -> PostEmbeddingDevelopmentReceipt:
    """Continue only after reverifying every complete immutable boundary."""

    return _execute_post_embedding_development(config, resume=True)


def post_embedding_development_status(
    config: PostEmbeddingDevelopmentConfig,
) -> Mapping[str, object]:
    """Report filesystem state without opening development label payloads."""

    if not os.path.lexists(config.output_root):
        return {
            "completed": False,
            "joint_power_interrupted": False,
            "output_exists": False,
            "present": [],
            "schema_version": POST_EMBEDDING_CLI_RESULT_SCHEMA,
        }
    _require_real_directory(config.output_root, label="operator output root", private=True)
    _assert_known_tree(config.output_root)
    present = sorted(child.name for child in config.output_root.iterdir())
    marker = JOINT_POWER_INVOCATION_FILENAME in present
    bundle = (config.output_root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY).is_dir()
    return {
        "completed": RECEIPT_FILENAME in present,
        "joint_power_interrupted": marker and not bundle,
        "output_exists": True,
        "present": present,
        "schema_version": POST_EMBEDDING_CLI_RESULT_SCHEMA,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-post-embedding-development",
        description="Build or verify the development chain after paired embeddings.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write-config")
    write.add_argument("--production-embedding-config", required=True)
    write.add_argument("--production-embedding-config-sha256", required=True)
    write.add_argument("--full-staged-root", required=True)
    write.add_argument("--full-staged-inventory-sha256", required=True)
    write.add_argument("--partition-audit", required=True)
    write.add_argument("--partition-audit-file-sha256", required=True)
    write.add_argument("--design-seed-sha256", required=True)
    write.add_argument("--output-root", required=True)
    write.add_argument("--output", required=True)
    for name in ("run", "resume", "verify", "status"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--config-sha256", required=True)
    commands.choices["verify"].add_argument("--receipt-sha256")
    return parser


def _write_result(value: Mapping[str, object]) -> None:
    sys_stdout = __import__("sys").stdout.buffer
    sys_stdout.write(_canonical_bytes(dict(value)))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write-config":
            config = write_post_embedding_development_config(
                production_embedding_config_path=args.production_embedding_config,
                production_embedding_config_sha256=(args.production_embedding_config_sha256),
                full_staged_root=args.full_staged_root,
                full_staged_inventory_sha256=args.full_staged_inventory_sha256,
                partition_audit_path=args.partition_audit,
                partition_audit_file_sha256=args.partition_audit_file_sha256,
                design_seed_sha256=args.design_seed_sha256,
                output_root=args.output_root,
                destination=args.output,
            )
            result = {
                "command": args.command,
                "config_sha256": config.file_sha256,
                "schema_version": POST_EMBEDDING_CLI_RESULT_SCHEMA,
            }
        else:
            config = load_post_embedding_development_config(
                args.config,
                expected_sha256=args.config_sha256,
            )
            if args.command == "run":
                receipt = run_post_embedding_development(config)
                result = {
                    "command": args.command,
                    "receipt_sha256": receipt.artifact_sha256,
                    "schema_version": POST_EMBEDDING_CLI_RESULT_SCHEMA,
                }
            elif args.command == "resume":
                receipt = resume_post_embedding_development(config)
                result = {
                    "command": args.command,
                    "receipt_sha256": receipt.artifact_sha256,
                    "schema_version": POST_EMBEDDING_CLI_RESULT_SCHEMA,
                }
            elif args.command == "verify":
                receipt = verify_post_embedding_development(
                    config.output_root,
                    expected_receipt_sha256=args.receipt_sha256,
                )
                result = {
                    "command": args.command,
                    "receipt_sha256": receipt.artifact_sha256,
                    "schema_version": POST_EMBEDDING_CLI_RESULT_SCHEMA,
                }
            else:
                result = {"command": args.command, **post_embedding_development_status(config)}
        _write_result(result)
        return 0
    except (
        AuthorizedIndexStoreError,
        DevelopmentCohortError,
        DevelopmentExecutionError,
        DevelopmentFreezeError,
        JointPowerDesignError,
        PolicyInterventionError,
        PostEmbeddingDevelopmentError,
        ProductionArtifactFactoryError,
        ProductionEmbeddingBuildError,
        ScalablePartitionAuditError,
    ) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
