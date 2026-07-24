"""Typed three-stage wrappers for policy and authorized-index artifacts.

The study manifest pins one directory per corpus for each artifact role.  This
module gives those directories a closed internal contract: exactly one
``fit``, ``calibration``, and ``sealed`` child plus one canonical bundle
receipt.  The receipt does not replace the outer directory-tree digest.  It
records the typed identities that the freeze compiler must reproduce before it
accepts that digest.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .authorized_index_store import (
    CONFIG_FILENAME as INDEX_CONFIG_FILENAME,
)
from .authorized_index_store import (
    INDEX_DIRECTORY,
    ROW_MAP_DIRECTORY,
    AuthorizedIndexConfig,
    AuthorizedIndexStoreError,
    AuthorizedIndexStoreReceipt,
    load_authorized_index_store_receipt,
)
from .authorized_index_store import (
    RECEIPT_FILENAME as INDEX_RECEIPT_FILENAME,
)
from .compiled_policy import (
    CompiledPolicyError,
    CompiledPolicyMaskStore,
    load_compiled_policy_catalog,
)
from .embedding_store import EmbeddingStoreError, EmbeddingStoreReceipt, verify_embedding_store
from .policy_intervention import (
    CATALOG_FILENAME,
    CONFIG_FILENAME,
    OPA_DATA_FILENAME,
    RECEIPT_FILENAME,
    SCHEDULE_FILENAME,
    PolicyInterventionError,
    load_canonical_trial_schedule,
    load_opa_compiled_mask_data,
    load_policy_intervention_config,
    load_policy_intervention_receipt,
)

ARTIFACT_STAGE_ORDER = ("fit", "calibration", "sealed")
POLICY_SOURCE_STAGE_BY_BUNDLE_STAGE = {
    "fit": "development-fit",
    "calibration": "development-calibration",
    "sealed": "sealed",
}
STAGE_BUNDLE_FILENAME = "stage-bundle.json"
POLICY_STAGE_BUNDLE_SCHEMA = "fractal-policy-stage-bundle-v1"
INDEX_STAGE_BUNDLE_SCHEMA = "fractal-authorized-index-stage-bundle-v1"
POLICY_STAGE_BUNDLE_KIND = "policy-data"
INDEX_STAGE_BUNDLE_KIND = "index-store"

StageName = Literal["fit", "calibration", "sealed"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CORPUS_ID = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
_POLICY_DESCRIPTOR_FIELDS = frozenset(
    {
        "assignment_map_sha256",
        "assignment_seed_sha256",
        "catalog_sha256",
        "config_sha256",
        "document_count",
        "document_universe_sha256",
        "execution_artifact_sha256",
        "mask_count",
        "policy_revision",
        "receipt_sha256",
        "relative_path",
        "schedule_sha256",
        "stage",
        "tree_sha256",
        "trial_count",
    }
)
_INDEX_DESCRIPTOR_FIELDS = frozenset(
    {
        "config_sha256",
        "current_truth_vector_sha256",
        "document_count",
        "document_row_order_sha256",
        "document_universe_sha256",
        "embedding_receipt_sha256",
        "index_count",
        "old_active_vector_sha256",
        "payload_tree_sha256",
        "policy_catalog_sha256",
        "policy_execution_artifact_sha256",
        "policy_receipt_sha256",
        "policy_revision",
        "receipt_sha256",
        "relative_path",
        "stage",
        "tree_sha256",
    }
)
_BUNDLE_FIELDS = frozenset({"artifact_kind", "corpus_id", "schema_version", "stages"})
_MAX_CONTROL_BYTES = 64 * 1024 * 1024


class ArtifactStageBundleError(RuntimeError):
    """Raised when a stage bundle is incomplete, mutable, or cross-bound incorrectly."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ArtifactStageBundleError(
            "stage-bundle metadata must be finite canonical JSON"
        ) from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactStageBundleError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_corpus_id(value: object) -> str:
    if not isinstance(value, str) or _CORPUS_ID.fullmatch(value) is None:
        raise ArtifactStageBundleError("corpus_id must be a lowercase filesystem-safe identifier")
    return value


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactStageBundleError(f"{name} must be a positive integer")
    return value


def _require_stage(value: object) -> StageName:
    if value not in ARTIFACT_STAGE_ORDER:
        raise ArtifactStageBundleError("stage must be one of fit, calibration, or sealed")
    return value  # type: ignore[return-value]


def _relative_stage_path(value: object, *, stage: StageName) -> str:
    if not isinstance(value, str) or value != stage:
        raise ArtifactStageBundleError("stage relative_path must equal its canonical stage name")
    path = PurePosixPath(value)
    if path.is_absolute() or path.parts != (stage,):
        raise ArtifactStageBundleError("stage relative_path is not canonical")
    return value


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ArtifactStageBundleError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise ArtifactStageBundleError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactStageBundleError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ArtifactStageBundleError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactStageBundleError(f"{label} must be canonical UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ArtifactStageBundleError(f"{label} must contain one object")
    return value


def _canonical_root(path: str | Path, *, label: str) -> Path:
    root = Path(path)
    if not root.is_absolute() or root.name in {"", ".", ".."}:
        raise ArtifactStageBundleError(f"{label} must be an absolute canonical path")
    raw = str(root)
    if (
        "\\" in raw
        or unicodedata.normalize("NFC", raw) != raw
        or PurePosixPath(raw).as_posix() != raw
        or any(part in {".", ".."} for part in root.parts)
    ):
        raise ArtifactStageBundleError(f"{label} must be an absolute canonical path")
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise ArtifactStageBundleError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ArtifactStageBundleError(f"{label} must be a real directory")
    return root


def _verify_bound_file(root: Path, relative_path: str, byte_count: int, sha256: str) -> None:
    target = root.joinpath(*PurePosixPath(relative_path).parts)
    _require_positive_integer(f"{relative_path} byte_count", byte_count)
    _require_sha256(f"{relative_path} sha256", sha256)
    try:
        metadata = target.lstat()
        digest = digest_regular_file(target, label=f"policy artifact {relative_path}")
    except (OSError, ArtifactIntegrityError) as exc:
        raise ArtifactStageBundleError(
            f"cannot verify policy artifact {relative_path!r}: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != byte_count
        or digest != sha256
    ):
        raise ArtifactStageBundleError(
            f"policy artifact {relative_path!r} differs from its receipt"
        )


@dataclass(frozen=True, order=True)
class PolicyStageDescriptor:
    """Typed identity of one policy-intervention stage package."""

    stage: StageName
    relative_path: str
    tree_sha256: str
    receipt_sha256: str
    execution_artifact_sha256: str
    config_sha256: str
    catalog_sha256: str
    schedule_sha256: str
    assignment_seed_sha256: str
    assignment_map_sha256: str
    policy_revision: str
    document_universe_sha256: str
    document_count: int
    mask_count: int
    trial_count: int

    def __post_init__(self) -> None:
        stage = _require_stage(self.stage)
        object.__setattr__(
            self,
            "relative_path",
            _relative_stage_path(self.relative_path, stage=stage),
        )
        for name in (
            "tree_sha256",
            "receipt_sha256",
            "execution_artifact_sha256",
            "config_sha256",
            "catalog_sha256",
            "schedule_sha256",
            "assignment_seed_sha256",
            "assignment_map_sha256",
            "document_universe_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            not isinstance(self.policy_revision, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.policy_revision) is None
        ):
            raise ArtifactStageBundleError("policy_revision must be an immutable SHA-256 revision")
        _require_positive_integer("document_count", self.document_count)
        _require_positive_integer("mask_count", self.mask_count)
        _require_positive_integer("trial_count", self.trial_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_map_sha256": self.assignment_map_sha256,
            "assignment_seed_sha256": self.assignment_seed_sha256,
            "catalog_sha256": self.catalog_sha256,
            "config_sha256": self.config_sha256,
            "document_count": self.document_count,
            "document_universe_sha256": self.document_universe_sha256,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "mask_count": self.mask_count,
            "policy_revision": self.policy_revision,
            "receipt_sha256": self.receipt_sha256,
            "relative_path": self.relative_path,
            "schedule_sha256": self.schedule_sha256,
            "stage": self.stage,
            "tree_sha256": self.tree_sha256,
            "trial_count": self.trial_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> PolicyStageDescriptor:
        row = _closed(value, _POLICY_DESCRIPTOR_FIELDS, label="policy stage descriptor")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PolicyStageBundleReceipt:
    """Canonical exact-stage receipt for one corpus policy artifact."""

    corpus_id: str
    stages: tuple[PolicyStageDescriptor, ...]
    schema_version: str = POLICY_STAGE_BUNDLE_SCHEMA
    artifact_kind: str = POLICY_STAGE_BUNDLE_KIND

    def __post_init__(self) -> None:
        _require_corpus_id(self.corpus_id)
        stages = tuple(self.stages)
        if (
            not all(isinstance(row, PolicyStageDescriptor) for row in stages)
            or tuple(row.stage for row in stages) != ARTIFACT_STAGE_ORDER
        ):
            raise ArtifactStageBundleError(
                "policy bundle must contain fit, calibration, sealed in order"
            )
        if self.schema_version != POLICY_STAGE_BUNDLE_SCHEMA:
            raise ArtifactStageBundleError("policy stage-bundle schema differs")
        if self.artifact_kind != POLICY_STAGE_BUNDLE_KIND:
            raise ArtifactStageBundleError("policy stage-bundle artifact kind differs")
        if len({row.execution_artifact_sha256 for row in stages}) != len(stages):
            raise ArtifactStageBundleError("policy stages must bind distinct execution artifacts")
        object.__setattr__(self, "stages", stages)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "corpus_id": self.corpus_id,
            "schema_version": self.schema_version,
            "stages": [row.to_dict() for row in self.stages],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> PolicyStageBundleReceipt:
        row = _closed(value, _BUNDLE_FIELDS, label="policy stage bundle")
        stages = row["stages"]
        if not isinstance(stages, list):
            raise ArtifactStageBundleError("policy stage bundle stages must be an array")
        return cls(
            corpus_id=row["corpus_id"],
            stages=tuple(PolicyStageDescriptor.from_dict(item) for item in stages),
            schema_version=row["schema_version"],
            artifact_kind=row["artifact_kind"],
        )


def inspect_policy_stage(
    stage_root: str | Path,
    *,
    expected_corpus_id: str,
    expected_stage: StageName,
) -> PolicyStageDescriptor:
    """Structurally and semantically inspect one stored intervention package."""

    root = _canonical_root(stage_root, label="policy stage root")
    corpus_id = _require_corpus_id(expected_corpus_id)
    stage = _require_stage(expected_stage)
    source_stage = POLICY_SOURCE_STAGE_BY_BUNDLE_STAGE[stage]
    try:
        tree = digest_directory_tree(root)
        config = load_policy_intervention_config(root / CONFIG_FILENAME)
        catalog = load_compiled_policy_catalog(root / CATALOG_FILENAME)
        opa = load_opa_compiled_mask_data(root / OPA_DATA_FILENAME)
        schedule = load_canonical_trial_schedule(root / SCHEDULE_FILENAME)
        receipt = load_policy_intervention_receipt(root / RECEIPT_FILENAME)
    except (
        ArtifactIntegrityError,
        CompiledPolicyError,
        PolicyInterventionError,
        OSError,
    ) as exc:
        raise ArtifactStageBundleError(f"invalid typed policy stage: {exc}") from exc

    expected_entries = {
        RECEIPT_FILENAME,
        "baseline-masks",
        "masks",
        *(row.path for row in receipt.artifacts),
    }
    if set(tree.entries) != expected_entries:
        raise ArtifactStageBundleError(
            "policy stage membership differs; "
            f"missing={sorted(expected_entries - set(tree.entries))}, "
            f"extra={sorted(set(tree.entries) - expected_entries)}"
        )
    for artifact in receipt.artifacts:
        _verify_bound_file(root, artifact.path, artifact.byte_count, artifact.sha256)
    try:
        mask_ids = CompiledPolicyMaskStore(root / CATALOG_FILENAME).verify_all()
    except CompiledPolicyError as exc:
        raise ArtifactStageBundleError(f"compiled policy masks failed verification: {exc}") from exc

    shared = (
        receipt.corpus,
        receipt.stage,
        receipt.document_count,
        receipt.document_universe_sha256,
        receipt.execution_artifact_sha256,
        receipt.config_sha256,
        receipt.policy_bundle_revision,
    )
    if shared != (
        corpus_id,
        source_stage,
        schedule.document_count,
        schedule.document_universe_sha256,
        schedule.execution_artifact_sha256,
        schedule.config_sha256,
        schedule.policy_bundle_revision,
    ):
        raise ArtifactStageBundleError("policy receipt and schedule bindings differ")
    if (
        receipt.config_sha256 != config.config_sha256
        or receipt.seed_sha256 != config.seed_sha256
        or receipt.baseline_seed_sha256 != config.baseline_seed_sha256
        or receipt.policy_bundle_revision != config.policy_bundle_revision
        or receipt.baseline_policy_revision != config.baseline_policy_revision
        or schedule.mask_catalog_sha256 != catalog.artifact_sha256
        or schedule.baseline_policy_revision != receipt.baseline_policy_revision
        or catalog.document_count != receipt.document_count
        or catalog.document_universe_sha256 != receipt.document_universe_sha256
        or catalog.policy_revision != receipt.policy_bundle_revision
        or opa.document_count != receipt.document_count
        or opa.document_universe_sha256 != receipt.document_universe_sha256
        or opa.mask_catalog_sha256 != catalog.artifact_sha256
        or opa.policy_revision != receipt.policy_bundle_revision
        or set(mask_ids) != {row.mask_id for row in catalog.masks}
        or {row.mask_id for row in opa.assignments} != set(mask_ids)
        or {row.mask_id for row in schedule.rows} != set(mask_ids)
    ):
        raise ArtifactStageBundleError("policy stage typed objects are not mutually bound")
    try:
        final_tree = digest_directory_tree(root)
    except ArtifactIntegrityError as exc:
        raise ArtifactStageBundleError(f"cannot rehash policy stage: {exc}") from exc
    if final_tree != tree:
        raise ArtifactStageBundleError("policy stage changed during typed inspection")
    return PolicyStageDescriptor(
        stage=stage,
        relative_path=stage,
        tree_sha256=tree.sha256,
        receipt_sha256=receipt.artifact_sha256,
        execution_artifact_sha256=receipt.execution_artifact_sha256,
        config_sha256=config.config_sha256,
        catalog_sha256=catalog.artifact_sha256,
        schedule_sha256=schedule.artifact_sha256,
        assignment_seed_sha256=schedule.assignment_seed_sha256,
        assignment_map_sha256=schedule.assignment_map_sha256,
        policy_revision=receipt.policy_bundle_revision,
        document_universe_sha256=receipt.document_universe_sha256,
        document_count=receipt.document_count,
        mask_count=len(mask_ids),
        trial_count=len(schedule.rows),
    )


def _read_bundle(path: Path, *, kind: str) -> bytes:
    try:
        return read_secure_regular_file(
            path,
            max_bytes=_MAX_CONTROL_BYTES,
            label=f"{kind} stage-bundle receipt",
        )
    except ArtifactIntegrityError as exc:
        raise ArtifactStageBundleError(f"cannot read {kind} stage-bundle receipt: {exc}") from exc


def load_policy_stage_bundle(root: str | Path) -> PolicyStageBundleReceipt:
    bundle_root = _canonical_root(root, label="policy stage-bundle root")
    encoded = _read_bundle(bundle_root / STAGE_BUNDLE_FILENAME, kind="policy")
    receipt = PolicyStageBundleReceipt.from_dict(
        _decode_object(encoded, label="policy stage-bundle receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ArtifactStageBundleError("policy stage-bundle receipt is not canonical")
    return receipt


def verify_policy_stage_bundle(
    root: str | Path,
    *,
    expected_corpus_id: str | None = None,
) -> PolicyStageBundleReceipt:
    """Reproduce every policy stage descriptor and reject any extra entry."""

    bundle_root = _canonical_root(root, label="policy stage-bundle root")
    receipt = load_policy_stage_bundle(bundle_root)
    if expected_corpus_id is not None and receipt.corpus_id != _require_corpus_id(
        expected_corpus_id
    ):
        raise ArtifactStageBundleError("policy stage-bundle corpus differs")
    reproduced = tuple(
        inspect_policy_stage(
            bundle_root / stage,
            expected_corpus_id=receipt.corpus_id,
            expected_stage=stage,
        )
        for stage in ARTIFACT_STAGE_ORDER
    )
    if reproduced != receipt.stages:
        raise ArtifactStageBundleError("policy stage descriptors differ from stored artifacts")
    try:
        tree = digest_directory_tree(bundle_root)
    except ArtifactIntegrityError as exc:
        raise ArtifactStageBundleError(f"cannot hash policy stage bundle: {exc}") from exc
    expected_entries = {STAGE_BUNDLE_FILENAME}
    for descriptor in receipt.stages:
        expected_entries.add(descriptor.relative_path)
        stage_tree = digest_directory_tree(bundle_root / descriptor.relative_path)
        if stage_tree.sha256 != descriptor.tree_sha256:
            raise ArtifactStageBundleError("policy stage changed during bundle verification")
        expected_entries.update(
            f"{descriptor.relative_path}/{entry}" for entry in stage_tree.entries
        )
    if set(tree.entries) != expected_entries:
        raise ArtifactStageBundleError("policy stage-bundle has undeclared membership")
    if load_policy_stage_bundle(bundle_root) != receipt:
        raise ArtifactStageBundleError("policy stage-bundle receipt changed during verification")
    if digest_directory_tree(bundle_root) != tree:
        raise ArtifactStageBundleError("policy stage-bundle changed during verification")
    return receipt


def seal_policy_stage_bundle(
    root: str | Path,
    *,
    corpus_id: str,
) -> PolicyStageBundleReceipt:
    """Write the sole canonical policy bundle receipt after all stages verify."""

    bundle_root = _canonical_root(root, label="policy stage-bundle root")
    target = bundle_root / STAGE_BUNDLE_FILENAME
    if target.exists() or target.is_symlink():
        return verify_policy_stage_bundle(bundle_root, expected_corpus_id=corpus_id)
    try:
        before = digest_directory_tree(bundle_root)
    except ArtifactIntegrityError as exc:
        raise ArtifactStageBundleError(f"cannot inspect unsealed policy bundle: {exc}") from exc
    expected = set(ARTIFACT_STAGE_ORDER)
    top_level = {entry for entry in before.entries if "/" not in entry}
    if top_level != expected:
        raise ArtifactStageBundleError(
            "unsealed policy bundle must contain exactly fit, calibration, and sealed"
        )
    receipt = PolicyStageBundleReceipt(
        corpus_id=_require_corpus_id(corpus_id),
        stages=tuple(
            inspect_policy_stage(
                bundle_root / stage,
                expected_corpus_id=corpus_id,
                expected_stage=stage,
            )
            for stage in ARTIFACT_STAGE_ORDER
        ),
    )
    try:
        write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), target)
    except ArtifactIntegrityError as exc:
        raise ArtifactStageBundleError(f"cannot seal policy stage bundle: {exc}") from exc
    return verify_policy_stage_bundle(bundle_root, expected_corpus_id=corpus_id)


@dataclass(frozen=True, order=True)
class IndexStageDescriptor:
    """Typed identity of one authorized-index stage store."""

    stage: StageName
    relative_path: str
    tree_sha256: str
    receipt_sha256: str
    payload_tree_sha256: str
    config_sha256: str
    old_active_vector_sha256: str
    current_truth_vector_sha256: str
    embedding_receipt_sha256: str
    policy_receipt_sha256: str
    policy_catalog_sha256: str
    policy_execution_artifact_sha256: str
    policy_revision: str
    document_universe_sha256: str
    document_row_order_sha256: str
    document_count: int
    index_count: int

    def __post_init__(self) -> None:
        stage = _require_stage(self.stage)
        object.__setattr__(
            self,
            "relative_path",
            _relative_stage_path(self.relative_path, stage=stage),
        )
        for name in (
            "tree_sha256",
            "receipt_sha256",
            "payload_tree_sha256",
            "config_sha256",
            "old_active_vector_sha256",
            "current_truth_vector_sha256",
            "embedding_receipt_sha256",
            "policy_receipt_sha256",
            "policy_catalog_sha256",
            "policy_execution_artifact_sha256",
            "document_universe_sha256",
            "document_row_order_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            not isinstance(self.policy_revision, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.policy_revision) is None
        ):
            raise ArtifactStageBundleError("index policy_revision must be immutable")
        _require_positive_integer("document_count", self.document_count)
        _require_positive_integer("index_count", self.index_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "current_truth_vector_sha256": self.current_truth_vector_sha256,
            "document_count": self.document_count,
            "document_row_order_sha256": self.document_row_order_sha256,
            "document_universe_sha256": self.document_universe_sha256,
            "embedding_receipt_sha256": self.embedding_receipt_sha256,
            "index_count": self.index_count,
            "old_active_vector_sha256": self.old_active_vector_sha256,
            "payload_tree_sha256": self.payload_tree_sha256,
            "policy_catalog_sha256": self.policy_catalog_sha256,
            "policy_execution_artifact_sha256": self.policy_execution_artifact_sha256,
            "policy_receipt_sha256": self.policy_receipt_sha256,
            "policy_revision": self.policy_revision,
            "receipt_sha256": self.receipt_sha256,
            "relative_path": self.relative_path,
            "stage": self.stage,
            "tree_sha256": self.tree_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> IndexStageDescriptor:
        row = _closed(value, _INDEX_DESCRIPTOR_FIELDS, label="index stage descriptor")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class IndexStageBundleReceipt:
    """Canonical exact-stage receipt for one corpus authorized-index artifact."""

    corpus_id: str
    stages: tuple[IndexStageDescriptor, ...]
    schema_version: str = INDEX_STAGE_BUNDLE_SCHEMA
    artifact_kind: str = INDEX_STAGE_BUNDLE_KIND

    def __post_init__(self) -> None:
        _require_corpus_id(self.corpus_id)
        stages = tuple(self.stages)
        if (
            not all(isinstance(row, IndexStageDescriptor) for row in stages)
            or tuple(row.stage for row in stages) != ARTIFACT_STAGE_ORDER
        ):
            raise ArtifactStageBundleError(
                "index bundle must contain fit, calibration, sealed in order"
            )
        if self.schema_version != INDEX_STAGE_BUNDLE_SCHEMA:
            raise ArtifactStageBundleError("index stage-bundle schema differs")
        if self.artifact_kind != INDEX_STAGE_BUNDLE_KIND:
            raise ArtifactStageBundleError("index stage-bundle artifact kind differs")
        object.__setattr__(self, "stages", stages)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "corpus_id": self.corpus_id,
            "schema_version": self.schema_version,
            "stages": [row.to_dict() for row in self.stages],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> IndexStageBundleReceipt:
        row = _closed(value, _BUNDLE_FIELDS, label="index stage bundle")
        stages = row["stages"]
        if not isinstance(stages, list):
            raise ArtifactStageBundleError("index stage bundle stages must be an array")
        return cls(
            corpus_id=row["corpus_id"],
            stages=tuple(IndexStageDescriptor.from_dict(item) for item in stages),
            schema_version=row["schema_version"],
            artifact_kind=row["artifact_kind"],
        )


def _load_index_config(root: Path) -> AuthorizedIndexConfig:
    try:
        encoded = read_secure_regular_file(
            root / INDEX_CONFIG_FILENAME,
            max_bytes=_MAX_CONTROL_BYTES,
            label="authorized index config",
        )
    except ArtifactIntegrityError as exc:
        raise ArtifactStageBundleError(f"cannot read authorized index config: {exc}") from exc
    try:
        config = AuthorizedIndexConfig.from_dict(
            _decode_object(encoded, label="authorized index config")
        )
    except AuthorizedIndexStoreError as exc:
        raise ArtifactStageBundleError(f"invalid authorized index config: {exc}") from exc
    if encoded != config.canonical_bytes() + b"\n":
        raise ArtifactStageBundleError("authorized index config is not canonical")
    return config


def _verify_index_payloads(root: Path, receipt: AuthorizedIndexStoreReceipt) -> None:
    for item in receipt.indexes:
        for label, relative_path, expected_size, expected_sha256 in (
            (
                "authorized index",
                item.index_path,
                item.index_byte_count,
                item.index_sha256,
            ),
            (
                "authorized row map",
                item.row_map_path,
                item.row_map_byte_count,
                item.row_map_sha256,
            ),
        ):
            target = root.joinpath(*PurePosixPath(relative_path).parts)
            try:
                metadata = target.lstat()
                observed = digest_regular_file(target, label=f"{label} {item.mask_id}")
            except (OSError, ArtifactIntegrityError) as exc:
                raise ArtifactStageBundleError(f"cannot verify {label}: {exc}") from exc
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != expected_size
                or observed != expected_sha256
            ):
                raise ArtifactStageBundleError(f"{label} differs from its typed receipt")
        row_map_path = root.joinpath(*PurePosixPath(item.row_map_path).parts)
        try:
            row_map = np.load(row_map_path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise ArtifactStageBundleError("authorized row map is not a valid NPY array") from exc
        try:
            values = np.asarray(row_map)
            if (
                values.dtype != np.dtype(item.row_map_dtype)
                or values.shape != item.row_map_shape
                or not values.flags.c_contiguous
                or int(values[0]) < 0
                or int(values[-1]) >= receipt.document_count
                or (len(values) > 1 and not np.all(values[1:] > values[:-1]))
            ):
                raise ArtifactStageBundleError(
                    "authorized row map geometry differs from its typed receipt"
                )
        finally:
            del row_map


def inspect_index_stage(
    stage_root: str | Path,
    *,
    expected_stage: StageName,
    embedding_receipt: EmbeddingStoreReceipt,
    policy_stage: PolicyStageDescriptor,
    policy_stage_root: str | Path,
) -> IndexStageDescriptor:
    """Inspect one store and cross-bind it to its policy stage and embedding store."""

    root = _canonical_root(stage_root, label="authorized index stage root")
    stage = _require_stage(expected_stage)
    if not isinstance(embedding_receipt, EmbeddingStoreReceipt):
        raise TypeError("embedding_receipt must be an EmbeddingStoreReceipt")
    if not isinstance(policy_stage, PolicyStageDescriptor) or policy_stage.stage != stage:
        raise ArtifactStageBundleError("index stage needs the matching typed policy stage")
    policy_root = _canonical_root(policy_stage_root, label="matching policy stage root")
    try:
        full_tree = digest_directory_tree(root)
        receipt = load_authorized_index_store_receipt(root)
        config = _load_index_config(root)
        catalog = load_compiled_policy_catalog(policy_root / CATALOG_FILENAME)
    except (
        ArtifactIntegrityError,
        AuthorizedIndexStoreError,
        CompiledPolicyError,
        OSError,
    ) as exc:
        raise ArtifactStageBundleError(f"invalid authorized index stage: {exc}") from exc
    expected_entries = {
        INDEX_CONFIG_FILENAME,
        INDEX_RECEIPT_FILENAME,
        INDEX_DIRECTORY,
        ROW_MAP_DIRECTORY,
        *(row.index_path for row in receipt.indexes),
        *(row.row_map_path for row in receipt.indexes),
    }
    if set(full_tree.entries) != expected_entries:
        raise ArtifactStageBundleError(
            "authorized index stage membership differs; "
            f"missing={sorted(expected_entries - set(full_tree.entries))}, "
            f"extra={sorted(set(full_tree.entries) - expected_entries)}"
        )
    payload_entries = sorted(expected_entries - {INDEX_RECEIPT_FILENAME})
    try:
        payload_tree = digest_directory_tree(root, included_entries=payload_entries)
    except ArtifactIntegrityError as exc:
        raise ArtifactStageBundleError(
            f"cannot verify authorized index payload tree: {exc}"
        ) from exc
    _verify_index_payloads(root, receipt)
    masks = {mask.mask_id: mask for mask in catalog.masks}
    indexed_masks = {item.mask_id: item for item in receipt.indexes}
    if set(indexed_masks) != set(masks):
        raise ArtifactStageBundleError(
            "authorized index stage does not cover the exact compiled mask catalog"
        )
    for mask_id, mask in masks.items():
        item = indexed_masks[mask_id]
        expected_build_binding = _sha256(
            _canonical_bytes(
                {
                    "backend": {
                        "build_sha256": config.backend_build_sha256,
                        "id": config.backend_id,
                        "version": config.backend_version,
                    },
                    "builder_identity": config.builder_identity,
                    "config_sha256": config.config_sha256,
                    "current_truth_vector": receipt.current_truth_vector.to_dict(),
                    "document_universe_sha256": policy_stage.document_universe_sha256,
                    "embedding_receipt_sha256": embedding_receipt.receipt_sha256,
                    "mask": mask.to_dict(),
                    "old_active_vector": receipt.old_active_vector.to_dict(),
                    "policy_catalog_sha256": catalog.artifact_sha256,
                    "policy_receipt_sha256": policy_stage.receipt_sha256,
                }
            )
        )
        if (
            item.mask_sha256 != mask.sha256
            or item.authorized_count != mask.authorized_count
            or item.index_path != f"{INDEX_DIRECTORY}/{mask_id}.hnsw"
            or item.row_map_path != f"{ROW_MAP_DIRECTORY}/{mask_id}.npy"
            or item.build_binding_sha256 != expected_build_binding
        ):
            raise ArtifactStageBundleError(
                "authorized index artifact differs from its compiled policy mask"
            )
    if (
        receipt.payload_tree_sha256 != payload_tree.sha256
        or receipt.config_sha256 != config.config_sha256
        or receipt.backend_id != config.backend_id
        or receipt.backend_version != config.backend_version
        or receipt.backend_build_sha256 != config.backend_build_sha256
        or receipt.builder_identity != config.builder_identity
        or receipt.failure_policy != config.failure_policy
        or receipt.embedding_receipt_sha256 != embedding_receipt.receipt_sha256
        or receipt.document_count != embedding_receipt.document_count
        or receipt.document_universe_sha256
        != embedding_receipt.row_orders["documents"].row_order_sha256
        or receipt.document_row_order_sha256
        != embedding_receipt.row_orders["documents"].row_order_sha256
        or receipt.policy_receipt_sha256 != policy_stage.receipt_sha256
        or receipt.policy_catalog_sha256 != policy_stage.catalog_sha256
        or catalog.artifact_sha256 != policy_stage.catalog_sha256
        or receipt.policy_execution_artifact_sha256 != policy_stage.execution_artifact_sha256
        or receipt.policy_revision != policy_stage.policy_revision
        or receipt.document_universe_sha256 != policy_stage.document_universe_sha256
        or receipt.document_count != policy_stage.document_count
        or len(receipt.indexes) != policy_stage.mask_count
    ):
        raise ArtifactStageBundleError("authorized index stage source bindings differ")
    try:
        final_tree = digest_directory_tree(root)
    except ArtifactIntegrityError as exc:
        raise ArtifactStageBundleError(f"cannot rehash authorized index stage: {exc}") from exc
    if final_tree != full_tree:
        raise ArtifactStageBundleError("authorized index stage changed during typed inspection")
    return IndexStageDescriptor(
        stage=stage,
        relative_path=stage,
        tree_sha256=full_tree.sha256,
        receipt_sha256=receipt.artifact_sha256,
        payload_tree_sha256=receipt.payload_tree_sha256,
        config_sha256=receipt.config_sha256,
        old_active_vector_sha256=receipt.old_active_vector.file_sha256,
        current_truth_vector_sha256=receipt.current_truth_vector.file_sha256,
        embedding_receipt_sha256=receipt.embedding_receipt_sha256,
        policy_receipt_sha256=receipt.policy_receipt_sha256,
        policy_catalog_sha256=receipt.policy_catalog_sha256,
        policy_execution_artifact_sha256=receipt.policy_execution_artifact_sha256,
        policy_revision=receipt.policy_revision,
        document_universe_sha256=receipt.document_universe_sha256,
        document_row_order_sha256=receipt.document_row_order_sha256,
        document_count=receipt.document_count,
        index_count=len(receipt.indexes),
    )


def load_index_stage_bundle(root: str | Path) -> IndexStageBundleReceipt:
    bundle_root = _canonical_root(root, label="authorized index stage-bundle root")
    encoded = _read_bundle(bundle_root / STAGE_BUNDLE_FILENAME, kind="index")
    receipt = IndexStageBundleReceipt.from_dict(
        _decode_object(encoded, label="index stage-bundle receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ArtifactStageBundleError("index stage-bundle receipt is not canonical")
    return receipt


def verify_index_stage_bundle(
    root: str | Path,
    *,
    embedding_store_root: str | Path,
    policy_bundle_root: str | Path,
    expected_corpus_id: str | None = None,
) -> IndexStageBundleReceipt:
    """Reproduce all index descriptors and their policy/embedding cross-bindings."""

    bundle_root = _canonical_root(root, label="authorized index stage-bundle root")
    receipt = load_index_stage_bundle(bundle_root)
    if expected_corpus_id is not None and receipt.corpus_id != _require_corpus_id(
        expected_corpus_id
    ):
        raise ArtifactStageBundleError("index stage-bundle corpus differs")
    try:
        embedding = verify_embedding_store(embedding_store_root)
    except EmbeddingStoreError as exc:
        raise ArtifactStageBundleError(f"embedding store failed typed verification: {exc}") from exc
    policy = verify_policy_stage_bundle(
        policy_bundle_root,
        expected_corpus_id=receipt.corpus_id,
    )
    policy_by_stage = {row.stage: row for row in policy.stages}
    reproduced = tuple(
        inspect_index_stage(
            bundle_root / stage,
            expected_stage=stage,
            embedding_receipt=embedding,
            policy_stage=policy_by_stage[stage],
            policy_stage_root=Path(policy_bundle_root) / stage,
        )
        for stage in ARTIFACT_STAGE_ORDER
    )
    if reproduced != receipt.stages:
        raise ArtifactStageBundleError("index stage descriptors differ from stored artifacts")
    try:
        tree = digest_directory_tree(bundle_root)
    except ArtifactIntegrityError as exc:
        raise ArtifactStageBundleError(f"cannot hash index stage bundle: {exc}") from exc
    expected_entries = {STAGE_BUNDLE_FILENAME}
    for descriptor in receipt.stages:
        expected_entries.add(descriptor.relative_path)
        stage_tree = digest_directory_tree(bundle_root / descriptor.relative_path)
        if stage_tree.sha256 != descriptor.tree_sha256:
            raise ArtifactStageBundleError("index stage changed during bundle verification")
        expected_entries.update(
            f"{descriptor.relative_path}/{entry}" for entry in stage_tree.entries
        )
    if set(tree.entries) != expected_entries:
        raise ArtifactStageBundleError("index stage-bundle has undeclared membership")
    if load_index_stage_bundle(bundle_root) != receipt:
        raise ArtifactStageBundleError("index stage-bundle receipt changed during verification")
    if digest_directory_tree(bundle_root) != tree:
        raise ArtifactStageBundleError("index stage-bundle changed during verification")
    return receipt


def seal_index_stage_bundle(
    root: str | Path,
    *,
    corpus_id: str,
    embedding_store_root: str | Path,
    policy_bundle_root: str | Path,
) -> IndexStageBundleReceipt:
    """Write one index bundle receipt after every stage and source pin verifies."""

    bundle_root = _canonical_root(root, label="authorized index stage-bundle root")
    target = bundle_root / STAGE_BUNDLE_FILENAME
    if target.exists() or target.is_symlink():
        return verify_index_stage_bundle(
            bundle_root,
            embedding_store_root=embedding_store_root,
            policy_bundle_root=policy_bundle_root,
            expected_corpus_id=corpus_id,
        )
    try:
        before = digest_directory_tree(bundle_root)
        embedding = verify_embedding_store(embedding_store_root)
    except (ArtifactIntegrityError, EmbeddingStoreError) as exc:
        raise ArtifactStageBundleError(f"cannot inspect unsealed index bundle: {exc}") from exc
    top_level = {entry for entry in before.entries if "/" not in entry}
    if top_level != set(ARTIFACT_STAGE_ORDER):
        raise ArtifactStageBundleError(
            "unsealed index bundle must contain exactly fit, calibration, and sealed"
        )
    policy = verify_policy_stage_bundle(
        policy_bundle_root,
        expected_corpus_id=corpus_id,
    )
    policy_by_stage = {row.stage: row for row in policy.stages}
    receipt = IndexStageBundleReceipt(
        corpus_id=_require_corpus_id(corpus_id),
        stages=tuple(
            inspect_index_stage(
                bundle_root / stage,
                expected_stage=stage,
                embedding_receipt=embedding,
                policy_stage=policy_by_stage[stage],
                policy_stage_root=Path(policy_bundle_root) / stage,
            )
            for stage in ARTIFACT_STAGE_ORDER
        ),
    )
    try:
        write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), target)
    except ArtifactIntegrityError as exc:
        raise ArtifactStageBundleError(f"cannot seal index stage bundle: {exc}") from exc
    return verify_index_stage_bundle(
        bundle_root,
        embedding_store_root=embedding_store_root,
        policy_bundle_root=policy_bundle_root,
        expected_corpus_id=corpus_id,
    )
