"""Deterministic five-corpus packaging for label-free online artifacts.

The expensive embedding, policy, index, and query-runtime builders retain their
specialized APIs.  This module closes the packaging boundary around their
outputs.  It first admits the exact online staging projection, then seals or
verifies each corpus in the registered order.  No command accepts a qrels,
evidence, answer, or custody path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .artifact_stage_bundles import (
    ArtifactStageBundleError,
    IndexStageBundleReceipt,
    PolicyStageBundleReceipt,
    seal_index_stage_bundle,
    seal_policy_stage_bundle,
    verify_index_stage_bundle,
    verify_policy_stage_bundle,
)
from .embedding_store import (
    EmbeddingStoreError,
    EmbeddingStoreReceipt,
    VectorDescriptor,
    verify_embedding_store,
)
from .joint_power_design import FIXED_CORPORA
from .scalable_execution import (
    ONLINE_EXECUTION_PLAN_FILENAME,
    ScalableExecutionError,
    load_sharded_online_execution_plan,
)
from .study_data import StudyDataError, verify_online_staging_projection
from .trial_runtime import (
    QUERY_TRIAL_FILENAME,
    QUERY_TRIAL_RECEIPT_FILENAME,
    CanonicalQueryTrialRow,
    QueryTrialStoreReceipt,
    QueryVectorEpochBinding,
    TrialRuntimeError,
    load_trial_runtime_receipt,
)

ARTIFACT_PIPELINE_SCHEMA = "fractal-five-corpus-artifact-pipeline-v1"
CORPUS_PIPELINE_SCHEMA = "fractal-corpus-artifact-pipeline-v1"
RUNTIME_PACKAGE_SCHEMA = "fractal-runtime-package-verification-v1"
SHARDED_PLAN_FILENAME = ONLINE_EXECUTION_PLAN_FILENAME
TRIAL_RUNTIME_ADMISSION_FILENAME = "trial-runtime-admission-receipt.json"
QUERY_PACKAGE_DIRECTORY = "query-package"
ARTIFACT_PIPELINE_ORDER = ("embedding", "policy", "index", "runtime")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_RUNTIME_EXPECTED_ENTRIES = frozenset(
    {
        QUERY_PACKAGE_DIRECTORY,
        f"{QUERY_PACKAGE_DIRECTORY}/{QUERY_TRIAL_FILENAME}",
        f"{QUERY_PACKAGE_DIRECTORY}/{QUERY_TRIAL_RECEIPT_FILENAME}",
        SHARDED_PLAN_FILENAME,
        TRIAL_RUNTIME_ADMISSION_FILENAME,
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "execution_plan_sha256",
        "query_count",
        "query_receipt_sha256",
        "runtime_receipt_sha256",
        "schema_version",
        "tree_sha256",
    }
)
_CORPUS_FIELDS = frozenset(
    {
        "corpus_id",
        "embedding_receipt_sha256",
        "embedding_tree_sha256",
        "index_bundle_receipt_sha256",
        "index_bundle_tree_sha256",
        "policy_bundle_receipt_sha256",
        "policy_bundle_tree_sha256",
        "runtime",
        "schema_version",
    }
)
_PIPELINE_FIELDS = frozenset(
    {
        "artifact_order",
        "corpora",
        "online_inventory_sha256",
        "projected_artifact_set_sha256",
        "schema_version",
    }
)


class ArtifactPipelineError(RuntimeError):
    """Raised when the fixed artifact sequence cannot be sealed or reproduced."""


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
        raise ArtifactPipelineError("pipeline records must be finite canonical JSON") from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactPipelineError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactPipelineError(f"{name} must be a positive integer")
    return value


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ArtifactPipelineError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise ArtifactPipelineError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactPipelineError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ArtifactPipelineError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactPipelineError(f"{label} must be UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ArtifactPipelineError(f"{label} must contain one object")
    return value


def _absolute_directory(path: str | Path, *, label: str) -> Path:
    root = Path(path)
    if not root.is_absolute() or root.name in {"", ".", ".."}:
        raise ArtifactPipelineError(f"{label} must be an absolute directory")
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise ArtifactPipelineError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ArtifactPipelineError(f"{label} must be a real directory")
    return root


def _read_control(path: Path, *, label: str) -> bytes:
    try:
        return read_secure_regular_file(path, max_bytes=_MAX_CONTROL_BYTES, label=label)
    except ArtifactIntegrityError as exc:
        raise ArtifactPipelineError(f"cannot read {label}: {exc}") from exc


def _load_query_receipt(path: Path) -> QueryTrialStoreReceipt:
    encoded = _read_control(path, label="query/trial receipt")
    try:
        receipt = QueryTrialStoreReceipt.from_dict(
            _decode_object(encoded, label="query/trial receipt")
        )
    except TrialRuntimeError as exc:
        raise ArtifactPipelineError(f"invalid query/trial receipt: {exc}") from exc
    if encoded != receipt.canonical_file_bytes():
        raise ArtifactPipelineError("query/trial receipt is not canonical")
    return receipt


def _load_query_rows(
    path: Path,
    receipt: QueryTrialStoreReceipt,
) -> tuple[CanonicalQueryTrialRow, ...]:
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=receipt.query_trial_store_byte_count,
            label="query/trial store",
        )
    except ArtifactIntegrityError as exc:
        raise ArtifactPipelineError(f"cannot read query/trial store: {exc}") from exc
    if (
        len(encoded) != receipt.query_trial_store_byte_count
        or _sha256(encoded) != receipt.query_trial_store_sha256
        or not encoded.endswith(b"\n")
    ):
        raise ArtifactPipelineError("query/trial store differs from its receipt")
    rows: list[CanonicalQueryTrialRow] = []
    for position, line in enumerate(encoded.splitlines(keepends=True)):
        if not line.endswith(b"\n") or line == b"\n":
            raise ArtifactPipelineError("query/trial store has a non-canonical line")
        value = _decode_object(line[:-1], label=f"query/trial row {position}")
        try:
            row = CanonicalQueryTrialRow.from_dict(value)
        except TrialRuntimeError as exc:
            raise ArtifactPipelineError(f"invalid query/trial row {position}: {exc}") from exc
        if line != _canonical_bytes(row.to_dict()) + b"\n":
            raise ArtifactPipelineError("query/trial row is not canonical")
        rows.append(row)
    values = tuple(rows)
    if (
        len(values) != receipt.record_count
        or tuple(row.opaque_row for row in values) != receipt.opaque_trials
    ):
        raise ArtifactPipelineError("query/trial rows differ from their typed receipt")
    return values


def _epoch_matches_vector(
    epoch: QueryVectorEpochBinding,
    vector: VectorDescriptor,
    *,
    role: str,
) -> bool:
    return (
        epoch.role,
        epoch.file_sha256,
        epoch.row_order_sha256,
        epoch.model_tree_sha256,
        epoch.model_revision,
        epoch.prompt_sha256,
        epoch.dtype,
        epoch.shape,
    ) == (
        role,
        vector.file_sha256,
        vector.row_order_sha256,
        vector.model_tree_sha256,
        vector.model_revision,
        vector.prompt_sha256,
        vector.dtype,
        vector.shape,
    )


@dataclass(frozen=True)
class RuntimePackageVerification:
    """Closed identity for the label-free sealed runtime package."""

    tree_sha256: str
    execution_plan_sha256: str
    query_receipt_sha256: str
    runtime_receipt_sha256: str
    query_count: int
    schema_version: str = RUNTIME_PACKAGE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "tree_sha256",
            "execution_plan_sha256",
            "query_receipt_sha256",
            "runtime_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _positive_integer("query_count", self.query_count)
        if self.schema_version != RUNTIME_PACKAGE_SCHEMA:
            raise ArtifactPipelineError("runtime package schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_plan_sha256": self.execution_plan_sha256,
            "query_count": self.query_count,
            "query_receipt_sha256": self.query_receipt_sha256,
            "runtime_receipt_sha256": self.runtime_receipt_sha256,
            "schema_version": self.schema_version,
            "tree_sha256": self.tree_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimePackageVerification:
        row = _closed(value, _RUNTIME_FIELDS, label="runtime package verification")
        return cls(**row)  # type: ignore[arg-type]


def verify_runtime_package(
    root: str | Path,
    *,
    corpus_id: str,
    online_inventory_sha256: str,
    embedding_receipt: EmbeddingStoreReceipt,
    policy_bundle: PolicyStageBundleReceipt,
    index_bundle: IndexStageBundleReceipt,
) -> RuntimePackageVerification:
    """Verify the sealed runtime package without opening staged source data."""

    runtime_root = _absolute_directory(root, label="runtime package root")
    if corpus_id not in FIXED_CORPORA:
        raise ArtifactPipelineError("runtime corpus is outside the fixed suite")
    _require_sha256("online_inventory_sha256", online_inventory_sha256)
    try:
        tree = digest_directory_tree(runtime_root)
    except ArtifactIntegrityError as exc:
        raise ArtifactPipelineError(f"cannot hash runtime package: {exc}") from exc
    if set(tree.entries) != set(_RUNTIME_EXPECTED_ENTRIES):
        raise ArtifactPipelineError(
            "runtime package membership differs; "
            f"missing={sorted(_RUNTIME_EXPECTED_ENTRIES - set(tree.entries))}, "
            f"extra={sorted(set(tree.entries) - _RUNTIME_EXPECTED_ENTRIES)}"
        )
    try:
        plan = load_sharded_online_execution_plan(runtime_root / SHARDED_PLAN_FILENAME)
        runtime = load_trial_runtime_receipt(runtime_root / TRIAL_RUNTIME_ADMISSION_FILENAME)
    except (ScalableExecutionError, TrialRuntimeError) as exc:
        raise ArtifactPipelineError(f"invalid typed runtime control: {exc}") from exc
    query_root = runtime_root / QUERY_PACKAGE_DIRECTORY
    query = _load_query_receipt(query_root / QUERY_TRIAL_RECEIPT_FILENAME)
    _load_query_rows(query_root / QUERY_TRIAL_FILENAME, query)
    sealed_policy = policy_bundle.stages[-1]
    sealed_index = index_bundle.stages[-1]
    old_documents = embedding_receipt.vectors.get("old_documents")
    current_documents = embedding_receipt.vectors.get("current_documents")
    old_queries = embedding_receipt.vectors.get("old_queries")
    current_queries = embedding_receipt.vectors.get("current_queries")
    if any(
        descriptor is None
        for descriptor in (old_documents, current_documents, old_queries, current_queries)
    ):
        raise ArtifactPipelineError("runtime requires a closed dual-model embedding store")
    assert old_documents is not None
    assert current_documents is not None
    assert old_queries is not None
    assert current_queries is not None
    if (
        plan.corpus != corpus_id
        or plan.stage != "sealed"
        or query.corpus != corpus_id
        or query.stage != "sealed"
        or plan.query_trial_store.artifact.relative_path
        != f"{QUERY_PACKAGE_DIRECTORY}/{QUERY_TRIAL_FILENAME}"
        or plan.query_trial_store.receipt.relative_path
        != f"{QUERY_PACKAGE_DIRECTORY}/{QUERY_TRIAL_RECEIPT_FILENAME}"
        or plan.query_trial_store.artifact.sha256 != query.query_trial_store_sha256
        or plan.query_trial_store.artifact.byte_count != query.query_trial_store_byte_count
        or plan.query_trial_store.receipt.sha256 != query.receipt_sha256
        or plan.query_trial_store.receipt.byte_count != query.receipt_byte_count
        or plan.trials != tuple(sorted(query.opaque_trials, key=lambda row: row.trial_key))
        or runtime.execution_artifact_sha256 != plan.artifact_sha256
        or runtime.query_trial_store_sha256 != query.query_trial_store_sha256
        or runtime.query_count != query.record_count
        or sealed_policy.trial_count != query.record_count
        or query.query_partition_audit_sha256 != plan.query_partition_audit_sha256
        or runtime.query_partition_audit_sha256 != plan.query_partition_audit_sha256
        or runtime.permutation_seed != plan.permutation_seed
        or query.embedding_store_receipt_sha256 != embedding_receipt.receipt_sha256
        or runtime.embedding_store_receipt_sha256 != embedding_receipt.receipt_sha256
        or query.staged_inventory_sha256 != online_inventory_sha256
        or runtime.staged_inventory_sha256 != online_inventory_sha256
        or query.source_inventory_sha256 != embedding_receipt.source_inventory_sha256
        or runtime.source_inventory_sha256 != embedding_receipt.source_inventory_sha256
        or runtime.active_query_epoch != query.active_query_epoch
        or runtime.current_truth_query_epoch != query.current_truth_query_epoch
        or not _epoch_matches_vector(
            runtime.active_query_epoch,
            old_queries,
            role="active-migration",
        )
        or not _epoch_matches_vector(
            runtime.current_truth_query_epoch,
            current_queries,
            role="current-exact-truth",
        )
        or plan.document_count != embedding_receipt.document_count
        or plan.ordered_document_universe_sha256
        != embedding_receipt.row_orders["documents"].row_order_sha256
        or plan.active_vector_store.artifact.sha256 != old_documents.file_sha256
        or plan.current_truth_vector_store.artifact.sha256 != current_documents.file_sha256
        or runtime.active_query_epoch.file_sha256 != old_queries.file_sha256
        or runtime.current_truth_query_epoch.file_sha256 != current_queries.file_sha256
        or runtime.policy_bundle_revision != sealed_policy.policy_revision
        or runtime.policy_config_sha256 != sealed_policy.config_sha256
        or runtime.mask_catalog_sha256 != sealed_policy.catalog_sha256
        or runtime.schedule_sha256 != sealed_policy.schedule_sha256
        or runtime.assignment_seed_sha256 != sealed_policy.assignment_seed_sha256
        or runtime.assignment_map_sha256 != sealed_policy.assignment_map_sha256
        or sealed_index.embedding_receipt_sha256 != embedding_receipt.receipt_sha256
        or sealed_index.policy_receipt_sha256 != sealed_policy.receipt_sha256
        or sealed_index.old_active_vector_sha256 != old_documents.file_sha256
        or sealed_index.current_truth_vector_sha256 != current_documents.file_sha256
        or sealed_index.document_universe_sha256 != plan.ordered_document_universe_sha256
    ):
        raise ArtifactPipelineError("runtime package differs from its staged artifact chain")
    try:
        final_tree = digest_directory_tree(runtime_root)
    except ArtifactIntegrityError as exc:
        raise ArtifactPipelineError(f"cannot rehash runtime package: {exc}") from exc
    if final_tree != tree:
        raise ArtifactPipelineError("runtime package changed during typed verification")
    return RuntimePackageVerification(
        tree_sha256=tree.sha256,
        execution_plan_sha256=plan.artifact_sha256,
        query_receipt_sha256=query.receipt_sha256,
        runtime_receipt_sha256=runtime.receipt_sha256,
        query_count=query.record_count,
    )


@dataclass(frozen=True)
class CorpusPipelineReceipt:
    """One corpus row in the fixed embedding-policy-index-runtime chain."""

    corpus_id: str
    embedding_tree_sha256: str
    embedding_receipt_sha256: str
    policy_bundle_tree_sha256: str
    policy_bundle_receipt_sha256: str
    index_bundle_tree_sha256: str
    index_bundle_receipt_sha256: str
    runtime: RuntimePackageVerification
    schema_version: str = CORPUS_PIPELINE_SCHEMA

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ArtifactPipelineError("pipeline corpus is outside the fixed suite")
        for name in (
            "embedding_tree_sha256",
            "embedding_receipt_sha256",
            "policy_bundle_tree_sha256",
            "policy_bundle_receipt_sha256",
            "index_bundle_tree_sha256",
            "index_bundle_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.runtime, RuntimePackageVerification):
            raise TypeError("runtime must be a RuntimePackageVerification")
        if self.schema_version != CORPUS_PIPELINE_SCHEMA:
            raise ArtifactPipelineError("corpus pipeline schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "embedding_receipt_sha256": self.embedding_receipt_sha256,
            "embedding_tree_sha256": self.embedding_tree_sha256,
            "index_bundle_receipt_sha256": self.index_bundle_receipt_sha256,
            "index_bundle_tree_sha256": self.index_bundle_tree_sha256,
            "policy_bundle_receipt_sha256": self.policy_bundle_receipt_sha256,
            "policy_bundle_tree_sha256": self.policy_bundle_tree_sha256,
            "runtime": self.runtime.to_dict(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> CorpusPipelineReceipt:
        row = _closed(value, _CORPUS_FIELDS, label="corpus pipeline receipt")
        return cls(
            corpus_id=row["corpus_id"],
            embedding_tree_sha256=row["embedding_tree_sha256"],
            embedding_receipt_sha256=row["embedding_receipt_sha256"],
            policy_bundle_tree_sha256=row["policy_bundle_tree_sha256"],
            policy_bundle_receipt_sha256=row["policy_bundle_receipt_sha256"],
            index_bundle_tree_sha256=row["index_bundle_tree_sha256"],
            index_bundle_receipt_sha256=row["index_bundle_receipt_sha256"],
            runtime=RuntimePackageVerification.from_dict(row["runtime"]),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class ArtifactPipelineReceipt:
    """Canonical receipt for all five corpus artifact chains."""

    online_inventory_sha256: str
    projected_artifact_set_sha256: str
    corpora: tuple[CorpusPipelineReceipt, ...]
    artifact_order: tuple[str, ...] = ARTIFACT_PIPELINE_ORDER
    schema_version: str = ARTIFACT_PIPELINE_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("online_inventory_sha256", self.online_inventory_sha256)
        _require_sha256("projected_artifact_set_sha256", self.projected_artifact_set_sha256)
        corpora = tuple(self.corpora)
        if (
            not all(isinstance(row, CorpusPipelineReceipt) for row in corpora)
            or tuple(row.corpus_id for row in corpora) != FIXED_CORPORA
        ):
            raise ArtifactPipelineError("pipeline receipt must cover the fixed corpus order")
        if tuple(self.artifact_order) != ARTIFACT_PIPELINE_ORDER:
            raise ArtifactPipelineError("pipeline artifact order differs")
        if self.schema_version != ARTIFACT_PIPELINE_SCHEMA:
            raise ArtifactPipelineError("artifact pipeline schema differs")
        object.__setattr__(self, "corpora", corpora)
        object.__setattr__(self, "artifact_order", tuple(self.artifact_order))

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_order": list(self.artifact_order),
            "corpora": [row.to_dict() for row in self.corpora],
            "online_inventory_sha256": self.online_inventory_sha256,
            "projected_artifact_set_sha256": self.projected_artifact_set_sha256,
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ArtifactPipelineReceipt:
        row = _closed(value, _PIPELINE_FIELDS, label="artifact pipeline receipt")
        corpora = row["corpora"]
        order = row["artifact_order"]
        if not isinstance(corpora, list) or not isinstance(order, list):
            raise ArtifactPipelineError("pipeline corpora and artifact_order must be arrays")
        return cls(
            online_inventory_sha256=row["online_inventory_sha256"],
            projected_artifact_set_sha256=row["projected_artifact_set_sha256"],
            corpora=tuple(CorpusPipelineReceipt.from_dict(item) for item in corpora),
            artifact_order=tuple(order),
            schema_version=row["schema_version"],
        )


def inspect_artifact_pipeline(
    artifact_root: str | Path,
    online_staging_root: str | Path,
    *,
    expected_online_inventory_sha256: str,
    seal_bundles: bool = False,
) -> ArtifactPipelineReceipt:
    """Inspect or one-time seal all package rows after label-free staging admission."""

    artifacts = _absolute_directory(artifact_root, label="artifact root")
    staging = _absolute_directory(online_staging_root, label="online staging root")
    try:
        projection = verify_online_staging_projection(
            staging,
            expected_inventory_sha256=expected_online_inventory_sha256,
        )
    except StudyDataError as exc:
        raise ArtifactPipelineError(f"online staging projection failed: {exc}") from exc
    corpus_rows: list[CorpusPipelineReceipt] = []
    for corpus_id in FIXED_CORPORA:
        embedding_root = artifacts / "embedding-stores" / corpus_id
        policy_root = artifacts / "policy-workloads" / corpus_id
        index_root = artifacts / "authorized-index-stores" / corpus_id
        runtime_root = artifacts / "trial-runtime" / corpus_id
        try:
            embedding = verify_embedding_store(embedding_root)
            embedding_tree = digest_directory_tree(embedding_root)
            policy = (
                seal_policy_stage_bundle(policy_root, corpus_id=corpus_id)
                if seal_bundles
                else verify_policy_stage_bundle(
                    policy_root,
                    expected_corpus_id=corpus_id,
                )
            )
            policy_tree = digest_directory_tree(policy_root)
            index = (
                seal_index_stage_bundle(
                    index_root,
                    corpus_id=corpus_id,
                    embedding_store_root=embedding_root,
                    policy_bundle_root=policy_root,
                )
                if seal_bundles
                else verify_index_stage_bundle(
                    index_root,
                    embedding_store_root=embedding_root,
                    policy_bundle_root=policy_root,
                    expected_corpus_id=corpus_id,
                )
            )
            index_tree = digest_directory_tree(index_root)
        except (
            ArtifactIntegrityError,
            ArtifactStageBundleError,
            EmbeddingStoreError,
        ) as exc:
            raise ArtifactPipelineError(f"{corpus_id} artifact package failed: {exc}") from exc
        if embedding.staged_inventory_sha256 != projection.inventory_sha256:
            raise ArtifactPipelineError(
                f"{corpus_id} embedding store was not built from the admitted online inventory"
            )
        runtime = verify_runtime_package(
            runtime_root,
            corpus_id=corpus_id,
            online_inventory_sha256=projection.inventory_sha256,
            embedding_receipt=embedding,
            policy_bundle=policy,
            index_bundle=index,
        )
        corpus_rows.append(
            CorpusPipelineReceipt(
                corpus_id=corpus_id,
                embedding_tree_sha256=embedding_tree.sha256,
                embedding_receipt_sha256=embedding.receipt_sha256,
                policy_bundle_tree_sha256=policy_tree.sha256,
                policy_bundle_receipt_sha256=policy.receipt_sha256,
                index_bundle_tree_sha256=index_tree.sha256,
                index_bundle_receipt_sha256=index.receipt_sha256,
                runtime=runtime,
            )
        )
    return ArtifactPipelineReceipt(
        online_inventory_sha256=projection.inventory_sha256,
        projected_artifact_set_sha256=projection.projected_artifact_set_sha256,
        corpora=tuple(corpus_rows),
    )


def load_artifact_pipeline_receipt(path: str | Path) -> ArtifactPipelineReceipt:
    encoded = _read_control(Path(path), label="artifact pipeline receipt")
    receipt = ArtifactPipelineReceipt.from_dict(
        _decode_object(encoded, label="artifact pipeline receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ArtifactPipelineError("artifact pipeline receipt is not canonical")
    return receipt


def build_artifact_pipeline(
    artifact_root: str | Path,
    online_staging_root: str | Path,
    receipt_path: str | Path,
    *,
    expected_online_inventory_sha256: str,
) -> ArtifactPipelineReceipt:
    """Seal stage metadata, verify the full sequence, and publish one receipt once."""

    receipt = inspect_artifact_pipeline(
        artifact_root,
        online_staging_root,
        expected_online_inventory_sha256=expected_online_inventory_sha256,
        seal_bundles=True,
    )
    try:
        write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), receipt_path)
    except ArtifactIntegrityError as exc:
        raise ArtifactPipelineError(f"cannot publish artifact pipeline receipt: {exc}") from exc
    return receipt


def verify_artifact_pipeline(
    artifact_root: str | Path,
    online_staging_root: str | Path,
    receipt_path: str | Path,
) -> ArtifactPipelineReceipt:
    """Reproduce a prior pipeline receipt without writing any artifact."""

    expected = load_artifact_pipeline_receipt(receipt_path)
    observed = inspect_artifact_pipeline(
        artifact_root,
        online_staging_root,
        expected_online_inventory_sha256=expected.online_inventory_sha256,
        seal_bundles=False,
    )
    if observed.canonical_file_bytes() != expected.canonical_file_bytes():
        raise ArtifactPipelineError("artifact pipeline differs from its sealed receipt")
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.artifact_pipeline",
        description="seal or verify the fixed label-free artifact pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="seal stage bundles and write a suite receipt")
    build.add_argument("--artifact-root", required=True, type=Path)
    build.add_argument("--online-staging-root", required=True, type=Path)
    build.add_argument("--expected-online-inventory-sha256", required=True)
    build.add_argument("--receipt", required=True, type=Path)
    verify = subparsers.add_parser("verify", help="reproduce an existing suite receipt")
    verify.add_argument("--artifact-root", required=True, type=Path)
    verify.add_argument("--online-staging-root", required=True, type=Path)
    verify.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            receipt = build_artifact_pipeline(
                args.artifact_root,
                args.online_staging_root,
                args.receipt,
                expected_online_inventory_sha256=(args.expected_online_inventory_sha256),
            )
        else:
            receipt = verify_artifact_pipeline(
                args.artifact_root,
                args.online_staging_root,
                args.receipt,
            )
    except ArtifactPipelineError as exc:
        parser = _parser()
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "corpus_count": len(receipt.corpora),
                "receipt_sha256": receipt.receipt_sha256,
                "schema_version": receipt.schema_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
