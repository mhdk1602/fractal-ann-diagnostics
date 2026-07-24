"""Closed, resumable construction of post-embedding production artifacts.

This module owns the mechanical boundary between the verified five-corpus
paired embedding suite and the immutable online packages.  It derives the
corpus and stage order from the protocol, calls the typed policy, index, query,
runtime, and artifact-pipeline builders, and records resource evidence.  The
command has no label, qrel, evidence, custody, plugin, or callback argument.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import platform
import re
import resource
import secrets
import shutil
import stat
import sys
import time
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal

import numpy as np

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .artifact_pipeline import (
    ArtifactPipelineError,
    ArtifactPipelineReceipt,
    build_artifact_pipeline,
    verify_artifact_pipeline,
)
from .artifact_stage_bundles import (
    ARTIFACT_STAGE_ORDER,
    POLICY_SOURCE_STAGE_BY_BUNDLE_STAGE,
    ArtifactStageBundleError,
    seal_index_stage_bundle,
    seal_policy_stage_bundle,
    verify_index_stage_bundle,
    verify_policy_stage_bundle,
)
from .authorized_index_store import (
    AuthorizedIndexConfig,
    AuthorizedIndexStoreError,
    AuthorizedIndexStoreReceipt,
    HnswlibBackend,
    build_authorized_index_store,
    load_authorized_index_store_receipt,
    open_verified_document_matrices,
    verify_authorized_index_store,
)
from .development_cohort import (
    DevelopmentCohortError,
    DevelopmentExecutionPlan,
    load_development_execution_plan,
    verify_materialized_development_cohort,
)
from .embedding_store import EmbeddingStoreError, EmbeddingStoreReceipt, verify_embedding_store
from .joint_power_design import (
    FIXED_CORPORA,
    JointPowerDesignError,
    load_joint_power_report,
)
from .policy_intervention import (
    SCHEDULE_FILENAME,
    PolicyInterventionConfig,
    PolicyInterventionError,
    load_canonical_trial_schedule,
    load_policy_intervention_receipt,
    verify_policy_intervention_package,
    write_policy_intervention_package,
)
from .production_embedding_build import (
    ProductionCorpusSources,
    ProductionEmbeddingBuildError,
    ProductionEmbeddingConfig,
    ProductionEmbeddingSuiteReceipt,
    admit_frozen_production_embedding_suite,
    load_production_embedding_config,
)
from .scalable_custody import _content_sha256 as custody_document_content_sha256
from .scalable_execution import (
    ONLINE_EXECUTION_PLAN_FILENAME,
    CorpusShard,
    CorpusShardInventory,
    ImmutableArtifactPin,
    IndexArtifactDescriptor,
    ProvenanceSidecarDescriptor,
    ScalableExecutionError,
    ShardedOnlineExecutionPlan,
    VectorStoreDescriptor,
    finalize_online_execution_package,
    load_sharded_online_execution_plan,
    verify_online_execution_package,
    write_corpus_shard_inventory,
    write_sharded_online_execution_plan,
)
from .scalable_partition_audit import (
    ScalablePartitionAuditError,
    ScalableQueryPartitionAuditReceipt,
    load_scalable_partition_audit,
)
from .study_data import ASSIGNMENT_SCHEMA, StudyDataError, verify_online_staging_projection
from .trial_runtime import (
    QUERY_TRIAL_FILENAME,
    QUERY_TRIAL_RECEIPT_FILENAME,
    RuntimeFeatureBinding,
    TrialRuntimeError,
    admit_trial_runtime,
    build_query_trial_store,
    load_trial_runtime_receipt,
    verify_query_trial_store,
)

PRODUCTION_FACTORY_CONFIG_SCHEMA = "fractal-production-artifact-factory-config-v1"
PRODUCTION_FACTORY_CORPUS_EVIDENCE_SCHEMA = "fractal-production-artifact-factory-corpus-evidence-v1"
PRODUCTION_FACTORY_SUITE_SCHEMA = "fractal-production-artifact-factory-suite-v1"
PRODUCTION_FACTORY_SHARD_REQUEST_SCHEMA = "fractal-production-artifact-factory-shard-request-v1"
PRODUCTION_FACTORY_SHARD_RECEIPT_SCHEMA = "fractal-production-artifact-factory-shard-receipt-v1"
PRODUCTION_FACTORY_SHARD_ARTIFACT_SCHEMA = "fractal-production-artifact-factory-shard-artifact-v1"
INDEX_REPRODUCIBILITY_STAGE_SCHEMA = "fractal-authorized-index-reproducibility-stage-v1"
FULL_HNSW_REPLICATE_SCHEMA = "fractal-full-hnsw-replicate-v1"
FULL_HNSW_REPRODUCIBILITY_SCHEMA = "fractal-full-hnsw-reproducibility-v1"
INDEX_REPRODUCIBILITY_SUITE_SCHEMA = "fractal-index-reproducibility-suite-v2"

FACTORY_EVIDENCE_DIRECTORY = "production-factory-evidence"
FACTORY_SUITE_FILENAME = "production-artifact-factory-receipt.json"
ARTIFACT_PIPELINE_RECEIPT_FILENAME = "artifact-pipeline-receipt.json"
INDEX_REPRODUCIBILITY_DIRECTORY = "authorized-index-reproducibility"
INDEX_REPRODUCIBILITY_SUITE_FILENAME = "authorized-index-reproducibility-receipt.json"
INDEX_REPLICATE_COUNT = 3
SELECTED_INDEX_REPLICATE = 1
INDEX_REPLICATE_DIRECTORIES = tuple(
    f"replicate-{replicate:02d}" for replicate in range(1, INDEX_REPLICATE_COUNT + 1)
)
FACTORY_RUNNER_PLATFORM = "linux/arm64"
FACTORY_INDEX_METRIC = "cosine"
FACTORY_INDEX_M = 16
FACTORY_INDEX_EF_CONSTRUCTION = 128
FACTORY_INDEX_RANDOM_SEED = 20260714
FACTORY_INDEX_BATCH_SIZE = 512
FACTORY_INDEX_VERIFICATION_EF = 64
FACTORY_INDEX_NUM_THREADS = 1
FACTORY_DESIGN_DERIVATION = "fractal-production-artifact-factory-design-v1"

RUNTIME_QUERY_DIRECTORY = "query-package"
RUNTIME_RECEIPT_FILENAME = "trial-runtime-admission-receipt.json"
ONLINE_INVENTORY_PATH = "corpus/shard-inventory.json"
ONLINE_PROVENANCE_PATH = "provenance/document-content-sha256.bin"
ONLINE_ACTIVE_VECTOR_PATH = "vectors/old-documents.f32"
ONLINE_TRUTH_VECTOR_PATH = "vectors/current-documents.f32"
ONLINE_HNSW_PATH = "indexes/full-active.hnsw"
FULL_HNSW_REPRODUCIBILITY_DIRECTORY = "full-active"
FULL_HNSW_REPLICATE_FILENAME = "full-active.hnsw"
FULL_HNSW_REPLICATE_RECEIPT_FILENAME = "replicate-evidence.json"
FULL_HNSW_REPRODUCIBILITY_RECEIPT_FILENAME = "reproducibility-receipt.json"
POLICY_SUBJECT_ID = "confirmatory-reader"
RUNTIME_BACKEND = "hnsw"
RUNTIME_DRIFT_FAMILY = "qwen-revision-lag"
VERSION_LAG = 1.0

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OCI_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 256 * 1024 * 1024
_READ_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_LINE_BYTES = 16 * 1024 * 1024
_CONFIG_FIELDS = frozenset(
    {
        "artifact_root",
        "artifact_stage_order",
        "baseline_policy_bundle_revision",
        "baseline_policy_seed_sha256",
        "corpora",
        "development_materialization_receipt_sha256",
        "development_materialization_root",
        "development_operator_joint_power_report_tree_sha256",
        "development_operator_receipt_sha256",
        "development_operator_root",
        "design_seed_sha256",
        "embedding_build_config_path",
        "embedding_build_config_sha256",
        "embedding_source_root",
        "embedding_source_tree_sha256",
        "embedding_suite_receipt_sha256",
        "hmac_key_id",
        "hmac_secret_sha256",
        "index_config",
        "index_replicate_count",
        "index_replicate_directories",
        "joint_power_report_path",
        "joint_power_report_sha256",
        "partition_audit_path",
        "partition_audit_sha256",
        "permutation_seed",
        "policy_bundle_revision",
        "policy_seed_sha256",
        "runner_image",
        "runner_platform",
        "schema_version",
        "selection_seed_sha256",
        "selected_family_count",
        "selected_index_replicate",
    }
)
_CORPUS_CONFIG_FIELDS = frozenset({"available_family_count", "corpus_id"})
_EVIDENCE_FIELDS = frozenset(
    {
        "completed_at_utc",
        "corpus_id",
        "elapsed_monotonic_ns",
        "embedding_receipt_sha256",
        "factory_config_sha256",
        "index_bundle_receipt_sha256",
        "online_execution_plan_sha256",
        "online_execution_tree_sha256",
        "policy_bundle_receipt_sha256",
        "process_peak_rss_bytes",
        "query_receipt_sha256",
        "runtime_receipt_sha256",
        "schema_version",
        "started_at_utc",
        "status",
    }
)
_SUITE_CORPUS_FIELDS = frozenset(
    {
        "corpus_id",
        "evidence_file_sha256",
        "evidence_sha256",
        "index_bundle_receipt_sha256",
        "online_execution_plan_sha256",
        "online_execution_tree_sha256",
        "policy_bundle_receipt_sha256",
        "query_receipt_sha256",
        "runtime_receipt_sha256",
    }
)
_SUITE_FIELDS = frozenset(
    {
        "artifact_pipeline_receipt_sha256",
        "corpora",
        "embedding_destination_tree_sha256",
        "embedding_source_tree_sha256",
        "embedding_suite_receipt_sha256",
        "factory_config_sha256",
        "hmac_key_id",
        "hmac_secret_sha256",
        "index_reproducibility_receipt_sha256",
        "online_inventory_sha256",
        "runner_image",
        "runner_platform",
        "schema_version",
    }
)
_SHARD_REQUEST_FIELDS = frozenset(
    {
        "artifact_root",
        "corpus_id",
        "embedding_source_tree_sha256",
        "embedding_suite_receipt_sha256",
        "factory_config_sha256",
        "hmac_key_id",
        "hmac_secret_sha256",
        "owned_relative_paths",
        "runner_image",
        "runner_platform",
        "schema_version",
    }
)
_SHARD_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_kind",
        "relative_path",
        "schema_version",
        "sha256",
    }
)
_SHARD_RECEIPT_FIELDS = frozenset(
    {
        "artifacts",
        "corpus_evidence_file_sha256",
        "corpus_evidence_sha256",
        "corpus_id",
        "factory_config_sha256",
        "hmac_key_id",
        "hmac_secret_sha256",
        "request_sha256",
        "runner_image",
        "runner_platform",
        "schema_version",
    }
)
_REPLICATE_FIELDS = frozenset(
    {
        "elapsed_monotonic_ns",
        "index_payloads",
        "process_peak_rss_bytes",
        "replicate",
        "receipt_sha256",
        "tree_sha256",
    }
)
_INDEX_PAYLOAD_FIELDS = frozenset(
    {
        "build_binding_sha256",
        "index_sha256",
        "mask_id",
        "row_map_sha256",
    }
)
_REPRO_STAGE_FIELDS = frozenset(
    {
        "backend_build_sha256",
        "backend_id",
        "backend_version",
        "corpus_id",
        "factory_config_sha256",
        "replicates",
        "runner_image",
        "runner_platform",
        "schema_version",
        "selected_final_receipt_sha256",
        "selected_replicate",
        "stage",
    }
)
_REPRO_SUITE_FIELDS = frozenset(
    {
        "factory_config_sha256",
        "full_hnsw_indexes",
        "replicate_count",
        "runner_image",
        "runner_platform",
        "schema_version",
        "selected_replicate",
        "stages",
    }
)
_FULL_HNSW_REPLICATE_FIELDS = frozenset(
    {
        "byte_count",
        "elapsed_monotonic_ns",
        "process_peak_rss_bytes",
        "relative_path",
        "replicate",
        "schema_version",
        "sha256",
    }
)
_FULL_HNSW_REPRODUCIBILITY_FIELDS = frozenset(
    {
        "backend_build_sha256",
        "backend_id",
        "backend_version",
        "corpus_id",
        "dimension",
        "document_count",
        "factory_config_sha256",
        "format_revision",
        "replicates",
        "runner_image",
        "runner_platform",
        "schema_version",
        "selected_final_sha256",
        "selected_replicate",
        "source_vector_sha256",
    }
)
_ASSIGNMENT_FIELDS = frozenset(
    {
        "assignment_key_sha256",
        "dataset",
        "domain",
        "partition_component_sha256",
        "query_id",
        "query_text_sha256",
        "schema_version",
        "source_split",
        "stage",
    }
)


class ProductionArtifactFactoryError(RuntimeError):
    """Raised when the fixed post-embedding artifact chain cannot close."""


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
        raise ProductionArtifactFactoryError(
            "factory records must be finite canonical JSON"
        ) from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _hash_parts(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductionArtifactFactoryError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionArtifactFactoryError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionArtifactFactoryError(f"{name} must be a non-negative integer")
    return value


def _require_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ProductionArtifactFactoryError(f"{name} must be canonical non-empty text")
    return value


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProductionArtifactFactoryError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise ProductionArtifactFactoryError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProductionArtifactFactoryError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ProductionArtifactFactoryError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionArtifactFactoryError(f"{label} must be UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ProductionArtifactFactoryError(f"{label} must contain one object")
    return value


def _canonical_absolute_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProductionArtifactFactoryError(f"{label} must be an absolute POSIX path")
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or str(pure) != value
        or pure.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in pure.parts[1:])
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ProductionArtifactFactoryError(f"{label} must be an absolute canonical path")
    path = Path(value)
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot resolve {label}: {exc}") from exc
    if resolved != path:
        raise ProductionArtifactFactoryError(f"{label} crosses an alias or symbolic link")
    return path


def _read_control(path: Path, *, label: str) -> bytes:
    try:
        return read_secure_regular_file(path, max_bytes=_MAX_CONTROL_BYTES, label=label)
    except ArtifactIntegrityError as exc:
        raise ProductionArtifactFactoryError(f"cannot read {label}: {exc}") from exc


def _require_real_directory(path: Path, *, label: str, private: bool = False) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ProductionArtifactFactoryError(f"{label} must be a real directory")
    if private and (
        (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ProductionArtifactFactoryError(
            f"{label} must be runner-owned and not writable by group or others"
        )


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProductionArtifactFactoryError(f"{label} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProductionArtifactFactoryError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo != timezone.utc or parsed.isoformat().replace("+00:00", "Z") != value:
        raise ProductionArtifactFactoryError(f"{label} must be canonical UTC")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


@dataclass(frozen=True)
class ProductionCorpusFactoryConfig:
    """Only the frozen qrel-derived cohort denominator varies by corpus."""

    corpus_id: str
    available_family_count: int

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionArtifactFactoryError("factory corpus is outside the fixed suite")
        _require_positive_integer("available_family_count", self.available_family_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "available_family_count": self.available_family_count,
            "corpus_id": self.corpus_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProductionCorpusFactoryConfig:
        row = _closed(value, _CORPUS_CONFIG_FIELDS, label="factory corpus config")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _FactoryDesignBindings:
    policy_seed_sha256: str
    baseline_policy_seed_sha256: str
    selection_seed_sha256: str
    permutation_seed: int
    policy_bundle_revision: str
    baseline_policy_bundle_revision: str


def _derive_factory_design_bindings(design_seed_sha256: str) -> _FactoryDesignBindings:
    seed = _require_sha256("design_seed_sha256", design_seed_sha256)

    def derive(purpose: str) -> str:
        return _hash_parts(FACTORY_DESIGN_DERIVATION, seed, purpose)

    permutation_digest = derive("permutation-seed-u64-be")
    return _FactoryDesignBindings(
        policy_seed_sha256=derive("policy-seed"),
        baseline_policy_seed_sha256=derive("baseline-policy-seed"),
        selection_seed_sha256=derive("sealed-family-selection-seed"),
        permutation_seed=int.from_bytes(bytes.fromhex(permutation_digest)[:8], "big"),
        policy_bundle_revision=f"sha256:{derive('policy-bundle-revision')}",
        baseline_policy_bundle_revision=(f"sha256:{derive('baseline-policy-bundle-revision')}"),
    )


def derive_production_policy_config(
    design_seed_sha256: str,
    corpus_id: str,
    stage: str,
) -> PolicyInterventionConfig:
    """Derive the exact policy config shared by development and factory execution."""

    design = _derive_factory_design_bindings(design_seed_sha256)
    if corpus_id not in FIXED_CORPORA:
        raise ProductionArtifactFactoryError("policy corpus is outside FIXED_CORPORA")
    if stage not in ARTIFACT_STAGE_ORDER:
        raise ProductionArtifactFactoryError("policy stage is outside ARTIFACT_STAGE_ORDER")
    return PolicyInterventionConfig(
        seed_sha256=_hash_parts(
            "fractal-production-policy-seed-v1",
            design.policy_seed_sha256,
            corpus_id,
            stage,
        ),
        baseline_seed_sha256=_hash_parts(
            "fractal-production-baseline-policy-seed-v1",
            design.baseline_policy_seed_sha256,
            corpus_id,
            stage,
        ),
        policy_bundle_revision=design.policy_bundle_revision,
        baseline_policy_revision=design.baseline_policy_bundle_revision,
        subject_ids=(POLICY_SUBJECT_ID,),
        assignment_repetitions=1,
    )


@dataclass(frozen=True)
class ProductionArtifactFactoryConfig:
    """Canonical identity for one closed five-corpus artifact build."""

    artifact_root: Path
    artifact_stage_order: tuple[str, ...]
    embedding_build_config_path: Path
    embedding_build_config_sha256: str
    embedding_source_root: Path
    embedding_source_tree_sha256: str
    embedding_suite_receipt_sha256: str
    development_materialization_root: Path
    development_materialization_receipt_sha256: str
    development_operator_root: Path
    development_operator_receipt_sha256: str
    development_operator_joint_power_report_tree_sha256: str
    design_seed_sha256: str
    partition_audit_path: Path
    partition_audit_sha256: str
    joint_power_report_path: Path
    joint_power_report_sha256: str
    runner_image: str
    runner_platform: str
    policy_seed_sha256: str
    baseline_policy_seed_sha256: str
    policy_bundle_revision: str
    baseline_policy_bundle_revision: str
    selection_seed_sha256: str
    permutation_seed: int
    hmac_key_id: str
    hmac_secret_sha256: str
    index_config: AuthorizedIndexConfig
    index_replicate_count: int
    index_replicate_directories: tuple[str, ...]
    selected_family_count: int
    selected_index_replicate: int
    corpora: tuple[ProductionCorpusFactoryConfig, ...]
    schema_version: str = PRODUCTION_FACTORY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "artifact_root",
            "embedding_build_config_path",
            "embedding_source_root",
            "development_materialization_root",
            "development_operator_root",
            "partition_audit_path",
            "joint_power_report_path",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_absolute_path(str(getattr(self, name)), label=name),
            )
        for name in (
            "embedding_build_config_sha256",
            "embedding_source_tree_sha256",
            "embedding_suite_receipt_sha256",
            "development_materialization_receipt_sha256",
            "development_operator_receipt_sha256",
            "development_operator_joint_power_report_tree_sha256",
            "design_seed_sha256",
            "partition_audit_sha256",
            "joint_power_report_sha256",
            "policy_seed_sha256",
            "baseline_policy_seed_sha256",
            "selection_seed_sha256",
            "hmac_secret_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        design = _derive_factory_design_bindings(self.design_seed_sha256)
        if (
            self.policy_seed_sha256 != design.policy_seed_sha256
            or self.baseline_policy_seed_sha256 != design.baseline_policy_seed_sha256
            or self.selection_seed_sha256 != design.selection_seed_sha256
            or self.permutation_seed != design.permutation_seed
            or self.policy_bundle_revision != design.policy_bundle_revision
            or self.baseline_policy_bundle_revision != design.baseline_policy_bundle_revision
        ):
            raise ProductionArtifactFactoryError(
                "factory randomized values differ from the design-seed derivation"
            )
        if (
            not isinstance(self.runner_image, str)
            or _OCI_IMAGE.fullmatch(self.runner_image) is None
        ):
            raise ProductionArtifactFactoryError("runner_image must be an immutable OCI digest")
        if self.runner_platform != FACTORY_RUNNER_PLATFORM:
            raise ProductionArtifactFactoryError(
                f"runner_platform must equal {FACTORY_RUNNER_PLATFORM}"
            )
        for name in ("policy_bundle_revision", "baseline_policy_bundle_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise ProductionArtifactFactoryError(f"{name} must equal sha256:<digest>")
            _require_sha256(name, value[7:])
        if self.policy_bundle_revision == self.baseline_policy_bundle_revision:
            raise ProductionArtifactFactoryError(
                "current and baseline policy revisions must differ"
            )
        if (
            isinstance(self.permutation_seed, bool)
            or not isinstance(self.permutation_seed, int)
            or not 0 <= self.permutation_seed < 2**64
        ):
            raise ProductionArtifactFactoryError(
                "permutation_seed must be an unsigned 64-bit integer"
            )
        _require_text("hmac_key_id", self.hmac_key_id)
        expected_key_id = f"sealed-online-ephemeral-sha256-{self.hmac_secret_sha256}"
        if self.hmac_key_id != expected_key_id:
            raise ProductionArtifactFactoryError(
                "hmac_key_id must be derived from hmac_secret_sha256"
            )
        if not isinstance(self.index_config, AuthorizedIndexConfig):
            raise ProductionArtifactFactoryError("index_config must be typed")
        expected_index_values = (
            FACTORY_INDEX_METRIC,
            FACTORY_INDEX_M,
            FACTORY_INDEX_EF_CONSTRUCTION,
            FACTORY_INDEX_RANDOM_SEED,
            FACTORY_INDEX_BATCH_SIZE,
            FACTORY_INDEX_VERIFICATION_EF,
            FACTORY_INDEX_NUM_THREADS,
        )
        observed_index_values = (
            self.index_config.metric,
            self.index_config.m,
            self.index_config.ef_construction,
            self.index_config.random_seed,
            self.index_config.batch_size,
            self.index_config.verification_ef,
            self.index_config.num_threads,
        )
        if observed_index_values != expected_index_values:
            raise ProductionArtifactFactoryError(
                "index_config differs from the C0 production constants"
            )
        artifact_stage_order = tuple(self.artifact_stage_order)
        if artifact_stage_order != ARTIFACT_STAGE_ORDER:
            raise ProductionArtifactFactoryError(
                "artifact_stage_order must follow ARTIFACT_STAGE_ORDER"
            )
        object.__setattr__(self, "artifact_stage_order", artifact_stage_order)
        if self.index_replicate_count != INDEX_REPLICATE_COUNT:
            raise ProductionArtifactFactoryError("index_replicate_count must equal three")
        directories = tuple(self.index_replicate_directories)
        if directories != INDEX_REPLICATE_DIRECTORIES or len(set(directories)) != len(directories):
            raise ProductionArtifactFactoryError(
                "index_replicate_directories must name three distinct fixed roots"
            )
        object.__setattr__(self, "index_replicate_directories", directories)
        _require_positive_integer("selected_family_count", self.selected_family_count)
        if self.selected_index_replicate != SELECTED_INDEX_REPLICATE:
            raise ProductionArtifactFactoryError("selected_index_replicate must equal one")
        corpora = tuple(self.corpora)
        if tuple(row.corpus_id for row in corpora) != FIXED_CORPORA:
            raise ProductionArtifactFactoryError("factory corpora must follow FIXED_CORPORA")
        if any(row.available_family_count < self.selected_family_count for row in corpora):
            raise ProductionArtifactFactoryError(
                "selected_family_count exceeds an audited corpus denominator"
            )
        object.__setattr__(self, "corpora", corpora)
        inputs = (
            self.embedding_build_config_path,
            self.embedding_source_root,
            self.development_materialization_root,
            self.development_operator_root,
            self.partition_audit_path,
            self.joint_power_report_path,
        )
        if any(_paths_overlap(self.artifact_root, source) for source in inputs):
            raise ProductionArtifactFactoryError("artifact_root cannot overlap a source path")
        expected_development_root, expected_power_path = _development_operator_paths(
            self.development_operator_root
        )
        if self.development_materialization_root != expected_development_root:
            raise ProductionArtifactFactoryError(
                "development_materialization_root must be derived from development_operator_root"
            )
        if self.joint_power_report_path != expected_power_path:
            raise ProductionArtifactFactoryError(
                "joint_power_report_path must be derived from development_operator_root"
            )
        if self.schema_version != PRODUCTION_FACTORY_CONFIG_SCHEMA:
            raise ProductionArtifactFactoryError(
                f"schema_version must equal {PRODUCTION_FACTORY_CONFIG_SCHEMA!r}"
            )

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @property
    def pipeline_receipt_path(self) -> Path:
        return self.artifact_root / ARTIFACT_PIPELINE_RECEIPT_FILENAME

    @property
    def reproducibility_root(self) -> Path:
        return self.artifact_root / INDEX_REPRODUCIBILITY_DIRECTORY

    @property
    def reproducibility_receipt_path(self) -> Path:
        return self.artifact_root / INDEX_REPRODUCIBILITY_SUITE_FILENAME

    @property
    def evidence_root(self) -> Path:
        return self.artifact_root / FACTORY_EVIDENCE_DIRECTORY

    @property
    def suite_receipt_path(self) -> Path:
        return self.artifact_root / FACTORY_SUITE_FILENAME

    def corpus(self, corpus_id: str) -> ProductionCorpusFactoryConfig:
        try:
            return next(row for row in self.corpora if row.corpus_id == corpus_id)
        except StopIteration as exc:
            raise ProductionArtifactFactoryError("corpus is outside the factory config") from exc

    def policy_config(self, corpus_id: str, stage: str) -> PolicyInterventionConfig:
        return derive_production_policy_config(
            self.design_seed_sha256,
            corpus_id,
            stage,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_root": str(self.artifact_root),
            "artifact_stage_order": list(self.artifact_stage_order),
            "baseline_policy_bundle_revision": self.baseline_policy_bundle_revision,
            "baseline_policy_seed_sha256": self.baseline_policy_seed_sha256,
            "corpora": [row.to_dict() for row in self.corpora],
            "development_materialization_receipt_sha256": (
                self.development_materialization_receipt_sha256
            ),
            "development_materialization_root": str(self.development_materialization_root),
            "development_operator_joint_power_report_tree_sha256": (
                self.development_operator_joint_power_report_tree_sha256
            ),
            "development_operator_receipt_sha256": (self.development_operator_receipt_sha256),
            "development_operator_root": str(self.development_operator_root),
            "design_seed_sha256": self.design_seed_sha256,
            "embedding_build_config_path": str(self.embedding_build_config_path),
            "embedding_build_config_sha256": self.embedding_build_config_sha256,
            "embedding_source_root": str(self.embedding_source_root),
            "embedding_source_tree_sha256": self.embedding_source_tree_sha256,
            "embedding_suite_receipt_sha256": self.embedding_suite_receipt_sha256,
            "hmac_key_id": self.hmac_key_id,
            "hmac_secret_sha256": self.hmac_secret_sha256,
            "index_config": self.index_config.to_dict(),
            "index_replicate_count": self.index_replicate_count,
            "index_replicate_directories": list(self.index_replicate_directories),
            "joint_power_report_path": str(self.joint_power_report_path),
            "joint_power_report_sha256": self.joint_power_report_sha256,
            "partition_audit_path": str(self.partition_audit_path),
            "partition_audit_sha256": self.partition_audit_sha256,
            "permutation_seed": self.permutation_seed,
            "policy_bundle_revision": self.policy_bundle_revision,
            "policy_seed_sha256": self.policy_seed_sha256,
            "runner_image": self.runner_image,
            "runner_platform": self.runner_platform,
            "schema_version": self.schema_version,
            "selection_seed_sha256": self.selection_seed_sha256,
            "selected_family_count": self.selected_family_count,
            "selected_index_replicate": self.selected_index_replicate,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_dict(cls, value: object) -> ProductionArtifactFactoryConfig:
        row = _closed(value, _CONFIG_FIELDS, label="production artifact factory config")
        corpora = row["corpora"]
        if not isinstance(corpora, list):
            raise ProductionArtifactFactoryError("factory corpora must be an array")
        artifact_stage_order = row["artifact_stage_order"]
        if not isinstance(artifact_stage_order, list):
            raise ProductionArtifactFactoryError("artifact_stage_order must be an array")
        replicate_directories = row["index_replicate_directories"]
        if not isinstance(replicate_directories, list):
            raise ProductionArtifactFactoryError("index_replicate_directories must be an array")
        try:
            index_config = AuthorizedIndexConfig.from_dict(row["index_config"])
        except AuthorizedIndexStoreError as exc:
            raise ProductionArtifactFactoryError(f"invalid index config: {exc}") from exc
        return cls(
            artifact_root=_canonical_absolute_path(row["artifact_root"], label="artifact_root"),
            artifact_stage_order=tuple(artifact_stage_order),
            embedding_build_config_path=_canonical_absolute_path(
                row["embedding_build_config_path"], label="embedding_build_config_path"
            ),
            embedding_build_config_sha256=row["embedding_build_config_sha256"],
            embedding_source_root=_canonical_absolute_path(
                row["embedding_source_root"], label="embedding_source_root"
            ),
            embedding_source_tree_sha256=row["embedding_source_tree_sha256"],
            embedding_suite_receipt_sha256=row["embedding_suite_receipt_sha256"],
            development_materialization_root=_canonical_absolute_path(
                row["development_materialization_root"],
                label="development_materialization_root",
            ),
            development_materialization_receipt_sha256=(
                row["development_materialization_receipt_sha256"]
            ),
            development_operator_root=_canonical_absolute_path(
                row["development_operator_root"], label="development_operator_root"
            ),
            development_operator_receipt_sha256=row["development_operator_receipt_sha256"],
            development_operator_joint_power_report_tree_sha256=row[
                "development_operator_joint_power_report_tree_sha256"
            ],
            design_seed_sha256=row["design_seed_sha256"],
            partition_audit_path=_canonical_absolute_path(
                row["partition_audit_path"], label="partition_audit_path"
            ),
            partition_audit_sha256=row["partition_audit_sha256"],
            joint_power_report_path=_canonical_absolute_path(
                row["joint_power_report_path"], label="joint_power_report_path"
            ),
            joint_power_report_sha256=row["joint_power_report_sha256"],
            runner_image=row["runner_image"],
            runner_platform=row["runner_platform"],
            policy_seed_sha256=row["policy_seed_sha256"],
            baseline_policy_seed_sha256=row["baseline_policy_seed_sha256"],
            policy_bundle_revision=row["policy_bundle_revision"],
            baseline_policy_bundle_revision=row["baseline_policy_bundle_revision"],
            selection_seed_sha256=row["selection_seed_sha256"],
            permutation_seed=row["permutation_seed"],
            hmac_key_id=row["hmac_key_id"],
            hmac_secret_sha256=row["hmac_secret_sha256"],
            index_config=index_config,
            index_replicate_count=row["index_replicate_count"],
            index_replicate_directories=tuple(replicate_directories),
            selected_family_count=row["selected_family_count"],
            selected_index_replicate=row["selected_index_replicate"],
            corpora=tuple(ProductionCorpusFactoryConfig.from_dict(item) for item in corpora),
            schema_version=row["schema_version"],
        )


def load_production_artifact_factory_config(
    path: str | Path,
    *,
    expected_sha256: str,
) -> ProductionArtifactFactoryConfig:
    """Load one canonical self-contained config through an external digest pin."""

    _require_sha256("expected config SHA-256", expected_sha256)
    encoded = _read_control(Path(path), label="production artifact factory config")
    if _sha256(encoded) != expected_sha256:
        raise ProductionArtifactFactoryError("factory config differs from its caller pin")
    config = ProductionArtifactFactoryConfig.from_dict(
        _decode_object(encoded, label="production artifact factory config")
    )
    if encoded != config.canonical_file_bytes():
        raise ProductionArtifactFactoryError("factory config is not canonical")
    return config


def _shard_owned_relative_paths(corpus_id: str) -> tuple[str, ...]:
    if corpus_id not in FIXED_CORPORA:
        raise ProductionArtifactFactoryError("factory shard corpus is outside FIXED_CORPORA")
    return (
        f"policy-workloads/{corpus_id}",
        f"authorized-index-stores/{corpus_id}",
        f"trial-runtime/{corpus_id}",
        f"custody/online/{corpus_id}",
        f"{INDEX_REPRODUCIBILITY_DIRECTORY}/{corpus_id}",
        f"{FACTORY_EVIDENCE_DIRECTORY}/{corpus_id}.json",
    )


@dataclass(frozen=True)
class ProductionArtifactFactoryShardRequest:
    """One corpus lane derived solely from the complete factory config."""

    factory_config_sha256: str
    artifact_root: Path
    corpus_id: str
    runner_image: str
    runner_platform: str
    embedding_source_tree_sha256: str
    embedding_suite_receipt_sha256: str
    hmac_key_id: str
    hmac_secret_sha256: str
    owned_relative_paths: tuple[str, ...]
    schema_version: str = PRODUCTION_FACTORY_SHARD_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("factory_config_sha256", self.factory_config_sha256)
        object.__setattr__(
            self,
            "artifact_root",
            _canonical_absolute_path(str(self.artifact_root), label="shard artifact_root"),
        )
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionArtifactFactoryError("factory shard corpus is outside FIXED_CORPORA")
        if _OCI_IMAGE.fullmatch(self.runner_image) is None:
            raise ProductionArtifactFactoryError("factory shard runner image is not immutable")
        if self.runner_platform != FACTORY_RUNNER_PLATFORM:
            raise ProductionArtifactFactoryError("factory shard runner platform differs")
        for name in (
            "embedding_source_tree_sha256",
            "embedding_suite_receipt_sha256",
            "hmac_secret_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_text("hmac_key_id", self.hmac_key_id)
        if self.hmac_key_id != f"sealed-online-ephemeral-sha256-{self.hmac_secret_sha256}":
            raise ProductionArtifactFactoryError(
                "factory shard HMAC key ID differs from its secret commitment"
            )
        paths = tuple(self.owned_relative_paths)
        if paths != _shard_owned_relative_paths(self.corpus_id):
            raise ProductionArtifactFactoryError(
                "factory shard owned paths differ from the corpus derivation"
            )
        object.__setattr__(self, "owned_relative_paths", paths)
        if self.schema_version != PRODUCTION_FACTORY_SHARD_REQUEST_SCHEMA:
            raise ProductionArtifactFactoryError("factory shard request schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_root": str(self.artifact_root),
            "corpus_id": self.corpus_id,
            "embedding_source_tree_sha256": self.embedding_source_tree_sha256,
            "embedding_suite_receipt_sha256": self.embedding_suite_receipt_sha256,
            "factory_config_sha256": self.factory_config_sha256,
            "hmac_key_id": self.hmac_key_id,
            "hmac_secret_sha256": self.hmac_secret_sha256,
            "owned_relative_paths": list(self.owned_relative_paths),
            "runner_image": self.runner_image,
            "runner_platform": self.runner_platform,
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def request_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionArtifactFactoryShardRequest:
        row = _closed(value, _SHARD_REQUEST_FIELDS, label="factory shard request")
        paths = row["owned_relative_paths"]
        if not isinstance(paths, list):
            raise ProductionArtifactFactoryError("factory shard owned paths must be an array")
        return cls(
            factory_config_sha256=row["factory_config_sha256"],
            artifact_root=_canonical_absolute_path(
                row["artifact_root"], label="shard artifact_root"
            ),
            corpus_id=row["corpus_id"],
            runner_image=row["runner_image"],
            runner_platform=row["runner_platform"],
            embedding_source_tree_sha256=row["embedding_source_tree_sha256"],
            embedding_suite_receipt_sha256=row["embedding_suite_receipt_sha256"],
            hmac_key_id=row["hmac_key_id"],
            hmac_secret_sha256=row["hmac_secret_sha256"],
            owned_relative_paths=tuple(paths),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True, order=True)
class ProductionArtifactFactoryShardArtifact:
    relative_path: str
    artifact_kind: Literal["file", "tree"]
    sha256: str
    schema_version: str = PRODUCTION_FACTORY_SHARD_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if (
            not isinstance(self.relative_path, str)
            or not self.relative_path
            or str(PurePosixPath(self.relative_path)) != self.relative_path
            or PurePosixPath(self.relative_path).is_absolute()
            or any(part in {"", ".", ".."} for part in PurePosixPath(self.relative_path).parts)
        ):
            raise ProductionArtifactFactoryError("factory shard artifact path is not canonical")
        if self.artifact_kind not in {"file", "tree"}:
            raise ProductionArtifactFactoryError("factory shard artifact kind differs")
        _require_sha256("factory shard artifact SHA-256", self.sha256)
        if self.schema_version != PRODUCTION_FACTORY_SHARD_ARTIFACT_SCHEMA:
            raise ProductionArtifactFactoryError("factory shard artifact schema differs")

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_kind": self.artifact_kind,
            "relative_path": self.relative_path,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ProductionArtifactFactoryShardArtifact:
        row = _closed(value, _SHARD_ARTIFACT_FIELDS, label="factory shard artifact")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductionArtifactFactoryShardReceipt:
    """Deterministic evidence for exactly one completed corpus-owned lane."""

    request_sha256: str
    factory_config_sha256: str
    corpus_id: str
    runner_image: str
    runner_platform: str
    hmac_key_id: str
    hmac_secret_sha256: str
    corpus_evidence_sha256: str
    corpus_evidence_file_sha256: str
    artifacts: tuple[ProductionArtifactFactoryShardArtifact, ...]
    schema_version: str = PRODUCTION_FACTORY_SHARD_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "factory_config_sha256",
            "hmac_secret_sha256",
            "corpus_evidence_sha256",
            "corpus_evidence_file_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionArtifactFactoryError("factory shard receipt corpus differs")
        if _OCI_IMAGE.fullmatch(self.runner_image) is None:
            raise ProductionArtifactFactoryError("factory shard receipt image is not immutable")
        if self.runner_platform != FACTORY_RUNNER_PLATFORM:
            raise ProductionArtifactFactoryError("factory shard receipt platform differs")
        _require_text("hmac_key_id", self.hmac_key_id)
        if self.hmac_key_id != f"sealed-online-ephemeral-sha256-{self.hmac_secret_sha256}":
            raise ProductionArtifactFactoryError(
                "factory shard receipt key ID differs from its commitment"
            )
        if self.corpus_evidence_sha256 != self.corpus_evidence_file_sha256:
            raise ProductionArtifactFactoryError(
                "factory shard evidence semantic and file digests differ"
            )
        artifacts = tuple(self.artifacts)
        expected_paths = _shard_owned_relative_paths(self.corpus_id)
        if tuple(row.relative_path for row in artifacts) != expected_paths or tuple(
            row.artifact_kind for row in artifacts
        ) != ("tree", "tree", "tree", "tree", "tree", "file"):
            raise ProductionArtifactFactoryError(
                "factory shard receipt artifacts differ from the exact owned set"
            )
        object.__setattr__(self, "artifacts", artifacts)
        if artifacts[-1].sha256 != self.corpus_evidence_file_sha256:
            raise ProductionArtifactFactoryError(
                "factory shard evidence artifact differs from its receipt pin"
            )
        if self.schema_version != PRODUCTION_FACTORY_SHARD_RECEIPT_SCHEMA:
            raise ProductionArtifactFactoryError("factory shard receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [row.to_dict() for row in self.artifacts],
            "corpus_evidence_file_sha256": self.corpus_evidence_file_sha256,
            "corpus_evidence_sha256": self.corpus_evidence_sha256,
            "corpus_id": self.corpus_id,
            "factory_config_sha256": self.factory_config_sha256,
            "hmac_key_id": self.hmac_key_id,
            "hmac_secret_sha256": self.hmac_secret_sha256,
            "request_sha256": self.request_sha256,
            "runner_image": self.runner_image,
            "runner_platform": self.runner_platform,
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionArtifactFactoryShardReceipt:
        row = _closed(value, _SHARD_RECEIPT_FIELDS, label="factory shard receipt")
        artifacts = row["artifacts"]
        if not isinstance(artifacts, list):
            raise ProductionArtifactFactoryError("factory shard artifacts must be an array")
        return cls(
            request_sha256=row["request_sha256"],
            factory_config_sha256=row["factory_config_sha256"],
            corpus_id=row["corpus_id"],
            runner_image=row["runner_image"],
            runner_platform=row["runner_platform"],
            hmac_key_id=row["hmac_key_id"],
            hmac_secret_sha256=row["hmac_secret_sha256"],
            corpus_evidence_sha256=row["corpus_evidence_sha256"],
            corpus_evidence_file_sha256=row["corpus_evidence_file_sha256"],
            artifacts=tuple(
                ProductionArtifactFactoryShardArtifact.from_dict(item) for item in artifacts
            ),
            schema_version=row["schema_version"],
        )


def derive_production_artifact_factory_shard_requests(
    config: ProductionArtifactFactoryConfig,
) -> tuple[ProductionArtifactFactoryShardRequest, ...]:
    """Derive the closed five-request set without accepting corpus parameters."""

    if not isinstance(config, ProductionArtifactFactoryConfig):
        raise ProductionArtifactFactoryError("factory config must be typed")
    return tuple(
        ProductionArtifactFactoryShardRequest(
            factory_config_sha256=config.file_sha256,
            artifact_root=config.artifact_root,
            corpus_id=corpus_id,
            runner_image=config.runner_image,
            runner_platform=config.runner_platform,
            embedding_source_tree_sha256=config.embedding_source_tree_sha256,
            embedding_suite_receipt_sha256=config.embedding_suite_receipt_sha256,
            hmac_key_id=config.hmac_key_id,
            hmac_secret_sha256=config.hmac_secret_sha256,
            owned_relative_paths=_shard_owned_relative_paths(corpus_id),
        )
        for corpus_id in FIXED_CORPORA
    )


def load_production_artifact_factory_shard_request(
    path: str | Path,
    *,
    expected_sha256: str,
) -> ProductionArtifactFactoryShardRequest:
    _require_sha256("expected shard request SHA-256", expected_sha256)
    encoded = _read_control(Path(path), label="factory shard request")
    if _sha256(encoded) != expected_sha256:
        raise ProductionArtifactFactoryError("factory shard request differs from its caller pin")
    request = ProductionArtifactFactoryShardRequest.from_dict(
        _decode_object(encoded, label="factory shard request")
    )
    if encoded != request.canonical_file_bytes():
        raise ProductionArtifactFactoryError("factory shard request is not canonical")
    return request


def load_production_artifact_factory_shard_receipt(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ProductionArtifactFactoryShardReceipt:
    encoded = _read_control(Path(path), label="factory shard receipt")
    if expected_sha256 is not None:
        _require_sha256("expected shard receipt SHA-256", expected_sha256)
        if _sha256(encoded) != expected_sha256:
            raise ProductionArtifactFactoryError(
                "factory shard receipt differs from its caller pin"
            )
    receipt = ProductionArtifactFactoryShardReceipt.from_dict(
        _decode_object(encoded, label="factory shard receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionArtifactFactoryError("factory shard receipt is not canonical")
    return receipt


@dataclass(frozen=True)
class ProductionCorpusFactoryEvidence:
    corpus_id: str
    factory_config_sha256: str
    embedding_receipt_sha256: str
    policy_bundle_receipt_sha256: str
    index_bundle_receipt_sha256: str
    query_receipt_sha256: str
    online_execution_plan_sha256: str
    online_execution_tree_sha256: str
    runtime_receipt_sha256: str
    started_at_utc: str
    completed_at_utc: str
    elapsed_monotonic_ns: int
    process_peak_rss_bytes: int
    status: Literal["built", "resumed", "verified-existing"]
    schema_version: str = PRODUCTION_FACTORY_CORPUS_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionArtifactFactoryError("evidence corpus is outside the fixed suite")
        for name in (
            "factory_config_sha256",
            "embedding_receipt_sha256",
            "policy_bundle_receipt_sha256",
            "index_bundle_receipt_sha256",
            "query_receipt_sha256",
            "online_execution_plan_sha256",
            "online_execution_tree_sha256",
            "runtime_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        started = _parse_utc(self.started_at_utc, label="started_at_utc")
        completed = _parse_utc(self.completed_at_utc, label="completed_at_utc")
        if completed < started:
            raise ProductionArtifactFactoryError("evidence completion precedes its start")
        _require_nonnegative_integer("elapsed_monotonic_ns", self.elapsed_monotonic_ns)
        _require_positive_integer("process_peak_rss_bytes", self.process_peak_rss_bytes)
        if self.status not in {"built", "resumed", "verified-existing"}:
            raise ProductionArtifactFactoryError("evidence status differs")
        if self.schema_version != PRODUCTION_FACTORY_CORPUS_EVIDENCE_SCHEMA:
            raise ProductionArtifactFactoryError("corpus evidence schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "completed_at_utc": self.completed_at_utc,
            "corpus_id": self.corpus_id,
            "elapsed_monotonic_ns": self.elapsed_monotonic_ns,
            "embedding_receipt_sha256": self.embedding_receipt_sha256,
            "factory_config_sha256": self.factory_config_sha256,
            "index_bundle_receipt_sha256": self.index_bundle_receipt_sha256,
            "online_execution_plan_sha256": self.online_execution_plan_sha256,
            "online_execution_tree_sha256": self.online_execution_tree_sha256,
            "policy_bundle_receipt_sha256": self.policy_bundle_receipt_sha256,
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
            "query_receipt_sha256": self.query_receipt_sha256,
            "runtime_receipt_sha256": self.runtime_receipt_sha256,
            "schema_version": self.schema_version,
            "started_at_utc": self.started_at_utc,
            "status": self.status,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionCorpusFactoryEvidence:
        row = _closed(value, _EVIDENCE_FIELDS, label="factory corpus evidence")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FactorySuiteCorpus:
    corpus_id: str
    evidence_sha256: str
    evidence_file_sha256: str
    policy_bundle_receipt_sha256: str
    index_bundle_receipt_sha256: str
    query_receipt_sha256: str
    online_execution_plan_sha256: str
    online_execution_tree_sha256: str
    runtime_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionArtifactFactoryError("suite corpus is outside the fixed order")
        for name in (
            "evidence_sha256",
            "evidence_file_sha256",
            "policy_bundle_receipt_sha256",
            "index_bundle_receipt_sha256",
            "query_receipt_sha256",
            "online_execution_plan_sha256",
            "online_execution_tree_sha256",
            "runtime_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.evidence_sha256 != self.evidence_file_sha256:
            raise ProductionArtifactFactoryError("evidence semantic and file digests differ")

    def to_dict(self) -> dict[str, str]:
        return {
            "corpus_id": self.corpus_id,
            "evidence_file_sha256": self.evidence_file_sha256,
            "evidence_sha256": self.evidence_sha256,
            "index_bundle_receipt_sha256": self.index_bundle_receipt_sha256,
            "online_execution_plan_sha256": self.online_execution_plan_sha256,
            "online_execution_tree_sha256": self.online_execution_tree_sha256,
            "policy_bundle_receipt_sha256": self.policy_bundle_receipt_sha256,
            "query_receipt_sha256": self.query_receipt_sha256,
            "runtime_receipt_sha256": self.runtime_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> FactorySuiteCorpus:
        row = _closed(value, _SUITE_CORPUS_FIELDS, label="factory suite corpus")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductionArtifactFactorySuiteReceipt:
    factory_config_sha256: str
    runner_image: str
    runner_platform: str
    embedding_source_tree_sha256: str
    embedding_destination_tree_sha256: str
    embedding_suite_receipt_sha256: str
    hmac_key_id: str
    hmac_secret_sha256: str
    online_inventory_sha256: str
    index_reproducibility_receipt_sha256: str
    artifact_pipeline_receipt_sha256: str
    corpora: tuple[FactorySuiteCorpus, ...]
    schema_version: str = PRODUCTION_FACTORY_SUITE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "factory_config_sha256",
            "embedding_source_tree_sha256",
            "embedding_destination_tree_sha256",
            "embedding_suite_receipt_sha256",
            "hmac_secret_sha256",
            "online_inventory_sha256",
            "index_reproducibility_receipt_sha256",
            "artifact_pipeline_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.embedding_destination_tree_sha256 != self.embedding_source_tree_sha256:
            raise ProductionArtifactFactoryError(
                "suite embedding destination differs from the admitted source tree"
            )
        if (
            not isinstance(self.runner_image, str)
            or _OCI_IMAGE.fullmatch(self.runner_image) is None
        ):
            raise ProductionArtifactFactoryError("suite runner image is not immutable")
        if self.runner_platform != "linux/arm64":
            raise ProductionArtifactFactoryError("suite runner platform differs")
        _require_text("hmac_key_id", self.hmac_key_id)
        if self.hmac_key_id != f"sealed-online-ephemeral-sha256-{self.hmac_secret_sha256}":
            raise ProductionArtifactFactoryError(
                "suite HMAC key ID differs from its secret commitment"
            )
        corpora = tuple(self.corpora)
        if tuple(row.corpus_id for row in corpora) != FIXED_CORPORA:
            raise ProductionArtifactFactoryError("suite rows must follow FIXED_CORPORA")
        object.__setattr__(self, "corpora", corpora)
        if self.schema_version != PRODUCTION_FACTORY_SUITE_SCHEMA:
            raise ProductionArtifactFactoryError("factory suite schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_pipeline_receipt_sha256": self.artifact_pipeline_receipt_sha256,
            "corpora": [row.to_dict() for row in self.corpora],
            "embedding_destination_tree_sha256": self.embedding_destination_tree_sha256,
            "embedding_source_tree_sha256": self.embedding_source_tree_sha256,
            "embedding_suite_receipt_sha256": self.embedding_suite_receipt_sha256,
            "factory_config_sha256": self.factory_config_sha256,
            "hmac_key_id": self.hmac_key_id,
            "hmac_secret_sha256": self.hmac_secret_sha256,
            "index_reproducibility_receipt_sha256": (self.index_reproducibility_receipt_sha256),
            "online_inventory_sha256": self.online_inventory_sha256,
            "runner_image": self.runner_image,
            "runner_platform": self.runner_platform,
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionArtifactFactorySuiteReceipt:
        row = _closed(value, _SUITE_FIELDS, label="factory suite receipt")
        corpora = row["corpora"]
        if not isinstance(corpora, list):
            raise ProductionArtifactFactoryError("suite corpora must be an array")
        return cls(
            factory_config_sha256=row["factory_config_sha256"],
            runner_image=row["runner_image"],
            runner_platform=row["runner_platform"],
            embedding_source_tree_sha256=row["embedding_source_tree_sha256"],
            embedding_destination_tree_sha256=row["embedding_destination_tree_sha256"],
            embedding_suite_receipt_sha256=row["embedding_suite_receipt_sha256"],
            hmac_key_id=row["hmac_key_id"],
            hmac_secret_sha256=row["hmac_secret_sha256"],
            online_inventory_sha256=row["online_inventory_sha256"],
            index_reproducibility_receipt_sha256=(row["index_reproducibility_receipt_sha256"]),
            artifact_pipeline_receipt_sha256=row["artifact_pipeline_receipt_sha256"],
            corpora=tuple(FactorySuiteCorpus.from_dict(item) for item in corpora),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True, order=True)
class ReproducibleIndexPayload:
    mask_id: str
    index_sha256: str
    row_map_sha256: str
    build_binding_sha256: str

    def __post_init__(self) -> None:
        _require_text("mask_id", self.mask_id)
        for name in ("index_sha256", "row_map_sha256", "build_binding_sha256"):
            _require_sha256(name, getattr(self, name))

    def to_dict(self) -> dict[str, str]:
        return {
            "build_binding_sha256": self.build_binding_sha256,
            "index_sha256": self.index_sha256,
            "mask_id": self.mask_id,
            "row_map_sha256": self.row_map_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ReproducibleIndexPayload:
        row = _closed(value, _INDEX_PAYLOAD_FIELDS, label="reproducible index payload")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class IndexReplicateEvidence:
    replicate: int
    receipt_sha256: str
    tree_sha256: str
    index_payloads: tuple[ReproducibleIndexPayload, ...]
    elapsed_monotonic_ns: int
    process_peak_rss_bytes: int

    def __post_init__(self) -> None:
        if self.replicate not in range(1, INDEX_REPLICATE_COUNT + 1):
            raise ProductionArtifactFactoryError("index replicate is outside the fixed set")
        _require_sha256("receipt_sha256", self.receipt_sha256)
        _require_sha256("tree_sha256", self.tree_sha256)
        payloads = tuple(self.index_payloads)
        if (
            not payloads
            or not all(isinstance(row, ReproducibleIndexPayload) for row in payloads)
            or payloads != tuple(sorted(payloads))
            or len({row.mask_id for row in payloads}) != len(payloads)
        ):
            raise ProductionArtifactFactoryError(
                "replicate index payloads must be non-empty, unique, and sorted"
            )
        object.__setattr__(self, "index_payloads", payloads)
        _require_nonnegative_integer("elapsed_monotonic_ns", self.elapsed_monotonic_ns)
        _require_positive_integer("process_peak_rss_bytes", self.process_peak_rss_bytes)

    def to_dict(self) -> dict[str, object]:
        return {
            "elapsed_monotonic_ns": self.elapsed_monotonic_ns,
            "index_payloads": [row.to_dict() for row in self.index_payloads],
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
            "receipt_sha256": self.receipt_sha256,
            "replicate": self.replicate,
            "tree_sha256": self.tree_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> IndexReplicateEvidence:
        row = _closed(value, _REPLICATE_FIELDS, label="index replicate evidence")
        payloads = row["index_payloads"]
        if not isinstance(payloads, list):
            raise ProductionArtifactFactoryError("index payloads must be an array")
        return cls(
            replicate=row["replicate"],
            receipt_sha256=row["receipt_sha256"],
            tree_sha256=row["tree_sha256"],
            index_payloads=tuple(ReproducibleIndexPayload.from_dict(item) for item in payloads),
            elapsed_monotonic_ns=row["elapsed_monotonic_ns"],
            process_peak_rss_bytes=row["process_peak_rss_bytes"],
        )


@dataclass(frozen=True)
class IndexReproducibilityStageReceipt:
    factory_config_sha256: str
    runner_image: str
    runner_platform: str
    corpus_id: str
    stage: str
    backend_id: str
    backend_version: str
    backend_build_sha256: str
    replicates: tuple[IndexReplicateEvidence, ...]
    selected_replicate: int
    selected_final_receipt_sha256: str
    schema_version: str = INDEX_REPRODUCIBILITY_STAGE_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("factory_config_sha256", self.factory_config_sha256)
        if (
            not isinstance(self.runner_image, str)
            or _OCI_IMAGE.fullmatch(self.runner_image) is None
        ):
            raise ProductionArtifactFactoryError("reproducibility runner image is not immutable")
        if self.runner_platform != "linux/arm64":
            raise ProductionArtifactFactoryError("reproducibility runner platform differs")
        if self.corpus_id not in FIXED_CORPORA or self.stage not in ARTIFACT_STAGE_ORDER:
            raise ProductionArtifactFactoryError("reproducibility stage is outside the protocol")
        _require_text("backend_id", self.backend_id)
        _require_text("backend_version", self.backend_version)
        _require_sha256("backend_build_sha256", self.backend_build_sha256)
        rows = tuple(self.replicates)
        if tuple(row.replicate for row in rows) != tuple(range(1, INDEX_REPLICATE_COUNT + 1)):
            raise ProductionArtifactFactoryError("reproducibility needs exactly three replicates")
        reference = rows[0].index_payloads
        if any(row.index_payloads != reference for row in rows[1:]):
            raise ProductionArtifactFactoryError(
                "HNSW or row-map bytes differ across isolated replicates"
            )
        if len({row.receipt_sha256 for row in rows}) != 1:
            raise ProductionArtifactFactoryError(
                "authorized-index receipts differ across isolated replicates"
            )
        if len({row.tree_sha256 for row in rows}) != 1:
            raise ProductionArtifactFactoryError(
                "authorized-index store bytes differ across isolated replicates"
            )
        if self.selected_replicate != SELECTED_INDEX_REPLICATE:
            raise ProductionArtifactFactoryError("selected index replicate differs")
        _require_sha256("selected_final_receipt_sha256", self.selected_final_receipt_sha256)
        if self.selected_final_receipt_sha256 != rows[self.selected_replicate - 1].receipt_sha256:
            raise ProductionArtifactFactoryError(
                "selected final index receipt differs from the registered replicate"
            )
        object.__setattr__(self, "replicates", rows)
        if self.schema_version != INDEX_REPRODUCIBILITY_STAGE_SCHEMA:
            raise ProductionArtifactFactoryError("reproducibility stage schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_build_sha256": self.backend_build_sha256,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "corpus_id": self.corpus_id,
            "factory_config_sha256": self.factory_config_sha256,
            "replicates": [row.to_dict() for row in self.replicates],
            "runner_image": self.runner_image,
            "runner_platform": self.runner_platform,
            "schema_version": self.schema_version,
            "selected_final_receipt_sha256": self.selected_final_receipt_sha256,
            "selected_replicate": self.selected_replicate,
            "stage": self.stage,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> IndexReproducibilityStageReceipt:
        row = _closed(value, _REPRO_STAGE_FIELDS, label="index reproducibility stage")
        replicates = row["replicates"]
        if not isinstance(replicates, list):
            raise ProductionArtifactFactoryError("reproducibility replicates must be an array")
        return cls(
            factory_config_sha256=row["factory_config_sha256"],
            runner_image=row["runner_image"],
            runner_platform=row["runner_platform"],
            corpus_id=row["corpus_id"],
            stage=row["stage"],
            backend_id=row["backend_id"],
            backend_version=row["backend_version"],
            backend_build_sha256=row["backend_build_sha256"],
            replicates=tuple(IndexReplicateEvidence.from_dict(item) for item in replicates),
            selected_replicate=row["selected_replicate"],
            selected_final_receipt_sha256=row["selected_final_receipt_sha256"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class FullHnswReplicateEvidence:
    """One retained build of the deployed full-active HNSW artifact."""

    replicate: int
    relative_path: str
    byte_count: int
    sha256: str
    elapsed_monotonic_ns: int
    process_peak_rss_bytes: int
    schema_version: str = FULL_HNSW_REPLICATE_SCHEMA

    def __post_init__(self) -> None:
        if self.replicate not in range(1, INDEX_REPLICATE_COUNT + 1):
            raise ProductionArtifactFactoryError("full HNSW replicate is outside the fixed set")
        expected_path = (
            f"{INDEX_REPLICATE_DIRECTORIES[self.replicate - 1]}/{FULL_HNSW_REPLICATE_FILENAME}"
        )
        if self.relative_path != expected_path:
            raise ProductionArtifactFactoryError("full HNSW replicate path differs")
        _require_positive_integer("full HNSW byte_count", self.byte_count)
        _require_sha256("full HNSW sha256", self.sha256)
        _require_nonnegative_integer("elapsed_monotonic_ns", self.elapsed_monotonic_ns)
        _require_positive_integer("process_peak_rss_bytes", self.process_peak_rss_bytes)
        if self.schema_version != FULL_HNSW_REPLICATE_SCHEMA:
            raise ProductionArtifactFactoryError("full HNSW replicate schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "elapsed_monotonic_ns": self.elapsed_monotonic_ns,
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
            "relative_path": self.relative_path,
            "replicate": self.replicate,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_dict(cls, value: object) -> FullHnswReplicateEvidence:
        row = _closed(value, _FULL_HNSW_REPLICATE_FIELDS, label="full HNSW replicate")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FullHnswReproducibilityReceipt:
    """Three-build equality proof and selected-copy identity for one corpus."""

    factory_config_sha256: str
    runner_image: str
    runner_platform: str
    corpus_id: str
    backend_id: str
    backend_version: str
    backend_build_sha256: str
    source_vector_sha256: str
    document_count: int
    dimension: int
    format_revision: str
    replicates: tuple[FullHnswReplicateEvidence, ...]
    selected_replicate: int
    selected_final_sha256: str
    schema_version: str = FULL_HNSW_REPRODUCIBILITY_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("factory_config_sha256", self.factory_config_sha256)
        if (
            not isinstance(self.runner_image, str)
            or _OCI_IMAGE.fullmatch(self.runner_image) is None
        ):
            raise ProductionArtifactFactoryError("full HNSW runner image is not immutable")
        if self.runner_platform != FACTORY_RUNNER_PLATFORM:
            raise ProductionArtifactFactoryError("full HNSW runner platform differs")
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionArtifactFactoryError("full HNSW corpus is outside the protocol")
        _require_text("backend_id", self.backend_id)
        _require_text("backend_version", self.backend_version)
        _require_sha256("backend_build_sha256", self.backend_build_sha256)
        _require_sha256("source_vector_sha256", self.source_vector_sha256)
        _require_positive_integer("document_count", self.document_count)
        _require_positive_integer("dimension", self.dimension)
        _require_text("format_revision", self.format_revision)
        rows = tuple(self.replicates)
        if tuple(row.replicate for row in rows) != tuple(range(1, INDEX_REPLICATE_COUNT + 1)):
            raise ProductionArtifactFactoryError("full HNSW needs exactly three replicas")
        if len({(row.byte_count, row.sha256) for row in rows}) != 1:
            raise ProductionArtifactFactoryError("full HNSW bytes differ across isolated replicas")
        if self.selected_replicate != SELECTED_INDEX_REPLICATE:
            raise ProductionArtifactFactoryError("selected full HNSW replicate differs")
        _require_sha256("selected_final_sha256", self.selected_final_sha256)
        if self.selected_final_sha256 != rows[self.selected_replicate - 1].sha256:
            raise ProductionArtifactFactoryError("selected full HNSW copy differs from replicate 1")
        object.__setattr__(self, "replicates", rows)
        if self.schema_version != FULL_HNSW_REPRODUCIBILITY_SCHEMA:
            raise ProductionArtifactFactoryError("full HNSW reproducibility schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_build_sha256": self.backend_build_sha256,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "corpus_id": self.corpus_id,
            "dimension": self.dimension,
            "document_count": self.document_count,
            "factory_config_sha256": self.factory_config_sha256,
            "format_revision": self.format_revision,
            "replicates": [row.to_dict() for row in self.replicates],
            "runner_image": self.runner_image,
            "runner_platform": self.runner_platform,
            "schema_version": self.schema_version,
            "selected_final_sha256": self.selected_final_sha256,
            "selected_replicate": self.selected_replicate,
            "source_vector_sha256": self.source_vector_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> FullHnswReproducibilityReceipt:
        row = _closed(
            value,
            _FULL_HNSW_REPRODUCIBILITY_FIELDS,
            label="full HNSW reproducibility receipt",
        )
        replicates = row["replicates"]
        if not isinstance(replicates, list):
            raise ProductionArtifactFactoryError("full HNSW replicates must be an array")
        return cls(
            factory_config_sha256=row["factory_config_sha256"],
            runner_image=row["runner_image"],
            runner_platform=row["runner_platform"],
            corpus_id=row["corpus_id"],
            backend_id=row["backend_id"],
            backend_version=row["backend_version"],
            backend_build_sha256=row["backend_build_sha256"],
            source_vector_sha256=row["source_vector_sha256"],
            document_count=row["document_count"],
            dimension=row["dimension"],
            format_revision=row["format_revision"],
            replicates=tuple(FullHnswReplicateEvidence.from_dict(item) for item in replicates),
            selected_replicate=row["selected_replicate"],
            selected_final_sha256=row["selected_final_sha256"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class IndexReproducibilitySuiteReceipt:
    factory_config_sha256: str
    runner_image: str
    runner_platform: str
    replicate_count: int
    selected_replicate: int
    stages: tuple[IndexReproducibilityStageReceipt, ...]
    full_hnsw_indexes: tuple[FullHnswReproducibilityReceipt, ...]
    schema_version: str = INDEX_REPRODUCIBILITY_SUITE_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("factory_config_sha256", self.factory_config_sha256)
        if (
            not isinstance(self.runner_image, str)
            or _OCI_IMAGE.fullmatch(self.runner_image) is None
        ):
            raise ProductionArtifactFactoryError("reproducibility suite image is not immutable")
        if self.runner_platform != "linux/arm64":
            raise ProductionArtifactFactoryError("reproducibility suite runner platform differs")
        if self.replicate_count != INDEX_REPLICATE_COUNT:
            raise ProductionArtifactFactoryError("reproducibility replicate count differs")
        if self.selected_replicate != SELECTED_INDEX_REPLICATE:
            raise ProductionArtifactFactoryError("reproducibility selection differs")
        stages = tuple(self.stages)
        expected = tuple(
            (corpus, stage) for corpus in FIXED_CORPORA for stage in ARTIFACT_STAGE_ORDER
        )
        if tuple((row.corpus_id, row.stage) for row in stages) != expected:
            raise ProductionArtifactFactoryError(
                "reproducibility stages must follow fixed corpus/stage order"
            )
        if any(
            row.factory_config_sha256 != self.factory_config_sha256
            or row.runner_image != self.runner_image
            or row.runner_platform != self.runner_platform
            or row.selected_replicate != self.selected_replicate
            for row in stages
        ):
            raise ProductionArtifactFactoryError(
                "reproducibility stage binding differs from its suite"
            )
        object.__setattr__(self, "stages", stages)
        full_hnsw = tuple(self.full_hnsw_indexes)
        if tuple(row.corpus_id for row in full_hnsw) != FIXED_CORPORA:
            raise ProductionArtifactFactoryError(
                "full HNSW reproducibility must follow fixed corpus order"
            )
        if any(
            row.factory_config_sha256 != self.factory_config_sha256
            or row.runner_image != self.runner_image
            or row.runner_platform != self.runner_platform
            or row.selected_replicate != self.selected_replicate
            for row in full_hnsw
        ):
            raise ProductionArtifactFactoryError(
                "full HNSW binding differs from its reproducibility suite"
            )
        object.__setattr__(self, "full_hnsw_indexes", full_hnsw)
        if self.schema_version != INDEX_REPRODUCIBILITY_SUITE_SCHEMA:
            raise ProductionArtifactFactoryError("reproducibility suite schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "factory_config_sha256": self.factory_config_sha256,
            "full_hnsw_indexes": [row.to_dict() for row in self.full_hnsw_indexes],
            "replicate_count": self.replicate_count,
            "runner_image": self.runner_image,
            "runner_platform": self.runner_platform,
            "schema_version": self.schema_version,
            "selected_replicate": self.selected_replicate,
            "stages": [row.to_dict() for row in self.stages],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> IndexReproducibilitySuiteReceipt:
        row = _closed(value, _REPRO_SUITE_FIELDS, label="index reproducibility suite")
        stages = row["stages"]
        if not isinstance(stages, list):
            raise ProductionArtifactFactoryError("reproducibility stages must be an array")
        full_hnsw = row["full_hnsw_indexes"]
        if not isinstance(full_hnsw, list):
            raise ProductionArtifactFactoryError("full HNSW indexes must be an array")
        return cls(
            factory_config_sha256=row["factory_config_sha256"],
            runner_image=row["runner_image"],
            runner_platform=row["runner_platform"],
            replicate_count=row["replicate_count"],
            selected_replicate=row["selected_replicate"],
            stages=tuple(IndexReproducibilityStageReceipt.from_dict(item) for item in stages),
            full_hnsw_indexes=tuple(
                FullHnswReproducibilityReceipt.from_dict(item) for item in full_hnsw
            ),
            schema_version=row["schema_version"],
        )


def _load_corpus_evidence(path: Path) -> ProductionCorpusFactoryEvidence:
    encoded = _read_control(path, label="factory corpus evidence")
    evidence = ProductionCorpusFactoryEvidence.from_dict(
        _decode_object(encoded, label="factory corpus evidence")
    )
    if encoded != evidence.canonical_file_bytes():
        raise ProductionArtifactFactoryError("factory corpus evidence is not canonical")
    return evidence


def load_production_artifact_factory_suite(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ProductionArtifactFactorySuiteReceipt:
    """Load one canonical terminal suite receipt, optionally through an external pin."""

    encoded = _read_control(Path(path), label="factory suite receipt")
    if expected_sha256 is not None:
        _require_sha256("expected factory suite SHA-256", expected_sha256)
        if _sha256(encoded) != expected_sha256:
            raise ProductionArtifactFactoryError(
                "factory suite receipt differs from its caller pin"
            )
    receipt = ProductionArtifactFactorySuiteReceipt.from_dict(
        _decode_object(encoded, label="factory suite receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionArtifactFactoryError("factory suite receipt is not canonical")
    return receipt


def _load_factory_suite(path: Path) -> ProductionArtifactFactorySuiteReceipt:
    return load_production_artifact_factory_suite(path)


def _load_repro_stage(path: Path) -> IndexReproducibilityStageReceipt:
    encoded = _read_control(path, label="index reproducibility stage receipt")
    receipt = IndexReproducibilityStageReceipt.from_dict(
        _decode_object(encoded, label="index reproducibility stage receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionArtifactFactoryError("index reproducibility stage receipt is not canonical")
    return receipt


def _load_full_hnsw_replicate(path: Path) -> FullHnswReplicateEvidence:
    encoded = _read_control(path, label="full HNSW replicate evidence")
    receipt = FullHnswReplicateEvidence.from_dict(
        _decode_object(encoded, label="full HNSW replicate evidence")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionArtifactFactoryError("full HNSW replicate evidence is not canonical")
    return receipt


def _load_full_hnsw_reproducibility(path: Path) -> FullHnswReproducibilityReceipt:
    encoded = _read_control(path, label="full HNSW reproducibility receipt")
    receipt = FullHnswReproducibilityReceipt.from_dict(
        _decode_object(encoded, label="full HNSW reproducibility receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionArtifactFactoryError("full HNSW reproducibility receipt is not canonical")
    return receipt


def _load_repro_suite(path: Path) -> IndexReproducibilitySuiteReceipt:
    encoded = _read_control(path, label="index reproducibility suite receipt")
    receipt = IndexReproducibilitySuiteReceipt.from_dict(
        _decode_object(encoded, label="index reproducibility suite receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionArtifactFactoryError("index reproducibility suite receipt is not canonical")
    return receipt


@dataclass(frozen=True)
class _FactoryInputs:
    embedding_config: ProductionEmbeddingConfig
    embedding_suite: ProductionEmbeddingSuiteReceipt
    embedding_source_tree_sha256: str
    embeddings: Mapping[str, EmbeddingStoreReceipt]
    development_operator: Any
    partition_audit: ScalableQueryPartitionAuditReceipt
    selected_family_count: int


def _prepare_private_directory(path: Path, *, label: str) -> None:
    if os.path.lexists(path):
        _require_real_directory(path, label=label, private=True)
        return
    _require_real_directory(path.parent, label=f"{label} parent", private=True)
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot create {label}: {exc}") from exc


def _require_read_only_filesystem(path: Path, *, label: str) -> None:
    try:
        flags = os.statvfs(path).f_flag
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot inspect {label} mount: {exc}") from exc
    read_only_flag = getattr(os, "ST_RDONLY", 1)
    if not flags & read_only_flag:
        raise ProductionArtifactFactoryError(f"{label} must be mounted read-only")


def _verify_development_operator_binding(
    *,
    root: Path,
    receipt_sha256: str,
    embedding_config_path: Path,
    embedding_config: ProductionEmbeddingConfig,
    embedding_suite: ProductionEmbeddingSuiteReceipt,
    partition_audit_path: Path,
    partition_audit: ScalableQueryPartitionAuditReceipt,
    embedding_suite_receipt_sha256: str,
    partition_audit_sha256: str,
    index_config_sha256: str,
    development_materialization_receipt_sha256: str | None = None,
    design_seed_sha256: str | None = None,
    joint_power_report_sha256: str | None = None,
    selected_family_count: int | None = None,
    joint_power_report_tree_sha256: str | None = None,
) -> Any:
    """Verify the typed development bridge and its exact factory-facing joins."""

    try:
        from .post_embedding_development import (
            PostEmbeddingDevelopmentError,
            admit_frozen_post_embedding_development,
        )
    except ImportError as exc:
        raise ProductionArtifactFactoryError(
            "post-embedding development verifier is unavailable"
        ) from exc
    try:
        receipt = admit_frozen_post_embedding_development(
            root,
            expected_receipt_sha256=receipt_sha256,
            production_embedding_config_path=embedding_config_path,
            embedding_config=embedding_config,
            embedding_suite=embedding_suite,
            partition_audit_path=partition_audit_path,
            partition_audit=partition_audit,
        )
    except PostEmbeddingDevelopmentError as exc:
        raise ProductionArtifactFactoryError(
            f"post-embedding development admission failed: {exc}"
        ) from exc
    expected = {
        "embedding_suite_receipt_sha256": embedding_suite_receipt_sha256,
        "index_config_sha256": index_config_sha256,
        "partition_audit_file_sha256": partition_audit_sha256,
        "partition_audit_sha256": partition_audit_sha256,
    }
    optional_expected = {
        "development_materialization_receipt_sha256": (development_materialization_receipt_sha256),
        "design_seed_sha256": design_seed_sha256,
        "joint_power_report_sha256": joint_power_report_sha256,
        "joint_power_report_tree_sha256": joint_power_report_tree_sha256,
        "selected_families_per_corpus": selected_family_count,
    }
    expected.update({name: value for name, value in optional_expected.items() if value is not None})
    try:
        mismatches = [name for name, value in expected.items() if getattr(receipt, name) != value]
        observed_receipt_sha256 = receipt.artifact_sha256
    except AttributeError as exc:
        raise ProductionArtifactFactoryError(
            "post-embedding development receipt lacks a required factory binding"
        ) from exc
    if observed_receipt_sha256 != receipt_sha256 or mismatches:
        raise ProductionArtifactFactoryError(
            "post-embedding development receipt differs at: "
            + ", ".join(sorted(mismatches or ["artifact_sha256"]))
        )
    return receipt


def _development_operator_paths(root: Path) -> tuple[Path, Path]:
    try:
        from .post_embedding_development import (
            ANALYSIS_DIRECTORY,
            JOINT_POWER_DIRECTORY,
            MATERIALIZATION_DIRECTORY,
        )
    except ImportError as exc:
        raise ProductionArtifactFactoryError(
            "post-embedding development path contract is unavailable"
        ) from exc
    return (
        root / MATERIALIZATION_DIRECTORY,
        root / ANALYSIS_DIRECTORY / JOINT_POWER_DIRECTORY / "report.json",
    )


def _admit_factory_inputs(config: ProductionArtifactFactoryConfig) -> _FactoryInputs:
    """Reproduce every upstream pin before inspecting a factory output."""

    expected_development_root, expected_power_path = _development_operator_paths(
        config.development_operator_root
    )
    redirected_operator_inputs = []
    if config.development_materialization_root != expected_development_root:
        redirected_operator_inputs.append("development_materialization_root")
    if config.joint_power_report_path != expected_power_path:
        redirected_operator_inputs.append("joint_power_report_path")
    if redirected_operator_inputs:
        raise ProductionArtifactFactoryError(
            "factory config redirects development-operator artifacts at: "
            + ", ".join(redirected_operator_inputs)
        )

    try:
        embedding_config = load_production_embedding_config(
            config.embedding_build_config_path,
            expected_sha256=config.embedding_build_config_sha256,
        )
        if embedding_config.output_root != config.embedding_source_root:
            raise ProductionArtifactFactoryError(
                "embedding config output_root differs from embedding_source_root"
            )
        dynamic_sources = (
            embedding_config.online_staging_root,
            embedding_config.current_model_root,
            embedding_config.stale_model_root,
        )
        if any(_paths_overlap(config.artifact_root, source) for source in dynamic_sources):
            raise ProductionArtifactFactoryError(
                "artifact_root cannot overlap an embedding build source"
            )
        _require_real_directory(
            config.embedding_source_root,
            label="embedding_source_root",
        )
        _require_read_only_filesystem(
            config.embedding_source_root,
            label="embedding_source_root",
        )
        source_tree = digest_directory_tree(config.embedding_source_root)
        if source_tree.sha256 != config.embedding_source_tree_sha256:
            raise ProductionArtifactFactoryError(
                "embedding source tree differs from its config pin"
            )
        embedding_suite = admit_frozen_production_embedding_suite(embedding_config)
        if embedding_suite.receipt_sha256 != config.embedding_suite_receipt_sha256:
            raise ProductionArtifactFactoryError(
                "embedding suite receipt differs from its config pin"
            )
        development = verify_materialized_development_cohort(
            config.development_materialization_root,
            expected_receipt_sha256=config.development_materialization_receipt_sha256,
            verify_label_payloads=False,
        )
        audit = load_scalable_partition_audit(
            config.partition_audit_path,
            expected_artifact_sha256=config.partition_audit_sha256,
            expected_inventory_sha256=embedding_config.online_inventory_sha256,
        )
        projection = verify_online_staging_projection(
            embedding_config.online_staging_root,
            expected_inventory_sha256=embedding_config.online_inventory_sha256,
        )
    except (
        DevelopmentCohortError,
        ProductionEmbeddingBuildError,
        ScalablePartitionAuditError,
        ArtifactIntegrityError,
        StudyDataError,
    ) as exc:
        raise ProductionArtifactFactoryError(f"upstream admission failed: {exc}") from exc
    if (
        embedding_suite.online_inventory_sha256 != audit.staged_inventory_sha256
        or embedding_suite.online_inventory_sha256 != projection.inventory_sha256
        or embedding_suite.projected_artifact_set_sha256 != projection.projected_artifact_set_sha256
        or development.staged_inventory_sha256 != projection.inventory_sha256
        or development.partition_audit_sha256 != audit.artifact_sha256
    ):
        raise ProductionArtifactFactoryError(
            "embedding, development, partition, and staging inputs are not one frozen cohort"
        )
    report_bytes = _read_control(config.joint_power_report_path, label="joint power report")
    if _sha256(report_bytes) != config.joint_power_report_sha256:
        raise ProductionArtifactFactoryError("joint power report differs from its config pin")
    try:
        report = load_joint_power_report(report_bytes)
    except JointPowerDesignError as exc:
        raise ProductionArtifactFactoryError(f"joint power report is invalid: {exc}") from exc
    selected = report.selected_families_per_corpus
    if not report.freeze_ready or selected is None:
        raise ProductionArtifactFactoryError("joint power report is not freeze-ready")
    if selected != config.selected_family_count:
        raise ProductionArtifactFactoryError(
            "joint power selection differs from selected_family_count"
        )
    development_operator = _verify_development_operator_binding(
        root=config.development_operator_root,
        receipt_sha256=config.development_operator_receipt_sha256,
        embedding_config_path=config.embedding_build_config_path,
        embedding_config=embedding_config,
        embedding_suite=embedding_suite,
        partition_audit_path=config.partition_audit_path,
        partition_audit=audit,
        embedding_suite_receipt_sha256=embedding_suite.receipt_sha256,
        development_materialization_receipt_sha256=(
            config.development_materialization_receipt_sha256
        ),
        design_seed_sha256=config.design_seed_sha256,
        partition_audit_sha256=config.partition_audit_sha256,
        joint_power_report_sha256=config.joint_power_report_sha256,
        index_config_sha256=config.index_config.config_sha256,
        selected_family_count=selected,
        joint_power_report_tree_sha256=(config.development_operator_joint_power_report_tree_sha256),
    )
    embeddings: dict[str, EmbeddingStoreReceipt] = {}
    suite_by_corpus = {row.corpus_id: row for row in embedding_suite.corpora}
    for corpus_id in FIXED_CORPORA:
        try:
            receipt = verify_embedding_store(config.embedding_source_root / corpus_id)
        except EmbeddingStoreError as exc:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} embedding store failed: {exc}"
            ) from exc
        if receipt.receipt_sha256 != suite_by_corpus[corpus_id].embedding_receipt_sha256:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} embedding receipt differs from the suite"
            )
        if config.corpus(corpus_id).available_family_count < selected:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} available family count is below the powered selection"
            )
        embeddings[corpus_id] = receipt
    return _FactoryInputs(
        embedding_config=embedding_config,
        embedding_suite=embedding_suite,
        embedding_source_tree_sha256=source_tree.sha256,
        embeddings=embeddings,
        development_operator=development_operator,
        partition_audit=audit,
        selected_family_count=selected,
    )


def _prepare_factory_roots(config: ProductionArtifactFactoryConfig) -> None:
    _require_real_directory(config.artifact_root, label="artifact_root", private=True)
    for relative in (
        "policy-workloads",
        "authorized-index-stores",
        "trial-runtime",
        "custody",
        "custody/online",
        INDEX_REPRODUCIBILITY_DIRECTORY,
        FACTORY_EVIDENCE_DIRECTORY,
    ):
        _prepare_private_directory(config.artifact_root / relative, label=relative)
    for corpus_id in FIXED_CORPORA:
        _prepare_private_directory(
            config.artifact_root / "policy-workloads" / corpus_id,
            label=f"{corpus_id} policy root",
        )
        _prepare_private_directory(
            config.artifact_root / "authorized-index-stores" / corpus_id,
            label=f"{corpus_id} index root",
        )
        _prepare_private_directory(
            config.reproducibility_root / corpus_id,
            label=f"{corpus_id} reproducibility root",
        )
        _prepare_private_directory(
            config.artifact_root / "trial-runtime" / corpus_id,
            label=f"{corpus_id} trial runtime root",
        )


def _require_no_terminal_factory_receipts(config: ProductionArtifactFactoryConfig) -> None:
    existing = [
        path.name
        for path in (
            config.pipeline_receipt_path,
            config.reproducibility_receipt_path,
            config.suite_receipt_path,
        )
        if os.path.lexists(path)
    ]
    if existing:
        raise ProductionArtifactFactoryError(
            f"factory aggregation has already started: {sorted(existing)}"
        )


def _require_prepared_shard_root(
    config: ProductionArtifactFactoryConfig,
    *,
    allow_terminal: bool = False,
) -> None:
    """Verify the sequential preparation boundary without creating a path."""

    _validate_factory_root_membership(config, fresh=False)
    if not allow_terminal:
        _require_no_terminal_factory_receipts(config)
    required_directories = (
        config.artifact_root / "embedding-stores",
        config.artifact_root / "policy-workloads",
        config.artifact_root / "authorized-index-stores",
        config.artifact_root / "trial-runtime",
        config.artifact_root / "custody",
        config.artifact_root / "custody" / "online",
        config.reproducibility_root,
        config.evidence_root,
    ) + tuple(
        path
        for corpus_id in FIXED_CORPORA
        for path in (
            config.artifact_root / "policy-workloads" / corpus_id,
            config.artifact_root / "authorized-index-stores" / corpus_id,
            config.artifact_root / "trial-runtime" / corpus_id,
            config.reproducibility_root / corpus_id,
        )
    )
    for path in required_directories:
        _require_real_directory(
            path,
            label=f"prepared shard directory {path.relative_to(config.artifact_root)}",
            private=True,
        )


@contextmanager
def _factory_corpus_locks(
    config: ProductionArtifactFactoryConfig,
    corpus_ids: Sequence[str],
) -> Iterator[None]:
    """Acquire non-mutating advisory locks on precreated corpus-owned directories."""

    corpus_ids = tuple(corpus_ids)
    if (
        not corpus_ids
        or any(corpus_id not in FIXED_CORPORA for corpus_id in corpus_ids)
        or len(set(corpus_ids)) != len(corpus_ids)
    ):
        raise ProductionArtifactFactoryError("factory lock corpus set is invalid")
    ordered = tuple(corpus_id for corpus_id in FIXED_CORPORA if corpus_id in corpus_ids)
    descriptors: list[int] = []
    try:
        for corpus_id in ordered:
            path = config.reproducibility_root / corpus_id
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor: int | None = None
            try:
                descriptor = os.open(path, flags)
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError) as exc:
                if descriptor is not None:
                    os.close(descriptor)
                raise ProductionArtifactFactoryError(
                    f"factory corpus lane {corpus_id} is already active"
                ) from exc
            assert descriptor is not None
            descriptors.append(descriptor)
        yield
    finally:
        for descriptor in reversed(descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _validate_factory_root_membership(
    config: ProductionArtifactFactoryConfig,
    *,
    fresh: bool,
    final: bool = False,
) -> None:
    _require_real_directory(config.artifact_root, label="artifact_root", private=True)
    allowed = {
        "embedding-stores",
        "policy-workloads",
        "authorized-index-stores",
        "trial-runtime",
        "custody",
        INDEX_REPRODUCIBILITY_DIRECTORY,
        FACTORY_EVIDENCE_DIRECTORY,
        ARTIFACT_PIPELINE_RECEIPT_FILENAME,
        INDEX_REPRODUCIBILITY_SUITE_FILENAME,
        FACTORY_SUITE_FILENAME,
    }
    required = set(allowed) if final else {"embedding-stores"}
    try:
        observed = {entry.name for entry in os.scandir(config.artifact_root)}
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot scan artifact_root: {exc}") from exc
    if ".embedding-stores.partial" in observed:
        raise ProductionArtifactFactoryError(
            "an interrupted embedding copy exists; this output root is terminal"
        )
    if fresh and observed:
        raise ProductionArtifactFactoryError("a new factory build requires an empty artifact_root")
    unknown = observed - allowed
    if unknown:
        raise ProductionArtifactFactoryError(
            f"artifact_root has undeclared members: {sorted(unknown)}"
        )
    if observed and "embedding-stores" not in observed:
        raise ProductionArtifactFactoryError(
            "factory outputs exist without the admitted embedding copy"
        )
    if final and not required.issubset(observed):
        raise ProductionArtifactFactoryError(
            f"terminal artifact_root is incomplete: {sorted(required - observed)}"
        )


def _development_plan(
    config: ProductionArtifactFactoryConfig,
    corpus_id: str,
    stage: Literal["fit", "calibration"],
    embedding: EmbeddingStoreReceipt,
) -> DevelopmentExecutionPlan:
    source_stage = POLICY_SOURCE_STAGE_BY_BUNDLE_STAGE[stage]
    path = (
        config.development_materialization_root / source_stage / corpus_id / "execution-plan.json"
    )
    try:
        plan = load_development_execution_plan(path)
    except DevelopmentCohortError as exc:
        raise ProductionArtifactFactoryError(
            f"cannot load {corpus_id} {stage} execution plan: {exc}"
        ) from exc
    if (
        plan.corpus != corpus_id
        or plan.stage != source_stage
        or plan.embedding_receipt_sha256 != embedding.receipt_sha256
        or plan.document_count != embedding.document_count
        or plan.document_universe_sha256 != embedding.row_orders["documents"].row_order_sha256
    ):
        raise ProductionArtifactFactoryError(
            f"{corpus_id} {stage} development plan differs from the production embedding store"
        )
    return plan


def _verify_development_stage_parity(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    *,
    corpus_id: str,
    stage: Literal["fit", "calibration"],
) -> None:
    """Require factory rebuilds to equal the artifacts used for development."""

    try:
        matches = tuple(
            row
            for row in inputs.development_operator.strata
            if row.corpus == corpus_id and row.source_stage == stage
        )
    except AttributeError as exc:
        raise ProductionArtifactFactoryError(
            "post-embedding development receipt lacks typed stratum parity evidence"
        ) from exc
    if len(matches) != 1:
        raise ProductionArtifactFactoryError(
            f"development operator has no unique {corpus_id} {stage} stratum"
        )
    expected = matches[0]
    policy_root = config.artifact_root / "policy-workloads" / corpus_id / stage
    index_root = config.artifact_root / "authorized-index-stores" / corpus_id / stage
    policy = load_policy_intervention_receipt(policy_root / "intervention-receipt.json")
    index = load_authorized_index_store_receipt(index_root)
    expected_policy_config_sha256 = config.policy_config(corpus_id, stage).config_sha256
    mismatches = []
    comparisons = {
        "embedding_receipt_sha256": (
            expected.embedding_receipt_sha256,
            inputs.embeddings[corpus_id].receipt_sha256,
        ),
        "policy_config_sha256": (
            expected.policy_config_sha256,
            expected_policy_config_sha256,
        ),
        "policy_intervention_receipt_sha256": (
            expected.policy_intervention_receipt_sha256,
            policy.artifact_sha256,
        ),
        "authorized_index_config_sha256": (
            expected.authorized_index_config_sha256,
            config.index_config.config_sha256,
        ),
        "authorized_index_receipt_sha256": (
            expected.authorized_index_receipt_sha256,
            index.artifact_sha256,
        ),
    }
    for name, (development_value, factory_value) in comparisons.items():
        if development_value != factory_value:
            mismatches.append(name)
    if mismatches:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} {stage} factory artifacts differ from development at: "
            + ", ".join(mismatches)
        )


def _ensure_policy_stage(
    config: ProductionArtifactFactoryConfig,
    *,
    corpus_id: str,
    stage: str,
    execution: object,
) -> None:
    target = config.artifact_root / "policy-workloads" / corpus_id / stage
    policy_config = config.policy_config(corpus_id, stage)
    try:
        if os.path.lexists(target):
            verify_policy_intervention_package(target, execution, policy_config)
        else:
            write_policy_intervention_package(execution, policy_config, target)
    except PolicyInterventionError as exc:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} {stage} policy stage failed: {exc}"
        ) from exc


def _index_payloads(receipt: AuthorizedIndexStoreReceipt) -> tuple[ReproducibleIndexPayload, ...]:
    return tuple(
        sorted(
            (
                ReproducibleIndexPayload(
                    mask_id=row.mask_id,
                    index_sha256=row.index_sha256,
                    row_map_sha256=row.row_map_sha256,
                    build_binding_sha256=row.build_binding_sha256,
                )
                for row in receipt.indexes
            )
        )
    )


def _rename_directory_noreplace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    *,
    label: str,
) -> None:
    """Rename two child names of one pinned parent without replacing a target."""

    if (
        source_name in {"", ".", ".."}
        or destination_name in {"", ".", ".."}
        or Path(source_name).name != source_name
        or Path(destination_name).name != destination_name
        or source_name == destination_name
    ):
        raise ProductionArtifactFactoryError(f"{label} has an invalid directory rename")
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise ProductionArtifactFactoryError("exclusive directory rename is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source,
            parent_descriptor,
            destination,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise ProductionArtifactFactoryError("exclusive directory rename is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source,
            parent_descriptor,
            destination,
            0x00000001,
        )
    else:
        raise ProductionArtifactFactoryError(
            f"exclusive directory rename is unsupported on {sys.platform!r}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ProductionArtifactFactoryError(f"{label} already exists")
        raise ProductionArtifactFactoryError(f"cannot rename {label}: {os.strerror(error_number)}")


def _exclusive_publish_directory(work: Path, output: Path, *, label: str) -> None:
    if (
        work.parent != output.parent
        or work.name in {"", ".", ".."}
        or output.name
        in {
            "",
            ".",
            "..",
        }
    ):
        raise ProductionArtifactFactoryError(
            f"{label} staging and target must be distinct names in one parent"
        )
    if work.name == output.name:
        raise ProductionArtifactFactoryError(f"{label} staging name equals its target")
    parent = work.parent
    parent_path_metadata = parent.lstat()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_descriptor = -1
    source_descriptor = -1
    destination_descriptor = -1
    try:
        parent_descriptor = os.open(parent, flags)
        parent_metadata = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or stat.S_ISLNK(parent_metadata.st_mode)
            or parent_metadata.st_dev != parent_path_metadata.st_dev
            or parent_metadata.st_ino != parent_path_metadata.st_ino
            or (hasattr(os, "geteuid") and parent_metadata.st_uid != os.geteuid())
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            raise ProductionArtifactFactoryError(f"{label} parent identity differs")
        source_path_metadata = work.lstat()
        source_descriptor = os.open(work.name, flags, dir_fd=parent_descriptor)
        source_metadata = os.fstat(source_descriptor)
        if (
            not stat.S_ISDIR(source_metadata.st_mode)
            or stat.S_ISLNK(source_metadata.st_mode)
            or source_metadata.st_dev != source_path_metadata.st_dev
            or source_metadata.st_ino != source_path_metadata.st_ino
            or source_metadata.st_uid != parent_metadata.st_uid
            or stat.S_IMODE(source_metadata.st_mode) & 0o022
        ):
            raise ProductionArtifactFactoryError(f"{label} staging identity differs")
        try:
            os.stat(output.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ProductionArtifactFactoryError(f"{label} already exists")
        _rename_directory_noreplace_at(
            parent_descriptor,
            work.name,
            output.name,
            label=label,
        )
        destination_descriptor = os.open(output.name, flags, dir_fd=parent_descriptor)
        destination_metadata = os.fstat(destination_descriptor)
        if (
            destination_metadata.st_dev != source_metadata.st_dev
            or destination_metadata.st_ino != source_metadata.st_ino
        ):
            raise ProductionArtifactFactoryError(f"{label} target is not the pinned staging inode")
        try:
            os.stat(work.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ProductionArtifactFactoryError(f"{label} staging name survived publication")
        current_parent = parent.lstat()
        if (
            current_parent.st_dev != parent_metadata.st_dev
            or current_parent.st_ino != parent_metadata.st_ino
        ):
            raise ProductionArtifactFactoryError(f"{label} parent changed during publication")
        os.fsync(destination_descriptor)
        os.fsync(parent_descriptor)
    except ProductionArtifactFactoryError:
        raise
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot publish {label}: {exc}") from exc
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _validate_and_remove_temporary_tree(path: Path, *, label: str) -> None:
    """Remove only a private, unpublished factory staging tree.

    Callers hold the corpus lane lock. Published target names and canonical
    receipts never enter this function.
    """

    if path.name in {"", ".", ".."}:
        raise ProductionArtifactFactoryError(f"{label} has an invalid temporary name")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_descriptor = -1
    root_descriptor = -1
    quarantine_descriptor = -1
    try:
        parent_path_metadata = path.parent.lstat()
        parent_descriptor = os.open(path.parent, flags)
        parent_metadata = os.fstat(parent_descriptor)
        current_uid = os.geteuid() if hasattr(os, "geteuid") else parent_metadata.st_uid
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or stat.S_ISLNK(parent_metadata.st_mode)
            or parent_metadata.st_dev != parent_path_metadata.st_dev
            or parent_metadata.st_ino != parent_path_metadata.st_ino
            or parent_metadata.st_uid != current_uid
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            raise ProductionArtifactFactoryError(f"{label} parent identity differs")
        root_path_metadata = path.lstat()
        root_descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        root_metadata = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_ISLNK(root_metadata.st_mode)
            or root_metadata.st_dev != root_path_metadata.st_dev
            or root_metadata.st_ino != root_path_metadata.st_ino
            or root_metadata.st_uid != current_uid
            or stat.S_IMODE(root_metadata.st_mode) & 0o022
        ):
            raise ProductionArtifactFactoryError(
                f"{label} is not a private runner-owned temporary directory"
            )
        for _root, directories, files, directory_descriptor in os.fwalk(
            ".",
            topdown=False,
            follow_symlinks=False,
            dir_fd=root_descriptor,
        ):
            for name in (*directories, *files):
                metadata = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or metadata.st_uid != current_uid
                    or metadata.st_dev != root_metadata.st_dev
                    or stat.S_IMODE(metadata.st_mode) & 0o022
                    or (
                        not stat.S_ISDIR(metadata.st_mode)
                        and (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1)
                    )
                ):
                    raise ProductionArtifactFactoryError(
                        f"{label} contains an unsafe temporary member"
                    )
        current = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _stat_signature(current) != _stat_signature(root_metadata):
            raise ProductionArtifactFactoryError(f"{label} changed during recovery inspection")
        quarantine_name = f"{path.name}.recovery-{secrets.token_hex(12)}"
        _rename_directory_noreplace_at(
            parent_descriptor,
            path.name,
            quarantine_name,
            label=f"{label} recovery quarantine",
        )
        quarantine_descriptor = os.open(
            quarantine_name,
            flags,
            dir_fd=parent_descriptor,
        )
        quarantine_metadata = os.fstat(quarantine_descriptor)
        pinned_metadata = os.fstat(root_descriptor)
        if (
            quarantine_metadata.st_dev != pinned_metadata.st_dev
            or quarantine_metadata.st_ino != pinned_metadata.st_ino
        ):
            raise ProductionArtifactFactoryError(
                f"{label} quarantine is not the pinned temporary inode"
            )
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ProductionArtifactFactoryError(f"{label} name survived quarantine")
        for _root, directories, files, directory_descriptor in os.fwalk(
            ".",
            topdown=False,
            follow_symlinks=False,
            dir_fd=quarantine_descriptor,
        ):
            for name in files:
                os.unlink(name, dir_fd=directory_descriptor)
            for name in directories:
                os.rmdir(name, dir_fd=directory_descriptor)
        os.rmdir(quarantine_name, dir_fd=parent_descriptor)
        try:
            os.stat(quarantine_name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ProductionArtifactFactoryError(f"{label} survived temporary cleanup")
        current_parent = path.parent.lstat()
        if (
            current_parent.st_dev != parent_metadata.st_dev
            or current_parent.st_ino != parent_metadata.st_ino
        ):
            raise ProductionArtifactFactoryError(f"{label} parent changed during cleanup")
        os.fsync(parent_descriptor)
    except ProductionArtifactFactoryError:
        raise
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot remove {label}: {exc}") from exc
    finally:
        if quarantine_descriptor >= 0:
            os.close(quarantine_descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _recover_factory_staging_directories(parent: Path, *, prefix: str, label: str) -> None:
    """Discard exact tokenized staging names after a corpus lock is acquired."""

    _require_real_directory(parent, label=f"{label} parent", private=True)
    try:
        candidates = tuple(path for path in parent.iterdir() if path.name.startswith(prefix))
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot scan {label} parent: {exc}") from exc
    token = re.compile(re.escape(prefix) + r"[0-9a-f]{24}(?:\.recovery-[0-9a-f]{24})*")
    for path in candidates:
        if token.fullmatch(path.name) is None:
            raise ProductionArtifactFactoryError(f"{label} has an undeclared temporary name")
        _validate_and_remove_temporary_tree(path, label=label)


def _recover_authorized_index_builder_residue(output: Path) -> None:
    """Recover a killed authorized-index build without touching a published store."""

    parent = output.parent
    prefix = f".{output.name}.staging-"
    lock = parent / f".{output.name}.authorized-index.lock"
    try:
        partials = tuple(path for path in parent.iterdir() if path.name.startswith(prefix))
    except OSError as exc:
        raise ProductionArtifactFactoryError(
            f"cannot scan {output.name} authorized-index parent: {exc}"
        ) from exc
    if not partials and not os.path.lexists(lock):
        return
    if partials and not os.path.lexists(lock):
        raise ProductionArtifactFactoryError(
            f"{output.name} has staging bytes without its builder lock"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_descriptor = -1
    try:
        parent_path_metadata = parent.lstat()
        parent_descriptor = os.open(parent, flags)
        parent_metadata = os.fstat(parent_descriptor)
        if (
            parent_metadata.st_dev != parent_path_metadata.st_dev
            or parent_metadata.st_ino != parent_path_metadata.st_ino
        ):
            raise ProductionArtifactFactoryError(
                f"{output.name} authorized-index parent identity differs"
            )
        lock_metadata = os.stat(
            lock.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except ProductionArtifactFactoryError:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        raise
    except OSError as exc:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        raise ProductionArtifactFactoryError(
            f"cannot inspect {output.name} builder lock: {exc}"
        ) from exc
    current_uid = os.geteuid() if hasattr(os, "geteuid") else lock_metadata.st_uid
    if (
        not stat.S_ISREG(lock_metadata.st_mode)
        or stat.S_ISLNK(lock_metadata.st_mode)
        or lock_metadata.st_nlink != 1
        or lock_metadata.st_uid != current_uid
        or stat.S_IMODE(lock_metadata.st_mode) != 0o600
        or lock_metadata.st_size != 0
    ):
        os.close(parent_descriptor)
        parent_descriptor = -1
        raise ProductionArtifactFactoryError(
            f"{output.name} builder lock is not the exact private lock format"
        )
    descriptor = -1
    acquired = False
    try:
        descriptor = os.open(
            lock.name,
            os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if _stat_signature(opened) != _stat_signature(lock_metadata):
            raise ProductionArtifactFactoryError(
                f"{output.name} builder lock changed during recovery"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProductionArtifactFactoryError(
                f"{output.name} authorized-index builder is still active"
            ) from exc
        acquired = True
        token = re.compile(re.escape(prefix) + r"[0-9a-f]{24}(?:\.recovery-[0-9a-f]{24})*")
        for path in partials:
            if token.fullmatch(path.name) is None:
                raise ProductionArtifactFactoryError(
                    f"{output.name} has an undeclared staging name"
                )
            _validate_and_remove_temporary_tree(
                path,
                label=f"{output.name} interrupted authorized-index staging",
            )
        current = os.stat(
            lock.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _stat_signature(current) != _stat_signature(opened):
            raise ProductionArtifactFactoryError(
                f"{output.name} builder lock changed before cleanup"
            )
        os.unlink(lock.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    except ProductionArtifactFactoryError:
        raise
    except OSError as exc:
        raise ProductionArtifactFactoryError(
            f"cannot recover {output.name} authorized-index builder: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _copy_tree_exclusive(source: Path, target: Path, *, label: str) -> None:
    if os.path.lexists(target):
        raise ProductionArtifactFactoryError(f"{label} already exists")
    work = target.parent / f".{target.name}.staging-{secrets.token_hex(12)}"
    try:
        shutil.copytree(source, work, symlinks=False)
        for path in work.rglob("*"):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ProductionArtifactFactoryError(f"{label} copy contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                os.chmod(path, 0o700)
            elif stat.S_ISREG(metadata.st_mode):
                os.chmod(path, 0o600)
            else:
                raise ProductionArtifactFactoryError(f"{label} copy contains a special file")
        _exclusive_publish_directory(work, target, label=label)
    finally:
        if os.path.lexists(work):
            _validate_and_remove_temporary_tree(work, label=f"{label} staging cleanup")


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _copy_regular_file_exclusive(source: Path, target: Path) -> None:
    required = ("O_CLOEXEC", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise ProductionArtifactFactoryError("secure copy flags are unavailable")
    try:
        source_metadata = source.lstat()
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot inspect embedding source: {exc}") from exc
    if not stat.S_ISREG(source_metadata.st_mode) or source_metadata.st_nlink != 1:
        raise ProductionArtifactFactoryError(
            f"embedding source file is not a single-link regular file: {source}"
        )
    try:
        source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot open embedding source: {exc}") from exc
    target_fd = -1
    try:
        opened_source = os.fstat(source_fd)
        if _stat_signature(opened_source) != _stat_signature(source_metadata):
            raise ProductionArtifactFactoryError("embedding source changed before copy")
        target_fd = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        byte_count = 0
        while True:
            chunk = os.read(source_fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise ProductionArtifactFactoryError("embedding copy made no progress")
                view = view[written:]
            byte_count += len(chunk)
        os.fsync(target_fd)
        final_source = os.fstat(source_fd)
        if (
            _stat_signature(final_source) != _stat_signature(opened_source)
            or byte_count != opened_source.st_size
            or os.fstat(target_fd).st_size != byte_count
        ):
            raise ProductionArtifactFactoryError("embedding source changed during copy")
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot copy embedding source: {exc}") from exc
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        os.close(source_fd)


def _copy_regular_tree_exclusive(source: Path, target: Path) -> None:
    try:
        target.mkdir(mode=0o700)
    except OSError as exc:
        raise ProductionArtifactFactoryError(
            f"cannot create embedding copy boundary: {exc}"
        ) from exc

    def copy_directory(source_directory: Path, target_directory: Path) -> None:
        try:
            entries = sorted(
                os.scandir(source_directory),
                key=lambda entry: entry.name.encode("utf-8", errors="strict"),
            )
        except (OSError, UnicodeEncodeError) as exc:
            raise ProductionArtifactFactoryError(
                f"cannot scan embedding source directory: {exc}"
            ) from exc
        for entry in entries:
            source_path = source_directory / entry.name
            target_path = target_directory / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ProductionArtifactFactoryError(
                    f"cannot inspect embedding source member: {exc}"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    target_path.mkdir(mode=0o700)
                except OSError as exc:
                    raise ProductionArtifactFactoryError(
                        f"cannot create embedding destination directory: {exc}"
                    ) from exc
                copy_directory(source_path, target_path)
            elif stat.S_ISREG(metadata.st_mode):
                _copy_regular_file_exclusive(source_path, target_path)
            else:
                raise ProductionArtifactFactoryError(
                    f"embedding source contains a non-regular member: {source_path}"
                )

    copy_directory(source, target)


def _verify_embedding_copy(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
) -> str:
    target = config.artifact_root / "embedding-stores"
    partial = config.artifact_root / ".embedding-stores.partial"
    if os.path.lexists(partial):
        raise ProductionArtifactFactoryError(
            "an interrupted embedding copy exists; this output root is terminal"
        )
    _require_real_directory(target, label="embedding destination", private=True)
    try:
        source_tree = digest_directory_tree(config.embedding_source_root)
        destination_tree = digest_directory_tree(target)
    except ArtifactIntegrityError as exc:
        raise ProductionArtifactFactoryError(
            f"cannot rehash embedding source and destination: {exc}"
        ) from exc
    if source_tree.sha256 != inputs.embedding_source_tree_sha256:
        raise ProductionArtifactFactoryError("embedding source tree changed after admission")
    if destination_tree.sha256 != source_tree.sha256:
        raise ProductionArtifactFactoryError(
            "embedding destination tree differs byte-for-byte from its source"
        )
    suite_by_corpus = {row.corpus_id: row for row in inputs.embedding_suite.corpora}
    for corpus_id in FIXED_CORPORA:
        try:
            receipt = verify_embedding_store(target / corpus_id)
        except EmbeddingStoreError as exc:
            raise ProductionArtifactFactoryError(
                f"copied {corpus_id} embedding store failed: {exc}"
            ) from exc
        if (
            receipt != inputs.embeddings[corpus_id]
            or receipt.receipt_sha256 != suite_by_corpus[corpus_id].embedding_receipt_sha256
        ):
            raise ProductionArtifactFactoryError(
                f"copied {corpus_id} embedding receipt differs from the admitted source"
            )
    return destination_tree.sha256


def _ensure_embedding_copy(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
) -> str:
    target = config.artifact_root / "embedding-stores"
    partial = config.artifact_root / ".embedding-stores.partial"
    if os.path.lexists(partial):
        raise ProductionArtifactFactoryError(
            "an interrupted embedding copy exists; this output root is terminal"
        )
    if os.path.lexists(target):
        return _verify_embedding_copy(config, inputs)
    _copy_regular_tree_exclusive(config.embedding_source_root, partial)
    try:
        copied = digest_directory_tree(partial)
    except ArtifactIntegrityError as exc:
        raise ProductionArtifactFactoryError(f"cannot hash embedding copy: {exc}") from exc
    if copied.sha256 != inputs.embedding_source_tree_sha256:
        raise ProductionArtifactFactoryError(
            "embedding copy differs byte-for-byte from its admitted source"
        )
    _exclusive_publish_directory(partial, target, label="embedding destination")
    return _verify_embedding_copy(config, inputs)


def _replicate_evidence(
    root: Path,
    *,
    replicate: int,
    elapsed_ns: int,
    peak_rss_bytes: int | None = None,
) -> IndexReplicateEvidence:
    receipt = load_authorized_index_store_receipt(root)
    try:
        tree = digest_directory_tree(root)
    except ArtifactIntegrityError as exc:
        raise ProductionArtifactFactoryError(f"cannot hash index replicate: {exc}") from exc
    return IndexReplicateEvidence(
        replicate=replicate,
        receipt_sha256=receipt.artifact_sha256,
        tree_sha256=tree.sha256,
        index_payloads=_index_payloads(receipt),
        elapsed_monotonic_ns=elapsed_ns,
        process_peak_rss_bytes=(_peak_rss_bytes() if peak_rss_bytes is None else peak_rss_bytes),
    )


def _ensure_reproducible_index_stage(
    config: ProductionArtifactFactoryConfig,
    *,
    corpus_id: str,
    stage: str,
    embedding: EmbeddingStoreReceipt,
    backend: HnswlibBackend,
    recover_partials: bool = False,
) -> IndexReproducibilityStageReceipt:
    policy_root = config.artifact_root / "policy-workloads" / corpus_id / stage
    policy_receipt = load_policy_intervention_receipt(policy_root / "intervention-receipt.json")
    stage_root = config.reproducibility_root / corpus_id / stage
    _prepare_private_directory(stage_root, label=f"{corpus_id} {stage} reproducibility")
    receipt_path = stage_root / "reproducibility-receipt.json"
    final_root = config.artifact_root / "authorized-index-stores" / corpus_id / stage
    existing_receipt = _load_repro_stage(receipt_path) if os.path.lexists(receipt_path) else None
    replicate_rows: list[IndexReplicateEvidence] = []
    for replicate, directory in enumerate(config.index_replicate_directories, start=1):
        root = stage_root / directory
        if recover_partials:
            _recover_authorized_index_builder_residue(root)
        started_ns = time.monotonic_ns()
        try:
            if os.path.lexists(root):
                verify_authorized_index_store(
                    root,
                    embedding_store_root=config.artifact_root / "embedding-stores" / corpus_id,
                    policy_intervention_root=policy_root,
                    expected_embedding_receipt_sha256=embedding.receipt_sha256,
                    expected_policy_receipt_sha256=policy_receipt.artifact_sha256,
                    backend=backend,
                )
            else:
                if existing_receipt is not None:
                    raise ProductionArtifactFactoryError(
                        f"{corpus_id} {stage} receipt exists but replicate {replicate} is missing"
                    )
                build_authorized_index_store(
                    config.artifact_root / "embedding-stores" / corpus_id,
                    policy_root,
                    root,
                    expected_embedding_receipt_sha256=embedding.receipt_sha256,
                    expected_policy_receipt_sha256=policy_receipt.artifact_sha256,
                    config=config.index_config,
                    backend=backend,
                )
        except AuthorizedIndexStoreError as exc:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} {stage} index replicate {replicate} failed: {exc}"
            ) from exc
        prior = None if existing_receipt is None else existing_receipt.replicates[replicate - 1]
        observed = _replicate_evidence(
            root,
            replicate=replicate,
            elapsed_ns=(
                prior.elapsed_monotonic_ns
                if prior is not None
                else time.monotonic_ns() - started_ns
            ),
            peak_rss_bytes=(None if prior is None else prior.process_peak_rss_bytes),
        )
        if prior is not None and observed != prior:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} {stage} replicate differs from its receipt"
            )
        replicate_rows.append(observed)
    reference = replicate_rows[0]
    if any(row.index_payloads != reference.index_payloads for row in replicate_rows[1:]):
        raise ProductionArtifactFactoryError(
            f"{corpus_id} {stage} HNSW or row-map digests are not reproducible"
        )
    if len({row.receipt_sha256 for row in replicate_rows}) != 1:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} {stage} index receipt is not reproducible"
        )
    selected_root = (
        stage_root / config.index_replicate_directories[config.selected_index_replicate - 1]
    )
    if recover_partials:
        _recover_factory_staging_directories(
            final_root.parent,
            prefix=f".{final_root.name}.staging-",
            label=f"{corpus_id} {stage} selected-index staging",
        )
    if not os.path.lexists(final_root):
        if existing_receipt is not None:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} {stage} receipt exists but selected index is missing"
            )
        _copy_tree_exclusive(
            selected_root,
            final_root,
            label=f"{corpus_id} {stage} selected index",
        )
    try:
        selected_verification = verify_authorized_index_store(
            final_root,
            embedding_store_root=config.artifact_root / "embedding-stores" / corpus_id,
            policy_intervention_root=policy_root,
            expected_embedding_receipt_sha256=embedding.receipt_sha256,
            expected_policy_receipt_sha256=policy_receipt.artifact_sha256,
            backend=backend,
            expected_store_receipt_sha256=reference.receipt_sha256,
        )
    except AuthorizedIndexStoreError as exc:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} {stage} selected index failed: {exc}"
        ) from exc
    try:
        selected_tree = digest_directory_tree(final_root)
    except ArtifactIntegrityError as exc:
        raise ProductionArtifactFactoryError(
            f"cannot hash {corpus_id} {stage} selected index: {exc}"
        ) from exc
    if selected_tree.sha256 != reference.tree_sha256:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} {stage} selected index is not an exact copy of replicate 1"
        )
    result = IndexReproducibilityStageReceipt(
        factory_config_sha256=config.file_sha256,
        runner_image=config.runner_image,
        runner_platform=config.runner_platform,
        corpus_id=corpus_id,
        stage=stage,
        backend_id=config.index_config.backend_id,
        backend_version=config.index_config.backend_version,
        backend_build_sha256=config.index_config.backend_build_sha256,
        replicates=tuple(replicate_rows),
        selected_replicate=config.selected_index_replicate,
        selected_final_receipt_sha256=selected_verification.receipt_sha256,
    )
    if existing_receipt is not None:
        if existing_receipt != result:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} {stage} reproducibility receipt differs"
            )
    else:
        try:
            write_exclusive_receipt_bytes(result.canonical_file_bytes(), receipt_path)
        except ArtifactIntegrityError as exc:
            raise ProductionArtifactFactoryError(
                f"cannot write {corpus_id} {stage} reproducibility receipt: {exc}"
            ) from exc
    expected_members = {
        "reproducibility-receipt.json",
        *config.index_replicate_directories,
    }
    observed_members = {path.name for path in stage_root.iterdir()}
    if observed_members != expected_members:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} {stage} reproducibility membership differs"
        )
    return result


def _source_row_by_path(
    audit: ScalableQueryPartitionAuditReceipt,
) -> Mapping[str, Any]:
    return {row.path: row for row in audit.source_artifacts}


def _document_sources(
    inputs: _FactoryInputs,
    corpus_id: str,
) -> tuple[tuple[ProductionCorpusSources, Any], ...]:
    config_row = next(row for row in inputs.embedding_config.corpora if row.corpus_id == corpus_id)
    by_path = _source_row_by_path(inputs.partition_audit)
    result: list[tuple[ProductionCorpusSources, Any]] = []
    for path in config_row.document_paths:
        source = by_path.get(path)
        if (
            source is None
            or source.dataset != corpus_id
            or source.stage is not None
            or source.role not in {"corpus", "corpus-shard"}
            or source.visibility != "online"
        ):
            raise ProductionArtifactFactoryError(
                f"{corpus_id} document source {path!r} differs from the partition audit"
            )
        result.append((config_row, source))
    if not result:
        raise ProductionArtifactFactoryError(f"{corpus_id} has no document sources")
    return tuple(result)


def _open_relative_source(root: Path, relative_path: str) -> int:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ProductionArtifactFactoryError("source path is not canonical relative POSIX")
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise ProductionArtifactFactoryError("secure source-open flags are unavailable")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(root, directory_flags)
    try:
        for component in pure.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(pure.parts[-1], file_flags, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    metadata = os.fstat(file_descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(file_descriptor)
        raise ProductionArtifactFactoryError("source must be one non-linked regular file")
    return file_descriptor


def _decode_assignment_line(line: bytes, *, label: str) -> Mapping[str, Any]:
    if len(line) > _MAX_LINE_BYTES or not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
        raise ProductionArtifactFactoryError(f"{label} is not bounded canonical JSONL")
    row = _closed(
        _decode_object(line[:-1], label=label),
        _ASSIGNMENT_FIELDS,
        label=label,
    )
    if line != _canonical_bytes(row) + b"\n":
        raise ProductionArtifactFactoryError(f"{label} bytes are not canonical")
    if row["schema_version"] != ASSIGNMENT_SCHEMA:
        raise ProductionArtifactFactoryError(f"{label} schema differs")
    if row["dataset"] not in FIXED_CORPORA or row["stage"] not in ARTIFACT_STAGE_ORDER:
        raise ProductionArtifactFactoryError(f"{label} corpus or stage differs")
    for name in ("query_id", "source_split"):
        _require_text(f"{label} {name}", row[name])
    for name in (
        "assignment_key_sha256",
        "partition_component_sha256",
        "query_text_sha256",
    ):
        _require_sha256(f"{label} {name}", row[name])
    if row["domain"] is not None:
        _require_text(f"{label} domain", row["domain"])
    return row


def _derive_available_family_counts(
    staging_root: Path,
    audit: ScalableQueryPartitionAuditReceipt,
) -> tuple[ProductionCorpusFactoryConfig, ...]:
    assignment_sources = [
        row
        for row in audit.source_artifacts
        if row.role == "assignments"
        and row.path == "assignments.jsonl"
        and row.visibility == "online"
        and row.dataset is None
        and row.stage is None
    ]
    if len(assignment_sources) != 1:
        raise ProductionArtifactFactoryError(
            "partition audit does not bind one online assignment store"
        )
    source = assignment_sources[0]
    if source.sha256 != audit.assignment_artifact_sha256:
        raise ProductionArtifactFactoryError("assignment source differs from the partition audit")
    expected_query_counts = {
        (row.dataset, row.stage): row.query_count for row in audit.query_counts
    }
    expected_keys = {
        (corpus_id, stage) for corpus_id in FIXED_CORPORA for stage in ARTIFACT_STAGE_ORDER
    }
    if set(expected_query_counts) != expected_keys:
        raise ProductionArtifactFactoryError(
            "partition-audit query counts do not cover the fixed corpus/stage product"
        )
    descriptor = _open_relative_source(staging_root, source.path)
    initial = os.fstat(descriptor)
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    query_ids: set[str] = set()
    observed_query_counts: dict[tuple[str, str], int] = {}
    components: dict[tuple[str, str], set[str]] = {}
    try:
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as reader:
            while True:
                line = reader.readline(_MAX_LINE_BYTES + 1)
                if not line:
                    break
                record_count += 1
                byte_count += len(line)
                digest.update(line)
                row = _decode_assignment_line(
                    line,
                    label=f"assignment line {record_count}",
                )
                query_id = row["query_id"]
                if query_id in query_ids:
                    raise ProductionArtifactFactoryError(
                        "assignment store repeats a query identifier"
                    )
                query_ids.add(query_id)
                key = (row["dataset"], row["stage"])
                observed_query_counts[key] = observed_query_counts.get(key, 0) + 1
                components.setdefault(key, set()).add(row["partition_component_sha256"])
        final = os.fstat(descriptor)
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot read audited assignments: {exc}") from exc
    finally:
        os.close(descriptor)
    if (
        _stat_signature(final) != _stat_signature(initial)
        or byte_count != source.byte_count
        or record_count != source.record_count
        or record_count != audit.assignment_count
        or digest.hexdigest() != source.sha256
        or observed_query_counts != expected_query_counts
    ):
        raise ProductionArtifactFactoryError(
            "assignment bytes, counts, or coverage differ from the partition audit"
        )
    result: list[ProductionCorpusFactoryConfig] = []
    for corpus_id in FIXED_CORPORA:
        available = len(components.get((corpus_id, "sealed"), set()))
        if available <= 0:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} has no audited sealed assignment families"
            )
        result.append(
            ProductionCorpusFactoryConfig(
                corpus_id=corpus_id,
                available_family_count=available,
            )
        )
    return tuple(result)


def _open_output(path: Path) -> int:
    try:
        return os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot create output {path.name!r}: {exc}") from exc


def _decode_document_line(line: bytes, *, label: str) -> Mapping[str, str]:
    if len(line) > _MAX_LINE_BYTES or not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
        raise ProductionArtifactFactoryError(f"{label} is not bounded canonical JSONL")
    row = _decode_object(line[:-1], label=label)
    if line != _canonical_bytes(row) + b"\n" or set(row) != {"id", "text", "title"}:
        raise ProductionArtifactFactoryError(f"{label} fields or bytes differ")
    if not all(isinstance(row[name], str) for name in ("id", "text", "title")):
        raise ProductionArtifactFactoryError(f"{label} values must be strings")
    if not row["id"] or not row["text"]:
        raise ProductionArtifactFactoryError(f"{label} has an empty ID or text")
    return row  # type: ignore[return-value]


def _copy_document_source(
    *,
    staging_root: Path,
    source: Any,
    target: Path,
    corpus_id: str,
    provenance_descriptor: int,
    provenance_digest: Any,
    previous_document_id: bytes | None,
) -> tuple[int, bytes]:
    source_descriptor = _open_relative_source(staging_root, source.path)
    target_descriptor = _open_output(target)
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    previous = previous_document_id
    try:
        with os.fdopen(os.dup(source_descriptor), "rb", closefd=True) as reader:
            while True:
                line = reader.readline(_MAX_LINE_BYTES + 1)
                if not line:
                    break
                digest.update(line)
                byte_count += len(line)
                written = os.write(target_descriptor, line)
                if written != len(line):
                    raise ProductionArtifactFactoryError("document copy made partial progress")
                record_count += 1
                row = _decode_document_line(
                    line,
                    label=f"{source.path} line {record_count}",
                )
                document_id = row["id"].encode("utf-8", errors="strict")
                if previous is not None and document_id <= previous:
                    raise ProductionArtifactFactoryError(
                        f"{corpus_id} document IDs are not globally bytewise sorted"
                    )
                content_digest = custody_document_content_sha256(
                    corpus_id,
                    row["title"],
                    row["text"],
                )
                if os.write(provenance_descriptor, content_digest) != len(content_digest):
                    raise ProductionArtifactFactoryError("provenance write made partial progress")
                provenance_digest.update(content_digest)
                previous = document_id
        os.fsync(target_descriptor)
        after = os.fstat(source_descriptor)
        if (
            after.st_size != source.byte_count
            or byte_count != source.byte_count
            or digest.hexdigest() != source.sha256
            or record_count != source.record_count
        ):
            raise ProductionArtifactFactoryError(
                f"document source {source.path!r} differs from its audit pin"
            )
    finally:
        os.close(target_descriptor)
        os.close(source_descriptor)
    assert previous is not None
    return record_count, previous


def _write_corpus_leaves(
    work: Path,
    *,
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    corpus_id: str,
    embedding: EmbeddingStoreReceipt,
) -> tuple[CorpusShardInventory, ImmutableArtifactPin, ProvenanceSidecarDescriptor]:
    shard_root = work / "corpus" / "shards"
    provenance_root = work / "provenance"
    shard_root.mkdir(parents=True, mode=0o700)
    provenance_root.mkdir(mode=0o700)
    provenance_path = work / ONLINE_PROVENANCE_PATH
    provenance_descriptor = _open_output(provenance_path)
    provenance_digest = hashlib.sha256()
    previous: bytes | None = None
    first_document = 0
    shards: list[CorpusShard] = []
    try:
        for position, (_config_row, source) in enumerate(_document_sources(inputs, corpus_id)):
            relative = f"corpus/shards/{position:05d}.jsonl"
            count, previous = _copy_document_source(
                staging_root=inputs.embedding_config.online_staging_root,
                source=source,
                target=work / relative,
                corpus_id=corpus_id,
                provenance_descriptor=provenance_descriptor,
                provenance_digest=provenance_digest,
                previous_document_id=previous,
            )
            shards.append(
                CorpusShard(
                    artifact=ImmutableArtifactPin(
                        artifact_id=f"{corpus_id}-corpus-shard-{position:05d}",
                        relative_path=relative,
                        kind="file",
                        byte_count=source.byte_count,
                        sha256=source.sha256,
                    ),
                    first_document_id=first_document,
                    document_count=count,
                )
            )
            first_document += count
        os.fsync(provenance_descriptor)
    finally:
        os.close(provenance_descriptor)
    if first_document != embedding.document_count:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} document source count differs from the embedding store"
        )
    universe = embedding.row_orders["documents"].row_order_sha256
    inventory = CorpusShardInventory(
        corpus=corpus_id,
        stage="sealed",
        document_count=embedding.document_count,
        ordered_document_universe_sha256=universe,
        shards=tuple(shards),
    )
    inventory_path = work / ONLINE_INVENTORY_PATH
    write_corpus_shard_inventory(inventory, inventory_path)
    inventory_pin = ImmutableArtifactPin(
        artifact_id=f"{corpus_id}-corpus-shard-inventory",
        relative_path=ONLINE_INVENTORY_PATH,
        kind="file",
        byte_count=len(inventory.canonical_file_bytes()),
        sha256=inventory.file_sha256,
    )
    provenance_pin = ImmutableArtifactPin(
        artifact_id=f"{corpus_id}-document-content-provenance",
        relative_path=ONLINE_PROVENANCE_PATH,
        kind="file",
        byte_count=embedding.document_count * 32,
        sha256=provenance_digest.hexdigest(),
    )
    return (
        inventory,
        inventory_pin,
        ProvenanceSidecarDescriptor(
            artifact=provenance_pin,
            record_count=embedding.document_count,
            document_universe_sha256=universe,
        ),
    )


def _write_raw_matrix(
    matrix: np.memmap,
    target: Path,
    *,
    artifact_id: str,
    relative_path: str,
) -> ImmutableArtifactPin:
    descriptor = _open_output(target)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        for start in range(0, matrix.shape[0], 8192):
            stop = min(start + 8192, matrix.shape[0])
            payload = np.asarray(matrix[start:stop], dtype=np.dtype("<f4"), order="C").tobytes(
                order="C"
            )
            digest.update(payload)
            byte_count += len(payload)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ProductionArtifactFactoryError("raw vector write made no progress")
                view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    expected = matrix.shape[0] * matrix.shape[1] * 4
    if byte_count != expected:
        raise ProductionArtifactFactoryError("raw vector byte count differs from its shape")
    return ImmutableArtifactPin(
        artifact_id=artifact_id,
        relative_path=relative_path,
        kind="file",
        byte_count=byte_count,
        sha256=digest.hexdigest(),
    )


def _build_full_hnsw(
    matrix: np.memmap,
    target: Path,
    *,
    corpus_id: str,
    config: AuthorizedIndexConfig,
    backend: HnswlibBackend,
) -> ImmutableArtifactPin:
    index = backend.create_index(metric=config.metric, dimension=matrix.shape[1])
    index.init_index(
        max_elements=matrix.shape[0],
        ef_construction=config.ef_construction,
        M=config.m,
        random_seed=config.random_seed,
        allow_replace_deleted=False,
    )
    index.set_num_threads(1)
    for start in range(0, matrix.shape[0], config.batch_size):
        stop = min(start + config.batch_size, matrix.shape[0])
        vectors = np.asarray(matrix[start:stop], dtype=np.float32, order="C")
        labels = np.arange(start, stop, dtype=np.int64)
        index.add_items(vectors, labels, num_threads=1)
    if os.path.lexists(target):
        raise ProductionArtifactFactoryError("full HNSW target already exists")
    index.save_index(str(target))
    os.chmod(target, 0o600)
    descriptor = os.open(target, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        metadata = target.lstat()
        digest = digest_regular_file(target, label="full active HNSW")
    except (OSError, ArtifactIntegrityError) as exc:
        raise ProductionArtifactFactoryError(f"cannot pin full HNSW index: {exc}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
    ):
        raise ProductionArtifactFactoryError("full HNSW index is not one nonempty regular file")
    del index
    check = backend.create_index(metric=config.metric, dimension=matrix.shape[1])
    check.load_index(str(target), max_elements=matrix.shape[0])
    check.set_num_threads(1)
    check.set_ef(min(config.verification_ef, matrix.shape[0]))
    labels, distances = check.knn_query(
        np.asarray(matrix[:1], dtype=np.float32, order="C"),
        k=1,
        num_threads=1,
    )
    if np.asarray(labels).shape != (1, 1) or not np.isfinite(np.asarray(distances)).all():
        raise ProductionArtifactFactoryError("full HNSW smoke query failed")
    return ImmutableArtifactPin(
        artifact_id=f"{corpus_id}-full-active-hnsw",
        relative_path=ONLINE_HNSW_PATH,
        kind="file",
        byte_count=metadata.st_size,
        sha256=digest,
    )


def _full_hnsw_format_revision(config: AuthorizedIndexConfig) -> str:
    return (
        f"{config.backend_id}-{config.backend_version}-{config.backend_build_sha256}-"
        f"{config.metric}-m{config.m}-ef{config.ef_construction}-"
        f"seed{config.random_seed}-full-v1"
    )


def _fsync_directory_path(path: Path, *, label: str) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot open {label} for fsync: {exc}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot fsync {label}: {exc}") from exc
    finally:
        os.close(descriptor)


def _verify_full_hnsw_file(
    matrix: np.memmap,
    path: Path,
    *,
    config: AuthorizedIndexConfig,
    backend: HnswlibBackend,
    expected_byte_count: int,
    expected_sha256: str,
) -> None:
    try:
        metadata = path.lstat()
        observed_sha256 = digest_regular_file(path, label="full active HNSW replicate")
    except (OSError, ArtifactIntegrityError) as exc:
        raise ProductionArtifactFactoryError(f"cannot verify full HNSW replicate: {exc}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size != expected_byte_count
        or observed_sha256 != expected_sha256
    ):
        raise ProductionArtifactFactoryError("full HNSW replicate differs from its byte pin")
    check = backend.create_index(metric=config.metric, dimension=matrix.shape[1])
    try:
        check.load_index(str(path), max_elements=matrix.shape[0])
        check.set_num_threads(FACTORY_INDEX_NUM_THREADS)
        check.set_ef(min(config.verification_ef, matrix.shape[0]))
        labels, distances = check.knn_query(
            np.asarray(matrix[:1], dtype=np.float32, order="C"),
            k=1,
            num_threads=FACTORY_INDEX_NUM_THREADS,
        )
        if (
            np.asarray(labels).shape != (1, 1)
            or np.asarray(distances).shape != (1, 1)
            or not np.isfinite(np.asarray(distances)).all()
            or type(np.asarray(labels)[0, 0].item()) is not int
            or not 0 <= int(np.asarray(labels)[0, 0]) < matrix.shape[0]
        ):
            raise ProductionArtifactFactoryError("full HNSW replicate smoke query failed")
    finally:
        del check
    try:
        after = path.lstat()
    except OSError as exc:
        raise ProductionArtifactFactoryError(
            f"cannot reinspect full HNSW replicate: {exc}"
        ) from exc
    if _stat_signature(after) != _stat_signature(metadata):
        raise ProductionArtifactFactoryError("full HNSW replicate changed during verification")


def _full_hnsw_reproducibility_root(
    config: ProductionArtifactFactoryConfig,
    corpus_id: str,
) -> Path:
    return config.reproducibility_root / corpus_id / FULL_HNSW_REPRODUCIBILITY_DIRECTORY


def _verify_full_hnsw_replicate(
    matrix: np.memmap,
    root: Path,
    *,
    replicate: int,
    config: ProductionArtifactFactoryConfig,
    backend: HnswlibBackend,
) -> FullHnswReplicateEvidence:
    _require_real_directory(root, label="full HNSW replicate root", private=True)
    expected_members = {
        FULL_HNSW_REPLICATE_FILENAME,
        FULL_HNSW_REPLICATE_RECEIPT_FILENAME,
    }
    try:
        members = {path.name for path in root.iterdir()}
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot scan full HNSW replicate: {exc}") from exc
    if members != expected_members:
        raise ProductionArtifactFactoryError("full HNSW replicate membership differs")
    evidence = _load_full_hnsw_replicate(root / FULL_HNSW_REPLICATE_RECEIPT_FILENAME)
    if evidence.replicate != replicate:
        raise ProductionArtifactFactoryError("full HNSW replicate number differs")
    _verify_full_hnsw_file(
        matrix,
        root / FULL_HNSW_REPLICATE_FILENAME,
        config=config.index_config,
        backend=backend,
        expected_byte_count=evidence.byte_count,
        expected_sha256=evidence.sha256,
    )
    return evidence


def _build_full_hnsw_replicate(
    matrix: np.memmap,
    parent: Path,
    *,
    corpus_id: str,
    replicate: int,
    config: ProductionArtifactFactoryConfig,
    backend: HnswlibBackend,
) -> FullHnswReplicateEvidence:
    directory = config.index_replicate_directories[replicate - 1]
    output = parent / directory
    work = parent / f".{directory}.staging-{secrets.token_hex(12)}"
    if os.path.lexists(output):
        raise ProductionArtifactFactoryError("full HNSW replicate target already exists")
    started_ns = time.monotonic_ns()
    try:
        work.mkdir(mode=0o700)
        pin = _build_full_hnsw(
            matrix,
            work / FULL_HNSW_REPLICATE_FILENAME,
            corpus_id=corpus_id,
            config=config.index_config,
            backend=backend,
        )
        evidence = FullHnswReplicateEvidence(
            replicate=replicate,
            relative_path=f"{directory}/{FULL_HNSW_REPLICATE_FILENAME}",
            byte_count=pin.byte_count,
            sha256=pin.sha256,
            elapsed_monotonic_ns=time.monotonic_ns() - started_ns,
            process_peak_rss_bytes=_peak_rss_bytes(),
        )
        write_exclusive_receipt_bytes(
            evidence.canonical_file_bytes(),
            work / FULL_HNSW_REPLICATE_RECEIPT_FILENAME,
        )
        _fsync_directory_path(work, label="full HNSW replicate staging")
        _exclusive_publish_directory(work, output, label="full HNSW replicate")
        _fsync_directory_path(parent, label="full HNSW reproducibility root")
        observed = _verify_full_hnsw_replicate(
            matrix,
            output,
            replicate=replicate,
            config=config,
            backend=backend,
        )
        if observed != evidence:
            raise ProductionArtifactFactoryError("published full HNSW evidence differs")
        return observed
    except (ArtifactIntegrityError, OSError) as exc:
        raise ProductionArtifactFactoryError(
            f"cannot build full HNSW replicate {replicate}: {exc}"
        ) from exc
    finally:
        if os.path.lexists(work):
            _validate_and_remove_temporary_tree(work, label="full HNSW staging cleanup")


def _admit_full_hnsw_reproducibility(
    config: ProductionArtifactFactoryConfig,
    *,
    corpus_id: str,
    embedding: EmbeddingStoreReceipt,
    matrix: np.memmap,
    source_vector_sha256: str,
    backend: HnswlibBackend,
    build_missing: bool,
    recover_partials: bool = False,
) -> FullHnswReproducibilityReceipt:
    _require_sha256("full HNSW source_vector_sha256", source_vector_sha256)
    root = _full_hnsw_reproducibility_root(config, corpus_id)
    if build_missing:
        _prepare_private_directory(root, label=f"{corpus_id} full HNSW reproducibility")
    else:
        _require_real_directory(
            root,
            label=f"{corpus_id} full HNSW reproducibility",
            private=True,
        )
    receipt_path = root / FULL_HNSW_REPRODUCIBILITY_RECEIPT_FILENAME
    existing = (
        _load_full_hnsw_reproducibility(receipt_path) if os.path.lexists(receipt_path) else None
    )
    rows: list[FullHnswReplicateEvidence] = []
    for replicate, directory in enumerate(config.index_replicate_directories, start=1):
        if recover_partials:
            _recover_factory_staging_directories(
                root,
                prefix=f".{directory}.staging-",
                label=f"{corpus_id} full HNSW replicate staging",
            )
        replicate_root = root / directory
        if os.path.lexists(replicate_root):
            observed = _verify_full_hnsw_replicate(
                matrix,
                replicate_root,
                replicate=replicate,
                config=config,
                backend=backend,
            )
        else:
            if existing is not None:
                raise ProductionArtifactFactoryError(
                    f"{corpus_id} full HNSW receipt exists but replicate {replicate} is missing"
                )
            if not build_missing:
                raise ProductionArtifactFactoryError(
                    f"{corpus_id} full HNSW replicate {replicate} is missing"
                )
            observed = _build_full_hnsw_replicate(
                matrix,
                root,
                corpus_id=corpus_id,
                replicate=replicate,
                config=config,
                backend=backend,
            )
        if existing is not None and observed != existing.replicates[replicate - 1]:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} full HNSW replicate {replicate} differs from its receipt"
            )
        rows.append(observed)
    vector = embedding.vectors.get("old_documents")
    if vector is None or tuple(vector.shape) != tuple(matrix.shape):
        raise ProductionArtifactFactoryError("full HNSW source vector binding differs")
    result = FullHnswReproducibilityReceipt(
        factory_config_sha256=config.file_sha256,
        runner_image=config.runner_image,
        runner_platform=config.runner_platform,
        corpus_id=corpus_id,
        backend_id=config.index_config.backend_id,
        backend_version=config.index_config.backend_version,
        backend_build_sha256=config.index_config.backend_build_sha256,
        source_vector_sha256=source_vector_sha256,
        document_count=matrix.shape[0],
        dimension=matrix.shape[1],
        format_revision=_full_hnsw_format_revision(config.index_config),
        replicates=tuple(rows),
        selected_replicate=config.selected_index_replicate,
        selected_final_sha256=rows[config.selected_index_replicate - 1].sha256,
    )
    if existing is not None:
        if existing != result:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} full HNSW reproducibility receipt differs"
            )
    elif build_missing:
        try:
            write_exclusive_receipt_bytes(result.canonical_file_bytes(), receipt_path)
            _fsync_directory_path(root, label="full HNSW reproducibility root")
        except ArtifactIntegrityError as exc:
            raise ProductionArtifactFactoryError(
                f"cannot write {corpus_id} full HNSW reproducibility receipt: {exc}"
            ) from exc
    else:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} full HNSW reproducibility receipt is missing"
        )
    expected_members = {
        FULL_HNSW_REPRODUCIBILITY_RECEIPT_FILENAME,
        *config.index_replicate_directories,
    }
    try:
        observed_members = {path.name for path in root.iterdir()}
    except OSError as exc:
        raise ProductionArtifactFactoryError(
            f"cannot scan {corpus_id} full HNSW reproducibility root: {exc}"
        ) from exc
    if observed_members != expected_members:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} full HNSW reproducibility membership differs"
        )
    return result


def _copy_file_exclusive(source: Path, target: Path, *, expected_sha256: str) -> None:
    source_descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    target_descriptor = _open_output(target)
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(source_descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_descriptor, view)
                if written <= 0:
                    raise ProductionArtifactFactoryError("file copy made no progress")
                view = view[written:]
        os.fsync(target_descriptor)
    finally:
        os.close(target_descriptor)
        os.close(source_descriptor)
    if digest.hexdigest() != expected_sha256:
        raise ProductionArtifactFactoryError("copied file differs from its source pin")


def _install_selected_full_hnsw(
    matrix: np.memmap,
    work: Path,
    reproducibility_root: Path,
    receipt: FullHnswReproducibilityReceipt,
    *,
    config: AuthorizedIndexConfig,
    backend: HnswlibBackend,
) -> ImmutableArtifactPin:
    """Copy and re-admit only the registered full-active replica."""

    selected = receipt.replicates[receipt.selected_replicate - 1]
    source = reproducibility_root / selected.relative_path
    target = work / ONLINE_HNSW_PATH
    _copy_file_exclusive(source, target, expected_sha256=selected.sha256)
    _verify_full_hnsw_file(
        matrix,
        target,
        config=config,
        backend=backend,
        expected_byte_count=selected.byte_count,
        expected_sha256=selected.sha256,
    )
    try:
        index_members = {path.name for path in target.parent.iterdir()}
        hnsw_paths = {
            path.relative_to(work).as_posix() for path in work.rglob("*.hnsw") if path.is_file()
        }
    except OSError as exc:
        raise ProductionArtifactFactoryError(f"cannot scan selected online HNSW: {exc}") from exc
    if index_members != {target.name} or hnsw_paths != {ONLINE_HNSW_PATH}:
        raise ProductionArtifactFactoryError("online package contains another HNSW copy")
    return ImmutableArtifactPin(
        artifact_id=f"{receipt.corpus_id}-full-active-hnsw",
        relative_path=ONLINE_HNSW_PATH,
        kind="file",
        byte_count=selected.byte_count,
        sha256=selected.sha256,
    )


def _ensure_query_package(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    *,
    corpus_id: str,
    hmac_secret: bytes | None,
) -> Any:
    runtime_root = config.artifact_root / "trial-runtime" / corpus_id
    _prepare_private_directory(runtime_root, label=f"{corpus_id} trial runtime root")
    query_root = runtime_root / RUNTIME_QUERY_DIRECTORY
    corpus_config = config.corpus(corpus_id)
    embedding_root = config.artifact_root / "embedding-stores" / corpus_id
    try:
        if os.path.lexists(query_root):
            receipt = verify_query_trial_store(
                query_root,
                inputs.embedding_config.online_staging_root,
                embedding_root,
                partition_audit_path=config.partition_audit_path,
            )
        else:
            if hmac_secret is None:
                raise ProductionArtifactFactoryError(
                    f"{corpus_id} query package is absent and no HMAC secret FD was supplied"
                )
            receipt = build_query_trial_store(
                inputs.embedding_config.online_staging_root,
                embedding_root,
                query_root,
                partition_audit_path=config.partition_audit_path,
                corpus=corpus_id,
                stage="sealed",
                hmac_key_id=config.hmac_key_id,
                hmac_secret=hmac_secret,
                selection_seed_sha256=config.selection_seed_sha256,
                available_family_count=corpus_config.available_family_count,
                selected_family_count=inputs.selected_family_count,
            )
    except TrialRuntimeError as exc:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} query/trial package failed: {exc}"
        ) from exc
    if receipt.hmac_key_id != config.hmac_key_id:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} query receipt HMAC key ID differs from the frozen commitment"
        )
    return receipt


def _online_package_root(config: ProductionArtifactFactoryConfig, corpus_id: str) -> Path:
    return config.artifact_root / "custody" / "online" / corpus_id


def _verify_online_source_binding(
    root: Path,
    *,
    corpus_id: str,
    embedding: EmbeddingStoreReceipt,
    query_receipt: Any,
    config: ProductionArtifactFactoryConfig,
    full_hnsw: FullHnswReproducibilityReceipt,
) -> tuple[ShardedOnlineExecutionPlan, str]:
    try:
        tree = digest_directory_tree(root)
        plan = load_sharded_online_execution_plan(root / ONLINE_EXECUTION_PLAN_FILENAME)
        package = verify_online_execution_package(
            root,
            expected_tree_sha256=tree.sha256,
            expected_plan_revision=f"sha256:{plan.artifact_sha256}",
        )
    except (ArtifactIntegrityError, ScalableExecutionError) as exc:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} online execution package failed: {exc}"
        ) from exc
    old = embedding.vectors.get("old_documents")
    current = embedding.vectors.get("current_documents")
    if old is None or current is None:
        raise ProductionArtifactFactoryError("online plan needs paired document embeddings")
    selected_full_hnsw = full_hnsw.replicates[full_hnsw.selected_replicate - 1]
    if (
        package.plan != plan
        or plan.corpus != corpus_id
        or plan.stage != "sealed"
        or query_receipt.hmac_key_id != config.hmac_key_id
        or plan.key_id != config.hmac_key_id
        or plan.permutation_seed != config.permutation_seed
        or plan.query_partition_audit_sha256 != query_receipt.query_partition_audit_sha256
        or plan.trials != tuple(sorted(query_receipt.opaque_trials, key=lambda row: row.trial_key))
        or plan.document_count != embedding.document_count
        or plan.ordered_document_universe_sha256
        != embedding.row_orders["documents"].row_order_sha256
        or plan.active_vector_store.shape != old.shape
        or plan.current_truth_vector_store.shape != current.shape
        or full_hnsw.factory_config_sha256 != config.file_sha256
        or full_hnsw.runner_image != config.runner_image
        or full_hnsw.runner_platform != config.runner_platform
        or full_hnsw.corpus_id != corpus_id
        or full_hnsw.backend_id != config.index_config.backend_id
        or full_hnsw.backend_version != config.index_config.backend_version
        or full_hnsw.backend_build_sha256 != config.index_config.backend_build_sha256
        or full_hnsw.source_vector_sha256 != plan.active_vector_store.artifact.sha256
        or full_hnsw.document_count != embedding.document_count
        or full_hnsw.dimension != old.shape[1]
        or full_hnsw.format_revision != _full_hnsw_format_revision(config.index_config)
        or plan.hnsw_index.format_revision != full_hnsw.format_revision
        or plan.hnsw_index.artifact.byte_count != selected_full_hnsw.byte_count
        or plan.hnsw_index.artifact.sha256 != selected_full_hnsw.sha256
        or plan.hnsw_index.source_vector_sha256 != full_hnsw.source_vector_sha256
    ):
        raise ProductionArtifactFactoryError(
            f"{corpus_id} online execution plan differs from its frozen sources"
        )
    return plan, tree.sha256


def _build_online_package(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    *,
    corpus_id: str,
    embedding: EmbeddingStoreReceipt,
    query_receipt: Any,
    backend: HnswlibBackend,
    recover_partials: bool = False,
) -> tuple[ShardedOnlineExecutionPlan, str]:
    output = _online_package_root(config, corpus_id)
    parent = output.parent
    _require_real_directory(parent, label="online execution parent", private=True)
    if recover_partials:
        _recover_factory_staging_directories(
            parent,
            prefix=f".{corpus_id}.staging-",
            label=f"{corpus_id} online execution staging",
        )
    fit_index_receipt = load_authorized_index_store_receipt(
        config.artifact_root / "authorized-index-stores" / corpus_id / "fit"
    )
    if os.path.lexists(output):
        candidate_plan = load_sharded_online_execution_plan(output / ONLINE_EXECUTION_PLAN_FILENAME)
        with open_verified_document_matrices(
            config.artifact_root / "embedding-stores" / corpus_id,
            index_receipt=fit_index_receipt,
            expected_embedding_receipt_sha256=embedding.receipt_sha256,
        ) as matrices:
            full_hnsw = _admit_full_hnsw_reproducibility(
                config,
                corpus_id=corpus_id,
                embedding=embedding,
                matrix=matrices.old_active,
                source_vector_sha256=(candidate_plan.active_vector_store.artifact.sha256),
                backend=backend,
                build_missing=False,
                recover_partials=recover_partials,
            )
        return _verify_online_source_binding(
            output,
            corpus_id=corpus_id,
            embedding=embedding,
            query_receipt=query_receipt,
            config=config,
            full_hnsw=full_hnsw,
        )
    work = parent / f".{corpus_id}.staging-{secrets.token_hex(12)}"
    work.mkdir(mode=0o700)
    try:
        query_root = work / RUNTIME_QUERY_DIRECTORY
        query_root.mkdir(mode=0o700)
        runtime_query_root = (
            config.artifact_root / "trial-runtime" / corpus_id / RUNTIME_QUERY_DIRECTORY
        )
        _copy_file_exclusive(
            runtime_query_root / QUERY_TRIAL_FILENAME,
            query_root / QUERY_TRIAL_FILENAME,
            expected_sha256=query_receipt.query_trial_store_sha256,
        )
        _copy_file_exclusive(
            runtime_query_root / QUERY_TRIAL_RECEIPT_FILENAME,
            query_root / QUERY_TRIAL_RECEIPT_FILENAME,
            expected_sha256=query_receipt.receipt_sha256,
        )
        _inventory, inventory_pin, provenance = _write_corpus_leaves(
            work,
            config=config,
            inputs=inputs,
            corpus_id=corpus_id,
            embedding=embedding,
        )
        vector_root = work / "vectors"
        index_root = work / "indexes"
        vector_root.mkdir(mode=0o700)
        index_root.mkdir(mode=0o700)
        with open_verified_document_matrices(
            config.artifact_root / "embedding-stores" / corpus_id,
            index_receipt=fit_index_receipt,
            expected_embedding_receipt_sha256=embedding.receipt_sha256,
        ) as matrices:
            active_pin = _write_raw_matrix(
                matrices.old_active,
                work / ONLINE_ACTIVE_VECTOR_PATH,
                artifact_id=f"{corpus_id}-active-document-vectors",
                relative_path=ONLINE_ACTIVE_VECTOR_PATH,
            )
            truth_pin = _write_raw_matrix(
                matrices.current_truth,
                work / ONLINE_TRUTH_VECTOR_PATH,
                artifact_id=f"{corpus_id}-current-truth-document-vectors",
                relative_path=ONLINE_TRUTH_VECTOR_PATH,
            )
            full_hnsw = _admit_full_hnsw_reproducibility(
                config,
                corpus_id=corpus_id,
                embedding=embedding,
                matrix=matrices.old_active,
                source_vector_sha256=active_pin.sha256,
                backend=backend,
                build_missing=True,
                recover_partials=recover_partials,
            )
            hnsw_pin = _install_selected_full_hnsw(
                matrices.old_active,
                work,
                _full_hnsw_reproducibility_root(config, corpus_id),
                full_hnsw,
                config=config.index_config,
                backend=backend,
            )
            if active_pin.sha256 != full_hnsw.source_vector_sha256:
                raise ProductionArtifactFactoryError(
                    f"{corpus_id} selected full HNSW source vector differs"
                )
        universe = embedding.row_orders["documents"].row_order_sha256
        plan = ShardedOnlineExecutionPlan(
            key_id=config.hmac_key_id,
            corpus=corpus_id,
            stage="sealed",
            document_count=embedding.document_count,
            ordered_document_universe_sha256=universe,
            permutation_seed=config.permutation_seed,
            trials=query_receipt.opaque_trials,
            query_partition_audit_sha256=query_receipt.query_partition_audit_sha256,
            corpus_shard_inventory=inventory_pin,
            query_trial_store=query_receipt.store_descriptor(
                artifact_id=f"{corpus_id}-query-trials",
                relative_path=f"{RUNTIME_QUERY_DIRECTORY}/{QUERY_TRIAL_FILENAME}",
                receipt_artifact_id=f"{corpus_id}-query-trial-receipt",
                receipt_relative_path=(f"{RUNTIME_QUERY_DIRECTORY}/{QUERY_TRIAL_RECEIPT_FILENAME}"),
            ),
            active_vector_store=VectorStoreDescriptor(
                artifact=active_pin,
                role="active-migration",
                dtype="<f4",
                shape=embedding.vectors["old_documents"].shape,
                document_universe_sha256=universe,
            ),
            current_truth_vector_store=VectorStoreDescriptor(
                artifact=truth_pin,
                role="current-exact-truth",
                dtype="<f4",
                shape=embedding.vectors["current_documents"].shape,
                document_universe_sha256=universe,
            ),
            provenance_sha256_sidecar=provenance,
            hnsw_index=IndexArtifactDescriptor(
                artifact=hnsw_pin,
                document_count=embedding.document_count,
                document_universe_sha256=universe,
                source_vector_sha256=active_pin.sha256,
                format_revision=full_hnsw.format_revision,
            ),
        )
        write_sharded_online_execution_plan(plan, work / ONLINE_EXECUTION_PLAN_FILENAME)
        finalize_online_execution_package(work)
        _exclusive_publish_directory(work, output, label=f"{corpus_id} online execution")
        return _verify_online_source_binding(
            output,
            corpus_id=corpus_id,
            embedding=embedding,
            query_receipt=query_receipt,
            config=config,
            full_hnsw=full_hnsw,
        )
    except (
        ArtifactIntegrityError,
        AuthorizedIndexStoreError,
        OSError,
        ScalableExecutionError,
    ) as exc:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} online execution construction failed: {exc}"
        ) from exc
    finally:
        if os.path.lexists(work):
            _validate_and_remove_temporary_tree(
                work,
                label=f"{corpus_id} online-package staging cleanup",
            )


def _ensure_runtime_plan_copy(
    config: ProductionArtifactFactoryConfig,
    corpus_id: str,
    plan: ShardedOnlineExecutionPlan,
) -> Path:
    target = config.artifact_root / "trial-runtime" / corpus_id / ONLINE_EXECUTION_PLAN_FILENAME
    if os.path.lexists(target):
        loaded = load_sharded_online_execution_plan(target)
        if loaded != plan:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} runtime plan copy differs from online execution"
            )
    else:
        write_sharded_online_execution_plan(plan, target)
    return target


def _runtime_feature_bindings(schedule: Any) -> tuple[RuntimeFeatureBinding, ...]:
    rows: dict[tuple[int, str, int, str], RuntimeFeatureBinding] = {}
    for row in schedule.rows:
        key = (row.group_order, row.subject, row.repetition, row.policy_state)
        candidate = RuntimeFeatureBinding(
            group_order=row.group_order,
            subject=row.subject,
            repetition=row.repetition,
            policy_state=row.policy_state,
            version_lag=VERSION_LAG,
            backend=RUNTIME_BACKEND,
            drift_family=RUNTIME_DRIFT_FAMILY,
            policy_complexity=row.realized_allow_rate,
        )
        prior = rows.setdefault(key, candidate)
        if prior != candidate:
            raise ProductionArtifactFactoryError(
                "one policy block has inconsistent derived runtime features"
            )
    return tuple(rows[key] for key in sorted(rows))


def _ensure_runtime_admission(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    *,
    corpus_id: str,
    plan: ShardedOnlineExecutionPlan,
) -> Any:
    runtime_root = config.artifact_root / "trial-runtime" / corpus_id
    schedule_path = (
        config.artifact_root / "policy-workloads" / corpus_id / "sealed" / SCHEDULE_FILENAME
    )
    try:
        schedule = load_canonical_trial_schedule(schedule_path)
        features = _runtime_feature_bindings(schedule)
        expected = admit_trial_runtime(
            plan,
            runtime_root / RUNTIME_QUERY_DIRECTORY,
            inputs.embedding_config.online_staging_root,
            config.artifact_root / "embedding-stores" / corpus_id,
            schedule_path,
            features,
            partition_audit_path=config.partition_audit_path,
        )
        receipt_path = runtime_root / RUNTIME_RECEIPT_FILENAME
        if os.path.lexists(receipt_path):
            observed = load_trial_runtime_receipt(receipt_path)
            if observed != expected.receipt:
                raise ProductionArtifactFactoryError(
                    f"{corpus_id} runtime receipt differs from its reproduced admission"
                )
            return observed
        admitted = admit_trial_runtime(
            plan,
            runtime_root / RUNTIME_QUERY_DIRECTORY,
            inputs.embedding_config.online_staging_root,
            config.artifact_root / "embedding-stores" / corpus_id,
            schedule_path,
            features,
            partition_audit_path=config.partition_audit_path,
            receipt_target=receipt_path,
        )
        return admitted.receipt
    except (PolicyInterventionError, ScalableExecutionError, TrialRuntimeError) as exc:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} runtime admission failed: {exc}"
        ) from exc


def _corpus_had_outputs(config: ProductionArtifactFactoryConfig, corpus_id: str) -> bool:
    paths = (
        config.artifact_root / "policy-workloads" / corpus_id,
        config.artifact_root / "authorized-index-stores" / corpus_id,
        config.artifact_root / "trial-runtime" / corpus_id,
        _online_package_root(config, corpus_id),
        config.reproducibility_root / corpus_id,
        config.evidence_root / f"{corpus_id}.json",
    )
    for path in paths:
        if not os.path.lexists(path):
            continue
        if path.is_dir():
            try:
                if any(path.iterdir()):
                    return True
            except OSError:
                return True
        else:
            return True
    return False


def _evidence_matches(
    evidence: ProductionCorpusFactoryEvidence,
    *,
    config: ProductionArtifactFactoryConfig,
    corpus_id: str,
    embedding: EmbeddingStoreReceipt,
    policy_receipt_sha256: str,
    index_receipt_sha256: str,
    query_receipt_sha256: str,
    plan_sha256: str,
    online_tree_sha256: str,
    runtime_receipt_sha256: str,
) -> None:
    if (
        evidence.corpus_id != corpus_id
        or evidence.factory_config_sha256 != config.file_sha256
        or evidence.embedding_receipt_sha256 != embedding.receipt_sha256
        or evidence.policy_bundle_receipt_sha256 != policy_receipt_sha256
        or evidence.index_bundle_receipt_sha256 != index_receipt_sha256
        or evidence.query_receipt_sha256 != query_receipt_sha256
        or evidence.online_execution_plan_sha256 != plan_sha256
        or evidence.online_execution_tree_sha256 != online_tree_sha256
        or evidence.runtime_receipt_sha256 != runtime_receipt_sha256
    ):
        raise ProductionArtifactFactoryError(
            f"{corpus_id} factory evidence differs from reproduced artifacts"
        )


def _build_one_corpus(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    *,
    corpus_id: str,
    backend: HnswlibBackend,
    hmac_secret: bytes | None,
    was_existing: bool,
) -> ProductionCorpusFactoryEvidence:
    started_at = _utc_now()
    started_ns = time.monotonic_ns()
    embedding = inputs.embeddings[corpus_id]
    for stage in ("fit", "calibration"):
        execution = _development_plan(
            config,
            corpus_id,
            stage,  # type: ignore[arg-type]
            embedding,
        )
        _ensure_policy_stage(
            config,
            corpus_id=corpus_id,
            stage=stage,
            execution=execution,
        )
        _ensure_reproducible_index_stage(
            config,
            corpus_id=corpus_id,
            stage=stage,
            embedding=embedding,
            backend=backend,
            recover_partials=was_existing,
        )
        _verify_development_stage_parity(
            config,
            inputs,
            corpus_id=corpus_id,
            stage=stage,  # type: ignore[arg-type]
        )
    query_receipt = _ensure_query_package(
        config,
        inputs,
        corpus_id=corpus_id,
        hmac_secret=hmac_secret,
    )
    plan, online_tree_sha256 = _build_online_package(
        config,
        inputs,
        corpus_id=corpus_id,
        embedding=embedding,
        query_receipt=query_receipt,
        backend=backend,
        recover_partials=was_existing,
    )
    _ensure_runtime_plan_copy(config, corpus_id, plan)
    _ensure_policy_stage(
        config,
        corpus_id=corpus_id,
        stage="sealed",
        execution=plan,
    )
    _ensure_reproducible_index_stage(
        config,
        corpus_id=corpus_id,
        stage="sealed",
        embedding=embedding,
        backend=backend,
        recover_partials=was_existing,
    )
    policy_root = config.artifact_root / "policy-workloads" / corpus_id
    index_root = config.artifact_root / "authorized-index-stores" / corpus_id
    try:
        policy_bundle = seal_policy_stage_bundle(policy_root, corpus_id=corpus_id)
        index_bundle = seal_index_stage_bundle(
            index_root,
            corpus_id=corpus_id,
            embedding_store_root=config.artifact_root / "embedding-stores" / corpus_id,
            policy_bundle_root=policy_root,
        )
    except ArtifactStageBundleError as exc:
        raise ProductionArtifactFactoryError(f"{corpus_id} stage bundle failed: {exc}") from exc
    runtime_receipt = _ensure_runtime_admission(
        config,
        inputs,
        corpus_id=corpus_id,
        plan=plan,
    )
    evidence_path = config.evidence_root / f"{corpus_id}.json"
    if os.path.lexists(evidence_path):
        evidence = _load_corpus_evidence(evidence_path)
        _evidence_matches(
            evidence,
            config=config,
            corpus_id=corpus_id,
            embedding=embedding,
            policy_receipt_sha256=policy_bundle.receipt_sha256,
            index_receipt_sha256=index_bundle.receipt_sha256,
            query_receipt_sha256=query_receipt.receipt_sha256,
            plan_sha256=plan.artifact_sha256,
            online_tree_sha256=online_tree_sha256,
            runtime_receipt_sha256=runtime_receipt.receipt_sha256,
        )
        return evidence
    evidence = ProductionCorpusFactoryEvidence(
        corpus_id=corpus_id,
        factory_config_sha256=config.file_sha256,
        embedding_receipt_sha256=embedding.receipt_sha256,
        policy_bundle_receipt_sha256=policy_bundle.receipt_sha256,
        index_bundle_receipt_sha256=index_bundle.receipt_sha256,
        query_receipt_sha256=query_receipt.receipt_sha256,
        online_execution_plan_sha256=plan.artifact_sha256,
        online_execution_tree_sha256=online_tree_sha256,
        runtime_receipt_sha256=runtime_receipt.receipt_sha256,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        elapsed_monotonic_ns=time.monotonic_ns() - started_ns,
        process_peak_rss_bytes=_peak_rss_bytes(),
        status="resumed" if was_existing else "built",
    )
    try:
        write_exclusive_receipt_bytes(evidence.canonical_file_bytes(), evidence_path)
    except ArtifactIntegrityError as exc:
        raise ProductionArtifactFactoryError(
            f"cannot write {corpus_id} factory evidence: {exc}"
        ) from exc
    return evidence


def _verify_reproducibility_stage_outputs(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    receipt: IndexReproducibilityStageReceipt,
    *,
    backend: HnswlibBackend,
) -> None:
    corpus_id = receipt.corpus_id
    stage = receipt.stage
    policy_root = config.artifact_root / "policy-workloads" / corpus_id / stage
    policy = load_policy_intervention_receipt(policy_root / "intervention-receipt.json")
    stage_root = config.reproducibility_root / corpus_id / stage
    for row in receipt.replicates:
        root = stage_root / config.index_replicate_directories[row.replicate - 1]
        try:
            verify_authorized_index_store(
                root,
                embedding_store_root=config.artifact_root / "embedding-stores" / corpus_id,
                policy_intervention_root=policy_root,
                expected_embedding_receipt_sha256=inputs.embeddings[corpus_id].receipt_sha256,
                expected_policy_receipt_sha256=policy.artifact_sha256,
                backend=backend,
                expected_store_receipt_sha256=row.receipt_sha256,
            )
        except AuthorizedIndexStoreError as exc:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} {stage} replicate {row.replicate} failed verification: {exc}"
            ) from exc
        observed = _replicate_evidence(
            root,
            replicate=row.replicate,
            elapsed_ns=row.elapsed_monotonic_ns,
            peak_rss_bytes=row.process_peak_rss_bytes,
        )
        if observed != row:
            raise ProductionArtifactFactoryError(f"{corpus_id} {stage} replicate evidence differs")
    final = config.artifact_root / "authorized-index-stores" / corpus_id / stage
    try:
        verified = verify_authorized_index_store(
            final,
            embedding_store_root=config.artifact_root / "embedding-stores" / corpus_id,
            policy_intervention_root=policy_root,
            expected_embedding_receipt_sha256=inputs.embeddings[corpus_id].receipt_sha256,
            expected_policy_receipt_sha256=policy.artifact_sha256,
            backend=backend,
            expected_store_receipt_sha256=receipt.selected_final_receipt_sha256,
        )
    except AuthorizedIndexStoreError as exc:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} {stage} selected index failed verification: {exc}"
        ) from exc
    try:
        final_tree = digest_directory_tree(final)
    except ArtifactIntegrityError as exc:
        raise ProductionArtifactFactoryError(
            f"cannot hash {corpus_id} {stage} selected index: {exc}"
        ) from exc
    selected = receipt.replicates[config.selected_index_replicate - 1]
    if verified.receipt_sha256 != selected.receipt_sha256:
        raise ProductionArtifactFactoryError("selected index differs from replicate 1")
    if final_tree.sha256 != selected.tree_sha256:
        raise ProductionArtifactFactoryError("selected index bytes differ from replicate 1")


def _assemble_reproducibility_suite(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    *,
    backend: HnswlibBackend,
    publish: bool,
) -> IndexReproducibilitySuiteReceipt:
    rows: list[IndexReproducibilityStageReceipt] = []
    full_hnsw_rows: list[FullHnswReproducibilityReceipt] = []
    for corpus_id in FIXED_CORPORA:
        for stage in config.artifact_stage_order:
            receipt = _load_repro_stage(
                config.reproducibility_root / corpus_id / stage / "reproducibility-receipt.json"
            )
            _verify_reproducibility_stage_outputs(
                config,
                inputs,
                receipt,
                backend=backend,
            )
            rows.append(receipt)
        embedding = inputs.embeddings[corpus_id]
        fit_index = load_authorized_index_store_receipt(
            config.artifact_root / "authorized-index-stores" / corpus_id / "fit"
        )
        with open_verified_document_matrices(
            config.artifact_root / "embedding-stores" / corpus_id,
            index_receipt=fit_index,
            expected_embedding_receipt_sha256=embedding.receipt_sha256,
        ) as matrices:
            candidate_plan = load_sharded_online_execution_plan(
                _online_package_root(config, corpus_id) / ONLINE_EXECUTION_PLAN_FILENAME
            )
            full_hnsw_rows.append(
                _admit_full_hnsw_reproducibility(
                    config,
                    corpus_id=corpus_id,
                    embedding=embedding,
                    matrix=matrices.old_active,
                    source_vector_sha256=(candidate_plan.active_vector_store.artifact.sha256),
                    backend=backend,
                    build_missing=False,
                )
            )
    result = IndexReproducibilitySuiteReceipt(
        factory_config_sha256=config.file_sha256,
        runner_image=config.runner_image,
        runner_platform=config.runner_platform,
        replicate_count=config.index_replicate_count,
        selected_replicate=config.selected_index_replicate,
        stages=tuple(rows),
        full_hnsw_indexes=tuple(full_hnsw_rows),
    )
    path = config.reproducibility_receipt_path
    if os.path.lexists(path):
        if _load_repro_suite(path) != result:
            raise ProductionArtifactFactoryError("reproducibility suite receipt differs")
    elif publish:
        try:
            write_exclusive_receipt_bytes(result.canonical_file_bytes(), path)
        except ArtifactIntegrityError as exc:
            raise ProductionArtifactFactoryError(
                f"cannot write reproducibility suite receipt: {exc}"
            ) from exc
    else:
        raise ProductionArtifactFactoryError("reproducibility suite receipt is missing")
    return result


def _verify_one_corpus(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    *,
    corpus_id: str,
    backend: HnswlibBackend,
) -> ProductionCorpusFactoryEvidence:
    embedding = inputs.embeddings[corpus_id]
    policy_root = config.artifact_root / "policy-workloads" / corpus_id
    index_root = config.artifact_root / "authorized-index-stores" / corpus_id
    for stage in ("fit", "calibration"):
        execution = _development_plan(
            config,
            corpus_id,
            stage,  # type: ignore[arg-type]
            embedding,
        )
        try:
            verify_policy_intervention_package(
                policy_root / stage,
                execution,
                config.policy_config(corpus_id, stage),
            )
        except PolicyInterventionError as exc:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} {stage} policy verification failed: {exc}"
            ) from exc
        _verify_development_stage_parity(
            config,
            inputs,
            corpus_id=corpus_id,
            stage=stage,  # type: ignore[arg-type]
        )
    query_root = config.artifact_root / "trial-runtime" / corpus_id / RUNTIME_QUERY_DIRECTORY
    try:
        query = verify_query_trial_store(
            query_root,
            inputs.embedding_config.online_staging_root,
            config.artifact_root / "embedding-stores" / corpus_id,
            partition_audit_path=config.partition_audit_path,
        )
    except TrialRuntimeError as exc:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} query package verification failed: {exc}"
        ) from exc
    candidate_plan = load_sharded_online_execution_plan(
        _online_package_root(config, corpus_id) / ONLINE_EXECUTION_PLAN_FILENAME
    )
    fit_index_receipt = load_authorized_index_store_receipt(index_root / "fit")
    with open_verified_document_matrices(
        config.artifact_root / "embedding-stores" / corpus_id,
        index_receipt=fit_index_receipt,
        expected_embedding_receipt_sha256=embedding.receipt_sha256,
    ) as matrices:
        full_hnsw = _admit_full_hnsw_reproducibility(
            config,
            corpus_id=corpus_id,
            embedding=embedding,
            matrix=matrices.old_active,
            source_vector_sha256=candidate_plan.active_vector_store.artifact.sha256,
            backend=backend,
            build_missing=False,
        )
    plan, online_tree = _verify_online_source_binding(
        _online_package_root(config, corpus_id),
        corpus_id=corpus_id,
        embedding=embedding,
        query_receipt=query,
        config=config,
        full_hnsw=full_hnsw,
    )
    plan_copy = config.artifact_root / "trial-runtime" / corpus_id / ONLINE_EXECUTION_PLAN_FILENAME
    if not os.path.lexists(plan_copy):
        raise ProductionArtifactFactoryError(f"{corpus_id} runtime plan copy is missing")
    if load_sharded_online_execution_plan(plan_copy) != plan:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} runtime plan copy differs from online execution"
        )
    try:
        verify_policy_intervention_package(
            policy_root / "sealed",
            plan,
            config.policy_config(corpus_id, "sealed"),
        )
        policy_bundle = verify_policy_stage_bundle(
            policy_root,
            expected_corpus_id=corpus_id,
        )
        index_bundle = verify_index_stage_bundle(
            index_root,
            embedding_store_root=config.artifact_root / "embedding-stores" / corpus_id,
            policy_bundle_root=policy_root,
            expected_corpus_id=corpus_id,
        )
    except (ArtifactStageBundleError, PolicyInterventionError) as exc:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} policy/index bundle verification failed: {exc}"
        ) from exc
    receipt_path = config.artifact_root / "trial-runtime" / corpus_id / RUNTIME_RECEIPT_FILENAME
    if not os.path.lexists(receipt_path):
        raise ProductionArtifactFactoryError(f"{corpus_id} runtime receipt is missing")
    runtime = _ensure_runtime_admission(
        config,
        inputs,
        corpus_id=corpus_id,
        plan=plan,
    )
    evidence = _load_corpus_evidence(config.evidence_root / f"{corpus_id}.json")
    _evidence_matches(
        evidence,
        config=config,
        corpus_id=corpus_id,
        embedding=embedding,
        policy_receipt_sha256=policy_bundle.receipt_sha256,
        index_receipt_sha256=index_bundle.receipt_sha256,
        query_receipt_sha256=query.receipt_sha256,
        plan_sha256=plan.artifact_sha256,
        online_tree_sha256=online_tree,
        runtime_receipt_sha256=runtime.receipt_sha256,
    )
    return evidence


def _suite_from_evidence(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    evidence: Sequence[ProductionCorpusFactoryEvidence],
    reproducibility: IndexReproducibilitySuiteReceipt,
    pipeline: ArtifactPipelineReceipt,
    *,
    embedding_destination_tree_sha256: str,
) -> ProductionArtifactFactorySuiteReceipt:
    evidence = tuple(evidence)
    if (
        len(evidence) != len(FIXED_CORPORA)
        or not all(isinstance(item, ProductionCorpusFactoryEvidence) for item in evidence)
        or tuple(item.corpus_id for item in evidence) != FIXED_CORPORA
    ):
        raise ProductionArtifactFactoryError(
            "factory suite evidence must contain exactly FIXED_CORPORA in protocol order"
        )
    rows: list[FactorySuiteCorpus] = []
    for corpus_id, item in zip(FIXED_CORPORA, evidence):
        path = config.evidence_root / f"{corpus_id}.json"
        try:
            file_sha256 = digest_regular_file(path, label=f"{corpus_id} factory evidence")
        except ArtifactIntegrityError as exc:
            raise ProductionArtifactFactoryError(
                f"cannot hash {corpus_id} factory evidence: {exc}"
            ) from exc
        rows.append(
            FactorySuiteCorpus(
                corpus_id=corpus_id,
                evidence_sha256=item.sha256,
                evidence_file_sha256=file_sha256,
                policy_bundle_receipt_sha256=item.policy_bundle_receipt_sha256,
                index_bundle_receipt_sha256=item.index_bundle_receipt_sha256,
                query_receipt_sha256=item.query_receipt_sha256,
                online_execution_plan_sha256=item.online_execution_plan_sha256,
                online_execution_tree_sha256=item.online_execution_tree_sha256,
                runtime_receipt_sha256=item.runtime_receipt_sha256,
            )
        )
    return ProductionArtifactFactorySuiteReceipt(
        factory_config_sha256=config.file_sha256,
        runner_image=config.runner_image,
        runner_platform=config.runner_platform,
        embedding_source_tree_sha256=inputs.embedding_source_tree_sha256,
        embedding_destination_tree_sha256=embedding_destination_tree_sha256,
        embedding_suite_receipt_sha256=inputs.embedding_suite.receipt_sha256,
        hmac_key_id=config.hmac_key_id,
        hmac_secret_sha256=config.hmac_secret_sha256,
        online_inventory_sha256=inputs.embedding_config.online_inventory_sha256,
        index_reproducibility_receipt_sha256=reproducibility.receipt_sha256,
        artifact_pipeline_receipt_sha256=pipeline.receipt_sha256,
        corpora=tuple(rows),
    )


def _require_factory_platform() -> None:
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
        raise ProductionArtifactFactoryError(
            "production artifact construction and verification require Linux arm64"
        )


def _fixed_index_config(backend: HnswlibBackend) -> AuthorizedIndexConfig:
    try:
        return AuthorizedIndexConfig(
            backend_version=backend.package_version,
            backend_build_sha256=backend.build_sha256,
            metric=FACTORY_INDEX_METRIC,
            m=FACTORY_INDEX_M,
            ef_construction=FACTORY_INDEX_EF_CONSTRUCTION,
            random_seed=FACTORY_INDEX_RANDOM_SEED,
            batch_size=FACTORY_INDEX_BATCH_SIZE,
            verification_ef=FACTORY_INDEX_VERIFICATION_EF,
            num_threads=FACTORY_INDEX_NUM_THREADS,
        )
    except AuthorizedIndexStoreError as exc:
        raise ProductionArtifactFactoryError(
            f"installed HNSW backend cannot form the fixed index config: {exc}"
        ) from exc


def production_authorized_index_components() -> tuple[AuthorizedIndexConfig, HnswlibBackend]:
    """Pin the C0 HNSW extension and return its fixed production config and backend."""

    _require_factory_platform()
    try:
        backend = HnswlibBackend()
    except AuthorizedIndexStoreError as exc:
        raise ProductionArtifactFactoryError(f"cannot pin installed HNSW backend: {exc}") from exc
    return _fixed_index_config(backend), backend


def _backend(config: ProductionArtifactFactoryConfig) -> HnswlibBackend:
    _require_factory_platform()
    try:
        backend = HnswlibBackend()
    except AuthorizedIndexStoreError as exc:
        raise ProductionArtifactFactoryError(f"cannot load pinned HNSW backend: {exc}") from exc
    if (
        backend.backend_id != config.index_config.backend_id
        or backend.package_version != config.index_config.backend_version
        or backend.build_sha256 != config.index_config.backend_build_sha256
    ):
        raise ProductionArtifactFactoryError(
            "installed HNSW extension differs from the factory config"
        )
    return backend


def _expected_shard_request(
    config: ProductionArtifactFactoryConfig,
    corpus_id: str,
) -> ProductionArtifactFactoryShardRequest:
    try:
        return next(
            request
            for request in derive_production_artifact_factory_shard_requests(config)
            if request.corpus_id == corpus_id
        )
    except StopIteration as exc:  # pragma: no cover - guarded by the typed corpus set
        raise ProductionArtifactFactoryError("factory shard corpus is outside the config") from exc


def _verify_shard_request_binding(
    config: ProductionArtifactFactoryConfig,
    request: ProductionArtifactFactoryShardRequest,
) -> None:
    if not isinstance(request, ProductionArtifactFactoryShardRequest):
        raise ProductionArtifactFactoryError("factory shard request must be typed")
    expected = _expected_shard_request(config, request.corpus_id)
    if request != expected:
        raise ProductionArtifactFactoryError(
            f"{request.corpus_id} shard request differs from the full pinned factory config"
        )


def _verify_shard_hmac_custody(
    config: ProductionArtifactFactoryConfig,
    request: ProductionArtifactFactoryShardRequest,
    hmac_secret: bytes,
) -> None:
    if not isinstance(hmac_secret, bytes) or not 32 <= len(hmac_secret) <= 4096:
        raise ProductionArtifactFactoryError("HMAC secret FD must contain 32 to 4096 bytes")
    observed_sha256 = hashlib.sha256(hmac_secret).hexdigest()
    observed_key_id = f"sealed-online-ephemeral-sha256-{observed_sha256}"
    if (
        observed_sha256 != request.hmac_secret_sha256
        or observed_sha256 != config.hmac_secret_sha256
        or observed_key_id != request.hmac_key_id
        or observed_key_id != config.hmac_key_id
    ):
        raise ProductionArtifactFactoryError(
            "factory shard HMAC bytes, commitment, and key ID do not identify one secret"
        )


def _derive_factory_shard_receipt(
    config: ProductionArtifactFactoryConfig,
    request: ProductionArtifactFactoryShardRequest,
    evidence: ProductionCorpusFactoryEvidence,
) -> ProductionArtifactFactoryShardReceipt:
    _verify_shard_request_binding(config, request)
    corpus_id = request.corpus_id
    if evidence.corpus_id != corpus_id or evidence.factory_config_sha256 != config.file_sha256:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} evidence does not belong to its shard request"
        )
    artifacts: list[ProductionArtifactFactoryShardArtifact] = []
    for index, relative_path in enumerate(request.owned_relative_paths):
        path = config.artifact_root / relative_path
        try:
            if index == len(request.owned_relative_paths) - 1:
                digest = digest_regular_file(path, label=f"{corpus_id} shard evidence")
                kind: Literal["file", "tree"] = "file"
            else:
                digest = digest_directory_tree(path).sha256
                kind = "tree"
        except ArtifactIntegrityError as exc:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} shard artifact {relative_path!r} failed hashing: {exc}"
            ) from exc
        artifacts.append(
            ProductionArtifactFactoryShardArtifact(
                relative_path=relative_path,
                artifact_kind=kind,
                sha256=digest,
            )
        )
    evidence_file_sha256 = artifacts[-1].sha256
    if evidence_file_sha256 != evidence.sha256:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} shard evidence file differs from its canonical record"
        )
    online_artifact = artifacts[3]
    if online_artifact.sha256 != evidence.online_execution_tree_sha256:
        raise ProductionArtifactFactoryError(
            f"{corpus_id} online tree differs from its corpus evidence"
        )
    return ProductionArtifactFactoryShardReceipt(
        request_sha256=request.request_sha256,
        factory_config_sha256=config.file_sha256,
        corpus_id=corpus_id,
        runner_image=config.runner_image,
        runner_platform=config.runner_platform,
        hmac_key_id=config.hmac_key_id,
        hmac_secret_sha256=config.hmac_secret_sha256,
        corpus_evidence_sha256=evidence.sha256,
        corpus_evidence_file_sha256=evidence_file_sha256,
        artifacts=tuple(artifacts),
    )


def _require_factory_control_destination(
    config: ProductionArtifactFactoryConfig,
    destination: Path,
    *,
    embedding_config: ProductionEmbeddingConfig,
    label: str,
) -> None:
    source_paths = (
        config.artifact_root,
        config.embedding_build_config_path,
        config.embedding_source_root,
        config.development_operator_root,
        config.partition_audit_path,
        config.joint_power_report_path,
        embedding_config.online_staging_root,
        embedding_config.current_model_root,
        embedding_config.stale_model_root,
    )
    if any(_paths_overlap(destination, source) for source in source_paths):
        raise ProductionArtifactFactoryError(f"{label} cannot overlap an input or artifact root")


def prepare_production_artifact_factory_shards(
    config: ProductionArtifactFactoryConfig,
    *,
    request_directory: str | Path,
) -> tuple[ProductionArtifactFactoryShardRequest, ...]:
    """Admit once, copy embeddings, and publish the exact five shard requests."""

    if not isinstance(config, ProductionArtifactFactoryConfig):
        raise ProductionArtifactFactoryError("factory config must be typed")
    request_root = _canonical_absolute_path(
        str(request_directory), label="factory shard request directory"
    )
    inputs = _admit_factory_inputs(config)
    _require_factory_control_destination(
        config,
        request_root,
        embedding_config=inputs.embedding_config,
        label="factory shard request directory",
    )
    if os.path.lexists(request_root):
        raise ProductionArtifactFactoryError("factory shard request directory already exists")
    _validate_factory_root_membership(config, fresh=True)
    _backend(config)
    _require_real_directory(
        request_root.parent,
        label="factory shard request directory parent",
        private=True,
    )
    requests = derive_production_artifact_factory_shard_requests(config)
    work = request_root.parent / f".{request_root.name}.partial-{secrets.token_hex(12)}"
    try:
        work.mkdir(mode=0o700)
        for index, request in enumerate(requests, start=1):
            path = work / f"{index:02d}-{request.corpus_id}.json"
            write_exclusive_receipt_bytes(request.canonical_file_bytes(), path)
        _ensure_embedding_copy(config, inputs)
        _prepare_factory_roots(config)
        _require_prepared_shard_root(config)
        _exclusive_publish_directory(
            work,
            request_root,
            label="factory shard request set",
        )
    except (ArtifactIntegrityError, OSError) as exc:
        raise ProductionArtifactFactoryError(
            f"cannot prepare factory shard request set: {exc}"
        ) from exc
    finally:
        if os.path.lexists(work):
            _validate_and_remove_temporary_tree(
                work,
                label="factory shard-request staging cleanup",
            )
    return requests


def build_production_artifact_factory_shard(
    config: ProductionArtifactFactoryConfig,
    request: ProductionArtifactFactoryShardRequest,
    *,
    hmac_secret: bytes,
    resume: bool,
) -> ProductionArtifactFactoryShardReceipt:
    """Build one locked corpus lane without writing a suite-level artifact."""

    if not isinstance(config, ProductionArtifactFactoryConfig):
        raise ProductionArtifactFactoryError("factory config must be typed")
    _verify_shard_request_binding(config, request)
    _verify_shard_hmac_custody(config, request, hmac_secret)
    inputs = _admit_factory_inputs(config)
    _require_prepared_shard_root(config)
    _verify_embedding_copy(config, inputs)
    backend = _backend(config)
    corpus_id = request.corpus_id
    with _factory_corpus_locks(config, (corpus_id,)):
        _require_no_terminal_factory_receipts(config)
        was_existing = _corpus_had_outputs(config, corpus_id)
        if was_existing and not resume:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} shard outputs already exist; use resume-shard"
            )
        evidence = _build_one_corpus(
            config,
            inputs,
            corpus_id=corpus_id,
            backend=backend,
            hmac_secret=hmac_secret,
            was_existing=was_existing,
        )
        verified = _verify_one_corpus(
            config,
            inputs,
            corpus_id=corpus_id,
            backend=backend,
        )
        if verified != evidence:
            raise ProductionArtifactFactoryError(
                f"{corpus_id} shard verification differs from its build evidence"
            )
        return _derive_factory_shard_receipt(config, request, verified)


def _publish_production_artifact_factory_terminal(
    config: ProductionArtifactFactoryConfig,
    inputs: _FactoryInputs,
    evidence: Sequence[ProductionCorpusFactoryEvidence],
    *,
    backend: HnswlibBackend,
    embedding_destination_tree_sha256: str,
) -> ProductionArtifactFactorySuiteReceipt:
    """One terminal assembler shared by sequential and sharded construction."""

    evidence = tuple(evidence)
    if tuple(item.corpus_id for item in evidence) != FIXED_CORPORA:
        raise ProductionArtifactFactoryError("terminal factory evidence differs from FIXED_CORPORA")
    reproducibility = _assemble_reproducibility_suite(
        config,
        inputs,
        backend=backend,
        publish=True,
    )
    try:
        if os.path.lexists(config.pipeline_receipt_path):
            pipeline = verify_artifact_pipeline(
                config.artifact_root,
                inputs.embedding_config.online_staging_root,
                config.pipeline_receipt_path,
            )
        else:
            pipeline = build_artifact_pipeline(
                config.artifact_root,
                inputs.embedding_config.online_staging_root,
                config.pipeline_receipt_path,
                expected_online_inventory_sha256=(inputs.embedding_config.online_inventory_sha256),
            )
    except ArtifactPipelineError as exc:
        raise ProductionArtifactFactoryError(f"suite artifact pipeline failed: {exc}") from exc
    expected = _suite_from_evidence(
        config,
        inputs,
        evidence,
        reproducibility,
        pipeline,
        embedding_destination_tree_sha256=embedding_destination_tree_sha256,
    )
    if os.path.lexists(config.suite_receipt_path):
        if _load_factory_suite(config.suite_receipt_path) != expected:
            raise ProductionArtifactFactoryError("factory suite receipt differs")
    else:
        try:
            write_exclusive_receipt_bytes(
                expected.canonical_file_bytes(),
                config.suite_receipt_path,
            )
        except ArtifactIntegrityError as exc:
            raise ProductionArtifactFactoryError(
                f"cannot write factory suite receipt: {exc}"
            ) from exc
    _validate_factory_root_membership(config, fresh=False, final=True)
    return expected


def _order_and_validate_factory_shard_receipts(
    config: ProductionArtifactFactoryConfig,
    receipts: Sequence[ProductionArtifactFactoryShardReceipt],
) -> tuple[ProductionArtifactFactoryShardReceipt, ...]:
    receipts = tuple(receipts)
    if not all(isinstance(receipt, ProductionArtifactFactoryShardReceipt) for receipt in receipts):
        raise ProductionArtifactFactoryError("factory shard receipts must be typed")
    corpus_ids = tuple(receipt.corpus_id for receipt in receipts)
    duplicates = sorted(
        corpus_id for corpus_id in set(corpus_ids) if corpus_ids.count(corpus_id) > 1
    )
    missing = [corpus_id for corpus_id in FIXED_CORPORA if corpus_id not in corpus_ids]
    extra = sorted(set(corpus_ids) - set(FIXED_CORPORA))
    if len(receipts) != len(FIXED_CORPORA) or duplicates or missing or extra:
        raise ProductionArtifactFactoryError(
            "factory shard receipt set differs; "
            f"missing={missing}, extra={extra}, duplicate={duplicates}"
        )
    by_corpus = {receipt.corpus_id: receipt for receipt in receipts}
    ordered = tuple(by_corpus[corpus_id] for corpus_id in FIXED_CORPORA)
    for receipt in ordered:
        request = _expected_shard_request(config, receipt.corpus_id)
        mismatches = []
        expected_values = {
            "request_sha256": request.request_sha256,
            "factory_config_sha256": config.file_sha256,
            "runner_image": config.runner_image,
            "runner_platform": config.runner_platform,
            "hmac_key_id": config.hmac_key_id,
            "hmac_secret_sha256": config.hmac_secret_sha256,
        }
        for field, expected in expected_values.items():
            if getattr(receipt, field) != expected:
                mismatches.append(field)
        if mismatches:
            raise ProductionArtifactFactoryError(
                f"{receipt.corpus_id} shard receipt differs from the pinned config at: "
                + ", ".join(mismatches)
            )
    return ordered


def _verify_factory_shard_receipt_current(
    config: ProductionArtifactFactoryConfig,
    receipt: ProductionArtifactFactoryShardReceipt,
    evidence: ProductionCorpusFactoryEvidence,
) -> None:
    expected = _derive_factory_shard_receipt(
        config,
        _expected_shard_request(config, receipt.corpus_id),
        evidence,
    )
    if receipt != expected:
        raise ProductionArtifactFactoryError(
            f"{receipt.corpus_id} shard receipt differs from the current corpus tree"
        )


def aggregate_production_artifact_factory_shards(
    config: ProductionArtifactFactoryConfig,
    receipts: Sequence[ProductionArtifactFactoryShardReceipt],
) -> ProductionArtifactFactorySuiteReceipt:
    """Verify five completed lanes and publish the reference terminal receipts."""

    if not isinstance(config, ProductionArtifactFactoryConfig):
        raise ProductionArtifactFactoryError("factory config must be typed")
    ordered = _order_and_validate_factory_shard_receipts(config, receipts)
    inputs = _admit_factory_inputs(config)
    _require_prepared_shard_root(config, allow_terminal=True)
    embedding_destination_tree_sha256 = _verify_embedding_copy(config, inputs)
    backend = _backend(config)
    with _factory_corpus_locks(config, FIXED_CORPORA):
        evidence: list[ProductionCorpusFactoryEvidence] = []
        for receipt in ordered:
            verified = _verify_one_corpus(
                config,
                inputs,
                corpus_id=receipt.corpus_id,
                backend=backend,
            )
            _verify_factory_shard_receipt_current(config, receipt, verified)
            evidence.append(verified)
        return _publish_production_artifact_factory_terminal(
            config,
            inputs,
            evidence,
            backend=backend,
            embedding_destination_tree_sha256=embedding_destination_tree_sha256,
        )


def build_production_artifact_factory(
    config: ProductionArtifactFactoryConfig,
    *,
    hmac_secret: bytes | None,
    resume: bool,
) -> ProductionArtifactFactorySuiteReceipt:
    """Build or resume the fixed factory and publish one terminal receipt."""

    if not isinstance(config, ProductionArtifactFactoryConfig):
        raise ProductionArtifactFactoryError("config must be typed")
    if hmac_secret is not None and (
        not isinstance(hmac_secret, bytes) or len(hmac_secret) < 32 or len(hmac_secret) > 4096
    ):
        raise ProductionArtifactFactoryError("HMAC secret FD must contain 32 to 4096 bytes")
    if (
        hmac_secret is not None
        and hashlib.sha256(hmac_secret).hexdigest() != config.hmac_secret_sha256
    ):
        raise ProductionArtifactFactoryError("HMAC secret differs from its frozen commitment")
    if hmac_secret is None:
        if not resume:
            raise ProductionArtifactFactoryError("a new factory build requires the HMAC secret FD")
        missing_queries = [
            corpus_id
            for corpus_id in FIXED_CORPORA
            if not os.path.lexists(
                config.artifact_root / "trial-runtime" / corpus_id / RUNTIME_QUERY_DIRECTORY
            )
        ]
        if missing_queries:
            raise ProductionArtifactFactoryError(
                "resume without the HMAC secret requires every query package; "
                f"missing={missing_queries}"
            )
    inputs = _admit_factory_inputs(config)
    _validate_factory_root_membership(config, fresh=not resume)
    existing = {corpus: _corpus_had_outputs(config, corpus) for corpus in FIXED_CORPORA}
    terminal_exists = any(
        os.path.lexists(path)
        for path in (
            config.pipeline_receipt_path,
            config.reproducibility_receipt_path,
            config.suite_receipt_path,
        )
    )
    if not resume and (terminal_exists or any(existing.values())):
        raise ProductionArtifactFactoryError(
            "factory outputs already exist; use the resume command after verifying custody"
        )
    backend = _backend(config)
    embedding_destination_tree_sha256 = _ensure_embedding_copy(config, inputs)
    _prepare_factory_roots(config)
    evidence: list[ProductionCorpusFactoryEvidence] = []
    for corpus_id in FIXED_CORPORA:
        with _factory_corpus_locks(config, (corpus_id,)):
            evidence.append(
                _build_one_corpus(
                    config,
                    inputs,
                    corpus_id=corpus_id,
                    backend=backend,
                    hmac_secret=hmac_secret,
                    was_existing=existing[corpus_id],
                )
            )
    return _publish_production_artifact_factory_terminal(
        config,
        inputs,
        evidence,
        backend=backend,
        embedding_destination_tree_sha256=embedding_destination_tree_sha256,
    )


def verify_production_artifact_factory(
    config: ProductionArtifactFactoryConfig,
) -> ProductionArtifactFactorySuiteReceipt:
    """Reproduce the complete terminal receipt without writing an artifact."""

    inputs = _admit_factory_inputs(config)
    _validate_factory_root_membership(config, fresh=False, final=True)
    embedding_destination_tree_sha256 = _verify_embedding_copy(config, inputs)
    backend = _backend(config)
    evidence = [
        _verify_one_corpus(
            config,
            inputs,
            corpus_id=corpus_id,
            backend=backend,
        )
        for corpus_id in FIXED_CORPORA
    ]
    reproducibility = _assemble_reproducibility_suite(
        config,
        inputs,
        backend=backend,
        publish=False,
    )
    try:
        pipeline = verify_artifact_pipeline(
            config.artifact_root,
            inputs.embedding_config.online_staging_root,
            config.pipeline_receipt_path,
        )
    except ArtifactPipelineError as exc:
        raise ProductionArtifactFactoryError(
            f"artifact pipeline verification failed: {exc}"
        ) from exc
    expected = _suite_from_evidence(
        config,
        inputs,
        evidence,
        reproducibility,
        pipeline,
        embedding_destination_tree_sha256=embedding_destination_tree_sha256,
    )
    observed = _load_factory_suite(config.suite_receipt_path)
    if observed != expected:
        raise ProductionArtifactFactoryError("factory suite receipt differs from reproduced state")
    return observed


def production_artifact_factory_status(
    config: ProductionArtifactFactoryConfig,
) -> dict[str, object]:
    """Return a read-only phase inventory without opening the HMAC secret."""

    corpora: list[dict[str, object]] = []
    for corpus_id in FIXED_CORPORA:
        policy_root = config.artifact_root / "policy-workloads" / corpus_id
        index_root = config.artifact_root / "authorized-index-stores" / corpus_id
        runtime_root = config.artifact_root / "trial-runtime" / corpus_id
        corpora.append(
            {
                "corpus_id": corpus_id,
                "evidence": os.path.lexists(config.evidence_root / f"{corpus_id}.json"),
                "index_stages": [
                    stage
                    for stage in config.artifact_stage_order
                    if os.path.lexists(index_root / stage)
                ],
                "online_execution": os.path.lexists(_online_package_root(config, corpus_id)),
                "policy_stages": [
                    stage
                    for stage in config.artifact_stage_order
                    if os.path.lexists(policy_root / stage)
                ],
                "query_package": os.path.lexists(runtime_root / RUNTIME_QUERY_DIRECTORY),
                "runtime_receipt": os.path.lexists(runtime_root / RUNTIME_RECEIPT_FILENAME),
            }
        )
    return {
        "artifact_pipeline_receipt": os.path.lexists(config.pipeline_receipt_path),
        "corpora": corpora,
        "embedding_copy": os.path.lexists(config.artifact_root / "embedding-stores"),
        "embedding_copy_partial": os.path.lexists(
            config.artifact_root / ".embedding-stores.partial"
        ),
        "embedding_source_tree_sha256": config.embedding_source_tree_sha256,
        "factory_config_sha256": config.file_sha256,
        "index_reproducibility_receipt": os.path.lexists(config.reproducibility_receipt_path),
        "schema_version": PRODUCTION_FACTORY_CONFIG_SCHEMA,
        "suite_receipt": os.path.lexists(config.suite_receipt_path),
    }


def _read_hmac_secret(fd: int | None) -> bytes | None:
    if fd is None:
        return None
    if isinstance(fd, bool) or not isinstance(fd, int) or fd != 0:
        raise ProductionArtifactFactoryError("hmac-secret-fd must equal stdin descriptor 0")
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = os.read(fd, min(4097 - total, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 4096:
                raise ProductionArtifactFactoryError("HMAC secret FD exceeds 4096 bytes")
    except OSError as exc:
        raise ProductionArtifactFactoryError("cannot read HMAC secret from stdin") from exc
    secret = b"".join(chunks)
    if len(secret) < 32:
        raise ProductionArtifactFactoryError("HMAC secret FD contains fewer than 32 bytes")
    return secret


def write_production_artifact_factory_config(
    *,
    artifact_root: str | Path,
    embedding_build_config_path: str | Path,
    embedding_build_config_sha256: str,
    development_operator_root: str | Path,
    development_operator_receipt_sha256: str,
    partition_audit_path: str | Path,
    partition_audit_sha256: str,
    runner_image: str,
    hmac_secret_fd: int,
    output: str | Path,
) -> ProductionArtifactFactoryConfig:
    """Derive every non-operator field, verify the candidate, and write it once."""

    secret = _read_hmac_secret(hmac_secret_fd)
    if secret is None:
        raise ProductionArtifactFactoryError("write-config requires HMAC secret bytes on stdin")
    hmac_secret_sha256 = hashlib.sha256(secret).hexdigest()
    del secret
    artifact = _canonical_absolute_path(str(artifact_root), label="artifact_root")
    embedding_config_path = _canonical_absolute_path(
        str(embedding_build_config_path),
        label="embedding_build_config_path",
    )
    development_operator = _canonical_absolute_path(
        str(development_operator_root),
        label="development_operator_root",
    )
    audit_path = _canonical_absolute_path(str(partition_audit_path), label="partition_audit_path")
    destination = _canonical_absolute_path(str(output), label="output")
    for name, value in (
        ("embedding_build_config_sha256", embedding_build_config_sha256),
        ("development_operator_receipt_sha256", development_operator_receipt_sha256),
        ("partition_audit_sha256", partition_audit_sha256),
    ):
        _require_sha256(name, value)
    if os.path.lexists(destination):
        raise ProductionArtifactFactoryError("factory config output already exists")
    try:
        embedding_config = load_production_embedding_config(
            embedding_config_path,
            expected_sha256=embedding_build_config_sha256,
        )
        _require_real_directory(
            embedding_config.output_root,
            label="embedding_source_root",
        )
        _require_read_only_filesystem(
            embedding_config.output_root,
            label="embedding_source_root",
        )
        embedding_source_tree = digest_directory_tree(embedding_config.output_root)
        embedding_suite = admit_frozen_production_embedding_suite(embedding_config)
        projection = verify_online_staging_projection(
            embedding_config.online_staging_root,
            expected_inventory_sha256=embedding_config.online_inventory_sha256,
        )
        audit = load_scalable_partition_audit(
            audit_path,
            expected_artifact_sha256=partition_audit_sha256,
            expected_inventory_sha256=embedding_config.online_inventory_sha256,
        )
    except (
        ArtifactIntegrityError,
        ProductionEmbeddingBuildError,
        ScalablePartitionAuditError,
        StudyDataError,
    ) as exc:
        raise ProductionArtifactFactoryError(f"write-config input admission failed: {exc}") from exc
    if (
        projection.projected_artifact_set_sha256 != embedding_config.projected_artifact_set_sha256
        or embedding_suite.online_inventory_sha256 != projection.inventory_sha256
        or embedding_suite.projected_artifact_set_sha256 != projection.projected_artifact_set_sha256
        or audit.staged_inventory_sha256 != projection.inventory_sha256
    ):
        raise ProductionArtifactFactoryError(
            "embedding suite, online projection, and partition audit do not share one cohort"
        )
    index_config, _ = production_authorized_index_components()
    development_receipt = _verify_development_operator_binding(
        root=development_operator,
        receipt_sha256=development_operator_receipt_sha256,
        embedding_config_path=embedding_config_path,
        embedding_config=embedding_config,
        embedding_suite=embedding_suite,
        partition_audit_path=audit_path,
        partition_audit=audit,
        embedding_suite_receipt_sha256=embedding_suite.receipt_sha256,
        partition_audit_sha256=partition_audit_sha256,
        index_config_sha256=index_config.config_sha256,
    )
    development_root, power_path = _development_operator_paths(development_operator)
    try:
        development_materialization_receipt_sha256 = (
            development_receipt.development_materialization_receipt_sha256
        )
        design_seed_sha256 = development_receipt.design_seed_sha256
        joint_power_report_sha256 = development_receipt.joint_power_report_sha256
        joint_power_report_tree_sha256 = development_receipt.joint_power_report_tree_sha256
        operator_selected_family_count = development_receipt.selected_families_per_corpus
    except AttributeError as exc:
        raise ProductionArtifactFactoryError(
            "post-embedding development receipt lacks a derived factory field"
        ) from exc
    design = _derive_factory_design_bindings(design_seed_sha256)
    report_bytes = _read_control(power_path, label="joint power report")
    if _sha256(report_bytes) != joint_power_report_sha256:
        raise ProductionArtifactFactoryError("joint power report differs from its caller pin")
    try:
        report = load_joint_power_report(report_bytes)
    except JointPowerDesignError as exc:
        raise ProductionArtifactFactoryError(f"joint power report is invalid: {exc}") from exc
    selected_family_count = report.selected_families_per_corpus
    if not report.freeze_ready or selected_family_count is None:
        raise ProductionArtifactFactoryError("joint power report is not freeze-ready")
    if selected_family_count != operator_selected_family_count:
        raise ProductionArtifactFactoryError(
            "joint power report selection differs from the development operator receipt"
        )
    corpora = _derive_available_family_counts(embedding_config.online_staging_root, audit)
    if any(row.available_family_count < selected_family_count for row in corpora):
        raise ProductionArtifactFactoryError(
            "joint power selection exceeds an audited sealed-family denominator"
        )
    config = ProductionArtifactFactoryConfig(
        artifact_root=artifact,
        artifact_stage_order=ARTIFACT_STAGE_ORDER,
        embedding_build_config_path=embedding_config_path,
        embedding_build_config_sha256=embedding_build_config_sha256,
        embedding_source_root=embedding_config.output_root,
        embedding_source_tree_sha256=embedding_source_tree.sha256,
        embedding_suite_receipt_sha256=embedding_suite.receipt_sha256,
        development_materialization_root=development_root,
        development_materialization_receipt_sha256=(development_materialization_receipt_sha256),
        development_operator_root=development_operator,
        development_operator_receipt_sha256=development_operator_receipt_sha256,
        development_operator_joint_power_report_tree_sha256=(joint_power_report_tree_sha256),
        design_seed_sha256=design_seed_sha256,
        partition_audit_path=audit_path,
        partition_audit_sha256=partition_audit_sha256,
        joint_power_report_path=power_path,
        joint_power_report_sha256=joint_power_report_sha256,
        runner_image=runner_image,
        runner_platform=FACTORY_RUNNER_PLATFORM,
        policy_seed_sha256=design.policy_seed_sha256,
        baseline_policy_seed_sha256=design.baseline_policy_seed_sha256,
        policy_bundle_revision=design.policy_bundle_revision,
        baseline_policy_bundle_revision=design.baseline_policy_bundle_revision,
        selection_seed_sha256=design.selection_seed_sha256,
        permutation_seed=design.permutation_seed,
        hmac_key_id=f"sealed-online-ephemeral-sha256-{hmac_secret_sha256}",
        hmac_secret_sha256=hmac_secret_sha256,
        index_config=index_config,
        index_replicate_count=INDEX_REPLICATE_COUNT,
        index_replicate_directories=INDEX_REPLICATE_DIRECTORIES,
        selected_family_count=selected_family_count,
        selected_index_replicate=SELECTED_INDEX_REPLICATE,
        corpora=corpora,
    )
    forbidden_destination_roots = (
        config.artifact_root,
        config.embedding_source_root,
        config.development_materialization_root,
        config.development_operator_root,
        embedding_config.online_staging_root,
        embedding_config.current_model_root,
        embedding_config.stale_model_root,
    )
    if any(_paths_overlap(destination, root) for root in forbidden_destination_roots) or any(
        destination == source
        for source in (
            config.embedding_build_config_path,
            config.partition_audit_path,
            config.joint_power_report_path,
        )
    ):
        raise ProductionArtifactFactoryError(
            "factory config output cannot overlap an input or artifact tree"
        )
    _admit_factory_inputs(config)
    _validate_factory_root_membership(config, fresh=True)
    if ProductionArtifactFactoryConfig.from_dict(config.to_dict()) != config:
        raise ProductionArtifactFactoryError("derived factory config failed canonical round-trip")
    try:
        write_exclusive_receipt_bytes(config.canonical_file_bytes(), destination)
    except ArtifactIntegrityError as exc:
        raise ProductionArtifactFactoryError(f"cannot write factory config: {exc}") from exc
    observed = load_production_artifact_factory_config(
        destination,
        expected_sha256=config.file_sha256,
    )
    if observed != config:
        raise ProductionArtifactFactoryError("written factory config differs from its derivation")
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-production-artifacts",
        description="build or verify the closed five-corpus post-embedding artifact chain",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    writer = subparsers.add_parser(
        "write-config",
        help="derive and write one closed production factory config",
    )
    writer.add_argument("--artifact-root", required=True, type=Path)
    writer.add_argument("--embedding-config", required=True, type=Path)
    writer.add_argument("--embedding-config-sha256", required=True)
    writer.add_argument("--development-operator-root", required=True, type=Path)
    writer.add_argument("--development-operator-receipt-sha256", required=True)
    writer.add_argument("--partition-audit", required=True, type=Path)
    writer.add_argument("--partition-audit-sha256", required=True)
    writer.add_argument("--runner-image", required=True)
    writer.add_argument("--hmac-secret-fd", type=int, choices=(0,), required=True)
    writer.add_argument("--output", required=True, type=Path)
    for command, help_text in (
        ("build", "start a new fixed factory build"),
        ("resume", "resume only at verified whole-artifact boundaries"),
        ("prepare-shards", "prepare the exact five independent corpus lanes"),
        ("build-shard", "build one derived corpus lane"),
        ("resume-shard", "resume one derived corpus lane"),
        ("aggregate-shards", "verify and aggregate exactly five corpus lanes"),
        ("verify", "reproduce the terminal suite receipt without writes"),
        ("status", "report the read-only phase inventory"),
        ("reproducibility", "verify all three-build HNSW determinism receipts"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--config-sha256", required=True)
        if command in {"build", "resume"}:
            child.add_argument(
                "--hmac-secret-fd",
                type=int,
                choices=(0,),
                required=command == "build",
            )
        elif command == "prepare-shards":
            child.add_argument("--request-directory", required=True, type=Path)
        elif command in {"build-shard", "resume-shard"}:
            child.add_argument("--request", required=True, type=Path)
            child.add_argument("--request-sha256", required=True)
            child.add_argument("--hmac-secret-fd", type=int, choices=(0,), required=True)
            child.add_argument("--receipt-output", required=True, type=Path)
        elif command == "aggregate-shards":
            child.add_argument(
                "--shard-receipt",
                action="append",
                required=True,
                type=Path,
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "write-config":
            config = write_production_artifact_factory_config(
                artifact_root=args.artifact_root,
                embedding_build_config_path=args.embedding_config,
                embedding_build_config_sha256=args.embedding_config_sha256,
                development_operator_root=args.development_operator_root,
                development_operator_receipt_sha256=(args.development_operator_receipt_sha256),
                partition_audit_path=args.partition_audit,
                partition_audit_sha256=args.partition_audit_sha256,
                runner_image=args.runner_image,
                hmac_secret_fd=args.hmac_secret_fd,
                output=args.output,
            )
            payload = {
                "config_sha256": config.file_sha256,
                "output": str(args.output.resolve()),
                "status": "written",
            }
        else:
            config = load_production_artifact_factory_config(
                args.config,
                expected_sha256=args.config_sha256,
            )
        if args.command == "status":
            payload = production_artifact_factory_status(config)
        elif args.command == "prepare-shards":
            requests = prepare_production_artifact_factory_shards(
                config,
                request_directory=args.request_directory,
            )
            payload = {
                "request_count": len(requests),
                "request_directory": str(args.request_directory.resolve()),
                "request_sha256": [request.request_sha256 for request in requests],
                "status": "prepared",
            }
        elif args.command in {"build-shard", "resume-shard"}:
            request = load_production_artifact_factory_shard_request(
                args.request,
                expected_sha256=args.request_sha256,
            )
            secret = _read_hmac_secret(args.hmac_secret_fd)
            if secret is None:  # pragma: no cover - descriptor is required by argparse
                raise ProductionArtifactFactoryError("factory shard HMAC secret is missing")
            receipt = build_production_artifact_factory_shard(
                config,
                request,
                hmac_secret=secret,
                resume=args.command == "resume-shard",
            )
            receipt_output = _canonical_absolute_path(
                str(args.receipt_output), label="factory shard receipt output"
            )
            try:
                embedding_config = load_production_embedding_config(
                    config.embedding_build_config_path,
                    expected_sha256=config.embedding_build_config_sha256,
                )
            except ProductionEmbeddingBuildError as exc:
                raise ProductionArtifactFactoryError(
                    f"cannot re-admit factory control destination: {exc}"
                ) from exc
            _require_factory_control_destination(
                config,
                receipt_output,
                embedding_config=embedding_config,
                label="factory shard receipt output",
            )
            try:
                write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), receipt_output)
            except ArtifactIntegrityError as exc:
                raise ProductionArtifactFactoryError(
                    f"cannot write factory shard receipt: {exc}"
                ) from exc
            payload = {
                "corpus_id": receipt.corpus_id,
                "receipt_output": str(receipt_output),
                "receipt_sha256": receipt.receipt_sha256,
                "status": "complete",
            }
        elif args.command == "aggregate-shards":
            shard_receipts = tuple(
                load_production_artifact_factory_shard_receipt(path) for path in args.shard_receipt
            )
            receipt = aggregate_production_artifact_factory_shards(
                config,
                shard_receipts,
            )
            payload = {
                "corpus_count": len(receipt.corpora),
                "receipt_sha256": receipt.receipt_sha256,
                "status": "complete",
            }
        elif args.command == "verify":
            receipt = verify_production_artifact_factory(config)
            payload = {
                "factory_config_sha256": config.file_sha256,
                "receipt_sha256": receipt.receipt_sha256,
                "status": "verified",
            }
        elif args.command == "reproducibility":
            inputs = _admit_factory_inputs(config)
            receipt = _assemble_reproducibility_suite(
                config,
                inputs,
                backend=_backend(config),
                publish=False,
            )
            payload = {
                "receipt_sha256": receipt.receipt_sha256,
                "replicate_count": receipt.replicate_count,
                "stage_count": len(receipt.stages),
                "status": "verified",
            }
        elif args.command in {"build", "resume"}:
            secret = _read_hmac_secret(args.hmac_secret_fd)
            receipt = build_production_artifact_factory(
                config,
                hmac_secret=secret,
                resume=args.command == "resume",
            )
            payload = {
                "corpus_count": len(receipt.corpora),
                "receipt_sha256": receipt.receipt_sha256,
                "status": "complete",
            }
    except ProductionArtifactFactoryError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
