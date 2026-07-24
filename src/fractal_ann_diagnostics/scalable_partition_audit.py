"""Inventory-derived query-partition audit for staged confirmatory data.

The audit reads only the staged assignment, query, and qrel artifacts named by
one verified inventory.  It recomputes query coverage, assignment components,
normalized text joins, positive-document joins, and the registered near-
duplicate graph before any freeze package is admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .partition_audit import (
    FROZEN_QUERY_PARTITION_CONFIG,
    FROZEN_QUERY_PARTITION_CONFIG_SHA256,
)
from .scalable_custody import (
    ScalableCustodyError,
    SourceArtifactPin,
    _absolute_path,
    _canonical_bytes,
    _closed_mapping,
    _decode_json,
    _fsync_directory,
    _iter_canonical_jsonl,
    _open_root,
    _read_secure_file,
    _write_exclusive,
)
from .study_data import (
    ASSIGNMENT_ALGORITHM,
    ASSIGNMENT_SCHEMA,
    INVENTORY_SCHEMA,
    STAGES,
    StudyDataError,
    verify_staged_data,
)

SCALABLE_PARTITION_AUDIT_SCHEMA = "fractal-scalable-query-partition-audit-v1"
SCALABLE_PARTITION_ALGORITHM_SCHEMA = "fractal-scalable-query-partition-algorithm-v1"
QUERY_COVERAGE_ALGORITHM = "inventory-query-assignment-exact-coverage-v1"
ASSIGNMENT_COMPONENT_ALGORITHM = "sha256-canonical-query-id-array-v1"
NORMALIZED_TEXT_DIGEST_ALGORITHM = "sha256-utf8-normalized-text-v1"
POSITIVE_DOCUMENT_ALGORITHM = "corpus-plus-external-document-id-v1"
POSITIVE_DOCUMENT_CONTENT_ALGORITHM = "suite-global-canonical-inline-document-content-v2"
CANONICAL_DOCUMENT_CONTENT_ALGORITHM = "length-prefixed-title-text-sha256-v1"
AUDIT_COMPONENT_ALGORITHM = "sha256-canonical-query-identity-array-v1"
STRUCTURAL_EXCLUSION_SCHEMA = "fractal-study-query-partition-exclusion-v1"
STRUCTURAL_EXCLUSION_RULE_ID = "source-split-component-isolation-v1"
STRUCTURAL_EXCLUSION_REASON = "cross-source-split-component"
STRUCTURAL_EXCLUSION_POLICY = "exclude-entire-component-v1"
STRUCTURAL_EXCLUSION_MEMBERSHIP_ALGORITHM = "sha256-canonical-structural-exclusion-components-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_STAGE_SET = frozenset(STAGES)
_CORPUS_SHARD_PATH = re.compile(r"^datasets/([^/]+)/corpus/part-[0-9]{5}\.jsonl$")
_INVENTORY_FIELDS = frozenset(
    {
        "artifacts",
        "assignment_algorithm",
        "assignment_seed_sha256",
        "bright_document_identity",
        "bright_domains",
        "config_sha256",
        "counts",
        "hotpotqa_fullwiki_scope",
        "schema_version",
        "sources",
        "withhold_sealed_labels_from_online_process",
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
_QUERY_FIELDS = frozenset({"id", "text"})
_QREL_FIELDS = frozenset({"document_id", "query_id", "relevance"})
_CORPUS_FIELDS = frozenset({"id", "text", "title"})
_STRUCTURAL_EXCLUSION_FIELDS = frozenset(
    {
        "dataset",
        "normalized_query_text_sha256",
        "partition_component_sha256",
        "positive_relevance_identity_sha256s",
        "query_id",
        "query_text_sha256",
        "reason",
        "rule_id",
        "schema_version",
        "source_split",
    }
)
_ASSIGNMENT_ALGORITHM_FIELDS = frozenset(
    {
        "component_edges",
        "cross_source_split_policy",
        "fit_calibration_component_ratio",
        "name",
        "three_way_component_ratio",
    }
)
_QUERY_COUNT_ROW_FIELDS = frozenset({"dataset", "stage", "query_count"})
_RECEIPT_FIELDS = frozenset(
    {
        "algorithm_sha256",
        "assignment_artifact_sha256",
        "assignment_component_count",
        "assignment_count",
        "assignment_seed_sha256",
        "audit_component_count",
        "component_membership_sha256",
        "corpus_artifact_count",
        "cross_stage_component_count",
        "exact_text_edge_count",
        "near_duplicate_config_sha256",
        "near_duplicate_edge_count",
        "normalized_text_membership_sha256",
        "positive_document_membership_sha256",
        "positive_document_content_membership_sha256",
        "positive_qrel_count",
        "qrel_artifact_count",
        "qrel_count",
        "query_artifact_count",
        "query_count",
        "query_counts",
        "query_coverage_sha256",
        "schema_version",
        "shared_positive_document_edge_count",
        "shared_positive_document_content_edge_count",
        "source_artifact_set_sha256",
        "source_artifacts",
        "staged_inventory_sha256",
        "staging_config_sha256",
        "structural_exclusion_artifact_sha256",
        "structural_exclusion_component_count",
        "structural_exclusion_counts",
        "structural_exclusion_membership_sha256",
        "structural_exclusion_query_count",
    }
)
_STRUCTURAL_EXCLUSION_COUNT_ROW_FIELDS = frozenset(
    {"component_count", "dataset", "query_count", "reason", "rule_id"}
)


class ScalablePartitionAuditError(RuntimeError):
    """Raised when staged query independence cannot be proved exactly."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ScalablePartitionAuditError(f"{name} must be 64 lowercase hex characters")
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ScalablePartitionAuditError(f"{name} must be a non-empty string")
    return value


def _require_body_text(name: str, value: object) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise ScalablePartitionAuditError(f"{name} must be a string without NUL")
    return value


def _require_integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScalablePartitionAuditError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _algorithm_descriptor() -> dict[str, object]:
    return {
        "assignment_component_algorithm": ASSIGNMENT_COMPONENT_ALGORITHM,
        "audit_component_algorithm": AUDIT_COMPONENT_ALGORITHM,
        "canonical_document_content_algorithm": (CANONICAL_DOCUMENT_CONTENT_ALGORITHM),
        "component_edges": [
            "normalized-query-text-equality",
            "registered-near-duplicate-token-rule",
            "shared-positive-document-content",
            "shared-positive-relevance-document",
        ],
        "near_duplicate_config": FROZEN_QUERY_PARTITION_CONFIG.to_dict(),
        "normalized_text_digest_algorithm": NORMALIZED_TEXT_DIGEST_ALGORITHM,
        "positive_document_algorithm": POSITIVE_DOCUMENT_ALGORITHM,
        "positive_document_content_algorithm": (POSITIVE_DOCUMENT_CONTENT_ALGORITHM),
        "positive_document_content_source_roles": ["corpus", "corpus-shard"],
        "query_coverage_algorithm": QUERY_COVERAGE_ALGORITHM,
        "schema_version": SCALABLE_PARTITION_ALGORITHM_SCHEMA,
        "source_rule": "verified-inventory-pins-with-custody-qrels-v1",
        "structural_exclusion_membership_algorithm": (STRUCTURAL_EXCLUSION_MEMBERSHIP_ALGORITHM),
        "structural_exclusion_reason": STRUCTURAL_EXCLUSION_REASON,
        "structural_exclusion_rule_id": STRUCTURAL_EXCLUSION_RULE_ID,
        "structural_exclusion_schema": STRUCTURAL_EXCLUSION_SCHEMA,
        "structural_exclusion_source": {
            "path": "partition-exclusions.jsonl",
            "role": "query-partition-structural-exclusions",
            "visibility": "protocol",
        },
        "whole_component_source_split_policy": STRUCTURAL_EXCLUSION_POLICY,
    }


SCALABLE_PARTITION_ALGORITHM_SHA256 = _sha256(_canonical_bytes(_algorithm_descriptor()))


def _source_artifact_set_sha256(
    sources: tuple[SourceArtifactPin, ...],
) -> str:
    return _sha256(_canonical_bytes([source.to_dict() for source in sources]))


@dataclass(frozen=True, order=True)
class QueryCountRow:
    """Exact query count for one staged corpus and study stage."""

    dataset: str
    stage: str
    query_count: int

    def __post_init__(self) -> None:
        _require_text("query count dataset", self.dataset)
        if self.stage not in _STAGE_SET:
            raise ScalablePartitionAuditError("query count stage is not registered")
        _require_integer("query_count", self.query_count, minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "query_count": self.query_count,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, value: object) -> QueryCountRow:
        row = _closed_mapping(
            value,
            fields=_QUERY_COUNT_ROW_FIELDS,
            label="query count row",
        )
        return cls(
            dataset=row["dataset"],
            stage=row["stage"],
            query_count=row["query_count"],
        )


@dataclass(frozen=True, order=True)
class StructuralExclusionCountRow:
    """Registered structural exclusions for one staged corpus."""

    dataset: str
    rule_id: str
    reason: str
    query_count: int
    component_count: int

    def __post_init__(self) -> None:
        _require_text("structural exclusion dataset", self.dataset)
        if self.rule_id != STRUCTURAL_EXCLUSION_RULE_ID:
            raise ScalablePartitionAuditError("structural exclusion rule differs")
        if self.reason != STRUCTURAL_EXCLUSION_REASON:
            raise ScalablePartitionAuditError("structural exclusion reason differs")
        _require_integer("structural exclusion query_count", self.query_count, minimum=1)
        _require_integer(
            "structural exclusion component_count",
            self.component_count,
            minimum=1,
        )
        if self.component_count > self.query_count:
            raise ScalablePartitionAuditError(
                "structural exclusion component count exceeds query count"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "component_count": self.component_count,
            "dataset": self.dataset,
            "query_count": self.query_count,
            "reason": self.reason,
            "rule_id": self.rule_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> StructuralExclusionCountRow:
        row = _closed_mapping(
            value,
            fields=_STRUCTURAL_EXCLUSION_COUNT_ROW_FIELDS,
            label="structural exclusion count row",
        )
        return cls(
            dataset=row["dataset"],
            rule_id=row["rule_id"],
            reason=row["reason"],
            query_count=row["query_count"],
            component_count=row["component_count"],
        )


@dataclass(frozen=True)
class ScalableQueryPartitionAuditReceipt:
    """Label-free commitment to one complete staged partition audit."""

    staged_inventory_sha256: str
    staging_config_sha256: str
    assignment_seed_sha256: str
    algorithm_sha256: str
    near_duplicate_config_sha256: str
    source_artifacts: tuple[SourceArtifactPin, ...]
    source_artifact_set_sha256: str
    assignment_artifact_sha256: str
    query_counts: tuple[QueryCountRow, ...]
    query_artifact_count: int
    qrel_artifact_count: int
    corpus_artifact_count: int
    assignment_count: int
    query_count: int
    qrel_count: int
    positive_qrel_count: int
    assignment_component_count: int
    audit_component_count: int
    exact_text_edge_count: int
    near_duplicate_edge_count: int
    shared_positive_document_edge_count: int
    shared_positive_document_content_edge_count: int
    cross_stage_component_count: int
    structural_exclusion_artifact_sha256: str
    structural_exclusion_query_count: int
    structural_exclusion_component_count: int
    structural_exclusion_counts: tuple[StructuralExclusionCountRow, ...]
    structural_exclusion_membership_sha256: str
    query_coverage_sha256: str
    normalized_text_membership_sha256: str
    component_membership_sha256: str
    positive_document_membership_sha256: str
    positive_document_content_membership_sha256: str
    schema_version: str = SCALABLE_PARTITION_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "staged_inventory_sha256",
            "staging_config_sha256",
            "assignment_seed_sha256",
            "algorithm_sha256",
            "source_artifact_set_sha256",
            "assignment_artifact_sha256",
            "query_coverage_sha256",
            "normalized_text_membership_sha256",
            "component_membership_sha256",
            "positive_document_membership_sha256",
            "structural_exclusion_artifact_sha256",
            "structural_exclusion_membership_sha256",
            "positive_document_content_membership_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.algorithm_sha256 != SCALABLE_PARTITION_ALGORITHM_SHA256:
            raise ScalablePartitionAuditError("partition-audit algorithm digest differs")
        if self.near_duplicate_config_sha256 != FROZEN_QUERY_PARTITION_CONFIG_SHA256:
            raise ScalablePartitionAuditError("near-duplicate config digest differs")
        sources = tuple(sorted(self.source_artifacts, key=lambda item: item.path.encode()))
        if (
            not sources
            or not all(isinstance(item, SourceArtifactPin) for item in sources)
            or len({item.path for item in sources}) != len(sources)
        ):
            raise ScalablePartitionAuditError(
                "source_artifacts must be typed, non-empty, and unique"
            )
        if _source_artifact_set_sha256(sources) != self.source_artifact_set_sha256:
            raise ScalablePartitionAuditError("source artifact-set digest differs")
        assignment_sources = [item for item in sources if item.role == "assignments"]
        if (
            len(assignment_sources) != 1
            or assignment_sources[0].path != "assignments.jsonl"
            or assignment_sources[0].sha256 != self.assignment_artifact_sha256
        ):
            raise ScalablePartitionAuditError("assignment artifact binding differs")
        exclusion_sources = [
            item for item in sources if item.role == "query-partition-structural-exclusions"
        ]
        if (
            len(exclusion_sources) != 1
            or exclusion_sources[0].path != "partition-exclusions.jsonl"
            or exclusion_sources[0].visibility != "protocol"
            or exclusion_sources[0].dataset is not None
            or exclusion_sources[0].stage is not None
            or exclusion_sources[0].sha256 != self.structural_exclusion_artifact_sha256
            or exclusion_sources[0].record_count != self.structural_exclusion_query_count
        ):
            raise ScalablePartitionAuditError("structural exclusion artifact binding differs")
        query_sources = [item for item in sources if item.role == "queries"]
        qrel_sources = [item for item in sources if item.role == "qrels"]
        corpus_sources = [item for item in sources if item.role in {"corpus", "corpus-shard"}]
        if (
            len(query_sources) != self.query_artifact_count
            or len(qrel_sources) != self.qrel_artifact_count
            or len(corpus_sources) != self.corpus_artifact_count
        ):
            raise ScalablePartitionAuditError("source artifact counts differ")
        counts = tuple(self.query_counts)
        if (
            not counts
            or not all(isinstance(item, QueryCountRow) for item in counts)
            or counts != tuple(sorted(counts))
            or len({(item.dataset, item.stage) for item in counts}) != len(counts)
        ):
            raise ScalablePartitionAuditError(
                "query_counts must be typed, unique, and canonically sorted"
            )
        for name in (
            "query_artifact_count",
            "qrel_artifact_count",
            "assignment_count",
            "query_count",
            "qrel_count",
            "positive_qrel_count",
            "assignment_component_count",
            "audit_component_count",
        ):
            _require_integer(name, getattr(self, name), minimum=1)
        for name in (
            "corpus_artifact_count",
            "exact_text_edge_count",
            "near_duplicate_edge_count",
            "shared_positive_document_edge_count",
            "shared_positive_document_content_edge_count",
            "cross_stage_component_count",
            "structural_exclusion_query_count",
            "structural_exclusion_component_count",
        ):
            _require_integer(name, getattr(self, name))
        exclusion_counts = tuple(self.structural_exclusion_counts)
        if (
            (
                not all(isinstance(item, StructuralExclusionCountRow) for item in exclusion_counts)
                or exclusion_counts != tuple(sorted(exclusion_counts))
                or len({item.dataset for item in exclusion_counts}) != len(exclusion_counts)
                or sum(item.query_count for item in exclusion_counts)
                != self.structural_exclusion_query_count
                or sum(item.component_count for item in exclusion_counts)
                != self.structural_exclusion_component_count
            )
            or (
                self.structural_exclusion_query_count == 0
                and (self.structural_exclusion_component_count != 0 or exclusion_counts)
            )
            or (
                self.structural_exclusion_query_count > 0
                and not (
                    1
                    <= self.structural_exclusion_component_count
                    <= self.structural_exclusion_query_count
                    and exclusion_counts
                )
            )
        ):
            raise ScalablePartitionAuditError("structural exclusion counts are inconsistent")
        if (
            self.assignment_count != self.query_count
            or sum(item.query_count for item in counts) != self.query_count
        ):
            raise ScalablePartitionAuditError("assignment/query coverage counts differ")
        if self.qrel_count < self.query_count or self.positive_qrel_count < self.query_count:
            raise ScalablePartitionAuditError("qrel coverage counts are incomplete")
        if not (
            1 <= self.audit_component_count <= self.assignment_component_count <= self.query_count
        ):
            raise ScalablePartitionAuditError("component counts are inconsistent")
        if self.cross_stage_component_count != 0:
            raise ScalablePartitionAuditError(
                "a passing partition-audit receipt must record zero crossings"
            )
        if self.schema_version != SCALABLE_PARTITION_AUDIT_SCHEMA:
            raise ScalablePartitionAuditError("partition-audit receipt schema differs")
        object.__setattr__(self, "source_artifacts", sources)
        object.__setattr__(self, "query_counts", counts)
        object.__setattr__(self, "structural_exclusion_counts", exclusion_counts)

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm_sha256": self.algorithm_sha256,
            "assignment_artifact_sha256": self.assignment_artifact_sha256,
            "assignment_component_count": self.assignment_component_count,
            "assignment_count": self.assignment_count,
            "assignment_seed_sha256": self.assignment_seed_sha256,
            "audit_component_count": self.audit_component_count,
            "component_membership_sha256": self.component_membership_sha256,
            "corpus_artifact_count": self.corpus_artifact_count,
            "cross_stage_component_count": self.cross_stage_component_count,
            "exact_text_edge_count": self.exact_text_edge_count,
            "near_duplicate_config_sha256": self.near_duplicate_config_sha256,
            "near_duplicate_edge_count": self.near_duplicate_edge_count,
            "normalized_text_membership_sha256": (self.normalized_text_membership_sha256),
            "positive_document_membership_sha256": (self.positive_document_membership_sha256),
            "positive_document_content_membership_sha256": (
                self.positive_document_content_membership_sha256
            ),
            "positive_qrel_count": self.positive_qrel_count,
            "qrel_artifact_count": self.qrel_artifact_count,
            "qrel_count": self.qrel_count,
            "query_artifact_count": self.query_artifact_count,
            "query_count": self.query_count,
            "query_counts": [item.to_dict() for item in self.query_counts],
            "query_coverage_sha256": self.query_coverage_sha256,
            "schema_version": self.schema_version,
            "shared_positive_document_edge_count": (self.shared_positive_document_edge_count),
            "shared_positive_document_content_edge_count": (
                self.shared_positive_document_content_edge_count
            ),
            "source_artifact_set_sha256": self.source_artifact_set_sha256,
            "source_artifacts": [item.to_dict() for item in self.source_artifacts],
            "staged_inventory_sha256": self.staged_inventory_sha256,
            "staging_config_sha256": self.staging_config_sha256,
            "structural_exclusion_artifact_sha256": (self.structural_exclusion_artifact_sha256),
            "structural_exclusion_component_count": (self.structural_exclusion_component_count),
            "structural_exclusion_counts": [
                item.to_dict() for item in self.structural_exclusion_counts
            ],
            "structural_exclusion_membership_sha256": (self.structural_exclusion_membership_sha256),
            "structural_exclusion_query_count": (self.structural_exclusion_query_count),
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def artifact_sha256(self) -> str:
        """Digest pinned by the freeze manifest for this canonical receipt file."""

        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ScalableQueryPartitionAuditReceipt:
        row = _closed_mapping(
            value,
            fields=_RECEIPT_FIELDS,
            label="scalable query-partition audit receipt",
        )
        source_values = row["source_artifacts"]
        count_values = row["query_counts"]
        exclusion_count_values = row["structural_exclusion_counts"]
        if (
            not isinstance(source_values, list)
            or not isinstance(count_values, list)
            or not isinstance(exclusion_count_values, list)
        ):
            raise ScalablePartitionAuditError(
                "partition-audit source artifacts and count rows must be arrays"
            )
        return cls(
            staged_inventory_sha256=row["staged_inventory_sha256"],
            staging_config_sha256=row["staging_config_sha256"],
            assignment_seed_sha256=row["assignment_seed_sha256"],
            algorithm_sha256=row["algorithm_sha256"],
            near_duplicate_config_sha256=row["near_duplicate_config_sha256"],
            source_artifacts=tuple(SourceArtifactPin.from_dict(item) for item in source_values),
            source_artifact_set_sha256=row["source_artifact_set_sha256"],
            assignment_artifact_sha256=row["assignment_artifact_sha256"],
            query_counts=tuple(QueryCountRow.from_dict(item) for item in count_values),
            query_artifact_count=row["query_artifact_count"],
            qrel_artifact_count=row["qrel_artifact_count"],
            corpus_artifact_count=row["corpus_artifact_count"],
            assignment_count=row["assignment_count"],
            query_count=row["query_count"],
            qrel_count=row["qrel_count"],
            positive_qrel_count=row["positive_qrel_count"],
            assignment_component_count=row["assignment_component_count"],
            audit_component_count=row["audit_component_count"],
            exact_text_edge_count=row["exact_text_edge_count"],
            near_duplicate_edge_count=row["near_duplicate_edge_count"],
            shared_positive_document_edge_count=row["shared_positive_document_edge_count"],
            shared_positive_document_content_edge_count=row[
                "shared_positive_document_content_edge_count"
            ],
            cross_stage_component_count=row["cross_stage_component_count"],
            structural_exclusion_artifact_sha256=row["structural_exclusion_artifact_sha256"],
            structural_exclusion_query_count=row["structural_exclusion_query_count"],
            structural_exclusion_component_count=row["structural_exclusion_component_count"],
            structural_exclusion_counts=tuple(
                StructuralExclusionCountRow.from_dict(item) for item in exclusion_count_values
            ),
            structural_exclusion_membership_sha256=row["structural_exclusion_membership_sha256"],
            query_coverage_sha256=row["query_coverage_sha256"],
            normalized_text_membership_sha256=row["normalized_text_membership_sha256"],
            component_membership_sha256=row["component_membership_sha256"],
            positive_document_membership_sha256=row["positive_document_membership_sha256"],
            positive_document_content_membership_sha256=row[
                "positive_document_content_membership_sha256"
            ],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class _Assignment:
    dataset: str
    stage: str
    query_id: str
    query_text_sha256: str
    component_sha256: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.dataset, self.stage, self.query_id


@dataclass(frozen=True)
class _QueryNode:
    dataset: str
    stage: str
    query_id: str
    text_sha256: str
    normalized_text: str
    normalized_text_sha256: str
    tokens: tuple[str, ...]
    assignment_component_sha256: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.dataset, self.stage, self.query_id

    @property
    def node_id(self) -> str:
        return "query://" + "/".join(quote(value, safe="") for value in self.key)

    @property
    def identity_sha256(self) -> str:
        return _sha256(_canonical_bytes(list(self.key)))


@dataclass(frozen=True)
class _StructuralExclusion:
    dataset: str
    query_id: str
    source_split: str
    component_sha256: str
    query_text_sha256: str
    normalized_query_text_sha256: str
    positive_relevance_identity_sha256s: tuple[str, ...]

    @property
    def sort_key(self) -> tuple[bytes, bytes, bytes, bytes]:
        return (
            self.dataset.encode("utf-8"),
            self.component_sha256.encode("ascii"),
            self.source_split.encode("utf-8"),
            self.query_id.encode("utf-8"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "normalized_query_text_sha256": self.normalized_query_text_sha256,
            "partition_component_sha256": self.component_sha256,
            "positive_relevance_identity_sha256s": list(self.positive_relevance_identity_sha256s),
            "query_id": self.query_id,
            "query_text_sha256": self.query_text_sha256,
            "reason": STRUCTURAL_EXCLUSION_REASON,
            "rule_id": STRUCTURAL_EXCLUSION_RULE_ID,
            "schema_version": STRUCTURAL_EXCLUSION_SCHEMA,
            "source_split": self.source_split,
        }


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self._parents = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parents[value]
        while parent != self._parents[parent]:
            parent = self._parents[parent]
        while value != parent:
            next_value = self._parents[value]
            self._parents[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self._parents[second] = first


def _normalize_query_text(text: str) -> tuple[str, tuple[str, ...]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))
    if tokens:
        return " ".join(tokens), tokens
    return " ".join(normalized.split()), tokens


def _connect_group(
    node_ids: Iterable[str],
    *,
    disjoint_set: _DisjointSet,
    edges: set[tuple[str, str]],
) -> None:
    ordered = sorted(set(node_ids))
    if len(ordered) < 2:
        return
    anchor = ordered[0]
    for other in ordered[1:]:
        pair = (anchor, other)
        edges.add(pair)
        disjoint_set.union(*pair)


def _near_duplicate_pairs(nodes: tuple[_QueryNode, ...]) -> set[tuple[str, str]]:
    config = FROZEN_QUERY_PARTITION_CONFIG
    eligible = [node for node in nodes if len(node.tokens) >= config.minimum_near_duplicate_tokens]
    exact_tokens: dict[tuple[str, ...], list[str]] = {}
    substitutions: dict[tuple[int, int, tuple[str, ...]], dict[str, list[str]]] = {}
    for node in eligible:
        exact_tokens.setdefault(node.tokens, []).append(node.node_id)
        for position, token in enumerate(node.tokens):
            deleted = node.tokens[:position] + node.tokens[position + 1 :]
            variants = substitutions.setdefault((len(node.tokens), position, deleted), {})
            variants.setdefault(token, []).append(node.node_id)

    pairs: set[tuple[str, str]] = set()
    for _, variants in sorted(substitutions.items(), key=lambda item: repr(item[0])):
        if len(variants) < 2:
            continue
        representatives = sorted(min(node_ids) for node_ids in variants.values())
        anchor = representatives[0]
        for other in representatives[1:]:
            pairs.add(tuple(sorted((anchor, other))))

    numerator = config.minimum_length_ratio_numerator
    denominator = config.minimum_length_ratio_denominator
    for node in eligible:
        shorter_length = len(node.tokens) - 1
        if shorter_length < config.minimum_near_duplicate_tokens:
            continue
        if shorter_length * denominator < len(node.tokens) * numerator:
            continue
        seen_deletions: set[tuple[str, ...]] = set()
        for position in range(len(node.tokens)):
            deleted = node.tokens[:position] + node.tokens[position + 1 :]
            if deleted in seen_deletions:
                continue
            seen_deletions.add(deleted)
            shorter_nodes = exact_tokens.get(deleted)
            if shorter_nodes:
                pairs.add(tuple(sorted((node.node_id, min(shorter_nodes)))))
    return pairs


def _validate_inventory(
    root_descriptor: int,
    *,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], tuple[SourceArtifactPin, ...]]:
    encoded = _read_secure_file(
        root_descriptor,
        "inventory.json",
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="staged inventory",
    )
    if _sha256(encoded) != expected_sha256:
        raise ScalablePartitionAuditError("staged inventory changed after package verification")
    checksum = _read_secure_file(
        root_descriptor,
        "inventory.sha256",
        maximum_bytes=1024,
        label="staged inventory checksum",
    )
    if checksum != f"{expected_sha256}  inventory.json\n".encode("ascii"):
        raise ScalablePartitionAuditError("staged inventory checksum differs")
    value = _decode_json(encoded, label="staged inventory")
    inventory = _closed_mapping(
        value,
        fields=_INVENTORY_FIELDS,
        label="staged inventory",
    )
    if inventory["schema_version"] != INVENTORY_SCHEMA:
        raise ScalablePartitionAuditError("staged inventory schema differs")
    if encoded != _canonical_bytes(value) + b"\n":
        raise ScalablePartitionAuditError("staged inventory is not canonical")
    if inventory["withhold_sealed_labels_from_online_process"] is not True:
        raise ScalablePartitionAuditError("staged inventory must keep sealed labels under custody")
    assignment_algorithm = _closed_mapping(
        inventory["assignment_algorithm"],
        fields=_ASSIGNMENT_ALGORITHM_FIELDS,
        label="assignment algorithm",
    )
    if assignment_algorithm != {
        "component_edges": [
            "normalized-query-text-equality",
            "registered-near-duplicate-token-rule",
            "shared-positive-document-content",
            "shared-positive-relevance-document",
        ],
        "cross_source_split_policy": STRUCTURAL_EXCLUSION_POLICY,
        "fit_calibration_component_ratio": "4:1",
        "name": ASSIGNMENT_ALGORITHM,
        "three_way_component_ratio": "3:1:1",
    }:
        raise ScalablePartitionAuditError("staged assignment algorithm differs")
    artifact_values = inventory["artifacts"]
    if not isinstance(artifact_values, list):
        raise ScalablePartitionAuditError("staged inventory artifacts must be an array")
    sources = tuple(SourceArtifactPin.from_dict(item) for item in artifact_values)
    return inventory, sources


def _select_sources(
    sources: tuple[SourceArtifactPin, ...],
) -> tuple[
    SourceArtifactPin,
    tuple[SourceArtifactPin, ...],
    tuple[SourceArtifactPin, ...],
    tuple[SourceArtifactPin, ...],
    SourceArtifactPin,
]:
    assignments = [
        source
        for source in sources
        if source.path == "assignments.jsonl"
        and source.dataset is None
        and source.stage is None
        and source.role == "assignments"
        and source.visibility == "online"
    ]
    if len(assignments) != 1:
        raise ScalablePartitionAuditError(
            "staged package must contain exactly one online assignment artifact"
        )
    exclusions = [
        source
        for source in sources
        if source.path == "partition-exclusions.jsonl"
        and source.dataset is None
        and source.stage is None
        and source.role == "query-partition-structural-exclusions"
        and source.visibility == "protocol"
    ]
    if len(exclusions) != 1:
        raise ScalablePartitionAuditError(
            "staged package must contain one protocol structural-exclusion artifact"
        )
    query_sources = tuple(
        sorted(
            (
                source
                for source in sources
                if source.role == "queries"
                and source.visibility == "online"
                and source.dataset is not None
                and source.stage in _STAGE_SET
            ),
            key=lambda item: item.path.encode(),
        )
    )
    qrel_sources = tuple(
        sorted(
            (
                source
                for source in sources
                if source.role == "qrels"
                and source.dataset is not None
                and source.stage in _STAGE_SET
                and (
                    (source.stage == "sealed" and source.visibility == "custody")
                    or (source.stage != "sealed" and source.visibility == "online")
                )
            ),
            key=lambda item: item.path.encode(),
        )
    )
    if not query_sources or not qrel_sources:
        raise ScalablePartitionAuditError("staged query/qrel source set is empty")
    query_keys = {(item.dataset, item.stage) for item in query_sources}
    qrel_keys = {(item.dataset, item.stage) for item in qrel_sources}
    if len(query_keys) != len(query_sources) or len(qrel_keys) != len(qrel_sources):
        raise ScalablePartitionAuditError("staged package repeats a query or qrel stage")
    if query_keys != qrel_keys:
        raise ScalablePartitionAuditError(
            "query and qrel artifacts do not cover the same corpus/stage set"
        )
    if {stage for _, stage in query_keys} != _STAGE_SET:
        raise ScalablePartitionAuditError(
            "staged query artifacts must cover fit, calibration, and sealed"
        )
    query_datasets = {dataset for dataset, _ in query_keys}
    corpus_sources = tuple(
        sorted(
            (
                source
                for source in sources
                if source.role == "corpus"
                and source.visibility == "online"
                and source.stage is None
                and source.dataset in query_datasets
            ),
            key=lambda item: item.path.encode(),
        )
    )
    corpus_shard_sources = tuple(
        sorted(
            (
                source
                for source in sources
                if source.role == "corpus-shard"
                and source.visibility == "online"
                and source.stage is None
                and source.dataset in query_datasets
            ),
            key=lambda item: item.path.encode(),
        )
    )
    inline_datasets = {source.dataset for source in corpus_sources}
    sharded_datasets = {source.dataset for source in corpus_shard_sources}
    if (
        len(inline_datasets) != len(corpus_sources)
        or inline_datasets & sharded_datasets
        or inline_datasets | sharded_datasets != query_datasets
    ):
        raise ScalablePartitionAuditError(
            "each query dataset must have one inline corpus or corpus shards"
        )
    for source in query_sources:
        expected = (
            f"datasets/{source.dataset}/sealed/online/queries.jsonl"
            if source.stage == "sealed"
            else f"datasets/{source.dataset}/{source.stage}/queries.jsonl"
        )
        if source.path != expected:
            raise ScalablePartitionAuditError("staged query path differs from the contract")
    for source in qrel_sources:
        expected = (
            f"datasets/{source.dataset}/sealed/custody/qrels.jsonl"
            if source.stage == "sealed"
            else f"datasets/{source.dataset}/{source.stage}/qrels.jsonl"
        )
        if source.path != expected:
            raise ScalablePartitionAuditError("staged qrel path differs from the contract")
    for source in corpus_sources:
        if source.path != f"datasets/{source.dataset}/corpus.jsonl":
            raise ScalablePartitionAuditError(
                "staged corpus path differs from the content-identity contract"
            )
    for source in corpus_shard_sources:
        match = _CORPUS_SHARD_PATH.fullmatch(source.path)
        if match is None or match.group(1) != source.dataset:
            raise ScalablePartitionAuditError(
                "staged corpus-shard path differs from the content-identity contract"
            )
    all_corpus_sources = tuple(
        sorted((*corpus_sources, *corpus_shard_sources), key=lambda item: item.path.encode())
    )
    selected = (
        assignments[0],
        exclusions[0],
        *query_sources,
        *qrel_sources,
        *all_corpus_sources,
    )
    if any(
        source.role
        in {
            "assignments",
            "queries",
            "qrels",
            "query-partition-structural-exclusions",
        }
        and source not in selected
        for source in sources
    ):
        raise ScalablePartitionAuditError("inventory contains an unadmitted query-partition source")
    return (
        assignments[0],
        query_sources,
        qrel_sources,
        all_corpus_sources,
        exclusions[0],
    )


def _load_assignments(
    root_descriptor: int,
    source: SourceArtifactPin,
) -> dict[tuple[str, str, str], _Assignment]:
    assignments: dict[tuple[str, str, str], _Assignment] = {}
    query_ids: set[str] = set()
    for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
        row = _closed_mapping(
            value,
            fields=_ASSIGNMENT_FIELDS,
            label=f"assignment {line_number}",
        )
        if row["schema_version"] != ASSIGNMENT_SCHEMA:
            raise ScalablePartitionAuditError("assignment schema differs")
        dataset = _require_text("assignment dataset", row["dataset"])
        stage = _require_text("assignment stage", row["stage"])
        if stage not in _STAGE_SET:
            raise ScalablePartitionAuditError("assignment stage is not registered")
        query_id = _require_text("assignment query_id", row["query_id"])
        if query_id in query_ids:
            raise ScalablePartitionAuditError("assignment query IDs must be globally unique")
        query_ids.add(query_id)
        _require_sha256("assignment_key_sha256", row["assignment_key_sha256"])
        _require_text("assignment source_split", row["source_split"])
        if row["domain"] is not None:
            _require_text("assignment domain", row["domain"])
        assignment = _Assignment(
            dataset=dataset,
            stage=stage,
            query_id=query_id,
            query_text_sha256=_require_sha256(
                "assignment query_text_sha256",
                row["query_text_sha256"],
            ),
            component_sha256=_require_sha256(
                "assignment partition_component_sha256",
                row["partition_component_sha256"],
            ),
        )
        if assignment.key in assignments:
            raise ScalablePartitionAuditError("assignment rows repeat a query key")
        assignments[assignment.key] = assignment
    if not assignments:
        raise ScalablePartitionAuditError("assignment artifact contains no rows")
    return assignments


def _load_structural_exclusions(
    root_descriptor: int,
    source: SourceArtifactPin,
    *,
    assignments: Mapping[tuple[str, str, str], _Assignment],
) -> tuple[
    tuple[_StructuralExclusion, ...],
    tuple[StructuralExclusionCountRow, ...],
    int,
    str,
]:
    exclusions: list[_StructuralExclusion] = []
    admitted_query_ids = {assignment.query_id for assignment in assignments.values()}
    observed_query_ids: set[str] = set()
    previous_sort_key: tuple[bytes, bytes, bytes, bytes] | None = None
    for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
        row = _closed_mapping(
            value,
            fields=_STRUCTURAL_EXCLUSION_FIELDS,
            label=f"structural exclusion {line_number}",
        )
        if row["schema_version"] != STRUCTURAL_EXCLUSION_SCHEMA:
            raise ScalablePartitionAuditError("structural exclusion schema differs")
        if row["rule_id"] != STRUCTURAL_EXCLUSION_RULE_ID:
            raise ScalablePartitionAuditError("structural exclusion rule differs")
        if row["reason"] != STRUCTURAL_EXCLUSION_REASON:
            raise ScalablePartitionAuditError("structural exclusion reason differs")
        positive_values = row["positive_relevance_identity_sha256s"]
        if not isinstance(positive_values, list) or not positive_values:
            raise ScalablePartitionAuditError(
                "structural exclusion positive identities must be a non-empty array"
            )
        positive_identities = tuple(
            _require_sha256("structural exclusion positive identity", value)
            for value in positive_values
        )
        if positive_identities != tuple(sorted(set(positive_identities))):
            raise ScalablePartitionAuditError(
                "structural exclusion positive identities must be unique and sorted"
            )
        exclusion = _StructuralExclusion(
            dataset=_require_text("structural exclusion dataset", row["dataset"]),
            query_id=_require_text("structural exclusion query_id", row["query_id"]),
            source_split=_require_text("structural exclusion source_split", row["source_split"]),
            component_sha256=_require_sha256(
                "structural exclusion partition_component_sha256",
                row["partition_component_sha256"],
            ),
            query_text_sha256=_require_sha256(
                "structural exclusion query_text_sha256",
                row["query_text_sha256"],
            ),
            normalized_query_text_sha256=_require_sha256(
                "structural exclusion normalized_query_text_sha256",
                row["normalized_query_text_sha256"],
            ),
            positive_relevance_identity_sha256s=positive_identities,
        )
        if previous_sort_key is not None and exclusion.sort_key <= previous_sort_key:
            raise ScalablePartitionAuditError(
                "structural exclusions must be unique and canonically sorted"
            )
        previous_sort_key = exclusion.sort_key
        if exclusion.query_id in admitted_query_ids or exclusion.query_id in observed_query_ids:
            raise ScalablePartitionAuditError(
                "structural exclusion query identities are not disjoint"
            )
        observed_query_ids.add(exclusion.query_id)
        exclusions.append(exclusion)

    groups: dict[tuple[str, str], list[_StructuralExclusion]] = {}
    for exclusion in exclusions:
        groups.setdefault((exclusion.dataset, exclusion.component_sha256), []).append(exclusion)
    for (_, component_sha256), members in groups.items():
        if len({member.source_split for member in members}) < 2:
            raise ScalablePartitionAuditError(
                "a structural exclusion component does not span source splits"
            )
        query_ids = sorted(
            (member.query_id for member in members),
            key=lambda value: value.encode("utf-8"),
        )
        if _sha256(_canonical_bytes(query_ids)) != component_sha256:
            raise ScalablePartitionAuditError(
                "structural exclusion component digest differs from membership"
            )

    counts: list[StructuralExclusionCountRow] = []
    for dataset in sorted(
        {exclusion.dataset for exclusion in exclusions},
        key=lambda value: value.encode("utf-8"),
    ):
        dataset_exclusions = [exclusion for exclusion in exclusions if exclusion.dataset == dataset]
        counts.append(
            StructuralExclusionCountRow(
                dataset=dataset,
                rule_id=STRUCTURAL_EXCLUSION_RULE_ID,
                reason=STRUCTURAL_EXCLUSION_REASON,
                query_count=len(dataset_exclusions),
                component_count=len(
                    {exclusion.component_sha256 for exclusion in dataset_exclusions}
                ),
            )
        )
    membership_rows = [exclusion.to_dict() for exclusion in exclusions]
    return (
        tuple(exclusions),
        tuple(counts),
        len(groups),
        _sha256(_canonical_bytes(membership_rows)),
    )


def _load_queries(
    root_descriptor: int,
    sources: tuple[SourceArtifactPin, ...],
    *,
    assignments: Mapping[tuple[str, str, str], _Assignment],
    inventory_counts: object,
) -> tuple[tuple[_QueryNode, ...], tuple[QueryCountRow, ...]]:
    nodes: list[_QueryNode] = []
    observed_keys: set[tuple[str, str, str]] = set()
    observed_ids: set[str] = set()
    count_rows: list[QueryCountRow] = []
    if not isinstance(inventory_counts, Mapping):
        raise ScalablePartitionAuditError("inventory counts must be an object")
    for source in sources:
        assert source.dataset is not None and source.stage is not None
        previous_id: bytes | None = None
        count = 0
        for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
            row = _closed_mapping(
                value,
                fields=_QUERY_FIELDS,
                label=f"query {source.path}:{line_number}",
            )
            query_id = _require_text("query id", row["id"])
            text = _require_text("query text", row["text"])
            query_id_bytes = query_id.encode("utf-8", errors="strict")
            if previous_id is not None and query_id_bytes <= previous_id:
                raise ScalablePartitionAuditError(
                    "query IDs must be unique and strictly bytewise sorted per file"
                )
            previous_id = query_id_bytes
            key = (source.dataset, source.stage, query_id)
            assignment = assignments.get(key)
            if assignment is None:
                raise ScalablePartitionAuditError(
                    "query artifact contains a row absent from assignments"
                )
            text_sha256 = _sha256(text.encode("utf-8", errors="strict"))
            if text_sha256 != assignment.query_text_sha256:
                raise ScalablePartitionAuditError("query text digest differs from its assignment")
            if key in observed_keys or query_id in observed_ids:
                raise ScalablePartitionAuditError("query coverage repeats an identity")
            observed_keys.add(key)
            observed_ids.add(query_id)
            normalized_text, tokens = _normalize_query_text(text)
            nodes.append(
                _QueryNode(
                    dataset=source.dataset,
                    stage=source.stage,
                    query_id=query_id,
                    text_sha256=text_sha256,
                    normalized_text=normalized_text,
                    normalized_text_sha256=_sha256(normalized_text.encode("utf-8")),
                    tokens=tokens,
                    assignment_component_sha256=assignment.component_sha256,
                )
            )
            count += 1
        if count <= 0:
            raise ScalablePartitionAuditError("query artifact contains no rows")
        dataset_counts = inventory_counts.get(source.dataset)
        if (
            not isinstance(dataset_counts, Mapping)
            or dataset_counts.get(f"{source.stage}_queries") != count
        ):
            raise ScalablePartitionAuditError(
                "inventory query count differs from recomputed coverage"
            )
        count_rows.append(
            QueryCountRow(
                dataset=source.dataset,
                stage=source.stage,
                query_count=count,
            )
        )
    if observed_keys != set(assignments):
        raise ScalablePartitionAuditError(
            "assignment and query artifacts do not have exact query coverage"
        )
    return (
        tuple(sorted(nodes, key=lambda item: item.node_id)),
        tuple(sorted(count_rows)),
    )


def _validate_assignment_components(
    nodes: tuple[_QueryNode, ...],
) -> dict[str, tuple[_QueryNode, ...]]:
    groups: dict[str, list[_QueryNode]] = {}
    for node in nodes:
        groups.setdefault(node.assignment_component_sha256, []).append(node)
    result: dict[str, tuple[_QueryNode, ...]] = {}
    for component_sha256, members in groups.items():
        datasets = {node.dataset for node in members}
        stages = {node.stage for node in members}
        if len(datasets) != 1 or len(stages) != 1:
            raise ScalablePartitionAuditError(
                "an assignment component spans a corpus or stage boundary"
            )
        query_ids = sorted(
            (node.query_id for node in members),
            key=lambda value: value.encode("utf-8"),
        )
        expected = _sha256(_canonical_bytes(query_ids))
        if expected != component_sha256:
            raise ScalablePartitionAuditError(
                "assignment component digest differs from exact membership"
            )
        result[component_sha256] = tuple(sorted(members, key=lambda item: item.node_id))
    return result


def _load_qrels(
    root_descriptor: int,
    sources: tuple[SourceArtifactPin, ...],
    *,
    nodes_by_key: Mapping[tuple[str, str, str], _QueryNode],
) -> tuple[
    int,
    int,
    dict[tuple[str, str], tuple[_QueryNode, ...]],
]:
    qrel_count = 0
    positive_count = 0
    observed_pairs: set[tuple[str, str, str, str]] = set()
    observed_queries: set[tuple[str, str, str]] = set()
    positive_queries: set[tuple[str, str, str]] = set()
    positive_documents: dict[tuple[str, str], list[_QueryNode]] = {}
    for source in sources:
        assert source.dataset is not None and source.stage is not None
        for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
            row = _closed_mapping(
                value,
                fields=_QREL_FIELDS,
                label=f"qrel {source.path}:{line_number}",
            )
            query_id = _require_text("qrel query_id", row["query_id"])
            document_id = _require_text("qrel document_id", row["document_id"])
            relevance = row["relevance"]
            if isinstance(relevance, bool) or not isinstance(relevance, int):
                raise ScalablePartitionAuditError("qrel relevance must be an integer")
            key = (source.dataset, source.stage, query_id)
            node = nodes_by_key.get(key)
            if node is None:
                raise ScalablePartitionAuditError("qrel names an unknown staged query")
            pair = (*key, document_id)
            if pair in observed_pairs:
                raise ScalablePartitionAuditError("qrel sources repeat a query/document pair")
            observed_pairs.add(pair)
            observed_queries.add(key)
            qrel_count += 1
            if relevance > 0:
                positive_count += 1
                positive_queries.add(key)
                positive_documents.setdefault((source.dataset, document_id), []).append(node)
    if observed_queries != set(nodes_by_key) or positive_queries != set(nodes_by_key):
        raise ScalablePartitionAuditError(
            "qrel sources do not exactly cover every query with positive relevance"
        )
    return (
        qrel_count,
        positive_count,
        {
            key: tuple(sorted(set(values), key=lambda item: item.node_id))
            for key, values in positive_documents.items()
        },
    )


def _load_positive_document_content_groups(
    root_descriptor: int,
    sources: tuple[SourceArtifactPin, ...],
    *,
    positive_documents: Mapping[tuple[str, str], tuple[_QueryNode, ...]],
) -> dict[str, tuple[_QueryNode, ...]]:
    content_datasets = {source.dataset for source in sources}
    needed: dict[str, set[str]] = {}
    for dataset, document_id in positive_documents:
        if dataset in content_datasets:
            needed.setdefault(dataset, set()).add(document_id)
    observed: dict[tuple[str, str], str] = {}
    previous_ids: dict[str, bytes] = {}
    for source in sources:
        assert source.dataset is not None
        for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
            row = _closed_mapping(
                value,
                fields=_CORPUS_FIELDS,
                label=f"corpus {source.path}:{line_number}",
            )
            document_id = _require_text("corpus document id", row["id"])
            document_id_bytes = document_id.encode("utf-8", errors="strict")
            previous_id = previous_ids.get(source.dataset)
            if previous_id is not None and document_id_bytes <= previous_id:
                raise ScalablePartitionAuditError(
                    "corpus document IDs must be globally unique and bytewise sorted"
                )
            previous_ids[source.dataset] = document_id_bytes
            if document_id not in needed.get(source.dataset, set()):
                continue
            title = _require_body_text("corpus document title", row["title"])
            text = _require_body_text("corpus document text", row["text"])
            content_sha256 = _hash_parts(title, text)
            observed[(source.dataset, document_id)] = _sha256(
                _canonical_bytes(["suite-global-canonical-document-content-v2", content_sha256])
            )
    expected = {
        (dataset, document_id)
        for dataset, document_ids in needed.items()
        for document_id in document_ids
    }
    if set(observed) != expected:
        missing = len(expected - set(observed))
        raise ScalablePartitionAuditError(
            f"positive qrels name {missing} documents absent from pinned corpora"
        )
    groups: dict[str, list[_QueryNode]] = {}
    for identity, members in positive_documents.items():
        content_identity = observed.get(identity)
        if content_identity is None:
            continue
        groups.setdefault(content_identity, []).extend(members)
    return {
        key: tuple(sorted(set(values), key=lambda item: item.node_id))
        for key, values in groups.items()
    }


def _validate_structural_exclusion_isolation(
    exclusions: tuple[_StructuralExclusion, ...],
    *,
    nodes: tuple[_QueryNode, ...],
    positive_documents: Mapping[tuple[str, str], tuple[_QueryNode, ...]],
    positive_document_contents: Mapping[str, tuple[_QueryNode, ...]],
) -> None:
    admitted_normalized = {(node.dataset, node.normalized_text_sha256) for node in nodes}
    admitted_positive = {
        (dataset, _sha256(_canonical_bytes([dataset, document_id])))
        for dataset, document_id in positive_documents
    }
    admitted_content = set(positive_document_contents)
    for exclusion in exclusions:
        if (
            exclusion.dataset,
            exclusion.normalized_query_text_sha256,
        ) in admitted_normalized:
            raise ScalablePartitionAuditError(
                "a structurally excluded normalized query remains admitted"
            )
        if any(
            (exclusion.dataset, identity) in admitted_positive or identity in admitted_content
            for identity in exclusion.positive_relevance_identity_sha256s
        ):
            raise ScalablePartitionAuditError(
                "a structurally excluded positive-document component remains admitted"
            )


def _audit_graph(
    nodes: tuple[_QueryNode, ...],
    *,
    assignment_components: Mapping[str, tuple[_QueryNode, ...]],
    positive_documents: Mapping[tuple[str, str], tuple[_QueryNode, ...]],
    positive_document_contents: Mapping[str, tuple[_QueryNode, ...]],
) -> tuple[int, int, int, int, int, str, str, str]:
    disjoint_set = _DisjointSet(node.node_id for node in nodes)
    for members in assignment_components.values():
        component_edges: set[tuple[str, str]] = set()
        _connect_group(
            (node.node_id for node in members),
            disjoint_set=disjoint_set,
            edges=component_edges,
        )

    exact_groups: dict[str, list[_QueryNode]] = {}
    for node in nodes:
        exact_groups.setdefault(node.normalized_text, []).append(node)
    exact_edges: set[tuple[str, str]] = set()
    for members in exact_groups.values():
        _connect_group(
            (node.node_id for node in members),
            disjoint_set=disjoint_set,
            edges=exact_edges,
        )

    positive_edges: set[tuple[str, str]] = set()
    for members in positive_documents.values():
        stages = {node.stage for node in members}
        components = {node.assignment_component_sha256 for node in members}
        if len(stages) != 1:
            raise ScalablePartitionAuditError("a shared positive document crosses study stages")
        if len(components) != 1:
            raise ScalablePartitionAuditError(
                "a shared positive document is split across assignment components"
            )
        _connect_group(
            (node.node_id for node in members),
            disjoint_set=disjoint_set,
            edges=positive_edges,
        )

    positive_content_edges: set[tuple[str, str]] = set()
    for members in positive_document_contents.values():
        stages = {node.stage for node in members}
        components = {node.assignment_component_sha256 for node in members}
        if len(stages) != 1:
            raise ScalablePartitionAuditError(
                "shared positive document content crosses study stages"
            )
        if len(components) != 1:
            raise ScalablePartitionAuditError(
                "shared positive document content is split across assignment components"
            )
        _connect_group(
            (node.node_id for node in members),
            disjoint_set=disjoint_set,
            edges=positive_content_edges,
        )

    near_edges = _near_duplicate_pairs(nodes)
    for left, right in near_edges:
        disjoint_set.union(left, right)

    by_root: dict[str, list[_QueryNode]] = {}
    for node in nodes:
        by_root.setdefault(disjoint_set.find(node.node_id), []).append(node)
    crossings = [
        members for members in by_root.values() if len({node.stage for node in members}) > 1
    ]
    if crossings:
        first = sorted(crossings, key=lambda values: min(node.node_id for node in values))[0]
        stages = sorted({node.stage for node in first})
        raise ScalablePartitionAuditError(
            "query-partition component crosses stages " + ", ".join(stages)
        )

    audit_component_by_node: dict[str, str] = {}
    for members in by_root.values():
        identities = sorted(node.identity_sha256 for node in members)
        component_sha256 = _sha256(_canonical_bytes(identities))
        for node in members:
            audit_component_by_node[node.node_id] = component_sha256
    component_rows = [
        {
            "assignment_component_sha256": node.assignment_component_sha256,
            "audit_component_sha256": audit_component_by_node[node.node_id],
            "dataset": node.dataset,
            "query_identity_sha256": node.identity_sha256,
            "stage": node.stage,
        }
        for node in nodes
    ]
    positive_rows = [
        {
            "document_identity_sha256": _sha256(_canonical_bytes(list(identity))),
            "query_identity_sha256": sorted(node.identity_sha256 for node in members),
        }
        for identity, members in sorted(positive_documents.items())
    ]
    positive_content_rows = [
        {
            "document_content_identity_sha256": identity_sha256,
            "query_identity_sha256": sorted(node.identity_sha256 for node in members),
        }
        for identity_sha256, members in sorted(positive_document_contents.items())
    ]
    return (
        len(by_root),
        len(exact_edges),
        len(near_edges),
        len(positive_edges),
        len(positive_content_edges),
        _sha256(_canonical_bytes(component_rows)),
        _sha256(_canonical_bytes(positive_rows)),
        _sha256(_canonical_bytes(positive_content_rows)),
    )


def audit_staged_query_partitions(
    staged_root: str | Path,
) -> ScalableQueryPartitionAuditReceipt:
    """Recompute one passing audit from a verified staged package."""

    staged = _absolute_path(str(staged_root), name="staged package root")
    try:
        verified = verify_staged_data(staged)
    except (StudyDataError, OSError) as exc:
        raise ScalablePartitionAuditError(f"staged package verification failed: {exc}") from exc
    try:
        root_descriptor = _open_root(staged, label="staged package root")
    except ScalableCustodyError as exc:
        raise ScalablePartitionAuditError(str(exc)) from exc
    try:
        inventory, inventory_sources = _validate_inventory(
            root_descriptor,
            expected_sha256=verified.inventory_sha256,
        )
        (
            assignment_source,
            query_sources,
            qrel_sources,
            corpus_sources,
            exclusion_source,
        ) = _select_sources(inventory_sources)
        assignments = _load_assignments(root_descriptor, assignment_source)
        (
            structural_exclusions,
            structural_exclusion_counts,
            structural_exclusion_component_count,
            structural_exclusion_membership_sha256,
        ) = _load_structural_exclusions(
            root_descriptor,
            exclusion_source,
            assignments=assignments,
        )
        nodes, query_counts = _load_queries(
            root_descriptor,
            query_sources,
            assignments=assignments,
            inventory_counts=inventory["counts"],
        )
        assignment_components = _validate_assignment_components(nodes)
        nodes_by_key = {node.key: node for node in nodes}
        qrel_count, positive_qrel_count, positive_documents = _load_qrels(
            root_descriptor,
            qrel_sources,
            nodes_by_key=nodes_by_key,
        )
        positive_document_contents = _load_positive_document_content_groups(
            root_descriptor,
            corpus_sources,
            positive_documents=positive_documents,
        )
        _validate_structural_exclusion_isolation(
            structural_exclusions,
            nodes=nodes,
            positive_documents=positive_documents,
            positive_document_contents=positive_document_contents,
        )
        (
            audit_component_count,
            exact_text_edge_count,
            near_duplicate_edge_count,
            shared_positive_document_edge_count,
            shared_positive_document_content_edge_count,
            component_membership_sha256,
            positive_document_membership_sha256,
            positive_document_content_membership_sha256,
        ) = _audit_graph(
            nodes,
            assignment_components=assignment_components,
            positive_documents=positive_documents,
            positive_document_contents=positive_document_contents,
        )
    except ScalableCustodyError as exc:
        raise ScalablePartitionAuditError(str(exc)) from exc
    finally:
        os.close(root_descriptor)

    query_coverage_rows = [
        {
            "dataset": node.dataset,
            "query_identity_sha256": node.identity_sha256,
            "query_text_sha256": node.text_sha256,
            "stage": node.stage,
        }
        for node in nodes
    ]
    normalized_rows = [
        {
            "normalized_text_sha256": node.normalized_text_sha256,
            "query_identity_sha256": node.identity_sha256,
        }
        for node in nodes
    ]
    selected_sources = tuple(
        sorted(
            (
                assignment_source,
                exclusion_source,
                *query_sources,
                *qrel_sources,
                *corpus_sources,
            ),
            key=lambda item: item.path.encode(),
        )
    )
    return ScalableQueryPartitionAuditReceipt(
        staged_inventory_sha256=verified.inventory_sha256,
        staging_config_sha256=_require_sha256(
            "inventory config_sha256", inventory["config_sha256"]
        ),
        assignment_seed_sha256=_require_sha256(
            "inventory assignment_seed_sha256",
            inventory["assignment_seed_sha256"],
        ),
        algorithm_sha256=SCALABLE_PARTITION_ALGORITHM_SHA256,
        near_duplicate_config_sha256=FROZEN_QUERY_PARTITION_CONFIG_SHA256,
        source_artifacts=selected_sources,
        source_artifact_set_sha256=_source_artifact_set_sha256(selected_sources),
        assignment_artifact_sha256=assignment_source.sha256,
        query_counts=query_counts,
        query_artifact_count=len(query_sources),
        qrel_artifact_count=len(qrel_sources),
        corpus_artifact_count=len(corpus_sources),
        assignment_count=len(assignments),
        query_count=len(nodes),
        qrel_count=qrel_count,
        positive_qrel_count=positive_qrel_count,
        assignment_component_count=len(assignment_components),
        audit_component_count=audit_component_count,
        exact_text_edge_count=exact_text_edge_count,
        near_duplicate_edge_count=near_duplicate_edge_count,
        shared_positive_document_edge_count=(shared_positive_document_edge_count),
        shared_positive_document_content_edge_count=(shared_positive_document_content_edge_count),
        cross_stage_component_count=0,
        structural_exclusion_artifact_sha256=exclusion_source.sha256,
        structural_exclusion_query_count=len(structural_exclusions),
        structural_exclusion_component_count=(structural_exclusion_component_count),
        structural_exclusion_counts=structural_exclusion_counts,
        structural_exclusion_membership_sha256=(structural_exclusion_membership_sha256),
        query_coverage_sha256=_sha256(_canonical_bytes(query_coverage_rows)),
        normalized_text_membership_sha256=_sha256(_canonical_bytes(normalized_rows)),
        component_membership_sha256=component_membership_sha256,
        positive_document_membership_sha256=(positive_document_membership_sha256),
        positive_document_content_membership_sha256=(positive_document_content_membership_sha256),
    )


def load_scalable_partition_audit(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
    expected_inventory_sha256: str | None = None,
) -> ScalableQueryPartitionAuditReceipt:
    """Load and verify one canonical typed audit through a no-follow path walk."""

    try:
        audit_path = _absolute_path(str(path), name="query-partition audit path")
        parent_descriptor = _open_root(
            audit_path.parent,
            label="query-partition audit parent",
        )
        try:
            encoded = _read_secure_file(
                parent_descriptor,
                audit_path.name,
                maximum_bytes=_MAX_CONTROL_BYTES,
                label="query-partition audit",
            )
        finally:
            os.close(parent_descriptor)
        value = _decode_json(encoded, label="query-partition audit")
        receipt = ScalableQueryPartitionAuditReceipt.from_dict(value)
    except ScalableCustodyError as exc:
        raise ScalablePartitionAuditError(str(exc)) from exc
    if encoded != receipt.canonical_file_bytes():
        raise ScalablePartitionAuditError(
            "query-partition audit must be canonical JSON with one terminal newline"
        )
    if expected_artifact_sha256 is not None and receipt.artifact_sha256 != _require_sha256(
        "expected_artifact_sha256",
        expected_artifact_sha256,
    ):
        raise ScalablePartitionAuditError("query-partition audit artifact digest differs")
    if expected_inventory_sha256 is not None and (
        receipt.staged_inventory_sha256
        != _require_sha256(
            "expected_inventory_sha256",
            expected_inventory_sha256,
        )
    ):
        raise ScalablePartitionAuditError("query-partition audit names another staged inventory")
    return receipt


def build_scalable_partition_audit(
    staged_root: str | Path,
    output_path: str | Path,
) -> ScalableQueryPartitionAuditReceipt:
    """Build and exclusively write one canonical staged partition audit."""

    output = _absolute_path(str(output_path), name="query-partition audit output")
    if os.path.lexists(output):
        raise ScalablePartitionAuditError("query-partition audit output already exists")
    parent_descriptor = _open_root(
        output.parent,
        label="query-partition audit output parent",
    )
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise ScalablePartitionAuditError(
                "query-partition audit output parent is not a directory"
            )
    finally:
        os.close(parent_descriptor)
    receipt = audit_staged_query_partitions(staged_root)
    _write_exclusive(output, receipt.canonical_file_bytes())
    _fsync_directory(output.parent)
    loaded = load_scalable_partition_audit(
        output,
        expected_artifact_sha256=receipt.artifact_sha256,
        expected_inventory_sha256=receipt.staged_inventory_sha256,
    )
    if loaded != receipt:
        raise ScalablePartitionAuditError("written query-partition audit differs")
    return receipt


def verify_scalable_partition_audit_against_staged(
    audit_path: str | Path,
    staged_root: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
) -> ScalableQueryPartitionAuditReceipt:
    """Recompute staged evidence and require exact equality with a typed receipt."""

    receipt = load_scalable_partition_audit(
        audit_path,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    recomputed = audit_staged_query_partitions(staged_root)
    if recomputed != receipt:
        raise ScalablePartitionAuditError(
            "query-partition audit differs from recomputed staged evidence"
        )
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.scalable_partition_audit",
        description="Build or verify an inventory-derived query-partition audit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--staged-root", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--audit", required=True, type=Path)
    verify_parser.add_argument("--expected-sha256")
    staged_parser = subparsers.add_parser("verify-staged")
    staged_parser.add_argument("--audit", required=True, type=Path)
    staged_parser.add_argument("--staged-root", required=True, type=Path)
    staged_parser.add_argument("--expected-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "build":
            receipt = build_scalable_partition_audit(
                arguments.staged_root,
                arguments.output,
            )
        elif arguments.command == "verify":
            receipt = load_scalable_partition_audit(
                arguments.audit,
                expected_artifact_sha256=arguments.expected_sha256,
            )
        else:
            receipt = verify_scalable_partition_audit_against_staged(
                arguments.audit,
                arguments.staged_root,
                expected_artifact_sha256=arguments.expected_sha256,
            )
    except (OSError, ScalablePartitionAuditError) as exc:
        parser.exit(2, f"scalable-partition-audit: {exc}\n")
    print(
        json.dumps(
            {
                "artifact_sha256": receipt.artifact_sha256,
                "assignment_component_count": receipt.assignment_component_count,
                "audit_component_count": receipt.audit_component_count,
                "command": arguments.command,
                "cross_stage_component_count": receipt.cross_stage_component_count,
                "query_count": receipt.query_count,
                "staged_inventory_sha256": receipt.staged_inventory_sha256,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
