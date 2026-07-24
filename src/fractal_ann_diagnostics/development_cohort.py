"""Label-payload-excluded selection and later development-label materialization.

``select`` commits one representative query from each deterministically ranked
assignment component. Those components were constructed from positive-qrel
edges during staging, but ``select`` is intentionally incapable of opening
qrels or evidence payloads. ``materialize`` first reproduces that commitment
from the same frozen component graph, verifies the paired embedding stores, and
only then filters the development labels into a no-replace package.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .artifact_integrity import read_secure_regular_file
from .embedding_store import EmbeddingStoreReceipt, verify_embedding_store
from .joint_power_design import EVIDENCE_CORPORA, FIXED_CORPORA
from .query_cohort import (
    FAMILY_SELECTION_ALGORITHM,
    NESTED_ROWS_PER_FAMILY,
    REPRESENTATIVE_SELECTION_ALGORITHM,
    family_selection_rank,
    nested_trial_source_value,
    representative_selection_rank,
)
from .scalable_custody import (
    SourceArtifactPin,
    _absolute_path,
    _closed_mapping,
    _decode_json,
    _iter_canonical_jsonl,
    _open_root,
    _read_secure_file,
)
from .scalable_partition_audit import (
    ScalableQueryPartitionAuditReceipt,
    load_scalable_partition_audit,
)
from .study_data import ASSIGNMENT_SCHEMA, INVENTORY_SCHEMA

DEVELOPMENT_COHORT_SELECTION_SCHEMA = "fractal-development-cohort-selection-v1"
DEVELOPMENT_COHORT_MATERIALIZATION_SCHEMA = "fractal-development-cohort-materialization-v1"
DEVELOPMENT_EXECUTION_PLAN_SCHEMA = "fractal-development-execution-plan-v1"
DEVELOPMENT_EXECUTION_TRIAL_SCHEMA = "fractal-development-execution-trial-v1"
DEVELOPMENT_EMBEDDING_CONFIG_SCHEMA = "fractal-development-embedding-bindings-v1"
DEVELOPMENT_COHORT_CLI_RESULT_SCHEMA = "fractal-development-cohort-cli-result-v1"

FIT_FAMILY_COUNT = 200
CALIBRATION_FAMILY_COUNT = 75
FIT_SELECTION_SEED_SHA256 = "b4ce31a68caf104a0a81a8e3d2745ac91b980b269ddc14417c4fbe15cb34a33f"
CALIBRATION_SELECTION_SEED_SHA256 = (
    "287cfbc31f6108a0cd3a244826db49cb828b218c243358e13cc6686f901a1617"
)
DEVELOPMENT_TRIAL_DOMAIN = "fractal-development-trial-key-v1"

_STAGE_SPECS = {
    "fit": ("development-fit", FIT_FAMILY_COUNT, FIT_SELECTION_SEED_SHA256),
    "calibration": (
        "development-calibration",
        CALIBRATION_FAMILY_COUNT,
        CALIBRATION_SELECTION_SEED_SHA256,
    ),
}
_DEVELOPMENT_TO_SOURCE_STAGE = {
    development: source for source, (development, _count, _seed) in _STAGE_SPECS.items()
}
_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_MAX_LABEL_BYTES = 8 * 1024 * 1024 * 1024
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
_EVIDENCE_FIELDS = frozenset({"answer", "evidence_bundles", "label_metadata", "query_id"})
_ROW_ORDER_FIELDS = frozenset({"dataset", "id", "kind", "source_path", "source_row", "stage"})


class DevelopmentCohortError(RuntimeError):
    """Raised when the development cohort boundary cannot be reproduced."""


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DevelopmentCohortError("cohort artifacts require finite canonical JSON") from exc
    return encoded + (b"\n" if newline else b"")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_parts(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DevelopmentCohortError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise DevelopmentCohortError(f"{name} must be canonical non-empty text")
    return value


def _require_integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DevelopmentCohortError(f"{name} must be an integer >= {minimum}")
    return value


def _require_absolute_path(name: str, value: str | Path) -> Path:
    try:
        return _absolute_path(str(value), name=name)
    except Exception as exc:
        raise DevelopmentCohortError(str(exc)) from exc


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    try:
        return _closed_mapping(value, fields=fields, label=label)
    except Exception as exc:
        raise DevelopmentCohortError(str(exc)) from exc


def _decode(encoded: bytes, *, label: str) -> Any:
    try:
        return _decode_json(encoded, label=label)
    except Exception as exc:
        raise DevelopmentCohortError(str(exc)) from exc


@dataclass(frozen=True, order=True)
class SelectedDevelopmentFamily:
    """One frozen qrel-derived component and its fixed representative query."""

    family_rank_sha256: str
    component_sha256: str
    representative_rank_sha256: str
    query_id_sha256: str
    query_id: str
    query_text_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "family_rank_sha256",
            "component_sha256",
            "representative_rank_sha256",
            "query_id_sha256",
            "query_text_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_text("selected query_id", self.query_id)
        if _sha256(self.query_id.encode("utf-8")) != self.query_id_sha256:
            raise DevelopmentCohortError("selected query ID digest differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "component_sha256": self.component_sha256,
            "family_rank_sha256": self.family_rank_sha256,
            "query_id": self.query_id,
            "query_id_sha256": self.query_id_sha256,
            "query_text_sha256": self.query_text_sha256,
            "representative_rank_sha256": self.representative_rank_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> SelectedDevelopmentFamily:
        row = _closed(
            value,
            frozenset(
                {
                    "component_sha256",
                    "family_rank_sha256",
                    "query_id",
                    "query_id_sha256",
                    "query_text_sha256",
                    "representative_rank_sha256",
                }
            ),
            label="selected development family",
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True, order=True)
class DevelopmentStageSelection:
    """The fixed family selection for one corpus and development stage."""

    corpus: str
    development_stage: str
    source_stage: str
    requested_family_count: int
    available_component_count: int
    selection_seed_sha256: str
    selected_families: tuple[SelectedDevelopmentFamily, ...]

    def __post_init__(self) -> None:
        if self.corpus not in FIXED_CORPORA:
            raise DevelopmentCohortError("selection corpus is outside the fixed suite")
        spec = _STAGE_SPECS.get(self.source_stage)
        if spec is None or spec[0] != self.development_stage:
            raise DevelopmentCohortError("selection stage mapping differs")
        if self.requested_family_count != spec[1]:
            raise DevelopmentCohortError("selection requested count differs from the protocol")
        if self.selection_seed_sha256 != spec[2]:
            raise DevelopmentCohortError("selection seed differs from the protocol")
        _require_integer("available_component_count", self.available_component_count, minimum=1)
        families = tuple(self.selected_families)
        if (
            len(families) != self.requested_family_count
            or self.available_component_count < len(families)
            or len({row.component_sha256 for row in families}) != len(families)
            or len({row.query_id for row in families}) != len(families)
        ):
            raise DevelopmentCohortError("selection family counts or identities differ")
        expected_order = tuple(
            sorted(families, key=lambda row: (row.family_rank_sha256, row.component_sha256))
        )
        if families != expected_order:
            raise DevelopmentCohortError("selected families are not in registered rank order")
        for row in families:
            if row.family_rank_sha256 != family_selection_rank(
                corpus=self.corpus,
                stage=self.development_stage,
                selection_seed_sha256=self.selection_seed_sha256,
                component_sha256=row.component_sha256,
            ):
                raise DevelopmentCohortError("stored family rank does not reproduce")
            if row.representative_rank_sha256 != representative_selection_rank(
                corpus=self.corpus,
                stage=self.development_stage,
                selection_seed_sha256=self.selection_seed_sha256,
                component_sha256=row.component_sha256,
                query_id_sha256=row.query_id_sha256,
            ):
                raise DevelopmentCohortError("stored representative rank does not reproduce")
        object.__setattr__(self, "selected_families", families)

    @property
    def selected_query_ids(self) -> tuple[str, ...]:
        return tuple(row.query_id for row in self.selected_families)

    def to_dict(self) -> dict[str, object]:
        return {
            "available_component_count": self.available_component_count,
            "corpus": self.corpus,
            "development_stage": self.development_stage,
            "requested_family_count": self.requested_family_count,
            "selected_families": [row.to_dict() for row in self.selected_families],
            "selection_seed_sha256": self.selection_seed_sha256,
            "source_stage": self.source_stage,
        }

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentStageSelection:
        row = _closed(
            value,
            frozenset(
                {
                    "available_component_count",
                    "corpus",
                    "development_stage",
                    "requested_family_count",
                    "selected_families",
                    "selection_seed_sha256",
                    "source_stage",
                }
            ),
            label="development stage selection",
        )
        families = row["selected_families"]
        if not isinstance(families, list):
            raise DevelopmentCohortError("selected_families must be an array")
        return cls(
            corpus=row["corpus"],
            development_stage=row["development_stage"],
            source_stage=row["source_stage"],
            requested_family_count=row["requested_family_count"],
            available_component_count=row["available_component_count"],
            selection_seed_sha256=row["selection_seed_sha256"],
            selected_families=tuple(SelectedDevelopmentFamily.from_dict(item) for item in families),
        )


@dataclass(frozen=True)
class DevelopmentCohortSelectionReceipt:
    """A payload-excluded commitment to every qrel-derived representative."""

    staged_inventory_sha256: str
    assignment_artifact_sha256: str
    partition_audit_sha256: str
    partition_component_membership_sha256: str
    audit_source_artifact_set_sha256: str
    query_artifacts: tuple[SourceArtifactPin, ...]
    selections: tuple[DevelopmentStageSelection, ...]
    family_selection_algorithm: str = FAMILY_SELECTION_ALGORITHM
    representative_selection_algorithm: str = REPRESENTATIVE_SELECTION_ALGORITHM
    schema_version: str = DEVELOPMENT_COHORT_SELECTION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "staged_inventory_sha256",
            "assignment_artifact_sha256",
            "partition_audit_sha256",
            "partition_component_membership_sha256",
            "audit_source_artifact_set_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.family_selection_algorithm != FAMILY_SELECTION_ALGORITHM:
            raise DevelopmentCohortError("family selection algorithm differs")
        if self.representative_selection_algorithm != REPRESENTATIVE_SELECTION_ALGORITHM:
            raise DevelopmentCohortError("representative selection algorithm differs")
        if self.schema_version != DEVELOPMENT_COHORT_SELECTION_SCHEMA:
            raise DevelopmentCohortError("development selection schema differs")
        artifacts = tuple(sorted(self.query_artifacts, key=lambda row: row.path.encode("utf-8")))
        expected_artifacts = {
            f"datasets/{corpus}/{source_stage}/queries.jsonl": (corpus, source_stage)
            for source_stage in _STAGE_SPECS
            for corpus in FIXED_CORPORA
        }
        if (
            len(artifacts) != len(expected_artifacts)
            or {row.path for row in artifacts} != set(expected_artifacts)
            or any(
                (row.dataset, row.stage) != expected_artifacts[row.path]
                or row.role != "queries"
                or row.visibility != "online"
                for row in artifacts
            )
        ):
            raise DevelopmentCohortError("selection query artifact set differs")
        selections = tuple(
            sorted(self.selections, key=lambda row: (row.development_stage, row.corpus))
        )
        expected = {
            (development_stage, corpus)
            for development_stage, _count, _seed in _STAGE_SPECS.values()
            for corpus in FIXED_CORPORA
        }
        if (
            len(selections) != len(expected)
            or {(row.development_stage, row.corpus) for row in selections} != expected
        ):
            raise DevelopmentCohortError("selection does not cover the fixed ten strata")
        artifacts_by_key = {(row.dataset, row.stage): row for row in artifacts}
        for selection in selections:
            artifact = artifacts_by_key[(selection.corpus, selection.source_stage)]
            if artifact.record_count < selection.available_component_count:
                raise DevelopmentCohortError(
                    "selection component denominator exceeds its query artifact"
                )
        for corpus in FIXED_CORPORA:
            fit_components = {
                row.component_sha256
                for row in next(
                    item
                    for item in selections
                    if item.corpus == corpus and item.development_stage == "development-fit"
                ).selected_families
            }
            calibration_components = {
                row.component_sha256
                for row in next(
                    item
                    for item in selections
                    if item.corpus == corpus and item.development_stage == "development-calibration"
                ).selected_families
            }
            if fit_components.intersection(calibration_components):
                raise DevelopmentCohortError(
                    "development fit and calibration selections overlap a family"
                )
        object.__setattr__(self, "query_artifacts", artifacts)
        object.__setattr__(self, "selections", selections)

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_artifact_sha256": self.assignment_artifact_sha256,
            "audit_source_artifact_set_sha256": self.audit_source_artifact_set_sha256,
            "family_selection_algorithm": self.family_selection_algorithm,
            "partition_audit_sha256": self.partition_audit_sha256,
            "partition_component_membership_sha256": (self.partition_component_membership_sha256),
            "query_artifacts": [row.to_dict() for row in self.query_artifacts],
            "representative_selection_algorithm": self.representative_selection_algorithm,
            "schema_version": self.schema_version,
            "selections": [row.to_dict() for row in self.selections],
            "staged_inventory_sha256": self.staged_inventory_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    def selection(self, corpus: str, development_stage: str) -> DevelopmentStageSelection:
        matches = [
            row
            for row in self.selections
            if row.corpus == corpus and row.development_stage == development_stage
        ]
        if len(matches) != 1:
            raise DevelopmentCohortError("selection lookup is not singular")
        return matches[0]

    def selected_query_ids(self, corpus: str, development_stage: str) -> tuple[str, ...]:
        return self.selection(corpus, development_stage).selected_query_ids

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentCohortSelectionReceipt:
        row = _closed(
            value,
            frozenset(
                {
                    "assignment_artifact_sha256",
                    "audit_source_artifact_set_sha256",
                    "family_selection_algorithm",
                    "partition_audit_sha256",
                    "partition_component_membership_sha256",
                    "query_artifacts",
                    "representative_selection_algorithm",
                    "schema_version",
                    "selections",
                    "staged_inventory_sha256",
                }
            ),
            label="development cohort selection receipt",
        )
        artifacts = row["query_artifacts"]
        selections = row["selections"]
        if not isinstance(artifacts, list) or not isinstance(selections, list):
            raise DevelopmentCohortError("selection receipt arrays differ")
        return cls(
            staged_inventory_sha256=row["staged_inventory_sha256"],
            assignment_artifact_sha256=row["assignment_artifact_sha256"],
            partition_audit_sha256=row["partition_audit_sha256"],
            partition_component_membership_sha256=row["partition_component_membership_sha256"],
            audit_source_artifact_set_sha256=row["audit_source_artifact_set_sha256"],
            query_artifacts=tuple(SourceArtifactPin.from_dict(item) for item in artifacts),
            selections=tuple(DevelopmentStageSelection.from_dict(item) for item in selections),
            family_selection_algorithm=row["family_selection_algorithm"],
            representative_selection_algorithm=row["representative_selection_algorithm"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class _Assignment:
    component_sha256: str
    query_text_sha256: str


@dataclass(frozen=True)
class _QueryCandidate:
    query_id: str
    text: str
    component_sha256: str
    query_id_sha256: str
    query_text_sha256: str
    representative_rank_sha256: str


def _inventory_sources(
    root_descriptor: int,
    *,
    expected_inventory_sha256: str,
) -> tuple[Mapping[str, Any], tuple[SourceArtifactPin, ...]]:
    encoded = _read_secure_file(
        root_descriptor,
        "inventory.json",
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="staged inventory",
    )
    expected = _require_sha256("expected staged inventory SHA-256", expected_inventory_sha256)
    if _sha256(encoded) != expected:
        raise DevelopmentCohortError("staged inventory differs from its external pin")
    checksum = _read_secure_file(
        root_descriptor,
        "inventory.sha256",
        maximum_bytes=1024,
        label="staged inventory checksum",
    )
    if checksum != f"{expected}  inventory.json\n".encode("ascii"):
        raise DevelopmentCohortError("staged inventory checksum differs")
    value = _decode(encoded, label="staged inventory")
    inventory = _closed(value, _INVENTORY_FIELDS, label="staged inventory")
    if inventory["schema_version"] != INVENTORY_SCHEMA or encoded != _canonical_bytes(value):
        raise DevelopmentCohortError("staged inventory schema or canonical bytes differ")
    values = inventory["artifacts"]
    if not isinstance(values, list) or not values:
        raise DevelopmentCohortError("staged inventory has no artifacts")
    try:
        sources = tuple(SourceArtifactPin.from_dict(item) for item in values)
    except Exception as exc:
        raise DevelopmentCohortError(f"invalid staged artifact pin: {exc}") from exc
    if sources != tuple(sorted(sources, key=lambda row: row.path.encode("utf-8"))) or len(
        {row.path for row in sources}
    ) != len(sources):
        raise DevelopmentCohortError("staged artifacts are repeated or not canonically sorted")
    return inventory, sources


def _select_label_free_sources(
    sources: tuple[SourceArtifactPin, ...],
    audit: ScalableQueryPartitionAuditReceipt,
) -> tuple[SourceArtifactPin, tuple[SourceArtifactPin, ...]]:
    assignments = [
        row
        for row in sources
        if row.path == "assignments.jsonl"
        and row.dataset is None
        and row.stage is None
        and row.role == "assignments"
        and row.visibility == "online"
    ]
    if len(assignments) != 1:
        raise DevelopmentCohortError("staged inventory lacks one canonical assignment ledger")
    query_sources: list[SourceArtifactPin] = []
    for source_stage in _STAGE_SPECS:
        for corpus in FIXED_CORPORA:
            expected_path = f"datasets/{corpus}/{source_stage}/queries.jsonl"
            matches = [
                row
                for row in sources
                if row.path == expected_path
                and row.dataset == corpus
                and row.stage == source_stage
                and row.role == "queries"
                and row.visibility == "online"
            ]
            if len(matches) != 1:
                raise DevelopmentCohortError(
                    f"staged inventory lacks one query source for {source_stage}:{corpus}"
                )
            query_sources.append(matches[0])

    audit_by_path = {row.path: row for row in audit.source_artifacts}
    for source in (assignments[0], *query_sources):
        if audit_by_path.get(source.path) != source:
            raise DevelopmentCohortError(
                f"payload-excluded source {source.path!r} differs from the partition audit"
            )
    if assignments[0].sha256 != audit.assignment_artifact_sha256:
        raise DevelopmentCohortError("assignment digest differs from the partition audit")
    audit_counts = {(row.stage, row.dataset): row.query_count for row in audit.query_counts}
    for source in query_sources:
        if audit_counts.get((source.stage, source.dataset)) != source.record_count:
            raise DevelopmentCohortError(
                f"query count for {source.stage}:{source.dataset} differs from the audit"
            )
    return assignments[0], tuple(sorted(query_sources, key=lambda row: row.path.encode("utf-8")))


def _load_development_assignments(
    root_descriptor: int,
    source: SourceArtifactPin,
) -> dict[tuple[str, str, str], _Assignment]:
    assignments: dict[tuple[str, str, str], _Assignment] = {}
    for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
        row = _closed(value, _ASSIGNMENT_FIELDS, label=f"assignment line {line_number}")
        if row["schema_version"] != ASSIGNMENT_SCHEMA:
            raise DevelopmentCohortError("assignment schema differs")
        corpus = _require_text("assignment dataset", row["dataset"])
        stage = _require_text("assignment stage", row["stage"])
        query_id = _require_text("assignment query_id", row["query_id"])
        _require_sha256("assignment key", row["assignment_key_sha256"])
        component = _require_sha256(
            "assignment partition component", row["partition_component_sha256"]
        )
        text_sha256 = _require_sha256("assignment query text", row["query_text_sha256"])
        _require_text("assignment source_split", row["source_split"])
        if row["domain"] is not None:
            _require_text("assignment domain", row["domain"])
        if stage not in _STAGE_SPECS:
            continue
        if corpus not in FIXED_CORPORA:
            raise DevelopmentCohortError("development assignment names an unknown corpus")
        key = (corpus, stage, query_id)
        if key in assignments:
            raise DevelopmentCohortError("development assignment repeats a query")
        assignments[key] = _Assignment(component, text_sha256)
    if not assignments:
        raise DevelopmentCohortError("assignment ledger has no development rows")
    return assignments


def _select_one_stratum(
    root_descriptor: int,
    source: SourceArtifactPin,
    *,
    assignments: dict[tuple[str, str, str], _Assignment],
) -> DevelopmentStageSelection:
    if source.dataset is None or source.stage is None:
        raise DevelopmentCohortError("development query pin lacks corpus or stage")
    corpus = source.dataset
    source_stage = source.stage
    development_stage, requested, seed = _STAGE_SPECS[source_stage]
    candidates: dict[str, list[_QueryCandidate]] = defaultdict(list)
    observed_ids: set[str] = set()
    previous_id: bytes | None = None
    for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
        row = _closed(value, _QUERY_FIELDS, label=f"query {source.path}:{line_number}")
        query_id = _require_text("query id", row["id"])
        text = row["text"]
        if not isinstance(text, str) or unicodedata.normalize("NFC", text) != text:
            raise DevelopmentCohortError("query text must be NFC text")
        encoded_id = query_id.encode("utf-8")
        if previous_id is not None and encoded_id <= previous_id:
            raise DevelopmentCohortError("query IDs must be unique and bytewise sorted")
        previous_id = encoded_id
        assignment = assignments.pop((corpus, source_stage, query_id), None)
        if assignment is None:
            raise DevelopmentCohortError("development query lacks its assignment")
        text_sha256 = _sha256(text.encode("utf-8"))
        if text_sha256 != assignment.query_text_sha256:
            raise DevelopmentCohortError("query text differs from its assignment digest")
        query_id_sha256 = _sha256(encoded_id)
        representative_rank = representative_selection_rank(
            corpus=corpus,
            stage=development_stage,
            selection_seed_sha256=seed,
            component_sha256=assignment.component_sha256,
            query_id_sha256=query_id_sha256,
        )
        candidates[assignment.component_sha256].append(
            _QueryCandidate(
                query_id=query_id,
                text=text,
                component_sha256=assignment.component_sha256,
                query_id_sha256=query_id_sha256,
                query_text_sha256=text_sha256,
                representative_rank_sha256=representative_rank,
            )
        )
        observed_ids.add(query_id)
    if len(observed_ids) != source.record_count:
        raise DevelopmentCohortError("query source count differs from its inventory pin")
    available = len(candidates)
    if available < requested:
        raise DevelopmentCohortError(
            f"development component count underflows {development_stage}:{corpus}; "
            f"requested={requested}, available={available}"
        )
    ranked_components = sorted(
        candidates,
        key=lambda component: (
            family_selection_rank(
                corpus=corpus,
                stage=development_stage,
                selection_seed_sha256=seed,
                component_sha256=component,
            ),
            component,
        ),
    )
    selected: list[SelectedDevelopmentFamily] = []
    for component in ranked_components[:requested]:
        representative = min(
            candidates[component],
            key=lambda row: (row.representative_rank_sha256, row.query_id_sha256),
        )
        selected.append(
            SelectedDevelopmentFamily(
                family_rank_sha256=family_selection_rank(
                    corpus=corpus,
                    stage=development_stage,
                    selection_seed_sha256=seed,
                    component_sha256=component,
                ),
                component_sha256=component,
                representative_rank_sha256=representative.representative_rank_sha256,
                query_id_sha256=representative.query_id_sha256,
                query_id=representative.query_id,
                query_text_sha256=representative.query_text_sha256,
            )
        )
    return DevelopmentStageSelection(
        corpus=corpus,
        development_stage=development_stage,
        source_stage=source_stage,
        requested_family_count=requested,
        available_component_count=available,
        selection_seed_sha256=seed,
        selected_families=tuple(selected),
    )


def _derive_selection_receipt(
    staged_root: str | Path,
    *,
    staged_inventory_sha256: str,
    partition_audit_path: str | Path,
    partition_audit_sha256: str,
) -> DevelopmentCohortSelectionReceipt:
    expected_inventory = _require_sha256("staged_inventory_sha256", staged_inventory_sha256)
    audit = load_scalable_partition_audit(
        partition_audit_path,
        expected_artifact_sha256=partition_audit_sha256,
        expected_inventory_sha256=expected_inventory,
    )
    root = _require_absolute_path("staged root", staged_root)
    root_descriptor = _open_root(root, label="development staged root")
    try:
        _inventory, sources = _inventory_sources(
            root_descriptor,
            expected_inventory_sha256=expected_inventory,
        )
        assignment_source, query_sources = _select_label_free_sources(sources, audit)
        assignments = _load_development_assignments(root_descriptor, assignment_source)
        selections = tuple(
            _select_one_stratum(root_descriptor, source, assignments=assignments)
            for source in query_sources
        )
        remaining_development = [
            key for key in assignments if key[0] in FIXED_CORPORA and key[1] in _STAGE_SPECS
        ]
        if remaining_development:
            raise DevelopmentCohortError(
                "development assignments contain queries absent from their query source"
            )
    finally:
        os.close(root_descriptor)
    return DevelopmentCohortSelectionReceipt(
        staged_inventory_sha256=expected_inventory,
        assignment_artifact_sha256=audit.assignment_artifact_sha256,
        partition_audit_sha256=audit.artifact_sha256,
        partition_component_membership_sha256=audit.component_membership_sha256,
        audit_source_artifact_set_sha256=audit.source_artifact_set_sha256,
        query_artifacts=query_sources,
        selections=selections,
    )


def _ensure_real_directory(path: Path, *, label: str) -> int:
    """Open or create an absolute directory without traversing symbolic links."""

    target = _require_absolute_path(label, path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open("/", flags)
        for component in target.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise DevelopmentCohortError(
                    f"{label} component {component!r} is not a real directory"
                )
            os.close(descriptor)
            descriptor = child
    except OSError as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise DevelopmentCohortError(
            f"cannot open or create {label} without symbolic links: {exc}"
        ) from exc
    except BaseException:
        if "descriptor" in locals():
            os.close(descriptor)
        raise
    return descriptor


def _write_exclusive_file(path: Path, encoded: bytes) -> None:
    parent_descriptor = _ensure_real_directory(
        path.parent,
        label=f"{path.name} output parent",
    )
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
        except OSError as exc:
            raise DevelopmentCohortError(f"cannot create {path}: {exc}") from exc
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def select_development_cohort(
    staged_root: str | Path,
    output_path: str | Path,
    *,
    staged_inventory_sha256: str,
    partition_audit_path: str | Path,
    partition_audit_sha256: str,
) -> DevelopmentCohortSelectionReceipt:
    """Publish the fixed cohort without opening qrel or evidence payloads."""

    output = _require_absolute_path("selection receipt output", output_path)
    if os.path.lexists(output):
        raise DevelopmentCohortError("selection receipt output already exists")
    output_parent = _ensure_real_directory(
        output.parent,
        label="selection receipt output parent",
    )
    os.close(output_parent)
    receipt = _derive_selection_receipt(
        staged_root,
        staged_inventory_sha256=staged_inventory_sha256,
        partition_audit_path=partition_audit_path,
        partition_audit_sha256=partition_audit_sha256,
    )
    _write_exclusive_file(output, receipt.canonical_file_bytes())
    loaded = load_development_cohort_selection(
        output,
        expected_artifact_sha256=receipt.artifact_sha256,
        expected_inventory_sha256=receipt.staged_inventory_sha256,
    )
    if loaded != receipt:
        raise DevelopmentCohortError("published selection receipt differs")
    return receipt


def load_development_cohort_selection(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
    expected_inventory_sha256: str | None = None,
) -> DevelopmentCohortSelectionReceipt:
    """Load one exact canonical development selection receipt."""

    receipt_path = _require_absolute_path("selection receipt path", path)
    encoded = read_secure_regular_file(
        receipt_path,
        max_bytes=_MAX_CONTROL_BYTES,
        label="development cohort selection receipt",
    )
    value = _decode(encoded, label="development cohort selection receipt")
    receipt = DevelopmentCohortSelectionReceipt.from_dict(value)
    if encoded != receipt.canonical_file_bytes():
        raise DevelopmentCohortError("development selection receipt is not canonical")
    if expected_artifact_sha256 is not None and receipt.artifact_sha256 != _require_sha256(
        "expected selection receipt SHA-256", expected_artifact_sha256
    ):
        raise DevelopmentCohortError("development selection receipt digest differs")
    if expected_inventory_sha256 is not None and (
        receipt.staged_inventory_sha256
        != _require_sha256("expected staged inventory SHA-256", expected_inventory_sha256)
    ):
        raise DevelopmentCohortError("development selection names another inventory")
    return receipt


@dataclass(frozen=True, order=True)
class DevelopmentEmbeddingBinding:
    """One exact paired embedding store admitted for materialization."""

    corpus: str
    development_stage: str
    root: Path
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.corpus not in FIXED_CORPORA:
            raise DevelopmentCohortError("embedding binding corpus is outside the fixed suite")
        if self.development_stage not in _DEVELOPMENT_TO_SOURCE_STAGE:
            raise DevelopmentCohortError("embedding binding stage is not a development stage")
        object.__setattr__(self, "root", _require_absolute_path("embedding root", self.root))
        _require_sha256("embedding receipt_sha256", self.receipt_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "development_stage": self.development_stage,
            "receipt_sha256": self.receipt_sha256,
            "root": str(self.root),
        }

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentEmbeddingBinding:
        row = _closed(
            value,
            frozenset({"corpus", "development_stage", "receipt_sha256", "root"}),
            label="development embedding binding",
        )
        return cls(
            corpus=row["corpus"],
            development_stage=row["development_stage"],
            root=Path(row["root"]),
            receipt_sha256=row["receipt_sha256"],
        )


@dataclass(frozen=True, order=True)
class DevelopmentExecutionTrial:
    """One of the three policy-state rows for a selected development family."""

    family_key: str
    trial_key: str
    query_id: str
    query_row: int
    nested_index: int
    schema_version: str = DEVELOPMENT_EXECUTION_TRIAL_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("development family_key", self.family_key)
        _require_sha256("development trial_key", self.trial_key)
        _require_text("development trial query_id", self.query_id)
        _require_integer("development trial query_row", self.query_row)
        if self.nested_index not in range(NESTED_ROWS_PER_FAMILY):
            raise DevelopmentCohortError("development nested_index is outside [0, 3)")
        if self.schema_version != DEVELOPMENT_EXECUTION_TRIAL_SCHEMA:
            raise DevelopmentCohortError("development execution trial schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "family_key": self.family_key,
            "nested_index": self.nested_index,
            "query_id": self.query_id,
            "query_row": self.query_row,
            "schema_version": self.schema_version,
            "trial_key": self.trial_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentExecutionTrial:
        row = _closed(
            value,
            frozenset(
                {
                    "family_key",
                    "nested_index",
                    "query_id",
                    "query_row",
                    "schema_version",
                    "trial_key",
                }
            ),
            label="development execution trial",
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DevelopmentExecutionPlan:
    """A policy-compiler-compatible three-row plan bound to paired embeddings."""

    corpus: str
    stage: str
    document_count: int
    document_universe_sha256: str
    document_row_order_sha256: str
    query_row_order_sha256: str
    embedding_receipt_sha256: str
    selection_receipt_sha256: str
    selected_family_count: int
    trials: tuple[DevelopmentExecutionTrial, ...]
    nested_rows_per_family: int = NESTED_ROWS_PER_FAMILY
    schema_version: str = DEVELOPMENT_EXECUTION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.corpus not in FIXED_CORPORA:
            raise DevelopmentCohortError("development plan corpus is outside the fixed suite")
        if self.stage not in _DEVELOPMENT_TO_SOURCE_STAGE:
            raise DevelopmentCohortError("development plan stage differs")
        _require_integer("development plan document_count", self.document_count, minimum=1)
        for name in (
            "document_universe_sha256",
            "document_row_order_sha256",
            "query_row_order_sha256",
            "embedding_receipt_sha256",
            "selection_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.document_universe_sha256 != self.document_row_order_sha256:
            raise DevelopmentCohortError("development document universe binding differs")
        _require_integer("selected_family_count", self.selected_family_count, minimum=1)
        source_stage = _DEVELOPMENT_TO_SOURCE_STAGE[self.stage]
        if self.selected_family_count != _STAGE_SPECS[source_stage][1]:
            raise DevelopmentCohortError(
                "development plan selected family count differs from the protocol"
            )
        if self.nested_rows_per_family != NESTED_ROWS_PER_FAMILY:
            raise DevelopmentCohortError("development plan nested row count differs")
        trials = tuple(self.trials)
        if (
            len(trials) != self.selected_family_count * self.nested_rows_per_family
            or not all(isinstance(row, DevelopmentExecutionTrial) for row in trials)
            or len({row.trial_key for row in trials}) != len(trials)
        ):
            raise DevelopmentCohortError("development plan trial counts differ")
        by_family: dict[str, list[DevelopmentExecutionTrial]] = defaultdict(list)
        for trial in trials:
            by_family[trial.family_key].append(trial)
        if len(by_family) != self.selected_family_count:
            raise DevelopmentCohortError("development plan family count differs")
        for rows in by_family.values():
            if (
                {row.nested_index for row in rows} != set(range(NESTED_ROWS_PER_FAMILY))
                or len({row.query_id for row in rows}) != 1
                or len({row.query_row for row in rows}) != 1
            ):
                raise DevelopmentCohortError("development family is not exactly three nested rows")
            for row in rows:
                expected_trial_key = _hash_parts(
                    DEVELOPMENT_TRIAL_DOMAIN,
                    self.corpus,
                    self.stage,
                    row.family_key,
                    nested_trial_source_value(row.query_id, row.nested_index),
                )
                if row.trial_key != expected_trial_key:
                    raise DevelopmentCohortError("development trial key does not reproduce")
        if (
            len({rows[0].query_id for rows in by_family.values()}) != self.selected_family_count
            or len({rows[0].query_row for rows in by_family.values()}) != self.selected_family_count
        ):
            raise DevelopmentCohortError(
                "development families must use distinct query IDs and embedding rows"
            )
        expected_order = tuple(sorted(trials, key=lambda row: (row.family_key, row.nested_index)))
        if trials != expected_order:
            raise DevelopmentCohortError("development trials are not canonically ordered")
        if self.schema_version != DEVELOPMENT_EXECUTION_PLAN_SCHEMA:
            raise DevelopmentCohortError("development execution plan schema differs")
        object.__setattr__(self, "trials", trials)

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "document_count": self.document_count,
            "document_row_order_sha256": self.document_row_order_sha256,
            "document_universe_sha256": self.document_universe_sha256,
            "embedding_receipt_sha256": self.embedding_receipt_sha256,
            "nested_rows_per_family": self.nested_rows_per_family,
            "query_row_order_sha256": self.query_row_order_sha256,
            "schema_version": self.schema_version,
            "selected_family_count": self.selected_family_count,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "stage": self.stage,
            "trials": [row.to_dict() for row in self.trials],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentExecutionPlan:
        row = _closed(
            value,
            frozenset(
                {
                    "corpus",
                    "document_count",
                    "document_row_order_sha256",
                    "document_universe_sha256",
                    "embedding_receipt_sha256",
                    "nested_rows_per_family",
                    "query_row_order_sha256",
                    "schema_version",
                    "selected_family_count",
                    "selection_receipt_sha256",
                    "stage",
                    "trials",
                }
            ),
            label="development execution plan",
        )
        trials = row["trials"]
        if not isinstance(trials, list):
            raise DevelopmentCohortError("development execution trials must be an array")
        return cls(
            corpus=row["corpus"],
            stage=row["stage"],
            document_count=row["document_count"],
            document_universe_sha256=row["document_universe_sha256"],
            document_row_order_sha256=row["document_row_order_sha256"],
            query_row_order_sha256=row["query_row_order_sha256"],
            embedding_receipt_sha256=row["embedding_receipt_sha256"],
            selection_receipt_sha256=row["selection_receipt_sha256"],
            selected_family_count=row["selected_family_count"],
            trials=tuple(DevelopmentExecutionTrial.from_dict(item) for item in trials),
            nested_rows_per_family=row["nested_rows_per_family"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True, order=True)
class MaterializedDevelopmentArtifact:
    path: str
    sha256: str
    byte_count: int
    record_count: int
    role: str
    corpus: str | None
    stage: str | None

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if (
            not self.path
            or path.is_absolute()
            or str(path) != self.path
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise DevelopmentCohortError("materialized artifact path is not canonical")
        _require_sha256("materialized artifact SHA-256", self.sha256)
        _require_integer("materialized byte_count", self.byte_count, minimum=1)
        _require_integer("materialized record_count", self.record_count, minimum=1)
        _require_text("materialized role", self.role)
        if self.corpus is not None and self.corpus not in FIXED_CORPORA:
            raise DevelopmentCohortError("materialized corpus differs")
        if self.stage is not None and self.stage not in _DEVELOPMENT_TO_SOURCE_STAGE:
            raise DevelopmentCohortError("materialized stage differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "corpus": self.corpus,
            "path": self.path,
            "record_count": self.record_count,
            "role": self.role,
            "sha256": self.sha256,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, value: object) -> MaterializedDevelopmentArtifact:
        row = _closed(
            value,
            frozenset({"byte_count", "corpus", "path", "record_count", "role", "sha256", "stage"}),
            label="materialized development artifact",
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DevelopmentCohortMaterializationReceipt:
    staged_inventory_sha256: str
    partition_audit_sha256: str
    selection_receipt_sha256: str
    artifacts: tuple[MaterializedDevelopmentArtifact, ...]
    embedding_bindings: tuple[DevelopmentEmbeddingBinding, ...]
    schema_version: str = DEVELOPMENT_COHORT_MATERIALIZATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "staged_inventory_sha256",
            "partition_audit_sha256",
            "selection_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        artifacts = tuple(sorted(self.artifacts, key=lambda row: row.path.encode("utf-8")))
        if len({row.path for row in artifacts}) != len(artifacts):
            raise DevelopmentCohortError("materialized artifact set is repeated")
        expected_artifacts = _materialized_artifact_contract()
        if {row.path for row in artifacts} != set(expected_artifacts):
            raise DevelopmentCohortError("materialized artifact set is not protocol-complete")
        for artifact in artifacts:
            role, corpus, stage, minimum_records, exact_records = expected_artifacts[artifact.path]
            if (
                (artifact.role, artifact.corpus, artifact.stage) != (role, corpus, stage)
                or artifact.record_count < minimum_records
                or (exact_records is not None and artifact.record_count != exact_records)
            ):
                raise DevelopmentCohortError(
                    f"materialized artifact contract differs for {artifact.path!r}"
                )
        bindings = tuple(
            sorted(self.embedding_bindings, key=lambda row: (row.development_stage, row.corpus))
        )
        expected = {
            (stage, corpus) for stage in _DEVELOPMENT_TO_SOURCE_STAGE for corpus in FIXED_CORPORA
        }
        if (
            len(bindings) != len(expected)
            or {(row.development_stage, row.corpus) for row in bindings} != expected
        ):
            raise DevelopmentCohortError("materialization embedding binding set differs")
        if self.schema_version != DEVELOPMENT_COHORT_MATERIALIZATION_SCHEMA:
            raise DevelopmentCohortError("materialization receipt schema differs")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "embedding_bindings", bindings)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [row.to_dict() for row in self.artifacts],
            "embedding_bindings": [row.to_dict() for row in self.embedding_bindings],
            "partition_audit_sha256": self.partition_audit_sha256,
            "schema_version": self.schema_version,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "staged_inventory_sha256": self.staged_inventory_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentCohortMaterializationReceipt:
        row = _closed(
            value,
            frozenset(
                {
                    "artifacts",
                    "embedding_bindings",
                    "partition_audit_sha256",
                    "schema_version",
                    "selection_receipt_sha256",
                    "staged_inventory_sha256",
                }
            ),
            label="development cohort materialization receipt",
        )
        artifacts = row["artifacts"]
        bindings = row["embedding_bindings"]
        if not isinstance(artifacts, list) or not isinstance(bindings, list):
            raise DevelopmentCohortError("materialization receipt arrays differ")
        return cls(
            staged_inventory_sha256=row["staged_inventory_sha256"],
            partition_audit_sha256=row["partition_audit_sha256"],
            selection_receipt_sha256=row["selection_receipt_sha256"],
            artifacts=tuple(MaterializedDevelopmentArtifact.from_dict(item) for item in artifacts),
            embedding_bindings=tuple(
                DevelopmentEmbeddingBinding.from_dict(item) for item in bindings
            ),
            schema_version=row["schema_version"],
        )


def _materialized_artifact_contract() -> dict[
    str, tuple[str, str | None, str | None, int, int | None]
]:
    """Return the exact path and metadata contract for one development package."""

    expected: dict[str, tuple[str, str | None, str | None, int, int | None]] = {
        "selection-receipt.json": (
            "development-cohort-selection",
            None,
            None,
            1,
            1,
        )
    }
    for development_stage, family_count, _seed in _STAGE_SPECS.values():
        for corpus in FIXED_CORPORA:
            root = f"{development_stage}/{corpus}"
            expected[f"{root}/queries.jsonl"] = (
                "queries",
                corpus,
                development_stage,
                family_count,
                family_count,
            )
            expected[f"{root}/qrels.jsonl"] = (
                "qrels",
                corpus,
                development_stage,
                family_count,
                None,
            )
            expected[f"{root}/execution-plan.json"] = (
                "development-execution-plan",
                corpus,
                development_stage,
                1,
                1,
            )
            if corpus in EVIDENCE_CORPORA:
                expected[f"{root}/evidence-bundles.jsonl"] = (
                    "evidence-bundles",
                    corpus,
                    development_stage,
                    family_count,
                    family_count,
                )
    return expected


def _decode_canonical_jsonl(encoded: bytes, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not encoded or not encoded.endswith(b"\n"):
        raise DevelopmentCohortError(f"{label} must be non-empty canonical JSONL")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(encoded.splitlines(keepends=True), start=1):
        value = _decode(line, label=f"{label} line {line_number}")
        if not isinstance(value, Mapping) or line != _canonical_bytes(value):
            raise DevelopmentCohortError(f"{label} line {line_number} is not canonical JSON")
        rows.append(value)
    return tuple(rows)


def _embedding_query_positions(
    binding: DevelopmentEmbeddingBinding,
    receipt: EmbeddingStoreReceipt,
    selection: DevelopmentStageSelection,
) -> dict[str, int]:
    descriptor = receipt.row_orders["queries"]
    encoded = read_secure_regular_file(
        binding.root / descriptor.relative_path,
        max_bytes=descriptor.byte_count,
        label=f"{binding.development_stage}:{binding.corpus} query row order",
    )
    if len(encoded) != descriptor.byte_count or _sha256(encoded) != descriptor.file_sha256:
        raise DevelopmentCohortError("embedding query row order differs from its receipt")
    selected = set(selection.selected_query_ids)
    positions: dict[str, int] = {}
    source_stage = _DEVELOPMENT_TO_SOURCE_STAGE[binding.development_stage]
    rows = _decode_canonical_jsonl(encoded, label="embedding query row order")
    if len(rows) != receipt.query_count:
        raise DevelopmentCohortError("embedding query row count differs")
    for position, value in enumerate(rows):
        row = _closed(value, _ROW_ORDER_FIELDS, label="embedding query row")
        if row["kind"] not in {"query", "queries"}:
            raise DevelopmentCohortError("embedding query row kind differs")
        query_id = _require_text("embedding query ID", row["id"])
        if (
            row["dataset"] == binding.corpus
            and row["stage"] == source_stage
            and query_id in selected
        ):
            if query_id in positions:
                raise DevelopmentCohortError("embedding query row order repeats a selected ID")
            positions[query_id] = position
    if set(positions) != selected:
        raise DevelopmentCohortError(
            "paired embedding query rows do not cover the exact selected representatives"
        )
    return positions


def _verify_embedding_bindings(
    bindings: Sequence[DevelopmentEmbeddingBinding],
    selection: DevelopmentCohortSelectionReceipt,
    *,
    inventory_counts: Mapping[str, Any],
) -> tuple[
    tuple[DevelopmentEmbeddingBinding, ...],
    dict[tuple[str, str], tuple[EmbeddingStoreReceipt, dict[str, int]]],
]:
    ordered = tuple(sorted(bindings, key=lambda row: (row.development_stage, row.corpus)))
    expected = {
        (stage, corpus) for stage in _DEVELOPMENT_TO_SOURCE_STAGE for corpus in FIXED_CORPORA
    }
    if (
        len(ordered) != len(expected)
        or {(row.development_stage, row.corpus) for row in ordered} != expected
    ):
        raise DevelopmentCohortError("embedding bindings must cover the fixed ten strata")
    verified: dict[tuple[str, str], tuple[EmbeddingStoreReceipt, dict[str, int]]] = {}
    for binding in ordered:
        try:
            receipt = verify_embedding_store(binding.root)
        except Exception as exc:
            raise DevelopmentCohortError(
                f"cannot verify embedding store {binding.development_stage}:{binding.corpus}: {exc}"
            ) from exc
        if receipt.receipt_sha256 != binding.receipt_sha256:
            raise DevelopmentCohortError("paired embedding receipt differs from its exact pin")
        if receipt.staged_inventory_sha256 != selection.staged_inventory_sha256:
            raise DevelopmentCohortError("paired embedding store names another staged inventory")
        if receipt.old_model is None or set(receipt.vectors) != {
            "current_documents",
            "current_queries",
            "old_documents",
            "old_queries",
        }:
            raise DevelopmentCohortError("embedding store is not an exact paired old/current store")
        corpus_counts = inventory_counts.get(binding.corpus)
        if not isinstance(corpus_counts, Mapping) or (
            corpus_counts.get("documents") != receipt.document_count
        ):
            raise DevelopmentCohortError("embedding document count differs from staged inventory")
        stratum = selection.selection(binding.corpus, binding.development_stage)
        positions = _embedding_query_positions(binding, receipt, stratum)
        verified[(binding.development_stage, binding.corpus)] = (receipt, positions)
    return ordered, verified


def _selected_query_payloads(
    root_descriptor: int,
    sources: Sequence[SourceArtifactPin],
    selection: DevelopmentCohortSelectionReceipt,
) -> dict[tuple[str, str], tuple[dict[str, str], ...]]:
    result: dict[tuple[str, str], tuple[dict[str, str], ...]] = {}
    for source in sources:
        if source.dataset is None or source.stage is None:
            raise DevelopmentCohortError("query artifact lacks stage or corpus")
        development_stage = _STAGE_SPECS[source.stage][0]
        stratum = selection.selection(source.dataset, development_stage)
        expected = {row.query_id: row.query_text_sha256 for row in stratum.selected_families}
        observed: dict[str, dict[str, str]] = {}
        for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
            row = _closed(value, _QUERY_FIELDS, label=f"query {source.path}:{line_number}")
            query_id = _require_text("query id", row["id"])
            text = row["text"]
            if query_id not in expected:
                continue
            if not isinstance(text, str) or _sha256(text.encode("utf-8")) != expected[query_id]:
                raise DevelopmentCohortError("selected query text differs from its receipt")
            if query_id in observed:
                raise DevelopmentCohortError("selected query is repeated")
            observed[query_id] = {"id": query_id, "text": text}
        if set(observed) != set(expected):
            raise DevelopmentCohortError("query source omits a selected representative")
        result[(development_stage, source.dataset)] = tuple(
            observed[query_id] for query_id in sorted(observed, key=lambda item: item.encode())
        )
    return result


def _plan_for_stratum(
    stratum: DevelopmentStageSelection,
    *,
    selection_receipt_sha256: str,
    embedding: EmbeddingStoreReceipt,
    query_positions: Mapping[str, int],
) -> DevelopmentExecutionPlan:
    trials: list[DevelopmentExecutionTrial] = []
    for family in stratum.selected_families:
        for nested_index in range(NESTED_ROWS_PER_FAMILY):
            source_value = nested_trial_source_value(family.query_id, nested_index)
            trials.append(
                DevelopmentExecutionTrial(
                    family_key=family.component_sha256,
                    trial_key=_hash_parts(
                        DEVELOPMENT_TRIAL_DOMAIN,
                        stratum.corpus,
                        stratum.development_stage,
                        family.component_sha256,
                        source_value,
                    ),
                    query_id=family.query_id,
                    query_row=query_positions[family.query_id],
                    nested_index=nested_index,
                )
            )
    document_order = embedding.row_orders["documents"].row_order_sha256
    return DevelopmentExecutionPlan(
        corpus=stratum.corpus,
        stage=stratum.development_stage,
        document_count=embedding.document_count,
        document_universe_sha256=document_order,
        document_row_order_sha256=document_order,
        query_row_order_sha256=embedding.row_orders["queries"].row_order_sha256,
        embedding_receipt_sha256=embedding.receipt_sha256,
        selection_receipt_sha256=selection_receipt_sha256,
        selected_family_count=stratum.requested_family_count,
        trials=tuple(sorted(trials, key=lambda row: (row.family_key, row.nested_index))),
    )


def _label_sources(
    sources: Sequence[SourceArtifactPin],
    audit: ScalableQueryPartitionAuditReceipt,
) -> dict[tuple[str, str, str], SourceArtifactPin]:
    audit_by_path = {row.path: row for row in audit.source_artifacts}
    selected: dict[tuple[str, str, str], SourceArtifactPin] = {}
    for source_stage in _STAGE_SPECS:
        development_stage = _STAGE_SPECS[source_stage][0]
        for corpus in FIXED_CORPORA:
            roles = ["qrels"]
            if corpus in EVIDENCE_CORPORA:
                roles.append("evidence-bundles")
            for role in roles:
                expected_path = f"datasets/{corpus}/{source_stage}/{role}.jsonl"
                matches = [
                    row
                    for row in sources
                    if row.path == expected_path
                    and row.dataset == corpus
                    and row.stage == source_stage
                    and row.role == role
                    and row.visibility == "online"
                ]
                if len(matches) != 1 or audit_by_path.get(expected_path) != matches[0]:
                    raise DevelopmentCohortError(
                        f"label source {development_stage}:{corpus}:{role} differs from the audit"
                    )
                selected[(development_stage, corpus, role)] = matches[0]
    return selected


def _materialize_qrels(
    root_descriptor: int,
    source: SourceArtifactPin,
    *,
    selected_query_ids: set[str],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    positive: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
        row = _closed(value, _QREL_FIELDS, label=f"qrel {source.path}:{line_number}")
        query_id = _require_text("qrel query_id", row["query_id"])
        document_id = _require_text("qrel document_id", row["document_id"])
        relevance = row["relevance"]
        if isinstance(relevance, bool) or not isinstance(relevance, int):
            raise DevelopmentCohortError("qrel relevance must be an integer")
        if query_id not in selected_query_ids:
            continue
        key = (query_id, document_id)
        if key in seen:
            raise DevelopmentCohortError("selected qrels repeat a query-document pair")
        seen.add(key)
        rows.append({"document_id": document_id, "query_id": query_id, "relevance": relevance})
        if relevance > 0:
            positive.add(query_id)
    if positive != selected_query_ids:
        raise DevelopmentCohortError("every selected representative needs a positive qrel")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                str(row["query_id"]).encode(),
                str(row["document_id"]).encode(),
            ),
        )
    )


def _materialize_evidence(
    root_descriptor: int,
    source: SourceArtifactPin,
    *,
    selected_query_ids: set[str],
) -> tuple[dict[str, object], ...]:
    rows: dict[str, dict[str, object]] = {}
    for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
        row = _closed(value, _EVIDENCE_FIELDS, label=f"evidence {source.path}:{line_number}")
        query_id = _require_text("evidence query_id", row["query_id"])
        if query_id not in selected_query_ids:
            continue
        if query_id in rows:
            raise DevelopmentCohortError("selected evidence repeats a query")
        rows[query_id] = dict(row)
    if set(rows) != selected_query_ids:
        raise DevelopmentCohortError("selected evidence coverage differs")
    return tuple(rows[query_id] for query_id in sorted(rows, key=lambda item: item.encode()))


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise DevelopmentCohortError("materialized JSONL cannot be empty")
    return b"".join(_canonical_bytes(dict(row)) for row in rows)


def _artifact(
    path: str,
    payload: bytes,
    *,
    record_count: int,
    role: str,
    corpus: str | None,
    stage: str | None,
) -> MaterializedDevelopmentArtifact:
    return MaterializedDevelopmentArtifact(
        path=path,
        sha256=_sha256(payload),
        byte_count=len(payload),
        record_count=record_count,
        role=role,
        corpus=corpus,
        stage=stage,
    )


def _exclusive_publish_directory(work: Path, output: Path) -> None:
    if os.path.lexists(output):
        raise DevelopmentCohortError("materialization output already exists")
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise DevelopmentCohortError("exclusive directory rename is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-2, os.fsencode(work), -2, os.fsencode(output), 0x00000004)
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise DevelopmentCohortError("exclusive directory rename is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, os.fsencode(work), -100, os.fsencode(output), 0x00000001)
    else:
        raise DevelopmentCohortError(
            f"exclusive directory rename is unsupported on {sys.platform!r}"
        )
    if result != 0:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise DevelopmentCohortError("materialization output already exists")
        raise DevelopmentCohortError(
            f"cannot publish materialization package: {os.strerror(number)}"
        )


def materialize_development_cohort(
    staged_root: str | Path,
    selection_receipt_path: str | Path,
    output_root: str | Path,
    *,
    selection_receipt_sha256: str,
    partition_audit_path: str | Path,
    embedding_bindings: Sequence[DevelopmentEmbeddingBinding],
) -> DevelopmentCohortMaterializationReceipt:
    """Materialize labels after reproducing the payload-excluded receipt."""

    output = _require_absolute_path("materialization output root", output_root)
    if os.path.lexists(output):
        raise DevelopmentCohortError("materialization output already exists")
    output_parent = _ensure_real_directory(
        output.parent,
        label="materialization output parent",
    )
    os.close(output_parent)
    expected_selection_sha256 = _require_sha256(
        "selection_receipt_sha256", selection_receipt_sha256
    )
    selection = load_development_cohort_selection(
        selection_receipt_path,
        expected_artifact_sha256=expected_selection_sha256,
    )

    # This full reproduction is the gate. No label-bearing source is selected
    # or opened before these exact payload-excluded bytes match.
    reproduced = _derive_selection_receipt(
        staged_root,
        staged_inventory_sha256=selection.staged_inventory_sha256,
        partition_audit_path=partition_audit_path,
        partition_audit_sha256=selection.partition_audit_sha256,
    )
    if reproduced.canonical_file_bytes() != selection.canonical_file_bytes():
        raise DevelopmentCohortError("selection receipt does not reproduce before materialization")

    audit = load_scalable_partition_audit(
        partition_audit_path,
        expected_artifact_sha256=selection.partition_audit_sha256,
        expected_inventory_sha256=selection.staged_inventory_sha256,
    )
    root = _require_absolute_path("staged root", staged_root)
    root_descriptor = _open_root(root, label="development staged root")
    try:
        inventory, sources = _inventory_sources(
            root_descriptor,
            expected_inventory_sha256=selection.staged_inventory_sha256,
        )
        _assignment_source, query_sources = _select_label_free_sources(sources, audit)
        query_payloads = _selected_query_payloads(root_descriptor, query_sources, selection)
        counts = inventory["counts"]
        if not isinstance(counts, Mapping):
            raise DevelopmentCohortError("staged inventory counts must be an object")
        ordered_bindings, verified_embeddings = _verify_embedding_bindings(
            embedding_bindings,
            selection,
            inventory_counts=counts,
        )

        # Label source resolution and reads occur only below this line.
        label_sources = _label_sources(sources, audit)
        payloads: dict[str, bytes] = {"selection-receipt.json": selection.canonical_file_bytes()}
        artifacts: list[MaterializedDevelopmentArtifact] = [
            _artifact(
                "selection-receipt.json",
                payloads["selection-receipt.json"],
                record_count=1,
                role="development-cohort-selection",
                corpus=None,
                stage=None,
            )
        ]
        for stratum in selection.selections:
            key = (stratum.development_stage, stratum.corpus)
            relative_root = f"{stratum.development_stage}/{stratum.corpus}"
            queries = query_payloads[key]
            query_path = f"{relative_root}/queries.jsonl"
            query_bytes = _jsonl_bytes(queries)
            payloads[query_path] = query_bytes
            artifacts.append(
                _artifact(
                    query_path,
                    query_bytes,
                    record_count=len(queries),
                    role="queries",
                    corpus=stratum.corpus,
                    stage=stratum.development_stage,
                )
            )
            selected_ids = set(stratum.selected_query_ids)
            qrels = _materialize_qrels(
                root_descriptor,
                label_sources[(stratum.development_stage, stratum.corpus, "qrels")],
                selected_query_ids=selected_ids,
            )
            qrel_path = f"{relative_root}/qrels.jsonl"
            qrel_bytes = _jsonl_bytes(qrels)
            payloads[qrel_path] = qrel_bytes
            artifacts.append(
                _artifact(
                    qrel_path,
                    qrel_bytes,
                    record_count=len(qrels),
                    role="qrels",
                    corpus=stratum.corpus,
                    stage=stratum.development_stage,
                )
            )
            if stratum.corpus in EVIDENCE_CORPORA:
                evidence = _materialize_evidence(
                    root_descriptor,
                    label_sources[(stratum.development_stage, stratum.corpus, "evidence-bundles")],
                    selected_query_ids=selected_ids,
                )
                evidence_path = f"{relative_root}/evidence-bundles.jsonl"
                evidence_bytes = _jsonl_bytes(evidence)
                payloads[evidence_path] = evidence_bytes
                artifacts.append(
                    _artifact(
                        evidence_path,
                        evidence_bytes,
                        record_count=len(evidence),
                        role="evidence-bundles",
                        corpus=stratum.corpus,
                        stage=stratum.development_stage,
                    )
                )
            embedding, positions = verified_embeddings[key]
            plan = _plan_for_stratum(
                stratum,
                selection_receipt_sha256=selection.artifact_sha256,
                embedding=embedding,
                query_positions=positions,
            )
            plan_path = f"{relative_root}/execution-plan.json"
            plan_bytes = plan.canonical_file_bytes()
            payloads[plan_path] = plan_bytes
            artifacts.append(
                _artifact(
                    plan_path,
                    plan_bytes,
                    record_count=1,
                    role="development-execution-plan",
                    corpus=stratum.corpus,
                    stage=stratum.development_stage,
                )
            )
    finally:
        os.close(root_descriptor)

    receipt = DevelopmentCohortMaterializationReceipt(
        staged_inventory_sha256=selection.staged_inventory_sha256,
        partition_audit_sha256=selection.partition_audit_sha256,
        selection_receipt_sha256=selection.artifact_sha256,
        artifacts=tuple(artifacts),
        embedding_bindings=ordered_bindings,
    )
    payloads["materialization-receipt.json"] = receipt.canonical_file_bytes()
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for relative_path, payload in sorted(payloads.items()):
            target = temporary.joinpath(*PurePosixPath(relative_path).parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_exclusive_file(target, payload)
        descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _exclusive_publish_directory(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verified = verify_materialized_development_cohort(
        output,
        expected_receipt_sha256=receipt.artifact_sha256,
        verify_label_payloads=True,
    )
    if verified != receipt:
        raise DevelopmentCohortError("published materialization receipt differs")
    return receipt


def load_development_execution_plan(path: str | Path) -> DevelopmentExecutionPlan:
    """Load one canonical plan suitable for ``compile_policy_intervention``."""

    plan_path = _require_absolute_path("development execution plan path", path)
    encoded = read_secure_regular_file(
        plan_path,
        max_bytes=_MAX_CONTROL_BYTES,
        label="development execution plan",
    )
    value = _decode(encoded, label="development execution plan")
    plan = DevelopmentExecutionPlan.from_dict(value)
    if encoded != plan.canonical_file_bytes():
        raise DevelopmentCohortError("development execution plan is not canonical")
    return plan


def load_development_cohort_materialization(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
) -> DevelopmentCohortMaterializationReceipt:
    """Load one exact canonical materialization receipt file."""

    receipt_path = _require_absolute_path("development materialization receipt", path)
    encoded = read_secure_regular_file(
        receipt_path,
        max_bytes=_MAX_CONTROL_BYTES,
        label="development materialization receipt",
    )
    value = _decode(encoded, label="development materialization receipt")
    receipt = DevelopmentCohortMaterializationReceipt.from_dict(value)
    if encoded != receipt.canonical_file_bytes():
        raise DevelopmentCohortError("materialization receipt is not canonical")
    if expected_artifact_sha256 is not None and receipt.artifact_sha256 != _require_sha256(
        "expected materialization SHA-256", expected_artifact_sha256
    ):
        raise DevelopmentCohortError("materialization receipt digest differs")
    return receipt


def verify_materialized_development_cohort(
    root: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
    verify_label_payloads: bool = False,
) -> DevelopmentCohortMaterializationReceipt:
    """Verify package closure without opening labels unless explicitly requested."""

    if type(verify_label_payloads) is not bool:
        raise DevelopmentCohortError("verify_label_payloads must be boolean")

    package = _require_absolute_path("development materialization root", root)
    try:
        package_descriptor = _open_root(package, label="development materialization root")
    except Exception as exc:
        raise DevelopmentCohortError(
            "cannot securely open development materialization root; "
            f"a symlink or non-directory ancestor may be present: {exc}"
        ) from exc
    else:
        os.close(package_descriptor)
    try:
        metadata = package.lstat()
    except OSError as exc:
        raise DevelopmentCohortError(f"cannot inspect materialization package: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise DevelopmentCohortError("materialization root must be a real directory")
    receipt_path = package / "materialization-receipt.json"
    receipt = load_development_cohort_materialization(
        receipt_path,
        expected_artifact_sha256=expected_receipt_sha256,
    )
    observed: set[str] = set()
    for path in package.rglob("*"):
        relative = path.relative_to(package).as_posix()
        path_metadata = path.lstat()
        if stat.S_ISLNK(path_metadata.st_mode):
            raise DevelopmentCohortError("materialization package contains a symbolic link")
        if stat.S_ISREG(path_metadata.st_mode):
            observed.add(relative)
        elif not stat.S_ISDIR(path_metadata.st_mode):
            raise DevelopmentCohortError("materialization package contains a special file")
    expected = {row.path for row in receipt.artifacts} | {"materialization-receipt.json"}
    if observed != expected:
        raise DevelopmentCohortError(
            "materialization package membership differs; "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )
    selection_artifact = next(
        row for row in receipt.artifacts if row.role == "development-cohort-selection"
    )
    if selection_artifact.sha256 != receipt.selection_receipt_sha256:
        raise DevelopmentCohortError("materialization does not directly pin its selection")
    selection = load_development_cohort_selection(
        package / selection_artifact.path,
        expected_artifact_sha256=receipt.selection_receipt_sha256,
        expected_inventory_sha256=receipt.staged_inventory_sha256,
    )
    if selection.partition_audit_sha256 != receipt.partition_audit_sha256:
        raise DevelopmentCohortError("materialization selection names another partition audit")
    bindings = {(row.development_stage, row.corpus): row for row in receipt.embedding_bindings}
    for artifact in receipt.artifacts:
        artifact_path = package.joinpath(*PurePosixPath(artifact.path).parts)
        artifact_metadata = artifact_path.lstat()
        if (
            not stat.S_ISREG(artifact_metadata.st_mode)
            or stat.S_ISLNK(artifact_metadata.st_mode)
            or artifact_metadata.st_nlink != 1
            or artifact_metadata.st_size != artifact.byte_count
        ):
            raise DevelopmentCohortError(
                f"materialized artifact {artifact.path!r} file contract differs"
            )
        if artifact.role in {"qrels", "evidence-bundles"} and not verify_label_payloads:
            continue
        payload = read_secure_regular_file(
            artifact_path,
            max_bytes=max(artifact.byte_count, 1),
            label=artifact.path,
        )
        if len(payload) != artifact.byte_count or _sha256(payload) != artifact.sha256:
            raise DevelopmentCohortError(f"materialized artifact {artifact.path!r} differs")
        if artifact.role in {"queries", "qrels", "evidence-bundles"}:
            rows = _decode_canonical_jsonl(payload, label=artifact.path)
            if len(rows) != artifact.record_count:
                raise DevelopmentCohortError(
                    f"materialized artifact {artifact.path!r} record count differs"
                )
            if artifact.corpus is None or artifact.stage is None:
                raise DevelopmentCohortError("materialized row artifact lacks its stratum")
            stratum = selection.selection(artifact.corpus, artifact.stage)
            selected_ids = set(stratum.selected_query_ids)
            selected_text_sha256 = {
                family.query_id: family.query_text_sha256 for family in stratum.selected_families
            }
            observed_ids: list[str] = []
            positive_ids: set[str] = set()
            observed_qrel_pairs: list[tuple[str, str]] = []
            for position, value in enumerate(rows, start=1):
                if artifact.role == "queries":
                    row = _closed(value, _QUERY_FIELDS, label=f"{artifact.path}:{position}")
                    query_id = _require_text("materialized query ID", row["id"])
                    text = row["text"]
                    expected_text = selected_text_sha256.get(query_id)
                    if (
                        not isinstance(text, str)
                        or expected_text is None
                        or _sha256(text.encode("utf-8")) != expected_text
                    ):
                        raise DevelopmentCohortError(
                            "materialized query differs from the selection receipt"
                        )
                elif artifact.role == "qrels":
                    row = _closed(value, _QREL_FIELDS, label=f"{artifact.path}:{position}")
                    query_id = _require_text("materialized qrel query ID", row["query_id"])
                    document_id = _require_text("materialized qrel document ID", row["document_id"])
                    observed_qrel_pairs.append((query_id, document_id))
                    relevance = row["relevance"]
                    if isinstance(relevance, bool) or not isinstance(relevance, int):
                        raise DevelopmentCohortError("materialized qrel relevance must be integer")
                    if relevance > 0:
                        positive_ids.add(query_id)
                else:
                    row = _closed(value, _EVIDENCE_FIELDS, label=f"{artifact.path}:{position}")
                    query_id = _require_text("materialized evidence query ID", row["query_id"])
                if query_id not in selected_ids:
                    raise DevelopmentCohortError(
                        "materialized labels or queries name an unselected query"
                    )
                observed_ids.append(query_id)
            if artifact.role in {"queries", "evidence-bundles"} and (
                set(observed_ids) != selected_ids or len(observed_ids) != len(selected_ids)
            ):
                raise DevelopmentCohortError(
                    f"materialized {artifact.role} coverage differs from the selection"
                )
            if artifact.role in {"queries", "evidence-bundles"} and observed_ids != sorted(
                observed_ids, key=lambda item: item.encode("utf-8")
            ):
                raise DevelopmentCohortError(
                    f"materialized {artifact.role} rows are not canonically ordered"
                )
            if artifact.role == "qrels" and positive_ids != selected_ids:
                raise DevelopmentCohortError(
                    "materialized qrels lack a positive row for a selected query"
                )
            if artifact.role == "qrels" and (
                len(observed_qrel_pairs) != len(set(observed_qrel_pairs))
                or observed_qrel_pairs
                != sorted(
                    observed_qrel_pairs,
                    key=lambda item: (item[0].encode("utf-8"), item[1].encode("utf-8")),
                )
            ):
                raise DevelopmentCohortError(
                    "materialized qrels are repeated or not canonically ordered"
                )
        elif artifact.role == "development-execution-plan":
            plan = load_development_execution_plan(artifact_path)
            if (
                plan.artifact_sha256 != artifact.sha256
                or plan.corpus != artifact.corpus
                or plan.stage != artifact.stage
                or plan.selection_receipt_sha256 != receipt.selection_receipt_sha256
            ):
                raise DevelopmentCohortError("materialized execution plan binding differs")
            if plan.embedding_receipt_sha256 != bindings[(plan.stage, plan.corpus)].receipt_sha256:
                raise DevelopmentCohortError("development plan names another embedding receipt")
            stratum = selection.selection(plan.corpus, plan.stage)
            expected_families = {
                row.component_sha256: row.query_id for row in stratum.selected_families
            }
            observed_families: dict[str, str] = {}
            for trial in plan.trials:
                expected_query = expected_families.get(trial.family_key)
                if expected_query is None or trial.query_id != expected_query:
                    raise DevelopmentCohortError(
                        "development plan differs from the selected representatives"
                    )
                observed_families[trial.family_key] = trial.query_id
            if observed_families != expected_families:
                raise DevelopmentCohortError(
                    "development plan family coverage differs from the selection"
                )
    return receipt


def verify_development_cohort_materialization(
    root: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
) -> DevelopmentCohortMaterializationReceipt:
    """Compatibility wrapper that verifies all payload bytes, including labels."""

    return verify_materialized_development_cohort(
        root,
        expected_receipt_sha256=expected_artifact_sha256,
        verify_label_payloads=True,
    )


def load_development_embedding_bindings(
    path: str | Path,
) -> tuple[DevelopmentEmbeddingBinding, ...]:
    """Load the closed ten-store materialization input config."""

    config_path = _require_absolute_path("embedding bindings config", path)
    encoded = read_secure_regular_file(
        config_path,
        max_bytes=_MAX_CONTROL_BYTES,
        label="development embedding bindings",
    )
    value = _decode(encoded, label="development embedding bindings")
    row = _closed(
        value,
        frozenset({"bindings", "schema_version"}),
        label="development embedding bindings",
    )
    if row["schema_version"] != DEVELOPMENT_EMBEDDING_CONFIG_SCHEMA:
        raise DevelopmentCohortError("embedding bindings schema differs")
    values = row["bindings"]
    if not isinstance(values, list):
        raise DevelopmentCohortError("embedding bindings must be an array")
    bindings = tuple(DevelopmentEmbeddingBinding.from_dict(item) for item in values)
    canonical = _canonical_bytes(
        {
            "bindings": [
                item.to_dict()
                for item in sorted(
                    bindings,
                    key=lambda binding: (binding.development_stage, binding.corpus),
                )
            ],
            "schema_version": DEVELOPMENT_EMBEDDING_CONFIG_SCHEMA,
        }
    )
    if encoded != canonical:
        raise DevelopmentCohortError("embedding bindings config is not canonical")
    return bindings


def canonical_development_embedding_bindings_bytes(
    bindings: Sequence[DevelopmentEmbeddingBinding],
) -> bytes:
    """Serialize the closed ten-store materialization input config."""

    ordered = tuple(sorted(bindings, key=lambda row: (row.development_stage, row.corpus)))
    expected = {
        (stage, corpus) for stage in _DEVELOPMENT_TO_SOURCE_STAGE for corpus in FIXED_CORPORA
    }
    if (
        len(ordered) != len(expected)
        or {(row.development_stage, row.corpus) for row in ordered} != expected
    ):
        raise DevelopmentCohortError("embedding bindings must cover the fixed ten strata")
    return _canonical_bytes(
        {
            "bindings": [row.to_dict() for row in ordered],
            "schema_version": DEVELOPMENT_EMBEDDING_CONFIG_SCHEMA,
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-development-cohort",
        description=(
            "Select from frozen qrel-derived components without opening label payloads, "
            "or materialize the fixed development cohort."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--staged-root", required=True, type=Path)
    select.add_argument("--staged-inventory-sha256", required=True)
    select.add_argument("--partition-audit", required=True, type=Path)
    select.add_argument("--partition-audit-sha256", required=True)
    select.add_argument("--output", required=True, type=Path)

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--staged-root", required=True, type=Path)
    materialize.add_argument("--selection-receipt", required=True, type=Path)
    materialize.add_argument("--selection-receipt-sha256", required=True)
    materialize.add_argument("--partition-audit", required=True, type=Path)
    materialize.add_argument("--embedding-bindings", required=True, type=Path)
    materialize.add_argument("--output-root", required=True, type=Path)

    verify_selection = commands.add_parser("verify-selection")
    verify_selection.add_argument("--receipt", required=True, type=Path)
    verify_selection.add_argument("--expected-sha256", required=True)
    verify_selection.add_argument("--expected-inventory-sha256")

    verify_materialization = commands.add_parser("verify-materialization")
    verify_materialization.add_argument("--root", required=True, type=Path)
    verify_materialization.add_argument("--expected-sha256", required=True)
    verify_materialization.add_argument("--verify-label-payloads", action="store_true")
    return parser


def _write_cli_result(command: str, receipt: object) -> None:
    if isinstance(receipt, DevelopmentCohortSelectionReceipt):
        result = {
            "artifact_sha256": receipt.artifact_sha256,
            "command": command,
            "selected_family_count": sum(row.requested_family_count for row in receipt.selections),
            "staged_inventory_sha256": receipt.staged_inventory_sha256,
        }
    elif isinstance(receipt, DevelopmentCohortMaterializationReceipt):
        result = {
            "artifact_count": len(receipt.artifacts),
            "artifact_sha256": receipt.artifact_sha256,
            "command": command,
            "selection_receipt_sha256": receipt.selection_receipt_sha256,
            "staged_inventory_sha256": receipt.staged_inventory_sha256,
        }
    else:
        raise TypeError("CLI result receipt type differs")
    result["schema_version"] = DEVELOPMENT_COHORT_CLI_RESULT_SCHEMA
    sys.stdout.buffer.write(_canonical_bytes(result))
    sys.stdout.buffer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "select":
            receipt: object = select_development_cohort(
                arguments.staged_root,
                arguments.output,
                staged_inventory_sha256=arguments.staged_inventory_sha256,
                partition_audit_path=arguments.partition_audit,
                partition_audit_sha256=arguments.partition_audit_sha256,
            )
        elif arguments.command == "materialize":
            receipt = materialize_development_cohort(
                arguments.staged_root,
                arguments.selection_receipt,
                arguments.output_root,
                selection_receipt_sha256=arguments.selection_receipt_sha256,
                partition_audit_path=arguments.partition_audit,
                embedding_bindings=load_development_embedding_bindings(
                    arguments.embedding_bindings
                ),
            )
        elif arguments.command == "verify-selection":
            receipt = load_development_cohort_selection(
                arguments.receipt,
                expected_artifact_sha256=arguments.expected_sha256,
                expected_inventory_sha256=arguments.expected_inventory_sha256,
            )
        else:
            receipt = verify_materialized_development_cohort(
                arguments.root,
                expected_receipt_sha256=arguments.expected_sha256,
                verify_label_payloads=arguments.verify_label_payloads,
            )
        _write_cli_result(arguments.command, receipt)
        return 0
    except (DevelopmentCohortError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"development-cohort: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
