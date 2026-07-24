"""Label-free query/trial assembly for sharded sealed execution.

The builder creates the small query/trial artifact needed by a
``ShardedOnlineExecutionPlan``.  The admission path later joins that plan to a
verified embedding store and a frozen policy schedule.  Runtime blocks are
loaded one at a time, so repeated policy assignments never require a
schedule-sized vector map.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from .artifact_integrity import (
    ArtifactIntegrityError,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .embedding_store import (
    EmbeddingStoreError,
    EmbeddingStoreReceipt,
    verify_embedding_store,
)
from .label_separation import OnlineTrial
from .online_runner import FrozenFeatureContext, OnlineTrialRuntime
from .policy_intervention import (
    NESTED_TRIALS_PER_FAMILY,
    TRIAL_STATE_ASSIGNMENT_ALGORITHM,
    CanonicalTrialSchedule,
    PolicyInterventionError,
    TrialScheduleRow,
    load_canonical_trial_schedule,
)
from .query_cohort import (
    FAMILY_SELECTION_ALGORITHM,
    NESTED_ROWS_PER_FAMILY,
    REPRESENTATIVE_SELECTION_ALGORITHM,
    family_selection_rank,
    nested_trial_source_value,
    representative_selection_rank,
)
from .scalable_execution import (
    QUERY_TRIAL_STORE_FORMAT,
    ImmutableArtifactPin,
    OpaqueTrialRow,
    QueryTrialStoreDescriptor,
    ScalableExecutionError,
    ShardedOnlineExecutionPlan,
    load_sharded_online_execution_plan,
)
from .scalable_partition_audit import (
    ScalablePartitionAuditError,
    ScalableQueryPartitionAuditReceipt,
    load_scalable_partition_audit,
)

QUERY_TRIAL_ROW_SCHEMA = "fractal-query-trial-row-v1"
QUERY_TRIAL_RECEIPT_SCHEMA = "fractal-query-trial-receipt-v3"
TRIAL_RUNTIME_RECEIPT_SCHEMA = "fractal-trial-runtime-admission-v3"
OPAQUE_KEY_ALGORITHM = "fractal-label-separation-v2"
GROUP_ORDER_ALGORITHM = "schedule-partitioned-state-blocks-v2"
QUERY_TRIAL_FILENAME = "query-trials.jsonl"
QUERY_TRIAL_RECEIPT_FILENAME = "query-trial-receipt.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_MAX_LINE_BYTES = 16 * 1024 * 1024
_FORBIDDEN_FIELD_PARTS = (
    "answer",
    "evidence",
    "gold",
    "judgment",
    "label",
    "qrel",
    "relevance",
    "supporting_fact",
)
_SOURCE_BINDING_FIELDS = frozenset(
    {
        "active_query_row_order_sha256",
        "current_truth_query_row_order_sha256",
        "embedding_query_row",
        "source_file_sha256",
        "source_path",
        "source_query_id_sha256",
        "source_record_sha256",
        "source_row",
    }
)
_QUERY_VECTOR_EPOCH_FIELDS = frozenset(
    {
        "dtype",
        "file_sha256",
        "model_revision",
        "model_tree_sha256",
        "prompt_sha256",
        "role",
        "row_order_sha256",
        "shape",
    }
)
_QUERY_TRIAL_FIELDS = frozenset(
    {
        "corpus",
        "family_key",
        "query_row",
        "schema_version",
        "source",
        "stage",
        "text",
        "trial_key",
    }
)
_QUERY_TRIAL_RECEIPT_FIELDS = frozenset(
    {
        "available_family_count",
        "assignment_store_sha256",
        "active_query_epoch",
        "embedding_store_receipt_sha256",
        "hmac_key_id",
        "family_selection_algorithm",
        "key_derivation_algorithm",
        "nested_rows_per_family",
        "opaque_trials",
        "query_trial_store_byte_count",
        "query_trial_store_format",
        "query_trial_store_sha256",
        "query_partition_audit_sha256",
        "record_count",
        "representative_selection_algorithm",
        "schema_version",
        "selected_family_count",
        "selection_seed_sha256",
        "source_inventory_sha256",
        "stage",
        "staged_inventory_sha256",
        "corpus",
        "current_truth_query_epoch",
    }
)
_FEATURE_BINDING_FIELDS = frozenset(
    {
        "backend",
        "drift_family",
        "group_order",
        "policy_complexity",
        "policy_state",
        "repetition",
        "subject",
        "version_lag",
    }
)
_RUNTIME_GROUP_FIELDS = frozenset(
    {
        "authorized_count",
        "block_order",
        "environment_sha256",
        "expected_policy_revision",
        "group_order",
        "mask_id",
        "mask_sha256",
        "policy_state",
        "realized_allow_rate",
        "repetition",
        "schedule_rows_sha256",
        "subject",
    }
)
_RUNTIME_RECEIPT_FIELDS = frozenset(
    {
        "active_query_epoch",
        "assignment_map_sha256",
        "assignment_seed_sha256",
        "embedding_store_receipt_sha256",
        "execution_artifact_sha256",
        "feature_bindings_sha256",
        "group_order_algorithm",
        "groups",
        "mask_catalog_sha256",
        "policy_config_sha256",
        "policy_bundle_revision",
        "permutation_seed",
        "query_count",
        "query_partition_audit_sha256",
        "query_trial_store_sha256",
        "schedule_sha256",
        "schema_version",
        "source_inventory_sha256",
        "staged_inventory_sha256",
        "trial_state_assignment_algorithm",
        "current_truth_query_epoch",
    }
)
_ROW_ORDER_FIELDS = frozenset({"dataset", "id", "kind", "source_path", "source_row", "stage"})
_SOURCE_INVENTORY_FIELDS = frozenset(
    {"documents", "queries", "schema_version", "staged_inventory_sha256"}
)
_SOURCE_DESCRIPTOR_FIELDS = frozenset(
    {"byte_count", "dataset", "kind", "path", "record_count", "sha256", "stage"}
)
_STAGED_ARTIFACT_FIELDS = frozenset(
    {
        "byte_count",
        "dataset",
        "path",
        "record_count",
        "role",
        "sha256",
        "stage",
        "visibility",
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


class TrialRuntimeError(RuntimeError):
    """Raised when a query/trial source or runtime block cannot be admitted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrialRuntimeError("trial runtime data must be finite canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _closed_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TrialRuntimeError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise TrialRuntimeError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _decode_json(encoded: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise TrialRuntimeError(f"{label} repeats key {key!r}")
            parsed[key] = value
        return parsed

    def reject_nonfinite(value: str) -> None:
        raise TrialRuntimeError(f"{label} contains non-finite number {value!r}")

    try:
        return json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise TrialRuntimeError(f"{label} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise TrialRuntimeError(f"{label} is not valid JSON: {exc.msg}") from exc


def _canonical_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise TrialRuntimeError(f"{label} needs exactly one terminal newline")
    value = _decode_json(encoded[:-1], label=label)
    if not isinstance(value, Mapping) or encoded != _canonical_bytes(value) + b"\n":
        raise TrialRuntimeError(f"{label} is not canonical JSON")
    return value


def _canonical_jsonl(encoded: bytes, *, label: str) -> list[Mapping[str, Any]]:
    if not encoded or not encoded.endswith(b"\n"):
        raise TrialRuntimeError(f"{label} must be non-empty canonical JSONL")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(encoded.splitlines(keepends=True), start=1):
        if len(line) > _MAX_LINE_BYTES or not line.endswith(b"\n") or line == b"\n":
            raise TrialRuntimeError(f"{label} line {line_number} is not a bounded record")
        value = _decode_json(line[:-1], label=f"{label} line {line_number}")
        if not isinstance(value, Mapping) or line != _canonical_bytes(value) + b"\n":
            raise TrialRuntimeError(f"{label} line {line_number} is not canonical JSON")
        rows.append(value)
    return rows


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrialRuntimeError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise TrialRuntimeError(f"{name} must be a canonical non-empty string")
    return value


def _require_integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrialRuntimeError(f"{name} must be an integer of at least {minimum}")
    return value


def _require_rate(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrialRuntimeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise TrialRuntimeError(f"{name} must be finite and in [0, 1]")
    return result


def _relative_path(name: str, value: object) -> str:
    text = _require_text(name, value)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise TrialRuntimeError(f"{name} must be a canonical relative POSIX path")
    return text


def _assert_label_free(value: object, *, path: str = "runtime") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold()
            if any(part in normalized for part in _FORBIDDEN_FIELD_PARTS):
                raise TrialRuntimeError(f"outcome-bearing field leaked into {path}: {key!r}")
            _assert_label_free(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, nested in enumerate(value):
            _assert_label_free(nested, path=f"{path}[{position}]")


def _read(path: Path, *, expected_bytes: int, label: str) -> bytes:
    if expected_bytes < 1:
        raise TrialRuntimeError(f"{label} expected byte count must be positive")
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=expected_bytes,
            label=label,
        )
    except ArtifactIntegrityError as exc:
        raise TrialRuntimeError(f"cannot read {label} safely: {exc}") from exc
    if len(encoded) != expected_bytes:
        raise TrialRuntimeError(f"{label} byte count differs from its descriptor")
    return encoded


def _bounded_control_size(path: Path, *, label: str) -> int:
    try:
        size = path.lstat().st_size
    except OSError as exc:
        raise TrialRuntimeError(f"cannot inspect {label}: {exc}") from exc
    if size < 1 or size > _MAX_CONTROL_BYTES:
        raise TrialRuntimeError(f"{label} must be between 1 and {_MAX_CONTROL_BYTES} bytes")
    return size


@dataclass(frozen=True)
class QuerySourceBinding:
    """Exact staged row and embedding-row identity for one online query."""

    embedding_query_row: int
    active_query_row_order_sha256: str
    current_truth_query_row_order_sha256: str
    source_path: str
    source_row: int
    source_file_sha256: str
    source_record_sha256: str
    source_query_id_sha256: str

    def __post_init__(self) -> None:
        _require_integer("embedding_query_row", self.embedding_query_row)
        _require_sha256(
            "active_query_row_order_sha256",
            self.active_query_row_order_sha256,
        )
        _require_sha256(
            "current_truth_query_row_order_sha256",
            self.current_truth_query_row_order_sha256,
        )
        object.__setattr__(
            self,
            "source_path",
            _relative_path("source_path", self.source_path),
        )
        _require_integer("source_row", self.source_row, minimum=1)
        _require_sha256("source_file_sha256", self.source_file_sha256)
        _require_sha256("source_record_sha256", self.source_record_sha256)
        _require_sha256("source_query_id_sha256", self.source_query_id_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "active_query_row_order_sha256": self.active_query_row_order_sha256,
            "current_truth_query_row_order_sha256": (self.current_truth_query_row_order_sha256),
            "embedding_query_row": self.embedding_query_row,
            "source_file_sha256": self.source_file_sha256,
            "source_path": self.source_path,
            "source_query_id_sha256": self.source_query_id_sha256,
            "source_record_sha256": self.source_record_sha256,
            "source_row": self.source_row,
        }

    @classmethod
    def from_dict(cls, value: object) -> QuerySourceBinding:
        row = _closed_mapping(value, fields=_SOURCE_BINDING_FIELDS, label="query source")
        return cls(
            embedding_query_row=row["embedding_query_row"],
            active_query_row_order_sha256=row["active_query_row_order_sha256"],
            current_truth_query_row_order_sha256=row["current_truth_query_row_order_sha256"],
            source_path=row["source_path"],
            source_row=row["source_row"],
            source_file_sha256=row["source_file_sha256"],
            source_record_sha256=row["source_record_sha256"],
            source_query_id_sha256=row["source_query_id_sha256"],
        )


@dataclass(frozen=True)
class QueryVectorEpochBinding:
    """One query-vector epoch with its independent model and row-order pins."""

    role: str
    file_sha256: str
    row_order_sha256: str
    model_tree_sha256: str
    model_revision: str
    prompt_sha256: str
    dtype: str
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.role not in {"active-migration", "current-exact-truth"}:
            raise TrialRuntimeError("query vector epoch has an unsupported role")
        for name in (
            "file_sha256",
            "row_order_sha256",
            "model_tree_sha256",
            "prompt_sha256",
        ):
            _require_sha256(f"query epoch {name}", getattr(self, name))
        _require_text("query epoch model_revision", self.model_revision)
        if self.dtype not in {"float16", "float32"}:
            raise TrialRuntimeError("query epoch dtype must be float16 or float32")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.shape
            )
        ):
            raise TrialRuntimeError("query epoch shape must contain two positive integers")

    def to_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "file_sha256": self.file_sha256,
            "model_revision": self.model_revision,
            "model_tree_sha256": self.model_tree_sha256,
            "prompt_sha256": self.prompt_sha256,
            "role": self.role,
            "row_order_sha256": self.row_order_sha256,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: object) -> QueryVectorEpochBinding:
        row = _closed_mapping(
            value,
            fields=_QUERY_VECTOR_EPOCH_FIELDS,
            label="query vector epoch",
        )
        shape = row["shape"]
        if not isinstance(shape, list) or len(shape) != 2:
            raise TrialRuntimeError("query vector epoch shape must be a two-item array")
        return cls(
            role=row["role"],
            file_sha256=row["file_sha256"],
            row_order_sha256=row["row_order_sha256"],
            model_tree_sha256=row["model_tree_sha256"],
            model_revision=row["model_revision"],
            prompt_sha256=row["prompt_sha256"],
            dtype=row["dtype"],
            shape=(shape[0], shape[1]),
        )


@dataclass(frozen=True)
class CanonicalQueryTrialRow:
    """One label-free query record pinned by an opaque plan row."""

    trial_key: str
    family_key: str
    query_row: int
    text: str
    corpus: str
    stage: str
    source: QuerySourceBinding
    schema_version: str = QUERY_TRIAL_ROW_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("trial_key", self.trial_key)
        _require_sha256("family_key", self.family_key)
        _require_integer("query_row", self.query_row)
        _require_text("query text", self.text)
        _require_text("query corpus", self.corpus)
        _require_text("query stage", self.stage)
        if not isinstance(self.source, QuerySourceBinding):
            raise TrialRuntimeError("source must be a QuerySourceBinding")
        if self.schema_version != QUERY_TRIAL_ROW_SCHEMA:
            raise TrialRuntimeError(f"schema_version must equal {QUERY_TRIAL_ROW_SCHEMA!r}")
        _assert_label_free(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "family_key": self.family_key,
            "query_row": self.query_row,
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "stage": self.stage,
            "text": self.text,
            "trial_key": self.trial_key,
        }

    def canonical_line(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def query_record_sha256(self) -> str:
        return _sha256(self.canonical_line())

    @property
    def opaque_row(self) -> OpaqueTrialRow:
        return OpaqueTrialRow(
            trial_key=self.trial_key,
            family_key=self.family_key,
            query_row=self.query_row,
            query_record_sha256=self.query_record_sha256,
        )

    @classmethod
    def from_dict(cls, value: object) -> CanonicalQueryTrialRow:
        row = _closed_mapping(value, fields=_QUERY_TRIAL_FIELDS, label="query/trial row")
        return cls(
            trial_key=row["trial_key"],
            family_key=row["family_key"],
            query_row=row["query_row"],
            text=row["text"],
            corpus=row["corpus"],
            stage=row["stage"],
            source=QuerySourceBinding.from_dict(row["source"]),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class QueryTrialStoreReceipt:
    """Pinned build evidence for one canonical query/trial JSONL file."""

    hmac_key_id: str
    corpus: str
    stage: str
    staged_inventory_sha256: str
    source_inventory_sha256: str
    assignment_store_sha256: str
    query_partition_audit_sha256: str
    selection_seed_sha256: str
    available_family_count: int
    selected_family_count: int
    nested_rows_per_family: int
    embedding_store_receipt_sha256: str
    active_query_epoch: QueryVectorEpochBinding
    current_truth_query_epoch: QueryVectorEpochBinding
    query_trial_store_sha256: str
    query_trial_store_byte_count: int
    record_count: int
    opaque_trials: tuple[OpaqueTrialRow, ...]
    family_selection_algorithm: str = FAMILY_SELECTION_ALGORITHM
    representative_selection_algorithm: str = REPRESENTATIVE_SELECTION_ALGORITHM
    key_derivation_algorithm: str = OPAQUE_KEY_ALGORITHM
    query_trial_store_format: str = QUERY_TRIAL_STORE_FORMAT
    schema_version: str = QUERY_TRIAL_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("hmac_key_id", "corpus", "stage"):
            _require_text(name, getattr(self, name))
        for name in (
            "staged_inventory_sha256",
            "source_inventory_sha256",
            "assignment_store_sha256",
            "query_partition_audit_sha256",
            "selection_seed_sha256",
            "embedding_store_receipt_sha256",
            "query_trial_store_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            not isinstance(self.active_query_epoch, QueryVectorEpochBinding)
            or self.active_query_epoch.role != "active-migration"
            or not isinstance(self.current_truth_query_epoch, QueryVectorEpochBinding)
            or self.current_truth_query_epoch.role != "current-exact-truth"
        ):
            raise TrialRuntimeError("query/trial receipt needs both typed vector epochs")
        if (
            self.active_query_epoch.shape != self.current_truth_query_epoch.shape
            or self.active_query_epoch.dtype != self.current_truth_query_epoch.dtype
            or self.active_query_epoch.row_order_sha256
            != self.current_truth_query_epoch.row_order_sha256
            or (
                self.active_query_epoch.model_tree_sha256,
                self.active_query_epoch.model_revision,
            )
            == (
                self.current_truth_query_epoch.model_tree_sha256,
                self.current_truth_query_epoch.model_revision,
            )
            or self.active_query_epoch.file_sha256 == self.current_truth_query_epoch.file_sha256
        ):
            raise TrialRuntimeError(
                "active and current-truth query epochs need aligned rows and distinct models"
            )
        _require_integer(
            "query_trial_store_byte_count",
            self.query_trial_store_byte_count,
            minimum=1,
        )
        _require_integer("record_count", self.record_count, minimum=1)
        _require_integer(
            "available_family_count",
            self.available_family_count,
            minimum=1,
        )
        _require_integer(
            "selected_family_count",
            self.selected_family_count,
            minimum=1,
        )
        if self.selected_family_count > self.available_family_count:
            raise TrialRuntimeError("selected_family_count cannot exceed available_family_count")
        if self.nested_rows_per_family != NESTED_ROWS_PER_FAMILY:
            raise TrialRuntimeError(f"nested_rows_per_family must equal {NESTED_ROWS_PER_FAMILY}")
        if self.record_count != self.selected_family_count * self.nested_rows_per_family:
            raise TrialRuntimeError("record_count must equal selected families times nested rows")
        trials = tuple(self.opaque_trials)
        if len(trials) != self.record_count or not all(
            isinstance(row, OpaqueTrialRow) for row in trials
        ):
            raise TrialRuntimeError("opaque_trials must cover record_count")
        if [row.query_row for row in trials] != list(range(self.record_count)):
            raise TrialRuntimeError("opaque trial query rows must be contiguous and ordered")
        if len({row.trial_key for row in trials}) != len(trials):
            raise TrialRuntimeError("opaque_trials repeat a trial key")
        family_sequence = [row.family_key for row in trials]
        family_blocks = [
            family_key
            for position, family_key in enumerate(family_sequence)
            if position == 0 or family_key != family_sequence[position - 1]
        ]
        if (
            len(set(family_sequence)) != self.selected_family_count
            or len(family_blocks) != self.selected_family_count
            or any(
                family_sequence.count(family_key) != self.nested_rows_per_family
                for family_key in family_blocks
            )
        ):
            raise TrialRuntimeError(
                "opaque trials must form one exact nested block per selected family"
            )
        object.__setattr__(self, "opaque_trials", trials)
        if self.family_selection_algorithm != FAMILY_SELECTION_ALGORITHM:
            raise TrialRuntimeError("family selection algorithm differs")
        if self.representative_selection_algorithm != REPRESENTATIVE_SELECTION_ALGORITHM:
            raise TrialRuntimeError("representative selection algorithm differs")
        if self.key_derivation_algorithm != OPAQUE_KEY_ALGORITHM:
            raise TrialRuntimeError("key derivation algorithm differs")
        if self.query_trial_store_format != QUERY_TRIAL_STORE_FORMAT:
            raise TrialRuntimeError("query/trial store format differs")
        if self.schema_version != QUERY_TRIAL_RECEIPT_SCHEMA:
            raise TrialRuntimeError("query/trial receipt schema differs")
        _assert_label_free(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "active_query_epoch": self.active_query_epoch.to_dict(),
            "available_family_count": self.available_family_count,
            "assignment_store_sha256": self.assignment_store_sha256,
            "corpus": self.corpus,
            "current_truth_query_epoch": self.current_truth_query_epoch.to_dict(),
            "embedding_store_receipt_sha256": self.embedding_store_receipt_sha256,
            "family_selection_algorithm": self.family_selection_algorithm,
            "hmac_key_id": self.hmac_key_id,
            "key_derivation_algorithm": self.key_derivation_algorithm,
            "nested_rows_per_family": self.nested_rows_per_family,
            "opaque_trials": [row.to_dict() for row in self.opaque_trials],
            "query_trial_store_byte_count": self.query_trial_store_byte_count,
            "query_trial_store_format": self.query_trial_store_format,
            "query_trial_store_sha256": self.query_trial_store_sha256,
            "query_partition_audit_sha256": self.query_partition_audit_sha256,
            "record_count": self.record_count,
            "representative_selection_algorithm": (self.representative_selection_algorithm),
            "schema_version": self.schema_version,
            "selected_family_count": self.selected_family_count,
            "selection_seed_sha256": self.selection_seed_sha256,
            "source_inventory_sha256": self.source_inventory_sha256,
            "stage": self.stage,
            "staged_inventory_sha256": self.staged_inventory_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @property
    def receipt_byte_count(self) -> int:
        return len(self.canonical_file_bytes())

    def store_descriptor(
        self,
        *,
        artifact_id: str,
        relative_path: str,
        receipt_artifact_id: str,
        receipt_relative_path: str,
    ) -> QueryTrialStoreDescriptor:
        return QueryTrialStoreDescriptor(
            artifact=ImmutableArtifactPin(
                artifact_id=artifact_id,
                relative_path=relative_path,
                kind="file",
                byte_count=self.query_trial_store_byte_count,
                sha256=self.query_trial_store_sha256,
            ),
            receipt=ImmutableArtifactPin(
                artifact_id=receipt_artifact_id,
                relative_path=receipt_relative_path,
                kind="file",
                byte_count=self.receipt_byte_count,
                sha256=self.receipt_sha256,
            ),
            record_count=self.record_count,
        )

    @classmethod
    def from_dict(cls, value: object) -> QueryTrialStoreReceipt:
        row = _closed_mapping(
            value,
            fields=_QUERY_TRIAL_RECEIPT_FIELDS,
            label="query/trial receipt",
        )
        trials = row["opaque_trials"]
        if not isinstance(trials, list):
            raise TrialRuntimeError("opaque_trials must be an array")
        return cls(
            hmac_key_id=row["hmac_key_id"],
            corpus=row["corpus"],
            stage=row["stage"],
            staged_inventory_sha256=row["staged_inventory_sha256"],
            source_inventory_sha256=row["source_inventory_sha256"],
            assignment_store_sha256=row["assignment_store_sha256"],
            query_partition_audit_sha256=row["query_partition_audit_sha256"],
            selection_seed_sha256=row["selection_seed_sha256"],
            available_family_count=row["available_family_count"],
            selected_family_count=row["selected_family_count"],
            nested_rows_per_family=row["nested_rows_per_family"],
            embedding_store_receipt_sha256=row["embedding_store_receipt_sha256"],
            active_query_epoch=QueryVectorEpochBinding.from_dict(row["active_query_epoch"]),
            current_truth_query_epoch=QueryVectorEpochBinding.from_dict(
                row["current_truth_query_epoch"]
            ),
            query_trial_store_sha256=row["query_trial_store_sha256"],
            query_trial_store_byte_count=row["query_trial_store_byte_count"],
            record_count=row["record_count"],
            opaque_trials=tuple(OpaqueTrialRow.from_dict(item) for item in trials),
            family_selection_algorithm=row["family_selection_algorithm"],
            representative_selection_algorithm=(row["representative_selection_algorithm"]),
            key_derivation_algorithm=row["key_derivation_algorithm"],
            query_trial_store_format=row["query_trial_store_format"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class RuntimeFeatureBinding:
    """One frozen context shared by every query in one schedule block."""

    group_order: int
    subject: str
    repetition: int
    policy_state: str
    version_lag: float
    backend: str
    drift_family: str
    policy_complexity: float

    def __post_init__(self) -> None:
        _require_integer("feature group_order", self.group_order)
        _require_text("feature subject", self.subject)
        _require_integer("feature repetition", self.repetition)
        _require_text("feature policy_state", self.policy_state)
        context = self.context
        object.__setattr__(self, "version_lag", context.version_lag)
        object.__setattr__(self, "policy_complexity", context.policy_complexity)

    @property
    def block_key(self) -> tuple[int, str, int, str]:
        return (self.group_order, self.subject, self.repetition, self.policy_state)

    @property
    def context(self) -> FrozenFeatureContext:
        return FrozenFeatureContext(
            version_lag=self.version_lag,
            backend=self.backend,
            drift_family=self.drift_family,
            policy_complexity=self.policy_complexity,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "drift_family": self.drift_family,
            "group_order": self.group_order,
            "policy_complexity": self.policy_complexity,
            "policy_state": self.policy_state,
            "repetition": self.repetition,
            "subject": self.subject,
            "version_lag": self.version_lag,
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimeFeatureBinding:
        row = _closed_mapping(value, fields=_FEATURE_BINDING_FIELDS, label="feature binding")
        return cls(
            group_order=row["group_order"],
            subject=row["subject"],
            repetition=row["repetition"],
            policy_state=row["policy_state"],
            version_lag=row["version_lag"],
            backend=row["backend"],
            drift_family=row["drift_family"],
            policy_complexity=row["policy_complexity"],
        )


@dataclass(frozen=True)
class RuntimeGroupDescriptor:
    """Deterministic block order, independent of within-trial action order."""

    block_order: int
    group_order: int
    subject: str
    repetition: int
    policy_state: str
    environment_sha256: str
    mask_id: str
    mask_sha256: str
    authorized_count: int
    realized_allow_rate: float
    expected_policy_revision: str
    schedule_rows_sha256: str

    def __post_init__(self) -> None:
        _require_integer("block_order", self.block_order)
        _require_integer("group_order", self.group_order)
        _require_text("group subject", self.subject)
        _require_integer("group repetition", self.repetition)
        _require_text("group policy_state", self.policy_state)
        _require_sha256("group environment_sha256", self.environment_sha256)
        _require_text("group mask_id", self.mask_id)
        _require_sha256("group mask_sha256", self.mask_sha256)
        _require_integer("group authorized_count", self.authorized_count, minimum=1)
        object.__setattr__(
            self,
            "realized_allow_rate",
            _require_rate("group realized_allow_rate", self.realized_allow_rate),
        )
        _require_text("group expected_policy_revision", self.expected_policy_revision)
        _require_sha256("group schedule_rows_sha256", self.schedule_rows_sha256)

    @property
    def block_key(self) -> tuple[int, str, int, str]:
        return (self.group_order, self.subject, self.repetition, self.policy_state)

    def to_dict(self) -> dict[str, object]:
        return {
            "authorized_count": self.authorized_count,
            "block_order": self.block_order,
            "environment_sha256": self.environment_sha256,
            "expected_policy_revision": self.expected_policy_revision,
            "group_order": self.group_order,
            "mask_id": self.mask_id,
            "mask_sha256": self.mask_sha256,
            "policy_state": self.policy_state,
            "realized_allow_rate": self.realized_allow_rate,
            "repetition": self.repetition,
            "schedule_rows_sha256": self.schedule_rows_sha256,
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimeGroupDescriptor:
        row = _closed_mapping(value, fields=_RUNTIME_GROUP_FIELDS, label="runtime group")
        return cls(**row)


@dataclass(frozen=True)
class TrialRuntimeAdmissionReceipt:
    """Closed evidence for a plan, schedule, query source, and feature join."""

    execution_artifact_sha256: str
    query_trial_store_sha256: str
    query_partition_audit_sha256: str
    schedule_sha256: str
    staged_inventory_sha256: str
    source_inventory_sha256: str
    embedding_store_receipt_sha256: str
    active_query_epoch: QueryVectorEpochBinding
    current_truth_query_epoch: QueryVectorEpochBinding
    policy_bundle_revision: str
    policy_config_sha256: str
    mask_catalog_sha256: str
    feature_bindings_sha256: str
    assignment_seed_sha256: str
    assignment_map_sha256: str
    permutation_seed: int
    query_count: int
    groups: tuple[RuntimeGroupDescriptor, ...]
    trial_state_assignment_algorithm: str = TRIAL_STATE_ASSIGNMENT_ALGORITHM
    group_order_algorithm: str = GROUP_ORDER_ALGORITHM
    schema_version: str = TRIAL_RUNTIME_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "execution_artifact_sha256",
            "query_trial_store_sha256",
            "query_partition_audit_sha256",
            "schedule_sha256",
            "staged_inventory_sha256",
            "source_inventory_sha256",
            "embedding_store_receipt_sha256",
            "policy_config_sha256",
            "mask_catalog_sha256",
            "feature_bindings_sha256",
            "assignment_seed_sha256",
            "assignment_map_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            not isinstance(self.active_query_epoch, QueryVectorEpochBinding)
            or self.active_query_epoch.role != "active-migration"
            or not isinstance(self.current_truth_query_epoch, QueryVectorEpochBinding)
            or self.current_truth_query_epoch.role != "current-exact-truth"
            or self.active_query_epoch.shape != self.current_truth_query_epoch.shape
            or self.active_query_epoch.row_order_sha256
            != self.current_truth_query_epoch.row_order_sha256
        ):
            raise TrialRuntimeError("runtime receipt needs aligned active and truth query epochs")
        _require_text("policy_bundle_revision", self.policy_bundle_revision)
        _require_integer("permutation_seed", self.permutation_seed)
        _require_integer("query_count", self.query_count, minimum=1)
        groups = tuple(self.groups)
        if not groups or not all(isinstance(row, RuntimeGroupDescriptor) for row in groups):
            raise TrialRuntimeError("groups must contain runtime group descriptors")
        if [row.block_order for row in groups] != list(range(len(groups))):
            raise TrialRuntimeError("runtime group block_order must be contiguous")
        if len({row.block_key for row in groups}) != len(groups):
            raise TrialRuntimeError("runtime groups repeat a block key")
        if (
            len(groups) != NESTED_TRIALS_PER_FAMILY
            or tuple(row.group_order for row in groups) != (0, 1, 2)
            or len({row.policy_state for row in groups}) != 3
            or self.query_count % NESTED_TRIALS_PER_FAMILY != 0
        ):
            raise TrialRuntimeError(
                "runtime receipt must bind three policy blocks and nested triples"
            )
        object.__setattr__(self, "groups", groups)
        if self.group_order_algorithm != GROUP_ORDER_ALGORITHM:
            raise TrialRuntimeError("group order algorithm differs")
        if self.trial_state_assignment_algorithm != TRIAL_STATE_ASSIGNMENT_ALGORITHM:
            raise TrialRuntimeError("trial-state assignment algorithm differs")
        if self.schema_version != TRIAL_RUNTIME_RECEIPT_SCHEMA:
            raise TrialRuntimeError("trial runtime receipt schema differs")
        _assert_label_free(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "active_query_epoch": self.active_query_epoch.to_dict(),
            "assignment_map_sha256": self.assignment_map_sha256,
            "assignment_seed_sha256": self.assignment_seed_sha256,
            "current_truth_query_epoch": self.current_truth_query_epoch.to_dict(),
            "embedding_store_receipt_sha256": self.embedding_store_receipt_sha256,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "feature_bindings_sha256": self.feature_bindings_sha256,
            "group_order_algorithm": self.group_order_algorithm,
            "groups": [row.to_dict() for row in self.groups],
            "mask_catalog_sha256": self.mask_catalog_sha256,
            "policy_bundle_revision": self.policy_bundle_revision,
            "policy_config_sha256": self.policy_config_sha256,
            "permutation_seed": self.permutation_seed,
            "query_count": self.query_count,
            "query_partition_audit_sha256": self.query_partition_audit_sha256,
            "query_trial_store_sha256": self.query_trial_store_sha256,
            "schedule_sha256": self.schedule_sha256,
            "schema_version": self.schema_version,
            "source_inventory_sha256": self.source_inventory_sha256,
            "staged_inventory_sha256": self.staged_inventory_sha256,
            "trial_state_assignment_algorithm": self.trial_state_assignment_algorithm,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> TrialRuntimeAdmissionReceipt:
        row = _closed_mapping(value, fields=_RUNTIME_RECEIPT_FIELDS, label="runtime receipt")
        groups = row["groups"]
        if not isinstance(groups, list):
            raise TrialRuntimeError("runtime receipt groups must be an array")
        return cls(
            execution_artifact_sha256=row["execution_artifact_sha256"],
            query_trial_store_sha256=row["query_trial_store_sha256"],
            query_partition_audit_sha256=row["query_partition_audit_sha256"],
            schedule_sha256=row["schedule_sha256"],
            staged_inventory_sha256=row["staged_inventory_sha256"],
            source_inventory_sha256=row["source_inventory_sha256"],
            embedding_store_receipt_sha256=row["embedding_store_receipt_sha256"],
            active_query_epoch=QueryVectorEpochBinding.from_dict(row["active_query_epoch"]),
            current_truth_query_epoch=QueryVectorEpochBinding.from_dict(
                row["current_truth_query_epoch"]
            ),
            policy_bundle_revision=row["policy_bundle_revision"],
            policy_config_sha256=row["policy_config_sha256"],
            mask_catalog_sha256=row["mask_catalog_sha256"],
            feature_bindings_sha256=row["feature_bindings_sha256"],
            assignment_seed_sha256=row["assignment_seed_sha256"],
            assignment_map_sha256=row["assignment_map_sha256"],
            permutation_seed=row["permutation_seed"],
            query_count=row["query_count"],
            groups=tuple(RuntimeGroupDescriptor.from_dict(item) for item in groups),
            trial_state_assignment_algorithm=row["trial_state_assignment_algorithm"],
            group_order_algorithm=row["group_order_algorithm"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class ShardedQueryExecutionAdapter:
    """Query-bearing full or partitioned view that preserves the plan digest."""

    plan: ShardedOnlineExecutionPlan
    trials: tuple[OnlineTrial, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ShardedOnlineExecutionPlan):
            raise TrialRuntimeError("plan must be a ShardedOnlineExecutionPlan")
        trials = tuple(sorted(self.trials, key=lambda row: row.trial_key))
        if not trials or not all(isinstance(row, OnlineTrial) for row in trials):
            raise TrialRuntimeError("adapter trials must contain OnlineTrial values")
        plan_families = {row.trial_key: row.family_key for row in self.plan.trials}
        trial_keys = tuple(row.trial_key for row in trials)
        if (
            len(trial_keys) != len(set(trial_keys))
            or not set(trial_keys).issubset(plan_families)
            or any(row.family_key != plan_families[row.trial_key] for row in trials)
        ):
            raise TrialRuntimeError("adapter trials differ from the sharded plan")
        if any(row.corpus != self.plan.corpus or row.stage != self.plan.stage for row in trials):
            raise TrialRuntimeError("adapter trial corpus or stage differs")
        object.__setattr__(self, "trials", trials)

    @property
    def artifact_sha256(self) -> str:
        return self.plan.artifact_sha256

    @property
    def corpus(self) -> str:
        return self.plan.corpus

    @property
    def stage(self) -> str:
        return self.plan.stage

    @property
    def document_count(self) -> int:
        return self.plan.document_count

    @property
    def document_universe_sha256(self) -> str:
        return self.plan.document_universe_sha256

    @property
    def trial_keys(self) -> tuple[str, ...]:
        return tuple(row.trial_key for row in self.trials)

    def canonical_bytes(self) -> bytes:
        return self.plan.canonical_bytes()


@dataclass(frozen=True)
class LoadedRuntimeBlock:
    """One disjoint policy block ready for verified combination."""

    descriptor: RuntimeGroupDescriptor
    execution: ShardedQueryExecutionAdapter
    trial_runtimes: Mapping[str, OnlineTrialRuntime]

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, RuntimeGroupDescriptor):
            raise TrialRuntimeError("descriptor must be a RuntimeGroupDescriptor")
        if not isinstance(self.execution, ShardedQueryExecutionAdapter):
            raise TrialRuntimeError("execution must be a ShardedQueryExecutionAdapter")
        runtimes = dict(self.trial_runtimes)
        if set(runtimes) != set(self.execution.trial_keys) or not all(
            isinstance(value, OnlineTrialRuntime) for value in runtimes.values()
        ):
            raise TrialRuntimeError("runtime mapping must cover the exact query set")
        if len(runtimes) > len(self.execution.trials):
            raise TrialRuntimeError("runtime mapping exceeds the query count")
        object.__setattr__(self, "trial_runtimes", MappingProxyType(runtimes))


@dataclass(frozen=True)
class LoadedTrialRuntime:
    """All disjoint policy blocks combined for one online action-matrix call."""

    descriptors: tuple[RuntimeGroupDescriptor, ...]
    execution: ShardedQueryExecutionAdapter
    trial_runtimes: Mapping[str, OnlineTrialRuntime]

    def __post_init__(self) -> None:
        descriptors = tuple(self.descriptors)
        if not descriptors or not all(
            isinstance(row, RuntimeGroupDescriptor) for row in descriptors
        ):
            raise TrialRuntimeError("descriptors must contain runtime group descriptors")
        if [row.block_order for row in descriptors] != list(range(len(descriptors))):
            raise TrialRuntimeError("runtime descriptors are not canonical")
        if not isinstance(self.execution, ShardedQueryExecutionAdapter):
            raise TrialRuntimeError("execution must be a ShardedQueryExecutionAdapter")
        runtimes = dict(self.trial_runtimes)
        if set(runtimes) != set(self.execution.plan.trial_keys) or set(
            self.execution.trial_keys
        ) != set(self.execution.plan.trial_keys):
            raise TrialRuntimeError("combined runtime must cover every plan trial exactly once")
        if not all(isinstance(value, OnlineTrialRuntime) for value in runtimes.values()):
            raise TrialRuntimeError("combined runtime mapping contains an invalid value")
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "trial_runtimes", MappingProxyType(runtimes))


@dataclass(frozen=True)
class TrialRuntimeAdmission:
    """Verified paths and frozen metadata for lazy block loading."""

    plan: ShardedOnlineExecutionPlan
    partition_audit_path: Path
    query_package_root: Path
    staged_root: Path
    embedding_store_root: Path
    schedule_path: Path
    feature_bindings: tuple[RuntimeFeatureBinding, ...]
    receipt: TrialRuntimeAdmissionReceipt


@dataclass(frozen=True)
class _QueryMaterial:
    embedding_row: int
    query_id: str
    text: str
    source_path: str
    source_row: int
    source_file_sha256: str
    source_record_sha256: str
    component_sha256: str


@dataclass(frozen=True)
class _SourceJoin:
    embedding_receipt: EmbeddingStoreReceipt
    partition_audit: ScalableQueryPartitionAuditReceipt
    materials: tuple[_QueryMaterial, ...]
    assignment_store_sha256: str


@dataclass(frozen=True)
class _NestedQueryMaterial:
    material: _QueryMaterial
    nested_index: int


def _query_epoch_binding(
    receipt: EmbeddingStoreReceipt,
    *,
    matrix: str,
    role: str,
) -> QueryVectorEpochBinding:
    descriptor = receipt.vectors.get(matrix)
    if descriptor is None:
        raise TrialRuntimeError(f"embedding store lacks required matrix {matrix!r}")
    return QueryVectorEpochBinding(
        role=role,
        file_sha256=descriptor.file_sha256,
        row_order_sha256=descriptor.row_order_sha256,
        model_tree_sha256=descriptor.model_tree_sha256,
        model_revision=descriptor.model_revision,
        prompt_sha256=descriptor.prompt_sha256,
        dtype=descriptor.dtype,
        shape=descriptor.shape,
    )


def _validate_secret(secret: bytes) -> bytes:
    if type(secret) is not bytes or len(secret) < 32 or len(set(secret)) < 8:
        raise TrialRuntimeError("HMAC secret must be immutable, diverse, and at least 32 bytes")
    return secret


def _derive_opaque_key(
    secret: bytes,
    *,
    domain: str,
    key_id: str,
    corpus: str,
    stage: str,
    source_value: str,
) -> str:
    digest = hmac.new(secret, digestmod=hashlib.sha256)
    for value in (
        OPAQUE_KEY_ALGORITHM,
        domain,
        key_id,
        corpus,
        stage,
        source_value,
    ):
        encoded = value.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _secure_binding_file(
    path: Path,
    *,
    expected_logical_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    encoded = _read(
        path,
        expected_bytes=_bounded_control_size(path, label=label),
        label=label,
    )
    value = _canonical_object(encoded, label=label)
    if _sha256(encoded[:-1]) != expected_logical_sha256:
        raise TrialRuntimeError(f"{label} differs from its logical digest")
    return value


def _load_source_join(
    staged_root: Path,
    embedding_store_root: Path,
    partition_audit_path: Path,
    *,
    corpus: str,
    stage: str,
    expected_partition_audit_sha256: str | None = None,
) -> _SourceJoin:
    _require_text("corpus", corpus)
    if stage != "sealed":
        raise TrialRuntimeError("trial runtime stage must equal 'sealed'")
    try:
        embedding_receipt = verify_embedding_store(embedding_store_root)
    except (EmbeddingStoreError, OSError) as exc:
        raise TrialRuntimeError(f"embedding store verification failed: {exc}") from exc
    try:
        partition_audit = load_scalable_partition_audit(
            partition_audit_path,
            expected_artifact_sha256=expected_partition_audit_sha256,
            expected_inventory_sha256=embedding_receipt.staged_inventory_sha256,
        )
    except (ScalablePartitionAuditError, OSError) as exc:
        raise TrialRuntimeError(f"typed query-partition audit verification failed: {exc}") from exc
    active_epoch = _query_epoch_binding(
        embedding_receipt,
        matrix="old_queries",
        role="active-migration",
    )
    truth_epoch = _query_epoch_binding(
        embedding_receipt,
        matrix="current_queries",
        role="current-exact-truth",
    )
    if (
        active_epoch.shape != truth_epoch.shape
        or active_epoch.dtype != truth_epoch.dtype
        or active_epoch.row_order_sha256 != truth_epoch.row_order_sha256
        or (active_epoch.model_tree_sha256, active_epoch.model_revision)
        == (truth_epoch.model_tree_sha256, truth_epoch.model_revision)
        or active_epoch.file_sha256 == truth_epoch.file_sha256
    ):
        raise TrialRuntimeError(
            "embedding store needs aligned query rows from two distinct model epochs"
        )

    source_inventory = _secure_binding_file(
        embedding_store_root / "source-inventory.json",
        expected_logical_sha256=embedding_receipt.source_inventory_sha256,
        label="embedding source inventory",
    )
    source_inventory = _closed_mapping(
        source_inventory,
        fields=_SOURCE_INVENTORY_FIELDS,
        label="embedding source inventory",
    )
    if source_inventory["staged_inventory_sha256"] != embedding_receipt.staged_inventory_sha256:
        raise TrialRuntimeError("embedding source inventory names another staged inventory")

    staged_inventory_path = staged_root / "inventory.json"
    staged_encoded = _read(
        staged_inventory_path,
        expected_bytes=_bounded_control_size(
            staged_inventory_path,
            label="staged inventory",
        ),
        label="staged inventory",
    )
    if _sha256(staged_encoded) != embedding_receipt.staged_inventory_sha256:
        raise TrialRuntimeError("staged inventory differs from the embedding receipt")
    staged_inventory = _canonical_object(staged_encoded, label="staged inventory")
    artifacts = staged_inventory.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TrialRuntimeError("staged inventory needs an artifacts array")
    staged_by_path: dict[str, Mapping[str, Any]] = {}
    for position, value in enumerate(artifacts):
        row = _closed_mapping(
            value,
            fields=_STAGED_ARTIFACT_FIELDS,
            label=f"staged artifact {position}",
        )
        path = _relative_path("staged artifact path", row["path"])
        if path in staged_by_path:
            raise TrialRuntimeError("staged inventory repeats an artifact path")
        staged_by_path[path] = row
    for source in partition_audit.source_artifacts:
        staged_source = staged_by_path.get(source.path)
        if staged_source is None or dict(staged_source) != source.to_dict():
            raise TrialRuntimeError(
                "typed query-partition audit source pins differ from staged inventory"
            )

    query_values = source_inventory["queries"]
    if not isinstance(query_values, list) or not query_values:
        raise TrialRuntimeError("embedding source inventory needs query sources")
    query_sources: dict[str, Mapping[str, Any]] = {}
    for position, value in enumerate(query_values):
        source = _closed_mapping(
            value,
            fields=_SOURCE_DESCRIPTOR_FIELDS,
            label=f"embedding query source {position}",
        )
        path = _relative_path("embedding query source path", source["path"])
        if source["kind"] != "queries" or path in query_sources:
            raise TrialRuntimeError("embedding query sources are invalid or repeated")
        staged = staged_by_path.get(path)
        if staged is None or (
            staged["role"] != "queries"
            or staged["visibility"] != "online"
            or staged["dataset"] != source["dataset"]
            or staged["stage"] != source["stage"]
            or staged["sha256"] != source["sha256"]
            or staged["byte_count"] != source["byte_count"]
            or staged["record_count"] != source["record_count"]
        ):
            raise TrialRuntimeError("embedding query source differs from staged inventory")
        query_sources[path] = source

    row_order_descriptor = embedding_receipt.row_orders["queries"]
    row_order_encoded = _read(
        embedding_store_root / row_order_descriptor.relative_path,
        expected_bytes=row_order_descriptor.byte_count,
        label="embedding query row order",
    )
    if _sha256(row_order_encoded) != row_order_descriptor.file_sha256:
        raise TrialRuntimeError("embedding query row order digest differs")
    row_order_values = _canonical_jsonl(row_order_encoded, label="embedding query row order")
    if len(row_order_values) != row_order_descriptor.row_count:
        raise TrialRuntimeError("embedding query row order count differs")

    source_rows: dict[str, list[Mapping[str, Any]]] = {}
    for path, source in query_sources.items():
        if source["dataset"] != corpus or source["stage"] != stage:
            continue
        byte_count = _require_integer(f"{path} byte_count", source["byte_count"], minimum=1)
        encoded = _read(
            staged_root.joinpath(*PurePosixPath(path).parts),
            expected_bytes=byte_count,
            label=f"staged query source {path}",
        )
        if _sha256(encoded) != _require_sha256(f"{path} sha256", source["sha256"]):
            raise TrialRuntimeError(f"staged query source {path!r} digest differs")
        values = _canonical_jsonl(encoded, label=f"staged query source {path}")
        if len(values) != source["record_count"]:
            raise TrialRuntimeError(f"staged query source {path!r} count differs")
        for line_number, value in enumerate(values, start=1):
            if set(value) != {"id", "text"} or not all(
                isinstance(item, str) and item for item in value.values()
            ):
                raise TrialRuntimeError(
                    f"staged query source {path!r} line {line_number} has invalid fields"
                )
            _assert_label_free(value, path=f"query_source[{line_number}]")
        source_rows[path] = values
    if not source_rows:
        raise TrialRuntimeError("embedding store contains no selected sealed query source")

    assignment_rows = [
        row
        for row in staged_by_path.values()
        if row["role"] == "assignments" and row["visibility"] == "online"
    ]
    if len(assignment_rows) != 1:
        raise TrialRuntimeError("staged inventory must name one online assignment store")
    assignment_artifact = assignment_rows[0]
    assignment_path = _relative_path("assignment path", assignment_artifact["path"])
    assignment_encoded = _read(
        staged_root.joinpath(*PurePosixPath(assignment_path).parts),
        expected_bytes=_require_integer(
            "assignment byte_count",
            assignment_artifact["byte_count"],
            minimum=1,
        ),
        label="query assignment store",
    )
    assignment_sha256 = _require_sha256(
        "assignment sha256",
        assignment_artifact["sha256"],
    )
    if _sha256(assignment_encoded) != assignment_sha256:
        raise TrialRuntimeError("query assignment store digest differs")
    assignment_values = _canonical_jsonl(
        assignment_encoded,
        label="query assignment store",
    )
    if len(assignment_values) != assignment_artifact["record_count"]:
        raise TrialRuntimeError("query assignment store count differs")
    assignments: dict[str, Mapping[str, Any]] = {}
    for position, value in enumerate(assignment_values):
        row = _closed_mapping(
            value,
            fields=_ASSIGNMENT_FIELDS,
            label=f"query assignment {position}",
        )
        if row["dataset"] != corpus or row["stage"] != stage:
            continue
        query_id = _require_text("assignment query_id", row["query_id"])
        _require_sha256("assignment query_text_sha256", row["query_text_sha256"])
        _require_sha256(
            "assignment partition_component_sha256",
            row["partition_component_sha256"],
        )
        if query_id in assignments:
            raise TrialRuntimeError("query assignment store repeats a selected query")
        assignments[query_id] = row

    materials: list[_QueryMaterial] = []
    selected_ids: set[str] = set()
    for embedding_row, value in enumerate(row_order_values):
        row = _closed_mapping(
            value,
            fields=_ROW_ORDER_FIELDS,
            label=f"embedding query row {embedding_row}",
        )
        if row["kind"] != "queries":
            raise TrialRuntimeError("query row-order artifact contains a non-query row")
        if row["dataset"] != corpus or row["stage"] != stage:
            continue
        path = _relative_path("query row source_path", row["source_path"])
        values = source_rows.get(path)
        source = query_sources.get(path)
        source_row = _require_integer("query row source_row", row["source_row"], minimum=1)
        if values is None or source is None or source_row > len(values):
            raise TrialRuntimeError("query row order names an unverified source row")
        source_value = values[source_row - 1]
        query_id = _require_text("query row id", row["id"])
        if source_value["id"] != query_id:
            raise TrialRuntimeError("query row order identifier differs from staged source")
        if query_id in selected_ids:
            raise TrialRuntimeError("selected embedding rows repeat a query identifier")
        assignment = assignments.get(query_id)
        text = _require_text("query text", source_value["text"])
        if assignment is None or assignment["query_text_sha256"] != _sha256(text.encode()):
            raise TrialRuntimeError("selected query lacks its exact assignment binding")
        source_line = _canonical_bytes(source_value) + b"\n"
        materials.append(
            _QueryMaterial(
                embedding_row=embedding_row,
                query_id=query_id,
                text=text,
                source_path=path,
                source_row=source_row,
                source_file_sha256=source["sha256"],
                source_record_sha256=_sha256(source_line),
                component_sha256=assignment["partition_component_sha256"],
            )
        )
        selected_ids.add(query_id)
    if not materials or set(assignments) != selected_ids:
        raise TrialRuntimeError("selected query rows and sealed assignments differ")
    selected_audit_counts = [
        row for row in partition_audit.query_counts if row.dataset == corpus and row.stage == stage
    ]
    if (
        partition_audit.assignment_artifact_sha256 != assignment_sha256
        or len(selected_audit_counts) != 1
        or selected_audit_counts[0].query_count != len(materials)
    ):
        raise TrialRuntimeError("typed query-partition audit differs from selected query sources")
    return _SourceJoin(
        embedding_receipt=embedding_receipt,
        partition_audit=partition_audit,
        materials=tuple(materials),
        assignment_store_sha256=assignment_sha256,
    )


def _select_nested_materials(
    source_join: _SourceJoin,
    *,
    corpus: str,
    stage: str,
    selection_seed_sha256: str,
    available_family_count: int,
    selected_family_count: int,
    nested_rows_per_family: int,
) -> tuple[_NestedQueryMaterial, ...]:
    seed = _require_sha256("selection_seed_sha256", selection_seed_sha256)
    available = _require_integer(
        "available_family_count",
        available_family_count,
        minimum=1,
    )
    selected = _require_integer(
        "selected_family_count",
        selected_family_count,
        minimum=1,
    )
    if selected > available:
        raise TrialRuntimeError("selected_family_count cannot exceed available_family_count")
    if nested_rows_per_family != NESTED_ROWS_PER_FAMILY:
        raise TrialRuntimeError(f"nested_rows_per_family must equal {NESTED_ROWS_PER_FAMILY}")
    components = sorted(
        {material.component_sha256 for material in source_join.materials},
        key=lambda value: value.encode("ascii"),
    )
    if len(components) != available:
        raise TrialRuntimeError("available family count differs from the registered cohort")
    ordered_components = tuple(
        sorted(
            components,
            key=lambda component: (
                family_selection_rank(
                    corpus=corpus,
                    stage=stage,
                    selection_seed_sha256=seed,
                    component_sha256=component,
                ),
                component,
            ),
        )[:selected]
    )
    if len(ordered_components) != selected:
        raise TrialRuntimeError("registered family selection underflowed")

    nested: list[_NestedQueryMaterial] = []
    query_digest_sources: dict[str, str] = {}
    for component in ordered_components:
        candidates: list[tuple[str, str, _QueryMaterial]] = []
        for material in source_join.materials:
            if material.component_sha256 != component:
                continue
            query_id_sha256 = _sha256(material.query_id.encode("utf-8", errors="strict"))
            prior_source = query_digest_sources.setdefault(
                query_id_sha256,
                material.query_id,
            )
            if prior_source != material.query_id:
                raise TrialRuntimeError("representative query-ID digest collision")
            candidates.append(
                (
                    representative_selection_rank(
                        corpus=corpus,
                        stage=stage,
                        selection_seed_sha256=seed,
                        component_sha256=component,
                        query_id_sha256=query_id_sha256,
                    ),
                    query_id_sha256,
                    material,
                )
            )
        if not candidates:
            raise TrialRuntimeError("selected family has no embedded query candidate")
        representative = min(candidates, key=lambda row: (row[0], row[1]))[2]
        nested.extend(
            _NestedQueryMaterial(material=representative, nested_index=nested_index)
            for nested_index in range(nested_rows_per_family)
        )
    if len(nested) != selected * nested_rows_per_family:
        raise TrialRuntimeError("nested query cohort has the wrong denominator")
    return tuple(nested)


def _query_rows(
    source_join: _SourceJoin,
    *,
    corpus: str,
    stage: str,
    key_id: str,
    secret: bytes,
    selection_seed_sha256: str,
    available_family_count: int,
    selected_family_count: int,
    nested_rows_per_family: int,
) -> tuple[CanonicalQueryTrialRow, ...]:
    receipt = source_join.embedding_receipt
    active_epoch = _query_epoch_binding(
        receipt,
        matrix="old_queries",
        role="active-migration",
    )
    truth_epoch = _query_epoch_binding(
        receipt,
        matrix="current_queries",
        role="current-exact-truth",
    )
    rows: list[CanonicalQueryTrialRow] = []
    nested_materials = _select_nested_materials(
        source_join,
        corpus=corpus,
        stage=stage,
        selection_seed_sha256=selection_seed_sha256,
        available_family_count=available_family_count,
        selected_family_count=selected_family_count,
        nested_rows_per_family=nested_rows_per_family,
    )
    for query_row, nested_material in enumerate(nested_materials):
        material = nested_material.material
        source_binding = QuerySourceBinding(
            embedding_query_row=material.embedding_row,
            active_query_row_order_sha256=active_epoch.row_order_sha256,
            current_truth_query_row_order_sha256=truth_epoch.row_order_sha256,
            source_path=material.source_path,
            source_row=material.source_row,
            source_file_sha256=material.source_file_sha256,
            source_record_sha256=material.source_record_sha256,
            source_query_id_sha256=_sha256(material.query_id.encode("utf-8")),
        )
        rows.append(
            CanonicalQueryTrialRow(
                trial_key=_derive_opaque_key(
                    secret,
                    domain="trial",
                    key_id=key_id,
                    corpus=corpus,
                    stage=stage,
                    source_value=nested_trial_source_value(
                        material.query_id,
                        nested_material.nested_index,
                    ),
                ),
                family_key=_derive_opaque_key(
                    secret,
                    domain="family",
                    key_id=key_id,
                    corpus=corpus,
                    stage=stage,
                    source_value=material.component_sha256,
                ),
                query_row=query_row,
                text=material.text,
                corpus=corpus,
                stage=stage,
                source=source_binding,
            )
        )
    if len({row.trial_key for row in rows}) != len(rows):
        raise TrialRuntimeError("HMAC derivation produced a repeated trial key")
    return tuple(rows)


def _load_query_rows(
    path: Path, *, expected_bytes: int, expected_sha256: str
) -> tuple[CanonicalQueryTrialRow, ...]:
    encoded = _read(path, expected_bytes=expected_bytes, label="query/trial store")
    if _sha256(encoded) != expected_sha256:
        raise TrialRuntimeError("query/trial store digest differs")
    values = _canonical_jsonl(encoded, label="query/trial store")
    rows = tuple(CanonicalQueryTrialRow.from_dict(value) for value in values)
    if [row.query_row for row in rows] != list(range(len(rows))):
        raise TrialRuntimeError("query/trial rows must be contiguous and ordered")
    return rows


def _load_query_receipt(path: Path) -> QueryTrialStoreReceipt:
    size = _bounded_control_size(path, label="query/trial receipt")
    encoded = _read(path, expected_bytes=size, label="query/trial receipt")
    receipt = QueryTrialStoreReceipt.from_dict(
        _canonical_object(encoded, label="query/trial receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise TrialRuntimeError("query/trial receipt bytes differ from its typed form")
    return receipt


def verify_query_trial_store(
    package_root: str | Path,
    staged_root: str | Path,
    embedding_store_root: str | Path,
    *,
    partition_audit_path: str | Path,
    secret: bytes | None = None,
) -> QueryTrialStoreReceipt:
    """Reverify package membership, every query source, and all declared digests."""

    package = Path(package_root)
    try:
        metadata = package.lstat()
    except OSError as exc:
        raise TrialRuntimeError(f"cannot inspect query/trial package: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise TrialRuntimeError("query/trial package must be a real directory")
    try:
        membership = {child.name for child in package.iterdir()}
    except OSError as exc:
        raise TrialRuntimeError(f"cannot scan query/trial package: {exc}") from exc
    expected = {QUERY_TRIAL_FILENAME, QUERY_TRIAL_RECEIPT_FILENAME}
    if membership != expected:
        raise TrialRuntimeError(
            f"query/trial package membership differs; missing={sorted(expected - membership)}, "
            f"unexpected={sorted(membership - expected)}"
        )
    receipt = _load_query_receipt(package / QUERY_TRIAL_RECEIPT_FILENAME)
    join = _load_source_join(
        Path(staged_root),
        Path(embedding_store_root),
        Path(partition_audit_path),
        corpus=receipt.corpus,
        stage=receipt.stage,
        expected_partition_audit_sha256=(receipt.query_partition_audit_sha256),
    )
    embedding = join.embedding_receipt
    active_epoch = _query_epoch_binding(
        embedding,
        matrix="old_queries",
        role="active-migration",
    )
    truth_epoch = _query_epoch_binding(
        embedding,
        matrix="current_queries",
        role="current-exact-truth",
    )
    if (
        receipt.staged_inventory_sha256 != embedding.staged_inventory_sha256
        or receipt.source_inventory_sha256 != embedding.source_inventory_sha256
        or receipt.assignment_store_sha256 != join.assignment_store_sha256
        or receipt.query_partition_audit_sha256 != join.partition_audit.artifact_sha256
        or receipt.embedding_store_receipt_sha256 != embedding.receipt_sha256
        or receipt.active_query_epoch != active_epoch
        or receipt.current_truth_query_epoch != truth_epoch
    ):
        raise TrialRuntimeError("query/trial receipt differs from its reverified sources")
    rows = _load_query_rows(
        package / QUERY_TRIAL_FILENAME,
        expected_bytes=receipt.query_trial_store_byte_count,
        expected_sha256=receipt.query_trial_store_sha256,
    )
    if len(rows) != receipt.record_count or tuple(row.opaque_row for row in rows) != (
        receipt.opaque_trials
    ):
        raise TrialRuntimeError("query/trial rows differ from the receipt")
    nested_materials = _select_nested_materials(
        join,
        corpus=receipt.corpus,
        stage=receipt.stage,
        selection_seed_sha256=receipt.selection_seed_sha256,
        available_family_count=receipt.available_family_count,
        selected_family_count=receipt.selected_family_count,
        nested_rows_per_family=receipt.nested_rows_per_family,
    )
    for row, nested_material in zip(rows, nested_materials, strict=True):
        material = nested_material.material
        expected_source = QuerySourceBinding(
            embedding_query_row=material.embedding_row,
            active_query_row_order_sha256=receipt.active_query_epoch.row_order_sha256,
            current_truth_query_row_order_sha256=(
                receipt.current_truth_query_epoch.row_order_sha256
            ),
            source_path=material.source_path,
            source_row=material.source_row,
            source_file_sha256=material.source_file_sha256,
            source_record_sha256=material.source_record_sha256,
            source_query_id_sha256=_sha256(material.query_id.encode("utf-8")),
        )
        if (
            row.corpus != receipt.corpus
            or row.stage != receipt.stage
            or row.text != material.text
            or row.source != expected_source
        ):
            raise TrialRuntimeError("query/trial row differs from its staged source")
    if secret is not None:
        expected_rows = _query_rows(
            join,
            corpus=receipt.corpus,
            stage=receipt.stage,
            key_id=receipt.hmac_key_id,
            secret=_validate_secret(secret),
            selection_seed_sha256=receipt.selection_seed_sha256,
            available_family_count=receipt.available_family_count,
            selected_family_count=receipt.selected_family_count,
            nested_rows_per_family=receipt.nested_rows_per_family,
        )
        if rows != expected_rows:
            raise TrialRuntimeError("query/trial opaque keys differ from the supplied HMAC key")
    return receipt


def _exclusive_publish(work: Path, output: Path) -> None:
    if os.path.lexists(output):
        raise TrialRuntimeError("query/trial package already exists and cannot be overwritten")
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(work)
    destination = os.fsencode(output)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise TrialRuntimeError("exclusive directory rename is unavailable on macOS")
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
            raise TrialRuntimeError("exclusive directory rename is unavailable on Linux")
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
        raise TrialRuntimeError(
            f"exclusive directory rename is unsupported on platform {sys.platform!r}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise TrialRuntimeError("query/trial package already exists and cannot be overwritten")
        raise TrialRuntimeError(
            f"cannot publish query/trial package exclusively: {os.strerror(error_number)}"
        )


def build_query_trial_store(
    staged_root: str | Path,
    embedding_store_root: str | Path,
    target_root: str | Path,
    *,
    partition_audit_path: str | Path,
    corpus: str,
    stage: str,
    hmac_key_id: str,
    hmac_secret: bytes,
    selection_seed_sha256: str,
    available_family_count: int,
    selected_family_count: int,
    nested_rows_per_family: int = NESTED_ROWS_PER_FAMILY,
) -> QueryTrialStoreReceipt:
    """Build, self-verify, and publish one immutable query/trial package."""

    key_id = _require_text("hmac_key_id", hmac_key_id)
    secret = _validate_secret(hmac_secret)
    target = Path(target_root)
    if not target.is_absolute() or target.name in {"", ".", ".."}:
        raise TrialRuntimeError("target_root must be an absolute directory path")
    try:
        parent = target.parent.lstat()
    except OSError as exc:
        raise TrialRuntimeError(f"cannot inspect package parent: {exc}") from exc
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or (hasattr(os, "geteuid") and parent.st_uid != os.geteuid())
        or stat.S_IMODE(parent.st_mode) & 0o022
    ):
        raise TrialRuntimeError("package parent must be a private runner-owned directory")
    if os.path.lexists(target):
        raise TrialRuntimeError("target_root already exists and cannot be overwritten")

    join = _load_source_join(
        Path(staged_root),
        Path(embedding_store_root),
        Path(partition_audit_path),
        corpus=corpus,
        stage=stage,
    )
    rows = _query_rows(
        join,
        corpus=corpus,
        stage=stage,
        key_id=key_id,
        secret=secret,
        selection_seed_sha256=selection_seed_sha256,
        available_family_count=available_family_count,
        selected_family_count=selected_family_count,
        nested_rows_per_family=nested_rows_per_family,
    )
    store_payload = b"".join(row.canonical_line() for row in rows)
    embedding = join.embedding_receipt
    active_epoch = _query_epoch_binding(
        embedding,
        matrix="old_queries",
        role="active-migration",
    )
    truth_epoch = _query_epoch_binding(
        embedding,
        matrix="current_queries",
        role="current-exact-truth",
    )
    receipt = QueryTrialStoreReceipt(
        hmac_key_id=key_id,
        corpus=corpus,
        stage=stage,
        staged_inventory_sha256=embedding.staged_inventory_sha256,
        source_inventory_sha256=embedding.source_inventory_sha256,
        assignment_store_sha256=join.assignment_store_sha256,
        query_partition_audit_sha256=join.partition_audit.artifact_sha256,
        selection_seed_sha256=selection_seed_sha256,
        available_family_count=available_family_count,
        selected_family_count=selected_family_count,
        nested_rows_per_family=nested_rows_per_family,
        embedding_store_receipt_sha256=embedding.receipt_sha256,
        active_query_epoch=active_epoch,
        current_truth_query_epoch=truth_epoch,
        query_trial_store_sha256=_sha256(store_payload),
        query_trial_store_byte_count=len(store_payload),
        record_count=len(rows),
        opaque_trials=tuple(row.opaque_row for row in rows),
    )
    staging = target.parent / f".{target.name}.staging-{secrets.token_hex(12)}"
    try:
        staging.mkdir(mode=0o700)
        write_exclusive_receipt_bytes(store_payload, staging / QUERY_TRIAL_FILENAME)
        write_exclusive_receipt_bytes(
            receipt.canonical_file_bytes(),
            staging / QUERY_TRIAL_RECEIPT_FILENAME,
        )
        verify_query_trial_store(
            staging,
            staged_root,
            embedding_store_root,
            partition_audit_path=partition_audit_path,
            secret=secret,
        )
        directory_descriptor = os.open(
            staging,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _exclusive_publish(staging, target)
        parent_descriptor = os.open(
            target.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _coerce_plan(value: ShardedOnlineExecutionPlan | str | Path) -> ShardedOnlineExecutionPlan:
    if isinstance(value, ShardedOnlineExecutionPlan):
        if _sha256(value.canonical_bytes()) != value.artifact_sha256:
            raise TrialRuntimeError("sharded plan digest differs from its canonical bytes")
        return value
    try:
        return load_sharded_online_execution_plan(value)
    except (ScalableExecutionError, TypeError) as exc:
        raise TrialRuntimeError(f"cannot load sharded plan: {exc}") from exc


def _coerce_schedule(value: str | Path) -> tuple[CanonicalTrialSchedule, Path]:
    try:
        return load_canonical_trial_schedule(value), Path(value)
    except (PolicyInterventionError, TypeError) as exc:
        raise TrialRuntimeError(f"cannot load policy schedule: {exc}") from exc


def _group_schedule(
    schedule: CanonicalTrialSchedule,
    *,
    trial_families: Mapping[str, str],
) -> tuple[tuple[RuntimeGroupDescriptor, tuple[TrialScheduleRow, ...]], ...]:
    expected = dict(trial_families)
    if not expected or len(expected) != len(trial_families):
        raise TrialRuntimeError("plan trial-family mapping is empty or repeated")
    groups: list[tuple[RuntimeGroupDescriptor, tuple[TrialScheduleRow, ...]]] = []
    assigned: set[str] = set()
    prior_key: tuple[int, str, int, str] | None = None
    rows: list[TrialScheduleRow] = []

    def finalize() -> None:
        if not rows:
            return
        first = rows[0]
        observed = [row.trial_key for row in rows]
        if len(observed) != len(set(observed)) or not set(observed).issubset(expected):
            raise TrialRuntimeError("one schedule block repeats or invents a plan trial")
        if any(row.family_key != expected[row.trial_key] for row in rows):
            raise TrialRuntimeError("schedule family mapping differs from the sharded plan")
        if assigned.intersection(observed):
            raise TrialRuntimeError("partitioned schedule assigns one plan trial twice")
        if tuple(observed) != tuple(sorted(observed)):
            raise TrialRuntimeError("schedule block trial order is not canonical")
        invariant = {
            (
                row.environment_sha256,
                row.mask_id,
                row.mask_sha256,
                row.authorized_count,
                row.realized_allow_rate,
                row.expected_policy_revision,
            )
            for row in rows
        }
        if len(invariant) != 1:
            raise TrialRuntimeError("schedule block changes its environment or mask")
        payload = b"".join(_canonical_bytes(row.to_dict()) + b"\n" for row in rows)
        groups.append(
            (
                RuntimeGroupDescriptor(
                    block_order=len(groups),
                    group_order=first.group_order,
                    subject=first.subject,
                    repetition=first.repetition,
                    policy_state=first.policy_state,
                    environment_sha256=first.environment_sha256,
                    mask_id=first.mask_id,
                    mask_sha256=first.mask_sha256,
                    authorized_count=first.authorized_count,
                    realized_allow_rate=first.realized_allow_rate,
                    expected_policy_revision=first.expected_policy_revision,
                    schedule_rows_sha256=_sha256(payload),
                ),
                tuple(rows),
            )
        )
        assigned.update(observed)

    for row in schedule.rows:
        key = (row.group_order, row.subject, row.repetition, row.policy_state)
        if prior_key is not None and key != prior_key:
            finalize()
            rows = []
        rows.append(row)
        prior_key = key
    finalize()
    if not groups:
        raise TrialRuntimeError("policy schedule contains no runtime blocks")
    if assigned != set(expected):
        raise TrialRuntimeError("partitioned schedule does not cover every plan trial exactly once")
    return tuple(groups)


def admit_trial_runtime(
    plan: ShardedOnlineExecutionPlan | str | Path,
    query_package_root: str | Path,
    staged_root: str | Path,
    embedding_store_root: str | Path,
    schedule: str | Path,
    feature_bindings: Sequence[RuntimeFeatureBinding],
    *,
    partition_audit_path: str | Path,
    receipt_target: str | Path | None = None,
) -> TrialRuntimeAdmission:
    """Join the plan and schedule without loading corpus rows or query vectors."""

    admitted_plan = _coerce_plan(plan)
    if admitted_plan.stage != "sealed":
        raise TrialRuntimeError("sharded plan stage must equal 'sealed'")
    query_receipt = verify_query_trial_store(
        query_package_root,
        staged_root,
        embedding_store_root,
        partition_audit_path=partition_audit_path,
    )
    query_pin = admitted_plan.query_trial_store.artifact
    query_receipt_pin = admitted_plan.query_trial_store.receipt
    if (
        admitted_plan.key_id != query_receipt.hmac_key_id
        or admitted_plan.corpus != query_receipt.corpus
        or admitted_plan.stage != query_receipt.stage
        or admitted_plan.trials
        != tuple(sorted(query_receipt.opaque_trials, key=lambda row: row.trial_key))
        or admitted_plan.query_trial_store.record_count != query_receipt.record_count
        or query_pin.sha256 != query_receipt.query_trial_store_sha256
        or query_pin.byte_count != query_receipt.query_trial_store_byte_count
        or query_receipt_pin.sha256 != query_receipt.receipt_sha256
        or query_receipt_pin.byte_count != query_receipt.receipt_byte_count
        or admitted_plan.query_partition_audit_sha256 != query_receipt.query_partition_audit_sha256
    ):
        raise TrialRuntimeError("sharded plan differs from the query/trial package")
    admitted_schedule, schedule_path = _coerce_schedule(schedule)
    if (
        admitted_schedule.execution_artifact_sha256 != admitted_plan.artifact_sha256
        or admitted_schedule.corpus != admitted_plan.corpus
        or admitted_schedule.stage != admitted_plan.stage
        or admitted_schedule.document_count != admitted_plan.document_count
        or admitted_schedule.document_universe_sha256 != admitted_plan.document_universe_sha256
    ):
        raise TrialRuntimeError("policy schedule differs from the sharded plan")
    grouped = _group_schedule(
        admitted_schedule,
        trial_families={row.trial_key: row.family_key for row in admitted_plan.trials},
    )
    bindings = tuple(feature_bindings)
    if not bindings or not all(isinstance(row, RuntimeFeatureBinding) for row in bindings):
        raise TrialRuntimeError("feature_bindings must contain RuntimeFeatureBinding values")
    binding_by_key = {row.block_key: row for row in bindings}
    if len(binding_by_key) != len(bindings) or set(binding_by_key) != {
        descriptor.block_key for descriptor, _rows in grouped
    }:
        raise TrialRuntimeError("feature bindings must cover the exact schedule blocks")
    feature_payload = b"".join(
        _canonical_bytes(binding_by_key[descriptor.block_key].to_dict()) + b"\n"
        for descriptor, _rows in grouped
    )
    receipt = TrialRuntimeAdmissionReceipt(
        execution_artifact_sha256=admitted_plan.artifact_sha256,
        query_trial_store_sha256=query_receipt.query_trial_store_sha256,
        query_partition_audit_sha256=(query_receipt.query_partition_audit_sha256),
        schedule_sha256=admitted_schedule.artifact_sha256,
        staged_inventory_sha256=query_receipt.staged_inventory_sha256,
        source_inventory_sha256=query_receipt.source_inventory_sha256,
        embedding_store_receipt_sha256=query_receipt.embedding_store_receipt_sha256,
        active_query_epoch=query_receipt.active_query_epoch,
        current_truth_query_epoch=query_receipt.current_truth_query_epoch,
        policy_bundle_revision=admitted_schedule.policy_bundle_revision,
        policy_config_sha256=admitted_schedule.config_sha256,
        mask_catalog_sha256=admitted_schedule.mask_catalog_sha256,
        feature_bindings_sha256=_sha256(feature_payload),
        assignment_seed_sha256=admitted_schedule.assignment_seed_sha256,
        assignment_map_sha256=admitted_schedule.assignment_map_sha256,
        permutation_seed=admitted_plan.permutation_seed,
        query_count=query_receipt.record_count,
        groups=tuple(descriptor for descriptor, _rows in grouped),
        trial_state_assignment_algorithm=(admitted_schedule.assignment_algorithm),
    )
    if receipt_target is not None:
        try:
            write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), receipt_target)
        except ArtifactIntegrityError as exc:
            raise TrialRuntimeError(f"cannot publish runtime receipt: {exc}") from exc
    return TrialRuntimeAdmission(
        plan=admitted_plan,
        partition_audit_path=Path(partition_audit_path),
        query_package_root=Path(query_package_root),
        staged_root=Path(staged_root),
        embedding_store_root=Path(embedding_store_root),
        schedule_path=schedule_path,
        feature_bindings=bindings,
        receipt=receipt,
    )


def load_trial_runtime_receipt(path: str | Path) -> TrialRuntimeAdmissionReceipt:
    """Load one exclusive admission receipt and reject non-canonical bytes."""

    target = Path(path)
    size = _bounded_control_size(target, label="runtime receipt")
    encoded = _read(target, expected_bytes=size, label="trial runtime receipt")
    receipt = TrialRuntimeAdmissionReceipt.from_dict(
        _canonical_object(encoded, label="trial runtime receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise TrialRuntimeError("runtime receipt bytes differ from its typed form")
    return receipt


def _canonical_absolute_runtime_path(name: str, value: str | Path) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str):
        raise TrialRuntimeError(f"{name} must be a text path")
    if (
        not raw.startswith("/")
        or "\\" in raw
        or unicodedata.normalize("NFC", raw) != raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
        or PurePosixPath(raw).as_posix() != raw
    ):
        raise TrialRuntimeError(f"{name} must be a canonical absolute POSIX path")
    return Path(raw)


def load_trial_runtime_admission(
    *,
    plan_path: str | Path,
    receipt_path: str | Path,
    partition_audit_path: str | Path,
    query_package_root: str | Path,
    staged_root: str | Path,
    embedding_store_root: str | Path,
    schedule_path: str | Path,
    feature_bindings: Sequence[RuntimeFeatureBinding],
) -> TrialRuntimeAdmission:
    """Restore a frozen lazy admission without opening workload sources.

    Only the immutable execution plan and prior admission receipt are read.
    Query rows, matrices, schedule rows, policy data, and corpus artifacts stay
    unopened until :func:`load_trial_runtime` runs after the one-shot marker.
    """

    admitted_plan = load_sharded_online_execution_plan(
        _canonical_absolute_runtime_path("plan_path", plan_path)
    )
    receipt = load_trial_runtime_receipt(
        _canonical_absolute_runtime_path("receipt_path", receipt_path)
    )
    return reconstruct_trial_runtime_admission(
        plan=admitted_plan,
        receipt=receipt,
        partition_audit_path=partition_audit_path,
        query_package_root=query_package_root,
        staged_root=staged_root,
        embedding_store_root=embedding_store_root,
        schedule_path=schedule_path,
        feature_bindings=feature_bindings,
    )


def reconstruct_trial_runtime_admission(
    *,
    plan: ShardedOnlineExecutionPlan,
    receipt: TrialRuntimeAdmissionReceipt,
    partition_audit_path: str | Path,
    query_package_root: str | Path,
    staged_root: str | Path,
    embedding_store_root: str | Path,
    schedule_path: str | Path,
    feature_bindings: Sequence[RuntimeFeatureBinding],
) -> TrialRuntimeAdmission:
    """Bind already parsed controls to lazy source paths without source I/O."""

    if not isinstance(plan, ShardedOnlineExecutionPlan):
        raise TrialRuntimeError("plan must be a ShardedOnlineExecutionPlan")
    if not isinstance(receipt, TrialRuntimeAdmissionReceipt):
        raise TrialRuntimeError("receipt must be a TrialRuntimeAdmissionReceipt")
    admitted_plan = plan
    if admitted_plan.stage != "sealed":
        raise TrialRuntimeError("sharded plan stage must equal 'sealed'")
    if receipt.execution_artifact_sha256 != admitted_plan.artifact_sha256:
        raise TrialRuntimeError("runtime receipt differs from the execution plan")
    if (
        receipt.query_trial_store_sha256 != admitted_plan.query_trial_store.artifact.sha256
        or receipt.query_partition_audit_sha256 != admitted_plan.query_partition_audit_sha256
        or receipt.permutation_seed != admitted_plan.permutation_seed
        or receipt.query_count != len(admitted_plan.trials)
        or receipt.query_count != admitted_plan.query_trial_store.record_count
    ):
        raise TrialRuntimeError("runtime receipt differs from the plan-bound query cohort")

    bindings = tuple(feature_bindings)
    if not bindings or not all(isinstance(row, RuntimeFeatureBinding) for row in bindings):
        raise TrialRuntimeError("feature_bindings must contain RuntimeFeatureBinding values")
    expected_keys = tuple(group.block_key for group in receipt.groups)
    observed_keys = tuple(binding.block_key for binding in bindings)
    if observed_keys != expected_keys or len(set(observed_keys)) != len(observed_keys):
        raise TrialRuntimeError("feature bindings must follow the exact receipt-bound block order")
    feature_payload = b"".join(_canonical_bytes(binding.to_dict()) + b"\n" for binding in bindings)
    if _sha256(feature_payload) != receipt.feature_bindings_sha256:
        raise TrialRuntimeError("feature bindings differ from the runtime receipt")

    source_paths = {
        "partition_audit_path": _canonical_absolute_runtime_path(
            "partition_audit_path", partition_audit_path
        ),
        "query_package_root": _canonical_absolute_runtime_path(
            "query_package_root", query_package_root
        ),
        "staged_root": _canonical_absolute_runtime_path("staged_root", staged_root),
        "embedding_store_root": _canonical_absolute_runtime_path(
            "embedding_store_root", embedding_store_root
        ),
        "schedule_path": _canonical_absolute_runtime_path("schedule_path", schedule_path),
    }
    if len(set(source_paths.values())) != len(source_paths):
        raise TrialRuntimeError("runtime source paths must be distinct")
    return TrialRuntimeAdmission(
        plan=admitted_plan,
        partition_audit_path=source_paths["partition_audit_path"],
        query_package_root=source_paths["query_package_root"],
        staged_root=source_paths["staged_root"],
        embedding_store_root=source_paths["embedding_store_root"],
        schedule_path=source_paths["schedule_path"],
        feature_bindings=bindings,
        receipt=receipt,
    )


def _load_query_matrix(
    embedding_store_root: Path,
    embedding_receipt: EmbeddingStoreReceipt,
    *,
    matrix: str,
    epoch: QueryVectorEpochBinding,
) -> np.ndarray:
    descriptor = embedding_receipt.vectors.get(matrix)
    if (
        descriptor is None
        or _query_epoch_binding(
            embedding_receipt,
            matrix=matrix,
            role=epoch.role,
        )
        != epoch
    ):
        raise TrialRuntimeError(f"query vector matrix {matrix!r} changed binding")
    encoded = _read(
        embedding_store_root / descriptor.relative_path,
        expected_bytes=descriptor.byte_count,
        label=f"{epoch.role} query vector matrix",
    )
    if _sha256(encoded) != descriptor.file_sha256:
        raise TrialRuntimeError(f"{epoch.role} query vector matrix digest differs")
    try:
        array = np.load(io.BytesIO(encoded), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TrialRuntimeError(f"cannot load {epoch.role} query vector matrix: {exc}") from exc
    if (
        not isinstance(array, np.ndarray)
        or array.shape != descriptor.shape
        or array.dtype != np.dtype(descriptor.dtype)
        or array.ndim != 2
        or not array.flags.c_contiguous
        or not np.all(np.isfinite(array))
    ):
        raise TrialRuntimeError(f"{epoch.role} query vector matrix differs from its descriptor")
    array.setflags(write=False)
    return array


def load_trial_runtime_block(
    admission: TrialRuntimeAdmission,
    block_order: int,
) -> LoadedRuntimeBlock:
    """Reverify all sources, then load one disjoint policy block."""

    if not isinstance(admission, TrialRuntimeAdmission):
        raise TrialRuntimeError("admission must be a TrialRuntimeAdmission")
    index = _require_integer("block_order", block_order)
    if index >= len(admission.receipt.groups):
        raise TrialRuntimeError("block_order is outside the admitted group sequence")
    query_receipt = verify_query_trial_store(
        admission.query_package_root,
        admission.staged_root,
        admission.embedding_store_root,
        partition_audit_path=admission.partition_audit_path,
    )
    query_pin = admission.plan.query_trial_store.artifact
    query_receipt_pin = admission.plan.query_trial_store.receipt
    try:
        embedding_receipt = verify_embedding_store(admission.embedding_store_root)
    except EmbeddingStoreError as exc:
        raise TrialRuntimeError(f"embedding store verification failed: {exc}") from exc
    schedule = load_canonical_trial_schedule(admission.schedule_path)
    grouped = _group_schedule(
        schedule,
        trial_families={row.trial_key: row.family_key for row in admission.plan.trials},
    )
    descriptor, schedule_rows = grouped[index]
    if descriptor != admission.receipt.groups[index]:
        raise TrialRuntimeError("runtime group changed after admission")
    bindings = {row.block_key: row for row in admission.feature_bindings}
    binding = bindings.get(descriptor.block_key)
    if binding is None:
        raise TrialRuntimeError("runtime group lacks its frozen feature context")
    feature_payload = b"".join(
        _canonical_bytes(bindings[group.block_key].to_dict()) + b"\n"
        for group in admission.receipt.groups
    )
    if _sha256(feature_payload) != admission.receipt.feature_bindings_sha256:
        raise TrialRuntimeError("frozen feature contexts changed after admission")
    if (
        admission.plan.artifact_sha256 != admission.receipt.execution_artifact_sha256
        or admission.plan.key_id != query_receipt.hmac_key_id
        or admission.plan.corpus != query_receipt.corpus
        or admission.plan.stage != query_receipt.stage
        or admission.plan.trials
        != tuple(sorted(query_receipt.opaque_trials, key=lambda row: row.trial_key))
        or admission.plan.query_trial_store.record_count != query_receipt.record_count
        or query_pin.sha256 != query_receipt.query_trial_store_sha256
        or query_pin.byte_count != query_receipt.query_trial_store_byte_count
        or query_receipt.query_trial_store_sha256 != admission.receipt.query_trial_store_sha256
        or query_receipt_pin.sha256 != query_receipt.receipt_sha256
        or query_receipt_pin.byte_count != query_receipt.receipt_byte_count
        or query_receipt.query_partition_audit_sha256
        != admission.receipt.query_partition_audit_sha256
        or admission.plan.query_partition_audit_sha256
        != admission.receipt.query_partition_audit_sha256
        or admission.plan.permutation_seed != admission.receipt.permutation_seed
        or embedding_receipt.receipt_sha256 != admission.receipt.embedding_store_receipt_sha256
        or schedule.artifact_sha256 != admission.receipt.schedule_sha256
        or schedule.execution_artifact_sha256 != admission.plan.artifact_sha256
        or schedule.assignment_seed_sha256 != admission.receipt.assignment_seed_sha256
        or schedule.assignment_map_sha256 != admission.receipt.assignment_map_sha256
        or schedule.assignment_algorithm != admission.receipt.trial_state_assignment_algorithm
    ):
        raise TrialRuntimeError("one runtime source changed after admission")
    query_rows = _load_query_rows(
        admission.query_package_root / QUERY_TRIAL_FILENAME,
        expected_bytes=query_receipt.query_trial_store_byte_count,
        expected_sha256=query_receipt.query_trial_store_sha256,
    )
    active_matrix = _load_query_matrix(
        admission.embedding_store_root,
        embedding_receipt,
        matrix="old_queries",
        epoch=admission.receipt.active_query_epoch,
    )
    truth_matrix = _load_query_matrix(
        admission.embedding_store_root,
        embedding_receipt,
        matrix="current_queries",
        epoch=admission.receipt.current_truth_query_epoch,
    )
    schedule_by_trial = {row.trial_key: row for row in schedule_rows}
    online_trials: list[OnlineTrial] = []
    runtimes: dict[str, OnlineTrialRuntime] = {}
    for row in query_rows:
        schedule_row = schedule_by_trial.get(row.trial_key)
        if schedule_row is None:
            continue
        active_vector = active_matrix[row.source.embedding_query_row]
        truth_vector = truth_matrix[row.source.embedding_query_row]
        runtime = OnlineTrialRuntime(
            active_query_vector=active_vector,
            current_truth_query_vector=truth_vector,
            feature_context=binding.context,
            environment=dict(schedule_row.environment),
        )
        if runtime.environment_sha256 != schedule_row.environment_sha256:
            raise TrialRuntimeError("runtime environment differs from the schedule")
        online_trials.append(
            OnlineTrial(
                trial_key=row.trial_key,
                family_key=row.family_key,
                text=row.text,
                corpus=row.corpus,
                stage=row.stage,
            )
        )
        runtimes[row.trial_key] = runtime
    del active_matrix, truth_matrix
    execution = ShardedQueryExecutionAdapter(
        plan=admission.plan,
        trials=tuple(online_trials),
    )
    if len(runtimes) != len(schedule_rows):
        raise TrialRuntimeError("runtime mapping differs from its schedule block")
    return LoadedRuntimeBlock(
        descriptor=descriptor,
        execution=execution,
        trial_runtimes=runtimes,
    )


def load_trial_runtime(admission: TrialRuntimeAdmission) -> LoadedTrialRuntime:
    """Reverify and combine every disjoint block for one sealed online run."""

    if not isinstance(admission, TrialRuntimeAdmission):
        raise TrialRuntimeError("admission must be a TrialRuntimeAdmission")
    blocks = tuple(
        load_trial_runtime_block(admission, block_order)
        for block_order in range(len(admission.receipt.groups))
    )
    online_trials: dict[str, OnlineTrial] = {}
    runtimes: dict[str, OnlineTrialRuntime] = {}
    for block in blocks:
        if block.execution.plan.artifact_sha256 != admission.plan.artifact_sha256:
            raise TrialRuntimeError("one runtime block changed the admitted plan digest")
        for trial in block.execution.trials:
            if trial.trial_key in online_trials or trial.trial_key in runtimes:
                raise TrialRuntimeError("runtime blocks assign one plan trial more than once")
            runtime = block.trial_runtimes.get(trial.trial_key)
            if runtime is None:
                raise TrialRuntimeError("runtime block lacks its trial runtime")
            online_trials[trial.trial_key] = trial
            runtimes[trial.trial_key] = runtime
    if set(online_trials) != set(admission.plan.trial_keys):
        raise TrialRuntimeError("runtime blocks do not cover every admitted plan trial")
    execution = ShardedQueryExecutionAdapter(
        plan=admission.plan,
        trials=tuple(online_trials.values()),
    )
    if execution.artifact_sha256 != admission.plan.artifact_sha256:
        raise TrialRuntimeError("combined execution changed the admitted plan digest")
    return LoadedTrialRuntime(
        descriptors=tuple(block.descriptor for block in blocks),
        execution=execution,
        trial_runtimes=runtimes,
    )
