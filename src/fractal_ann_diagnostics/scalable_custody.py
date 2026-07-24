"""Streaming label custody for inventory-bound sealed corpora.

The builder maps bytewise-ordered external document identifiers to canonical
integer rows without retaining the corpus in memory.  Online output contains
only query text, opaque trial/family keys, and fixed-width content digests.
Qrels, answers, evidence, and raw source identifiers remain in the custody
artifact accepted by the existing post-run scoring path.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import quote

from .label_separation import (
    SEALED_LABEL_SCHEMA,
    SealedEvidenceBundle,
    SealedEvidenceLocation,
    SealedLabelArtifact,
    SealedTrialLabels,
)
from .query_cohort import (
    FAMILY_SELECTION_ALGORITHM,
    FAMILY_SELECTION_DOMAIN,  # noqa: F401 - compatibility re-export
    NESTED_ROWS_PER_FAMILY,
    NESTED_TRIAL_SOURCE_DOMAIN,  # noqa: F401 - compatibility re-export
    REPRESENTATIVE_SELECTION_ALGORITHM,
    REPRESENTATIVE_SELECTION_DOMAIN,  # noqa: F401 - compatibility re-export
    family_selection_rank,
    nested_trial_source_value,
    representative_selection_rank,
)
from .study_data import (
    ASSIGNMENT_SCHEMA,
    INVENTORY_SCHEMA,
    StudyDataError,
    verify_staged_data,
)

SCALABLE_CUSTODY_PLAN_SCHEMA = "fractal-scalable-custody-plan-v1"
SCALABLE_CUSTODY_CONFIG_SCHEMA = "fractal-scalable-custody-config-v1"
SCALABLE_CUSTODY_RECEIPT_SCHEMA = "fractal-scalable-custody-receipt-v1"
CUSTODY_QUERY_KEY_ROW_SCHEMA = "fractal-custody-query-key-row-v1"
KEY_DERIVATION_ALGORITHM = "fractal-label-separation-v2"
DOCUMENT_ROW_ORDER_ALGORITHM = "fractal-document-row-order-length-prefixed-v1"
PROVENANCE_RECORD_SIZE_BYTES = 32

QUERY_KEY_MAP_PATH = "online/query-key-map.jsonl"
PROVENANCE_PATH = "online/provenance-sha256.bin"
SEALED_LABEL_PATH = "custody/sealed-labels.json"
RECEIPT_PATH = "receipt.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CORPUS_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_SHARD_PATH = re.compile(r"^datasets/([^/]+)/corpus/part-([0-9]{5})\.jsonl$")
_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_MAX_LINE_BYTES = 16 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_MAX_KEY_BYTES = 4096
_EVIDENCE_CORPORA = frozenset({"scifact", "hotpotqa-fullwiki", "t2-ragbench"})
_PLACEHOLDERS = frozenset({"", "latest", "main", "master", "tbd", "todo", "unassigned"})

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
_SOURCE_ARTIFACT_FIELDS = frozenset(
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
_DOCUMENT_FIELDS = frozenset({"id", "text", "title"})
_QUERY_FIELDS = frozenset({"id", "text"})
_QREL_FIELDS = frozenset({"document_id", "query_id", "relevance"})
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
_EVIDENCE_ROW_FIELDS = frozenset({"answer", "evidence_bundles", "label_metadata", "query_id"})
_EVIDENCE_BUNDLE_FIELDS = frozenset({"bundle_id", "locations"})
_EVIDENCE_LOCATION_FIELDS = frozenset({"document_id", "locator"})
_QUERY_KEY_ROW_FIELDS = frozenset(
    {
        "corpus",
        "family_key",
        "nested_index",
        "query_row",
        "schema_version",
        "stage",
        "text",
        "trial_key",
    }
)
_PUBLISHED_ARTIFACT_FIELDS = frozenset(
    {
        "byte_count",
        "path",
        "record_count",
        "record_size_bytes",
        "role",
        "sha256",
        "visibility",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "allowlisted_paths",
        "available_families",
        "corpus",
        "execution_artifact_sha256",
        "expected_document_count",
        "family_selection_algorithm",
        "hmac_key_id",
        "nested_rows_per_family",
        "representative_selection_algorithm",
        "schema_version",
        "selected_families",
        "selection_seed_sha256",
        "stage",
        "staged_inventory_sha256",
    }
)
_HMAC_KEY_PIN_FIELDS = frozenset({"byte_count", "path", "sha256"})
_CONFIG_FIELDS = frozenset(
    {
        "corpus",
        "available_families",
        "execution_artifact_sha256",
        "expected_document_count",
        "family_selection_algorithm",
        "hmac_key",
        "hmac_key_id",
        "nested_rows_per_family",
        "representative_selection_algorithm",
        "schema_version",
        "selected_families",
        "selection_seed_sha256",
        "source_artifacts",
        "stage",
        "staged_inventory_sha256",
        "staged_root",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "artifacts",
        "available_family_count",
        "corpus",
        "document_count",
        "document_row_order_algorithm",
        "execution_artifact_sha256",
        "family_selection_algorithm",
        "hmac_key_id",
        "key_derivation_algorithm",
        "nested_rows_per_family",
        "ordered_document_row_sha256",
        "query_count",
        "representative_selection_algorithm",
        "schema_version",
        "selected_family_count",
        "selection_seed_sha256",
        "source_artifacts",
        "stage",
        "staged_inventory_sha256",
    }
)


class ScalableCustodyError(RuntimeError):
    """Raised when a custody package cannot be built without ambiguity."""


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
        raise ScalableCustodyError("custody metadata must be finite canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _closed_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ScalableCustodyError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise ScalableCustodyError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _decode_json(value: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ScalableCustodyError(f"{label} repeats key {key!r}")
            result[key] = item
        return result

    def reject_nonfinite(value: str) -> None:
        raise ScalableCustodyError(f"{label} contains non-finite value {value!r}")

    try:
        text = value.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise ScalableCustodyError(f"{label} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ScalableCustodyError(f"{label} is not valid JSON: {exc.msg}") from exc


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ScalableCustodyError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScalableCustodyError(f"{name} must be an integer >= {minimum}")
    return value


def _require_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ScalableCustodyError(f"{name} must be a canonical non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ScalableCustodyError(f"{name} must be valid UTF-8") from exc
    return value


def _require_body_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScalableCustodyError(f"{name} must be non-empty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ScalableCustodyError(f"{name} must be valid UTF-8") from exc
    return value


def _relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ScalableCustodyError(f"{name} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ScalableCustodyError(f"{name} must be a canonical relative POSIX path")
    return value


def _absolute_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ScalableCustodyError(f"{name} must be an absolute POSIX path string")
    path = Path(value)
    if (
        not path.is_absolute()
        or value.startswith("//")
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or unicodedata.normalize("NFC", value) != value
        or any(part.casefold() in _PLACEHOLDERS for part in path.parts[1:])
    ):
        raise ScalableCustodyError(
            f"{name} must be a canonical absolute path without placeholder segments"
        )
    return path


def _require_nonplaceholder(name: str, value: object) -> str:
    text = _require_text(name, value)
    if text.casefold() in _PLACEHOLDERS:
        raise ScalableCustodyError(f"{name} cannot be a movable placeholder")
    return text


def _require_nonplaceholder_sha256(name: str, value: object) -> str:
    digest = _require_sha256(name, value)
    if len(set(digest)) == 1:
        raise ScalableCustodyError(f"{name} cannot be a placeholder digest")
    return digest


@dataclass(frozen=True)
class ScalableCustodyPlan:
    """Externally pinned inputs for one corpus custody build."""

    corpus: str
    staged_inventory_sha256: str
    execution_artifact_sha256: str
    hmac_key_id: str
    expected_document_count: int
    available_families: int
    selected_families: int
    selection_seed_sha256: str
    allowlisted_paths: tuple[str, ...]
    family_selection_algorithm: str = FAMILY_SELECTION_ALGORITHM
    representative_selection_algorithm: str = REPRESENTATIVE_SELECTION_ALGORITHM
    nested_rows_per_family: int = NESTED_ROWS_PER_FAMILY
    stage: str = "sealed"
    schema_version: str = SCALABLE_CUSTODY_PLAN_SCHEMA

    def __post_init__(self) -> None:
        corpus = _require_text("corpus", self.corpus)
        if _CORPUS_NAME.fullmatch(corpus) is None:
            raise ScalableCustodyError("corpus must be a lowercase URI-safe identifier")
        _require_sha256("staged_inventory_sha256", self.staged_inventory_sha256)
        _require_sha256("execution_artifact_sha256", self.execution_artifact_sha256)
        _require_text("hmac_key_id", self.hmac_key_id)
        _require_integer("expected_document_count", self.expected_document_count, minimum=1)
        _require_integer("available_families", self.available_families, minimum=1)
        _require_integer("selected_families", self.selected_families, minimum=1)
        if self.selected_families > self.available_families:
            raise ScalableCustodyError("selected_families cannot exceed available_families")
        _require_sha256("selection_seed_sha256", self.selection_seed_sha256)
        if self.family_selection_algorithm != FAMILY_SELECTION_ALGORITHM:
            raise ScalableCustodyError("family selection algorithm differs")
        if self.representative_selection_algorithm != REPRESENTATIVE_SELECTION_ALGORITHM:
            raise ScalableCustodyError("representative selection algorithm differs")
        if self.nested_rows_per_family != NESTED_ROWS_PER_FAMILY:
            raise ScalableCustodyError(
                f"nested_rows_per_family must equal {NESTED_ROWS_PER_FAMILY}"
            )
        if self.stage != "sealed":
            raise ScalableCustodyError("scalable custody accepts only stage='sealed'")
        if self.schema_version != SCALABLE_CUSTODY_PLAN_SCHEMA:
            raise ScalableCustodyError(
                f"schema_version must equal {SCALABLE_CUSTODY_PLAN_SCHEMA!r}"
            )
        paths = tuple(
            sorted(
                (_relative_path(path, name="allowlisted path") for path in self.allowlisted_paths),
                key=lambda item: item.encode("utf-8"),
            )
        )
        if not paths or len(paths) != len(set(paths)):
            raise ScalableCustodyError("allowlisted_paths must be non-empty and unique")
        object.__setattr__(self, "allowlisted_paths", paths)

    def to_dict(self) -> dict[str, object]:
        return {
            "allowlisted_paths": list(self.allowlisted_paths),
            "available_families": self.available_families,
            "corpus": self.corpus,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "expected_document_count": self.expected_document_count,
            "family_selection_algorithm": self.family_selection_algorithm,
            "hmac_key_id": self.hmac_key_id,
            "nested_rows_per_family": self.nested_rows_per_family,
            "representative_selection_algorithm": (self.representative_selection_algorithm),
            "schema_version": self.schema_version,
            "selected_families": self.selected_families,
            "selection_seed_sha256": self.selection_seed_sha256,
            "stage": self.stage,
            "staged_inventory_sha256": self.staged_inventory_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> ScalableCustodyPlan:
        row = _closed_mapping(value, fields=_PLAN_FIELDS, label="scalable custody plan")
        paths = row["allowlisted_paths"]
        if not isinstance(paths, list):
            raise ScalableCustodyError("allowlisted_paths must be an array")
        return cls(
            corpus=row["corpus"],
            available_families=row["available_families"],
            staged_inventory_sha256=row["staged_inventory_sha256"],
            execution_artifact_sha256=row["execution_artifact_sha256"],
            hmac_key_id=row["hmac_key_id"],
            expected_document_count=row["expected_document_count"],
            selected_families=row["selected_families"],
            selection_seed_sha256=row["selection_seed_sha256"],
            allowlisted_paths=tuple(paths),
            family_selection_algorithm=row["family_selection_algorithm"],
            representative_selection_algorithm=row["representative_selection_algorithm"],
            nested_rows_per_family=row["nested_rows_per_family"],
            stage=row["stage"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class SourceArtifactPin:
    """One exact staged source row copied into the custody receipt."""

    path: str
    sha256: str
    byte_count: int
    record_count: int
    dataset: str | None
    stage: str | None
    role: str
    visibility: str

    def __post_init__(self) -> None:
        _relative_path(self.path, name="source artifact path")
        _require_sha256("source artifact sha256", self.sha256)
        _require_integer("source artifact byte_count", self.byte_count)
        _require_integer("source artifact record_count", self.record_count)
        if self.dataset is not None:
            _require_text("source artifact dataset", self.dataset)
        if self.stage is not None:
            _require_text("source artifact stage", self.stage)
        _require_text("source artifact role", self.role)
        _require_text("source artifact visibility", self.visibility)

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "dataset": self.dataset,
            "path": self.path,
            "record_count": self.record_count,
            "role": self.role,
            "sha256": self.sha256,
            "stage": self.stage,
            "visibility": self.visibility,
        }

    @classmethod
    def from_dict(cls, value: object) -> SourceArtifactPin:
        row = _closed_mapping(value, fields=_SOURCE_ARTIFACT_FIELDS, label="source artifact")
        return cls(
            path=row["path"],
            sha256=row["sha256"],
            byte_count=row["byte_count"],
            record_count=row["record_count"],
            dataset=row["dataset"],
            stage=row["stage"],
            role=row["role"],
            visibility=row["visibility"],
        )


@dataclass(frozen=True)
class HmacKeyFilePin:
    """Local binary HMAC key file pinned only inside the custodian config."""

    path: Path
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        path = _absolute_path(str(self.path), name="hmac_key.path")
        digest = _require_nonplaceholder_sha256("hmac_key.sha256", self.sha256)
        byte_count = _require_integer("hmac_key.byte_count", self.byte_count, minimum=32)
        if byte_count > _MAX_KEY_BYTES:
            raise ScalableCustodyError(f"hmac_key.byte_count cannot exceed {_MAX_KEY_BYTES} bytes")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "path": str(self.path),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> HmacKeyFilePin:
        row = _closed_mapping(value, fields=_HMAC_KEY_PIN_FIELDS, label="hmac_key")
        return cls(
            path=Path(row["path"]) if isinstance(row["path"], str) else row["path"],
            sha256=row["sha256"],
            byte_count=row["byte_count"],
        )


@dataclass(frozen=True)
class ScalableCustodyConfig:
    """Closed canonical configuration for one production custody build."""

    staged_root: Path
    corpus: str
    staged_inventory_sha256: str
    execution_artifact_sha256: str
    hmac_key_id: str
    hmac_key: HmacKeyFilePin
    expected_document_count: int
    available_families: int
    selected_families: int
    selection_seed_sha256: str
    source_artifacts: tuple[SourceArtifactPin, ...]
    family_selection_algorithm: str = FAMILY_SELECTION_ALGORITHM
    representative_selection_algorithm: str = REPRESENTATIVE_SELECTION_ALGORITHM
    nested_rows_per_family: int = NESTED_ROWS_PER_FAMILY
    stage: str = "sealed"
    schema_version: str = SCALABLE_CUSTODY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        staged_root = _absolute_path(str(self.staged_root), name="staged_root")
        corpus = _require_nonplaceholder("corpus", self.corpus)
        if _CORPUS_NAME.fullmatch(corpus) is None:
            raise ScalableCustodyError("corpus must be a lowercase URI-safe identifier")
        staged_inventory_sha256 = _require_nonplaceholder_sha256(
            "staged_inventory_sha256",
            self.staged_inventory_sha256,
        )
        execution_artifact_sha256 = _require_nonplaceholder_sha256(
            "execution_artifact_sha256",
            self.execution_artifact_sha256,
        )
        hmac_key_id = _require_nonplaceholder("hmac_key_id", self.hmac_key_id)
        if not isinstance(self.hmac_key, HmacKeyFilePin):
            raise ScalableCustodyError("hmac_key must be a typed file pin")
        _require_integer("expected_document_count", self.expected_document_count, minimum=1)
        _require_integer("available_families", self.available_families, minimum=1)
        _require_integer("selected_families", self.selected_families, minimum=1)
        if self.selected_families > self.available_families:
            raise ScalableCustodyError("selected_families cannot exceed available_families")
        selection_seed_sha256 = _require_nonplaceholder_sha256(
            "selection_seed_sha256",
            self.selection_seed_sha256,
        )
        if self.family_selection_algorithm != FAMILY_SELECTION_ALGORITHM:
            raise ScalableCustodyError("family selection algorithm differs")
        if self.representative_selection_algorithm != REPRESENTATIVE_SELECTION_ALGORITHM:
            raise ScalableCustodyError("representative selection algorithm differs")
        if self.nested_rows_per_family != NESTED_ROWS_PER_FAMILY:
            raise ScalableCustodyError(
                f"nested_rows_per_family must equal {NESTED_ROWS_PER_FAMILY}"
            )
        sources = tuple(sorted(self.source_artifacts, key=lambda item: item.path.encode()))
        if (
            not sources
            or not all(isinstance(item, SourceArtifactPin) for item in sources)
            or len({item.path for item in sources}) != len(sources)
        ):
            raise ScalableCustodyError("source_artifacts must be typed, non-empty, and unique")
        if self.stage != "sealed":
            raise ScalableCustodyError("custody config stage must equal 'sealed'")
        if self.schema_version != SCALABLE_CUSTODY_CONFIG_SCHEMA:
            raise ScalableCustodyError(
                f"schema_version must equal {SCALABLE_CUSTODY_CONFIG_SCHEMA!r}"
            )
        object.__setattr__(self, "staged_root", staged_root)
        object.__setattr__(self, "corpus", corpus)
        object.__setattr__(self, "staged_inventory_sha256", staged_inventory_sha256)
        object.__setattr__(self, "execution_artifact_sha256", execution_artifact_sha256)
        object.__setattr__(self, "hmac_key_id", hmac_key_id)
        object.__setattr__(self, "selection_seed_sha256", selection_seed_sha256)
        object.__setattr__(self, "source_artifacts", sources)

    @property
    def plan(self) -> ScalableCustodyPlan:
        return ScalableCustodyPlan(
            corpus=self.corpus,
            staged_inventory_sha256=self.staged_inventory_sha256,
            execution_artifact_sha256=self.execution_artifact_sha256,
            hmac_key_id=self.hmac_key_id,
            expected_document_count=self.expected_document_count,
            available_families=self.available_families,
            selected_families=self.selected_families,
            selection_seed_sha256=self.selection_seed_sha256,
            allowlisted_paths=tuple(item.path for item in self.source_artifacts),
            family_selection_algorithm=self.family_selection_algorithm,
            representative_selection_algorithm=(self.representative_selection_algorithm),
            nested_rows_per_family=self.nested_rows_per_family,
            stage=self.stage,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "available_families": self.available_families,
            "corpus": self.corpus,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "expected_document_count": self.expected_document_count,
            "family_selection_algorithm": self.family_selection_algorithm,
            "hmac_key": self.hmac_key.to_dict(),
            "hmac_key_id": self.hmac_key_id,
            "nested_rows_per_family": self.nested_rows_per_family,
            "representative_selection_algorithm": (self.representative_selection_algorithm),
            "schema_version": self.schema_version,
            "selected_families": self.selected_families,
            "selection_seed_sha256": self.selection_seed_sha256,
            "source_artifacts": [item.to_dict() for item in self.source_artifacts],
            "stage": self.stage,
            "staged_inventory_sha256": self.staged_inventory_sha256,
            "staged_root": str(self.staged_root),
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ScalableCustodyConfig:
        row = _closed_mapping(value, fields=_CONFIG_FIELDS, label="scalable custody config")
        source_values = row["source_artifacts"]
        if not isinstance(source_values, list):
            raise ScalableCustodyError("source_artifacts must be an array")
        staged_root = row["staged_root"]
        return cls(
            staged_root=(Path(staged_root) if isinstance(staged_root, str) else staged_root),
            available_families=row["available_families"],
            corpus=row["corpus"],
            staged_inventory_sha256=row["staged_inventory_sha256"],
            execution_artifact_sha256=row["execution_artifact_sha256"],
            hmac_key_id=row["hmac_key_id"],
            hmac_key=HmacKeyFilePin.from_dict(row["hmac_key"]),
            expected_document_count=row["expected_document_count"],
            selected_families=row["selected_families"],
            selection_seed_sha256=row["selection_seed_sha256"],
            source_artifacts=tuple(SourceArtifactPin.from_dict(item) for item in source_values),
            stage=row["stage"],
            family_selection_algorithm=row["family_selection_algorithm"],
            representative_selection_algorithm=row["representative_selection_algorithm"],
            nested_rows_per_family=row["nested_rows_per_family"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class PublishedArtifactPin:
    """One emitted online or custody artifact."""

    path: str
    sha256: str
    byte_count: int
    record_count: int
    role: str
    visibility: str
    record_size_bytes: int | None = None

    def __post_init__(self) -> None:
        _relative_path(self.path, name="published artifact path")
        _require_sha256("published artifact sha256", self.sha256)
        _require_integer("published artifact byte_count", self.byte_count)
        _require_integer("published artifact record_count", self.record_count)
        _require_text("published artifact role", self.role)
        if self.visibility not in {"online", "custody"}:
            raise ScalableCustodyError("published artifact visibility is invalid")
        if self.record_size_bytes is not None:
            _require_integer("record_size_bytes", self.record_size_bytes, minimum=1)
            if self.byte_count != self.record_count * self.record_size_bytes:
                raise ScalableCustodyError("fixed-width artifact byte count is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "path": self.path,
            "record_count": self.record_count,
            "record_size_bytes": self.record_size_bytes,
            "role": self.role,
            "sha256": self.sha256,
            "visibility": self.visibility,
        }

    @classmethod
    def from_dict(cls, value: object) -> PublishedArtifactPin:
        row = _closed_mapping(
            value,
            fields=_PUBLISHED_ARTIFACT_FIELDS,
            label="published artifact",
        )
        return cls(
            path=row["path"],
            sha256=row["sha256"],
            byte_count=row["byte_count"],
            record_count=row["record_count"],
            role=row["role"],
            visibility=row["visibility"],
            record_size_bytes=row["record_size_bytes"],
        )


@dataclass(frozen=True)
class ScalableCustodyReceipt:
    """Build evidence for one label-separated corpus package."""

    corpus: str
    stage: str
    staged_inventory_sha256: str
    execution_artifact_sha256: str
    hmac_key_id: str
    document_count: int
    query_count: int
    available_family_count: int
    selected_family_count: int
    selection_seed_sha256: str
    nested_rows_per_family: int
    ordered_document_row_sha256: str
    source_artifacts: tuple[SourceArtifactPin, ...]
    artifacts: tuple[PublishedArtifactPin, ...]
    key_derivation_algorithm: str = KEY_DERIVATION_ALGORITHM
    family_selection_algorithm: str = FAMILY_SELECTION_ALGORITHM
    representative_selection_algorithm: str = REPRESENTATIVE_SELECTION_ALGORITHM
    document_row_order_algorithm: str = DOCUMENT_ROW_ORDER_ALGORITHM
    schema_version: str = SCALABLE_CUSTODY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_text("corpus", self.corpus)
        if self.stage != "sealed":
            raise ScalableCustodyError("custody receipt stage must equal 'sealed'")
        for name in (
            "staged_inventory_sha256",
            "execution_artifact_sha256",
            "ordered_document_row_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_text("hmac_key_id", self.hmac_key_id)
        _require_integer("document_count", self.document_count, minimum=1)
        _require_integer("query_count", self.query_count, minimum=1)
        _require_integer("available_family_count", self.available_family_count, minimum=1)
        _require_integer("selected_family_count", self.selected_family_count, minimum=1)
        if self.selected_family_count > self.available_family_count:
            raise ScalableCustodyError("selected_family_count cannot exceed available_family_count")
        _require_sha256("selection_seed_sha256", self.selection_seed_sha256)
        if self.nested_rows_per_family != NESTED_ROWS_PER_FAMILY:
            raise ScalableCustodyError(
                f"nested_rows_per_family must equal {NESTED_ROWS_PER_FAMILY}"
            )
        if self.query_count != self.selected_family_count * self.nested_rows_per_family:
            raise ScalableCustodyError("query_count must equal selected families times nested rows")
        sources = tuple(sorted(self.source_artifacts, key=lambda item: item.path.encode()))
        outputs = tuple(sorted(self.artifacts, key=lambda item: item.path.encode()))
        if not sources or len({row.path for row in sources}) != len(sources):
            raise ScalableCustodyError("source_artifacts must be non-empty and unique")
        if {row.path for row in outputs} != {
            QUERY_KEY_MAP_PATH,
            PROVENANCE_PATH,
            SEALED_LABEL_PATH,
        }:
            raise ScalableCustodyError("published artifact membership differs")
        outputs_by_path = {row.path: row for row in outputs}
        output_contract = {
            QUERY_KEY_MAP_PATH: (
                "custody-query-key-map",
                "online",
                self.query_count,
                None,
            ),
            PROVENANCE_PATH: (
                "document-content-provenance",
                "online",
                self.document_count,
                PROVENANCE_RECORD_SIZE_BYTES,
            ),
            SEALED_LABEL_PATH: (
                "sealed-labels",
                "custody",
                self.query_count,
                None,
            ),
        }
        for path, (role, visibility, record_count, record_size_bytes) in output_contract.items():
            artifact = outputs_by_path[path]
            if (
                artifact.role != role
                or artifact.visibility != visibility
                or artifact.record_count != record_count
                or artifact.record_size_bytes != record_size_bytes
            ):
                raise ScalableCustodyError(f"published artifact contract differs for {path!r}")
        if self.key_derivation_algorithm != KEY_DERIVATION_ALGORITHM:
            raise ScalableCustodyError("key derivation algorithm differs")
        if self.family_selection_algorithm != FAMILY_SELECTION_ALGORITHM:
            raise ScalableCustodyError("family selection algorithm differs")
        if self.representative_selection_algorithm != REPRESENTATIVE_SELECTION_ALGORITHM:
            raise ScalableCustodyError("representative selection algorithm differs")
        if self.document_row_order_algorithm != DOCUMENT_ROW_ORDER_ALGORITHM:
            raise ScalableCustodyError("document row-order algorithm differs")
        if self.schema_version != SCALABLE_CUSTODY_RECEIPT_SCHEMA:
            raise ScalableCustodyError("custody receipt schema differs")
        object.__setattr__(self, "source_artifacts", sources)
        object.__setattr__(self, "artifacts", outputs)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [row.to_dict() for row in self.artifacts],
            "available_family_count": self.available_family_count,
            "corpus": self.corpus,
            "document_count": self.document_count,
            "document_row_order_algorithm": self.document_row_order_algorithm,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "family_selection_algorithm": self.family_selection_algorithm,
            "hmac_key_id": self.hmac_key_id,
            "key_derivation_algorithm": self.key_derivation_algorithm,
            "nested_rows_per_family": self.nested_rows_per_family,
            "ordered_document_row_sha256": self.ordered_document_row_sha256,
            "query_count": self.query_count,
            "representative_selection_algorithm": (self.representative_selection_algorithm),
            "schema_version": self.schema_version,
            "selected_family_count": self.selected_family_count,
            "selection_seed_sha256": self.selection_seed_sha256,
            "source_artifacts": [row.to_dict() for row in self.source_artifacts],
            "stage": self.stage,
            "staged_inventory_sha256": self.staged_inventory_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ScalableCustodyReceipt:
        row = _closed_mapping(value, fields=_RECEIPT_FIELDS, label="scalable custody receipt")
        sources = row["source_artifacts"]
        artifacts = row["artifacts"]
        if not isinstance(sources, list) or not isinstance(artifacts, list):
            raise ScalableCustodyError("receipt artifact fields must be arrays")
        return cls(
            corpus=row["corpus"],
            stage=row["stage"],
            staged_inventory_sha256=row["staged_inventory_sha256"],
            execution_artifact_sha256=row["execution_artifact_sha256"],
            hmac_key_id=row["hmac_key_id"],
            document_count=row["document_count"],
            query_count=row["query_count"],
            available_family_count=row["available_family_count"],
            selected_family_count=row["selected_family_count"],
            selection_seed_sha256=row["selection_seed_sha256"],
            nested_rows_per_family=row["nested_rows_per_family"],
            ordered_document_row_sha256=row["ordered_document_row_sha256"],
            source_artifacts=tuple(SourceArtifactPin.from_dict(item) for item in sources),
            artifacts=tuple(PublishedArtifactPin.from_dict(item) for item in artifacts),
            key_derivation_algorithm=row["key_derivation_algorithm"],
            family_selection_algorithm=row["family_selection_algorithm"],
            representative_selection_algorithm=row["representative_selection_algorithm"],
            document_row_order_algorithm=row["document_row_order_algorithm"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class _Selection:
    corpus: tuple[SourceArtifactPin, ...]
    queries: SourceArtifactPin
    qrels: SourceArtifactPin
    evidence: SourceArtifactPin | None
    assignments: SourceArtifactPin

    @property
    def sources(self) -> tuple[SourceArtifactPin, ...]:
        rows = (*self.corpus, self.queries, self.qrels, self.assignments)
        if self.evidence is not None:
            rows = (*rows, self.evidence)
        return tuple(sorted(rows, key=lambda item: item.path.encode()))


@dataclass(frozen=True)
class _Assignment:
    component_sha256: str
    query_text_sha256: str


@dataclass(frozen=True)
class _FamilySelection:
    components: frozenset[str]
    ordered_components: tuple[str, ...]
    available_count: int
    selected_count: int


@dataclass(frozen=True)
class _QueryMaterial:
    source_id: str
    text: str
    query_row: int
    nested_index: int
    trial_key: str
    family_key: str


@dataclass(frozen=True)
class _Representative:
    source_id: str
    text: str
    component_sha256: str
    query_id_sha256: str
    rank_sha256: str


def _open_root(path: Path, *, label: str) -> int:
    path = _absolute_path(str(path), name=label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open("/", flags)
        for part in path.parts[1:]:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                raise ScalableCustodyError(f"{label} component {part!r} must be a real directory")
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise ScalableCustodyError(f"cannot open {label}: {exc}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ScalableCustodyError(f"{label} must be a real directory")
    return descriptor


def _open_relative_file(root_descriptor: int, relative_path: str) -> tuple[int, os.stat_result]:
    parts = PurePosixPath(_relative_path(relative_path, name="source path")).parts
    parent = os.dup(root_descriptor)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for part in parts[:-1]:
            next_parent = os.open(part, directory_flags, dir_fd=parent)
            metadata = os.fstat(next_parent)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_parent)
                raise ScalableCustodyError(f"source path component {part!r} is not a directory")
            os.close(parent)
            parent = next_parent
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(parts[-1], flags, dir_fd=parent)
        metadata = os.fstat(descriptor)
        entry = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not stat.S_ISREG(entry.st_mode)
            or metadata.st_nlink != 1
            or entry.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (entry.st_dev, entry.st_ino)
        ):
            os.close(descriptor)
            raise ScalableCustodyError(
                f"source {relative_path!r} must be one unlinked regular file"
            )
        return descriptor, metadata
    except OSError as exc:
        raise ScalableCustodyError(f"cannot open source {relative_path!r}: {exc}") from exc
    finally:
        os.close(parent)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _read_secure_file(
    root_descriptor: int,
    relative_path: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    descriptor, opened = _open_relative_file(root_descriptor, relative_path)
    if opened.st_size > maximum_bytes:
        os.close(descriptor)
        raise ScalableCustodyError(f"{label} exceeds the control-file limit")
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ScalableCustodyError(f"{label} exceeds the control-file limit")
        after = os.fstat(descriptor)
        if _stat_signature(after) != _stat_signature(opened) or total != opened.st_size:
            raise ScalableCustodyError(f"{label} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _iter_canonical_jsonl(
    root_descriptor: int,
    source: SourceArtifactPin,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    descriptor, opened = _open_relative_file(root_descriptor, source.path)
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                line = handle.readline(_MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > _MAX_LINE_BYTES:
                    raise ScalableCustodyError(f"source {source.path!r} has an oversized line")
                digest.update(line)
                byte_count += len(line)
                record_count += 1
                if not line.endswith(b"\n") or line == b"\n":
                    raise ScalableCustodyError(
                        f"source {source.path!r} line {record_count} is not canonical JSONL"
                    )
                value = _decode_json(
                    line[:-1],
                    label=f"source {source.path} line {record_count}",
                )
                if not isinstance(value, Mapping):
                    raise ScalableCustodyError(
                        f"source {source.path!r} line {record_count} must contain an object"
                    )
                if _canonical_bytes(value) + b"\n" != line:
                    raise ScalableCustodyError(
                        f"source {source.path!r} line {record_count} is not canonical JSON"
                    )
                yield record_count, value
        after = os.fstat(descriptor)
        if _stat_signature(after) != _stat_signature(opened):
            raise ScalableCustodyError(f"source {source.path!r} changed while it was read")
        if byte_count != source.byte_count or opened.st_size != source.byte_count:
            raise ScalableCustodyError(f"source {source.path!r} byte count differs")
        if record_count != source.record_count:
            raise ScalableCustodyError(f"source {source.path!r} record count differs")
        if digest.hexdigest() != source.sha256:
            raise ScalableCustodyError(f"source {source.path!r} SHA-256 differs")
    finally:
        os.close(descriptor)


def _source_pin(value: object, *, position: int) -> SourceArtifactPin:
    row = _closed_mapping(
        value,
        fields=_SOURCE_ARTIFACT_FIELDS,
        label=f"staged artifact {position}",
    )
    return SourceArtifactPin.from_dict(row)


def _load_inventory(
    root_descriptor: int,
    plan: ScalableCustodyPlan,
) -> Mapping[str, Any]:
    encoded = _read_secure_file(
        root_descriptor,
        "inventory.json",
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="staged inventory",
    )
    if _sha256(encoded) != plan.staged_inventory_sha256:
        raise ScalableCustodyError("staged inventory differs from the external pin")
    checksum = _read_secure_file(
        root_descriptor,
        "inventory.sha256",
        maximum_bytes=1024,
        label="staged inventory checksum",
    )
    expected_checksum = f"{plan.staged_inventory_sha256}  inventory.json\n".encode("ascii")
    if checksum != expected_checksum:
        raise ScalableCustodyError("inventory.sha256 differs from the pinned inventory")
    value = _decode_json(encoded, label="staged inventory")
    row = _closed_mapping(value, fields=_INVENTORY_FIELDS, label="staged inventory")
    if row["schema_version"] != INVENTORY_SCHEMA:
        raise ScalableCustodyError("staged inventory schema differs")
    if encoded != _canonical_bytes(value) + b"\n":
        raise ScalableCustodyError("staged inventory bytes are not canonical")
    if row["withhold_sealed_labels_from_online_process"] is not True:
        raise ScalableCustodyError(
            "staged inventory does not withhold sealed labels from the online process"
        )
    return row


def _single_source(rows: Sequence[SourceArtifactPin], *, label: str) -> SourceArtifactPin:
    if len(rows) != 1:
        raise ScalableCustodyError(f"staged inventory must contain exactly one {label}")
    return rows[0]


def _select_sources(
    inventory: Mapping[str, Any],
    plan: ScalableCustodyPlan,
) -> _Selection:
    values = inventory["artifacts"]
    if not isinstance(values, list) or not values:
        raise ScalableCustodyError("staged inventory artifacts must be a non-empty array")
    sources = tuple(_source_pin(value, position=position) for position, value in enumerate(values))
    if len({source.path for source in sources}) != len(sources):
        raise ScalableCustodyError("staged inventory repeats an artifact path")

    corpus_sources = tuple(
        source
        for source in sources
        if source.dataset == plan.corpus
        and source.stage is None
        and source.role in {"corpus", "corpus-shard"}
        and source.visibility == "online"
    )
    if not corpus_sources:
        raise ScalableCustodyError("staged inventory has no selected corpus source")
    roles = {source.role for source in corpus_sources}
    if roles == {"corpus"}:
        if len(corpus_sources) != 1 or corpus_sources[0].path != (
            f"datasets/{plan.corpus}/corpus.jsonl"
        ):
            raise ScalableCustodyError("single-file corpus path differs from the staging contract")
    elif roles == {"corpus-shard"}:
        ordered = tuple(sorted(corpus_sources, key=lambda item: item.path.encode()))
        for index, source in enumerate(ordered):
            match = _SHARD_PATH.fullmatch(source.path)
            if match is None or match.group(1) != plan.corpus or int(match.group(2)) != index:
                raise ScalableCustodyError("corpus shard paths are not contiguous and canonical")
        corpus_sources = ordered
    else:
        raise ScalableCustodyError("staged inventory mixes corpus and corpus-shard roles")

    queries = _single_source(
        [
            source
            for source in sources
            if source.dataset == plan.corpus
            and source.stage == plan.stage
            and source.role == "queries"
            and source.visibility == "online"
        ],
        label="selected online query source",
    )
    qrels = _single_source(
        [
            source
            for source in sources
            if source.dataset == plan.corpus
            and source.stage == plan.stage
            and source.role == "qrels"
            and source.visibility == "custody"
        ],
        label="selected custody qrel source",
    )
    assignments = _single_source(
        [
            source
            for source in sources
            if source.dataset is None
            and source.stage is None
            and source.role == "assignments"
            and source.visibility == "online"
        ],
        label="online assignment source",
    )
    evidence_candidates = [
        source
        for source in sources
        if source.dataset == plan.corpus
        and source.stage == plan.stage
        and source.role == "evidence-bundles"
        and source.visibility == "custody"
    ]
    if plan.corpus in _EVIDENCE_CORPORA:
        evidence = _single_source(evidence_candidates, label="selected custody evidence source")
    else:
        if evidence_candidates:
            raise ScalableCustodyError(
                "BRIGHT/MIRACL-style relevance corpora must leave evidence undefined"
            )
        evidence = None

    expected_queries = f"datasets/{plan.corpus}/sealed/online/queries.jsonl"
    expected_qrels = f"datasets/{plan.corpus}/sealed/custody/qrels.jsonl"
    expected_evidence = f"datasets/{plan.corpus}/sealed/custody/evidence-bundles.jsonl"
    if queries.path != expected_queries or qrels.path != expected_qrels:
        raise ScalableCustodyError("sealed query/qrel paths differ from the staging contract")
    if evidence is not None and evidence.path != expected_evidence:
        raise ScalableCustodyError("sealed evidence path differs from the staging contract")
    if assignments.path != "assignments.jsonl":
        raise ScalableCustodyError("assignment path differs from the staging contract")

    selection = _Selection(
        corpus=tuple(corpus_sources),
        queries=queries,
        qrels=qrels,
        evidence=evidence,
        assignments=assignments,
    )
    observed_allowlist = tuple(source.path for source in selection.sources)
    if observed_allowlist != plan.allowlisted_paths:
        raise ScalableCustodyError(
            "exact source allowlist differs; "
            f"missing={sorted(set(observed_allowlist) - set(plan.allowlisted_paths))}, "
            f"unexpected={sorted(set(plan.allowlisted_paths) - set(observed_allowlist))}"
        )
    corpus_count = sum(source.record_count for source in selection.corpus)
    if corpus_count != plan.expected_document_count:
        raise ScalableCustodyError("corpus source count differs from expected_document_count")
    counts = inventory["counts"]
    if not isinstance(counts, Mapping) or not isinstance(counts.get(plan.corpus), Mapping):
        raise ScalableCustodyError("staged inventory lacks selected corpus counts")
    corpus_counts = counts[plan.corpus]
    if corpus_counts.get("documents") != plan.expected_document_count:
        raise ScalableCustodyError("inventory document count differs from the external plan")
    if corpus_counts.get("sealed_queries") != queries.record_count:
        raise ScalableCustodyError("inventory sealed-query count differs from its source")
    return selection


def _validate_secret(secret: bytes) -> bytes:
    if type(secret) is not bytes or len(secret) < 32 or len(set(secret)) < 8:
        raise ScalableCustodyError("HMAC secret must be immutable, diverse, and at least 32 bytes")
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
        KEY_DERIVATION_ALGORITHM,
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


def _family_selection_rank(plan: ScalableCustodyPlan, component_sha256: str) -> str:
    return family_selection_rank(
        corpus=plan.corpus,
        stage=plan.stage,
        selection_seed_sha256=plan.selection_seed_sha256,
        component_sha256=component_sha256,
    )


def _representative_selection_rank(
    plan: ScalableCustodyPlan,
    *,
    component_sha256: str,
    query_id_sha256: str,
) -> str:
    return representative_selection_rank(
        corpus=plan.corpus,
        stage=plan.stage,
        selection_seed_sha256=plan.selection_seed_sha256,
        component_sha256=component_sha256,
        query_id_sha256=query_id_sha256,
    )


def _nested_trial_source_value(source_id: str, nested_index: int) -> str:
    return nested_trial_source_value(source_id, nested_index)


def _select_assignment_families(
    assignments: Mapping[str, _Assignment],
    plan: ScalableCustodyPlan,
) -> _FamilySelection:
    components = sorted(
        {assignment.component_sha256 for assignment in assignments.values()},
        key=lambda value: value.encode("ascii"),
    )
    available_count = len(components)
    if available_count < plan.selected_families:
        raise ScalableCustodyError(
            "selected family count underflows the sealed assignment cohort; "
            f"requested={plan.selected_families}, available={available_count}"
        )
    if available_count != plan.available_families:
        raise ScalableCustodyError(
            "available family count differs from the external plan; "
            f"expected={plan.available_families}, observed={available_count}"
        )
    ranked = sorted(
        components,
        key=lambda component: (
            _family_selection_rank(plan, component),
            component,
        ),
    )
    ordered_selected = tuple(ranked[: plan.selected_families])
    selected = frozenset(ordered_selected)
    if len(selected) != plan.selected_families:
        raise ScalableCustodyError("family selection did not produce the registered count")
    return _FamilySelection(
        components=selected,
        ordered_components=ordered_selected,
        available_count=available_count,
        selected_count=len(selected),
    )


def _content_sha256(corpus: str, title: str, text: str) -> bytes:
    parts = (title, *text.split("\n")) if corpus == "scifact" else (title, text)
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _source_uri(corpus: str, external_id: str) -> str:
    encoded = quote(external_id, safe="")
    if corpus == "scifact":
        return f"scifact://document/{encoded}"
    if corpus == "hotpotqa-fullwiki":
        return f"hotpotqa-fullwiki://title/{encoded}"
    if corpus == "t2-ragbench":
        return f"t2-ragbench://context/{encoded}"
    return f"{quote(corpus, safe='')}://document/{encoded}"


def _write_all(handle: BinaryIO, value: bytes) -> None:
    written = handle.write(value)
    if written != len(value):
        raise ScalableCustodyError("artifact write made incomplete progress")


def _open_output(path: Path) -> BinaryIO:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ScalableCustodyError(f"cannot create output {path.name!r}: {exc}") from exc
    return os.fdopen(descriptor, "wb", closefd=True)


def _create_document_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-8192")
    connection.execute(
        "CREATE TABLE documents ("
        "external_id BLOB PRIMARY KEY, "
        "row_id INTEGER NOT NULL UNIQUE, "
        "content_sha256 BLOB NOT NULL"
        ") WITHOUT ROWID"
    )
    return connection


def _build_document_index(
    root_descriptor: int,
    selection: _Selection,
    plan: ScalableCustodyPlan,
    *,
    connection: sqlite3.Connection,
    target: Path,
) -> tuple[PublishedArtifactPin, str]:
    digest = hashlib.sha256()
    row_order = hashlib.sha256()
    algorithm = DOCUMENT_ROW_ORDER_ALGORITHM.encode("ascii")
    row_order.update(len(algorithm).to_bytes(8, "big"))
    row_order.update(algorithm)
    document_count = 0
    previous_id: bytes | None = None
    with _open_output(target) as handle:
        for source in selection.corpus:
            for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
                row = _closed_mapping(
                    value,
                    fields=_DOCUMENT_FIELDS,
                    label=f"document {source.path}:{line_number}",
                )
                external_id = _require_text("document id", row["id"])
                title = _require_body_text("document title", row["title"])
                text = _require_body_text("document text", row["text"])
                external_bytes = external_id.encode("utf-8", errors="strict")
                if previous_id is not None and external_bytes <= previous_id:
                    raise ScalableCustodyError(
                        "corpus document IDs must be unique and strictly bytewise sorted"
                    )
                content_sha256 = _content_sha256(plan.corpus, title, text)
                try:
                    connection.execute(
                        "INSERT INTO documents(external_id, row_id, content_sha256) "
                        "VALUES (?, ?, ?)",
                        (
                            sqlite3.Binary(external_bytes),
                            document_count,
                            sqlite3.Binary(content_sha256),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ScalableCustodyError("corpus document mapping is not one-to-one") from exc
                _write_all(handle, content_sha256)
                digest.update(content_sha256)
                row_order.update(len(external_bytes).to_bytes(8, "big"))
                row_order.update(external_bytes)
                previous_id = external_bytes
                document_count += 1
                if document_count % 10_000 == 0:
                    connection.commit()
            connection.commit()
        handle.flush()
        os.fsync(handle.fileno())
    if document_count != plan.expected_document_count:
        raise ScalableCustodyError("streamed document count differs from the external plan")
    return (
        PublishedArtifactPin(
            path=PROVENANCE_PATH,
            sha256=digest.hexdigest(),
            byte_count=document_count * PROVENANCE_RECORD_SIZE_BYTES,
            record_count=document_count,
            record_size_bytes=PROVENANCE_RECORD_SIZE_BYTES,
            role="document-content-provenance",
            visibility="online",
        ),
        row_order.hexdigest(),
    )


def _lookup_document(
    connection: sqlite3.Connection,
    *,
    corpus: str,
    external_id: str,
) -> tuple[int, str, str]:
    row = connection.execute(
        "SELECT row_id, content_sha256 FROM documents WHERE external_id = ?",
        (sqlite3.Binary(external_id.encode("utf-8", errors="strict")),),
    ).fetchone()
    if row is None:
        raise ScalableCustodyError(f"label source names unknown document {external_id!r}")
    content = bytes(row[1])
    if len(content) != PROVENANCE_RECORD_SIZE_BYTES:
        raise ScalableCustodyError("document index contains an invalid content digest")
    return int(row[0]), _source_uri(corpus, external_id), f"sha256:{content.hex()}"


def _load_assignments(
    root_descriptor: int,
    source: SourceArtifactPin,
    plan: ScalableCustodyPlan,
) -> dict[str, _Assignment]:
    assignments: dict[str, _Assignment] = {}
    for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
        row = _closed_mapping(
            value,
            fields=_ASSIGNMENT_FIELDS,
            label=f"assignment {line_number}",
        )
        if row["dataset"] != plan.corpus or row["stage"] != plan.stage:
            continue
        if row["schema_version"] != ASSIGNMENT_SCHEMA:
            raise ScalableCustodyError("selected assignment schema differs")
        query_id = _require_text("assignment query_id", row["query_id"])
        _require_sha256("assignment_key_sha256", row["assignment_key_sha256"])
        component = _require_sha256(
            "partition_component_sha256",
            row["partition_component_sha256"],
        )
        text_sha256 = _require_sha256("query_text_sha256", row["query_text_sha256"])
        _require_text("assignment source_split", row["source_split"])
        if row["domain"] is not None:
            _require_text("assignment domain", row["domain"])
        if query_id in assignments:
            raise ScalableCustodyError("selected assignment rows repeat a query")
        assignments[query_id] = _Assignment(
            component_sha256=component,
            query_text_sha256=text_sha256,
        )
    if not assignments:
        raise ScalableCustodyError("assignment source has no selected sealed rows")
    return assignments


def _build_query_key_map(
    root_descriptor: int,
    source: SourceArtifactPin,
    plan: ScalableCustodyPlan,
    *,
    assignments: dict[str, _Assignment],
    family_selection: _FamilySelection,
    secret: bytes,
    target: Path,
) -> tuple[PublishedArtifactPin, tuple[_QueryMaterial, ...], frozenset[str]]:
    previous_id: bytes | None = None
    all_query_ids: set[str] = set()
    representatives: dict[str, _Representative] = {}
    query_digest_sources: dict[str, str] = {}
    for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
        row = _closed_mapping(
            value,
            fields=_QUERY_FIELDS,
            label=f"query {line_number}",
        )
        source_id = _require_text("query id", row["id"])
        text = _require_body_text("query text", row["text"])
        source_id_bytes = source_id.encode("utf-8", errors="strict")
        if previous_id is not None and source_id_bytes <= previous_id:
            raise ScalableCustodyError(
                "sealed query IDs must be unique and strictly bytewise sorted"
            )
        assignment = assignments.pop(source_id, None)
        if assignment is None:
            raise ScalableCustodyError("sealed query lacks its assignment row")
        all_query_ids.add(source_id)
        if _sha256(text.encode("utf-8", errors="strict")) != assignment.query_text_sha256:
            raise ScalableCustodyError("sealed query text differs from its assignment")
        previous_id = source_id_bytes
        if assignment.component_sha256 not in family_selection.components:
            continue
        query_id_sha256 = _sha256(source_id_bytes)
        prior_source = query_digest_sources.setdefault(query_id_sha256, source_id)
        if prior_source != source_id:
            raise ScalableCustodyError("representative query-ID digest collision")
        candidate = _Representative(
            source_id=source_id,
            text=text,
            component_sha256=assignment.component_sha256,
            query_id_sha256=query_id_sha256,
            rank_sha256=_representative_selection_rank(
                plan,
                component_sha256=assignment.component_sha256,
                query_id_sha256=query_id_sha256,
            ),
        )
        prior = representatives.get(assignment.component_sha256)
        if prior is None or (candidate.rank_sha256, candidate.query_id_sha256) < (
            prior.rank_sha256,
            prior.query_id_sha256,
        ):
            representatives[assignment.component_sha256] = candidate
    if assignments:
        raise ScalableCustodyError("selected assignments contain queries absent from the source")
    if set(representatives) != family_selection.components:
        raise ScalableCustodyError("query source does not realize every selected family")

    digest = hashlib.sha256()
    byte_count = 0
    materials: list[_QueryMaterial] = []
    trial_sources: dict[str, str] = {}
    family_sources: dict[str, str] = {}
    with _open_output(target) as handle:
        for component_sha256 in family_selection.ordered_components:
            representative = representatives[component_sha256]
            family_key = _derive_opaque_key(
                secret,
                domain="family",
                key_id=plan.hmac_key_id,
                corpus=plan.corpus,
                stage=plan.stage,
                source_value=component_sha256,
            )
            if family_sources.setdefault(family_key, component_sha256) != component_sha256:
                raise ScalableCustodyError("HMAC family-key collision")
            for nested_index in range(plan.nested_rows_per_family):
                trial_source = _nested_trial_source_value(
                    representative.source_id,
                    nested_index,
                )
                trial_key = _derive_opaque_key(
                    secret,
                    domain="trial",
                    key_id=plan.hmac_key_id,
                    corpus=plan.corpus,
                    stage=plan.stage,
                    source_value=trial_source,
                )
                if trial_sources.setdefault(trial_key, trial_source) != trial_source:
                    raise ScalableCustodyError("HMAC trial-key collision")
                query_row = len(materials)
                online_row = {
                    "corpus": plan.corpus,
                    "family_key": family_key,
                    "nested_index": nested_index,
                    "query_row": query_row,
                    "schema_version": CUSTODY_QUERY_KEY_ROW_SCHEMA,
                    "stage": plan.stage,
                    "text": representative.text,
                    "trial_key": trial_key,
                }
                if set(online_row) != _QUERY_KEY_ROW_FIELDS:
                    raise ScalableCustodyError("custody query-key schema drifted")
                encoded = _canonical_bytes(online_row) + b"\n"
                _write_all(handle, encoded)
                digest.update(encoded)
                byte_count += len(encoded)
                materials.append(
                    _QueryMaterial(
                        source_id=representative.source_id,
                        text=representative.text,
                        query_row=query_row,
                        nested_index=nested_index,
                        trial_key=trial_key,
                        family_key=family_key,
                    )
                )
        handle.flush()
        os.fsync(handle.fileno())
    if len(family_sources) != family_selection.selected_count:
        raise ScalableCustodyError("query-key map omits a selected family")
    if len(materials) != family_selection.selected_count * plan.nested_rows_per_family:
        raise ScalableCustodyError("query-key map has the wrong nested trial count")
    return (
        PublishedArtifactPin(
            path=QUERY_KEY_MAP_PATH,
            sha256=digest.hexdigest(),
            byte_count=byte_count,
            record_count=len(materials),
            role="custody-query-key-map",
            visibility="online",
        ),
        tuple(materials),
        frozenset(all_query_ids),
    )


def _load_relevance(
    root_descriptor: int,
    source: SourceArtifactPin,
    plan: ScalableCustodyPlan,
    *,
    connection: sqlite3.Connection,
    queries: Mapping[str, _QueryMaterial],
    all_query_ids: frozenset[str],
) -> dict[str, set[int]]:
    relevant = {query_id: set() for query_id in queries}
    observed_pairs: set[tuple[str, str]] = set()
    for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
        row = _closed_mapping(value, fields=_QREL_FIELDS, label=f"qrel {line_number}")
        query_id = _require_text("qrel query_id", row["query_id"])
        external_id = _require_text("qrel document_id", row["document_id"])
        relevance = _require_integer("qrel relevance", row["relevance"])
        if query_id not in all_query_ids:
            raise ScalableCustodyError("qrel names an unknown sealed query")
        pair = (query_id, external_id)
        if pair in observed_pairs:
            raise ScalableCustodyError("qrel source repeats a query/document pair")
        observed_pairs.add(pair)
        if query_id not in queries:
            continue
        document_id, _, _ = _lookup_document(
            connection,
            corpus=plan.corpus,
            external_id=external_id,
        )
        if relevance > 0:
            relevant[query_id].add(document_id)
    missing = sorted(query_id for query_id, values in relevant.items() if not values)
    if missing:
        raise ScalableCustodyError(
            f"sealed qrel coverage lacks positive relevance for {len(missing)} queries"
        )
    return relevant


def _metadata_pairs(value: object, *, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ScalableCustodyError(f"{label} must be an array")
    result: list[tuple[str, str]] = []
    for position, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise ScalableCustodyError(f"{label}[{position}] must be a two-string array")
        result.append(
            (
                _require_text(f"{label}[{position}][0]", item[0]),
                _require_text(f"{label}[{position}][1]", item[1]),
            )
        )
    if len({key for key, _ in result}) != len(result):
        raise ScalableCustodyError(f"{label} repeats a key")
    return tuple(result)


def _load_evidence(
    root_descriptor: int,
    source: SourceArtifactPin,
    plan: ScalableCustodyPlan,
    *,
    connection: sqlite3.Connection,
    queries: Mapping[str, _QueryMaterial],
    all_query_ids: frozenset[str],
    relevant: Mapping[str, set[int]],
) -> dict[
    str,
    tuple[str | None, tuple[SealedEvidenceBundle, ...], tuple[tuple[str, str], ...]],
]:
    result: dict[
        str,
        tuple[str | None, tuple[SealedEvidenceBundle, ...], tuple[tuple[str, str], ...]],
    ] = {}
    observed_query_ids: set[str] = set()
    for line_number, value in _iter_canonical_jsonl(root_descriptor, source):
        row = _closed_mapping(
            value,
            fields=_EVIDENCE_ROW_FIELDS,
            label=f"evidence row {line_number}",
        )
        query_id = _require_text("evidence query_id", row["query_id"])
        if query_id not in all_query_ids or query_id in observed_query_ids:
            raise ScalableCustodyError("evidence rows name an unknown or repeated query")
        observed_query_ids.add(query_id)
        if query_id not in queries:
            continue
        answer = row["answer"]
        if answer is not None and not isinstance(answer, str):
            raise ScalableCustodyError("evidence answer must be a string or null")
        bundle_values = row["evidence_bundles"]
        if not isinstance(bundle_values, list) or not bundle_values:
            raise ScalableCustodyError("evidence row must contain at least one bundle")
        bundles: list[SealedEvidenceBundle] = []
        for bundle_position, bundle_value in enumerate(bundle_values):
            bundle_row = _closed_mapping(
                bundle_value,
                fields=_EVIDENCE_BUNDLE_FIELDS,
                label=f"evidence bundle {bundle_position}",
            )
            bundle_id = _require_text("evidence bundle_id", bundle_row["bundle_id"])
            location_values = bundle_row["locations"]
            if not isinstance(location_values, list) or not location_values:
                raise ScalableCustodyError("evidence bundle needs at least one location")
            locations: list[SealedEvidenceLocation] = []
            for location_position, location_value in enumerate(location_values):
                location_row = _closed_mapping(
                    location_value,
                    fields=_EVIDENCE_LOCATION_FIELDS,
                    label=f"evidence location {location_position}",
                )
                external_id = _require_text(
                    "evidence document_id",
                    location_row["document_id"],
                )
                locator = _require_text("evidence locator", location_row["locator"])
                document_id, source_uri, content_hash = _lookup_document(
                    connection,
                    corpus=plan.corpus,
                    external_id=external_id,
                )
                if document_id not in relevant[query_id]:
                    raise ScalableCustodyError(
                        "evidence location lacks a positive relevance judgment"
                    )
                locations.append(
                    SealedEvidenceLocation(
                        document_id=document_id,
                        source_uri=source_uri,
                        locator=locator,
                        content_hash=content_hash,
                    )
                )
            bundles.append(
                SealedEvidenceBundle(
                    bundle_id=bundle_id,
                    locations=tuple(locations),
                )
            )
        result[query_id] = (
            answer,
            tuple(bundles),
            _metadata_pairs(row["label_metadata"], label="evidence label_metadata"),
        )
    if set(result) != set(queries):
        raise ScalableCustodyError("evidence rows do not exactly cover sealed queries")
    return result


def _write_sealed_labels(
    plan: ScalableCustodyPlan,
    *,
    materials: tuple[_QueryMaterial, ...],
    relevant: Mapping[str, set[int]],
    evidence: Mapping[
        str,
        tuple[str | None, tuple[SealedEvidenceBundle, ...], tuple[tuple[str, str], ...]],
    ],
    target: Path,
) -> tuple[PublishedArtifactPin, SealedLabelArtifact]:
    labels: list[SealedTrialLabels] = []
    for material in materials:
        answer: str | None = None
        bundles: tuple[SealedEvidenceBundle, ...] = ()
        metadata: tuple[tuple[str, str], ...] = ()
        if material.source_id in evidence:
            answer, bundles, metadata = evidence[material.source_id]
        labels.append(
            SealedTrialLabels(
                trial_key=material.trial_key,
                family_key=material.family_key,
                answer=answer,
                relevant_document_ids=tuple(relevant[material.source_id]),
                evidence_bundles=bundles,
                label_metadata=metadata,
            )
        )
    artifact = SealedLabelArtifact(
        execution_artifact_sha256=plan.execution_artifact_sha256,
        key_id=plan.hmac_key_id,
        corpus=plan.corpus,
        stage=plan.stage,
        document_count=plan.expected_document_count,
        labels=tuple(labels),
        schema_version=SEALED_LABEL_SCHEMA,
    )
    encoded = artifact.canonical_bytes() + b"\n"
    with _open_output(target) as handle:
        _write_all(handle, encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return (
        PublishedArtifactPin(
            path=SEALED_LABEL_PATH,
            sha256=_sha256(encoded),
            byte_count=len(encoded),
            record_count=len(labels),
            role="sealed-labels",
            visibility="custody",
        ),
        artifact,
    )


def _write_exclusive(path: Path, value: bytes) -> None:
    with _open_output(path) as handle:
        _write_all(handle, value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = _open_root(path, label=f"directory {path}")
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_publish(work: Path, output: Path) -> None:
    if os.path.lexists(output):
        raise ScalableCustodyError("final custody package already exists")
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(work)
    destination = os.fsencode(output)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise ScalableCustodyError("exclusive directory rename is unavailable on macOS")
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
            raise ScalableCustodyError("exclusive directory rename is unavailable on Linux")
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
        raise ScalableCustodyError(f"exclusive directory rename is unsupported on {sys.platform!r}")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ScalableCustodyError("final custody package already exists")
        raise ScalableCustodyError(
            f"cannot publish custody package exclusively: {os.strerror(error_number)}"
        )


def _fingerprint_file(
    root_descriptor: int,
    path: str,
    *,
    maximum_bytes: int | None = None,
) -> tuple[int, str, int, bytes]:
    descriptor, opened = _open_relative_file(root_descriptor, path)
    if maximum_bytes is not None and opened.st_size > maximum_bytes:
        os.close(descriptor)
        raise ScalableCustodyError(f"package artifact {path!r} exceeds its size limit")
    digest = hashlib.sha256()
    byte_count = 0
    newline_count = 0
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            newline_count += chunk.count(b"\n")
            if maximum_bytes is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_signature(after) != _stat_signature(opened) or byte_count != opened.st_size:
            raise ScalableCustodyError(f"package artifact {path!r} changed while read")
        return byte_count, digest.hexdigest(), newline_count, b"".join(chunks)
    finally:
        os.close(descriptor)


def load_scalable_custody_receipt(root: str | Path) -> ScalableCustodyReceipt:
    """Load one canonical custody receipt through a no-follow path walk."""

    package = Path(root)
    descriptor = _open_root(package, label="custody package root")
    try:
        encoded = _read_secure_file(
            descriptor,
            RECEIPT_PATH,
            maximum_bytes=_MAX_CONTROL_BYTES,
            label="scalable custody receipt",
        )
    finally:
        os.close(descriptor)
    value = _decode_json(encoded, label="scalable custody receipt")
    receipt = ScalableCustodyReceipt.from_dict(value)
    if encoded != receipt.canonical_file_bytes():
        raise ScalableCustodyError("scalable custody receipt is not canonical")
    return receipt


def verify_scalable_custody_package(
    root: str | Path,
    *,
    expected_execution_artifact_sha256: str | None = None,
) -> ScalableCustodyReceipt:
    """Verify emitted bytes, schemas, key joins, and fixed-width provenance."""

    package = Path(root)
    receipt = load_scalable_custody_receipt(package)
    if expected_execution_artifact_sha256 is not None and (
        receipt.execution_artifact_sha256
        != _require_sha256(
            "expected_execution_artifact_sha256",
            expected_execution_artifact_sha256,
        )
    ):
        raise ScalableCustodyError("custody receipt names another execution artifact")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for directory, directory_names, file_names in os.walk(package, followlinks=False):
        parent = Path(directory)
        for name in directory_names:
            path = parent / name
            relative = path.relative_to(package).as_posix()
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ScalableCustodyError("custody package contains an unsafe directory")
            observed_directories.add(relative)
        for name in file_names:
            path = parent / name
            relative = path.relative_to(package).as_posix()
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ScalableCustodyError("custody package contains an unsafe file")
            observed_files.add(relative)
    if observed_directories != {"custody", "online"} or observed_files != {
        RECEIPT_PATH,
        QUERY_KEY_MAP_PATH,
        PROVENANCE_PATH,
        SEALED_LABEL_PATH,
    }:
        raise ScalableCustodyError("custody package membership differs")

    descriptor = _open_root(package, label="custody package root")
    encoded_by_path: dict[str, bytes] = {}
    try:
        for artifact in receipt.artifacts:
            maximum = _MAX_CONTROL_BYTES if artifact.path != PROVENANCE_PATH else None
            byte_count, sha256, newlines, encoded = _fingerprint_file(
                descriptor,
                artifact.path,
                maximum_bytes=maximum,
            )
            if byte_count != artifact.byte_count or sha256 != artifact.sha256:
                raise ScalableCustodyError(f"published artifact {artifact.path!r} differs")
            if artifact.path == QUERY_KEY_MAP_PATH and newlines != artifact.record_count:
                raise ScalableCustodyError("custody query-key record count differs")
            if artifact.path == PROVENANCE_PATH and (
                artifact.record_size_bytes != PROVENANCE_RECORD_SIZE_BYTES
                or byte_count != receipt.document_count * PROVENANCE_RECORD_SIZE_BYTES
            ):
                raise ScalableCustodyError("provenance sidecar width differs")
            if encoded:
                encoded_by_path[artifact.path] = encoded
    finally:
        os.close(descriptor)

    online = encoded_by_path[QUERY_KEY_MAP_PATH]
    online_keys: set[str] = set()
    online_families: set[str] = set()
    family_rows: dict[str, list[tuple[int, str, str]]] = {}
    family_sequence: list[str] = []
    rows = online.splitlines(keepends=True)
    for position, line in enumerate(rows):
        value = _decode_json(line[:-1], label=f"online query row {position}")
        row = _closed_mapping(
            value,
            fields=_QUERY_KEY_ROW_FIELDS,
            label="custody query-key row",
        )
        if _canonical_bytes(value) + b"\n" != line:
            raise ScalableCustodyError("custody query-key bytes are not canonical")
        if row["schema_version"] != CUSTODY_QUERY_KEY_ROW_SCHEMA:
            raise ScalableCustodyError("custody query-key schema differs")
        if row["corpus"] != receipt.corpus or row["stage"] != receipt.stage:
            raise ScalableCustodyError("custody query-key corpus binding differs")
        if row["query_row"] != position:
            raise ScalableCustodyError("online query rows are not contiguous")
        trial_key = _require_sha256("online trial_key", row["trial_key"])
        family_key = _require_sha256("online family_key", row["family_key"])
        nested_index = _require_integer("online nested_index", row["nested_index"])
        if nested_index >= receipt.nested_rows_per_family:
            raise ScalableCustodyError("online nested_index is outside the registered range")
        _require_body_text("online query text", row["text"])
        if trial_key in online_keys:
            raise ScalableCustodyError("custody query-key map repeats a trial key")
        online_keys.add(trial_key)
        online_families.add(family_key)
        family_sequence.append(family_key)
        family_rows.setdefault(family_key, []).append((nested_index, row["text"], trial_key))
    if len(rows) != receipt.query_count:
        raise ScalableCustodyError("custody query-key count differs from the receipt")
    if len(online_families) != receipt.selected_family_count:
        raise ScalableCustodyError("custody query-key families differ from the receipt")
    family_blocks = [
        family_key
        for position, family_key in enumerate(family_sequence)
        if position == 0 or family_key != family_sequence[position - 1]
    ]
    if len(family_blocks) != len(online_families):
        raise ScalableCustodyError(
            "custody query-key rows must form one contiguous block per family"
        )
    expected_nested = list(range(receipt.nested_rows_per_family))
    for rows_for_family in family_rows.values():
        if [nested for nested, _, _ in rows_for_family] != expected_nested or len(
            {text for _, text, _ in rows_for_family}
        ) != 1:
            raise ScalableCustodyError(
                "each selected family must have ordered nested rows with one text"
            )

    sealed_encoded = encoded_by_path[SEALED_LABEL_PATH]
    sealed_value = _decode_json(sealed_encoded, label="sealed label artifact")
    sealed = SealedLabelArtifact.from_dict(sealed_value)
    if sealed_encoded != sealed.canonical_bytes() + b"\n":
        raise ScalableCustodyError("sealed label artifact is not canonical")
    if (
        sealed.execution_artifact_sha256 != receipt.execution_artifact_sha256
        or sealed.key_id != receipt.hmac_key_id
        or sealed.corpus != receipt.corpus
        or sealed.stage != receipt.stage
        or sealed.document_count != receipt.document_count
        or len(sealed.labels) != receipt.query_count
        or {row.trial_key for row in sealed.labels} != online_keys
    ):
        raise ScalableCustodyError("sealed labels do not join exactly to the query-key map")
    sealed_by_trial = {row.trial_key: row for row in sealed.labels}
    sealed_pairs = {(row.trial_key, row.family_key) for row in sealed.labels}
    online_pairs = {
        (trial_key, family_key)
        for family_key, rows_for_family in family_rows.items()
        for _, _, trial_key in rows_for_family
    }
    if sealed_pairs != online_pairs:
        raise ScalableCustodyError(
            "sealed trial/family pairs do not join exactly to the query-key map"
        )
    for rows_for_family in family_rows.values():
        payloads: set[bytes] = set()
        for _, _, trial_key in rows_for_family:
            payload = sealed_by_trial[trial_key].to_dict()
            payload.pop("trial_key")
            payload.pop("family_key")
            payloads.add(_canonical_bytes(payload))
        if len(payloads) != 1:
            raise ScalableCustodyError(
                "nested rows in one family must carry identical sealed labels"
            )
    return receipt


def verify_query_trial_key_parity(
    custody_root: str | Path,
    runtime_query_trial_path: str | Path,
    *,
    expected_runtime_sha256: str,
    expected_runtime_byte_count: int,
) -> str:
    """Require exact key/order parity with a verified trial-runtime store.

    The caller first verifies the trial-runtime package against its staged and
    embedding sources.  This second check binds its executable rows to the
    custodian's pre-embedding key map without exposing query identifiers.
    """

    from .trial_runtime import CanonicalQueryTrialRow, TrialRuntimeError

    receipt = verify_scalable_custody_package(custody_root)
    runtime_sha256 = _require_sha256(
        "expected_runtime_sha256",
        expected_runtime_sha256,
    )
    runtime_byte_count = _require_integer(
        "expected_runtime_byte_count",
        expected_runtime_byte_count,
        minimum=1,
    )
    if runtime_byte_count > _MAX_CONTROL_BYTES:
        raise ScalableCustodyError("runtime query/trial store exceeds the parity-check limit")
    runtime_path = Path(runtime_query_trial_path)
    if not runtime_path.is_absolute() or runtime_path.name in {"", ".", ".."}:
        raise ScalableCustodyError("runtime_query_trial_path must be an absolute file path")
    runtime_parent = _open_root(runtime_path.parent, label="runtime query/trial parent")
    try:
        runtime_encoded = _read_secure_file(
            runtime_parent,
            runtime_path.name,
            maximum_bytes=runtime_byte_count,
            label="runtime query/trial store",
        )
    finally:
        os.close(runtime_parent)
    if len(runtime_encoded) != runtime_byte_count or _sha256(runtime_encoded) != runtime_sha256:
        raise ScalableCustodyError("runtime query/trial bytes differ from their pin")

    package = Path(custody_root)
    package_descriptor = _open_root(package, label="custody package root")
    try:
        query_key_pin = next(
            artifact for artifact in receipt.artifacts if artifact.path == QUERY_KEY_MAP_PATH
        )
        key_map_encoded = _read_secure_file(
            package_descriptor,
            QUERY_KEY_MAP_PATH,
            maximum_bytes=query_key_pin.byte_count,
            label="custody query-key map",
        )
    finally:
        os.close(package_descriptor)
    if (
        len(key_map_encoded) != query_key_pin.byte_count
        or _sha256(key_map_encoded) != query_key_pin.sha256
    ):
        raise ScalableCustodyError("custody query-key map differs from its receipt")

    key_lines = key_map_encoded.splitlines(keepends=True)
    runtime_lines = runtime_encoded.splitlines(keepends=True)
    if len(key_lines) != len(runtime_lines) or len(key_lines) != receipt.query_count:
        raise ScalableCustodyError("runtime query/trial count differs from the query-key map")
    for position, (key_line, runtime_line) in enumerate(zip(key_lines, runtime_lines, strict=True)):
        key_value = _decode_json(key_line[:-1], label=f"custody query key {position}")
        key_row = _closed_mapping(
            key_value,
            fields=_QUERY_KEY_ROW_FIELDS,
            label=f"custody query key {position}",
        )
        if _canonical_bytes(key_value) + b"\n" != key_line:
            raise ScalableCustodyError("custody query-key map is not canonical")
        runtime_value = _decode_json(
            runtime_line[:-1],
            label=f"runtime query/trial row {position}",
        )
        try:
            runtime_row = CanonicalQueryTrialRow.from_dict(runtime_value)
        except TrialRuntimeError as exc:
            raise ScalableCustodyError(
                f"runtime query/trial row {position} is invalid: {exc}"
            ) from exc
        if _canonical_bytes(runtime_value) + b"\n" != runtime_line:
            raise ScalableCustodyError("runtime query/trial store is not canonical")
        if (
            runtime_row.query_row != position
            or key_row["query_row"] != position
            or runtime_row.corpus != key_row["corpus"]
            or runtime_row.stage != key_row["stage"]
            or runtime_row.text != key_row["text"]
            or runtime_row.trial_key != key_row["trial_key"]
            or runtime_row.family_key != key_row["family_key"]
        ):
            raise ScalableCustodyError(
                f"runtime query/trial key parity differs at query row {position}"
            )
    return runtime_sha256


def _read_private_absolute_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
    allowed_modes: frozenset[int],
) -> bytes:
    path = _absolute_path(str(path), name=label)
    parent_descriptor = _open_root(path.parent, label=f"{label} parent")
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if (hasattr(os, "geteuid") and parent_metadata.st_uid != os.geteuid()) or stat.S_IMODE(
            parent_metadata.st_mode
        ) & 0o022:
            raise ScalableCustodyError(
                f"{label} parent must be a private custodian-owned directory"
            )
        file_descriptor, file_metadata = _open_relative_file(
            parent_descriptor,
            path.name,
        )
        os.close(file_descriptor)
        if (hasattr(os, "geteuid") and file_metadata.st_uid != os.geteuid()) or stat.S_IMODE(
            file_metadata.st_mode
        ) not in allowed_modes:
            raise ScalableCustodyError(
                f"{label} must be custodian-owned with an approved private mode"
            )
        return _read_secure_file(
            parent_descriptor,
            path.name,
            maximum_bytes=maximum_bytes,
            label=label,
        )
    finally:
        os.close(parent_descriptor)


def load_scalable_custody_config(path: str | Path) -> ScalableCustodyConfig:
    """Load one mode-private, canonical, closed-schema custody configuration."""

    config_path = _absolute_path(str(path), name="custody config path")
    encoded = _read_private_absolute_file(
        config_path,
        maximum_bytes=_MAX_CONTROL_BYTES,
        label="custody config",
        allowed_modes=frozenset({0o400, 0o600}),
    )
    value = _decode_json(encoded, label="custody config")
    config = ScalableCustodyConfig.from_dict(value)
    if encoded != config.canonical_file_bytes():
        raise ScalableCustodyError(
            "custody config must be canonical JSON with exactly one trailing newline"
        )
    return config


def _read_hmac_key(pin: HmacKeyFilePin) -> bytes:
    encoded = _read_private_absolute_file(
        pin.path,
        maximum_bytes=pin.byte_count,
        label="HMAC key file",
        allowed_modes=frozenset({0o400, 0o600}),
    )
    if len(encoded) != pin.byte_count or _sha256(encoded) != pin.sha256:
        raise ScalableCustodyError("HMAC key file differs from its private config pin")
    return _validate_secret(encoded)


def _validate_config_source_pins(config: ScalableCustodyConfig) -> None:
    root_descriptor = _open_root(config.staged_root, label="staged package root")
    try:
        inventory = _load_inventory(root_descriptor, config.plan)
        selection = _select_sources(inventory, config.plan)
    finally:
        os.close(root_descriptor)
    if selection.sources != config.source_artifacts:
        raise ScalableCustodyError(
            "config source_artifacts differ from the pinned staged inventory"
        )


def build_scalable_custody_from_config(
    config_path: str | Path,
    output_root: str | Path,
) -> ScalableCustodyReceipt:
    """Build from one canonical config without serializing HMAC key material."""

    config = load_scalable_custody_config(config_path)
    _validate_config_source_pins(config)
    secret = _read_hmac_key(config.hmac_key)
    try:
        receipt = build_scalable_custody_package(
            config.staged_root,
            output_root,
            plan=config.plan,
            hmac_secret=secret,
        )
    finally:
        del secret
    if receipt.source_artifacts != config.source_artifacts:
        raise ScalableCustodyError("published receipt sources differ from the canonical config")
    return receipt


def build_scalable_custody_package(
    staged_root: str | Path,
    output_root: str | Path,
    *,
    plan: ScalableCustodyPlan,
    hmac_secret: bytes,
) -> ScalableCustodyReceipt:
    """Build and exclusively publish one streaming, label-separated package."""

    if not isinstance(plan, ScalableCustodyPlan):
        raise ScalableCustodyError("plan must be a ScalableCustodyPlan")
    secret = _validate_secret(hmac_secret)
    staged = Path(staged_root)
    output = Path(output_root)
    if not staged.is_absolute() or not output.is_absolute():
        raise ScalableCustodyError("staged_root and output_root must be absolute paths")
    if os.path.lexists(output):
        raise ScalableCustodyError("final custody package already exists")
    try:
        verified = verify_staged_data(staged)
    except (StudyDataError, OSError) as exc:
        raise ScalableCustodyError(f"staged package verification failed: {exc}") from exc
    if verified.inventory_sha256 != plan.staged_inventory_sha256:
        raise ScalableCustodyError("verified staged inventory differs from the plan")
    try:
        parent_metadata = output.parent.lstat()
    except OSError as exc:
        raise ScalableCustodyError(f"cannot inspect output parent: {exc}") from exc
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or stat.S_ISLNK(parent_metadata.st_mode)
        or (hasattr(os, "geteuid") and parent_metadata.st_uid != os.geteuid())
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise ScalableCustodyError("output parent must be a private custodian-owned directory")
    parent_descriptor = _open_root(output.parent, label="custody output parent")
    os.close(parent_descriptor)

    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.custody-", dir=output.parent))
    root_descriptor = _open_root(staged, label="staged package root")
    connection: sqlite3.Connection | None = None
    try:
        inventory = _load_inventory(root_descriptor, plan)
        selection = _select_sources(inventory, plan)
        (work / "online").mkdir(mode=0o700)
        (work / "custody").mkdir(mode=0o700)
        scratch = work / ".scratch"
        scratch.mkdir(mode=0o700)
        connection = _create_document_index(scratch / "documents.sqlite3")
        provenance_pin, row_order_sha256 = _build_document_index(
            root_descriptor,
            selection,
            plan,
            connection=connection,
            target=work / PROVENANCE_PATH,
        )
        assignments = _load_assignments(
            root_descriptor,
            selection.assignments,
            plan,
        )
        family_selection = _select_assignment_families(assignments, plan)
        online_pin, materials, all_query_ids = _build_query_key_map(
            root_descriptor,
            selection.queries,
            plan,
            assignments=assignments,
            family_selection=family_selection,
            secret=secret,
            target=work / QUERY_KEY_MAP_PATH,
        )
        queries = {material.source_id: material for material in materials}
        relevance = _load_relevance(
            root_descriptor,
            selection.qrels,
            plan,
            connection=connection,
            queries=queries,
            all_query_ids=all_query_ids,
        )
        evidence: dict[
            str,
            tuple[
                str | None,
                tuple[SealedEvidenceBundle, ...],
                tuple[tuple[str, str], ...],
            ],
        ] = {}
        if selection.evidence is not None:
            evidence = _load_evidence(
                root_descriptor,
                selection.evidence,
                plan,
                connection=connection,
                queries=queries,
                all_query_ids=all_query_ids,
                relevant=relevance,
            )
        sealed_pin, _ = _write_sealed_labels(
            plan,
            materials=materials,
            relevant=relevance,
            evidence=evidence,
            target=work / SEALED_LABEL_PATH,
        )
        connection.close()
        connection = None
        shutil.rmtree(scratch)
        receipt = ScalableCustodyReceipt(
            corpus=plan.corpus,
            stage=plan.stage,
            staged_inventory_sha256=plan.staged_inventory_sha256,
            execution_artifact_sha256=plan.execution_artifact_sha256,
            hmac_key_id=plan.hmac_key_id,
            document_count=plan.expected_document_count,
            query_count=len(materials),
            available_family_count=family_selection.available_count,
            selected_family_count=family_selection.selected_count,
            selection_seed_sha256=plan.selection_seed_sha256,
            nested_rows_per_family=plan.nested_rows_per_family,
            ordered_document_row_sha256=row_order_sha256,
            source_artifacts=selection.sources,
            artifacts=(online_pin, provenance_pin, sealed_pin),
        )
        _write_exclusive(work / RECEIPT_PATH, receipt.canonical_file_bytes())
        _fsync_directory(work / "online")
        _fsync_directory(work / "custody")
        _fsync_directory(work)
        verify_scalable_custody_package(
            work,
            expected_execution_artifact_sha256=plan.execution_artifact_sha256,
        )
        _exclusive_publish(work, output)
        _fsync_directory(output.parent)
        return receipt
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(work, ignore_errors=True)
        raise
    finally:
        os.close(root_descriptor)


def _load_runtime_query_trial_receipt(
    trial_root: Path,
) -> tuple[object, Path]:
    from .trial_runtime import (
        QUERY_TRIAL_FILENAME,
        QUERY_TRIAL_RECEIPT_FILENAME,
        QueryTrialStoreReceipt,
        TrialRuntimeError,
    )

    root = _absolute_path(str(trial_root), name="trial runtime root")
    descriptor = _open_root(root, label="trial runtime root")
    try:
        metadata = os.fstat(descriptor)
        if (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid()) or stat.S_IMODE(
            metadata.st_mode
        ) & 0o022:
            raise ScalableCustodyError(
                "trial runtime root must be a private runner-owned directory"
            )
        membership = set(os.listdir(descriptor))
        expected = {QUERY_TRIAL_FILENAME, QUERY_TRIAL_RECEIPT_FILENAME}
        if membership != expected:
            raise ScalableCustodyError(
                "trial runtime package membership differs; "
                f"missing={sorted(expected - membership)}, "
                f"unexpected={sorted(membership - expected)}"
            )
        encoded = _read_secure_file(
            descriptor,
            QUERY_TRIAL_RECEIPT_FILENAME,
            maximum_bytes=_MAX_CONTROL_BYTES,
            label="query/trial receipt",
        )
    finally:
        os.close(descriptor)
    value = _decode_json(encoded, label="query/trial receipt")
    try:
        receipt = QueryTrialStoreReceipt.from_dict(value)
    except TrialRuntimeError as exc:
        raise ScalableCustodyError(f"query/trial receipt is invalid: {exc}") from exc
    if encoded != receipt.canonical_file_bytes():
        raise ScalableCustodyError("query/trial receipt is not canonical")
    return receipt, root / QUERY_TRIAL_FILENAME


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.scalable_custody",
        description="Build or verify an inventory-bound scalable custody package.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser(
        "build",
        help="build a no-replace custody package from one canonical config",
    )
    build_parser.add_argument("--config", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    verify_parser = subparsers.add_parser(
        "verify",
        help="verify a published custody package",
    )
    verify_parser.add_argument("--root", required=True, type=Path)
    parity_parser = subparsers.add_parser(
        "verify-query-parity",
        help="compare the custody key map with a verified trial-runtime store",
    )
    parity_parser.add_argument("--custody-root", required=True, type=Path)
    parity_parser.add_argument("--trial-root", required=True, type=Path)
    return parser


def _print_result(value: Mapping[str, object]) -> None:
    print(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "build":
            receipt = build_scalable_custody_from_config(
                arguments.config,
                arguments.output,
            )
            result: dict[str, object] = {
                "available_family_count": receipt.available_family_count,
                "command": "build",
                "corpus": receipt.corpus,
                "document_count": receipt.document_count,
                "output_root": str(arguments.output),
                "query_count": receipt.query_count,
                "receipt_sha256": receipt.receipt_sha256,
                "selected_family_count": receipt.selected_family_count,
            }
        elif arguments.command == "verify":
            receipt = verify_scalable_custody_package(arguments.root)
            result = {
                "available_family_count": receipt.available_family_count,
                "command": "verify",
                "corpus": receipt.corpus,
                "document_count": receipt.document_count,
                "query_count": receipt.query_count,
                "receipt_sha256": receipt.receipt_sha256,
                "selected_family_count": receipt.selected_family_count,
            }
        else:
            runtime_receipt, runtime_path = _load_runtime_query_trial_receipt(arguments.trial_root)
            custody_receipt = verify_scalable_custody_package(arguments.custody_root)
            if (
                runtime_receipt.corpus != custody_receipt.corpus
                or runtime_receipt.stage != custody_receipt.stage
                or runtime_receipt.hmac_key_id != custody_receipt.hmac_key_id
                or runtime_receipt.staged_inventory_sha256
                != custody_receipt.staged_inventory_sha256
            ):
                raise ScalableCustodyError(
                    "trial-runtime receipt differs from the custody identity pins"
                )
            runtime_sha256 = verify_query_trial_key_parity(
                arguments.custody_root,
                runtime_path,
                expected_runtime_sha256=runtime_receipt.query_trial_store_sha256,
                expected_runtime_byte_count=(runtime_receipt.query_trial_store_byte_count),
            )
            result = {
                "command": "verify-query-parity",
                "corpus": custody_receipt.corpus,
                "query_count": custody_receipt.query_count,
                "runtime_query_trial_sha256": runtime_sha256,
            }
    except (ScalableCustodyError, OSError) as exc:
        parser.exit(2, f"scalable-custody: {exc}\n")
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
