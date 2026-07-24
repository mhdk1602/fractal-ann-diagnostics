"""Bounded control artifacts for FullWiki-scale online execution.

The execution plan names immutable data-plane artifacts. It never serializes the
corpus or vector matrices into the control file. The provenance registry keeps
one verified binary sidecar open and reads a single SHA-256 record per lookup.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from .artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactVerificationReceipt,
    DirectoryDigest,
    LocalArtifactSpec,
    VerifiedArtifact,
    digest_directory_tree,
    load_verification_receipt,
    read_secure_regular_file,
    verify_local_artifacts,
    write_exclusive_receipt_bytes,
)
from .policy import policy_document_universe_sha256

SHARDED_EXECUTION_SCHEMA = "fractal-sharded-online-execution-v4"
SHARD_INVENTORY_SCHEMA = "fractal-corpus-shard-inventory-v1"
QUERY_TRIAL_STORE_FORMAT = "fractal-query-trial-store-v1"
RAW_VECTOR_STORE_FORMAT = "raw-c-order-v1"
PROVENANCE_SIDECAR_FORMAT = "raw-sha256-by-document-id-v1"
HNSW_BACKEND = "hnsw"
DOCUMENT_ROW_ORDER = "document-id-ascending"
SHA256_RECORD_BYTES = 32
ONLINE_EXECUTION_PLAN_FILENAME = "sharded-online-execution-plan.json"
EXECUTION_LEAF_RECEIPT_FILENAME = "leaf-verification-receipt.json"
EXECUTION_LEAF_RECEIPT_SCHEMA = "fractal-execution-leaf-verification-v1"

_MAX_PLAN_BYTES = 8 * 1024 * 1024
_MAX_INVENTORY_BYTES = 8 * 1024 * 1024
_MAX_LEAF_RECEIPT_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_COMPONENT_NAMES = frozenset(
    {
        "application",
        "controller",
        "corpus",
        "embedding",
        "index",
        "policy",
    }
)
_PORTABLE_DTYPES = {
    "|i1": 1,
    "|u1": 1,
    "<i2": 2,
    "<u2": 2,
    "<f2": 2,
    "<i4": 4,
    "<u4": 4,
    "<f4": 4,
    "<i8": 8,
    "<u8": 8,
    "<f8": 8,
}

_PIN_FIELDS = frozenset({"artifact_id", "byte_count", "kind", "relative_path", "sha256"})
_SHARD_FIELDS = frozenset(
    {
        "artifact",
        "document_count",
        "first_document_id",
        "record_format",
    }
)
_INVENTORY_FIELDS = frozenset(
    {
        "corpus",
        "document_count",
        "ordered_document_universe_sha256",
        "schema_version",
        "shards",
        "stage",
    }
)
_TRIAL_FIELDS = frozenset({"family_key", "query_record_sha256", "query_row", "trial_key"})
_QUERY_STORE_FIELDS = frozenset({"artifact", "receipt", "record_count", "record_format"})
_VECTOR_STORE_FIELDS = frozenset(
    {
        "artifact",
        "document_universe_sha256",
        "dtype",
        "role",
        "row_order",
        "shape",
        "storage_format",
    }
)
_SIDECAR_FIELDS = frozenset(
    {
        "artifact",
        "document_universe_sha256",
        "record_count",
        "record_size_bytes",
        "row_order",
        "storage_format",
    }
)
_INDEX_FIELDS = frozenset(
    {
        "artifact",
        "backend",
        "document_count",
        "document_universe_sha256",
        "format_revision",
        "source_vector_sha256",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "active_vector_store",
        "corpus",
        "corpus_shard_inventory",
        "current_truth_vector_store",
        "document_count",
        "hnsw_index",
        "key_id",
        "ordered_document_universe_sha256",
        "permutation_seed",
        "provenance_sha256_sidecar",
        "query_partition_audit_sha256",
        "query_trial_store",
        "schema_version",
        "stage",
        "trials",
    }
)
_LEAF_RECEIPT_FIELDS = frozenset(
    {
        "artifacts",
        "plan_file_sha256",
        "plan_sha256",
        "schema_version",
    }
)

ArtifactKind = Literal["file", "directory"]
VectorRole = Literal["active-migration", "current-exact-truth"]


class ScalableExecutionError(ValueError):
    """Raised when a sharded execution control or pinned artifact is inadmissible."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScalableExecutionError("control object must be finite canonical JSON") from exc


def _closed_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScalableExecutionError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ScalableExecutionError(f"{label} keys must be strings")
    observed = set(value)
    missing = fields - observed
    unknown = observed - fields
    if missing or unknown:
        raise ScalableExecutionError(
            f"{label} fields differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _parse_json_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ScalableExecutionError(f"{label} contains duplicate key {key!r}")
            parsed[key] = value
        return parsed

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ScalableExecutionError(f"{label} contains non-finite number {value!r}")
        return parsed

    def reject_nonfinite(value: str) -> None:
        raise ScalableExecutionError(f"{label} contains non-finite number {value!r}")

    try:
        text = encoded.decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_float=parse_float,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise ScalableExecutionError(f"{label} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ScalableExecutionError(f"{label} must be valid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ScalableExecutionError(f"{label} must contain one JSON object")
    return payload


def _as_canonical_file_bytes(
    value: bytes | str,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    if isinstance(value, str):
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ScalableExecutionError(f"{label} must be valid UTF-8") from exc
    elif isinstance(value, bytes):
        encoded = value
    else:
        raise ScalableExecutionError(f"{label} must be bytes or text")
    if not encoded:
        raise ScalableExecutionError(f"{label} cannot be empty")
    if len(encoded) > max_bytes:
        raise ScalableExecutionError(f"{label} exceeds {max_bytes} bytes")
    return encoded


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ScalableExecutionError(f"{name} must be a non-empty canonical string")
    if unicodedata.normalize("NFC", value) != value:
        raise ScalableExecutionError(f"{name} must use NFC Unicode normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ScalableExecutionError(f"{name} cannot contain control characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ScalableExecutionError(f"{name} must be valid UTF-8") from exc
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ScalableExecutionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ScalableExecutionError(f"{name} must be an integer {qualifier}")
    return value


def _relative_path(name: str, value: object) -> str:
    text = _require_text(name, value)
    if chr(92) in text:
        raise ScalableExecutionError(f"{name} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or str(path) != text:
        raise ScalableExecutionError(f"{name} must be a canonical relative POSIX path")
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ScalableExecutionError(f"{name} cannot contain dot or parent traversal")
    return text


def _require_json_array(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise ScalableExecutionError(f"{name} must be a JSON array")
    return value


@dataclass(frozen=True)
class ImmutableArtifactPin:
    """One local artifact identity with no mutable locator or placeholder digest."""

    artifact_id: str
    relative_path: str
    kind: ArtifactKind
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _require_text("artifact_id", self.artifact_id)
        object.__setattr__(
            self,
            "relative_path",
            _relative_path("relative_path", self.relative_path),
        )
        if self.kind not in {"file", "directory"}:
            raise ScalableExecutionError("artifact kind must be 'file' or 'directory'")
        _require_integer("artifact byte_count", self.byte_count)
        _require_sha256("artifact sha256", self.sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "byte_count": self.byte_count,
            "kind": self.kind,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ImmutableArtifactPin:
        row = _closed_mapping(payload, fields=_PIN_FIELDS, label="artifact pin")
        return cls(
            artifact_id=row["artifact_id"],
            relative_path=row["relative_path"],
            kind=row["kind"],
            byte_count=row["byte_count"],
            sha256=row["sha256"],
        )


@dataclass(frozen=True)
class CorpusShard:
    """A contiguous document-ID interval stored outside the inventory."""

    artifact: ImmutableArtifactPin
    first_document_id: int
    document_count: int
    record_format: str = "canonical-jsonl-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ImmutableArtifactPin):
            raise ScalableExecutionError("shard artifact must be an immutable pin")
        if self.artifact.kind != "file":
            raise ScalableExecutionError("corpus shards must be file artifacts")
        _require_integer("first_document_id", self.first_document_id)
        _require_integer("shard document_count", self.document_count, minimum=1)
        if self.record_format != "canonical-jsonl-v1":
            raise ScalableExecutionError("shard record_format must equal 'canonical-jsonl-v1'")

    @property
    def stop_document_id(self) -> int:
        return self.first_document_id + self.document_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "document_count": self.document_count,
            "first_document_id": self.first_document_id,
            "record_format": self.record_format,
        }

    @classmethod
    def from_dict(cls, payload: object) -> CorpusShard:
        row = _closed_mapping(payload, fields=_SHARD_FIELDS, label="corpus shard")
        return cls(
            artifact=ImmutableArtifactPin.from_dict(row["artifact"]),
            first_document_id=row["first_document_id"],
            document_count=row["document_count"],
            record_format=row["record_format"],
        )


@dataclass(frozen=True)
class CorpusShardInventory:
    """Canonical shard boundaries and file pins, without document records."""

    corpus: str
    stage: str
    document_count: int
    ordered_document_universe_sha256: str
    shards: tuple[CorpusShard, ...]
    schema_version: str = SHARD_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        _require_text("inventory corpus", self.corpus)
        _require_text("inventory stage", self.stage)
        _require_integer("inventory document_count", self.document_count, minimum=1)
        _require_sha256(
            "inventory ordered_document_universe_sha256",
            self.ordered_document_universe_sha256,
        )
        if self.schema_version != SHARD_INVENTORY_SCHEMA:
            raise ScalableExecutionError(
                f"inventory schema_version must equal {SHARD_INVENTORY_SCHEMA!r}"
            )
        shards = tuple(self.shards)
        if not shards or not all(isinstance(row, CorpusShard) for row in shards):
            raise ScalableExecutionError("shards must contain corpus-shard records")
        shards = tuple(sorted(shards, key=lambda row: row.first_document_id))
        expected_start = 0
        for position, shard in enumerate(shards):
            if shard.first_document_id != expected_start:
                raise ScalableExecutionError(
                    "shards must form one contiguous document-ID partition; "
                    f"shard {position} starts at {shard.first_document_id}, "
                    f"expected {expected_start}"
                )
            expected_start = shard.stop_document_id
        if expected_start != self.document_count:
            raise ScalableExecutionError("shard counts do not cover inventory document_count")
        identifiers = [row.artifact.artifact_id for row in shards]
        paths = [row.artifact.relative_path for row in shards]
        if len(identifiers) != len(set(identifiers)):
            raise ScalableExecutionError("shard inventory repeats an artifact ID")
        if len(paths) != len(set(paths)):
            raise ScalableExecutionError("shard inventory repeats an artifact path")
        object.__setattr__(self, "shards", shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "document_count": self.document_count,
            "ordered_document_universe_sha256": self.ordered_document_universe_sha256,
            "schema_version": self.schema_version,
            "shards": [row.to_dict() for row in self.shards],
            "stage": self.stage,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: object) -> CorpusShardInventory:
        row = _closed_mapping(
            payload,
            fields=_INVENTORY_FIELDS,
            label="corpus shard inventory",
        )
        shards = _require_json_array("corpus shard inventory shards", row["shards"])
        return cls(
            corpus=row["corpus"],
            stage=row["stage"],
            document_count=row["document_count"],
            ordered_document_universe_sha256=row["ordered_document_universe_sha256"],
            shards=tuple(CorpusShard.from_dict(item) for item in shards),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class OpaqueTrialRow:
    """Small query binding; query text and vectors remain in the pinned store."""

    trial_key: str
    family_key: str
    query_row: int
    query_record_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("trial_key", self.trial_key)
        _require_sha256("family_key", self.family_key)
        _require_integer("query_row", self.query_row)
        _require_sha256("query_record_sha256", self.query_record_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_key": self.family_key,
            "query_record_sha256": self.query_record_sha256,
            "query_row": self.query_row,
            "trial_key": self.trial_key,
        }

    @classmethod
    def from_dict(cls, payload: object) -> OpaqueTrialRow:
        row = _closed_mapping(payload, fields=_TRIAL_FIELDS, label="opaque trial row")
        return cls(
            trial_key=row["trial_key"],
            family_key=row["family_key"],
            query_row=row["query_row"],
            query_record_sha256=row["query_record_sha256"],
        )


@dataclass(frozen=True)
class QueryTrialStoreDescriptor:
    """Pins for the external query/trial store and its typed build receipt."""

    artifact: ImmutableArtifactPin
    receipt: ImmutableArtifactPin
    record_count: int
    record_format: str = QUERY_TRIAL_STORE_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ImmutableArtifactPin):
            raise ScalableExecutionError("query store artifact must be an immutable pin")
        if self.artifact.kind != "file":
            raise ScalableExecutionError("query/trial store must be a file artifact")
        if not isinstance(self.receipt, ImmutableArtifactPin):
            raise ScalableExecutionError("query store receipt must be an immutable pin")
        if self.receipt.kind != "file":
            raise ScalableExecutionError("query/trial receipt must be a file artifact")
        if (
            self.receipt.artifact_id == self.artifact.artifact_id
            or self.receipt.relative_path == self.artifact.relative_path
        ):
            raise ScalableExecutionError(
                "query/trial store and receipt need distinct artifact identities"
            )
        _require_integer("query store record_count", self.record_count, minimum=1)
        if self.record_format != QUERY_TRIAL_STORE_FORMAT:
            raise ScalableExecutionError(
                f"query store record_format must equal {QUERY_TRIAL_STORE_FORMAT!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "receipt": self.receipt.to_dict(),
            "record_count": self.record_count,
            "record_format": self.record_format,
        }

    @classmethod
    def from_dict(cls, payload: object) -> QueryTrialStoreDescriptor:
        row = _closed_mapping(
            payload,
            fields=_QUERY_STORE_FIELDS,
            label="query/trial store descriptor",
        )
        return cls(
            artifact=ImmutableArtifactPin.from_dict(row["artifact"]),
            receipt=ImmutableArtifactPin.from_dict(row["receipt"]),
            record_count=row["record_count"],
            record_format=row["record_format"],
        )


@dataclass(frozen=True)
class VectorStoreDescriptor:
    """Raw matrix identity with explicit semantics for the vector epoch."""

    artifact: ImmutableArtifactPin
    role: VectorRole
    dtype: str
    shape: tuple[int, int]
    document_universe_sha256: str
    row_order: str = DOCUMENT_ROW_ORDER
    storage_format: str = RAW_VECTOR_STORE_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ImmutableArtifactPin):
            raise ScalableExecutionError("vector store artifact must be an immutable pin")
        if self.artifact.kind != "file":
            raise ScalableExecutionError("raw vector store must be a file artifact")
        if self.role not in {"active-migration", "current-exact-truth"}:
            raise ScalableExecutionError(
                "vector role must be 'active-migration' or 'current-exact-truth'"
            )
        if self.dtype not in _PORTABLE_DTYPES:
            raise ScalableExecutionError(
                f"dtype must be an explicit portable value in {sorted(_PORTABLE_DTYPES)}"
            )
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.shape
            )
        ):
            raise ScalableExecutionError("vector shape must contain two positive integers")
        _require_sha256(
            "vector document_universe_sha256",
            self.document_universe_sha256,
        )
        if self.row_order != DOCUMENT_ROW_ORDER:
            raise ScalableExecutionError(f"vector row_order must equal {DOCUMENT_ROW_ORDER!r}")
        if self.storage_format != RAW_VECTOR_STORE_FORMAT:
            raise ScalableExecutionError(
                f"vector storage_format must equal {RAW_VECTOR_STORE_FORMAT!r}"
            )
        expected_bytes = self.shape[0] * self.shape[1] * _PORTABLE_DTYPES[self.dtype]
        if self.artifact.byte_count != expected_bytes:
            raise ScalableExecutionError(
                "vector byte_count does not equal shape product times dtype width"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "document_universe_sha256": self.document_universe_sha256,
            "dtype": self.dtype,
            "role": self.role,
            "row_order": self.row_order,
            "shape": list(self.shape),
            "storage_format": self.storage_format,
        }

    @classmethod
    def from_dict(cls, payload: object) -> VectorStoreDescriptor:
        row = _closed_mapping(
            payload,
            fields=_VECTOR_STORE_FIELDS,
            label="vector store descriptor",
        )
        shape = _require_json_array("vector store shape", row["shape"])
        if len(shape) != 2:
            raise ScalableExecutionError("vector store shape must have two entries")
        return cls(
            artifact=ImmutableArtifactPin.from_dict(row["artifact"]),
            role=row["role"],
            dtype=row["dtype"],
            shape=(shape[0], shape[1]),
            document_universe_sha256=row["document_universe_sha256"],
            row_order=row["row_order"],
            storage_format=row["storage_format"],
        )


@dataclass(frozen=True)
class ProvenanceSidecarDescriptor:
    """Fixed-width content digests in document-ID order."""

    artifact: ImmutableArtifactPin
    record_count: int
    document_universe_sha256: str
    record_size_bytes: int = SHA256_RECORD_BYTES
    row_order: str = DOCUMENT_ROW_ORDER
    storage_format: str = PROVENANCE_SIDECAR_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ImmutableArtifactPin):
            raise ScalableExecutionError("sidecar artifact must be an immutable pin")
        if self.artifact.kind != "file":
            raise ScalableExecutionError("provenance sidecar must be a file artifact")
        _require_integer("sidecar record_count", self.record_count, minimum=1)
        _require_sha256(
            "sidecar document_universe_sha256",
            self.document_universe_sha256,
        )
        if self.record_size_bytes != SHA256_RECORD_BYTES:
            raise ScalableExecutionError(
                f"sidecar record_size_bytes must equal {SHA256_RECORD_BYTES}"
            )
        if self.row_order != DOCUMENT_ROW_ORDER:
            raise ScalableExecutionError(f"sidecar row_order must equal {DOCUMENT_ROW_ORDER!r}")
        if self.storage_format != PROVENANCE_SIDECAR_FORMAT:
            raise ScalableExecutionError(
                f"sidecar storage_format must equal {PROVENANCE_SIDECAR_FORMAT!r}"
            )
        expected_bytes = self.record_count * self.record_size_bytes
        if self.artifact.byte_count != expected_bytes:
            raise ScalableExecutionError(
                "sidecar byte_count must equal record_count times record_size_bytes"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "document_universe_sha256": self.document_universe_sha256,
            "record_count": self.record_count,
            "record_size_bytes": self.record_size_bytes,
            "row_order": self.row_order,
            "storage_format": self.storage_format,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ProvenanceSidecarDescriptor:
        row = _closed_mapping(
            payload,
            fields=_SIDECAR_FIELDS,
            label="provenance sidecar descriptor",
        )
        return cls(
            artifact=ImmutableArtifactPin.from_dict(row["artifact"]),
            record_count=row["record_count"],
            document_universe_sha256=row["document_universe_sha256"],
            record_size_bytes=row["record_size_bytes"],
            row_order=row["row_order"],
            storage_format=row["storage_format"],
        )


@dataclass(frozen=True)
class IndexArtifactDescriptor:
    """Index pin bound to the active vector bytes and ordered document universe."""

    artifact: ImmutableArtifactPin
    document_count: int
    document_universe_sha256: str
    source_vector_sha256: str
    format_revision: str
    backend: str = HNSW_BACKEND

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ImmutableArtifactPin):
            raise ScalableExecutionError("index artifact must be an immutable pin")
        _require_integer("index document_count", self.document_count, minimum=1)
        _require_sha256(
            "index document_universe_sha256",
            self.document_universe_sha256,
        )
        _require_sha256("index source_vector_sha256", self.source_vector_sha256)
        _require_text("index format_revision", self.format_revision)
        if self.backend != HNSW_BACKEND:
            raise ScalableExecutionError(f"index backend must equal {HNSW_BACKEND!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "backend": self.backend,
            "document_count": self.document_count,
            "document_universe_sha256": self.document_universe_sha256,
            "format_revision": self.format_revision,
            "source_vector_sha256": self.source_vector_sha256,
        }

    @classmethod
    def from_dict(cls, payload: object) -> IndexArtifactDescriptor:
        row = _closed_mapping(
            payload,
            fields=_INDEX_FIELDS,
            label="index artifact descriptor",
        )
        return cls(
            artifact=ImmutableArtifactPin.from_dict(row["artifact"]),
            document_count=row["document_count"],
            document_universe_sha256=row["document_universe_sha256"],
            source_vector_sha256=row["source_vector_sha256"],
            format_revision=row["format_revision"],
            backend=row["backend"],
        )


@dataclass(frozen=True)
class ShardedOnlineExecutionPlan:
    """Bounded execution control for a corpus that remains in sharded storage."""

    key_id: str
    corpus: str
    stage: str
    document_count: int
    ordered_document_universe_sha256: str
    permutation_seed: int
    trials: tuple[OpaqueTrialRow, ...]
    query_partition_audit_sha256: str
    corpus_shard_inventory: ImmutableArtifactPin
    query_trial_store: QueryTrialStoreDescriptor
    active_vector_store: VectorStoreDescriptor
    current_truth_vector_store: VectorStoreDescriptor
    provenance_sha256_sidecar: ProvenanceSidecarDescriptor
    hnsw_index: IndexArtifactDescriptor
    schema_version: str = SHARDED_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        _require_text("key_id", self.key_id)
        _require_text("corpus", self.corpus)
        _require_text("stage", self.stage)
        _require_integer("document_count", self.document_count, minimum=1)
        _require_integer("permutation_seed", self.permutation_seed)
        _require_sha256(
            "ordered_document_universe_sha256",
            self.ordered_document_universe_sha256,
        )
        _require_sha256(
            "query_partition_audit_sha256",
            self.query_partition_audit_sha256,
        )
        if self.schema_version != SHARDED_EXECUTION_SCHEMA:
            raise ScalableExecutionError(f"schema_version must equal {SHARDED_EXECUTION_SCHEMA!r}")
        if (
            not isinstance(self.corpus_shard_inventory, ImmutableArtifactPin)
            or self.corpus_shard_inventory.kind != "file"
        ):
            raise ScalableExecutionError("corpus_shard_inventory must be a pinned file artifact")
        for name, expected_type in (
            ("query_trial_store", QueryTrialStoreDescriptor),
            ("active_vector_store", VectorStoreDescriptor),
            ("current_truth_vector_store", VectorStoreDescriptor),
            ("provenance_sha256_sidecar", ProvenanceSidecarDescriptor),
            ("hnsw_index", IndexArtifactDescriptor),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise ScalableExecutionError(f"{name} has the wrong descriptor type")

        trials = tuple(self.trials)
        if not trials or not all(isinstance(row, OpaqueTrialRow) for row in trials):
            raise ScalableExecutionError("trials must contain opaque trial rows")
        trials = tuple(sorted(trials, key=lambda row: row.trial_key))
        trial_keys = [row.trial_key for row in trials]
        family_rows = [(row.family_key, row.query_row) for row in trials]
        query_rows = [row.query_row for row in trials]
        if len(trial_keys) != len(set(trial_keys)):
            raise ScalableExecutionError("execution plan repeats a trial key")
        if len(family_rows) != len(set(family_rows)):
            raise ScalableExecutionError("execution plan repeats a family/query-row pairing")
        if len(query_rows) != len(set(query_rows)):
            raise ScalableExecutionError("execution plan repeats a query row")
        if self.query_trial_store.record_count != len(trials):
            raise ScalableExecutionError(
                "query/trial store record_count must equal the plan trial count"
            )
        if set(query_rows) != set(range(len(trials))):
            raise ScalableExecutionError(
                "query rows must be a contiguous permutation starting at zero"
            )

        active = self.active_vector_store
        truth = self.current_truth_vector_store
        if active.role != "active-migration":
            raise ScalableExecutionError("active_vector_store must carry role 'active-migration'")
        if truth.role != "current-exact-truth":
            raise ScalableExecutionError(
                "current_truth_vector_store must carry role 'current-exact-truth'"
            )
        if active.shape[0] != self.document_count or truth.shape[0] != self.document_count:
            raise ScalableExecutionError("both vector stores must have document_count matrix rows")
        if active.shape[1] != truth.shape[1]:
            raise ScalableExecutionError(
                "active and current-truth vectors must share an embedding dimension"
            )
        if (
            active.artifact.artifact_id == truth.artifact.artifact_id
            or active.artifact.relative_path == truth.artifact.relative_path
        ):
            raise ScalableExecutionError(
                "active and current-truth vectors need distinct artifact identities"
            )

        universe_bound = (
            active.document_universe_sha256,
            truth.document_universe_sha256,
            self.provenance_sha256_sidecar.document_universe_sha256,
            self.hnsw_index.document_universe_sha256,
        )
        if any(value != self.ordered_document_universe_sha256 for value in universe_bound):
            raise ScalableExecutionError(
                "every row-addressed artifact must bind the ordered document universe"
            )
        if self.provenance_sha256_sidecar.record_count != self.document_count:
            raise ScalableExecutionError(
                "provenance sidecar record_count must equal document_count"
            )
        if self.hnsw_index.document_count != self.document_count:
            raise ScalableExecutionError("HNSW index document_count must equal plan document_count")
        if self.hnsw_index.source_vector_sha256 != active.artifact.sha256:
            raise ScalableExecutionError(
                "HNSW index must bind the active migration vector artifact"
            )

        direct_pins = self.direct_artifact_pins
        identifiers = [pin.artifact_id for pin in direct_pins]
        paths = [pin.relative_path for pin in direct_pins]
        if len(identifiers) != len(set(identifiers)):
            raise ScalableExecutionError("execution plan repeats an artifact ID")
        if len(paths) != len(set(paths)):
            raise ScalableExecutionError("execution plan repeats an artifact path")
        object.__setattr__(self, "trials", trials)

    @property
    def direct_artifact_pins(self) -> tuple[ImmutableArtifactPin, ...]:
        return (
            self.corpus_shard_inventory,
            self.query_trial_store.artifact,
            self.query_trial_store.receipt,
            self.active_vector_store.artifact,
            self.current_truth_vector_store.artifact,
            self.provenance_sha256_sidecar.artifact,
            self.hnsw_index.artifact,
        )

    @property
    def trial_keys(self) -> tuple[str, ...]:
        return tuple(row.trial_key for row in self.trials)

    @property
    def document_universe_sha256(self) -> str:
        """Alias used by authorization and audit interfaces."""

        return self.ordered_document_universe_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_vector_store": self.active_vector_store.to_dict(),
            "corpus": self.corpus,
            "corpus_shard_inventory": self.corpus_shard_inventory.to_dict(),
            "current_truth_vector_store": self.current_truth_vector_store.to_dict(),
            "document_count": self.document_count,
            "hnsw_index": self.hnsw_index.to_dict(),
            "key_id": self.key_id,
            "ordered_document_universe_sha256": (self.ordered_document_universe_sha256),
            "permutation_seed": self.permutation_seed,
            "provenance_sha256_sidecar": self.provenance_sha256_sidecar.to_dict(),
            "query_partition_audit_sha256": self.query_partition_audit_sha256,
            "query_trial_store": self.query_trial_store.to_dict(),
            "schema_version": self.schema_version,
            "stage": self.stage,
            "trials": [row.to_dict() for row in self.trials],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def artifact_sha256(self) -> str:
        """Logical execution digest, matching the inline artifact convention."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: object) -> ShardedOnlineExecutionPlan:
        row = _closed_mapping(
            payload,
            fields=_PLAN_FIELDS,
            label="sharded online execution plan",
        )
        trials = _require_json_array("execution plan trials", row["trials"])
        return cls(
            key_id=row["key_id"],
            corpus=row["corpus"],
            stage=row["stage"],
            document_count=row["document_count"],
            ordered_document_universe_sha256=row["ordered_document_universe_sha256"],
            permutation_seed=row["permutation_seed"],
            trials=tuple(OpaqueTrialRow.from_dict(item) for item in trials),
            query_partition_audit_sha256=row["query_partition_audit_sha256"],
            corpus_shard_inventory=ImmutableArtifactPin.from_dict(row["corpus_shard_inventory"]),
            query_trial_store=QueryTrialStoreDescriptor.from_dict(row["query_trial_store"]),
            active_vector_store=VectorStoreDescriptor.from_dict(row["active_vector_store"]),
            current_truth_vector_store=VectorStoreDescriptor.from_dict(
                row["current_truth_vector_store"]
            ),
            provenance_sha256_sidecar=ProvenanceSidecarDescriptor.from_dict(
                row["provenance_sha256_sidecar"]
            ),
            hnsw_index=IndexArtifactDescriptor.from_dict(row["hnsw_index"]),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class ExecutionLeafVerificationReceipt:
    """Plan-bound verification rows for the package leaves, without C1."""

    plan_sha256: str
    plan_file_sha256: str
    artifacts: tuple[VerifiedArtifact, ...]
    schema_version: str = EXECUTION_LEAF_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("leaf receipt plan_sha256", self.plan_sha256)
        _require_sha256("leaf receipt plan_file_sha256", self.plan_file_sha256)
        if self.schema_version != EXECUTION_LEAF_RECEIPT_SCHEMA:
            raise ScalableExecutionError(
                "leaf receipt schema_version differs from the registered schema"
            )
        artifacts = tuple(
            sorted(
                self.artifacts,
                key=lambda row: (
                    row.artifact_id.encode("utf-8"),
                    row.relative_path.encode("utf-8"),
                ),
            )
        )
        if not artifacts or not all(isinstance(row, VerifiedArtifact) for row in artifacts):
            raise ScalableExecutionError(
                "leaf receipt artifacts must contain verified artifact rows"
            )
        identifiers = [row.artifact_id for row in artifacts]
        paths = [row.relative_path for row in artifacts]
        if len(identifiers) != len(set(identifiers)):
            raise ScalableExecutionError("leaf receipt repeats an artifact ID")
        if len(paths) != len(set(paths)):
            raise ScalableExecutionError("leaf receipt repeats an artifact path")
        if any(row.kind != "file" or not row.exact for row in artifacts):
            raise ScalableExecutionError("execution package leaves must be exact file artifacts")
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [row.to_dict() for row in self.artifacts],
            "plan_file_sha256": self.plan_file_sha256,
            "plan_sha256": self.plan_sha256,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    def as_artifact_verification_receipt(self) -> ArtifactVerificationReceipt:
        """Adapt leaf rows to the existing no-follow artifact readers."""

        return ArtifactVerificationReceipt(
            manifest_sha256=self.plan_sha256,
            artifacts=self.artifacts,
        )

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: object) -> ExecutionLeafVerificationReceipt:
        row = _closed_mapping(
            payload,
            fields=_LEAF_RECEIPT_FIELDS,
            label="execution leaf verification receipt",
        )
        artifacts = _require_json_array("execution leaf receipt artifacts", row["artifacts"])
        try:
            verified = tuple(VerifiedArtifact.from_dict(value) for value in artifacts)
        except ArtifactIntegrityError as exc:
            raise ScalableExecutionError(f"invalid execution leaf receipt row: {exc}") from exc
        return cls(
            plan_sha256=row["plan_sha256"],
            plan_file_sha256=row["plan_file_sha256"],
            artifacts=verified,
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class OnlineExecutionPackage:
    """Verified outer tree plus its logical plan and exact leaf closure."""

    root: Path
    tree_sha256: str
    plan: ShardedOnlineExecutionPlan
    inventory: CorpusShardInventory
    leaf_receipt: ExecutionLeafVerificationReceipt

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            raise ScalableExecutionError("package root must be absolute")
        _require_sha256("package tree_sha256", self.tree_sha256)
        if not isinstance(self.plan, ShardedOnlineExecutionPlan):
            raise ScalableExecutionError("package plan has the wrong type")
        if self.tree_sha256 == self.plan.artifact_sha256:
            raise ScalableExecutionError(
                "package tree SHA-256 must differ from its logical plan revision"
            )
        if not isinstance(self.inventory, CorpusShardInventory):
            raise ScalableExecutionError("package inventory has the wrong type")
        if not isinstance(self.leaf_receipt, ExecutionLeafVerificationReceipt):
            raise ScalableExecutionError("package leaf receipt has the wrong type")

    @property
    def revision(self) -> str:
        return f"sha256:{self.plan.artifact_sha256}"


def loads_corpus_shard_inventory(payload: bytes | str) -> CorpusShardInventory:
    """Parse one bounded canonical inventory with exactly one trailing newline."""

    encoded = _as_canonical_file_bytes(
        payload,
        label="corpus shard inventory",
        max_bytes=_MAX_INVENTORY_BYTES,
    )
    parsed = _parse_json_object(encoded, label="corpus shard inventory")
    inventory = CorpusShardInventory.from_dict(parsed)
    if encoded != inventory.canonical_file_bytes():
        raise ScalableExecutionError("corpus shard inventory bytes are not canonical")
    return inventory


def load_corpus_shard_inventory(path: str | Path) -> CorpusShardInventory:
    """Read one inventory without following links or accepting a hard link."""

    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=_MAX_INVENTORY_BYTES,
            label="corpus shard inventory",
        )
    except ArtifactIntegrityError as exc:
        raise ScalableExecutionError(f"cannot read corpus shard inventory safely: {exc}") from exc
    return loads_corpus_shard_inventory(encoded)


def write_corpus_shard_inventory(
    inventory: CorpusShardInventory,
    target: str | Path,
) -> None:
    """Write one canonical inventory through exclusive, no-follow creation."""

    if not isinstance(inventory, CorpusShardInventory):
        raise ScalableExecutionError("inventory must be a CorpusShardInventory")
    encoded = inventory.canonical_file_bytes()
    if len(encoded) > _MAX_INVENTORY_BYTES:
        raise ScalableExecutionError(f"corpus shard inventory exceeds {_MAX_INVENTORY_BYTES} bytes")
    try:
        write_exclusive_receipt_bytes(encoded, target)
    except ArtifactIntegrityError as exc:
        raise ScalableExecutionError(f"cannot write corpus shard inventory safely: {exc}") from exc


def loads_sharded_online_execution_plan(
    payload: bytes | str,
) -> ShardedOnlineExecutionPlan:
    """Parse one bounded canonical plan with exactly one trailing newline."""

    encoded = _as_canonical_file_bytes(
        payload,
        label="sharded online execution plan",
        max_bytes=_MAX_PLAN_BYTES,
    )
    parsed = _parse_json_object(encoded, label="sharded online execution plan")
    plan = ShardedOnlineExecutionPlan.from_dict(parsed)
    if encoded != plan.canonical_file_bytes():
        raise ScalableExecutionError("sharded online execution plan bytes are not canonical")
    return plan


def load_sharded_online_execution_plan(
    path: str | Path,
) -> ShardedOnlineExecutionPlan:
    """Read one plan without following links or accepting a hard link."""

    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=_MAX_PLAN_BYTES,
            label="sharded online execution plan",
        )
    except ArtifactIntegrityError as exc:
        raise ScalableExecutionError(
            f"cannot read sharded online execution plan safely: {exc}"
        ) from exc
    return loads_sharded_online_execution_plan(encoded)


def write_sharded_online_execution_plan(
    plan: ShardedOnlineExecutionPlan,
    target: str | Path,
) -> None:
    """Write one canonical plan through exclusive, no-follow creation."""

    if not isinstance(plan, ShardedOnlineExecutionPlan):
        raise ScalableExecutionError("plan must be a ShardedOnlineExecutionPlan")
    encoded = plan.canonical_file_bytes()
    if len(encoded) > _MAX_PLAN_BYTES:
        raise ScalableExecutionError(
            f"sharded online execution plan exceeds {_MAX_PLAN_BYTES} bytes"
        )
    try:
        write_exclusive_receipt_bytes(encoded, target)
    except ArtifactIntegrityError as exc:
        raise ScalableExecutionError(
            f"cannot write sharded online execution plan safely: {exc}"
        ) from exc


def _coerce_receipt(
    value: ArtifactVerificationReceipt | ExecutionLeafVerificationReceipt | str | Path,
) -> ArtifactVerificationReceipt:
    if isinstance(value, ArtifactVerificationReceipt):
        return value
    if isinstance(value, ExecutionLeafVerificationReceipt):
        return value.as_artifact_verification_receipt()
    try:
        return load_verification_receipt(value)
    except (ArtifactIntegrityError, TypeError) as exc:
        raise ScalableExecutionError(
            f"cannot load artifact verification receipt safely: {exc}"
        ) from exc


def _receipt_artifacts(
    receipt: ArtifactVerificationReceipt,
) -> dict[str, Any]:
    return {row.artifact_id: row for row in receipt.artifacts}


def _require_verified_pin(
    pin: ImmutableArtifactPin,
    *,
    artifacts: Mapping[str, Any],
) -> None:
    row = artifacts.get(pin.artifact_id)
    if row is None:
        raise ScalableExecutionError(f"artifact receipt omits pin {pin.artifact_id!r}")
    if (
        not row.exact
        or row.kind != pin.kind
        or row.relative_path != pin.relative_path
        or row.expected_sha256 != pin.sha256
        or row.verified_sha256 != pin.sha256
        or row.byte_count != pin.byte_count
        or row.observed_byte_count != pin.byte_count
    ):
        raise ScalableExecutionError(
            f"artifact receipt does not exactly attest pin {pin.artifact_id!r}"
        )


def _validate_receipt_header(
    plan: ShardedOnlineExecutionPlan,
    receipt: ArtifactVerificationReceipt,
) -> dict[str, Any]:
    artifacts = _receipt_artifacts(receipt)
    for pin in plan.direct_artifact_pins:
        _require_verified_pin(pin, artifacts=artifacts)
    return artifacts


def _artifact_path(
    artifact_root: str | Path,
    relative_path: str,
) -> Path:
    root = Path(artifact_root)
    if not root.is_absolute():
        raise ScalableExecutionError("artifact_root must be an absolute directory path")
    if any(part in {".", ".."} for part in root.parts):
        raise ScalableExecutionError("artifact_root cannot contain dot or parent traversal")
    parts = PurePosixPath(relative_path).parts
    return root.joinpath(*parts)


def loads_execution_leaf_verification_receipt(
    payload: bytes | str,
) -> ExecutionLeafVerificationReceipt:
    """Parse one canonical plan-bound leaf receipt."""

    encoded = _as_canonical_file_bytes(
        payload,
        label="execution leaf verification receipt",
        max_bytes=_MAX_LEAF_RECEIPT_BYTES,
    )
    parsed = _parse_json_object(encoded, label="execution leaf verification receipt")
    receipt = ExecutionLeafVerificationReceipt.from_dict(parsed)
    if encoded != receipt.canonical_file_bytes():
        raise ScalableExecutionError("execution leaf verification receipt bytes are not canonical")
    return receipt


def load_execution_leaf_verification_receipt(
    path: str | Path,
) -> ExecutionLeafVerificationReceipt:
    """Load one leaf receipt without links or mutable aliases."""

    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=_MAX_LEAF_RECEIPT_BYTES,
            label="execution leaf verification receipt",
        )
    except ArtifactIntegrityError as exc:
        raise ScalableExecutionError(f"cannot read execution leaf receipt safely: {exc}") from exc
    return loads_execution_leaf_verification_receipt(encoded)


def write_execution_leaf_verification_receipt(
    receipt: ExecutionLeafVerificationReceipt,
    target: str | Path,
) -> None:
    """Publish one plan-bound leaf receipt without replacement."""

    if not isinstance(receipt, ExecutionLeafVerificationReceipt):
        raise ScalableExecutionError("receipt must be an ExecutionLeafVerificationReceipt")
    try:
        write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), target)
    except ArtifactIntegrityError as exc:
        raise ScalableExecutionError(f"cannot write execution leaf receipt safely: {exc}") from exc


def _load_inventory_leaf(
    plan: ShardedOnlineExecutionPlan,
    package_root: str | Path,
) -> CorpusShardInventory:
    pin = plan.corpus_shard_inventory
    path = _artifact_path(package_root, pin.relative_path)
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=_MAX_INVENTORY_BYTES,
            label="package corpus shard inventory",
        )
    except ArtifactIntegrityError as exc:
        raise ScalableExecutionError(f"cannot read package shard inventory safely: {exc}") from exc
    if len(encoded) != pin.byte_count or hashlib.sha256(encoded).hexdigest() != pin.sha256:
        raise ScalableExecutionError("package shard inventory differs from its plan pin")
    inventory = loads_corpus_shard_inventory(encoded)
    if (
        inventory.corpus != plan.corpus
        or inventory.stage != plan.stage
        or inventory.document_count != plan.document_count
        or inventory.ordered_document_universe_sha256 != plan.ordered_document_universe_sha256
        or inventory.file_sha256 != pin.sha256
    ):
        raise ScalableExecutionError("package shard inventory identity differs from its plan")
    return inventory


def _execution_leaf_pins(
    plan: ShardedOnlineExecutionPlan,
    inventory: CorpusShardInventory,
) -> tuple[ImmutableArtifactPin, ...]:
    pins = (*plan.direct_artifact_pins, *(row.artifact for row in inventory.shards))
    identifiers = [pin.artifact_id for pin in pins]
    paths = [pin.relative_path for pin in pins]
    if len(identifiers) != len(set(identifiers)):
        raise ScalableExecutionError("execution package repeats a leaf artifact ID")
    if len(paths) != len(set(paths)):
        raise ScalableExecutionError("execution package repeats a leaf path")
    reserved = {
        ONLINE_EXECUTION_PLAN_FILENAME,
        EXECUTION_LEAF_RECEIPT_FILENAME,
    }
    if reserved.intersection(paths):
        raise ScalableExecutionError("execution leaf path collides with a package control file")
    if any(pin.kind != "file" for pin in pins):
        raise ScalableExecutionError("execution package currently admits only file leaves")
    all_files = (*paths, *reserved)
    path_parts = [PurePosixPath(path).parts for path in all_files]
    for position, first in enumerate(path_parts):
        for second in path_parts[position + 1 :]:
            common = min(len(first), len(second))
            if first[:common] == second[:common]:
                raise ScalableExecutionError("execution package contains overlapping file paths")
    return tuple(
        sorted(
            pins,
            key=lambda pin: (
                pin.artifact_id.encode("utf-8"),
                pin.relative_path.encode("utf-8"),
            ),
        )
    )


def _package_entries(
    pins: Sequence[ImmutableArtifactPin],
    *,
    include_leaf_receipt: bool,
) -> tuple[str, ...]:
    files = [ONLINE_EXECUTION_PLAN_FILENAME, *(pin.relative_path for pin in pins)]
    if include_leaf_receipt:
        files.append(EXECUTION_LEAF_RECEIPT_FILENAME)
    entries = set(files)
    for path in files:
        parts = PurePosixPath(path).parts
        entries.update("/".join(parts[:stop]) for stop in range(1, len(parts)))
    return tuple(sorted(entries, key=lambda value: value.encode("utf-8")))


def _assert_package_entries(
    observed: Sequence[str],
    expected: Sequence[str],
) -> None:
    observed_set = set(observed)
    expected_set = set(expected)
    if observed_set != expected_set or len(observed) != len(observed_set):
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        raise ScalableExecutionError(
            f"execution package membership differs; missing={missing[:5]}, extra={extra[:5]}"
        )


def _digest_execution_package(root: Path) -> DirectoryDigest:
    """Hash one package tree and expose integrity failures through this API."""

    try:
        return digest_directory_tree(root)
    except ArtifactIntegrityError as exc:
        raise ScalableExecutionError(f"execution package tree verification failed: {exc}") from exc


def _fresh_leaf_receipt(
    plan: ShardedOnlineExecutionPlan,
    inventory: CorpusShardInventory,
    package_root: str | Path,
) -> ExecutionLeafVerificationReceipt:
    pins = _execution_leaf_pins(plan, inventory)
    try:
        verified = verify_local_artifacts(
            package_root,
            manifest_sha256=plan.artifact_sha256,
            artifacts=tuple(
                LocalArtifactSpec(
                    artifact_id=pin.artifact_id,
                    relative_path=pin.relative_path,
                    kind="file",
                    expected_sha256=pin.sha256,
                )
                for pin in pins
            ),
        )
    except ArtifactIntegrityError as exc:
        raise ScalableExecutionError(f"execution package leaf verification failed: {exc}") from exc
    return ExecutionLeafVerificationReceipt(
        plan_sha256=plan.artifact_sha256,
        plan_file_sha256=plan.file_sha256,
        artifacts=verified.artifacts,
    )


def finalize_online_execution_package(
    package_root: str | Path,
) -> OnlineExecutionPackage:
    """Write the leaf receipt, then return the immutable outer-tree identity."""

    root = Path(package_root)
    if not root.is_absolute():
        raise ScalableExecutionError("package_root must be absolute")
    plan = load_sharded_online_execution_plan(root / ONLINE_EXECUTION_PLAN_FILENAME)
    inventory = _load_inventory_leaf(plan, root)
    pins = _execution_leaf_pins(plan, inventory)
    before = _digest_execution_package(root)
    _assert_package_entries(
        before.entries,
        _package_entries(pins, include_leaf_receipt=False),
    )
    receipt = _fresh_leaf_receipt(plan, inventory, root)
    write_execution_leaf_verification_receipt(receipt, root / EXECUTION_LEAF_RECEIPT_FILENAME)
    tree = _digest_execution_package(root)
    return verify_online_execution_package(
        root,
        expected_tree_sha256=tree.sha256,
        expected_plan_revision=f"sha256:{plan.artifact_sha256}",
    )


def verify_online_execution_package(
    package_root: str | Path,
    *,
    expected_tree_sha256: str,
    expected_plan_revision: str,
) -> OnlineExecutionPackage:
    """Verify the outer Merkle tree and the exact internal leaf closure."""

    root = Path(package_root)
    if not root.is_absolute():
        raise ScalableExecutionError("package_root must be absolute")
    expected_tree = _require_sha256("expected package tree SHA-256", expected_tree_sha256)
    if not isinstance(expected_plan_revision, str) or not expected_plan_revision.startswith(
        "sha256:"
    ):
        raise ScalableExecutionError(
            "expected_plan_revision must equal 'sha256:<logical-plan-digest>'"
        )
    expected_plan_sha256 = _require_sha256(
        "expected logical plan SHA-256", expected_plan_revision[7:]
    )
    tree = _digest_execution_package(root)
    if tree.sha256 != expected_tree:
        raise ScalableExecutionError("execution package outer directory-tree SHA-256 differs")
    plan = load_sharded_online_execution_plan(root / ONLINE_EXECUTION_PLAN_FILENAME)
    if plan.artifact_sha256 != expected_plan_sha256:
        raise ScalableExecutionError(
            "execution package logical plan differs from the manifest revision"
        )
    inventory = _load_inventory_leaf(plan, root)
    pins = _execution_leaf_pins(plan, inventory)
    _assert_package_entries(
        tree.entries,
        _package_entries(pins, include_leaf_receipt=True),
    )
    receipt = load_execution_leaf_verification_receipt(root / EXECUTION_LEAF_RECEIPT_FILENAME)
    if receipt.plan_sha256 != plan.artifact_sha256 or receipt.plan_file_sha256 != plan.file_sha256:
        raise ScalableExecutionError("execution leaf receipt binds a different logical plan")
    fresh = _fresh_leaf_receipt(plan, inventory, root)
    if fresh != receipt:
        raise ScalableExecutionError(
            "execution leaf receipt does not cover the exact required leaves"
        )
    final_tree = _digest_execution_package(root)
    if final_tree != tree:
        raise ScalableExecutionError("execution package changed during verification")
    return OnlineExecutionPackage(
        root=root,
        tree_sha256=final_tree.sha256,
        plan=plan,
        inventory=inventory,
        leaf_receipt=receipt,
    )


def _load_pinned_inventory(
    plan: ShardedOnlineExecutionPlan,
    *,
    artifact_root: str | Path,
    receipt: ArtifactVerificationReceipt,
) -> CorpusShardInventory:
    artifacts = _validate_receipt_header(plan, receipt)
    pin = plan.corpus_shard_inventory
    path = _artifact_path(artifact_root, pin.relative_path)
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=_MAX_INVENTORY_BYTES,
            label="pinned corpus shard inventory",
        )
    except ArtifactIntegrityError as exc:
        raise ScalableExecutionError(
            f"cannot read pinned corpus shard inventory safely: {exc}"
        ) from exc
    if len(encoded) != pin.byte_count:
        raise ScalableExecutionError(
            "pinned corpus shard inventory byte count differs from its plan pin"
        )
    if hashlib.sha256(encoded).hexdigest() != pin.sha256:
        raise ScalableExecutionError(
            "pinned corpus shard inventory digest differs from its plan pin"
        )
    inventory = loads_corpus_shard_inventory(encoded)
    if (
        inventory.corpus != plan.corpus
        or inventory.stage != plan.stage
        or inventory.document_count != plan.document_count
        or inventory.ordered_document_universe_sha256 != plan.ordered_document_universe_sha256
    ):
        raise ScalableExecutionError(
            "corpus shard inventory identity differs from the execution plan"
        )
    if (
        inventory.file_sha256 != pin.sha256
        or len(inventory.canonical_file_bytes()) != pin.byte_count
    ):
        raise ScalableExecutionError(
            "corpus shard inventory canonical bytes differ from its plan pin"
        )
    direct_ids = {row.artifact_id for row in plan.direct_artifact_pins}
    direct_paths = {row.relative_path for row in plan.direct_artifact_pins}
    for shard in inventory.shards:
        if shard.artifact.artifact_id in direct_ids or shard.artifact.relative_path in direct_paths:
            raise ScalableExecutionError(
                "shard artifact identity collides with a plan control artifact"
            )
        _require_verified_pin(shard.artifact, artifacts=artifacts)
    return inventory


def load_pinned_corpus_shard_inventory(
    plan: ShardedOnlineExecutionPlan,
    *,
    artifact_root: str | Path,
    verification_receipt: (
        ArtifactVerificationReceipt | ExecutionLeafVerificationReceipt | str | Path
    ),
) -> CorpusShardInventory:
    """Load and admit the inventory named by a plan and its verification receipt."""

    if not isinstance(plan, ShardedOnlineExecutionPlan):
        raise ScalableExecutionError("plan must be a ShardedOnlineExecutionPlan")
    receipt = _coerce_receipt(verification_receipt)
    return _load_pinned_inventory(
        plan,
        artifact_root=artifact_root,
        receipt=receipt,
    )


def validate_sharded_execution_receipt(
    plan: ShardedOnlineExecutionPlan,
    inventory: CorpusShardInventory,
    receipt: ArtifactVerificationReceipt,
) -> None:
    """Check every direct plan pin and every shard pin against one receipt."""

    if not isinstance(plan, ShardedOnlineExecutionPlan):
        raise ScalableExecutionError("plan must be a ShardedOnlineExecutionPlan")
    if not isinstance(inventory, CorpusShardInventory):
        raise ScalableExecutionError("inventory must be a CorpusShardInventory")
    if not isinstance(receipt, ArtifactVerificationReceipt):
        raise ScalableExecutionError("receipt must be an ArtifactVerificationReceipt")
    artifacts = _validate_receipt_header(plan, receipt)
    pin = plan.corpus_shard_inventory
    if (
        inventory.file_sha256 != pin.sha256
        or len(inventory.canonical_file_bytes()) != pin.byte_count
        or inventory.corpus != plan.corpus
        or inventory.stage != plan.stage
        or inventory.document_count != plan.document_count
        or inventory.ordered_document_universe_sha256 != plan.ordered_document_universe_sha256
    ):
        raise ScalableExecutionError("inventory does not match its plan identity and file pin")
    seen_ids = {row.artifact_id for row in plan.direct_artifact_pins}
    seen_paths = {row.relative_path for row in plan.direct_artifact_pins}
    for shard in inventory.shards:
        pin = shard.artifact
        if pin.artifact_id in seen_ids or pin.relative_path in seen_paths:
            raise ScalableExecutionError("shard pin collides with another execution artifact")
        _require_verified_pin(pin, artifacts=artifacts)
        seen_ids.add(pin.artifact_id)
        seen_paths.add(pin.relative_path)


def _require_secure_io() -> None:
    missing_flags = [
        name
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        if not hasattr(os, name)
    ]
    if (
        missing_flags
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
        or not hasattr(os, "pread")
    ):
        missing = ", ".join(missing_flags) or "required descriptor operations"
        raise ScalableExecutionError(f"fixed-width sidecar admission is unavailable: {missing}")


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _file_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def _secure_open_error(label: str, exc: OSError) -> ScalableExecutionError:
    if exc.errno == errno.ENOENT:
        return ScalableExecutionError(f"{label} is missing")
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        return ScalableExecutionError(f"{label} crosses a symlink or non-directory ancestor")
    return ScalableExecutionError(f"cannot open {label}: {exc.strerror or str(exc)}")


def _open_absolute_directory(path: str | Path, *, label: str) -> int:
    _require_secure_io()
    target = Path(path)
    if not target.is_absolute() or target.anchor != "/":
        raise ScalableExecutionError(f"{label} must be an absolute POSIX path")
    if any(part in {".", ".."} for part in target.parts):
        raise ScalableExecutionError(f"{label} cannot contain dot or parent traversal")
    try:
        descriptor = os.open("/", _directory_flags())
    except OSError as exc:
        raise _secure_open_error(label, exc) from exc
    try:
        for component in target.parts[1:]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise _secure_open_error(label, exc) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ScalableExecutionError(f"{label} must be a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_at(
    root_descriptor: int,
    parts: Sequence[str],
    *,
    label: str,
) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for component in parts:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise _secure_open_error(label, exc) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ScalableExecutionError(f"{label} must be a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stable_stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_fixed_width_sidecar(
    artifact_root: str | Path,
    descriptor: ProvenanceSidecarDescriptor,
) -> tuple[int, int, str, os.stat_result]:
    root_descriptor = _open_absolute_directory(
        artifact_root,
        label="artifact_root",
    )
    parts = PurePosixPath(descriptor.artifact.relative_path).parts
    try:
        parent_descriptor = _open_directory_at(
            root_descriptor,
            parts[:-1],
            label="provenance sidecar parent",
        )
    finally:
        os.close(root_descriptor)
    filename = parts[-1]
    try:
        try:
            before_entry = os.stat(
                filename,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise _secure_open_error("provenance sidecar", exc) from exc
        if stat.S_ISLNK(before_entry.st_mode):
            raise ScalableExecutionError("provenance sidecar cannot be a symlink")
        try:
            file_descriptor = os.open(
                filename,
                _file_flags(),
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise _secure_open_error("provenance sidecar", exc) from exc
        try:
            opened = os.fstat(file_descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ScalableExecutionError("provenance sidecar must be a regular file")
            if opened.st_nlink != 1:
                raise ScalableExecutionError("hard-linked provenance sidecar is forbidden")
            if (
                opened.st_dev,
                opened.st_ino,
            ) != (
                before_entry.st_dev,
                before_entry.st_ino,
            ):
                raise ScalableExecutionError("provenance sidecar was substituted during open")
            if opened.st_size != descriptor.artifact.byte_count:
                raise ScalableExecutionError(
                    "provenance sidecar length differs from its descriptor"
                )
            digest = hashlib.sha256()
            observed_bytes = 0
            while True:
                chunk = os.read(file_descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                observed_bytes += len(chunk)
            after = os.fstat(file_descriptor)
            if (
                _stable_stat_signature(opened) != _stable_stat_signature(after)
                or observed_bytes != opened.st_size
            ):
                raise ScalableExecutionError(
                    "provenance sidecar changed during digest verification"
                )
            if digest.hexdigest() != descriptor.artifact.sha256:
                raise ScalableExecutionError(
                    "provenance sidecar SHA-256 differs from its descriptor"
                )
            return parent_descriptor, file_descriptor, filename, after
        except BaseException:
            os.close(file_descriptor)
            raise
    except BaseException:
        os.close(parent_descriptor)
        raise


class DigestOnlyProvenanceRegistry:
    """O(1) positional lookup over one receipt-bound SHA-256 sidecar."""

    def __init__(
        self,
        *,
        plan: ShardedOnlineExecutionPlan,
        inventory: CorpusShardInventory,
        execution_receipt: ArtifactVerificationReceipt,
        component_verification_receipt: ArtifactVerificationReceipt,
        component_revisions: tuple[tuple[str, str], ...],
        parent_descriptor: int,
        file_descriptor: int,
        filename: str,
        opened_metadata: os.stat_result,
    ) -> None:
        self.corpus_name = plan.corpus
        self.corpus_stage = plan.stage
        self.document_count = plan.document_count
        self.document_universe_sha256 = plan.ordered_document_universe_sha256
        self.verification_receipt_sha256 = component_verification_receipt.receipt_sha256
        self.execution_verification_receipt_sha256 = execution_receipt.receipt_sha256
        self.execution_artifact_sha256 = plan.artifact_sha256
        self.shard_count = len(inventory.shards)
        self.component_revisions = component_revisions
        self._record_size_bytes = plan.provenance_sha256_sidecar.record_size_bytes
        self._parent_descriptor = parent_descriptor
        self._file_descriptor = file_descriptor
        self._filename = filename
        self._opened_signature = _stable_stat_signature(opened_metadata)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _assert_stable_binding(self) -> None:
        if self._closed:
            raise ScalableExecutionError("provenance registry is closed")
        try:
            opened = os.fstat(self._file_descriptor)
            entry = os.stat(
                self._filename,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ScalableExecutionError("provenance sidecar path changed after admission") from exc
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(entry.st_mode)
            or opened.st_nlink != 1
            or entry.st_nlink != 1
            or _stable_stat_signature(opened) != self._opened_signature
            or _stable_stat_signature(entry) != self._opened_signature
        ):
            raise ScalableExecutionError(
                "provenance sidecar was linked, mutated, or substituted after admission"
            )

    def content_sha256(self, document_id: int) -> str:
        """Return one bare lowercase digest without reading any corpus record."""

        if (
            isinstance(document_id, bool)
            or not isinstance(document_id, int)
            or not 0 <= document_id < self.document_count
        ):
            raise ScalableExecutionError("document_id is outside the verified document universe")
        self._assert_stable_binding()
        offset = document_id * self._record_size_bytes
        try:
            record = os.pread(
                self._file_descriptor,
                self._record_size_bytes,
                offset,
            )
        except OSError as exc:
            raise ScalableExecutionError("cannot read the admitted provenance sidecar") from exc
        if len(record) != self._record_size_bytes:
            raise ScalableExecutionError("provenance sidecar returned a truncated digest record")
        self._assert_stable_binding()
        return record.hex()

    def content_hash(self, document_id: int) -> str:
        """Return the corpus-facing sha256-prefixed content hash."""

        return f"sha256:{self.content_sha256(document_id)}"

    def lookup_content_hash(self, document_id: int) -> str:
        """Named alias for content-hash consumers."""

        return self.content_hash(document_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._file_descriptor)
        os.close(self._parent_descriptor)

    def __enter__(self) -> DigestOnlyProvenanceRegistry:
        if self._closed:
            raise ScalableExecutionError("provenance registry is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


def _verified_audit_component_revisions(
    receipt: ArtifactVerificationReceipt,
    bindings: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(receipt, ArtifactVerificationReceipt):
        raise ScalableExecutionError(
            "component_verification_receipt must be an ArtifactVerificationReceipt"
        )
    if not isinstance(bindings, tuple):
        raise ScalableExecutionError("component_artifact_ids must be a tuple")
    parsed: list[tuple[str, str]] = []
    for position, binding in enumerate(bindings):
        if not isinstance(binding, tuple) or len(binding) != 2:
            raise ScalableExecutionError(
                f"component_artifact_ids[{position}] must be one component/ID pair"
            )
        component = _require_text(
            f"component_artifact_ids[{position}].component",
            binding[0],
        )
        artifact_id = _require_text(
            f"component_artifact_ids[{position}].artifact_id",
            binding[1],
        )
        parsed.append((component, artifact_id))
    canonical = tuple(
        sorted(
            parsed,
            key=lambda item: (item[0].encode("utf-8"), item[1].encode("utf-8")),
        )
    )
    names = [name for name, _ in canonical]
    artifact_ids = [artifact_id for _, artifact_id in canonical]
    if (
        tuple(parsed) != canonical
        or set(names) != _AUDIT_COMPONENT_NAMES
        or len(names) != len(set(names))
        or len(artifact_ids) != len(set(artifact_ids))
    ):
        raise ScalableExecutionError(
            "component artifact bindings must be the exact six sorted audit components"
        )
    receipt_by_id = _receipt_artifacts(receipt)
    missing = set(artifact_ids) - set(receipt_by_id)
    if missing:
        raise ScalableExecutionError(
            f"component artifact bindings name unverified IDs {sorted(missing)}"
        )
    revisions: list[tuple[str, str]] = []
    for component, artifact_id in canonical:
        row = receipt_by_id[artifact_id]
        if not row.exact:
            raise ScalableExecutionError(f"audit component {component!r} was not verified exactly")
        revisions.append((component, row.verified_sha256))
    return tuple(revisions)


def open_digest_provenance_registry(
    plan: ShardedOnlineExecutionPlan,
    *,
    artifact_root: str | Path,
    verification_receipt: (
        ArtifactVerificationReceipt | ExecutionLeafVerificationReceipt | str | Path
    ),
    component_verification_receipt: ArtifactVerificationReceipt,
    component_artifact_ids: tuple[tuple[str, str], ...],
) -> DigestOnlyProvenanceRegistry:
    """Admit a fixed-width sidecar, then retain only random-access descriptors."""

    if not isinstance(plan, ShardedOnlineExecutionPlan):
        raise ScalableExecutionError("plan must be a ShardedOnlineExecutionPlan")
    receipt = _coerce_receipt(verification_receipt)
    component_revisions = _verified_audit_component_revisions(
        component_verification_receipt,
        component_artifact_ids,
    )
    inventory = _load_pinned_inventory(
        plan,
        artifact_root=artifact_root,
        receipt=receipt,
    )
    validate_sharded_execution_receipt(plan, inventory, receipt)
    parent_descriptor, file_descriptor, filename, metadata = _open_fixed_width_sidecar(
        artifact_root,
        plan.provenance_sha256_sidecar,
    )
    try:
        return DigestOnlyProvenanceRegistry(
            plan=plan,
            inventory=inventory,
            execution_receipt=receipt,
            component_verification_receipt=component_verification_receipt,
            component_revisions=component_revisions,
            parent_descriptor=parent_descriptor,
            file_descriptor=file_descriptor,
            filename=filename,
            opened_metadata=metadata,
        )
    except BaseException:
        os.close(file_descriptor)
        os.close(parent_descriptor)
        raise


@dataclass(frozen=True)
class ExecutionCompatibilityView:
    """The common fields needed by action-panel and prediction binders."""

    document_count: int
    document_universe_sha256: str
    trial_keys: tuple[str, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require_integer("execution document_count", self.document_count, minimum=1)
        _require_sha256(
            "execution document_universe_sha256",
            self.document_universe_sha256,
        )
        keys = tuple(self.trial_keys)
        if not keys:
            raise ScalableExecutionError("execution must contain at least one trial key")
        for key in keys:
            _require_sha256("execution trial key", key)
        canonical = tuple(sorted(keys, key=lambda value: value.encode("utf-8")))
        if keys != canonical or len(keys) != len(set(keys)):
            raise ScalableExecutionError("execution trial keys must be unique and bytewise sorted")
        _require_sha256("execution artifact_sha256", self.artifact_sha256)


def execution_document_count(execution: object) -> int:
    """Return a count from either an inline artifact or a sharded plan."""

    if isinstance(execution, ShardedOnlineExecutionPlan):
        return execution.document_count
    try:
        declared = execution.document_count  # type: ignore[attr-defined]
    except AttributeError:
        declared = None
    if declared is not None:
        return _require_integer("execution document_count", declared, minimum=1)
    try:
        documents = execution.documents  # type: ignore[attr-defined]
        count = len(documents)
    except (AttributeError, TypeError) as exc:
        raise ScalableExecutionError(
            "execution exposes neither document_count nor sized documents"
        ) from exc
    return _require_integer("execution document_count", count, minimum=1)


def execution_trial_keys(execution: object) -> tuple[str, ...]:
    """Return sorted opaque trial keys without touching corpus documents."""

    if isinstance(execution, ShardedOnlineExecutionPlan):
        return execution.trial_keys
    try:
        trials = execution.trials  # type: ignore[attr-defined]
        keys = tuple(row.trial_key for row in trials)
    except (AttributeError, TypeError) as exc:
        raise ScalableExecutionError("execution does not expose iterable trial rows") from exc
    for key in keys:
        _require_sha256("execution trial key", key)
    if not keys or len(keys) != len(set(keys)):
        raise ScalableExecutionError("execution trial keys must be non-empty and unique")
    return tuple(sorted(keys, key=lambda value: value.encode("utf-8")))


def execution_document_universe_sha256(execution: object) -> str:
    """Return the ordered provenance identity of inline or sharded documents."""

    if isinstance(execution, ShardedOnlineExecutionPlan):
        return _require_sha256(
            "execution document_universe_sha256",
            execution.document_universe_sha256,
        )
    try:
        documents = tuple(execution.documents)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        try:
            observed = execution.document_universe_sha256  # type: ignore[attr-defined]
        except AttributeError:
            raise ScalableExecutionError(
                "execution exposes neither inline documents nor a document-universe digest"
            ) from exc
        return _require_sha256("execution document_universe_sha256", observed)
    identities: list[str] = []
    for position, document in enumerate(documents):
        try:
            document_id = document.document_id
            payload = {
                "content_hash": document.content_hash,
                "document_id": document_id,
                "external_id": document.external_id,
                "source_uri": document.source_uri,
            }
        except AttributeError as exc:
            raise ScalableExecutionError(
                "inline execution document lacks provenance identity fields"
            ) from exc
        if type(document_id) is not int or document_id != position:
            raise ScalableExecutionError(
                "inline execution document IDs must be contiguous and ordered"
            )
        identities.append(_canonical_bytes(payload).decode("utf-8"))
    try:
        return policy_document_universe_sha256(identities)
    except ValueError as exc:
        raise ScalableExecutionError("inline execution document identities must be unique") from exc


def execution_artifact_sha256(execution: object) -> str:
    """Return and verify the logical artifact digest for either plan shape."""

    try:
        observed = execution.artifact_sha256  # type: ignore[attr-defined]
        canonical = execution.canonical_bytes()  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise ScalableExecutionError("execution lacks artifact_sha256 and canonical_bytes") from exc
    digest = _require_sha256("execution artifact_sha256", observed)
    if not isinstance(canonical, bytes):
        raise ScalableExecutionError("execution canonical_bytes must return bytes")
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise ScalableExecutionError("execution artifact digest differs from its canonical bytes")
    return digest


def execution_compatibility_view(execution: object) -> ExecutionCompatibilityView:
    """Build the small binding surface shared by inline and sharded execution."""

    return ExecutionCompatibilityView(
        document_count=execution_document_count(execution),
        document_universe_sha256=execution_document_universe_sha256(execution),
        trial_keys=execution_trial_keys(execution),
        artifact_sha256=execution_artifact_sha256(execution),
    )
