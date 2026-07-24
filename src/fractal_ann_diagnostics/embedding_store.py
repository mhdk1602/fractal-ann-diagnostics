"""Streaming construction of pinned, label-payload-excluded embedding stores."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import random
import re
import stat
import sys
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

import numpy as np

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    read_secure_regular_file,
)

EMBEDDING_STORE_SCHEMA = "fractal-embedding-store-v1"
EMBEDDING_CHECKPOINT_SCHEMA = "fractal-embedding-store-checkpoint-v1"
EMBEDDING_SOURCE_BINDING_SCHEMA = "fractal-embedding-source-binding-v1"
EMBEDDING_BUILDER_VERSION = "fractal-embedding-store-builder-v1"
SENTENCE_TRANSFORMERS_ENCODER_ID = "sentence-transformers-local-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDERS = {"", "latest", "main", "master", "tbd", "todo", "unassigned"}
_INVENTORY_MAX_BYTES = 64 * 1024 * 1024
_CONTROL_MAX_BYTES = 4 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_FINAL_VALIDATION_ROWS = 8192
_SOURCE_ROW_FIELDS = {
    "byte_count",
    "dataset",
    "path",
    "record_count",
    "role",
    "sha256",
    "stage",
    "visibility",
}
_DOCUMENT_FIELDS = {"id", "text", "title"}
_QUERY_FIELDS = {"id", "text"}
_FORBIDDEN_PATH_PARTS = ("custody", "label", "qrel")
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
_ROW_ORDER_FIELDS = {
    "byte_count",
    "file_sha256",
    "relative_path",
    "row_count",
    "row_order_sha256",
}
_VECTOR_FIELDS = {
    "builder_version",
    "byte_count",
    "dtype",
    "file_sha256",
    "model_revision",
    "model_tree_sha256",
    "prompt_sha256",
    "relative_path",
    "row_order_sha256",
    "shape",
}
_RECEIPT_FIELDS = {
    "builder_version",
    "config_sha256",
    "current_model",
    "document_count",
    "old_model",
    "query_count",
    "row_orders",
    "schema_version",
    "source_inventory_sha256",
    "staged_inventory_sha256",
    "vectors",
}
_MODEL_BINDING_FIELDS = {"encoder_id", "revision", "tree_sha256"}
_CHECKPOINT_FIELDS = {
    "build_sha256",
    "builder_version",
    "config_sha256",
    "document_count",
    "progress",
    "query_count",
    "row_orders",
    "schema_version",
    "source_inventory_sha256",
    "staged_inventory_sha256",
    "vector_files",
}


class EmbeddingStoreError(RuntimeError):
    """Raised when an embedding store cannot be built without ambiguity."""


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
        raise EmbeddingStoreError("embedding metadata must be finite canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EmbeddingStoreError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EmbeddingStoreError(f"{name} must be a canonical non-empty string")
    return value


def _require_prompt(name: str, value: object, *, allow_empty: bool) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise EmbeddingStoreError(f"{name} has the wrong prompt type")
    if unicodedata.normalize("NFC", value) != value:
        raise EmbeddingStoreError(f"{name} must use NFC Unicode normalization")
    if "\r" in value or any(
        ord(character) < 32 and character not in {"\n", "\t"} for character in value
    ):
        raise EmbeddingStoreError(f"{name} contains an unsupported control character")
    return value


def _positive_integer(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise EmbeddingStoreError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _closed_mapping(value: object, *, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EmbeddingStoreError(f"{name} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise EmbeddingStoreError(
            f"{name} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _decode_json(encoded: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise EmbeddingStoreError(f"{name} repeats key {key!r}")
            value[key] = item
        return value

    def reject_nonfinite(value: str) -> None:
        raise EmbeddingStoreError(f"{name} contains non-finite value {value!r}")

    try:
        decoded = encoded.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise EmbeddingStoreError(f"{name} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise EmbeddingStoreError(f"{name} must be valid JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise EmbeddingStoreError(f"{name} must contain one object")
    return value


def _relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EmbeddingStoreError(f"{name} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise EmbeddingStoreError(f"{name} must be a canonical relative POSIX path")
    if unicodedata.normalize("NFC", value) != value:
        raise EmbeddingStoreError(f"{name} must use NFC Unicode normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise EmbeddingStoreError(f"{name} cannot contain control characters")
    if any(
        forbidden in part.casefold() for part in path.parts for forbidden in _FORBIDDEN_PATH_PARTS
    ):
        raise EmbeddingStoreError(f"{name} crosses a forbidden outcome path")
    if path.suffix != ".jsonl":
        raise EmbeddingStoreError(f"{name} must identify canonical JSONL")
    return value


def _artifact_filename(value: object, *, name: str, suffix: str) -> str:
    filename = _require_identifier(name, value)
    path = PurePosixPath(filename)
    if len(path.parts) != 1 or path.name != filename or path.suffix != suffix:
        raise EmbeddingStoreError(f"{name} must be one local {suffix} filename")
    return filename


def _require_sorted_paths(name: str, values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise EmbeddingStoreError(f"{name} must be a non-empty tuple")
    paths = tuple(
        _relative_path(value, name=f"{name}[{position}]") for position, value in enumerate(values)
    )
    ordered = tuple(sorted(paths, key=lambda path: path.encode("utf-8")))
    if paths != ordered or len(paths) != len(set(paths)):
        raise EmbeddingStoreError(f"{name} must be unique and bytewise sorted")
    return paths


@dataclass(frozen=True)
class StagedEmbeddingSources:
    """An inventory-bound allowlist of document and query JSONL files."""

    root: Path
    inventory_sha256: str
    document_paths: tuple[str, ...]
    query_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute() or any(part in {".", ".."} for part in root.parts):
            raise EmbeddingStoreError("staged source root must be an absolute canonical path")
        object.__setattr__(self, "root", root)
        _require_sha256("inventory_sha256", self.inventory_sha256)
        object.__setattr__(
            self,
            "document_paths",
            _require_sorted_paths("document_paths", self.document_paths),
        )
        object.__setattr__(
            self,
            "query_paths",
            _require_sorted_paths("query_paths", self.query_paths),
        )
        overlap = set(self.document_paths) & set(self.query_paths)
        if overlap:
            raise EmbeddingStoreError(f"document and query selections overlap: {sorted(overlap)}")


@dataclass(frozen=True)
class LocalModelSpec:
    """One exact local model tree and its immutable upstream revision."""

    path: Path
    revision: str
    tree_sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
            raise EmbeddingStoreError("model path must be absolute and canonical")
        object.__setattr__(self, "path", path)
        revision = _require_identifier("model revision", self.revision)
        if revision.casefold() in _PLACEHOLDERS:
            raise EmbeddingStoreError("model revision cannot be movable or a placeholder")
        _require_sha256("model tree_sha256", self.tree_sha256)

    def binding(self, *, encoder_id: str) -> dict[str, str]:
        return {
            "encoder_id": _require_identifier("encoder_id", encoder_id),
            "revision": self.revision,
            "tree_sha256": self.tree_sha256,
        }


@dataclass(frozen=True)
class EmbeddingStoreConfig:
    """Frozen text, geometry, numeric, and execution parameters."""

    query_prompt: str
    document_prompt: str
    max_sequence_length: int
    output_dimension: int
    normalize: bool
    batch_size: int
    output_dtype: Literal["float16", "float32"]
    device: str
    deterministic_seed: int
    builder_version: str = EMBEDDING_BUILDER_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_prompt",
            _require_prompt("query_prompt", self.query_prompt, allow_empty=False),
        )
        object.__setattr__(
            self,
            "document_prompt",
            _require_prompt("document_prompt", self.document_prompt, allow_empty=True),
        )
        _positive_integer("max_sequence_length", self.max_sequence_length, maximum=1_048_576)
        _positive_integer("output_dimension", self.output_dimension, maximum=65_536)
        if self.normalize is not True:
            raise EmbeddingStoreError("normalize must be fixed to true")
        _positive_integer("batch_size", self.batch_size, maximum=4096)
        if self.output_dtype not in {"float16", "float32"}:
            raise EmbeddingStoreError("output_dtype must be 'float16' or 'float32'")
        _require_identifier("device", self.device)
        if (
            isinstance(self.deterministic_seed, bool)
            or not isinstance(self.deterministic_seed, int)
            or not 0 <= self.deterministic_seed < 2**63
        ):
            raise EmbeddingStoreError("deterministic_seed must be an unsigned 63-bit integer")
        if self.builder_version != EMBEDDING_BUILDER_VERSION:
            raise EmbeddingStoreError(f"builder_version must equal {EMBEDDING_BUILDER_VERSION!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "builder_version": self.builder_version,
            "deterministic_seed": self.deterministic_seed,
            "device": self.device,
            "document_prompt": self.document_prompt,
            "max_sequence_length": self.max_sequence_length,
            "normalize": self.normalize,
            "output_dimension": self.output_dimension,
            "output_dtype": self.output_dtype,
            "query_prompt": self.query_prompt,
        }

    @property
    def sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))


class EmbeddingBatchEncoder(Protocol):
    """Injected batch encoder used by the streaming builder."""

    implementation_id: str

    def encode(
        self,
        texts: Sequence[str],
        *,
        model_path: Path,
        prompt: str,
        max_sequence_length: int,
        output_dimension: int,
        normalize: bool,
        device: str,
        seed: int,
    ) -> np.ndarray: ...


class PairedEmbeddingBatchEncoder(Protocol):
    """Encoder that emits current and old rows in one causally paired call."""

    current_implementation_id: str
    old_implementation_id: str

    def encode_pair(
        self,
        texts: Sequence[str],
        *,
        current_model_path: Path,
        old_model_path: Path,
        prompt: str,
        max_sequence_length: int,
        output_dimension: int,
        normalize: bool,
        device: str,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class SentenceTransformersLocalEncoder:
    """Lazy Sentence Transformers adapter restricted to a local model tree."""

    implementation_id = SENTENCE_TRANSFORMERS_ENCODER_ID

    def __init__(self) -> None:
        self._loaded_path: Path | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load(self, model_path: Path, *, device: str, output_dimension: int) -> None:
        if self._model is not None:
            if self._loaded_path != model_path:
                raise EmbeddingStoreError(
                    "one SentenceTransformersLocalEncoder instance cannot switch model trees"
                )
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingStoreError(
                "sentence-transformers and torch are required for the production encoder"
            ) from exc

        previous = {
            name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        }
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            model = SentenceTransformer(
                str(model_path),
                device=device,
                local_files_only=True,
                trust_remote_code=False,
                truncate_dim=output_dimension,
            )
        except Exception as exc:
            raise EmbeddingStoreError("cannot load the pinned local model tree") from exc
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        model.eval()
        self._loaded_path = model_path
        self._model = model
        self._torch = torch

    def encode(
        self,
        texts: Sequence[str],
        *,
        model_path: Path,
        prompt: str,
        max_sequence_length: int,
        output_dimension: int,
        normalize: bool,
        device: str,
        seed: int,
    ) -> np.ndarray:
        self._load(model_path, device=device, output_dimension=output_dimension)
        model = self._model
        torch = self._torch
        if model is None or torch is None:
            raise EmbeddingStoreError("local encoder did not initialize")
        model.max_seq_length = max_sequence_length
        model.eval()
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        with torch.inference_mode():
            vectors = model.encode(
                list(texts),
                prompt=prompt,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                device=device,
                truncate_dim=output_dimension,
            )
        return np.asarray(vectors)


@dataclass(frozen=True)
class _SourceFile:
    relative_path: str
    kind: Literal["documents", "queries"]
    dataset: str
    stage: str | None
    sha256: str
    byte_count: int
    record_count: int

    def binding(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "dataset": self.dataset,
            "kind": self.kind,
            "path": self.relative_path,
            "record_count": self.record_count,
            "sha256": self.sha256,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class _ResolvedSources:
    staged_inventory_sha256: str
    source_inventory_sha256: str
    documents: tuple[_SourceFile, ...]
    queries: tuple[_SourceFile, ...]

    @property
    def document_count(self) -> int:
        return sum(source.record_count for source in self.documents)

    @property
    def query_count(self) -> int:
        return sum(source.record_count for source in self.queries)

    def binding_payload(self) -> dict[str, object]:
        return {
            "documents": [source.binding() for source in self.documents],
            "queries": [source.binding() for source in self.queries],
            "schema_version": EMBEDDING_SOURCE_BINDING_SCHEMA,
            "staged_inventory_sha256": self.staged_inventory_sha256,
        }


@dataclass(frozen=True)
class RowOrderDescriptor:
    relative_path: str
    row_count: int
    byte_count: int
    row_order_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _artifact_filename(
                self.relative_path,
                name="row-order relative_path",
                suffix=".jsonl",
            ),
        )
        for name in ("row_count", "byte_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EmbeddingStoreError(f"row-order {name} must be non-negative")
        _require_sha256("row_order_sha256", self.row_order_sha256)
        _require_sha256("row-order file_sha256", self.file_sha256)
        if self.row_order_sha256 != self.file_sha256:
            raise EmbeddingStoreError("row order digest must equal its canonical file digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "file_sha256": self.file_sha256,
            "relative_path": self.relative_path,
            "row_count": self.row_count,
            "row_order_sha256": self.row_order_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> RowOrderDescriptor:
        row = _closed_mapping(value, fields=_ROW_ORDER_FIELDS, name="row-order descriptor")
        return cls(
            relative_path=row["relative_path"],
            row_count=row["row_count"],
            byte_count=row["byte_count"],
            row_order_sha256=row["row_order_sha256"],
            file_sha256=row["file_sha256"],
        )


@dataclass(frozen=True)
class VectorDescriptor:
    relative_path: str
    dtype: str
    shape: tuple[int, int]
    row_order_sha256: str
    byte_count: int
    file_sha256: str
    model_tree_sha256: str
    model_revision: str
    prompt_sha256: str
    builder_version: str = EMBEDDING_BUILDER_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _artifact_filename(
                self.relative_path,
                name="vector relative_path",
                suffix=".npy",
            ),
        )
        if self.dtype not in {"float16", "float32"}:
            raise EmbeddingStoreError("vector dtype must be float16 or float32")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.shape
            )
        ):
            raise EmbeddingStoreError("vector shape must contain two positive integers")
        _require_sha256("vector row_order_sha256", self.row_order_sha256)
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count <= 0
        ):
            raise EmbeddingStoreError("vector byte_count must be positive")
        _require_sha256("vector file_sha256", self.file_sha256)
        _require_sha256("vector model_tree_sha256", self.model_tree_sha256)
        _require_identifier("vector model_revision", self.model_revision)
        _require_sha256("vector prompt_sha256", self.prompt_sha256)
        if self.builder_version != EMBEDDING_BUILDER_VERSION:
            raise EmbeddingStoreError("vector builder_version differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "builder_version": self.builder_version,
            "byte_count": self.byte_count,
            "dtype": self.dtype,
            "file_sha256": self.file_sha256,
            "model_revision": self.model_revision,
            "model_tree_sha256": self.model_tree_sha256,
            "prompt_sha256": self.prompt_sha256,
            "relative_path": self.relative_path,
            "row_order_sha256": self.row_order_sha256,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: object) -> VectorDescriptor:
        row = _closed_mapping(value, fields=_VECTOR_FIELDS, name="vector descriptor")
        shape = row["shape"]
        if not isinstance(shape, list):
            raise EmbeddingStoreError("vector shape must be an array")
        return cls(
            relative_path=row["relative_path"],
            dtype=row["dtype"],
            shape=tuple(shape),
            row_order_sha256=row["row_order_sha256"],
            byte_count=row["byte_count"],
            file_sha256=row["file_sha256"],
            model_tree_sha256=row["model_tree_sha256"],
            model_revision=row["model_revision"],
            prompt_sha256=row["prompt_sha256"],
            builder_version=row["builder_version"],
        )


@dataclass(frozen=True)
class EmbeddingStoreReceipt:
    staged_inventory_sha256: str
    source_inventory_sha256: str
    config_sha256: str
    document_count: int
    query_count: int
    current_model: Mapping[str, str]
    old_model: Mapping[str, str] | None
    row_orders: Mapping[str, RowOrderDescriptor]
    vectors: Mapping[str, VectorDescriptor]
    builder_version: str = EMBEDDING_BUILDER_VERSION
    schema_version: str = EMBEDDING_STORE_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("staged_inventory_sha256", self.staged_inventory_sha256)
        _require_sha256("source_inventory_sha256", self.source_inventory_sha256)
        _require_sha256("config_sha256", self.config_sha256)
        for name in ("document_count", "query_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise EmbeddingStoreError(f"{name} must be positive")
        _validate_model_binding(self.current_model, name="current_model")
        if self.old_model is not None:
            _validate_model_binding(self.old_model, name="old_model")
        row_orders = dict(self.row_orders)
        if set(row_orders) != {"documents", "queries"} or not all(
            isinstance(value, RowOrderDescriptor) for value in row_orders.values()
        ):
            raise EmbeddingStoreError("row_orders must bind documents and queries")
        vectors = dict(self.vectors)
        expected = {"current_documents", "current_queries"}
        if self.old_model is not None:
            expected |= {"old_documents", "old_queries"}
        if set(vectors) != expected or not all(
            isinstance(value, VectorDescriptor) for value in vectors.values()
        ):
            raise EmbeddingStoreError("vectors do not match the declared model set")
        if row_orders["documents"].row_count != self.document_count:
            raise EmbeddingStoreError("document row-order count differs from receipt")
        if row_orders["queries"].row_count != self.query_count:
            raise EmbeddingStoreError("query row-order count differs from receipt")
        dimensions: set[int] = set()
        dtypes: set[str] = set()
        prompts: dict[str, set[str]] = {"documents": set(), "queries": set()}
        for matrix, descriptor in vectors.items():
            kind = "documents" if matrix.endswith("documents") else "queries"
            expected_rows = self.document_count if kind == "documents" else self.query_count
            if descriptor.shape[0] != expected_rows:
                raise EmbeddingStoreError(f"vector {matrix!r} row count differs")
            if descriptor.row_order_sha256 != row_orders[kind].row_order_sha256:
                raise EmbeddingStoreError(f"vector {matrix!r} row order differs")
            model_binding = self.old_model if matrix.startswith("old_") else self.current_model
            if model_binding is None or (
                descriptor.model_revision != model_binding["revision"]
                or descriptor.model_tree_sha256 != model_binding["tree_sha256"]
            ):
                raise EmbeddingStoreError(f"vector {matrix!r} model binding differs")
            dimensions.add(descriptor.shape[1])
            dtypes.add(descriptor.dtype)
            prompts[kind].add(descriptor.prompt_sha256)
        if len(dimensions) != 1 or len(dtypes) != 1:
            raise EmbeddingStoreError("all vector matrices must share one dimension and dtype")
        if any(len(values) != 1 for values in prompts.values()):
            raise EmbeddingStoreError("current and old matrices must share prompts by row kind")
        object.__setattr__(self, "current_model", dict(self.current_model))
        object.__setattr__(
            self, "old_model", None if self.old_model is None else dict(self.old_model)
        )
        object.__setattr__(self, "row_orders", row_orders)
        object.__setattr__(self, "vectors", vectors)
        if self.builder_version != EMBEDDING_BUILDER_VERSION:
            raise EmbeddingStoreError("receipt builder_version differs")
        if self.schema_version != EMBEDDING_STORE_SCHEMA:
            raise EmbeddingStoreError("receipt schema_version differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "builder_version": self.builder_version,
            "config_sha256": self.config_sha256,
            "current_model": dict(self.current_model),
            "document_count": self.document_count,
            "old_model": None if self.old_model is None else dict(self.old_model),
            "query_count": self.query_count,
            "row_orders": {key: value.to_dict() for key, value in sorted(self.row_orders.items())},
            "schema_version": self.schema_version,
            "source_inventory_sha256": self.source_inventory_sha256,
            "staged_inventory_sha256": self.staged_inventory_sha256,
            "vectors": {key: value.to_dict() for key, value in sorted(self.vectors.items())},
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> EmbeddingStoreReceipt:
        row = _closed_mapping(value, fields=_RECEIPT_FIELDS, name="embedding store receipt")
        row_orders = row["row_orders"]
        vectors = row["vectors"]
        if not isinstance(row_orders, Mapping) or not isinstance(vectors, Mapping):
            raise EmbeddingStoreError("receipt row_orders and vectors must be objects")
        return cls(
            staged_inventory_sha256=row["staged_inventory_sha256"],
            source_inventory_sha256=row["source_inventory_sha256"],
            config_sha256=row["config_sha256"],
            document_count=row["document_count"],
            query_count=row["query_count"],
            current_model=row["current_model"],
            old_model=row["old_model"],
            row_orders={
                key: RowOrderDescriptor.from_dict(item) for key, item in row_orders.items()
            },
            vectors={key: VectorDescriptor.from_dict(item) for key, item in vectors.items()},
            builder_version=row["builder_version"],
            schema_version=row["schema_version"],
        )


def _validate_model_binding(value: object, *, name: str) -> Mapping[str, str]:
    row = _closed_mapping(value, fields=_MODEL_BINDING_FIELDS, name=name)
    _require_identifier(f"{name}.encoder_id", row["encoder_id"])
    _require_identifier(f"{name}.revision", row["revision"])
    _require_sha256(f"{name}.tree_sha256", row["tree_sha256"])
    return row


def _load_sources(selection: StagedEmbeddingSources) -> _ResolvedSources:
    inventory_path = selection.root / "inventory.json"
    try:
        encoded = read_secure_regular_file(
            inventory_path,
            max_bytes=_INVENTORY_MAX_BYTES,
            label="staged inventory",
        )
    except ArtifactIntegrityError as exc:
        raise EmbeddingStoreError(f"cannot read staged inventory: {exc}") from exc
    observed_sha256 = _sha256(encoded)
    if observed_sha256 != selection.inventory_sha256:
        raise EmbeddingStoreError("staged inventory digest differs from its caller pin")
    inventory = _decode_json(encoded, name="staged inventory")
    if encoded != _canonical_bytes(inventory) + b"\n":
        raise EmbeddingStoreError("staged inventory must be canonical JSON plus one newline")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EmbeddingStoreError("staged inventory needs a non-empty artifacts array")
    by_path: dict[str, Mapping[str, Any]] = {}
    for position, value in enumerate(artifacts):
        row = _closed_mapping(
            value,
            fields=_SOURCE_ROW_FIELDS,
            name=f"staged inventory artifact {position}",
        )
        path = row["path"]
        if not isinstance(path, str) or path in by_path:
            raise EmbeddingStoreError("staged inventory contains an invalid or repeated path")
        by_path[path] = row

    def resolve(path: str, kind: Literal["documents", "queries"]) -> _SourceFile:
        row = by_path.get(path)
        if row is None:
            raise EmbeddingStoreError(f"selected source {path!r} is absent from inventory")
        role = _require_identifier(f"{path}.role", row["role"])
        visibility = _require_identifier(f"{path}.visibility", row["visibility"])
        allowed_roles = {"corpus", "corpus-shard"} if kind == "documents" else {"queries"}
        if role not in allowed_roles or visibility != "online":
            raise EmbeddingStoreError(f"selected source {path!r} has a forbidden role")
        dataset = _require_identifier(f"{path}.dataset", row["dataset"])
        stage = row["stage"]
        if kind == "documents":
            if stage is not None:
                raise EmbeddingStoreError(f"document source {path!r} cannot declare a stage")
        elif not isinstance(stage, str) or not stage:
            raise EmbeddingStoreError(f"query source {path!r} must declare its stage")
        sha256 = _require_sha256(f"{path}.sha256", row["sha256"])
        byte_count = _positive_integer(f"{path}.byte_count", row["byte_count"], maximum=2**63 - 1)
        record_count = _positive_integer(
            f"{path}.record_count", row["record_count"], maximum=2**63 - 1
        )
        return _SourceFile(
            relative_path=path,
            kind=kind,
            dataset=dataset,
            stage=stage,
            sha256=sha256,
            byte_count=byte_count,
            record_count=record_count,
        )

    documents = tuple(resolve(path, "documents") for path in selection.document_paths)
    queries = tuple(resolve(path, "queries") for path in selection.query_paths)
    binding = {
        "documents": [source.binding() for source in documents],
        "queries": [source.binding() for source in queries],
        "schema_version": EMBEDDING_SOURCE_BINDING_SCHEMA,
        "staged_inventory_sha256": observed_sha256,
    }
    return _ResolvedSources(
        staged_inventory_sha256=observed_sha256,
        source_inventory_sha256=_sha256(_canonical_bytes(binding)),
        documents=documents,
        queries=queries,
    )


def _directory_flags() -> int:
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise EmbeddingStoreError("secure streaming flags are unavailable on this platform")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _file_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def _stable_stat(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_error(name: str, exc: OSError) -> EmbeddingStoreError:
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        return EmbeddingStoreError(f"{name} crosses a link or non-directory ancestor")
    return EmbeddingStoreError(f"cannot open {name}: {exc.strerror or exc}")


def _open_absolute_directory(path: Path, *, name: str) -> int:
    if not path.is_absolute() or path.anchor != "/":
        raise EmbeddingStoreError(f"{name} must be an absolute POSIX directory")
    try:
        descriptor = os.open("/", _directory_flags())
    except OSError as exc:
        raise _open_error(name, exc) from exc
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise _open_error(name, exc) from exc
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise EmbeddingStoreError(f"{name} must be a real directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _open_source_stream(root: Path, source: _SourceFile) -> Iterator[Any]:
    root_descriptor = _open_absolute_directory(root, name="staged source root")
    descriptor = os.dup(root_descriptor)
    try:
        parts = PurePosixPath(source.relative_path).parts
        for component in parts[:-1]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise _open_error(source.relative_path, exc) from exc
            os.close(descriptor)
            descriptor = child
        try:
            file_descriptor = os.open(parts[-1], _file_flags(), dir_fd=descriptor)
        except OSError as exc:
            raise _open_error(source.relative_path, exc) from exc
        try:
            before = os.fstat(file_descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise EmbeddingStoreError(
                    f"source {source.relative_path!r} must be one non-linked regular file"
                )
            if before.st_size != source.byte_count:
                raise EmbeddingStoreError(f"source {source.relative_path!r} byte count changed")
            with os.fdopen(file_descriptor, "rb", closefd=False) as handle:
                yield handle
            after = os.fstat(file_descriptor)
            if _stable_stat(before) != _stable_stat(after):
                raise EmbeddingStoreError(f"source {source.relative_path!r} changed during read")
        finally:
            os.close(file_descriptor)
    finally:
        os.close(descriptor)
        os.close(root_descriptor)


def _validate_source_row(
    row: Mapping[str, Any],
    *,
    source: _SourceFile,
    line_number: int,
) -> Mapping[str, str]:
    forbidden = sorted(
        key for key in row if any(part in key.casefold() for part in _FORBIDDEN_FIELD_PARTS)
    )
    if forbidden:
        raise EmbeddingStoreError(
            f"source {source.relative_path!r} line {line_number} contains forbidden "
            f"fields {forbidden}"
        )
    expected = _DOCUMENT_FIELDS if source.kind == "documents" else _QUERY_FIELDS
    if set(row) != expected:
        raise EmbeddingStoreError(
            f"source {source.relative_path!r} line {line_number} fields differ from "
            f"{sorted(expected)}"
        )
    for name, value in row.items():
        if not isinstance(value, str):
            raise EmbeddingStoreError(
                f"source {source.relative_path!r} line {line_number} field {name!r} "
                "must be a string"
            )
        if unicodedata.normalize("NFC", value) != value:
            raise EmbeddingStoreError(
                f"source {source.relative_path!r} line {line_number} must use NFC text"
            )
    if not row["id"] or not row["text"]:
        raise EmbeddingStoreError(
            f"source {source.relative_path!r} line {line_number} has an empty ID or text"
        )
    return row  # type: ignore[return-value]


def _iter_source_rows(
    root: Path,
    source: _SourceFile,
) -> Iterator[tuple[int, Mapping[str, str]]]:
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    with _open_source_stream(root, source) as handle:
        while True:
            line = handle.readline(_MAX_JSONL_LINE_BYTES + 1)
            if not line:
                break
            record_count += 1
            if len(line) > _MAX_JSONL_LINE_BYTES:
                raise EmbeddingStoreError(
                    f"source {source.relative_path!r} line {record_count} exceeds the line bound"
                )
            if not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
                raise EmbeddingStoreError(
                    f"source {source.relative_path!r} line {record_count} is not canonical"
                )
            digest.update(line)
            byte_count += len(line)
            row = _decode_json(line[:-1], name=f"{source.relative_path} line {record_count}")
            if line != _canonical_bytes(row) + b"\n":
                raise EmbeddingStoreError(
                    f"source {source.relative_path!r} line {record_count} is not canonical JSON"
                )
            yield (
                record_count,
                _validate_source_row(
                    row,
                    source=source,
                    line_number=record_count,
                ),
            )
    if byte_count != source.byte_count or digest.hexdigest() != source.sha256:
        raise EmbeddingStoreError(f"source {source.relative_path!r} differs from inventory")
    if record_count != source.record_count:
        raise EmbeddingStoreError(f"source {source.relative_path!r} row count changed")


def _verify_sources(root: Path, sources: _ResolvedSources) -> None:
    for source in (*sources.documents, *sources.queries):
        for _line_number, _row in _iter_source_rows(root, source):
            pass


def _verify_model(spec: LocalModelSpec) -> None:
    try:
        digest = digest_directory_tree(spec.path)
    except ArtifactIntegrityError as exc:
        raise EmbeddingStoreError(f"cannot verify local model tree: {exc}") from exc
    if digest.file_count <= 0 or digest.sha256 != spec.tree_sha256:
        raise EmbeddingStoreError("local model tree differs from its verified canonical digest")


def _row_order_payload(
    source: _SourceFile,
    *,
    line_number: int,
    row_id: str,
) -> dict[str, object]:
    return {
        "dataset": source.dataset,
        "id": row_id,
        "kind": source.kind,
        "source_path": source.relative_path,
        "source_row": line_number,
        "stage": source.stage,
    }


def _row_order_descriptor(
    root: Path,
    sources: Sequence[_SourceFile],
    *,
    relative_path: str,
    target: Path | None,
) -> RowOrderDescriptor:
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    handle = None
    if target is not None:
        try:
            handle = target.open("xb")
        except OSError as exc:
            raise EmbeddingStoreError(f"cannot create row-order file {target}: {exc}") from exc
    try:
        for source in sources:
            for line_number, row in _iter_source_rows(root, source):
                encoded = (
                    _canonical_bytes(
                        _row_order_payload(source, line_number=line_number, row_id=row["id"])
                    )
                    + b"\n"
                )
                digest.update(encoded)
                byte_count += len(encoded)
                row_count += 1
                if handle is not None:
                    handle.write(encoded)
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if handle is not None:
            handle.close()
    file_sha256 = digest.hexdigest()
    return RowOrderDescriptor(
        relative_path=relative_path,
        row_count=row_count,
        byte_count=byte_count,
        row_order_sha256=file_sha256,
        file_sha256=file_sha256,
    )


def _checkpoint_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(f".{output.name}.partial"),
        output.with_name(f".{output.name}.checkpoint.json"),
    )


def _ensure_output_parent(output: Path) -> None:
    if not output.is_absolute() or output.name in {"", ".", ".."}:
        raise EmbeddingStoreError("output root must be an absolute directory path")
    try:
        parent = output.parent.lstat()
    except OSError as exc:
        raise EmbeddingStoreError(f"cannot inspect output parent: {exc}") from exc
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise EmbeddingStoreError("output parent must be a real directory")
    descriptor = _open_absolute_directory(output.parent, name="output parent")
    os.close(descriptor)


def _write_exclusive(path: Path, encoded: bytes) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
    except OSError as exc:
        raise EmbeddingStoreError(f"cannot create {path.name!r} exclusively: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_bytes(payload) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.next")
    _write_exclusive(temporary, encoded)
    try:
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise EmbeddingStoreError(f"cannot replace checkpoint: {exc}") from exc
    _fsync_directory(path.parent)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=_CONTROL_MAX_BYTES,
            label="embedding checkpoint",
        )
    except ArtifactIntegrityError as exc:
        raise EmbeddingStoreError(f"cannot load embedding checkpoint: {exc}") from exc
    value = _decode_json(encoded, name="embedding checkpoint")
    if encoded != _canonical_bytes(value) + b"\n":
        raise EmbeddingStoreError("embedding checkpoint must be canonical JSON")
    return _closed_mapping(value, fields=_CHECKPOINT_FIELDS, name="embedding checkpoint")


def _verify_binding_file(path: Path, *, expected_sha256: str, name: str) -> None:
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=_CONTROL_MAX_BYTES,
            label=name,
        )
    except ArtifactIntegrityError as exc:
        raise EmbeddingStoreError(f"cannot load {name}: {exc}") from exc
    if not encoded.endswith(b"\n"):
        raise EmbeddingStoreError(f"{name} must have one canonical terminal newline")
    value = _decode_json(encoded[:-1], name=name)
    canonical = _canonical_bytes(value)
    if encoded != canonical + b"\n" or _sha256(canonical) != expected_sha256:
        raise EmbeddingStoreError(f"{name} differs from its canonical binding")


def _matrix_files(include_old: bool) -> dict[str, str]:
    files = {
        "current_documents": "current-documents.npy",
        "current_queries": "current-queries.npy",
    }
    if include_old:
        files.update(
            {
                "old_documents": "old-documents.npy",
                "old_queries": "old-queries.npy",
            }
        )
    return files


def _create_memmap(path: Path, *, dtype: str, shape: tuple[int, int]) -> None:
    try:
        array = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.dtype(dtype),
            shape=shape,
            fortran_order=False,
            version=(2, 0),
        )
        array.flush()
        del array
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except (OSError, ValueError) as exc:
        raise EmbeddingStoreError(f"cannot allocate vector file {path.name!r}: {exc}") from exc


def _open_vector_memmap(
    path: Path,
    *,
    dtype: str,
    shape: tuple[int, int],
    writable: bool,
) -> np.memmap:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EmbeddingStoreError(f"cannot inspect vector file {path.name!r}: {exc}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise EmbeddingStoreError(f"vector file {path.name!r} must be a non-linked regular file")
    try:
        array = np.load(path, mmap_mode="r+" if writable else "r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise EmbeddingStoreError(f"cannot load vector file {path.name!r}: {exc}") from exc
    if not isinstance(array, np.memmap):
        raise EmbeddingStoreError(f"vector file {path.name!r} is not a memory-mappable array")
    if array.dtype != np.dtype(dtype) or array.shape != shape or array.ndim != 2:
        raise EmbeddingStoreError(f"vector file {path.name!r} shape or dtype changed")
    if array.flags.f_contiguous and not array.flags.c_contiguous:
        raise EmbeddingStoreError(f"vector file {path.name!r} changed storage order")
    return array


def _fsync_regular_file(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise EmbeddingStoreError(f"cannot open {path.name!r} for durable flush: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise EmbeddingStoreError(f"{path.name!r} must remain a non-linked regular file")
        os.fsync(descriptor)
    except OSError as exc:
        raise EmbeddingStoreError(f"cannot flush {path.name!r} durably: {exc}") from exc
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = _open_absolute_directory(path, name="durability directory")
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise EmbeddingStoreError(f"cannot flush directory metadata: {exc}") from exc
    finally:
        os.close(descriptor)


def _batch_seed(config: EmbeddingStoreConfig, *, matrix: str, start: int) -> int:
    payload = _canonical_bytes(
        {
            "deterministic_seed": config.deterministic_seed,
            "matrix": matrix,
            "start": start,
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _validate_batch(
    vectors: object,
    *,
    rows: int,
    config: EmbeddingStoreConfig,
) -> np.ndarray:
    array = np.asarray(vectors)
    if array.shape != (rows, config.output_dimension):
        raise EmbeddingStoreError(
            f"encoder returned shape {array.shape}; expected {(rows, config.output_dimension)}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise EmbeddingStoreError("encoder output must be floating point")
    array = np.asarray(array, dtype=np.dtype(config.output_dtype), order="C")
    if not np.all(np.isfinite(array)):
        raise EmbeddingStoreError("encoder returned a non-finite vector")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    tolerance = 5e-3 if config.output_dtype == "float16" else 1e-5
    if not np.allclose(norms, 1.0, rtol=0.0, atol=tolerance):
        raise EmbeddingStoreError("encoder output is not unit normalized")
    return array


def _iter_texts(root: Path, sources: Sequence[_SourceFile]) -> Iterator[str]:
    for source in sources:
        for _line_number, row in _iter_source_rows(root, source):
            yield row["text"]


def _encode_matrix(
    *,
    matrix: str,
    vector_path: Path,
    root: Path,
    sources: Sequence[_SourceFile],
    row_count: int,
    model: LocalModelSpec,
    encoder: EmbeddingBatchEncoder,
    prompt: str,
    config: EmbeddingStoreConfig,
    start: int,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    array = _open_vector_memmap(
        vector_path,
        dtype=config.output_dtype,
        shape=(row_count, config.output_dimension),
        writable=True,
    )
    batch: list[str] = []
    batch_start = start
    observed = 0
    try:
        for text in _iter_texts(root, sources):
            if observed < start:
                observed += 1
                continue
            if not batch:
                batch_start = observed
            batch.append(text)
            observed += 1
            if len(batch) < config.batch_size:
                continue
            encoded = encoder.encode(
                tuple(batch),
                model_path=model.path,
                prompt=prompt,
                max_sequence_length=config.max_sequence_length,
                output_dimension=config.output_dimension,
                normalize=config.normalize,
                device=config.device,
                seed=_batch_seed(config, matrix=matrix, start=batch_start),
            )
            values = _validate_batch(encoded, rows=len(batch), config=config)
            array[batch_start:observed] = values
            array.flush()
            _fsync_regular_file(vector_path)
            checkpoint["progress"][matrix] = observed
            _write_checkpoint(checkpoint_path, checkpoint)
            batch.clear()
        if batch:
            encoded = encoder.encode(
                tuple(batch),
                model_path=model.path,
                prompt=prompt,
                max_sequence_length=config.max_sequence_length,
                output_dimension=config.output_dimension,
                normalize=config.normalize,
                device=config.device,
                seed=_batch_seed(config, matrix=matrix, start=batch_start),
            )
            values = _validate_batch(encoded, rows=len(batch), config=config)
            array[batch_start:observed] = values
            array.flush()
            _fsync_regular_file(vector_path)
            checkpoint["progress"][matrix] = observed
            _write_checkpoint(checkpoint_path, checkpoint)
    finally:
        del array
    if observed != row_count:
        raise EmbeddingStoreError(
            f"matrix {matrix!r} streamed {observed} rows; expected {row_count}"
        )


def _encode_paired_matrices(
    *,
    kind: Literal["documents", "queries"],
    current_vector_path: Path,
    old_vector_path: Path,
    root: Path,
    sources: Sequence[_SourceFile],
    row_count: int,
    current_model: LocalModelSpec,
    old_model: LocalModelSpec,
    encoder: PairedEmbeddingBatchEncoder,
    prompt: str,
    config: EmbeddingStoreConfig,
    start: int,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    current_matrix = f"current_{kind}"
    old_matrix = f"old_{kind}"
    current_array = _open_vector_memmap(
        current_vector_path,
        dtype=config.output_dtype,
        shape=(row_count, config.output_dimension),
        writable=True,
    )
    old_array = _open_vector_memmap(
        old_vector_path,
        dtype=config.output_dtype,
        shape=(row_count, config.output_dimension),
        writable=True,
    )
    batch: list[str] = []
    batch_start = start
    observed = 0

    def encode_batch(end: int) -> None:
        nonlocal batch, current_array, old_array
        encoded = encoder.encode_pair(
            tuple(batch),
            current_model_path=current_model.path,
            old_model_path=old_model.path,
            prompt=prompt,
            max_sequence_length=config.max_sequence_length,
            output_dimension=config.output_dimension,
            normalize=config.normalize,
            device=config.device,
            seed=_batch_seed(config, matrix=f"paired_{kind}", start=batch_start),
        )
        if not isinstance(encoded, tuple) or len(encoded) != 2:
            raise EmbeddingStoreError("paired encoder must return current and old matrices")
        current_values = _validate_batch(encoded[0], rows=len(batch), config=config)
        old_values = _validate_batch(encoded[1], rows=len(batch), config=config)
        current_array[batch_start:end] = current_values
        old_array[batch_start:end] = old_values
        current_array.flush()
        old_array.flush()
        _fsync_regular_file(current_vector_path)
        _fsync_regular_file(old_vector_path)
        checkpoint["progress"][current_matrix] = end
        checkpoint["progress"][old_matrix] = end
        _write_checkpoint(checkpoint_path, checkpoint)
        batch = []

    try:
        for text in _iter_texts(root, sources):
            if observed < start:
                observed += 1
                continue
            if not batch:
                batch_start = observed
            batch.append(text)
            observed += 1
            if len(batch) == config.batch_size:
                encode_batch(observed)
        if batch:
            encode_batch(observed)
    finally:
        del current_array
        del old_array
    if observed != row_count:
        raise EmbeddingStoreError(
            f"paired {kind!r} matrices streamed {observed} rows; expected {row_count}"
        )


def _validate_complete_vectors(
    path: Path,
    *,
    dtype: str,
    shape: tuple[int, int],
) -> tuple[int, str]:
    array = _open_vector_memmap(path, dtype=dtype, shape=shape, writable=False)
    tolerance = 5e-3 if dtype == "float16" else 1e-5
    try:
        for start in range(0, shape[0], _FINAL_VALIDATION_ROWS):
            values = np.asarray(array[start : start + _FINAL_VALIDATION_ROWS])
            if not np.all(np.isfinite(values)):
                raise EmbeddingStoreError(f"vector file {path.name!r} contains non-finite data")
            norms = np.linalg.norm(values.astype(np.float64), axis=1)
            if not np.allclose(norms, 1.0, rtol=0.0, atol=tolerance):
                raise EmbeddingStoreError(f"vector file {path.name!r} contains a non-unit vector")
    finally:
        del array
    try:
        byte_count = path.lstat().st_size
        file_sha256 = digest_regular_file(path, label=path.name)
    except (OSError, ArtifactIntegrityError) as exc:
        raise EmbeddingStoreError(f"cannot finalize vector file {path.name!r}: {exc}") from exc
    return byte_count, file_sha256


def _checkpoint_static(
    *,
    build_sha256: str,
    config: EmbeddingStoreConfig,
    sources: _ResolvedSources,
    row_orders: Mapping[str, RowOrderDescriptor],
    vector_files: Mapping[str, str],
) -> dict[str, object]:
    return {
        "build_sha256": build_sha256,
        "builder_version": EMBEDDING_BUILDER_VERSION,
        "config_sha256": config.sha256,
        "document_count": sources.document_count,
        "query_count": sources.query_count,
        "row_orders": {key: value.to_dict() for key, value in sorted(row_orders.items())},
        "schema_version": EMBEDDING_CHECKPOINT_SCHEMA,
        "source_inventory_sha256": sources.source_inventory_sha256,
        "staged_inventory_sha256": sources.staged_inventory_sha256,
        "vector_files": dict(sorted(vector_files.items())),
    }


def _validate_checkpoint(
    value: Mapping[str, Any],
    *,
    expected_static: Mapping[str, object],
    vector_files: Mapping[str, str],
    document_count: int,
    query_count: int,
) -> dict[str, Any]:
    observed_static = {key: value[key] for key in expected_static}
    if observed_static != expected_static:
        changed = sorted(
            key for key in expected_static if observed_static.get(key) != expected_static[key]
        )
        raise EmbeddingStoreError(f"checkpoint binding changed: {changed}")
    progress = value["progress"]
    if not isinstance(progress, Mapping) or set(progress) != set(vector_files):
        raise EmbeddingStoreError("checkpoint progress does not cover the exact matrices")
    normalized: dict[str, int] = {}
    for matrix, count in progress.items():
        maximum = document_count if matrix.endswith("documents") else query_count
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= maximum:
            raise EmbeddingStoreError(f"checkpoint progress for {matrix!r} is invalid")
        normalized[matrix] = count
    checkpoint = dict(value)
    checkpoint["progress"] = normalized
    return checkpoint


def _row_order_files_match(
    work: Path,
    expected: Mapping[str, RowOrderDescriptor],
) -> None:
    for descriptor in expected.values():
        path = work / descriptor.relative_path
        try:
            metadata = path.lstat()
            observed = digest_regular_file(path, label=descriptor.relative_path)
        except (OSError, ArtifactIntegrityError) as exc:
            raise EmbeddingStoreError(f"cannot verify row-order file: {exc}") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != descriptor.byte_count
            or observed != descriptor.file_sha256
        ):
            raise EmbeddingStoreError("row-order checkpoint artifact changed")


def _build_binding_sha256(
    *,
    config: EmbeddingStoreConfig,
    sources: _ResolvedSources,
    current_model: LocalModelSpec,
    current_encoder_id: str,
    old_model: LocalModelSpec | None,
    old_encoder_id: str | None,
) -> str:
    payload = {
        "builder_version": EMBEDDING_BUILDER_VERSION,
        "config": config.to_dict(),
        "current_model": current_model.binding(encoder_id=current_encoder_id),
        "old_model": (
            None
            if old_model is None
            else old_model.binding(encoder_id=_require_identifier("old encoder ID", old_encoder_id))
        ),
        "source_inventory_sha256": sources.source_inventory_sha256,
        "staged_inventory_sha256": sources.staged_inventory_sha256,
    }
    return _sha256(_canonical_bytes(payload))


def _initialize_checkpoint(
    *,
    work: Path,
    checkpoint_path: Path,
    selection: StagedEmbeddingSources,
    sources: _ResolvedSources,
    config: EmbeddingStoreConfig,
    vector_files: Mapping[str, str],
    build_sha256: str,
) -> tuple[dict[str, Any], dict[str, RowOrderDescriptor]]:
    try:
        work.mkdir(mode=0o700)
    except OSError as exc:
        raise EmbeddingStoreError(f"cannot create partial embedding store: {exc}") from exc
    _write_exclusive(work / "config.json", _canonical_bytes(config.to_dict()) + b"\n")
    _write_exclusive(
        work / "source-inventory.json",
        _canonical_bytes(sources.binding_payload()) + b"\n",
    )
    row_orders = {
        "documents": _row_order_descriptor(
            selection.root,
            sources.documents,
            relative_path="document-rows.jsonl",
            target=work / "document-rows.jsonl",
        ),
        "queries": _row_order_descriptor(
            selection.root,
            sources.queries,
            relative_path="query-rows.jsonl",
            target=work / "query-rows.jsonl",
        ),
    }
    for matrix, filename in vector_files.items():
        rows = sources.document_count if matrix.endswith("documents") else sources.query_count
        _create_memmap(
            work / filename,
            dtype=config.output_dtype,
            shape=(rows, config.output_dimension),
        )
    static = _checkpoint_static(
        build_sha256=build_sha256,
        config=config,
        sources=sources,
        row_orders=row_orders,
        vector_files=vector_files,
    )
    checkpoint: dict[str, Any] = {
        **static,
        "progress": {matrix: 0 for matrix in sorted(vector_files)},
    }
    _write_checkpoint(checkpoint_path, checkpoint)
    return checkpoint, row_orders


def _resume_checkpoint(
    *,
    work: Path,
    checkpoint_path: Path,
    selection: StagedEmbeddingSources,
    sources: _ResolvedSources,
    config: EmbeddingStoreConfig,
    vector_files: Mapping[str, str],
    build_sha256: str,
) -> tuple[dict[str, Any], dict[str, RowOrderDescriptor]]:
    try:
        metadata = work.lstat()
    except OSError as exc:
        raise EmbeddingStoreError(f"cannot inspect partial embedding store: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise EmbeddingStoreError("partial embedding store must be a real directory")
    observed = _load_checkpoint(checkpoint_path)
    row_order_values = observed["row_orders"]
    if not isinstance(row_order_values, Mapping):
        raise EmbeddingStoreError("checkpoint row_orders must be an object")
    row_orders = {
        key: RowOrderDescriptor.from_dict(value) for key, value in row_order_values.items()
    }
    static = _checkpoint_static(
        build_sha256=build_sha256,
        config=config,
        sources=sources,
        row_orders=row_orders,
        vector_files=vector_files,
    )
    checkpoint = _validate_checkpoint(
        observed,
        expected_static=static,
        vector_files=vector_files,
        document_count=sources.document_count,
        query_count=sources.query_count,
    )
    _verify_binding_file(
        work / "config.json",
        expected_sha256=config.sha256,
        name="embedding configuration",
    )
    _verify_binding_file(
        work / "source-inventory.json",
        expected_sha256=sources.source_inventory_sha256,
        name="embedding source inventory",
    )
    recomputed = {
        "documents": _row_order_descriptor(
            selection.root,
            sources.documents,
            relative_path="document-rows.jsonl",
            target=None,
        ),
        "queries": _row_order_descriptor(
            selection.root,
            sources.queries,
            relative_path="query-rows.jsonl",
            target=None,
        ),
    }
    if row_orders != recomputed:
        raise EmbeddingStoreError("checkpoint row order differs from the staged sources")
    _row_order_files_match(work, row_orders)
    for matrix, filename in vector_files.items():
        rows = sources.document_count if matrix.endswith("documents") else sources.query_count
        array = _open_vector_memmap(
            work / filename,
            dtype=config.output_dtype,
            shape=(rows, config.output_dimension),
            writable=True,
        )
        del array
    return checkpoint, row_orders


def _exclusive_publish(work: Path, output: Path) -> None:
    if os.path.lexists(output):
        raise EmbeddingStoreError("final embedding store already exists and cannot be overwritten")
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(work)
    destination = os.fsencode(output)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise EmbeddingStoreError("exclusive directory rename is unavailable on macOS")
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
            raise EmbeddingStoreError("exclusive directory rename is unavailable on Linux")
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
        raise EmbeddingStoreError(
            f"exclusive directory rename is unsupported on platform {sys.platform!r}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise EmbeddingStoreError(
                "final embedding store already exists and cannot be overwritten"
            )
        raise EmbeddingStoreError(
            f"cannot publish embedding store exclusively: {os.strerror(error_number)}"
        )


def build_embedding_store(
    selection: StagedEmbeddingSources,
    output_root: str | Path,
    *,
    current_model: LocalModelSpec,
    current_encoder: EmbeddingBatchEncoder | None = None,
    config: EmbeddingStoreConfig,
    old_model: LocalModelSpec | None = None,
    old_encoder: EmbeddingBatchEncoder | None = None,
    paired_encoder: PairedEmbeddingBatchEncoder | None = None,
) -> EmbeddingStoreReceipt:
    """Build or resume one immutable, inventory-bound embedding store."""

    if not isinstance(selection, StagedEmbeddingSources):
        raise EmbeddingStoreError("selection must be StagedEmbeddingSources")
    if not isinstance(current_model, LocalModelSpec):
        raise EmbeddingStoreError("current_model must be LocalModelSpec")
    if not isinstance(config, EmbeddingStoreConfig):
        raise EmbeddingStoreError("config must be EmbeddingStoreConfig")
    if paired_encoder is not None:
        if current_encoder is not None or old_encoder is not None or old_model is None:
            raise EmbeddingStoreError(
                "paired_encoder requires old_model and excludes individual encoders"
            )
        current_encoder_id = _require_identifier(
            "paired current encoder implementation ID",
            getattr(paired_encoder, "current_implementation_id", None),
        )
        old_encoder_id = _require_identifier(
            "paired old encoder implementation ID",
            getattr(paired_encoder, "old_implementation_id", None),
        )
    else:
        if current_encoder is None:
            raise EmbeddingStoreError("current_encoder is required without paired_encoder")
        current_encoder_id = _require_identifier(
            "current encoder implementation_id",
            getattr(current_encoder, "implementation_id", None),
        )
        if (old_model is None) != (old_encoder is None):
            raise EmbeddingStoreError("old_model and old_encoder must be supplied together")
        old_encoder_id = (
            None
            if old_encoder is None
            else _require_identifier(
                "old encoder implementation_id", getattr(old_encoder, "implementation_id", None)
            )
        )
    if old_model is not None:
        if not isinstance(old_model, LocalModelSpec):
            raise EmbeddingStoreError("old_model must be LocalModelSpec")
        if (old_model.revision, old_model.tree_sha256) == (
            current_model.revision,
            current_model.tree_sha256,
        ):
            raise EmbeddingStoreError("old and current model bindings must differ")

    output = Path(output_root)
    _ensure_output_parent(output)
    work, checkpoint_path = _checkpoint_paths(output)
    if os.path.lexists(output):
        raise EmbeddingStoreError("final embedding store already exists and cannot be overwritten")
    work_exists = os.path.lexists(work)
    checkpoint_exists = os.path.lexists(checkpoint_path)
    if work_exists != checkpoint_exists:
        raise EmbeddingStoreError("partial store and checkpoint sidecar must appear together")

    sources = _load_sources(selection)
    _verify_sources(selection.root, sources)
    _verify_model(current_model)
    if old_model is not None:
        _verify_model(old_model)
    vector_files = _matrix_files(old_model is not None)
    build_sha256 = _build_binding_sha256(
        config=config,
        sources=sources,
        current_model=current_model,
        current_encoder_id=current_encoder_id,
        old_model=old_model,
        old_encoder_id=old_encoder_id,
    )

    if work_exists:
        checkpoint, row_orders = _resume_checkpoint(
            work=work,
            checkpoint_path=checkpoint_path,
            selection=selection,
            sources=sources,
            config=config,
            vector_files=vector_files,
            build_sha256=build_sha256,
        )
    else:
        checkpoint, row_orders = _initialize_checkpoint(
            work=work,
            checkpoint_path=checkpoint_path,
            selection=selection,
            sources=sources,
            config=config,
            vector_files=vector_files,
            build_sha256=build_sha256,
        )

    plans: list[
        tuple[
            str,
            Sequence[_SourceFile],
            int,
            LocalModelSpec,
            EmbeddingBatchEncoder,
            str,
        ]
    ] = []
    if paired_encoder is None:
        if current_encoder is None:
            raise EmbeddingStoreError("current encoder disappeared after validation")
        plans.extend(
            [
                (
                    "current_documents",
                    sources.documents,
                    sources.document_count,
                    current_model,
                    current_encoder,
                    config.document_prompt,
                ),
                (
                    "current_queries",
                    sources.queries,
                    sources.query_count,
                    current_model,
                    current_encoder,
                    config.query_prompt,
                ),
            ]
        )
    if paired_encoder is None and old_model is not None and old_encoder is not None:
        plans.extend(
            [
                (
                    "old_documents",
                    sources.documents,
                    sources.document_count,
                    old_model,
                    old_encoder,
                    config.document_prompt,
                ),
                (
                    "old_queries",
                    sources.queries,
                    sources.query_count,
                    old_model,
                    old_encoder,
                    config.query_prompt,
                ),
            ]
        )

    if paired_encoder is not None:
        if old_model is None:
            raise EmbeddingStoreError("paired encoder lost its old model binding")
        for kind, source_files, row_count, prompt in (
            ("documents", sources.documents, sources.document_count, config.document_prompt),
            ("queries", sources.queries, sources.query_count, config.query_prompt),
        ):
            current_matrix = f"current_{kind}"
            old_matrix = f"old_{kind}"
            current_completed = checkpoint["progress"][current_matrix]
            old_completed = checkpoint["progress"][old_matrix]
            if current_completed != old_completed:
                raise EmbeddingStoreError(f"paired checkpoint progress differs for {kind}")
            if current_completed < row_count:
                _encode_paired_matrices(
                    kind=kind,  # type: ignore[arg-type]
                    current_vector_path=work / vector_files[current_matrix],
                    old_vector_path=work / vector_files[old_matrix],
                    root=selection.root,
                    sources=source_files,
                    row_count=row_count,
                    current_model=current_model,
                    old_model=old_model,
                    encoder=paired_encoder,
                    prompt=prompt,
                    config=config,
                    start=current_completed,
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                )

    for matrix, source_files, row_count, model, encoder, prompt in plans:
        completed = checkpoint["progress"][matrix]
        if completed < row_count:
            _encode_matrix(
                matrix=matrix,
                vector_path=work / vector_files[matrix],
                root=selection.root,
                sources=source_files,
                row_count=row_count,
                model=model,
                encoder=encoder,
                prompt=prompt,
                config=config,
                start=completed,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
            )

    if any(
        checkpoint["progress"][matrix]
        != (sources.document_count if matrix.endswith("documents") else sources.query_count)
        for matrix in vector_files
    ):
        raise EmbeddingStoreError("checkpoint is incomplete after matrix construction")

    _verify_model(current_model)
    if old_model is not None:
        _verify_model(old_model)

    vectors: dict[str, VectorDescriptor] = {}
    current_binding = current_model.binding(encoder_id=current_encoder_id)
    old_binding = (
        None
        if old_model is None
        else old_model.binding(encoder_id=_require_identifier("old encoder ID", old_encoder_id))
    )
    for matrix, filename in vector_files.items():
        is_document = matrix.endswith("documents")
        rows = sources.document_count if is_document else sources.query_count
        model = old_model if matrix.startswith("old_") else current_model
        if model is None:
            raise EmbeddingStoreError("old vector matrix has no old model binding")
        prompt = config.document_prompt if is_document else config.query_prompt
        byte_count, file_sha256 = _validate_complete_vectors(
            work / filename,
            dtype=config.output_dtype,
            shape=(rows, config.output_dimension),
        )
        vectors[matrix] = VectorDescriptor(
            relative_path=filename,
            dtype=config.output_dtype,
            shape=(rows, config.output_dimension),
            row_order_sha256=row_orders["documents" if is_document else "queries"].row_order_sha256,
            byte_count=byte_count,
            file_sha256=file_sha256,
            model_tree_sha256=model.tree_sha256,
            model_revision=model.revision,
            prompt_sha256=_sha256(prompt.encode("utf-8")),
        )

    receipt = EmbeddingStoreReceipt(
        staged_inventory_sha256=sources.staged_inventory_sha256,
        source_inventory_sha256=sources.source_inventory_sha256,
        config_sha256=config.sha256,
        document_count=sources.document_count,
        query_count=sources.query_count,
        current_model=current_binding,
        old_model=old_binding,
        row_orders=row_orders,
        vectors=vectors,
    )
    _write_exclusive(work / "receipt.json", receipt.canonical_bytes() + b"\n")
    _fsync_directory(work)
    _exclusive_publish(work, output)
    _fsync_directory(output.parent)
    try:
        checkpoint_path.unlink()
    except OSError as exc:
        raise EmbeddingStoreError(
            "embedding store was published but checkpoint cleanup failed"
        ) from exc
    _fsync_directory(output.parent)
    return receipt


def load_embedding_store_receipt(root: str | Path) -> EmbeddingStoreReceipt:
    """Load one canonical final receipt without opening vector payloads."""

    path = Path(root) / "receipt.json"
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=_CONTROL_MAX_BYTES,
            label="embedding store receipt",
        )
    except ArtifactIntegrityError as exc:
        raise EmbeddingStoreError(f"cannot load embedding store receipt: {exc}") from exc
    value = _decode_json(encoded, name="embedding store receipt")
    receipt = EmbeddingStoreReceipt.from_dict(value)
    if encoded != receipt.canonical_bytes() + b"\n":
        raise EmbeddingStoreError("embedding store receipt is not canonical")
    return receipt


def verify_embedding_store(root: str | Path) -> EmbeddingStoreReceipt:
    """Verify final membership, descriptors, vector geometry, and receipt bytes."""

    store = Path(root)
    try:
        metadata = store.lstat()
    except OSError as exc:
        raise EmbeddingStoreError(f"cannot inspect embedding store: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise EmbeddingStoreError("embedding store must be a real directory")
    receipt = load_embedding_store_receipt(store)
    expected = {"config.json", "receipt.json", "source-inventory.json"}
    expected.update(descriptor.relative_path for descriptor in receipt.row_orders.values())
    expected.update(descriptor.relative_path for descriptor in receipt.vectors.values())
    try:
        observed = {child.name for child in store.iterdir()}
    except OSError as exc:
        raise EmbeddingStoreError(f"cannot scan embedding store: {exc}") from exc
    if observed != expected:
        raise EmbeddingStoreError(
            f"embedding store membership differs; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    _verify_binding_file(
        store / "config.json",
        expected_sha256=receipt.config_sha256,
        name="embedding configuration",
    )
    _verify_binding_file(
        store / "source-inventory.json",
        expected_sha256=receipt.source_inventory_sha256,
        name="embedding source inventory",
    )
    for descriptor in receipt.row_orders.values():
        path = store / descriptor.relative_path
        try:
            file_metadata = path.lstat()
            digest = digest_regular_file(path, label=descriptor.relative_path)
        except (OSError, ArtifactIntegrityError) as exc:
            raise EmbeddingStoreError(f"cannot verify row-order artifact: {exc}") from exc
        if (
            not stat.S_ISREG(file_metadata.st_mode)
            or file_metadata.st_nlink != 1
            or file_metadata.st_size != descriptor.byte_count
            or digest != descriptor.file_sha256
        ):
            raise EmbeddingStoreError("row-order artifact differs from its descriptor")
    for descriptor in receipt.vectors.values():
        byte_count, file_sha256 = _validate_complete_vectors(
            store / descriptor.relative_path,
            dtype=descriptor.dtype,
            shape=descriptor.shape,
        )
        if byte_count != descriptor.byte_count or file_sha256 != descriptor.file_sha256:
            raise EmbeddingStoreError("vector artifact differs from its descriptor")
    return receipt
