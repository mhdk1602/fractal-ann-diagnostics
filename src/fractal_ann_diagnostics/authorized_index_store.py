"""Immutable authorized HNSW stores for sealed confirmatory execution.

The builder indexes only the active old-model document matrix.  The current
document matrix is verified and pinned as exact-search truth, but its values are
never supplied to the HNSW backend.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import secrets
import shutil
import stat
import sys
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, Literal, Protocol

import numpy as np

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    read_secure_regular_file,
)
from .compiled_policy import CompiledMaskDescriptor, load_compiled_policy_catalog
from .embedding_store import (
    EMBEDDING_BUILDER_VERSION,
    EmbeddingStoreReceipt,
    VectorDescriptor,
    verify_embedding_store,
)
from .policy import PolicyDecision
from .policy_intervention import (
    CATALOG_FILENAME,
    PolicyInterventionReceipt,
    load_policy_intervention_receipt,
)
from .policy_intervention import (
    RECEIPT_FILENAME as POLICY_RECEIPT_FILENAME,
)
from .retrieval import AuthorizedHNSWIndex, DistanceMetric

AUTHORIZED_INDEX_CONFIG_SCHEMA = "fractal-authorized-index-config-v1"
AUTHORIZED_INDEX_RECEIPT_SCHEMA = "fractal-authorized-index-store-v1"
AUTHORIZED_INDEX_BUILDER_IDENTITY = "fractal-authorized-index-builder-v1"
HNSWLIB_BACKEND_ID = "hnswlib-python-v1"
FAILURE_POLICY = "fail-clean-no-resume-v1"
INDEX_NUM_THREADS = 1
ROW_MAP_DTYPE = "<i8"
INDEX_INPUT_DTYPE = "<f4"

CONFIG_FILENAME = "config.json"
RECEIPT_FILENAME = "receipt.json"
INDEX_DIRECTORY = "indexes"
ROW_MAP_DIRECTORY = "row-maps"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:[a-z0-9.+-]*)?$")
_CONFIG_FIELDS = {
    "backend_build_sha256",
    "backend_id",
    "backend_version",
    "batch_size",
    "builder_identity",
    "ef_construction",
    "failure_policy",
    "m",
    "metric",
    "num_threads",
    "random_seed",
    "schema_version",
    "verification_ef",
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
    "role",
    "row_order_sha256",
    "shape",
}
_INDEX_FIELDS = {
    "authorized_count",
    "build_binding_sha256",
    "index_byte_count",
    "index_path",
    "index_sha256",
    "mask_id",
    "mask_sha256",
    "row_map_byte_count",
    "row_map_dtype",
    "row_map_path",
    "row_map_sha256",
    "row_map_shape",
}
_RECEIPT_FIELDS = {
    "backend_build_sha256",
    "backend_id",
    "backend_version",
    "builder_identity",
    "config_sha256",
    "current_truth_vector",
    "document_count",
    "document_row_order_sha256",
    "document_universe_sha256",
    "embedding_receipt_sha256",
    "failure_policy",
    "indexes",
    "old_active_vector",
    "payload_tree_sha256",
    "policy_catalog_sha256",
    "policy_execution_artifact_sha256",
    "policy_receipt_sha256",
    "policy_revision",
    "schema_version",
}
_CONTROL_MAX_BYTES = 16 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_BIT_COUNTS = tuple(value.bit_count() for value in range(256))


class AuthorizedIndexStoreError(RuntimeError):
    """Raised when an authorized index store is not exactly admissible."""


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
        raise AuthorizedIndexStoreError("index metadata must be finite canonical JSON") from exc


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuthorizedIndexStoreError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise AuthorizedIndexStoreError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizedIndexStoreError(f"cannot decode {label}") from exc
    if not isinstance(value, Mapping):
        raise AuthorizedIndexStoreError(f"{label} must contain one object")
    return value


def _closed_mapping(value: object, *, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AuthorizedIndexStoreError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise AuthorizedIndexStoreError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _require_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AuthorizedIndexStoreError(f"{name} must be a canonical non-empty string")
    return value


def _require_identifier(name: str, value: object) -> str:
    text = _require_text(name, value)
    if _IDENTIFIER.fullmatch(text) is None:
        raise AuthorizedIndexStoreError(f"{name} must be a lowercase safe identifier")
    return text


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AuthorizedIndexStoreError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_integer(name: str, value: object, *, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise AuthorizedIndexStoreError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _relative_path(name: str, value: object, *, suffix: str) -> str:
    text = _require_text(name, value)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
        or path.suffix != suffix
    ):
        raise AuthorizedIndexStoreError(f"{name} must be a canonical relative {suffix} path")
    return text


@dataclass(frozen=True)
class AuthorizedIndexConfig:
    """Frozen backend and construction parameters for every policy-state index."""

    backend_version: str
    backend_build_sha256: str
    metric: Literal["cosine", "l2", "ip"]
    m: int
    ef_construction: int
    random_seed: int
    batch_size: int
    verification_ef: int = 64
    backend_id: str = HNSWLIB_BACKEND_ID
    num_threads: int = INDEX_NUM_THREADS
    builder_identity: str = AUTHORIZED_INDEX_BUILDER_IDENTITY
    failure_policy: str = FAILURE_POLICY
    schema_version: str = AUTHORIZED_INDEX_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        version = _require_text("backend_version", self.backend_version)
        if _VERSION.fullmatch(version) is None:
            raise AuthorizedIndexStoreError("backend_version must be an exact release version")
        _require_sha256("backend_build_sha256", self.backend_build_sha256)
        if self.metric not in {"cosine", "l2", "ip"}:
            raise AuthorizedIndexStoreError("metric must be cosine, l2, or ip")
        _require_positive_integer("m", self.m, maximum=4096)
        _require_positive_integer("ef_construction", self.ef_construction, maximum=1_000_000)
        if self.ef_construction < self.m:
            raise AuthorizedIndexStoreError("ef_construction must be at least m")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed < 2**31
        ):
            raise AuthorizedIndexStoreError("random_seed must be an unsigned 31-bit integer")
        _require_positive_integer("batch_size", self.batch_size, maximum=1_000_000)
        _require_positive_integer("verification_ef", self.verification_ef, maximum=1_000_000)
        if self.backend_id != HNSWLIB_BACKEND_ID:
            raise AuthorizedIndexStoreError(f"backend_id must equal {HNSWLIB_BACKEND_ID!r}")
        if self.num_threads != INDEX_NUM_THREADS:
            raise AuthorizedIndexStoreError("authorized indexes require exactly one thread")
        if self.builder_identity != AUTHORIZED_INDEX_BUILDER_IDENTITY:
            raise AuthorizedIndexStoreError("builder_identity differs from the frozen builder")
        if self.failure_policy != FAILURE_POLICY:
            raise AuthorizedIndexStoreError("failure_policy differs from fail-clean execution")
        if self.schema_version != AUTHORIZED_INDEX_CONFIG_SCHEMA:
            raise AuthorizedIndexStoreError("authorized index config schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_build_sha256": self.backend_build_sha256,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "batch_size": self.batch_size,
            "builder_identity": self.builder_identity,
            "ef_construction": self.ef_construction,
            "failure_policy": self.failure_policy,
            "m": self.m,
            "metric": self.metric,
            "num_threads": self.num_threads,
            "random_seed": self.random_seed,
            "schema_version": self.schema_version,
            "verification_ef": self.verification_ef,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> AuthorizedIndexConfig:
        row = _closed_mapping(value, fields=_CONFIG_FIELDS, label="authorized index config")
        return cls(
            backend_version=row["backend_version"],
            backend_build_sha256=row["backend_build_sha256"],
            metric=row["metric"],
            m=row["m"],
            ef_construction=row["ef_construction"],
            random_seed=row["random_seed"],
            batch_size=row["batch_size"],
            verification_ef=row["verification_ef"],
            backend_id=row["backend_id"],
            num_threads=row["num_threads"],
            builder_identity=row["builder_identity"],
            failure_policy=row["failure_policy"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class VectorSourceBinding:
    """One embedding matrix identity without access to its values."""

    role: Literal["active-old-stale", "current-exact-truth"]
    relative_path: str
    dtype: str
    shape: tuple[int, int]
    row_order_sha256: str
    byte_count: int
    file_sha256: str
    model_tree_sha256: str
    model_revision: str
    prompt_sha256: str
    builder_version: str

    def __post_init__(self) -> None:
        if self.role not in {"active-old-stale", "current-exact-truth"}:
            raise AuthorizedIndexStoreError("vector role differs")
        object.__setattr__(
            self,
            "relative_path",
            _relative_path("vector relative_path", self.relative_path, suffix=".npy"),
        )
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise AuthorizedIndexStoreError("vector dtype is invalid") from exc
        if dtype not in {np.dtype("float16"), np.dtype("float32")}:
            raise AuthorizedIndexStoreError("vector dtype must be float16 or float32")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(type(value) is not int or value <= 0 for value in self.shape)
        ):
            raise AuthorizedIndexStoreError("vector shape needs two positive integers")
        _require_sha256("vector row_order_sha256", self.row_order_sha256)
        _require_positive_integer("vector byte_count", self.byte_count)
        _require_sha256("vector file_sha256", self.file_sha256)
        _require_sha256("vector model_tree_sha256", self.model_tree_sha256)
        _require_text("vector model_revision", self.model_revision)
        _require_sha256("vector prompt_sha256", self.prompt_sha256)
        if self.builder_version != EMBEDDING_BUILDER_VERSION:
            raise AuthorizedIndexStoreError("vector builder version differs")

    @classmethod
    def from_descriptor(
        cls,
        descriptor: VectorDescriptor,
        *,
        role: Literal["active-old-stale", "current-exact-truth"],
    ) -> VectorSourceBinding:
        return cls(
            role=role,
            relative_path=descriptor.relative_path,
            dtype=descriptor.dtype,
            shape=descriptor.shape,
            row_order_sha256=descriptor.row_order_sha256,
            byte_count=descriptor.byte_count,
            file_sha256=descriptor.file_sha256,
            model_tree_sha256=descriptor.model_tree_sha256,
            model_revision=descriptor.model_revision,
            prompt_sha256=descriptor.prompt_sha256,
            builder_version=descriptor.builder_version,
        )

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
            "role": self.role,
            "row_order_sha256": self.row_order_sha256,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: object) -> VectorSourceBinding:
        row = _closed_mapping(value, fields=_VECTOR_FIELDS, label="vector source binding")
        shape = row["shape"]
        if not isinstance(shape, list):
            raise AuthorizedIndexStoreError("vector shape must be a JSON array")
        return cls(
            role=row["role"],
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


@dataclass(frozen=True, order=True)
class AuthorizedIndexArtifact:
    """Frozen HNSW bytes and their local-to-global row map."""

    mask_id: str
    mask_sha256: str
    authorized_count: int
    index_path: str
    index_sha256: str
    index_byte_count: int
    row_map_path: str
    row_map_sha256: str
    row_map_byte_count: int
    row_map_shape: tuple[int]
    build_binding_sha256: str
    row_map_dtype: str = ROW_MAP_DTYPE

    def __post_init__(self) -> None:
        _require_identifier("mask_id", self.mask_id)
        _require_sha256("mask_sha256", self.mask_sha256)
        _require_positive_integer("authorized_count", self.authorized_count)
        object.__setattr__(
            self, "index_path", _relative_path("index_path", self.index_path, suffix=".hnsw")
        )
        _require_sha256("index_sha256", self.index_sha256)
        _require_positive_integer("index_byte_count", self.index_byte_count)
        object.__setattr__(
            self,
            "row_map_path",
            _relative_path("row_map_path", self.row_map_path, suffix=".npy"),
        )
        _require_sha256("row_map_sha256", self.row_map_sha256)
        _require_positive_integer("row_map_byte_count", self.row_map_byte_count)
        if self.row_map_shape != (self.authorized_count,):
            raise AuthorizedIndexStoreError("row map shape differs from authorized_count")
        _require_sha256("build_binding_sha256", self.build_binding_sha256)
        if self.row_map_dtype != ROW_MAP_DTYPE:
            raise AuthorizedIndexStoreError(f"row_map_dtype must equal {ROW_MAP_DTYPE!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "authorized_count": self.authorized_count,
            "build_binding_sha256": self.build_binding_sha256,
            "index_byte_count": self.index_byte_count,
            "index_path": self.index_path,
            "index_sha256": self.index_sha256,
            "mask_id": self.mask_id,
            "mask_sha256": self.mask_sha256,
            "row_map_byte_count": self.row_map_byte_count,
            "row_map_dtype": self.row_map_dtype,
            "row_map_path": self.row_map_path,
            "row_map_sha256": self.row_map_sha256,
            "row_map_shape": list(self.row_map_shape),
        }

    @classmethod
    def from_dict(cls, value: object) -> AuthorizedIndexArtifact:
        row = _closed_mapping(value, fields=_INDEX_FIELDS, label="authorized index artifact")
        shape = row["row_map_shape"]
        if not isinstance(shape, list):
            raise AuthorizedIndexStoreError("row_map_shape must be a JSON array")
        return cls(
            mask_id=row["mask_id"],
            mask_sha256=row["mask_sha256"],
            authorized_count=row["authorized_count"],
            index_path=row["index_path"],
            index_sha256=row["index_sha256"],
            index_byte_count=row["index_byte_count"],
            row_map_path=row["row_map_path"],
            row_map_sha256=row["row_map_sha256"],
            row_map_byte_count=row["row_map_byte_count"],
            row_map_shape=tuple(shape),
            build_binding_sha256=row["build_binding_sha256"],
            row_map_dtype=row["row_map_dtype"],
        )


@dataclass(frozen=True)
class AuthorizedIndexStoreReceipt:
    """Canonical binding for all authorized index payloads and their sources."""

    config_sha256: str
    embedding_receipt_sha256: str
    policy_receipt_sha256: str
    policy_catalog_sha256: str
    policy_execution_artifact_sha256: str
    policy_revision: str
    document_count: int
    document_universe_sha256: str
    document_row_order_sha256: str
    old_active_vector: VectorSourceBinding
    current_truth_vector: VectorSourceBinding
    indexes: tuple[AuthorizedIndexArtifact, ...]
    payload_tree_sha256: str
    backend_version: str
    backend_build_sha256: str
    backend_id: str = HNSWLIB_BACKEND_ID
    builder_identity: str = AUTHORIZED_INDEX_BUILDER_IDENTITY
    failure_policy: str = FAILURE_POLICY
    schema_version: str = AUTHORIZED_INDEX_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "config_sha256",
            "embedding_receipt_sha256",
            "policy_receipt_sha256",
            "policy_catalog_sha256",
            "policy_execution_artifact_sha256",
            "document_universe_sha256",
            "document_row_order_sha256",
            "payload_tree_sha256",
            "backend_build_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_text("policy_revision", self.policy_revision)
        _require_positive_integer("document_count", self.document_count)
        if not isinstance(self.old_active_vector, VectorSourceBinding) or (
            self.old_active_vector.role != "active-old-stale"
        ):
            raise AuthorizedIndexStoreError("old_active_vector has the wrong role")
        if not isinstance(self.current_truth_vector, VectorSourceBinding) or (
            self.current_truth_vector.role != "current-exact-truth"
        ):
            raise AuthorizedIndexStoreError("current_truth_vector has the wrong role")
        if (
            self.old_active_vector.shape != self.current_truth_vector.shape
            or self.old_active_vector.shape[0] != self.document_count
            or self.old_active_vector.row_order_sha256 != self.document_row_order_sha256
            or self.current_truth_vector.row_order_sha256 != self.document_row_order_sha256
        ):
            raise AuthorizedIndexStoreError("vector matrices differ in shape or row order")
        if self.document_universe_sha256 != self.document_row_order_sha256:
            raise AuthorizedIndexStoreError(
                "document universe must equal the canonical document row-order digest"
            )
        if (
            self.old_active_vector.file_sha256 == self.current_truth_vector.file_sha256
            or self.old_active_vector.model_tree_sha256
            == self.current_truth_vector.model_tree_sha256
        ):
            raise AuthorizedIndexStoreError("old active and current truth vectors must differ")
        indexes = tuple(self.indexes)
        if not indexes or not all(isinstance(item, AuthorizedIndexArtifact) for item in indexes):
            raise AuthorizedIndexStoreError("indexes must contain typed artifact records")
        canonical = tuple(sorted(indexes, key=lambda item: item.mask_id.encode("ascii")))
        if indexes != canonical or len({item.mask_id for item in indexes}) != len(indexes):
            raise AuthorizedIndexStoreError("indexes must be uniquely sorted by mask_id")
        paths = [path for item in indexes for path in (item.index_path, item.row_map_path)]
        if len(paths) != len(set(paths)):
            raise AuthorizedIndexStoreError("authorized index artifacts repeat a path")
        object.__setattr__(self, "indexes", indexes)
        version = _require_text("backend_version", self.backend_version)
        if _VERSION.fullmatch(version) is None:
            raise AuthorizedIndexStoreError("receipt backend_version is not exact")
        if self.backend_id != HNSWLIB_BACKEND_ID:
            raise AuthorizedIndexStoreError("receipt backend_id differs")
        if self.builder_identity != AUTHORIZED_INDEX_BUILDER_IDENTITY:
            raise AuthorizedIndexStoreError("receipt builder_identity differs")
        if self.failure_policy != FAILURE_POLICY:
            raise AuthorizedIndexStoreError("receipt failure_policy differs")
        if self.schema_version != AUTHORIZED_INDEX_RECEIPT_SCHEMA:
            raise AuthorizedIndexStoreError("authorized index receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_build_sha256": self.backend_build_sha256,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "builder_identity": self.builder_identity,
            "config_sha256": self.config_sha256,
            "current_truth_vector": self.current_truth_vector.to_dict(),
            "document_count": self.document_count,
            "document_row_order_sha256": self.document_row_order_sha256,
            "document_universe_sha256": self.document_universe_sha256,
            "embedding_receipt_sha256": self.embedding_receipt_sha256,
            "failure_policy": self.failure_policy,
            "indexes": [item.to_dict() for item in self.indexes],
            "old_active_vector": self.old_active_vector.to_dict(),
            "payload_tree_sha256": self.payload_tree_sha256,
            "policy_catalog_sha256": self.policy_catalog_sha256,
            "policy_execution_artifact_sha256": self.policy_execution_artifact_sha256,
            "policy_receipt_sha256": self.policy_receipt_sha256,
            "policy_revision": self.policy_revision,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> AuthorizedIndexStoreReceipt:
        row = _closed_mapping(value, fields=_RECEIPT_FIELDS, label="authorized index receipt")
        indexes = row["indexes"]
        if not isinstance(indexes, list):
            raise AuthorizedIndexStoreError("receipt indexes must be a JSON array")
        return cls(
            config_sha256=row["config_sha256"],
            embedding_receipt_sha256=row["embedding_receipt_sha256"],
            policy_receipt_sha256=row["policy_receipt_sha256"],
            policy_catalog_sha256=row["policy_catalog_sha256"],
            policy_execution_artifact_sha256=row["policy_execution_artifact_sha256"],
            policy_revision=row["policy_revision"],
            document_count=row["document_count"],
            document_universe_sha256=row["document_universe_sha256"],
            document_row_order_sha256=row["document_row_order_sha256"],
            old_active_vector=VectorSourceBinding.from_dict(row["old_active_vector"]),
            current_truth_vector=VectorSourceBinding.from_dict(row["current_truth_vector"]),
            indexes=tuple(AuthorizedIndexArtifact.from_dict(item) for item in indexes),
            payload_tree_sha256=row["payload_tree_sha256"],
            backend_version=row["backend_version"],
            backend_build_sha256=row["backend_build_sha256"],
            backend_id=row["backend_id"],
            builder_identity=row["builder_identity"],
            failure_policy=row["failure_policy"],
            schema_version=row["schema_version"],
        )


class HNSWIndex(Protocol):
    """Minimal index surface needed by the sealed builder and verifier."""

    def init_index(
        self,
        *,
        max_elements: int,
        ef_construction: int,
        M: int,
        random_seed: int,
        allow_replace_deleted: bool,
    ) -> None: ...

    def set_num_threads(self, count: int) -> None: ...

    def add_items(self, vectors: np.ndarray, labels: np.ndarray, *, num_threads: int) -> None: ...

    def save_index(self, path: str) -> None: ...

    def load_index(self, path: str, *, max_elements: int) -> None: ...

    def set_ef(self, value: int) -> None: ...

    def knn_query(
        self, vectors: np.ndarray, *, k: int, num_threads: int
    ) -> tuple[np.ndarray, np.ndarray]: ...


class AuthorizedIndexBackend(Protocol):
    """Injected backend whose exact installed bytes are configuration-bound."""

    backend_id: str
    package_version: str
    build_sha256: str

    def create_index(self, *, metric: str, dimension: int) -> HNSWIndex: ...


class HnswlibBackend:
    """Lazy adapter that pins the installed hnswlib extension file."""

    backend_id = HNSWLIB_BACKEND_ID

    def __init__(self) -> None:
        try:
            import hnswlib
        except ImportError as exc:
            raise AuthorizedIndexStoreError(
                "hnswlib is required for production index builds"
            ) from exc
        module_path = Path(hnswlib.__file__ or "")
        if not module_path.is_absolute():
            raise AuthorizedIndexStoreError("hnswlib module path is not absolute")
        try:
            version = importlib.metadata.version("hnswlib")
            build_sha256 = digest_regular_file(module_path, label="hnswlib extension")
        except (importlib.metadata.PackageNotFoundError, ArtifactIntegrityError) as exc:
            raise AuthorizedIndexStoreError("cannot pin the installed hnswlib build") from exc
        self.package_version = version
        self.build_sha256 = build_sha256
        self._module = hnswlib

    def create_index(self, *, metric: str, dimension: int) -> HNSWIndex:
        return self._module.Index(space=metric, dim=dimension)


@dataclass(frozen=True)
class _SourceAdmission:
    embedding: EmbeddingStoreReceipt
    policy: PolicyInterventionReceipt
    catalog_sha256: str
    masks: tuple[CompiledMaskDescriptor, ...]
    old_vector: VectorSourceBinding
    current_vector: VectorSourceBinding


@dataclass(frozen=True)
class AuthorizedIndexStoreVerification:
    root: Path
    receipt_sha256: str
    payload_tree_sha256: str
    mask_ids: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedDocumentMatrices:
    """Receipt-bound read-only document matrices held open for one execution."""

    old_active: np.memmap
    current_truth: np.memmap
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("old_active", "current_truth"):
            value = getattr(self, name)
            if (
                not isinstance(value, np.memmap)
                or value.ndim != 2
                or value.dtype != np.dtype(np.float32)
                or value.flags.writeable
                or not value.flags.c_contiguous
            ):
                raise AuthorizedIndexStoreError(
                    f"{name} must be one read-only C-contiguous float32 memmap"
                )
        if self.old_active.shape != self.current_truth.shape:
            raise AuthorizedIndexStoreError("verified document matrix shapes differ")
        _require_sha256("receipt_sha256", self.receipt_sha256)


class VerifiedAuthorizedIndexProvider:
    """Load only receipt-bound HNSW indexes selected by a live policy mask.

    Construction verifies the full index store and both source packages. Later
    lookups re-open the selected index and row map through no-follow file
    descriptors, check their immutable pins again, and compare the row map with
    the exact live authorization decision before exposing a queryable object.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        embedding_store_root: str | Path,
        policy_intervention_root: str | Path,
        expected_embedding_receipt_sha256: str,
        expected_policy_receipt_sha256: str,
        expected_store_receipt_sha256: str,
        backend: AuthorizedIndexBackend,
    ) -> None:
        self.root = Path(root)
        self.embedding_store_root = Path(embedding_store_root)
        self.policy_intervention_root = Path(policy_intervention_root)
        for name, path in (
            ("root", self.root),
            ("embedding_store_root", self.embedding_store_root),
            ("policy_intervention_root", self.policy_intervention_root),
        ):
            if (
                not path.is_absolute()
                or path.anchor != "/"
                or any(part in {".", ".."} for part in path.parts)
            ):
                raise AuthorizedIndexStoreError(f"{name} must be an absolute canonical path")
        _require_sha256(
            "expected_embedding_receipt_sha256",
            expected_embedding_receipt_sha256,
        )
        _require_sha256(
            "expected_policy_receipt_sha256",
            expected_policy_receipt_sha256,
        )
        _require_sha256(
            "expected_store_receipt_sha256",
            expected_store_receipt_sha256,
        )
        verification = verify_authorized_index_store(
            self.root,
            embedding_store_root=self.embedding_store_root,
            policy_intervention_root=self.policy_intervention_root,
            expected_embedding_receipt_sha256=expected_embedding_receipt_sha256,
            expected_policy_receipt_sha256=expected_policy_receipt_sha256,
            backend=backend,
            expected_store_receipt_sha256=expected_store_receipt_sha256,
        )
        if verification.receipt_sha256 != expected_store_receipt_sha256:
            raise AuthorizedIndexStoreError(
                "authorized index verification returned another receipt"
            )
        self.receipt = load_authorized_index_store_receipt(self.root)
        self.config = _load_config(self.root)
        _check_backend(backend, self.config)
        self.backend = backend
        self._by_mask_sha256 = {item.mask_sha256: item for item in self.receipt.indexes}
        if len(self._by_mask_sha256) != len(self.receipt.indexes):
            raise AuthorizedIndexStoreError("authorized index masks do not have unique digests")
        self._cache: dict[str, AuthorizedHNSWIndex] = {}
        self._lock = RLock()

    @property
    def retrieval_metric(self) -> DistanceMetric:
        """Return the exact-search metric bound by the verified store config."""

        if self.config.metric == "l2":
            return "euclidean"
        if self.config.metric == "cosine":
            return "cosine"
        raise AuthorizedIndexStoreError("inner-product indexes are not admitted by the retriever")

    def index_for(self, authorization: PolicyDecision) -> AuthorizedHNSWIndex:
        """Return the exact prebuilt index selected by one admitted decision."""

        if not isinstance(authorization, PolicyDecision) or not authorization.available:
            raise AuthorizedIndexStoreError("authorization must be one available policy decision")
        if (
            authorization.policy_version != self.receipt.policy_revision
            or authorization.document_universe_sha256 != self.receipt.document_universe_sha256
            or authorization.authorized_mask.shape != (self.receipt.document_count,)
            or authorization.authorized_count <= 0
        ):
            raise AuthorizedIndexStoreError(
                "authorization differs from the frozen index-store universe"
            )
        packed = np.packbits(authorization.authorized_mask, bitorder="little").tobytes()
        mask_sha256 = hashlib.sha256(packed).hexdigest()
        try:
            artifact = self._by_mask_sha256[mask_sha256]
        except KeyError as exc:
            raise AuthorizedIndexStoreError(
                "live authorization did not select a frozen policy mask"
            ) from exc
        if artifact.authorized_count != authorization.authorized_count:
            raise AuthorizedIndexStoreError("live mask count differs from the frozen index")

        with self._lock:
            cached = self._cache.get(mask_sha256)
            if cached is not None:
                return cached
            with _secure_npy(
                self.root,
                artifact.row_map_path,
                expected_byte_count=artifact.row_map_byte_count,
                expected_sha256=artifact.row_map_sha256,
                expected_dtype=artifact.row_map_dtype,
                expected_shape=artifact.row_map_shape,
                label=f"authorized row map {artifact.mask_id}",
            ) as mapped_rows:
                rows = np.array(mapped_rows, dtype=np.int64, copy=True)
            expected_rows = np.flatnonzero(authorization.authorized_mask).astype(np.int64)
            if not np.array_equal(rows, expected_rows):
                raise AuthorizedIndexStoreError(
                    "authorized row map differs from the live policy decision"
                )
            index = self.backend.create_index(
                metric=self.config.metric,
                dimension=self.receipt.old_active_vector.shape[1],
            )
            with _secure_backend_path(
                self.root,
                artifact.index_path,
                expected_byte_count=artifact.index_byte_count,
                expected_sha256=artifact.index_sha256,
                label=f"authorized index {artifact.mask_id}",
            ) as backend_path:
                index.load_index(backend_path, max_elements=artifact.authorized_count)
            index.set_num_threads(INDEX_NUM_THREADS)
            index.set_ef(min(self.config.verification_ef, artifact.authorized_count))
            admitted = AuthorizedHNSWIndex.from_loaded_backend(
                index,
                rows,
                n_documents=self.receipt.document_count,
                dimension=self.receipt.old_active_vector.shape[1],
                metric=self.retrieval_metric,
            )
            self._cache[mask_sha256] = admitted
            return admitted


def _directory_flags() -> int:
    missing = [name for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW") if not hasattr(os, name)]
    if missing or os.open not in os.supports_dir_fd:
        raise AuthorizedIndexStoreError("secure POSIX no-follow directory access is unavailable")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _file_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def _open_error(label: str, exc: OSError) -> AuthorizedIndexStoreError:
    if exc.errno == errno.ENOENT:
        return AuthorizedIndexStoreError(f"{label} is missing")
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        return AuthorizedIndexStoreError(f"{label} crosses a link or non-directory ancestor")
    return AuthorizedIndexStoreError(f"cannot open {label}: {exc.strerror or exc}")


def _open_absolute_directory(path: Path, *, label: str) -> int:
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise AuthorizedIndexStoreError(f"{label} must be an absolute canonical POSIX path")
    try:
        descriptor = os.open("/", _directory_flags())
    except OSError as exc:
        raise _open_error(label, exc) from exc
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise _open_error(label, exc) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise AuthorizedIndexStoreError(f"{label} must be a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_file(root: Path, relative_path: str, *, label: str) -> int:
    parts = PurePosixPath(relative_path).parts
    root_descriptor = _open_absolute_directory(root, label=f"{label} root")
    descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise _open_error(label, exc) from exc
            os.close(descriptor)
            descriptor = child
        try:
            file_descriptor = os.open(parts[-1], _file_flags(), dir_fd=descriptor)
        except OSError as exc:
            raise _open_error(label, exc) from exc
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(file_descriptor)
            raise AuthorizedIndexStoreError(f"{label} must be a non-linked regular file")
        return file_descriptor
    finally:
        os.close(descriptor)
        os.close(root_descriptor)


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


def _hash_fd(descriptor: int, *, label: str) -> tuple[int, str]:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        chunk = os.read(descriptor, _READ_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        byte_count += len(chunk)
    after = os.fstat(descriptor)
    if _stat_signature(before) != _stat_signature(after) or byte_count != before.st_size:
        raise AuthorizedIndexStoreError(f"{label} changed during digest verification")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return byte_count, digest.hexdigest()


def _verify_relative_digest(
    root: Path,
    relative_path: str,
    *,
    expected_byte_count: int,
    expected_sha256: str,
    label: str,
) -> None:
    descriptor = _open_relative_file(root, relative_path, label=label)
    try:
        observed = _hash_fd(descriptor, label=label)
    finally:
        os.close(descriptor)
    if observed != (expected_byte_count, expected_sha256):
        raise AuthorizedIndexStoreError(f"{label} differs from its immutable pin")


@contextmanager
def _secure_npy(
    root: Path,
    relative_path: str,
    *,
    expected_byte_count: int,
    expected_sha256: str,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    label: str,
) -> Iterator[np.memmap]:
    descriptor = _open_relative_file(root, relative_path, label=label)
    opened_signature = _stat_signature(os.fstat(descriptor))
    array: np.memmap | None = None
    stream = None
    try:
        if _hash_fd(descriptor, label=label) != (expected_byte_count, expected_sha256):
            raise AuthorizedIndexStoreError(f"{label} differs from its immutable pin")
        stream = os.fdopen(os.dup(descriptor), "rb")
        version = np.lib.format.read_magic(stream)
        if version != (2, 0):
            raise AuthorizedIndexStoreError(f"{label} must use NPY format 2.0")
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        if (
            tuple(shape) != expected_shape
            or fortran_order
            or dtype != np.dtype(expected_dtype)
            or dtype.hasobject
        ):
            raise AuthorizedIndexStoreError(f"{label} header differs from its descriptor")
        array = np.memmap(
            stream,
            dtype=dtype,
            mode="r",
            offset=stream.tell(),
            shape=shape,
            order="C",
        )
        array.setflags(write=False)
        yield array
        if _hash_fd(descriptor, label=label) != (expected_byte_count, expected_sha256):
            raise AuthorizedIndexStoreError(f"{label} changed while mapped")
        rebound = _open_relative_file(root, relative_path, label=label)
        try:
            if (
                _stat_signature(os.fstat(descriptor)) != opened_signature
                or _stat_signature(os.fstat(rebound)) != opened_signature
                or _hash_fd(rebound, label=label) != (expected_byte_count, expected_sha256)
            ):
                raise AuthorizedIndexStoreError(f"{label} was mutated or substituted while mapped")
        finally:
            os.close(rebound)
    finally:
        if array is not None:
            del array
        if stream is not None:
            stream.close()
        os.close(descriptor)


@contextmanager
def open_verified_document_matrices(
    embedding_store_root: str | Path,
    *,
    index_receipt: AuthorizedIndexStoreReceipt,
    expected_embedding_receipt_sha256: str,
) -> Iterator[VerifiedDocumentMatrices]:
    """Hold both document epochs through no-follow descriptors for one run.

    The embedding store is freshly verified, both descriptors must equal the
    corresponding bindings in ``index_receipt``, and each path is checked again
    when the context exits. Replacing a matrix after admission therefore
    invalidates the attempt before result publication.
    """

    root = Path(embedding_store_root)
    if (
        not root.is_absolute()
        or root.anchor != "/"
        or any(part in {".", ".."} for part in root.parts)
    ):
        raise AuthorizedIndexStoreError("embedding_store_root must be an absolute canonical path")
    if not isinstance(index_receipt, AuthorizedIndexStoreReceipt):
        raise AuthorizedIndexStoreError("index_receipt must be typed")
    _require_sha256(
        "expected_embedding_receipt_sha256",
        expected_embedding_receipt_sha256,
    )
    try:
        embedding = verify_embedding_store(root)
    except Exception as exc:
        raise AuthorizedIndexStoreError(f"embedding store admission failed: {exc}") from exc
    if embedding.receipt_sha256 != expected_embedding_receipt_sha256:
        raise AuthorizedIndexStoreError("embedding receipt differs from its frozen pin")
    if embedding.old_model is None or "old_documents" not in embedding.vectors:
        raise AuthorizedIndexStoreError("embedding store lacks the old document epoch")
    old_binding = VectorSourceBinding.from_descriptor(
        embedding.vectors["old_documents"],
        role="active-old-stale",
    )
    truth_binding = VectorSourceBinding.from_descriptor(
        embedding.vectors["current_documents"],
        role="current-exact-truth",
    )
    if (
        old_binding != index_receipt.old_active_vector
        or truth_binding != index_receipt.current_truth_vector
        or embedding.receipt_sha256 != index_receipt.embedding_receipt_sha256
    ):
        raise AuthorizedIndexStoreError(
            "document matrices differ from the authorized-index receipt"
        )
    if np.dtype(old_binding.dtype) != np.dtype(np.float32):
        raise AuthorizedIndexStoreError(
            "sealed retrieval requires native float32 document matrices"
        )

    with _secure_npy(
        root,
        old_binding.relative_path,
        expected_byte_count=old_binding.byte_count,
        expected_sha256=old_binding.file_sha256,
        expected_dtype=old_binding.dtype,
        expected_shape=old_binding.shape,
        label="old active document vectors",
    ) as old_active:
        with _secure_npy(
            root,
            truth_binding.relative_path,
            expected_byte_count=truth_binding.byte_count,
            expected_sha256=truth_binding.file_sha256,
            expected_dtype=truth_binding.dtype,
            expected_shape=truth_binding.shape,
            label="current truth document vectors",
        ) as current_truth:
            yield VerifiedDocumentMatrices(
                old_active=old_active,
                current_truth=current_truth,
                receipt_sha256=embedding.receipt_sha256,
            )


def _mask_count_and_digest(
    root: Path,
    mask: CompiledMaskDescriptor,
    *,
    document_count: int,
) -> None:
    descriptor = _open_relative_file(root, mask.path, label=f"compiled mask {mask.mask_id}")
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        byte_count = 0
        count = 0
        last_byte = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            count += sum(_BIT_COUNTS[value] for value in chunk)
            last_byte = chunk[-1]
        after = os.fstat(descriptor)
        if _stat_signature(before) != _stat_signature(after):
            raise AuthorizedIndexStoreError("compiled mask changed during verification")
    finally:
        os.close(descriptor)
    remainder = document_count % 8
    if remainder and last_byte & (~((1 << remainder) - 1) & 0xFF):
        raise AuthorizedIndexStoreError("compiled mask has nonzero trailing bits")
    if (
        byte_count != mask.byte_count
        or digest.hexdigest() != mask.sha256
        or count != mask.authorized_count
    ):
        raise AuthorizedIndexStoreError("compiled mask differs from its catalog descriptor")


def _mask_batch(descriptor: int, start: int, stop: int) -> np.ndarray:
    byte_start = start // 8
    byte_stop = (stop + 7) // 8
    try:
        encoded = os.pread(descriptor, byte_stop - byte_start, byte_start)
    except OSError as exc:
        raise AuthorizedIndexStoreError(f"cannot read compiled mask batch: {exc}") from exc
    if len(encoded) != byte_stop - byte_start:
        raise AuthorizedIndexStoreError("compiled mask ended during batch read")
    unpacked = np.unpackbits(np.frombuffer(encoded, dtype=np.uint8), bitorder="little")
    offset = start - byte_start * 8
    return unpacked[offset : offset + (stop - start)].astype(bool, copy=False)


def _policy_tree_entries(receipt: PolicyInterventionReceipt) -> set[str]:
    entries = {POLICY_RECEIPT_FILENAME}
    for artifact in receipt.artifacts:
        path = PurePosixPath(artifact.path)
        entries.add(str(path))
        for position in range(1, len(path.parts)):
            entries.add(str(PurePosixPath(*path.parts[:position])))
    return entries


def _admit_sources(
    embedding_root: Path,
    policy_root: Path,
    *,
    expected_embedding_receipt_sha256: str,
    expected_policy_receipt_sha256: str,
) -> _SourceAdmission:
    _require_sha256("expected_embedding_receipt_sha256", expected_embedding_receipt_sha256)
    _require_sha256("expected_policy_receipt_sha256", expected_policy_receipt_sha256)
    try:
        embedding = verify_embedding_store(embedding_root)
    except Exception as exc:
        raise AuthorizedIndexStoreError(f"embedding store admission failed: {exc}") from exc
    if embedding.receipt_sha256 != expected_embedding_receipt_sha256:
        raise AuthorizedIndexStoreError("embedding receipt differs from the caller pin")
    if embedding.old_model is None or "old_documents" not in embedding.vectors:
        raise AuthorizedIndexStoreError("embedding store has no pinned old document matrix")
    old_descriptor = embedding.vectors["old_documents"]
    current_descriptor = embedding.vectors["current_documents"]
    old_vector = VectorSourceBinding.from_descriptor(old_descriptor, role="active-old-stale")
    current_vector = VectorSourceBinding.from_descriptor(
        current_descriptor, role="current-exact-truth"
    )
    for binding in (old_vector, current_vector):
        _verify_relative_digest(
            embedding_root,
            binding.relative_path,
            expected_byte_count=binding.byte_count,
            expected_sha256=binding.file_sha256,
            label=f"{binding.role} document matrix",
        )

    try:
        policy = load_policy_intervention_receipt(policy_root / POLICY_RECEIPT_FILENAME)
        catalog = load_compiled_policy_catalog(policy_root / CATALOG_FILENAME)
        tree = digest_directory_tree(policy_root)
    except Exception as exc:
        raise AuthorizedIndexStoreError(f"policy intervention admission failed: {exc}") from exc
    if policy.artifact_sha256 != expected_policy_receipt_sha256:
        raise AuthorizedIndexStoreError("policy receipt differs from the caller pin")
    expected_entries = _policy_tree_entries(policy)
    if set(tree.entries) != expected_entries:
        raise AuthorizedIndexStoreError("policy intervention tree membership differs from receipt")
    for artifact in policy.artifacts:
        _verify_relative_digest(
            policy_root,
            artifact.path,
            expected_byte_count=artifact.byte_count,
            expected_sha256=artifact.sha256,
            label=f"policy artifact {artifact.path}",
        )
    catalog_row = next((row for row in policy.artifacts if row.path == CATALOG_FILENAME), None)
    if catalog_row is None or catalog_row.sha256 != catalog.artifact_sha256:
        raise AuthorizedIndexStoreError("policy receipt does not bind its mask catalog")
    document_row_order_sha256 = embedding.row_orders["documents"].row_order_sha256
    if (
        policy.document_count != embedding.document_count
        or catalog.document_count != embedding.document_count
        or policy.document_universe_sha256 != document_row_order_sha256
        or catalog.document_universe_sha256 != document_row_order_sha256
        or policy.policy_bundle_revision != catalog.policy_revision
    ):
        raise AuthorizedIndexStoreError(
            "embedding row order and compiled policy universe are not the same frozen universe"
        )
    masks = tuple(catalog.masks)
    for mask in masks:
        row = next((item for item in policy.artifacts if item.path == mask.path), None)
        if row is None or row.sha256 != mask.sha256 or row.byte_count != mask.byte_count:
            raise AuthorizedIndexStoreError("policy receipt does not bind every compiled mask")
        _mask_count_and_digest(policy_root, mask, document_count=embedding.document_count)
    return _SourceAdmission(
        embedding=embedding,
        policy=policy,
        catalog_sha256=catalog.artifact_sha256,
        masks=masks,
        old_vector=old_vector,
        current_vector=current_vector,
    )


def _check_backend(backend: AuthorizedIndexBackend, config: AuthorizedIndexConfig) -> None:
    observed = (
        getattr(backend, "backend_id", None),
        getattr(backend, "package_version", None),
        getattr(backend, "build_sha256", None),
    )
    expected = (config.backend_id, config.backend_version, config.backend_build_sha256)
    if observed != expected:
        raise AuthorizedIndexStoreError("runtime backend differs from the frozen backend binding")


def _write_exclusive(path: Path, encoded: bytes) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        raise AuthorizedIndexStoreError(f"cannot create {path.name!r} exclusively: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = _open_absolute_directory(path, label="index output directory")
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_publish(work: Path, output: Path) -> None:
    if os.path.lexists(output):
        raise AuthorizedIndexStoreError("authorized index target already exists")
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(work)
    destination = os.fsencode(output)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise AuthorizedIndexStoreError("exclusive directory rename is unavailable on macOS")
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
            raise AuthorizedIndexStoreError("exclusive directory rename is unavailable on Linux")
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
        raise AuthorizedIndexStoreError(
            f"exclusive directory rename is unsupported on platform {sys.platform!r}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise AuthorizedIndexStoreError("authorized index target already exists")
        raise AuthorizedIndexStoreError(
            f"cannot publish authorized index store: {os.strerror(error_number)}"
        )


def _assert_private_parent(path: Path) -> None:
    descriptor = _open_absolute_directory(path, label="authorized index target parent")
    try:
        metadata = os.fstat(descriptor)
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise AuthorizedIndexStoreError("target parent must be owned by the builder identity")
        if stat.S_IMODE(metadata.st_mode) & 0o022:
            raise AuthorizedIndexStoreError("target parent cannot be writable by group or others")
    finally:
        os.close(descriptor)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _build_binding(
    admission: _SourceAdmission,
    mask: CompiledMaskDescriptor,
    config: AuthorizedIndexConfig,
) -> str:
    payload = {
        "backend": {
            "build_sha256": config.backend_build_sha256,
            "id": config.backend_id,
            "version": config.backend_version,
        },
        "builder_identity": config.builder_identity,
        "config_sha256": config.config_sha256,
        "current_truth_vector": admission.current_vector.to_dict(),
        "document_universe_sha256": admission.policy.document_universe_sha256,
        "embedding_receipt_sha256": admission.embedding.receipt_sha256,
        "mask": mask.to_dict(),
        "old_active_vector": admission.old_vector.to_dict(),
        "policy_catalog_sha256": admission.catalog_sha256,
        "policy_receipt_sha256": admission.policy.artifact_sha256,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _create_row_map(path: Path, authorized_count: int) -> np.memmap:
    if os.path.lexists(path):
        raise AuthorizedIndexStoreError("row-map target already exists")
    try:
        array = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.dtype(ROW_MAP_DTYPE),
            shape=(authorized_count,),
            fortran_order=False,
            version=(2, 0),
        )
        os.chmod(path, 0o600)
        return array
    except (OSError, ValueError) as exc:
        raise AuthorizedIndexStoreError(f"cannot allocate row map: {exc}") from exc


def _finalize_backend_file(path: Path, *, label: str) -> tuple[int, str]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise AuthorizedIndexStoreError(f"cannot inspect {label}: {exc}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
    ):
        raise AuthorizedIndexStoreError(f"{label} must be a nonempty non-linked regular file")
    os.chmod(path, 0o600)
    descriptor = os.open(path, _file_flags())
    try:
        os.fsync(descriptor)
        return _hash_fd(descriptor, label=label)
    finally:
        os.close(descriptor)


def _build_one_index(
    *,
    work: Path,
    policy_root: Path,
    old_vectors: np.memmap,
    mask: CompiledMaskDescriptor,
    config: AuthorizedIndexConfig,
    backend: AuthorizedIndexBackend,
    build_binding_sha256: str,
) -> AuthorizedIndexArtifact:
    safe_id = mask.mask_id
    index_relative = f"{INDEX_DIRECTORY}/{safe_id}.hnsw"
    map_relative = f"{ROW_MAP_DIRECTORY}/{safe_id}.npy"
    index_path = work / index_relative
    map_path = work / map_relative
    row_map = _create_row_map(map_path, mask.authorized_count)
    index = backend.create_index(metric=config.metric, dimension=old_vectors.shape[1])
    index.init_index(
        max_elements=mask.authorized_count,
        ef_construction=config.ef_construction,
        M=config.m,
        random_seed=config.random_seed,
        allow_replace_deleted=False,
    )
    index.set_num_threads(INDEX_NUM_THREADS)
    mask_descriptor = _open_relative_file(
        policy_root, mask.path, label=f"compiled mask {mask.mask_id}"
    )
    local_position = 0
    try:
        for start in range(0, old_vectors.shape[0], config.batch_size):
            stop = min(start + config.batch_size, old_vectors.shape[0])
            flags = _mask_batch(mask_descriptor, start, stop)
            offsets = np.flatnonzero(flags)
            if not offsets.size:
                continue
            global_rows = (offsets + start).astype(np.int64, copy=False)
            next_position = local_position + len(global_rows)
            if next_position > mask.authorized_count:
                raise AuthorizedIndexStoreError("compiled mask exceeds its authorized count")
            row_map[local_position:next_position] = global_rows
            vectors = np.asarray(
                old_vectors[global_rows], dtype=np.dtype(INDEX_INPUT_DTYPE), order="C"
            )
            if (
                vectors.shape != (len(global_rows), old_vectors.shape[1])
                or not np.isfinite(vectors).all()
            ):
                raise AuthorizedIndexStoreError("old vector batch is non-finite or malformed")
            labels = np.arange(local_position, next_position, dtype=np.int64)
            index.add_items(vectors, labels, num_threads=INDEX_NUM_THREADS)
            local_position = next_position
    finally:
        os.close(mask_descriptor)
    if local_position != mask.authorized_count:
        raise AuthorizedIndexStoreError("compiled mask count changed during index build")
    row_map.flush()
    del row_map
    map_byte_count, map_sha256 = _finalize_backend_file(map_path, label="authorized row map")
    if os.path.lexists(index_path):
        raise AuthorizedIndexStoreError("backend index target appeared before save")
    index.save_index(str(index_path))
    index_byte_count, index_sha256 = _finalize_backend_file(
        index_path, label="authorized HNSW index"
    )
    return AuthorizedIndexArtifact(
        mask_id=mask.mask_id,
        mask_sha256=mask.sha256,
        authorized_count=mask.authorized_count,
        index_path=index_relative,
        index_sha256=index_sha256,
        index_byte_count=index_byte_count,
        row_map_path=map_relative,
        row_map_sha256=map_sha256,
        row_map_byte_count=map_byte_count,
        row_map_shape=(mask.authorized_count,),
        build_binding_sha256=build_binding_sha256,
    )


def _load_config(root: Path) -> AuthorizedIndexConfig:
    try:
        encoded = read_secure_regular_file(
            root / CONFIG_FILENAME,
            max_bytes=_CONTROL_MAX_BYTES,
            label="authorized index config",
        )
    except ArtifactIntegrityError as exc:
        raise AuthorizedIndexStoreError(f"cannot read authorized index config: {exc}") from exc
    config = AuthorizedIndexConfig.from_dict(
        _decode_object(encoded, label="authorized index config")
    )
    if encoded != config.canonical_bytes() + b"\n":
        raise AuthorizedIndexStoreError("authorized index config is not canonical")
    return config


def load_authorized_index_store_receipt(root: str | Path) -> AuthorizedIndexStoreReceipt:
    """Load one canonical store receipt without opening an index."""

    path = Path(root) / RECEIPT_FILENAME
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=_CONTROL_MAX_BYTES,
            label="authorized index receipt",
        )
    except ArtifactIntegrityError as exc:
        raise AuthorizedIndexStoreError(f"cannot read authorized index receipt: {exc}") from exc
    receipt = AuthorizedIndexStoreReceipt.from_dict(
        _decode_object(encoded, label="authorized index receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise AuthorizedIndexStoreError("authorized index receipt is not canonical")
    return receipt


@contextmanager
def _secure_backend_path(
    root: Path,
    relative_path: str,
    *,
    expected_byte_count: int,
    expected_sha256: str,
    label: str,
) -> Iterator[str]:
    descriptor = _open_relative_file(root, relative_path, label=label)
    try:
        if _hash_fd(descriptor, label=label) != (expected_byte_count, expected_sha256):
            raise AuthorizedIndexStoreError(f"{label} differs from its immutable pin")
        fd_path = f"/dev/fd/{descriptor}"
        if not os.path.exists("/dev/fd"):
            raise AuthorizedIndexStoreError("secure descriptor paths are unavailable")
        yield fd_path
        if _hash_fd(descriptor, label=label) != (expected_byte_count, expected_sha256):
            raise AuthorizedIndexStoreError(f"{label} changed during backend load")
    finally:
        os.close(descriptor)


def _expected_store_entries(indexes: Sequence[AuthorizedIndexArtifact]) -> set[str]:
    return {
        CONFIG_FILENAME,
        RECEIPT_FILENAME,
        INDEX_DIRECTORY,
        ROW_MAP_DIRECTORY,
        *(item.index_path for item in indexes),
        *(item.row_map_path for item in indexes),
    }


def _verify_row_map_against_mask(
    row_map: np.memmap,
    *,
    policy_root: Path,
    mask: CompiledMaskDescriptor,
    document_count: int,
    batch_size: int,
) -> int:
    mask_descriptor = _open_relative_file(
        policy_root, mask.path, label=f"compiled mask {mask.mask_id}"
    )
    local_position = 0
    first_global_row = -1
    previous = -1
    try:
        for start in range(0, document_count, batch_size):
            stop = min(start + batch_size, document_count)
            expected = np.flatnonzero(_mask_batch(mask_descriptor, start, stop)) + start
            observed = np.asarray(
                row_map[local_position : local_position + len(expected)], dtype=np.int64
            )
            if observed.shape != expected.shape or not np.array_equal(observed, expected):
                raise AuthorizedIndexStoreError("row map differs from its compiled mask")
            if len(observed):
                if first_global_row < 0:
                    first_global_row = int(observed[0])
                if int(observed[0]) <= previous:
                    raise AuthorizedIndexStoreError("row map is not strictly increasing")
                previous = int(observed[-1])
            local_position += len(expected)
    finally:
        os.close(mask_descriptor)
    if local_position != mask.authorized_count or first_global_row < 0:
        raise AuthorizedIndexStoreError("row map coverage differs from the mask")
    return first_global_row


def verify_authorized_index_store(
    root: str | Path,
    *,
    embedding_store_root: str | Path,
    policy_intervention_root: str | Path,
    expected_embedding_receipt_sha256: str,
    expected_policy_receipt_sha256: str,
    backend: AuthorizedIndexBackend,
    expected_store_receipt_sha256: str | None = None,
) -> AuthorizedIndexStoreVerification:
    """Verify source pins, exact payloads, row maps, and one query per loaded index."""

    store = Path(root)
    embedding_root = Path(embedding_store_root)
    policy_root = Path(policy_intervention_root)
    admission = _admit_sources(
        embedding_root,
        policy_root,
        expected_embedding_receipt_sha256=expected_embedding_receipt_sha256,
        expected_policy_receipt_sha256=expected_policy_receipt_sha256,
    )
    config = _load_config(store)
    _check_backend(backend, config)
    receipt = load_authorized_index_store_receipt(store)
    if expected_store_receipt_sha256 is not None:
        _require_sha256("expected_store_receipt_sha256", expected_store_receipt_sha256)
        if receipt.artifact_sha256 != expected_store_receipt_sha256:
            raise AuthorizedIndexStoreError("authorized index receipt differs from its caller pin")
    expected_source_values = (
        admission.embedding.receipt_sha256,
        admission.policy.artifact_sha256,
        admission.catalog_sha256,
        admission.policy.execution_artifact_sha256,
        admission.policy.policy_bundle_revision,
        admission.embedding.document_count,
        admission.policy.document_universe_sha256,
        admission.embedding.row_orders["documents"].row_order_sha256,
        admission.old_vector,
        admission.current_vector,
        config.config_sha256,
        config.backend_version,
        config.backend_build_sha256,
    )
    observed_source_values = (
        receipt.embedding_receipt_sha256,
        receipt.policy_receipt_sha256,
        receipt.policy_catalog_sha256,
        receipt.policy_execution_artifact_sha256,
        receipt.policy_revision,
        receipt.document_count,
        receipt.document_universe_sha256,
        receipt.document_row_order_sha256,
        receipt.old_active_vector,
        receipt.current_truth_vector,
        receipt.config_sha256,
        receipt.backend_version,
        receipt.backend_build_sha256,
    )
    if observed_source_values != expected_source_values:
        raise AuthorizedIndexStoreError("authorized index receipt source binding differs")
    masks = {mask.mask_id: mask for mask in admission.masks}
    if set(masks) != {item.mask_id for item in receipt.indexes}:
        raise AuthorizedIndexStoreError("authorized indexes do not cover the exact mask catalog")
    expected_entries = _expected_store_entries(receipt.indexes)
    try:
        full_tree = digest_directory_tree(store)
        payload_entries = sorted(expected_entries - {RECEIPT_FILENAME})
        payload_tree = digest_directory_tree(store, included_entries=payload_entries)
    except ArtifactIntegrityError as exc:
        raise AuthorizedIndexStoreError(f"cannot verify authorized index tree: {exc}") from exc
    if set(full_tree.entries) != expected_entries:
        raise AuthorizedIndexStoreError("authorized index store membership differs from receipt")
    if payload_tree.sha256 != receipt.payload_tree_sha256:
        raise AuthorizedIndexStoreError("authorized index payload tree digest differs")

    with _secure_npy(
        embedding_root,
        admission.old_vector.relative_path,
        expected_byte_count=admission.old_vector.byte_count,
        expected_sha256=admission.old_vector.file_sha256,
        expected_dtype=admission.old_vector.dtype,
        expected_shape=admission.old_vector.shape,
        label="old active document vectors",
    ) as old_vectors:
        for item in receipt.indexes:
            mask = masks[item.mask_id]
            if (
                item.mask_sha256 != mask.sha256
                or item.authorized_count != mask.authorized_count
                or item.build_binding_sha256 != _build_binding(admission, mask, config)
            ):
                raise AuthorizedIndexStoreError("authorized index build binding differs")
            _verify_relative_digest(
                store,
                item.index_path,
                expected_byte_count=item.index_byte_count,
                expected_sha256=item.index_sha256,
                label=f"authorized index {item.mask_id}",
            )
            with _secure_npy(
                store,
                item.row_map_path,
                expected_byte_count=item.row_map_byte_count,
                expected_sha256=item.row_map_sha256,
                expected_dtype=item.row_map_dtype,
                expected_shape=item.row_map_shape,
                label=f"authorized row map {item.mask_id}",
            ) as row_map:
                first_global_row = _verify_row_map_against_mask(
                    row_map,
                    policy_root=policy_root,
                    mask=mask,
                    document_count=receipt.document_count,
                    batch_size=config.batch_size,
                )
            index = backend.create_index(metric=config.metric, dimension=old_vectors.shape[1])
            with _secure_backend_path(
                store,
                item.index_path,
                expected_byte_count=item.index_byte_count,
                expected_sha256=item.index_sha256,
                label=f"authorized index {item.mask_id}",
            ) as backend_path:
                index.load_index(backend_path, max_elements=item.authorized_count)
            index.set_num_threads(INDEX_NUM_THREADS)
            index.set_ef(min(config.verification_ef, item.authorized_count))
            query = np.asarray(
                old_vectors[first_global_row : first_global_row + 1],
                dtype=np.dtype(INDEX_INPUT_DTYPE),
                order="C",
            )
            labels, distances = index.knn_query(query, k=1, num_threads=INDEX_NUM_THREADS)
            labels = np.asarray(labels)
            distances = np.asarray(distances)
            if (
                labels.shape != (1, 1)
                or distances.shape != (1, 1)
                or not np.isfinite(distances).all()
                or type(labels[0, 0].item()) is not int
                or not 0 <= int(labels[0, 0]) < item.authorized_count
            ):
                raise AuthorizedIndexStoreError(
                    "loaded authorized index failed its query smoke test"
                )
    return AuthorizedIndexStoreVerification(
        root=store,
        receipt_sha256=receipt.artifact_sha256,
        payload_tree_sha256=receipt.payload_tree_sha256,
        mask_ids=tuple(item.mask_id for item in receipt.indexes),
    )


def build_authorized_index_store(
    embedding_store_root: str | Path,
    policy_intervention_root: str | Path,
    output_root: str | Path,
    *,
    expected_embedding_receipt_sha256: str,
    expected_policy_receipt_sha256: str,
    config: AuthorizedIndexConfig,
    backend: AuthorizedIndexBackend,
) -> AuthorizedIndexStoreVerification:
    """Build, self-verify, and exclusively publish one fail-clean index package."""

    if not isinstance(config, AuthorizedIndexConfig):
        raise AuthorizedIndexStoreError("config must be AuthorizedIndexConfig")
    _check_backend(backend, config)
    embedding_root = Path(embedding_store_root)
    policy_root = Path(policy_intervention_root)
    output = Path(output_root)
    for label, source in (
        ("embedding_store_root", embedding_root),
        ("policy_intervention_root", policy_root),
    ):
        if (
            not source.is_absolute()
            or source.anchor != "/"
            or any(part in {".", ".."} for part in source.parts)
        ):
            raise AuthorizedIndexStoreError(f"{label} must be an absolute canonical path")
    if (
        not output.is_absolute()
        or output.name in {"", ".", ".."}
        or any(part in {".", ".."} for part in output.parts)
    ):
        raise AuthorizedIndexStoreError("output_root must be an absolute canonical path")
    for source in (embedding_root, policy_root):
        if _paths_overlap(output, source):
            raise AuthorizedIndexStoreError("authorized index output cannot overlap a source root")
    _assert_private_parent(output.parent)
    if os.path.lexists(output):
        raise AuthorizedIndexStoreError("authorized index target already exists")
    admission = _admit_sources(
        embedding_root,
        policy_root,
        expected_embedding_receipt_sha256=expected_embedding_receipt_sha256,
        expected_policy_receipt_sha256=expected_policy_receipt_sha256,
    )
    lock = output.parent / f".{output.name}.authorized-index.lock"
    work = output.parent / f".{output.name}.staging-{secrets.token_hex(12)}"
    lock_descriptor: int | None = None
    finalized = False
    try:
        try:
            lock_descriptor = os.open(
                lock,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except FileExistsError as exc:
            raise AuthorizedIndexStoreError(
                "another authorized index builder holds the lock"
            ) from exc
        work.mkdir(mode=0o700)
        (work / INDEX_DIRECTORY).mkdir(mode=0o700)
        (work / ROW_MAP_DIRECTORY).mkdir(mode=0o700)
        _write_exclusive(work / CONFIG_FILENAME, config.canonical_bytes() + b"\n")
        artifacts: list[AuthorizedIndexArtifact] = []
        with _secure_npy(
            embedding_root,
            admission.old_vector.relative_path,
            expected_byte_count=admission.old_vector.byte_count,
            expected_sha256=admission.old_vector.file_sha256,
            expected_dtype=admission.old_vector.dtype,
            expected_shape=admission.old_vector.shape,
            label="old active document vectors",
        ) as old_vectors:
            for mask in admission.masks:
                artifacts.append(
                    _build_one_index(
                        work=work,
                        policy_root=policy_root,
                        old_vectors=old_vectors,
                        mask=mask,
                        config=config,
                        backend=backend,
                        build_binding_sha256=_build_binding(admission, mask, config),
                    )
                )
        _fsync_directory(work / INDEX_DIRECTORY)
        _fsync_directory(work / ROW_MAP_DIRECTORY)
        payload_tree = digest_directory_tree(work)
        receipt = AuthorizedIndexStoreReceipt(
            config_sha256=config.config_sha256,
            embedding_receipt_sha256=admission.embedding.receipt_sha256,
            policy_receipt_sha256=admission.policy.artifact_sha256,
            policy_catalog_sha256=admission.catalog_sha256,
            policy_execution_artifact_sha256=admission.policy.execution_artifact_sha256,
            policy_revision=admission.policy.policy_bundle_revision,
            document_count=admission.embedding.document_count,
            document_universe_sha256=admission.policy.document_universe_sha256,
            document_row_order_sha256=admission.embedding.row_orders["documents"].row_order_sha256,
            old_active_vector=admission.old_vector,
            current_truth_vector=admission.current_vector,
            indexes=tuple(artifacts),
            payload_tree_sha256=payload_tree.sha256,
            backend_version=config.backend_version,
            backend_build_sha256=config.backend_build_sha256,
        )
        _write_exclusive(work / RECEIPT_FILENAME, receipt.canonical_file_bytes())
        _fsync_directory(work)
        verification = verify_authorized_index_store(
            work,
            embedding_store_root=embedding_root,
            policy_intervention_root=policy_root,
            expected_embedding_receipt_sha256=expected_embedding_receipt_sha256,
            expected_policy_receipt_sha256=expected_policy_receipt_sha256,
            backend=backend,
            expected_store_receipt_sha256=receipt.artifact_sha256,
        )
        _exclusive_publish(work, output)
        finalized = True
        _fsync_directory(output.parent)
        return AuthorizedIndexStoreVerification(
            root=output,
            receipt_sha256=verification.receipt_sha256,
            payload_tree_sha256=verification.payload_tree_sha256,
            mask_ids=verification.mask_ids,
        )
    finally:
        if not finalized and work.exists():
            shutil.rmtree(work)
        if lock_descriptor is not None:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                if finalized:
                    raise AuthorizedIndexStoreError(
                        "store published but builder lock cleanup failed"
                    ) from exc
