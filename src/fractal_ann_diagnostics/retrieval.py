"""Exact and HNSW retrieval paths with explicit authorization boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import perf_counter_ns
from typing import Literal, Protocol

import numpy as np

DistanceMetric = Literal["euclidean", "cosine"]
QUERY_CONTROL_FEATURE_SCHEMA = "fractal-query-control-features-v1"


def _lower_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_mask(mask: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(mask)
    if value.dtype != np.dtype(bool) or value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be one non-empty Boolean decision vector")
    return value


def packed_policy_mask_sha256(mask: np.ndarray) -> str:
    """Hash one decision vector using the compiled-policy wire encoding."""

    value = _canonical_mask(mask, name="policy mask")
    encoded = np.packbits(value, bitorder="little").tobytes(order="C")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def policy_mask_churn(baseline: np.ndarray, current: np.ndarray) -> float:
    """Return the registered document-level Hamming fraction for one subject.

    The two vectors are complete authorization decisions over the same ordered
    document universe. An allow-rate scalar, changed-count scalar, or aggregate
    percentage is not an admissible substitute for either vector.
    """

    before = _canonical_mask(baseline, name="baseline policy mask")
    after = _canonical_mask(current, name="current policy mask")
    if before.shape != after.shape:
        raise ValueError("baseline and current policy masks cover different universes")
    return float(np.count_nonzero(before != after) / before.size)


def dual_epoch_query_drift(active: np.ndarray, current: np.ndarray) -> float:
    """Derive per-query drift as one minus dual-epoch cosine similarity."""

    before = np.asarray(active, dtype=np.float64)
    after = np.asarray(current, dtype=np.float64)
    if (
        before.ndim != 1
        or before.size == 0
        or after.shape != before.shape
        or not np.all(np.isfinite(before))
        or not np.all(np.isfinite(after))
    ):
        raise ValueError("active and current query epochs must be finite matched rows")
    denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
    if not np.isfinite(denominator) or denominator == 0.0:
        raise ValueError("dual-epoch query rows must have finite non-zero norms")
    cosine = float(np.clip(np.dot(before, after) / denominator, -1.0, 1.0))
    drift = 1.0 - cosine
    if not np.isfinite(drift):
        raise ValueError("query drift derivation produced a non-finite value")
    return drift


@dataclass(frozen=True, init=False)
class PolicyTransitionEvidence:
    """Source-bound synthetic policy mutation for one frozen environment.

    Instances can be created only from complete baseline and current masks via
    :meth:`derive`. The class deliberately has no constructor that accepts a
    churn scalar.
    """

    environment_sha256: str
    baseline_policy_revision: str
    current_policy_revision: str
    baseline_mask_sha256: str
    current_mask_sha256: str
    baseline_authorized_count: int
    current_authorized_count: int
    document_count: int
    policy_churn: float
    schema_version: str = QUERY_CONTROL_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        _lower_sha256("environment_sha256", self.environment_sha256)
        _lower_sha256("baseline_mask_sha256", self.baseline_mask_sha256)
        _lower_sha256("current_mask_sha256", self.current_mask_sha256)
        for name in ("baseline_policy_revision", "current_policy_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise ValueError(f"{name} must be an immutable sha256 revision")
            _lower_sha256(name, value.removeprefix("sha256:"))
        if self.baseline_policy_revision == self.current_policy_revision:
            raise ValueError("baseline and current policy revisions must differ")
        if type(self.document_count) is not int or self.document_count <= 0:
            raise ValueError("document_count must be a positive integer")
        for name in ("baseline_authorized_count", "current_authorized_count"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < self.document_count:
                raise ValueError(f"{name} must be strictly within the document universe")
        if not np.isfinite(self.policy_churn) or not 0.0 < self.policy_churn <= 1.0:
            raise ValueError("policy_churn must be finite and non-zero in (0, 1]")
        if self.schema_version != QUERY_CONTROL_FEATURE_SCHEMA:
            raise ValueError("query control feature schema differs")

    @classmethod
    def derive(
        cls,
        *,
        environment_sha256: str,
        baseline_policy_revision: str,
        current_policy_revision: str,
        baseline_mask: np.ndarray,
        current_mask: np.ndarray,
        expected_baseline_mask_sha256: str,
        expected_current_mask_sha256: str,
        expected_baseline_authorized_count: int,
        expected_current_authorized_count: int,
    ) -> PolicyTransitionEvidence:
        """Recompute all numeric values from complete pinned decision vectors."""

        before = _canonical_mask(baseline_mask, name="baseline policy mask")
        after = _canonical_mask(current_mask, name="current policy mask")
        if before.shape != after.shape:
            raise ValueError("baseline and current policy masks cover different universes")
        before_sha256 = packed_policy_mask_sha256(before)
        after_sha256 = packed_policy_mask_sha256(after)
        if before_sha256 != _lower_sha256(
            "expected_baseline_mask_sha256", expected_baseline_mask_sha256
        ):
            raise ValueError("baseline policy mask differs from its frozen digest")
        if after_sha256 != _lower_sha256(
            "expected_current_mask_sha256", expected_current_mask_sha256
        ):
            raise ValueError("current policy mask differs from its frozen digest")
        baseline_count = int(np.count_nonzero(before))
        current_count = int(np.count_nonzero(after))
        if baseline_count != expected_baseline_authorized_count:
            raise ValueError("baseline authorized count differs from its frozen row")
        if current_count != expected_current_authorized_count:
            raise ValueError("current authorized count differs from its frozen row")
        instance = object.__new__(cls)
        values: dict[str, object] = {
            "environment_sha256": environment_sha256,
            "baseline_policy_revision": baseline_policy_revision,
            "current_policy_revision": current_policy_revision,
            "baseline_mask_sha256": before_sha256,
            "current_mask_sha256": after_sha256,
            "baseline_authorized_count": baseline_count,
            "current_authorized_count": current_count,
            "document_count": before.size,
            "policy_churn": policy_mask_churn(before, after),
            "schema_version": QUERY_CONTROL_FEATURE_SCHEMA,
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def verify_current_mask(
        self,
        mask: np.ndarray,
        *,
        policy_revision: str,
        environment_sha256: str,
    ) -> None:
        """Bind the live PDP decision to the frozen current mutation endpoint."""

        if environment_sha256 != self.environment_sha256:
            raise ValueError("policy transition belongs to another environment")
        if policy_revision != self.current_policy_revision:
            raise ValueError("live policy revision differs from the transition endpoint")
        current = _canonical_mask(mask, name="live current policy mask")
        if current.size != self.document_count:
            raise ValueError("live policy mask covers another document universe")
        if packed_policy_mask_sha256(current) != self.current_mask_sha256:
            raise ValueError("live policy mask differs from the transition endpoint")
        if int(np.count_nonzero(current)) != self.current_authorized_count:
            raise ValueError("live authorized count differs from the transition endpoint")


@dataclass(frozen=True)
class SearchWork:
    """Backend work that was observed, kept separate from configured effort.

    ``hnswlib`` does not expose visited-node or distance-evaluation counters. Those
    fields therefore remain ``None`` for HNSW instead of treating ``efSearch`` as
    measured work. Exact search can report both counters without approximation.
    """

    returned_candidates: int
    visited_candidates: int | None = None
    distance_evaluations: int | None = None
    configured_ef_search: int | None = None

    def __post_init__(self) -> None:
        for name in ("returned_candidates", "visited_candidates", "distance_evaluations"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.visited_candidates is not None
            and self.visited_candidates < self.returned_candidates
        ):
            raise ValueError("visited_candidates cannot be smaller than returned_candidates")
        if (
            self.distance_evaluations is not None
            and self.distance_evaluations < self.returned_candidates
        ):
            raise ValueError("distance_evaluations cannot be smaller than returned_candidates")
        if self.configured_ef_search is not None and self.configured_ef_search <= 0:
            raise ValueError("configured_ef_search must be positive")


@dataclass(frozen=True)
class SearchResult:
    """One retrieval action plus its policy-boundary accounting.

    ``candidates_examined`` is retained as the v0.2 compatibility proxy. New
    analysis must use ``work`` and treat an unavailable backend counter as
    missing, not as ``efSearch``.
    """

    ids: np.ndarray
    distances: np.ndarray
    strategy: str
    requested_k: int
    candidates_examined: int
    unauthorized_candidates: int
    unauthorized_context: int
    latency_ms: float
    work: SearchWork | None = None

    def __post_init__(self) -> None:
        ids = np.array(self.ids, dtype=np.int64, copy=True)
        distances = np.array(self.distances, dtype=np.float32, copy=True)
        if ids.ndim != 1 or distances.ndim != 1 or ids.shape != distances.shape:
            raise ValueError("search ids and distances must be equal-length vectors")
        if self.requested_k <= 0:
            raise ValueError("requested_k must be positive")
        if len(ids) > self.requested_k:
            raise ValueError("search result cannot contain more than requested_k items")
        if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("search ids must be unique non-negative integers")
        if not np.all(np.isfinite(distances)) or np.any(distances < 0):
            raise ValueError("search distances must be finite and non-negative")
        for name in (
            "candidates_examined",
            "unauthorized_candidates",
            "unauthorized_context",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.unauthorized_context > len(ids):
            raise ValueError("unauthorized_context cannot exceed returned items")
        if not np.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if self.work is not None and self.work.returned_candidates < len(ids):
            raise ValueError("work.returned_candidates cannot be smaller than returned items")
        ids.setflags(write=False)
        distances.setflags(write=False)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "distances", distances)

    @property
    def shortfall(self) -> int:
        return max(0, self.requested_k - len(self.ids))


@dataclass(frozen=True)
class ProbeTelemetry:
    """Bounded, authorized neighbor evidence available to online geometry.

    The telemetry contains only IDs and distances returned by one authorized
    search. It deliberately has no vector matrix or query handle, so downstream
    feature code cannot expand the candidate universe.
    """

    ids: np.ndarray
    distances: np.ndarray
    metric: DistanceMetric
    authorized_count: int
    corpus_count: int
    max_neighbors: int
    search_latency_ms: float
    work: SearchWork

    def __post_init__(self) -> None:
        ids = np.array(self.ids, dtype=np.int64, copy=True)
        distances = np.array(self.distances, dtype=np.float32, copy=True)
        if ids.ndim != 1 or distances.ndim != 1 or ids.shape != distances.shape:
            raise ValueError("probe ids and distances must be equal-length vectors")
        if self.max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")
        if len(ids) > self.max_neighbors:
            raise ValueError("probe result exceeds max_neighbors")
        if self.authorized_count <= 0 or self.authorized_count > self.corpus_count:
            raise ValueError("authorized_count must be within the corpus")
        if np.any(ids < 0) or np.any(ids >= self.corpus_count):
            raise ValueError("probe contains an out-of-range document id")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("probe document ids must be unique")
        if not np.all(np.isfinite(distances)) or np.any(distances < 0):
            raise ValueError("probe distances must be finite and non-negative")
        if not np.isfinite(self.search_latency_ms) or self.search_latency_ms < 0:
            raise ValueError("search_latency_ms must be finite and non-negative")
        if self.metric not in {"euclidean", "cosine"}:
            raise ValueError(f"unsupported distance metric: {self.metric!r}")
        if self.work.returned_candidates != len(ids):
            raise ValueError("work.returned_candidates must equal probe result size")
        ids.setflags(write=False)
        distances.setflags(write=False)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "distances", distances)


def probe_telemetry_from_search(
    search: SearchResult,
    authorized_mask: np.ndarray,
    *,
    metric: DistanceMetric,
    max_neighbors: int,
) -> ProbeTelemetry:
    """Validate and freeze an authorized search result as bounded probe input."""
    mask = np.asarray(authorized_mask, dtype=bool)
    ids = np.asarray(search.ids, dtype=np.int64)
    if mask.ndim != 1:
        raise ValueError("authorized_mask must be one-dimensional")
    if search.unauthorized_context or search.unauthorized_candidates:
        raise ValueError("probe search contains unauthorized material")
    if np.any(ids < 0) or np.any(ids >= len(mask)) or not mask[ids].all():
        raise ValueError("probe ids must all be authorized")
    work = search.work or SearchWork(returned_candidates=len(ids))
    return ProbeTelemetry(
        ids=ids.copy(),
        distances=np.asarray(search.distances, dtype=np.float32).copy(),
        metric=metric,
        authorized_count=int(mask.sum()),
        corpus_count=len(mask),
        max_neighbors=max_neighbors,
        search_latency_ms=float(search.latency_ms),
        work=work,
    )


class SearchIndex(Protocol):
    """Minimal interface shared by exact and approximate indexes."""

    n_documents: int

    def query(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


class LoadedHNSWBackend(Protocol):
    """Minimal interface for one digest-verified, already-built HNSW index."""

    def set_ef(self, value: int) -> None: ...

    def knn_query(
        self,
        vectors: np.ndarray,
        *,
        k: int,
        num_threads: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def snapshot_query(query: np.ndarray, dimension: int) -> np.ndarray:
    """Own and freeze one validated query vector for a governed request.

    The copy prevents a caller from changing the request between authorization,
    probe, controller, and retrieval stages. Callers should create one snapshot
    at the request boundary and pass that snapshot to every downstream stage.
    """
    vector = np.array(query, dtype=np.float32, copy=True).reshape(-1)
    if vector.size != dimension:
        raise ValueError(f"query dimension {vector.size} does not match index {dimension}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("query contains non-finite values")
    vector.setflags(write=False)
    return vector


def _as_query(query: np.ndarray, dimension: int) -> np.ndarray:
    return snapshot_query(query, dimension)


def _distances(vectors: np.ndarray, query: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    if metric == "euclidean":
        delta = vectors - query
        return np.einsum("ij,ij->i", delta, delta)
    if metric == "cosine":
        vector_norms = np.linalg.norm(vectors, axis=1)
        query_norm = float(np.linalg.norm(query))
        denom = np.clip(vector_norms * query_norm, 1e-12, None)
        return 1.0 - (vectors @ query) / denom
    raise ValueError(f"unsupported distance metric: {metric!r}")


class ExactSearchIndex:
    """Numpy exact search, used both as a baseline and policy oracle."""

    def __init__(self, vectors: np.ndarray, metric: DistanceMetric = "euclidean") -> None:
        matrix = np.array(vectors, dtype=np.float32, copy=True)
        if matrix.ndim != 2 or len(matrix) == 0:
            raise ValueError("vectors must have shape (n_documents, dimension), n > 0")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("vectors contain non-finite values")
        matrix.setflags(write=False)
        self.vectors = matrix
        self.metric = metric
        self.n_documents, self.dimension = matrix.shape

    def query(
        self,
        query: np.ndarray,
        k: int,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be positive")
        vector = _as_query(query, self.dimension)
        if mask is None:
            candidate_ids = np.arange(self.n_documents)
        else:
            authorized = np.asarray(mask, dtype=bool)
            if authorized.shape != (self.n_documents,):
                raise ValueError("mask must have shape (n_documents,)")
            candidate_ids = np.flatnonzero(authorized)
        if candidate_ids.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        distances = _distances(self.vectors[candidate_ids], vector, self.metric)
        keep = min(k, candidate_ids.size)
        local = np.argpartition(distances, keep - 1)[:keep]
        local = local[np.argsort(distances[local], kind="stable")]
        return candidate_ids[local].astype(np.int64), distances[local].astype(np.float32)


class AuthorizedExactIndex:
    """Exact index that owns only one policy-authorized vector slice.

    ``original_ids`` maps local rows back to the corpus document universe. The
    inner exact index has no handle to denied vectors, so a query cannot read or
    score them after the authorization boundary has been established.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        authorized_mask: np.ndarray,
        metric: DistanceMetric = "euclidean",
    ) -> None:
        shape = np.shape(vectors)
        if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
            raise ValueError("vectors must have shape (n_documents, dimension), n > 0")
        mask = np.asarray(authorized_mask, dtype=bool)
        if mask.shape != (shape[0],):
            raise ValueError("authorized_mask must have shape (n_documents,)")
        original_ids = np.flatnonzero(mask).astype(np.int64)
        if original_ids.size == 0:
            raise ValueError("authorized universe is empty")

        # Perform the only source-vector read after authorization, selecting
        # permitted rows before constructing the queryable index.
        authorized_vectors = np.asarray(vectors[original_ids], dtype=np.float32)
        self._inner = ExactSearchIndex(authorized_vectors, metric=metric)
        original_ids.setflags(write=False)
        self.original_ids = original_ids
        self.n_documents = int(shape[0])
        self.n_authorized = len(original_ids)
        self.dimension = int(shape[1])
        self.metric = metric

    def query(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        local_ids, distances = self._inner.query(query, k)
        return self.original_ids[local_ids], distances


class HNSWSearchIndex:
    """A thin hnswlib backend used by the empirical benchmark."""

    def __init__(
        self,
        vectors: np.ndarray,
        metric: DistanceMetric = "euclidean",
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 42,
    ) -> None:
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError(
                "hnswlib is required for HNSWSearchIndex; install with `pip install -e .[hnsw]`"
            ) from exc

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) == 0:
            raise ValueError("vectors must have shape (n_documents, dimension), n > 0")
        if metric not in {"euclidean", "cosine"}:
            raise ValueError(f"unsupported distance metric: {metric!r}")
        self.n_documents, self.dimension = matrix.shape
        self.metric = metric
        space = "l2" if metric == "euclidean" else "cosine"
        index = hnswlib.Index(space=space, dim=self.dimension)
        index.init_index(
            max_elements=self.n_documents,
            ef_construction=ef_construction,
            M=m,
            random_seed=seed,
        )
        index.set_num_threads(1)
        index.add_items(matrix, np.arange(self.n_documents), num_threads=1)
        index.set_ef(max(ef_search, 10))
        self._index = index
        self._query_lock = RLock()

    def set_ef(self, ef_search: int) -> None:
        if ef_search <= 0:
            raise ValueError("ef_search must be positive")
        with self._query_lock:
            self._index.set_ef(max(ef_search, 10))

    def _query_unlocked(
        self,
        vector: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        keep = min(k, self.n_documents)
        labels, distances = self._index.knn_query(vector.reshape(1, -1), k=keep, num_threads=1)
        return labels[0].astype(np.int64), distances[0].astype(np.float32)

    def query(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be positive")
        vector = _as_query(query, self.dimension)
        with self._query_lock:
            return self._query_unlocked(vector, k)

    def query_with_ef(
        self,
        query: np.ndarray,
        k: int,
        *,
        ef_search: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Set request effort and execute its query in one critical section."""
        if k <= 0:
            raise ValueError("k must be positive")
        if ef_search <= 0:
            raise ValueError("ef_search must be positive")
        vector = _as_query(query, self.dimension)
        with self._query_lock:
            self._index.set_ef(max(ef_search, 10))
            return self._query_unlocked(vector, k)


class LoadedHNSWSearchIndex:
    """Query wrapper for immutable HNSW bytes loaded by the freeze verifier."""

    def __init__(
        self,
        backend: LoadedHNSWBackend,
        *,
        n_documents: int,
        dimension: int,
        metric: DistanceMetric,
    ) -> None:
        if not isinstance(n_documents, int) or isinstance(n_documents, bool) or n_documents <= 0:
            raise ValueError("n_documents must be a positive integer")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if metric not in {"euclidean", "cosine"}:
            raise ValueError(f"unsupported distance metric: {metric!r}")
        self._backend = backend
        self.n_documents = n_documents
        self.dimension = dimension
        self.metric = metric
        self._query_lock = RLock()

    def _query_unlocked(
        self,
        vector: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        keep = min(k, self.n_documents)
        labels, distances = self._backend.knn_query(
            vector.reshape(1, -1),
            k=keep,
            num_threads=1,
        )
        labels = np.asarray(labels)
        distances = np.asarray(distances)
        if labels.shape != (1, keep) or distances.shape != (1, keep):
            raise RuntimeError("loaded HNSW backend returned a malformed result")
        ids = labels[0].astype(np.int64, copy=True)
        values = distances[0].astype(np.float32, copy=True)
        if (
            np.any(ids < 0)
            or np.any(ids >= self.n_documents)
            or len(np.unique(ids)) != len(ids)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise RuntimeError("loaded HNSW backend returned an invalid result")
        return ids, values

    def query(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.query_with_ef(query, k, ef_search=max(k, 10))

    def query_with_ef(
        self,
        query: np.ndarray,
        k: int,
        *,
        ef_search: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be positive")
        if ef_search <= 0:
            raise ValueError("ef_search must be positive")
        vector = _as_query(query, self.dimension)
        with self._query_lock:
            self._backend.set_ef(max(ef_search, 10))
            return self._query_unlocked(vector, k)


class AuthorizedHNSWIndex:
    """HNSW built only over one live policy-authorized universe."""

    def __init__(
        self,
        vectors: np.ndarray,
        authorized_mask: np.ndarray,
        metric: DistanceMetric = "euclidean",
        **params: int,
    ) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        mask = np.asarray(authorized_mask, dtype=bool)
        if mask.shape != (len(matrix),):
            raise ValueError("authorized_mask must have shape (n_documents,)")
        self.original_ids = np.flatnonzero(mask).astype(np.int64)
        if self.original_ids.size == 0:
            raise ValueError("authorized universe is empty")
        self._inner = HNSWSearchIndex(matrix[self.original_ids], metric=metric, **params)
        self.n_documents = len(matrix)
        self.n_authorized = len(self.original_ids)
        self.metric = metric

    @classmethod
    def from_loaded_backend(
        cls,
        backend: LoadedHNSWBackend,
        original_ids: np.ndarray,
        *,
        n_documents: int,
        dimension: int,
        metric: DistanceMetric,
    ) -> AuthorizedHNSWIndex:
        """Bind a verified local-label index to its ordered global row map."""

        if not isinstance(n_documents, int) or isinstance(n_documents, bool) or n_documents <= 0:
            raise ValueError("n_documents must be a positive integer")
        rows = np.array(original_ids, dtype=np.int64, copy=True)
        if (
            rows.ndim != 1
            or len(rows) == 0
            or np.any(rows < 0)
            or np.any(rows >= n_documents)
            or np.any(rows[1:] <= rows[:-1])
        ):
            raise ValueError("original_ids must be strictly increasing rows in the universe")
        rows.setflags(write=False)
        instance = cls.__new__(cls)
        instance.original_ids = rows
        instance._inner = LoadedHNSWSearchIndex(
            backend,
            n_documents=len(rows),
            dimension=dimension,
            metric=metric,
        )
        instance.n_documents = n_documents
        instance.n_authorized = len(rows)
        instance.metric = metric
        return instance

    def set_ef(self, ef_search: int) -> None:
        self._inner.set_ef(ef_search)

    def query(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        local_ids, distances = self._inner.query(query, min(k, self.n_authorized))
        return self.original_ids[local_ids], distances

    def query_with_ef(
        self,
        query: np.ndarray,
        k: int,
        *,
        ef_search: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_ids, distances = self._inner.query_with_ef(
            query,
            min(k, self.n_authorized),
            ef_search=ef_search,
        )
        return self.original_ids[local_ids], distances


def authorized_hnsw_search(
    index: AuthorizedHNSWIndex,
    query: np.ndarray,
    authorized_mask: np.ndarray,
    k: int,
    *,
    ef_search: int,
    strategy: str,
) -> SearchResult:
    """Search an index whose candidate universe was authorized before build."""
    start = perf_counter_ns()
    ids, distances = index.query_with_ef(query, k, ef_search=ef_search)
    latency_ms = (perf_counter_ns() - start) / 1_000_000
    mask = np.asarray(authorized_mask, dtype=bool)
    unauthorized = int((~mask[ids]).sum())
    if unauthorized:
        raise RuntimeError("authorization-first index returned an unauthorized document")
    return SearchResult(
        ids=ids,
        distances=distances,
        strategy=strategy,
        requested_k=k,
        candidates_examined=ef_search,
        unauthorized_candidates=0,
        unauthorized_context=0,
        latency_ms=latency_ms,
        work=SearchWork(
            returned_candidates=len(ids),
            # hnswlib exposes the configured efSearch but not actual visits or
            # distance computations. Do not relabel the configuration as work.
            configured_ef_search=ef_search,
        ),
    )


def authorized_hnsw_probe(
    index: AuthorizedHNSWIndex,
    query: np.ndarray,
    authorized_mask: np.ndarray,
    *,
    probe_k: int = 101,
    ef_search: int = 128,
    max_neighbors: int = 101,
) -> ProbeTelemetry:
    """Run one bounded authorized search and freeze its online telemetry.

    The default bound leaves room for a zero-distance self match while exposing
    100 positive neighbors for the registered LID scales 20, 50, and 100.
    """
    if probe_k <= 0 or probe_k > max_neighbors:
        raise ValueError("probe_k must be positive and no larger than max_neighbors")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")
    search = authorized_hnsw_search(
        index,
        query,
        authorized_mask,
        probe_k,
        ef_search=max(ef_search, probe_k),
        strategy="hnsw-probe",
    )
    return probe_telemetry_from_search(
        search,
        authorized_mask,
        metric=index.metric,
        max_neighbors=max_neighbors,
    )


def search_result_from_probe(
    probe: ProbeTelemetry,
    k: int,
    *,
    strategy: str = "hnsw-low",
) -> SearchResult:
    """Reuse a bounded probe as the low-effort result without a second search."""
    if k <= 0:
        raise ValueError("k must be positive")
    keep = min(k, len(probe.ids))
    configured = probe.work.configured_ef_search
    return SearchResult(
        ids=probe.ids[:keep].copy(),
        distances=probe.distances[:keep].copy(),
        strategy=strategy,
        requested_k=k,
        candidates_examined=(configured if configured is not None else len(probe.ids)),
        unauthorized_candidates=0,
        unauthorized_context=0,
        latency_ms=probe.search_latency_ms,
        work=probe.work,
    )


def authorized_exact_search(
    index: AuthorizedExactIndex,
    query: np.ndarray,
    k: int,
) -> SearchResult:
    """Search a pre-authorized exact index and account for its full slice scan."""
    start = perf_counter_ns()
    ids, distances = index.query(query, k)
    latency_ms = (perf_counter_ns() - start) / 1_000_000
    return SearchResult(
        ids=ids,
        distances=distances,
        strategy="exact-authorized",
        requested_k=k,
        candidates_examined=index.n_authorized,
        unauthorized_candidates=0,
        unauthorized_context=0,
        latency_ms=latency_ms,
        work=SearchWork(
            returned_candidates=len(ids),
            visited_candidates=index.n_authorized,
            distance_evaluations=index.n_authorized,
        ),
    )


def exact_authorized_search(
    index: ExactSearchIndex,
    query: np.ndarray,
    authorized_mask: np.ndarray,
    k: int,
) -> SearchResult:
    """Offline masked oracle retained for benchmark and development comparisons.

    Governed online retrieval must use :func:`authorized_exact_search` so its
    queryable index never owns denied vectors.
    """
    start = perf_counter_ns()
    ids, distances = index.query(query, k, mask=authorized_mask)
    latency_ms = (perf_counter_ns() - start) / 1_000_000
    authorized_count = int(np.asarray(authorized_mask, dtype=bool).sum())
    return SearchResult(
        ids=ids,
        distances=distances,
        strategy="exact-authorized",
        requested_k=k,
        candidates_examined=authorized_count,
        unauthorized_candidates=0,
        unauthorized_context=0,
        latency_ms=latency_ms,
        work=SearchWork(
            returned_candidates=len(ids),
            visited_candidates=authorized_count,
            distance_evaluations=authorized_count,
        ),
    )


def safe_post_filter_search(
    index: SearchIndex,
    query: np.ndarray,
    authorized_mask: np.ndarray,
    k: int,
    *,
    candidate_k: int,
) -> SearchResult:
    """Search globally, enforce IAM before context assembly, then truncate."""
    mask = np.asarray(authorized_mask, dtype=bool)
    if mask.shape != (index.n_documents,):
        raise ValueError("authorized_mask must have shape (n_documents,)")
    start = perf_counter_ns()
    candidate_ids, candidate_distances = index.query(query, min(candidate_k, index.n_documents))
    allowed = mask[candidate_ids]
    unauthorized_candidates = int((~allowed).sum())
    ids = candidate_ids[allowed][:k]
    distances = candidate_distances[allowed][:k]
    latency_ms = (perf_counter_ns() - start) / 1_000_000
    return SearchResult(
        ids=ids,
        distances=distances,
        strategy="safe-post-filter",
        requested_k=k,
        candidates_examined=len(candidate_ids),
        unauthorized_candidates=unauthorized_candidates,
        unauthorized_context=0,
        latency_ms=latency_ms,
        work=SearchWork(returned_candidates=len(ids)),
    )


def unsafe_unfiltered_search(
    index: SearchIndex,
    query: np.ndarray,
    authorized_mask: np.ndarray,
    k: int,
) -> SearchResult:
    """Explicitly unsafe comparator: global candidates cross the context boundary."""
    mask = np.asarray(authorized_mask, dtype=bool)
    start = perf_counter_ns()
    ids, distances = index.query(query, k)
    unauthorized = int((~mask[ids]).sum())
    latency_ms = (perf_counter_ns() - start) / 1_000_000
    return SearchResult(
        ids=ids,
        distances=distances,
        strategy="unsafe-unfiltered",
        requested_k=k,
        candidates_examined=len(ids),
        unauthorized_candidates=unauthorized,
        unauthorized_context=unauthorized,
        latency_ms=latency_ms,
        work=SearchWork(returned_candidates=len(ids)),
    )
