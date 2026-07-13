"""Exact and HNSW retrieval paths with explicit authorization boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Literal, Protocol

import numpy as np

DistanceMetric = Literal["euclidean", "cosine"]


@dataclass(frozen=True)
class SearchResult:
    """One retrieval action plus its policy-boundary accounting."""

    ids: np.ndarray
    distances: np.ndarray
    strategy: str
    requested_k: int
    candidates_examined: int
    unauthorized_candidates: int
    unauthorized_context: int
    latency_ms: float

    @property
    def shortfall(self) -> int:
        return max(0, self.requested_k - len(self.ids))


class SearchIndex(Protocol):
    """Minimal interface shared by exact and approximate indexes."""

    n_documents: int

    def query(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


def _as_query(query: np.ndarray, dimension: int) -> np.ndarray:
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    if vector.size != dimension:
        raise ValueError(f"query dimension {vector.size} does not match index {dimension}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("query contains non-finite values")
    return vector


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
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) == 0:
            raise ValueError("vectors must have shape (n_documents, dimension), n > 0")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("vectors contain non-finite values")
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

    def set_ef(self, ef_search: int) -> None:
        if ef_search <= 0:
            raise ValueError("ef_search must be positive")
        self._index.set_ef(max(ef_search, 10))

    def query(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be positive")
        vector = _as_query(query, self.dimension)
        keep = min(k, self.n_documents)
        labels, distances = self._index.knn_query(
            vector.reshape(1, -1), k=keep, num_threads=1
        )
        return labels[0].astype(np.int64), distances[0].astype(np.float32)


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

    def set_ef(self, ef_search: int) -> None:
        self._inner.set_ef(ef_search)

    def query(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        local_ids, distances = self._inner.query(query, min(k, self.n_authorized))
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
    index.set_ef(ef_search)
    start = perf_counter_ns()
    ids, distances = index.query(query, k)
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
    )


def exact_authorized_search(
    index: ExactSearchIndex,
    query: np.ndarray,
    authorized_mask: np.ndarray,
    k: int,
) -> SearchResult:
    start = perf_counter_ns()
    ids, distances = index.query(query, k, mask=authorized_mask)
    latency_ms = (perf_counter_ns() - start) / 1_000_000
    return SearchResult(
        ids=ids,
        distances=distances,
        strategy="exact-authorized",
        requested_k=k,
        candidates_examined=int(np.asarray(authorized_mask, dtype=bool).sum()),
        unauthorized_candidates=0,
        unauthorized_context=0,
        latency_ms=latency_ms,
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
    )
