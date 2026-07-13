"""Compatibility wrappers around measured retrieval backends.

The diagnostic recommender outputs one of {HNSW, IVF, flat-NSW, DiskANN}; this
module isolates the actual library calls so the rest of the package is
library-agnostic. HNSW is implemented in v0.2.0; the other index-selection
stubs remain part of the retired v0.1 research path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .retrieval import HNSWSearchIndex


@dataclass(frozen=True)
class IndexHandle:
    """Opaque handle to a built ANN index."""

    name: str
    backend: str
    index: Any


class ANNIndex(Protocol):
    """Common interface for every ANN backend wrapper."""

    name: str

    def build(self, vectors: np.ndarray, **params) -> IndexHandle: ...

    def query(self, handle: IndexHandle, queries: np.ndarray, k: int) -> np.ndarray: ...


def build_hnsw(vectors: np.ndarray, M: int = 16, ef_construction: int = 200) -> IndexHandle:
    """Build an HNSW index via hnswlib."""
    index = HNSWSearchIndex(vectors, m=M, ef_construction=ef_construction)
    return IndexHandle(name="hnsw", backend="hnswlib", index=index)


def build_ivf(vectors: np.ndarray, nlist: int = 1024) -> IndexHandle:
    """Build a FAISS IVF index. Stub in v0.0.1."""
    raise NotImplementedError("IVF build unimplemented in v0.0.1; integrate faiss at v0.1.0.")


def build_flat_nsw(vectors: np.ndarray, M: int = 16) -> IndexHandle:
    """Build a flat NSW (no hierarchy) graph index. Stub in v0.0.1.

    The Hub Highway Hypothesis (2024) argues that flat NSW matches HNSW in
    high-dimensional regimes; this wrapper exists to make that empirically
    testable from the diagnostic output.
    """
    raise NotImplementedError("flat-NSW build unimplemented in v0.0.1.")


def build_diskann(vectors: np.ndarray) -> IndexHandle:
    """Build a DiskANN index. Stub in v0.0.1."""
    raise NotImplementedError("DiskANN build unimplemented in v0.0.1.")
