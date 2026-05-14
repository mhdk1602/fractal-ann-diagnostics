"""ANN-benchmarks corpus harness for descriptor calibration.

Reads the standard ANN-benchmarks HDF5 datasets (SIFT, GIST, GloVe, DEEP1B
subsets, MS-MARCO embeddings, etc.) and provides a uniform interface for
descriptor extraction and downstream evaluation.

Reference
---------
- Aumüller, M., Bernhardsson, E., Faithfull, A. (2020). ANN-Benchmarks:
  A benchmarking tool for approximate nearest neighbor algorithms.
  Information Systems, 87.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AnnDataset:
    """One ANN-benchmarks dataset, loaded and ready for descriptor extraction."""

    name: str
    train: np.ndarray
    test: np.ndarray
    neighbors: np.ndarray
    distances: np.ndarray
    distance_metric: str


CANONICAL_DATASETS: tuple[str, ...] = (
    "sift-128-euclidean",
    "gist-960-euclidean",
    "glove-25-angular",
    "glove-100-angular",
    "glove-200-angular",
    "deep-image-96-angular",
    "fashion-mnist-784-euclidean",
    "lastfm-64-dot",
    "mnist-784-euclidean",
    "nytimes-256-angular",
)


def load_ann_benchmark(name: str, cache_dir: Path) -> AnnDataset:
    """Load a single ANN-benchmarks dataset by name.

    Downloads the HDF5 to ``cache_dir`` on first call. The set of canonical
    names is in CANONICAL_DATASETS.
    """
    raise NotImplementedError("load_ann_benchmark unimplemented in v0.0.1; HDF5 loader lands at v0.1.0.")


def evaluate_recall(
    index_handle, queries: np.ndarray, ground_truth: np.ndarray, k: int = 10
) -> float:
    """Compute recall@k against ground-truth neighbour ids."""
    raise NotImplementedError("evaluate_recall unimplemented in v0.0.1.")


def descriptor_panel_for_corpus(cache_dir: Path) -> dict[str, dict]:
    """Compute the descriptor panel for every canonical ANN-benchmarks dataset.

    Returns a nested dict[dataset_name][descriptor_name] -> value, suitable
    for the calibration step at v0.1.0.
    """
    raise NotImplementedError("descriptor_panel_for_corpus unimplemented in v0.0.1.")
