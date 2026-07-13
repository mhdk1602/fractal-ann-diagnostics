"""Permutation-invariant query geometry for filtered retrieval risk."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .retrieval import DistanceMetric, _distances


@dataclass(frozen=True)
class QueryGeometry:
    """Observable geometry and policy features for one query."""

    lid: float
    lid_scale_instability: float
    authorized_selectivity: float
    relative_contrast: float
    radius_expansion: float
    policy_churn: float
    embedding_drift: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.lid,
                self.lid_scale_instability,
                self.authorized_selectivity,
                self.relative_contrast,
                self.radius_expansion,
                self.policy_churn,
                self.embedding_drift,
            ],
            dtype=np.float64,
        )


def _lid_from_sorted_distances(distances: np.ndarray, k: int) -> float:
    positive = np.asarray(distances, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size < 3:
        return float("nan")
    use_k = min(k, positive.size)
    neighborhood = positive[:use_k]
    radius = float(neighborhood[-1])
    if radius <= 0:
        return float("nan")
    logs = np.log(np.clip(neighborhood[:-1] / radius, 1e-12, 1.0))
    mean_log = float(logs.mean())
    if not np.isfinite(mean_log) or mean_log >= 0:
        return float("nan")
    return float(-1.0 / mean_log)


def query_geometry(
    vectors: np.ndarray,
    query: np.ndarray,
    authorized_mask: np.ndarray,
    *,
    metric: DistanceMetric = "euclidean",
    scales: tuple[int, ...] = (10, 20, 40),
    policy_churn: float = 0.0,
    embedding_drift: float = 0.0,
) -> QueryGeometry:
    """Measure geometry inside the policy-authorized universe.

    The previous MFDFA-on-pair-order statistic was not invariant to row
    permutation. This replacement uses neighbor-distance order statistics;
    permuting corpus rows cannot change the feature values. Unauthorized
    vectors are removed before any query-specific geometry is computed.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    mask = np.asarray(authorized_mask, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[1] != vector.size:
        raise ValueError("vectors and query dimensions do not match")
    if mask.shape != (len(matrix),):
        raise ValueError("authorized_mask must have shape (n_documents,)")
    if not scales or min(scales) < 3:
        raise ValueError("scales must contain integers >= 3")

    authorized_vectors = matrix[mask]
    if len(authorized_vectors) < max(scales):
        raise ValueError("authorized universe is smaller than the largest LID scale")
    distances = _distances(authorized_vectors, vector, metric)
    order = np.argsort(distances, kind="stable")
    sorted_distances = (
        np.sqrt(np.clip(distances[order], 0.0, None))
        if metric == "euclidean"
        else distances[order]
    )
    lids = np.asarray([_lid_from_sorted_distances(sorted_distances, k) for k in scales])
    finite_lids = lids[np.isfinite(lids)]
    if finite_lids.size == 0:
        lid = float("nan")
        instability = float("nan")
    else:
        lid = float(finite_lids[-1])
        instability = float(np.std(finite_lids) / max(np.mean(finite_lids), 1e-12))

    selectivity = float(mask.mean()) if mask.size else 0.0
    positive = sorted_distances[np.isfinite(sorted_distances) & (sorted_distances > 0)]
    if positive.size:
        relative_contrast = float(np.median(positive) / max(positive[0], 1e-12))
        lo_idx = min(scales[0] - 1, positive.size - 1)
        hi_idx = min(scales[-1] - 1, positive.size - 1)
        radius_expansion = float(positive[hi_idx] / max(positive[lo_idx], 1e-12))
    else:
        relative_contrast = float("nan")
        radius_expansion = float("nan")
    return QueryGeometry(
        lid=lid,
        lid_scale_instability=instability,
        authorized_selectivity=selectivity,
        relative_contrast=relative_contrast,
        radius_expansion=radius_expansion,
        policy_churn=float(policy_churn),
        embedding_drift=float(embedding_drift),
    )


def multiscale_lid_dispersion(
    vectors: np.ndarray,
    *,
    metric: DistanceMetric = "euclidean",
    scales: tuple[int, ...] = (10, 20, 40),
    sample_size: int = 256,
    seed: int = 0,
) -> float:
    """Median cross-scale LID coefficient of variation over sampled points."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < max(scales) + 1:
        return float("nan")
    count = min(sample_size, len(matrix))
    # Deterministic content-based selection keeps the estimator invariant to
    # corpus row order while retaining a bounded sample.
    weights = np.sin(np.arange(matrix.shape[1], dtype=np.float64) + float(seed) + 1.0)
    sample_scores = matrix @ weights
    sample_ids = np.argsort(sample_scores, kind="stable")[:count]
    values: list[float] = []
    for sample_id in sample_ids:
        distances = _distances(matrix, matrix[sample_id], metric)
        distances[sample_id] = np.inf
        nearest = np.partition(distances, max(scales) - 1)[: max(scales)]
        nearest.sort()
        if metric == "euclidean":
            nearest = np.sqrt(np.clip(nearest, 0.0, None))
        lids = np.asarray([_lid_from_sorted_distances(nearest, k) for k in scales])
        lids = lids[np.isfinite(lids)]
        if lids.size >= 2 and float(lids.mean()) > 0:
            values.append(float(lids.std() / lids.mean()))
    return float(np.median(values)) if values else float("nan")
