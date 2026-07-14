"""Permutation-invariant query geometry for filtered retrieval risk."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Literal

import numpy as np

from .retrieval import DistanceMetric, ProbeTelemetry, _distances

PREREGISTERED_LID_SCALES = (20, 50, 100)
PRIMARY_LID_SCALE = 50
LEGACY_ORACLE_LID_SCALES = (10, 20, 40)


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
    source: Literal["unspecified", "bounded-probe", "offline-oracle"] = "unspecified"
    lid_by_scale: tuple[tuple[int, float], ...] = ()
    probe_neighbors: int | None = None
    search_latency_ms: float = 0.0
    feature_latency_ms: float = 0.0
    visited_candidates: int | None = None
    distance_evaluations: int | None = None
    configured_ef_search: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "lid",
            "lid_scale_instability",
            "relative_contrast",
            "radius_expansion",
        ):
            if np.isinf(float(getattr(self, name))):
                raise ValueError(f"{name} cannot be infinite")
        if not np.isfinite(self.authorized_selectivity) or not (
            0.0 <= self.authorized_selectivity <= 1.0
        ):
            raise ValueError("authorized_selectivity must be finite and in [0, 1]")
        if not np.isfinite(self.policy_churn) or not 0.0 <= self.policy_churn <= 1.0:
            raise ValueError("policy_churn must be finite and in [0, 1]")
        if not np.isfinite(self.embedding_drift) or self.embedding_drift < 0.0:
            raise ValueError("embedding_drift must be finite and non-negative")
        for name in ("search_latency_ms", "feature_latency_ms"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "probe_neighbors",
            "visited_candidates",
            "distance_evaluations",
            "configured_ef_search",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        scales = [scale for scale, _ in self.lid_by_scale]
        if scales != sorted(set(scales)):
            raise ValueError("lid_by_scale must use unique increasing scales")
        if any(scale < 3 or np.isinf(float(value)) for scale, value in self.lid_by_scale):
            raise ValueError("lid_by_scale contains an invalid scale or value")

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

    @property
    def accounted_latency_ms(self) -> float:
        """Search plus feature time included in the online decision boundary."""
        return self.search_latency_ms + self.feature_latency_ms


def _lid_from_sorted_distances(
    distances: np.ndarray,
    k: int,
    *,
    require_full_scale: bool = False,
) -> float:
    positive = np.asarray(distances, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size < 3:
        return float("nan")
    if require_full_scale and positive.size < k:
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


def _summarize_sorted_distances(
    sorted_distances: np.ndarray,
    *,
    scales: tuple[int, ...],
    primary_scale: int,
    require_full_scale: bool,
) -> tuple[float, float, float, float, tuple[tuple[int, float], ...]]:
    if not scales or min(scales) < 3 or len(set(scales)) != len(scales):
        raise ValueError("scales must contain unique integers >= 3")
    if tuple(sorted(scales)) != scales:
        raise ValueError("scales must be in increasing order")
    if primary_scale not in scales:
        raise ValueError("primary_scale must be one of scales")

    lid_by_scale = tuple(
        (
            scale,
            _lid_from_sorted_distances(
                sorted_distances,
                scale,
                require_full_scale=require_full_scale,
            ),
        )
        for scale in scales
    )
    lids = np.asarray([value for _, value in lid_by_scale], dtype=np.float64)
    finite_lids = lids[np.isfinite(lids)]
    lid = dict(lid_by_scale)[primary_scale]
    instability = (
        float(np.std(finite_lids) / max(np.mean(finite_lids), 1e-12))
        if finite_lids.size >= 2
        else float("nan")
    )

    positive = sorted_distances[np.isfinite(sorted_distances) & (sorted_distances > 0)]
    if positive.size:
        relative_contrast = float(np.median(positive) / max(positive[0], 1e-12))
        if require_full_scale and positive.size < scales[-1]:
            radius_expansion = float("nan")
        else:
            lo_idx = min(scales[0] - 1, positive.size - 1)
            hi_idx = min(scales[-1] - 1, positive.size - 1)
            radius_expansion = float(positive[hi_idx] / max(positive[lo_idx], 1e-12))
    else:
        relative_contrast = float("nan")
        radius_expansion = float("nan")
    return lid, instability, relative_contrast, radius_expansion, lid_by_scale


def query_geometry_from_probe(
    probe: ProbeTelemetry,
    *,
    scales: tuple[int, ...] = PREREGISTERED_LID_SCALES,
    primary_scale: int = PRIMARY_LID_SCALE,
    policy_churn: float = 0.0,
    embedding_drift: float = 0.0,
) -> QueryGeometry:
    """Compute online features from one bounded authorized probe, and nothing else.

    LID at 50 is the registered primary feature. Values at 20 and 100 are
    retained in ``lid_by_scale`` for the registered sensitivity analysis.
    """
    start = perf_counter_ns()
    distances = np.sort(np.asarray(probe.distances, dtype=np.float64), kind="stable")
    if probe.metric == "euclidean":
        distances = np.sqrt(np.clip(distances, 0.0, None))
    lid, instability, contrast, expansion, lid_by_scale = _summarize_sorted_distances(
        distances,
        scales=scales,
        primary_scale=primary_scale,
        require_full_scale=True,
    )
    feature_latency_ms = (perf_counter_ns() - start) / 1_000_000
    return QueryGeometry(
        lid=lid,
        lid_scale_instability=instability,
        authorized_selectivity=probe.authorized_count / probe.corpus_count,
        relative_contrast=contrast,
        radius_expansion=expansion,
        policy_churn=float(policy_churn),
        embedding_drift=float(embedding_drift),
        source="bounded-probe",
        lid_by_scale=lid_by_scale,
        probe_neighbors=len(probe.ids),
        search_latency_ms=probe.search_latency_ms,
        feature_latency_ms=feature_latency_ms,
        visited_candidates=probe.work.visited_candidates,
        distance_evaluations=probe.work.distance_evaluations,
        configured_ef_search=probe.work.configured_ef_search,
    )


def offline_oracle_query_geometry(
    vectors: np.ndarray,
    query: np.ndarray,
    authorized_mask: np.ndarray,
    *,
    metric: DistanceMetric = "euclidean",
    scales: tuple[int, ...] = LEGACY_ORACLE_LID_SCALES,
    primary_scale: int | None = None,
    policy_churn: float = 0.0,
    embedding_drift: float = 0.0,
) -> QueryGeometry:
    """Compute exact authorized geometry for offline analysis only.

    The previous MFDFA-on-pair-order statistic was not invariant to row
    permutation. This replacement uses neighbor-distance order statistics;
    permuting corpus rows cannot change the feature values. Unauthorized
    vectors are removed before any query-specific geometry is computed. This
    function scans the full authorized universe and reports that work; it is not
    an online controller feature source.
    """
    start = perf_counter_ns()
    matrix = np.asarray(vectors, dtype=np.float32)
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    mask = np.asarray(authorized_mask, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[1] != vector.size:
        raise ValueError("vectors and query dimensions do not match")
    if mask.shape != (len(matrix),):
        raise ValueError("authorized_mask must have shape (n_documents,)")
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
    selected_primary_scale = scales[-1] if primary_scale is None else primary_scale
    lid, instability, contrast, expansion, lid_by_scale = _summarize_sorted_distances(
        sorted_distances,
        scales=scales,
        primary_scale=selected_primary_scale,
        require_full_scale=False,
    )
    selectivity = float(mask.mean()) if mask.size else 0.0
    feature_latency_ms = (perf_counter_ns() - start) / 1_000_000
    return QueryGeometry(
        lid=lid,
        lid_scale_instability=instability,
        authorized_selectivity=selectivity,
        relative_contrast=contrast,
        radius_expansion=expansion,
        policy_churn=float(policy_churn),
        embedding_drift=float(embedding_drift),
        source="offline-oracle",
        lid_by_scale=lid_by_scale,
        feature_latency_ms=feature_latency_ms,
        visited_candidates=len(authorized_vectors),
        distance_evaluations=len(authorized_vectors),
    )


def query_geometry(
    vectors: np.ndarray,
    query: np.ndarray,
    authorized_mask: np.ndarray,
    *,
    metric: DistanceMetric = "euclidean",
    scales: tuple[int, ...] = LEGACY_ORACLE_LID_SCALES,
    policy_churn: float = 0.0,
    embedding_drift: float = 0.0,
) -> QueryGeometry:
    """Compatibility alias for the exact offline oracle.

    New online code must call :func:`query_geometry_from_probe`.
    """
    warnings.warn(
        "query_geometry performs an exact offline scan; use "
        "offline_oracle_query_geometry explicitly or query_geometry_from_probe online",
        DeprecationWarning,
        stacklevel=2,
    )
    return offline_oracle_query_geometry(
        vectors,
        query,
        authorized_mask,
        metric=metric,
        scales=scales,
        policy_churn=policy_churn,
        embedding_drift=embedding_drift,
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
    # Explicit float64 contraction avoids backend-specific mixed-precision GEMV
    # behavior and keeps content-based ordering stable across supported NumPy builds.
    sample_scores = np.einsum(
        "ij,j->i",
        matrix,
        weights,
        dtype=np.float64,
        optimize=False,
    )
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
