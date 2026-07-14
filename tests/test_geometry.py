from __future__ import annotations

import numpy as np
import pytest

from fractal_ann_diagnostics.geometry import (
    multiscale_lid_dispersion,
    offline_oracle_query_geometry,
    query_geometry,
    query_geometry_from_probe,
)
from fractal_ann_diagnostics.retrieval import (
    ExactSearchIndex,
    ProbeTelemetry,
    SearchResult,
    SearchWork,
    probe_telemetry_from_search,
)


def test_multiscale_lid_dispersion_is_permutation_invariant() -> None:
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(180, 12))
    with np.errstate(all="raise"):
        baseline = multiscale_lid_dispersion(vectors, sample_size=80, seed=9)
        permuted = multiscale_lid_dispersion(
            vectors[rng.permutation(len(vectors))],
            sample_size=80,
            seed=9,
        )
    assert np.isclose(baseline, permuted, rtol=0.08, atol=0.02)


def test_query_geometry_never_reads_denied_vectors() -> None:
    rng = np.random.default_rng(12)
    authorized = rng.normal(size=(100, 10))
    denied = rng.normal(50, 10, size=(100, 10))
    vectors = np.vstack([authorized, denied])
    mask = np.r_[np.ones(100, dtype=bool), np.zeros(100, dtype=bool)]
    query = authorized[0] + 0.01
    first = query_geometry(vectors, query, mask)
    changed = vectors.copy()
    changed[~mask] = rng.normal(-1000, 500, size=changed[~mask].shape)
    second = query_geometry(changed, query, mask)
    assert np.allclose(first.as_array(), second.as_array(), equal_nan=True)


def _probe(order: np.ndarray | None = None) -> ProbeTelemetry:
    ids = np.arange(101, dtype=np.int64)
    # hnswlib's Euclidean result uses squared L2 distances.
    distances = np.square(np.linspace(0.05, 5.05, 101, dtype=np.float32))
    if order is not None:
        ids = ids[order]
        distances = distances[order]
    return ProbeTelemetry(
        ids=ids,
        distances=distances,
        metric="euclidean",
        authorized_count=150,
        corpus_count=300,
        max_neighbors=101,
        search_latency_ms=1.25,
        work=SearchWork(
            returned_candidates=101,
            visited_candidates=137,
            distance_evaluations=141,
            configured_ef_search=160,
        ),
    )


def test_probe_geometry_uses_registered_scales_and_accounts_for_work() -> None:
    geometry = query_geometry_from_probe(_probe(), policy_churn=0.02)
    per_scale = dict(geometry.lid_by_scale)
    assert tuple(per_scale) == (20, 50, 100)
    assert geometry.lid == per_scale[50]
    assert geometry.source == "bounded-probe"
    assert geometry.probe_neighbors == 101
    assert geometry.visited_candidates == 137
    assert geometry.distance_evaluations == 141
    assert geometry.configured_ef_search == 160
    assert geometry.accounted_latency_ms >= 1.25


def test_probe_geometry_is_invariant_to_candidate_order() -> None:
    rng = np.random.default_rng(91)
    first = query_geometry_from_probe(_probe())
    second = query_geometry_from_probe(_probe(rng.permutation(101)))
    assert np.allclose(first.as_array(), second.as_array(), rtol=1e-12, atol=1e-12)
    assert np.allclose(
        [value for _, value in first.lid_by_scale],
        [value for _, value in second.lid_by_scale],
        rtol=1e-12,
        atol=1e-12,
    )


def test_probe_rejects_results_over_the_frozen_bound() -> None:
    result = SearchResult(
        ids=np.arange(102),
        distances=np.linspace(0.1, 2.0, 102),
        strategy="test-probe",
        requested_k=102,
        candidates_examined=102,
        unauthorized_candidates=0,
        unauthorized_context=0,
        latency_ms=0.2,
        work=SearchWork(returned_candidates=102),
    )
    mask = np.ones(200, dtype=bool)
    with pytest.raises(ValueError, match="exceeds max_neighbors"):
        probe_telemetry_from_search(result, mask, metric="euclidean", max_neighbors=101)


def test_probe_rejects_denied_candidate_before_geometry() -> None:
    result = SearchResult(
        ids=np.asarray([0, 1, 2]),
        distances=np.asarray([0.1, 0.2, 0.3]),
        strategy="test-probe",
        requested_k=3,
        candidates_examined=3,
        unauthorized_candidates=0,
        unauthorized_context=0,
        latency_ms=0.1,
        work=SearchWork(returned_candidates=3),
    )
    mask = np.asarray([True, False, True])
    with pytest.raises(ValueError, match="all be authorized"):
        probe_telemetry_from_search(result, mask, metric="euclidean", max_neighbors=3)


def test_offline_oracle_reports_full_authorized_scan_work() -> None:
    rng = np.random.default_rng(92)
    vectors = rng.normal(size=(180, 12))
    mask = np.r_[np.ones(150, dtype=bool), np.zeros(30, dtype=bool)]
    geometry = offline_oracle_query_geometry(
        vectors,
        rng.normal(size=12),
        mask,
        scales=(20, 50, 100),
        primary_scale=50,
    )
    assert geometry.source == "offline-oracle"
    assert geometry.visited_candidates == 150
    assert geometry.distance_evaluations == 150
    assert geometry.lid == dict(geometry.lid_by_scale)[50]


def test_exact_search_respects_dataset_metric() -> None:
    vectors = np.asarray([[10.0, 0.0], [1.0, 0.1], [0.8, 0.2]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    euclidean_ids, _ = ExactSearchIndex(vectors, "euclidean").query(query, 1)
    cosine_ids, _ = ExactSearchIndex(vectors, "cosine").query(query, 1)
    assert euclidean_ids.tolist() == [1]
    assert cosine_ids.tolist() == [0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_churn", float("nan")),
        ("policy_churn", 1.1),
        ("embedding_drift", float("nan")),
        ("embedding_drift", -0.1),
    ],
)
def test_query_geometry_rejects_invalid_control_covariates(
    field: str,
    value: float,
) -> None:
    values = {
        "lid": 4.0,
        "lid_scale_instability": 0.1,
        "authorized_selectivity": 0.5,
        "relative_contrast": 2.0,
        "radius_expansion": 1.2,
        "policy_churn": 0.0,
        "embedding_drift": 0.0,
    }
    values[field] = value
    with pytest.raises(ValueError):
        from fractal_ann_diagnostics.geometry import QueryGeometry

        QueryGeometry(**values)
