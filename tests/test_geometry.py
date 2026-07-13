from __future__ import annotations

import numpy as np

from fractal_ann_diagnostics.geometry import multiscale_lid_dispersion, query_geometry
from fractal_ann_diagnostics.retrieval import ExactSearchIndex


def test_multiscale_lid_dispersion_is_permutation_invariant() -> None:
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(180, 12))
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


def test_exact_search_respects_dataset_metric() -> None:
    vectors = np.asarray([[10.0, 0.0], [1.0, 0.1], [0.8, 0.2]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    euclidean_ids, _ = ExactSearchIndex(vectors, "euclidean").query(query, 1)
    cosine_ids, _ = ExactSearchIndex(vectors, "cosine").query(query, 1)
    assert euclidean_ids.tolist() == [1]
    assert cosine_ids.tolist() == [0]
