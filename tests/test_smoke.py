"""Smoke tests for the v0.1.1 working recommender (IVF rule tightened to n<5e4)."""
from __future__ import annotations

import numpy as np

from fractal_ann_diagnostics import __version__
from fractal_ann_diagnostics.descriptors import (
    correlation_dimension,
    hubness,
    lid_mle,
    multifractal_width,
)
from fractal_ann_diagnostics.diagnostic import (
    DiagnosticResult,
    compute_descriptors,
    diagnose,
)


def test_version_pinned() -> None:
    assert __version__ == "0.1.1"


def test_correlation_dimension_on_uniform_2d() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.uniform(0, 1, size=(500, 2))
    d2 = correlation_dimension(vectors, sample_size=500, rng=rng)
    # On uniform 2D data, D2 should be close to 2.0
    assert 1.5 < d2 < 2.3


def test_correlation_dimension_on_line_in_high_dim() -> None:
    rng = np.random.default_rng(1)
    n = 400
    t = rng.uniform(0, 1, size=n)
    # 1D structure embedded in R^20
    vectors = np.column_stack([t] + [0.01 * rng.standard_normal(n) for _ in range(19)])
    d2 = correlation_dimension(vectors, sample_size=400, rng=rng)
    # D2 should be close to 1.0 even though ambient is 20
    assert 0.7 < d2 < 1.5


def test_lid_mle_shape_and_finite() -> None:
    rng = np.random.default_rng(2)
    vectors = rng.standard_normal((300, 10))
    lid = lid_mle(vectors, k=50, sample_size=300, rng=rng)
    assert lid.shape == (300,)
    assert np.all(np.isfinite(lid))
    assert np.all(lid > 0)


def test_hubness_returns_finite_float() -> None:
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((200, 30))
    skew_value = hubness(vectors, k=10, sample_size=200, rng=rng)
    assert isinstance(skew_value, float)
    assert np.isfinite(skew_value)


def test_multifractal_width_finite_on_random() -> None:
    rng = np.random.default_rng(5)
    vectors = rng.standard_normal((200, 8))
    width = multifractal_width(vectors, sample_size=200, rng=rng)
    # Should be a finite, non-negative float on well-behaved random data.
    assert isinstance(width, float)
    assert np.isfinite(width)
    assert width >= 0.0


def test_multifractal_width_handles_degenerate_input() -> None:
    # A handful of identical points produces a constant distance series and
    # should yield NaN rather than crash.
    vectors = np.ones((10, 4))
    width = multifractal_width(vectors, sample_size=10)
    assert np.isnan(width)


def test_compute_descriptors_returns_panel() -> None:
    rng = np.random.default_rng(4)
    vectors = rng.standard_normal((200, 8))
    panel = compute_descriptors(vectors, sample_size=200, rng=rng)
    assert panel.ambient_dimension == 8
    assert panel.n_points == 200
    assert np.isfinite(panel.correlation_dimension)
    assert panel.lid_distribution.shape == (200,)
    # multifractal_width is now wired up; should be a finite float or NaN.
    assert isinstance(panel.multifractal_width, float)


def test_diagnose_returns_recommended_index() -> None:
    rng = np.random.default_rng(6)
    vectors = rng.standard_normal((200, 8))
    result = diagnose(vectors, sample_size=200)
    assert isinstance(result, DiagnosticResult)
    assert result.recommended_index in {"hnsw", "ivf", "flat-nsw", "diskann"}
    assert 0.0 <= result.predicted_recall_drop <= 1.0
    assert result.confidence == 0.5
    assert isinstance(result.rationale, str) and len(result.rationale) > 0


def test_diagnose_flat_nsw_on_high_dim_gaussian() -> None:
    # A Gaussian sample in R^32 has heavy hubness skew (well above 2.0) on
    # 400 points, which fires the Radovanović-rule for flat-NSW.
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((400, 32))
    result = diagnose(vectors, sample_size=400)
    assert result.recommended_index == "flat-nsw"


def test_diagnose_ivf_on_low_intrinsic_dim() -> None:
    # A 2D manifold embedded in R^20 with low ambient noise has D2 ~ 2 and
    # n < 5e4, which should trigger the IVF rule (rules 1 and 2 cleanly
    # decline because D2/ambient ~ 0.1 and hubness is low).
    rng = np.random.default_rng(8)
    n = 500
    t = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float64)
    M = rng.standard_normal((2, 20)).astype(np.float64)
    lifted = np.einsum("ij,jk->ik", t, M)
    vectors = lifted + 0.001 * rng.standard_normal((n, 20))
    result = diagnose(vectors, sample_size=n)
    assert result.recommended_index == "ivf"


def test_recommend_ivf_cutoff_tightened_to_5e4() -> None:
    """v0.1.1 regression: 60k-point datasets like MNIST/Fashion-MNIST must
    no longer be classified as IVF on the cardinality rule alone. The v0.1.0
    cutoff at n<1e5 sent both to IVF; v0.1.1 tightens to n<5e4."""
    from fractal_ann_diagnostics.descriptors import DescriptorPanel
    from fractal_ann_diagnostics.diagnostic import _recommend

    # Construct a panel that mimics MNIST after the descriptor pass:
    # n=60000, d=784, D2 small enough to pass the "<10" half of rule 4,
    # hubness and LID otherwise quiet so rules 1/2/3 don't fire.
    rng = np.random.default_rng(0)
    panel = DescriptorPanel(
        correlation_dimension=9.3,
        lid_distribution=rng.uniform(5.0, 25.0, size=200),
        multifractal_width=0.5,
        hubness_skew=0.6,
        ambient_dimension=784,
        n_points=60_000,
    )
    index, _drop, _rationale = _recommend(panel)
    assert index == "hnsw", (
        f"60k-point MNIST-like panel should fall through to HNSW now that the IVF "
        f"cutoff is tightened to 5e4; got {index!r} instead."
    )

    # And confirm that a strictly smaller dataset still fires IVF.
    small_panel = DescriptorPanel(
        correlation_dimension=2.0,
        lid_distribution=rng.uniform(2.0, 5.0, size=200),
        multifractal_width=0.1,
        hubness_skew=0.5,
        ambient_dimension=20,
        n_points=10_000,
    )
    index_small, _, _ = _recommend(small_panel)
    assert index_small == "ivf"
