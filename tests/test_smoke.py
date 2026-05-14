"""Smoke tests for the v0.0.1 scaffold."""
from __future__ import annotations

import numpy as np
import pytest

from fractal_ann_diagnostics import __version__
from fractal_ann_diagnostics.descriptors import (
    correlation_dimension,
    hubness,
    lid_mle,
    multifractal_width,
)
from fractal_ann_diagnostics.diagnostic import compute_descriptors, diagnose


def test_version_pinned() -> None:
    assert __version__ == "0.0.1"


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


def test_compute_descriptors_returns_panel() -> None:
    rng = np.random.default_rng(4)
    vectors = rng.standard_normal((200, 8))
    panel = compute_descriptors(vectors, sample_size=200, rng=rng)
    assert panel.ambient_dimension == 8
    assert panel.n_points == 200
    assert np.isfinite(panel.correlation_dimension)
    assert panel.lid_distribution.shape == (200,)


def test_multifractal_width_unimplemented_in_scaffold() -> None:
    rng = np.random.default_rng(5)
    vectors = rng.standard_normal((50, 4))
    with pytest.raises(NotImplementedError):
        multifractal_width(vectors)


def test_diagnose_raises_at_recommender_step() -> None:
    rng = np.random.default_rng(6)
    vectors = rng.standard_normal((100, 8))
    with pytest.raises(NotImplementedError):
        diagnose(vectors, sample_size=100)
