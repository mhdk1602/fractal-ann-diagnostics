"""Quickstart: compute the descriptor panel on a synthetic mixture.

Run from the repo root after `pip install -e .`:

    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from fractal_ann_diagnostics.descriptors import (
    correlation_dimension,
    hubness,
    lid_mle,
)


def main() -> None:
    rng = np.random.default_rng(20260514)

    # A mixture: 500 points on a 3-dimensional manifold embedded in R^50, plus
    # 500 points uniformly distributed in R^50. The mixture has heterogeneous
    # local intrinsic dimensionality, which the descriptor panel should detect.
    n_per = 500
    d_ambient = 50

    # Low-intrinsic-dim component: 3D coordinates lifted to R^50 via random linear map
    z = rng.standard_normal((n_per, 3))
    M = rng.standard_normal((3, d_ambient))
    low_d = z @ M + 0.05 * rng.standard_normal((n_per, d_ambient))

    # High-intrinsic-dim component: full-rank R^50 Gaussian
    high_d = rng.standard_normal((n_per, d_ambient))

    vectors = np.vstack([low_d, high_d])
    rng.shuffle(vectors)

    print(f"Dataset: {vectors.shape[0]} points in R^{vectors.shape[1]}")
    print()

    d2 = correlation_dimension(vectors, sample_size=500, rng=rng)
    print(f"Correlation dimension D2 = {d2:.3f}")
    print(f"  (ambient = {vectors.shape[1]}; the mixture should yield a D2 much smaller than 50)")
    print()

    lid = lid_mle(vectors, k=100, sample_size=500, rng=rng)
    print("Local intrinsic dimensionality:")
    print(f"  median = {np.median(lid):.3f}")
    print(f"  p95    = {np.quantile(lid, 0.95):.3f}")
    print("  (expect a bimodal distribution: low-LID cluster from the manifold,")
    print("   high-LID cluster from the full-rank component)")
    print()

    skew = hubness(vectors, k=10, sample_size=500, rng=rng)
    print(f"Hubness skewness = {skew:.3f}")
    print("  (>1 indicates emerging hub structure; flat-NSW backbone is forming)")


if __name__ == "__main__":
    main()
