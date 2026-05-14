"""Fractal and intrinsic-dimension descriptors for vector datasets.

References
----------
- Grassberger, P., Procaccia, I. (1983). Characterization of strange attractors.
  Physical Review Letters, 50(5), 346.
- Amsaleg, L., Chelly, O., Furon, T., Girard, S., Houle, M.E., Kawarabayashi, K.,
  Nett, M. (2015). Estimating local intrinsic dimensionality. KDD.
- Radovanović, M., Nanopoulos, A., Ivanović, M. (2010). Hubs in space: Popular
  nearest neighbors in high-dimensional data. JMLR 11.
- Belussi, A., Faloutsos, C. (1995). Estimating the selectivity of spatial
  queries using the correlation fractal dimension. VLDB.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class DescriptorPanel:
    """The full descriptor panel for one vector dataset."""

    correlation_dimension: float
    lid_distribution: np.ndarray
    multifractal_width: float
    hubness_skew: float
    ambient_dimension: int
    n_points: int


def correlation_dimension(
    vectors: np.ndarray,
    n_scales: int = 16,
    sample_size: int | None = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """Grassberger-Procaccia correlation dimension D₂.

    Computes the correlation sum C(r) = (2 / (N(N-1))) * |{(i,j) : i<j, ||x_i - x_j|| < r}|
    over a log-spaced grid of r values, then fits the slope in the scaling region.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    n_scales : int
        Number of log-spaced r values.
    sample_size : int, optional
        If not None, randomly subsample to this many vectors before computing
        pairwise distances (memory bound; full pairwise on n=1e5 is infeasible).
    rng : np.random.Generator, optional
        For reproducible subsampling.

    Returns
    -------
    float
        Estimated D₂. In low-noise self-similar data, this is the slope of
        log C(r) vs log r in the linear scaling region.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(vectors)
    if sample_size is not None and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        x = vectors[idx]
    else:
        x = vectors

    # Pairwise distances (upper triangle only)
    n_eff = len(x)
    diffs = x[:, None, :] - x[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    iu = np.triu_indices(n_eff, k=1)
    pair_dists = dists[iu]
    pair_dists = pair_dists[pair_dists > 0]

    r_min = np.quantile(pair_dists, 0.02)
    r_max = np.quantile(pair_dists, 0.5)
    r_grid = np.geomspace(r_min, r_max, n_scales)

    c_of_r = np.array([(pair_dists < r).mean() for r in r_grid])
    log_r = np.log(r_grid)
    log_c = np.log(np.clip(c_of_r, 1e-12, None))

    # Fit the middle 60% of the curve (avoid finite-size effects at the tails)
    lo = int(0.2 * n_scales)
    hi = int(0.8 * n_scales)
    slope, _ = np.polyfit(log_r[lo:hi], log_c[lo:hi], 1)
    return float(slope)


def lid_mle(
    vectors: np.ndarray,
    k: int = 100,
    sample_size: int | None = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """MLE estimator of local intrinsic dimensionality (Amsaleg et al., 2015).

    For each point x in the (sub)sample, fit the LID via the maximum likelihood
    estimator over the k nearest neighbours::

        LID_MLE(x) = - ( (1/k) * sum_{i=1..k} log(d_i / d_k) )^(-1)

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    k : int
        Number of nearest neighbours. The standard recommendation is k = 100.
    sample_size : int, optional
        Subsample size for LID computation (the descriptor is per-point but
        we report the distribution; full computation is expensive).
    rng : np.random.Generator, optional

    Returns
    -------
    ndarray of LID estimates, one per (subsampled) point.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(vectors)
    if sample_size is not None and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        query = vectors[idx]
    else:
        query = vectors

    knn = NearestNeighbors(n_neighbors=k + 1).fit(vectors)
    dists, _ = knn.kneighbors(query, n_neighbors=k + 1)
    # Drop the self-distance at column 0
    dists = dists[:, 1:]
    d_k = dists[:, -1:]
    log_ratio = np.log(np.clip(dists / d_k, 1e-12, 1.0))
    lid = -1.0 / log_ratio.mean(axis=1)
    return lid


def multifractal_width(
    vectors: np.ndarray,
    q_range: tuple[float, float] = (-5.0, 5.0),
    n_q: int = 21,
    sample_size: int | None = 2000,
) -> float:
    """Width of the multifractal singularity spectrum on the kNN graph distances.

    The descriptor is the difference α_max − α_min of the singularity spectrum
    computed via MFDFA on the sequence of all-pairs distances treated as a
    one-dimensional series. Width close to zero indicates monofractal
    (single scaling regime); wide spectra indicate multifractality
    (mixture of local dimensions).

    Unimplemented in v0.0.1 — wraps the MFDFA library at v0.1.0.
    """
    raise NotImplementedError(
        "multifractal_width awaits the MFDFA-on-graph-distances pipeline planned for v0.1.0."
    )


def hubness(
    vectors: np.ndarray,
    k: int = 10,
    sample_size: int | None = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """Hubness skewness (Radovanović et al., 2010).

    N_k(x) is the number of times x appears among the k-nearest neighbours of
    all other points. The skewness of the N_k distribution measures hubness:
    high positive skew indicates that a small number of points absorb a
    disproportionate share of nearest-neighbour relationships.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    k : int
        Reverse-kNN count parameter. Default 10 matches the original paper.
    sample_size : int, optional
    rng : np.random.Generator, optional

    Returns
    -------
    float
        Skewness of the N_k distribution.
    """
    from scipy.stats import skew

    if rng is None:
        rng = np.random.default_rng(0)
    n = len(vectors)
    if sample_size is not None and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        x = vectors[idx]
    else:
        x = vectors

    n_eff = len(x)
    knn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    _, indices = knn.kneighbors(x, n_neighbors=k + 1)
    # Drop self at column 0
    neighbour_ids = indices[:, 1:].ravel()
    counts = np.bincount(neighbour_ids, minlength=n_eff)
    return float(skew(counts))
