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

import warnings
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
    lid_scale_instability: float = float("nan")
    metric: str = "euclidean"


def _validate_vectors(vectors: np.ndarray, *, minimum: int = 4) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("vectors must have shape (n_points, dimension)")
    if len(matrix) < minimum:
        raise ValueError(f"at least {minimum} vectors are required")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("vectors contain non-finite values")
    return matrix


def _sample_pair_distances(
    vectors: np.ndarray,
    *,
    metric: str,
    max_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample unordered pairs without allocating an (n, n, d) tensor."""
    n = len(vectors)
    total_pairs = n * (n - 1) // 2
    count = min(max_pairs, total_pairs)
    if count == total_pairs and n <= 1200:
        from scipy.spatial.distance import pdist

        sklearn_metric = "cosine" if metric == "cosine" else "euclidean"
        return pdist(vectors, metric=sklearn_metric)

    left = rng.integers(0, n, size=count * 2)
    right = rng.integers(0, n, size=count * 2)
    valid = left != right
    left, right = left[valid][:count], right[valid][:count]
    while len(left) < count:
        extra_left = rng.integers(0, n, size=count - len(left))
        extra_right = rng.integers(0, n, size=count - len(left))
        valid = extra_left != extra_right
        left = np.concatenate([left, extra_left[valid]])[:count]
        right = np.concatenate([right, extra_right[valid]])[:count]

    if metric == "euclidean":
        delta = vectors[left] - vectors[right]
        return np.sqrt(np.einsum("ij,ij->i", delta, delta))
    if metric == "cosine":
        a, b = vectors[left], vectors[right]
        denominator = np.clip(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12, None)
        return 1.0 - np.einsum("ij,ij->i", a, b) / denominator
    raise ValueError(f"unsupported metric: {metric!r}")


def correlation_dimension(
    vectors: np.ndarray,
    n_scales: int = 16,
    sample_size: int | None = 2000,
    rng: np.random.Generator | None = None,
    metric: str = "euclidean",
    max_pairs: int = 250_000,
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
    matrix = _validate_vectors(vectors)
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(matrix)
    if sample_size is not None and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        x = matrix[idx]
    else:
        x = matrix

    pair_dists = _sample_pair_distances(x, metric=metric, max_pairs=max_pairs, rng=rng)
    pair_dists = pair_dists[pair_dists > 0]

    r_min = np.quantile(pair_dists, 0.02)
    r_max = np.quantile(pair_dists, 0.5)
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_min <= 0 or r_max <= r_min:
        return float("nan")
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
    metric: str = "euclidean",
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
    matrix = _validate_vectors(vectors)
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(matrix)
    if sample_size is not None and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        query = matrix[idx]
    else:
        query = matrix

    use_k = min(k, n - 1)
    knn = NearestNeighbors(n_neighbors=use_k + 1, metric=metric, algorithm="brute").fit(matrix)
    dists, _ = knn.kneighbors(query, n_neighbors=use_k + 1)
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
    rng: np.random.Generator | None = None,
) -> float:
    """Retired non-invariant descriptor, retained only for API compatibility.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    q_range : tuple of float
        Inclusive (q_min, q_max) range of fractal exponents to sweep. q = 0
        is dropped by the MFDFA library because the estimator diverges there.
    n_q : int
        Number of q points across q_range.
    sample_size : int, optional
        Subsample size for the underlying pairwise distance series; full
        all-pairs distances on n = 1e5 are infeasible. The descriptor is an
        intrinsic property and converges in N.
    rng : np.random.Generator, optional

    Returns
    -------
    float
        Always NaN. Row permutations changed the old estimate even though the
        point cloud was unchanged. Use ``multiscale_lid_dispersion`` instead.

    References
    ----------
    Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin, S.,
    Bunde, A., Stanley, H. E. (2002). Multifractal detrended fluctuation
    analysis of nonstationary time series. Physica A, 316(1-4), 87–114.
    """
    del vectors, q_range, n_q, sample_size, rng
    warnings.warn(
        "multifractal_width was retired in v0.2.0 because it is not permutation-invariant; "
        "use geometry.multiscale_lid_dispersion",
        DeprecationWarning,
        stacklevel=2,
    )
    return float("nan")


def hubness(
    vectors: np.ndarray,
    k: int = 10,
    sample_size: int | None = 2000,
    rng: np.random.Generator | None = None,
    metric: str = "euclidean",
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

    matrix = _validate_vectors(vectors)
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(matrix)
    if sample_size is not None and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        x = matrix[idx]
    else:
        x = matrix

    n_eff = len(x)
    use_k = min(k, n_eff - 1)
    knn = NearestNeighbors(n_neighbors=use_k + 1, metric=metric, algorithm="brute").fit(x)
    _, indices = knn.kneighbors(x, n_neighbors=use_k + 1)
    # Drop self at column 0
    neighbour_ids = indices[:, 1:].ravel()
    counts = np.bincount(neighbour_ids, minlength=n_eff)
    return float(skew(counts))
