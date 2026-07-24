"""Legacy v0.1 rule diagnostic.

Given a vector dataset, compute the descriptor panel and return a
recommendation among HNSW / IVF / flat-NSW / DiskANN with a predicted
recall-degradation estimate.

This API is preserved for reproducibility. Its index recommendations and
recall-drop values were never calibrated against measured backend outcomes;
the v0.2 governance controller does not use them. Rule sources:

- High-D2 -> flat-NSW: the Hub Highway Hypothesis (2024) argues that the
  HNSW hierarchy stops helping once intrinsic dimension is a substantial
  fraction of the ambient dimension; the flat NSW backbone is sufficient.
- High-hubness -> flat-NSW: Radovanović, M., Nanopoulos, A., Ivanović, M.
  (2010). "Hubs in space: Popular nearest neighbors in high-dimensional
  data." JMLR 11. Heavy hubness already organises the search graph; the
  hierarchy buys little.
- Heterogeneous LID (p95 >> p50) on large data -> DiskANN: Elliott et al.,
  SIGIR 2024 show that LID-sorted insertion shifts HNSW recall up to 12 pp,
  so heterogeneous LID with > 1e6 points motivates DiskANN's tighter graph
  construction discipline.
- Low D2, modest size -> IVF: when intrinsic dimension is low the dataset
  partitions naturally into IVF cells without paying the graph-construction
  cost.
- Else HNSW: the default. Predicted recall drop is a linear ramp on
  lid_p95, calibrated against ANN-benchmarks at v0.2.0.

Confidence is fixed at 0.5 for v0.1.0 because the rules are uncalibrated
heuristics; v0.2.0 will deliver Bayesian posteriors over recall drop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .descriptors import (
    DescriptorPanel,
    correlation_dimension,
    hubness,
    lid_mle,
)
from .geometry import multiscale_lid_dispersion

IndexChoice = Literal["hnsw", "ivf", "flat-nsw", "diskann"]
Workload = Literal["recall@1", "recall@10", "recall@100"]


@dataclass(frozen=True)
class DiagnosticResult:
    """Output of the recommender."""

    descriptors: DescriptorPanel
    recommended_index: IndexChoice
    predicted_recall_drop: float
    confidence: float
    rationale: str


def compute_descriptors(
    vectors: np.ndarray,
    sample_size: int = 2000,
    rng: np.random.Generator | None = None,
    skip_multifractal: bool = False,
    metric: str = "euclidean",
) -> DescriptorPanel:
    """Compute the full descriptor panel from a vector dataset.

    The panel is computed on a subsample of size ``sample_size`` for runtime
    reasons; the descriptors are intrinsic properties and converge in N.
    """
    n, d = vectors.shape
    d2 = correlation_dimension(vectors, sample_size=sample_size, rng=rng, metric=metric)
    lid = lid_mle(vectors, sample_size=sample_size, rng=rng, metric=metric)
    hub = hubness(vectors, sample_size=sample_size, rng=rng, metric=metric)
    mfw = float("nan")
    instability = multiscale_lid_dispersion(
        vectors,
        metric=metric,
        sample_size=min(sample_size, 256),
    )
    return DescriptorPanel(
        correlation_dimension=d2,
        lid_distribution=lid,
        multifractal_width=mfw,
        hubness_skew=hub,
        ambient_dimension=d,
        n_points=n,
        lid_scale_instability=instability,
        metric=metric,
    )


def _recommend(
    panel: DescriptorPanel,
) -> tuple[IndexChoice, float, str]:
    """Rule-based mapping from descriptor panel to (index, predicted_drop, rationale).

    Rule precedence (first match wins):

    1. ``D2 / ambient_d > 0.7`` -> ``flat-nsw``
       Hub Highway Hypothesis (2024): when intrinsic dimension approaches
       ambient dimension the HNSW hierarchy is no longer useful.
    2. ``hubness_skew > 2.0`` -> ``flat-nsw``
       Radovanović et al. (2010): heavy hubness means the flat NSW backbone
       is already organising the search graph.
    3. ``lid_p95 > 2 * lid_p50`` AND ``n_points > 1e6`` -> ``diskann``
       Elliott et al. (SIGIR 2024): heterogeneous LID at scale benefits
       from DiskANN's stricter construction.
    4. ``D2 < 10`` AND ``n_points < 5e4`` -> ``ivf``
       Low intrinsic dimension and *small* cardinality partition cleanly
       into IVF cells. The cutoff was tightened from 1e5 to 5e4 in v0.1.1
       after the v0.1.0 calibration showed 60k MNIST and Fashion-MNIST
       both flipped to IVF, while in practice both are canonical HNSW
       workloads. v0.2.0 will replace this hand-set cutoff with a
       boundary learned from ANN-benchmarks.
    5. else -> ``hnsw`` with predicted recall drop
       ``min(0.3, max(0.0, (lid_p95 - 5) / 50))``. The ramp is a v0.1.x
       heuristic; v0.2.0 will calibrate it against ANN-benchmarks.
    """
    d2 = panel.correlation_dimension
    ambient = panel.ambient_dimension
    n = panel.n_points
    lid_p50 = float(np.quantile(panel.lid_distribution, 0.5))
    lid_p95 = float(np.quantile(panel.lid_distribution, 0.95))
    skew = panel.hubness_skew

    if ambient > 0 and (d2 / ambient) > 0.7:
        return (
            "flat-nsw",
            0.0,
            "D2/ambient > 0.7 (Hub Highway Hypothesis 2024: the HNSW hierarchy "
            "is uninformative when intrinsic dimension approaches ambient).",
        )
    if skew > 2.0:
        return (
            "flat-nsw",
            0.0,
            "Hubness skew > 2.0 (Radovanović et al. 2010: hubs already form a "
            "navigational backbone, so the flat NSW graph suffices).",
        )
    if lid_p95 > 2.0 * lid_p50 and n > 1_000_000:
        return (
            "diskann",
            0.0,
            "Heterogeneous LID (p95 > 2 x p50) at large scale (n > 1e6) "
            "motivates DiskANN's tighter graph construction (Elliott et al. 2024).",
        )
    if d2 < 10.0 and n < 50_000:
        return (
            "ivf",
            0.0,
            "Low intrinsic dimension (D2 < 10) and small cardinality "
            "(n < 5e4) partition naturally into IVF cells.",
        )

    drop = float(min(0.3, max(0.0, (lid_p95 - 5.0) / 50.0)))
    return (
        "hnsw",
        drop,
        f"Default: HNSW. Predicted recall drop {drop:.3f} from a linear ramp "
        "on lid_p95 (uncalibrated v0.1.0 heuristic; v0.2.0 will calibrate against "
        "ANN-benchmarks).",
    )


def diagnose(
    vectors: np.ndarray,
    workload: Workload = "recall@10",
    sample_size: int = 2000,
    metric: str = "euclidean",
) -> DiagnosticResult:
    """Compute descriptors and recommend an index.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    workload : str
        Reserved for v0.2.0 (recall@k may change the recommendation). Not
        used by the v0.1.0 rule set.
    sample_size : int
        Subsample size for the descriptor pipeline. Defaults to 2000.

    Returns
    -------
    DiagnosticResult
        ``recommended_index``, ``predicted_recall_drop``, ``confidence``
        (fixed at 0.5 in v0.1.0 since the rules are uncalibrated), and a
        one-sentence ``rationale``.
    """
    panel = compute_descriptors(vectors, sample_size=sample_size, metric=metric)
    index, drop, rationale = _recommend(panel)
    return DiagnosticResult(
        descriptors=panel,
        recommended_index=index,
        predicted_recall_drop=drop,
        confidence=0.5,
        rationale=rationale,
    )
