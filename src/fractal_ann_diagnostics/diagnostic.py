"""Diagnostic recommender — the public face of the package.

Given a vector dataset, compute the descriptor panel, classify the workload
regime, and return a recommendation among HNSW / IVF / flat-NSW / DiskANN
with a predicted recall-degradation estimate.

The recommender logic is unimplemented in v0.0.1; only the interface and the
descriptor pipeline are wired up.
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
) -> DescriptorPanel:
    """Compute the full descriptor panel from a vector dataset.

    The panel is computed on a subsample of size ``sample_size`` for runtime
    reasons; the descriptors are intrinsic properties and converge in N.
    """
    n, d = vectors.shape
    d2 = correlation_dimension(vectors, sample_size=sample_size, rng=rng)
    lid = lid_mle(vectors, sample_size=sample_size, rng=rng)
    hub = hubness(vectors, sample_size=sample_size, rng=rng)
    # multifractal_width pending v0.1.0
    return DescriptorPanel(
        correlation_dimension=d2,
        lid_distribution=lid,
        multifractal_width=float("nan"),
        hubness_skew=hub,
        ambient_dimension=d,
        n_points=n,
    )


def diagnose(
    vectors: np.ndarray,
    workload: Workload = "recall@10",
    sample_size: int = 2000,
) -> DiagnosticResult:
    """Compute descriptors and recommend an index.

    The recommender logic (mapping descriptors -> index choice and recall
    prediction) is unimplemented in v0.0.1; this function computes the
    descriptors and raises NotImplementedError on the recommendation step.
    """
    panel = compute_descriptors(vectors, sample_size=sample_size)
    raise NotImplementedError(
        "Recommender mapping descriptors -> index choice is unimplemented in v0.0.1. "
        "The descriptors themselves are computed:\n"
        f"  correlation_dimension = {panel.correlation_dimension:.3f}\n"
        f"  ambient_dimension     = {panel.ambient_dimension}\n"
        f"  lid p95               = {np.quantile(panel.lid_distribution, 0.95):.3f}\n"
        f"  hubness_skew          = {panel.hubness_skew:.3f}\n"
        "Calibration on the ANN-benchmarks corpus lands at v0.1.0."
    )
