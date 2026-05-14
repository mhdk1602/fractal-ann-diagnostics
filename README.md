# Fractal ANN Diagnostics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-scaffold-orange.svg)](https://github.com/mhdk1602/fractal-ann-diagnostics)

**Predict approximate-nearest-neighbor (ANN) index failure modes from intrinsic-dimension descriptors.**

Given a vector dataset, this tool computes a panel of fractal and intrinsic-dimension descriptors and returns a workload-classed recommendation among **HNSW**, **IVF**, **flat-NSW**, and **DiskANN** — with a calibrated estimate of recall degradation under default parameters. The goal is not to invent a new index; it is to tell a practitioner *which existing index to use, and what to expect from it*, before they spend a week tuning the wrong one.

> **Status.** Pre-alpha scaffold. The descriptor panel has a reference implementation; the recommender, the workload classifier, and the ANN-benchmarks evaluation pipeline are stubs awaiting v0.1.0.

**Author:** [Dineshkumar Malempati Hari](https://orcid.org/0009-0003-1036-9477).

## Position in the literature

The closest related work has been published in 2024-2026:

- [Elliott et al., SIGIR 2024](https://arxiv.org/abs/2405.17813) — links HNSW recall to LID and insertion order; up to 12 pp recall shift from LID-sorted insertion.
- **Dual-Branch HNSW + LID** (Nguyen et al., 2025) — uses LID values to drive HNSW construction.
- [MCGI](https://arxiv.org/abs/2601.01930) (2026) — manifold-consistent graph indexing that modulates beam search budget based on in-situ LID analysis.
- **Hub Highway Hypothesis** (2024) — challenges whether HNSW's hierarchy even helps in high-dimensional regimes.

This project does **not** propose a new index variant. It is the diagnostic layer that sits *before* the index choice — it engages MCGI rather than competes with it. The same LID infrastructure modern variants use internally is exposed here as a workload-time decision aid.

## Descriptor panel

| Descriptor | Reference | What it predicts |
|---|---|---|
| Correlation dimension D₂ (Grassberger-Procaccia 1983) | `descriptors.correlation_dimension` | Effective intrinsic dimension; relative to ambient dimension, governs HNSW recall (Faloutsos & Kamel, 1994) |
| Local intrinsic dimensionality, MLE estimator | `descriptors.lid_mle` | Per-point hardness; high-LID points are search dead-ends in HNSW (Elliott et al., 2024) |
| Multifractal spectrum width on graph distances | `descriptors.multifractal_width` | Heterogeneity of local dimensions; predicts whether the dataset has a single global scaling or a mixture |
| Hubness ratio (Radovanović et al., 2010) | `descriptors.hubness` | High hubness indicates a flat-NSW backbone is forming; supports the Hub Highway Hypothesis |

## Quickstart (intended)

```python
from fractal_ann_diagnostics import diagnose

# vectors: np.ndarray of shape (n, d)
result = diagnose(vectors, workload="recall@10")
print(result.recommended_index)        # e.g., "flat-nsw"
print(result.predicted_recall_drop)    # e.g., 0.08 (expected recall degradation under defaults)
print(result.descriptors.correlation_dimension)
print(result.descriptors.lid_distribution.quantile(0.95))
```

## Repository structure

```
fractal-ann-diagnostics/
├── src/fractal_ann_diagnostics/
│   ├── descriptors.py       # D2, LID, multifractal width, hubness
│   ├── models.py            # Thin wrappers around hnswlib, faiss, etc.
│   ├── diagnostic.py        # The recommender
│   ├── benchmark.py         # ANN-benchmarks corpus harness
│   └── io.py                # Vector dataset loaders (hdf5, fvecs)
├── examples/
│   └── quickstart.py
├── tests/
│   └── test_smoke.py
├── research/paper/
│   └── outline.md
├── pyproject.toml
├── CITATION.cff
├── .zenodo.json
└── LICENSE                  # MIT
```

## Roadmap

| Version | Scope |
|---|---|
| v0.0.1 (this) | Scaffolding, descriptor reference implementations, project structure |
| v0.1.0 | Working recommender on synthetic mixtures with known intrinsic dimension; ANN-benchmarks integration; first paper figures |
| v0.2.0 | Calibration on the full ANN-benchmarks corpus; recall-degradation predictions reported as Bayesian posteriors |
| v1.0.0 | Paper-ready: target SISAP 2027 short paper or SIGIR 2027 short paper |

## Connection to the wider program

This is the **(b)** track of the fractal-indexing research program ([master plan](https://github.com/mhdk1602/hurst-aware-partitioning), [H2 pre-registration](https://doi.org/10.5281/zenodo.20188013)). It is the H3 pivot: the original H3 hypothesis ("intrinsic-dimension-driven HNSW construction is novel") was substantially preempted by Elliott 2024, Dual-Branch HNSW 2025, and MCGI 2026. The diagnostics framing reframes the contribution as a *complementary utility*, not a *competing index*.

## License

[MIT](./LICENSE).
