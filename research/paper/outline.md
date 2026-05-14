# Paper Outline (Post-Empirical)

**Working title:** Fractal Diagnostics for Approximate Nearest Neighbor Index Selection.

**Target venues:** SISAP 2027 short paper (primary) or SIGIR 2027 short paper (alternative).

**Status:** Outline only. Empirical work begins at v0.1.0 of the package; the paper is drafted after calibration on the ANN-benchmarks corpus.

## Section plan

1. **Introduction.** Practitioners pick HNSW by default and discover recall problems only after deployment. The literature documents *why* (intrinsic dimensionality, hubness, manifold mismatch) but does not give them a workload-time diagnostic. We provide one.
2. **Background.** HNSW (Malkov & Yashunin 2018); LID (Houle 2017, Amsaleg et al. 2015); correlation fractal dimension (Belussi & Faloutsos 1995); hubness (Radovanović et al. 2010); recent LID-aware variants (Elliott et al. SIGIR 2024, Dual-Branch HNSW 2025, MCGI 2026); Hub Highway Hypothesis (2024).
3. **Descriptor panel.** Definitions, estimators, finite-sample properties. Why four descriptors and not one: each captures a distinct failure mode (D₂ for ambient/intrinsic mismatch, LID for hard-point distribution, multifractal width for mixture detection, hubness for emergent flat-NSW backbone).
4. **Recommender.** From descriptor panel to index choice. Two flavours considered: (i) rule-based, derived from theory, (ii) calibrated, trained on the ANN-benchmarks corpus. Section reports both.
5. **Calibration.** ANN-benchmarks corpus, descriptor panel for each dataset, recall under each of {HNSW default, HNSW tuned, IVF default, flat-NSW, DiskANN}.
6. **Evaluation.** Held-out datasets (or held-out splits). The diagnostic's selection vs. an oracle that knows the best index post hoc. Reports the recall gap.
7. **Discussion.** Where the diagnostic fails, what it does not predict (latency, memory), how to combine with cost-based query planning.
8. **Limitations.** Static dataset assumption; the diagnostic is computed once at index build time, not refreshed as data drifts.
9. **Reproducibility statement.** Code, ANN-benchmarks HDF5 hashes, seeds, fixed splits.

## Figures (planned)

- F1. Descriptor panel for each canonical ANN-benchmarks dataset, scatter plot in (D₂ / ambient_d, LID p95) space, coloured by best-index-post-hoc.
- F2. Per-dataset recall comparison: HNSW default vs. diagnostic recommendation.
- F3. Confusion matrix of recommendation vs. oracle.
- F4. Calibration curve of predicted-recall-drop vs. observed.
- F5. Ablation: each descriptor removed singly.

## Pre-commitment statement (informal, not formal pre-registration)

The descriptor panel is fixed at v0.0.1. Adding descriptors after seeing calibration results would inflate Type-I error. If the rule-based recommender fails calibration, we report it as a null result rather than search for a recommender that passes.
