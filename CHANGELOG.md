# Changelog

All material changes to this research codebase are recorded here.

## 0.2.0 — 2026-07-13

### Retrieval-governance pivot

- Reframed the project around authorization-first, geometry-aware retrieval for RAG systems.
- Added immutable, versioned role policies and exact search over the authorized universe.
- Added physically partitioned HNSW search, query-local multiscale geometry, drift signals, and a fail-closed controller.
- Added counterfactual replay across low-effort HNSW, high-effort HNSW, exact authorized search, and an explicitly unsafe global-search comparator.
- Added synthetic aligned-policy, scrambled-policy, and embedding-drift scenarios with machine-readable reports.
- Added a preregistered evaluation protocol, threat model, evidence ledger, and revised paper outline.

### Scientific corrections

- Made descriptor distances match the dataset metric. Angular workloads no longer use Euclidean ground truth.
- Retired the former `multifractal_width` estimator. Its output depended on arbitrary row order because pairwise distances were treated as a time series. The compatibility function now warns and returns `NaN`.
- Replaced placeholder index construction and recall routines with working HNSW and exact-recall implementations.
- Removed the quadratic broadcast used by correlation-dimension sampling.

### Release engineering

- Added Python 3.10, 3.12, and 3.14 CI.
- Added a command-line pilot runner and protected local paths for corpora, embeddings, indexes, fitted models, and large experiment runs.
