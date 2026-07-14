# Changelog

All material changes to this research codebase are recorded here.

## 0.3.0 — 2026-07-13

### Confirmatory study apparatus

- Added a closed study-manifest contract with freeze blockers, artifact pins,
  hardware declarations, receipt templates, and exclusive sealed-run receipts.
- Added corpus normalizers for SciFact, HotpotQA, T2-RAGBench, and qrels-style
  datasets, with document registries and disjoint query-family checks.
- Added explicit evidence bundles, answer outcomes, equal-corpus estimands,
  paired family bootstrap intervals, decision rules, and a joint H2/H3 design-assurance
  contract.
- Added frozen orientation-diagnostic and H2 modeling contracts, train/evaluation partition
  checks, corpus-level predictive metrics, and the prespecified geometry-risk contrast.
- Added the closed orientation-diagnostic plus H2/H3 analysis runner, connected-component split
  audit, label-independent policy-workload contract, and hash-locked Python environment.
- Defined the modeled outcome as intent-to-treat low-effort action failure. Runner failures remain
  failures; successful governed no-result responses remain valid outcomes.
- Moved probe latency and declared work into the system block so H2 measures the increment from
  LID, LID-CV, relative contrast, and radius expansion. One-class corpus outcomes now return a
  conservative failed AUPRC gate instead of aborting the one-shot analysis.
- Required joint design simulation for every H2/H3 success gate and prohibited reserve rescue of
  a consumed confirmatory attempt.

### Governance boundaries

- Bound authorization decisions to policy version, environment, request, and
  document-universe fingerprints before geometry or retrieval work begins.
- Added physically authorized retrieval, counterfactual work accounting, and
  noninterference checks for denied documents.
- Added drift records for corpus, embedding, and policy changes, plus chained
  audit records with pseudonymized subjects.
- Added local artifact verification and custodian-separated labels. Online
  execution emits predictions and the complete raw action panel without protected
  labels; joining occurs only after an externally anchored completion receipt.
- Added typed action-panel admission that derives completed and
  governed-abstention rows from a `GovernedResult` matched to a self-hashed
  audit record. Failed actions carry runner-timed evidence and no invented
  retrieval output.
- Bound each action panel to its controller decisions, authorization context,
  audit-chain head, query-partition audit, and a detached admission receipt.
- Added external protocol-registration admission with bounded HTTPS
  revalidation, exact remote/local byte comparison, pinned OPA bundle revision,
  authenticated remote OPA transport, redirect rejection, and verified audit
  provenance derived from normalized corpora and admitted artifacts.
- Removed the circular manifest-to-generated-artifact hash dependency. The
  outer manifest now pins distinct source inputs, online executions, and sealed
  label payloads; the post-release join checks the exact admitted label objects.
- Added a single-attempt analysis boundary. An exclusive attempt receipt is
  durable before scoring, failed attempts cannot be retried silently, and the
  result has a detached receipt bound to its canonical bytes.
- Production sealed-run admission now reopens and hashes every locally mapped
  artifact, then compares that fresh receipt with the admitted verification
  receipt before creating the run receipt.
- Restricted the optional ANN-benchmark downloader to HTTPS, HTTPS redirects,
  and non-traversing dataset slugs.

### Release boundary

- Exposed an explicit Python API for the protocol, execution, evidence,
  statistics, modeling, audit, integrity, drift, and label-separation layers.
- Synchronized package, citation, and archive metadata at version 0.3.0.
- No sealed confirmatory study was run for version 0.3.0. Existing pilot tables
  remain synthetic mechanism checks, not confirmatory findings.

## 0.2.0 — 2026-07-13

### Retrieval-governance pivot

- Reframed the project around authorization-first, geometry-aware retrieval for RAG systems.
- Added immutable, versioned role policies and exact search over the authorized universe.
- Added physically partitioned HNSW search, query-local multiscale geometry, drift signals, and a fail-closed controller.
- Added counterfactual replay across low-effort HNSW, high-effort HNSW, exact authorized search, and an explicitly unsafe global-search comparator.
- Added synthetic aligned-policy, scrambled-policy, and embedding-drift scenarios with machine-readable reports.
- Added a draft evaluation protocol, threat model, evidence ledger, and revised paper outline.

### Scientific corrections

- Made descriptor distances match the dataset metric. Angular workloads no longer use Euclidean ground truth.
- Retired the former `multifractal_width` estimator. Its output depended on arbitrary row order because pairwise distances were treated as a time series. The compatibility function now warns and returns `NaN`.
- Replaced placeholder index construction and recall routines with working HNSW and exact-recall implementations.
- Removed the quadratic broadcast used by correlation-dimension sampling.

### Release engineering

- Added Python 3.10, 3.12, and 3.14 CI.
- Added a command-line pilot runner and protected local paths for corpora, embeddings, indexes, fitted models, and large experiment runs.
