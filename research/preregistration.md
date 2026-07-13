# Adaptive policy-aware vector retrieval under drift

**Protocol version:** 0.2.0-draft<br>
**Protocol date:** 2026-07-13<br>
**Status:** Design and synthetic development pilot. No confirmatory corpus has been opened.<br>
**Sole author:** [mhdk1602](https://github.com/mhdk1602)

## Research question

Can query-local intrinsic geometry predict failure in policy-constrained approximate retrieval,
and can an adaptive controller meet a locked evidence-recall target at lower serving cost than a
cost-matched static action?

The controller never decides authorization. A live policy decision point defines the authorized
universe before a query-specific descriptor, index, reranker, logger, or generator can inspect a
document. Geometry controls effort inside that universe.

This ordering follows the noninterference criterion in
[Authorization-First Retrieval](https://aclanthology.org/2026.trustnlp-main.15.pdf), the subject,
object, action, and environment model in
[NIST SP 800-162](https://doi.org/10.6028/NIST.SP.800-162), and the resource-access ordering in
[NIST SP 800-207](https://doi.org/10.6028/NIST.SP.800-207).

## Novelty boundary

This study does not claim to invent permission-aware RAG, filtered ANN, query-local search
difficulty, or adaptive `efSearch`.

- Permission-Aware RAG validates access through provider-native IAM endpoints
  ([Jeong and Lee, 2025](https://doi.org/10.1109/ACCESS.2025.3628960)).
- Authorization-First Retrieval formalizes authorization before retrieval and reports structural
  leakage for retrieve-then-filter baselines
  ([TrustNLP 2026](https://aclanthology.org/2026.trustnlp-main.15.pdf)).
- Filtered-DiskANN and ACORN perform predicate-constrained graph search
  ([Gollapudi et al., 2023](https://doi.org/10.1145/3543507.3583552);
  [Patel et al., 2024](https://doi.org/10.1145/3654923)).
- Global-Local Selectivity already describes vector-filter correlation and filtered-query hardness
  ([Amanbayev et al., 2026](https://arxiv.org/abs/2602.11443)).
- Ada-ef selects per-query HNSW effort for a requested recall target
  ([PACMMOD 2026](https://doi.org/10.1145/3786639)).
- Fiber-Navigable Search already uses local geometry to diagnose filtered-graph failure regimes
  ([Dang, 2026](https://arxiv.org/abs/2604.00102)).

The tested contribution is narrower: authorization-first exact truth, query-local multiscale
geometry inside the authorized universe, counterfactual replay of every action, three drift
families, and a cost-constrained fail-closed controller.

## System contract

For trial \(i=(q,u,t)\), let:

- \(C_t\) be the authoritative corpus at time \(t\).
- \(m_t\) be the approved embedding model and revision.
- \(P_t(u,d,a,e)\in\{0,1\}\) be the live authorization decision.
- \(C_{u,t}=\{d\in C_t:P_t(u,d,a,e)=1\}\) be the authorized universe.
- \(G_i^K\) be exact current top-\(K\) retrieval over \(C_{u,t}\).
- \(S_i^K(a)\) be the result returned by action \(a\).

The exact authorized reference is

\[
G_i^K=\operatorname{TopK}_{d\in C_{u,t}}
\operatorname{sim}(e_{m_t}(q),e_{m_t}(d)).
\]

ANN fidelity is

\[
R_i(a)=\frac{|S_i^K(a)\cap G_i^K|}{|G_i^K|}.
\]

A structural entitlement violation is

\[
U_i(a)=\mathbf{1}\left[
\exists d\in \operatorname{learned\_context}_i(a):P_t(u,d,a,e)=0
\right].
\]

The security requirement is exact: \(\sum_i U_i(a)=0\) for every proposed action. Zero observed
violations does not prove safety outside the tested systems; the report will include an exact
one-sided binomial upper bound.

## Evidence outcomes

For corpora with annotated evidence bundles, let \(\mathcal E_q\) contain every accepted
alternative evidence set. An authorized solution exists when at least one complete evidence set is
authorized. Returned evidence is sufficient only when one complete authorized set is present.

False permits and false denials remain separate:

- **False permit:** answer when no authorized complete evidence set was retrieved.
- **False denial:** abstain when the selected action contained a complete authorized evidence set.

For corpora without evidence bundles, the study reports authorized recall, nDCG, and rank error.
It will not infer answer sufficiency from nearest-neighbor recall alone.

## Confirmatory hypotheses

### H1: multiscale geometric mechanism

For the default low-effort authorized HNSW action, failure risk increases with query-local LID and
cross-scale LID instability after conditioning on corpus, backend, allow rate, search budget,
dimension, and drift severity.

The primary coefficient is the interaction between standardized LID at \(k=50\) and standardized
cross-scale instability over \(k\in\{10,20,50,100\}\). A hierarchical logistic model contains
corpus and backend varying intercepts.

### H2: incremental predictive value

Models are compared on untouched corpora and policy seeds:

1. System-only: corpus size, dimension, backend, effort, version lag, and drift.
2. Policy-only: system model plus global allow rate and predicate complexity.
3. Geometry-only: system model plus the preregistered geometry panel.
4. Full: system, policy, and geometry.

The full model passes the fractal gate only if all of the following hold:

- Paired 95% interval for held-out log-loss improvement over policy-only excludes zero.
- Relative Brier-score reduction is at least 5%.
- AUPRC improves by at least 0.02.
- Improvement has the same direction in at least four of five sealed corpora.

If policy-only matches the full model, “fractal” is removed from the paper title.

### H3: controller value

Every executable action is replayed for every frozen trial. The comparison therefore uses observed
counterfactual action outcomes rather than propensity weighting.

The static comparator is the best single action chosen on development data under the same mean
compute budget. The controller passes only if it:

- reduces evidence-policy violations by at least 20% relative and one percentage point absolute;
- has a paired 95% interval excluding zero;
- loses no more than one percentage point of answer coverage;
- stays within the locked mean-compute allowance;
- increases p95 latency by no more than 25%; and
- produces zero entitlement violations.

### H4: drift transfer

H3 is repeated separately under corpus, embedding, and policy drift. Combined drift is a declared
stress test, not a primary endpoint.

## Data plan

The development tier will use MS MARCO, Natural Questions, FiQA, NFCorpus, TREC-COVID, and
MultiHop-RAG. The sealed tier will use SciFact, HotpotQA, BRIGHT, T²-RAGBench, and one untouched
large retrieval shard.

[BEIR](https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html)
supplies heterogeneous retrieval tasks.
[SciFact](https://arxiv.org/abs/2004.14974) and
[HotpotQA](https://aclanthology.org/D18-1259/) provide explicit evidence.
[BRIGHT](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7a0f8055c838df8e62329a76c7c6403d-Abstract-Conference.html)
contains reasoning-intensive retrieval queries, while
[T²-RAGBench](https://aclanthology.org/2026.eacl-long.8/) adds text-and-table evidence.

The sealed suite must contain at least 10,000 distinct queries and 1,000 low-effort failures.
Multiple user-policy draws attached to one query are nested replicates, not independent queries.

Corpora will be fetched through versioned manifests. Each record will include source URL, license,
content hash, embedding model revision, chunking revision, and exclusion reason. Restricted corpora
will not be redistributed.

## Authorization workload

The reference evaluator implements RBAC and ABAC fields for tenant, project, role, clearance,
region, purpose, classification, embargo, and temporary-grant expiry.

Three fixed generators are used:

1. Independent ACL labels as a negative control.
2. Cluster-aligned ACLs at locked correlation strengths.
3. Evidence-boundary cases where complete evidence is authorized, partial, or unavailable.

Global allow rates are \(\{0.001,0.01,0.05,0.20,0.50\}\). Reports use “allow rate” rather than
“selectivity” where ambiguity would reverse the interpretation.

## Action set

- Low-effort authorization-first HNSW.
- Two-times and four-times HNSW effort.
- Filter-native traversal using ACORN or Filtered-DiskANN.
- Exact scan over the authorized subset.
- Authorized sparse+dense fusion followed by an authorized reranker.
- Abstention or human review.

The current v0.2 reference implementation contains low/high HNSW, exact authorized search, and
abstention. The other actions remain out of the confirmatory protocol until their conformance tests
pass.

No reranker or generator may inspect a denied vector, text, identifier, or derived query-specific
summary. Geometry predicts retrieval failure; it never predicts permission.

## Query-local features

### Online controller panel

- LID MLE at \(k=50\), with sensitivity values at \(k=20\) and \(k=100\).
- Cross-scale LID coefficient of variation.
- Relative contrast and rank-distance curvature.
- Neighbor-radius expansion.
- Authorized-universe size and allow rate.
- Policy version, version lag, and churn.
- Corpus-index lag and embedding-revision mixture.
- Authorized pilot-search yield and early termination telemetry where the backend exposes them.

Every online geometric feature is computed only after authorization and only within the authorized
universe.

### Offline mechanism panel

The offline analysis may measure Global-Local Selectivity and vector-policy correlation over an
isolated research copy. These variables are used to explain failure strata, not as production
controller inputs. This separation prevents a learned component from inspecting denied vectors.

### Rejected feature

The v0.1 MFDFA feature converted an arbitrarily ordered upper triangle of pairwise distances into a
time series. Row permutations changed the estimate while leaving the point cloud unchanged. It is
retired before confirmation and replaced with multiscale neighbor-radius statistics.

All descriptors face 100 row permutations, sample-size sensitivity tests, `NaN` accounting, and
metric-conformance tests before sealed data are opened.

## Drift interventions

### Corpus drift

- Append 1%, 5%, 10%, and 20% new documents.
- Delete or supersede 1%, 5%, and 10%.
- Add duplicate topic bursts.
- Delay ingestion by one and five update batches.
- Rechunk a fixed fraction without changing source truth.

### Embedding drift

Use three pinned open embedding models. Evaluate old/old, new-query/old-corpus, 25%, 50%, and 75%
migration, and fully rebuilt current embeddings. Exact truth always uses the fully approved model.
Stale target embeddings are a documented dense-retrieval problem
([Monath et al., 2024](https://proceedings.mlr.press/v235/monath24a.html)).

### Policy drift

Apply grants, revocations, role transfers, reclassification, region changes, and embargo expiry to
1%, 5%, and 20% of eligible records. Test fresh policy metadata, one-revision lag, five-revision lag,
live final validation, and policy-engine failure. Policy-engine failure must abstain.

## Splits and freezing

Four disjoint stages are used:

1. Engineering split for backend correctness.
2. Development corpora for feature and action design.
3. Calibration split for risk thresholds and compute allowance.
4. Sealed corpora, model migrations, policy seeds, and future snapshots.

No source document, paraphrased query, user, topic family, policy seed, or embedding migration pair
may cross these stages. Hyperparameter changes stop before sealed hashes are unlocked.

## Statistical analysis

- All actions are paired within query.
- Query families are the bootstrap unit; users, policies, seeds, and actions stay attached.
- Corpus and backend receive varying intercepts; LID and allow-rate effects may vary by corpus.
- Primary controller intervals use 10,000 paired cluster bootstrap replicates.
- Results are shown per corpus before any pooled estimate.
- Ordered gatekeeping is H1, then H2, then H3. Secondary ablations use Holm correction within family.
- Crashes count as service failures and remain separately visible in retrieval-quality tables.
- A backend-condition cell is invalid if more than 1% of trials are missing for non-policy reasons.

## Required ablations

- System-only, policy-only, geometry-only, and full models.
- LID alone, rank-distance features alone, and no drift/version features.
- Oracle geometry versus online authorized-pilot geometry.
- No exact-search action, no abstention, and one action removed at a time.
- Held-out backend and held-out embedding family.
- Independent versus cluster-aligned policies.
- ANN fidelity versus annotated evidence sufficiency.
- Retired MFDFA permutation test as a recorded negative control.

## Decision gates

**Security gate:** any denied item reaching a learned component blocks a safety claim.

**Fractal gate:** H1 and every H2 minimum-effect condition must pass. Otherwise the paper is framed
as policy-aware retrieval without a fractal claim.

**Controller gate:** H3 must beat a cost-matched static action. Predictability without better action
selection is not a controller result.

**Drift gate:** the controller advantage must reproduce under each single-drift family. Static-only
success does not support migration safety.

**Publication gate:** a full null still releases the benchmark, action-outcome matrix, drift traces,
backend conformance tests, and preregistered analysis.

## Interpretation of null outcomes

- No H1: the proposed geometric mechanism is unsupported.
- H1 passes but H2 fails: the association is too weak or unstable for prediction.
- H2 passes but H3 fails: failure is predictable but the action policy adds no cost-adjusted value.
- Policy-only equals full: remove the fractal claim.
- Only synthetic ACLs work: report a controlled stress test, not enterprise prevalence.
- One backend works: report an implementation-specific effect.
- Static success but drift failure: the controller is not migration-safe.
- Retrieval improves but answer evidence does not: claim retrieval control only.
- Any entitlement violation: reject the safe-RAG claim.

## Development pilot boundary

`experiments/run_governance_pilot.py` is an engineering and mechanism pilot. Its synthetic mixtures
test metric handling, authorization-first candidate construction, exact authorized truth, action
replay, audit records, and the unsafe comparator. Pilot outcomes cannot confirm H1–H4 and will not
be pooled with sealed results. The v0.2 rule thresholds were adjusted on this synthetic tier so the
pilot exercises low-effort, widened, and exact actions. That controller is an executable reference,
not a confirmatory estimator. Confirmatory thresholds require the disjoint calibration split above.
