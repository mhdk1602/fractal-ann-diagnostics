# Paper outline

**Proposed confirmatory title:** Adaptive Policy-Aware Vector Retrieval Under Corpus, Embedding, and
Authorization Drift

**Conditional title:** Fractal Risk Control for Policy-Aware RAG

The conditional title is permitted only if the H2+H3 primary intersection passes. That includes
H2's four-of-five corpus point rule, every H2 directional bound, and every H3 cost, fidelity, tail,
and authorization condition. H1 is a descriptive orientation diagnostic and has no title-selection
role. Otherwise the paper uses a neutral title and reports the failed gate without retuning it.

## Claim hierarchy

1. Authorization is deterministic. The controller cannot grant access.
2. Exact top-k inside the live authorized set supplies retrieval truth.
3. Query-local multiscale geometry means LID at k=50, LID-CV, relative contrast, and radius
   expansion. Probe latency and work are system telemetry, not geometry.
4. The modeling target is intent-to-treat low-effort action failure: a non-completed action or a
   completed action with authorized recall below 0.90. It is not a pure ANN-failure label.
5. Adaptive value requires an equal-corpus mean family-level relative latency reduction above 10%
   at noninferior retrieval target attainment and complete-evidence sufficiency.
6. Zero emitted entitlement violations is reported with a family-level upper confidence bound.
7. Answer correctness, faithfulness, and extraction resistance are outside the primary study.

## Sections

### 1. Problem

Approximate vector search is often joined to live, heterogeneous IAM. A globally accurate index
can still fail an authorized query when permitted evidence occupies a sparse or difficult subset.
Retrieve-then-filter can also expose denied material before the filter acts.

### 2. Prior work and novelty boundary

Position the paper after Authorization-First Retrieval, Permission-Aware RAG, Filtered-DiskANN,
ACORN, Global-Local Selectivity, Ada-ef, Fiber-Navigable Search, and retrieval-extraction attacks.
State what each already resolves. Do not claim the first permission-aware or geometry-aware
filtered retrieval system.

### 3. Formal contract

Define the authorized universe, exact authorized neighbor truth, complete-evidence sufficiency,
structural entitlement violation, measured request latency, action regret, and point-of-emission
authorization. State that downstream retention after revocation lies outside this API.

### 4. Reference architecture

Describe request-bound OPA decisions, authorized-only exact and HNSW indexes, bounded query probe,
frozen controller, final authorization, pseudonymous audit chain, external protocol registration,
custodian split, typed governed-result/audit admission, anchored pre-label action panel, and offline
post-receipt label join.

### 5. Candidate geometric mechanism

Define LID, cross-scale instability, relative contrast, radius expansion, and offline
vector-policy correlation. Document metric handling and permutation tests. Retain the rejected
MFDFA statistic as an integrity case study.

### 6. Fixed benchmark

Describe SciFact, HotpotQA FullWiki with a separately acquired corpus, T2-RAGBench, BRIGHT, and the
fixed MIRACL transfer slice. Record input and label artifacts separately. Include policy
revisions, embedding revision, exact truth, action grid, drift interventions, hardware, warmups,
licenses, hashes, the connected-component partition audit, and the five label-independent policy
workloads.

### 7. Prespecified analysis

Report in this order:

1. H1 diagnostic: label-free frozen full-model high-minus-low geometry-profile predictive-risk
   contrast, reported without a primary success decision.
2. H2: held-out log-loss, Brier-score, and AUPRC gain for `full` versus `system-policy`, including
   the four-of-five corpus point rule. `system-policy` contains probe telemetry; `full` adds only the
   four geometric descriptors.
3. H3: paired adaptive-versus-static equal-corpus mean family-relative request-latency reduction,
   retrieval-target attainment, complete-evidence sufficiency, the equal-corpus mean of
   within-corpus p95 ratios of family-mean latency, and entitlement violations.

Show corpus-specific estimates before the equal-corpus aggregate. Query family is the resampling
unit. Corpora are fixed strata, not sampled clusters.

### 8. Controller results

Compare the adaptive policy with frozen low-effort, high-effort, and exact actions. Report every
counterfactual action, exact fallback, index-refresh cost, probe cost, abstention, and regret. Keep
configured `efSearch` separate from observed work counters.

### 9. Drift and transfer

Separate corpus, embedding, and policy drift. Revision tuples must be internally consistent.
Failed transfer stays visible and limits the claim.

### 10. Limitations

Cover generated rather than enterprise IAM, fixed-corpus inference, backend specificity, timing
sensitivity, authorization-oracle assumptions, evaluator error, label-custodian trust, finite
zero-event bounds, the absence of external attestation from ordinary Python objects, the external
anchors required by the receipt and audit chain, and the rule that a technical failure ends v0.3
rather than releasing a confirmatory reserve.

## Planned figures

1. Request-bound authorization and retrieval boundary.
2. Query geometry under aligned and fragmented policy subsets.
3. Low-effort failure by LID, instability, and allow-rate strata.
4. Model-block comparison on each sealed corpus.
5. Paired request-latency and retrieval-quality frontier.
6. Controller regret against the per-query action oracle.
7. Corpus, embedding, and policy drift trajectories.
8. Entitlement and complete-evidence outcome matrix with exact upper bounds.

## Release rule

Release the protocol, code, artifact declarations, exclusions, action matrix, drift traces, and
negative results when any primary gate fails. The title and abstract follow the prespecified gates.
The data and exclusions do not follow the preferred story.
