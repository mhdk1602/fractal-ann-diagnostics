# Adaptive authorization-first vector retrieval under drift

**Protocol version:** 0.3.0-draft<br>
**Protocol date:** 2026-07-13<br>
**Status:** Protocol and calibration work only. The study manifest is not frozen, and no v0.3
sealed run is permitted.<br>
**Claim scope:** Suite-conditional retrieval control<br>
**Sole author:** [mhdk1602](https://github.com/mhdk1602)

## Primary claim to be registered

The claim recorded in the [study manifest](study-manifest.json) is:

> On the fixed five-corpus suite, a frozen full model that adds LID at k=50, LID-CV, relative
> contrast, and radius expansion to the frozen system-policy baseline improves held-out prediction
> of intent-to-treat low-effort action failure beyond the frozen H2 thresholds; and a frozen adaptive
> controller achieves an equal-corpus mean family-level relative end-to-end request-latency
> reduction greater than 10% relative to a frozen static action while authorized retrieval-target
> attainment and complete-evidence sufficiency remain noninferior within one percentage point, the
> equal-corpus mean of within-corpus proposed-to-comparator p95 ratios of family-mean end-to-end
> request latency remains below 1.25, and no denied item is emitted at the controlled retrieval
> boundary.

The population is the fixed corpus suite, pinned workload, pinned embedding revisions, tested
backends, and frozen policy contract. Equal weighting across five named corpora does not turn
the suite into a random sample of organizations, policies, or retrieval systems.

Answer emission, answer coverage, false permit, and false denial remain available as evaluation
outputs. They are declared secondary analyses, with no role in the primary claim or success
gate. Any later confirmatory answer claim requires its own frozen answer/refusal policy, estimand,
margin, and analysis rule before sealed execution and any answer-label join used for analysis.

The prospective ordering between model construction and scoring of sealed-stage payloads is
motivated by [Nosek et al.](https://doi.org/10.1073/pnas.1708274114). It is a process-order claim,
not evidence that the observations were unseen to the operator: benchmark labels are publicly
accessible, and the registered component graph uses qrel-derived positive-relevance edges.
Artifact disclosure and executable reporting follow the practices described by
[Pineau et al.](https://www.jmlr.org/papers/v22/20-303.html).

## Prior work and tested conjunction

This protocol does not claim invention of permission-aware RAG, authorization-before-retrieval,
filtered ANN, query-local difficulty, or adaptive HNSW effort.

- Permission-Aware RAG validates access through provider-native IAM endpoints
  ([Jeong and Lee, 2025](https://doi.org/10.1109/ACCESS.2025.3628960)).
- Authorization-First Retrieval formalizes authorization before retrieval and reports structural
  leakage in retrieve-then-filter baselines
  ([TrustNLP 2026](https://aclanthology.org/2026.trustnlp-main.15.pdf)).
- Filtered-DiskANN and ACORN study predicate-constrained graph search
  ([Gollapudi et al., 2023](https://doi.org/10.1145/3543507.3583552);
  [Patel et al., 2024](https://doi.org/10.1145/3654923)).
- Global-Local Selectivity describes vector-filter correlation and filtered-query difficulty
  ([Amanbayev et al., 2026](https://arxiv.org/abs/2602.11443)).
- Ada-ef selects per-query HNSW effort for a requested recall target
  ([PACMMOD 2026](https://doi.org/10.1145/3786639)).
- RACORN-1 adds graph and exact fallback modes for low-selectivity, correlated filtered search
  ([Kim and Choe, 2026](https://arxiv.org/abs/2607.00768)).
- Fiber-Navigable Search connects local geometry to filtered-graph failure regimes
  ([Dang, 2026](https://arxiv.org/abs/2604.00102)).

The proposed contribution is the conjunction of exact authorized truth, an authorization-filtered
bounded probe, multiscale geometry computed only from that probe, paired replay of executable
actions, frozen action control, evidence-bundle scoring, and a fresh policy decision at the return
boundary.

## Fixed corpus suite

The suite has five strata. Corpus-level estimates receive equal weight when an endpoint is defined
on all five.

| Corpus | Normalizer | Retrieval labels | Complete-evidence endpoint |
|---|---|---|---|
| SciFact | `normalize_scifact` | Relevant document IDs | Yes, including alternative rationale bundles |
| HotpotQA FullWiki | `normalize_hotpotqa_fullwiki` | External-corpus document IDs | Yes, from supporting facts |
| T2-RAGBench | `normalize_t2_ragbench` | Relevant document IDs | Yes, source context as complete document evidence |
| BRIGHT | `normalize_qrels_corpus` | Qrels | No |
| MIRACL transfer | `normalize_qrels_corpus` | Qrels | No |

SciFact, HotpotQA FullWiki, and T2-RAGBench form the fixed evidence subset. Complete-evidence
sufficiency is averaged equally across those three corpora. All five corpora use exact authorized
top-k as primary ANN truth. BRIGHT and MIRACL relevance judgments support secondary IR reporting;
they do not substitute for exact ANN truth or enter the complete-evidence gate. Missing evidence
annotations are undefined, never negative.

All five adapters are implemented. Their presence is not sufficient for a sealed run: each input,
label artifact, normalizer revision, source revision, hash, license, chunking rule, and exclusion
rule must be pinned in the frozen manifest. `normalize_hotpotqa` is a development fixture for
supplied contexts. It rejects `stage="sealed"` because those contexts preselect paragraphs and
cannot stand in for FullWiki retrieval.

## Frozen authorization workloads

Each corpus has a separate `policy-workload` artifact. It fixes subjects and attributes,
environment rows, document attributes, policy rules, policy mutations, seeds, and every planned
subject-query-policy pairing. The workload is generated without relevance, answer, or evidence
labels. Its claim is conditional: public-corpus permissions do not estimate any enterprise's
entitlement distribution.

The ordered document-universe digest is derived from stable external document identity plus pinned
content, chunking, and embedding revisions. A digest of integer positions or document count is not
admissible. The OPA bundle, policy data, and expected bundle revision are separately hashed. Every
decision must echo that revision and the ordered-universe digest.

Before freeze, the development owner must pin the policy generator and its output, allow-rate and
policy-complexity strata, mutation schedule, excluded subjects or empty grants, and corpus-specific
counts. After freeze, no workload may be changed to improve low-effort action-failure prevalence,
controller latency, relevance, or evidence sufficiency.

Each policy state has a separately seeded baseline mask and one current mask over the same ordered
document universe. Both immutable policy identities and both complete mask digests are frozen.
`policy_churn` is their exact subject-specific Hamming fraction; the live OPA serves only the current
mask. This seeded mutation is an experimental factor, not an estimate of enterprise policy change.

## Trial, target, and paired action matrix

For a trial \(i=(q,u,t)\), define:

- \(C_t\): the corpus snapshot at time \(t\);
- \(m_t\): the approved embedding model and revision;
- \(P_t(u,d,o,e)\in\{0,1\}\): the policy decision for subject \(u\), document \(d\),
  operation \(o\), and environment \(e\);
- \(C_{u,t}=\{d\in C_t:P_t(u,d,o,e)=1\}\): the authorized universe;
- \(G_i^K\): exact top-\(K\) retrieval over \(C_{u,t}\); and
- \(S_i^K(a)\): the result for prespecified action \(a\).

The exact authorized reference is

\[
G_i^K=\operatorname{TopK}_{d\in C_{u,t}}
\operatorname{sim}(e_{m_t}(q),e_{m_t}(d)).
\]

Authorized ANN recall uses the following total convention:

\[
R_i(a)=
\begin{cases}
\frac{|S_i^K(a)\cap G_i^K|}{|G_i^K|}, & |G_i^K|>0,\\
1, & |G_i^K|=0\ \text{and}\ |S_i^K(a)|=0,\\
0, & |G_i^K|=0\ \text{and}\ |S_i^K(a)|>0.
\end{cases}
\]

The intent-to-treat low-effort action-failure label used for modeling is

\[
Y_i=\mathbf 1[\operatorname{state}_i(\texttt{hnsw-low})\ne\texttt{completed}
\;\lor\;
(\operatorname{state}_i(\texttt{hnsw-low})=\texttt{completed}
\land R_i(\texttt{hnsw-low})<0.90)].
\]

Every prespecified trial remains in this endpoint. A failed or otherwise non-completed low-effort
action is a failure even when recall is unavailable. An empty authorized universe with a completed
empty result is a valid governed no-result service outcome under the recall convention above. The
endpoint is therefore a service-and-retrieval composite; it must not be reported as pure ANN failure.

Every trial executes the same frozen action set:

- `hnsw-low`, which reuses the bounded probe;
- `hnsw-high`;
- `exact-authorized`; and
- `abstain`.

The adaptive controller and frozen static comparator select from realized paired outcomes. Pair IDs
must be unique, identically ordered, and identical across proposed and comparator inputs. A missing
or duplicated pair is a protocol error, not an independent observation.

`nested_rows_per_family` is a trial-design cardinality, pinned independently of design assurance.
It fixes the exact number of subject-policy draws, seeds, drift rows, or timing repeats carried by
each query family. The sealed panel must match it exactly; it is not inferred from a power model.

## Authorization contract

`OpenPolicyAgentDecisionPoint` is the primary policy-decision-point artifact. The benchmark
`AuthorizationPolicy` remains a controlled role-by-document oracle for engineering and policy
workload tests; it is not the primary PDP in the sealed manifest.

The OPA adapter sends subject, action, environment, and the complete controlled document universe
through the bulk Data API. A valid response must include a decision ID, policy revision, echoed
subject/action/environment digest, document-universe digest, request nonce, and request SHA.
Malformed, unavailable, replayed, redirected, or mismatched output becomes an unavailable deny-all
decision. The expected policy-bundle revision is pinned. The built-in remote transport requires
HTTPS and a bearer credential; local HTTP is allowed only for a literal loopback IP.

The first authoritative decision occurs before authorized index construction, probe search,
geometry calculation, or action selection. The index is built only over the permitted subset and
is keyed by policy revision and authorization-mask digest. A cached index is never final authority.

A second, fresh decision occurs immediately before document IDs cross the controlled return
boundary. The request fails closed if the policy revision or mask changes, if a decision ID,
request nonce, or request SHA is reused, or if any selected document is no longer permitted.
Geometry may increase effort, choose exact search, or abstain. It cannot grant access.

The primary safety observation is therefore

\[
U_i(a)=\mathbf 1[\text{a denied document ID crosses the controlled return boundary}].
\]

The success condition is zero observed events. A family-level one-sided Clopper-Pearson upper
bound is also reported after nested rows are collapsed to “any violation in this family”
([Clopper and Pearson, 1934](https://doi.org/10.1093/biomet/26.4.404)).

This guarantee ends when `GovernedRetriever.query` returns its result object. The implementation
does not continuously reauthorize that object at a generator, UI, network sink, or later consumer.
It does not establish end-to-end RAG authorization, side-channel freedom, or downstream data-flow
control.

Sending the complete benchmark document universe in one OPA request tests the bulk-decision
contract. It is not evidence of production-scale IAM throughput or equivalence to AWS IAM, Azure
RBAC, Google Cloud IAM, SharePoint ACLs, or an organization's entitlement graph. The authorization
model is consistent with the subject, object, operation, and environment terms in
[NIST SP 800-162](https://doi.org/10.6028/NIST.SP.800-162) and the resource-access ordering in
[NIST SP 800-207](https://doi.org/10.6028/NIST.SP.800-207).

## Bounded multiscale geometry

The online feature source is one authorization-filtered HNSW probe. It returns at most 101
authorized neighbors. The low and high configured efforts are `efSearch=128` and `efSearch=512`.
`hnsw-low` reuses the first ten probe results, so the probe is charged once.

The prespecified geometry panel contains only:

- LID MLE at \(k=50\), recorded as `lid_k50`;
- LID at \(k=20\) and \(k=100\) for the cross-scale calculation;
- `lid_cv`, the coefficient of variation over \(k\in\{20,50,100\}\);
- relative contrast; and
- radius expansion.

Probe latency and declared work are pre-outcome system telemetry. Allow rate, authorized-universe
size, policy churn, and embedding drift remain system or policy variables according to the frozen
feature schema. None may be described as geometric evidence.

Embedding drift is derived separately for every query as one minus cosine similarity between its
receipt-bound active and current-truth query rows. Neither embedding drift nor policy churn may enter
the sealed runner as a caller-supplied scalar.

The telemetry object contains authorized IDs, distances, metric, corpus counts, neighbor bound,
latency, and work. It contains no vector matrix or query handle. An unavailable scale yields `NaN`;
the row stays in the analysis, and the reference rule controller treats undefined geometry as risk.

Full-authorized-universe geometry is an offline oracle and ablation only. It cannot enter the
controller. The retired v0.1 MFDFA statistic remains a negative control because row permutation
changed its value without changing the point cloud.

## Model artifacts and leakage controls

Four logistic artifacts are fit before sealed execution and outcome scoring:

1. `system-only`;
2. `system-policy`;
3. `geometry-only`; and
4. `full`.

System features are corpus size, authorized-universe size, embedding dimension, version lag,
drift severity, corpus stratum, backend, drift family, probe latency, and probe work. Policy features
are allow rate, policy complexity, and policy churn. Geometry features are only `lid_k50`, `lid_cv`,
relative contrast, and radius expansion. `system-policy` is the H2 reference and includes all system,
policy, and probe-telemetry variables. `full` adds only the four geometric descriptors and the
`lid_k50__x__lid_cv` interaction.

Development fitting and development calibration use separate query-family groups. Both remain
group-disjoint from the sealed stage under the registered connected-component graph. That graph
contains shared-positive-document edges derived from qrels. Component digests enter deterministic
stage allocation for SciFact, MIRACL transfer, BRIGHT, and the HotpotQA FullWiki training split;
HotpotQA development questions are fixed to sealed. Positive-qrel changes can therefore alter
component membership, digest, rank, and assigned stage.
The serialized JSON artifact records feature schema, imputation, standardization, categorical
levels, coefficients, Platt calibration, model digests, suite digest, and group hashes. No sealed
intercept, coefficient, category level, threshold, or calibration parameter may be refit after
sealed execution begins or after sealed payloads enter the label-authorized analysis process.

The development cohort ranking step is prospectively fixed. For each corpus, the selector ranks
qrel-derived assignment components with the shared `query_cohort` algorithm and fixed SHA-256
seeds, then keeps exactly 200 fit families and 75 calibration families. It chooses one
representative per component by the registered representative rank. The selector itself opens no
development qrel or evidence payload. Its canonical receipt is published before materialization
opens those payloads. Materialization must reproduce that receipt byte for byte, verify the paired
old/current embedding receipt and document universe, and only then filter development labels. This
ordering limits post-selection discretion but does not make component construction
label-independent. The development-freeze config directly pins the receipt and rejects a query set
that differs from it. The exact operational contract is specified in
[`development-cohort.md`](development-cohort.md).

The corpus conformance check rejects an exact query family assigned to two stages. Before freeze,
the custodian must also pin the prespecified near-duplicate procedure and its result.

## Prespecified estimands

### H1: descriptive frozen-model orientation diagnostic

H1 asks whether the frozen full model assigns greater low-effort action-failure risk to the
prespecified high-geometry profile than to the low-geometry profile while holding the sealed
covariate distribution fixed. It uses no sealed outcome labels. The contrast is predictive,
noncausal, and descriptive: it checks the orientation of a model fitted before sealed execution
and outcome scoring rather than testing a held-out geometry-outcome association.

Within `run_confirmatory_analysis_once`, the pinned computation derives the per-row high-minus-low
predictive-risk contrast and resamples query families inside each fixed corpus. Its directional
bound and the legacy `h1_minimum_risk_increase` value are reported as diagnostics only. H1 has no
role in confirmatory success, title selection, or the primary claim. The draft manifest still lacks
numeric high- and low-geometry profiles, an admitted H1 artifact, and an admitted model suite.

### H2: added held-out predictive information

H2 compares `full` with `system-policy` on paired sealed families. `system-policy` contains system,
policy, probe-latency, and probe-work variables; `full` adds only the four declared geometry
descriptors. Log loss and Brier loss are `system-policy` minus `full`; AUPRC gain is `full` minus
`system-policy`. Row weights are equalized within query family, corpus metrics are computed
separately, and the fixed-corpus summary weights the five corpora equally. The prespecified
consistency rule requires geometry gain in at least four of five corpora.

Within `run_confirmatory_analysis_once`, the pinned computation reports the corpus-specific point
summaries, applies the four-of-five consistency rule, and computes equal-corpus directional
family-bootstrap bounds for all three metrics. H2 requires the point rule and every aggregate
directional gate to pass. All three draft thresholds remain `TBD`, and the H2 suite is unpinned. The
runner cannot instantiate an H2 decision from the current manifest.

### H3: controller cost and constrained fidelity

The static comparator is chosen on the calibration split as the lowest-latency single action that
satisfies the frozen fidelity constraints. Its runtime action name and artifact digest are pinned
before sealed execution and outcome scoring.

Before the first action timer starts, the runner obtains one authorization for every distinct
registered trial environment, loads the corresponding exact and HNSW objects, and seals the
multi-mask cache. A canonical cache-preparation receipt binds each environment digest to its policy
revision, mask digest, and authorized count. A timed decision that selects an unprepared mask or
rebinds a prepared mask to another environment aborts the matrix.

`GovernedRetriever.query` starts the request timer before input validation and the first timed
policy decision. It stops after the fresh point-of-return decision, immediately before returning
the result. Governed request latency includes both timed PDP decisions, the sealed-cache lookup,
bounded probe, geometry, controller choice, selected search, and local request overhead. It
excludes one-time index loading. H3 therefore estimates warm-service request latency under a
fully admitted policy-state set, not cold-start deployment latency.

The query arrives as an embedding vector. The timer excludes upstream query embedding, answer
generation, UI work, and any network or consumer work after the result is returned. “End-to-end”
in this protocol means the governed retrieval request boundary just defined.

For corpus \(c\), query family \(f\), and nested row \(r\), let
\(T^{A}_{cfr}\) and \(T^{S}_{cfr}\) denote proposed and static-comparator request latency. First
average nested rows separately by action:

\[
\bar T^{a}_{cf}=\frac{1}{n_{cf}}\sum_r T^{a}_{cfr}.
\]

The family relative reduction is

\[
D_{cf}=1-\frac{\bar T^{A}_{cf}}{\bar T^{S}_{cf}}.
\]

The prespecified `end-to-end-request-latency-family-relative-reduction` estimand is

\[
\Delta_C=\frac{1}{5}\sum_{c=1}^{5}
\frac{1}{F_c}\sum_{f=1}^{F_c}D_{cf}.
\]

This is a ratio of family-mean action latencies, not a mean of row-wise ratios. Extra policy draws,
seeds, or timing repeats within a family do not gain inferential weight.

For the tail constraint, compute the p95 of family-mean latency for each action inside each corpus,
form the proposed-to-comparator ratio, then average the five corpus ratios equally.

Retrieval-target attainment is the favorable binary event that authorized recall reaches the
prespecified target. It is defined on all five corpora. Complete-evidence sufficiency is the
favorable event that at least one authorized gold bundle is fully present; it is defined only on
SciFact, HotpotQA FullWiki, and T2-RAGBench. Both differences are proposed minus comparator and use
the same family-then-corpus weighting.

H3 succeeds only if all conditions hold:

- the one-sided 95% lower bound for the equal-corpus mean of family-level relative request-latency
  reductions, \(\Delta_C\), is greater than 0.10;
- the one-sided 95% lower bound for retrieval-target attainment difference is greater than
  \(-0.01\);
- the one-sided 95% lower bound for complete-evidence sufficiency difference is greater than
  \(-0.01\);
- the one-sided 95% upper bound for the equal-corpus mean of within-corpus proposed-to-comparator
  p95 ratios of family-mean request latency is less than 1.25; and
- no denied item is emitted at the controlled retrieval boundary.

These are intersection-union conditions at \(\alpha=0.05\). Every condition must pass
([Berger and Hsu, 1996](https://doi.org/10.1214/ss/1032280304)).

A prespecified carryover sensitivity fits paired log request-latency ratios against the actual
proposed-minus-comparator execution position inside each corpus. Query families are bootstrapped,
and the five zero-position-difference corpus contrasts receive equal weight. Its one-sided upper
bound is compared with `log(1 - minimum_cost_reduction)`. The endpoint is always reported but
cannot alter H3, the primary H2/H3 intersection, family-count selection, or title selection. The
joint-power simulation reports the same endpoint's operating probability outside the primary-gate
conjunction.

## Directional family bootstrap

The sealed confirmatory analysis will use 10,000 deterministic paired bootstrap replicates unless a
larger frozen count is recorded. The draft protocol fixes base seed `20260713`; endpoint-specific
offsets are deterministic in the pinned runner. For each replicate:

1. keep the five corpora fixed;
2. sample query families with replacement inside each corpus;
3. carry all nested rows and both paired actions with the sampled family;
4. average nested rows within family and action; and
5. average family estimates within corpus, then fixed-corpus estimates equally.

Corpora are not resampled. Nested rows and action rows are not sampled independently. Inference is
conditional on the five-corpus suite.

For a directional 95% lower bound, use the fifth percentile of the bootstrap distribution. For a
directional 95% upper bound, use the ninety-fifth percentile. The two reported directional
percentiles are not a two-sided 95% interval. The frozen runner must preserve endpoint direction
and reject pair-order, duplicate-pair, or corpus-set mismatches.

Secondary ablation families use [Holm correction](https://doi.org/10.2307/4615733) when a family of
hypotheses is interpreted inferentially. Crashes remain service failures; undefined geometry stays
in its frozen missing-value path. No row can be dropped because its outcome is inconvenient.

## Event-yield sensitivity and joint-gate design

Raw query count is not the effective sample size. User-policy draws, seeds, drift levels, and timing
repeats attached to one query family are nested observations.

The implemented `paired-beta-binomial-common-shock` utility remains an event-yield sensitivity
analysis for low-effort action success, the complement of the intent-to-treat composite. It can
study family clustering and paired residual coupling. It does not estimate power for H2, H3, or
their conjunction, and its lower Monte Carlo bound cannot select the confirmatory family count.

Before freeze, the pinned design report must use `development-family-cluster-resampling` with
`registered-percentile-family-bootstrap-plug-in-calibration`. Whole families are resampled within
the fixed corpus suite. Continuous lower gates add the fifth percentile of centered calibration
error to an independently simulated study estimate; the p95 safety gate adds the 95th percentile.
This approximates the registered directional percentile bounds for power planning. It does not
replace the exact 10,000-replicate bootstrap used on sealed observations.

The design module exposes `audit_percentile_approximation`, which reconstructs a seeded simulated
study and compares all seven continuous plug-in gate decisions with the exact registered
bootstrap. A frozen family-count claim requires the canonical exact selection certificate defined
below. Any primary decision disagreement in a checked study aborts certification. The report
retains the exact registered estimands, fixed-corpus weighting, thresholds, noninferiority margins,
and intersection-union rule. Its joint success event is `h2-and-h3-all-gates-pass`. The registered
endpoints are, in order: H2 log-loss reduction, H2
Brier-score reduction, H2 AUPRC gain, H2 four-of-five consistency, H3 family-relative latency reduction,
H3 retrieval-target noninferiority, H3 complete-evidence noninferiority, H3 family-mean p95 latency
ratio, and H3 zero entitlement violations.

The report must pin its development-data dependence source, effect scenarios, simulation seed, and
at least 5,000 simulated studies for each candidate count. Candidate family counts are 25, 50, 75,
100, 150, and 200 per corpus. It reports endpoint-specific operating characteristics, the joint
H2+H3 pass probability, Monte Carlo uncertainty, and the zero-event condition's finite-sample
behavior. The selected family count is the maximum requirement across the endpoint-specific and
joint-gate calculations; its one-sided lower Monte Carlo bound must reach the frozen 0.90 design
target. Until those fields and the report are pinned, no power claim or family-count selection is
permitted. Simulation-based planning for clustered models follows the approach illustrated by
[Green and MacLeod](https://doi.org/10.1111/2041-210X.12504).

The production selection certificate is fail-closed. The multiplicity family is fixed before
simulation at six candidate counts by two required scenarios, or 12 cells. Let $M=5{,}000$, target
$p_0=0.90$, familywise confidence 0.95, and Bonferroni cellwise alpha $0.05/12$. Let $k$ be the
smallest integer whose one-sided Clopper-Pearson lower bound at that cellwise alpha is at least
$p_0$; here $k=4{,}556$. A scenario-candidate cell qualifies only after 4,556 checked studies pass
the exact joint gate. It is blocked after 445 checked studies fail the exact joint gate. This gives
simultaneous coverage of at least 0.95 over the fixed grid without assuming independence among
cells. Candidate counts are processed in the registered ascending order. Required scenarios are
processed in canonical order. Approximate-pass indices are checked first for a provisional pass;
approximate-fail indices are checked first for a provisional failure; ties retain ascending study
index. Each checked study uses the registered 10,000-replicate bootstrap and records its exact
family-draw digest. This stopping rule is sufficient because unchecked studies cannot lower a
known pass count or erase a known failure count.

`selection-audit.json` must bind the config SHA-256, all panel SHA-256 values, the complete plug-in
selection basis, stopping thresholds, every checked index and draw digest, exact and approximate
bounds and gate decisions, family size, familywise confidence, cellwise alpha, multiplicity method,
the selected count, and the exact-bootstrap settings. Missing, partial, duplicate, reordered,
stale, or substituted records are inadmissible. The freeze verifier must reproduce the file byte
for byte before accepting `freeze_ready=true`. The action-position sensitivity remains outside the
selection conjunction; its pointwise 95% result and any approximation disagreement are reported
but cannot alter the family count.

Observed sealed inference uses the directional family bootstrap, not the design simulator.

## Drift and noninterference

Corpus snapshot, embedding pair, migration fraction, policy revision, policy seed, index seed,
query family, pseudonymous subject, action, evidence outcome, request latency, work counters, and
failure state are recorded for each trial. Corpus, embedding, and policy drift are frozen design
factors and reported separately.

The current manifest does not define an additional H4 success threshold. Combined-drift and
transfer stress tests are secondary unless an amended protocol and frozen runner define a gate
before sealed execution and outcome scoring.

The paired-world conformance test holds authorized vectors, query, policy, controller, and index
seed fixed while changing denied vectors. It compares deterministic visible fields. An unsafe
global comparator is the positive control.

A passing test supports only an extensional statement for those inputs and fields. It is not a
universal noninterference proof and does not cover timing, cache state, process memory, network
traces, policy logs, or hardware side channels.

## Evidence labels and custody

Gold evidence is one or more alternative complete bundles. Every location records document ID,
source URI, exact locator, and optional content hash. A bundle is authorized only when every member
is permitted. Returned evidence is sufficient only when it covers every location in at least one
authorized bundle, including a pinned hash where supplied.

For each `stage="sealed"` normalized corpus, the custodian emits two digest-bound artifacts:

- a label-separated online artifact containing documents and query text under opaque HMAC-SHA256
  trial and family keys; and
- a sealed label artifact containing answers, relevant IDs, evidence bundles, label metadata, and
  the execution-artifact SHA.

The frozen manifest separately pins the source `sealed-inputs`, the `online-execution` package
delivered to the runner, the canonical `sealed-labels` artifact, and a `sealed-label-ciphertext` for
every corpus. For sharded execution, `sha256` pins the complete package tree and `revision` carries
the canonical logical plan digest. The execution controls and label file do not embed the manifest
digest, which would make their own pinned digest circular. The sealed label file binds the logical
execution digest; the outer manifest binds the package and files to the study.

A closed custody-seal receipt binds, per corpus, distinct online-execution, plaintext-label, and
ciphertext digests. It also records one exact drand chain hash and positive integer release round,
plus immutable timelock-tool and custody-builder digests. The manifest pins the exact
newline-terminated receipt file. Durations and moving round aliases are inadmissible. The receipt
proves agreement among named bytes; it does not prove correct encryption, deletion of every other
plaintext copy, public-label ignorance, or independent administration.

The online artifact rejects original IDs, labels, answers, evidence, relevance fields, and label
metadata. It still contains query text. Public benchmark queries may therefore be reidentified by
an operator who already knows the benchmark. Label custody prevents direct label delivery; it does
not make public queries unknowable. A frozen noninteractive runner, restricted egress, immutable
artifacts, and post-receipt label joining attenuate this residual channel. The custodian retains the
sealed label artifact outside the online process. Independent organizational custody would require
separate ultimate administrative authority, not merely a second job or credential under the same
administrator. This study has common administrative control and does not claim that independence.

After the one-shot run receipt exists, the online runner may emit immutable prediction and raw
action-panel artifacts. The prediction artifact binds manifest digest, receipt digest,
execution-artifact digest, and the exact complete set of opaque trial keys. The action panel records
every prespecified action's returned IDs, recorded request latency, execution state, controller
selection, entitlement count, pre-outcome features, and the audit-record digest for every completed
or governed-abstention row. `GovernedActionExecution` admits such a row only after its
`GovernedResult` agrees with the self-hashed `AuditRecord`; returned IDs, request latency, and
entitlement count are derived from that pair. Audit records cannot be reused. Governed
counterfactuals must share the selected policy revision, and final decisions must share policy
revision, environment digest, document-universe digest, and authorization mask. The panel cannot
contain relevance, gold evidence, recall, evidence sufficiency, answer labels, or derived failure
targets. The receipt starts the externally registered run; it neither executes retrieval nor
grants general label access.

A failed action uses `FailedActionExecution`: trial, controller decision, authorization decision,
one of `backend-error`, `backend-timeout`, `invalid-result`, `resource-exhausted`, or
`runner-interruption`, monotonic start and finish times, and runner identity. Admission derives the
latency and recomputes a failure-timing digest over those fields. The timing evidence is supplied by
the pinned runner; it is not an independent clock. A failure has no audit-record digest, returned
IDs, or entitlement claim. Every `hnsw-low` row must carry its supplied pre-outcome feature tuple,
including a failed row, while other actions cannot carry one. Panel admission checks action
placement rather than computational provenance; the anchor fixes the bytes before label release,
and later analysis admission checks dimension.

The panel builder requires exactly one admitted outcome per trial-action cell and keeps a
selected-action failure in the intention-to-treat analysis. The `abstain` cell must be a governed
abstention, and every trial requires a completed `exact-authorized` oracle. The caller supplies the
trusted audit-chain head, frozen query-partition-audit digest, and `primary` partition label. The
receipt schema may encode `reserve` for nonconfirmatory engineering work, but v0.3 cannot admit it
as a replacement confirmatory sample. Every governed row must form the verified chain ending at
that head. Registered `action_order` and actual `execution_position` are separate fields. The
frozen seed produces SHA-256-ranked cyclic Latin rows. Admission rejects an incomplete per-trial
permutation or position counts that differ by more than one within a corpus or opaque query
family.

`action_panel_from_governed_executions` returns the panel with a detached
`ActionPanelAdmissionReceipt`. The receipt binds the exact panel digest, manifest, run, online
execution artifact, corpus, partition, query-partition audit, audit head, ordered audit-record
digests, and one admission record per trial-action cell. Each record binds the controller action,
risk score, reasons, and policy revision; the policy decision ID, request digest, mask digest and
size, availability, environment, and document universe; and either governed audit position or the
runner-bound failure timing. The receipt is canonical, newline-terminated, and written exclusively.

An independently administered HTTPS anchor then issues a canonical prediction-completion receipt
bound to the exact prediction digest and a typed action-panel binding: manifest, run receipt,
online-execution digest, corpus, stage, and raw panel digest. The receipt also records prediction
count, anchor identity, URI, and UTC time. The custodian releases labels only after that completion
receipt exists and the registered drand round is available, and the offline join must present that
same panel binding. The drand condition is a time embargo, not an event gate. If the registered
round becomes available before a valid completion anchor, the confirmatory run terminates without
scoring.
`join_predictions_after_receipt` verifies both receipts, all artifact digests, and exact trial keys
before it joins predictions to labels for scoring. For each action, the analysis input builder
derives ANN recall against the completed `exact-authorized` row in the same anchored action panel;
relevance labels do not define ANN recall. It derives complete-bundle coverage by matching anchored
returned IDs to the sealed evidence bundles rather than accepting either outcome from the online
runner.

The scorer must load each custody file through `load_sealed_label_artifact`; reconstructing an
object from copied label fields or a digest string is not byte admission.
`ConfirmatoryInputArtifact` derives the analysis configuration from the frozen manifest and checks
the run receipt, exact artifact-verification receipt, fixed corpus set, each panel's execution
digest against the `online-execution` logical revision, and each actual sealed-label artifact's
canonical bytes. The artifact-verification receipt separately binds the outer package tree. It also
requires the joined label objects to equal the admitted sealed payload, rather than trusting a
copied digest string. One detached action-panel admission receipt is mandatory for each corpus. The
input verifies its exact panel coverage, primary-partition label, frozen query-partition-audit
digest, governed audit chain, and failed-action runner identity. `run_confirmatory_analysis_once`
admits the H1 model and H2 suite only when their canonical bytes match the verified manifest
artifacts. A caller-supplied digest string or replacement analysis configuration is insufficient.

Model pins use the exact output of `canonical_h1_model_artifact_bytes` and
`canonical_h2_model_suite_artifact_bytes`: UTF-8 canonical JSON with no trailing newline. The H1 pin
covers the full-model artifact, and the H2 pin covers the complete suite. This byte contract is
distinct from the newline-terminated custody files below.

Custody files are canonical JSON followed by exactly one newline. The online, sealed-label,
custody-seal, online-custody-admission, prediction, completion-receipt, offline-evaluation,
action-panel, action-panel-admission, analysis-attempt, analysis-result-receipt, and
confirmatory-result loaders reject the applicable duplicate-key, nonfinite, schema, canonical-byte,
location, digest, symlink, and hard-link failures. Their writers create new files exclusively rather
than replacing an existing path. The machine-readable custody procedure and its proof limits are in
[label-custody.md](label-custody.md).

## Freeze, receipt, and single analysis

The manifest remains `draft`. Freeze requires every declared input and component role to have a
non-placeholder URI, immutable revision, SHA-256, and license; separate sealed-input,
online-execution, sealed-label, and sealed-label-ciphertext artifacts for every corpus; a pinned
custody receipt, timelock tool, and custody builder; a pinned hardware object; exact runner identity;
code commit; OCI image digest; stores; and no unresolved blockers.

Required component roles include corpus normalizers, corpus-specific policy workloads,
development-fit and calibration data, the connected query-partition audit with exact and
near-duplicate edges,
primary embedding, authorized exact oracle, strict HNSW backend, OPA PDP, frozen controller, static
comparator, H1 diagnostic artifact, H2 model suite, endpoint-specific joint-gate design report,
analysis runner, and source code.
The closed manifest also pins the action names, corpus suite, endpoints, weights, margins,
resampling counts, power model, and receipt URI template.

The root `production_workloads` field is part of C1. While the repository is a draft, its sole
permitted unresolved value is `unresolved-before-c1`. A frozen manifest instead contains exactly
five rows in the registered corpus order. Each row discloses the complete label-payload-excluded
`ProductionCorpusWorkloadSpec` object and the SHA-256 of its canonical UTF-8 JSON file, including
the terminal newline. The wrapper corpus, embedded corpus, runner image, runner identity, code
commit, and platform cannot vary independently. Before C1, the materializer writes those exact
files, one canonical `production_workloads` fragment, and the runtime-plan templates. Its blueprint
receipt binds the fragment hash and the exact blueprint inventory. After registration, the
finalizer requires equality among the public object, the preserved pre-C1 fragment, each individual
WorkloadSpec file, the blueprint binding, and the transitioned plan's workload digest. C1 therefore
registers the executable scientific workload rather than only naming its surrounding artifact
packages.

The manifest also contains a closed `sealed_execution.production_controls` object. It records the
materialization-config file SHA-256, the blueprint receipt's canonical-object SHA-256, and the
blueprint receipt file SHA-256. Those fields are part of the signed C1 manifest and its public
registry digest. They prevent a replacement config and replacement blueprint from becoming a new
local authority merely because the two replacement files agree with each other. No extra Zenodo
member is needed; the existing manifest, predicate, and registry record carry the binding.

The same pre-C1 blueprint writes a canonical `sealed_execution.hardware` fragment. Provider,
instance type, accelerator, and region are closed operator claims; CPU model and operating-system
identity are also declared before C1, then checked against all five independent preflight receipts.
Logical cores come from the admitted CPU set, and memory GiB comes from the byte-exact container
limit. The finalizer requires the five observations to agree, verifies an ARM64 final plan for each
corpus, and rejects any difference between those observations, the pre-C1 fragment, and C1.
It also reproduces every launcher-geometry field, mount, root, environment value, image, commit,
resource limit, provisional plan byte, control-tree digest, and preflight-contract field from C0
factory evidence, the C1 WorkloadSpecs, the pre-C1 config and blueprint, and fixed runtime
constants. Equality of a few selected hashes is not sufficient for launch admission.

The sole run-receipt URI is derived from the canonical frozen-manifest SHA-256. Before opening, the
frozen digest must appear in an independently administered registration. OSF describes a
preregistration as a time-stamped, read-only study plan posted before collection or analysis
([OSF Support](https://help.osf.io/article/330-welcome-to-registrations)). The runner receives a
canonical local copy of the registry record plus a closed receipt that binds the registry identity,
HTTPS URI, UTC timestamp, and exact record-file digest. The local receipt is not evidence of
registration without the matching registry record.

The online provider plan also registers `registered_online_runtime_budget_seconds`. This value is
fixed from development-only capacity planning before sealed confirmatory inputs are opened; it is
not an empirical full-suite timing from those inputs. The registered value must be positive and no
greater than the 72,000-second online phase ceiling.
The runner must also verify exact local coverage of every manifest artifact and write a canonical
artifact-verification receipt. The approved runner must match the pinned identity and present the
manifest lock plus both receipts. In production, `begin-sealed-run` also requires `--artifact-root`
and `--artifact-map`. It rereads every mapped local artifact immediately before opening, rebuilds the
verification receipt, and requires exact canonical-byte equality with the admitted receipt. A
stored receipt cannot stand in for current bytes.

`begin-sealed-run` validates those bindings and writes the run receipt exclusively. Before that
write, its built-in path performs one fresh certificate-validated HTTPS GET of the receipt's
registry URI, refuses redirects and responses over 64 KiB, and requires the fetched digest and bytes
to equal the secure local canonical record. The admission preflight runs before the network-disabled
sealed execution boundary. A pre-existing receipt, unavailable or redirected registry,
remote-record mismatch, incomplete or changed local artifacts, digest mismatch, or symlinked
receipt parent fails the opening. The fetch proves byte availability at admission time; independent
review remains responsible for registry ownership and custody.
The Python-only `trusted_registry_record_fetcher` parameter is an explicit test/integration seam;
the production CLI never supplies it, and injected transport authentication becomes part of the
trusted computing base. Size, digest, and exact-byte checks still apply to its output.

The run receipt records manifest digest, protocol, UTC timestamp, runner identity, code commit, OCI
image, protocol-registration and artifact-verification pointers, and its own URI. It does not run
retrieval or statistics.

After the prediction-label join, `run_confirmatory_analysis_once` accepts only a canonical local
`file:` results directory. It derives the attempt, detached result-receipt, and result paths from the
manifest digest. Before any H1 diagnostic or H2–H3 outcome computation, it validates the pinned model suite and
creates the attempt receipt with `O_EXCL`. That receipt binds manifest, run, confirmatory-input
digest, model-suite digest, runner identity, and intended result URI. An existing attempt aborts
before computation, and a failed attempt remains on disk.

After computation, the runner checks the result against the admitted attempt. It creates a detached
result receipt, also with exclusive creation, before it creates the canonical result file. The
detached receipt binds the attempt digest, result digest, manifest, and result URI. A crash may leave
an attempt receipt or result receipt without a complete result; this is retained failure evidence,
not permission to repeat the same admitted attempt. The built-in path rejects `s3:` and `gs:` result
stores. A remote store requires a separately pinned authenticated create-if-absent adapter and
conformance evidence. Operational steps are in
[confirmatory-execution.md](confirmatory-execution.md).

## Audit and release

A governed request can produce a canonical hash-chained audit record with policy decision IDs and
revisions, authorization-mask digests, component revisions, pseudonymous subject, controller
action and reason, work accounting, evidence document IDs and content hashes, output digest, and
predecessor hash.

The record omits query text, vectors, distances, raw subject, evidence text, policy masks, and raw
generated output. Verification detects middle deletion, reordering, changed records, and broken
links. Tail deletion requires a trusted expected head or record count. A hash chain is
tamper-evident; it is not an external signature or timestamp.

After the one-shot analysis decision, release the frozen manifest and digest, run receipt, permitted
data references, exclusions, paired action matrix, action-panel admission receipts, analysis-attempt
receipt, detached result receipt, result digest, estimates, directional bounds, all gate decisions,
event-yield sensitivity and joint-gate design results, conformance results, incident records, and
audit anchors. Null gates are released with passing gates. Restricted evidence text, raw subjects,
masks, query vectors, and policy secrets remain withheld.

## Interpretation fixed before results

- H1 is reported as a descriptive orientation diagnostic regardless of its sign and cannot change
  primary success.
- If H2 does not pass, geometry did not add the prespecified held-out predictive information beyond
  system, policy, and probe-telemetry variables.
- If H2 passes but H3 does not, prediction did not produce a controller result that met every cost,
  fidelity, tail-latency, and authorization condition.
- If `system-policy` matches `full`, remove the incremental geometry claim.
- If a single primary H3 condition fails, the joint controller claim fails.
- If any denied item crosses the controlled retrieval boundary, the authorization condition fails.
- If retrieval-target attainment passes but evidence sufficiency does not, no complete-evidence
  noninferiority claim is made.

The protocol cannot establish organization-wide authorization, provider-native IAM equivalence,
production scalability, downstream generator safety, continuous authorization after return, or
resistance to every side channel.

## Development evidence boundary

`experiments/run_governance_pilot.py` is synthetic engineering evidence. It checks metric handling,
authorization-first construction, bounded-probe reuse, exact authorized truth, paired action
replay, audit accounting, and an unsafe comparator. Its thresholds were adjusted on synthetic
data, and its rows cannot enter the sealed analysis.

These remain synthetic mechanism results, not confirmatory findings. Confirmatory support requires
a frozen, externally registered, label-separated, single-opening study that executes the
five-corpus design, preserves the paired action matrix, and passes the pinned H2+H3 intersection
without post-outcome revision. H1 remains descriptive even after that run.
