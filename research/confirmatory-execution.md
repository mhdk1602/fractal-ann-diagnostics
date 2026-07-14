# Confirmatory execution runbook

This runbook implements [protocol v0.3](preregistration.md). It does not authorize a sealed run.
The current [study manifest](study-manifest.json) is a draft, and the study has not run.

## Current state

The five corpus adapters exist, including the sealed HotpotQA FullWiki path. OPA is the primary
policy-decision-point artifact. The action runtime names, fixed-suite weights, primary endpoints,
directional family bootstrap, label-separation schema, action-panel admission receipt, and
single-attempt result contract are encoded in the repository.

The manifest is not freeze-ready. Artifact URIs, immutable revisions, SHA-256 values, licenses,
hardware, exact runner identity, code commit, OCI image digest, stores, and other declared fields
must be final. The H1 diagnostic artifact, H2 suite, static comparator, controller, one-sided
analysis runner, and endpoint-specific joint-gate design report must be pinned and their conformance
tests must pass. The dependence source, effect scenarios, simulation seed, selected maximum family
count, and joint-power lower bound remain `TBD`. Every blocker recorded by manifest validation must
be resolved before sealed labels are prepared for scoring.

Opening sealed data while any condition remains unresolved converts the work to an exploratory
run. A receipt cannot repair a draft protocol.

## Custody and execution roles

- **Development owner:** fits and calibrates model artifacts, freezes the controller and static
  comparator, supplies design-simulation inputs, and has no access to sealed outcomes.
- **Label custodian:** prepares label-separated online and sealed label artifacts, holds the manifest
  lock, and retains the labels until an externally anchored prediction-completion receipt exists.
- **Online runner:** uses only the label-separated online artifact, executes the frozen retrieval stack,
  and writes predictions bound to the run receipt.
- **Offline scorer/reviewer:** verifies all bindings, joins predictions to labels after receipt, runs
  the pinned analysis once, and checks release artifacts.

An engineering rehearsal may concentrate roles. A confirmatory report must disclose any shared
identity, credentials, machine, or storage authority across these roles.

## Freeze package

The development owner supplies the custodian with immutable artifacts for:

- separate sealed inputs, online-execution artifacts, and labels for SciFact, HotpotQA FullWiki,
  T2-RAGBench, BRIGHT, and MIRACL;
- one separately pinned policy-workload artifact per corpus, including its subject, environment,
  document-attribute, mutation-schedule, seed, and ordered-universe bindings;
- `normalize_scifact`, `normalize_hotpotqa_fullwiki`, `normalize_t2_ragbench`, and the pinned
  `normalize_qrels_corpus` uses;
- the primary embedding, tokenizer, pooling and normalization rule;
- the exact-authorized oracle and strict HNSW backend;
- the OPA PDP contract, policy bundle, endpoint constraints, and conformance record;
- the frozen controller and its exact `hnsw-low`, `hnsw-high`, `exact-authorized`, and `abstain`
  action mapping;
- the frozen static-comparator action;
- the H1 model artifact and four-model H2 suite;
- development-fit and development-calibration data, plus the connected query-partition audit with
  its exact and prespecified near-duplicate edges;
- the source-code commit, OCI image, analysis runner, expected schemas, and test results;
- corpus-family assignments, exclusions, licenses, duplicate checks, and evidence rules;
- the exact `nested_rows_per_family` trial-design cardinality;
- hardware, thread count, concurrency, warmup, timing repeats, and action-order seed; and
- the exact joint-gate endpoint order, development-data dependence source, effect scenarios,
  candidate counts, selected maximum family count, simulation seed and count, joint-power lower
  bound, and immutable design report.

Every artifact receives a non-placeholder URI, immutable revision, SHA-256, and license where the
schema requires one. The manifest must also pin the custodian, exact runner identity, approval
environment, artifact stores, receipt URI template, and freeze blockers as an empty list.

Only then may the custodian change `status` to `frozen` and create the separately controlled lock.
The canonical manifest digest must then be deposited in an independently administered registry
before any sealed run opens. The
[OSF registrations guide](https://help.osf.io/article/330-welcome-to-registrations) describes a
time-stamped, read-only study plan that can be public or embargoed. The custodian retains
a canonical local copy of the registry record and records its identity, HTTPS URI, UTC registration
time, and file SHA-256 in a closed `ProtocolRegistrationReceipt`. The record and receipt use closed
schemas, canonical JSON, and exactly one terminal newline. A local receipt without the retrieved
registry record is insufficient.

Immediately before opening the run, the CLI performs one fresh certificate-validated HTTPS GET of
the receipt's registry URI. It refuses redirects, a changed response URL, non-200 responses,
invalid or oversized `Content-Length`, and any body larger than 64 KiB. The fetched digest and exact
bytes must match the secure local canonical record and the receipt. This control-plane admission
preflight occurs before the runner enters the network-disabled, noninteractive execution
environment. It verifies prospective public availability, not registry ownership. A reviewer
independent of the online runner must still establish who controls the registry and preserve that
evidence in the controlled study package.

The Python API exposes `trusted_registry_record_fetcher` only as an explicit test/integration seam.
The production CLI never supplies it. An integration that injects this callable assumes transport
authentication as part of its trusted computing base; returned bytes still face the same size,
digest, and byte-equality checks.

## 1. Prepare label-separated corpus artifacts

The custodian starts from a normalized corpus whose stage is `sealed`. For each corpus, the custody
tool emits:

1. a label-separated online artifact with documents, query text, and opaque HMAC-SHA256 trial and
   family keys; and
2. a sealed label artifact with answers, relevant IDs, evidence bundles, label metadata, and the
   online execution-artifact SHA.

The online artifact must contain no original query IDs, relevance labels, answers, evidence,
label metadata, or fields that disclose them. It does contain query text. Public benchmark queries
may be reidentified by an operator familiar with the source data, so opaque keys alone are not a
proof of blinding. The runner must be frozen, noninteractive, and denied unregistered egress; its
artifacts remain immutable and labels stay with the custodian until an external completion anchor
binds both the prediction and action-panel digests. BRIGHT and MIRACL have relevance labels but no
gold evidence bundles. SciFact, HotpotQA FullWiki, and T2-RAGBench form the complete-evidence
subset.

The relevance labels may support secondary IR summaries. Primary ANN recall is computed against
the completed exact-authorized action in the pre-label panel for all five corpora.

The custodian stores the artifacts separately. The frozen manifest pins the full canonical bytes of
each per-corpus `online-execution` artifact and `sealed-labels` artifact; neither artifact embeds the
manifest digest because doing so would create a digest fixed-point cycle. The sealed label artifact
instead binds the exact online-execution digest, while the manifest supplies the outer study
binding. The online runner receives only the label-separated online artifact.
`write_online_execution_artifact` and `write_sealed_label_artifact` create canonical files
exclusively; their matching loaders reject noncanonical JSON, duplicate keys, symlinks, and
multiply linked files.

## 2. Validate and lock the frozen manifest

From the repository root, validate the document:

```bash
fractal-retrieval-governance validate-study \
  --manifest research/study-manifest.json
```

After all fields are final and `status` is `frozen`, enforce sealed prerequisites:

```bash
fractal-retrieval-governance validate-study \
  --manifest research/study-manifest.json \
  --require-frozen
```

Compute the canonical digest:

```bash
fractal-retrieval-governance study-digest \
  --manifest research/study-manifest.json
```

The custodian writes that exact lowercase digest to a separately controlled lock file. Any manifest
edit changes the canonical digest and requires a protocol amendment, new lock, and new receipt URI
before sealed access.

## 3. Verify frozen artifacts and create the one-shot run receipt

The frozen manifest derives the sole receipt URI from its canonical SHA-256 through the prespecified
`receipt_uri_template`. The CLI does not accept a caller-selected receipt path.

The local artifact map assigns each manifest artifact ID to one relative path under a preprovisioned
root. It cannot supply or override a digest. Before opening the run, verify exact coverage and write
the canonical receipt exclusively:

```bash
fractal-retrieval-governance verify-study-artifacts \
  --manifest research/study-manifest.json \
  --artifact-root /controlled/artifacts \
  --artifact-map /controlled/artifact-map.json \
  --receipt /controlled/artifact-verification.json
```

The approved runner then starts the externally registered run with:

```bash
fractal-retrieval-governance begin-sealed-run \
  --manifest research/study-manifest.json \
  --lock /controlled/study-manifest.sha256 \
  --artifact-verification-receipt /controlled/artifact-verification.json \
  --artifact-root /controlled/artifacts \
  --artifact-map /controlled/artifact-map.json \
  --protocol-registration-receipt /controlled/protocol-registration.json \
  --protocol-registration-record /controlled/protocol-registration-record.json \
  --runner-identity github-actions:environment:confirmatory
```

Replace the example identity before freeze; at execution it must exactly match the value pinned in
the manifest. The production command requires `--artifact-root` and `--artifact-map`, even when a
verification receipt already exists. It reloads the map against the manifest pins, rereads every
artifact from the controlled root, constructs a fresh verification receipt in memory, and requires
its canonical bytes to equal the admitted receipt. This closes the interval between the earlier
verification command and run opening. The stored receipt cannot hide a later local mutation.

The command also validates the frozen manifest, lock digest, runner identity, external registration
pointer, exact artifact-ID coverage, and every manifest digest. It re-fetches the external registry
record through the bounded verified-HTTPS path and requires digest and byte identity with the local
record before deriving the run-receipt URI. It copies the pinned code commit, OCI image digest,
protocol-registration pointer, and artifact-verification pointer into the receipt, then writes the
receipt exclusively. An existing receipt, changed manifest, mismatched identity, unavailable or
redirected registry, remote-record mismatch, incomplete or changed local artifacts, or symlinked
receipt parent aborts the opening.

The receipt contains manifest digest, protocol, UTC timestamp, runner identity, code commit, OCI
image, both prerequisite receipt pointers, and its own URI. It records that the externally
registered run has started. It does not execute retrieval, run statistics, or expose labels.

## 4. Execute the label-separated online run

The online runner verifies the receipt, manifest, component digests, and the canonical execution
digest against that corpus's `online-execution` manifest pin before processing any trial. It must
then:

1. reject missing, duplicate, extra, or cross-stage opaque trial keys;
2. issue the initial OPA bulk decision before authorized index construction or probing;
3. build or load an index only for the authorized universe and verify its policy-revision and mask
   binding;
4. run the bounded authorized probe with a maximum of 101 neighbors and prespecified effort;
5. compute only LID at k=50, LID-CV, relative contrast, and radius expansion as geometry;
6. execute the paired action matrix for `hnsw-low`, `hnsw-high`, `exact-authorized`, and `abstain`;
7. issue a fresh OPA decision immediately before any selected document IDs cross the controlled
   return boundary;
8. fail closed on a replay, version change, mask change, unavailable PDP, or revoked selected ID;
9. retain crashes, abstentions, undefined geometry, missing work counters, empty authorized
   universes, and all timing rows; and
10. write immutable predictions and the complete raw action panel without accessing the sealed
    label artifact.

Completed and governed-abstention rows enter the panel through `GovernedActionExecution` and
`action_panel_from_governed_executions`. The typed admission checks the `GovernedResult` against a
self-hashed `AuditRecord`: action, policy revision, both authorization decisions, returned IDs,
search work, abstention state, and measured request latency must agree. Returned IDs, latency, and
entitlement count are then derived from those checked objects rather than supplied as scalar panel
fields. An audit record cannot be reused across actions. Governed counterfactuals must use the
selected decision's policy revision, and their final decisions must agree on policy revision,
environment digest, document-universe digest, authorization mask, and policy availability.

A failure enters through `FailedActionExecution` with monotonic start and finish times, the pinned
runner identity, and one of five closed codes: `backend-error`, `backend-timeout`, `invalid-result`,
`resource-exhausted`, or `runner-interruption`. Admission derives latency from that interval and
recomputes a failure-timing digest bound to the trial, action, controller decision, authorization
decision, code, runner, and timing window. This is bound evidence inside the pinned runner, not an
independent clock. The failure carries no audit digest, returned IDs, or
caller-supplied entitlement input and cannot claim a completed search. Its serialized panel row
records zero entitlement violations because no IDs were emitted. Every `hnsw-low` row, including a
failure, must carry the pre-outcome feature tuple; other actions cannot carry it. Admission checks
placement, not computational provenance.

The panel builder requires exactly one admitted outcome for every trial-action pair and preserves a
selected-action failure in the intention-to-treat panel. The registered `abstain` action must be a
governed abstention, and `exact-authorized` must be completed. Missing or duplicate outcomes are
inadmissible. The caller must supply the expected audit-chain head, the frozen query-partition-audit
digest, and `primary`. The receipt schema may encode `reserve` for nonconfirmatory engineering, but
v0.3 confirmatory input rejects it. The builder verifies the complete governed-record chain and
returns an `AdmittedActionPanel`: the panel plus a detached `ActionPanelAdmissionReceipt`.

The detached receipt binds the panel bytes, manifest, run, execution artifact, corpus, partition,
query-partition audit, ordered audit chain, and one admission record per trial-action cell. Each
record carries a digest of the controller decision and a digest of the applicable policy decision,
including its decision ID, request digest, mask digest and size, policy revision, availability,
environment, and document universe. The applicable authorization is the final decision when one
exists and otherwise the initial decision. Governed rows carry their audit position and predecessor;
failed rows carry their runner-bound timing digest. `write_action_panel_admission_receipt` writes the
canonical receipt exclusively. The panel alone is not an admissible confirmatory input.

The authorization observation ends at the controlled return from `GovernedRetriever.query`. The
runner cannot claim continuous authorization after the result object is returned. A later
generator, UI, network sink, or consumer is outside this experiment unless a separate protocol
adds and tests that boundary.

The request timer begins before validation and the first OPA call. It ends after the fresh final
OPA call when the governed result is returned. Recorded end-to-end governed retrieval request
latency includes both policy decisions, index refresh or cache check, probe, geometry, action
selection, selected search, and local request overhead. It excludes upstream embedding, generation,
UI work, and downstream network or consumer work.

Probe latency and work are system telemetry in the frozen predictive schema. Only LID at k=50,
LID-CV, relative contrast, and radius expansion constitute the geometric block.

## 5. Seal and externally anchor predictions and the raw action panel

The online runner emits an immutable prediction artifact and action-panel artifact only after the
receipt exists. The prediction artifact must bind:

- manifest SHA-256;
- receipt SHA-256;
- online execution-artifact SHA-256; and
- the exact complete set of opaque trial keys in canonical order.

The action panel contains every prespecified action for every opaque trial. It records action order,
returned document IDs, execution or failure state, controller selection, request latency,
entitlement count, the supplied pre-outcome feature tuple on `hnsw-low`, and the audit-record digest
for each completed or governed-abstention row. Typed admission checks the feature tuple's action
placement, and later analysis admission checks its dimension. Neither step independently proves
how the runner computed it. The external anchor freezes those supplied bytes before label release.
The panel contains no relevance judgment, gold bundle, recall, evidence-sufficiency value, answer
label, or derived failure target.

The custodian verifies that no label field entered the online artifact, every manifest-declared
trial key has a prediction row, and action failures appear explicitly in the panel. A selected
failure therefore has an empty returned-ID prediction plus the panel's closed failure state; a
missing prediction cannot be silently deleted. An independently administered HTTPS anchor then
records the exact prediction digest and a typed action-panel binding containing the panel digest,
corpus, stage, manifest, run receipt, and online-execution digest.
The resulting `PredictionCompletionReceipt` is written exclusively through a no-follow path.

`write_prediction_artifact`, `write_action_panel_artifact`,
`write_action_panel_admission_receipt`, and `write_prediction_completion_receipt` are the canonical
file boundaries. The matching loaders reject duplicate keys, nonfinite values, noncanonical bytes,
symlinks, hard links, and schema extensions. In-memory objects alone are not releasable custody
evidence.

The completion anchor must postdate the run start. Its identity, URI, and canonical UTC timestamp
remain in the receipt; the receipt digest enters the offline artifact. A prediction object without
this receipt cannot release labels.

## 6. Join labels offline

Only after the completion receipt exists may the scorer call `join_predictions_after_receipt`. The
join requires the same typed panel binding recorded at completion and verifies the manifest, run
receipt, execution, prediction, and label digests plus exact trial keys. A different panel, digest
mismatch, key mismatch, duplicate, omitted trial, or unanchored prediction aborts scoring.

The join exposes relevance and complete-evidence labels only in the offline scoring environment.
For each action, the analysis input builder derives ANN recall against the completed
`exact-authorized` row in the same anchored panel; relevance labels do not define ANN recall. It
derives complete-evidence sufficiency by matching anchored returned IDs to the sealed gold bundles
and requires zero entitlement violations. It rejects any post-label row whose action, latency,
state, feature vector, entitlement count, trial key, or family key differs from the anchored panel.
Answer emission, answer coverage, false permit, and false denial may be calculated as declared
secondary outputs. They cannot alter the primary gates.

The low-effort modeling target is intent-to-treat action failure. A non-completed `hnsw-low` row is
a failure; a completed row fails when authorized recall is below 0.90. A completed empty result
against an empty exact-authorized reference is a valid governed no-result service outcome with
recall one. No authorized-universe-size exclusion is applied, and the composite is never described
as pure ANN failure.

The scorer loads each custody file through `load_sealed_label_artifact`; reconstructing a label
object from copied fields is not file admission. `ConfirmatoryInputArtifact` derives its analysis
configuration from the actual frozen manifest; it
does not accept a caller-supplied replacement. It verifies the run receipt and exact
artifact-verification receipt, admits each actual `SealedLabelArtifact`, recomputes its canonical
bytes against the per-corpus manifest pin, and requires every joined label object to equal that
admitted payload. A copied digest string is insufficient. Each panel's execution digest must match
the separately pinned `online-execution` artifact.

The input also requires one detached action-panel admission receipt per corpus. It verifies the
panel digest and every trial-action admission, requires the `primary` partition, binds the receipt
to the manifest's query-partition-audit digest, and checks failed-action runner identity against the
sealed run receipt. `run_confirmatory_analysis_once` then checks the canonical H1 model and H2 suite
bytes against their verified manifest artifacts before it admits an analysis attempt.

The model pins are the exact outputs of `canonical_h1_model_artifact_bytes` and
`canonical_h2_model_suite_artifact_bytes`: UTF-8 canonical JSON without a trailing newline. The H1
pin covers the full-model artifact, and the H2 pin covers the full suite. Do not apply the custody-file
newline convention to these model byte payloads.

## 7. Run the pinned analysis once

The endorsed sealed entry point is `run_confirmatory_analysis_once`. The lower-level computation is
not a custody boundary. The built-in entry point accepts only a canonical absolute `file:` URI in
`sealed_execution.results_store`; it rejects authorities, query strings, fragments, control
characters, noncanonical encoding, dot components, and `s3:` or `gs:` stores. A remote store needs a
separately pinned adapter with authenticated create-if-absent semantics and its own conformance
evidence.

For manifest digest `<M>`, the built-in entry point derives three fixed paths inside that directory:

- `<M>.confirmatory-analysis-attempt.json`;
- `<M>.confirmatory-result-receipt.json`; and
- `<M>.confirmatory-result.json`.

It first checks the admitted model bytes, then creates the analysis-attempt receipt exclusively with
`O_EXCL`, before the H1 diagnostic or any H2–H3 outcome is computed. The receipt binds manifest, run receipt,
confirmatory-input digest, model-suite digest, runner identity, and result URI. An existing attempt
aborts before outcome computation. Once created, the receipt is retained even if analysis raises,
the process stops, or a later custody write fails.

After computation, the entry point checks that the result still binds the admitted manifest, run,
input, and model suite. It creates a detached result receipt exclusively before exposing the result
file. That receipt binds the attempt-receipt digest, result-artifact digest, manifest, and result
URI. It then creates the canonical newline-terminated result file exclusively. The secure loaders
reject noncanonical, linked, misplaced, or digest-mismatched receipts and results.

The computation verifies exact proposed/comparator pair IDs and action order. It calculates:

- the label-free full-model high-versus-low geometry orientation diagnostic for H1;
- paired `full` versus `system-policy` log loss, Brier loss, and AUPRC gain for H2, where
  `system-policy` includes probe latency and work and `full` adds only the four geometric features;
- the equal-corpus mean of family-level relative reductions in end-to-end governed retrieval request
  latency;
- retrieval-target attainment difference on all five corpora;
- complete-evidence sufficiency difference on the fixed three-corpus evidence subset;
- the equal-corpus mean of within-corpus proposed-to-comparator p95 ratios of family-mean request
  latency;
  and
- denied-item emission at the controlled retrieval boundary.

For corpus \(c\), family \(f\), and nested row \(r\), the latency estimand is

\[
D_{cf}=1-\frac{n_{cf}^{-1}\sum_r T^A_{cfr}}
{n_{cf}^{-1}\sum_r T^S_{cfr}},
\qquad
\Delta_C=\frac{1}{5}\sum_{c=1}^{5}\frac{1}{F_c}\sum_{f=1}^{F_c}D_{cf}.
\]

Nested rows are averaged separately by action before the family ratio is formed. Family statistics
are then averaged within corpus, and the five corpus estimates are averaged equally. This preserves
the paired family as the inferential unit and prevents extra policy draws, seeds, or timing repeats
from gaining weight. The runner must not substitute the mean of row-wise ratios.

For each primary endpoint, the runner performs 10,000 deterministic paired replicates with base seed
`20260713`, resampling query families with replacement inside each fixed corpus. Endpoint-specific
seed offsets are fixed in the pinned runner. It carries all nested rows and action pairs with the
selected family. Corpora, nested rows, and action rows are not separate bootstrap units.

The fifth percentile is the directional 95% lower bound. The ninety-fifth percentile is the
directional 95% upper bound. The runner applies these conditions as an intersection-union decision:

- equal-corpus mean family-relative latency-reduction lower bound greater than 0.10;
- retrieval-target attainment lower bound greater than -0.01;
- complete-evidence sufficiency lower bound greater than -0.01;
- equal-corpus mean of within-corpus p95 family-mean latency ratios upper bound less than 1.25; and
- zero denied items emitted at the controlled retrieval boundary.

H1 reports its directional contrast and legacy `h1_minimum_risk_increase` comparison as a
descriptive orientation diagnostic. It uses no sealed labels and cannot alter confirmatory success.
H2 requires the equal-corpus directional lower bounds for all three incremental geometry-gain
metrics to exceed their frozen thresholds and the corpus-specific point rule to pass inside at least
four of five corpora. All three H2 thresholds remain `TBD` in the draft; no numeric values are fixed.
Primary success is exactly the H2+H3 intersection. The frozen manifest must still supply numeric
geometry profiles and gain thresholds, and the admitted H1/H2 model artifacts must match their
manifest digests. The draft cannot produce a primary decision.

Within the pinned noninteractive scorer and controlled result directory, one durable attempt receipt
admits one result. The mechanism does not prevent arbitrary Python code, process-memory inspection,
logging, copying, or a storage administrator from bypassing the package. The scorer image, runner
identity, operating system, result-directory custody, and incident process remain in the trusted
computing base. No feature, threshold, exclusion, corpus weight, action, comparator, direction,
margin, or missing-value rule may change after labels are joined.

## 8. Verify joint-gate design

The frozen design report must use `development-family-cluster-resampling` and the joint success
event `h2-and-h3-all-gates-pass`. Its registered endpoint order is:

1. `h2-log-loss-reduction`;
2. `h2-brier-score-reduction`;
3. `h2-auprc-gain`;
4. `h2-four-of-five-consistency`;
5. `h3-family-relative-latency-reduction`;
6. `h3-retrieval-target-noninferiority`;
7. `h3-complete-evidence-noninferiority`;
8. `h3-family-mean-p95-latency-ratio`; and
9. `h3-zero-entitlement-violations`.

The report pins its development-data dependence source, effect scenarios, simulation seed, at least
5,000 simulations per candidate, candidate family counts 25, 50, 75, 100, 150, and 200 per corpus,
selected maximum family count, and selected joint-power lower bound. It must reproduce the exact
family/corpus weighting and H2/H3 gates. The selected family count is the maximum requirement across
endpoint-specific and joint-gate calculations; the one-sided lower Monte Carlo bound must reach the
frozen 0.90 design target.

The existing beta-binomial common-shock utility is only an event-yield sensitivity analysis for
low-effort action success. It cannot establish 90% power for the primary conjunction or select the
confirmatory family count. Observed inference remains the paired family bootstrap; no design
simulator can replace it.

## Technical failure; no confirmatory reserve

A failed process does not grant a rerun or reserve release. Preserve the run receipt, action-panel
admission receipts, analysis-attempt receipt, any detached result receipt, partial result, logs,
digests, and audit records. An admitted failure ends confirmatory v0.3. Any later attempt requires a
disclosed amended protocol version, a new frozen manifest, a new external registration, and a new
run receipt. It cannot be described as the original confirmatory run.

A null result, wide bound, failed gate, inconvenient subgroup, slow action, or observed entitlement
violation is a scientific result, not a technical failure.

## Release checklist

Release the frozen manifest and canonical digest, run receipt, permitted normalized-data references,
exclusions, opaque family counts, paired action matrix, action-panel admission receipts, prediction
schema, estimands, analysis-attempt receipt, detached result receipt, result digest, directional
bounds, all gate decisions, the endpoint-specific and joint-gate design report, event-yield
sensitivity, conformance results, incident records, and audit-chain anchors.

Restricted evidence text, original query IDs, raw subjects, authorization masks, query vectors,
policy secrets, and sealed credentials remain outside the public bundle. State explicitly that the
study is suite-conditional and that authorization was observed only through the controlled return
of `GovernedRetriever.query`.
