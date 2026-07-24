# Label-free paired-action online runner

`online_runner.py` bridges either an admitted inline execution artifact or a control-only sharded
execution view to `action_panel_from_governed_executions`. Corpus rows need not enter memory. It
executes retrieval only. The module has no custody-data loader, no outcome type, and no path
argument for protected artifacts.

`run_online_action_matrix` is the object-level mechanism. It is not the production entrypoint.
Production uses `sealed_online_execution.run_sealed_online_once`, which reconstructs that object
graph only after a file-backed runtime attestation and an exclusive corpus attempt both succeed.
External suite-state attestation remains a later admission layer.

## Closed production entrypoint

The operator-facing boundary is `run-sealed-corpus`, described below. It starts with an exactly
empty output directory. The same Python process loads the externally hash-pinned config and the
canonical runtime plan, proves the plan's argv, workload digest, and marker path, and calls
`attest_runtime_once(plan, probe=LinuxRuntimeProbe(), receipt_target=...)`. Only after reloading and
verifying the persisted receipt does it reconstruct the remaining typed controls and call the
internal `run_sealed_online_once` boundary. That Python function accepts a `TrialRuntimeAdmission`,
canonical runtime-attestation plan and receipt paths with their frozen digests, other receipt pins,
immutable absolute roots, the custody and run receipts, and the admitted artifact-ID closure.
Neither boundary has parameters for
an execution object, runtime map, retriever, controller configuration, policy revision, feature
context, policy environment, permutation seed, exact-truth array, K, policy action, partition,
runtime observation, mount namespace, environment, or probe.

The attestation function creates the invocation marker with `O_EXCL` before it reads launcher
identity, hashes a mount, or invokes the Linux probe. A failed probe therefore consumes the runtime
attempt. Before it opens an embedding, query, policy, provenance, HNSW, or secret-key source, the
inner boundary reloads the closed runtime plan and receipt and verifies their pins against canonical
content. The pair must reproduce the sealed run's manifest, runner identity, C0 commit, and OCI
image. A fresh internal `LinuxRuntimeProbe` reobserves the mount namespace, environment, argv,
resource facts, and network namespace. The invocation marker is rehashed. Every workload source
path must sit inside exactly one read-only mount named by the plan.

Only after that gate succeeds does the entrypoint write the manifest-scoped attempt receipt with
`O_EXCL`. The attempt directly binds the runtime-plan and runtime-receipt digests, then derives the
action-order seed, policy revision, query-partition digest, and execution digest from the trial
plan and runtime receipt. The fixed runner revision supplies `K=10`, action `retrieve`, the primary
partition, the default frozen rule controller, and the loopback OPA decision path.

After the token exists, the entrypoint performs these checks:

1. Load the authorized-index receipt and require its exact frozen digest.
2. Match its policy, embedding, execution, document-universe, active-document, and current-truth
   bindings to the plan and runtime receipt.
3. Reverify the complete authorized index store and construct `VerifiedAuthorizedIndexProvider`.
4. Call `load_trial_runtime` internally. Query epochs, feature contexts, and policy environments
   therefore come only from the receipt-bound source join.
5. Reverify every compiled mask, policy transition, and `opa-data.json` assignment against the
   policy receipt, catalog, schedule, and runtime groups.
6. Copy the canonical OPA data into a private `/tmp` directory, then start the image-baked OPA
   1.18.2 binary with the image-baked Rego module. The only decision URL is
   `http://127.0.0.1:8181/v1/data/fractal_auth/retrieval/mask_decision`.
7. Require both OPA plugin health and one exact decision-contract probe before constructing the
   policy decision point.
8. Open both document matrices through held no-follow descriptors. The current matrix descriptor
   must equal `AuthorizedIndexStoreReceipt.current_truth_vector`, including bytes, model identity,
   row order, dtype, and shape. Construct the rule controller, governed retriever, and digest-only
   provenance registry inside these held contexts.

OPA is not a launcher prerequisite and cannot be supplied by the caller. The production signature
has no binary, Rego, endpoint, process, transport, timeout, or readiness parameter. The runtime
attestation must pin `/usr/local/bin/opa`; C0 pins the Rego digest. Before spawning, the runner
rejects an occupied port. OPA receives `fractal:<private-path>` so the canonical data value appears
only at `data.fractal`. The child uses no shell, starts a new process session with a minimal fixed
environment, and has stdin and stdout closed. A drain thread prevents stderr backpressure while
retaining no more than 64 KiB for bounded diagnostics.

Before leaving the held-input context, it also compares every pre-timing cache row with the
schedule's environment-to-mask assignment. A live OPA process may select only the exact mask and
authorized count registered for that environment; selecting another valid catalog mask aborts the
attempt.

The document paths and open file descriptors are checked again after the online matrix returns.
An in-place mutation or same-shape path replacement raises before any result file is published.
The attempt remains, so the failed package cannot be rerun under the same registration.

The result writer runs only after OPA has terminated and its private data directory has been
removed. A cleanup timeout triggers forced termination. A surviving process, stuck stderr drain,
or scratch-removal error blocks every result file.

The runtime-plan digest passed to this Python boundary must equal the corpus-specific plan already
bound by the suite `OPENED` state. Its runtime receipt is created by the same noninteractive Linux
process before workload-source I/O. `complete_online_suite` later checks five separate plan/receipt
pairs and durable invocation markers, one for each fixed corpus. Caller-supplied pins alone are not
independent freeze evidence. The external state transition and launcher evidence supply that
separate evidentiary chain.

The object-level engineering helper has no runtime gate and stamps zero runtime-attestation
digests into its attempt record. Such an attempt is intentionally ineligible for suite completion.

### Canonical command closure

The image now exposes one corpus command:

```text
/opt/venv/bin/python -m fractal_ann_diagnostics.cli run-sealed-corpus \
  --config /input/control/corpus-run-config.json
```

That config path is its sole option. The launcher writes the freshly verified canonical
`RuntimeClaimReceipt` bytes to the container's standard input; the process validates them before
loading the config. There is no flag for a corpus root, seed, policy revision, feature row,
partition, action, K, controller, endpoint, binary, retry, or config digest. The
runtime-attestation plan must contain that exact six-element argv. A different ordering,
executable, module path, config path, or trailing argument fails the live process check. The plan's
`workload_sha256` pins the config bytes rather than adding a caller-supplied digest argument.

`fractal-production-corpus-run-config-v2` is a closed canonical JSON file. It names the immutable
source paths, three source-package receipt hashes, five pre-existing control-file hashes, and the
three typed `RuntimeFeatureBinding` rows. It does not contain K, action, partition, controller
settings, permutation seed, policy version, trial IDs, or execution digest. The runner derives those
facts from fixed code, the sharded plan, and the trial-runtime receipt.

The control directory is flat and has exactly these entries:

```text
corpus-run-config.json
online-custody-admission.json
required-artifact-bindings.json
runtime-attestation-plan.json
sealed-run-receipt.json
sharded-online-execution-plan.json
trial-runtime-admission-receipt.json
```

At process entry, the output directory must be empty. Successful same-process runtime attestation
then leaves exactly:

```text
runtime-attestation-receipt.json
runtime-invocation-marker.json
```

A pre-existing file, subdirectory, symbolic link, hard link, alternate config location, or
noncanonical filename is fatal. The empty-directory check makes a completed or failed prior runtime
attempt visible before a second call can construct the Linux probe. After attestation, an absent
marker or receipt and any extra entry are also fatal. Once the remaining control closure is typed
and cross-bound, the command creates `sealed-corpus-command-attempt.json` with `O_EXCL`. That marker
binds the config, manifest, workload, runtime plan, and runtime receipt before the inner live
reobservation or workload-source I/O. It remains after any later failure. The inner corpus-attempt
writer also uses `O_EXCL`, so a race cannot turn either replay check into an overwrite.

A successful production directory contains those three control and consumption files, the sealed
attempt and result receipts, and the six result-pinned artifacts. Suite closure admits exactly
those eleven names, parses the command marker through its closed schema, and checks its config,
manifest, plan, receipt, and workload bindings. Any missing or twelfth entry is fatal.

`RequiredArtifactIdBindings` now has a canonical file schema and no-follow loader. Its single file
contains the nested artifact-verification receipt, execution artifact ID, runner and source ID
sets, retriever ID set, and component-to-artifact map. Unknown fields, duplicate keys, alternate
JSON bytes, noncanonical ordering, links, and digest drift are rejected.

`load_trial_runtime_admission` reconstructs the lazy `TrialRuntimeAdmission` from the canonical
sharded plan and trial-runtime receipt. It verifies the plan digest, query-store digest, partition
digest, permutation seed, query count, exact block-key order, and canonical hash of every feature
binding. It records the immutable source paths but does not open a query row, schedule row, matrix,
policy object, corpus artifact, or secret. Those sources remain unopened until runtime attestation
has passed and the corpus attempt file exists.

The pin chain is intentionally acyclic:

1. The external suite `OPENED` state pins one runtime-attestation plan per fixed corpus.
2. The plan's `workload_sha256` equals the exact config-file SHA-256, and the sixth and final argv
   element repeats the config path. The digest is not an argv element.
3. The config file pins the five controls that exist before config construction.
4. The runtime receipt binds the plan digest and the invocation-marker digest.
5. The corpus attempt binds the runtime plan, runtime receipt, trial-runtime receipt, required
   artifact closure, custody receipt, source-package receipts, and sealed run.

The config cannot also hash the runtime plan: the plan already hashes the config. Adding the reverse
edge would require a cryptographic fixed point. Independent plan custody belongs to the external
suite state, not to a self-referential local file.

## Execution contract

`run_online_action_matrix` accepts:

- one already admitted online execution object;
- the sealed-run receipt and verified provenance registry;
- an exclusively owned `GovernedRetriever` with the frozen `RuleController`;
- exactly one `OnlineTrialRuntime` per opaque trial key;
- the registered policy revision, query-partition-audit digest, permutation seed, and pseudonym
  key; and
- an optional timestamp factory for a pinned runtime clock adapter.

Each trial runtime contains two precomputed float32 query vectors: the active query from the stale
model revision and the current-truth query from the current revision. It also carries a finite
policy environment and four non-derived covariates: version lag, backend identity, drift family,
and policy complexity. The runtime owns separate immutable copies of both vectors and stores the
environment as canonical JSON. Missing, extra, non-finite, shared, writeable, digest-changed, or
dimensionally wrong queries fail before the retriever is modified.

The query epochs have distinct execution roles. Every HNSW probe, low-effort search, and
high-effort search receives the active query. `exact-authorized` receives the current-truth query.
When no migration is being studied, direct `GovernedRetriever.query` callers may omit
`current_truth_query`; the retriever then uses its active-query snapshot for both roles. The sealed
online runner requires both vectors explicitly.

The runner checks that the execution, provenance registry, policy decision point, and retriever
share the same ordered document-universe digest. For an inline execution it recomputes that digest
from the document rows. For a sharded execution it admits only the pinned document count and
ordered-universe digest; no document sequence is requested or materialized. The retriever's
original controller and policy object are restored on every return or exception.

## One selection, four measured actions

The first action in the seeded order performs the ordinary authorized probe. A controller wrapper
calls the frozen controller once and retains that exact `ControllerDecision`. It then returns the
decision assigned to the current matrix cell. When the assigned action equals the selected action,
the original decision object is retained. The other three cells receive explicit counterfactual
decisions with the frozen risk score and policy revision.

The registered panel action set remains:

```text
hnsw-low, hnsw-high, exact-authorized, abstain
```

All four cells must exist. `exact-authorized` must complete. A non-exact backend exception becomes
one of the five registered failure codes and emits no IDs or audit claim. Authorization drift,
decision replay, missing selection evidence, or an unexpected governed abstention aborts the
matrix. Such events are not relabeled as backend failures.

Every completed retrieval and the governed abstention receive a new `AuditRecord`. Records form one
chain in actual execution order across all trials. Request and trace IDs are SHA-256 bindings of
the execution artifact, opaque trial key, query binding, assigned action, and actual ordinal.

## Pre-timing authorization cache

The runner prepares each distinct trial environment before any action timer starts. A live policy
decision selects the mask, the exact authorized slice is materialized, and the matching
receipt-verified HNSW object is loaded. The retriever retains all prepared masks rather than only
the last one, then seals the cache against later misses.

`CachePreparationReceipt` records the execution and run-receipt bindings, document universe, role,
policy action, expected policy revision, and one row per environment. Each row carries the
environment SHA-256, raw Boolean-mask SHA-256, and authorized count. Timed actions must reuse these
objects. The runner rejects a rebuild, a new mask, or a prepared mask attached to another
environment. The sealed result pins the cache receipt as a separate pre-label output.

## Portable balanced action positions

The panel keeps two different integers. `action_order` is the registered position in the fixed
action set. `execution_position` is the zero-based ordinal at which that action actually ran for
the trial. Conflating them would erase the evidence needed to diagnose warm-service carryover.

`sha256-ranked-family-latin-square-v1` first ranks the four action identities, query families, and
trials inside each family from canonical SHA-256 inputs containing the frozen permutation seed and
execution-artifact digest. It then assigns successive cyclic rows of the resulting four-by-four
Latin square to the ranked trial stream. Each action occupies each execution position either
`floor(n/4)` or `ceil(n/4)` times across the corpus and inside every query-family block. The latter
is the finest pre-outcome blocking unit available to the runner. Observed geometric descriptors do
not define the schedule because they are produced by the first timed probe.

The detached `ExecutionOrderReceipt` records every actual action sequence and declares the corpus
and query-family balance units. Reconstruction checks the entire receipt at once; a row cannot be
validated as an independent per-trial permutation. The procedure does not depend on NumPy's random
generator, Python's hash seed, or mapping iteration order. The receipt also binds the canonical
query text digest, separate
little-endian float32 digests for the active and current-truth query vectors, policy-environment
digest, family key, and derived query-binding digest. The role-named fields prevent a stale/current
swap from preserving the query binding.

The execution-order receipt binds the cache-preparation receipt SHA-256. Its writer emits canonical
JSON plus one terminal newline through the same
exclusive, no-follow receipt writer used elsewhere. Loading rejects duplicate keys, extra fields,
noncanonical bytes, path symlinks, a changed permutation, or a changed query binding.

## Registered feature row

Only the `hnsw-low` panel row carries predictive features. Values are emitted in
`REGISTERED_FEATURE_SCHEMA.input_features` order:

| Position | Feature | Source |
|---:|---|---|
| 1 | `corpus_size` | retriever vector count |
| 2 | `authorized_universe_size` | first frozen controller call |
| 3 | `embedding_dimension` | retriever vector width |
| 4 | `version_lag` | frozen trial context |
| 5 | `drift_severity` | per-query `1 - cosine(active query row, current query row)` |
| 6 | `probe_latency_ms` | bounded authorized probe |
| 7 | `probe_work` | measured distance evaluations, then visited candidates; missing if neither exists |
| 8–10 | `corpus_stratum`, `backend`, `drift_family` | execution and frozen trial context |
| 11 | `allow_rate` | authorized selectivity |
| 12 | `policy_complexity` | frozen trial context |
| 13 | `policy_churn` | exact Hamming fraction from the environment's complete baseline/current masks |
| 14–17 | `lid_k50`, `lid_cv`, `relative_contrast`, `radius_expansion` | first authorized probe geometry |

Configured `efSearch` is not recast as measured work. When the backend exposes neither visits nor
distance evaluations, `probe_work` remains missing and serializes through the panel's registered
missing-number representation.

The closed production entrypoint accepts neither feature as a scalar. Query drift is recomputed from
the two receipt-bound query rows. Policy churn is recomputed from the baseline mask stored outside
the OPA catalog and the current mask selected by the live environment. The attempt receipt binds the
query row orders, vector epochs, schedule, embedding receipt, and policy-intervention receipt. A
missing environment, duplicate transition, changed mask, wrong revision, aggregate-only percentage,
or nonfinite query row aborts the run.

## Outputs and anchoring

`OnlineRunArtifacts` returns:

- the admitted action panel and its detached admission receipt;
- the execution-order receipt;
- the complete ordered audit chain;
- one selected decision per trial; and
- any admitted non-exact failures.

Its `anchoring_digests` property provides the action-panel digest, admission-receipt digest,
execution-order-receipt digest, and audit-chain head. A production writer should persist all four
objects immutably before asking the external completion-anchor service to attest them.

The runner does not create a completion anchor itself. It also does not open the one-shot receipt;
that authority remains with the existing sealed-run admission path.

## Mechanism-level invocation

Direct object construction is reserved for mechanism tests and development diagnostics:

```python
from fractal_ann_diagnostics.online_runner import OnlineTrialRuntime, run_online_action_matrix

runtime = OnlineTrialRuntime(
    active_query_vector=old_query_vectors[row],
    current_truth_query_vector=current_query_vectors[row],
    feature_context=feature_context,
    environment=policy_environment,
)

outputs = run_online_action_matrix(
    execution=execution,
    run_receipt=run_receipt,
    retriever=retriever,
    provenance_registry=provenance_registry,
    trial_runtimes=trial_runtimes,
    permutation_seed=20260714,
    expected_policy_version="policy-registered",
    query_partition_audit_sha256=query_partition_audit_sha256,
    pseudonym_key=pseudonym_key,
    pseudonym_key_id="online-runner-key-1",
    k=10,
)
```

The production callable is deliberately narrower:

```python
from fractal_ann_diagnostics.sealed_online_execution import run_sealed_online_once

persisted = run_sealed_online_once(
    output_root="/sealed/results/corpus-a",
    admission_receipt=online_custody_admission,
    required_artifacts=required_artifact_bindings,
    run_receipt=sealed_run_receipt,
    runtime_admission=trial_runtime_admission,
    runtime_attestation_plan_path="/sealed/control/runtime-plan.json",
    expected_runtime_attestation_plan_sha256=frozen_runtime_plan_sha256,
    runtime_attestation_receipt_path="/sealed/results/runtime-receipt.json",
    expected_runtime_attestation_receipt_sha256=frozen_runtime_attestation_receipt_sha256,
    expected_runtime_receipt_sha256=frozen_runtime_receipt_sha256,
    artifact_root="/sealed/execution-package",
    authorized_index_store_root="/sealed/authorized-indexes",
    expected_authorized_index_store_receipt_sha256=frozen_index_receipt_sha256,
    policy_intervention_root="/sealed/policy",
    expected_policy_intervention_receipt_sha256=frozen_policy_receipt_sha256,
    pseudonym_key_path="/run/secrets/audit-pseudonym.key",
    expected_pseudonym_key_sha256=frozen_pseudonym_key_sha256,
)
```

Run the bounded tests with:

```bash
uv run pytest tests/test_online_runner.py -q
```

They cover the fixed portable permutation vector, all four actions, dual-epoch query routing and
digest evidence, feature order, complete audit linkage, canonical receipt loading, symlink refusal,
timeout admission, mandatory exact completion, policy-mask drift, query mutability drift, inexact
query maps, and the absence of protected-data imports or paths.
