# Sealed query/trial runtime

The corpus is too large to place in an online execution object. Query data is
small enough to admit in memory, but only after its staged source, embedding
row, opaque identifier, policy assignment, and model epoch have been joined
without outcome data. `trial_runtime.py` performs that join.

## Two-phase construction

The custodian first calls `build_query_trial_store` with the registered
selection-seed digest, expected available-family count, selected-family count,
and `nested_rows_per_family=3`. The builder reads the verified embedding row
order and checks every sealed row against the staged query file and
`assignments.jsonl`. It then applies the same outcome-blind family and
representative ranking functions as the separated custody builder. One
representative query per selected assignment component expands to three
runtime rows. Those rows retain one embedding source row and one query text,
but receive distinct nested trial keys.

Trial and family identifiers use the label-separation v2 contract:
length-prefixed HMAC-SHA-256 over the v2 domain, key ID, corpus, stage, and
source value. The family source is
`assignments.jsonl.partition_component_sha256`. Each trial source is the
canonical JSON value
`["fractal-custody-nested-trial-v1", query_id, nested_index]`. The build-time
secret is absent from every published file. The shared functions in
`query_cohort.py` prevent the runtime and custody implementations from
silently diverging.

Each `query-trials.jsonl` record contains the query text, corpus, stage, local
query row, opaque keys, and its source binding. It does not contain the source
query ID. Instead, the record pins its SHA-256 digest, source path, source
line, source-file digest, and embedding row. The SHA-256 of the full canonical
JSONL line becomes `OpaqueTrialRow.query_record_sha256`; the local row becomes
`OpaqueTrialRow.query_row`.

The query package is published with an exclusive directory rename. An existing
target is never replaced. Its v3 receipt pins both ranking algorithms, the
selection-seed digest, available and selected family counts, the three-row
cardinality, and the identity
`record_count = selected_family_count * nested_rows_per_family`. The receipt
also requires one contiguous three-row block per opaque family. It can produce
the `QueryTrialStoreDescriptor` and opaque rows required to construct a
`ShardedOnlineExecutionPlan`. The descriptor has separate immutable file pins
for `query-trials.jsonl` and `query-trial-receipt.json`; both become direct
leaves of the v4 plan and its verification receipt.

After the sharded plan and policy schedule have been frozen,
`admit_trial_runtime` checks these equalities:

- plan key ID, corpus, stage, query store pin, query receipt pin, and opaque rows;
- schedule execution digest, document count, document-universe digest, and
  policy revision;
- exactly three trials per opaque family and one trial from each family in each
  policy-state block;
- a disjoint union of the three blocks equal to the plan trial keys;
- one frozen `FrozenFeatureContext` for every such block;
- staged inventory, embedding source inventory, assignment store, query row
  order, query vectors, query source files, query/trial file, plan, and
  schedule digests.

The resulting admission receipt binds the assignment algorithm, assignment
seed digest, canonical trial-to-state map digest, schedule digest, and block
order. It has no action permutation field. The online runner still derives the
within-trial action order from its separately frozen permutation seed.

## Separate migration and truth epochs

Every admitted query has two vectors:

| Runtime field | Embedding matrix | Scientific use |
|---|---|---|
| `active_query_vector` | `old_queries` | Query the stale migration index |
| `current_truth_query_vector` | `current_queries` | Compute current-space exact truth |

The receipt pins each epoch independently: vector-file SHA-256, row-order
SHA-256, model-tree SHA-256, immutable model revision, prompt SHA-256, dtype,
and shape. The two epochs must share query rows, dtype, and dimension. They
must have different model identities and different vector bytes. Reusing one
query vector for both paths is rejected.

`load_trial_runtime_block` rechecks the sources, loads both `.npy` matrices
through no-follow, single-link reads, and returns read-only float32 copies in
`OnlineTrialRuntime`. Each block contains its assigned third of the plan. Its
query-bearing execution adapter retains the full plan digest while exposing
only that block's trials.

`load_trial_runtime` loads the three admitted blocks, rejects any repeated or
missing trial, and combines them into one full `ShardedQueryExecutionAdapter`
and runtime map. The result preserves the plan digest and is the object passed
to one `run_online_action_matrix` call.

## Excluded data and filesystem rules

The module accepts no qrels, answers, judgments, supporting facts, evidence
bundles, or decrypted custody path. Closed schemas reject fields whose names
carry those meanings. Every control or source file is read through an absolute
no-follow path, must be a single-link regular file, and must remain stable for
the duration of the read. Canonical JSON rejects duplicate keys and non-finite
numbers.

Package directories and admission receipts are write-once. Query-package
publication uses `renameatx_np(RENAME_EXCL)` on macOS or
`renameat2(RENAME_NOREPLACE)` on Linux. Receipt creation uses `O_EXCL`.

## Production sequence

1. Build and verify the staged study data and dual-model embedding store.
2. Build the query/trial package with the registered cohort parameters and
   sealed HMAC key, then remove that key from the online runner environment.
3. Construct and pin the `ShardedOnlineExecutionPlan` with the package receipt.
4. Independently build the custody package with the same registered cohort
   parameters and plan digest. Require `verify_query_trial_key_parity` to pass.
5. Compile the policy intervention and freeze its schedule.
6. Admit the plan, schedule, and registered feature contexts; publish the
   canonical admission receipt.
7. Load all three disjoint runtime blocks in receipt order, combine them once,
   and call the online action-matrix runner once. The runner derives the
   corpus- and query-family-balanced within-trial Latin schedule.

Any digest change, omitted query, cohort-count mismatch, repeated trial,
noncontiguous or incomplete three-row family, model-epoch collapse, link
substitution, or existing publication target stops execution before the first
retrieval call.
