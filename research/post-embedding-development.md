# Post-embedding development operator

The post-embedding operator closes the gap between the five verified paired-Qwen stores and the
production artifact factory. It performs the full development sequence under one typed receipt:
cohort selection, development-label materialization, policy compilation, authorized HNSW builds,
paired execution, model freeze, and joint-power design.

The implementation is
[`post_embedding_development.py`](../src/fractal_ann_diagnostics/post_embedding_development.py).
The installed command is `fractal-post-embedding-development`.

## Boundary

The canonical config contains exactly eight operator-controlled values:

| Field | Meaning |
| --- | --- |
| `production_embedding_config_path` | Exact canonical config used to build all five embedding stores |
| `production_embedding_config_sha256` | File SHA-256 for that config |
| `full_staged_root` | Full staged-data root used for development selection and labels |
| `full_staged_inventory_sha256` | Exact `inventory.json` SHA-256 |
| `partition_audit_path` | Canonical query-partition audit path |
| `partition_audit_file_sha256` | Exact canonical audit-file SHA-256 |
| `design_seed_sha256` | Single seed from which policy identities are derived |
| `output_root` | New private operator root |

Paths must be absolute, normalized, free of symbolic-link aliases, and pairwise disjoint. The
operator rejects path tokens that denote sealed, custody, held-out, label, outcome, or result
boundaries. Its interface has no path or parameter for a sealed label, confirmatory outcome,
runtime callback, plugin, or alternate policy/index config.

The full staged root is admitted because development qrels and evidence are required after the
selection gate. The label-free selection receipt is reproduced byte for byte before the
materializer resolves or opens those development label sources. Sealed qrels remain outside every
development type and path admitted downstream.

## Fixed derivation

The operator first verifies the production embedding config and rehashes all five stores through
`verify_production_embedding_suite`. The production suite, staged inventory, and partition audit
must name the same inventory digest. The canonical audit's semantic digest must equal its exact
file digest.

Each corpus store supplies two `DevelopmentEmbeddingBinding` rows, one for
`development-fit` and one for `development-calibration`. Both rows name the same store root and
receipt. There are ten rows, no per-stage embedding copies.

For each stratum, policy identity comes only from:

```python
derive_production_policy_config(design_seed_sha256, corpus, source_stage)
```

The source stages are `fit` and `calibration`. The policy compiler receives the exact materialized
execution plan. An existing package is recompiled and compared byte for byte, so a hand-authored
policy, schedule, mask, or receipt is inadmissible.

Authorized indexes come only from `production_authorized_index_components()`. That factory-owned
helper requires the C0 Linux/arm64 runtime, pins the installed `hnswlib` extension bytes, and fixes:

```text
metric              cosine
M                   16
efConstruction      128
random seed         20260714
batch size          512
verification ef     64
builder threads     1
```

The development operator and production factory therefore use one policy derivation and one index
constructor. Drift between a preliminary development index and a factory index becomes a receipt
failure rather than an undocumented implementation choice.

## Execution and freeze

The operator derives `DevelopmentPairedExecutionConfig`; it does not accept one from the command
line. The ten strata contain exactly:

- 1,375 selected development families;
- 4,125 nested policy-state trials; and
- 16,500 paired action rows across the four registered actions.

The execution runner's admitted inputs exclude qrels and evidence. It reads materialized queries
and execution plans, paired embeddings, policies, and authorized indexes. Development qrels and
evidence first re-enter at the freeze compiler. This is a process-input claim; it does not assert
that the administrator is unaware of public benchmark labels. The operator derives the freeze
config from the verified execution receipt, then requires `verify_development_freeze` to
reconstruct its controller, models, profiles, panels, and joint-power config.

## Single joint-power publication

The freeze package must expose a production config with exactly 5,000 bound-calibration
simulations and 5,000 evaluation simulations. `test_mode` is forbidden.

Immediately before the simulator call, the operator exclusively writes
`joint-power-invocation.json`. It binds the freeze-tree digest, config digest, both panel digests,
and `authorized_invocation_count=1`. If the process stops after this marker but before an atomically
published bundle, resume fails permanently. It cannot infer whether the simulator already ran and
will not authorize a second publication attempt.

The published tree is exactly:

```text
analysis/joint-power-design/
  config.json
  report.json
  selection-audit.json
  panels/
    <expected-panel-sha256>.json
    <conservative-panel-sha256>.json
```

Publication uses `renameat2(RENAME_NOREPLACE)` on Linux or
`renameatx_np(RENAME_EXCL)` on macOS. A competing or preexisting target is never replaced.

After a successful new calculation, the operator canonical-loads the published config, audit,
report, and panels, checks closed membership, reruns the deterministic outer design against the
stored audit, and rehashes the tree. The terminal receipt may reuse this same-process generation
readback only through an in-memory token bound to the freeze-tree digest, invocation bytes, bundle
path, and bundle-tree digest. Any intervening mutation invalidates the token. No token is
serialized.

The independent freeze-package verifier then freshly reproduces the exact selection audit and the
deterministic joint-power calculation from the pinned config and panels. It requires byte-identical
audit and report output and rehashes the directory after typed inspection. A successful fresh chain
therefore executes the exact selection calculation twice in total: once to create the audit and
once in the freeze verifier. Resume after a process boundary and every standalone operator `verify`
call still perform one fresh exact reproduction before admitting the terminal receipt. These checks
do not publish a second report or alter the exclusive invocation marker.

## Resume contract

`run` requires an absent output root. `resume` requires an existing private root and the exact
operator-config copy. Resume crosses only a completed immutable boundary:

1. selection receipt;
2. embedding-binding config;
3. materialization package;
4. each policy package in fixed stage/corpus order;
5. each authorized-index package in the same order;
6. paired-execution config and package;
7. freeze config and package;
8. invocation marker and joint-power bundle; and
9. terminal receipt.

Every existing boundary is reverified before the next write. Missing entries may be built only by
`run` or `resume`. `verify` is read-only. It passes `allow_writes=false` at each stage, rejects
missing or extra paths, rejects links and special files, rehashes every stage root, and compares a
freshly reconstructed terminal receipt with the stored bytes.

The terminal schema is `fractal-post-embedding-development-receipt-v1`. It binds the production
embedding suite, both partition-audit digests, selection and materialization receipts, design seed,
fixed index config, all ten policy/index receipts, paired execution, freeze config and tree,
joint-power invocation/config/report/tree, cardinalities, selected family count, and twelve exact
file or directory pins. Its `artifact_sha256` is also the exact SHA-256 of
`post-embedding-development-receipt.json`.

The production factory accepts the operator root plus that one receipt digest. It calls
`verify_post_embedding_development(root, expected_receipt_sha256=...)` and derives its
materialization, design, audit, index, and power inputs from the verified receipt. Those fields are
not repeated as operator choices in the factory command.

## Commands

Write the canonical config after the five embedding stores are final:

```bash
fractal-post-embedding-development write-config \
  --production-embedding-config /controlled/embedding-build-config.json \
  --production-embedding-config-sha256 <sha256> \
  --full-staged-root /controlled/study-data-v2 \
  --full-staged-inventory-sha256 <sha256> \
  --partition-audit /controlled/suite-partition-audit-v2.json \
  --partition-audit-file-sha256 <sha256> \
  --design-seed-sha256 <sha256> \
  --output-root /controlled/development/operator-v1 \
  --output /controlled/development/operator-v1-config.json
```

Run inside the exact C0 Linux/arm64 image:

```bash
fractal-post-embedding-development run \
  --config /controlled/development/operator-v1-config.json \
  --config-sha256 <sha256>
```

After a clean stop at a completed boundary:

```bash
fractal-post-embedding-development resume \
  --config /controlled/development/operator-v1-config.json \
  --config-sha256 <sha256>
```

Read status without opening development labels:

```bash
fractal-post-embedding-development status \
  --config /controlled/development/operator-v1-config.json \
  --config-sha256 <sha256>
```

Reverify the terminal package and deterministic power report:

```bash
fractal-post-embedding-development verify \
  --config /controlled/development/operator-v1-config.json \
  --config-sha256 <sha256> \
  --receipt-sha256 <sha256>
```

An interrupted joint-power marker is terminal. Do not delete it, replace it, or call the simulator
outside this operator. Retain the entire root for the failure record.
