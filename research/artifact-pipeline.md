# Five-corpus online artifact pipeline

The online artifact pipeline closes the packaging boundary before the C1 freeze. It admits one
label-free staging projection, visits the five registered corpora in their fixed order, verifies
each artifact transition, and emits one canonical suite receipt. It does not read qrels, answers,
evidence bundles, plaintext custody material, or timelock ciphertext.

The four-step order is fixed:

```text
embedding store
  -> fit | calibration | sealed policy packages
  -> fit | calibration | sealed authorized-index stores
  -> sealed query/runtime package
```

The corpus order is `scifact`, `hotpotqa-fullwiki`, `t2-ragbench`, `bright`, then
`miracl-transfer`. A receipt with a different order, a missing corpus, or an extra corpus is
invalid even if its directory digests are otherwise correct.

## Controlled layout

The command expects these paths below `--artifact-root`:

```text
embedding-stores/
  <corpus>/
policy-workloads/
  <corpus>/
    fit/
    calibration/
    sealed/
    stage-bundle.json
authorized-index-stores/
  <corpus>/
    fit/
    calibration/
    sealed/
    stage-bundle.json
trial-runtime/
  <corpus>/
    query-package/
      query-trials.jsonl
      query-trial-receipt.json
    sharded-online-execution-plan.json
    trial-runtime-admission-receipt.json
```

There is one `<corpus>` directory for each fixed corpus. Each policy and index bundle contains
exactly the three named stage directories and one canonical bundle receipt. Symlinks, special
files, undeclared directories, additional files, duplicate JSON keys, non-finite numbers, and
non-canonical receipt bytes fail verification.

The wrapper names map to source partitions without ambiguity: `fit` must carry a
`development-fit` policy receipt, `calibration` must carry `development-calibration`, and `sealed`
must carry `sealed`. Renaming one source package into another stage directory cannot pass.

### Policy stage binding

For every stage, the bundle verifier loads the policy config, compiled-mask catalog, OPA data,
trial schedule, intervention receipt, and mask bytes. It then checks:

- corpus, stage, execution artifact, document universe, document count, policy revision, and
  config digest agree across the typed objects;
- the catalog, OPA assignments, and schedule contain the same mask IDs;
- every file named by the intervention receipt has the recorded byte count and SHA-256;
- the stage directory has no membership beyond the typed package;
- `fit`, `calibration`, and `sealed` bind distinct execution artifacts.

`stage-bundle.json` records the reproduced tree digest and typed digests for each stage. It is
written with exclusive-create semantics. If it already exists, `build` verifies it instead of
replacing it.

### Authorized-index stage binding

Each authorized-index stage is tied to the shared dual-model embedding store and to the matching
policy stage. The verifier checks the config and receipt, hashes every HNSW payload and row map,
loads every row map as a non-pickle NPY array, and requires a strictly increasing in-range global
row sequence.

The index set must equal the compiled mask catalog. For every mask, the mask digest, authorized
count, canonical HNSW path, canonical row-map path, and deterministic build-binding digest are
recomputed. The store also has to bind the exact embedding receipt, current and stale document
vectors, policy receipt, policy catalog, execution artifact, document universe, and document row
order.

This wrapper does not rerun HNSW queries. The authorized-index builder already does that before it
publishes a store. C1 pins the resulting exact directory tree; the online provider repeats its own
backend-aware verification before use.

## Build the suite receipt

Build the five paired-Qwen embedding stores with the production controller first. It derives every
source path from the admitted staging inventory and resumes from per-corpus checkpoints:

```bash
fractal-production-embeddings build \
  --config /controlled/fractal-v0.3/production-embeddings.json \
  --config-sha256 <config-sha256>

fractal-production-embeddings verify \
  --config /controlled/fractal-v0.3/production-embeddings.json \
  --config-sha256 <config-sha256>
```

The production embedding config must set its `output_root` to
`<artifact-root>/embedding-stores`. The final embedding-suite receipt and five resource-evidence
files remain beside the corpus directories; the artifact pipeline reads only the five closed store
directories.

Next build the three policy packages, three authorized-index stores, and sealed runtime package for
each corpus through their typed builders. Those directories must match the controlled layout above.
Then seal the stage wrappers and verify the full chain:

```bash
fractal-artifact-pipeline build \
  --artifact-root /controlled/fractal-v0.3/artifacts \
  --online-staging-root /controlled/fractal-v0.3/artifacts/study-data/online-projection \
  --expected-online-inventory-sha256 <64-hex-digest> \
  --receipt /controlled/fractal-v0.3/artifact-pipeline.json
```

`build` is a package-finalization operation, not a model-training command. It verifies the
embedding store first, seals or verifies the policy bundle second, seals or verifies the index
bundle third, and checks the flat runtime package last. If a later corpus fails, a rerun verifies
any stage receipts already written and resumes at the same deterministic boundary. The suite
receipt itself is exclusive-create and is never overwritten.

The receipt records:

- the admitted online inventory and projected artifact-set digests;
- the fixed corpus and artifact orders;
- each embedding tree and receipt digest;
- each policy and index bundle tree and receipt digest;
- each runtime tree, sharded plan, query receipt, runtime admission receipt, and query count.

Runtime admission is checked against the sealed policy assignment seed and assignment-map digest,
the plan's query-partition audit and permutation seed, and both complete query-vector epoch
bindings. A matching vector-file hash with a different model, prompt, shape, dtype, or row order is
not sufficient.

## Reproduce the receipt

Verification performs no writes:

```bash
fractal-artifact-pipeline verify \
  --artifact-root /controlled/fractal-v0.3/artifacts \
  --online-staging-root /controlled/fractal-v0.3/artifacts/study-data/online-projection \
  --receipt /controlled/fractal-v0.3/artifact-pipeline.json
```

The verifier first re-admits the online staging projection against the inventory digest in the
receipt. It then reproduces every corpus row. A different byte, typed binding, member path, corpus
order, or receipt byte causes a closed failure.

## Label boundary

The online staging verifier accepts the projection receipt and its declared non-label artifacts.
The runtime package has an exact five-entry contract: one query-package directory, its two files,
and the two root control files. A file such as
`sealed-labels.json`, a qrels file, an evidence bundle, or a custody receipt is therefore an extra
member and fails before any runtime control file is parsed.

The command line has no label path, custody path, decryption key, timelock tool, network endpoint,
or generic callback option. It cannot release labels or open a sealed attempt. Those operations
remain in the separately attested custody and suite-attempt state machines.

## Freeze integration

The freeze compiler now treats policy workloads, embedding stores, authorized-index stores,
online execution packages, query partition audits, and the joint-power bundle as typed artifacts.
For policy and index directories, it invokes the same three-stage verifiers described here and
records the bundle receipt digest as the observed revision. A tree with plausible bytes but an
invalid stage binding does not reach `present` state.

C1 should pin both the exact outer directory-tree digests and the revisions returned by typed
inspection. The suite pipeline receipt is supporting evidence for those pins; it does not replace
the study manifest, registration record, artifact map, runtime admission, or one-shot attempt
ledger.
