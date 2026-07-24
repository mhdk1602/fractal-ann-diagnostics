# Label-payload-excluded development cohort ranking

## Why this boundary exists

Development labels are allowed to shape the frozen models, controller, and power inputs. They are
therefore a source of analytic discretion. If an engineer can inspect qrels, answers, or evidence
before choosing the development queries, the apparent fit/calibration split no longer constrains
that discretion.

[`development_cohort.py`](../src/fractal_ann_diagnostics/development_cohort.py) divides the operation
into two commands:

1. `select` fixes qrel-derived assignment components and representatives without opening a
   label-bearing file.
2. `materialize` reproduces the selection, verifies the paired embedding bindings, then filters the
   development qrels and evidence.

Both commands use exclusive creation. Neither replaces an existing receipt or output directory.
The assignment components and partition-audit bindings were constructed earlier using registered
shared-positive-document edges from qrels. This boundary constrains direct payload exposure and
post-selection discretion. It does not establish outcome-independent cohort construction or human
blinding.

## Registered allocation

The allocation is fixed in code:

| Source stage | Output stage | Families per corpus | Selection seed SHA-256 |
|---|---|---:|---|
| `fit` | `development-fit` | 200 | `b4ce31a68caf104a0a81a8e3d2745ac91b980b269ddc14417c4fbe15cb34a33f` |
| `calibration` | `development-calibration` | 75 | `287cfbc31f6108a0cd3a244826db49cb828b218c243358e13cc6686f901a1617` |

These denominators apply separately to SciFact, HotpotQA FullWiki, T2-RAGBench, BRIGHT, and MIRACL
Transfer. The selector computes the number of assignment components actually present in each
stratum. It aborts when that number is below the registered denominator; it does not duplicate a
family or silently reduce the count.

## Phase 1: select

The selector may open only:

- `inventory.json` and `inventory.sha256`;
- the typed scalable partition-audit receipt;
- `assignments.jsonl`; and
- the ten fit/calibration query files.

The inventory and partition audit supply exact SHA-256, byte-count, record-count, corpus, stage,
role, and visibility pins for those files. The selector checks that the assignment and query pins
are identical in both sources. The audit's query denominators must equal the inventory record
counts.

The selector has no qrel or evidence reader in this phase. A sentinel test replaces the shared
JSONL iterator and raises if a `qrels` or `evidence-bundles` role is requested. Selection completes
with the sentinel installed.

For corpus (c), development stage (s), fixed seed (z_s), and assignment component (g), the
family rank is the shared `query_cohort.family_selection_rank` digest:

```text
sha256-rank-v1(c, s, z_s, g)
```

Components are ordered by `(family_rank_sha256, component_sha256)`. The first 200 or 75 are kept.
Within each kept component, every candidate query receives the shared representative rank:

```text
sha256-rank-v1(c, s, z_s, g, sha256(query_id))
```

The minimum `(representative_rank_sha256, query_id_sha256)` is the sole representative. Query text
is checked against the assignment ledger before ranking. The receipt records the text digest, not
an outcome.

The canonical `fractal-development-cohort-selection-v1` receipt binds:

- the staged inventory and assignment ledger;
- the typed partition audit and its component-membership digest;
- all ten query-file pins;
- the fixed algorithms and seeds;
- each available component denominator; and
- every selected component, representative ID, text digest, and reproduced rank.

## Phase 2: materialize

Materialization accepts the selection receipt only by an external SHA-256 pin. Before it resolves a
qrel or evidence source, it performs these checks in order:

1. load the receipt through a no-follow regular-file read;
2. reproduce the entire receipt from inventory, assignments, queries, and the typed audit;
3. require byte equality with the published receipt;
4. verify ten exact paired old/current embedding stores;
5. require each embedding receipt to name the same staged inventory;
6. require each selected query ID to occur once in the pinned query row order; and
7. bind the document count and document-universe digest from the paired receipt.

Only then does the code resolve the exact fit/calibration qrel and evidence pins from the inventory
and partition audit. It reads no sealed-stage path. Every selected query must have at least one
positive qrel. The three evidence corpora must have exactly one evidence row per selected query.

## Materialized package

The package contains no vector matrix:

```text
materialization-receipt.json
selection-receipt.json
development-fit/
  <corpus>/
    execution-plan.json
    queries.jsonl
    qrels.jsonl
    evidence-bundles.jsonl        # three evidence corpora only
development-calibration/
  <corpus>/
    execution-plan.json
    queries.jsonl
    qrels.jsonl
    evidence-bundles.jsonl        # three evidence corpora only
```

`fractal-development-cohort-materialization-v1` pins every emitted byte and all ten paired
embedding receipts. It also carries the selection and partition-audit digests. Extra files,
missing files, links, and special files are rejected. The receipt itself is closed over exactly 37
artifacts: one embedded selection, ten query files, ten qrel files, ten execution plans, and six
evidence files. A rehashed receipt cannot omit a stratum, rename a role, or relax the registered
record denominator.

`verify_materialized_development_cohort(..., verify_label_payloads=False)` is the online admission
path. It verifies exact tree membership, regular-file sizes, and all query, plan, selection, and
receipt bytes without opening qrels or evidence. It parses the embedded selection, requires the
same inventory and partition-audit pins, checks query IDs and text digests, and proves that each
plan names the selected components, representatives, and embedding receipt. The materializer
itself calls the same verifier with `verify_label_payloads=True` after label access has already
passed the phase boundary. That mode also checks label record counts, selected-query coverage, and
the positive-qrel condition.

## Execution plans

Each `fractal-development-execution-plan-v1` object exposes the exact interface consumed by
`compile_policy_intervention`:

- `corpus` and development `stage`;
- `document_count` and `document_universe_sha256`;
- `artifact_sha256`; and
- `trials`, where each row exposes `family_key` and `trial_key`.

The plan adds the embedding receipt, document row order, query row order, selection receipt,
selected-family count, representative query ID, and embedding query-row index. A family has exactly
three rows with nested indices 0, 1, and 2. Trial keys are deterministic length-prefixed SHA-256
digests over the plan domain, corpus, development stage, component, representative ID, and nested
index. The loader recomputes every trial key and requires 200 distinct fit query rows or 75 distinct
calibration query rows. A missing, repeated, or fourth nested row is invalid.

The plan points into the paired embedding store. It does not copy current or stale vectors. This
keeps one vector identity across cohort materialization, policy compilation, paired action
execution, and the development freeze.

The next admitted step is the
[`development_execution.py`](../src/fractal_ann_diagnostics/development_execution.py) runner. Its
[operator contract](development-execution.md) executes the four registered actions for all 4,125
planned trials without opening qrels or evidence, then binds the resulting 16,500 action rows back
to these plans and the exact selection receipt.

## CLI

Publish the label-payload-excluded receipt:

```bash
python -m fractal_ann_diagnostics.development_cohort select \
  --staged-root /controlled/study-data-v2 \
  --staged-inventory-sha256 <inventory-sha256> \
  --partition-audit /controlled/suite-partition-audit-v2.json \
  --partition-audit-sha256 <audit-sha256> \
  --output /controlled/development/selection-receipt.json
```

An installed package exposes the same interface as `fractal-development-cohort`.

Materialize after the receipt has been recorded:

```bash
python -m fractal_ann_diagnostics.development_cohort materialize \
  --staged-root /controlled/study-data-v2 \
  --selection-receipt /controlled/development/selection-receipt.json \
  --selection-receipt-sha256 <selection-sha256> \
  --partition-audit /controlled/suite-partition-audit-v2.json \
  --embedding-bindings /controlled/development/embedding-bindings.json \
  --output-root /controlled/development/materialized-v1
```

The embedding config is canonical JSON with schema
`fractal-development-embedding-bindings-v1`. Its `bindings` array contains one exact
`{corpus, development_stage, root, receipt_sha256}` object for each of the ten strata.

Online code can verify package closure without label reads:

```bash
python -m fractal_ann_diagnostics.development_cohort verify-materialization \
  --root /controlled/development/materialized-v1 \
  --expected-sha256 <materialization-receipt-sha256>
```

Add `--verify-label-payloads` only in an operation already authorized to read development labels.

## Freeze binding

`DevelopmentFreezeConfig` schema v2 contains a required top-level `selection_receipt` file pin.
The compiler verifies its path, byte count, SHA-256, canonical schema, and selected query IDs before
it opens qrels. Each materialized query file must equal that receipt's representative set for the
same corpus and stage. A caller cannot replace the queries, policy schedule, or paired actions with
an outcome-chosen set while retaining the original selection pin.

Do not construct the freeze config directly after cohort materialization. Run the
label-payload-excluded paired execution first, then use
`fractal-development-execution write-freeze-config`. That command derives every query, qrel,
evidence, schedule, action-panel, embedding, and selection pin from the closed receipt chain.
