# Scalable staged query-partition audit

This audit is the last process allowed to inspect all staged relevance judgments before label
custody closes. It proves that fit, calibration, and sealed queries occupy disjoint connected
components under the registered partition graph. The output is a canonical, label-free receipt.
The receipt can enter the freeze package and the online runtime without exposing query IDs,
document IDs, relevance values, answers, or evidence.

The audit does not accept a caller's description of the staged cohort. It first verifies
`inventory.json`, `inventory.sha256`, exact package membership, artifact sizes, record counts, and
artifact SHA-256 values. It then securely reopens only inventory-pinned files through a no-follow
path walk. A changed file, added file, symlink, hard link, noncanonical JSON row, or omitted stage
aborts the audit.

## Registered source contract

The audit reads these inventory rows:

- `assignments.jsonl`, online
- every fit and calibration `queries.jsonl` and `qrels.jsonl`
- every sealed `sealed/online/queries.jsonl`
- every sealed `sealed/custody/qrels.jsonl`
- each inline `corpus.jsonl` needed for canonical document-content identities
- the always-present `partition-exclusions.jsonl`, protocol visibility

The sealed qrels are opened here because cross-stage shared-positive-document edges cannot be
proved from query text alone. They are not opened again by the online runtime.

The inventory's assignment declaration must name algorithm `component-ranked-sha256-v2`, policy
`exclude-entire-component-v1`, and these four edges in exact order:

1. `normalized-query-text-equality`
2. `registered-near-duplicate-token-rule`
3. `shared-positive-document-content`
4. `shared-positive-relevance-document`

Changing the declaration changes the audit algorithm digest. A receipt produced under another
partition policy therefore cannot be replayed.

## Recomputed graph

Every query file is parsed. Query IDs must be unique and bytewise sorted within each file. The
auditor recomputes the raw UTF-8 text SHA-256, Unicode NFKC/casefold normalization, alphanumeric
tokens, and normalized-text SHA-256. The result must join exactly to one assignment row. There can
be no extra assignment and no unassigned query.

Assignment components are checked rather than trusted. A component may name one corpus and one
stage only, and its digest must equal SHA-256 of the canonical, bytewise-sorted query-ID array.

The auditor then constructs edges from four independent observations:

- exact normalized-text equality;
- the registered one-token insertion, deletion, or substitution rule;
- shared positive external document identity, encoded as SHA-256 of canonical JSON
  `[dataset, external_document_id]`;
- shared canonical inline document content, encoded as SHA-256 of canonical JSON
  `["suite-global-canonical-document-content-v2", content_sha256]`.

`content_sha256` is the SHA-256 of the title and text after each UTF-8 field is prefixed by its
eight-byte big-endian length. Content identity is independent of corpus name. Two external IDs
with identical canonical content remain one statistical unit even when they occur in different
corpora. Dataset-scoped external ID remains a second, independent join.

Inline `corpus` files and streamed `corpus-shard` files both contribute suite-global content
identity. Shards are scanned in pinned path order and document-ID order; only documents named by a
positive qrel are retained in memory. The dataset-scoped external identity remains an independent
edge for every corpus.

All qrels are parsed, including negative judgments. Every admitted query must appear in qrels and
must have at least one positive judgment. Duplicate query/document pairs, unknown query IDs, a
positive-document edge split across assignment components, or any connected component spanning
more than one study stage aborts the audit.

## Registered structural exclusions

Official source-split semantics are preserved. Rows are not moved from train to development or
from development to test to repair leakage. Instead, staging removes the entire connected
component whenever that component spans source splits.

`partition-exclusions.jsonl` is present even when empty. Each nonempty row uses schema
`fractal-study-query-partition-exclusion-v1`, rule
`source-split-component-isolation-v1`, and reason `cross-source-split-component`. A closed row
contains:

- dataset, query ID, and original source split;
- full component digest;
- raw and normalized query-text digests;
- sorted positive external-document and inline-content identity digests.

The audit requires canonical row order, unique query identities, disjointness from admitted
assignments, at least two source splits per excluded component, and an exact component digest
recomputed from all declared query IDs. It also rejects an excluded normalized-text or positive
document identity that remains in the admitted cohort.

The receipt records the structural exclusion artifact SHA-256, query count, component count,
per-dataset typed counts, and a digest over the exact closed rows. It does not serialize the rows
or their identifiers.

## Label-free receipt

The canonical receipt records:

- staged inventory, staging config, assignment seed, assignment artifact, and algorithm digests;
- the frozen near-duplicate configuration digest;
- exact source pins and their set digest;
- query, qrel, assignment, corpus-artifact, and positive-qrel counts;
- assignment-component and recomputed audit-component counts;
- exact-text, near-duplicate, shared-document, and shared-content edge counts;
- exact query-coverage, normalized-text, positive-document, and component-membership digests;
- structural exclusion counts and membership digest;
- `cross_stage_component_count: 0`.

The file does not contain a self-referential digest. `artifact_sha256` is SHA-256 of the complete
canonical receipt bytes, including the terminal newline. The freeze manifest pins that value.

## Commands

Build once from a verified staged package:

```bash
python -m fractal_ann_diagnostics.scalable_partition_audit build \
  --staged-root /absolute/path/to/staged-data \
  --output /absolute/path/to/query-partition-audit.json
```

The builder uses exclusive creation and never overwrites an existing receipt.

Verify canonical bytes and a pinned artifact digest without reopening staged labels:

```bash
python -m fractal_ann_diagnostics.scalable_partition_audit verify \
  --audit /absolute/path/to/query-partition-audit.json \
  --expected-sha256 ARTIFACT_SHA256
```

Before custody closure, recompute the complete audit and require byte-for-byte receipt equality:

```bash
python -m fractal_ann_diagnostics.scalable_partition_audit verify-staged \
  --audit /absolute/path/to/query-partition-audit.json \
  --staged-root /absolute/path/to/staged-data \
  --expected-sha256 ARTIFACT_SHA256
```

## Admission boundary

Full recomputation belongs before freeze because it reads sealed custody qrels. Freeze compilation
parses the typed canonical receipt, verifies its manifest digest, and may repeat full recomputation
only when an explicit staged root is supplied before custody closes.

Runtime admission loads the typed receipt by its expected artifact and inventory digests, then
checks only label-free bindings: assignment SHA-256, selected query counts, and online source pins.
It must not reopen sealed qrels. The query package, execution plan, and runtime receipt carry the
same `query_partition_audit_sha256`. Sealed execution derives the value from those admitted
objects; it does not accept a free SHA-256 argument.

Mutation tests cover assignment omission, query-text substitution, component corruption,
cross-stage exact and near duplicates, shared external documents, canonical-content aliases,
missing custody qrels, malformed structural exclusions, noncanonical receipts, symlinked receipt
paths, and output overwrite attempts. A known one-token pair is also run through the earlier
in-memory partition audit to prove rule parity.
