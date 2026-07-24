# Scalable label custody

Status: executable apparatus component
Implementation: `src/fractal_ann_diagnostics/scalable_custody.py`
Focused tests: `tests/test_scalable_custody.py`

## Purpose

The sealed study cannot construct an in-memory `NormalizedCorpus` for HotpotQA
FullWiki. Its document table contains millions of rows, while the labels refer
to a small query cohort. The custody packager resolves that asymmetry directly:
it streams the corpus without retaining it, assigns canonical integer rows,
and keeps the external-ID lookup in a temporary SQLite index on disk. One
inventory-verification pass precedes the mapping pass; both have bounded memory.

The output is split at the filesystem boundary. The `online` directory has no
qrels, answers, evidence bundles, raw query IDs, raw document IDs, or partition
component IDs. The `custody` directory contains the canonical
`fractal-sealed-labels-v2` object used by the existing post-receipt scoring
code.

This component does not register the protocol, freeze the execution plan, or
release labels. It accepts the digest of the already-frozen execution artifact
and binds that digest into the sealed labels and its receipt. There is no
future-receipt digest in the plan or custody package.

## Input contract

`ScalableCustodyPlan` names one sealed corpus and fixes:

- the SHA-256 digest of the canonical staged inventory;
- the SHA-256 digest of the frozen online execution artifact;
- the HMAC key ID, never the HMAC secret;
- the expected document count;
- the exact available assignment-family count and positive selected count;
- the family-selection seed digest and fixed `sha256-rank-v1` algorithm;
- the fixed representative-selection algorithm and three nested rows per family;
- every staged path the custodian permits this build to read.

The allowlist must equal the inventory-derived source set exactly. It includes
the corpus file or all contiguous corpus shards, the sealed online query file,
the sealed custody qrels, the global assignment file, and the custody evidence
file where the registered corpus defines evidence. Missing and surplus paths
both stop the build.

The staged inventory must set
`withhold_sealed_labels_from_online_process=true`. Selected qrels and evidence
must have `visibility="custody"`; queries, assignments, and corpus records must
have `visibility="online"`. This controls the online process boundary and makes
no claim of human outcome blindness. The standard staging paths are part of the
contract, so a role cannot be moved to an innocuous-looking path after freeze.

### Canonical CLI configuration

The production entry point accepts one
`fractal-scalable-custody-config-v1` object. Its top-level fields are closed:

| Field | Type | Constraint |
|---|---|---|
| `schema_version` | string | exactly `fractal-scalable-custody-config-v1` |
| `staged_root` | string | canonical absolute path to a real staged-package directory |
| `corpus` | string | lowercase URI-safe corpus name |
| `stage` | string | exactly `sealed` |
| `staged_inventory_sha256` | string | non-placeholder lowercase SHA-256 |
| `execution_artifact_sha256` | string | non-placeholder lowercase SHA-256 |
| `hmac_key_id` | string | registered non-placeholder key identifier |
| `hmac_key` | object | exact key-file pin described below |
| `expected_document_count` | integer | positive frozen corpus count |
| `available_families` | integer | exact number of unique sealed assignment components |
| `selected_families` | integer | positive registered family count |
| `selection_seed_sha256` | string | non-placeholder lowercase SHA-256 |
| `family_selection_algorithm` | string | exactly `sha256-rank-v1` |
| `representative_selection_algorithm` | string | exactly `sha256-rank-v1` |
| `nested_rows_per_family` | integer | exactly `3` |
| `source_artifacts` | array | exact inventory rows for every allowlisted source |

The `hmac_key` object has exactly `path`, `sha256`, and `byte_count`. `path` is
an absolute path to the raw binary key file. `byte_count` is in `[32, 4096]`.
The key file and config file must be owned by the custodian, use mode 0400 or
0600, have link count one, and reside below custodian-owned directories that
are not group- or other-writable.

Every `source_artifacts` entry has exactly these inventory fields:

```json
{
  "byte_count": 1234,
  "dataset": "hotpotqa-fullwiki",
  "path": "datasets/hotpotqa-fullwiki/sealed/online/queries.jsonl",
  "record_count": 7404,
  "role": "queries",
  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "stage": "sealed",
  "visibility": "online"
}
```

`dataset` and `stage` may be null only where the staging inventory permits it,
such as the global assignment file. The array must equal the inventory-derived
allowlist field for field. A matching path with a different digest, byte count,
role, visibility, dataset, stage, or record count is rejected before key use.

The config itself is canonical UTF-8 JSON: sorted keys, compact separators, no
duplicate keys, no non-finite numbers, and exactly one terminal newline.
Unknown fields are fatal. Paths containing dot traversal, relative paths,
symlink components, and placeholder path segments are rejected.

The key file SHA-256 exists only in this private config. Neither secret bytes,
the key-file path, nor the key-file digest is copied to `receipt.json` or CLI
output. The receipt contains only `hmac_key_id`.

### Module commands

```bash
python -m fractal_ann_diagnostics.scalable_custody build \
  --config /absolute/private/scalable-custody-config.json \
  --output /absolute/private/hotpotqa-custody

python -m fractal_ann_diagnostics.scalable_custody verify \
  --root /absolute/private/hotpotqa-custody

python -m fractal_ann_diagnostics.scalable_custody verify-query-parity \
  --custody-root /absolute/private/hotpotqa-custody \
  --trial-root /absolute/private/hotpotqa-query-trials
```

The trial root must contain exactly `query-trials.jsonl` and
`query-trial-receipt.json`. The parity command validates their canonical
receipt and pinned store bytes, then compares the executable rows with the
custody key map. The full trial-runtime source/embedding verification remains a
separate prerequisite.

### Label-payload-excluded ranking over qrel-derived components

The builder does not package every sealed assignment by default. It enumerates
the unique `partition_component_sha256` values in the pinned assignment file
and first requires that count to equal `available_families`. It computes a rank
for each component with length-prefixed SHA-256:

```text
LP("fractal-custody-family-selection-v1")
LP("sha256-rank-v1")
LP(corpus)
LP("sealed")
LP(selection_seed_sha256)
LP(partition_component_sha256)
```

`staged_inventory_sha256` is deliberately absent from both ranking formulas.
That digest commits qrel and evidence bytes. Excluding it prevents later
payload-byte changes from altering ranks once the component identifiers are
fixed. The component identifiers are not label-independent: staging constructs them partly from
positive-qrel document edges, and the partition audit verifies that graph. The plan and receipt
still bind the staged inventory separately, so inventory substitution remains fatal.

Components sort by `(rank_sha256, partition_component_sha256)`. The first
`selected_families` components define the cohort. Ranking reads no qrel,
answer, evidence, text, prediction, or other outcome. A count mismatch or fewer
components than `selected_families` stops the build without publication. This
direct no-read property does not turn the qrel-derived component graph into an
outcome-independent sample.

### Representative query and nested trials

Each selected component contributes one representative query. For every query
assigned to that component, the builder computes:

```text
query_id_sha256 = SHA256(UTF8(query_id))

LP("fractal-custody-representative-selection-v1")
LP("sha256-rank-v1")
LP(corpus)
LP("sealed")
LP(selection_seed_sha256)
LP(partition_component_sha256)
LP(query_id_sha256)
```

Candidates sort by `(rank_sha256, query_id_sha256)`. The first candidate is the
representative. Query text is checked against its assignment digest before the
choice. Qrels and evidence are not read by either ranking procedure.

The representative then expands to exactly three rows with `nested_index` 0,
1, and 2. All three rows share its query text and opaque family key. Their trial
keys differ because each HMAC receives this canonical JSON string as its
`source_value`:

```json
["fractal-custody-nested-trial-v1","<raw representative query ID>",0]
```

The final integer changes for the other two rows. The raw query ID remains
inside custody and never appears in the online key map. Selected families are
ordered by family rank, nested indices are ordered numerically, and published
`query_row` values are contiguous. The later trial-runtime record retains the
representative's original embedding row separately.

`receipt.json` records both ranking algorithms, the seed digest, available and
selected family counts, `nested_rows_per_family=3`, and the resulting query
count. The required identity is:

```text
query_count = selected_family_count * 3
```

The verifier checks that identity, the declared number of opaque family keys,
one contiguous `0,1,2` block per family, and identical query text within each
block.

## Published package

```text
<corpus-custody-package>/
├── online/
│   ├── query-key-map.jsonl
│   └── provenance-sha256.bin
├── custody/
│   └── sealed-labels.json
└── receipt.json
```

`online/query-key-map.jsonl` has three closed-schema rows per selected family:

```json
{
  "corpus": "hotpotqa-fullwiki",
  "family_key": "<64 lowercase hex>",
  "nested_index": 0,
  "query_row": 0,
  "schema_version": "fractal-custody-query-key-row-v1",
  "stage": "sealed",
  "text": "Which city ...?",
  "trial_key": "<64 lowercase hex>"
}
```

There is deliberately no generic metadata map. Adding one would create a route
for a source ID or outcome to cross the custody boundary.

`online/provenance-sha256.bin` is a positional array of 32-byte SHA-256
digests. Record `n` is the content digest for canonical document row `n`. The
format matches `DigestOnlyProvenanceRegistry`, which performs `pread` at
`n * 32` and does not open a corpus shard during online execution.

The key map is not an executable query store. `trial_runtime.py` reserves
`query-trials.jsonl` for the source-row- and vector-epoch-bound runtime record.
Both builders import the rank and nested-source functions from
`query_cohort.py`; neither implementation maintains a private copy of those
formulas. The runtime v3 receipt pins the registered seed, available and
selected family counts, both algorithm identifiers, and the three-row
cardinality. After both packages are independently verified,
`verify_query_trial_key_parity` requires row-by-row equality of query order,
corpus, stage, text, trial key, and family key. The runtime store is the sole
query authority admitted by `ShardedOnlineExecutionPlan`. The v4 plan pins the
runtime JSONL and its v3 typed receipt as distinct direct artifacts, so the
registered cohort metadata cannot move while the query rows remain fixed.

`custody/sealed-labels.json` is a canonical `SealedLabelArtifact`. Each row
contains the same opaque trial/family keys as the online key map, positive qrel
row IDs, and, where defined, the answer and complete evidence alternatives.
The three rows for a family repeat the representative query's labels exactly;
only their trial keys and nested indices differ. The artifact carries the frozen
execution digest. Existing prediction completion and offline join functions can
consume it without a translation layer.

`receipt.json` pins all three output files and every selected staged source. It
also records the staged-inventory digest, execution digest, key ID, document
count, query count, available/selected family counts, selection seed digest,
fixed nested count, document-row-order digest, and the derivation algorithm
identifiers. It contains no HMAC secret and no label values.

## Canonical document rows

Corpus records must already be strictly ordered by the UTF-8 bytes of their
external IDs. The packager checks that order across shard boundaries and
assigns rows `0..N-1`. Equality or regression is fatal, so duplicate IDs and
reordered shards cannot silently change the retrieval universe.

The row-order digest is length-prefixed SHA-256:

```text
LP("fractal-document-row-order-length-prefixed-v1")
LP(external_id_0)
LP(external_id_1)
...
LP(external_id_N-1)
```

`LP(x)` is the eight-byte unsigned big-endian byte length followed by the UTF-8
bytes of `x`. The digest exposes no identifier but commits to their count,
values, and order.

The content digest and source URI follow the corpus adapters:

| Corpus | Source URI | Content digest parts |
|---|---|---|
| SciFact | `scifact://document/<quoted-id>` | title, then each newline-delimited abstract sentence |
| HotpotQA FullWiki | `hotpotqa-fullwiki://title/<quoted-id>` | title, staged text |
| T2-RAGBench | `t2-ragbench://context/<quoted-id>` | file name, context |
| BRIGHT | `bright://document/<quoted-id>` | title, text |
| MIRACL transfer | `miracl-transfer://document/<quoted-id>` | title, text |

Each part uses the same eight-byte length prefix as `corpora._content_hash`.
The sidecar stores the raw 32 digest bytes; sealed evidence stores the familiar
`sha256:<hex>` form.

SciFact staging joins the upstream abstract sentence array with newline
characters. Its pinned release does not contain embedded newlines inside an
abstract sentence, so splitting the staged text on newline reconstructs the
normalizer input. A future SciFact source revision that permits embedded
newlines needs a staging-schema revision that preserves sentence boundaries.
It must not reuse this package format on assumption alone.

## Opaque identity

The packager and trial runtime share the existing v2 derivation contract. For
each value below, append an eight-byte big-endian length and then its UTF-8
bytes to HMAC-SHA-256:

```text
"fractal-label-separation-v2"
domain
key_id
corpus
"sealed"
source_value
```

For a trial key, `domain="trial"` and `source_value` is the compact canonical
JSON array `["fractal-custody-nested-trial-v1","<raw query ID>",nested_index]`.
The nested index is one of 0, 1, or 2. For a family key, `domain="family"` and
`source_value` is the assignment row's
`partition_component_sha256`. The component choice preserves connected-query
family identity while withholding the component itself from online bytes.

The assignment row must match the query text SHA-256. Missing assignments,
surplus selected assignments, repeated selected queries, and HMAC collisions
stop publication. The HMAC secret must be immutable `bytes`, at least 32 bytes
long, with at least eight distinct byte values. Only its registered key ID is
serialized.

## Label construction

Qrels are joined through the disk-backed external-ID index after representative
selection. The packager validates every source query and document identity but
retains labels only for the selected representatives. Each representative must
have at least one positive judgment. Positive relevance becomes the sorted
`relevant_document_ids` tuple expected by the current scorer. Graded values
greater than zero retain the registered binary-positive interpretation; zero is
not relevant. The same tuple is copied to all three nested rows.

SciFact, HotpotQA FullWiki, and T2-RAGBench require one evidence row for every
selected representative. Every evidence location must name a mapped document
with a positive qrel. The locator is retained byte for byte; canonical integer
row, source URI, and content hash come from the same document mapping used for
the provenance sidecar. Complete alternatives remain grouped by bundle ID and
are copied unchanged to all three nested rows.

BRIGHT and MIRACL transfer define relevance but no registered complete-evidence
annotation. Their sealed labels therefore contain empty evidence tuples and
null answers. The appearance of an evidence artifact for either corpus is
treated as schema drift.

## Memory and filesystem bounds

Corpus JSONL is read one line at a time with a 16 MiB maximum line size. The
builder retains one document record plus a small SQLite page cache. Its durable
memory cost does not grow with FullWiki document count. The temporary SQLite
table stores external ID, integer row, and 32-byte content digest; it is deleted
before publication.

Queries, qrels, and evidence remain in memory because their size follows the
sealed query cohort rather than the document universe. The canonical sealed
label object is also assembled in memory because the existing scoring schema
is one JSON object. If a future registered cohort makes query labels large,
that schema needs a versioned streaming replacement.

Every selected file is opened by walking directory descriptors with
`O_NOFOLLOW`. The final file must be regular, have link count one, and retain
the same device, inode, size, timestamps, and link count for the read. Canonical
bytes, inventory byte count, record count, and SHA-256 must all agree.

Output is assembled in a mode-0700 temporary sibling. Files are created with
exclusive mode-0600 opens, flushed, and fsynced. The package verifier runs
before publication. Publication uses the platform's no-replace directory
rename (`renameatx_np(RENAME_EXCL)` on macOS or `renameat2(RENAME_NOREPLACE)` on
Linux), followed by a parent-directory fsync.

## Execution order

1. Register the external protocol and custody roles.
2. Pin and verify the staged data inventory.
3. Freeze the sharded online execution plan; record its artifact digest.
4. Register `available_families`, `selected_families`, both fixed ranking
   algorithms, `nested_rows_per_family=3`, and a separately registered
   selection-seed digest from the power design. Freeze these values before the
   custody build. This order constrains post-freeze changes; it does not show
   that the operator never inspected publicly available labels.
5. Write one canonical CLI config per sealed corpus with exact source pins.
6. Run the module `build` command inside the custodian boundary.
7. Verify the embedding-bound trial-runtime store, then run
   `verify_query_trial_key_parity`. Query order, text, corpus/stage binding, and
   trial/family keys must match the custody key map exactly.
8. Transfer only the required online artifacts to the runner. Keep
   `custody/sealed-labels.json`, the HMAC secret, qrels, and evidence inaccessible
   to the runner identity.
9. Execute once. Anchor the prediction completion and sealed-run receipts.
10. Release the matching sealed-label artifact to the offline scorer only after
   receipt verification.

## Acceptance checks

The focused tests establish the following executable claims:

- SciFact, HotpotQA FullWiki, and T2 evidence locations map to the expected
  canonical row, source URI, locator, and content hash;
- BRIGHT produces relevance labels with undefined evidence;
- one representative is selected reproducibly when a partition component has
  multiple queries;
- every selected family emits exactly three distinct trial keys, one shared
  family key, identical query text and labels, and nested indices `0,1,2`;
- a runtime query store with one altered family key is rejected even when its
  supplied file digest matches the altered bytes;
- family ranking is deterministic from its registered pins, does not open qrel
  payloads, and rejects selected-count underflow or available-count drift;
- changing qrel bytes, the staged-inventory digest, and the resulting sealed
  labels cannot change family selection, representative choice, or online key
  map bytes when the registered seed and structural identifiers are fixed;
- verification rejects interleaved family blocks, altered trial/family joins,
  unequal nested labels, and forged output roles or record counts;
- the module `build`, `verify`, and `verify-query-parity` commands execute
  against mode-private fixtures;
- configs reject unknown and duplicate keys, noncanonical JSON, relative and
  symlinked paths, placeholder IDs/digests, and source-pin drift;
- emitted receipts and CLI output contain no key bytes, key path, or key-file
  digest;
- raw query IDs, raw document IDs, answers, qrels, and evidence do not occur in
  online artifact bytes in the sentinel fixtures;
- bytewise document order, mapping coverage, evidence coverage, source hashes,
  hard links, and output replacement all fail closed;
- corpus paths use the streaming reader, not the bounded control-file reader;
- the emitted package re-verifies its bytes and exact online-to-sealed key join.

These checks establish apparatus integrity. They do not authorize label
release or turn operator self-review into independent scientific review. A
researcher who did not operate the apparatus can independently reproduce the
checks from read-only copies. Separate administrative authority is required
only for a claim of independent custody or host control.
