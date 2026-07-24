# Confirmatory study-data staging

`fractal_ann_diagnostics.study_data` converts locally acquired benchmark releases into the frozen
data package consumed by the confirmatory apparatus. It does not download, sample, relabel, or
silently repair upstream data. Each input must carry an immutable upstream revision and an exact
SHA-256 digest before staging starts.

This boundary matters. A result is confirmatory only if the corpus, queries, labels, allocation
rule, and exclusions were fixed before the sealed run. A checksum calculated after looking at an
outcome is merely a checksum.

## Package contract

The command publishes one new directory by rename after every source, schema, identity, and
leakage check passes:

```text
study-data/
├── assignments.jsonl
├── exclusions.jsonl                            # present when a registered cohort rule excludes rows
├── datasets/
│   └── <dataset>/
│       ├── corpus.jsonl                         # SciFact, T2, MIRACL
│       ├── corpus/part-00000.jsonl              # BRIGHT, HotpotQA FullWiki
│       ├── corpus/part-00001.jsonl
│       ├── fit/{queries,qrels}.jsonl
│       ├── calibration/{queries,qrels}.jsonl
│       └── sealed/
│           ├── online/queries.jsonl
│           └── custody/qrels.jsonl
├── inventory.json
└── inventory.sha256
```

Only stages registered for a source are written. HotpotQA uses the two immutable Parquet shards
derived from the v1.1 training release for fit and calibration, while FullWiki development
questions remain sealed. BRIGHT and FullWiki corpora are emitted in lexicographic 100,000-record
parts. This avoids the one-GiB materialization ceiling of the current inline execution loader and
gives a later vector-store builder stable shard boundaries.

Every artifact row in `inventory.json` binds its relative path, role, dataset, stage, visibility,
byte count, record count, and SHA-256. `inventory.sha256` binds the canonical inventory bytes. The
package verifier recomputes all three artifact measurements, checks terminal-newline form, rejects
symbolic links before artifact reads, and requires exact package membership.

The local checksum is not an external witness. The freeze procedure must publish the inventory
digest to the registered repository release, transparency receipt, and time authority before the
sealed executor receives its allowlist.

## Registered source allocation

| Study stratum | Pinned source files | Allocation |
|---|---|---|
| SciFact | corpus, train claims, development claims | all evidence-bearing claims form one 3:1:1 component allocation; upstream split remains descriptive |
| T2-RAGBench FinQA | train, development, test JSONL | train is fit, development is calibration, test is sealed |
| MIRACL Swahili | compressed corpus, train/development topics, train/development qrels | all queries form one 3:1:1 component allocation; upstream split remains descriptive |
| BRIGHT | official `documents/<domain>.parquet` and `examples/<domain>.parquet` for all 12 domains | one cross-domain component graph divides 3:1:1; domain remains a reported stratum |
| HotpotQA FullWiki | two-shard v1.1 training Parquet conversion at an immutable `hotpotqa/hotpot_qa` commit, Wikipedia abstracts archive, extracted shard tree, FullWiki validation Parquet | training components divide 4:1 between fit and calibration; validation questions are sealed; the entire corpus is searchable |

Primary source descriptions are available from [SciFact](https://github.com/allenai/scifact),
[T2-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench),
[MIRACL](https://github.com/project-miracl/miracl),
[BRIGHT](https://github.com/xlang-ai/BRIGHT), and
[HotpotQA](https://hotpotqa.github.io/).

### BRIGHT domain closure

The confirmatory configuration must name this exact, UTF-8-sorted set:

```text
aops
biology
earth_science
economics
leetcode
pony
psychology
robotics
stackoverflow
sustainable_living
theoremqa_questions
theoremqa_theorems
```

Missing or additional domains are rejected. The accepted official example fields are `id`,
`query`, and `gold_ids`. Reasoning traces, answers, `excluded_ids`, and `gold_ids_long` do not enter
the online query or qrel records. In particular, `gold_ids_long=["N/A"]` is not interpreted as a
judgment.

BRIGHT repeats document pools across some domains. Deduplication therefore uses the exact pair
`(upstream document ID, canonical content digest)`, not content alone. A repeated pair maps to one
global document ID. The inventory reports source rows, unique documents, duplicate rows, collision
policy, and the SHA-256 of the complete domain-to-global-ID map. The normal confirmatory policy is
`error`: an upstream ID carrying different content in two domains stops staging. The alternative
`domain-scoped` policy is accepted only when that choice has already been registered.

### FullWiki streaming receipt

The acquired HotpotQA materials currently bind:

```text
source release URL             http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
archived source SHA-256        26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316
training mirror revision       1908d6afbbead072334abe2965f91bd2709910ab
training shard 00000 SHA-256   76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68
training shard 00000 bytes     165,624,177
training shard 00000 rows      45,224
training shard 00001 SHA-256   713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f
training shard 00001 bytes     166,162,479
training shard 00001 rows      45,223
training rows                  90,447
official archive SHA-256       1acca1c5cc93c4890ea51091d2bad7c3ef6987aead127ab88728dc9e26555729
validation Parquet SHA-256     78933c0a31a5f7b420d4effdf4cd4eed573b28c6a3da6179dcf7a02b39e51d03
validation rows                7,405
extracted shard files          15,517
compressed shard bytes         1,550,020,166
FullWiki document records      5,233,329
canonical shard-tree SHA       9ae92bbb75168e5001fe7115688a69b128bc7713f418bcc57340b5d162c4c612
```

The [HotpotQA reference implementation](https://github.com/hotpotqa/hotpot) identifies
`hotpot_train_v1.1.json` as the training release. The same 90,447 questions supply the training
split of both the distractor and FullWiki configurations; “FullWiki training” here does not denote
a separately published annotation file. The original CMU endpoint repeatedly returned HTTP 504
during acquisition, so it is provenance rather than a consumed input. The actual inputs are the
two Parquet shards in the [`hotpotqa/hotpot_qa` FullWiki tree](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab/fullwiki),
fixed at commit `1908d6afbbead072334abe2965f91bd2709910ab`. Their two row counts sum to
90,447, and their schema carries the release identifiers, questions, answers, type, level,
supporting facts, and context. Staging verifies each acquired shard independently. It does not
trust a filename, HTTP timestamp, aggregate row count, or mutable dataset branch.

The tree digest is the SHA-256 of canonical rows sorted by relative shard path. Each row binds the
path, compressed byte count, and compressed-file SHA-256. During staging, one `.bz2` shard is read
and decompressed at a time into a disk-backed SQLite index. The converter checks the registered
shard count, tree digest, corpus record count, unique document IDs, unique titles, and unique
canonical content. Supporting-fact titles and sentence offsets from the sealed Parquet file must
resolve against that complete index.

Some official FullWiki rows retain a title and ID but have an empty abstract string. The registered
`hotpot-title-only-document-v1` rule preserves each row and substitutes its non-empty title as the
embedding text. The original sentence count remains unchanged for supporting-fact validation, and
the inventory records the exact number of title fallbacks. No document is removed from the
5,233,329-row universe.

The pinned validation release contains one structurally invalid supporting fact. Question
`5ae61bfd5542992663a4f261` assigns sentence `902` to `Jimmy Butler (basketball)`, whose pinned
FullWiki abstract has five sentences. It is the sole supporting-fact index above 100 and the sole
index outside its named paragraph among all 18,005 supporting-fact rows. The registered
`hotpotqa-supporting-fact-range-v1` rule excludes the complete question before query, qrel, answer,
or evidence construction. It does not guess that `902` meant `2`, truncate the index, or retain the
query only for endpoints with usable labels. The exclusion receipt records the query and rule, so
the sealed HotpotQA cohort contains 7,404 rather than 7,405 questions.

Training and development questions first enter one component graph. A normalized-question match or
shared positive FullWiki document across the source-split boundary excludes the complete connected
component before allocation. Remaining training components receive the registered 4:1
fit/calibration assignment. Remaining development components are fixed to sealed. This order keeps
training labels available for development without allowing a shared positive target to cross into
the sealed cohort.

`corpus_archive` and `corpus_shards` are separate pins by design. The first proves which upstream
archive was acquired. The second proves that the extracted bytes used for conversion are exactly
the registered extraction.

## Assignment and leakage controls

Hashing an isolated query is insufficient because two queries may share their answer document.
The assignment unit is a connected component. An edge joins two queries when either condition is
true within the registered allocation partition:

1. their NFKC-casefolded token sequences are identical;
2. they share a positively judged document.

The component digest is the SHA-256 of its sorted stable query IDs. Components are then ranked by
a length-delimited SHA-256 over:

```text
algorithm | assignment seed | dataset | allocation stratum | component digest
```

The assignment ledger records the stable query ID, source split, domain, stage, normalized query
text digest, component digest, and assignment-key digest. The post-allocation audit rejects:

- duplicate query IDs;
- duplicate normalized query text outside one shared component and stage;
- unknown query or document references;
- repeated qrel pairs;
- conflicting document IDs or duplicate content aliases;
- any positive document crossing fit, calibration, or sealed stages.

T2 retains its fixed train, development, and test stages because its context IDs do not cross those
boundaries in the pinned release. SciFact and MIRACL do have shared positive content across their
upstream train/development splits. Their component graphs therefore span both splits before the
3:1:1 assignment. This keeps content-equivalent positive targets in one stage while retaining the
upstream split as a reported covariate.

### Registered acquisition rules discovered before freeze

SciFact contributes a retrieval trial only when its `evidence` object contains at least one
document mapped to a non-empty rationale array. The rule is identified as
`scifact-evidence-bearing-v1`; it does not inspect the claim label or model output. Applied to the
exact pinned release, it admits 505 of 809 train claims and 188 of 300 development claims. The
remaining 304 train and 112 development IDs enter `exclusions.jsonl` with the rule identifier and
reason. The staged inventory pins that receipt, the admitted query files, and their counts. A
source update that changes any of those bytes needs a new protocol version.

T2 FinQA contributes a row only when `question` is non-empty. The pinned test split contains one
empty question, `finqa_test_490`; `t2-finqa-nonempty-question-v1` excludes that row before document
or query construction and records the stable query ID in the same exclusion receipt. The rule does
not substitute an answer, program, context fragment, or guessed question.

BRIGHT uses one connected-component graph across every query view in the 12-domain suite. This
couples the 76 normalized-query duplicates shared by `theoremqa_questions` and
`theoremqa_theorems`; shared positive documents can couple other domain views as well. Each global
component receives one 3:1:1 assignment. Domain remains a recorded stratum, but it does not enter
the assignment hash. Duplicate normalized text is admissible only when every copy has the same
dataset, component digest, and stage.

The official `theoremqa_questions` examples repeat a `gold_ids` value within 120 query rows.
Relevance is binary here, so `bright-binary-qrel-dedup-v1` collapses repeated `(query_id,
document_id)` pairs after validating every identifier. It preserves the query, emits one positive
judgment per pair, and records the 178 removed repeated pairs in the pinned staged inventory. The
120 figure counts affected query rows; the 178 figure counts surplus pair occurrences. The rule does
not combine different document IDs or reinterpret `gold_ids_long`.

Both rules were fixed after source acquisition and before protocol registration, controller
selection, sealed prediction, or label release. They are prospective parts of v0.3.0 rather than
post-result repairs.

The MIRACL Swahili corpus also contains distinct passage IDs with byte-identical title and text.
The stager preserves those IDs because qrels address them independently, counts every alias in the
inventory, and uses canonical content identity when constructing query components. Two queries
whose positive IDs contain identical content therefore cannot enter different stages.

## Canonical records

All emitted JSON uses UTF-8, lexicographically sorted object keys, no insignificant whitespace,
finite numbers, and one newline per record. Identifiers must be non-empty NFC strings without
leading or trailing whitespace or control characters.

```json
{"id":"document-17","text":"Document body","title":"Document title"}
{"id":"dataset:query-9","text":"Question text"}
{"document_id":"document-17","query_id":"dataset:query-9","relevance":1}
{"answer":null,"evidence_bundles":[{"bundle_id":"rationale-0","locations":[{"document_id":"document-17","locator":"sentence:2"}]}],"label_metadata":[],"query_id":"dataset:query-9"}
```

Answers, rationales, evidence polarity, and supporting-sentence coordinates never enter online
query rows. Evidence-bearing corpora instead receive a separate `evidence-bundles.jsonl` artifact.
Development copies are online because they are admitted only for fitting and calibration. The
sealed copy is written under `sealed/custody/` and receives `visibility="custody"`; no matching
file may appear under `sealed/online/`.

SciFact retains each upstream rationale as one alternative bundle with exact sentence locators.
HotpotQA retains one `supporting-facts` bundle containing every registered supporting sentence.
T2-RAGBench retains one `source-context` bundle whose locator is the complete document. BRIGHT and
MIRACL have relevance judgments but no complete-evidence file. The stager rejects a missing bundle,
an evidence location without a positive relevance link, an unknown document, repeated locations,
or incomplete query coverage in any of the three evidence corpora.

The staged evidence rows use stable external document IDs. The custody packager maps those IDs to
the bytewise-sorted integer row order and attaches the canonical source URI and content hash. This
keeps outcome-bearing evidence out of the embedding and online-execution paths while retaining the
information needed for the registered post-anchor sufficiency endpoint.

## Closed configuration

The configuration root accepts no undeclared fields:

```json
{
  "assignment_seed": "64 lowercase hexadecimal characters",
  "datasets": {
    "bright": {
      "document_id_collision_policy": "error",
      "domain_order": [
        "aops",
        "biology",
        "earth_science",
        "economics",
        "leetcode",
        "pony",
        "psychology",
        "robotics",
        "stackoverflow",
        "sustainable_living",
        "theoremqa_questions",
        "theoremqa_theorems"
      ],
      "domains": {}
    },
    "hotpotqa_fullwiki": {},
    "miracl_sw": {},
    "scifact": {},
    "t2_finqa": {}
  },
  "schema_version": "fractal-study-data-staging-config-v3",
  "withhold_sealed_labels_from_online_process": true
}
```

Every ordinary source declaration has exactly three fields:

```json
{
  "path": "non-git-files/confirmatory-v0.3/upstream/scifact-data/data/corpus.jsonl",
  "revision": "0123456789abcdef0123456789abcdef01234567",
  "sha256": "64 lowercase hexadecimal characters"
}
```

Paths resolve relative to the configuration directory unless absolute. Final-component symbolic
links and non-regular files are rejected. `latest`, `main`, `master`, `tbd`, and similar movable
revision labels are invalid even when a digest is present.

FullWiki adds an extracted-tree pin and a closed scope assertion:

```json
{
  "corpus_archive": {
    "path": "upstream/hotpotqa/enwiki-abstracts.tar.bz2",
    "revision": "hotpotqa-official-fullwiki-archive",
    "sha256": "1acca1c5cc93c4890ea51091d2bad7c3ef6987aead127ab88728dc9e26555729"
  },
  "corpus_scope": {
    "expected_document_count": 5233329,
    "name": "fullwiki",
    "sampling": "none"
  },
  "corpus_shards": {
    "file_count": 15517,
    "path": "upstream/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts",
    "revision": "hotpotqa-official-fullwiki-extraction",
    "sha256": "9ae92bbb75168e5001fe7115688a69b128bc7713f418bcc57340b5d162c4c612"
  },
  "train_questions": [
    {
      "path": "upstream/hotpotqa/train-00000-of-00002.parquet",
      "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
      "sha256": "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68"
    },
    {
      "path": "upstream/hotpotqa/train-00001-of-00002.parquet",
      "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
      "sha256": "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f"
    }
  ],
  "dev_questions": {
    "path": "upstream/hotpotqa/fullwiki-validation.parquet",
    "revision": "hotpotqa-fullwiki-validation",
    "sha256": "78933c0a31a5f7b420d4effdf4cd4eed573b28c6a3da6179dcf7a02b39e51d03"
  }
}
```

The record count above was recomputed across every pinned compressed shard. A parser cannot infer
corpus completeness from a directory name, so the stager checks this exact count before publishing.

## Run and verify

Parquet sources require the declared benchmark extra:

```bash
python -m pip install -e '.[benchmarks]'
```

Build a package at a path that does not yet exist:

```bash
python -m fractal_ann_diagnostics.study_data stage \
  --config research/study-data-config.json \
  --output artifacts/study-data-v1
```

Then verify it independently:

```bash
python -m fractal_ann_diagnostics.study_data verify \
  --root artifacts/study-data-v1
```

Create the online mount as a separate no-overwrite projection:

```bash
python -m fractal_ann_diagnostics.study_data project-online \
  --source-root artifacts/study-data-v1 \
  --output artifacts/study-data-online-v1

python -m fractal_ann_diagnostics.study_data verify-online \
  --root artifacts/study-data-online-v1 \
  --expected-inventory-sha256 <full-staged-inventory-sha256>
```

The projector first verifies every byte in the custody-complete source tree. It then copies only
corpus rows, query rows, assignments, and the two partition-control ledgers. Qrels and evidence
bundles are absent at every stage, including development partitions. `inventory.json` and its
checksum remain byte-identical to the custody-complete package so downstream receipts retain the
same staged-inventory identity. The inventory therefore exposes digests and counts for omitted
files, but no label payload bytes. `projection-receipt.json` pins the exact selected artifact rows,
selection rule, source inventory, and artifact-set digest. Verification rejects an extra file,
missing file, link, changed byte, new role without an explicit projection decision, or any injected
qrel or evidence payload.

Staging uses a temporary sibling directory. Any digest, decoding, split, identity, count, or
leakage failure removes that directory and publishes nothing. An existing destination is never
accepted as mutable state.

## Custody handoff

With `withhold_sealed_labels_from_online_process=true`, sealed query IDs and text are placed below
`online`, while sealed qrels are placed below `custody`. This is a process-boundary control, not a
claim that the public benchmark outcomes are unavailable to people. The online execution principal
receives the separately verified projection, containing only:

- the inventory-derived allowlist;
- corpus shards;
- sealed online queries;
- the assignment ledger;
- frozen environment and model artifacts registered elsewhere in the apparatus.

It does not receive the package root, custody paths, answers, or label keys. Directory separation
is not cryptographic custody. The custody procedure must encrypt and time-lock each sealed qrels
artifact, publish the ciphertext receipt, and retain decryption authority with an independent
custodian until predictions and execution telemetry have been externally anchored.
