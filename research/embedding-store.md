# Label-blind embedding store

Status: the streaming store and the fixed five-corpus production controller are implemented. The
controller derives source paths from the admitted online inventory, resumes at the store's flushed
checkpoint boundary, and records a terminal suite receipt.

## Production suite command

Embedding construction is a pre-C1, label-blind artifact operation. It does not run in the Linux C0
sealed-execution image. The measured and admitted builder is a separately pinned macOS arm64 MPS
environment. C1 freezes its builder receipt, production config, five vector trees, five shard
receipts, and terminal suite receipt before any outcome label is released. The Linux C0 runner later
consumes those immutable vector bytes.

The exact optional dependency set is `fractal-ann-diagnostics[production-embedding]`: PyTorch
2.13.0 and Transformers 5.13.1 plus the transitives resolved in `uv.lock`. The lock carries the
official macOS arm64 PyTorch wheel and the platform-neutral Transformers wheel with their PyPI
SHA-256 values. See the primary [PyTorch 2.13.0 release](https://pypi.org/project/torch/2.13.0/)
and [Transformers 5.13.1 release](https://pypi.org/project/transformers/5.13.1/).

A 2026-07-18 dependency review found Transformers 5.14.1 as the latest release. Its published
changes concern Inkling position bias, assisted generation with `EncoderDecoderCache` and
OlmoHybrid, FP8 kernels, and multi-device DeepGEMM; none is used by this Qwen3 embedding path.
The apparatus therefore retains 5.13.1, the version used for the development throughput and
memory measurements below. Changing it would require a new label-blind benchmark and a new
artifact closure. See the official [5.14.1 release
record](https://github.com/huggingface/transformers/releases/tag/v5.14.1).

After bootstrap source commit P is committed and its candidate image closure fixes P/T/D, use a
clean checkout of P. Create the environment outside the repository so environment files cannot
enter the source commit. Apparatus commit A does not exist yet because these embeddings become
inputs to the raw candidate manifest later committed at A:

```bash
REPO=/controlled/fractal-v0.3/source/fractal-ann-diagnostics
CONTROL=/controlled/fractal-v0.3
SOURCE_P=<full-40-character-bootstrap-source-commit>
mkdir -p "$CONTROL/controls" "$CONTROL/artifacts"
chmod 0700 "$CONTROL" "$CONTROL/controls" "$CONTROL/artifacts"
export UV_PROJECT_ENVIRONMENT="$CONTROL/embedding-builder-venv"
export UV_LINK_MODE=copy

test "$(/usr/bin/git -C "$REPO" rev-parse HEAD)" = "$SOURCE_P"
test -z "$(/usr/bin/git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -z "$(/usr/bin/git -C "$REPO" status --porcelain=v1 --ignored=matching \
  --untracked-files=all -- src)"
uv sync \
  --project "$REPO" \
  --frozen \
  --no-dev \
  --extra production-embedding \
  --python /opt/homebrew/bin/python3.12

BUILDER_PY="$CONTROL/embedding-builder-venv/bin/python"
builder_python() {
  env -i \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    HOME=/private/var/empty \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PATH=/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONSAFEPATH=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=0 \
    TOKENIZERS_PARALLELISM=false \
    TMPDIR=/private/tmp \
    TRANSFORMERS_OFFLINE=1 \
    TZ=UTC \
    VECLIB_MAXIMUM_THREADS=1 \
    __CF_USER_TEXT_ENCODING="$(printf '0x%X:0x0:0x0' "$(/usr/bin/id -u)")" \
    "$BUILDER_PY" -P -s "$@"
}
builder_command() {
  builder_python -m fractal_ann_diagnostics.production_embedding_build "$@"
}

test "$(builder_python -c 'import platform; print(platform.python_version())')" = 3.12.13
SITE_PACKAGES="$(builder_python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
test -z "$(find "$SITE_PACKAGES" \( -type d -name __pycache__ -o -type f \
  \( -name '*.pyc' -o -name '*.pyo' \) \) -print -quit)"

# The receipt writer and every production command require immutable executable inputs.
find "$REPO/src" -type f -exec chmod a-w {} +
find "$REPO/src" -type d -exec chmod a-w {} +
chmod a-w "$REPO" "$REPO/pyproject.toml" "$REPO/uv.lock"
for model_root in "$CONTROL/models/qwen-current" "$CONTROL/models/qwen-stale"; do
  find "$model_root" -type f -exec chmod a-w {} +
  find "$model_root" -type d -exec chmod a-w {} +
done
find "$SITE_PACKAGES" -type f -exec chmod a-w {} +
find "$SITE_PACKAGES" -type d -exec chmod a-w {} +
chmod a-w \
  "$CONTROL/embedding-builder-venv" \
  "$CONTROL/embedding-builder-venv/lib" \
  "$CONTROL/embedding-builder-venv/lib/python3.12" \
  "$CONTROL/embedding-builder-venv/pyvenv.cfg"
```

The first command that can load either model is `write-builder-receipt`. It refuses a dirty
checkout (including ignored import shadows), a source commit mismatch, another lockfile, another
Python or package tree, a writable executable-input tree, a changed minimal process environment, a
non-Darwin platform, a non-arm64 machine, absent MPS, model-tree drift, or a changed fixed probe. It
records the macOS product and build, Mac model, chip, logical cores, physical memory, fixed system
Git and SHA-256, resolved Python binary and SHA-256, `pyvenv.cfg`, every base `sys.path` file/tree,
the full site-packages tree (including `.pth`, distribution metadata, and native libraries), exact
import origins, source-tree identity, Qwen arm configs, and byte-identical repeated probe vectors.

```bash
BUILDER_RECEIPT="$CONTROL/controls/production-embedding-builder.json"
BUILDER_RESULT="$(builder_command write-builder-receipt \
  --repository-root "$REPO" \
  --expected-source-commit "$SOURCE_P" \
  --uv-lock "$REPO/uv.lock" \
  --current-model-root "$CONTROL/models/qwen-current" \
  --stale-model-root "$CONTROL/models/qwen-stale" \
  --batch-size 64 \
  --seed 20260714 \
  --output "$BUILDER_RECEIPT")"
BUILDER_RECEIPT_SHA256="$(printf '%s' "$BUILDER_RESULT" | jq -r .builder_receipt_sha256)"
```

The production entry point accepts no document path, query path, label path, qrels path, image
alias, or generic callback. `write-config` freshly revalidates the builder receipt, rehashes the
source, venv, base Python import roots, and model trees, and reruns the four-text MPS probe. It then
validates the online staging projection and derives one document allowlist and the `fit`,
`calibration`, and `sealed` query allowlists for each registered corpus. A query-stage omission or
duplicate stops config creation. Create the config outside every source, model, and output tree:

```bash
CONFIG="$CONTROL/controls/production-embeddings.json"
CONFIG_RESULT="$(builder_command write-config \
  --online-staging-root "$CONTROL/study-data/online-projection" \
  --expected-inventory-sha256 <64-hex-digest> \
  --builder-receipt "$BUILDER_RECEIPT" \
  --builder-receipt-sha256 "$BUILDER_RECEIPT_SHA256" \
  --current-model-root "$CONTROL/models/qwen-current" \
  --stale-model-root "$CONTROL/models/qwen-stale" \
  --output-root "$CONTROL/artifacts/embedding-stores" \
  --batch-size 64 \
  --device mps \
  --seed 20260714 \
  --output-dtype float32 \
  --output "$CONFIG")"
CONFIG_SHA256="$(printf '%s' "$CONFIG_RESULT" | jq -r .config_sha256)"
```

The config embeds the complete builder receipt and its SHA-256. Its own printed digest covers the
exact canonical config file, including the terminal LF. Pin that value in each later command:

```bash
builder_command status \
  --config "$CONFIG" \
  --config-sha256 "$CONFIG_SHA256"

builder_command build \
  --config "$CONFIG" \
  --config-sha256 "$CONFIG_SHA256"

builder_command verify \
  --config "$CONFIG" \
  --config-sha256 "$CONFIG_SHA256"
```

The same derivation can run as five fixed-corpus jobs. Each job still builds both registered Qwen
revisions through the paired encoder; `--corpus-id` is only a scheduling selector. It cannot change
source paths, query stages, prompts, model revisions, model-tree digests, batch settings, dtype, or
row order. Run every command under the same verified MPS builder receipt, with the same read-only
config and staging data, checkout, absolute paths, environment, and shared output filesystem:

```bash
# Run once per pinned worker. The five allowed values are:
# scifact, hotpotqa-fullwiki, t2-ragbench, bright, miracl-transfer
CORPUS_ID=<one-fixed-corpus-id>
builder_command build-shard \
  --config "$CONFIG" \
  --config-sha256 "$CONFIG_SHA256" \
  --corpus-id "$CORPUS_ID"

builder_command aggregate \
  --config "$CONFIG" \
  --config-sha256 "$CONFIG_SHA256"

builder_command verify \
  --config "$CONFIG" \
  --config-sha256 "$CONFIG_SHA256"
```

Use one worker per corpus. Running five MPS processes on the one measured M4 Max creates memory and
compute contention, so the single-host schedule remains sequential. Multi-host execution is
admissible only when every worker freshly matches the exact receipt, including the absolute checkout
and environment paths, Mac model and chip, macOS build, Python binary, base-import and site-packages
tree digests, minimal process environment, import origins, and source digest. The shared output
filesystem must provide correct POSIX `flock` semantics.
Every worker must run as the same numeric owner of the shared output parent. That parent cannot be
group- or other-writable; the derived worker-lock directory must remain owned by that identity at
mode `0700`. These checks run before and after each locked critical section.
Distinct workers may complete in any order. `aggregate` scans the closed output tree, requires
exactly the five registered stores and five typed shard-evidence receipts, rejects partial, missing,
repeated, or undeclared members, re-admits the builder, projection, and both model trees, and rehashes
every vector store before writing the suite receipt in `FIXED_CORPORA` order. The resulting receipt
is the same derivation produced by `build`; aggregation does not introduce a second scientific path.

Each shard first takes a nonblocking OS lock derived from its corpus ID, then re-admits the full
builder and scientific inputs inside the critical section before a model forward or output write.
Persistent lock inodes live beside the output tree under `.<output-name>.worker-locks/`; they are
operational controls and never enter the immutable embedding tree or its digest. The lock checks
that the directory pathname and corpus lock pathname still name the opened empty, singly linked
inode before acquisition, after acquisition, and after the critical section. Process exit and
crashes release the lock. A duplicate live worker for one corpus fails before opening its partial
store. Monolithic build, aggregation, and terminal verification take all five locks in fixed corpus
order, preventing cross-mode races without serializing different corpus workers.

The independent unit is deliberately one corpus with the paired current/stale model. Splitting the
two revision arms would change the encoder implementation identities embedded in `receipt.json`.
Relabeling independent-arm output as paired output would be false provenance. Splitting a corpus by
row range would also change source bindings, batch boundaries, checkpoint seeds, and possibly float
bytes unless a separately registered builder proved exact equivalence. Neither shortcut is used
here.

`status` inspects filesystem state only. Its `diagnostic-only` marker means that `complete` reports
the presence of five stores, five evidence records, and the suite receipt; it does not rehash those
objects. The producer-side `verify` command freshly re-observes the Darwin/MPS builder, re-admits the
staging projection and model trees, re-derives all allowlists, verifies every vector matrix and
model/config binding, rehashes every store and evidence file, and reproduces the suite receipt
without writes.

After that producer check, Linux post-embedding and factory stages call
`admit_frozen_production_embedding_suite`. The online projection and completed embedding root must
be mounted read-only at the literal absolute paths recorded in the embedding config. Frozen
admission rehashes the source projection, all five store trees, vector descriptors, model and
encoder bindings, evidence files, and terminal suite. It treats the embedded builder receipt as
producer-time provenance. It never opens the recorded Mac checkout, venv, model paths, or MPS
device, and it makes no claim that those mutable resources still exist after handoff.

`ST_RDONLY` proves that the container cannot write through its admitted mount; it is not a defense
against a privileged host retaining another writable alias to the same backing files. The
production handoff must therefore use a closed snapshot or volume with no writable alias during
admission. If the host itself is outside the trust boundary, filesystem immutability needs an
independent storage control rather than another Python check.

The output root is closed to these names during either execution mode:

```text
embedding-stores/
├── .<corpus>.partial/                 # only while one store is unfinished
├── .<corpus>.checkpoint.json          # paired with the partial directory
├── <corpus>/                          # one final store per fixed corpus
├── build-evidence/
│   └── <corpus>.json                  # five immutable timing/resource records
└── production-embedding-suite-receipt.json
```

Each corpus evidence file binds the production config, online and selected-source inventories,
embedding receipt, complete tree digest, row counts, UTC start and completion timestamps, elapsed
monotonic nanoseconds, and process peak RSS through corpus completion. The status distinguishes a
fresh build, a resumed build, and verification of a store that existed before its evidence record.
Timing and RSS never enter vector computation or statistical analysis.

## Artifact contract

`fractal_ann_diagnostics.embedding_store` turns allowlisted staged JSONL into memory-mapped vector
matrices. It never receives a corpus object. Documents and queries are read one canonical line at a
time, grouped into the registered batch size, and released after each batch.

The final directory has this shape:

```text
embedding-store/
├── config.json
├── current-documents.npy
├── current-queries.npy
├── document-rows.jsonl
├── query-rows.jsonl
├── source-inventory.json
├── old-documents.npy       # present only when an old model is registered
├── old-queries.npy         # present only when an old model is registered
└── receipt.json
```

The current matrices are the confirmatory retrieval truth. Optional old matrices exist only for
revision-drift measurements. Current and old matrices use the same dimension and the same row-order
digests, but their model revisions, model-tree digests, vector bytes, and file digests remain
separate.

`config.json` preserves the full frozen encoder configuration. `source-inventory.json` preserves
the selected staged paths, roles, byte counts, row counts, and SHA-256 values. Their canonical
payload digests are recorded in `receipt.json`, so the receipt does not depend on an unavailable
hash preimage.

## Source boundary

`StagedEmbeddingSources` carries four inputs:

| Field | Binding |
| --- | --- |
| `root` | Absolute staged-data directory |
| `inventory_sha256` | Exact SHA-256 of canonical `inventory.json` bytes |
| `document_paths` | Unique, bytewise-sorted corpus or corpus-shard allowlist |
| `query_paths` | Unique, bytewise-sorted online query allowlist |

Every selected path must appear in the staged inventory with `visibility="online"` and the expected
role. The builder rejects path components containing `custody`, `label`, or `qrel`. Document records
must contain exactly `id`, `text`, and `title`; query records must contain exactly `id` and `text`.
Fields for answers, evidence, gold judgments, relevance, or supporting facts stop the build before a
checkpoint is created.

The inventory may describe outcome files because it is the package ledger. Those files are not
opened or scanned. This distinction is tested with an unselected qrels path that is itself a
symbolic link. The build succeeds while the link remains untouched.

Each selected file is opened through no-follow directory descriptors. Symbolic links, hard links,
non-regular objects, changed byte counts, changed hashes, noncanonical JSON, repeated JSON keys,
non-finite JSON values, and lines over 16 MiB are rejected. A full streaming fingerprint pass
finishes before any partial store is created.

## Frozen encoder configuration

`EmbeddingStoreConfig` is immutable and included in the build digest. It fixes:

| Parameter | Enforcement |
| --- | --- |
| Query prompt | Exact UTF-8 string; suitable for Qwen's `Instruct: ...\nQuery: ` convention |
| Document prompt | Exact UTF-8 string, including the registered empty prompt |
| Maximum sequence length | Applied to the local model before encoding |
| MRL output dimension | Passed as `truncate_dim`; every returned batch must have that width |
| Normalization | Fixed to `true`; each stored row must have unit norm |
| Batch size | Outer streaming bound and inner encoder batch size |
| Output dtype | `float16` or `float32`, recorded per matrix |
| Device | Exact adapter device string |
| Deterministic seed | Combined with matrix name and batch start to derive a resume-stable seed |
| Builder version | Fixed to `fractal-embedding-store-builder-v1` |

Batch-specific seeds matter during recovery. A resumed batch receives the same seed it would have
received in an uninterrupted run, independent of earlier encoder calls.

## Local model binding

`LocalModelSpec` requires an absolute directory, a caller-supplied immutable revision, and the exact
canonical tree SHA-256. The tree digest covers ordered paths, entry types, empty directories, file
sizes, and file hashes. It rejects symbolic links, hard links, and irregular filesystem entries.

The tree is verified before encoding and again before finalization. An optional old model needs its
own `LocalModelSpec` and encoder instance. A byte-identical model and revision cannot be relabeled as
old.

`SentenceTransformersLocalEncoder` imports PyTorch and Sentence Transformers only on the first
batch. It passes an absolute local directory with `local_files_only=True`, sets the Hugging Face and
Transformers offline flags while loading, disables remote code, calls `eval()`, and encodes inside
`torch.inference_mode()`. It fixes normalization, prompt, dimension, batch size, device, sequence
length, and seed on each call. The relevant upstream interfaces are documented in the
[Sentence Transformers API](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)
and the [Qwen3 Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

The confirmatory Qwen path uses `QwenRevisionEmbeddingAdapter` instead. Its implementation identity
contains the full SHA-256 of the closed arm configuration. The adapter refuses prompt, sequence
length, output width, normalization, device, batch-seed, or model-root substitution before calling
the pinned offline encoder. Query and document calls are selected only by the two registered prompt
byte strings. One adapter instance cannot switch model trees.

Production builds with both Qwen tokenizer revisions use
`QwenPairedRevisionEmbeddingAdapter`. The two local trees retain separate revisions, tokenizer
digests, canonical tree digests, and receipt identities. The adapter verifies both complete trees
before model load. Per batch it checks a metadata snapshot of every tree entry; the store repeats
the cryptographic tree verification before finalization. This removes repeated reads of the
1.19 GB weight file without weakening the final content check.

The paired path is exact for the frozen decoder-only intervention. Position IDs are derived from
each row's attention mask, so every content sequence starts at position zero regardless of its
batch companions. This removes a roughly `1e-6` numeric shift observed when Qwen's default absolute
left-pad positions changed with batch composition. Token lengths are partitioned into the fixed
`64, 128, 256, 384, 511, 512` buckets. Both independent-arm and paired encoders use these buckets,
then restore the original row order.

For sequences of at most 511 pre-terminal tokens, the current input is the stale token sequence
followed by token `151643`. Causal attention leaves the preceding content state unchanged by that
future token. One forward supplies the stale embedding at the penultimate active position and the
current embedding at the terminal position. The loader runs a prefix-invariance probe before
accepting this path, and each batch rechecks both tokenizers' active token IDs.

A 512-token stale sequence cannot share the current row because the current arm reserves its last
position for the terminal token. That bucket uses separate current and stale forwards. The paired
backend transfers only the selected 256 hidden-state coordinates to host memory, then applies the
same float32 normalization routine used by the independent encoder.

Tests inject a deterministic encoder, so no model download or external connection occurs in CI.

## Builder-route decision

The fastest presently evidenced route is the separately pinned macOS arm64 MPS builder. It retains
the same five corpora, paired model revisions, source rows, prompts, float32 output, batch size, and
artifact verification. The scheduling choice changes no scientific unit. Its output is complete and
frozen before C1, so the later sealed C0 run receives immutable matrices rather than an opportunity
to recompute them after registration or label access.

A 2026-07-16 check used the first 16 canonical rows of HotpotQA FullWiki
`part-00000.jsonl`; their exact line bytes have SHA-256
`c6c3daf9f2f0b849cc9bc8a7d1ecf0c99ff6f4f9f828fb2ab431db1edfc2ad60`. On the retained
Mac16,6 M4 Max environment (Python 3.12.13, PyTorch 2.13.0, Transformers 5.13.1), the second
paired MPS forward took 0.173340 seconds, or 92.304 source rows/s. The repeated current and stale
vector bytes had combined SHA-256
`e838778d057292d42317f7475fafb7ed5396fa240875af87a2354b56b56f7331`.

The matching macOS CPU attempt did not reach a timed batch. It stopped at the unchanged causal
prefix-invariance gate: maximum absolute cross-shape drift was `2.7418137e-05`, mean absolute drift
was `7.2561670e-06`, and 15 of 256 components exceeded the registered `rtol=2e-5`,
`atol=2e-6` bound. The gate was not relaxed. This is not a Linux CPU benchmark, and no MPS result is
used to claim Linux CPU throughput. A Linux arm64 CPU route would need its own successful probe,
exact environment receipt, output freeze, and measured capacity before C1.

The 16-row check is too small to revise the corpus-scale budget. The 90–110 hour estimate below
therefore remains the planning range for the exact measured M4 Max MPS host. It is neither a CPU
estimate nor a guarantee for another Mac.

## Production feasibility benchmark

This section records a benchmark, not a frozen execution runtime or a guarantee about another
host. It was run on 2026-07-14 on a MacBook Pro `Mac16,6`, Apple M4 Max (14 CPU cores), 36 GB unified
memory, arm64, macOS 26.3.1 build `25D771280a`. The runtime was Python 3.12.13, PyTorch 2.13.0,
Transformers 5.13.1, and NumPy 2.5.1. MPS was available. Both model trees were local, offline, and
bound to the digests above.

An eight-query comparison first loaded the current and stale arms independently, then loaded the
paired adapter. A second comparison included a short document beside a text truncated to 512 stale
tokens. In both comparisons, paired current vectors were byte-identical to independent current
vectors, and paired stale vectors were byte-identical to independent stale vectors. Maximum
absolute difference and maximum L2 distance were both `0.0`.

The controlled scaling test repeated one 64-document HotpotQA FullWiki mix so the length
distribution stayed fixed:

| Outer batch | Seconds | Source rows/s | Output vectors/s | Result |
| ---: | ---: | ---: | ---: | --- |
| 64 | 2.040 | 31.38 | 62.76 | completed |
| 128 | 4.779 | 26.79 | 53.57 | completed |
| 256 | 10.130 | 25.27 | 50.54 | completed |

Batch 64 is the production choice on this host. Batch 256 completed without an MPS allocation
error, but it was slower per source row.

A separate batch-64 test used 64 staged documents per corpus, spread across shards where shards
exist. Results include paired tokenization, model forward, selected-state transfer, normalization,
and per-call tree metadata checks. They exclude store-level source verification, memmap flushes,
checkpoint replacement, and final hashing.

| Corpus | Documents in staged store | Sample rows/s | Forward projection |
| --- | ---: | ---: | ---: |
| BRIGHT | 1,145,164 | 12.42 | 25.61 h |
| HotpotQA FullWiki | 5,233,329 | 30.37 | 47.86 h |
| MIRACL transfer | 131,924 | 59.86 | 0.61 h |
| SciFact | 5,183 | 5.69 | 0.25 h |
| T2-RAGBench | 2,789 | 2.26 | 0.34 h |

The count-weighted model-forward projection is 74.7 hours for 6,518,389 documents. The execution
budget is 90–110 wall-clock hours on this host. That range allows for corpus-wide token-length
variation, the initial source scan, memmap flush and checkpoint costs, query matrices, and final
artifact hashing. A real run may fall outside the range; the terminal receipt records observed
times and bytes rather than this estimate.

### Single-host concurrency rejection

A 2026-07-17 development check tested the tempting two-process schedule on the same M4 Max. Both
processes used the paired encoder, batch 64, MPS, offline local model trees, and independently loaded
model state. Five isolated timed batches produced 58.885 HotpotQA rows/s and 8.684 BRIGHT rows/s.
The sampled source-line digests were
`ee585116175c399ff2db38150f3053ca4cfbd16fc940d0b59b79f325c145de53` and
`f5d3bc6750dedabecfb71615abaead255ab51c219236d043f6545240d925a5d6`.

During sustained overlap, BRIGHT fell to 3.945 rows/s and the first 21 HotpotQA batches averaged
14.187 rows/s. Expressed in isolated-work equivalents, the two processes delivered only 0.695 of
the sequential one-process rate. After the BRIGHT process exited, the remaining HotpotQA batches
recovered to 29.032 rows/s, still below the isolated baseline. The vector-byte digests remained
unchanged for both samples:
`7c6140b207ee77d6d5965978c8c875aa17aef4bfff1fb5fd66ddd24fc568040b` for HotpotQA and
`6f3f6ba3efb827ac22502698ccf37023f2669fa6d125f48ff2735becd335dd97` for BRIGHT.

The result rejects two concurrent production MPS workers on this host. It is a scheduling result,
not an output-equivalence failure. The one-Mac route remains sequential. Corpus-level parallelism
is reserved for independently admitted builders that share the exact POSIX-locking output
filesystem; assigning HotpotQA to one builder and BRIGHT plus the three small corpora to another
reduces the model-forward floor from 74.7 hours to approximately 47.9 hours before scan, flush,
checkpoint, query, and hashing costs.

## Row order and vector descriptors

`document-rows.jsonl` and `query-rows.jsonl` retain no text. Each row records:

```json
{"dataset":"demo","id":"document-17","kind":"documents","source_path":"datasets/demo/corpus/part-00000.jsonl","source_row":18,"stage":null}
```

Source paths are processed in their declared bytewise order. Lines retain file order. The SHA-256 of
the canonical row-order file becomes `row_order_sha256` for every corresponding current or old
matrix.

Each vector descriptor records:

- relative `.npy` path;
- dtype and two-dimensional shape;
- row-order SHA-256;
- exact byte count and file SHA-256;
- model revision and canonical model-tree SHA-256;
- prompt SHA-256;
- builder version.

NumPy `.npy` version 2.0 provides a typed header plus a C-order matrix. The builder writes through
`open_memmap`, so resident memory is bounded by one text batch, one encoded batch, and small control
objects. The entire corpus is never accumulated as a Python collection or NumPy array.

## Recovery and publication

An interrupted build leaves two objects beside the requested final path:

```text
.<name>.partial/
.<name>.checkpoint.json
```

The sidecar uses the distinct schema `fractal-embedding-store-checkpoint-v1`; the partial directory
contains no `receipt.json`. The checkpoint binds the staged inventory, selected source inventory,
full encoder configuration, current and old model bindings, encoder implementation IDs, row-order
descriptors, matrix filenames, and completed row counts.

Resume starts by repeating source and model verification. It then compares the checkpoint binding,
recomputes row-order digests from staged JSONL, verifies partial vector shapes and dtypes, and resumes
at the last flushed batch boundary. Config replacement, source replacement, model replacement,
tampered row order, and invalid progress all fail closed.

For a paired build, current and old memmaps are written and flushed before either progress value is
advanced. One checkpoint replacement records both equal row counts. A crash before that replacement
causes both slices to be recomputed. Unequal paired progress is rejected, as is substitution of
either arm-specific encoder identity. No intermediate vector cache is written.

Publication occurs only after every matrix reaches its registered row count and a full memmap scan
confirms finite, unit-normalized vectors. File sizes and SHA-256 values are computed after that scan.
The canonical receipt is written exclusively inside the partial directory, then the directory is
published with the operating system's no-replace rename primitive (`RENAME_EXCL` on macOS or
`RENAME_NOREPLACE` on Linux). An existing final path is never replaced, including an empty directory
created during a race.

## What the receipt establishes

The receipt establishes byte agreement between selected staged inputs, fixed encoder settings,
pinned local model trees, row order, and final vector files. It also distinguishes current and old
model products without relying on filenames alone.

It does not authenticate the caller-supplied upstream revision, prove that a GPU kernel is
bit-identical across hardware, or prove who controlled the machine. Those claims belong to external
source provenance, the scientific image and hardware receipt, and the provider-claimed runtime
evidence. None of those records establishes independent administration under the current
single-owner design.
