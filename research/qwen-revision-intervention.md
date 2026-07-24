# Qwen tokenizer-revision intervention

Status: acquisition and mechanism audit completed on 2026-07-14. This is an admissibility record,
not a sealed retrieval result.

## Decision

Admit the contrast, but name it precisely: it is a **tokenizer post-processor revision**, not a
weight revision. Both arms use `Qwen/Qwen3-Embedding-0.6B`. The stale arm is revision
`99cabfa1346cbf4ac8b0e73079bb2e286cff3a1f`; the current arm is revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

The model weights, model configuration, generation configuration, tokenizer configuration, merge
rules, and vocabulary are byte-identical. The consequential change is in `tokenizer.json`: the
current revision appends token `151643` (`<|endoftext|>`) through a `TemplateProcessing`
post-processor, while the stale revision ends after its `ByteLevel` processor. Since Qwen3
Embedding pools the final token, the stale tokenizer moves the pooling site from the trained
terminal token to the last content token.

This is a scientifically coherent stale-revision intervention because it holds the learned model
fixed and changes one executable preprocessing rule. It measures configuration drift in the same
model family. It must not be described as evidence about model retraining or weight drift.

## Official source evidence

The audit used the official Hugging Face repository and revision-addressed API responses:

- [current revision tree](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3)
- [stale revision tree](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/99cabfa1346cbf4ac8b0e73079bb2e286cff3a1f)
- [current revision metadata](https://huggingface.co/api/models/Qwen/Qwen3-Embedding-0.6B/revision/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3?blobs=true)
- [stale revision metadata](https://huggingface.co/api/models/Qwen/Qwen3-Embedding-0.6B/revision/99cabfa1346cbf4ac8b0e73079bb2e286cff3a1f?blobs=true)
- [commit that introduced automatic terminal-token processing and Sentence Transformers files](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/commit/b22da495047858cce924d27d76261e96be6febc0)
- [upstream discussion of the terminal-token and pooling change](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/discussions/2)

The raw API responses were retained outside Git:

| Receipt | SHA-256 |
| --- | --- |
| `non-git-files/confirmatory-v0.3/qwen3-current-hf-api.json` | `d6fc06e436c413ad7128356b4729b16a12b889697831001245e1f42594cbe1e6` |
| `non-git-files/confirmatory-v0.3/qwen3-pre-eos-hf-api.json` | `5f5f0c88e994dae4b9bdb6f9a67b85e108a18b25225e6f870ff8d21f86cd2903` |

The API resolves the requested SHAs exactly. It reports 12 repository entries and 1,207,489,041
bytes for the current revision, and 11 entries and 1,207,484,316 bytes for the stale revision. Each
count includes the identical 1,570-byte `.gitattributes` file. The local executable trees omit that
Git policy file in both arms.

## Byte audit

The following table covers every executable model or tokenizer file and the documentation change.
An equality sign means a direct local `cmp` succeeded in addition to matching the official object
metadata.

| Path | Stale revision | Current revision | Result |
| --- | --- | --- | --- |
| `model.safetensors` | 1,191,586,416 bytes; `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` | same | byte-identical weights |
| `config.json` | 727 bytes; `b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9` | same | byte-identical model config |
| `generation_config.json` | 117 bytes; `28396d421a2108acce96383f6a7de78008f7f1b17f807958f3c14c51dbfb65fb` | same | byte-identical |
| `tokenizer_config.json` | 9,706 bytes; `253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0` | same | byte-identical |
| `merges.txt` | 1,671,853 bytes; `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` | same | byte-identical |
| `vocab.json` | 2,776,833 bytes; `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` | same | byte-identical |
| `tokenizer.json` | 11,422,654 bytes; `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` | 11,423,705 bytes; `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a` | post-processor changed |
| `added_tokens.json` | 707 bytes; `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` | absent | stale-only sidecar |
| `special_tokens_map.json` | 613 bytes; `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` | absent | stale-only sidecar |
| `1_Pooling/config.json` | absent | 313 bytes; `37bf193fa101f19101bfad9c31d3eb0f786e247b7b1e5cb7f007d730eed1ddbd` | current-only last-token pooling config |
| `config_sentence_transformers.json` | absent | 215 bytes; `10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961` | current-only prompt and similarity config |
| `modules.json` | absent | 349 bytes; `84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf` | current-only wrapper graph |
| `README.md` | 13,120 bytes; `98b4ef4d1655aec0fc28ea60eebfbe4ff146abf9cb076d5f6232d2e996291769` | 17,237 bytes; `c34d9b7e5a267ad3fdd13227a253686bc90844ff4744a2a6a86c7c905e3d06f3` | documentation changed |

Canonicalizing both tokenizer JSON files and diffing them yields one semantic change. The stale
value is:

```json
{
  "post_processor": {
    "add_prefix_space": false,
    "trim_offsets": false,
    "type": "ByteLevel",
    "use_regex": false
  }
}
```

The current value is a sequence containing the same `ByteLevel` processor followed by this rule:

```json
{
  "type": "TemplateProcessing",
  "single": ["A", "<|endoftext|>"],
  "pair": ["A", "B", "<|endoftext|>"],
  "special_token": {"id": 151643, "token": "<|endoftext|>"}
}
```

The compact fragment above transcribes the operative rule; the retained tokenizer files hold the
exact upstream JSON schema and bytes.

## Local artifact bindings

The stale revision was downloaded to:

```text
non-git-files/confirmatory-v0.3/upstream/qwen3-embedding-0.6b-pre-eos
```

The canonical directory digest uses `digest_directory_tree`, which binds ordered relative paths,
entry types, file sizes, and file SHA-256 values. Links and irregular entries are rejected by that
implementation.

| Arm | Files | Directories | Bytes | Canonical tree SHA-256 |
| --- | ---: | ---: | ---: | --- |
| current | 11 | 1 | 1,207,487,471 | `0d1d985a7fb0500d53ebd83d2516ab6324bc9ad92b4fc88487b5a05437aef951` |
| stale | 10 | 0 | 1,207,482,746 | `742aaae08f118ef62ac498dba01241dd254a05b75cd7ce3d903c64785a8231df` |

The official LFS metadata and the downloaded files agree on both large objects:

- weights: `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`;
- current tokenizer: `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a`;
- stale tokenizer: `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.

For ordinary Git blobs, locally computed Git object IDs also match every `blobId` returned by the
official API. This checks the small files against Hugging Face metadata without treating a Git SHA-1
as a content SHA-256.

## Production freeze decision

The production query prefix follows the immutable official current
`config_sentence_transformers.json` exactly:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:
```

There is no byte after the colon. The prefix is 91 UTF-8 bytes and has SHA-256
`9fab7feb99edb69f560d85cf1bb849dd556c37f383d0336574f78f1303e90740`. The document prompt remains
the empty byte string.

The isolated production adapter in
[`qwen_revision_encoder.py`](../src/fractal_ann_diagnostics/qwen_revision_encoder.py) binds this
prefix, both revision and tree pairs, the arm-specific tokenizer digests, 512-token truncation,
last-active-token pooling, the first 256 coordinates, float32 unit normalization, and deterministic
batch seeds. Encoder v2 assigns content-relative position IDs and uses fixed token-length buckets,
so an unrelated long row cannot change a short row through left-padding position or bucket width.
The paired adapter reuses the causal prefix state for both arms below the 512-token boundary and
has been checked for byte equality against both independent v2 arms. It loads only absolute local
directories with Hugging Face offline flags,
`local_files_only=true`, and `trust_remote_code=false`. Its closed configuration rejects the earlier
trailing-space prefix and every other prompt substitution.

## Executed mechanism comparison

The inference path deliberately sits below the Sentence Transformers wrapper. Loading the stale
directory through a convenience wrapper would introduce a second treatment because that revision
does not contain `modules.json` or `1_Pooling/config.json`. Both arms instead use `AutoTokenizer`
and `AutoModel`, followed by the same explicit last-token pooling code.

Fixed mechanism-audit settings:

| Setting | Frozen value |
| --- | --- |
| query prompt | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ` |
| query prompt SHA-256 | `df4b2898bf22e00bacddddd489243a3f8793730e38b842ec10161cebd94d36d6` |
| document prompt | empty UTF-8 string |
| document prompt SHA-256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| maximum sequence length | 512 tokens |
| tokenizer | fast tokenizer, `add_special_tokens=true`, left padding, truncation enabled |
| pooling | final active token |
| output | first 256 coordinates, then L2 normalization |
| numeric path | CPU, float32, eager attention, deterministic algorithms |
| runtime | Python 3.12.13; PyTorch 2.13.0; Transformers 5.13.1; NumPy 2.5.1; macOS arm64 |

This earlier mechanism audit uses a trailing space after `Query:` in both arms. That common-mode
input does not confound its tokenizer-postprocessor contrast, reciprocal controls, or causal
interpretation. Its token and vector hashes are mechanism-audit artifacts, not production embedding
receipts. Production execution uses the no-space prefix frozen above.

Four constructed, label-free texts were encoded. The current tokenizer added exactly one terminal
token to every input. The stale tokenizer ended on `?` for queries and `.` for documents.

| Sample | Role | Current/stale tokens | Cosine | L2 distance | Maximum absolute difference |
| --- | --- | ---: | ---: | ---: | ---: |
| `query-1` | query | 32 / 31 | 0.770129323 | 0.678042293 | 0.275284410 |
| `query-2` | query | 31 / 30 | 0.787532628 | 0.651870072 | 0.213124081 |
| `document-1` | document | 14 / 13 | 0.859214604 | 0.530632555 | 0.196605802 |
| `document-2` | document | 16 / 15 | 0.879280806 | 0.491363645 | 0.153245389 |

Mean cosine similarity was `0.824039340`. Every vector was unit-normalized within float32 error,
and every current/stale pair had different vector bytes.

### Reciprocal causal controls

Two controls were run for every sample:

1. Manually append token `151643` to the stale token sequence.
2. Disable special-token processing in the current tokenizer.

For all four samples, the stale-plus-terminal token IDs and vectors were byte-identical to the
current standard arm. The current-without-special-processing token IDs and vectors were
byte-identical to the stale standard arm. All eight controlled comparisons had L2 distance `0.0`
and maximum absolute difference `0.0`.

This reciprocal result identifies the automatic terminal-token rule as the cause of the observed
vector shift in this execution. It also rules out model deserialization as an explanation for the
four measured contrasts.

## Reproduction receipts

The ignored runner and its canonical JSON receipt are:

| Artifact | SHA-256 |
| --- | --- |
| `non-git-files/confirmatory-v0.3/qwen-revision-intervention.py` | `a9e191a8902d430ef8b60c30723cd9271627da4da26dd9e89dc66dd2c08cd0c6` |
| `non-git-files/confirmatory-v0.3/qwen-revision-intervention-receipt.json` | `2db17509f916d907a29209b68644894555ef9729d921ce7a9618c935ea4ed978` |

The runner was executed twice. Both executions produced the same receipt SHA-256. The receipt binds
the two revisions, both canonical model-tree digests, every local file digest, prompt bytes,
settings, environment, sample text digests, token-sequence digests, vector digests, comparison
statistics, and reciprocal controls.

Reproduction command:

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
  non-git-files/confirmatory-v0.3/tooling-venv/bin/python \
  non-git-files/confirmatory-v0.3/qwen-revision-intervention.py
```

## Scope and limitations

This audit establishes an executable mechanism and a pinned stale artifact. It does not establish a
retrieval effect on the sealed corpora.

- The four texts are mechanism probes, not a representative query or document sample.
- No index was built, and no recall, ranking, hubness, calibration, or policy metric was inspected.
- None of the probes approaches the 512-token boundary. Truncation-boundary behavior remains a
  separate test.
- The receipt binds one CPU float32 software and hardware path. It does not claim bitwise agreement
  with GPU, MPS, bfloat16, or another library release.
- The reciprocal controls establish causation for the fast-tokenizer path used here. They do not
  certify every slow-tokenizer or third-party wrapper implementation.
- The current-only Sentence Transformers files are excluded from execution by design. A wrapper
  comparison would answer a different question because pooling configuration and wrapper discovery
  would change alongside tokenization.
- The stale-only token sidecars are retained exactly. The unchanged vocabulary, unchanged tokenizer
  configuration, canonical JSON diff, and reciprocal controls show that they do not explain these
  four fast-tokenizer contrasts. Broader compatibility claims would need their own tests.
- Hugging Face revision SHAs and object hashes establish repository identity and byte agreement.
  They do not attest who controlled the execution host.

For the sealed study, register this as a binary implementation factor with the treatment stated in
executable terms: **automatic terminal token `151643` present versus absent before last-token
pooling**. Freeze both tree digests and the exact prompt bytes before label custody releases any
outcome data. The production prompt digest must be
`9fab7feb99edb69f560d85cf1bb849dd556c37f383d0336574f78f1303e90740`; the trailing-space mechanism
receipt is inadmissible as a production vector store. Build separate vector stores and indexes for
the two arms. Only the single sealed execution may estimate effects on geometry, retrieval,
governance behavior, or downstream answers.
