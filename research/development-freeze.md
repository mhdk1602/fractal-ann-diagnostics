# Development-only confirmatory freeze compiler

## What this closes

[`development_freeze.py`](../src/fractal_ann_diagnostics/development_freeze.py) turns admitted
development evidence into the files that must become immutable before any sealed label is opened.
It does not read sealed data, register a protocol, or start the confirmatory run. Its sole job is to
make the remaining development judgments explicit and byte-addressable.

The compiler produces:

- separate fit and calibration feature batches;
- separate fit and calibration paired-action outcome panels;
- the H1 full-model bytes and the four-model H2 suite;
- one controller chosen from a declared finite grid;
- the static comparator `hnsw-high`, fixed without calibration selection;
- low- and high-geometry profiles derived from fit rows only;
- expected and conservative raw panels for `joint_power_design`;
- a production joint-power config with the registered candidate family counts; and
- one receipt that pins every output and every source digest.

No output path is replaced. The compiler writes a private staging directory, flushes every file,
then publishes the directory with an operating-system no-replace rename.

## Admissible sources

Each of the five fixed corpora must have one `development-fit` source bundle and one
`development-calibration` source bundle. A bundle contains these exact inputs:

| Input | Content | Required pin |
|---|---|---|
| queries | Canonical staged `queries.jsonl` | SHA-256 and byte count |
| qrels | Canonical staged `qrels.jsonl` | SHA-256 and byte count |
| evidence bundles | Canonical staged evidence JSONL for SciFact, HotpotQA FullWiki, and T2-RAGBench | SHA-256 and byte count |
| policy schedule | Canonical compiled policy schedule | SHA-256 and byte count |
| paired actions | All four realized action outcomes | SHA-256 and byte count |
| embedding store | Old and current query/document vector matrices plus row orders | exact embedding-store receipt SHA-256 |

The config also carries one direct exact pin to the canonical
[`fractal-development-cohort-selection-v1`](development-cohort.md) receipt. The compiler loads this
receipt before qrels and requires every corpus-stage query file to contain exactly its selected
representatives. Policy schedules and paired actions therefore cannot substitute an
outcome-chosen query set.

The paired-action and policy-schedule pins should come from the
[outcome-blind development execution](development-execution.md). Its `write-freeze-config` command
reverifies the materialization, policy, embedding, authorized-index, execution-order, and action
receipts before it emits this compiler's config. A hand-assembled config remains schema-valid only
if it reproduces those exact pins, but it is not the registered operator path.

The evidence path must be absent for BRIGHT and MIRACL Transfer. It must be present for the three
evidence corpora.

Before opening a source byte, the compiler examines every supplied path. Any path token equal to
`sealed`, `custody`, `holdout`, `heldout`, `reserve`, or `reserved` aborts the compilation. This
check covers query, qrel, evidence, schedule, paired-action, and embedding-store paths. A test
replaces the source loader with a sentinel and proves the sentinel is never reached when such a
path is supplied.

The source role is also closed by type. There is no generic file role and no `sealed` partition
value in the compiler config.

## Paired-action input contract

The paired-action file is canonical JSONL. Every scheduled trial must have four rows in this order:

1. `hnsw-low`;
2. `hnsw-high`;
3. `exact-authorized`; and
4. `abstain`.

Each line has exactly these fields:

```json
{
  "action": "hnsw-low",
  "entitlement_violations": 0,
  "execution_state": "completed",
  "failure_state": null,
  "family_id": "development-family-id",
  "feature_values": {
    "allow_rate": 0.25,
    "authorized_universe_size": 250,
    "backend": "hnswlib-0.8.0",
    "corpus_size": 1000,
    "corpus_stratum": "scifact",
    "drift_family": "revision-lag-one",
    "drift_severity": 0.013,
    "embedding_dimension": 1024,
    "lid_cv": 0.08,
    "lid_k50": 11.4,
    "policy_churn": 0.01,
    "policy_complexity": 2,
    "probe_latency_ms": 0.74,
    "probe_work": 101,
    "radius_expansion": 1.17,
    "relative_contrast": 1.42,
    "version_lag": 1
  },
  "query_id": "development-query-id",
  "request_latency_ms": 4.12,
  "returned_document_rows": [12, 87, 103],
  "schedule_order": 0,
  "execution_position": 2,
  "schema_version": "fractal-development-paired-action-row-v2",
  "trial_key": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
}
```

The displayed object is formatted for reading. Production JSONL uses sorted keys, compact
separators, finite JSON values, and one terminal newline per row.

Only `hnsw-low` carries the model feature vector. The other three rows encode `feature_values` as
JSON `null`. Every action row must match the schedule's trial key. The exact action must complete.
Failed actions retain their measured request latency and an explicit failure state; they cannot
claim returned documents. `execution_position` is the zero-based position in the separately
validated balanced action order. All four rows for a trial must retain the permutation exactly.

## Source-derived feature checks

Six feature fields are recomputed or bound from independent artifacts:

- `corpus_size` equals the embedding-store and policy-schedule document count;
- `authorized_universe_size` equals the scheduled authorized count;
- `embedding_dimension` equals the pinned current-query vector width;
- `version_lag` equals one;
- `allow_rate` equals the registered target stratum; and
- `drift_severity` equals the query-specific dual-epoch cosine drift.

The drift value is

\[
d(q)=1-\frac{q_{\text{active}}^\top q_{\text{current}}}
{\lVert q_{\text{active}}\rVert_2\lVert q_{\text{current}}\rVert_2}.
\]

`q_active` comes from `old_queries.npy`. `q_current` comes from `current_queries.npy`. Zero-norm,
nonfinite, unmatched, or dimensionally inconsistent rows are rejected. The old document matrix is
the ANN epoch; the current document matrix is the exact-truth epoch. The compiler pins and verifies
both matrices. It does not rerun retrieval after seeing labels.

The remaining telemetry arrives from the completed label-free action run. The compiler rejects a
caller value that disagrees with any independently derived field above.

## Development trial design

The design has three nested rows per query family:

| Field | Frozen value |
|---|---:|
| subjects | 1 |
| repetitions | 1, encoded as repetition `0` |
| target allow rates | `0.25`, `0.50`, `0.75` |
| version lag | `1` |
| actions per nested row | 4 |

The realized allow rate may differ from its target by at most one document divided by corpus size.
Every family must contain all three target rates. Each corpus needs at least two families and both
binary low-effort outcomes in fit and calibration. Fit and calibration family identifiers must be
disjoint.

The low-effort action-failure label is one when `hnsw-low` fails to complete or its recall against
the completed current-epoch `exact-authorized` result is below `0.90`. Otherwise it is zero. Qrel
recall is retained separately in the outcome artifact. Complete-evidence sufficiency requires all
locations in at least one gold bundle to be present, with zero entitlement violations.

## Frozen model bytes

The compiler calls the existing `fit_frozen_model_suite` implementation with seed `20260713`:

- fit rows estimate imputation, scaling, categorical levels, and logistic coefficients;
- calibration rows estimate Platt calibration;
- H1 is the canonical `full` model object; and
- H2 is the canonical four-model suite.

The model artifact already binds the ordered feature schema, fitted parameters, scikit-learn
version, development group hashes, model digests, and suite digest. The freeze receipt repeats the
feature-schema, development-group, and suite digests.

## Controller selection

The compiler evaluates 45 rule-controller candidates. All candidates fix:

```text
low_ef = 128
high_ef = 512
probe_k = 101
```

The declared grid crosses:

```text
exact_scan_threshold in {128, 256, 512}
high_effort_threshold in {0.15, 0.20, 0.25, 0.30}
exact_threshold in {0.30, 0.35, 0.40, 0.45}
high_effort_threshold < exact_threshold
```

For each candidate, the compiler selects a realized action for every calibration row and evaluates
the choice against `exact-authorized`. A candidate is admissible only when all four point rules
hold:

- equal-corpus retrieval-target loss is at most `0.005`;
- equal-corpus evidence-sufficiency loss is at most `0.005`;
- selected actions emit zero denied documents; and
- the equal-corpus p95 request-latency ratio is strictly below `1.20`.

Among admissible candidates, selection maximizes equal-corpus family-level latency reduction.
Objective values are rounded to 15 decimal places only for tie detection. A tie follows the
registered least-complex order: narrower effort gap, smaller exact-scan threshold, then thresholds
that trigger fewer escalations. The output preserves every candidate, every metric, the pass flag,
and the selected row. If no candidate passes, nothing is published.

The static comparator does not enter this search. It is `hnsw-high` by prior declaration and is
written to its own artifact with `selection_data: null`.

## Geometry profiles and practical minima

Profile values use development-fit rows only. NumPy's linear quantile rule computes the 25th and
75th percentiles.

- low geometry uses the 25th percentile for `lid_k50`, `lid_cv`, and `radius_expansion`, and the
  75th percentile for `relative_contrast`;
- high geometry reverses those choices because lower relative contrast indicates greater search
  difficulty.

The H2 practical gain minima are fixed constants, never estimated from the development outcomes:

| Endpoint | Minimum gain |
|---|---:|
| log-loss reduction | `0.002` |
| Brier-score reduction | `0.001` |
| AUPRC gain | `0.005` |

## Expected and conservative power panels

The expected panel uses calibration rows exactly as observed after applying the frozen controller.
`hnsw-high` supplies the comparator columns. The frozen `system-policy` and `full` models supply
the two row probabilities.

Power also reports the prespecified position-adjusted log-latency sensitivity. Simulation assigns
successive cyclic Latin rows across the ordered family/row stream, fits the same within-corpus
linear proposed-minus-comparator position term as the sealed analysis, and calibrates its one-sided
upper bound. This endpoint is excluded from the family-count selection conjunction; it cannot
redefine the registered raw-latency estimand.

The conservative panel is generated by one fixed adverse map. No row receives a manual edit.

For each row, let \(z_R\) be the reference-model logit, \(z_F\) the full-model logit, and
\(r=T_A/T_S\) the proposed-to-static latency ratio. The conservative full-model probability is

\[
p_F^{C}=\operatorname{logit}^{-1}\left[z_R+0.75(z_F-z_R)\right].
\]

The conservative latency ratio is

\[
r_C=\begin{cases}
1-0.75(1-r), & r<1,\\
1+1.25(r-1), & r\ge 1.
\end{cases}
\]

Thus only 75% of an observed latency benefit remains, while an observed penalty is enlarged by
25%.

Retrieval and evidence successes face a deterministic 10% adverse drop. The compiler hashes
`endpoint + NUL + row_id`, reads the first unsigned 64 bits, and drops a success when that value is
below `0.10 * 2^64`. A false outcome never becomes true. Comparator outcomes, labels, and denied
emission counts remain unchanged. The exact attenuation constants and hash construction are stored
in `scenario-attenuation.json`.

Both panels are marked `selection_required=true` in the power config. The config fixes:

```text
candidate families per corpus = 25, 50, 75, 100, 150, 200
nested rows per family = 3
calibration simulations = 5,000
power simulations = 5,000
simulation seed = 20260713
```

The compiler prepares these inputs. It deliberately does not execute the production power run.

## Output package

The no-replace package contains exactly:

```text
controller.json
development-calibration-features.json
development-calibration-outcomes.json
development-fit-features.json
development-fit-outcomes.json
freeze-receipt.json
geometry-profiles.json
h1-model.json
h2-model-suite.json
joint-power-config.json
joint-power-conservative-panel.json
joint-power-expected-panel.json
scenario-attenuation.json
static-comparator.json
```

`verify_development_freeze` rejects extra or missing files, changed sizes or digests, alternate
canonical encodings, schema drift, a comparator other than `hnsw-high`, model-suite mismatch, H1
bytes that differ from the suite's full model, or scenario panels not pinned by the power config.
It then reconstructs typed fit and calibration partitions, refits the model suite, reevaluates all
45 controller candidates, recomputes the fit-only profiles, and regenerates both scenario panels
and the power config. Every derived byte must reproduce. Rewriting a changed file and updating its
receipt digest therefore remains insufficient.

## Canonical compiler config

The module CLI accepts one closed JSON document. The document must use UTF-8, compact separators,
lexicographically sorted object keys, ASCII escapes, finite JSON numbers, and exactly one final LF.
Duplicate keys, unknown keys, alternate whitespace, a byte-order mark, and trailing data are errors.
The top-level object has exactly these fields:

| Field | Exact value or type |
| --- | --- |
| `failure_recall_threshold` | JSON number `0.9` |
| `k` | JSON integer `10` |
| `model_seed` | JSON integer `20260713` |
| `output_root` | canonical absolute POSIX path |
| `schema_version` | `fractal-development-freeze-config-v2` |
| `selection_receipt` | exact `byte_count`, absolute `path`, and `sha256` pin |
| `sources` | array of the ten source objects described below |

Each source object has exactly these fields:

| Field | Exact value or type |
| --- | --- |
| `corpus_id` | `scifact`, `hotpotqa-fullwiki`, `t2-ragbench`, `bright`, or `miracl-transfer` |
| `embedding_store` | object with exactly `receipt_sha256` and `root` |
| `evidence_bundles` | file pin for an evidence corpus; otherwise JSON `null` |
| `paired_actions` | file pin |
| `policy_schedule` | file pin |
| `qrels` | file pin |
| `queries` | file pin |
| `stage` | `development-fit` or `development-calibration` |

A file pin has exactly `byte_count`, `path`, and `sha256`. `byte_count` is a positive JSON integer
no larger than 256 GiB. `sha256` and `receipt_sha256` are 64 lowercase hexadecimal characters.
An embedding-store pin has exactly `receipt_sha256` and `root`. The array contains one source for
each fixed corpus at each development stage, ordered canonically by `(stage, corpus_id)`. Evidence
bundles are present exactly for `scifact`, `hotpotqa-fullwiki`, and `t2-ragbench`; they are JSON
`null` for `bright` and `miracl-transfer`.

Every `path`, `root`, and `output_root` value is a canonical absolute POSIX path. The config path
itself obeys the same rule. A path is rejected before any source is opened if one of its components
contains `sealed`, `custody`, `holdout`, `heldout`, `reserve`, or `reserved` as a token. The tokens
`changeme`, `latest`, `placeholder`, `replace`, `tbd`, `todo`, `unassigned`, and `unset` are also
rejected. Backslashes, control characters, placeholder metacharacters (`<>{}$*?`), `.` or `..`
components, a root-only path, and noncanonical spellings are invalid.

This formatted fragment shows the complete object shapes. It is not an admissible config because
it contains symbolic values and formatting whitespace:

```json
{
  "failure_recall_threshold": 0.9,
  "k": 10,
  "model_seed": 20260713,
  "output_root": "/controlled/development/freeze-v1",
  "schema_version": "fractal-development-freeze-config-v2",
  "selection_receipt": {
    "byte_count": 123,
    "path": "/controlled/development/selection-receipt.json",
    "sha256": "<64-lowercase-hex>"
  },
  "sources": [
    {
      "corpus_id": "<fixed-corpus-id>",
      "embedding_store": {
        "receipt_sha256": "<64-lowercase-hex>",
        "root": "/controlled/development/<stage>/<corpus>/embedding-store"
      },
      "evidence_bundles": {
        "byte_count": 123,
        "path": "/controlled/development/<stage>/<corpus>/evidence-bundles.jsonl",
        "sha256": "<64-lowercase-hex>"
      },
      "paired_actions": {
        "byte_count": 123,
        "path": "/controlled/development/<stage>/<corpus>/paired-actions.jsonl",
        "sha256": "<64-lowercase-hex>"
      },
      "policy_schedule": {
        "byte_count": 123,
        "path": "/controlled/development/<stage>/<corpus>/policy-schedule.jsonl",
        "sha256": "<64-lowercase-hex>"
      },
      "qrels": {
        "byte_count": 123,
        "path": "/controlled/development/<stage>/<corpus>/qrels.jsonl",
        "sha256": "<64-lowercase-hex>"
      },
      "queries": {
        "byte_count": 123,
        "path": "/controlled/development/<stage>/<corpus>/queries.jsonl",
        "sha256": "<64-lowercase-hex>"
      },
      "stage": "development-fit"
    }
  ]
}
```

Create the typed value, then write
`canonical_development_freeze_config_bytes(config)` verbatim. Hand-formatted JSON is rejected.
`load_development_freeze_config` validates the entire document and every path before the compiler
can open a development source.

## Module CLI

Compile into the config's absent `output_root`:

```bash
python -m fractal_ann_diagnostics.development_freeze compile \
  --config /controlled/development/development-freeze-config.json
```

Verify an existing package by reconstructing every derived artifact:

```bash
python -m fractal_ann_diagnostics.development_freeze verify \
  --root /controlled/development/freeze-v1
```

Success writes one canonical JSON result to stdout. For example:

```json
{"command":"compile","receipt_sha256":"<freeze-receipt-sha256>","root":"/controlled/development/freeze-v1","schema_version":"fractal-development-freeze-cli-result-v1"}
```

Validation failures write a diagnostic to stderr and return a nonzero status. `compile` retains the
compiler's no-replace semantics: the output directory must not exist. `verify` accepts no config
and trusts no caller-supplied digest; it reads the package receipt, checks each pinned byte, and
reconstructs the freeze.

## Typed Python API

The module also exposes a typed Python API without adding this label-admitted operation to the
package-level CLI or package export surface. Construct ten `DevelopmentCorpusSources` values with
exact pins, then call:

```python
from pathlib import Path

from fractal_ann_diagnostics.development_freeze import (
    DevelopmentFreezeConfig,
    PinnedDevelopmentSelectionReceipt,
    compile_development_freeze,
    verify_development_freeze,
)

config = DevelopmentFreezeConfig(
    sources=tuple(all_fit_and_calibration_sources),
    selection_receipt=PinnedDevelopmentSelectionReceipt(
        path=Path("/controlled/development/selection-receipt.json"),
        sha256=selection_receipt_sha256,
        byte_count=selection_receipt_byte_count,
    ),
    output_root=Path("/controlled/development/freeze-v1"),
)

receipt = compile_development_freeze(config)
verified = verify_development_freeze(config.output_root)
assert verified == receipt
```

The output directory must not exist. Source SHA-256 values and byte counts must come from the
staging and action-run receipts, not from a digest computed ad hoc during this call.

## Freeze acceptance

Before these bytes enter the registered study manifest, an independent reviewer should confirm:

1. the checked-in config is byte-identical to
   `canonical_development_freeze_config_bytes(load_development_freeze_config(path))`;
2. all 57 source bindings resolve to the pre-label selection and intended fit or calibration
   artifacts;
3. no source artifact ID, path, row, or family crosses the development boundary;
4. the H1 and H2 bytes reproduce from the pinned development sources;
5. every controller candidate and metric reproduces byte-for-byte;
6. geometry profiles reproduce from fit rows alone;
7. expected and conservative panels reproduce from the pinned rule;
8. a second CLI compile to a new empty directory yields the same artifact bytes; and
9. the CLI `verify` command accepts both copies and rejects a one-byte mutation.

Only after that review should the artifact URIs and digests be inserted into the frozen manifest,
deposited with the external protocol record, and transferred to the independent label custodian.
