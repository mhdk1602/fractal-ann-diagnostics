# Post-label input and single-analysis operator

This operator is the only command-line route from the attested `LABELS_RELEASED` state to a local
`ANALYSIS_COMPLETE` state. It does not accept a corpus list, an analysis configuration, model
names, Python callables, plugins, endpoint definitions, or estimands. The verified suite state fixes
the five online directories and released plaintext files. The frozen manifest fixes the model
roles and all statistical choices.

The console command is `fractal-confirmatory-input`. Its three subcommands use the same closed
configuration file:

```bash
fractal-confirmatory-input materialize --config /controlled/analysis/input-operator.json
fractal-confirmatory-input verify      --config /controlled/analysis/input-operator.json
fractal-confirmatory-input analyze     --config /controlled/analysis/input-operator.json
```

Each command first invokes the registered GitHub suite verifier and requests exactly
`LABELS_RELEASED`. The verifier reconstructs the protected append-only branch, checks the local
state prefix against the live Git objects, verifies each Sigstore bundle and Rekor entry, and
returns a file-backed `VerifiedSuiteLabelsReleased` capability. A copied JSON state or caller-made
Python object cannot satisfy this gate.

## Locator-only configuration

The configuration supplies local locations for evidence that the state record identifies by
digest but does not locate. It contains no scientific choice:

```json
{
  "artifact_root_uri": "file:///controlled/artifacts",
  "artifact_verification_receipt_uri": "file:///controlled/run/artifact-verification.json",
  "corpus_evidence": [
    {
      "corpus_id": "scifact",
      "prediction_completion_anchor_receipt_uri": "file:///controlled/completion/scifact-anchor-receipt.json",
      "prediction_completion_anchor_record_uri": "file:///controlled/completion/scifact-anchor-record.json",
      "prediction_completion_receipt_uri": "file:///controlled/completion/scifact-completion.json",
      "timelock_decryption_receipt_uri": "file:///controlled/released/scifact-decryption.json"
    },
    {
      "corpus_id": "hotpotqa-fullwiki",
      "prediction_completion_anchor_receipt_uri": "file:///controlled/completion/hotpotqa-fullwiki-anchor-receipt.json",
      "prediction_completion_anchor_record_uri": "file:///controlled/completion/hotpotqa-fullwiki-anchor-record.json",
      "prediction_completion_receipt_uri": "file:///controlled/completion/hotpotqa-fullwiki-completion.json",
      "timelock_decryption_receipt_uri": "file:///controlled/released/hotpotqa-fullwiki-decryption.json"
    },
    {
      "corpus_id": "t2-ragbench",
      "prediction_completion_anchor_receipt_uri": "file:///controlled/completion/t2-ragbench-anchor-receipt.json",
      "prediction_completion_anchor_record_uri": "file:///controlled/completion/t2-ragbench-anchor-record.json",
      "prediction_completion_receipt_uri": "file:///controlled/completion/t2-ragbench-completion.json",
      "timelock_decryption_receipt_uri": "file:///controlled/released/t2-ragbench-decryption.json"
    },
    {
      "corpus_id": "bright",
      "prediction_completion_anchor_receipt_uri": "file:///controlled/completion/bright-anchor-receipt.json",
      "prediction_completion_anchor_record_uri": "file:///controlled/completion/bright-anchor-record.json",
      "prediction_completion_receipt_uri": "file:///controlled/completion/bright-completion.json",
      "timelock_decryption_receipt_uri": "file:///controlled/released/bright-decryption.json"
    },
    {
      "corpus_id": "miracl-transfer",
      "prediction_completion_anchor_receipt_uri": "file:///controlled/completion/miracl-transfer-anchor-receipt.json",
      "prediction_completion_anchor_record_uri": "file:///controlled/completion/miracl-transfer-anchor-record.json",
      "prediction_completion_receipt_uri": "file:///controlled/completion/miracl-transfer-completion.json",
      "timelock_decryption_receipt_uri": "file:///controlled/released/miracl-transfer-decryption.json"
    }
  ],
  "manifest_uri": "file:///controlled/frozen-study-manifest.json",
  "schema_version": "fractal-confirmatory-input-operator-config-v1",
  "sealed_run_receipt_uri": "file:///controlled/run/SEAL-MANIFEST-SHA.json",
  "suite_namespace_uri": "file:///controlled/suite-attempt-MANIFEST-DERIVED-ID"
}
```

Replace the two uppercase placeholders with the actual manifest-derived names. Encode the file as
canonical JSON with one final LF. The parser rejects extra keys, duplicate corpora, reused evidence
URIs, noncanonical file URIs, missing corpus rows, nonfinite values, and alternate JSON bytes.

## Materialization checks

For each corpus, the operator takes the online root from `ONLINE_COMPLETE`, not from the config. It
requires the directory to contain exactly eleven files: runtime receipt, runtime marker, production
command attempt, sealed attempt, sealed result, prediction, action panel, panel-admission receipt,
audit chain, cache receipt, and execution-order receipt. It rehashes every file and checks both the
semantic and file digests recorded by the online result and suite closure. An extra file is a
membership failure.

The plaintext path comes from `LABELS_RELEASED`. The operator securely loads the actual canonical
`SealedLabelArtifact`, rehashes its complete newline-terminated file, and checks that digest against
both the state closure and the manifest's `sealed-labels` pin. It records the label object's
semantic digest separately. These identities differ by design:

```text
semantic_sha256 = SHA256(canonical JSON)
file_sha256     = SHA256(canonical JSON || LF)
```

They are not interchangeable. The manifest, timelock receipt, and label-release state carry the
file digest. The offline evaluation and confirmatory input carry the semantic object digest.

The completion receipt and both local anchor files come from the locator config. Their identity is
not trusted from those paths. The operator performs a fresh certificate-validated fetch of the
external anchor record, requires byte equality with the local record, and follows the recorded
digest chain through the completion receipt, online result, prediction, panel, decryption receipt,
and released plaintext. It then performs the exact prediction-label join for all trial keys. No row
may be added, removed, or moved between corpora.

`materialize` constructs one typed `ConfirmatoryInputArtifact`. It writes two manifest-derived files
inside the frozen `sealed_execution.results_store`:

```text
<manifest-sha256>.confirmatory-input-receipt.json
<manifest-sha256>.confirmatory-input.json
```

The detached receipt is reserved first with `O_EXCL`. It binds the suite attempt, exact
`LABELS_RELEASED` record, attestation descriptor, manifest, run receipt, input artifact, and a
canonical inventory of every source URI with its semantic digest, file digest, and byte count. The
input file is created second, also with `O_EXCL`. A crash after receipt creation leaves terminal
evidence; deleting the receipt to retry is not an admitted operation.

`verify` repeats the GitHub, external-anchor, online-directory, release, join, and file checks. It
reconstructs the typed input from current source bytes, requires the detached receipt to equal that
closure exactly, and requires the persisted artifact bytes to equal canonical JSON plus one LF.

## One analysis and local closure

`analyze` starts by running the full `verify` path. It locates the sole
`h1-predictive-model` and `h2-model-suite` roles through the frozen manifest and artifact-verification
receipt. The config cannot name replacements. Both files must be singly linked regular files whose
raw bytes equal the manifest pins and the canonical model serializers.

The existing one-shot analysis boundary then:

1. checks the `VerifiedSuiteLabelsReleased` capability again;
2. checks both model files against the input's admitted pins;
3. creates the manifest-derived analysis-attempt receipt before outcome computation;
4. computes only the preregistered H1 diagnostic and H2–H3 gates;
5. creates the detached result receipt and result file exclusively; and
6. creates local state `003.state.json` with `ANALYSIS_COMPLETE`.

The final state binds the input digest, attempt receipt, result receipt, final result, and their
persisted file digests. Publishing that state to the protected GitHub ledger and producing its
Sigstore evidence remain separate provider operations. A pre-existing input receipt, input file,
attempt receipt, result receipt, result, or state record stops the command. There is no retry flag.

The machinery proves byte identity and ordering relative to the registered GitHub, external anchor,
and timelock records. It does not prove independent human custody, lack of an undisclosed plaintext
copy, or absence of administrator access outside the measured process.
