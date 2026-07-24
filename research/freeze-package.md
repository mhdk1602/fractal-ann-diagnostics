# Freeze-package compiler

The freeze-package compiler prepares controlled storage without changing the study manifest or
opening the sealed run. It reads every artifact ID and role from the validated manifest, assigns a
non-overlapping local path, copies only repository code that already exists, and hashes every
present file or directory tree.

It does not download corpora, fit models, invent placeholder bytes, register the protocol, release
labels, create a run receipt, or run the confirmatory analysis.

## Prepare a package

Keep the package outside the Git checkout. The command creates `artifacts/`, `artifact-map.json`,
and `freeze-readiness.json` under the selected package root:

```bash
python -m fractal_ann_diagnostics.freeze_package compile \
  --repository-root "$PWD" \
  --manifest research/study-manifest.json \
  --package-root /controlled/fractal-v0.3
```

Five corpus-normalizer copies are written to separate paths. The exact-oracle module, controller,
and any local custody-builder module are also copied when their manifest URIs resolve to regular
files inside the repository. No artifact path overlaps the source archive or another artifact.
The five runtime-attestation plan templates are assigned to
`artifacts/runtime/<corpus-id>/runtime-attestation-plan.template.json`; there is no suite-wide plan
slot that can be reused across corpus commands. The singleton `opa-runtime-binary` is assigned to
`artifacts/runtime/opa`, the host source for the exact read-only `/usr/local/bin/opa` file mount.

After the C0 image workflow has published its digest and retained artifact package, materialize
that file from the selected platform's attested bytes:

```bash
python -m fractal_ann_diagnostics.opa_runtime_binary materialize-retained \
  --c0-package /controlled/c0-image-record \
  --image ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@sha256:<C0-index-digest> \
  --plan-root /controlled/fractal-v0.3/artifacts/runtime \
  --output /controlled/fractal-v0.3/artifacts/runtime/opa
```

The materializer derives `linux-amd64` or `linux-arm64` from the five plans. It loads only that
platform's closed `runtime-extraction.json`, requires the exact OCI index, selected manifest,
platform, C0 commit, in-image paths, digests, and byte counts, then checks the receipt, OPA, Python,
and `uv.lock` rows in `C0-ARTIFACT-SHA256SUMS`. It calls `gh attestation verify` separately for all
four files against the retained `c0-artifact-attestation-bundle.json`, with the fixed repository,
C0 workflow identity, C0 tag, source commit, GitHub OIDC issuer, and hosted-runner restriction. Only
then does it copy the retained OPA bytes once and set the destination to mode `0555`.

Every template must pin those same bytes at `/usr/local/bin/opa` through one file-kind, read-only
mount with role `opa-runtime-binary`. The five templates must also agree on the C0 image,
architecture, and code commit. The frozen manifest pins the executable as its own artifact with
`revision = sha256:<file-sha256>`, while `sealed_execution.runner_image` pins the source OCI digest.
GitHub artifact transport may discard the source executable mode; the copied bytes, receipt,
Sigstore evidence, and newly verified `0555` destination are the custody contract. A host-installed
OPA binary is never substituted.

Use `--no-copy-code` to inspect the plan first. Such rows appear as `generatable`; absent data,
models, binaries, receipts, and encrypted labels appear as `missing`. A nonempty path that can be
hashed appears as `present`. Presence is only a byte-level fact. It is not evidence that the object
satisfies its registered scientific or custody contract.

Several roles receive typed inspection rather than presence-and-hash treatment:

- `query-partition-audit` must be the canonical audit receipt, not a file containing a bare digest;
- `online-execution` must reproduce its sharded plan and exact package revision;
- `embedding-store` must reproduce its model, row-order, vector, config, and inventory bindings;
- `policy-workload` must contain exactly `fit`, `calibration`, and `sealed` intervention packages
  plus their canonical stage receipt;
- `authorized-index-store` must contain the same three stages and bind each index set to the
  matching policy stage and embedding store;
- `runtime-attestation-plan-template` must be canonical except for its single C1 manifest token;
- `opa-runtime-binary` must be a non-writable executable whose SHA-256 equals the attested C0
  extraction receipt and exact OPA file mount in all five templates, with the same C0 image,
  platform, and code commit;
- `power-analysis-report` is a closed directory at `analysis/joint-power-design/`.

The policy and index rules are specified in
[Five-corpus online artifact pipeline](artifact-pipeline.md). The joint-power directory is:

```text
config.json
report.json
selection-audit.json
panels/
  <registered-panel-sha256>.json
```

No other entry is allowed. The compiler canonical-loads the config, report, exact selection audit,
and panels; rejects test mode; freshly reproduces the exact audit; reruns the joint design over
those exact panels; requires identical audit and report bytes; and rehashes the tree after typed
inspection. This path calls the exact audit verifier once. The final tree readback is an integrity
check, not a second statistical replay.
It then checks the manifest's target, candidates, seed, simulation counts, scenario IDs,
dependence-source URI, selected family count, selected joint lower bound, endpoint order, fixed
12-cell family size, 95% familywise confidence, Bonferroni cell alpha, multiplicity method, and the
scientific scalars consumed by the simulator. The manifest pins the exact directory-tree digest.

The compiler refuses to overwrite a code copy that differs from its repository source. After a
reviewed source change, refresh those copies explicitly:

```bash
python -m fractal_ann_diagnostics.freeze_package compile \
  --repository-root "$PWD" \
  --manifest research/study-manifest.json \
  --package-root /controlled/fractal-v0.3 \
  --refresh-code
```

## Validate exact coverage

Draft placeholders do not prevent structural validation. This command requires exactly one map row
for every manifest artifact and rejects extra IDs, missing IDs, duplicate or overlapping paths,
and any path or file-kind assignment that differs from the compiler's layout:

```bash
python -m fractal_ann_diagnostics.freeze_package validate-map \
  --repository-root "$PWD" \
  --manifest research/study-manifest.json \
  --artifact-map /controlled/fractal-v0.3/artifact-map.json
```

This validation neither calls `begin-sealed-run` nor requires a frozen manifest digest.
Map coverage is an inventory control, not a mount or read permission. The online runner may open
only its separately admitted whitelist; sealed inputs, plaintext labels, and custody secrets remain
outside that process even though their artifact IDs appear in the complete map.

## Read the readiness report

`freeze-readiness.json` is canonical JSON with no timestamp or machine-specific absolute paths. For
each artifact it records the controlled path, byte state, deterministic SHA-256, file and directory
counts, source digest when applicable, unresolved manifest pin fields, and whether a final manifest
pin matches the observed bytes. When all six OPA runtime inputs are present, the top-level
`opa_runtime_binding` records the common image, C0 commit, architecture, platform, executable
digest, byte count, and five template digests.

`ready_for_freeze_review` remains false while any path is missing or generatable, a manifest pin is
unfinished or mismatched, or a declared freeze blocker remains. `sealed_run_authorized` is always
false because package preparation is not run admission.

The report also records the circularity-safe sequence:

1. Finish C0 executable code and development-only generation recipes.
2. Build the OCI image, source archive, runtime binaries, and derived artifacts from C0 without the
   study manifest.
3. Materialize all five canonical WorkloadSpec files, run the hardware preflight, and preserve the
   post-preflight runtime-plan templates and transition receipts before C1.
4. Create C1 as a manifest-only freeze commit that pins C0, the already built bytes, the complete
   five-object `production_workloads` section, and each canonical WorkloadSpec file digest.
5. Register the exact canonical C1 manifest with the independent registry.
6. Substitute only the registered manifest digest and the attested final closure digest into the
   preserved templates. Mount C1 and admitted artifacts read-only, verify them again, and invoke the
   one-shot opener only
   after registration and custody checks pass.
