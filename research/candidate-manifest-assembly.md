# Candidate manifest assembly

The C0 candidate manifest is an evidence product. It is not a form that an operator fills in.
In particular, the assembly path has no option for an artifact digest, revision, role, corpus, URI,
or license. Those values come from inspected bytes and typed receipts.

## Artifact-pin producer

The first stage now emits `fractal-candidate-artifact-pin-inventory-v1`. It starts with the tracked
draft solely as a structural schema. `freeze_package.layout_from_manifest` derives the only allowed
ID, role, corpus, path, and file/tree assignments. The producer then calls the same typed
`_inspect_target` boundary used by freeze review for every controlled target.

```bash
fractal-candidate-manifest-assembler artifact-inventory \
  --template "$repo/research/study-manifest.json" \
  --repository-root "$repo" \
  --artifact-root "$controlled/artifacts" \
  --output-directory "$controlled/candidate-artifact-pins"

fractal-candidate-manifest-assembler verify-artifact-inventory \
  --directory "$controlled/candidate-artifact-pins"
```

The output directory contains exactly two canonical JSON files:

- `candidate-artifact-pin-inventory.json`
- `candidate-artifact-pin-inventory-receipt.json`

Both files are mode `0600`; the directory is assembled privately and published with a no-replace
directory rename. A failed second write leaves no output directory. Publication refuses an existing
path.

The inventory has exactly 79 rows. Corpus-bound roles follow the registered five-corpus order.
Every row records its derivation class:

- local generated files and trees use their observed SHA-256 revision unless their typed receipt
  supplies a more specific logical revision;
- source code carries the sole C0 commit sentinel;
- copied repository code must match its repository file and uses that digest;
- benchmark inputs and labels use the verified staged-study-data inventory revision, while their
  own controlled file/tree digest remains the outer artifact pin;
- Qwen directories must match the registered current or stale model-tree digest, then use that
  model's immutable upstream revision rather than relabeling the tree digest as a revision;
- HNSW, OPA, and `tle` use the inspected typed tool or lock content;
- online execution, policy, embedding, index, power, OPA, and runtime-plan roles pass their existing
  typed verifiers before a row is admitted.

After the first pass, all 79 targets are rehashed. Any mutation in digest or accounting aborts the
transaction. A `tbd` URI in the tracked schema can become only the `file:` URI of its controlled
layout target. Non-placeholder URI and license text can only be copied from the tracked schema.

## Assembly boundary

`apply_candidate_artifact_inventory` changes only `uri`, `revision`, `sha256`, and `license` on the
79 structurally matched artifact rows. It rejects another template digest, a missing or extra ID,
a role/corpus substitution, and candidate/production locator confusion. The final gate is
`assert_candidate_manifest_closed`, which calls
`validate_candidate_rehearsal_manifest(candidate, c0_commit=A)`. That validator requires the exact
13 C0 sentinels and applies frozen scientific, workload, runtime, image, hardware, provider-plan,
and artifact admission after resolving only those sentinels in memory.

Once the typed analysis, workload, hardware, custody, image, and provider-plan producers have
materialized the candidate, publish it through the same boundary:

```bash
fractal-candidate-manifest-assembler publish-closed \
  --candidate "$controlled/candidate-study-manifest.source.json" \
  --artifact-inventory "$controlled/candidate-artifact-pins" \
  --candidate-image-closure "$controlled/candidate-image-closure.json" \
  --output-directory "$controlled/candidate-manifest-package"
```

This command accepts paths and has no manifest-field, digest, or future-C0 override. It
secure-loads the canonical candidate, typed image closure, and inventory,
checks the exact sentinel set, artifact multiplicities, provider-plan closure, and P/T/D image
bindings, then writes exactly `candidate-study-manifest.json` and
`candidate-manifest-assembly-receipt.json` as one no-replace transaction.

Treat that directory as the candidate artifact. Do not extract the manifest for the C1 handoff.
Supply the package together with the exact C0 boundaries:

```bash
fractal-c1-manifest-transition \
  --candidate-package "$controlled/candidate-manifest-package" \
  --c0-commit "$C0_COMMIT" \
  --c0-evidence-release "$controlled/c0-evidence-release-binding.json" \
  --output-directory "$controlled/c1-frozen-transition"
```

The loader reopens the directory under the same closed-membership, private-mode, ownership,
canonical-byte, manifest-digest, and provider-plan-closure checks. Its transition receipt carries
the assembly receipt file digest into the frozen C1 predicate and public registration package.

Raw candidate publication is A-independent. Candidate image and production-control evidence may
name bootstrap source commit P, but P is not copied into any of the 13 C0 sentinel slots. The
publisher uses a fixed internal probe commit only to exercise the candidate validator in memory;
neither that probe nor future C0 commit A enters the manifest or receipt. After the exact candidate
bytes are committed, the resulting Git commit is A and rehearsal resolves the registered sentinels
to A externally.

The remaining manifest sections must come from their existing typed producers before the final
package writer is invoked: post-embedding development and development freeze for analysis,
production controls for workloads and hardware, candidate image closure v2 for P/T/D, custody and
drand registration for sealed execution, and the three provider-plan operator records. If one of
those producers cannot state a required value, extend that producer's closed schema. Do not add a
digest or manifest-field flag to this CLI.

## Failure interpretation

An inventory failure means the controlled artifact set is incomplete, stale, or internally
inconsistent. It does not authorize substituting a path, copying a digest from CI output, or editing
the JSON. Correct the upstream typed package and rerun into a new, absent output directory.
