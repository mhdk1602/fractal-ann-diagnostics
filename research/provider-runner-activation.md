# Production runner registration and activation

The runner lifecycle has two typed transitions. Before apparatus commit A exists, `register`
captures the fixed runner under candidate source commit P. After C1 is frozen, `write` publishes
the A-bound bootstrap receipt required by production runtime. Keeping the records separate avoids
claiming that the pre-A GitHub observation executed code that did not yet exist.

Both transitions require `approval_environment=confirmatory` and derive the sole accepted runner
identity as `github-actions:environment:confirmatory`. That pair enters the pre-A claim nonce,
registration receipt, bootstrap receipt, every phase plan, C0 instantiation check, and post-A
activation receipt. Neither command accepts an environment or identity override.

## Register under P before plan finalization

`register` admits the closed provider-plan blueprint, revalidates its raw candidate shell and
P/T/D source closure, and selects one phase expectation. It then requires two byte-identical
GitHub runner-list responses while that runner is offline, idle, and has exactly the four expected
labels. The output path, runner name, nonce, label, version, archive digest, P, T, and both image
digests come from typed inputs; none is accepted on the command line.

```bash
phase='online' # repeat for label-release and analysis
controlled='/opt/fractal-confirmatory/host-tools'
blueprint='/absolute/control/provider-plan-blueprint'

install -d -m 0700 "$controlled/production/runner-registrations/$phase"
fractal-provider-runner-activation register \
  --blueprint-directory "$blueprint" \
  --phase "$phase"
```

The fixed registration directory contains exactly `registration-receipt.json`,
`repository-runner-inventory.json`, `repository-runners-api.raw.json`, and
`provider-runner-registration-receipt.json`. The first file is the P-bound
`ProviderRunnerBootstrapReceipt`; the fourth binds that receipt to the blueprint and retained
GitHub bytes. All members are mode `0600`, and the directory is mode `0700`.

```bash
fractal-provider-runner-activation verify-registration \
  --blueprint-directory "$blueprint" \
  --phase "$phase"
```

Registration has no manifest, C1 commit, C0 instantiation receipt, or apparatus-commit argument.
It cannot cross the pre-A boundary.

`fractal-provider-plans finalize` derives each registration directory from the closed blueprint.
It calls the same closed-bundle admission path, so a copied `registration-receipt.json` or any
other partial, phase-swapped, or hand-authored directory is inadmissible. The canonical
four-member bundle digest and the registration-evidence file digest enter both the phase plan and
the finalization receipt. The finalizer accepts no registration path or digest on its command
line.

## Activate under A after C1

The provider-plan operator cannot write the later production bootstrap receipt. After C1 is
frozen, `write` performs that post-A step without changing the manifest or accepting
caller-supplied hashes.

The writer admits five authorities together:

- the exact frozen C1 manifest and its C1 commit;
- the typed C0 control-instantiation receipt, which binds A, P, T, both OCI index digests D, and the
  candidate bootstrap closure;
- the exact candidate-image closure named by the C0 receipt;
- the fixed mode-`0600` materialized provider plan derived from the frozen manifest; and
- two identical live reads of the repository-runner API while the registered runner is offline,
  idle, and carries only `self-hosted`, `macOS`, `ARM64`, and its claim-derived nonce label.

The controlled `gh` executable comes from the provider plan and must match its registered binary
digest. There is no CLI option for a GitHub executable, output directory, runner identity, A, P, T,
D, or any SHA-256 value.

## Preconditions

Materialize each resolved C1 provider plan at the exact `provider_plan_path` stored in that plan.
The file must be owned by the current operator, singly linked, and mode `0600`. The P-registered
runner named by the plan must still be present in the repository inventory. Leave its listener
stopped.

The plan's Python import closure must also remain byte- and mode-identical to the pre-C0
host-tool contract. Its venv, `lib/python3.12` path, and site-packages tree are
`root:wheel`, have no write bits or ACL grants for the runner, and contain no symlinks,
hardlinks, or bytecode caches. The installed `fractal_ann_diagnostics` content equals the
clean package subtree at P; the C0 rehearsal separately proves that the same package
subtree is present at A. Replacing the venv or making its parent writable invalidates the
provider plan.

Create only the parent directory for the phase bundle. The activation writer creates the final
nonce-label directory atomically:

```bash
phase='online' # then label-release and analysis when each runner is provisioned
controlled='/opt/fractal-confirmatory/host-tools'
install -d -m 0700 "$controlled/production/runners/$phase"
```

If the registered runner disappeared or its ID, name, labels, official runner version, archive
digest, or inventory lineage changed, stop. A replacement changes the frozen provider plan and
requires a new candidate/C0/C1 lineage.

## Write the A-bound receipt

```bash
fractal-provider-runner-activation write \
  --manifest /absolute/c1/research/study-manifest.json \
  --c1-commit '<full C1 commit>' \
  --phase "$phase" \
  --c0-instantiation-receipt \
    /absolute/c0-controls/c0-control-instantiation-receipt.json \
  --candidate-image-closure /absolute/c0-evidence/candidate-closure.json
```

The output path is fixed by the plan:

```text
<controlled-root>/production/runners/<phase>/<claim-derived-label>/
```

The directory is mode `0700` and has exactly four mode-`0600` members:

- `bootstrap-receipt.json`, the exact A-resolved `ProviderRunnerBootstrapReceipt` embedded in C1;
- `repository-runner-inventory.json`, the typed live inventory;
- `repository-runners-api.raw.json`, the retained GitHub response bytes; and
- `provider-runner-activation-receipt.json`, the A/P/T/D, plan, claim, image, runner, and inventory
  binding.

Publication writes those members under a random private sibling, rereads each file through its open
descriptor, revalidates every immutable source and the live runner inventory, then performs one
no-replace directory rename. A failed or killed writer cannot expose a partial canonical bundle.
The writer also refuses an existing target, so it cannot silently replace the first activation.

## Verify before listener start

Run the retained verifier against the same immutable inputs:

```bash
fractal-provider-runner-activation verify \
  --manifest /absolute/c1/research/study-manifest.json \
  --c1-commit '<full C1 commit>' \
  --phase "$phase" \
  --c0-instantiation-receipt \
    /absolute/c0-controls/c0-control-instantiation-receipt.json \
  --candidate-image-closure /absolute/c0-evidence/candidate-closure.json
```

The verifier opens the activation directory once, binds all reads to that inode, requires the exact
four-member set and private modes, parses the three typed records from the bytes it admitted, and
rechecks them against the frozen plan. Start `run.sh` only after verification succeeds. Production
phase runtime independently reloads `bootstrap-receipt.json` and requires byte equality with C1.
Its first Python process uses the C0-fixed verified launcher; after that pre-import check,
activation rehashes the full host-tool closure and retains the resulting receipt before any phase
driver can open input.

The activation inventory proves one repository API observation before listener start. It does not
prove that GitHub will schedule the intended job. The claim job and live Jobs API verification
remain separate execution boundaries.
