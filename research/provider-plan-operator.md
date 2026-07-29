# Provider-plan construction before C0

The three provider phase plans must exist before the confirmatory apparatus commit A can
exist. A plan may therefore contain the registered C0 sentinel, but it cannot contain A,
the final candidate-manifest digest, or a hash derived from either value. The
`fractal-provider-plans` operator constructs that pre-A object from typed records. It has
no command-line option for a digest or a future commit.

The construction separates three identities:

- P is the source commit recorded by the candidate-image closure;
- T is the candidate build-context Git-tree digest; and
- D is each immutable OCI index digest shared by the candidate and production image
  names.

P, T, and D authenticate the executable candidate lineage. None is renamed A. The final
raw candidate carries the literal `containing-confirmatory-apparatus-c0-commit` sentinel
at its 13 registered scalar paths; C1 resolves those paths to A. The production-control
config and raw blueprint use the same boundary: P appears only as
`candidate_image_source_commit`, while executable commit fields remain sentinel-bound
until post-A control instantiation.

The same typed chain binds the protected GitHub environment. Production controls,
the blueprint, all three phase plans, both runner receipts, and the activation evidence
must carry `approval_environment=confirmatory` and
`runner_identity=github-actions:environment:confirmatory`. The identity is rederived
from the environment at every boundary; a ref-based identity or another environment is
inadmissible.

## Stage 1: closed blueprint

`write-blueprint` admits the source manifest shell, production-control config and its
write receipt, production-control blueprint, candidate-image closure, execution-beacon
contract, factory design seed, one clean checkout at P, fixed host tools, and three
operator-selected runner names. The source manifest must have
`provider_phase_plans: "tbd"`; hand-written plan JSON is rejected. The five workload
specs must already authenticate their raw sentinel-bearing bytes.

The host package is not an editable install and it is not imported from the operator's
checkout. Before blueprint construction, install only the locked runtime dependencies
under the fixed Python 3.12 venv, copy `src/fractal_ann_diagnostics` byte-for-byte from a
clean checkout at P into the fixed site-packages root, and remove every write bit from the
venv, import root, and their intervening directories. The operator rejects symlinks,
hardlinks, bytecode caches, dirty or ignored source bytes, a different Git tree at P, and
any content difference between the source package and installed package. It records
separate identities for:

- the Git commit P and `P:src/fractal_ann_diagnostics` tree object;
- the mode-independent package content;
- the permission-sensitive package and complete import-root trees; and
- the complete venv tree, symlink inventory, and verified-launcher source digest.

On macOS, strip ACLs and extended attributes before making the venv `root:wheel` and
read-only. Its parent chain must not be writable by the runner. Pre-create the separate
phase output directories with their intended non-root owner; do not recursively change
their ownership when sealing the venv.

```bash
fractal-provider-plans write-blueprint \
  --candidate-manifest /absolute/control/candidate-source-shell.json \
  --candidate-source-root /absolute/clean/worktree-at-P \
  --production-control-config /absolute/control/production-control-config.json \
  --production-control-config-write-receipt /absolute/control/production-control-config-write-receipt.json \
  --candidate-image-closure /absolute/control/candidate-image-closure.json \
  --execution-beacon-contract /absolute/control/execution-beacon-contract.json \
  --registered-online-runtime-budget-seconds 68000 \
  --controlled-root /opt/fractal-confirmatory/host-tools \
  --python-executable /opt/fractal-confirmatory/host-tools/python/bin/python3.12 \
  --venv-root /opt/fractal-confirmatory/host-tools/venv \
  --python-import-root /opt/fractal-confirmatory/host-tools/venv/lib/python3.12/site-packages \
  --gh-executable /opt/fractal-confirmatory/host-tools/gh/bin/gh \
  --runner-listener-executable /opt/fractal-confirmatory/host-tools/runner/bin/Runner.Listener \
  --runner-listener-dll /opt/fractal-confirmatory/host-tools/runner/bin/Runner.Listener.dll \
  --runner-config-executable /opt/fractal-confirmatory/host-tools/runner/config.sh \
  --runner-run-executable /opt/fractal-confirmatory/host-tools/runner/run.sh \
  --docker-executable /usr/local/bin/docker \
  --host-probe /absolute/control/phase-host-probe.json \
  --docker-server-probe /absolute/control/docker-server-probe.json \
  --claim-root /var/fractal-confirmatory/claims \
  --evidence-root /var/fractal-confirmatory/evidence \
  --online-runner-name fractal-registration-online \
  --label-release-runner-name fractal-registration-label-release \
  --analysis-runner-name fractal-registration-analysis \
  --output-directory /absolute/control/provider-plan-blueprint
```

The output directory is mode `0700` and has exactly two mode-`0600` members:

- `provider-plan-blueprint.json`;
- `provider-plan-blueprint-write-receipt.json`.

The claim nonce is derived from the source-shell digest, P, T, both D values, the
candidate bootstrap closure, production-control config, host-tool contract, runner name,
approval environment, runner identity, and fixed mutable roots. A and the final manifest
digest are absent from the derivation.

Commit A may differ from P, but its package subtree may not. The hosted rehearsal checks
out A and requires
`A:src/fractal_ann_diagnostics == P:src/fractal_ann_diagnostics` by Git tree object before
any runner is admitted. The rehearsal receipt retains both identities. A can freeze the
manifest and workflows without silently changing the P-bound host control package.

## Stage 2: registration evidence and raw templates

Each runner is registered while P is still the workflow source. The typed runner operator
is the only writer of the three registration bundles accepted by finalization. It reads
the closed blueprint, reauthenticates its source shell and P/T/D closure, selects the
runner by the blueprint's fixed name, and takes two identical live GitHub inventory reads.
The runner must be offline, idle, and labeled with exactly `self-hosted`, `macOS`, `ARM64`,
and its claim-derived nonce label.

Create only each phase parent, then run the P-bound command. It has no A, C1, final
candidate, digest, output, or runner-identity argument:

```bash
controlled='/opt/fractal-confirmatory/host-tools'
blueprint='/absolute/control/provider-plan-blueprint'

for phase in online label-release analysis; do
  install -d -m 0700 "$controlled/production/runner-registrations/$phase"
  fractal-provider-runner-activation register \
    --blueprint-directory "$blueprint" \
    --phase "$phase"
done
```

The fixed output is:

```text
<controlled-root>/production/runner-registrations/<phase>/<runner-label>/
```

It is a mode-`0700` closed directory with four mode-`0600` members:

- `registration-receipt.json`, the P-bound `ProviderRunnerBootstrapReceipt` admitted as
  one member of the closed bundle;
- `repository-runner-inventory.json`, the typed inventory receipt;
- `repository-runners-api.raw.json`, the exact GitHub response bytes; and
- `provider-runner-registration-receipt.json`, the blueprint, P/T/D, nonce, runner, and
  inventory binding.

Verify each retained bundle before finalization:

```bash
fractal-provider-runner-activation verify-registration \
  --blueprint-directory "$blueprint" \
  --phase online
```

Finalization derives the three fixed bundle roots from the closed blueprint and invokes the same
closed-bundle admission used by `verify-registration`. It accepts no registration path, receipt,
digest, or phase override. All four members, their private modes, the retained GitHub bytes, and
their typed cross-bindings must verify. The three bundles must have distinct runner IDs, names,
labels, repository-inventory digests, bundle digests, and registration-evidence digests.

```bash
fractal-provider-plans finalize \
  --blueprint /absolute/control/provider-plan-blueprint/provider-plan-blueprint.json \
  --blueprint-write-receipt /absolute/control/provider-plan-blueprint/provider-plan-blueprint-write-receipt.json \
  --output-directory /absolute/control/provider-plan-finalization
```

The finalization directory is mode `0700` and has exactly three mode-`0600` members:

- `provider-phase-plans.json`;
- `candidate-manifest.json`;
- `provider-plan-finalization-receipt.json`.

Each phase plan records the derived registration-bundle root, the canonical four-member bundle
digest, and the typed registration-evidence file digest. The finalization receipt repeats those
maps for all phases, alongside the bootstrap-receipt path and digest. The operator re-admits every
bundle immediately before publication and rejects any change to a path or digest.

`activation_argv_template` and `activation_environment` describe the inner isolated
Python activation, not the complete GitHub runner process. The argv template fixes the
interpreter, isolation flags, launcher-source binding, allowlisted module, command, and
argument order. The environment object is an exact ten-key subset: the nine C1 host
closure bindings plus `PYTHONDONTWRITEBYTECODE=1`. The workflow's outer shell supplies
job-scoped GitHub evidence, claim-artifact coordinates, dynamic output paths, and, for
label release, the completion-anchor token descriptor. Those wrapper values are
validated by their own typed contracts; they are not import-path authority and cannot
replace a key in the inner isolation subset.

The operator copies the registration identity into each embedded bootstrap template and replaces
only its `workflow_sha` with the C0 sentinel. The plan stores the SHA-256 of those raw
sentinel-bearing template bytes. The plan's future activation path is separate:

```text
<controlled-root>/production/runners/<phase>/<runner-label>/bootstrap-receipt.json
```

After A exists, production runner activation must materialize a
`ProviderRunnerBootstrapReceipt` at that path with `workflow_sha=A`. The plan operator
does not create this future receipt, and the candidate rehearsal's separate
`RehearsalRunnerBootstrapReceipt` is not a substitute. Candidate admission resolves the
sentinel in memory and rederives the expected activation-receipt digest. The P
registration receipt remains immutable evidence; it is not misrepresented as the A
activation receipt. The authoritative writer and retained verifier are specified in
[Production runner activation after C1](provider-runner-activation.md).

## Commit-dependent hashes

Eight file hashes change as a deterministic consequence of resolving A:

- three embedded runner-bootstrap receipt file hashes;
- five production-workload spec file hashes.

The raw candidate authenticates each sentinel-bearing template before resolution. The C0
candidate loader and C1 transition then replace the 13 registered scalar sentinels and
rederive those eight hashes in memory. Frozen validation runs only on that normalized
object. Caller-supplied replacement hashes are not accepted.

Finalization derives two synthetic commit witnesses after the candidate bytes exist and
loads the same raw candidate against both. The witnesses are not written into the
candidate. This proves that candidate admission is a function of a later valid commit,
without pretending either witness is A. The finalization receipt records one witness,
its normalized plan-closure digest, and the A-independent raw plan-template digest.

## Publication and failure behavior

Each stage writes its fixed members under a random private staging directory. It reads
every member back through its open file descriptor, rehashes every admitted source, and
then performs one parent-directory `RENAME_NOREPLACE`. A killed process can leave a
random staging directory. It cannot expose a partial canonical bundle, and a new run can
publish the still-absent target directory.

Finalization fails on a missing or hand-authored partial bundle, phase swap, reused path, changed
source file, changed host-tool closure, registration reuse, pre-existing output directory, extra
bundle member, non-private mode, retained-byte substitution, or typed candidate-loader failure. It
never edits the source manifest.
