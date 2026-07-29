# Provider phase workflows

The confirmatory provider path has three manually dispatched workflows:

- online execution;
- label release;
- confirmatory analysis.

Each accepts one value: the 64-character lowercase hexadecimal suite-attempt ID derived
from the frozen manifest. Paths, runner labels, timestamps, beacon rounds, image
identities, and output locations are not dispatch inputs.

## State transition pattern

Each phase has the same four-job structure.

1. A GitHub-hosted claim job verifies the public C1 package and both C1 attestations,
   checks the exact C0 workflow identity, and publishes the phase claim with compare-and-
   swap state semantics. Verification emits two distinct provider-plan paths. The
   materialization path names the hosted evidence copy and is retained in the claim
   artifact. The provider path is the C1-fixed canonical absolute path on the ephemeral
   macOS host and is the only plan path forwarded to the execute job. Both paths bind the
   same file digest.
2. The claim derives a phase-specific runner label from the registered claim nonce. The
   execute job can run only on the ephemeral self-hosted runner carrying that label.
3. The execute job checks the C1-pinned host Python, GitHub CLI, Docker client and resolved
   target, Actions runner listener, bootstrap receipt, and provider plan before the guarded
   activation call. The first Python process uses `-I -S -P -s` and the workflow-fixed
   `fractal-host-python-verified-launcher-v1` program. That program verifies the
   permission-sensitive venv, import-root, and package identities before inserting the
   sole C1-bound site-packages root and invoking the allowlisted module with `runpy`.
   Its exact source SHA-256 is part of the pre-A host-tool contract; the workflow literal
   must hash to that value. There is no preliminary import and no second Python start.
   The activation call then
   re-reads GitHub job evidence, binds the actual runner ID, runner name, runner group, job
   identity, run ID, and run attempt to the C1 contract, then performs the fixed phase
   operation once. The short-lived GitHub job token is present only for this host-side API
   readback. The guard removes it before Docker starts; it cannot enter container
   environment, command arguments, receipts, artifacts, or logs.
4. A GitHub-hosted completion job downloads the subject file, attests that file by path,
   and publishes the terminal state only after it verifies the returned bundle and Rekor
   evidence. If any claimed phase does not complete, the failure job recovers the live
   claim by suite-attempt ID and uses the same prepare, attest, publish sequence.

The self-hosted jobs do not check out source, install Python, run `uv sync`, or accept a
tool path from the dispatcher. They do not inherit the hosted jobs' `PYTHONPATH`, admit
the working directory, process user-site files, or execute raw `python -m` and
`python -` forms. All eleven self-hosted calls, including the two post-execution
validators that formerly used stdin scripts, pass through the same launcher and
allowlist. The import closure was constructed before C0 from package P, is read-only to
the runner, and is fixed by C1. The rehearsal additionally proves that the package
subtree did not change between P and workflow commit A.

The authority boundary is narrower than a hostile-host proof. Construction seals and
checks the complete venv; each phase rehashes that venv's bytes and modes, then checks
root ownership and effective non-writability along the import-root ancestors and every
site-packages entry. The official standalone Python distribution and its standard
library sit outside that import root. C1 binds the registered archive and interpreter
binary, but the privileged seal and later administration of that distribution remain
in the honest-administrator trusted computing base. The apparatus does not claim
protection from a root administrator who replaces the interpreter while forging its
own observations.

The online job is capped at 1,380 minutes so its job token cannot cross GitHub's 24-hour
lifetime. Label release and analysis are capped at 360 and 720 minutes. Hosted claim,
completion, and failure jobs pin Python 3.14.6; C1 binds the official Ubuntu 24.04 x64
archive and installed verifier-environment digests.

After the guarded import, production activation rehashes the full C1 host-tool closure
before it reads a beacon, opens phase input, or starts Docker. It writes a canonical
host-tool receipt once and carries both the semantic receipt digest and the retained-file
digest through the runtime request and phase execution receipt. The evidence package
therefore records the host closure observed during the sealed phase, rather than only the
pre-C0 construction claim.

## Runtime image roles

The guarded core selects the runtime from the verified phase contract. Online execution
uses the ARM64 platform manifest of the multiarch scientific image. Analysis uses its
AMD64 platform manifest under the registered emulation path. Both scientific manifests
must prove that `tle` is absent and start with container networking disabled.

Label release cannot reuse either scientific manifest. Its contract names a separate
ARM64 timelock-release image and the exact `tle` binary digest. The workflow accepts that
digest only from the verified C1 provider plan; it is not a dispatch field. The workflow
also checks that value against the C0-registered source-build digest. The source build and
binary equality, static proof, offline tests, two-way Quicknet interoperability, and the
TLE-only raw scan are complete. C1 release-image registration remains blocked on the
second no-cache OCI projection, exact manifest/config/layer equality, full-image scan and
adjudication, signed receipt, and anonymous readback.

The phase binding is closed in the core contract: online is
`linux/arm64 + scientific + main`, label release is
`linux/arm64 + timelock-release + release`, and analysis is
`linux/amd64 + scientific + main`. The prerequisite receipt emits all three values; the
execute job compares them with the C0 workflow constants before it can open phase input.

## Attestation storage

Every `actions/attest` call sets `create-storage-record: false`. GitHub storage records
for artifact attestations are available only to organization-owned repositories; this
repository is owned by the personal account `mhdk1602`. Enabling that input would request
a repository storage path that this ownership model cannot supply. The jobs therefore do
not request `artifact-metadata: write`, which the action needs only when it creates that
storage record. See the
[`actions/attest` input contract](https://github.com/actions/attest#usage) and GitHub's
[artifact-attestation documentation](https://docs.github.com/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds).

This setting does not turn the attestation into a caller-provided digest assertion. The
action signs the downloaded subject path, returns the Sigstore bundle, and records the
public-repository signing event through the public-good transparency service. The phase
publisher verifies the subject bytes, bundle, certificate identity, and Rekor evidence
before it advances state. Claim, completion, and failure packages retain the returned
bundle as a 90-day Actions artifact. The terminal state binds its digest, so deleting or
substituting a retained bundle is detectable.

Subject and predicate bytes are checked against core-emitted SHA-256 values before every
attestation. Completion preparation writes checksum files beside the subject, predicate,
and preparation receipt. The hosted completion job checks those files after artifact
download and before `actions/attest`; publish then rehashes the same bytes against the
preparation receipt and returned Sigstore bundle. The final publication receipt, state
record digest, and ledger commit are checked before the completion evidence is uploaded.

## Evidence behavior on failure

All evidence-upload steps use `if: always()`. Missing receipts, activation output, or
attestation bundles produce explicit marker files instead of an empty artifact or a shell
error that hides the earlier fault. Failure preparation does not trust claim-job outputs:
it identifies the suite from the sole dispatch input, reads the provider ledger, and
mutates state only when one live claim exists for that phase. Failure attestation and
publication are skipped for the typed no-claim result. A successful terminal completion
prevents the failure job from running.

## Live provider rehearsal before C1

The production phase workflows must not be used as smoke tests. A rehearsal must not mint a
suite-attempt ID, read a beacon, create a claim, write the append-only provider ledger, or launch
scientific input. Deleting a rehearsal branch later is also inadmissible because the ledger rules
forbid that cleanup pattern.

The mechanism is the separately named
`.github/workflows/confirmatory-provider-rehearsal.yml` included in C0. It has no `contents: write`
permission and no call path into the claim, completion, or failure publishers. Its hosted admission
job verifies the candidate image run through the Actions API, then reads
`sealed_execution.provider_phase_plans` through the same typed loader used in production. The
loader's production mode accepts only a frozen manifest. Candidate mode is a separate fail-closed
admission path with four lifecycle rules: `status` is `draft`, `protocol_version` is
`0.3.0-draft`, `freeze_blockers` is nonempty, and `sealed_execution.c0_evidence_release` is exactly
`"tbd"`. Every analysis, artifact, workload, host, image, runtime, and provider-plan value otherwise
faces frozen-manifest validation.

Candidate mode recognizes one sentinel,
`containing-confirmatory-apparatus-c0-commit`, at exactly 13 registered paths:

- `sealed_execution.code_commit` and the source-code artifact revision;
- each of the five `production_workloads[*].spec.code_commit` values;
- each of the three provider-plan `workflow_sha` values; and
- each embedded runner-bootstrap `workflow_sha`.

A missing occurrence, an occurrence elsewhere, an extra occurrence, or any other placeholder fails
admission. Before resolution, the loader authenticates three raw sentinel-bearing runner-bootstrap
templates and five raw sentinel-bearing workload specs. During a rehearsal at C0 commit A, it
resolves the 13 scalar values and rederives those eight consequent file hashes in memory. It does
not rewrite the candidate manifest or accept replacement hashes from the caller. The raw manifest
digest therefore continues to identify the exact candidate bytes, while the materialized provider
plans and workloads identify the current C0 source.

Three non-scientific execute jobs then test the exact registered bindings:

- online: ARM64 scientific image and the `main` index role;
- label release: ARM64 timelock-release image and the `release` index role;
- analysis: AMD64 scientific image under the registered emulation path and the `main` index role.

The candidate image closure has a distinct source identity. Its `github_sha` P may precede C0 A.
The closure records the exact build-context Git-tree digest T, two OCI index digests, and three
platform-manifest digests. A canonical bootstrap-closure digest hashes T with those five OCI
digests. Build provenance, SBOM manifests, registry timestamps, and the source commit P remain
evidence about the candidate build; they are not substituted for the executable identity. The
rehearsal admission and aggregate record P, A, T, the candidate-closure file digest, and the
bootstrap-closure digest as separate fields. Equality between P and A is neither required nor
claimed.

The production-control boundary uses the same distinction. Its raw config and blueprint record P
as `candidate_image_source_commit`; every executable commit slot remains the registered sentinel.
After A exists, `fractal-production-controls instantiate-c0-controls` resolves only those slots and
publishes one A-bound five-corpus tree. The receipt binds A, P, T, D, both raw control objects, the
raw candidate, `approval_environment: confirmatory`, and the exact output tree. The materialization
config requires `runner_identity: github-actions:environment:confirmatory`; every workload is
rechecked against that identity. The receipt file digest enters C1 only through the verified C0
apparatus-evidence object.

The promotion-facing aggregate names are exact: `candidate_image_source_commit`, `c0_commit`,
`build_context_tree_sha256`, `candidate_image_closure_file_sha256`, and
`candidate_bootstrap_closure_sha256`. Phase admissions carry the same five fields, in addition to
their phase-specific index and platform-manifest digests.

Each job requests `self-hosted`, `macOS`, `ARM64`, and one rehearsal-only nonce label derived from
the candidate plan digest, phase, C0 workflow SHA, run ID, and attempt. The host verifier reads the
real run and jobs endpoints and checks the job ID, name, status, run attempt, runner ID, runner name,
runner group, and requested labels. It rehashes the fixed Python environment, GitHub CLI, Docker
client and resolved target, Actions runner members, bootstrap receipt, and registered provider path.
Hosted materialization paths may appear in the evidence package but may not cross into an execute
job.

The container launch has `--network none`, a read-only root, dropped capabilities, no secrets, no
GitHub token, and no study-data mounts. It runs a fixed image self-check only. The resulting typed
receipt records the API response digest, host-tool receipt, exact OCI index and platform manifest,
image role, index role, platform, Docker command digest, exit status, and the assertions
`scientific_inputs_opened=false`, `provider_state_mutated=false`, and `suite_attempt_id=null`.
Rehearsal failure produces an artifact-only incident receipt; it does not publish `FAILED` to the
confirmatory ledger.

A hosted completion job verifies the receipt bytes, attests the receipt by subject path, and retains
the Sigstore bundle and checksums for 90 days. The workflow creates no branch, tag, release, Zenodo
record, state ref, or suite namespace. The normalized plan-closure digest hashes the three provider
templates after the exact C0 sentinel resolution. C1 must produce the same digest. Finalization may
replace the 13 registered sentinel occurrences with A, change the lifecycle fields to `frozen`,
`0.3.0`, and an empty blocker list, and insert the verified C0 evidence-release binding. A canonical
candidate-to-frozen comparison rejects every other mutation. The templates' semantic binding tokens
for the enclosing manifest and future C1 commit remain literal contract values, so no Git hash fixed
point is introduced.

The pre-A construction procedure, including the P registration versus A activation boundary, is
specified in [Provider-plan construction before C0](provider-plan-operator.md). The fixed-path,
post-C1 production receipt writer is specified in
[Production runner activation after C1](provider-runner-activation.md). Its A-bound production
receipt is distinct from every candidate rehearsal receipt.

The rehearsal passes only when all three jobs use newly registered ephemeral runners, every live API
field matches, every fixed host path resolves on the self-hosted machine, all three exact image
bindings launch without network or data, the job token is absent before Docker starts, and anonymous
readback verifies the retained receipt and attestation bundle. Production dispatch remains blocked
until this receipt is included in C1.

## Repository-runner baseline

A read-only API capture on 2026-07-17 returned `total_count = 0` for repository-level
self-hosted runners. That observation is a provisioning baseline, not C0 or C1 evidence. Regenerate
it immediately before creating any candidate or production runner. The inventory command retains
the raw API response and a closed typed receipt outside the Git worktree:

```bash
test "$(gh api user --jq .login)" = mhdk1602
inventory_dir="$(mktemp -d /private/tmp/fractal-runner-inventory.XXXXXX)"
github_output="$inventory_dir/github-output"
: > "$github_output"
UV_CACHE_DIR=/private/tmp/fractal-uv-cache \
  uv run --frozen --no-sync \
  python -m fractal_ann_diagnostics.provider_rehearsal inventory \
  --gh-executable "$(command -v gh)" \
  --output-dir "$inventory_dir/evidence" \
  --github-output "$github_output"
(
  cd "$inventory_dir/evidence"
  printf '%s  %s\n' \
    "$(sed -n 's/^raw_response_sha256=//p' "$github_output")" \
    repository-runners-api.raw.json | sha256sum -c -
)
```

The typed capture rejects a truncated first page, duplicate runner IDs or names, malformed labels,
and more than 100 registered runners. A repository-owned runner has no organization runner group.
Accordingly, every rehearsal bootstrap receipt and every C1 phase plan for this personal repository
must use `runner_group_id: null`. The live Jobs API read in the execute job proves that null again;
the repository runner-list endpoint is not treated as group evidence.

## Candidate rehearsal operator sequence

Dispatch the candidate image workflow first. Once its candidate-closure artifact exists, dispatch
the provider rehearsal from the same `c0-candidate/*` branch and pass only the image run ID. The
hosted plan job verifies that source run, loads the production provider-plan schema, and uploads
`provider-rehearsal-plan-<run-id>`. Download that artifact while the three execute jobs are queued:

```bash
candidate_branch='c0-candidate/<candidate>'
candidate_image_run_id='<successful candidate image run ID>'
gh workflow run confirmatory-provider-rehearsal.yml \
  --ref "$candidate_branch" \
  -f mode=candidate-rehearsal \
  -f candidate_image_run_id="$candidate_image_run_id"

rehearsal_run_id='<new provider rehearsal run ID>'
plan_dir="$(mktemp -d /private/tmp/fractal-provider-plan.XXXXXX)"
gh run download "$rehearsal_run_id" \
  -n "provider-rehearsal-plan-$rehearsal_run_id" \
  -D "$plan_dir"
sha256sum -c "$plan_dir/PACKAGE-SHA256SUMS"
```

Retain `rehearsal_run_id` after the aggregate `complete` job succeeds. Production image publication
requires that numeric run ID alongside the candidate image run ID. It independently reads the run
and artifact APIs, verifies the aggregate artifact digest and fixed member set, rechecks the
candidate-closure binding and all three phase job identities, and verifies the aggregate's GitHub
attestation. A successful candidate image build without this aggregate cannot become C0.

For each phase, copy the hosted plan bytes to the exact absolute `provider_plan_path` recorded in
its admission. The source and target SHA-256 must equal `provider_plan_file_sha256`; do not rewrite
the JSON. Configure an Actions runner from the candidate-pinned 2.335.1 archive with
`--ephemeral --disableupdate --unattended` and only the admission's custom nonce label. Leave the
listener stopped. Then write the bootstrap receipt from a fresh repository inventory:

```bash
phase='online' # then label-release, then analysis
admissions="$plan_dir/rehearsal-admissions.json"
admission_json="$(jq -c --arg phase "$phase" '.admissions[$phase]' "$admissions")"
provider_plan_path="$(jq -r '.provider_plan_path' <<<"$admission_json")"
provider_plan_sha256="$(jq -r '.provider_plan_file_sha256' <<<"$admission_json")"
hosted_plan="$plan_dir/hosted-plan-materializations/$phase/provider-plan.json"

test ! -e "$provider_plan_path"
install -d -m 0700 "$(dirname "$provider_plan_path")"
install -m 0444 "$hosted_plan" "$provider_plan_path"
test "$(sha256sum "$provider_plan_path" | awk '{print $1}')" = \
  "$provider_plan_sha256"

runner_name="fractal-rehearsal-$phase-$rehearsal_run_id"
runner_label="$(jq -r '.runner_label' <<<"$admission_json")"
runner_config="$(jq -r '.host_tools.runner_config_executable' "$provider_plan_path")"
registration_token="$(gh api --method POST \
  repos/mhdk1602/fractal-ann-diagnostics/actions/runners/registration-token \
  --jq .token)"
(
  cd "$(dirname "$runner_config")"
  ./config.sh \
    --url https://github.com/mhdk1602/fractal-ann-diagnostics \
    --token "$registration_token" \
    --name "$runner_name" \
    --labels "$runner_label" \
    --ephemeral \
    --disableupdate \
    --unattended
)
unset registration_token

bootstrap_output="$(mktemp /private/tmp/fractal-bootstrap.XXXXXX)"
host_python="$(jq -r '.host_python_path' <<<"$admission_json")"
"$host_python" -m fractal_ann_diagnostics.provider_rehearsal \
  prepare-runner-bootstrap \
  --admission-json "$admission_json" \
  --runner-name "$runner_name" \
  --github-output "$bootstrap_output"

runner_run="$(jq -r '.host_tools.runner_run_executable' "$provider_plan_path")"
(
  cd "$(dirname "$runner_run")"
  ./run.sh
)
```

The bootstrap command refuses an online or busy runner, any extra label, a non-null group, more
than one matching name, or a runner absent from the raw API bytes. Start that runner only after the
command succeeds. One physical Mac may service the three queued phase jobs sequentially, but each
phase still needs a fresh ephemeral registration and its own nonce label. Never start two MPS or
Docker workloads concurrently on that host during the rehearsal.

The phase verifier generates `phase-host-probe.json` and `docker-server-probe.json` itself under a
new evidence directory. Neither file is a workflow input. Their canonical bytes must equal the
inline candidate `host_tools.host_probe` and `host_tools.docker_server_probe` objects and their file
digests. The same closed plan now binds `execution_claim_inputs`: online has the design-seed digest,
the registered online-runtime budget, and the complete drand contract; label release and analysis must carry
JSON null. The budget is prespecified from development-only capacity planning and capped
at 72,000 seconds. It is not a measurement over sealed confirmatory inputs. The plan semantic digest
covers that distinction before the rehearsal label is derived.
