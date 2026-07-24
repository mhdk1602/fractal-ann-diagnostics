# Sealed container launcher

The sealed workload has one host-side execution path: `fractal-sealed-container-launcher`.
It creates Docker objects through argument arrays, never a shell, and consumes the corpus attempt
before `docker create`. A failed preflight is repeatable because it does not open workload inputs.
A failed sealed invocation is not repeatable.

This launcher is an apparatus component, not an ergonomic wrapper around `docker run`. Its contracts
freeze the image, platform, identity, process vector, environment, CPU affinity, memory limit,
hostname, mount table, output subpath, tmpfs, and copy-out destination. Docker defaults cannot supply
any of those facts implicitly.

## State sequence

```text
typed provisional controls
        |
        v
preflight-launch-contract.json
        |
        | root-only volume initializer
        v
fresh named volume / private 0700 subpath / UID:GID 65532:65532
        |
        | label-free preflight container
        v
runtime-preflight-receipt.json
        |
        | admitted observation substitution only
        v
runtime-plan-transition-receipt.json + final template
        |
        | C1 registration + verified production-control finalization
        v
instantiated production closure + closure-binding receipt
        |
        | registered manifest and closure substitution
        v
runtime-attestation-plan.json + plan-instantiation-receipt.json
        |
        | external launcher attempt marker (O_EXCL)
        v
persist argv -> one docker create -> persist argv -> one start --attach --interactive -> no retry
        |
        v
retained container, inspect, logs, named volume, UID-65532 copy reader, host hashes
```

The provisional plan uses sentinels only for these host observations:

```text
architecture
cpu_model
kernel_release
logical_cpu_count
memory_limit_bytes
mount_namespace_sha256
operating_system_id
operating_system_version_id
python_version
```

`materialize-transition` reconstructs both plan forms and rejects any other semantic change, even
when an attacker recomputes every enclosing digest. The transition receipt binds the provisional
contract and tree, the preflight receipt's semantic and file digests, the final template and tree,
and the exact field allowlist.

All five newline-terminated WorkloadSpec files already exist at this point. Their complete objects
and file digests enter C1 in fixed corpus order. The post-registration path cannot admit a
replacement specification, select another corpus package, or change a scientific mount.

C1 also records the materialization-config file digest and both the semantic and file digests of
the blueprint receipt. Before it admits runtime evidence, the finalizer compares those public pins
with current bytes, then reproduces every provisional plan and `LauncherGeometry` field from the C0
factory, config, WorkloadSpec, hardware disclosure, and fixed launcher constants. The preserved
preflight contract must equal that full typed derivation. A replacement image, path, root, mount,
environment value, limit, volume, tmpfs setting, plan byte, or contract field fails even if the
replacement JSON remains well formed.

The manifest digest enters later through the verified production-control finalization request and
receipt. `instantiate-plan` reproduces that authority, replaces the single
`{manifest_sha256}` token and provisional closure-mount digest, removes the template, writes the
registered plan, and emits an instantiation receipt. It then derives the sealed launch contract.
This keeps the plan template C1-pinnable without creating a plan/manifest digest cycle.

## Output volume

The initializer is the sole root container in the protocol. It creates one new subdirectory in a
new named volume, changes ownership to `65532:65532`, and sets mode `0700`. The host verifies the
volume name and closed label map, then checks the initializer's image, identity, labels,
capabilities, mount, tmpfs, terminal state, exit code, and OOM flag from retained Docker inspect
bytes. The preflight and sealed containers mount the same exact volume and subpath. They run as
`65532:65532` with all capabilities dropped, no network, a read-only root filesystem, and one
registered tmpfs.

The launcher never removes the volume or a retained container. Copy-out uses a separate reader
running as `65532:65532`, with the output subpath mounted read-only. The reader computes its source
inventory before the host copy. `docker cp` names that reader ID, never the writable sealed
container ID. The host creates the contract-bound `<suite namespace>/online/<corpus-id>` directory,
pins its device and inode, and requires a real launcher-owned mode-`0700` parent and root. Copied
directories must be mode `0700`; files must be singly linked, launcher-owned, and mode `0600`.
File count, byte count, per-file hashes, and the tree digest must match the source inventory.

## Secret transport

The HMAC secret enters only through file descriptor 0 of:

```text
docker start --attach --interactive <sealed-container-id>
```

No Docker environment row, mount, command argument, marker, contract, inspect file, log file, or
receipt contains the secret. Evidence records only its byte count and SHA-256. After the container
returns, the launcher scans its retained evidence for the literal secret bytes and rejects a leak.

## Operator commands

The operator begins with a typed `PreflightLaunchContract`. The control directory and plan template
must already match the contract's provisional digests.

```bash
fractal-sealed-container-launcher initialize-volume \
  --contract /controlled/launcher/scifact/preflight-launch-contract.json \
  --audit-root /controlled/launcher/scifact/volume-initialization-evidence

fractal-sealed-container-launcher preflight \
  --contract /controlled/launcher/scifact/preflight-launch-contract.json \
  --volume-receipt /controlled/launcher/scifact/volume-initialization-evidence/volume-initialization-receipt.json \
  --audit-root /controlled/runtime-evidence/scifact

fractal-sealed-container-launcher materialize-transition \
  --contract /controlled/launcher/scifact/preflight-launch-contract.json \
  --preflight-receipt /controlled/runtime-evidence/scifact/runtime-preflight-receipt.json \
  --transition-receipt /controlled/runtime-evidence/scifact/runtime-plan-transition-receipt.json

fractal-sealed-container-launcher instantiate-plan \
  --preflight-contract /controlled/launcher/scifact/preflight-launch-contract.json \
  --preflight-receipt /controlled/runtime-evidence/scifact/runtime-preflight-receipt.json \
  --transition-receipt /controlled/runtime-evidence/scifact/runtime-plan-transition-receipt.json \
  --finalization-request /controlled/suite/finalization-request.json \
  --finalization-receipt /controlled/suite/production-control-finalization-receipt.json \
  --instantiation-receipt /controlled/launcher/scifact/plan-instantiation-receipt.json \
  --sealed-contract /controlled/launcher/scifact/sealed-launch-contract.json
```

`initialize-volume` and `preflight` choose their receipt names from the protocol and create them
inside their respective private audit roots. The caller cannot redirect those receipts to another
path. Each corpus preflight and transition pair sits directly under
`/controlled/runtime-evidence/<corpus-id>` so the production-control finalizer can admit one exact
five-corpus evidence root.

There is no operator-facing standalone `launch` command. A serialized runtime-claim receipt cannot
reconstruct the in-memory `VerifiedRunClaimCapability`, so the launcher rejects direct launch
invocations even when their files are otherwise valid. The only admitted production entry is the
fixed C0 workflow:

```bash
: "${SUITE_ATTEMPT_ID:?set the frozen 64-character suite attempt identifier}"
gh workflow run confirmatory-online-execution.yml \
  --repo mhdk1602/fractal-ann-diagnostics \
  --ref confirmatory-apparatus-c0 \
  --field suite_attempt_id="$SUITE_ATTEMPT_ID"
```

The hosted claim job admits the C1 plan and produces the one runtime claim. Its protected
self-hosted execute job calls `execution_claim verify-prerequisites --activate-and-execute`, which
retains the typed authority in memory through `provider_phase_runtime` and into
`launch_sealed_once`. The claim HMAC bytes never enter a process argument, environment variable,
temporary file, command substitution, or terminal paste buffer.

Label release uses the same non-serializable boundary. Retained phase-claim bytes and the public
`provider_phase_runtime` parser cannot reach decryption. The activation factory must supply a
fresh `VerifiedPhaseClaimCapability` and a verified pre-decryption admission marker separately for
each corpus immediately before the timelock payload is opened.

## Retained evidence

Before each mutating Docker operation, the launcher writes a canonical
`<operation>-docker-argv.json` record containing the exact argument array. Only then may it call
Docker. After the call it retains bounded stdout and stderr and writes a result record containing
the Docker return code, stream sizes, stream hashes, and argument-record digest. This applies to
volume create, every container create and start, and copy-out. No shell string is constructed.

The sealed container and output reader are inspected after exit. Their inspect records must show
terminal `exited` state, the observed exit code, the OOM flag, fixed labels, no restart policy, no
extra environment row, mount, or capability, and the contract's remaining geometry. A nonzero
sealed start still writes attached streams, inspect, logs, command result, and a closed `failed`
launch receipt. It then returns an error. Any other failure after the
attempt marker, including inspect, reader, inventory, destination, copy, or receipt verification,
writes `sealed-launch-failure-receipt.json`. That terminal record binds the failure stage, error
record, container identities reached so far, and the exact evidence inventory. The private error
record contains the exception class and a bounded, secret-redacted message. Independent
verification rederives its hash and size, the failure stage, every reached Docker argument array,
and every inspect record available at that stage. The attempt marker makes every such result
ineligible for retry.

The sealed receipt binds:

- the preflight, transition, instantiation, volume, and sealed contract digests;
- the marker's digest and its position before container creation;
- the sealed container ID, Docker return code, terminal exit code, OOM flag, inspect hash, and log
  hashes;
- the secret byte count and hash;
- the source inventory, host copy receipt, retained copy-reader container, and copied root;
- every retained evidence file through a closed canonical inventory digest.

The sorted inventory covers every regular file in the sealed audit directory except
`sealed-launch-receipt.json`; excluding the receipt avoids a self-digest cycle. Independent
verification requires exact directory membership equal to `inventory + receipt`, rehashes each
file, reconstructs every command/result pair, rechecks inspect geometry when given the sealed
contract, and rehashes the preserved host output:

```bash
fractal-sealed-container-launcher verify-launch-evidence \
  --receipt /controlled/launcher/scifact/sealed-evidence/sealed-launch-receipt.json \
  --audit-root /controlled/launcher/scifact/sealed-evidence \
  --sealed-contract /controlled/launcher/scifact/sealed-launch-contract.json \
  --expected-receipt-sha256 "$SEALED_LAUNCH_RECEIPT_SHA256"
```

For a host-stage failure, substitute the terminal failure verifier:

```bash
fractal-sealed-container-launcher verify-launch-failure-evidence \
  --receipt /controlled/launcher/scifact/sealed-evidence/sealed-launch-failure-receipt.json \
  --audit-root /controlled/launcher/scifact/sealed-evidence \
  --sealed-contract /controlled/launcher/scifact/sealed-launch-contract.json \
  --expected-receipt-sha256 "$SEALED_LAUNCH_FAILURE_RECEIPT_SHA256"
```

`ONLINE_COMPLETE` requires one provider-verified successful launch receipt per corpus. The
suite closure binds its semantic digest, file digest, evidence-inventory digest, sealed-contract
digest, copy-root URI, and output-tree digest before transferring the preserved staging bytes.

There is deliberately no cleanup or retry command. The provider CAS and registered retention policy
decide when retained Docker state may be released. A repository administrator can still intervene
outside that path; the apparatus treats such a lineage as inadmissible rather than physically
impossible.
