# Runtime attestation for the sealed process

The frozen protocol needs evidence about the process that actually opens the online workload. An
OCI reference in C1 proves which registry object was selected. It does not, by itself, show which
object the launcher started, whether the input mounts were writable, or whether the process had a
network path. `runtime_attestation.py` supplies that missing receipt.

No production receipt is stored in this repository. A receipt is admissible only when it is created
inside the registered Linux container immediately before the first workload operation.

## Two evidence sources

The attestation joins evidence from two administrative positions.

The launcher writes `fractal-launcher-runtime-identity-v1` after resolving the image through the
container runtime or orchestrator. That closed JSON file contains the digest-qualified OCI reference
and the full C0 Git commit. Its exact bytes are pinned in the frozen plan and mounted read-only with
the runtime controls. The launcher must retain its registry resolution record, image-inspection
output, and container creation record outside the container.

The process records what it can inspect from its own Linux namespace: Debian ID and version, kernel,
architecture, CPU model and affinity count, finite cgroup memory limit, exact mount points, mount
read-only flags, a reproducible security profile for every mount, the SHA-256 of the exact raw
`/proc/self/mountinfo` bytes, interfaces, routes,
network-namespace inode, argv, environment, Python executable, and Python version. It hashes the
mounted artifact trees, OPA, Python, `uv.lock`, and the launcher identity twice. A file change
between the two passes aborts the attempt.

This division is deliberate. The in-container receipt is not TPM, TEE, or cloud-verifier evidence.
It cannot independently discover the registry digest of its own root filesystem. The external
launcher record supplies that fact; the receipt binds it to the other runtime observations.

## Frozen plan

`RuntimeAttestationPlan` is the closed canonical `fractal-runtime-attestation-plan-v2` record. C1
pins one separately hashed template for each fixed corpus. The five plans share the registered
runner identity, image, commit, and hardware contract, but each plan binds its own corpus command,
config digest, workload identity, mounts, output namespace, and invocation marker.

The pre-C0 production-control blueprint is a different object. Its five plan templates and launcher
identity contain the exact registered C0 sentinel, so they do not assert A before A exists. Runtime
preflight and execution reject that sentinel. After A exists, the one-shot control-instantiation
operator derives the A-bound templates and contracts, then publishes them with a closed receipt.
C0 apparatus evidence pins that receipt. C1 and finalization admit the A-bound tree only when it
matches both the raw sentinel templates after exact resolution and the frozen workload bytes.

The admitted plan pins:

- manifest digest, runner identity, C0 commit, and digest-qualified OCI reference;
- operating-system ID and version, kernel release, architecture, CPU model, affinity count, and
  finite memory limit in bytes;
- every artifact mount by canonical absolute root, role, file-or-directory kind, tree digest, and
  `read_only=true`;
- the SHA-256 of a reproducible mount security profile, including every mount point, filesystem,
  effective option, and every stable backing root and source. Container-assigned mount IDs,
  device numbers, overlay paths, and Docker-managed host paths are excluded from the frozen digest;
- the complete argv plus its domain-separated digest;
- the exact environment-name allowlist and a domain-separated digest over every name and value;
- absolute paths and SHA-256 digests for OPA, Python, `uv.lock`, and the launcher identity;
- the admitted workload ID and digest; and
- an invocation-marker path on the dedicated writable output volume.

OPA, `uv.lock`, and the launcher identity must reside inside a declared read-only artifact
mount. Mount roots cannot overlap. The marker cannot reside below any immutable input root. Paths
with `..`, redundant separators, trailing separators, URI syntax, symlink components, or hard-linked
files are refused.

The environment digest does not disclose values in the receipt. Verification still requires the
process environment to contain exactly the registered names. An undeclared credential, proxy, cloud
metadata setting, or diagnostic flag aborts execution.

The time-lock executable is intentionally absent from both platforms of the main scientific image.
Online execution uses its ARM64 manifest and analysis uses its AMD64 manifest. A separate pinned
ARM64 release image contains `tle`; only the claimed label-release phase may select that image and
its controlled drand network path.

## One provider-claimed invocation

For each fixed corpus, `attest_runtime_once` creates `fractal-runtime-invocation-marker-v1` with
`O_EXCL` before reading the
launcher identity, probing Linux, or hashing an artifact. The marker binds the plan digest and the
workload identity. If any later check fails, the marker remains. Repeating the invocation under the
same plan then requires an intervention in the output volume, which falls outside the
provider-claimed lineage and is inadmissible.

This order matters. A failed environment check leaves evidence in the claimed output tree. A wrong
image, writable mount, connected interface, unexpected argument, changed binary, or malformed
receipt closes the provider phase as failed; it cannot be rescued inside the registered lineage.

After every check passes, the module creates `fractal-runtime-attestation-receipt-v3`. The receipt
contains the frozen identities, observed hardware and namespace facts, mount records, process
digests, binary pins, workload identity, invocation-marker digest, and SHA-256 of the exact raw
mount table observed by that process. The loader rejects duplicate
keys, unknown keys, missing keys, non-finite JSON values, alternate whitespace, alternate Unicode,
and a missing terminal newline. `verify_runtime_attestation_receipt` then compares every frozen field
with the exact plan.

The persisted receipt is not the final admission gate. Immediately before a production corpus run
opens any workload source, `run_sealed_online_once` reloads the canonical plan and receipt from
absolute paths, checks their frozen semantic digests, and calls
`verify_live_runtime_attestation` with a freshly constructed `LinuxRuntimeProbe`. This second
observation must reproduce the registered mount security-profile digest and the receipt's raw
mount-table digest, exact environment, argv, CPU and
memory facts, read-only mount table, network-namespace inode, interfaces, and route-table digest.
The production entrypoint has no probe, observation, environment, or namespace parameter.

The gate also rehashes the durable invocation marker and requires every artifact, index, policy,
embedding, query-package, staging, schedule, partition-audit, and pseudonym-key path to fall inside
exactly one declared read-only mount. It then writes the corpus attempt receipt. That receipt names
both the runtime-plan digest and runtime-receipt digest. A failed live check leaves the earlier
runtime marker in place. Within the admitted output lineage, that failed check cannot be erased and
recast as a first invocation even though the corpus attempt receipt has not yet been written.

The operator cannot supply those source paths or receipt pins as command-line facts. The only
corpus command has one option: `--config`, followed by the fixed `corpus-run-config.json` path. The
plan repeats that exact six-element argv and stores the config digest as `workload_sha256`; the
digest is not a command-line argument. After verifying `RUN_CLAIMED`, the launcher writes the exact
canonical bytes of the freshly verified `RuntimeClaimReceipt` to the container's standard input.
The command rejects missing, oversized, or noncanonical receipt bytes before loading the config. A
closed loader then derives the source paths and all subordinate pins from canonical controls. The
external suite state pins the plan; the plan pins the config. A reverse config-to-plan hash is
forbidden because it would create a digest cycle. After validating the standard-input receipt, the
command requires an empty output directory, loads the config and canonical plan, checks the plan's
argv, workload, and marker bindings, and performs `attest_runtime_once` in that same process. It
reloads and verifies the resulting receipt before admitting any subordinate control. A failed
attestation leaves the invocation marker behind and cannot be retried through the same command
package.

OPA is not running during either attestation pass. The plan pins its binary bytes and must name
`/usr/local/bin/opa`. Its exact file mount must use role `opa-runtime-binary`; the frozen manifest
pins the same file at `artifacts/runtime/opa`. The C1 materializer verifies the selected platform's
closed C0 extraction receipt, checksum rows, and GitHub attestation bundle for OPA, Python, and
`uv.lock` before copying the retained OPA bytes to that path at mode `0555`; it then requires all
five plan templates to agree on the digest, OCI index, platform, architecture, and C0 commit. The
same receipt is the source of the plans' Python and lock digests. Only after the corpus attempt receipt
exists may the same Python process
spawn that binary as one child inside the already attested `network_mode=none` namespace. The child
binds IPv4 loopback only, receives fixed arguments and a minimal fixed environment, and is stopped
before result persistence. An external OPA process would occupy the fixed port and cause admission
to fail rather than becoming an alternate policy service.

## Linux launch requirements

The registered launcher should use the digest-qualified image and, at minimum, the following
controls:

```text
--network none
--read-only
--memory <registered-byte-limit>
--cpuset-cpus <registered-affinity-set>
--hostname <registered-hostname>
--env HOSTNAME=<registered-hostname>
--mount type=bind,src=<controls>,dst=<registered-control-root>,readonly
--mount type=bind,src=<inputs>,dst=<registered-input-root>,readonly
--mount type=bind,src=<C1-pinned-opa>,dst=/usr/local/bin/opa,readonly
--mount type=bind,src=<new-empty-output>,dst=<registered-output-root>
```

The hostname is a frozen process input, not a launcher default. `--hostname` and the explicit
`HOSTNAME` value must agree with the corpus plan. The plan must enumerate the complete environment
materialized by the image and launcher, with every value fixed before C1. Do not pass `--env-file`,
inherit a host variable with `--env NAME`, or add proxy, credential, locale, tracing, or cloud-SDK
variables. Any extra name or one-byte value change consumes the attempt.

Each declared artifact root must be an exact mount point. A read-only ancestor does not substitute
for the registered root. An undeclared mount anywhere in the namespace changes the frozen namespace
digest and consumes the attempt. This matters for label custody: mounting plaintext labels at an
unregistered path cannot evade the input-root checks. The only accepted network state has one
interface named `lo` and zero
non-loopback IPv4 or IPv6 routes. An unlimited cgroup memory value is rejected. CPU count comes from
the process affinity mask, not the host's nominal processor count.

OPA and Python must be singly linked regular files, not symlinks, executable by their owner,
and unwritable by group or other identities. The lockfile and launcher identity follow the same
no-link and no-group-write rules without the executable requirement.

## Capture sequence

1. Outside the container, read A from the frozen apparatus and D from the admitted image closure.
   Load the C0-pinned control-instantiation receipt. Do not accept P or an unresolved sentinel as
   the executable commit. Preserve the independent launcher evidence.
2. Load the five A-bound plans from the exact instantiated tree. Each launcher-identity digest must match the
   file that will be mounted.
   Derive the mount security-profile digest from a label-free preflight container created from the
   same immutable launcher specification. Kernel-assigned mount IDs, device numbers, overlay backing
   paths, and Docker-managed host paths are excluded; every mount point, filesystem, and effective
   security option remains bound. The exact raw mount-table digest is created only by the admitted
   process and is reobserved immediately before workload access.
3. For each corpus, start a new container with the fixed six-element argv ending in
   `run-sealed-corpus --config <path>`, its own empty output directory, no network, a read-only root
   filesystem, the registered hostname and exact environment, finite cgroup memory, fixed CPU
   affinity, and exact read-only artifact mounts. Supply the freshly verified canonical
   `RuntimeClaimReceipt` bytes through standard input.
4. After parsing the standard-input claim, the command loads that corpus's externally hash-pinned
   config and canonical plan. It checks the config-bound argv, workload digest, and
   invocation-marker path, then calls `attest_runtime_once` with a freshly constructed
   `LinuxRuntimeProbe` in the same Python process.
   The function writes the marker before any probe, subordinate control, or corpus source access and
   publishes the receipt to the formerly empty output directory.
5. The command reloads and verifies the receipt, reconstructs the typed custody, artifact,
   execution, and trial-runtime controls, then enters the internal runner. That boundary reloads the
   plan and receipt, reobserves the live Linux process, verifies the invocation marker, checks the
   complete source-path mount closure, and binds both attestation digests into its exclusive attempt
   before any scientific source opens.
6. Before accepting the suite, independently load all five plan/receipt pairs, call
   `verify_runtime_attestation_receipt` on each pair, and compare the retained launcher records with
   the external registry evidence. `complete_online_suite` requires each corpus attempt to bind its
   own C1-pinned plan, receipt, durable invocation marker, and typed command-consumption marker.
   The command marker binds the exact config digest to the plan workload digest. Reusing any plan,
   receipt, marker, or file identity across corpora rejects the transition.

An injected `RuntimeProbe` exists for deterministic conformance tests. It does not authorize a
production substitution. Registered execution uses `LinuxRuntimeProbe`; any other probe is test or
integration code and must not support a confirmatory claim.

## Failure semantics

| Observation | Result |
|---|---|
| launcher image or commit differs | abort; marker remains |
| input root is absent, aliased, nested, or writable | abort; marker remains |
| mount namespace contains an added, removed, or altered mount | abort; marker remains |
| artifact, binary, lock, or identity digest differs | abort; marker remains |
| interface other than `lo` or a non-loopback route exists | abort; marker remains |
| cgroup memory is unlimited or CPU inventory is heterogeneous | abort; marker remains |
| argv or environment differs by one byte | abort; marker remains |
| live namespace or environment changed after receipt creation | abort before workload-source I/O; marker remains |
| workload source is outside the registered read-only mounts | abort before the corpus attempt; marker remains |
| corpus attempt binds another runtime plan or receipt | reject suite completion |
| receipt or plan has an extra field or alternate JSON encoding | reject the record |
| invocation marker already exists | refuse a second invocation |

These checks establish a reproducible process identity and a falsifiable custody record. They do not
prove that the host kernel, container runtime, registry, or launcher administrator was honest. That
stronger claim would require an independent verifier backed by signed runtime or hardware evidence.
