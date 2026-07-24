# Sealed online orchestrator

Status: the admission seam, frozen-manifest binding constructor, and closed five-corpus publisher
are implemented.

## Purpose

The online action runner already binds its output to a `SealedRunReceipt`. A second
receipt is needed before execution: `OnlineCustodyAdmissionReceipt` records the exact
artifact IDs revalidated at the online boundary. Calling the runner directly would
leave the ordering between these checks to convention.

`run_admitted_online_matrix(...)` makes that ordering executable. It admits no call to
`run_online_action_matrix(...)` until the receipt chain and explicit artifact bindings
agree. A failed check returns no partial matrix and never calls the retriever or its
policy decision point.

## Typed binding

`RequiredArtifactIdBindings` schema v2 carries seven facts:

| Field | Meaning |
| --- | --- |
| `verification_receipt` | Exact artifact-verification evidence already bound by the sealed run |
| `execution_artifact_id` | Manifest ID for the exact outer online-execution package tree |
| `execution_revision_sha256` | Logical plan digest encoded by the manifest artifact's `revision` |
| `runner_artifact_ids` | Frozen artifacts that constitute the executing runner |
| `source_artifact_ids` | Frozen source artifacts needed by that runner |
| `retriever_artifact_ids` | Exact dependency set used by the retriever |
| `provenance_component_artifact_ids` | Explicit component-to-artifact map used to construct audit provenance |

The object accepts canonical, bytewise-sorted ID tuples. Retriever IDs must equal the
IDs in the provenance component map. The outer package digest and logical execution
revision are separate by design: the artifact-verification receipt attests the former,
while the loaded execution object must equal the latter. The orchestrator does not
infer an artifact ID from a path, URI, filename, or naming pattern.

## Production publication

`fractal-production-controls write-required-artifact-bindings` is the sole production attachment.
It runs on the custodian-controlled full artifact tree and takes the materialization config, frozen
manifest, admitted verification receipt, artifact root, closed local map, and one absent output-root
name. It has no option for a corpus, artifact ID, component map, digest, or partial suite.

Before writing, the command revalidates the C0 factory and blueprint, matches all five disclosed
workloads and the hardware fragment to the frozen manifest, and rehashes the full local artifact map.
The fresh receipt must equal the admitted receipt byte for byte. The publisher derives the five
objects in `FIXED_CORPORA` order, builds an exact private staging tree, and exposes it with one
no-replace directory rename. It then repeats authority derivation and reloads the published tree.
Finalization applies the same five-corpus derivation and rejects missing, extra, changed, linked, or
noncanonical files. The exact command and directory layout are in
[confirmatory-execution.md](confirmatory-execution.md#4-execute-the-label-separated-online-run).

`derive_required_artifact_id_bindings(...)` accepts a frozen manifest, its exact
artifact-verification receipt, and one registered corpus ID. There is no caller
surface for substituting artifact IDs. Before it selects any role, the constructor:

1. validates the full study manifest with `require_frozen=True`;
2. recomputes the manifest digest and checks the receipt binding;
3. requires the receipt ID set to equal the full manifest artifact ID set;
4. requires every receipt row to be exact and equal to the manifest SHA-256; and
5. rejects missing, duplicate, suite/corpus scope mismatches, and any selected
   sealed-label or encryption-custody role.

The role table is closed:

| Binding class | Manifest roles |
| --- | --- |
| Audit components | `application` -> `source-code`; `controller` -> `frozen-controller`; `corpus` -> corpus `corpus-normalizer`; `embedding` -> `primary-embedding`; `index` -> `strict-authorized-hnsw`; `policy` -> `opa-pdp` |
| Execution | corpus `online-execution` |
| Runner | `source-code`, `opa-runtime-binary`, `strict-authorized-hnsw`, `exact-authorized-oracle`, `frozen-controller`, `opa-pdp` |
| Source | corpus `policy-workload`, `embedding-store`, `authorized-index-store`, `trial-runtime-package`; suite `online-staging-package`, `query-partition-audit` |

The execution revision must have the canonical `sha256:<digest>` form. The constructor
stores the digest part in `execution_revision_sha256`; it does not reinterpret the
outer artifact SHA-256 as the logical plan revision.

## Admission sequence

The preflight applies these checks in memory before it transfers control:

1. The admission receipt, sealed-run receipt, and artifact-verification receipt bind
   the same manifest digest.
2. The SHA-256 of the supplied sealed-run receipt equals the digest recorded by the
   admission receipt.
3. The runner identity is identical in both receipts.
4. The admission receipt, sealed-run receipt, provenance registry, and supplied
   artifact-verification receipt expose one identical verification-receipt digest.
5. Every admitted ID exists in that verification receipt.
6. The full caller-declared dependency closure is contained in the admitted ID set,
   and each required row was verified with `exact=True`.
7. The execution object's `artifact_sha256` equals the manifest-derived logical
   `execution_revision_sha256`. The receipt row for `execution_artifact_id` separately
   remains an exact outer package-tree attestation.
8. Every provenance component name is covered exactly once. Its registry revision
   must equal the verified digest of the explicitly bound artifact ID.

Only after step 8 does the wrapper call the existing runner. The valid path invokes it
once and passes every original argument unchanged.

```text
frozen manifest
      |
artifact verification receipt ---------+
      |                                 |
sealed-run receipt                      |
      |                                 |
online admission receipt                |
      |                                 |
      +--> typed artifact-ID bindings <-+
                       |
                 fail-closed preflight
                       |
                 online action matrix
```

## Deliberate exclusions

The module has no API or import path for outcome files, decryption, confirmatory
scoring, or network clients. It can produce only the existing in-memory online runner
output. Later analysis remains a separate process after the online artifacts have
been sealed and externally anchored.

This seam proves sequencing and digest agreement. It does not prove that two
administrators are independent or that an operator never retained another copy of a
file. Those claims still depend on the registered protocol and separated authority.

## Audit provenance boundary

The production provenance registry consumes two receipts for different purposes.
The execution leaf receipt admits the corpus shards, stores, index, and fixed-width
content-digest sidecar named by the logical plan. The manifest artifact-verification
receipt derives the six `AuditRecord` component revisions listed above. Low-level leaf
names cannot enter the audit component map.

The registry exposes both receipt digests explicitly:

| Registry field | Evidence |
| --- | --- |
| `verification_receipt_sha256` | Frozen-manifest artifact-verification receipt used by custody and audit admission |
| `execution_verification_receipt_sha256` | Execution leaf receipt used for plan and sidecar integrity |

This prevents a valid leaf receipt from being mistaken for the C1 manifest receipt,
and prevents the seven execution pins from masquerading as the six registered audit
components.

## Acceptance evidence

The focused orchestrator suite has 27 tests. It covers a valid call, all admission
mismatch classes, schema-v2 round trips, full manifest/receipt closure, hostile role
selection, scope errors, and the required distinction between the outer execution
tree pin and logical revision. Each admission rejection spies on
`run_online_action_matrix(...)` and records zero invocations.

The scalable-execution suite has 31 tests. Its provenance cases require the exact six
sorted audit components and reject missing, extra, duplicate-ID, unverified, and
non-exact component rows. A source-structure test rejects outcome processing,
decryption, and network-client imports and confirms that the wrapper preserves every
parameter of the existing runner after its two admission inputs.
