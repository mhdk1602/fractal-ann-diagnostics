# Candidate study-manifest source composer

The tracked `research/study-manifest.json` is a structural draft. The artifact inventory,
development freeze, joint-power report, production controls, and candidate image closure each
hold a different part of the evidence needed to close it. Copying values from those files into the
manifest by hand would create an unrecorded authority outside the typed producers.

`operators/candidate_study_manifest_composer.py` performs that join. Its `compose` command accepts
one canonical request file, the SHA-256 of that request, and an absent output directory. It has no
flag for a manifest field, producer digest, C0 commit, provider plan, custodian, output hash, or
scientific constant.

This operator is deliberately outside `src/fractal_ann_diagnostics` and outside the confirmatory
image context. It prepares the pre-provider source. It does not publish C0, construct provider
plans, freeze C1, register the study, open a suite attempt, release labels, or run analysis.

## Exact request

The request schema is `fractal-candidate-study-manifest-source-request-v1`:

```json
{
  "inputs": {
    "artifact_inventory": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "artifact_inventory_receipt": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "candidate_image_closure": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "deployment_fragment": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "development_freeze_receipt": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "geometry_profiles": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "joint_power_config": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "joint_power_report": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "post_embedding_receipt": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "production_control_blueprint_receipt": {
      "path": "/absolute/path",
      "sha256": "<64 lowercase hex>"
    },
    "production_control_config": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "production_control_config_write_receipt": {
      "path": "/absolute/path",
      "sha256": "<64 lowercase hex>"
    },
    "production_hardware_fragment": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "production_workloads_fragment": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "static_comparator": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"},
    "template": {"path": "/absolute/path", "sha256": "<64 lowercase hex>"}
  },
  "schema_version": "fractal-candidate-study-manifest-source-request-v1"
}
```

The actual file must be canonical JSON with one terminal LF. The role set is exact. Missing,
additional, or repeated roles fail admission. Every path is normalized and absolute. Every
producer file except the tracked template must be owned by the current operator, singly linked,
and mode `0400` or `0600`; the template may also be `0444` or `0644`. The request itself is a
private `0400` or `0600` file.

The inventory receipt is a separate request role because both members of that producer package
are authorities. The composer reads each member once through the exact request binding, then
constructs and validates the typed inventory from those captured bytes. It never asks the
inventory loader to reopen the package path, and it opens no undeclared sidecar.

## Non-scientific deployment fragment

Three required values have no scientific producer: the custodian identity, immutable results
location, and receipt path template. They enter through
`fractal-candidate-deployment-fragment-v1`:

```json
{
  "custodian": "custodian@example.org",
  "receipt_uri_template": "file:///controlled/receipts/{manifest_sha256}.json",
  "results_store": "file:///controlled/results/study-candidate",
  "schema_version": "fractal-candidate-deployment-fragment-v1"
}
```

The schema is closed. The custodian must be canonical non-placeholder text. The results store must
be one canonical absolute local `file:` URI, matching the registered built-in one-shot analysis
runner. The receipt template must be an absolute `file:` URI whose final member is exactly
`{manifest_sha256}.json`. Scientific values and digests cannot enter through this fragment. A
remote store requires a separately pinned create-if-absent adapter and is outside protocol v0.3.

## Producer cross-bindings

The composer does more than copy fields:

- it binds the 79-row artifact inventory and its receipt to the canonical semantic digest of the
  tracked template;
- it requires the development-freeze receipt, geometry profiles, static comparator, and
  joint-power config to sit under the post-embedding producer root and match that producer's
  hashes;
- it compares the joint-power dependence hash with the frozen calibration-outcomes pin without
  opening the outcome file;
- it compares the two fixed scenario IDs and panel hashes across the freeze receipt, joint-power
  config, and joint-power report without opening either panel;
- it requires the geometry thresholds to agree between the geometry-profile and joint-power
  producers;
- it binds the five workload objects and hardware fragment to the production-control config and
  blueprint;
- it binds candidate image commit P, scientific candidate image T, and promoted production image
  digest D across the image closure and production controls.

The tracked analysis constants must already equal the typed joint-power design. The composer will
not replace a conflicting registered constant. It uses a fixed all-zero commit only in an
in-memory validation probe, recalculates the five probe workload hashes, and discards the probe.
That value never enters the source or its receipt.

## Outcome-blind read set

The composer opens aggregate design metadata and producer receipts. It does not dereference any
artifact URI. In particular, it does not open:

- development fit or calibration outcome rows;
- either joint-power panel;
- sealed inputs, labels, label ciphertexts, or timelock plaintext;
- online action panels, predictions, confirmatory results, or analysis output.

The composition receipt records `outcome_payloads_opened: false`. That statement concerns this
operator's read set. It does not assert human blinding, independent organizational custody, or
label inaccessibility elsewhere.

## Closed CLI

First record the canonical request and its digest through the acquisition controller. Then compose
into an absent directory:

```bash
python3 operators/candidate_study_manifest_composer.py compose \
  --request "$controlled/candidate-source-request.json" \
  --request-sha256 "$CANDIDATE_SOURCE_REQUEST_SHA256" \
  --output-directory "$controlled/candidate-source-package"
```

The directory is published once and contains exactly:

- `candidate-study-manifest.source.json`
- `candidate-study-manifest.source-receipt.json`

The directory is mode `0700`; both files are mode `0600`, regardless of the caller's umask.
Composition and verification require those exact modes, not merely a lack of group write access.
Before derivation, the writer opens the canonical request, all sixteen producer files, and each
distinct producer parent with `O_NOFOLLOW`. It obtains nonblocking exclusive POSIX advisory locks
on those open descriptions, captures every file's exact bytes once, and retains the descriptors,
bytes, and locks through publication. Typed parsing uses the captured bytes; typed loaders do not
reopen producer paths. A second derivation from the same capture must reproduce the source and
receipt. The receipt records the capture-set digest and the exact custody contract.

This is a cooperative lease, not an operating-system write prohibition. Every producer, custodian,
and repair process permitted to mutate an admitted file or its parent namespace must obtain the
same exclusive lock first. The condition is testable: while composition or verification holds the
lease, a nonblocking exclusive-lock attempt through another open description must fail with
`EWOULDBLOCK`. A same-UID process that ignores advisory locks can still call `chmod`, write through
another descriptor, or rename a path after the operator's last observation. The receipt says
`noncooperating_same_uid_mutation_excluded: true`; it does not claim that such a process was
technically prevented. Where that exclusion is unacceptable, place the producer capture on a
read-only immutable snapshot or run the composer under a separate identity that cannot modify the
producer tree. Do not substitute another trailing `stat` call for that custody boundary.

The output parent is held under the same cooperative exclusive lease. The writer reopens every
component with `O_NOFOLLOW`, requires its device, inode, mode, owner, and group to match the initial
values, checks that the random staging name still binds the retained directory descriptor, and
uses a no-replace directory rename. Existing destinations are never replaced.

Rename success starts a post-publication verification phase. The composer fsyncs the parent,
reopens the complete parent chain, rechecks the captured parent-control identity, checks the
destination name against the retained staging descriptor, and compares both files with the
captured package. Both member descriptors are opened before either file is interpreted. They stay
open until both reads finish, then both descriptor signatures, both named identities, and the
two-member directory listing are checked as one closing scan. A receipt mutation after its read
and before the source read therefore fails the same compose and verify path.

If a check fails while the destination still binds the captured staging inode, the composer moves
it to a fresh rollback name without replacement. Cleanup first moves each candidate name to a
fresh quarantine name and identifies the quarantined inode. A foreign substitution is restored;
only the retained staging inode can have its known members unlinked and directory removed. The
parent is then fsynced and the destination proved absent. This is a clean composition failure only
if the parent-control identity also remains exact.

If provenance is no longer exact, the parent path was relocated, a foreign destination occupies
the name, rollback durability cannot be proved, or an early staging allocation cannot be proved
removed, the command exits `2` and reports `candidate source publication indeterminate`. Treat
that state as a custody incident. Quarantine the parent directory and inspect both the named
destination and any staging or rollback entries; do not retry into the same path. Ordinary
admission failures and proven-clean rollbacks exit `1`.

Verification captures the request and all sixteen producer files under the same leases, rederives
the source from those bytes in memory, and compares both published files. It retains the package
directory descriptor during that work and applies the same joint two-file scan:

```bash
python3 operators/candidate_study_manifest_composer.py verify \
  --directory "$controlled/candidate-source-package"
```

Producer inputs therefore remain part of the verification evidence. Moving or deleting one makes
verification fail rather than silently trusting the receipt. Moving or replacing the package
itself during authority reproduction also makes verification fail.

## Output boundary

The source has exactly seven
`containing-confirmatory-apparatus-c0-commit` sentinels:

- `sealed_execution.code_commit`;
- the source-code artifact revision;
- one `code_commit` in each of the five production workload specifications.

Exactly two unresolved `tbd` paths remain:

- `sealed_execution.c0_evidence_release`;
- `sealed_execution.provider_phase_plans`.

The sole freeze blocker is `the immutable C0 evidence release remains unresolved`. The provider
plan operator consumes this source and adds the six provider-plan workflow sentinels. The resulting
thirteen-sentinel candidate then proceeds through `publish-closed` and the C1 transition. Do not
send the two-file source package directly to the C1 transition.

## Failure rules

A digest mismatch, alternate producer root, unknown field, duplicate JSON key, non-finite number,
symlinked path component, hard link, FIFO, unsafe mode, changed inode, scenario substitution, or
occupied output path is a terminal composition failure. Correct the upstream producer package or
request and choose a new absent output directory. Do not edit the source JSON or add a field
override to the operator.

A publication-indeterminate result is different. The operator has refused to claim either success
or clean absence because a post-rename identity or durability fact became unprovable. Preserve the
directory as evidence and resolve the custody discrepancy before any later candidate-manifest,
C0, provider-plan, or C1 step.

The same rule applies if staging-directory creation succeeds but its first identity capture fails.
Without a captured inode, the operator will not delete an entry merely because it occupies the
random staging name: a concurrent rename could have installed a foreign replacement there. It
leaves the ambiguous entries untouched and reports publication indeterminate.

`SIGINT`, `SIGTERM`, and `SIGHUP` are converted into transaction exceptions while composition is
active. Before rename, a proved quarantine cleanup leaves no operator-owned staging inode. After
rename, a proved rollback leaves no destination. A clean signal interruption exits
`128 + signal`; an unprovable cleanup or rollback exits `2`. `SIGKILL` and machine loss cannot be
handled. If the process dies without a success record, preserve the directory and run `verify`;
do not infer success or absence from the missing terminal line.
