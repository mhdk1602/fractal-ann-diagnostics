# Development selection staging view

The host-side
[`development_staging_view.py`](../operators/development_staging_view.py) operator publishes the
smallest filesystem view from which `development_cohort select` can commit the registered
fit/calibration families. It runs outside the candidate image and is excluded from the Python
wheel. Its job is custody reduction, not scientific computation.

This is a phase-1 view. It contains no qrel, evidence, answer, label, outcome, custody, or sealed
payload file. It is not the later development-label view consumed by cohort materialization.

## Exact membership

The published tree contains 13 pinned input artifacts and one receipt:

```text
inventory.json
inventory.sha256
assignments.jsonl
datasets/
  <each registered corpus>/
    fit/queries.jsonl
    calibration/queries.jsonl
development-staging-view-receipt.json
```

The five corpus identifiers are fixed to `scifact`, `hotpotqa-fullwiki`, `t2-ragbench`, `bright`,
and `miracl-transfer`. No command-line option can add a corpus, stage, role, or path.

`inventory.json` remains the byte-exact custody inventory. It names sealed and label-bearing
artifacts as metadata because its SHA-256 is already bound by the embedding suite and partition
audit. Those payloads are absent. `assignments.jsonl` also remains byte exact: the selection code
filters its fit/calibration rows after checking the complete assignment-ledger digest. Rewriting
either control would sever the registered inventory and audit bindings.

The operator traverses the source projection from one pinned root descriptor and copies only the
11 registered payload files above. It does not open sealed query files. Directory enumeration
errors are fatal. Empty, unreadable, or otherwise unregistered directories cannot disappear from
the membership check. Any injected qrel, evidence, label, custody, or unregistered file makes
source membership fail before the temporary output is created.

## Admission

`build` requires five external facts:

| Input | Check |
| --- | --- |
| online projection root | real, operator-owned, read-only tree with exact receipt membership |
| staged inventory SHA-256 | exact `inventory.json` file digest |
| projection receipt SHA-256 | exact canonical `projection-receipt.json` digest |
| partition-audit path | real, operator-owned, read-only regular file |
| partition-audit file SHA-256 | exact canonical audit-file digest |

The projection receipt must use
`fractal-online-staging-projection-v1` and
`corpus-query-assignment-controls-only-v1`. The partition audit must use
`fractal-scalable-query-partition-audit-v1`, the registered algorithm and near-duplicate
configuration digests, and zero cross-stage components. Admission parses the complete typed audit
receipt rather than a field subset. It validates all source pins, aggregate counts, component
counts, structural exclusions, query coverage, and membership digests. It then binds the audit's
assignment seed, staging configuration, fifteen corpus/stage strata, source visibility, and count
rows to the inventory.

For `assignments.jsonl` and every fit/calibration query file, the inventory row, projection row,
and partition-audit source row must be identical. Query record counts must also equal the audit
stratum counts. The receipt binds:

- the staged inventory digest;
- projection receipt and projected-artifact-set digests;
- partition-audit file, source-artifact-set, and component-membership digests;
- the assignment-ledger digest;
- the absolute source, audit, and output paths;
- the captured input set and its cooperative lease premise; and
- all 13 output artifacts plus their artifact-set digest.

The receipt schema is `fractal-development-staging-view-receipt-v2`.

## Input custody premise

The custody contract is `fractal-exclusive-posix-advisory-custody-v1`. Before it copies the
selected bytes, the operator acquires nonblocking exclusive POSIX advisory leases on the source
root, every admitted source directory and file, and the audit file and its parent. It discovers
the exact projection membership once, acquires the complete lease set, then reads and admits the
same source again. Only this second, leased capture supplies the published artifacts. The operator
also leases the output parent and the complete temporary tree. These descriptors remain open
through the publication proof and transaction closure.

The receipt's `input_custody` object records:

- `capture_set_sha256`, which binds the captured artifact paths, byte counts, and digests plus the
  projection-receipt and partition-audit file digests;
- `producer_parent_and_file_leases_held_through_publication: true`; and
- `noncooperating_same_uid_mutation_excluded: true`.

That last field is a scope exclusion. POSIX advisory locks constrain only producers that follow the
same file-and-parent lease protocol. A same-UID process can ignore them, retain a writable
descriptor, call `pwrite`, or change names without acquiring a lease. No finite sequence of
`stat`, rehash, or name observations can rule out such a write after the final observation. If the
custody claim must include a hostile or defective same-UID process, run the producer under a
separate UID or publish the admitted inputs as a read-only immutable snapshot before this operator
starts.

## Publication and modes

The output parent must be owned by the invoking nonroot identity and grant no group or other
permissions. `build` writes under a private temporary directory, verifies exact membership and
every byte count, record count, and SHA-256, syncs the complete temporary tree, and rechecks the
source and output-parent bindings. Temporary directories and files start at `0700` and `0600`.
After materialization, the operator changes every file to `0400` and every directory to `0500`,
syncs the mode changes, and verifies the complete tree again. Drift before publication is an
ordinary fail-clean error only when the pinned tree is proved to remain at its temporary name.

Publication uses `renameat2(RENAME_NOREPLACE)` on Linux or
`renameatx_np(RENAME_EXCL)` on macOS. The output-parent descriptor remains pinned for temporary
creation, writes, directory fsyncs, sealing, verification, rename, and the parent fsync. macOS
requires the renamed directory itself to be owner-writable. The sealed-rename helper changes only
the pinned root to `0700` for that syscall, then restores `0500` and syncs it before returning or
propagating the native error; files and descendant directories remain read-only throughout.

Rename is part of an explicit guarded state transition. Two matching descriptor-relative
observations must classify the pinned inode as temporary, published, or rolled back. Success binds
the parent device, inode, mode, owner, group, and absolute path; proves that the output name alone
names the pinned inode; proves temporary-name absence; rereads the exact sealed artifact bytes
through both the output name and the pinned descriptor; and rechecks the read-only source names,
inodes, and bytes. The proof is repeated before success is returned.

Any error after rename triggers rollback with the same no-replace primitive. A process that races
to occupy the former temporary name cannot be overwritten. A successful rollback is synced and
proved twice to restore the pinned tree while leaving the public name absent; cleanup then removes
the temporary tree and proves the final name state. If rename outcome, rollback, cleanup, or any
later observation is ambiguous, the operator raises
`DevelopmentStagingPublicationIndeterminate` and preserves the filesystem evidence for an
operator decision. A preexisting output is never changed.

`SIGINT`, `SIGTERM`, and `SIGHUP` are translated into transaction-visible interruptions while
`build` is active. Before publication is proved, the normal cleanup or proved rollback path runs
and the interruption remains observable to the caller. After publication is proved, a
`BaseException` during descriptor closure, including `KeyboardInterrupt`, `SystemExit`, or one of
those signals, does not trigger rollback. The operator raises
`DevelopmentStagingPublicationIndeterminate` and leaves the public tree in place. The error
records the output path, former temporary name, and receipt digest so recovery can verify the
durable tree. `SIGKILL`, machine loss, and storage failure remain outside process-local signal
handling; a missing success acknowledgement requires recovery verification.

Published directories use mode `0500`; files use `0400`. The standalone verifier rejects a
writable publication. It requires these exact modes and the same owner-only boundary, rejects
links, hard links, special files, and mutation during reads, and verifies the bytes again after
checking that the absolute path still names its pinned root descriptor. It does not reopen the
source projection or partition audit. Every public verifier call requires the externally
recorded receipt SHA-256; self-authentication from the receipt file is not available.

Source, audit, receipt, and output-artifact reads use `O_NOFOLLOW` beneath pinned directory
descriptors and add `O_NONBLOCK` where the platform exposes it. A FIFO substituted before or
between inspection and open is rejected without waiting for a writer. Each regular file must have
one link. The operator compares device, inode, mode, owner, link count, size, modification time,
and change time before and after every read.

## Commands

Run the standalone module from a clean checkout. The operator has only `build` and `verify`
subcommands, and command-line option abbreviations are disabled:

```bash
set -euo pipefail

PROJECTION='/absolute/controlled/path/online-projection'
INVENTORY_SHA256='replace-with-staged-inventory-sha256'
PROJECTION_RECEIPT_SHA256='replace-with-projection-receipt-file-sha256'
PARTITION_AUDIT='/absolute/controlled/path/query-partition-audit.json'
PARTITION_AUDIT_SHA256='replace-with-partition-audit-file-sha256'
OUTPUT_PARENT='/absolute/private/path/development-selection'
OUTPUT_ROOT="$OUTPUT_PARENT/view-v1"

test "$(id -u)" -ne 0
test ! -e "$OUTPUT_ROOT"
mkdir -m 0700 "$OUTPUT_PARENT"

PYTHONDONTWRITEBYTECODE=1 python -m operators.development_staging_view build \
  --projection-root "$PROJECTION" \
  --staged-inventory-sha256 "$INVENTORY_SHA256" \
  --projection-receipt-sha256 "$PROJECTION_RECEIPT_SHA256" \
  --partition-audit "$PARTITION_AUDIT" \
  --partition-audit-file-sha256 "$PARTITION_AUDIT_SHA256" \
  --output-root "$OUTPUT_ROOT"
```

Record `receipt_sha256` from the canonical JSON result. A separate verifier needs only the
published tree and that digest:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m operators.development_staging_view verify \
  --root "$OUTPUT_ROOT" \
  --receipt-sha256 "$DEVELOPMENT_VIEW_RECEIPT_SHA256"
```

Run `fractal-development-cohort select` against this view to create the label-payload-excluded
selection receipt. The current all-in-one post-embedding operator still needs a separately
custody-produced development-label view for materialization. It reproduces the same selection
before opening fit/calibration qrels or evidence. This host operator does not construct or inspect
that second view.
