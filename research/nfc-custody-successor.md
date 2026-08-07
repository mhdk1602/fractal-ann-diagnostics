# NFC custody-successor assembly

- Status: executable host-side custody operator
- Implementation: [`operators/nfc_custody_successor.py`](../operators/nfc_custody_successor.py)
- Focused tests: [`tests/test_nfc_custody_successor_operator.py`](../tests/test_nfc_custody_successor_operator.py)

## Purpose

The production embedding pass created an NFC-normalized online projection. Its inventory names
all 110 staged artifacts, but the projection deliberately carries only 86 label-free artifacts.
The 24 qrel and evidence payloads remain in the original complete custody root. A partition audit
needs one complete namespace whose corpus bytes agree with the NFC inventory and whose label bytes
remain under the original custody boundary.

The host-side operator constructs that namespace without interpreting a payload record. It streams
files as opaque bytes and computes only byte count, newline count, and SHA-256. It is outside the
Python wheel and outside the confirmatory image build context.

## Exact 86/24 join

The successor inventory must contain exactly 110 bytewise path-sorted artifact rows:

- 86 rows whose roles are not `qrels` or `evidence-bundles`; and
- 24 outcome rows, exactly 15 `qrels` and 9 `evidence-bundles`.

The NFC projection receipt must use `fractal-online-staging-projection-v1` and
`corpus-query-assignment-controls-only-v1`. Its 86 rows must equal, field for field, the
non-outcome rows in the successor inventory. Its artifact-set digest, source inventory digest,
and both cardinalities are recalculated.

The original complete root has its own caller-pinned inventory. It must name the same 110 paths.
For every path, `dataset`, `stage`, `role`, and `visibility` must agree with the successor. The 24
outcome rows must agree in every field, including bytes, records, and SHA-256. Non-outcome bytes
may differ because they predate NFC normalization, but the operator verifies them against the
original inventory and never copies them.

The published root contains exactly:

```text
inventory.json                         # NFC projection, byte exact
inventory.sha256                       # NFC projection, byte exact
<86 non-outcome inventory artifacts>   # NFC projection
<24 qrel/evidence artifacts>           # original complete custody root
```

`projection-receipt.json` is an admitted source control, not a member of the reconstructed staged
root. The assembly receipt is also excluded from that root and must use a disjoint absolute path.

## Source admission

Both source roots must already be sealed with directories at `0500` and files at `0400`. Every
member must be owned by the invoking nonroot identity. Group or other permissions, symbolic links,
hard links, FIFOs, devices, sockets, extra files, missing files, noncanonical names, and path
aliases are fatal. Extended ACLs are fatal too: mode bits alone do not prove an owner-only custody
boundary when an ACL grants another principal access.

The operator opens roots component by component with no-follow flags. It takes nonblocking shared
advisory leases on each root, descendant directory, and file. All descriptors remain open through
copy, publication, and the post-publication proof. It checks device, inode, mode, owner, group,
link count, size, modification time, and change time before and after reads and again before the
transaction closes.

ACL admission is descriptor-bound. On macOS the operator queries `acl_get_fd_np` with
`ACL_TYPE_EXTENDED`; on Linux it enumerates descriptor xattrs and rejects
`system.posix_acl_access` or `system.posix_acl_default`. The same checks cover publication parents,
new temporary members, every sealed output member, the external receipt, and the final verifier.
An ACL inspection error closes admission rather than treating an unknown result as absence.

Each source artifact is hashed and newline-counted against its own inventory. Each selected file
is checked again while copied. Thus:

- every projection payload is proved against the successor inventory;
- every original payload is proved against the original inventory;
- every original outcome payload is also proved identical to its successor row; and
- every output payload is proved against the successor inventory.

The default aggregate artifact limit is 64 GiB per admitted tree. `--max-total-artifact-bytes`
can set a smaller or larger explicit ceiling. Canonical JSON controls are limited to 256 MiB, and
copy buffers are fixed at 1 MiB. The selected ceiling and fixed limits are recorded in the
receipt.

## Custody premise

The receipt records `fractal-exclusive-posix-advisory-custody-v1` and three claims:

- producer directories and files were leased through publication;
- publication parents were exclusively leased; and
- noncooperating same-UID mutation is excluded.

The third claim defines the threat boundary. Advisory locks constrain cooperating producers. A
same-UID process can ignore a lock or retain an earlier writable descriptor. Exact owner-only,
read-only modes and repeated metadata checks make accidental drift visible, but they cannot prove
absence of a hostile same-UID write after the last observation. Use a separate custodian UID or an
immutable filesystem snapshot if that adversary belongs in scope.

## Publication

The output and receipt parents must exist, be owned by the operator, and use mode `0700`. Existing
destinations are never changed. The operator creates a private temporary tree, copies every
artifact with exclusive file creation, syncs each file and directory, seals files to `0400` and
directories to `0500`, then verifies the exact temporary tree.

Publication uses `renameat2(RENAME_NOREPLACE)` on Linux or
`renameatx_np(RENAME_EXCL)` on macOS. The output root is published first and the external receipt
second. Both parent directories are synced. Two descriptor-relative observations classify each
rename by the pinned inode rather than trusting a syscall return alone. A receipt-publication or
post-rename verification failure moves the receipt and output back to their private temporary
names with the same no-replace primitive before cleanup. If either name transition cannot be
classified, the operator raises `NfcCustodyPublicationIndeterminate` instead of claiming a clean
failure.

Each move enters an indeterminate-safe state before the rename call. Cleanup may unlink or empty a
temporary object only after descriptor-relative classification proves that the pinned inode is at
its temporary name. An interruption after a successful rename but before classification therefore
leaves the published inode intact for custody review; it cannot turn a complete published tree into
an empty directory through descriptor-based cleanup.

`SIGINT`, `SIGTERM`, and `SIGHUP` become transaction-visible exceptions so the rollback and cleanup
path runs. `SIGKILL`, machine loss, and storage failure can still interrupt the two-name
transaction. After such an event, do not infer success from one surviving name: run `verify` with
the externally recorded receipt digest, or retain the names for custody review.

## Receipt

The canonical external receipt schema is `fractal-nfc-custody-successor-receipt-v1`. It binds:

- all four absolute paths: projection root, original root, output root, and receipt file;
- the successor inventory, original inventory, and projection-receipt SHA-256 pins;
- 110 path-sorted artifact rows, each marked `nfc-projection` or `original-custody`;
- projected, custody, output, and complete source-capture set digests;
- the 110/86/24 cardinalities;
- resource limits; and
- the cooperative custody declaration.

CLI output contains only receipt location, digest, cardinalities, output location, schema, and the
successor inventory digest. It never emits payload rows or payload content.

## Build and verification

Use physical, canonical absolute paths. Do not use a symlinked checkout or a `/tmp` alias.

```bash
set -euo pipefail

PROJECTION='/absolute/controlled/online-projection-nfc-v1'
SUCCESSOR_INVENTORY_SHA256='replace-with-exact-successor-inventory-sha256'
PROJECTION_RECEIPT_SHA256='replace-with-exact-projection-receipt-sha256'
ORIGINAL='/absolute/controlled/original-complete-staging-root'
ORIGINAL_INVENTORY_SHA256='replace-with-exact-original-inventory-sha256'
OUTPUT='/absolute/private/complete-nfc-custody-root'
RECEIPT='/absolute/private/receipts/nfc-custody-successor-receipt.json'

test "$(id -u)" -ne 0
test ! -e "$OUTPUT"
test ! -e "$RECEIPT"

PYTHONDONTWRITEBYTECODE=1 python -m operators.nfc_custody_successor build \
  --projection-root "$PROJECTION" \
  --successor-inventory-sha256 "$SUCCESSOR_INVENTORY_SHA256" \
  --projection-receipt-sha256 "$PROJECTION_RECEIPT_SHA256" \
  --original-root "$ORIGINAL" \
  --original-inventory-sha256 "$ORIGINAL_INVENTORY_SHA256" \
  --output-root "$OUTPUT" \
  --receipt-output "$RECEIPT"
```

Record `receipt_sha256` from the single canonical JSON result. Verification deliberately reopens
and re-admits both source trees; a receipt cannot substitute for live source custody evidence. The
standalone verifier holds exclusive cooperative leases on both publication parents while it reads
the receipt and proves the output. After the output proof it reopens the final output name through
the leased parent, compares the device and inode plus the complete stable-metadata tuple, retains a
shared lease on that rebound descriptor, rereads the receipt, and checks the source captures once
more before returning success.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m operators.nfc_custody_successor verify \
  --projection-root "$PROJECTION" \
  --successor-inventory-sha256 "$SUCCESSOR_INVENTORY_SHA256" \
  --projection-receipt-sha256 "$PROJECTION_RECEIPT_SHA256" \
  --original-root "$ORIGINAL" \
  --original-inventory-sha256 "$ORIGINAL_INVENTORY_SHA256" \
  --output-root "$OUTPUT" \
  --receipt-output "$RECEIPT" \
  --receipt-sha256 "$NFC_CUSTODY_SUCCESSOR_RECEIPT_SHA256"
```

Only after this verifier succeeds may the complete NFC root become the input to a fresh partition
audit. Assembly does not run that audit, open development labels, construct a development cohort,
or cross the freeze-before-labels boundary.
