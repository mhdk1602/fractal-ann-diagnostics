# Development phase-two view

The host-side [`development_phase2_view.py`](../operators/development_phase2_view.py) operator
publishes the only label-bearing tree admitted by the frozen post-embedding runtime. It runs after
the standalone phase-one selection has been committed and reproduced, and after a future Quicknet
round has fixed the design seed. The operator is outside the wheel and outside the confirmatory
Docker context.

This boundary has one purpose: the candidate image may read fit and calibration judgments, but it
must never receive a sealed query judgment, corpus source, or complete custody package.

## Closed membership

The publication contains one receipt beside a mounted `view/` directory. `view/` has exactly 29
files:

```text
view/
  inventory.json
  inventory.sha256
  assignments.jsonl
  datasets/<each registered corpus>/
    fit/queries.jsonl
    fit/qrels.jsonl
    calibration/queries.jsonl
    calibration/qrels.jsonl
  datasets/{scifact,hotpotqa-fullwiki,t2-ragbench}/
    fit/evidence-bundles.jsonl
    calibration/evidence-bundles.jsonl
phase-two-view-receipt.json
```

The inventory remains the byte-exact, 110-artifact NFC custody inventory. It names corpus and
sealed artifacts as upstream commitments, although those payloads are absent from `view/`. The
candidate container receives only `view/`; the receipt stays on the host.

No CLI option selects a corpus, stage, role, path, seed, or beacon round. The registered corpus
set, two development stages, three evidence-bearing corpora, and 29 relative paths are constants.

## Ordering gate

Before the complete NFC source root is opened, the operator performs the following label-free
chain:

1. Load the externally pinned partition-audit receipt and require zero cross-stage components.
2. Verify the read-only phase-one view from its external receipt digest.
3. Load the independently published selection receipt and require 1,375 selected families across
   the ten fixed corpus/stage strata.
4. Recompute that selection from the phase-one view into a temporary private file and require
   byte-for-byte equality.
5. Verify the attested pre-round design-seed commitment, its resolved Sigstore-attestation
   admission, and the BLS-verified Quicknet reveal.
6. Acquire exclusive advisory leases on every label-free control, its parent, and the complete
   phase-one tree. The attestation admission path discovered from the verified reveal joins this
   lease set.
7. Repeat the whole label-free admission while those leases remain held, and require the inventory,
   audit, phase-one view, selection, attestation, and seed scope to describe one cohort.

Only then does the operator open the custody-complete root. It parses the canonical inventory,
checks its checksum and exact 112-file tree, and selects the 27 payload files required beside the
two inventory controls. Assignment, query, and qrel rows must also equal their typed partition-
audit source pins. Evidence bundles are checked against the inventory because the partition graph
does not consume them.

The operator copies bytes without parsing judgment values. Every copy is rehashed and checked for
byte count, line count, terminal newline, and source metadata stability.

## Filesystem transaction

Inputs and output parents participate in
`fractal-exclusive-posix-advisory-custody-v1`. The operator takes nonblocking exclusive advisory
leases on every label-free control (including the resolved attestation admission) before it opens
the complete NFC source. It repeats label-free admission under lease, then acquires the source-tree
leases and retains all descriptors until publication finishes. The receipt explicitly excludes a
noncooperating same-UID writer. Use a separate producer UID or immutable snapshot when that threat
is in scope.

Mode bits are not accepted as the whole custody proof. Every retained control, parent, source
directory, source file, temporary member, published member, and bootstrap artifact is checked for
descriptor-bound ACLs. On macOS the operator uses `acl_get_fd_np(ACL_TYPE_EXTENDED)`; on Linux it
enumerates descriptor xattrs and rejects POSIX access or default ACLs. ACL absence and stable
descriptor metadata are checked again before publication completes. An inspection error closes the
gate.

The package is assembled below a private temporary name. Files start at `0600`, directories at
`0700`, and every directory is synced. Before publication, files become `0400` and directories
`0500`. Publication uses `renameatx_np(RENAME_EXCL)` on macOS or
`renameat2(RENAME_NOREPLACE)` on Linux. A pre-existing destination is never replaced.

The operator verifies the public name and every artifact after rename. A failed readback is rolled
back to the still-pinned temporary inode. If the rename or rollback state cannot be proved, it
raises `DevelopmentPhase2PublicationIndeterminate` and preserves the evidence for manual
adjudication. Every publish and rollback move enters an indeterminate state before the syscall.
Two descriptor-relative observations then prove whether the pinned inode is solely at the source or
destination name. Cleanup is allowed only after that proof places it at the private temporary name;
an indeterminate tree is never emptied.

The receipt schema is `fractal-development-phase-two-view-receipt-v2`. It binds the source and
output roots, inventory and audit, phase-one and selection receipts, seed commitment, resolved
attestation-admission path and digest, reveal, derived design seed, all 29 artifact descriptors,
their aggregate digest, and the cooperative capture-set digest. The capture-set digest also commits
to the admission path and digest, so neither can be substituted while preserving the receipt.

Standalone verification needs the published package, an externally recorded receipt digest, and
the still-sealed seed controls named by that receipt. It does not reopen the custody source. It does
fresh commitment, attestation, BLS reveal, and seed-derivation verification rather than accepting
the receipt's seed value as evidence of itself.

## Exact-P resume bootstrap

The independent selection must enter the frozen operator as an existing boundary. Calling
`fractal-post-embedding-development run` would create a new selection internally, so the host
operator instead creates the post root with exactly:

```text
operator-config.json
selection-receipt.json
```

The bootstrap freshly verifies the commitment, attestation admission, and reveal. The config must
name `view/`, the same inventory and partition audit, the seed from that verified reveal, and the
intended post root. The copied selection bytes must equal the independent phase-one receipt. The
host then invokes `fractal-post-embedding-development resume`.

The two-file prefix is published with the same no-replace name-state classification. Its external
bootstrap receipt is written through a private temporary inode, synced, sealed to `0400`, moved with
no replacement, rebound to the pinned inode, and reread before success. If receipt publication is
indeterminate, the prefix and receipt are retained together for custody review. A rollback whose
name state cannot be classified likewise preserves the prefix rather than deleting its only inode.

Exact P accepts this canonical prefix. Its materializer recomputes the selection from the mounted
phase-two view before it resolves or opens any qrel or evidence source. A mismatch stops before
development labels enter the materialized package.

## Commands

Build and record the emitted receipt digest:

```bash
python -m operators.development_phase2_view build \
  --source-root "$COMPLETE_NFC_ROOT" \
  --staged-inventory-sha256 "$INVENTORY_SHA256" \
  --partition-audit "$PARTITION_AUDIT" \
  --partition-audit-file-sha256 "$PARTITION_AUDIT_SHA256" \
  --phase1-view-root "$PHASE1_VIEW" \
  --phase1-view-receipt-sha256 "$PHASE1_RECEIPT_SHA256" \
  --selection-receipt "$SELECTION_RECEIPT" \
  --selection-receipt-sha256 "$SELECTION_RECEIPT_SHA256" \
  --seed-commitment "$SEED_COMMITMENT" \
  --seed-commitment-sha256 "$SEED_COMMITMENT_SHA256" \
  --seed-reveal "$SEED_REVEAL" \
  --seed-reveal-sha256 "$SEED_REVEAL_SHA256" \
  --output-root "$PHASE2_ROOT"
```

Verify without reopening source custody:

```bash
python -m operators.development_phase2_view verify \
  --root "$PHASE2_ROOT" \
  --receipt-sha256 "$PHASE2_RECEIPT_SHA256"
```

After the candidate image writes and pins its canonical post config while the post root is absent,
create the resume prefix:

```bash
python -m operators.development_phase2_view bootstrap-post-resume \
  --root "$PHASE2_ROOT" \
  --receipt-sha256 "$PHASE2_RECEIPT_SHA256" \
  --post-config "$POST_CONFIG" \
  --post-config-sha256 "$POST_CONFIG_SHA256" \
  --post-output-root "$POST_OUTPUT_ROOT" \
  --bootstrap-receipt-output "$BOOTSTRAP_RECEIPT"
```

Run only the frozen `resume` command after this point. `run` must reject the pre-existing root.
