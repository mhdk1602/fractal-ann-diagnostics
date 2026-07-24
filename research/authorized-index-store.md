# Authorized HNSW index store

The confirmatory runner does not apply a policy mask after unrestricted ANN
search. It searches an index whose membership already equals the authorized
document set.

`authorized_index_store.py` builds one immutable HNSW index for every mask in a
verified compiled-policy catalog. Each HNSW local label has one separately
pinned `int64` entry that maps it back to the global document row.

## Scientific boundary

The apparatus distinguishes two document matrices:

| Matrix | Role | HNSW access |
|---|---|---|
| old-model document vectors | active migration state, including stale-index effects | indexed |
| current-model document vectors | exact authorized reference action | never indexed |

Both matrices must come from the same verified embedding-store receipt. They
must have the same shape and canonical document row order, but distinct model
and file digests. The current matrix is hashed during source admission. Its
values are not passed to the HNSW backend.

The compiled-policy universe digest must equal the embedding store's canonical
document row-order digest. A matching row count is insufficient. This prevents
a mask created for one ordering from being applied to another ordering with the
same number of documents.

## Frozen inputs

Construction requires caller pins for both source receipts:

1. the verified embedding-store receipt SHA-256;
2. the verified policy-intervention receipt SHA-256.

The policy package is admitted as an exact tree. Every control file and mask
must match the path, byte count, and SHA-256 in the intervention receipt. Each
mask is counted from its packed bytes without expanding the whole mask in
memory. Nonzero trailing bits are rejected.

The index configuration fixes:

- hnswlib release version and installed extension-file SHA-256;
- metric, `M`, `efConstruction`, and random seed;
- one construction thread and one query thread;
- maximum vector batch size and verification `ef`;
- builder identity and the `fail-clean-no-resume-v1` failure rule.

Mutable release names and unpinned backend binaries are inadmissible.

## Bounded construction

The builder memory maps the old document matrix through an already-opened,
no-follow file descriptor. It reads the packed mask in bounded slices. For each
slice it performs four actions:

1. derive authorized global rows;
2. write those rows into the local-to-global map;
3. cast only the selected old vectors to contiguous `float32`;
4. add them with contiguous local HNSW labels.

The corpus, full vector matrix, and full authorized subset are never copied into
process memory. Peak construction data is bounded by the registered batch size
times the embedding dimension, plus a small mask slice.

Every index receipt row binds the exact mask digest and count, both vector
descriptors, document row order, universe digest, backend binary, configuration,
policy catalog, source receipts, and builder identity. The resulting binding
digest is specific to one mask and one source state.

## Publication and interruption

Construction occurs in a private, randomly named staging directory beside the
final target. The final path and a per-target lock must be absent. A backend
error removes the staging directory and its lock. There is no checkpoint to
resume and no partially valid output.

After every index and row map is written, the builder:

1. hashes each payload;
2. records the exact payload-tree digest;
3. writes a canonical receipt;
4. reopens every artifact through no-follow descriptors;
5. checks each row map against its packed mask;
6. loads every HNSW index and issues one bounded smoke query;
7. publishes with an operating-system no-replace rename.

An existing target is never replaced.

## Backend byte determinism

Single-thread construction, a fixed seed, and fixed parameters constrain the
build. They do not prove that hnswlib emits identical bytes across processors,
compilers, standard libraries, or package builds.

Before freezing the confirmatory package, build the same source state at least
three times in the registered OCI image. Compare every HNSW SHA-256 and row-map
SHA-256. Record the image digest, processor architecture, hnswlib wheel digest,
loaded extension digest, and observed hashes. If any HNSW digest differs, state
that byte determinism was not observed.

"Every HNSW" includes the full-active HNSW used by the online low- and
high-effort actions. The production factory retains three full-active builds per
corpus in its reproducibility tree and copies only registered replica 1 into the
online execution package. The terminal reproducibility receipt binds the three
source-equal digests and the selected package bytes.

The confirmatory run does not depend on a cross-host determinism claim. It uses
one selected set of index bytes as immutable registered input. Those bytes are
externally pinned with the rest of the freeze package. The sealed runner loads
them; it does not rebuild them.

The low-level builder remains `fail-clean-no-resume-v1`: an ordinary exception
removes its unpublished work, and a direct second builder does not infer that an
existing lock is stale. During production-factory resume, a higher-level corpus
lock supplies that missing exclusion boundary. The factory may remove an exact
unpublished staging name only after it proves the builder's advisory lock is no
longer held. It never removes a published index store or receipt.

## Verification call

```python
from fractal_ann_diagnostics.authorized_index_store import (
    HnswlibBackend,
    verify_authorized_index_store,
)

backend = HnswlibBackend()
verification = verify_authorized_index_store(
    "/absolute/frozen/authorized-indexes",
    embedding_store_root="/absolute/frozen/embeddings",
    policy_intervention_root="/absolute/frozen/policy",
    expected_embedding_receipt_sha256="<64 lowercase hex characters>",
    expected_policy_receipt_sha256="<64 lowercase hex characters>",
    expected_store_receipt_sha256="<64 lowercase hex characters>",
    backend=backend,
)
```

The verifier checks exact tree membership, payload digests, source bindings,
mask-to-map equality, backend identity, and load/query behavior. A change to an
index, row map, source matrix, mask, configuration, or receipt aborts admission.

The sealed entrypoint then calls `open_verified_document_matrices`. This second
boundary requires the embedding receipt to match the index receipt, requires
the active and current descriptors to equal the receipt fields exactly, and
holds both files through no-follow descriptors for the whole online matrix.
On exit it rehashes each open descriptor, reopens the named path, and compares
device, inode, mode, link count, size, and timestamps. Replacing the current
matrix with another float32 array of the same shape is therefore a failed
attempt, not an alternate exact-search reference.

## What this artifact does not establish

An authorized index removes post-filtering confounding from the ANN action. It
does not establish causal identification by itself. Confirmatory status still
depends on the registered cohort, withheld outcome custody, fixed action
schedule, sealed execution, and prespecified analysis.
