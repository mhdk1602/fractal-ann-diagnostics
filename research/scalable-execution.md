# FullWiki-scale sealed execution

An inline execution artifact is acceptable for small corpora. It is the wrong
control surface for FullWiki. Serializing every document into one Python object
would couple admission, hashing, audit provenance, and retrieval to a corpus-wide
memory load.

The sharded contract separates control data from bulk data. The control plan is
small, canonical JSON. Corpus text, matrices, and index bytes remain in pinned
artifacts. No plan field contains a document row.

This module defines the storage and admission contract. It does not build a
FullWiki index, execute the sealed trial, or assert a production result.

## Bound artifact graph

The plan is a pre-freeze artifact. Its direct pins cover:

| Role | Admitted identity |
| --- | --- |
| Corpus partition | Canonical shard inventory file |
| Query surface | External query/trial store and its canonical typed receipt |
| Deployed migration state | Active vector matrix |
| Current exact oracle | Current-truth vector matrix |
| Retrieval backend | HNSW index built from the active matrix |
| Audit provenance | Fixed-width SHA-256 sidecar |

The shard inventory contains ordered, contiguous document-ID intervals. Each
interval names one file by artifact ID, relative path, byte count, and SHA-256.
Changing a shard changes its receipt row, the inventory bytes, the inventory
pin, and the plan digest.

The dependency graph has no future-digest backreference. The plan pins its
child artifacts. A plan-bound leaf receipt records fresh verification of those
children. The C1 manifest then pins the SHA-256 Merkle identity of the complete
directory package. Neither the plan nor its leaf receipt embeds the C1 manifest
digest. The sealed-run admission receipt can therefore bind C1 without asking a
pre-C1 artifact to predict its future container digest.

The plan carries only four fields per trial:

- opaque trial key
- opaque family key
- query-store row number
- digest of that query-store record

Query text and query vectors remain in the query/trial store. Its distinct
receipt pin commits the selection seed, family counts, ranking algorithms,
nested-row cardinality, source bindings, and vector epochs used to create those
rows. A migration trial binds two query-vector rows with the same opaque trial
identity: one encoded by the active model revision and one encoded by the
current model revision.

## Two vector epochs

The active-vector store and current-truth-vector store are not aliases.

The active store is the matrix used to build the deployed HNSW artifact during
a migration interval. The HNSW descriptor names its source matrix SHA-256.
The current-truth store is the matrix used for exact authorized retrieval under
the current embedding state. Both descriptors bind:

- an explicit portable dtype such as little-endian float32
- a two-dimensional shape
- raw C-order storage
- document-ID row order
- exact byte count
- artifact SHA-256
- ordered document-universe SHA-256

The plan rejects a row-count mismatch, embedding-dimension mismatch, ambiguous
native-endian dtype, incorrect byte count, swapped vector role, or an HNSW pin
that names the wrong source matrix.

This distinction permits the registered comparison between stale deployed
geometry and current exact truth without silently replacing one with the other.

The same epoch split applies to queries. HNSW probe, low, and high actions use the active query
vector. The exact authorized action uses the current-truth query vector against the current-truth
document matrix. The execution-order receipt records separate digests for both query vectors. Both
role fields remain required even when the two revisions coincide.

## Digest-only provenance

The provenance sidecar is a raw binary file with one 32-byte SHA-256 value per
document. Record i begins at byte offset:

    i * 32

Admission performs three checks before the registry is returned:

1. The supplied receipt must exactly attest every direct pin and every shard pin.
2. The sidecar length must equal document_count multiplied by 32.
3. A streaming SHA-256 pass over the sidecar must match its immutable pin.

The initial digest pass is linear in sidecar bytes. The sidecar is never copied
into a Python list or tuple. After admission, one content-hash lookup performs a
single 32-byte positional read. It is O(1) in corpus size.

The registry retains open file and parent-directory descriptors. Each lookup
checks file identity, link count, size, mode, and mutation timestamps before and
after the positional read. A symlink, hard link, in-place mutation, or path
replacement causes a closed failure.

Corpus shards are not opened by registry admission. Their bytes were attested
when the artifact-verification receipt was created. The registry reads the
bounded inventory and binary digest sidecar only.

## Canonical control I/O

The v4 plan and shard inventory use closed JSON schemas. Their loaders reject:

- missing or unknown fields
- duplicate JSON keys
- non-finite numbers
- invalid UTF-8
- noncanonical key or row order
- indentation, alternate separators, or an incorrect trailing newline
- control files beyond the fixed byte limit
- file or ancestor symlinks
- hard-linked controls
- mutation during read

Writers use exclusive creation with mode 0600. They do not replace an existing
path, follow a link, or write into a parent directory that another identity can
modify.

Logical plan identity is SHA-256 over canonical JSON without the file newline.
This matches the existing inline execution-artifact convention. File pins use
SHA-256 over the bytes on disk, including the one required newline.

## Immutable package closure

Each `online-execution` manifest row identifies two different objects:

| Manifest field | Meaning |
| --- | --- |
| `sha256` | Merkle digest of the complete execution-package directory |
| `revision` | `sha256:<digest>` of the canonical logical plan |

These values must differ. Runtime receipts, action panels, predictions, and
offline joins bind the logical digest carried by `revision`. Artifact admission
binds the outer directory digest carried by `sha256`. This distinction prevents
a receipt file or packaging-path change from masquerading as a scientific plan
change, while still making every package byte immutable.

The package has two reserved control paths at its root:

- `sharded-online-execution-plan.json`
- `leaf-verification-receipt.json`

All other files must be named by a direct plan pin or by a shard pin in the
plan-pinned inventory. Finalization first scans the package without the receipt,
requires the exact declared file and directory set, verifies every leaf through
no-follow file descriptors, and writes the canonical receipt by exclusive
creation. It then computes the outer tree digest.

Admission repeats the tree scan against the C1 `sha256`, parses the plan against
the manifest `revision`, checks the exact membership set, loads the receipt, and
re-verifies every leaf. A second whole-tree scan must equal the first. Extra
files, missing files, empty undeclared directories, symlinks, hard links,
substituted bytes, a changed plan, a changed receipt, or mutation during either
scan closes admission.

The leaf receipt binds the logical plan digest, the exact plan-file digest, and
one exact verification row per required child. It deliberately contains no C1
manifest hash. C1 is the later immutable statement that pins the finished tree.

## Admission sequence

The sealed runner should perform these steps in order:

1. Load the canonical sharded plan.
2. Verify the package tree against the C1 artifact `sha256` and the plan against
   the C1 artifact `revision`.
3. Load the plan-bound leaf receipt and require exact package membership.
4. Load the plan-pinned shard inventory.
5. Check all direct and shard pins against fresh receipt rows.
6. Open the digest-only provenance registry.
7. Open vector and index artifacts through backend-specific no-follow readers.
8. Bind both active and current-truth query rows to the opaque trial rows.
9. Start the registered paired-action runner.

Steps 6 and 7 remain backend integration work. The contract supplies the
identities and invariants those readers must enforce.

## Compatibility surface

Action-panel and prediction code should consume the three-field compatibility
view:

- document count
- sorted opaque trial keys
- logical execution artifact SHA-256

The helper functions derive those values from either the existing inline
execution artifact or a sharded plan. They do not touch corpus shards when given
a sharded plan.

## Required operator evidence

A real FullWiki execution package still needs:

- exact shard-production command and tool digest
- document-universe construction rule and observed digest
- query/trial store builder and its sealed input
- canonical query/trial receipt and its direct plan pin
- active and current-truth vector builders
- HNSW build configuration, seed, library version, and observed index digest
- finalized package tree digest, logical revision, and leaf receipt
- external plan and receipt anchors
- machine, filesystem, and resource records for the single sealed run

Until those artifacts exist and are independently pinned, this file describes
an admissible execution mechanism rather than a completed confirmatory run.
