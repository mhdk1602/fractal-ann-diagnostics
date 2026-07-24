# Label-independent compiled policy intervention

This compiler turns a frozen experimental assignment into bit-packed policy
masks, OPA lookup data, and an execution schedule. It reads seven facts from the
admitted execution:

- corpus identifier
- stage
- document count
- ordered document-universe SHA-256
- opaque trial keys
- opaque family keys
- execution artifact SHA-256

It does not read document text, query text, qrels, response judgments, returned
documents, or generated answers.

## Frozen configuration

The configuration records:

| Field | Meaning |
| --- | --- |
| seed_sha256 | Prespecified 256-bit assignment seed |
| baseline_seed_sha256 | Separate prespecified seed for baseline decision vectors |
| allow_rate_strata | Probability thresholds, defaulting to 0.25, 0.50, and 0.75 |
| subject_ids | The single subject exposed to the compiled policy |
| policy_state_ids | OPA environment states aligned positionally with the rate strata |
| assignment_repetitions | Fixed at one for the confirmatory schedule |
| grouped_execution_order | Explicit state order for mask-grouped execution |
| policy_bundle_revision | Immutable sha256-prefixed OPA bundle revision |
| baseline_policy_revision | Distinct immutable identity for the synthetic pre-mutation epoch |

Rates must be unique and increasing. The design requires exactly three policy
states, one subject, and one repetition, hence three execution blocks. The group
order must be a permutation of those states. The two seeds and two revisions must differ. Movable
revisions such as latest, main, or a version tag are inadmissible.

The configuration digest is SHA-256 over canonical JSON without the file
newline. Every package receipt and trial schedule binds that digest.

## Mask assignment

Two pseudorandom ranks are computed for each ordered document position. The current rank uses
`seed_sha256`; the baseline rank uses `baseline_seed_sha256`. Each HMAC message contains a fixed
scheme identifier, corpus, ordered document-universe digest, and the zero-based row position.

For an allow-rate threshold p, row i is authorized when:

    uint64(HMAC-SHA256(seed, corpus || universe || i)[0:8]) < floor(p * 2^64)

The compiler never hashes a response-side datum. Subjects and policy states do not affect row rank.
Within each epoch, one ranking is thresholded at every prespecified rate, so its masks are nested.
Baseline masks are excluded from the OPA catalog and cannot authorize a live request.

For state \(s\), policy churn is recomputed from both complete Boolean vectors:

\[
\operatorname{churn}(s)=\frac{1}{|D|}\sum_{d\in D}
\mathbb{1}\{P_{baseline,s}(d)\neq P_{current,s}(d)\}.
\]

The schedule records both mask digests, both authorized counts, both policy revisions, and this
exact Hamming fraction. Every state must have non-zero mutation. An allow-rate difference,
changed-count field, or supplied percentage cannot replace the decision vectors.

The realized authorized count is measured after compilation. Empty and full
masks are rejected. Realized allow rate is authorized_count divided by the
frozen document count, and both values are written into the schedule. The
catalog carries the exact count used by the OPA adapter.

Masks use the numpy-packbits little-bit convention already enforced by
CompiledPolicyMaskStore. One FullWiki-scale mask therefore occupies roughly one
eighth of the document count in bytes.

## OPA data contract

The policy package is fractal_auth.retrieval. Its lookup data remains mounted at
data.fractal. The generated opa-data.json has exactly these top-level fields:

    {
      "assignments": {
        "reader-a": {
          "medium": {
            "authorized_count": 2616664,
            "mask_id": "allow-01-...",
            "mask_sha256": "..."
          }
        }
      },
      "document_count": 5233329,
      "document_universe_sha256": "...",
      "mask_catalog_sha256": "...",
      "policy_revision": "sha256:..."
    }

No document-ID array is sent to OPA. Each subject and policy-state entry contains
three scalars regardless of corpus size. OPA selects a mask identifier; the
local compiled-mask store admits the pinned bytes and reconstructs the Boolean
vector.

## Grouped trial schedule

Each admitted family must contain exactly three distinct trial keys. The
compiler ranks those keys by SHA-256 over a domain identifier, the frozen
configuration seed, the opaque family key, and the opaque trial key. It then
maps the ranked keys positionally to `grouped_execution_order`. Each family has
one trial under each policy state, and each trial occurs once in the entire
schedule.

Rows are emitted in the configured policy-state group order. Every row in one
group names the same mask, so an authorized HNSW view can remain warm across the
block. Trial keys are bytewise sorted inside each block. The three blocks are a
disjoint partition of the execution plan, not repeated copies of it.

Each row contains:

- global schedule order
- mask group order
- opaque trial key
- opaque family key
- repetition number
- subject
- finite environment with policy_state and assignment_repetition
- environment digest
- mask ID, mask digest, authorized count, and realized allow rate
- baseline mask ID, path, digest, byte count, authorized count, and policy revision
- exact baseline-to-current policy churn
- expected immutable policy revision

The schedule records the assignment algorithm, seed digest, and a canonical
digest of the complete trial-to-state map. Its constructor recomputes the rank
within every family. A missing trial, duplicate assignment, family-size change,
or altered state mapping is rejected before runtime admission.

The schedule contains no action order. The paired-action runner performs its
separate seeded action permutation within each scheduled trial.

## Package contents

One finalized directory contains:

    intervention-config.json
    compiled-policy-catalog.json
    opa-data.json
    trial-schedule.json
    intervention-receipt.json
    baseline-masks/baseline-allow-00-<binding>.bin
    baseline-masks/baseline-allow-01-<binding>.bin
    baseline-masks/baseline-allow-02-<binding>.bin
    masks/allow-00-<binding>.bin
    masks/allow-01-<binding>.bin
    masks/allow-02-<binding>.bin

Mask identifiers include a digest over the execution artifact, document universe, configuration,
policy revision, epoch, seed, state, and requested rate. The package receipt repeats both seeds and
revisions, records one typed transition per state, and pins every payload other than itself.

## Admission and atomic publication

Package construction follows this order:

1. Refuse an existing destination and acquire an exclusive sibling lock.
2. Compile all deterministic payloads in memory.
3. Create a private staging directory.
4. Write every file by exclusive creation.
5. Check the exact directory tree, canonical JSON, catalog, every mask, schedule
   coverage, OPA assignments, source bindings, file sizes, and file digests.
6. Atomically rename the verified staging directory to the final destination.

The parent directory must be owned by the runner and cannot be writable by group
or other identities. Loaders reject file links, ancestor links, hard links,
noncanonical JSON, extra files, mutation, and overwrite. A source interface that
changes between the initial and final binding read is rejected.

Public verification recompiles expected bytes from the supplied execution and
frozen configuration. A changed seed, document universe, execution digest, or
policy revision therefore cannot be admitted as the same package.

## Scope boundary

This compiler creates a synthetic experimental policy mutation. Its Hamming fractions describe the
seeded corpus masks. They do not estimate an enterprise entitlement distribution or production
policy change rate. The compiler does not launch OPA, build an authorized HNSW index, execute
retrieval actions, or score a trial.
