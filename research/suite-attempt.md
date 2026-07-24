# Externally attested suite attempt

The confirmatory unit is the fixed five-corpus suite, not one corpus. Its attempt
ID is `SHA256("fractal-suite-attempt-v1\0" || manifest_sha256)`. The production
finalization receipt derives the sole `suite-attempt-<attempt-id>/` namespace
from the publicly verified C1 manifest. Callers cannot supply another base
directory or attempt ID. Before `OPENED`, the suite revalidates all five guarded
production-closure capabilities and the single finalization receipt. It also
verifies each registered runtime-plan instantiation by reversing exactly two
post-C1 substitutions: the C1 manifest SHA-256 and the final shared closure-tree
SHA-256. The earlier manifest-only template check is not an admission path.

The launcher writes label-free results to the pre-C1 staging root fixed by that
same finalization receipt. The canonical namespace begins with an empty
`online/` placeholder. `ONLINE_COMPLETE` first validates all five staged corpus
closures, copies every admitted regular file into a private sibling tree, and
rehashes the unchanged source. It then atomically exchanges the complete sibling
tree with the empty canonical placeholder. The empty placeholder is retained.
The source tree is never moved or deleted.

The state chain uses the closed `fractal-suite-state-v7` schema:

1. `OPENED`
2. `ONLINE_COMPLETE` or terminal `FAILED`
3. `LABELS_RELEASED` or terminal `FAILED`
4. `ANALYSIS_COMPLETE` or terminal `FAILED`

Each sequence has one filename, `<sequence>.state.json`. Success and failure
therefore compete for the same exclusive file rather than using sibling paths.
Every non-genesis record binds the exact preceding state-file digest. A state is
usable only after the matching provider evidence and signature bundle exist and
the full chain has been reloaded from disk.

## Bound files

`OPENED` binds the frozen manifest, protocol-registration receipt and registry
record, sealed-run receipt, code commit, OCI image, production-finalization
receipt, provisional and final closure-tree digests, attestation descriptor,
five execution-artifact digests, and both the five staging and five canonical
output directories.

The execution digests are derived from the five validated C1 workload specs.
A caller map is comparison input only: missing, substituted, or non-string
values fail, and insertion order cannot alter the persisted UTF-8 corpus order.
Before publication, each derived digest is joined back to the workload spec in
the finalized closure; the verified executable plan must bind that exact spec
file. This closes the manifest-to-closure-to-plan chain before an attempt can be
consumed.

The runtime binding is corpus-scoped. Five ordered rows retain each plan's
semantic and file digests, production-closure binding, typed plan-instantiation
receipt, and sealed-launch contract. Duplicate plan identities or a missing
corpus make `OPENED` invalid.

`ONLINE_COMPLETE` reloads and checks one persisted runtime-attestation receipt against each
corpus-specific frozen plan. It rehashes all five durable one-shot invocation markers and requires
each corpus attempt to bind its own plan and receipt. Each corpus closure retains semantic and file
digests for that plan, receipt, and marker. A reused identity across corpora is rejected.
For each fixed corpus it requires exactly these files in the corpus output
directory:

- runtime-attestation receipt
- runtime one-shot invocation marker
- production command-attempt marker
- sealed-online attempt receipt
- sealed-online result receipt
- predictions
- action panel
- action-panel admission receipt
- audit chain
- cache-preparation receipt
- execution-order receipt

No twelfth file is admitted in a staging or canonical corpus directory. Typed loaders reconstruct each
object. The command marker binds the exact config digest to the runtime plan's workload digest and
binds that plan and receipt to the manifest. The suite verifier checks manifest, run, corpus, execution, result,
prediction, panel, admission, audit-head, cache, and execution-order links. It
also recomputes every audit self-hash and predecessor link. Before returning the
closure, it freezes an ordered eleven-row transfer map. Each row binds one
registered role to its filename, exact file digest, and byte count. The state
record retains that map alongside the semantic and file-byte digests.

The closed `fractal-suite-output-transfer-v2` receipt resides outside the
source and canonical namespaces. It
binds the finalization-receipt file, attempt ID, manifest, both online-tree
URIs, the retained empty placeholder, exact tree inventories, and every copied
file's byte count and SHA-256. One nonblocking, runner-owned lock serializes the
post-scientific transfer. Source and target parents are pinned by directory file
descriptors; candidate directories are mode `0700`, candidate files are mode
`0600`, and every admitted object must belong to the runner identity. The
exchange uses directory descriptors, proves that the two admitted inode
identities swapped, and synchronizes both parents before publishing the
receipt. Immediately before copying, the worker joins every observed
role/filename/digest/count tuple back to the frozen `OnlineCorpusClosure`; a
fresh internally consistent snapshot cannot replace that earlier closure.
State verification repeats the closure-to-transfer comparison and rehashes both
copies and the placeholder.

Recovery is closed over five observable filesystem states:

1. With an empty canonical placeholder and no candidate, copying starts.
2. With an empty canonical placeholder and a partial candidate, each existing
   file must be an exact source prefix. The worker appends only the missing
   suffixes. An extra corpus, extra filename, wrong prefix, link, wrong owner,
   permissive mode, oversized prefix, or changed source aborts recovery.
3. With an empty canonical placeholder and a complete candidate, the worker
   revalidates both trees and performs the one atomic exchange.
4. With the complete canonical tree and retained empty placeholder, the
   exchange has already happened. The worker must not exchange again; it writes
   the deterministically reconstructed receipt if that file is absent.
5. With the complete canonical tree, empty retained placeholder, and receipt,
   the worker reloads canonical receipt bytes and returns that same receipt. The
   subsequent state transition may then finish.

No recovery branch calls a corpus runner, evaluates a query, or changes the
staged scientific tree. Recovery can copy already admitted bytes, exchange the
two output directories once, publish the derived receipt, and finish state
plumbing. Source, canonical, placeholder, or receipt mutation after publication
is fatal. A receipt before exchange, an incomplete canonical tree, two complete
trees, or any state outside the five cases is also fatal.

`LABELS_RELEASED` requires all five persisted timelock-decryption receipts in one
transition. A per-corpus completion anchor is no longer sufficient to decrypt a
label file: `release_timelock_label` also requires a file-backed, externally
verified `ONLINE_COMPLETE` token and checks that corpus's result digest against
the all-five record. The transition also reloads each released plaintext file, requires its bytes
to match the frozen per-corpus label pin and decryption receipt, and retains its canonical file URI,
byte count, and digest. A syntactically valid receipt cannot substitute another plaintext file.

`ANALYSIS_COMPLETE` binds the confirmatory input digest, analysis-attempt receipt,
analysis-result receipt, and final result, including both semantic and file-byte
digests. A locally written final result is not an externally attestable result
until the final state record has its own verified provider evidence.

## Attestation policy

`SuiteAttestationDescriptor` is a closed schema. It fixes the expected GitHub
Actions certificate identity, OIDC issuer, repository, workflow path, Git ref,
signer commit, public transparency log and key, timestamp authority and key, and
exclusive state service. The verifier must reject self-hosted signing runners.
It must verify the canonical state record itself as the signed subject, not trust
a workflow-controlled predicate as a substitute for the subject digest.

The injected verifier returns typed claims only after it has checked:

- the signature and exact subject digest;
- GitHub-hosted signer identity, workflow commit, repository, workflow, and ref;
- a public transparency-log entry with a strictly increasing log index;
- a cryptographically verified signed timestamp no earlier than registration,
  run start, or the preceding transition;
- an exclusive compare-and-swap transition under the fixed state-service key,
  including the exact preceding transition ID.

An HTTPS URI proves only that bytes were available at a location. It does not
prove signing identity, publication time, or uniqueness. The library therefore
has no HTTPS-only verifier and no permissive adapter.

`GitHubSuiteEvidenceVerifier` supplies the production GitHub adapter. It checks
an exact C0 workflow certificate with `gh attestation verify`, parses one public
Rekor entry and its observer time, and reconstructs the protected
manifest-derived Git ref as an append-only compare-and-swap ledger. The paired
workflow has read-only repository permission and accepts only a Git commit ID;
it refuses to sign unless that commit is the current protected ledger tip and
its sole new blob is the canonical next state. The exact layout, policy flags,
and execution order are specified in
[`github-state-attestation.md`](github-state-attestation.md). A dispatch workflow
that accepts caller-supplied bytes remains inadmissible.

The C1 registration workflow is separate from state dispatch. It has no inputs
and accepts only the fixed frozen manifest at
`refs/tags/confirmatory-freeze-c1`, where C1 is the direct manifest-and-lock-only
child of C0. Its retained Sigstore bundle and GitHub verification record become
part of the prospective registry deposit before sequence zero can be opened.
The manifest attestation's verified Rekor time determines the canonical Zenodo
registry record. A second attestation pins those record bytes and their first
bundle dependency before publication.

## Independence claim

This protocol produces externally timestamped process evidence. If the same
administrator controls the repository, signing workflow, state service, and
verifier policy, it does not establish organizational independence. A stronger
claim requires separate administrative control of the state service and its
verification keys, fixed before `OPENED`.
