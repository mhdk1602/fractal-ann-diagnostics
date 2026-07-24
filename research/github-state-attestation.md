# GitHub state attestation for the sealed suite

This adapter turns each local suite-state record into three independently
checkable facts:

1. The exact bytes appeared in the sole manifest-derived Git ref as one
   fast-forward state transition.
2. A workflow at the registered C0 commit signed those bytes on a GitHub-hosted
   runner.
3. Sigstore Public Good recorded the signature and observer time in Rekor.

The implementation is
`fractal_ann_diagnostics.github_state_attestation.GitHubSuiteEvidenceVerifier`.
The signing workflow is
`.github/workflows/confirmatory-state-attestation.yml`.

## Closed ledger layout

For suite attempt ID `<A>`, the sole external state key is:

```text
refs/heads/confirmatory-ledger/<A>
```

The branch is an orphan, single-parent Git history. Sequence `n` has exactly
these blobs and no others:

```text
suite-attempts/<A>/000.state.json
...
suite-attempts/<A>/<n>.state.json
```

Every commit adds one canonical state file. Earlier blob object IDs must remain
unchanged. A commit message has one admitted form:

```text
confirmatory-state <A> <NNN> <STATE> <state-record-sha256>
```

The current branch must be protected by an active ruleset with `deletion`,
`non_fast_forward`, and `required_linear_history` rules. Configure that ruleset
without bypass actors. The verifier reads the ref, applied rule types and IDs,
commit ancestry, trees, and blobs through GitHub's APIs. It rejects merge
commits, gaps, rewritten blobs, unexpected tree entries, noncanonical state
JSON, a stale local chain, and any ref not derived from the frozen manifest.

The transition publisher must update the ref with `force=false`. Two commits
built from the same predecessor cannot both become a fast-forward tip: after the
first succeeds, the second is no longer a descendant of the live ref. For
genesis, `POST /git/refs` supplies the create-if-absent operation. GitHub
documents these operations in the [Git references REST
API](https://docs.github.com/en/rest/git/refs). The [rules REST
API](https://docs.github.com/en/rest/repos/rules#get-rules-for-a-branch) exposes
the active controls to the read-only workflow token.

The typed publisher is
`fractal_ann_diagnostics.github_state_attestation.publish_ledger_transition`.
It derives the ref and state path from the latest canonical local record. It
does not accept a repository, ref, branch, path, parent, commit message, author,
committer, or force flag from the caller. It creates the blob, prefix tree, and
single-parent commit through the Git Database API, updates only the derived ref,
then reloads the remote chain and compares every state byte. A retry after a
successful remote write is read-only and reproduces the same publication
receipt.

Before creating the genesis ref, install one active repository ruleset whose
include condition matches `refs/heads/confirmatory-ledger/*`. It needs the
`deletion`, `non_fast_forward`, and `required_linear_history` rules and no bypass
actors. The workflow-dispatch schema has exactly one input, `ledger_commit`, a
full lowercase 40-character Git object ID. No token with repository-write or
administration permission is supplied to the signing job.

Install or verify that fixed ruleset with the authenticated `mhdk1602` account:

```bash
python -m fractal_ann_diagnostics.github_state_attestation \
  install-ledger-ruleset
```

The command refuses a second ruleset with the fixed name, inactive enforcement,
another ref condition, an extra or missing rule, or any bypass actor. The
publisher and the later evidence verifier use an administrator-readable detail
response to recheck the empty bypass list. The read-only signing job cannot see
that field under GitHub's ruleset API, so it checks the fixed name, target,
enforcement, condition, rules, and applied ruleset ID. The applied rules
reported for the live branch must cite that exact ID.

The C0 and C1 refs have a separate repository ruleset. Install it before either
annotated tag is created:

```bash
python -m fractal_ann_diagnostics.github_state_attestation \
  install-freeze-tag-ruleset
```

This ruleset targets `tag`, includes only
`refs/tags/confirmatory-apparatus-c0` and
`refs/tags/confirmatory-freeze-c1`, and has the `deletion` and
`non_fast_forward` rules. Its bypass list is empty. The command accepts an
existing ruleset only when the name, target, enforcement, exact ordered ref
set, rule set, and empty bypass list all match. It neither protects unrelated
tags nor grants an administrator exception. GitHub documents `tag` as a
repository-ruleset target and defines the deletion rule as requiring bypass
permission to delete a matching ref. With no bypass actor, the protected
freeze refs cannot be deleted through the admitted GitHub path.

## Why the workflow is not a signing oracle

The dispatch surface accepts one value: a full Git commit object ID. The job has
`contents: read`, never `contents: write`. Before requesting an OIDC signing
certificate, the C0 code checks all of the following:

- repository `mhdk1602/fractal-ann-diagnostics`;
- ref `refs/tags/confirmatory-apparatus-c0`;
- `github.sha == github.workflow_sha`;
- exact `github.workflow_ref` for the state-attestation workflow at that tag;
- caller-selected commit equals the current protected ledger tip;
- closed, append-only ledger tree and canonical state-machine transition;
- manifest-derived branch and path;
- the three required active branch rules and GitHub's protected-branch marker.

Only outputs produced after those checks reach `actions/attest`: the computed
state-record SHA-256, its manifest-derived subject name, and a canonical ledger
predicate. Caller-supplied bytes, paths, digests, predicates, refs, and signing
identities have no route to the signing step.

Both attestation workflows grant `id-token: write`, `attestations: write`, and
`artifact-metadata: write`, as specified by the
[`actions/attest` v4 interface](https://github.com/actions/attest). Repository
contents remain read-only. `create-storage-record: false` is explicit because
these file subjects are retained as evidence packages rather than published
registry images. The permission remains declared because it is part of the v4
action's documented permission set.

The custom predicate records the observed commit, predecessor, tree, state key,
state sequence, state digest, and normalized branch controls. The predicate is
secondary evidence. Verification still passes the exact state file to `gh
attestation verify`, so a truthful predicate cannot compensate for a different
subject. GitHub's [artifact-attestation documentation](https://docs.github.com/en/actions/concepts/security/artifact-attestations)
explains the certificate identity and Sigstore trust model. The [CLI
manual](https://cli.github.com/manual/gh_attestation_verify) defines the policy
flags used here.

## Exact verification policy

`GitHubSuiteEvidenceVerifier` invokes `gh attestation verify` with the retained
bundle and all of these controls:

```text
--repo mhdk1602/fractal-ann-diagnostics
--cert-identity https://github.com/mhdk1602/fractal-ann-diagnostics/.github/workflows/confirmatory-state-attestation.yml@refs/tags/confirmatory-apparatus-c0
--cert-oidc-issuer https://token.actions.githubusercontent.com
--signer-digest <registered-C0-commit>
--source-digest <registered-C0-commit>
--source-ref refs/tags/confirmatory-apparatus-c0
--deny-self-hosted-runners
--predicate-type https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/confirmatory-state/v1
--format json
```

After `gh` verifies the signature and trust chain, the adapter parses the signed
DSSE statement and retained Sigstore v0.3 bundle. It requires one subject, one
Rekor entry, the exact live-ledger predicate, and the exact state digest. The
descriptor pins the Rekor log key ID as SHA-256 for both inclusion and observer
time. The provider claims returned to `verify_suite_state` come from the live
ledger and verified bundle, never from the untrusted evidence JSON.

GitHub stores the attestation and writes public-repository attestations to the
Sigstore Public Good transparency log. The workflow also uploads the exact
bundle, state, evidence JSON, predicate, validation receipt, attestation ID and
URL, workflow-run URL, and checksums for 90 days. Download that package
immediately and attach it to the externally registered protocol record. An
Actions artifact is a transfer channel, not permanent custody. The [Sigstore
bundle specification](https://github.com/sigstore/protobuf-specs/blob/main/protos/sigstore_bundle.proto)
defines the retained evidence format.

## Operational order

The sequence for each state is fixed. The repository ruleset described above
must already be active before sequence zero:

1. Write the local canonical state file with the apparatus.
2. Publish the transition and retain its canonical receipt:

   ```bash
   python -m fractal_ann_diagnostics.github_state_attestation \
     publish-transition \
     --namespace /controlled/fractal-v0.3/suite-attempt-<A> \
     --receipt /controlled/fractal-v0.3/suite-attempt-<A>/<NNN>.ledger-publication.json
   ```

   The command creates the Git blob, next prefix tree, and `mhdk1602`-authored
   single-parent commit. Sequence zero uses create-if-absent. Later sequences
   use one `force=false` update whose parent must be the observed live tip.
3. Read back the protected branch. The publication receipt records the blob,
   tree, commit, predecessor, state path, record digest, and ruleset ID.
4. Confirm the active `deletion`, `non_fast_forward`, and
   `required_linear_history` rules on the manifest-derived branch, with no
   configured bypass actor.
5. Dispatch the workflow at the registered C0 tag:

   ```bash
   gh workflow run confirmatory-state-attestation.yml \
     --ref confirmatory-apparatus-c0 \
     -f ledger_commit=<new-ledger-commit>
   ```

6. Wait for the exact run, download its evidence artifact, and place
   `<NNN>.attestation.json` and `<NNN>.sigstore.bundle.json` beside the local
   state record.
7. Reconstruct the local and remote chains with
   `GitHubSuiteEvidenceVerifier(namespace)` before issuing the typed state token.
8. Archive the whole evidence package with the external registration record.

No online execution, label release, or confirmatory analysis may start from a
state whose provider package has not passed step 7.

## Prospective protocol registration at C1

Before creating C1, publish the production C0 evidence package through
`.github/workflows/confirmatory-c0-evidence-release.yml`. Repository immutable releases must
already be enabled. The workflow drafts the existing `confirmatory-apparatus-c0` tag release,
attaches the deterministic evidence archive and checksum, publishes once, verifies GitHub's
release and asset attestations, and performs anonymous readback. Copy the resulting closed binding
object into `sealed_execution.c0_evidence_release`; the frozen manifest validator checks its C0
commit, URLs, sizes, asset hashes, immutable flag, embedded verification receipt, and receipt
digest. Do not add the archive or binding artifact to the Zenodo directory. The later C1 workflow
must independently repeat the public API, tag, attestation, and anonymous-byte checks and retain
their closed result as `c0-public-verification.json`.

Registration precedes `OPENED`. The C1 registration deposit must contain the
exact frozen study manifest, lock, and candidate-to-C1 transition receipt, the full C1 Git commit,
the fixed ref
`refs/tags/confirmatory-freeze-c1`, and two retained GitHub Sigstore bundles.
The first subject is the manifest; the second is the canonical protocol
registry record derived from the first bundle's verified observer time. The tag
must resolve to the cited commit, and both attestation certificates must cite
the same source ref and source digest. Each bundle supplies a Rekor entry, log
index, observer time, signer workflow, and subject digest. A repository URL or
tag page alone is not a registration deposit.

The operational route is:

1. Finish every development artifact and write the canonical frozen C1 manifest, lock, and the
   transition receipt emitted from the closed candidate package.
2. Create one direct child of the C0 commit. Its changed-path set must be exactly
   `research/study-manifest.json`, `research/study-manifest.sha256`, and
   `research/manifest-transition-receipt.json`. The lock
   contains the semantic manifest SHA-256 followed by one LF byte. Apply the
   fixed `confirmatory-freeze-c1` tag and record both its commit ID and tag object
   ID. Admission also requires an otherwise clean worktree. The manifest, lock,
   and fixed Zenodo reservation must each hash to the blob named by the C1 tree;
   untracked files and post-checkout edits are rejected before a predicate is
   written. Both raw Git identity headers must be
   `mhdk1602 <mhdk1602@users.noreply.github.com>`, and a `Co-authored-by` trailer
   is forbidden. If the C1 tag is annotated, its raw tagger header must use that
   same identity.
3. Dispatch `.github/workflows/confirmatory-registration-attestation.yml` at
   that exact tag. The dispatch has zero inputs:

   ```bash
   gh workflow run confirmatory-registration-attestation.yml \
     --ref confirmatory-freeze-c1
   ```

   The workflow rejects any other repository, ref, source commit, workflow
   identity, parent, or changed-path set. Before it writes the first predicate, it runs the pinned
   GitHub CLI 2.96.0 against the public C0 release, anonymously downloads both assets, and creates
   `c0-public-verification.json`. The predicate signs that receipt's SHA-256, release tag, C0
   target, binding digest, path, and schema. It also requires the canonical
   `research/zenodo-reservation.json` for record `21361837`, DOI
   `10.5281/zenodo.21361837`, and the exact `https://zenodo.org` direct-content
   URI. Its first attestation subject is the fixed checkout path
   `research/study-manifest.json`. No dispatch value can select artifact bytes,
   a path, a digest, a timestamp, a registry identity, or a predicate.
4. The workflow verifies the first bundle under the exact C1 identity before it
   uses the Rekor integrated time as `registered_at_utc`. It then writes the
   canonical `protocol-registry-record.json` for the fixed Zenodo identity and
   issues a second Sigstore/Rekor attestation over those record bytes. The
   second signed predicate binds the record digest to the first bundle digest,
   first Rekor entry and manifest digest. This two-entry order avoids a
   timestamp circularity: the manifest entry supplies time; the later record
   entry pins the object containing that time. The retained package requires
   distinct GitHub attestation IDs, Rekor entry IDs, and Rekor signed-timestamp
   digests. Equal UTC seconds are permitted because Rekor timestamps have
   second-level resolution; distinct entries still prove two log admissions.
5. Download both Sigstore bundles immediately. Verify each subject digest,
   repository, workflow identity, source ref, source digest, hosted-runner
   claim, Rekor entry, and observer time with `gh attestation verify`.

   ```bash
   gh attestation verify research/study-manifest.json \
     --bundle study-manifest.sigstore.bundle.json \
     --hostname github.com \
     --repo mhdk1602/fractal-ann-diagnostics \
     --cert-identity https://github.com/mhdk1602/fractal-ann-diagnostics/.github/workflows/confirmatory-registration-attestation.yml@refs/tags/confirmatory-freeze-c1 \
     --cert-oidc-issuer https://token.actions.githubusercontent.com \
     --signer-digest <C1-commit> \
     --source-digest <C1-commit> \
     --source-ref refs/tags/confirmatory-freeze-c1 \
     --deny-self-hosted-runners \
     --predicate-type https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/prospective-c1-registration/v2 \
     --format json
   ```

   Repeat the same command for `protocol-registry-record.json`, changing the
   bundle to `protocol-registry-record.sigstore.bundle.json` and the predicate
   type to
   `https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/prospective-c1-registry-record/v2`.
6. Download the `confirmatory-c1-registration-<C1>` artifact. It contains the
   manifest, lock, reservation, canonical registry record, tag and commit
   records, the fresh C0 public-verification receipt, two bundles, two signed predicates,
   validation receipts, two `gh` verification results, attestation IDs and URLs, workflow-run URL,
   verifier version, and `SHA256SUMS`. Deposit those exact 27 files in the already reserved
   Zenodo record `21361837` (`10.5281/zenodo.21361837`). Publish the record and
   verify its unauthenticated direct-content bytes. The public Zenodo record and
   both C1 Rekor entries must precede `OPENED`.
7. Put the registry's immutable record URI and exact record SHA-256 in
   `ProtocolRegistrationReceipt`. Re-fetch those bytes before the sealed run.

The registration package must also retain this machine-readable field. It is
present in both signed registration predicates, every signed state predicate,
and the workflow receipts:

```json
{
  "claim": "github-process-evidence-under-common-administration",
  "independent_organizational_custody": false,
  "same_administrator_controls": [
    "repository",
    "branch-protection",
    "workflow-dispatch",
    "evidence-retention",
    "verifier-policy"
  ]
}
```

The external registry provides prospective public registration. The GitHub
tag, commit, and Sigstore bundle provide byte identity, signer identity, and
public timing. They do not convert same-administrator GitHub control into
independent custody.

## Claim boundary

This path supplies public timestamping, exact C0 signer identity, current
append-only Git ancestry, and a provider-backed state key. It does not supply
independent administration when `mhdk1602` controls the repository, branch
rules, workflow dispatch, evidence download, and verifier policy. The same
administrator can disable branch rules or delete a ref after an attestation.
The verifier detects the resulting live-state mismatch, while Rekor preserves
evidence that a signature existed; neither fact proves that the administrator
was unable to create another history.

A claim of organizationally independent custody requires a state service under
another administrator, or an organization ruleset whose bypass authority is
held by an independent custodian. This GitHub path is strong process evidence
under disclosed common control. It is not proof of administrative separation.
