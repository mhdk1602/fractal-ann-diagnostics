# Future-Quicknet design-seed commitment

The development design seed must not be chosen by an operator, copied from an
earlier rehearsal, or selected after development labels are visible. This
host-side control commits the exact label-free development scope, admits a
GitHub/Sigstore timestamp for that commitment, and derives the seed from the
first eligible Quicknet beacon.

The operator is `operators/design_seed_commitment.py`; the attester is
`.github/workflows/design-seed-commitment.yml`. Both are host controls outside
the exact-P confirmatory image context. Every public operator entry point proves
that the loaded `src/fractal_ann_diagnostics` tree is byte-identical to source
commit `9061f09777b1af2346eebe3fb1ae21e6325cdf75`, tree
`33e7aa05527042bdba301310c62eb3dbaffde941`, before it uses the Quicknet or
Sigstore parser. A changed, untracked, or differently imported package tree is
a stop condition.

## Control sequence

There are four closed, canonical JSON schemas.

1. `fractal-design-seed-commitment-request-v1` binds the full staged inventory,
   query-partition audit file, phase-one label-free view receipt, and standalone
   selection receipt. It also pins the attesting workflow path, commit, and Git
   ref, source P, and its tree. `scope_sha256` is recomputed from the four scientific digests with
   domain-separated, unsigned-64-bit length framing; the request digest binds
   the signer pins as a separate control.
2. `fractal-design-seed-commitment-v1` copies that scope and fixes the exact-P
   Quicknet chain, the 900-second minimum lead, the target-round rule, and the
   seed derivation. It contains neither a round nor a seed.
3. `fractal-design-seed-attestation-admission-v1` admits one Sigstore bundle.
   Its signed custom predicate binds the commitment digest, scope, source P,
   first workflow attempt, immutable apparatus tag, repository, actor, and the
   scope-specific immutable release. Admission independently reads the public
   Actions run, release, and release-tag APIs. Their closed canonical
   projections and digests remain in the admission. The Rekor integrated time
   in the verified bundle selects the round.
4. `fractal-design-seed-reveal-v1` retains the exact beacon bytes, their digest,
   the verified signature and randomness, an absolute admission path and file
   digest, and the derived seed. Every local control is published through an
   atomic no-replace link, so a final pathname never exposes partial bytes.

The target round is

```text
deadline = rekor_integrated_time + 900
target_round = first Quicknet round whose publication time is >= deadline
```

Quicknet publishes every three seconds. The resulting lead is therefore 900,
901, or 902 seconds. A caller cannot supply a round through either the Python
API or CLI.

The final seed hashes a domain plus these named components, each encoded as
`u64be(length) || bytes`:

```text
scope_sha256
commitment_sha256
attestation_admission_sha256
target_round
quicknet_beacon_sha256
quicknet_randomness
quicknet_signature
```

There is no seed override. Changing a label-free input, the attestation, the
round, or any beacon byte changes the seed or causes verification to fail.

## Signed attestation predicate

The commitment must be the sole subject of a GitHub custom attestation with
predicate type:

```text
https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/design-seed-commitment/v1
```

The predicate has exactly these fields:

```json
{
  "actor": "mhdk1602",
  "commitment_sha256": "<commitment file SHA-256>",
  "event": "workflow_dispatch",
  "git_ref": "refs/tags/design-seed-apparatus-v1",
  "release_id": 987654321,
  "release_name": "design-seed-scope-<scope SHA-256>",
  "release_published_at_utc": "<GitHub UTC timestamp>",
  "release_tag": "design-seed-scope-<scope SHA-256>",
  "repository": "mhdk1602/fractal-ann-diagnostics",
  "run_attempt": 1,
  "run_id": 123456789,
  "schema_version": "fractal-design-seed-attestation-predicate-v1",
  "scope_sha256": "<scope SHA-256>",
  "source_p": "9061f09777b1af2346eebe3fb1ae21e6325cdf75",
  "source_tree": "33e7aa05527042bdba301310c62eb3dbaffde941",
  "triggering_actor": "mhdk1602",
  "workflow": ".github/workflows/design-seed-commitment.yml",
  "workflow_ref": "mhdk1602/fractal-ann-diagnostics/.github/workflows/design-seed-commitment.yml@refs/tags/design-seed-apparatus-v1",
  "workflow_sha": "<40-character workflow commit>"
}
```

The workflow constructs identity and run values from GitHub context, not from
dispatch inputs. Before it signs, it publishes an assetless release whose tag
and name are `design-seed-scope-${scope_sha256}` and whose target is the pinned
workflow commit. Repository release immutability must be enabled. The workflow
polls until the API reports `immutable=true`, verifies the generated lightweight
tag target, and only then invokes `actions/attest`.

Admission asks `gh attestation verify --bundle` to check the bundle under the
exact repository, workflow certificate identity, OIDC issuer, signer digest,
source digest, source ref, custom predicate type, and GitHub-hosted-runner
constraint. It removes GitHub tokens and points `gh` at an empty temporary
configuration directory; this check cannot start a device or PAT flow.

The signed predicate is not accepted as proof of its own run metadata. A
bounded anonymous reader connects only to `https://api.github.com` with the
system trust store, no redirects, the fixed API version, and no credentials.
It requires one completed successful attempt-1 run at the exact apparatus tag,
both actors `mhdk1602`, the fixed workflow path and commit, an assetless
immutable release authored by `github-actions[bot]`, the exact release tag
target, and release publication between run start and Rekor integration.
Canonical projections of all three responses remain in the admission and are
re-read on verification.

The temporal gate uses `published_at`, not `created_at`. GitHub defines a
release's `created_at` as the date of the commit used for that release, so it
normally precedes the workflow that publishes the release
([REST release semantics](https://docs.github.com/en/rest/releases/releases#get-the-latest-release)).
The admitted release projection omits that field rather than misrepresenting it
as provider evidence of release creation.

This ordering matters. The immutable scope release exists before attestation,
and the attestation's Rekor time exists before the target beacon. The future
beacon could not have informed selection.

## One-scope remote burn

A local marker cannot exclude a second attempt made from another directory or
host. `run_attempt == 1` is also insufficient because two distinct workflow
runs can both have attempt number one.

The scope-derived immutable release is the global burn primitive. Every
dispatch executes the same unconditional release-creation request. Two runs for
one scope race on one tag; at most one can publish it. A later failure leaves
the immutable release and tag in place, so another run stops before
attestation. The workflow has no cleanup, recovery, resume, or overwrite path.
The workflow token cannot read the repository's administration-only immutable
release setting, so it makes no such preflight call. Instead it publishes and
polls the public release record until `immutable=true`; failure to reach that
state stops before attestation. GitHub then locks the tag and target even if
the release record is later removed; GitHub documents that the same tag name
cannot be reused after deletion of an immutable release ([immutable release
semantics](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases)).
A failed scope requires a new registered protocol version, not a second random
draw. The settings endpoint is omitted because it requires repository
Administration read permission ([REST permission
contract](https://docs.github.com/en/enterprise-cloud@latest/rest/repos/repos#check-if-immutable-releases-are-enabled-for-a-repository)).

## Ceremony

First merge `.github/workflows/design-seed-commitment.yml` to the default
branch, then create immutable apparatus tag `design-seed-apparatus-v1` at that
exact commit. GitHub requires a `workflow_dispatch` workflow file to exist on
the default branch even when the dispatch ref is an immutable tag ([manual-run
contract](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow?tool=cli)).
Record that tag target as `ATTESTATION_WORKFLOW_SHA`.

Use one mode-`0700` local control directory. The commitment command writes a
scope-specific `LOCAL_DEFENSE_ONLY` marker and then the commitment through
atomic no-replace publication. This catches an accidental repeat in the same
directory; it is not the cross-host burn authority. The immutable GitHub
release created by the fixed workflow is that authority.

```bash
CONTROL_ROOT='/absolute/controlled/design-seed-v0.3'
mkdir -m 0700 "$CONTROL_ROOT"

PYTHONPATH=src .venv/bin/python operators/design_seed_commitment.py build-request \
  --staged-inventory-sha256 "$STAGED_INVENTORY_SHA256" \
  --partition-audit-file-sha256 "$PARTITION_AUDIT_FILE_SHA256" \
  --phase1-view-receipt-sha256 "$PHASE1_VIEW_RECEIPT_SHA256" \
  --selection-receipt-sha256 "$SELECTION_RECEIPT_SHA256" \
  --attestation-workflow '.github/workflows/design-seed-commitment.yml' \
  --attestation-workflow-sha "$ATTESTATION_WORKFLOW_SHA" \
  --attestation-git-ref 'refs/tags/design-seed-apparatus-v1' \
  --output-directory "$CONTROL_ROOT"
```

Take the emitted request path and build the one commitment:

```bash
PYTHONPATH=src .venv/bin/python operators/design_seed_commitment.py build-commitment \
  --request "$DESIGN_SEED_REQUEST" \
  --output-directory "$CONTROL_ROOT"
```

Dispatch exactly once. The commitment bytes are the only dispatch-carried
payload; the workflow derives identity, run, and release fields from GitHub.

```bash
DESIGN_SEED_COMMITMENT_BASE64="$(base64 < "$DESIGN_SEED_COMMITMENT" | tr -d '\n')"

gh workflow run design-seed-commitment.yml \
  --ref design-seed-apparatus-v1 \
  -f commitment_base64="$DESIGN_SEED_COMMITMENT_BASE64" \
  -f commitment_sha256="$DESIGN_SEED_COMMITMENT_SHA256" \
  -f scope_sha256="$DESIGN_SEED_SCOPE_SHA256"
```

Do not dispatch again after any failure. The scope-derived release or tag is a
permanent burn signal. On success, download the one
`design-seed-evidence-${DESIGN_SEED_SCOPE_SHA256}` artifact and retain its
Sigstore bundle. Host admission needs no GitHub credential: the run, release,
and tag evidence is public. Admit the bundle:

```bash
PYTHONPATH=src .venv/bin/python operators/design_seed_commitment.py admit-attestation \
  --commitment "$DESIGN_SEED_COMMITMENT" \
  --bundle "$DESIGN_SEED_SIGSTORE_BUNDLE" \
  --output-directory "$CONTROL_ROOT"
```

Read `target_round` from the admission. Fetch that exact public Quicknet
response without rewriting its bytes. Transport is not an authority: the next
command reconstructs the frozen Quicknet contract and performs the RFC 9380
BLS pairing check locally.

```bash
PYTHONPATH=src .venv/bin/python operators/design_seed_commitment.py build-reveal \
  --commitment "$DESIGN_SEED_COMMITMENT" \
  --admission "$DESIGN_SEED_ATTESTATION_ADMISSION" \
  --beacon "$EXACT_QUICKNET_BEACON" \
  --output-directory "$CONTROL_ROOT"
```

Before phase-two label custody is opened, verify the chain again:

```bash
PYTHONPATH=src .venv/bin/python operators/design_seed_commitment.py verify-commitment \
  --commitment "$DESIGN_SEED_COMMITMENT" \
  --expected-sha256 "$DESIGN_SEED_COMMITMENT_SHA256"

PYTHONPATH=src .venv/bin/python operators/design_seed_commitment.py verify-reveal \
  --commitment "$DESIGN_SEED_COMMITMENT" \
  --reveal "$DESIGN_SEED_REVEAL" \
  --expected-sha256 "$DESIGN_SEED_REVEAL_SHA256"
```

`verify-reveal` securely reopens the absolute admission path stored in the
reveal, checks its file digest, re-verifies the embedded Sigstore bundle, and
repeats the Quicknet BLS and seed derivations. Moving or replacing the
admission invalidates the chain.

## Admission and stop rules

Proceed to phase-two development labels only when all of these hold:

- the four scope pins match the phase-one and partition-audit receipts;
- source P and its package tree pass byte-identity verification;
- the local defense marker and commitment match, without treating that
  directory as a global registry;
- one assetless scope-derived release is public, immutable, and targets the
  fixed apparatus commit;
- `run_attempt` is exactly one, both actors are `mhdk1602`, and the event is
  `workflow_dispatch`;
- the public run, release, and release-tag projections match their retained
  admission copies;
- GitHub verifies exactly one attestation under the signed workflow identity;
- the attested subject is the exact commitment file;
- the first eligible Quicknet round has at least 900 seconds of Rekor-time
  lead;
- the exact-P verifier accepts the beacon's BLS signature; and
- an independent `verify-reveal` returns the same design-seed digest.

Stop without opening labels if any artifact is missing, group/other writable,
non-canonical, path-substituted, duplicated, or inconsistent. A failed or
interrupted remote burn or attestation does not authorize another run for the
same scope. Record the failure and create a new protocol version if a new draw
is scientifically necessary.

The Sigstore and Quicknet checks establish public timing, subject identity,
and unpredictable seed derivation. They do not create investigator
independence: the repository and workflow remain under the same ultimate
administrative control described in the threat model.
