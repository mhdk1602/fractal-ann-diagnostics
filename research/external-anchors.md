# Exact external anchors

This note specifies the production file boundary between public services and the sealed runner. An
external page, DOI, log entry, or local receipt is not enough. Protocol admission requires the fixed
C1 package, two GitHub attestations, and all 27 public Zenodo files. Prediction-completion admission
uses the narrower byte-exact HTTPS record check described below.

## Protocol registration

Finish every manifest pin before registration. The registered manifest must have
`status="frozen"`, `protocol_version="0.3.0"`, no freeze blockers, immutable revisions and
SHA-256 values for all declared artifacts, a full source commit, an OCI digest, and the exact
runner and storage configuration.

The production record is reserved in
[`zenodo-reservation.json`](zenodo-reservation.json): Zenodo record `21361837`, reserved DOI
`10.5281/zenodo.21361837`. Reservation is not registration. The draft remains unpublished and
mutable until C1 exists, so neither the DOI nor the reservation timestamp may authorize a run.

The supported publication sequence is:

1. publish and verify the two-asset immutable C0 GitHub evidence release, then place its complete
   `fractal-c0-evidence-release-binding-v2` object in
   `sealed_execution.c0_evidence_release`. Its closed apparatus subrecord names bootstrap commit
   P, C0 commit A, build-context tree T, both promoted OCI indexes, the candidate closure, the
   promotion receipt, the normalized provider-plan rehearsal closure, and the post-A
   production-control instantiation receipt. Its exact apparatus field is
   `production_control_instantiation_receipt_file_sha256`. It also binds the authenticated live
   environment readback as `github_environment_control_receipt_file_sha256`. The release archive is
   not a Zenodo package member;
2. finish and tag the C1 commit whose only changed paths are the frozen manifest, its lock, and the
   consumed candidate-to-C1 transition receipt;
3. run the zero-input C1 GitHub workflow at `confirmatory-freeze-c1`; it freshly verifies the
   immutable C0 release and retains `c0-public-verification.json`, signs that receipt's digest in
   the manifest predicate, verifies the first bundle, uses its Rekor integrated time to create the
   canonical registry record, and separately attests the registry-record bytes;
4. download the retained package and require the exact identity
   `zenodo-record:21361837;zenodo-doi:10.5281/zenodo.21361837`, record ID `21361837`, and URI
   `https://zenodo.org/api/records/21361837/files/protocol-registry-record.json/content`;
5. upload the registry record, frozen manifest and lock, C1 Git objects, two signed predicates, two
   Sigstore bundles, verification receipts, and package checksums to that reserved draft;
6. check the draft file inventory and metadata through the authenticated API, then publish it;
7. require the public API to report record `21361837` as submitted with the reserved DOI;
8. download every deposited file through its unauthenticated direct-content URI and require exact
   byte and SHA-256 equality with the closed package; and
9. create the local registration receipt from those downloaded bytes.

The production client implements that boundary. It accepts one directory containing exactly the
27 files emitted by the C1 workflow, including canonical `SHA256SUMS`,
`manifest-transition-receipt.json`, and `c0-public-verification.json`. Validation recomputes every
SHA-256, the frozen manifest digest and lock, both in-toto statements, both Rekor observations,
both typed GitHub verification results, the two workflow receipts, the registry record, the C0/C1
parent relation, the C1 tag object, the fresh C0 release readback, and the retained GitHub pointers.
A missing file, extra file, symlink, hard link, or rewritten receipt is a hard failure.

First, validate the downloaded package without network access:

```bash
python -m fractal_ann_diagnostics.zenodo_publication validate \
  --package /controlled/confirmatory-c1-registration
```

That offline check proves closed-schema and cross-file consistency. Before any authenticated
Zenodo request, run the cryptographic preflight. It asks `gh attestation verify` to verify both
retained bundles under the exact C1 workflow identity, C1 commit, tag ref, GitHub-hosted runner
restriction, OIDC issuer, and predicate types:

```bash
python -m fractal_ann_diagnostics.zenodo_publication preflight \
  --package /controlled/confirmatory-c1-registration
```

`stage`, `publish`, and `verify-public` repeat this preflight automatically. The C1 GitHub workflow
also runs it on the assembled directory before artifact retention. An internally consistent local
replacement therefore cannot cross the publication client unless both GitHub attestations still
verify for its exact manifest and registry-record bytes.

The directory is opened once with a no-follow descriptor, and all 27 members are read relative to
that descriptor. Provider verification runs on private snapshots made from those admitted bytes,
not on paths that can be replaced between offline validation and `gh`. The client then rereads the
closed directory and rejects any change before it returns from preflight. A verifier that mutates
its own subject or bundle snapshot also fails.

The C0 evidence archive, checksum asset, and post-publication binding artifact are deliberately not
accepted by this client. Their release identity is inside the frozen `study-manifest.json`, while
`c0-public-verification.json` records the later fresh public observation. Adding any excluded C0
artifact to the directory creates a 28th member and fails the closed inventory before upload.

Staging is resumable but not permissive. Before any file PUT, the client sends one authenticated
metadata PUT containing the fixed protocol title, description, ordered keywords, open CC BY 4.0
license, `publication-other` type, sole creator `mhdk1602`, ORCID `0009-0003-1036-9477`, and fixed
publication date `2026-07-14`. The creator object omits affiliation and the metadata forbids
contributors. The client verifies the PUT response, makes a separate authenticated GET, and requires
the exact metadata plus reserved-DOI object on both reads. A mismatch stops before the first file
write. Repository software-release metadata in `.zenodo.json` remains a separate record contract;
the client does not merge it into the protocol deposit.

The client then admits either an empty draft or a subset whose names, sizes, and MD5 values equal
the local package. It uploads only missing files, fetches the draft again, requires the full 27-file
inventory, and downloads every unpublished file through the authenticated bucket for byte and
SHA-256 equality. MD5 inventory fields alone do not authorize publication. It also requires record
`21361837` to remain unsubmitted with the reserved DOI and the exact metadata described above. The
repository's separate software-release record remains MIT licensed; the C1 protocol deposit is a
publication artifact.

The access token has no command-line or URL form. Pipe it from a password manager or pass an
already-open descriptor. The producer must write only the token and an optional final LF:

```bash
token-command | python -m fractal_ann_diagnostics.zenodo_publication stage \
  --package /controlled/confirmatory-c1-registration \
  --token-fd 0
```

Publication is a separate command. It will not upload or repair anything. It first performs the
anonymous full-record and 27-file byte check. An exact public record returns success without an
authenticated request. Only an exact public HTTP 404 permits the client to inspect the closed draft
and issue one publish POST; a 403, 5xx response, metadata mismatch, inventory mismatch, or changed
public byte stops without POST. The record number must be typed explicitly:

```bash
token-command | python -m fractal_ann_diagnostics.zenodo_publication publish \
  --package /controlled/confirmatory-c1-registration \
  --token-fd 0 \
  --confirm-record 21361837
```

After the POST attempt, the command polls only bounded integration statuses for at most five
minutes. Polling still occurs when the POST response was lost or could not be authenticated. Each
poll requires the exact public record and all 27 public file bytes. One invocation never sends a
second POST; a later invocation first repeats anonymous verification and therefore treats a
successfully published record as finished. The verifier requires ID `21361837`, DOI
`10.5281/zenodo.21361837`, submitted and published state, the same metadata and file inventory, and
byte-for-byte plus SHA-256 equality for every deposited file. The same check can be repeated without
a token:

```bash
python -m fractal_ann_diagnostics.zenodo_publication verify-public \
  --package /controlled/confirmatory-c1-registration
```

The client uses the authorization header prescribed by the [Zenodo REST API](https://developers.zenodo.org/),
certificate and hostname validation, TLS 1.2 or newer, identity response encoding, bounded bodies,
and no redirects. It permits only query-free URLs at `https://zenodo.org`, the fixed record paths,
and the UUID bucket returned by that draft. Tokens are read only from stdin or `--token-fd` and are
absent from command arguments, URLs, object representations, and emitted errors. The client
overwrites the mutable token buffer it owns when the request boundary closes. It cannot prove
erasure of transient immutable copies made by the Python runtime, TLS stack, or operating system;
the process must therefore be short-lived and run on the controlled publication host.

Zenodo supplies the prospective public protocol deposit and predictable direct-content URLs. The
first C1 Rekor entry supplies the record's observer time. The second pins the exact registry-record
bytes, so a later Zenodo file correction cannot erase that publicly logged digest. Anonymous
full-package verification detects any later change to the other 24 files against the controlled
local package. This matters because
Zenodo's [published-file policy](https://help.zenodo.org/docs/deposit/manage-files/#modify-files-after-publication)
permits minor file corrections during its post-publication window. The verifier rejects a host
alias, another record ID, credentials, ports, query strings, fragments, and redirects. A
transparency-log API response does not replace the exact record consumed by the runner.

All paths passed to the commands below must be absolute. Writers use exclusive creation and reject
existing files, symlinks, hard links, unsafe parent directories, and noncanonical records.

Do not construct the C1 registry record with free-form CLI values. Use the record from the fixed
workflow package. After publication, download those bytes from the direct-content URI and create
the local receipt:

```bash
python -m fractal_ann_diagnostics.external_anchors \
  write-protocol-registration-receipt \
  --manifest /controlled/research/study-manifest.json \
  --registry-record /controlled/protocol-registry-record.json \
  --output /controlled/protocol-registration.json
```

The registry record and local receipt are closed schemas. Both use compact, key-sorted JSON and
exactly one terminal newline. The record digest includes that newline.

Those two local objects do not authorize production by themselves. `begin-sealed-run` also requires
`--registration-package /controlled/confirmatory-c1-registration`. The CLI first mints an admission
capability by revalidating the closed package, both retained GitHub attestations, the fixed Zenodo
identity and record URI, and every anonymous public file. The run opener invokes that verifier again
immediately before it creates the one-shot receipt. No free-form HTTPS registry URI reaches the
production run-opening API.

## Prediction completion

The existing `PredictionCompletionReceipt` fixes the prediction artifact, action panel, corpus,
sealed run, execution artifact, row count, anchor identity, URI, and UTC time. The external anchor
layer adds two files:

- `PredictionCompletionAnchorRecord`, the exact object published by the independent service; and
- `PredictionCompletionAnchorReceipt`, the local custody object that binds the external record's
  exact digest to the prediction completion receipt.

Create the external record after the online prediction and action-panel artifacts are final:

```bash
python -m fractal_ann_diagnostics.external_anchors \
  write-prediction-completion-anchor-record \
  --completion-receipt /controlled/prediction-completion.json \
  --output /controlled/prediction-completion-anchor-record.json
```

Publish those exact bytes at the `external_anchor_uri` already present in the completion receipt.
Then create the local anchor receipt:

```bash
python -m fractal_ann_diagnostics.external_anchors \
  write-prediction-completion-anchor-receipt \
  --anchor-record /controlled/prediction-completion-anchor-record.json \
  --output /controlled/prediction-completion-anchor-receipt.json
```

Immediately before label release, run the production verifier:

```bash
python -m fractal_ann_diagnostics.external_anchors \
  verify-prediction-completion-anchor \
  --completion-receipt /controlled/prediction-completion.json \
  --anchor-record /controlled/prediction-completion-anchor-record.json \
  --anchor-receipt /controlled/prediction-completion-anchor-receipt.json
```

The verifier performs one HTTPS GET with certificate and hostname checks, requests identity
encoding, refuses redirects or a changed response URL, rejects malformed or excessive
`Content-Length`, caps the body at 64 KiB, and requires both SHA-256 and byte equality. The Python
API has a callable fetch seam for deterministic tests and controlled integrations. Injecting that
callable makes its transport authentication part of the trusted computing base.

The verified token returned by `verify_prediction_completion_anchor` is the object to admit at the
label-join boundary. A completion receipt, anchor record, anchor receipt, or remote page in
isolation does not satisfy that boundary.
