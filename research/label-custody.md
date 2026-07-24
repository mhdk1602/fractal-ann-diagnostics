# Label withholding and timed-release protocol

This document specifies the machine boundary between label-bearing storage and the online process
for protocol v0.3. It defines admissible evidence. Public benchmark labels remain accessible
outside the apparatus, and one repository administrator controls the registered phases. The study
therefore claims neither human outcome blindness nor independent organizational custody.

## Confirmatory process-separation criterion

The registered apparatus treats the label boundary as satisfied only when all four statements are
true:

1. Before manifest freeze, each plaintext label file, corresponding timelock ciphertext, and exact
   encryption-operation receipt file are separately hashed. A closed custody-seal receipt binds
   those fifteen digests to one exact drand chain, one positive integer round, the timelock-tool
   digest, custody-builder digest, and ARM64 release-image digest.
2. The frozen manifest pins the exact newline-terminated custody-seal file. The public prospective
   registration pins the frozen manifest before `RUN_CLAIMED` and before any admitted online input
   is opened.
3. The online process runs in the main ARM64 scientific image with no `tle` executable, no label or
   ciphertext mount, and no network. All five output trees must close under the provider-claimed
   lineage before the label-release workflow can claim its state transition.
4. A separate ARM64 release image may mount ciphertext only after
   `LABEL_RELEASE_CLAIMED` wins the provider CAS and before the registered future drand round is
   read. Released plaintext becomes an analysis input only after `ANALYSIS_CLAIMED` wins its own
   provider transition.

These controls make premature input access or an alternate result lineage inadmissible within the
registered process. They do not show that the administrator lacked an out-of-band copy, never read
the public labels, or never ran code elsewhere. The confirmatory claim rests on prospective
registration, fixed admissibility rules, provider CAS ordering, and one registered analysis, not on
human ignorance.

## Registered custody artifacts

The study manifest reserves these custody-specific roles:

| Manifest role | Cardinality | Required pin |
|---|---:|---|
| `sealed-label-ciphertext` | five, one per corpus | exact ciphertext URI, immutable revision, SHA-256 |
| `timelock-encryption-receipt` | five, one per corpus | SHA-256 of canonical JSON plus its one terminal newline |
| `custody-seal-receipt` | one | SHA-256 of canonical JSON plus its one terminal newline |
| `tlock-release-provenance` | one | exact release lineage, archive, binary, Quicknet parameters, and absolute round |
| `timelock-tool` | one | exact executable revision and file SHA-256 |
| `custody-builder` | one | exact builder revision and SHA-256 |

The receipt records, for every fixed corpus, four distinct commitments: the online-execution
artifact, plaintext sealed-label artifact, sealed-label ciphertext, and exact newline-terminated
encryption receipt. Reusing a URI or digest across those roles is a freeze error. The inner
encryption receipt omits the manifest and suite-seal digests; the outer manifest pins that receipt
and the completed suite seal. This direction prevents a digest fixed-point cycle.

The separate `fractal-tlock-release-provenance-v1` record binds tlock `v1.2.0` to annotated tag
object `6a94bf6b8200ab67f2b80af8000a55db64998d94`, source commit
`7b54141a9733fd6fa207587a11148280e6fb020d`, and the Linux ARM64 release archive. It also binds
the extracted executable digest and byte count plus Quicknet's chain hash, scheme, period, genesis,
and public key. The checked-in pre-freeze record at
`research/tlock-release-provenance.prefreeze.json` sets `drand_round` to JSON `null`. The custody
freeze must derive a new canonical record by setting one positive absolute round exactly once. C1
then pins that file. Because the provenance record does not contain the manifest or custody-seal
digest, this additional manifest edge is acyclic.

The receipt also records:

- `drand_chain_hash`, exactly 64 lowercase hexadecimal characters;
- `drand_round`, a positive JSON integer, never a duration or moving alias;
- `timelock_tool_sha256` and `custody_builder_sha256`; and
- protocol version `0.3.0` under the closed schema
  `fractal-custody-seal-receipt-v2`.

`receipt_sha256` hashes the canonical JSON object. The manifest artifact pin hashes the exact file,
which is the same canonical JSON followed by one newline. The CLI prints both values so that a
reviewer cannot silently substitute one digest convention for the other.

## Pre-C1 encryption ceremony

The artifact factory performs this ceremony under the controlled root before C1. No provider phase
runner is active. The same administrator controls this root and the later workflows; the receipts
establish byte and phase ordering, not organizational independence.

1. Obtain the five canonical plaintext label artifacts and verify their registered corpus and
   online-execution bindings.
2. Select an absolute release round on a named drand chain. Record the chain hash and integer round.
   Do not specify `@latest`, a wall-clock duration, or a command whose computed round is not retained.
   Load the canonical pre-freeze tlock provenance record, call
   `freeze_tlock_release_provenance` with that positive round, and write the result to
   `artifacts/custody/tlock-release-provenance.json`. Run
   `verify_tlock_release_binary` inside the pinned Linux ARM64 release image. The frozen provenance
   round must equal every encryption receipt and the suite custody seal. Insert the provenance file
   digest, executable digest, and release-image digest in their separate manifest rows.
3. Build each timelock ciphertext inside that release image with the digest-pinned `tle`
   executable. The adapter hashes the binary before and after execution, reads plaintext through
   the no-link boundary, passes those bytes on stdin, accepts bounded ciphertext only on stdout,
   disables shell execution, supplies no duration, decrypt, force, or output-path flag, and creates
   the output file exclusively. It emits a closed operation receipt containing the input, output,
   binary, image, network, chain, round, argument, and byte-count commitments.
4. Put every plaintext and ciphertext in separately administered storage. A storage URI is evidence
   only after its bytes are verified against the manifest.
5. Pin each plaintext digest and the exact `custody/bin/tle` executable digest in the draft manifest.
   Generate a ciphertext for each corpus:

   ```bash
   fractal-retrieval-governance encrypt-timelock-label \
     --manifest /controlled/study-manifest.json \
     --corpus-id scifact \
     --plaintext /controlled/custody/labels/scifact.json \
     --tle-binary /controlled/custody/bin/tle \
     --drand-network https://api2.drand.sh/ \
     --drand-chain-hash <64-lowercase-hex> \
     --drand-round <positive-integer> \
     --ciphertext /controlled/custody/ciphertext/scifact-sealed-labels.tlock \
     --receipt /controlled/custody/scifact-timelock-encryption.json
   ```

   Repeat for all five corpus IDs. Insert each printed `ciphertext artifact sha256` in its
   `sealed-label-ciphertext` manifest row. Hash the exact canonical operation-receipt file, including
   its terminal newline, and insert that digest in the corpus's
   `timelock-encryption-receipt` row. A failed process or existing output path blocks creation; use a
   new reviewed path for any engineering retry before freeze.
6. Pin the custody-builder digest. Then create the closed suite receipt:

   ```bash
   fractal-retrieval-governance create-custody-seal-receipt \
     --manifest /controlled/study-manifest.json \
     --drand-chain-hash <64-lowercase-hex> \
     --drand-round <positive-integer> \
     --receipt /controlled/custody-seal-receipt.json
   ```

7. Insert the printed `manifest artifact sha256` as the `custody-seal-receipt` artifact pin. Freeze
   only after every other blocker is resolved. Then verify the final binding:

   ```bash
   fractal-retrieval-governance verify-custody-seal-receipt \
     --manifest /controlled/frozen-study-manifest.json \
     --receipt /controlled/custody-seal-receipt.json
   ```

8. Verify each operation receipt against both the final manifest and suite seal:

   ```bash
   fractal-retrieval-governance verify-timelock-encryption-receipt \
     --manifest /controlled/frozen-study-manifest.json \
     --receipt /controlled/custody/scifact-timelock-encryption.json \
     --custody-seal /controlled/custody-seal-receipt.json
   ```

`--allow-draft` exists only to inspect commitments before the receipt-file pin is inserted. Its
output is not freeze evidence.

## Full custody verification and online process admission

`verify-study-artifacts` and `begin-sealed-run` are full custodian operations. They open and hash
every manifest artifact, including plaintext labels. They must not run inside the online execution
environment.

After the full verification receipt and local sealed-run receipt exist, provision the online phase
with:

- the frozen manifest, custody seal, run receipt, and full verification receipt;
- the complete local artifact map, which may name excluded paths but does not contain their bytes;
- only artifacts whose roles appear in `ONLINE_CUSTODY_REVALIDATION_ROLES`; the label-release
  `timelock-tool` is deliberately absent from that online allowlist; and
- no plaintext `sealed-labels`, `sealed-inputs`, development data, fitted analysis model, power
  report, analysis runner, or query-partition audit file.

Run the online boundary check before retrieval:

```bash
fractal-retrieval-governance verify-online-custody \
  --manifest /online/control/frozen-study-manifest.json \
  --custody-seal-receipt /online/artifacts/custody/custody-seal-receipt.json \
  --sealed-run-receipt /online/control/<manifest-sha256>.json \
  --artifact-verification-receipt /online/control/full-artifact-verification.json \
  --artifact-root /online/artifacts \
  --artifact-map /online/control/artifact-map.json \
  --runner-identity <exact-manifest-identity> \
  --receipt /online/control/online-custody-admission.json
```

Admission checks the frozen manifest and runner identity, all 79 rows in the custodian verification
receipt, its binding in the run receipt, the custody-seal commitments and manifest pin, and fresh
digests for the online subset. It then writes a closed receipt listing every file it opened. The
filesystem test suite proves that this operation succeeds when all plaintext-label and sealed-input
files are missing; an altered admitted online artifact fails before execution.

The admission receipt does not create another run. It is evidence about the already claimed
provider lineage and must be retained with that run's prediction and action-panel records.

## Release decision

Time and event are separate predicates:

```text
time_ready       = the registered drand round is available
completion_ready = the exact prediction and action-panel bytes have a verified external anchor
score_ready      = time_ready AND completion_ready AND every custody binding still verifies
```

Timelock encryption supplies only `time_ready`. It cannot make decryption conditional on the
completion anchor. Anyone who possesses the ciphertext can decrypt after the selected round once
the required drand randomness is available. Choose a round after the planned online window and
treat these events as terminal failures of v0.3:

- the round becomes available before a valid completion anchor exists;
- the admitted online process opens plaintext labels, ciphertext, or another excluded role;
- a receipt, ciphertext, manifest, runner identity, or external record fails verification; or
- a provider-claimed phase or the registered analysis terminates without its prescribed output.

A failure is retained and reported. Re-encryption, a later round, a replacement anchor, or another
attempt requires a protocol amendment, new frozen manifest, and new external registration.

## Sealed decryption ceremony

The release command is the only admitted plaintext ingress. It first reconstructs the protected
GitHub ledger and verifies the complete, externally attested `ONLINE_COMPLETE` suite chain. It then
verifies the completion-anchor record against the external HTTPS copy, frozen manifest, custody seal, exact
encryption receipt file, ciphertext, and `tle` executable. Only then does it fetch the pinned drand
chain metadata and exact target-round response over certificate-validated HTTPS with redirects
disabled.

The chain metadata must name the frozen chain hash and an unchained scheme. The target-round response
must name the exact integer round, and its randomness must equal SHA-256 of the response signature.
The release time is computed from the authenticated genesis time, period, and round. The external
completion anchor timestamp must be strictly earlier than that release time; decryption must start
at or after it.

The self-hosted release job runs this command inside the exact ARM64 release
image after its provider claim and future-beacon check. The workflow derives every path from the
C1-fixed provider plan; the dispatch caller supplies only the suite-attempt ID.

```bash
fractal-retrieval-governance release-timelock-label \
  --manifest /controlled/frozen-study-manifest.json \
  --corpus-id scifact \
  --custody-seal /controlled/custody/custody-seal-receipt.json \
  --encryption-receipt /controlled/custody/receipts/scifact-timelock-encryption.json \
  --completion-receipt /controlled/completion/scifact-completion.json \
  --completion-anchor-record /controlled/completion/scifact-anchor-record.json \
  --completion-anchor-receipt /controlled/completion/scifact-anchor-receipt.json \
  --suite-namespace /controlled/suite-attempt-<manifest-derived-id> \
  --ciphertext /controlled/custody/ciphertext/scifact-sealed-labels.tlock \
  --tle-binary /controlled/custody/bin/tle \
  --plaintext-output /controlled/released/scifact-labels.json \
  --receipt /controlled/released/scifact-decryption-receipt.json
```

The adapter passes only the ciphertext to `tle --decrypt` on stdin. Plaintext is accepted only from
bounded stdout and is created with exclusive filesystem semantics. Existing outputs, moving-round
aliases, duration flags, output-path flags, plaintext inputs, redirects, early rounds, and any byte
mismatch are fatal. After creation, the release capability reopens and rehashes the plaintext when
the claimed analysis input operator requests it.

The canonical `fractal-timelock-decryption-receipt-v1` records both exact drand response byte strings
in base64 plus their digests, all chain timing fields, ciphertext and plaintext digests and counts,
the exact `tle` argv, start and completion times, the externally verified completion record and
receipt digests, and the sealed online-result receipt digest carried by both anchor objects.
Ciphertext and plaintext bytes remain in their separately controlled files because embedding
corpus-scale payloads in the receipt would duplicate the custody material; digest and byte count
bind the exact files.

## What the evidence proves

| Evidence | Supported statement | Unsupported statement |
|---|---|---|
| Custody-seal receipt plus manifest pin | named plaintext, ciphertext, tool, builder, chain, and round digests agreed before freeze | ciphertext was built correctly; no other plaintext copy exists |
| Drand timelock ciphertext | under its cryptographic and beacon-availability assumptions, a holder cannot decrypt through the intended route before the round | independent custody; event-conditioned release; ignorance of public labels |
| Full artifact-verification receipt | the custodian process reported exact agreement for all manifest artifacts | the custodian is independent or its host administrator behaved honestly |
| Online custody-admission receipt | the admitted process freshly opened only the listed roles and their bytes matched the custodian receipt | host-level absence, memory erasure, or lack of an out-of-band copy |
| Network-disabled runner | the admitted process has no configured egress through the tested execution boundary | a trusted host administrator cannot inspect, inject, or copy files |
| External completion anchor | a specified external record committed to the exact prediction and action-panel bytes at a stated time | retrieval correctness, computation provenance, or custodian independence |

The apparatus makes substitution and premature in-process label access detectable relative to its
anchors. Claims about human knowledge, ultimate administrative authority, and undisclosed copies
remain matters for external custody evidence and audit.

The timed-release construction follows the security model described by the
[drand timelock documentation](https://docs.drand.love/docs/timelock-encryption/) and its
[open-source tlock implementation](https://github.com/drand/tlock). Those sources describe the
time-based cryptographic mechanism; they do not certify this study's custody arrangement.
