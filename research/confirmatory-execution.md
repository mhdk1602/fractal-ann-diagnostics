# Confirmatory execution runbook

This runbook implements [protocol v0.3](preregistration.md). It does not authorize a sealed run.
The current [study manifest](study-manifest.json) is a draft, and the study has not run.

## Current state

The five corpus adapters exist, including the sealed HotpotQA FullWiki path. OPA is the primary
policy-decision-point artifact. The action runtime names, fixed-suite weights, primary endpoints,
directional family bootstrap, label-separation schema, action-panel admission receipt, and
provider-claimed no-rescue result contract are encoded in the repository.

The manifest is not freeze-ready. Artifact URIs, immutable revisions, SHA-256 values, licenses,
hardware, exact runner identity, code commit, OCI image digest, stores, and other declared fields
must be final. The H1 diagnostic artifact, H2 suite, static comparator, controller, one-sided
analysis runner, and endpoint-specific joint-gate design report must be pinned and their conformance
tests must pass. The dependence source, effect scenarios, simulation seed, selected maximum family
count, and joint-power lower bound remain `TBD`. The five timelock ciphertexts, drand chain and exact
round, timelock tool, custody builder, and custody-seal receipt also remain unpinned. Every blocker
recorded by manifest validation must be resolved before sealed labels are prepared for scoring.

Opening label-bearing artifacts inside the registered process while any condition remains
unresolved converts that lineage to an exploratory run. A later receipt cannot repair a draft
protocol.

## Custody and execution roles

- **Pre-C1 artifact factory:** fits and calibrates model artifacts, freezes the controller and
  static comparator, and supplies design-simulation inputs. Its admitted input schemas exclude the
  label-bearing artifacts; this is a process property, not a claim about the administrator's
  knowledge.
- **Label storage boundary:** prepares separate online, plaintext-label, and ciphertext artifacts,
  then seals each label file to the registered drand round inside the pinned ARM64 release image.
- **Online provider phase:** uses the main ARM64 scientific image with no `tle`, label mount, or
  network; it writes five output trees under the provider-claimed run lineage.
- **Release and analysis phases:** claim separate provider transitions. Release alone mounts
  ciphertext and uses the ARM64 release image; analysis uses the no-network AMD64 scientific image
  after the released-label transition.

The `mhdk1602` repository administrator controls every role, the self-hosted machine, and the local
storage roots. Public benchmark labels are accessible outside the apparatus. Separate jobs and
cryptographic receipts establish an admissible process order, not independent custody or human
outcome blindness. The complete evidence and failure rules are specified in
[label-custody.md](label-custody.md).

The workflows bind candidate execution jobs to `confirmatory-rehearsal`. Production image
publication, C0 evidence publication, C1 and state attestations, and the production claim and
execution jobs bind to `confirmatory`. The YAML bindings are apparatus source, not evidence that
the live repository environments have protection rules. Preserve the repository API readback
before C0/C1 freeze. With `mhdk1602` as the sole eligible administrator and reviewer, a production
environment approval is recorded self-approval and prevention of self-review must remain disabled.
That gate records a deliberate pause by the operator. It does not establish independent review,
separation of duties, or independent custody.
The [GitHub environment-control receipt](github-environment-control.md) specifies the exact offline
REST readback, response hashes, rejection rules, and administrator-bypass visibility limit.

## Freeze package

The development owner supplies the custodian with immutable artifacts for:

- separate sealed inputs, online-execution artifacts, plaintext labels, and timelock ciphertexts for
  SciFact, HotpotQA FullWiki, T2-RAGBench, BRIGHT, and MIRACL;
- one separately pinned policy-workload artifact per corpus, including its subject, environment,
  document-attribute, mutation-schedule, seed, and ordered-universe bindings;
- `normalize_scifact`, `normalize_hotpotqa_fullwiki`, `normalize_t2_ragbench`, and the pinned
  `normalize_qrels_corpus` uses;
- the primary embedding, tokenizer, pooling and normalization rule;
- the exact-authorized oracle and strict HNSW backend;
- the OPA PDP contract, policy bundle, endpoint constraints, and conformance record;
- the frozen controller and its exact `hnsw-low`, `hnsw-high`, `exact-authorized`, and `abstain`
  action mapping;
- the frozen static-comparator action;
- the H1 model artifact and four-model H2 suite;
- development-fit and development-calibration data, plus the connected query-partition audit with
  its exact and prespecified near-duplicate edges;
- the source-code commit, main scientific OCI index, ARM64 release image, analysis runner, expected
  schemas, and test results;
- one exact drand chain hash and release round, plus immutable timelock-tool, custody-builder, and
  custody-seal-receipt pins;
- corpus-family assignments, exclusions, licenses, duplicate checks, and evidence rules;
- the exact `nested_rows_per_family` trial-design cardinality;
- hardware, thread count, concurrency, warmup, timing repeats, and action-order seed; and
- the exact joint-gate endpoint order, development-data dependence source, effect scenarios,
  candidate counts, selected maximum family count, simulation seed and count, joint-power lower
  bound, and immutable design report.

Every artifact receives a non-placeholder URI, immutable revision, SHA-256, and license where the
schema requires one. The manifest must also pin the custodian, exact runner identity, approval
environment, artifact stores, receipt URI template, and freeze blockers as an empty list.

Before C1, run the [five-corpus online artifact pipeline](artifact-pipeline.md). It verifies the
fixed embedding, policy, authorized-index, and runtime sequence against the label-free staging
projection and emits the canonical suite receipt used to review those manifest pins.

Do not hand-author the production-control materialization config. Set the declared hardware values,
create one private control directory, and let the controls command derive every available digest
from the canonical factory, C0 extraction, and retained runtime files:

```bash
: "${SCIENTIFIC_CANDIDATE_REFERENCE:?set the digest-qualified candidate C0 image}"
: "${SCIENTIFIC_PRODUCTION_REFERENCE:?set the future production locator at the same digest}"
APPROVAL_ENVIRONMENT=confirmatory
RUNNER_IDENTITY="github-actions:environment:${APPROVAL_ENVIRONMENT}"
: "${HARDWARE_PROVIDER:?set the provider}"
: "${HARDWARE_INSTANCE_TYPE:?set the instance type}"
: "${HARDWARE_CPU_MODEL:?set the exact CPU model}"
: "${HARDWARE_ACCELERATOR:?set the accelerator or none}"
: "${HARDWARE_REGION:?set the region}"
: "${HARDWARE_OPERATING_SYSTEM:?set the operating-system identity}"
: "${MEMORY_LIMIT_BYTES:?set an integral GiB byte count}"
: "${CPUSET_CPUS:?set canonical sorted CPU indices, for example 0,1,2,3}"
: "${TMPFS_SIZE_BYTES:?set the tmpfs byte count}"

install -d -m 0700 /controlled/config
CONFIG_RESULT="$(fractal-production-controls write-config \
  --factory-config /controlled/production-artifact-factory.json \
  --c0-runtime-extraction-receipt /controlled/c0-runtime-extraction-receipt.json \
  --opa-binary /controlled/runtime/opa \
  --uv-lock /controlled/runtime/uv.lock \
  --pseudonym-key /controlled/secrets/audit-pseudonym.key \
  --scientific-candidate-reference "$SCIENTIFIC_CANDIDATE_REFERENCE" \
  --scientific-production-reference "$SCIENTIFIC_PRODUCTION_REFERENCE" \
  --approval-environment "$APPROVAL_ENVIRONMENT" \
  --runner-platform linux/arm64 \
  --runner-identity "$RUNNER_IDENTITY" \
  --hostname sealed-runner \
  --hardware-provider "$HARDWARE_PROVIDER" \
  --hardware-instance-type "$HARDWARE_INSTANCE_TYPE" \
  --hardware-cpu-model "$HARDWARE_CPU_MODEL" \
  --hardware-accelerator "$HARDWARE_ACCELERATOR" \
  --hardware-region "$HARDWARE_REGION" \
  --hardware-operating-system "$HARDWARE_OPERATING_SYSTEM" \
  --memory-limit-bytes "$MEMORY_LIMIT_BYTES" \
  --cpuset-cpus "$CPUSET_CPUS" \
  --tmpfs-size-bytes "$TMPFS_SIZE_BYTES" \
  --blueprint-root /controlled/production-control-blueprint \
  --finalized-controls-root /controlled/production-run-closure \
  --suite-base-root /controlled/suite \
  --output /controlled/config/production-control-materialization.json \
  --receipt /controlled/config/production-control-materialization.write-receipt.json)"

CONFIG_FILE_SHA256="$(printf '%s' "$CONFIG_RESULT" | jq -er '.config_file_sha256')"
RECEIPT_FILE_SHA256="$(printf '%s' "$CONFIG_RESULT" | jq -er '.receipt_file_sha256')"
SCIENTIFIC_INDEX_DIGEST="$(printf '%s' "$CONFIG_RESULT" | \
  jq -er '.scientific_index_digest')"
printf '%s' "$CONFIG_RESULT" | jq -e '.oci_promotion_required == true' >/dev/null
```

The command accepts no digest override. It reproduces the terminal factory receipt, hashes the
complete factory tree, derives the factory-config, factory-suite, C0-extraction, OPA, `uv.lock`, and
pseudonym-key digests, and checks the candidate reference and platform against the factory and C0
records. The extraction receipt's `c0_sha` is candidate-image source commit P, not future apparatus
commit A. The config and its write receipt expose it only as `candidate_image_source_commit`. The
candidate and future production references must be distinct digest-qualified locators
ending in the same `sha256:<index>` value. The config records both, the shared
`scientific_index_digest`, and literal `oci_promotion_required: true`; it does not claim that the
future production locator exists yet. C1 admission later binds `oci_promotion_receipt_sha256` and
proves raw-index equality at that destination. The writer rejects a symlinked input or root,
overlapping roots, a noncanonical CPU list, an existing output, or a source mutation detected by the
final rehash. Both output files are canonical mode `0600` files; the write receipt binds the config
checksum, byte count, mode, source checksums, locator transition, P, approval environment, and typed
readback. The config admits only `approval_environment: confirmatory` and requires the runner
identity to equal `github-actions:environment:confirmatory`. The blueprint and post-A instantiation
receipt repeat the approval environment, while all five workload identities are checked against
that derived runner identity. A crash
after the config rename but before the receipt rename may publish only the uniquely derived missing
receipt. A byte difference at either existing name is terminal.

Then materialize the five closed launcher blueprints from that derived config:

```bash
BLUEPRINT_RESULT="$(fractal-production-controls materialize-blueprint \
  --materialization-config /controlled/config/production-control-materialization.json \
  --materialization-config-sha256 "$CONFIG_FILE_SHA256")"
BLUEPRINT_FILE_SHA256="$(printf '%s' "$BLUEPRINT_RESULT" | \
  jq -er '.blueprint_receipt_file_sha256')"
```

The raw blueprint contains five canonical WorkloadSpec templates, their runtime-plan templates, an
exact `production_workloads` fragment, and an exact hardware fragment. Every executable
`code_commit` slot contains the registered C0 sentinel. P appears only as
`candidate_image_source_commit` in the authority header. Workload loading, runtime preflight, and
container launch reject these unresolved templates. The receipt binds both
fragment hashes and the complete inventory. The materializer reports three distinct C1 values:
the materialization-config file SHA-256, the blueprint receipt's canonical-object SHA-256, and the
blueprint receipt file SHA-256 (canonical JSON plus one newline). Print the C1-ready sections, or
print their secure local paths for a byte-preserving manifest update:

```bash
fractal-production-controls print-manifest-fragments \
  --materialization-config /controlled/config/production-control-materialization.json \
  --materialization-config-sha256 "$CONFIG_FILE_SHA256" \
  --blueprint-receipt-file-sha256 "$BLUEPRINT_FILE_SHA256"

fractal-production-controls print-manifest-fragments \
  --materialization-config /controlled/config/production-control-materialization.json \
  --materialization-config-sha256 "$CONFIG_FILE_SHA256" \
  --blueprint-receipt-file-sha256 "$BLUEPRINT_FILE_SHA256" \
  --print-paths
```

The emitted `sealed_execution.production_controls` object has exactly those three digest fields.
They live inside `study-manifest.json`, so the signed manifest predicate, the registry record, and
the closed 27-file Zenodo package bind the apparatus without adding another control-specific
deposit member. A
locally consistent replacement config or blueprint after C1 has different bytes and fails against
these public pins.

The same manifest carries `sealed_execution.c0_evidence_release`. That closed object binds the
immutable GitHub C0 evidence release by tag, C0 commit, release URL, archive and checksum asset
URLs, byte counts, SHA-256 values, and the digest-checked post-publication verification receipt.
The registration directory separately retains `c0-public-verification.json`, a fresh C1-time public
readback bound by the signed predicate. The release assets remain outside the 27-file directory;
adding either asset to the Zenodo package fails its exact inventory.

Once A exists, derive the executable controls without changing the raw candidate, config, or
blueprint:

```bash
: "${C0_COMMIT:?set the full apparatus commit A}"
INSTANTIATION_RESULT="$(fractal-production-controls instantiate-c0-controls \
  --materialization-config /controlled/config/production-control-materialization.json \
  --candidate-package /controlled/c0/candidate-manifest-package \
  --candidate-image-closure /controlled/c0/candidate-image-closure.json \
  --apparatus-commit "$C0_COMMIT" \
  --output-root /controlled/c0-production-controls)"
C0_CONTROL_INSTANTIATION_RECEIPT="$(printf '%s' "$INSTANTIATION_RESULT" | \
  jq -er '.receipt_path')"
C0_CONTROL_INSTANTIATION_RECEIPT_SHA256="$(printf '%s' "$INSTANTIATION_RESULT" | \
  jq -er '.receipt_file_sha256')"
```

The command resolves only the A-dependent fields. It writes an A-bound launcher identity, five
workload specs, five runtime-plan templates, five preflight contracts, and two manifest fragments
under one private staging directory. It also copies the exact bytes of
`candidate-study-manifest.json` and `candidate-manifest-assembly-receipt.json` into the fixed
`candidate-manifest-package/` subtree. Those paths, both file hashes, and the two members themselves
enter the instantiated payload-tree digest. The operator reopens the snapshot as a closed candidate
package before it synchronizes every child directory and publishes the tree with one no-replace
rename. The receipt records A separately from P, plus T, the candidate and production locators,
scientific digest D, the release digest, raw candidate hashes, raw config and blueprint hashes,
exact output membership, and the output-tree digest.

The raw candidate cannot contain this receipt hash without a digest cycle. The C0 release records
it at
`sealed_execution.c0_evidence_release.apparatus_evidence.production_control_instantiation_receipt_file_sha256`.
The production image record also contains the canonical authenticated environment readback, whose
file digest enters the same apparatus object as
`github_environment_control_receipt_file_sha256`. The C1 transition inserts that verified C0
evidence object. Its receipt verifier independently recomputes the evidence-release file hash from
canonical JSON plus LF and compares the candidate manifest and assembly-receipt file hashes with the
immutable apparatus subrecord. A transition receipt cannot satisfy registration or Zenodo packaging
by repeating self-declared digests.

The materialization config declares provider, instance type, CPU model, accelerator, region, and
operating system before C1. It derives logical cores from `cpuset_cpus` and memory GiB from the
byte-exact container limit. During finalization, all five preflight receipts must agree with those
claims and with C1. A fragment edit, reordered workload, changed wrapper hash, or one-corpus
hardware drift aborts before the production closure is admitted.
For each corpus, finalization then regenerates the A-bound launcher identity, provisional plan bytes,
one-file control-tree digest, all bind mounts, environment, image, commit, CPU and memory limits,
writable roots, volume names, output-copy root, tmpfs settings, workload binding, preflight command,
and complete preflight contract. The regenerated typed objects must equal the post-A instantiation
tree byte for byte, while resolving the raw sentinel templates must reproduce the same workloads.
Runtime transition checks remain a second boundary, not a substitute for this preflight
reproduction.

The pre-C0 rehearsal manifest is not an ordinary permissive draft. Its dedicated validator requires
`draft` and `0.3.0-draft`, a nonempty blocker list, and a literal `"tbd"` C0 evidence-release field,
then applies frozen validation to every other section. One C0 commit sentinel must occur at exactly
13 registered code and workflow paths. Candidate loading resolves only those occurrences to the
current C0 commit A. The default provider-plan loader has no such escape hatch and accepts frozen
manifests only.

Image construction may occur at bootstrap source commit P before A exists. The v2 candidate-image
closure therefore separates P from executable identity: a build-context-tree digest T is hashed
with the scientific and release index digests and their three platform-manifest digests. Rehearsal
evidence records P, A, T, the closure-file digest, and that bootstrap-closure digest independently.
It never requires P to equal A. This permits the final C0 commit to bind already observed OCI
digests even when registry provenance or SBOM descriptors make the outer index byte-dependent.

Before C1, compare the candidate and frozen provider-plan closures after resolving the 13 sentinels
to A. Their normalized digests must match. The candidate-to-frozen transition permits only the
registered sentinel replacement, `draft` to `frozen`, `0.3.0-draft` to `0.3.0`, removal of the
declared blockers, and insertion of the verified C0 evidence-release binding. Any other byte-level
content change fails before the manifest lock is created.

Run that transition with the dedicated operator. Both inputs must be canonical JSON with one final
newline. The output parent must be private, and the destination directory must not exist:

```bash
install -d -m 0700 /controlled/c1
fractal-c1-manifest-transition \
  --candidate-package /controlled/c0/candidate-manifest-package \
  --c0-commit "$C0_COMMIT" \
  --c0-evidence-release /controlled/c0/c0-evidence-release-binding.json \
  --output-directory /controlled/c1/frozen-transition
```

The candidate input is the exact private two-member package emitted by
`fractal-candidate-manifest-assembler publish-closed`, not a naked manifest. The loader admits only
a mode-`0700` directory owned by the current operator containing mode-`0600`, singly linked
`candidate-study-manifest.json` and `candidate-manifest-assembly-receipt.json`. It retypes both
objects and checks the manifest byte digest, semantic digest, and provider-plan closure against the
assembly receipt before deriving C1.

The command accepts no caller-supplied digest and no field override. It first requires the v2 C0
binding's target and apparatus commits to equal A. The apparatus rehearsal-manifest digest must
equal the raw candidate's canonical-object digest, while its provider-plan closure must equal the
candidate plans after the exact 13-path A substitution. The operator then derives the frozen object,
validates the registered transition and full frozen schema, publishes both files with mode `0600`
inside a mode-`0700` directory through one no-replace directory rename, and performs an
inode-pinned typed readback. A stale candidate, v1 binding, substituted
evidence object, extra sentinel, existing destination, symlink, or interrupted temporary write
cannot yield a transition receipt. Preserve both outputs in the C1 commit:

```bash
install -m 0600 /controlled/c1/frozen-transition/study-manifest.json \
  research/study-manifest.json
install -m 0600 /controlled/c1/frozen-transition/manifest-transition-receipt.json \
  research/manifest-transition-receipt.json
umask 077
fractal-retrieval-governance study-digest \
  --manifest research/study-manifest.json > research/study-manifest.sha256
```

The C1 tag must name the direct child of C0 whose changed path set is exactly the frozen manifest,
its semantic-digest lock, and `research/manifest-transition-receipt.json`. Registration rejects a
two-file manifest-only commit, an uncommitted receipt, or a receipt whose candidate-package,
assembly, C0, frozen-manifest, or provider-plan bindings do not agree.

Only then may the custodian register the frozen state.
The canonical manifest digest must then be deposited in an independently administered registry
before any sealed run opens. The
[OSF registrations guide](https://help.osf.io/article/330-welcome-to-registrations) describes a
time-stamped, read-only study plan that can be public or embargoed. The custodian retains
a canonical local copy of the registry record and records its identity, HTTPS URI, UTC registration
time, and file SHA-256 in a closed `ProtocolRegistrationReceipt`. The record and receipt use closed
schemas, canonical JSON, and exactly one terminal newline. A local receipt without the retrieved
registry record is insufficient.

Immediately before opening the run, the CLI performs one fresh certificate-validated HTTPS GET of
the receipt's registry URI. It refuses redirects, a changed response URL, non-200 responses,
invalid or oversized `Content-Length`, and any body larger than 64 KiB. The fetched digest and exact
bytes must match the secure local canonical record and the receipt. This control-plane admission
preflight occurs before the runner enters the network-disabled, noninteractive execution
environment. It verifies prospective public availability, not registry ownership. A reviewer
independent of the online runner must still establish who controls the registry and preserve that
evidence in the controlled study package.

The Python API exposes `trusted_registry_record_fetcher` only as an explicit test/integration seam.
The production CLI never supplies it. An integration that injects this callable assumes transport
authentication as part of its trusted computing base; returned bytes still face the same size,
digest, and byte-equality checks.

Post-C1 control finalization holds one derived sibling lock for the entire state classification,
directory exchange, verification, and receipt publication. The lock is a runner-owned, singly
linked, mode-`0600` regular file and uses a nonblocking operating-system `flock`. It remains outside
the frozen closure. A concurrent finalizer fails before acting, while a later `--resume` process
reopens the same lock and reclassifies both directory names only after acquiring it. This prevents
two recovery processes from exchanging the receipt-only and full-final trees twice.

The exchange admits two distinct mode-`0700` sibling directories owned by the runner. It opens the
shared parent and both children without following links, proves that each name still resolves to
the opened inode immediately before the platform exchange, and passes only the parent descriptor
and child names to `renameatx_np` or `renameat2`. After the call, it proves that the two inode
identities changed names and synchronizes the parent directory before a finalization receipt can be
published. A symlink, permissive mode, different parent, or concurrent name substitution aborts the
transition without being treated as a recoverable post-exchange state.

Finalization also snapshots the exact sharded-plan and trial-runtime receipt bytes while checking
their registered file digests and typed semantic links. Closure staging uses those admitted byte
strings. It does not reopen a factory path after the context check, so a post-admission source
replacement cannot enter either the staged or canonical closure.

## 1. Prepare label-separated corpus artifacts

The custodian starts from a normalized corpus whose stage is `sealed`. For each corpus, the custody
tool emits:

1. a label-separated online artifact with documents, query text, and opaque HMAC-SHA256 trial and
   family keys; and
2. a sealed label artifact with answers, relevant IDs, evidence bundles, label metadata, and the
   online execution-artifact SHA.

The online artifact must contain no original query IDs, relevance labels, answers, evidence,
label metadata, or fields that disclose them. It does contain query text. Public benchmark queries
may be reidentified by an operator familiar with the source data, so opaque keys alone are not a
proof of blinding. The runner must be frozen, noninteractive, and denied unregistered egress; its
artifacts remain immutable and labels stay with the custodian until an external completion anchor
binds both the prediction and action-panel digests. BRIGHT and MIRACL have relevance labels but no
gold evidence bundles. SciFact, HotpotQA FullWiki, and T2-RAGBench form the complete-evidence
subset.

The relevance labels may support secondary IR summaries. Primary ANN recall is computed against
the completed exact-authorized action in the pre-label panel for all five corpora.

The custodian stores the artifacts separately. For a sharded `online-execution` package, the frozen
manifest `sha256` pins the complete directory tree and its `revision` carries the canonical logical
plan digest. The `sealed-labels` artifact binds that logical digest. Neither execution control
embeds the manifest digest because doing so would create a digest fixed-point cycle. The online
runner receives only the label-separated online package.
`write_online_execution_artifact` and `write_sealed_label_artifact` create canonical files
exclusively; their matching loaders reject noncanonical JSON, duplicate keys, symlinks, and
multiply linked files.

Before freeze, the custodian also encrypts each canonical label file to one absolute drand round and
pins the resulting `sealed-label-ciphertext`. The closed custody-seal receipt binds all five triples
of online, plaintext-label, and ciphertext digests to the chain hash, integer round, timelock-tool
digest, and custody-builder digest. Relative durations and moving aliases such as `@latest` are not
admissible. Follow the exact ceremony and digest conventions in
[label-custody.md](label-custody.md).

## 2. Validate and lock the frozen manifest

From the repository root, validate the document:

```bash
fractal-retrieval-governance validate-study \
  --manifest research/study-manifest.json
```

After all fields are final and `status` is `frozen`, enforce sealed prerequisites:

```bash
fractal-retrieval-governance validate-study \
  --manifest research/study-manifest.json \
  --require-frozen
```

Compute the canonical digest:

```bash
fractal-retrieval-governance study-digest \
  --manifest research/study-manifest.json
```

The custodian writes that exact lowercase digest to a separately controlled lock file. Any manifest
edit changes the canonical digest and requires a protocol amendment, new lock, and new receipt URI
before sealed execution and outcome scoring.

## 3. Verify frozen artifacts and create the exclusive local run receipt

The frozen manifest derives the sole receipt URI from its canonical SHA-256 through the prespecified
`receipt_uri_template`. The CLI does not accept a caller-selected receipt path.

The local artifact map assigns each manifest artifact ID to one relative path under a preprovisioned
root. It cannot supply or override a digest. Before opening the run, verify exact coverage and write
the canonical receipt exclusively:

```bash
fractal-retrieval-governance verify-study-artifacts \
  --manifest research/study-manifest.json \
  --artifact-root /controlled/artifacts \
  --artifact-map /controlled/artifact-map.json \
  --receipt /controlled/artifact-verification.json
```

Retain the canonical frozen manifest bytes as `/controlled/frozen-study-manifest.json`. On that same
custodian-controlled machine, build the label-free artifact-ID binding suite in one atomic
publication:

```bash
fractal-production-controls write-required-artifact-bindings \
  --materialization-config /controlled/config/production-control-materialization.json \
  --c0-control-instantiation-receipt \
    /controlled/c0-production-controls/c0-control-instantiation-receipt.json \
  --frozen-manifest /controlled/frozen-study-manifest.json \
  --artifact-verification-receipt /controlled/artifact-verification.json \
  --artifact-root /controlled/artifacts \
  --artifact-map /controlled/artifact-map.json \
  --output-root /controlled/required-artifact-bindings
```

This command accepts no corpus selector, artifact ID, or digest override. It reopens the C0 factory,
raw blueprint, and post-A instantiation receipt; checks the C0 apparatus digest pin; and checks the
disclosed five-workload and hardware sections against the frozen
manifest, rehashes every local artifact through the closed map, and requires byte equality with the
admitted verification receipt. It derives all five binding objects from manifest roles, writes them
under a private staging name, verifies exact membership, and performs one operating-system
no-replace rename. Any missing, extra, duplicated, stale, substituted, linked, or partial member
aborts admission. The output name must be absent inside a runner-owned parent that is not writable
by group or other identities. The result contains identifiers, revisions, and the verification
receipt, not artifact payloads. Copy or mount that exact root read-only into the online control
plane. Do not transfer the custodian's plaintext artifacts with it.

The full custodian admission then opens the externally registered run with:

```bash
fractal-retrieval-governance begin-sealed-run \
  --manifest research/study-manifest.json \
  --lock /controlled/study-manifest.sha256 \
  --artifact-verification-receipt /controlled/artifact-verification.json \
  --artifact-root /controlled/artifacts \
  --artifact-map /controlled/artifact-map.json \
  --protocol-registration-receipt /controlled/protocol-registration.json \
  --protocol-registration-record /controlled/protocol-registration-record.json \
  --registration-package /controlled/confirmatory-c1-registration \
  --runner-identity github-actions:environment:confirmatory
```

Replace the example identity before freeze; at execution it must exactly match the value pinned in
the manifest. The production command requires `--artifact-root` and `--artifact-map`, even when a
verification receipt already exists. It reloads the map against the manifest pins, rereads every
artifact from the controlled root, constructs a fresh verification receipt in memory, and requires
its canonical bytes to equal the admitted receipt. This closes the interval between the earlier
verification command and run opening. The stored receipt cannot hide a later local mutation.

This full operation opens the plaintext-label artifacts. It belongs on the custodian machine, not
inside the online execution environment. The online machine receives the resulting receipts and a
label-free artifact tree, then runs the separate admission below.

The command also validates the frozen manifest, lock digest, runner identity, exact artifact-ID
coverage, and every manifest digest. Its registration preflight accepts only the closed 27-file C1
package for Zenodo record `21361837`. It revalidates both retained GitHub attestations, anonymously
downloads all 27 public Zenodo files, and requires exact byte equality with the local package. The
local registry record and receipt must match the package and fixed Zenodo identity. It copies the
pinned code commit, OCI image digest, protocol-registration pointer, and artifact-verification
pointer into the receipt, then writes the receipt exclusively. An existing receipt, changed
manifest or package, mismatched identity, failed attestation, unavailable or incomplete public
record, changed local artifact, or symlinked receipt parent aborts the opening.

The receipt contains manifest digest, protocol, UTC timestamp, runner identity, code commit, OCI
image, both prerequisite receipt pointers, and its own URI. It records local preparation under the
publicly registered protocol. The later provider CAS records `RUN_CLAIMED`; this local file does not
prove that execution started or that no off-apparatus run exists.

## 4. Execute the label-separated online run

Before retrieval, create an online process-admission receipt:

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

The command verifies full receipt coverage but freshly opens only the registered online-safe roles.
Plaintext labels, sealed inputs, development data, fitted analysis models, and the query-partition
audit are excluded. Admission succeeds with those files absent and fails on a missing or altered
ciphertext. The receipt lists every artifact opened by the online process.

Next, complete `initialize-volume`, `preflight`, and `materialize-transition` for every fixed corpus
as specified by the [sealed container launcher](sealed-container-launcher.md). Put the two runtime
records for each corpus at these exact paths:

```text
/controlled/runtime-evidence/<corpus-id>/runtime-preflight-receipt.json
/controlled/runtime-evidence/<corpus-id>/runtime-plan-transition-receipt.json
```

The corpus directories must cover `scifact`, `hotpotqa-fullwiki`, `t2-ragbench`, `bright`, and
`miracl-transfer`, with no other member. The online control plane must now expose the exact
five-corpus binding root published by the custodian at `/controlled/required-artifact-bindings`.

The provider plan's `registered_online_runtime_budget_seconds` is an admission ceiling, not an
observed confirmatory result. It is prespecified from development-only capacity planning, fixed
before any sealed confirmatory input is opened, and cannot exceed 72,000 seconds. Online execution
is inadmissible when the registered budget exceeds that ceiling; the surrounding Actions job's
longer timeout does not enlarge the registered budget.

Derive the 18 authority fields plus the fixed `schema_version` from those inputs:

```bash
fractal-production-controls write-finalization-request \
  --materialization-config /controlled/config/production-control-materialization.json \
  --c0-control-instantiation-receipt \
    /controlled/c0-production-controls/c0-control-instantiation-receipt.json \
  --frozen-manifest /online/control/frozen-study-manifest.json \
  --manifest-lock /controlled/study-manifest.sha256 \
  --c1-package-root /controlled/confirmatory-c1-registration \
  --protocol-registry-record /controlled/protocol-registration-record.json \
  --protocol-registration-receipt /controlled/protocol-registration.json \
  --online-custody-admission /online/control/online-custody-admission.json \
  --custody-seal-receipt /online/artifacts/custody/custody-seal-receipt.json \
  --artifact-verification-receipt /online/control/full-artifact-verification.json \
  --artifact-root /online/artifacts \
  --artifact-map /online/control/artifact-map.json \
  --required-artifact-bindings-root /controlled/required-artifact-bindings \
  --runtime-evidence-root /controlled/runtime-evidence \
  --output /controlled/suite/finalization-request.json
```

The writer calculates the materialization-config and blueprint-receipt hashes itself. It derives the
blueprint path from the validated config and the sealed-run receipt path from the registered manifest
digest. The request names the fixed post-A receipt path, while its digest comes from C0 apparatus
evidence and is not a caller argument. Before publication it verifies C0, C1, registration freshness, the manifest lock, sealed-run
identity, online custody, the exact five-corpus binding root, every preflight transition, and the
shared hardware observation. It publishes canonical mode-`0600` bytes with a no-replace rename,
reloads them by the emitted `request_sha256`, and repeats the authority check. The request output
must be outside every admitted immutable tree.

Use only the digest printed by that writer to finalize the closure:

```bash
fractal-production-controls finalize \
  --request /controlled/suite/finalization-request.json \
  --request-sha256 <request_sha256-from-writer-output> \
  --receipt /controlled/suite/production-control-finalization-receipt.json
```

The finalizer reloads the request and all of its authorities before and after taking the sibling
process lock. Only then may the operator run `instantiate-plan` and `launch` for each corpus.

For each fixed corpus, create a separate exclusive runtime-attestation receipt in its network-disabled
Linux container before that corpus source is opened. Each `run_sealed_online_once` call reloads its
C1-pinned corpus plan and matching receipt, reobserves the live namespace and exact environment,
verifies its invocation marker, and checks that every source path belongs to a registered read-only
mount. Its exclusive attempt binds both attestation digests. The suite advances to
`ONLINE_COMPLETE` only after verifying all five distinct plan/receipt/marker closures. The exact
capture order and failure rules are in
[runtime-attestation.md](runtime-attestation.md).

The online runner verifies the receipt, manifest, component digests, outer package tree against the
manifest `sha256`, and canonical execution-plan digest against the manifest `revision` before
processing any trial. It must then:

1. reject missing, duplicate, extra, or cross-stage opaque trial keys;
2. issue the initial OPA bulk decision before authorized index construction or probing;
3. build or load an index only for the authorized universe and verify its policy-revision and mask
   binding;
4. run the bounded authorized probe with a maximum of 101 neighbors and prespecified effort;
5. compute only LID at k=50, LID-CV, relative contrast, and radius expansion as geometry;
6. execute the paired action matrix for `hnsw-low`, `hnsw-high`, `exact-authorized`, and `abstain`;
7. issue a fresh OPA decision immediately before any selected document IDs cross the controlled
   return boundary;
8. fail closed on a replay, version change, mask change, unavailable PDP, or revoked selected ID;
9. retain crashes, abstentions, undefined geometry, missing work counters, empty authorized
   universes, and all timing rows; and
10. write immutable predictions and the complete raw action panel without accessing the sealed
    label artifact.

Completed and governed-abstention rows enter the panel through `GovernedActionExecution` and
`action_panel_from_governed_executions`. The typed admission checks the `GovernedResult` against a
self-hashed `AuditRecord`: action, policy revision, both authorization decisions, returned IDs,
search work, abstention state, and measured request latency must agree. Returned IDs, latency, and
entitlement count are then derived from those checked objects rather than supplied as scalar panel
fields. An audit record cannot be reused across actions. Governed counterfactuals must use the
selected decision's policy revision, and their final decisions must agree on policy revision,
environment digest, document-universe digest, authorization mask, and policy availability.

A failure enters through `FailedActionExecution` with monotonic start and finish times, the pinned
runner identity, and one of five closed codes: `backend-error`, `backend-timeout`, `invalid-result`,
`resource-exhausted`, or `runner-interruption`. Admission derives latency from that interval and
recomputes a failure-timing digest bound to the trial, action, controller decision, authorization
decision, code, runner, and timing window. This is bound evidence inside the pinned runner, not an
independent clock. The failure carries no audit digest, returned IDs, or
caller-supplied entitlement input and cannot claim a completed search. Its serialized panel row
records zero entitlement violations because no IDs were emitted. Every `hnsw-low` row, including a
failure, must carry the pre-outcome feature tuple; other actions cannot carry it. Admission checks
placement, not computational provenance.

The panel builder requires exactly one admitted outcome for every trial-action pair and preserves a
selected-action failure in the intention-to-treat panel. The registered `abstain` action must be a
governed abstention, and `exact-authorized` must be completed. Missing or duplicate outcomes are
inadmissible. The caller must supply the expected audit-chain head, the frozen query-partition-audit
digest, and `primary`. The receipt schema may encode `reserve` for nonconfirmatory engineering, but
v0.3 confirmatory input rejects it. The builder verifies the complete governed-record chain and
returns an `AdmittedActionPanel`: the panel plus a detached `ActionPanelAdmissionReceipt`.

The detached receipt binds the panel bytes, manifest, run, execution artifact, corpus, partition,
query-partition audit, ordered audit chain, and one admission record per trial-action cell. Each
record carries a digest of the controller decision and a digest of the applicable policy decision,
including its decision ID, request digest, mask digest and size, policy revision, availability,
environment, and document universe. The applicable authorization is the final decision when one
exists and otherwise the initial decision. Governed rows carry their audit position and predecessor;
failed rows carry their runner-bound timing digest. `write_action_panel_admission_receipt` writes the
canonical receipt exclusively. The panel alone is not an admissible confirmatory input.

The authorization observation ends at the controlled return from `GovernedRetriever.query`. The
runner cannot claim continuous authorization after the result object is returned. A later
generator, UI, network sink, or consumer is outside this experiment unless a separate protocol
adds and tests that boundary.

After the provider-claimed invocation marker is durable, but before the first request timer, the runner prepares
every distinct authorization named by the admitted trial environments. It loads the exact slice and
receipt-verified HNSW object for each mask, emits a cache-preparation receipt, and seals the cache.
An unprepared mask or an environment-to-mask rebind during timed execution aborts the matrix.

The request timer begins before validation and the first timed OPA call. It ends after the fresh
final OPA call when the governed result is returned. Recorded governed retrieval request latency
includes both timed policy decisions, the sealed-cache lookup, probe, geometry, action selection,
selected search, and local request overhead. It excludes pre-timing index loading, upstream
embedding, generation, UI work, and downstream network or consumer work. The registered H3
estimand is warm-service latency over the admitted policy-state set.

Probe latency and work are system telemetry in the frozen predictive schema. Only LID at k=50,
LID-CV, relative contrast, and radius expansion constitute the geometric block.

Before the first request, the runner also reconstructs the query-level control covariates. It
computes embedding drift from each active/current query-row pair and computes policy churn from the
complete baseline/current masks bound to that trial environment. The baseline mask is evidence only;
it is absent from the OPA catalog. The attempt receipt binds the two query epochs, row order,
schedule, embedding receipt, and policy-intervention receipt. No production argument accepts either
numeric feature.

## 5. Seal and externally anchor predictions and the raw action panel

The online runner emits an immutable prediction artifact and action-panel artifact only after the
receipt exists. The prediction artifact must bind:

- manifest SHA-256;
- receipt SHA-256;
- online execution-artifact SHA-256; and
- the exact complete set of opaque trial keys in canonical order.

The action panel contains every prespecified action for every opaque trial. It records registered
action order and actual execution position as separate fields,
returned document IDs, execution or failure state, controller selection, request latency,
entitlement count, the supplied pre-outcome feature tuple on `hnsw-low`, and the audit-record digest
for each completed or governed-abstention row. Typed admission checks the feature tuple's action
placement, and later analysis admission checks its dimension. Neither step independently proves
how the runner computed it. The external anchor freezes those supplied bytes before label release.
The panel contains no relevance judgment, gold bundle, recall, evidence-sufficiency value, answer
label, or derived failure target.

The custodian verifies that no label field entered the online artifact, every manifest-declared
trial key has a prediction row, and action failures appear explicitly in the panel. A selected
failure therefore has an empty returned-ID prediction plus the panel's closed failure state; a
missing prediction cannot be silently deleted. An independently administered HTTPS anchor then
records the exact prediction digest and a typed action-panel binding containing the panel digest,
corpus, stage, manifest, run receipt, and online-execution digest.
The resulting `PredictionCompletionReceipt` is written exclusively through a no-follow path.

`write_prediction_artifact`, `write_action_panel_artifact`,
`write_action_panel_admission_receipt`, and `write_prediction_completion_receipt` are the canonical
file boundaries. The matching loaders reject duplicate keys, nonfinite values, noncanonical bytes,
symlinks, hard links, and schema extensions. In-memory objects alone are not releasable custody
evidence.

The completion anchor must postdate the run start. Its identity, URI, and canonical UTC timestamp
remain in the receipt; the receipt digest enters the offline artifact. A prediction object without
this receipt cannot release labels.

## 6. Join labels offline

Only after the completion receipt exists and the registered drand round is available may the scorer
call `join_predictions_after_receipt`. The join requires the same typed panel binding recorded at
completion and verifies the manifest, run receipt, execution, prediction, and label digests plus
exact trial keys. A different panel, digest mismatch, key mismatch, duplicate, omitted trial, or
unanchored prediction aborts scoring. Timelock encryption is time-conditioned, not
completion-conditioned: if the round arrives before a valid completion anchor, v0.3 ends without
scoring. Re-encryption or a later round requires a new registered protocol.

The join exposes relevance and complete-evidence labels only in the offline scoring environment.
For each action, the analysis input builder derives ANN recall against the completed
`exact-authorized` row in the same anchored panel; relevance labels do not define ANN recall. It
derives complete-evidence sufficiency by matching anchored returned IDs to the sealed gold bundles
and requires zero entitlement violations. It rejects any post-label row whose action, latency,
state, feature vector, entitlement count, trial key, or family key differs from the anchored panel.
Answer emission, answer coverage, false permit, and false denial may be calculated as declared
secondary outputs. They cannot alter the primary gates.

The low-effort modeling target is intent-to-treat action failure. A non-completed `hnsw-low` row is
a failure; a completed row fails when authorized recall is below 0.90. A completed empty result
against an empty exact-authorized reference is a valid governed no-result service outcome with
recall one. No authorized-universe-size exclusion is applied, and the composite is never described
as pure ANN failure.

The scorer loads each custody file through `load_sealed_label_artifact`; reconstructing a label
object from copied fields is not file admission. `ConfirmatoryInputArtifact` derives its analysis
configuration from the actual frozen manifest; it
does not accept a caller-supplied replacement. It verifies the run receipt and exact
artifact-verification receipt, admits each actual `SealedLabelArtifact`, recomputes its canonical
bytes against the per-corpus manifest pin, and requires every joined label object to equal that
admitted payload. A copied digest string is insufficient. Each panel's execution digest must match
the separately pinned `online-execution` artifact.

The input also requires one detached action-panel admission receipt per corpus. It verifies the
panel digest and every trial-action admission, requires the `primary` partition, binds the receipt
to the manifest's query-partition-audit digest, and checks failed-action runner identity against the
sealed run receipt. `run_confirmatory_analysis_once` then checks the canonical H1 model and H2 suite
bytes against their verified manifest artifacts before it admits an analysis attempt.

Production scoring uses the closed post-label operator rather than constructing these objects in an
interactive Python process:

```bash
fractal-confirmatory-input materialize --config /controlled/analysis/input-operator.json
fractal-confirmatory-input verify      --config /controlled/analysis/input-operator.json
fractal-confirmatory-input analyze     --config /controlled/analysis/input-operator.json
```

The command reconstructs `LABELS_RELEASED` through the registered GitHub verifier, takes each online
root and plaintext-label path from that state, performs a fresh external-anchor check, and persists
one input artifact plus its detached source-inventory receipt. The exact config, file identities,
exclusive-write order, and terminal failure rules are in
[confirmatory-input-operator.md](confirmatory-input-operator.md).

The model pins are the exact outputs of `canonical_h1_model_artifact_bytes` and
`canonical_h2_model_suite_artifact_bytes`: UTF-8 canonical JSON without a trailing newline. The H1
pin covers the full-model artifact, and the H2 pin covers the full suite. Do not apply the custody-file
newline convention to these model byte payloads.

## 7. Run the pinned analysis once

The endorsed sealed entry point is `run_confirmatory_analysis_once`. The lower-level computation is
not a custody boundary. The built-in entry point accepts only a canonical absolute `file:` URI in
`sealed_execution.results_store`; it rejects authorities, query strings, fragments, control
characters, noncanonical encoding, dot components, and `s3:` or `gs:` stores. A remote store needs a
separately pinned adapter with authenticated create-if-absent semantics and its own conformance
evidence.

For manifest digest `<M>`, the built-in entry point derives three fixed paths inside that directory:

- `<M>.confirmatory-analysis-attempt.json`;
- `<M>.confirmatory-result-receipt.json`; and
- `<M>.confirmatory-result.json`.

It first checks the admitted model bytes, then creates the analysis-attempt receipt exclusively with
`O_EXCL`, before the H1 diagnostic or any H2–H3 outcome is computed. The receipt binds manifest, run receipt,
confirmatory-input digest, model-suite digest, runner identity, and result URI. An existing attempt
aborts before outcome computation. Once created, the receipt is retained even if analysis raises,
the process stops, or a later custody write fails.

After computation, the entry point checks that the result still binds the admitted manifest, run,
input, and model suite. It creates a detached result receipt exclusively before exposing the result
file. That receipt binds the attempt-receipt digest, result-artifact digest, manifest, and result
URI. It then creates the canonical newline-terminated result file exclusively. The secure loaders
reject noncanonical, linked, misplaced, or digest-mismatched receipts and results.

The computation verifies exact proposed/comparator pair IDs, registered action order, complete
per-trial position permutations, and floor/ceiling position balance. It calculates:

- the label-free full-model high-versus-low geometry orientation diagnostic for H1;
- paired `full` versus `system-policy` log loss, Brier loss, and AUPRC gain for H2, where
  `system-policy` includes probe latency and work and `full` adds only the four geometric features;
- the equal-corpus mean of family-level relative reductions in end-to-end governed retrieval request
  latency;
- retrieval-target attainment difference on all five corpora;
- complete-evidence sufficiency difference on the fixed three-corpus evidence subset;
- the equal-corpus mean of within-corpus proposed-to-comparator p95 ratios of family-mean request
  latency;
- the non-gating, position-adjusted paired log-latency sensitivity; and
- denied-item emission at the controlled retrieval boundary.

For corpus \(c\), family \(f\), and nested row \(r\), the latency estimand is

\[
D_{cf}=1-\frac{n_{cf}^{-1}\sum_r T^A_{cfr}}
{n_{cf}^{-1}\sum_r T^S_{cfr}},
\qquad
\Delta_C=\frac{1}{5}\sum_{c=1}^{5}\frac{1}{F_c}\sum_{f=1}^{F_c}D_{cf}.
\]

Nested rows are averaged separately by action before the family ratio is formed. Family statistics
are then averaged within corpus, and the five corpus estimates are averaged equally. This preserves
the paired family as the inferential unit and prevents extra policy draws, seeds, or timing repeats
from gaining weight. The runner must not substitute the mean of row-wise ratios.

For each primary endpoint, the runner performs 10,000 deterministic paired replicates with base seed
`20260713`, resampling query families with replacement inside each fixed corpus. Endpoint-specific
seed offsets are fixed in the pinned runner. It carries all nested rows and action pairs with the
selected family. Corpora, nested rows, and action rows are not separate bootstrap units.

The fifth percentile is the directional 95% lower bound. The ninety-fifth percentile is the
directional 95% upper bound. The runner applies these conditions as an intersection-union decision:

- equal-corpus mean family-relative latency-reduction lower bound greater than 0.10;
- retrieval-target attainment lower bound greater than -0.01;
- complete-evidence sufficiency lower bound greater than -0.01;
- equal-corpus mean of within-corpus p95 family-mean latency ratios upper bound less than 1.25; and
- zero denied items emitted at the controlled retrieval boundary.

H1 reports its directional contrast and legacy `h1_minimum_risk_increase` comparison as a
descriptive orientation diagnostic. It uses no sealed labels and cannot alter confirmatory success.
H2 requires the equal-corpus directional lower bounds for all three incremental geometry-gain
metrics to exceed their frozen thresholds and the corpus-specific point rule to pass inside at least
four of five corpora. All three H2 thresholds remain `TBD` in the draft; no numeric values are fixed.
Primary success is exactly the H2+H3 intersection. The frozen manifest must still supply numeric
geometry profiles and gain thresholds, and the admitted H1/H2 model artifacts must match their
manifest digests. The draft cannot produce a primary decision.

Within the pinned noninteractive scorer and controlled result directory, one durable attempt receipt
admits one result. The mechanism does not prevent arbitrary Python code, process-memory inspection,
logging, copying, or a storage administrator from bypassing the package. The scorer image, runner
identity, operating system, result-directory custody, and incident process remain in the trusted
computing base. No feature, threshold, exclusion, corpus weight, action, comparator, direction,
margin, or missing-value rule may change after labels are joined.

## 8. Verify joint-gate design

The frozen design report must use `development-family-cluster-resampling` with
`registered-percentile-family-bootstrap-plug-in-calibration` and the joint success event
`h2-and-h3-all-gates-pass`. Calibration and evaluation use disjoint, SHA-256-derived PCG64 streams.
The plug-in construction estimates power for the directional gates; sealed inference still uses
the exact registered 10,000-replicate family bootstrap. Its registered endpoint order is:

1. `h2-log-loss-reduction`;
2. `h2-brier-score-reduction`;
3. `h2-auprc-gain`;
4. `h2-four-of-five-consistency`;
5. `h3-family-relative-latency-reduction`;
6. `h3-retrieval-target-noninferiority`;
7. `h3-complete-evidence-noninferiority`;
8. `h3-family-mean-p95-latency-ratio`; and
9. `h3-zero-entitlement-violations`.

The report pins its development-data dependence source, effect scenarios, simulation seed, at least
5,000 simulations per candidate, candidate family counts 25, 50, 75, 100, 150, and 200 per corpus,
selected maximum family count, and selected joint-power lower bound. It must preserve the registered
family/corpus estimands and H2/H3 decision directions. `selection-audit.json` must provide the
closed exact certificate over the fixed 12-cell selection family. Bonferroni cellwise alpha is
`0.05 / 12`: 4,556 exact joint passes qualify a 5,000-study cell and 445 exact joint failures block
it. This yields at least 95% simultaneous coverage across six candidate counts and two required
scenarios without an independence assumption. Every checked study uses the registered
10,000-replicate bootstrap. Any primary approximate/exact gate disagreement aborts certification.
The action-position sensitivity retains its pointwise 95% result and remains non-gating. The
selected family count is the maximum requirement across endpoint-specific and joint-gate
calculations; the multiplicity-adjusted one-sided lower Monte Carlo bound must reach the frozen
0.90 design target.

The existing beta-binomial common-shock utility is only an event-yield sensitivity analysis for
low-effort action success. It cannot establish 90% power for the primary conjunction or select the
confirmatory family count. Observed inference remains the paired family bootstrap; no design
simulator can replace it.

## Technical failure; no confirmatory reserve

A failed process does not grant a rerun or reserve release. Preserve the run receipt, action-panel
admission receipts, analysis-attempt receipt, any detached result receipt, partial result, logs,
digests, and audit records. An admitted failure ends confirmatory v0.3. Any later attempt requires a
disclosed amended protocol version, a new frozen manifest, a new external registration, and a new
run receipt. It cannot be described as the original confirmatory run.

A null result, wide bound, failed gate, inconvenient subgroup, slow action, or observed entitlement
violation is a scientific result, not a technical failure.

## Release checklist

Release the frozen manifest and canonical digest, run receipt, permitted normalized-data references,
exclusions, opaque family counts, paired action matrix, action-panel admission receipts, prediction
schema, estimands, analysis-attempt receipt, detached result receipt, result digest, directional
bounds, all gate decisions, the endpoint-specific and joint-gate design report, event-yield
sensitivity, conformance results, incident records, and audit-chain anchors.

Restricted evidence text, original query IDs, raw subjects, authorization masks, query vectors,
policy secrets, and sealed credentials remain outside the public bundle. State explicitly that the
study is suite-conditional and that authorization was observed only through the controlled return
of `GovernedRetriever.query`.
