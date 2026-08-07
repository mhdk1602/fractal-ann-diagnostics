# C0 production artifact factory

The production artifact factory turns the five verified paired-embedding stores into the exact
label-free packages cited by C1. Before A exists, it runs inside the digest-qualified candidate
scientific image built from bootstrap source P on `linux/arm64`. The candidate and production
locators share the same OCI index digest D; production later promotes those exact index bytes.
It does not bind a runtime preflight receipt. Runtime preflight occurs after the factory has
finished and the final artifact trees exist, which avoids a digest cycle between construction
and sealed execution.

The command has no label, qrel, evidence, custody-release, plugin, callback, network, or generic
subprocess argument. Its sole secret input is an ephemeral HMAC key on standard input. That key
creates opaque query-family and trial identifiers; it does not score an outcome.

## Closed inputs

The module's `write-config` command takes only the artifact root, the externally pinned
embedding config, the verified post-embedding development operator root and receipt, the
partition audit, candidate scientific image, HMAC secret on descriptor 0, and an output path. It derives the
materialization, design seed, joint-power report, selected family count, and every downstream
randomized value. Duplicate caller values are rejected. The config records:

- the empty writable artifact root;
- the production embedding config path and digest;
- an absolute read-only embedding-source root, its complete tree digest, and the production
  embedding-suite receipt digest;
- the development materialization, query-partition audit, and joint-power report pins;
- the post-embedding development root, terminal receipt, and joint-power tree pin;
- the digest-qualified candidate scientific image and the sole admitted platform, `linux/arm64`;
- the fixed corpus order: `scifact`, `hotpotqa-fullwiki`, `t2-ragbench`, `bright`, and
  `miracl-transfer`;
- the fixed stage order: `fit`, `calibration`, then `sealed`;
- the design seed and its domain-separated policy, family-selection, permutation, and revision
  values;
- the installed hnswlib version and extension digest plus the fixed cosine, `M=16`,
  `efConstruction=128`, seed `20260714`, batch `512`, verification `ef=64`, one-thread index
  parameters;
- three replicas for every authorized-index stage and for the deployed full-active HNSW in the
  distinct roots `replicate-01`, `replicate-02`, and `replicate-03`, with replica 1 as the selected
  copy;
- the HMAC commitment and its derived public identifier.

The public identifier has one valid form:

```text
hmac_secret_sha256 = sha256(secret_bytes)
hmac_key_id = "sealed-online-ephemeral-sha256-" + hmac_secret_sha256
```

The config and receipts contain both values above, but never `secret_bytes`. A different key ID,
a different commitment, another platform, a reordered corpus or stage, a second selected
replica, or an undeclared JSON field fails during config admission.

The embedding config's `output_root` must equal the factory's `embedding_source_root`. The source
root must be a real read-only filesystem mount. The factory verifies the canonical production
embedding config, hashes the whole source tree, admits the producer-frozen embedding suite, checks
all five typed embedding stores, and joins their inventory to the development materialization,
partition audit, online projection, and power decision. Frozen admission rehashes the recorded
source and output bytes but does not rerun the Mac/MPS probe inside the Linux image.

The post-embedding receipt closes the development-to-production provenance join. Its typed
verifier must reproduce the same embedding-suite receipt, materialization receipt, design seed,
partition-audit file and semantic digest, joint-power report and tree, fixed index config, and
selected family count. The factory derives the materialization root and report path from that
verified operator package. It does not accept parallel caller values for those fields.

The candidate image contract guarantees `/opt/venv/bin/python`; it does not promise that the
`fractal-production-artifacts` console script is installed. Every invocation therefore uses the
module entry point explicitly.

Absolute paths are evidence. A `/host/input` to `/input` alias changes those paths and cannot satisfy
the signed upstream configs. Define the exact producer paths, bind each one to itself, and expose
only the listed development and label-free inputs. Do not mount a broader control or custody tree.
The pre-C1 construction containers use the invoking host UID/GID so private host bind mounts remain
owned by the process. This override applies only to factory construction. Sealed execution retains
UID/GID `65532:65532` and the named-volume launcher in [runner-image.md](runner-image.md).

Create distinct empty mode-`0700` roots for factory artifacts and factory controls, then write the
config. Fill every input path from the accepted handoff record; do not let an unverified config
choose a bind source through `jq` or shell evaluation. The typed admission inside the container
will reject any disagreement between those paths and the pinned configs. `full_staged_root` is
deliberately absent: frozen operator admission needs the development package and label-free audit,
not the raw staged tree or its sealed-label custody subtree.

```bash
set -euo pipefail

IMAGE='ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-candidate@sha256:<scientific-index-digest>'
DEVELOPMENT_ROOT='/absolute/host/path/post-embedding-output/operator-v1'
DEVELOPMENT_CONFIG="$DEVELOPMENT_ROOT/operator-config.json"
DEVELOPMENT_RECEIPT_SHA256='replace-with-post-embedding-receipt-sha256'
EMBEDDING_CONFIG='/absolute/producer/path/production-embedding-config.json'
EMBEDDING_CONFIG_SHA256='replace-with-production-embedding-config-sha256'
ONLINE_STAGING='/absolute/producer/path/online-staging-projection'
EMBEDDING_SOURCE='/absolute/producer/path/production-embedding-suite'
PARTITION_AUDIT='/absolute/controlled/path/query-partition-audit.json'
PARTITION_AUDIT_SHA256='replace-with-partition-audit-file-sha256'
FACTORY_ROOT='/absolute/host/path/production-artifacts'
FACTORY_CONTROL_ROOT='/absolute/host/path/factory-control'
FACTORY_CONFIG="$FACTORY_CONTROL_ROOT/production-artifact-factory.json"
FACTORY_CONFIG_SHA256='replace-after-write-config'
SCIFACT_REQUEST_SHA256='replace-after-prepare-shards'
HMAC_FILE='/absolute/private/host/path/query-id-hmac.bin'
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

umask 077
test "$HOST_UID" -ne 0
mkdir -m 0700 "$FACTORY_ROOT" "$FACTORY_CONTROL_ROOT"
test -f "$HMAC_FILE"
test ! -L "$HMAC_FILE"
case "$(uname -s)" in
  Darwin) HMAC_METADATA="$(stat -f '%u %Lp %l' "$HMAC_FILE")" ;;
  Linux) HMAC_METADATA="$(stat -c '%u %a %h' "$HMAC_FILE")" ;;
  *) exit 1 ;;
esac
read -r HMAC_OWNER HMAC_MODE HMAC_LINKS <<< "$HMAC_METADATA"
test "$HMAC_OWNER" = "$HOST_UID"
test "$HMAC_LINKS" = 1
case "$HMAC_MODE" in 400|600) ;; *) exit 1 ;; esac
case "$HMAC_FILE/" in
  "$DEVELOPMENT_ROOT/"*|"$ONLINE_STAGING/"*|"$EMBEDDING_SOURCE/"*|\
  "$FACTORY_ROOT/"*|"$FACTORY_CONTROL_ROOT/"*) exit 1 ;;
esac

FACTORY_CONTAINER_GUARDS=(
  --network none
  --read-only
  --cap-drop ALL
  --security-opt no-new-privileges
  --env PYTHONDONTWRITEBYTECODE=1
  --tmpfs "/tmp:rw,noexec,nosuid,nodev,size=64m,uid=$HOST_UID,gid=$HOST_GID,mode=1777"
)

FACTORY_INPUT_MOUNTS=(
  --mount "type=bind,src=$EMBEDDING_CONFIG,dst=$EMBEDDING_CONFIG,readonly"
  --mount "type=bind,src=$ONLINE_STAGING,dst=$ONLINE_STAGING,readonly"
  --mount "type=bind,src=$EMBEDDING_SOURCE,dst=$EMBEDDING_SOURCE,readonly"
  --mount "type=bind,src=$DEVELOPMENT_ROOT,dst=$DEVELOPMENT_ROOT,readonly"
  --mount "type=bind,src=$PARTITION_AUDIT,dst=$PARTITION_AUDIT,readonly"
)

docker run --rm -i \
  --platform linux/arm64 \
  --user "$HOST_UID:$HOST_GID" \
  "${FACTORY_CONTAINER_GUARDS[@]}" \
  --entrypoint /opt/venv/bin/python \
  "${FACTORY_INPUT_MOUNTS[@]}" \
  --mount "type=bind,src=$FACTORY_ROOT,dst=$FACTORY_ROOT" \
  --mount "type=bind,src=$FACTORY_CONTROL_ROOT,dst=$FACTORY_CONTROL_ROOT" \
  "$IMAGE" \
  -m fractal_ann_diagnostics.production_artifact_factory \
  write-config \
  --artifact-root "$FACTORY_ROOT" \
  --embedding-config "$EMBEDDING_CONFIG" \
  --embedding-config-sha256 "$EMBEDDING_CONFIG_SHA256" \
  --development-operator-root "$DEVELOPMENT_ROOT" \
  --development-operator-receipt-sha256 "$DEVELOPMENT_RECEIPT_SHA256" \
  --partition-audit "$PARTITION_AUDIT" \
  --partition-audit-sha256 "$PARTITION_AUDIT_SHA256" \
  --runner-image "$IMAGE" \
  --hmac-secret-fd 0 \
  --output "$FACTORY_CONFIG" \
  < "$HMAC_FILE"
```

The guard array has been exercised with the non-root UID/GID recipe: the root filesystem is
read-only, networking and Linux capabilities are absent, privilege escalation is disabled, and
only a bounded `/tmp` tmpfs plus the declared output binds are writable.

## Copy boundary

The writable artifact root must already exist, be owned by the runner, have no group or other
write bit, and contain no entries when `build` starts. The factory copies the admitted embedding
tree into `.embedding-stores.partial` with exclusive file creation. Source files must be regular,
singly linked files; source directories may not contain symlinks, devices, sockets, or FIFOs.
Each source file is checked before and after its copy.

After the copy, the factory hashes the source tree again, hashes the destination tree, and
requires both digests to equal the admitted source pin. It then publishes the destination as
`embedding-stores/` with a no-replace directory rename. An existing partial tree is terminal for
that artifact root. The operator must retain it for inspection and start with a new empty root;
the factory never guesses which partial bytes are safe to keep.

## Fixed build sequence

For each corpus in protocol order, the factory performs these operations:

1. reproduce the `fit` and `calibration` development plans;
2. create or verify the matching compiled-policy package;
3. build the authorized-index store three times in distinct roots with the fixed one-thread
   hnswlib config;
4. compare every HNSW digest, row-map digest, build-binding digest, store receipt, and tree digest;
5. byte-copy replica 1 into the selected authorized-index root and rehash that copy;
6. create the sealed query package with the committed HMAC key, then assert that its receipt
   carries the exact derived key ID;
7. create the label-free online execution package, the sealed policy package, three sealed index
   replicas, the stage bundles, and the runtime-admission receipt;
8. write one exclusive corpus evidence receipt.

Before the online package is published, the factory also builds its full-active HNSW three times in
`authorized-index-reproducibility/<corpus>/full-active/`. A typed receipt binds the source-vector
digest, document count, dimension, backend extension, format revision, all three byte digests, and
replica 1 as the selected copy. Only that exact copy enters `custody/online/<corpus>/`; the other two
copies remain reproducibility evidence rather than deployable inputs.

The complete suite contains 60 isolated HNSW builds: 45 authorized-index builds (five corpora times
three stages times three replicas) plus 15 full-active builds (five corpora times three replicas). A
resumed run verifies every existing whole-artifact boundary before moving forward. It never
overwrites a receipt or silently rebuilds an object whose registered path is already present.

## Development HNSW feasibility measurement

A 2026-07-17 development measurement tested the registered HNSW numerical settings before C0. The
host was the Apple M4 Max used for the embedding feasibility work. The environment used hnswlib
`0.8.0` with extension SHA-256
`5132b1579ee553977a4a8b46c4a04c73e38146a0bba24c67d37f0096a863cf86`, NumPy `2.5.1`, macOS
`26.3.1`, and arm64. Each input was a seed-`20260714` matrix of normalized random float32 vectors;
the timed interval covered only `add_items`. Every index used cosine distance, dimension 256,
`M=16`, `efConstruction=128`, HNSW seed `20260714`, integer labels `0..N-1`, and one thread.

| Rows | Build time | Throughput | Serialized bytes/row | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 20,000 | 11.832 s | 1,690.36 rows/s | 1,172.551 | 139.1 MiB |
| 100,000 | 87.493 s | 1,142.95 rows/s | 1,172.663 | 540.8 MiB |
| 250,000 | 249.634 s | 1,001.47 rows/s | 1,172.577 | 1,285.5 MiB |

The serialized files contained 23,451,012, 117,266,304, and 293,144,196 bytes. Their SHA-256
values, in ascending row-count order, were
`8d64711c6fb7ddaf7870e99a0a616e47dc0232c1edb75679348fb042f7c8f5d8`,
`fd5599493f1b7c09de1a372a1d30763235d1c64cbe2e5c16686690ac6519d7d5`, and
`61e2a7f230d42cae6ed7ec2caeebd551ce1611afea9d0270936d072e4e0d0c70`.

A linear fit of seconds per row against `ln(N)` projects 665 rows/s and 2.18 hours for one
5.23-million-row build. Applying that rate to the deliberately conservative ceiling of twelve
full-corpus build equivalents per corpus gives 32.7 build-hours for 78,220,668 indexed-row
equivalents. Corpus-lane scheduling can reduce wall time because the five lanes are disjoint. The
projection excludes vector preparation, artifact copies, hashing, admission passes, and any change
caused by the real embedding geometry, so it does not turn the 12–36 hour planning interval into a
guarantee. Terminal receipts remain the authority for elapsed time and resource use.

The same measurement serialized between 1,172.551 and 1,172.663 bytes per indexed row. The retained
topology contains 40 HNSW files per corpus: three policy stages times four copies (three
reproducibility replicas and one selected authorized-index copy) times three policy masks produce
36 files, while three full-active reproducibility replicas and the selected online copy produce
four more. The masks admit `0.25N`, `0.50N`, and `0.75N` rows. Retained authorized-index files
therefore account for `18N` indexed rows and full-active files for another `4N`, or `22N` in total.
Across the current five-corpus receipts, `N = 6,518,389`; applying the measured 1,172.663-byte
ceiling gives 168,165,219,198 bytes, or 156.616 GiB, of retained HNSW payloads.

The current factory capacity snapshot adds 0.874 GiB of `int64` row maps, a 13.618-GiB exact copy
of the embedding tree, 12.433 GiB of online paired document vectors, and at most 2.780 GiB of
copied online-projection data. Those known components total 186.321 GiB; bounded controls and query
artifacts raise the operational planning figure to about 186.5 GiB. The preceding post-embedding
development stage retains another `3N` HNSW rows and `3N` row-map entries, or 21.502 GiB at the
same ceiling. A 260-GiB free-space start gate leaves 52.176 GiB above those two planning figures for
filesystem allocation, unpublished staging, custody views, and other bounded outputs. These are
capacity calculations, not claims about final production bytes. The operator must measure free
space again immediately before construction and retain that margin throughout the build.

If a process is killed during an authorized-index build, the kernel releases its advisory lock but
the exclusive lock file and tokenized staging directory may remain. `resume` or `resume-shard`
acquires the corpus lane first, proves that the old builder lock is no longer held, checks the exact
owner, mode, link count, name, and member types of the unpublished staging tree, and removes only
those temporary names. Cleanup first renames the checked inode to a fresh recovery-only name through
the pinned parent descriptor, proves that the renamed inode is still the open staging tree, and only
then deletes it. The interrupted replica is then rebuilt. A published replica, selected copy,
online package, reproducibility receipt, or corpus evidence file is never deleted or repaired. The
same corpus-locked rule discards an unpublished selected-copy, full-active-replica, or online-package
staging directory after a power interruption. A live lock, missing lock, malformed name, link,
special file, ownership change, or permissive mode fails closed.

The sequential `build` command remains the reference construction path. The sharded path below
calls the same corpus builder and the same terminal assembler. It changes scheduling, not the
policy, query, index, runtime, receipt, or final package semantics.

The final suite receipt binds the source and destination embedding-tree digests, embedding-suite
receipt, candidate image, platform, HMAC key ID and secret commitment, online inventory, index
reproducibility suite, artifact-pipeline receipt, and five corpus evidence rows. Verification
reproduces that receipt without writing.

## Five-corpus shard and aggregate path

Corpus construction can run as five independent C0 processes after one sequential preparation
step. Preparation admits the full factory config, rehashes the entire read-only embedding source,
copies that source once, creates the declared corpus roots, checks the installed HNSW binary, and
publishes exactly five canonical request files. Each request is derived from the full config. It
contains no caller-selected seed, stage, replica, model, family count, backend parameter, or output
redirect.

Each request binds:

- the factory-config digest and absolute artifact root;
- one member of `FIXED_CORPORA`;
- the candidate image and `linux/arm64` platform;
- the embedding source-tree and suite-receipt digests;
- both the HMAC secret commitment and its derived key ID;
- the exact six corpus-owned destinations for policy, index, runtime, online, reproducibility, and
  corpus-evidence artifacts.

The six destination sets are pairwise disjoint. A worker takes a nonblocking advisory lock on its
precreated reproducibility directory and never writes a suite receipt, aggregate reproducibility
receipt, or artifact-pipeline receipt. A second worker for the same corpus fails before it can
write. Workers for different corpora may run at the same time. They read the shared immutable
embedding copy and upstream source mounts, then write only corpus-namespaced entries.
All workers must see one filesystem with coherent POSIX `flock` behavior. Do not distribute these
lanes across clients whose advisory locks are host-local.

Run preparation once, with the request directory outside `artifact_root`:

```bash
docker run --rm \
  --platform linux/arm64 \
  --user "$HOST_UID:$HOST_GID" \
  "${FACTORY_CONTAINER_GUARDS[@]}" \
  --entrypoint /opt/venv/bin/python \
  "${FACTORY_INPUT_MOUNTS[@]}" \
  --mount "type=bind,src=$FACTORY_ROOT,dst=$FACTORY_ROOT" \
  --mount "type=bind,src=$FACTORY_CONTROL_ROOT,dst=$FACTORY_CONTROL_ROOT" \
  "$IMAGE" \
  -m fractal_ann_diagnostics.production_artifact_factory \
  prepare-shards \
  --config "$FACTORY_CONFIG" \
  --config-sha256 "$FACTORY_CONFIG_SHA256" \
  --request-directory "$FACTORY_CONTROL_ROOT/requests"
```

Preparation emits these files in protocol order:

```text
01-scifact.json
02-hotpotqa-fullwiki.json
03-t2-ragbench.json
04-bright.json
05-miracl-transfer.json
```

Create the private receipt directory once, before any worker starts:

```bash
mkdir -m 0700 "$FACTORY_CONTROL_ROOT/receipts"
```

Launch one worker per request. The request digest is an external byte pin, not a corpus selector.
The same private HMAC bytes go to descriptor 0 for every worker. Give each worker a distinct
receipt output outside `artifact_root`:

```bash
docker run --rm -i \
  --platform linux/arm64 \
  --user "$HOST_UID:$HOST_GID" \
  "${FACTORY_CONTAINER_GUARDS[@]}" \
  --entrypoint /opt/venv/bin/python \
  "${FACTORY_INPUT_MOUNTS[@]}" \
  --mount "type=bind,src=$FACTORY_ROOT,dst=$FACTORY_ROOT" \
  --mount "type=bind,src=$FACTORY_CONTROL_ROOT,dst=$FACTORY_CONTROL_ROOT" \
  "$IMAGE" \
  -m fractal_ann_diagnostics.production_artifact_factory \
  build-shard \
  --config "$FACTORY_CONFIG" \
  --config-sha256 "$FACTORY_CONFIG_SHA256" \
  --request "$FACTORY_CONTROL_ROOT/requests/01-scifact.json" \
  --request-sha256 "$SCIFACT_REQUEST_SHA256" \
  --hmac-secret-fd 0 \
  --receipt-output "$FACTORY_CONTROL_ROOT/receipts/01-scifact.json" \
  < "$HMAC_FILE"
```

Use `resume-shard` with the same request, request digest, secret bytes, and receipt destination if a
worker stops at a verified whole-artifact boundary. A completed receipt is deterministic over the
current corpus tree. It pins the request, config, runner, HMAC commitment and key ID, canonical
corpus evidence, and all six owned artifact digests. It contains no secret bytes.

After all workers exit, aggregate once. Receipt arguments may arrive in any order. Aggregation
requires exactly one receipt for each fixed corpus, acquires all five locks, re-admits the entire
upstream suite, rehashes the embedding copy, reproduces every corpus artifact, compares every
owned-tree digest, and only then writes the three shared terminal receipts:

```bash
docker run --rm \
  --platform linux/arm64 \
  --user "$HOST_UID:$HOST_GID" \
  "${FACTORY_CONTAINER_GUARDS[@]}" \
  --entrypoint /opt/venv/bin/python \
  "${FACTORY_INPUT_MOUNTS[@]}" \
  --mount "type=bind,src=$FACTORY_ROOT,dst=$FACTORY_ROOT" \
  --mount "type=bind,src=$FACTORY_CONTROL_ROOT,dst=$FACTORY_CONTROL_ROOT" \
  "$IMAGE" \
  -m fractal_ann_diagnostics.production_artifact_factory \
  aggregate-shards \
  --config "$FACTORY_CONFIG" \
  --config-sha256 "$FACTORY_CONFIG_SHA256" \
  --shard-receipt "$FACTORY_CONTROL_ROOT/receipts/05-miracl-transfer.json" \
  --shard-receipt "$FACTORY_CONTROL_ROOT/receipts/02-hotpotqa-fullwiki.json" \
  --shard-receipt "$FACTORY_CONTROL_ROOT/receipts/01-scifact.json" \
  --shard-receipt "$FACTORY_CONTROL_ROOT/receipts/04-bright.json" \
  --shard-receipt "$FACTORY_CONTROL_ROOT/receipts/03-t2-ragbench.json"
```

Missing, duplicate, extra, replayed, wrong-config, wrong-secret, and partial-tree evidence fails
before terminal publication. The aggregator orders verified evidence by `FIXED_CORPORA` and calls
the sequential builder's terminal assembler, so equal corpus bytes produce the same terminal suite
receipt regardless of worker completion order.

The three replicas of each HNSW artifact inside one corpus remain sequential. Their one-thread
numerical contract, per-replica peak-memory evidence, and selected-replica comparison are defined
inside one process. Concurrent child processes would need a separately frozen scheduler,
memory-isolation rule, and resource-evidence model. That extra mechanism is not part of C0.
Parallelizing the five closed corpus lanes captures the safe scheduling gain: the factory duration
becomes approximately the slowest corpus lane plus preparation and aggregation, rather than the sum
of all five lanes.

## Secret transport

For a new build, pass 32–4096 raw bytes only on file descriptor 0. The CLI rejects every other
descriptor. It reads to EOF before admitting an upstream input or creating an output, hashes the
bytes, and compares the digest with `hmac_secret_sha256`.

Before the first invocation, verify that `HMAC_FILE` is an owned, private, non-symbolic, singly
linked regular file outside every input, artifact, and control root. The path is never mounted into
the container; host-side stdin redirection is the only transport.

```bash
docker run --rm -i \
  --platform linux/arm64 \
  --user "$HOST_UID:$HOST_GID" \
  "${FACTORY_CONTAINER_GUARDS[@]}" \
  --entrypoint /opt/venv/bin/python \
  "${FACTORY_INPUT_MOUNTS[@]}" \
  --mount "type=bind,src=$FACTORY_ROOT,dst=$FACTORY_ROOT" \
  --mount "type=bind,src=$FACTORY_CONTROL_ROOT,dst=$FACTORY_CONTROL_ROOT" \
  "$IMAGE" \
  -m fractal_ann_diagnostics.production_artifact_factory \
  build \
  --config "$FACTORY_CONFIG" \
  --config-sha256 "$FACTORY_CONFIG_SHA256" \
  --hmac-secret-fd 0 \
  < "$HMAC_FILE"
```

Do not place the secret in an argument, environment variable, config, receipt, log, image layer,
Git tree, or shell history. Retain the original bytes in the private C1 custody set because the
sealed online runner must reproduce the same opaque identifiers.

`resume` may omit standard input only when all five query packages already exist and verify. If
even one is absent, supply the same secret through descriptor 0. The command checks its
commitment before any further write.

```bash
docker run --rm -i \
  --platform linux/arm64 \
  --user "$HOST_UID:$HOST_GID" \
  "${FACTORY_CONTAINER_GUARDS[@]}" \
  --entrypoint /opt/venv/bin/python \
  "${FACTORY_INPUT_MOUNTS[@]}" \
  --mount "type=bind,src=$FACTORY_ROOT,dst=$FACTORY_ROOT" \
  --mount "type=bind,src=$FACTORY_CONTROL_ROOT,dst=$FACTORY_CONTROL_ROOT" \
  "$IMAGE" \
  -m fractal_ann_diagnostics.production_artifact_factory \
  resume \
  --config "$FACTORY_CONFIG" \
  --config-sha256 "$FACTORY_CONFIG_SHA256" \
  --hmac-secret-fd 0 \
  < "$HMAC_FILE"

docker run --rm \
  --platform linux/arm64 \
  --user "$HOST_UID:$HOST_GID" \
  "${FACTORY_CONTAINER_GUARDS[@]}" \
  --entrypoint /opt/venv/bin/python \
  "${FACTORY_INPUT_MOUNTS[@]}" \
  --mount "type=bind,src=$FACTORY_ROOT,dst=$FACTORY_ROOT,readonly" \
  --mount "type=bind,src=$FACTORY_CONTROL_ROOT,dst=$FACTORY_CONTROL_ROOT,readonly" \
  "$IMAGE" \
  -m fractal_ann_diagnostics.production_artifact_factory \
  verify \
  --config "$FACTORY_CONFIG" \
  --config-sha256 "$FACTORY_CONFIG_SHA256"

docker run --rm \
  --platform linux/arm64 \
  --user "$HOST_UID:$HOST_GID" \
  "${FACTORY_CONTAINER_GUARDS[@]}" \
  --entrypoint /opt/venv/bin/python \
  "${FACTORY_INPUT_MOUNTS[@]}" \
  --mount "type=bind,src=$FACTORY_ROOT,dst=$FACTORY_ROOT,readonly" \
  --mount "type=bind,src=$FACTORY_CONTROL_ROOT,dst=$FACTORY_CONTROL_ROOT,readonly" \
  "$IMAGE" \
  -m fractal_ann_diagnostics.production_artifact_factory \
  status \
  --config "$FACTORY_CONFIG" \
  --config-sha256 "$FACTORY_CONFIG_SHA256"
```

`status` only inventories declared phase paths. It does not admit inputs, open the secret, repair
an interrupted tree, or imply that an artifact passed verification.

## P, C0, and C1 boundary

Bootstrap source commit P fixes this factory, the numerical stack, hnswlib extension, policy engine,
and candidate container identity. The factory config cites the published candidate OCI digest and
admits no alternate platform. The factory may run only after candidate publication and its
anonymous digest-read gate have passed. Its outputs help close the raw candidate manifest later
committed at apparatus commit A; they therefore cannot depend on A.

C1 can cite a factory output only after `verify` succeeds from the candidate image, the exact OCI
index D is promoted without rebuilding into the production repository at C0, and an independent
artifact inventory pins every resulting tree and receipt. The later runtime-preflight transition
may observe host facts and bind the final execution plan. It may not change a factory artifact,
query key identity, corpus order, policy stage, selected HNSW bytes, or embedding source. C0 alone
does not authorize a sealed attempt, and a complete factory receipt is not evidence that labels
were released or that an analysis ran.
