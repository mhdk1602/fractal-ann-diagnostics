# C0 confirmatory runner image

The C0 runner is the executable environment committed before registration. One source commit
produces two disjoint OCI subjects: a multi-platform scientific image and an ARM64-only timelock
release image. C1 cites both digest-qualified subjects. A mutable tag, including a 40-character
commit tag, is never sufficient evidence of identity.

This apparatus builds the image. It does not open labels, admit C1, start a trial, or imply
that a confirmatory execution has occurred.

## Frozen construction

`Dockerfile.confirmatory` is a build graph, not a four-stage wrapper. Separate discarded stages
build hnswlib, SQLite, zlib, OPA, and `tle` from pinned sources. A Python assembly stage inventories
the non-distroless shared libraries and Debian package records that the scientific runtime needs.
The final scientific image starts from a signed, digest-pinned distroless base and receives only
the virtual environment, admitted native objects, OPA, compiled Rego, `uv.lock`, runtime receipts,
and `src/`. It receives no compiler, package manager, Git history, research data, or credential.

The release image inherits that scientific filesystem and adds one source-built static `tle`
binary plus its build receipt. The default `runtime` target points back to the TLE-free scientific
base. This split is structural: label release can use the ARM64 release subject, while corpus
execution and analysis cannot recover a beacon client or TLE binary from their image. The Rego
test module enters only a networkless build-time bind mount and is absent from both runtime
layers.

| Input | Immutable identity | Official source |
|---|---|---|
| Python builder | `python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b` | [Python 3.12.13 release](https://www.python.org/downloads/release/python-31213/); [Docker Official Image](https://hub.docker.com/_/python) |
| uv builder | `ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc` | [uv container](https://github.com/astral-sh/uv/pkgs/container/uv) |
| Go builder | `golang:1.26.5-bookworm@sha256:1ecb7edf62a0408027bd5729dfd6b1b8766e578e8df93995b225dfd0944eb651` | [Go 1.26.5 release](https://go.dev/doc/devel/release#go1.26.5); [Docker Official Image](https://hub.docker.com/_/golang) |
| Final base | `gcr.io/distroless/base-nossl-debian12:nonroot@sha256:26cd77482910e221ff26cf7c480203ce97f8f01ad272e2dc8a9ae29c811e9efe` | [distroless](https://github.com/GoogleContainerTools/distroless) |
| OPA source | commit `e695c9ef8edb0f8b9f13d014d7bc8a7fbcc57297`, archive SHA-256 `a8b3ecdc925b75bdade52d315aa13efaa51c2de99acb78003ad353cce6e9e637` | [OPA repository](https://github.com/open-policy-agent/opa) |
| TLE source | v1.2.0 commit `7b54141a9733fd6fa207587a11148280e6fb020d`, archive SHA-256 `98b5edb760cffbe6edd392f004d2d51fcc7a8e6ef7ed7672c32b1a9e1ce3e32d` | [tlock v1.2.0](https://github.com/drand/tlock/releases/tag/v1.2.0) |
| Official TLE interoperability fixture | ARM64 archive SHA-256 `3b724032620587c2551ee857c98dc02690076f4972a4fe4389b0f6e0911a6a92`; binary SHA-256 `e153cfa8539e871f50143d1bde10fec7ec3fe82630f717c3c1bf166eb4975059` | [tlock v1.2.0 assets](https://github.com/drand/tlock/releases/tag/v1.2.0) |
| Compiled Rego | `examples/opa_compiled_masks.rego` SHA-256 `18f6eb8a7411a7a1415bd2425ad5720f28fcd3b428d9aa2c1e7d73f6e14e356c` | This C0 source tree |
| Rego contract tests | `examples/opa_compiled_masks_test.rego` SHA-256 `67370adfcba1c5180bdc99ae2cab900785ec5cee6fd91a9a4a9058415a7d4f00` | This C0 source tree |
| Python resolution | `uv.lock` SHA-256 `a7251c8ce2b54888a047daefb32a2584c6d3f596030dd6cd87e46693b7ca57d6` | [uv lock format](https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile) |
| hnswlib source | 0.8.0 sdist SHA-256 `cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c` | [PyPI release record](https://pypi.org/project/hnswlib/0.8.0/#files) |
| Native toolchain repository | Debian snapshot `20260714T000000Z`, Bookworm main | [Debian Snapshot](https://snapshot.debian.org/archive/debian/20260714T000000Z/) |
| hnswlib build wheels | `requirements.confirmatory-build.txt`, exact versions and SHA-256 hashes | [PyPI JSON API](https://docs.pypi.org/api/json/) |

The lock also resolves the `production-embedding` optional set for the separate macOS arm64 MPS
builder. `Dockerfile.confirmatory` does not select that extra, does not install PyTorch or
Transformers, and calls `uv pip check` after removing every build-only distribution. It then
requires the installed set to equal hnswlib 0.8.0, joblib 1.5.3, NumPy 2.5.1, scikit-learn 1.9.0,
SciPy 1.18.0, narwhals 2.24.0, threadpoolctl 3.6.0, and tqdm 4.68.4. The final nonroot,
networkless build gate also
proves that the `torch` and `transformers` module specifications are absent. C0 cannot claim to
have produced the pre-C1 embedding matrices. The exact MPS builder receipt and commands are
specified in [embedding-store.md](embedding-store.md). The shared lock hash binds both resolutions;
the selected extra and closed installed-package set distinguish their execution roles.

Snapshot transport is HTTP; transport encryption is not the trust boundary. The Dockerfile first
pins the dated `InRelease` bytes and Debian archive keyring by SHA-256, then verifies the Release
signature with `gpgv`. APT rejects insecure or unauthenticated repositories and authenticates the
package indexes against that Release file. `APT::Update::Error-Mode=any` converts an unavailable
or unauthenticated index into a build failure. The builder then checks that
`g++` version `4:12.2.0-3` was installed and is executable; an empty package index cannot
fall through to the Python build.

`HNSWLIB_NO_NATIVE=1` suppresses hnswlib's default `-march=native`. The resulting extension
does not inherit the instruction set of the GitHub-hosted builder. The builder extracts the
checksum-constrained sdist to `/build/hnswlib-source/hnswlib-0.8.0`; it never asks `uv` to unpack
the sdist into a random temporary path. `CFLAGS` and `CXXFLAGS` remove debug information and map
that stable build root to `/usr/src/hnswlib-source`. The builder creates exactly one wheel,
installs that wheel, and verifies that the wheel's extension bytes equal the imported extension
bytes. It records the sdist, wheel, extension, Python ABI, sizes, and SHA-256 values in
`/opt/artifacts/hnswlib-runtime-receipt.json`. Build-only Python packages are removed after
compilation. `uv_cache.json`, whose installation-time field is not runtime state, is removed with
its `RECORD` row. The final stage copies `libstdc++`, `libgcc_s`, and `libgomp` from the same
snapshot-backed builder because hnswlib links against C++ and OpenMP runtimes.

The workflow reads `SOURCE_DATE_EPOCH` from the tracked
`confirmatory-source-date-epoch.txt` build input and passes it to the build and image exporter.
The file contains one canonical positive Unix timestamp and is itself part of the admitted build
context. Every wheel member must carry the ZIP-representable source epoch or the build fails. The
registry exporter sets `rewrite-timestamp=true`, so generated layer metadata uses the same epoch.
The hnsw wheel and executable layers do not inherit either the workflow start time or the C0
commit timestamp.

The executable OCI config binds to the SHA-256 of the admitted Git tree records for the complete
build-context allowlist. That digest populates
`io.fractal-ann.confirmatory.build-context-tree-sha256` and the OCI revision/version labels.
The C0 Git SHA remains outside the executable closure in checkout checks, image tags, source
attestations, and `c0-sha.txt`. Consequently, two builds from identical admitted context produce
the same executable manifests even if only `research/study-manifest.json` changed between their
source commits.
The environment's `python` is a copied regular executable rather than a `uv`-created symlink or
hard link. Build checks require `lstat` to report a regular file, link count one, mode `0555`,
`sys.executable == "/opt/venv/bin/python"`, and `sys.prefix == "/opt/venv"`.

The final image copies `/opt/venv`, `/opt/native-libs`, `/opt/artifacts`, `/opt/app/src`, the exact
verified lock at `/opt/app/uv.lock`, the OPA binary, and the Rego module as `root:root`. It rehashes
the lock, removes every write bit from those paths, then drops
to UID/GID 65532 during the build and checks that none is writable. The OPA binary is mode `0555`;
the Rego module is mode `0444` and is rehashed in the image. Runner-owned directories are limited
to `/home/runner`, `/input`, `/output`, and `/workspace`. `/tmp` becomes writable only when the
launcher supplies the registered disposable tmpfs.

The executable comparison admits a closed runtime config rather than a blacklist.
Its only fields are `ArgsEscaped`, `Cmd`, `Entrypoint`, `Env`, `Labels`, `User`, and `WorkingDir`.
The [pinned BuildKit
frontend](https://github.com/moby/buildkit/blob/dd2170e156c9633da1b2d1a58a6188e3f7d36fa4/frontend/dockerfile/dockerfile2llb/convert.go#L1577-L1590)
emits `ArgsEscaped=true` when it processes the JSON-form `CMD`; the comparison requires that exact
compatibility marker rather than admitting another value. The label map must equal the C0
Dockerfile's declared map. The environment key set must equal the runner variables declared on the
distroless final stage. No
additional environment name, label, stop signal, port, volume, health check, shell override, or
`ONBUILD` instruction is admitted. The receipt retains every observed label and environment value.

The final assembly test in `source-runtime` runs without network access as UID/GID 65532. It imports the CLI and
the pinned numerical/native packages, evaluates the CLI help path, rechecks interpreter identity
and immutability, verifies the Rego test digest, and executes `opa test` against the shipped policy.
Any failed import, policy assertion, ownership check, or filesystem check aborts publication.

## Platform decision

The published object is a two-platform OCI index: `linux/amd64` and `linux/arm64`.
The pinned Python and uv indexes publish both variants. For CPython 3.12, `uv.lock` records
Linux wheels for both architectures for NumPy 2.5.1, SciPy 1.18.0, and scikit-learn 1.9.0.
hnswlib 0.8.0 is compiled once per target architecture with native instruction selection
disabled. No third platform is admitted by the workflow.

## Publication toolchain

The job runs only on GitHub's `ubuntu-24.04` hosted runner. A hosted runner image can change, so
the job records `ImageOS`, `ImageVersion`, kernel data, Docker versions, and the GitHub runner
identity. The build tools that affect image bytes are fixed separately:

| Tool | Fixed identity |
|---|---|
| Buildx | v0.34.1 linux-amd64 binary SHA-256 `f1332ddb9010bd0b72628266c3a906d9a6979848033df4c8d9bd2cd113bae12b` |
| BuildKit | `moby/buildkit@sha256:0168606be2315b7c807a03b3d8aa79beefdb31c98740cebdffdfeebf31190c9f` |
| binfmt/QEMU | `tonistiigi/binfmt@sha256:400a4873b838d1b89194d982c45e5fb3cda4593fbfd7e08a02e76b03b21166f0` |

Buildx verifies its own downloaded binary before use. The job checks the BuildKit and binfmt
versions inside the digest-pinned containers. QEMU is enabled only for `linux/arm64`; amd64 is the
host architecture. Binary, image, and build caches are disabled. The build action receives an
empty GitHub context token and the Docker context is constrained by the two identical sealed
allowlists, `.dockerignore` and `.dockerignore.confirmatory`. Before any setup action, the workflow
compares the checked-out regular-file inventory with `git ls-files`, then materializes the admitted
inputs from `git archive <C0-SHA>` into a separate context. Its NUL-delimited `git ls-tree` manifest
binds every path, Git mode, and blob ID. File timestamps are normalized to the source epoch;
directory modes are fixed at `0555`. A hash-bound verifier rejects symbolic links, multiply linked
files, special files, changed bytes, changed modes or timestamps, missing Git blobs, and any extra
path. Both ignore files are in the checked set. The verifier runs before and after each independent
arm64 build, after registry authentication and immediately before the multi-platform build, and
again when the registry output is read back. Every builder consumes this separate context; none
reads the mutable checkout. A clean `git status` is necessary but is not treated as proof that
ignored files are absent.

### Reproducibility acceptance

The C0 tag fixes the source and workflow commit. Before registry publication, the workflow creates
two fresh BuildKit daemons and builds `linux/arm64` twice without cache from that source, with the
same tag, source epoch, build arguments, and OCI exporter settings. It compares the executable
platform manifest digest, config digest, ordered layer descriptors, runtime files, retained hnsw
wheel, and imported extension. Any difference blocks publication. The two local exporters and the
later registry exporter all set `oci-mediatypes=true`; otherwise equivalent files could acquire
different manifest digests solely because one exporter selected Docker media types.

Report BuildKit provenance and SBOM attestation manifests separately. Those documents may contain
generation-time evidence, so their digests and the enclosing OCI index digest are not evidence of
executable-layer reproducibility. Each attestation must still name the exact executable platform
manifest produced by its own build. Never report the outer index as reproducible unless its digest
was also equal in the two retained exports.

The arm64 gate operates on the two retained OCI-layout tar archives, without extracting either
archive into the host filesystem:

```bash
python -m fractal_ann_diagnostics.c0_reproducibility \
  --archive-a /controlled/c0-proof/arm64-build-a.oci.tar \
  --archive-b /controlled/c0-proof/arm64-build-b.oci.tar \
  --expected-build-context-tree-sha256 "$(sha256sum build-context-git-tree.z | cut -d ' ' -f 1)" \
  --expected-source-date-epoch "$(<confirmatory-source-date-epoch.txt)" \
  --expected-uv-lock-sha256 a7251c8ce2b54888a047daefb32a2584c6d3f596030dd6cd87e46693b7ca57d6 \
  --expected-opa-policy-sha256 18f6eb8a7411a7a1415bd2425ad5720f28fcd3b428d9aa2c1e7d73f6e14e356c \
  --output /controlled/c0-proof/arm64-executable-reproducibility.json
```

Both input paths must be distinct, absolute, regular, singly linked files. The output parent must
already exist, and the receipt is created once at mode `0444`; an existing output is never
replaced. The gate accepts either direct platform descriptors or Buildx's single unplatformed
wrapper descriptor, then traverses that wrapper to the execution index. It requires one
`linux/arm64` executable manifest. Other execution-index members are admitted only as
`unknown/unknown` BuildKit attestation manifests that point back to that executable digest.
An amd64 image, a second arm64 image, an untyped index member, or an attestation referring to a
different executable closes the gate.

For every reachable descriptor, the gate verifies the declared size and SHA-256 against the exact
blob in the OCI tar. It rejects unreferenced blobs and unsupported executable-layer compression.
It then checks the arm64 config, C0 labels, source epoch, ordered layer descriptors, and
`rootfs.diff_ids`. It also requires UID/GID `65532:65532`, working directory `/workspace`, the
fixed exec-form Python entrypoint, the `--help` default command, the registered runtime environment,
and the absence of populated ports, volumes, health checks, and `ONBUILD` instructions. The OPA,
Python, and hnsw extension bytes must be AArch64 ELF objects. The layer reader implements OCI
whiteouts in memory and materializes no host path. Its final-state projection contains the exact
bytes, size, digest, and mode of:

```text
/usr/local/bin/opa
/opt/venv/bin/python
/opt/app/uv.lock
/opt/app/policy/opa_compiled_masks.rego
/opt/artifacts/hnswlib-runtime-receipt.json
/opt/artifacts/hnswlib/<receipt-named-wheel>.whl
```

The inner hnsw receipt must be canonical and must cross-hash the one retained wheel. The wheel is
opened as a ZIP archive, its member paths and source-epoch timestamps are checked, and its named
extension must have the byte count and digest claimed for the imported extension. A target reached
through a link, duplicate member, device, traversal component, malformed whiteout, or ambiguous
file/directory state is rejected.

Equality covers the executable manifest digest, config descriptor, ordered layer descriptors,
rootfs diff IDs, all six retained runtime files, and the imported-extension digest. If any field
differs, the command exits without creating the receipt. Each archive's full-file SHA-256, outer
index SHA-256, and attestation descriptor set is still written into the successful receipt.
`outer_index_equal` and `attestation_metadata_equal` are informational booleans. Their values do
not alter `executable_equal`, so generation-time provenance or SBOM variation cannot be mistaken
for executable drift.

The local OCI tar files are deleted after comparison because each can be large. Their full-file
hashes, builder descriptions, build metadata, plain logs, and the read-only comparison receipt are
retained. Candidate mode publishes one bootstrap scientific index and one bootstrap release index.
Their closure binds the admitted build-context tree SHA-256, both complete index digests, and all
three executable platform-manifest digests. BuildKit provenance and SBOM members may contain build
timestamps, so production does not attempt to reproduce those enclosing index bytes.

Production still runs the two local no-cache arm64 projection checks against its admitted context.
It then copies the authenticated bootstrap indexes into the final GHCR repositories without a
Docker build. The copy is admitted only when the final tags are absent. Candidate, promoted, and
anonymous-readback raw index bytes must be identical and hash to the closure digests; normalized
platform inventories must also agree. `oci-promotion-receipt.json` binds the bootstrap source
commit, current C0 commit, context-tree digest, index digests, platform manifests, raw-index hashes,
and anonymous readback. A partial copy emits `fractal-c0-oci-promotion-incident-v1` evidence and
cannot be retried over an existing tag. Recovery requires a separately recorded audit and tag
removal decision.

## Manual publication

Only `workflow_dispatch` can publish these images. Candidate mode runs first from the exact
`c0-candidate/*` branch head P. The `c0_sha` input is P in this mode:

```bash
gh workflow run confirmatory-image.yml \
  --repo mhdk1602/fractal-ann-diagnostics \
  --ref c0-candidate/v0.3-bootstrap \
  -f c0_sha=<40-character-bootstrap-source-P> \
  -f mode=candidate
```

Retain the successful candidate run ID and its closure. After raw candidate bytes are committed at
A, provider rehearsal succeeds, and the fixed C0 tag points to A, production mode promotes the
already authenticated index digests without rebuilding. The `c0_sha` input is A in this mode:

```bash
gh workflow run confirmatory-image.yml \
  --repo mhdk1602/fractal-ann-diagnostics \
  --ref confirmatory-apparatus-c0 \
  -f c0_sha=<40-character-C0-SHA> \
  -f mode=production \
  -f candidate_run_id=<successful-candidate-image-run-id> \
  -f rehearsal_run_id=<successful-provider-rehearsal-run-id>
```

Production admission authenticates both prior runs through the GitHub API. The candidate image run
may come from an earlier bootstrap commit. Its closure must carry the same
`build_context_tree_sha256` as the production checkout; its Git SHA remains external provenance.
The provider rehearsal must be a successful attempt-one run of
`confirmatory-provider-rehearsal.yml` bound to that bootstrap closure. Its aggregate artifact is
downloaded by numeric artifact ID, checked against GitHub's
artifact SHA-256, restricted to its three fixed members, and verified with its closed checksum
file. The canonical aggregate must bind all three phase jobs to the same candidate closure while
recording that neither scientific inputs nor provider state were opened. Finally, the retained
Sigstore bundle is verified under the exact provider-rehearsal workflow identity, source ref,
source digest, and GitHub-hosted completion job. Candidate mode rejects both run-ID inputs.

Before checkout, the workflow requires repository `mhdk1602/fractal-ann-diagnostics`, actor and
triggering actor `mhdk1602`, event `workflow_dispatch`, ref
`refs/tags/confirmatory-apparatus-c0`, and equality among the input SHA, source SHA, and workflow
SHA. This prevents running a modified workflow from `master` against an older source commit. It
then checks out the tag rather than the untrusted dispatch input, peels that exact tag to the
supplied commit, and requires the complete admitted build-input inventory to match Git. It builds
without cache and publishes exactly

```text
ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory:<C0-SHA>
```

and uploads a retained artifact package. C1 must record the digest-qualified reference from that
package, for example:

```text
ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@sha256:<64-hex-digest>
```

The first personal-account GHCR publication is private by default. During the publication job,
`mhdk1602` must open the package's settings, change its visibility to **Public**, and acknowledge
that GitHub does not permit a public package to return to private visibility. This is an explicit
human gate, not an inference from the source repository's visibility. See GitHub's
[package visibility procedure](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility#configuring-visibility-of-packages-for-your-personal-account).

The job waits up to 30 minutes for that change. On each attempt it invokes the pinned Buildx binary
directly with an isolated Docker configuration whose `config.json` is the empty object. The child
process starts with an empty environment, then receives only `DOCKER_CONFIG`, `HOME`, locale, and a
fixed `PATH`. It does not log out of the authenticated publication client because later extraction
still needs that client. The isolated client resolves the digest without `docker login` or a
credential helper:

```bash
buildx_path="$HOME/.docker/cli-plugins/docker-buildx"
image_name=ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory
image_digest="$(cat /controlled/c0-publication-record/digest.txt)"
anonymous_config="$(mktemp -d)"
printf '{}\n' > "$anonymous_config/config.json"
observed_digest="$(
  env -i \
    DOCKER_CONFIG="$anonymous_config" \
    HOME="$HOME" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/usr/bin:/bin \
    "$buildx_path" imagetools inspect "${image_name}@${image_digest}" \
    --format '{{json .Manifest}}' | jq -r .digest
)"
test "$observed_digest" = "$image_digest"
```

An authenticated pull is insufficient. C1 admission remains closed until the anonymous index is
byte-equivalent after canonicalization to the authenticated index and names the published digest.
The retained record includes the empty `config.json` and binds its hash, exact child-environment
allowlist, Buildx path, attempt log, stdout, stderr, exit status, observed digest, and verification
time. A timeout fails the publication job; rerunning the workflow would create a separate
publication attempt and is not an automatic retry of this one.

The workflow reads the registry index back and requires exactly two executable platform manifests
and one BuildKit attestation manifest per platform. It extracts and validates maximal SLSA
provenance and SPDX SBOM documents for both platforms. The SHA-pinned GitHub attestation action
then signs the exact OCI index digest through GitHub OIDC, pushes the attestation to GHCR, and
returns a Sigstore bundle. `gh attestation verify` requires the C0 workflow identity, source
digest, source ref, repository, and a GitHub-hosted runner.

The retained package contains more than the registry digest:

```text
C0-ARTIFACT-SHA256SUMS
PACKAGE-SHA256SUMS
anonymous-docker-config.json
anonymous-registry-attempts.log
anonymous-registry-index.json
anonymous-registry-index.raw.json
anonymous-registry-stderr.txt
anonymous-registry-verification.json
anonymous-release-registry-index.raw.json
c0-sha.txt
build-context-tree-sha256.txt
source-date-epoch.txt
digest.txt
image-reference.txt
platforms.txt
registry-index.json
buildkit-provenance.json
buildkit-sbom.spdx.json
oci-promotion-receipt.json                 # production only
arm64-build-a-builder.json
arm64-build-a-metadata.json
arm64-build-a.oci.tar.sha256
arm64-build-a.log
arm64-build-b-builder.json
arm64-build-b-metadata.json
arm64-build-b.oci.tar.sha256
arm64-build-b.log
arm64-executable-reproducibility.json
arm64-executable-reproducibility.stdout
published-arm64-projection.json
reproducibility-verifier-image.json
github-attestation-*
c0-artifact-attestation-*
c0-artifact-attestation-verification.jsonl
runner-environment.json
builder.json
docker-version.json
buildx-version.txt
buildkit-version.txt
binfmt-version.txt
kernel.txt
workflow-run.txt
source/fractal-ann-diagnostics-C0.tar
source/build-context-git-tree.z
source/build-context-verifier.py
runtime-artifacts/linux-amd64/{opa,python,uv.lock,runtime-extraction.json,hnswlib-runtime-receipt.json}
runtime-artifacts/linux-amd64/hnswlib/hnswlib-0.8.0-*.whl
runtime-artifacts/linux-arm64/{opa,python,uv.lock,runtime-extraction.json,hnswlib-runtime-receipt.json}
runtime-artifacts/linux-arm64/hnswlib/hnswlib-0.8.0-*.whl
```

The source tar is made by `git archive` from C0 and is rejected if it contains `.git` or
`non-git-files`. Each platform's OPA executable, Python executable, exact `uv.lock`, hnsw wheel, and
hnsw build receipt is copied from the stopped platform image addressed by its manifest digest. The
workflow also compares the extracted lock byte-for-byte with C0's checked-out `uv.lock`.
`runtime-extraction.json` binds these files to C0, the source epoch, OCI index digest, platform
manifest digest, platform, in-image paths, byte counts, and SHA-256 values. The workflow verifies
the inner hnsw receipt before writing that outer extraction receipt.
The closed schema is `fractal-c0-runtime-extraction-v2`. Version 1 is rejected because it did not
bind the Python executable or runtime lock and cannot supply their plan digests.

`C0-ARTIFACT-SHA256SUMS` covers every file present at extraction time. A second OIDC attestation
signs every subject in that list. Before package sealing, the workflow rehashes and verifies each
subject against that bundle, the exact C0 workflow identity, commit, tag, repository, and
GitHub-hosted-runner constraint. The canonical verification output is retained as JSON Lines.
`PACKAGE-SHA256SUMS` then covers those subjects, verification output, and returned attestation
bundle and identifiers. GitHub's artifact transport retains file bytes but does not retain Unix
executable modes. The C1 materializer must therefore verify the selected OPA receipt before copying
its bytes to `runtime/opa`, set the new file to mode `0555`, and rehash it. It must copy the selected
wheel bytes without rebuilding them to
`backends/hnsw/hnswlib-runtime.whl`.

After all five runtime-plan templates name one architecture, materialize OPA from the downloaded
retained package rather than pulling it anew:

```bash
python -m fractal_ann_diagnostics.opa_runtime_binary materialize-retained \
  --c0-package /controlled/c0-publication-record \
  --image "$(cat /controlled/c0-publication-record/image-reference.txt)" \
  --plan-root /controlled/fractal-v0.3/artifacts/runtime \
  --output /controlled/fractal-v0.3/artifacts/runtime/opa
```

The command selects the platform declared by all five plans. It requires regular, singly linked
package inputs; verifies the package checksum rows, closed extraction-receipt schema, C0 commit,
OCI index, platform manifest, and the size and digest of OPA, Python, and `uv.lock`; and calls
`gh attestation verify` for the receipt and all three subjects. It creates the OPA destination once
at mode `0555`, then repeats the five-plan binding check against the new file. An existing
destination, mixed platforms, modified package bytes, or a self-hosted provenance statement causes
failure before C1 can be frozen.

These records identify the GitHub-hosted workflow that published and extracted the named bytes.
They do not establish independent human custody. The Actions artifact is a 90-day transport copy,
not the durable evidence location and not a Zenodo registration member.

## Immutable C0 evidence release

After the candidate rehearsal passes and the production image workflow finishes at
`confirmatory-apparatus-c0`, enable immutable releases for the repository. Then dispatch
`.github/workflows/confirmatory-c0-evidence-release.yml` at that same tag with the full C0 commit
and successful production image run ID. The workflow rejects a disabled immutability setting,
another actor, another ref, a rerun, an unsuccessful image run, another workflow path, or an
artifact whose GitHub SHA-256 does not verify.

Production mode has two jobs. A `confirmatory` environment job on the controlled
`self-hosted`, `macOS`, `ARM64`, `confirmatory-control` runner reads the private materialization
config, instantiates the absent post-A output directory once, verifies the complete receipt-bound
tree, and uploads those exact bytes. The tree contains the two original candidate-package byte
strings at `candidate-manifest-package/candidate-study-manifest.json` and
`candidate-manifest-package/candidate-manifest-assembly-receipt.json`; both members and their parent
directory are part of the receipt-bound payload. The hosted publication job downloads that artifact,
rechecks its closed membership, payload digest, fixed candidate paths, and both candidate file
hashes, and inserts it under
`production-control-instantiation/` before either checksum manifest or GitHub artifact attestation
is created. The same hosted job reads the two live GitHub environments and their three fixed tag or
branch policies through authenticated REST calls. It admits those five response bodies with the
offline verifier and retains the canonical `github-environment-control-receipt.json` in the image
record.

The C0 release workflow rejects noncanonical ZIP names before extraction, including repeated-slash
and dot-segment aliases, then rechecks both checksum layers, the candidate-to-production identity
receipt, the environment-control receipt, and the post-A production-control instantiation tree.
That tree must contain its canonical receipt at
`production-control-instantiation/c0-control-instantiation-receipt.json`. The workflow checks A,
P, T, the candidate-closure digest, candidate-manifest digest, and fixed five-workload membership
before creating one deterministic `tar.gz`. In the archive, the candidate bytes therefore have the
fixed paths
`production-control-instantiation/candidate-manifest-package/candidate-study-manifest.json` and
`production-control-instantiation/candidate-manifest-package/candidate-manifest-assembly-receipt.json`.
The archive
uses sorted paths, normalized ownership and modes, the C0 commit epoch, PAX time-field deletion,
and timestamp-free gzip. It creates a draft release against the existing C0 tag, attaches exactly
the archive and its checksum file, inspects their draft metadata, and only then publishes. Once
published, the workflow requires the release API to report `immutable=true`, runs both
`gh release verify` and `gh release verify-asset`, and rechecks the exact two asset names, sizes, and
service SHA-256 digests. It then re-reads the remote tag, resolves either its lightweight target or
annotated peeled target, and requires the immutable target to equal C0. Both assets are downloaded
without credentials; the anonymous copies must equal the local sizes and SHA-256 values. A failed
pre-publication run can delete only the draft release ID created by that run, carrying its run-ID,
attempt, and C0 body marker, and cleanup remains behind the `confirmatory` environment. It never
looks up an unrelated draft by tag or attempts to delete a published immutable release.

The workflow retains `c0-evidence-release-binding.json`. Before C1, copy that JSON object verbatim
into `sealed_execution.c0_evidence_release`. Its closed schema binds the repository, release tag,
C0 commit, release and asset URLs, both asset names, sizes and SHA-256 values, the immutable flag,
and a canonical embedded verification receipt. Apparatus evidence also pins
`production_control_instantiation_receipt_file_sha256` and
`github_environment_control_receipt_file_sha256`, both candidate file SHA-256 values, and both fixed
candidate archive-member paths; the frozen schema accepts none of those values from a
caller-supplied digest. The receipt pins the release API readback, post-immutability tag readback
and resolved C0 target,
release-attestation result, asset-attestation result, and anonymous readback. The frozen manifest
validator recomputes the receipt digest and rejects any changed field.

The C0 archive and binding artifact remain outside the exact 27-file Zenodo package. C1 binds the
release twice: `study-manifest.json` contains the complete release object, and
`c0-public-verification.json` records a fresh public readback whose digest is signed in the C1
predicate. This preserves the closed registration inventory while making the durable C0 evidence
independently downloadable. GitHub documents the draft-attach-publish sequence and the
post-publication tag and asset protections in its [immutable release guide](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/establish-provenance-and-integrity/prevent-release-changes).

## Runtime confinement

The image runs as `runner` (`65532:65532`). Locale, timezone, Python hash seed, and numerical
thread counts are fixed. Its entrypoint is an exec-form Python invocation, so a shell does
not sit between the container runtime and the process. Python owns one OPA child during each
corpus attempt; no external daemon or supervisor is admitted. The child starts only after the
exclusive attempt marker exists. It binds `127.0.0.1:8181`, loads the image-baked Rego and an
exact private copy of the receipt-bound `opa-data.json`, and must pass both `/health?plugins` and
a decision-contract probe. Python drains OPA stderr continuously while retaining at most 64 KiB,
then terminates OPA and removes its scratch data before publishing a result receipt. Failure to
start, become ready, stay alive, terminate, drain, or clean up consumes the attempt and emits no
valid result.

The digest-qualified image runs with a read-only root filesystem, a bounded disposable `/tmp`,
read-only input and secret mounts, and a fresh Docker named volume fixed by the corpus-attempt
contract. Do not bind a host directory directly at `/output`; host ownership on macOS and Linux is
not an invariant of the container identity. The host must not assemble a `docker run` command from
this description. The closed [sealed-container launcher](sealed-container-launcher.md) derives the
exact Docker argument arrays from the admitted C1 contracts, records each array before the
corresponding mutation, and retains the containers and volume.

Commands that write receipts or results must name a path below `/output`. The default
working directory is intentionally unwritable under `--read-only`. No secret is accepted as
a build argument, copied into the image, or mounted by the publication workflow. The separate
pseudonym-key file remains mode `0600`; it is never made world-readable to simplify a bind mount.
The runtime preflight must show that UID 65532 can read but cannot write that exact mounted file.

Volume creation is a recorded preparation step. The launcher freezes the volume name and private
subpath in the preflight contract, creates the volume once with contract and corpus labels, then
uses the same C0 image to create the empty subdirectory as root. An existing volume is a terminal
conflict rather than state to clean or reuse. The initializer is retained and inspected; it is not
an unrecorded `docker run --rm` preparation command.

A non-mutating preflight mounts only
`type=volume,src="$output_volume",dst=/output,volume-subpath="$output_subpath",volume-nocopy`.
It requires an empty directory owned by UID/GID 65532, writable by UID 65532, and not writable by
group or other identities. The effective mount root must remain exactly `0700`; a `0755` root is a
contract failure even if its write bits appear safe. Record the ownership and mode from inside the
selected platform image. Reuse that exact volume and subpath for the sole corpus invocation. Docker
requires a volume subpath to exist before the mount; see the
[volume-subpath contract](https://docs.docker.com/engine/storage/volumes/#mount-a-volume-subdirectory).

After the invocation, run the copy-out reader as UID/GID 65532 with the retained subpath mounted
read-only. Copy its exact files plus checksum manifest into the host suite namespace, then rehash
the host copies before admitting them. Keep the named volume until publication and independent
custody are complete. The copy operation is not a second corpus invocation. A root reader with
added discretionary-access capabilities is forbidden.

The registered inner process has one admitted argument vector:

```text
/opt/venv/bin/python
-m
fractal_ann_diagnostics.cli
run-sealed-corpus
--config
<production-run-closure>/<corpus-id>/control/corpus-run-config.json
```

There is no `--config-sha256` runtime option. The C1 workload, instantiated plan, production
closure, and config loader bind the config bytes; the host launcher admits that chain before it
creates the sealed container.

The C1 package must include the byte-identical selected-platform OPA executable at
`c1-input/runtime/opa`; its read-only file mount makes `/usr/local/bin/opa` an exact declared
artifact mount rather than an unrecorded file inherited only from the image. The runtime plan pins
its SHA-256. C1 must also freeze the matching retained hnsw wheel at
`c1-input/backends/hnsw/hnswlib-runtime.whl`. Copy both from one selected platform directory in the
retained C0 package. Crossing the amd64 OPA bytes with the arm64 wheel, rebuilding either object,
or selecting a platform that differs from the five runtime plans invalidates the freeze.

The same selected-platform receipt supplies the only admissible Python and `uv.lock` digests for
the five plans. Both retained files have separate GitHub artifact-attestation subjects. The runtime
then rehashes `/opt/venv/bin/python` and `/opt/app/uv.lock` inside the digest-qualified image; an
operator-provided digest or alternate lock file is not a C0 derivation.

The fixed hostname and explicit `HOSTNAME` value must agree and appear in the plan's exact
environment digest. Every other image or launcher environment value is frozen in that same
allowlist; host-variable inheritance, environment files, and unregistered overrides are forbidden.

The runtime plan must pin that exact argv, including argument order. The config self-declares its
fixed path, and the plan's workload digest equals the config file digest. The output directory is
empty when the command starts. The command itself creates the invocation marker and receipt in the
same process before it admits subordinate controls. A second call, any pre-existing output, an extra
option, or an alternate control path is rejected.

## Action source pins

The workflow uses commit SHAs resolved from each action's official GitHub release on
2026-07-16. Tags appear only as comments for human comparison.

| Action release | Commit SHA | Official source |
|---|---|---|
| `actions/checkout` v7.0.0 | `9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0` | <https://github.com/actions/checkout/releases/tag/v7.0.0> |
| `docker/setup-qemu-action` v4.2.0 | `96fe6ef7f33517b61c61be40b68a1882f3264fb8` | <https://github.com/docker/setup-qemu-action/releases/tag/v4.2.0> |
| `docker/setup-buildx-action` v4.2.0 | `bb05f3f5519dd87d3ba754cc423b652a5edd6d2c` | <https://github.com/docker/setup-buildx-action/releases/tag/v4.2.0> |
| `docker/login-action` v4.4.0 | `af1e73f918a031802d376d3c8bbc3fe56130a9b0` | <https://github.com/docker/login-action/releases/tag/v4.4.0> |
| `docker/build-push-action` v7.3.0 | `53b7df96c91f9c12dcc8a07bcb9ccacbed38856a` | <https://github.com/docker/build-push-action/releases/tag/v7.3.0> |
| `actions/upload-artifact` v7.0.1 | `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a` | <https://github.com/actions/upload-artifact/releases/tag/v7.0.1> |
| `actions/attest` v4.2.0 | `f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6` | <https://github.com/actions/attest/releases/tag/v4.2.0> |

Before C0 is registered, run:

```bash
pytest tests/test_confirmatory_image_contract.py -q
ruff check tests/test_confirmatory_image_contract.py
ruby -e 'require "yaml"; YAML.load_file(".github/workflows/confirmatory-image.yml")'
git diff --check
```

Also complete one uncached single-platform Docker build and execute the resulting image with
`--read-only`, `--network none`, the registered UID/GID, and the registered tmpfs. The Dockerfile's
final instruction already exercises imports and OPA policy tests; the post-build run checks that
the image remains valid under the intended launcher restrictions.

After C0 is registered, changing any source pin, Docker instruction, lock, allowlist, platform,
test module, build-tool identity, or action SHA creates a different apparatus and requires a new
C0.
