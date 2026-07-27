from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile.confirmatory"
WORKFLOW = ROOT / ".github" / "workflows" / "confirmatory-image.yml"
BUILD_REQUIREMENTS = ROOT / "requirements.confirmatory-build.txt"
LOCKFILE = ROOT / "uv.lock"
OPA_REGO = ROOT / "examples" / "opa_compiled_masks.rego"
OPA_REGO_TEST = ROOT / "examples" / "opa_compiled_masks_test.rego"
RUNNER_IMAGE_DOC = ROOT / "research" / "runner-image.md"
SOURCE_DATE_EPOCH_FILE = ROOT / "confirmatory-source-date-epoch.txt"

PYTHON_DIGEST = "sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b"
UV_DIGEST = "sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc"
GO_DIGEST = "sha256:1ecb7edf62a0408027bd5729dfd6b1b8766e578e8df93995b225dfd0944eb651"
DISTROLESS_DIGEST = "sha256:26cd77482910e221ff26cf7c480203ce97f8f01ad272e2dc8a9ae29c811e9efe"
DOCKERFILE_FRONTEND_DIGEST = (
    "sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e"
)
OPA_COMMIT = "e695c9ef8edb0f8b9f13d014d7bc8a7fbcc57297"
OPA_SOURCE_SHA256 = "a8b3ecdc925b75bdade52d315aa13efaa51c2de99acb78003ad353cce6e9e637"
UV_LOCK_SHA256 = "a7251c8ce2b54888a047daefb32a2584c6d3f596030dd6cd87e46693b7ca57d6"
HNSWLIB_SHA256 = "cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c"
OPA_REGO_SHA256 = "18f6eb8a7411a7a1415bd2425ad5720f28fcd3b428d9aa2c1e7d73f6e14e356c"
OPA_REGO_TEST_SHA256 = "67370adfcba1c5180bdc99ae2cab900785ec5cee6fd91a9a4a9058415a7d4f00"
OPA_PATCHED_GO_SUM_SHA256 = "594c9098656b4b4b4a41f11093ff95babda2d0333077f8a7ad42528466da0903"
OPA_DEPENDENCY_DELTA_SHA256 = "2b66370c2620bea30ed5ed776a807ea9ac83ca7aef9b2214a2f444cbcf7a7524"
BUILDX_SHA256 = "f1332ddb9010bd0b72628266c3a906d9a6979848033df4c8d9bd2cd113bae12b"
BUILDKIT_DIGEST = "sha256:0168606be2315b7c807a03b3d8aa79beefdb31c98740cebdffdfeebf31190c9f"
BINFMT_DIGEST = "sha256:400a4873b838d1b89194d982c45e5fb3cda4593fbfd7e08a02e76b03b21166f0"

ACTION_PINS = {
    "actions/attest": "f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6",
    "actions/checkout": "9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
    "actions/download-artifact": "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
    "actions/setup-python": "ece7cb06caefa5fff74198d8649806c4678c61a1",
    "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    "astral-sh/setup-uv": "11f9893b081a58869d3b5fccaea48c9e9e46f990",
    "docker/build-push-action": "53b7df96c91f9c12dcc8a07bcb9ccacbed38856a",
    "docker/login-action": "af1e73f918a031802d376d3c8bbc3fe56130a9b0",
    "docker/setup-buildx-action": "bb05f3f5519dd87d3ba754cc423b652a5edd6d2c",
    "docker/setup-qemu-action": "96fe6ef7f33517b61c61be40b68a1882f3264fb8",
    "sigstore/cosign-installer": "6f9f17788090df1f26f669e9d70d6ae9567deba6",
}

CONFIRMATORY_RUNTIME_DISTRIBUTIONS = {
    "annotated-types": "0.7.0",
    "cytoolz": "1.1.0",
    "eth-hash": "0.8.0",
    "eth-typing": "6.0.0",
    "eth-utils": "6.0.0",
    "hnswlib": "0.8.0",
    "joblib": "1.5.3",
    "narwhals": "2.24.0",
    "numpy": "2.5.1",
    "py-ecc": "8.0.0",
    "pydantic": "2.13.4",
    "pydantic-core": "2.46.4",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
    "threadpoolctl": "3.6.0",
    "toolz": "1.1.0",
    "tqdm": "4.68.4",
    "truststore": "0.10.4",
    "typing-extensions": "4.16.0",
    "typing-inspection": "0.4.2",
}


def test_container_sources_and_dependency_inputs_are_immutable() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    lock_bytes = LOCKFILE.read_bytes()
    requirements = BUILD_REQUIREMENTS.read_text(encoding="utf-8")
    python_image = f"python:3.12.13-slim-bookworm@{PYTHON_DIGEST}"
    go_image = f"docker.io/library/golang:1.26.5-bookworm@{GO_DIGEST}"
    distroless_image = f"gcr.io/distroless/base-nossl-debian12:nonroot@{DISTROLESS_DIGEST}"

    assert dockerfile.splitlines()[0] == (
        f"# syntax=docker/dockerfile:1.7@{DOCKERFILE_FRONTEND_DIGEST}"
    )
    assert f"ARG PYTHON_IMAGE={python_image}" in dockerfile
    assert f"ARG GO_IMAGE={go_image}" in dockerfile
    assert f"ARG DISTROLESS_IMAGE={distroless_image}" in dockerfile
    assert f'org.opencontainers.image.base.name="{distroless_image}"' in dockerfile
    assert f'io.fractal-ann.confirmatory.python-builder-image="{python_image}"' in dockerfile
    assert f'io.fractal-ann.confirmatory.go-builder-image="{go_image}"' in dockerfile
    assert f"ghcr.io/astral-sh/uv:0.11.29@{UV_DIGEST}" in dockerfile
    assert f"ARG OPA_COMMIT={OPA_COMMIT}" in dockerfile
    assert f"ARG OPA_SOURCE_SHA256={OPA_SOURCE_SHA256}" in dockerfile
    assert f"https://github.com/open-policy-agent/opa/archive/{OPA_COMMIT}.tar.gz" in dockerfile
    assert "openpolicyagent/opa:1.18.2-static" not in dockerfile
    assert hashlib.sha256(lock_bytes).hexdigest() == UV_LOCK_SHA256
    assert f"ARG UV_LOCK_SHA256={UV_LOCK_SHA256}" in dockerfile
    assert HNSWLIB_SHA256 in lock_bytes.decode("utf-8")
    assert hashlib.sha256(OPA_REGO.read_bytes()).hexdigest() == OPA_REGO_SHA256
    assert hashlib.sha256(OPA_REGO_TEST.read_bytes()).hexdigest() == OPA_REGO_TEST_SHA256
    assert f"ARG OPA_REGO_SHA256={OPA_REGO_SHA256}" in dockerfile
    assert f"ARG OPA_REGO_TEST_SHA256={OPA_REGO_TEST_SHA256}" in dockerfile
    assert f"ARG OPA_PATCHED_GO_SUM_SHA256={OPA_PATCHED_GO_SUM_SHA256}" in dockerfile
    assert f"ARG OPA_DEPENDENCY_DELTA_SHA256={OPA_DEPENDENCY_DELTA_SHA256}" in dockerfile
    digest_arguments = re.findall(
        r"^ARG ([A-Z0-9_]*SHA256[A-Z0-9_]*)=([^\s]+)$",
        dockerfile,
        flags=re.MULTILINE,
    )
    assert digest_arguments
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value) is not None for _name, value in digest_arguments
    )
    assert "uv sync" in dockerfile
    assert "--frozen" in dockerfile
    assert "--no-extra production-embedding" in dockerfile
    assert "--extra production-embedding" not in dockerfile
    assert dockerfile.index("uv sync") < dockerfile.index("uv build")
    assert "--extra hnsw" not in dockerfile
    assert "ADD --checksum=sha256:" + HNSWLIB_SHA256 in dockerfile
    assert "uv build" in dockerfile
    assert "--file /tmp/hnswlib-0.8.0.tar.gz" in dockerfile
    assert "--directory /build/hnswlib-source" in dockerfile
    assert "/build/hnswlib-source/hnswlib-0.8.0" in dockerfile
    assert dockerfile.index("--directory /build/hnswlib-source") < dockerfile.index("uv build")
    assert "/opt/wheels" in dockerfile
    assert "fractal-hnswlib-runtime-artifact-v1" in dockerfile
    assert "HNSWLIB_NO_NATIVE=1" in dockerfile
    assert "-ffile-prefix-map=/build/hnswlib-source=/usr/src/hnswlib-source" in dockerfile
    assert "-fdebug-prefix-map=/build/hnswlib-source=/usr/src/hnswlib-source" in dockerfile
    assert "-fmacro-prefix-map=/build/hnswlib-source=/usr/src/hnswlib-source" in dockerfile
    assert "expected_epoch -= expected_epoch % 2" in dockerfile
    assert "timestamps == {expected_timestamp}" in dockerfile
    assert 'cache = dist_info / "uv_cache.json"' in dockerfile
    assert "cache.unlink()" in dockerfile
    assert "COPY --from=opa-builder --chown=0:0 /out/opa /usr/local/bin/opa" in dockerfile
    assert (
        "COPY --from=opa-builder --chown=0:0 /out/artifacts /opt/artifacts/opa-build" in dockerfile
    )
    assert "COPY --from=hnsw-builder --chown=0:0 /opt/venv /opt/venv" in dockerfile
    assert "COPY --from=hnsw-builder --chown=0:0 /opt/native-libs /opt/native-libs" in dockerfile
    assert "COPY --from=hnsw-builder --chown=0:0 /build/uv.lock /opt/app/uv.lock" in dockerfile
    assert "COPY --chown=0:0 src /opt/app/src" in dockerfile
    assert (
        "COPY --chown=0:0 examples/opa_compiled_masks.rego /opt/app/policy/opa_compiled_masks.rego"
    ) in dockerfile
    assert "chmod -R a-w /opt/venv /opt/native-libs /opt/artifacts /opt/app/src" in dockerfile
    assert "! -user root -print -quit" in dockerfile
    assert "-perm /222 -print -quit" in dockerfile
    assert "os.setgid(65532); os.setuid(65532)" in dockerfile
    assert "os.access(path, os.W_OK)" in dockerfile
    assert "chmod 0555 /usr/local/bin/opa" in dockerfile
    assert "chmod 0444 /opt/app/uv.lock" in dockerfile
    assert '"${UV_LOCK_SHA256}"' in dockerfile
    assert "/opt/app/uv.lock" in dockerfile
    assert "chmod 0444 /opt/app/policy/opa_compiled_masks.rego" in dockerfile
    assert "install -d -o 65532 -g 65532" not in dockerfile
    assert 'for raw_path in ("/home/runner", "/input", "/output", "/workspace"):' in dockerfile
    assert "os.chown(path, 65532, 65532)" in dockerfile
    assert "path.chmod(0o755)" in dockerfile
    assert dockerfile.count("    PATH=/opt/venv/bin:/usr/local/bin \\") == 2
    assert dockerfile.count("export PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin") == 4
    assert "opa version" in dockerfile
    assert "--network=none" in dockerfile
    assert "source=examples/opa_compiled_masks_test.rego" in dockerfile
    assert "python -m fractal_ann_diagnostics.cli --help" in dockerfile
    assert "uv pip check --python /opt/venv/bin/python" in dockerfile
    runtime_inventory_match = re.search(
        r"expected = (\{[^;\n]+\}); observed = ",
        dockerfile,
    )
    assert runtime_inventory_match is not None
    assert ast.literal_eval(runtime_inventory_match.group(1)) == CONFIRMATORY_RUNTIME_DISTRIBUTIONS
    lock_text = lock_bytes.decode("utf-8")
    for distribution, version in CONFIRMATORY_RUNTIME_DISTRIBUTIONS.items():
        assert f'name = "{distribution}"\nversion = "{version}"' in lock_text
    assert "assert observed == expected" in dockerfile
    assert 'assert "torch" not in observed and "transformers" not in observed' in dockerfile
    assert "import torch" not in dockerfile
    assert "import transformers" not in dockerfile
    assert "fractal_ann_diagnostics.drand_beacon" in dockerfile
    assert "fractal_ann_diagnostics.provider_activation_factory" in dockerfile
    assert (
        'importlib.util.find_spec(name) is None for name in ("torch", "transformers")' in dockerfile
    )
    assert "/usr/local/bin/opa test" in dockerfile
    assert "SOURCE_DATE_EPOCH" in dockerfile
    assert "ARG C0_SHA" not in dockerfile
    assert "${C0_SHA}" not in dockerfile
    assert "ARG BUILD_CONTEXT_TREE_SHA256" in dockerfile
    assert (
        'io.fractal-ann.confirmatory.build-context-tree-sha256="${BUILD_CONTEXT_TREE_SHA256}"'
        in dockerfile
    )
    assert "http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" in dockerfile
    assert "APT::Update::Error-Mode=any" in dockerfile
    assert """dpkg-query -W -f='${Version}' g++""" in dockerfile
    assert "= '4:12.2.0-3'" in dockerfile
    assert """test "$(command -v g++)" = '/usr/bin/g++' """.strip() in dockerfile

    requirement_lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pins = [line for line in requirement_lines if "==" in line]
    hashes = [line for line in requirement_lines if "--hash=sha256:" in line]
    assert len(pins) == len(hashes) == 3
    assert all(re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", line) for line in hashes)


def test_runtime_is_nonroot_deterministic_and_read_only_compatible() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    runtime = dockerfile.split("FROM ${DISTROLESS_IMAGE} AS scientific-runtime-base", maxsplit=1)[1]
    scientific_runtime, release_runtime = runtime.split(
        "FROM scientific-runtime-base AS release-runtime", maxsplit=1
    )

    assert "apt-get" not in runtime
    assert "USER 65532:65532" in scientific_runtime
    assert "--chown=65532:65532" not in scientific_runtime
    assert 'io.fractal-ann.confirmatory.runtime-role="scientific"' in scientific_runtime
    assert 'io.fractal-ann.confirmatory.tle-present="false"' in scientific_runtime
    assert "PYTHONDONTWRITEBYTECODE=1" in scientific_runtime
    assert "PYTHONHASHSEED=0" in scientific_runtime
    assert "TZ=UTC" in scientific_runtime
    for variable in (
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
    ):
        assert variable in scientific_runtime
    assert (
        'ENTRYPOINT ["/opt/venv/bin/python", "-m", "fractal_ann_diagnostics.cli"]'
        in scientific_runtime
    )
    assert 'io.fractal-ann.confirmatory.runtime-role="timelock-release"' in release_runtime
    assert 'io.fractal-ann.confirmatory.tle-present="true"' in release_runtime
    assert "COPY --from=release-rootfs /usr/local/bin/tle /usr/local/bin/tle" in release_runtime
    assert "/usr/local/bin/tle" not in scientific_runtime
    assert dockerfile.rstrip().endswith("FROM scientific-runtime-base AS runtime")
    assert "rm -f /opt/venv/bin/python" in dockerfile
    assert "cp -L /usr/local/bin/python /opt/venv/bin/python" in dockerfile
    assert "test ! -L /opt/venv/bin/python" in dockerfile
    assert "stat -c '%h' /opt/venv/bin/python" in dockerfile
    assert "sys.executable == str(path)" in dockerfile
    assert "COPY ." not in dockerfile
    assert "VOLUME" not in runtime
    assert "--mount=type=secret" not in dockerfile
    assert "local-prototype-system-certs" not in dockerfile
    assert "BCG" not in dockerfile
    assert dockerfile.count("os.chmod(status_path, 0o444)") == 1
    assert dockerfile.count("os.chmod(manifest_path, 0o444)") == 1


def test_locked_runtime_wheels_cover_both_published_platforms() -> None:
    lock_lines = LOCKFILE.read_text(encoding="utf-8").splitlines()
    packages = {
        "numpy": "2.5.1",
        "scikit_learn": "1.9.0",
        "scipy": "1.18.0",
    }

    for package, version in packages.items():
        prefix = f"{package}-{version}-cp312-cp312"
        for architecture in ("x86_64", "aarch64"):
            assert any(prefix in line and architecture in line for line in lock_lines), (
                package,
                version,
                architecture,
            )

    assert any(
        "torch-2.13.0-cp312-cp312-macosx_14_0_arm64.whl" in line
        and "sha256:2fe228aba290d14b9f31b049be550dbd469c3fd3013d7a19705b30454da97027" in line
        for line in lock_lines
    )
    assert any(
        "transformers-5.13.1-py3-none-any.whl" in line
        and "sha256:53f0ea8aa397e29244c2377ba981bcaf0c87adcf44fbdd447ef6306522afcacd" in line
        for line in lock_lines
    )


def test_workflow_is_manual_sha_pinned_and_records_the_registry_digest() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    uses = dict(re.findall(r"uses:\s+([^@\s]+)@([0-9a-f]{40})", workflow))

    assert uses == ACTION_PINS
    assert (
        """permissions:
  actions: read
  attestations: write
  contents: read
  id-token: write
  packages: write
"""
        in workflow
    )
    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow.split("jobs:", maxsplit=1)[0]
    assert "runs-on: ubuntu-24.04" in workflow
    assert f"C0_PYTHON_IMAGE: python:3.12.13-slim-bookworm@{PYTHON_DIGEST}" in workflow
    assert "persist-credentials: false" in workflow
    assert "ref: ${{ env.C0_REF }}" in workflow
    assert "^[0-9a-f]{40}$" in workflow
    assert "refs/tags/confirmatory-apparatus-c0" in workflow
    assert 'test "$GITHUB_SHA" = "$C0_SHA"' in workflow
    assert 'test "$GITHUB_WORKFLOW_SHA" = "$C0_SHA"' in workflow
    assert 'test "$(git rev-parse "${C0_REF}^{commit}")" = "$C0_SHA"' in workflow
    assert "test \"$GITHUB_ACTOR\" = 'mhdk1602'" in workflow
    assert "test \"$GITHUB_TRIGGERING_ACTOR\" = 'mhdk1602'" in workflow
    assert "platforms: linux/amd64,linux/arm64" in workflow
    assert "outputs: type=registry,rewrite-timestamp=true,oci-mediatypes=true" in workflow
    assert "push: true" not in workflow
    assert "no-cache: true" in workflow
    assert "provenance: mode=max" in workflow
    assert "sbom: true" in workflow
    assert "${{ env.IMAGE_NAME }}:${{ inputs.c0_sha }}" in workflow
    assert "${{ steps.image.outputs.digest }}" in workflow
    assert '"$GITHUB_SERVER_URL" "$GITHUB_REPOSITORY" "$GITHUB_RUN_ID"' in workflow
    assert "actions/upload-artifact@" in workflow
    assert "gh attestation verify" in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert "--source-digest" in workflow
    assert "github-attestation-verification.json" in workflow
    assert "registry-index.json" in workflow
    assert "buildkit-provenance.json" in workflow
    assert "buildkit-sbom.spdx.json" in workflow
    assert "runner-environment.json" in workflow
    assert "C0-ARTIFACT-SHA256SUMS" in workflow
    assert "PACKAGE-SHA256SUMS" in workflow
    assert "build-context-tree-sha256.txt" in workflow
    assert "git archive" in workflow
    assert "source/fractal-ann-diagnostics-C0.tar" in workflow
    assert "fractal-c0-runtime-extraction-v3" in workflow
    assert "hnswlib-runtime-receipt.json" in workflow
    assert "runtime-artifacts" in workflow
    assert ". as $index" in workflow
    assert "$index.manifests[]" in workflow
    assert 'docker pull --platform "$platform"' in workflow
    assert workflow.count("actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6") == 3
    assert workflow.count("create-storage-record: false") == 3
    assert "artifact-metadata: write" not in workflow
    assert "subject-checksums:" in workflow
    assert "SOURCE_DATE_EPOCH=${{ steps.source.outputs.source_date_epoch }}" in workflow
    assert "SOURCE_DATE_EPOCH: ${{ steps.source.outputs.source_date_epoch }}" in workflow
    assert "BUILD_CONTEXT_TREE_SHA256=${{ steps.source.outputs.context_tree_sha256 }}" in workflow


def test_workflow_compares_two_independent_arm64_builds_before_publication() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    gate_start = workflow.index(
        "- name: Build and compare two independent arm64 executable projections"
    )
    publication_start = workflow.index("- name: Authenticate to GHCR")
    gate = workflow[gate_start:publication_start]

    assert gate_start < publication_start
    assert "for replicate in a b" in gate
    assert 'active_builder="c0-repro-${replicate}"' in gate
    assert "docker buildx create" in gate
    assert "docker buildx rm" in gate
    assert "--platform linux/arm64" in gate
    assert "--pull" in gate
    assert "--no-cache" in gate
    assert "--provenance=mode=max" in gate
    assert "--sbom=true" in gate
    assert "rewrite-timestamp=true,oci-mediatypes=true" in gate
    assert "arm64-build-a.oci.tar" in gate
    assert "arm64-build-b.oci.tar" in gate
    assert "release-arm64-build-a.oci.tar" in gate
    assert "release-arm64-build-b.oci.tar" in gate
    assert 'active_builder="c0-release-repro-${replicate}"' in gate
    assert "--target release-runtime" in gate
    assert "--image-role timelock-release" in gate
    assert "fractal-tle-release-oci-reproducibility-v2" in gate
    assert ".image_closure_equal == true" in gate
    assert ".archive_a.executable_projection.config_descriptor.digest" in gate
    assert ".archive_a.executable_projection.ordered_layer_descriptors" in gate
    assert "--read-only" in gate
    assert "--network none" in gate
    assert "--cap-drop ALL" in gate
    assert "--expected-uv-lock-sha256" in gate
    assert "--expected-opa-policy-sha256" in gate
    assert 'schema_version == "fractal-c0-executable-reproducibility-v3"' in gate
    assert "--expected-build-context-tree-sha256" in gate
    assert "--expected-c0-sha" not in gate
    assert "rm --" in gate

    registry_gate_start = workflow.index(
        "- name: Verify registry identity, platforms, provenance, and SBOM"
    )
    anonymous_gate_start = workflow.index("- name: Require public anonymous digest access")
    registry_gate = workflow[registry_gate_start:anonymous_gate_start]
    assert "published_arm64_digest" in registry_gate
    assert '"$published_arm64_digest" != "$local_arm64_digest"' in registry_gate
    assert "fractal-c0-published-arm64-projection-v1" in registry_gate
    assert "published-arm64-projection.json" in workflow
    assert "published_release_arm64_digest" in registry_gate
    assert '"$published_release_arm64_digest"' in registry_gate
    assert "fractal-c0-published-release-arm64-projection-v1" in registry_gate
    assert "published-release-arm64-projection.json" in workflow


def test_workflow_separately_scans_and_attests_the_release_subject() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "id: release_attestation" in workflow
    assert "subject-name: ${{ env.RELEASE_IMAGE_NAME }}" in workflow
    assert "subject-digest: ${{ steps.release_image.outputs.digest }}" in workflow
    assert workflow.count("push-to-registry: true") == 2
    assert '"oci://${RELEASE_IMAGE_NAME}@${RELEASE_IMAGE_DIGEST}"' in workflow
    assert '--bundle "$RELEASE_ATTESTATION_BUNDLE"' in workflow
    assert "github-release-attestation-bundle.json" in workflow
    assert "github-release-attestation-verification.json" in workflow
    assert '--source-digest "$C0_SHA"' in workflow
    assert '--source-ref "$C0_REF"' in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert "release-linux-arm64-trivy-direct.json" in workflow
    assert "release-linux-arm64-trivy.cdx.json" in workflow
    assert "release-linux-arm64-trivy-sbom-rescan.json" in workflow
    assert "--image-role timelock-release" in workflow
    assert ".severity_counts.UNKNOWN == 1" in workflow
    assert 'vulnerability_id: "GO-2026-5932"' in workflow
    assert ".vex_required == false" in workflow
    assert ".vex_documents == []" in workflow


def test_offline_trivy_image_scans_use_memory_cache_with_read_only_db() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow[
        workflow.index(
            "- name: Retain and adjudicate raw Trivy and CycloneDX evidence"
        ) : workflow.index("- name: Retain govulncheck source and symbol reachability evidence")
    ]
    commands = tuple(re.finditer(r"(?m)^\s+run_trivy (image|sbom) \\\n", step))

    assert [match.group(1) for match in commands] == [
        "image",
        "image",
        "sbom",
        "image",
        "image",
        "sbom",
    ]
    for position, match in enumerate(commands):
        end = commands[position + 1].start() if position + 1 < len(commands) else len(step)
        command = step[match.start() : end]
        if match.group(1) == "image":
            assert "--cache-backend memory" in command
        else:
            assert "--cache-backend memory" not in command
    assert step.count("--cache-backend memory") == 4
    assert '--mount "type=bind,src=${trivy_cache},dst=/root/.cache/trivy,readonly"' in step


def test_candidate_closure_binds_the_admitted_build_context_tree() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow[
        workflow.index("- name: Materialize the candidate rehearsal closure") : workflow.index(
            "- name: Upload the candidate rehearsal closure"
        )
    ]

    binding = '--arg build_context_tree_sha256 "$BUILD_CONTEXT_TREE_SHA256" \\'
    assert step.count(binding) == 1
    assert step.count("$build_context_tree_sha256") == 1
    assert step.index(binding) < step.index(
        "{build_context_tree_sha256: $build_context_tree_sha256"
    )


def test_candidate_rehearsal_and_production_use_one_shared_execution_core() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "type: choice" in workflow
    assert "- candidate" in workflow
    assert "- production" in workflow
    assert "candidate_run_id:" in workflow
    assert "rehearsal_run_id:" in workflow
    assert "^refs/heads/c0-candidate/[a-z0-9._-]+$" in workflow
    assert "if: inputs.mode == 'production'" in workflow
    assert "if: inputs.mode == 'candidate'" in workflow
    assert "test \"$GITHUB_RUN_ATTEMPT\" = '1'" in workflow
    assert "run_attempt == 1" in workflow
    assert '.event == "workflow_dispatch"' in workflow
    assert '.status == "completed"' in workflow
    assert '.conclusion == "success"' in workflow
    assert '.actor.login == "mhdk1602"' in workflow
    assert '.triggering_actor.login == "mhdk1602"' in workflow
    assert 'candidate_sha="$(jq -r .head_sha' in workflow
    assert ".workflow_run.head_sha == $candidate_sha" in workflow
    assert '.path == ".github/workflows/confirmatory-image.yml"' in workflow
    assert "candidate-closure.zip" in workflow
    assert "printf '%s  %s\\n' \"${artifact_digest#sha256:}\"" in workflow
    assert "candidate-closure.json" in workflow
    assert "candidate-closure.sha256" in workflow
    assert "fractal-c0-candidate-closure-v2" in workflow
    assert "build_context_tree_sha256" in workflow
    assert "confirmatory-image-candidate-closure-${candidate_sha}" in workflow
    assert "production mode requires one provider rehearsal Actions run id" in workflow
    assert "provider-rehearsal-complete-${REHEARSAL_RUN_ID}" in workflow
    assert '.path == ".github/workflows/confirmatory-provider-rehearsal.yml"' in workflow
    assert "provider rehearsal receipt is not canonical JSON" in workflow
    assert "fractal-provider-rehearsal-aggregate-v1" in workflow
    assert "fractal-c0-provider-rehearsal-gate-v2" in workflow
    assert "candidate_bootstrap_closure_sha256" in workflow
    assert "candidate_image_source_commit" in workflow
    assert "candidate_image_closure_file_sha256" in workflow
    assert "provider_state_mutated" in workflow
    assert "scientific_inputs_opened" in workflow
    assert '--bundle "$evidence_dir/attestation-bundle.json"' in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert "confirmatory-provider-rehearsal.yml@refs/heads/${candidate_branch}" in workflow
    assert "provider-rehearsal-production-gate.json" in workflow
    assert 'test -z "$REHEARSAL_RUN_ID"' in workflow
    assert "fractal-ann-diagnostics-confirmatory-candidate" in workflow
    assert "fractal-ann-diagnostics-confirmatory-release-candidate" in workflow

    for field in (
        "candidate_package_checksums_sha256",
        "release_govulncheck_adjudication_sha256",
        "release_image_index_digest",
        "release_image_reference",
        "release_linux_arm64_manifest_digest",
        "release_oci_attestation_bundle_sha256",
        "release_oci_attestation_verification_sha256",
        "release_reproducibility_receipt_sha256",
        "release_security_adjudication_sha256",
        "release_tle_interoperability_receipt_sha256",
        "scientific_image_index_digest",
        "scientific_image_reference",
        "scientific_linux_amd64_manifest_digest",
        "scientific_linux_amd64_runtime_extraction_sha256",
        "scientific_linux_arm64_manifest_digest",
        "scientific_linux_arm64_runtime_extraction_sha256",
        "scientific_oci_attestation_bundle_sha256",
        "scientific_oci_attestation_verification_sha256",
    ):
        assert field in workflow

    local_gate = workflow[
        workflow.index(
            "- name: Build and compare two independent arm64 executable projections"
        ) : workflow.index("- name: Authenticate to GHCR")
    ]
    assert "inputs.mode" not in local_gate
    assert "PUBLICATION_MODE" not in local_gate
    assert "candidate-rehearsal" not in local_gate


def test_production_promotes_the_bootstrap_indexes_without_rebuilding() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    candidate_build_start = workflow.index("- name: Build and publish the candidate C0 image")
    promotion_start = workflow.index(
        "- name: Promote the verified candidate OCI indexes without rebuilding"
    )
    promotion_end = workflow.index("- name: Normalize the scientific image digest")
    promotion = workflow[promotion_start:promotion_end]

    assert "if: inputs.mode == 'candidate'" in workflow[candidate_build_start:promotion_start]
    assert "if: inputs.mode == 'production'" in promotion
    assert "docker buildx build" not in promotion
    assert (
        'docker buildx imagetools create --tag "$scientific_tag" "$scientific_source"' in promotion
    )
    assert 'docker buildx imagetools create --tag "$release_tag" "$release_source"' in promotion
    assert promotion.count("imagetools inspect") >= 8
    assert promotion.count("--raw") >= 5
    assert promotion.count("cmp --silent") >= 4
    assert "guard_absent" in promotion
    assert "refusing to overwrite existing immutable destination" in promotion
    assert "fractal-c0-oci-promotion-incident-v1" in promotion
    assert "operator_recovery" in promotion
    assert "candidate_source_commit" in promotion
    assert "current_c0_commit" in promotion
    assert "BUILD_CONTEXT_TREE_SHA256" in promotion
    assert "Retain failed production promotion incident evidence" in workflow
    assert "fractal-c0-oci-promotion-v1" in workflow
    assert "scientific_raw_index_equal: true" in workflow
    assert "release_raw_index_equal: true" in workflow
    assert "scientific_anonymous_readback_equal: true" in workflow
    assert "release_anonymous_readback_equal: true" in workflow
    assert "oci-promotion-receipt.json" in workflow


def test_workflow_proves_bidirectional_quicknet_release_interoperability() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step_start = workflow.index(
        "- name: Prove safe and official tlock v1.2.0 Quicknet interoperability"
    )
    step_end = workflow.index("- name: Require public anonymous digest access")
    step = workflow[step_start:step_end]

    assert step_start < step_end
    assert "tlock_1.2.0_linux_arm64.tar.gz" in step
    assert "3092e410128cd64b98bd4f50ce60503b7df91fa3d676f2b820b00403452b3e7a" in step
    assert "3b724032620587c2551ee857c98dc02690076f4972a4fe4389b0f6e0911a6a92" in step
    assert "e153cfa8539e871f50143d1bde10fec7ec3fe82630f717c3c1bf166eb4975059" in step
    assert "ca9d498b6a3c1ea8edff9ace7bf00eb0f90ce67166343161f9a53f21900a6ef5" in step
    assert "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971" in step
    assert "round=30485281" in step
    assert "3e690bc527c8a4e78232bc06b5a3cff057c51c68f51b208a1a21b2abd6d6b194" in step
    assert "4275f9ee6cd9f767fb82e9877880bbb5659c961306c40051675bc76dded4d9be" in step
    assert "docker network create --internal" in step
    assert "--pull never" in step
    assert "--read-only" in step
    assert "--cap-drop ALL" in step
    assert "--security-opt no-new-privileges" in step
    assert "--entrypoint /usr/local/bin/tle" in step
    assert "--entrypoint /work/official-tle" in step
    assert "safe-to-official.age" in step
    assert "official-to-safe.age" in step
    assert 'encryptor: "source-built-safe-v1.2.0"' in step
    assert 'encryptor: "official-v1.2.0"' in step
    assert 'schema_version: "fractal-tlock-quicknet-interoperability-v1"' in step
    assert "external_client_network_access: false" in step
    assert "interoperability-receipt.sha256" in step
    assert "tlock-quicknet-interoperability/interoperability-receipt.json" in workflow
    assert "release_tle_interoperability_receipt_sha256" in workflow


def test_workflow_retains_versioned_govulncheck_reachability_without_vex() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step_start = workflow.index(
        "- name: Retain govulncheck source and symbol reachability evidence"
    )
    step_end = workflow.index("- name: Attest the digest published by this job")
    step = workflow[step_start:step_end]

    assert "golang.org/x/vuln/cmd/govulncheck@v1.6.0" in step
    assert "1cf0bf22b6f9484c850380cd3065bffd9a6d6577181e281053ab2d6bcb8898f0" in step
    assert "b677bec1ea587aa03320e7d65520dd52cae824fd197ede36417a5f572be41cb3" in step
    assert "69ca051a3d3e14f6f405875dfdcb976c6be78cab66dc24c7191a949bd8257ff7" in step
    assert "d37ef9b9e10d3b3b17569653d5d3be68f5dba50f72d6494fcf63a360c952936b" in step
    assert "go list -deps ./cmd/tle" in step
    assert "! grep -i openpgp" in step
    assert "-mode=binary" in step
    assert "tlock_reachability.py" in step
    assert 'finding_trace_level == "module"' in step
    assert "package_or_symbol_reachable == false" in step
    assert ".vex_document == null" in step
    assert "tlock-govulncheck/reachability-adjudication.json" in workflow


def test_workflow_verifies_each_retained_subject_before_package_sealing() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    verification_start = workflow.index("- name: Verify every retained C0 artifact attestation")
    sealing_start = workflow.index("- name: Seal the retained artifact package")
    verification = workflow[verification_start:sealing_start]

    assert verification_start < sealing_start
    assert 'done < "$checksums"' in verification
    assert 'cd "$record_dir"' in verification
    assert 'gh attestation verify "$relative_path"' in verification
    assert 'test "$(sha256sum "$subject" | cut -d \' \' -f 1)" = "$expected_sha256"' in verification
    assert "--cert-identity" in verification
    assert "--source-digest" in verification
    assert "--source-ref" in verification
    assert "--deny-self-hosted-runners" in verification
    assert "c0-artifact-attestation-verification.jsonl" in verification
    assert workflow.count("--hostname github.com") == 4
    assert 'test "$(wc -l < "$verification")" -eq "$verified_count"' in verification


def test_workflow_blocks_until_the_digest_is_publicly_readable_without_credentials() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    gate_start = workflow.index("- name: Require public anonymous digest access")
    gate_end = workflow.index("- name: Attest the digest published by this job")
    gate = workflow[gate_start:gate_end]

    assert workflow.index("- name: Normalize the scientific image digest") < gate_start
    assert gate_start < gate_end
    assert "timeout-minutes: 360" in workflow
    assert "anonymous-docker-config" in gate
    assert "printf '{}\\n' > \"$anonymous_config/config.json\"" in gate
    assert "jq -e 'type == \"object\" and keys == []'" in gate
    assert 'DOCKER_CONFIG="$anonymous_config"' in gate
    assert '"$buildx_path" imagetools inspect "$digest_reference"' in gate
    assert '"$buildx_path" imagetools inspect "$digest_reference" --raw' in gate
    assert '"$buildx_path" imagetools inspect "$release_digest_reference" --raw' in gate
    assert "env -i" in gate
    assert 'HOME="$HOME"' in gate
    assert "LANG=C.UTF-8" in gate
    assert "LC_ALL=C.UTF-8" in gate
    assert "PATH=/usr/bin:/bin" in gate
    assert 'environment_allowlist: ["DOCKER_CONFIG", "HOME", "LANG", "LC_ALL", "PATH"]' in gate
    assert "for attempt in $(seq 1 60)" in gate
    assert "sleep 30" in gate
    assert 'package_name="${IMAGE_NAME##*/}"' in gate
    assert "packages/container/${package_name}/settings" in gate
    assert 'release_package_name="${RELEASE_IMAGE_NAME##*/}"' in gate
    assert "packages/container/${release_package_name}/settings" in gate
    assert "fractal-c0-anonymous-registry-verification-v1" in gate
    assert "public_anonymous_access: true" in gate
    assert "cmp --silent" in gate
    assert "docker login" not in gate
    assert "GITHUB_TOKEN" not in gate
    assert "GH_TOKEN" not in gate
    for retained_name in (
        "anonymous-docker-config.json",
        "anonymous-registry-attempts.log",
        "anonymous-registry-index.json",
        "anonymous-registry-stderr.txt",
        "anonymous-registry-verification.json",
    ):
        assert workflow.count(retained_name) >= 2
    for retained_name in (
        "anonymous-release-registry-attempts.log",
        "anonymous-release-registry-index.json",
        "anonymous-release-registry-stderr.txt",
        "anonymous-release-registry-verification.json",
    ):
        assert workflow.count(retained_name) >= 2


def test_runtime_document_routes_volume_preparation_through_closed_launcher() -> None:
    document = RUNNER_IMAGE_DOC.read_text(encoding="utf-8")
    preparation_start = document.index("Volume creation is a recorded preparation step")
    preparation_end = document.index("After the invocation", preparation_start)
    preparation = document[preparation_start:preparation_end]

    assert "closed [sealed-container launcher](sealed-container-launcher.md)" in document
    assert "must not assemble a `docker run` command" in document
    assert "creates the volume once with contract and corpus labels" in preparation
    assert "An existing volume is a terminal" in preparation
    assert "initializer is retained and inspected" in preparation
    assert "`docker run --rm`" in preparation
    assert "docker volume create" not in preparation
    assert "must remain exactly `0700`" in preparation


def test_retained_runtime_receipt_binds_c0_image_platform_and_bytes() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for field in (
        "c0_sha",
        "hnswlib_receipt_image_path",
        "hnswlib_receipt_sha256",
        "hnswlib_wheel_basename",
        "hnswlib_wheel_byte_count",
        "hnswlib_wheel_image_path",
        "hnswlib_wheel_sha256",
        "image_digest",
        "image_manifest_digest",
        "image_reference",
        "opa_byte_count",
        "opa_image_path",
        "opa_sha256",
        "platform",
        "python_binary_byte_count",
        "python_binary_image_path",
        "python_binary_sha256",
        "source_date_epoch",
        "uv_lock_byte_count",
        "uv_lock_image_path",
        "uv_lock_sha256",
    ):
        assert f"{field}: ${field}" in workflow
    assert 'schema_version: "fractal-c0-runtime-extraction-v3"' in workflow
    assert "/usr/local/bin/opa" in workflow
    assert "/opt/venv/bin/python" in workflow
    assert "/opt/app/uv.lock" in workflow
    assert 'docker cp "$container_id:/opt/venv/bin/python" "$runtime_dir/python"' in workflow
    assert 'docker cp "$container_id:/opt/app/uv.lock" "$runtime_dir/uv.lock"' in workflow
    assert 'cmp --silent uv.lock "$runtime_dir/uv.lock"' in workflow
    assert "/opt/artifacts/hnswlib-runtime-receipt.json" in workflow
    assert "/opt/artifacts/hnswlib/${wheel_basename}" in workflow
    assert "assert set(receipt) ==" in workflow
    assert 'receipt["extension_sha256"] == hashlib.sha256(extension_bytes).hexdigest()' in workflow
    assert 'receipt["wheel_sha256"] == hashlib.sha256(wheel_bytes).hexdigest()' in workflow
    assert "zipfile.ZipFile(wheel_path)" in workflow


def test_builder_toolchain_and_transitive_images_are_fixed() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert f"BUILDX_LINUX_AMD64_SHA256: {BUILDX_SHA256}" in workflow
    assert "BUILDX_VERSION: v0.34.1" in workflow
    assert f"BUILDKIT_IMAGE: docker.io/moby/buildkit@{BUILDKIT_DIGEST}" in workflow
    assert f"BINFMT_IMAGE: docker.io/tonistiigi/binfmt@{BINFMT_DIGEST}" in workflow
    assert "cache-binary: false" in workflow
    assert "cache-image: false" in workflow
    assert "platforms: arm64" in workflow
    assert "buildkitd-flags: --debug=false" in workflow
    assert "driver-opts: image=${{ env.BUILDKIT_IMAGE }}" in workflow
    assert "builder: c0-builder" in workflow
    assert 'test "$(docker run --rm "$BINFMT_IMAGE" --version 2>&1)" \\' in workflow
    assert (
        'docker run --rm "$BINFMT_IMAGE" --version \\\n'
        '            > "$record_dir/binfmt-version.txt" 2>&1' in workflow
    )
    assert "--bootstrap --format" not in workflow
    assert "docker buildx ls --format '{{json .}}'" not in workflow
    assert workflow.count("docker buildx ls --format json") == 3
    assert workflow.count("map(select(.Name == $builder))") == 3
    assert 'github-token: ""' in workflow
    assert 'DOCKER_BUILD_RECORD_UPLOAD: "false"' in workflow
    assert 'DOCKER_BUILD_SUMMARY: "false"' in workflow
    assert "secrets." not in workflow


def test_build_context_is_an_explicit_allowlist() -> None:
    ignore = (ROOT / ".dockerignore.confirmatory").read_text(encoding="utf-8").splitlines()
    assert (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines() == ignore
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert ignore[0] == "*"
    assert set(ignore[1:]) == {
        "!Dockerfile.confirmatory",
        "!confirmatory-source-date-epoch.txt",
        "!examples/opa_compiled_masks.rego",
        "!examples/opa_compiled_masks_test.rego",
        "!pyproject.toml",
        "!requirements.confirmatory-build.txt",
        "!src/",
        "!src/**",
        "!uv.lock",
    }
    assert "cmp --silent .dockerignore.confirmatory .dockerignore" in workflow
    assert "cp -- .dockerignore.confirmatory .dockerignore" not in workflow
    assert "context_roots=(" in workflow
    assert 'git archive --format=tar "$C0_SHA" -- "${context_roots[@]}"' in workflow
    assert 'git ls-tree -r -z "$C0_SHA"' in workflow
    assert ".dockerignore.confirmatory" in workflow
    assert "-type f -links +1" in workflow
    assert 'find "${context_roots[@]}" -xdev -type f -print0' in workflow
    assert 'git ls-files -z -- "${context_roots[@]}"' in workflow
    assert 'cmp --silent "$filesystem_members" "$tracked_members"' in workflow
    assert 'source_date_epoch="$(<confirmatory-source-date-epoch.txt)"' in workflow
    assert SOURCE_DATE_EPOCH_FILE.read_bytes() == b"1783987200\n"
    assert "sealed C0 context file bytes differ from Git" in workflow
    assert "sealed C0 context file inventory differs from Git" in workflow
    assert "sealed C0 context file mode differs from Git" in workflow
    assert "sealed C0 context file timestamp changed" in workflow
    assert "sealed SOURCE_DATE_EPOCH differs from its tracked input" in workflow
    assert (
        "research/study-manifest.json"
        not in workflow[
            workflow.index("context_roots=(") : workflow.index(
                ")", workflow.index("context_roots=(")
            )
        ]
    )
    assert "source/build-context-git-tree.z" in workflow
    assert "source/build-context-verifier.py" in workflow
    assert "context: ${{ runner.temp }}/confirmatory-c0-context" in workflow
    assert "file: ${{ runner.temp }}/confirmatory-c0-context/Dockerfile.confirmatory" in workflow


def test_sealed_context_is_revalidated_around_every_build() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("printf '%s  %s\\n' \"$CONTEXT_VERIFIER_SHA256\"") >= 4
    build_args_start = workflow.index("build-args: |")
    first_build_args = workflow[build_args_start : workflow.index("provenance:", build_args_start)]
    assert "C0_SHA=${{ inputs.c0_sha }}" not in first_build_args
    assert (
        workflow.count("BUILD_CONTEXT_TREE_SHA256=${{ steps.source.outputs.context_tree_sha256 }}")
        >= 2
    )
    assert (
        workflow.count("CONTEXT_TREE_SHA256: ${{ steps.source.outputs.context_tree_sha256 }}") >= 3
    )
    independent_gate = workflow[
        workflow.index(
            "- name: Build and compare two independent arm64 executable projections"
        ) : workflow.index("- name: Authenticate to GHCR")
    ]
    assert independent_gate.count("verify_context") == 5
    assert '"$sealed_context" 2>&1 | tee "$log"' in independent_gate
    assert "src=${sealed_context},dst=/source,readonly" in independent_gate
    assert workflow.index(
        "- name: Revalidate the sealed C0 context before multi-platform publication"
    ) < workflow.index("- name: Build and publish the candidate C0 image")
    assert workflow.index("- name: Normalize the scientific image digest") < workflow.index(
        "- name: Verify registry identity, platforms, provenance, and SBOM"
    )


def test_embedded_context_verifier_rejects_changed_git_bytes(tmp_path: Path) -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    script = textwrap.dedent(
        workflow.split("cat > \"$context_verifier\" <<'PY'\n", maxsplit=1)[1].split(
            "\n          PY\n", maxsplit=1
        )[0]
    )
    verifier = tmp_path / "verify-context.py"
    verifier.write_text(script, encoding="utf-8")

    source_epoch = 1_700_000_000
    context = tmp_path / "context"
    source = context / "src"
    source.mkdir(parents=True)
    payloads = {
        ".dockerignore": b"*\n!src/\n!src/**\n",
        ".dockerignore.confirmatory": b"*\n!src/\n!src/**\n",
        "confirmatory-source-date-epoch.txt": f"{source_epoch}\n".encode("ascii"),
        "src/module.py": b"VALUE = 1\n",
    }
    for relative, payload in payloads.items():
        target = context / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(0o644)
        os.utime(target, (source_epoch, source_epoch))
    for directory in (source, context):
        directory.chmod(0o555)
        os.utime(directory, (source_epoch, source_epoch))

    records = []
    for relative, payload in sorted(payloads.items()):
        object_id = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        records.append(f"100644 blob {object_id}\t{relative}".encode() + b"\0")
    tree = tmp_path / "context.tree"
    tree.write_bytes(b"".join(records))
    tree.chmod(0o444)
    tree_sha256 = hashlib.sha256(tree.read_bytes()).hexdigest()

    command = [
        sys.executable,
        str(verifier),
        str(context),
        str(tree),
        str(source_epoch),
        tree_sha256,
    ]
    admitted = subprocess.run(command, check=False, capture_output=True, text=True)
    assert admitted.returncode == 0, admitted.stderr

    changed = source / "module.py"
    changed.write_bytes(b"VALUE = 2\n")
    os.utime(changed, (source_epoch, source_epoch))
    rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "file bytes differ from Git" in rejected.stderr
