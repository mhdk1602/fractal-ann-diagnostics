from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from fractal_ann_diagnostics.opa_runtime_binary import (
    C0_ARTIFACT_ATTESTATION_BUNDLE_FILENAME,
    C0_ARTIFACT_CHECKSUMS_FILENAME,
    NATIVE_BUILD_RECEIPT_IMAGE_PATH,
    OPA_BUILD_RECEIPT_IMAGE_PATH,
    OPA_RUNTIME_BINARY_PATH,
    OPA_RUNTIME_MOUNT_ROLE,
    PYTHON_RUNTIME_BINARY_PATH,
    RUNTIME_LIBRARY_MANIFEST_IMAGE_PATH,
    SQLITE_RUNTIME_LIBRARY_IMAGE_PATH,
    UV_LOCK_RUNTIME_PATH,
    ZLIB_RUNTIME_LIBRARY_IMAGE_PATH,
    GhC0ArtifactAttestationVerifier,
    OpaRuntimeBinaryError,
    load_c0_runtime_extraction_receipt,
    load_runtime_attestation_plan_template,
    materialize_opa_runtime_binary,
    materialize_retained_opa_runtime_binary,
    plan_template_paths,
    verify_opa_runtime_binary,
)
from fractal_ann_diagnostics.runtime_attestation import (
    RuntimeArtifactMount,
    RuntimeAttestationPlan,
    RuntimeFilePin,
    argv_sha256,
    environment_sha256,
    runtime_attestation_plan_template_file_bytes,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA

_IMAGE = "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@sha256:" + "a" * 64
_COMMIT = "b" * 40
_OPA_BYTES = b"\x7fELF" + b"pinned-opa-1.18.2\n"
_OPA_SHA256 = hashlib.sha256(_OPA_BYTES).hexdigest()
_PYTHON_BYTES = b"\x7fELF" + b"pinned-python-3.12.11\n"
_PYTHON_SHA256 = hashlib.sha256(_PYTHON_BYTES).hexdigest()
_UV_LOCK_BYTES = b'version = 1\nrevision = 3\nrequires-python = ">=3.10"\n'
_UV_LOCK_SHA256 = hashlib.sha256(_UV_LOCK_BYTES).hexdigest()
_MANIFEST_DIGEST = "sha256:" + "c" * 64
_NATIVE_BUILD_RECEIPT_BYTES = b'{"schema_version":"fractal-native-build-receipt-v1"}\n'
_OPA_BUILD_RECEIPT_BYTES = b'{"schema_version":"fractal-opa-build-receipt-v2"}\n'
_RUNTIME_LIBRARY_MANIFEST_BYTES = b'{"schema_version":"fractal-runtime-library-manifest-v1"}\n'
_SQLITE_LIBRARY_BYTES = b"\x7fELFretained-sqlite-library\n"
_ZLIB_LIBRARY_BYTES = b"\x7fELFretained-zlib-library\n"
_HNSW_WHEEL_BASENAME = "hnswlib-0.8.0-cp312-cp312-linux_x86_64.whl"
_HNSW_WHEEL_BYTES = b"retained-hnswlib-wheel\n"


def test_c0_attestation_verifier_pins_public_github_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = tmp_path / "retained-artifact"
    bundle = tmp_path / "attestation.bundle.json"
    subject.write_bytes(b"subject\n")
    bundle.write_bytes(b"bundle\n")
    observed: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed.extend(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'[{"verificationResult":{}}]',
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    GhC0ArtifactAttestationVerifier().verify(
        subject_path=subject,
        bundle_path=bundle,
        c0_commit=_COMMIT,
    )
    assert observed[observed.index("--hostname") + 1] == "github.com"


def _plan(corpus_id: str, *, opa_sha256: str = _OPA_SHA256) -> RuntimeAttestationPlan:
    environment = {"HOSTNAME": "fractal-confirmatory", "LANG": "C.UTF-8"}
    argv = (
        "/opt/venv/bin/python",
        "-m",
        "fractal_ann_diagnostics.production_corpus_run",
        f"/input/{corpus_id}/config.json",
        f"/output/{corpus_id}",
    )
    mounts = tuple(
        sorted(
            (
                RuntimeArtifactMount(
                    root="/input",
                    role="c1-input",
                    kind="directory",
                    artifact_sha256="1" * 64,
                ),
                RuntimeArtifactMount(
                    root="/opt/app",
                    role="c0-launcher-controls",
                    kind="directory",
                    artifact_sha256="2" * 64,
                ),
                RuntimeArtifactMount(
                    root=OPA_RUNTIME_BINARY_PATH,
                    role=OPA_RUNTIME_MOUNT_ROLE,
                    kind="file",
                    artifact_sha256=opa_sha256,
                ),
            ),
            key=lambda item: item.root.encode("utf-8"),
        )
    )
    return RuntimeAttestationPlan(
        attestation_id=f"confirmatory-{corpus_id}",
        manifest_sha256="3" * 64,
        runner_identity="github-actions-confirmatory",
        oci_image_digest=_IMAGE,
        code_commit=_COMMIT,
        operating_system_id="debian",
        operating_system_version_id="12",
        kernel_release="6.12.0",
        architecture="x86_64",
        cpu_model="AMD EPYC 7763",
        logical_cpu_count=8,
        memory_limit_bytes=16 * 1024**3,
        mount_namespace_sha256="4" * 64,
        mounts=mounts,
        argv=argv,
        argv_sha256=argv_sha256(argv),
        environment_allowlist=tuple(sorted(environment)),
        environment_sha256=environment_sha256(environment),
        opa_binary=RuntimeFilePin(path=OPA_RUNTIME_BINARY_PATH, sha256=opa_sha256),
        python_binary=RuntimeFilePin(path="/opt/venv/bin/python", sha256="5" * 64),
        python_version="3.12.11",
        uv_lock=RuntimeFilePin(path="/opt/app/uv.lock", sha256="6" * 64),
        launcher_identity=RuntimeFilePin(
            path="/opt/app/launcher-identity.json",
            sha256="7" * 64,
        ),
        workload_id=f"sealed-{corpus_id}",
        workload_sha256="8" * 64,
        invocation_marker_path=f"/output/{corpus_id}/runtime-attempt.json",
    )


def _write_plans(root: Path, *, changed_corpus: str | None = None) -> dict[str, Path]:
    for corpus_id in FIXED_CORPORA:
        directory = root / corpus_id
        directory.mkdir(parents=True)
        digest = "f" * 64 if corpus_id == changed_corpus else _OPA_SHA256
        (directory / "runtime-attestation-plan.template.json").write_bytes(
            runtime_attestation_plan_template_file_bytes(_plan(corpus_id, opa_sha256=digest))
        )
    return plan_template_paths(root)


class _Extractor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def extract(self, *, image: str, platform: str) -> bytes:
        self.calls.append((image, platform))
        return _OPA_BYTES


class _AttestationVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, str]] = []

    def verify(
        self,
        *,
        subject_path: Path,
        bundle_path: Path,
        c0_commit: str,
    ) -> bytes:
        self.calls.append((subject_path, bundle_path, c0_commit))
        return b'[{"verificationResult":{}}]'


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _retained_package(root: Path, *, overrides: dict[str, object] | None = None) -> Path:
    package = (root / "c0-package").resolve()
    runtime = package / "runtime-artifacts" / "linux-amd64"
    runtime.mkdir(parents=True)
    opa = runtime / "opa"
    opa.write_bytes(_OPA_BYTES)
    opa.chmod(0o644)
    python = runtime / "python"
    python.write_bytes(_PYTHON_BYTES)
    python.chmod(0o644)
    uv_lock = runtime / "uv.lock"
    uv_lock.write_bytes(_UV_LOCK_BYTES)
    native_build_receipt = runtime / "native-build" / "native-build-receipt.json"
    native_build_receipt.parent.mkdir()
    native_build_receipt.write_bytes(_NATIVE_BUILD_RECEIPT_BYTES)
    opa_build_receipt = runtime / "opa-build" / "opa-build-receipt.json"
    opa_build_receipt.parent.mkdir()
    opa_build_receipt.write_bytes(_OPA_BUILD_RECEIPT_BYTES)
    runtime_library_manifest = runtime / "runtime-library-manifest.json"
    runtime_library_manifest.write_bytes(_RUNTIME_LIBRARY_MANIFEST_BYTES)
    sqlite_library = runtime / "libsqlite3.so.0"
    sqlite_library.write_bytes(_SQLITE_LIBRARY_BYTES)
    zlib_library = runtime / "libz.so.1"
    zlib_library.write_bytes(_ZLIB_LIBRARY_BYTES)
    hnswlib_wheel = runtime / "hnswlib" / _HNSW_WHEEL_BASENAME
    hnswlib_wheel.parent.mkdir()
    hnswlib_wheel.write_bytes(_HNSW_WHEEL_BYTES)
    hnswlib_receipt = runtime / "hnswlib-runtime-receipt.json"
    hnswlib_receipt.write_bytes(
        _canonical_bytes(
            {
                "extension_basename": "hnswlib.cpython-312-x86_64-linux-gnu.so",
                "extension_byte_count": 1,
                "extension_sha256": "9" * 64,
                "package": "hnswlib",
                "python_abi": "cp312",
                "schema_version": "fractal-hnswlib-runtime-artifact-v1",
                "sdist_sha256": (
                    "cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c"
                ),
                "version": "0.8.0",
                "wheel_basename": _HNSW_WHEEL_BASENAME,
                "wheel_byte_count": len(_HNSW_WHEEL_BYTES),
                "wheel_sha256": hashlib.sha256(_HNSW_WHEEL_BYTES).hexdigest(),
            }
        )
    )
    receipt_payload: dict[str, object] = {
        "c0_sha": _COMMIT,
        "hnswlib_receipt_image_path": "/opt/artifacts/hnswlib-runtime-receipt.json",
        "hnswlib_receipt_sha256": hashlib.sha256(hnswlib_receipt.read_bytes()).hexdigest(),
        "hnswlib_wheel_basename": _HNSW_WHEEL_BASENAME,
        "hnswlib_wheel_byte_count": len(_HNSW_WHEEL_BYTES),
        "hnswlib_wheel_image_path": f"/opt/artifacts/hnswlib/{_HNSW_WHEEL_BASENAME}",
        "hnswlib_wheel_sha256": hashlib.sha256(_HNSW_WHEEL_BYTES).hexdigest(),
        "image_digest": _IMAGE.rsplit("@", 1)[1],
        "image_manifest_digest": _MANIFEST_DIGEST,
        "image_reference": _IMAGE,
        "native_build_receipt_image_path": NATIVE_BUILD_RECEIPT_IMAGE_PATH,
        "native_build_receipt_sha256": hashlib.sha256(_NATIVE_BUILD_RECEIPT_BYTES).hexdigest(),
        "opa_build_receipt_image_path": OPA_BUILD_RECEIPT_IMAGE_PATH,
        "opa_build_receipt_sha256": hashlib.sha256(_OPA_BUILD_RECEIPT_BYTES).hexdigest(),
        "opa_byte_count": len(_OPA_BYTES),
        "opa_image_path": OPA_RUNTIME_BINARY_PATH,
        "opa_sha256": _OPA_SHA256,
        "platform": "linux/amd64",
        "python_binary_byte_count": len(_PYTHON_BYTES),
        "python_binary_image_path": PYTHON_RUNTIME_BINARY_PATH,
        "python_binary_sha256": _PYTHON_SHA256,
        "runtime_library_manifest_image_path": RUNTIME_LIBRARY_MANIFEST_IMAGE_PATH,
        "runtime_library_manifest_sha256": hashlib.sha256(
            _RUNTIME_LIBRARY_MANIFEST_BYTES
        ).hexdigest(),
        "schema_version": "fractal-c0-runtime-extraction-v3",
        "source_date_epoch": 1_750_000_000,
        "sqlite_library_byte_count": len(_SQLITE_LIBRARY_BYTES),
        "sqlite_library_image_path": SQLITE_RUNTIME_LIBRARY_IMAGE_PATH,
        "sqlite_library_sha256": hashlib.sha256(_SQLITE_LIBRARY_BYTES).hexdigest(),
        "uv_lock_byte_count": len(_UV_LOCK_BYTES),
        "uv_lock_image_path": UV_LOCK_RUNTIME_PATH,
        "uv_lock_sha256": _UV_LOCK_SHA256,
        "zlib_library_byte_count": len(_ZLIB_LIBRARY_BYTES),
        "zlib_library_image_path": ZLIB_RUNTIME_LIBRARY_IMAGE_PATH,
        "zlib_library_sha256": hashlib.sha256(_ZLIB_LIBRARY_BYTES).hexdigest(),
    }
    if overrides:
        receipt_payload.update(overrides)
    receipt = runtime / "runtime-extraction.json"
    receipt.write_bytes(_canonical_bytes(receipt_payload))
    checksums = {
        "runtime-artifacts/linux-amd64/opa": hashlib.sha256(opa.read_bytes()).hexdigest(),
        "runtime-artifacts/linux-amd64/python": hashlib.sha256(python.read_bytes()).hexdigest(),
        "runtime-artifacts/linux-amd64/runtime-extraction.json": hashlib.sha256(
            receipt.read_bytes()
        ).hexdigest(),
        "runtime-artifacts/linux-amd64/uv.lock": hashlib.sha256(uv_lock.read_bytes()).hexdigest(),
        "runtime-artifacts/linux-amd64/hnswlib-runtime-receipt.json": hashlib.sha256(
            hnswlib_receipt.read_bytes()
        ).hexdigest(),
        f"runtime-artifacts/linux-amd64/hnswlib/{_HNSW_WHEEL_BASENAME}": hashlib.sha256(
            hnswlib_wheel.read_bytes()
        ).hexdigest(),
        "runtime-artifacts/linux-amd64/native-build/native-build-receipt.json": (
            hashlib.sha256(native_build_receipt.read_bytes()).hexdigest()
        ),
        "runtime-artifacts/linux-amd64/opa-build/opa-build-receipt.json": (
            hashlib.sha256(opa_build_receipt.read_bytes()).hexdigest()
        ),
        "runtime-artifacts/linux-amd64/runtime-library-manifest.json": (
            hashlib.sha256(runtime_library_manifest.read_bytes()).hexdigest()
        ),
        "runtime-artifacts/linux-amd64/libsqlite3.so.0": hashlib.sha256(
            sqlite_library.read_bytes()
        ).hexdigest(),
        "runtime-artifacts/linux-amd64/libz.so.1": hashlib.sha256(
            zlib_library.read_bytes()
        ).hexdigest(),
    }
    (package / C0_ARTIFACT_CHECKSUMS_FILENAME).write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(checksums.items())),
        encoding="ascii",
    )
    (package / C0_ARTIFACT_ATTESTATION_BUNDLE_FILENAME).write_bytes(b"{}\n")
    return package


def test_materialize_extracts_once_and_binds_all_five_templates(tmp_path: Path) -> None:
    plan_root = (tmp_path / "runtime").resolve()
    plan_root.mkdir()
    paths = _write_plans(plan_root)
    extractor = _Extractor()
    output = plan_root / "opa"

    result = materialize_opa_runtime_binary(
        image=_IMAGE,
        plan_paths=paths,
        output_path=output,
        extractor=extractor,
    )

    assert extractor.calls == [(_IMAGE, "linux/amd64")]
    assert output.read_bytes() == _OPA_BYTES
    assert output.stat().st_mode & 0o777 == 0o555
    assert result.binary_sha256 == _OPA_SHA256
    assert result.image == _IMAGE
    assert result.code_commit == _COMMIT
    assert dict(result.plan_template_sha256_by_corpus) == {
        corpus_id: hashlib.sha256(
            (plan_root / corpus_id / "runtime-attestation-plan.template.json").read_bytes()
        ).hexdigest()
        for corpus_id in FIXED_CORPORA
    }
    assert verify_opa_runtime_binary(output, image=_IMAGE, plan_paths=paths) == result

    with pytest.raises(OpaRuntimeBinaryError, match="already exists"):
        materialize_opa_runtime_binary(
            image=_IMAGE,
            plan_paths=paths,
            output_path=output,
            extractor=extractor,
        )


def test_materialize_rejects_any_plan_or_image_disagreement(tmp_path: Path) -> None:
    plan_root = (tmp_path / "runtime").resolve()
    plan_root.mkdir()
    paths = _write_plans(plan_root, changed_corpus="bright")
    with pytest.raises(OpaRuntimeBinaryError, match="bright runtime plan OPA digest"):
        materialize_opa_runtime_binary(
            image=_IMAGE,
            plan_paths=paths,
            output_path=plan_root / "opa",
            extractor=_Extractor(),
        )
    assert not (plan_root / "opa").exists()

    paths = _write_plans((tmp_path / "second-runtime").resolve())
    with pytest.raises(OpaRuntimeBinaryError, match="another C0 OCI image"):
        materialize_opa_runtime_binary(
            image="ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@sha256:" + "c" * 64,
            plan_paths=paths,
            output_path=(tmp_path / "second-runtime" / "opa").resolve(),
            extractor=_Extractor(),
        )


def test_template_loader_rejects_changes_outside_the_manifest_token(tmp_path: Path) -> None:
    plan = _plan("scifact")
    path = (tmp_path / "plan.json").resolve()
    path.write_bytes(runtime_attestation_plan_template_file_bytes(plan))
    assert load_runtime_attestation_plan_template(path).opa_binary.sha256 == _OPA_SHA256

    path.write_bytes(path.read_bytes().replace(b'"{manifest_sha256}"', b'"' + b"9" * 64 + b'"'))
    with pytest.raises(OpaRuntimeBinaryError, match="manifest token"):
        load_runtime_attestation_plan_template(path)


def test_verifier_rejects_writable_or_hard_linked_binary(tmp_path: Path) -> None:
    plan_root = (tmp_path / "runtime").resolve()
    plan_root.mkdir()
    paths = _write_plans(plan_root)
    opa = plan_root / "opa"
    opa.write_bytes(_OPA_BYTES)
    opa.chmod(0o755)
    with pytest.raises(OpaRuntimeBinaryError, match="non-writable"):
        verify_opa_runtime_binary(opa, image=_IMAGE, plan_paths=paths)

    opa.chmod(0o555)
    alias = plan_root / "opa-alias"
    alias.hardlink_to(opa)
    with pytest.raises(OpaRuntimeBinaryError, match="non-writable"):
        verify_opa_runtime_binary(opa, image=_IMAGE, plan_paths=paths)


def test_plan_set_cannot_omit_or_reassign_a_corpus(tmp_path: Path) -> None:
    root = (tmp_path / "runtime").resolve()
    root.mkdir()
    paths = _write_plans(root)
    paths.pop("bright")
    with pytest.raises(OpaRuntimeBinaryError, match="every fixed corpus"):
        materialize_opa_runtime_binary(
            image=_IMAGE,
            plan_paths=paths,
            output_path=root / "opa",
            extractor=_Extractor(),
        )


def test_plan_contract_rejects_mutable_or_wrong_role_mount(tmp_path: Path) -> None:
    root = (tmp_path / "runtime").resolve()
    root.mkdir()
    plan = _plan("scifact")
    opa_mount = next(mount for mount in plan.mounts if mount.root == OPA_RUNTIME_BINARY_PATH)
    changed = replace(opa_mount, role="image-baked-tool")
    mounts = tuple(changed if mount == opa_mount else mount for mount in plan.mounts)
    path = root / "scifact"
    path.mkdir()
    (path / "runtime-attestation-plan.template.json").write_bytes(
        runtime_attestation_plan_template_file_bytes(replace(plan, mounts=mounts))
    )
    for corpus_id in FIXED_CORPORA[1:]:
        directory = root / corpus_id
        directory.mkdir()
        (directory / "runtime-attestation-plan.template.json").write_bytes(
            runtime_attestation_plan_template_file_bytes(_plan(corpus_id))
        )
    with pytest.raises(OpaRuntimeBinaryError, match="mount differs"):
        materialize_opa_runtime_binary(
            image=_IMAGE,
            plan_paths=plan_template_paths(root),
            output_path=root / "opa",
            extractor=_Extractor(),
        )


def test_retained_materializer_verifies_receipt_attestations_and_five_plans(
    tmp_path: Path,
) -> None:
    plan_root = (tmp_path / "runtime").resolve()
    plan_root.mkdir()
    paths = _write_plans(plan_root)
    package = _retained_package(tmp_path)
    verifier = _AttestationVerifier()
    output = plan_root / "opa"

    result = materialize_retained_opa_runtime_binary(
        c0_package_root=package,
        image=_IMAGE,
        plan_paths=paths,
        output_path=output,
        attestation_verifier=verifier,
    )

    receipt = package / "runtime-artifacts/linux-amd64/runtime-extraction.json"
    retained_opa = package / "runtime-artifacts/linux-amd64/opa"
    retained_python = package / "runtime-artifacts/linux-amd64/python"
    retained_uv_lock = package / "runtime-artifacts/linux-amd64/uv.lock"
    retained_hnswlib_receipt = (
        package / "runtime-artifacts/linux-amd64/hnswlib-runtime-receipt.json"
    )
    retained_hnswlib_wheel = (
        package / "runtime-artifacts/linux-amd64/hnswlib" / _HNSW_WHEEL_BASENAME
    )
    retained_native_build_receipt = (
        package / "runtime-artifacts/linux-amd64/native-build/native-build-receipt.json"
    )
    retained_opa_build_receipt = (
        package / "runtime-artifacts/linux-amd64/opa-build/opa-build-receipt.json"
    )
    retained_runtime_library_manifest = (
        package / "runtime-artifacts/linux-amd64/runtime-library-manifest.json"
    )
    retained_sqlite_library = package / "runtime-artifacts/linux-amd64/libsqlite3.so.0"
    retained_zlib_library = package / "runtime-artifacts/linux-amd64/libz.so.1"
    bundle = package / C0_ARTIFACT_ATTESTATION_BUNDLE_FILENAME
    assert verifier.calls == [
        (receipt, bundle, _COMMIT),
        (retained_opa, bundle, _COMMIT),
        (retained_python, bundle, _COMMIT),
        (retained_uv_lock, bundle, _COMMIT),
        (retained_hnswlib_receipt, bundle, _COMMIT),
        (retained_hnswlib_wheel, bundle, _COMMIT),
        (retained_native_build_receipt, bundle, _COMMIT),
        (retained_opa_build_receipt, bundle, _COMMIT),
        (retained_runtime_library_manifest, bundle, _COMMIT),
        (retained_sqlite_library, bundle, _COMMIT),
        (retained_zlib_library, bundle, _COMMIT),
    ]
    assert output.read_bytes() == _OPA_BYTES
    assert output.stat().st_mode & 0o777 == 0o555
    assert result.runtime_binding.binary_sha256 == _OPA_SHA256
    assert result.runtime_binding.platform == "linux/amd64"
    assert result.selected_manifest_digest == _MANIFEST_DIGEST
    assert result.extraction_receipt_sha256 == hashlib.sha256(receipt.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image_reference": _IMAGE.replace("a" * 64, "f" * 64)}, "index digest disagree"),
        ({"platform": "linux/arm64"}, "another OCI platform"),
        ({"c0_sha": "f" * 40}, "another C0 commit"),
        ({"opa_sha256": "f" * 64}, "retained OPA bytes differ"),
        ({"opa_byte_count": len(_OPA_BYTES) + 1}, "retained OPA bytes differ"),
        ({"python_binary_sha256": "f" * 64}, "retained Python bytes differ"),
        (
            {"python_binary_byte_count": len(_PYTHON_BYTES) + 1},
            "retained Python bytes differ",
        ),
        ({"uv_lock_sha256": "f" * 64}, "retained uv lock differs"),
        ({"uv_lock_byte_count": len(_UV_LOCK_BYTES) + 1}, "retained uv lock differs"),
        ({"hnswlib_receipt_sha256": "f" * 64}, "retained hnswlib receipt differs"),
        ({"hnswlib_wheel_sha256": "f" * 64}, "retained hnswlib wheel differs"),
        (
            {"hnswlib_wheel_byte_count": len(_HNSW_WHEEL_BYTES) + 1},
            "retained hnswlib wheel differs",
        ),
        (
            {"native_build_receipt_sha256": "f" * 64},
            "retained native-build receipt differs",
        ),
        (
            {"opa_build_receipt_sha256": "f" * 64},
            "retained OPA-build receipt differs",
        ),
        (
            {"runtime_library_manifest_sha256": "f" * 64},
            "retained runtime-library manifest differs",
        ),
        ({"sqlite_library_sha256": "f" * 64}, "retained SQLite library differs"),
        (
            {"sqlite_library_byte_count": len(_SQLITE_LIBRARY_BYTES) + 1},
            "retained SQLite library differs",
        ),
        ({"zlib_library_sha256": "f" * 64}, "retained zlib library differs"),
        (
            {"zlib_library_byte_count": len(_ZLIB_LIBRARY_BYTES) + 1},
            "retained zlib library differs",
        ),
        ({"python_binary_image_path": "/usr/bin/python"}, "another Python image path"),
        ({"uv_lock_image_path": "/workspace/uv.lock"}, "another uv lock image path"),
        (
            {"native_build_receipt_image_path": "/tmp/native.json"},
            "another native-build receipt path",
        ),
        (
            {"opa_build_receipt_image_path": "/tmp/opa.json"},
            "another OPA-build receipt path",
        ),
        (
            {"runtime_library_manifest_image_path": "/tmp/libraries.json"},
            "another runtime-library manifest path",
        ),
        (
            {"sqlite_library_image_path": "/tmp/libsqlite3.so.0"},
            "another SQLite library path",
        ),
        (
            {"zlib_library_image_path": "/tmp/libz.so.1"},
            "another zlib library path",
        ),
    ],
)
def test_retained_materializer_rejects_receipt_disagreement_before_attestation(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    plan_root = (tmp_path / "runtime").resolve()
    plan_root.mkdir()
    paths = _write_plans(plan_root)
    package = _retained_package(tmp_path, overrides=overrides)
    verifier = _AttestationVerifier()

    with pytest.raises(OpaRuntimeBinaryError, match=message):
        materialize_retained_opa_runtime_binary(
            c0_package_root=package,
            image=_IMAGE,
            plan_paths=paths,
            output_path=plan_root / "opa",
            attestation_verifier=verifier,
        )
    assert verifier.calls == []
    assert not (plan_root / "opa").exists()


@pytest.mark.parametrize(
    ("retained_name", "message"),
    [
        ("python", "retained Python bytes differ"),
        ("uv.lock", "retained uv lock differs"),
        ("hnswlib-runtime-receipt.json", "retained hnswlib receipt differs"),
        (f"hnswlib/{_HNSW_WHEEL_BASENAME}", "retained hnswlib wheel differs"),
        (
            "native-build/native-build-receipt.json",
            "retained native-build receipt differs",
        ),
        ("opa-build/opa-build-receipt.json", "retained OPA-build receipt differs"),
        (
            "runtime-library-manifest.json",
            "retained runtime-library manifest differs",
        ),
        ("libsqlite3.so.0", "retained SQLite library differs"),
        ("libz.so.1", "retained zlib library differs"),
    ],
)
def test_retained_materializer_rejects_bound_runtime_file_substitution(
    tmp_path: Path,
    retained_name: str,
    message: str,
) -> None:
    plan_root = (tmp_path / "runtime").resolve()
    plan_root.mkdir()
    paths = _write_plans(plan_root)
    package = _retained_package(tmp_path)
    target = package / "runtime-artifacts/linux-amd64" / retained_name
    target.write_bytes(target.read_bytes() + b"substitution")
    verifier = _AttestationVerifier()

    with pytest.raises(OpaRuntimeBinaryError, match=message):
        materialize_retained_opa_runtime_binary(
            c0_package_root=package,
            image=_IMAGE,
            plan_paths=paths,
            output_path=plan_root / "opa",
            attestation_verifier=verifier,
        )
    assert verifier.calls == []


def test_retained_materializer_rejects_checksum_or_receipt_schema_substitution(
    tmp_path: Path,
) -> None:
    plan_root = (tmp_path / "runtime").resolve()
    plan_root.mkdir()
    paths = _write_plans(plan_root)
    package = _retained_package(tmp_path)
    checksums = package / C0_ARTIFACT_CHECKSUMS_FILENAME
    checksums.write_text(
        checksums.read_text(encoding="ascii").replace(_OPA_SHA256, "f" * 64),
        encoding="ascii",
    )
    with pytest.raises(OpaRuntimeBinaryError, match="checksums do not bind"):
        materialize_retained_opa_runtime_binary(
            c0_package_root=package,
            image=_IMAGE,
            plan_paths=paths,
            output_path=plan_root / "opa",
            attestation_verifier=_AttestationVerifier(),
        )

    second = _retained_package(tmp_path / "second", overrides={"unknown": "field"})
    receipt_path = second / "runtime-artifacts/linux-amd64/runtime-extraction.json"
    with pytest.raises(OpaRuntimeBinaryError, match="schema mismatch"):
        load_c0_runtime_extraction_receipt(receipt_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            field,
            value,
            message,
        )
        for field, message in (
            ("hnswlib_wheel_byte_count", "hnsw wheel byte count is invalid"),
            ("sqlite_library_byte_count", "SQLite library byte count is invalid"),
            ("zlib_library_byte_count", "zlib library byte count is invalid"),
        )
        for value in (True, 0, 512 * 1024 * 1024 + 1)
    ],
)
def test_runtime_extraction_loader_rejects_invalid_retained_byte_count_boundaries(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    package = _retained_package(tmp_path, overrides={field: value})
    receipt_path = package / "runtime-artifacts/linux-amd64/runtime-extraction.json"

    with pytest.raises(OpaRuntimeBinaryError, match=message):
        load_c0_runtime_extraction_receipt(receipt_path)


def test_retained_materializer_rejects_receipt_change_after_typed_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal_ann_diagnostics import opa_runtime_binary as runtime_module

    plan_root = (tmp_path / "runtime").resolve()
    plan_root.mkdir()
    paths = _write_plans(plan_root)
    package = _retained_package(tmp_path)
    receipt_path = package / "runtime-artifacts/linux-amd64/runtime-extraction.json"
    verifier = _AttestationVerifier()
    original_read = runtime_module.read_secure_regular_file
    receipt_reads = 0

    def changing_read(path: str | Path, **kwargs: object) -> bytes:
        nonlocal receipt_reads
        encoded = original_read(path, **kwargs)  # type: ignore[arg-type]
        if Path(path) == receipt_path:
            receipt_reads += 1
            if receipt_reads == 2:
                return encoded + b" "
        return encoded

    monkeypatch.setattr(runtime_module, "read_secure_regular_file", changing_read)
    with pytest.raises(OpaRuntimeBinaryError, match="changed after typed admission"):
        materialize_retained_opa_runtime_binary(
            c0_package_root=package,
            image=_IMAGE,
            plan_paths=paths,
            output_path=plan_root / "opa",
            attestation_verifier=verifier,
        )
    assert receipt_reads == 2
    assert verifier.calls == []
    assert not (plan_root / "opa").exists()


def test_retained_materializer_rejects_linked_source_and_existing_output(tmp_path: Path) -> None:
    plan_root = (tmp_path / "runtime").resolve()
    plan_root.mkdir()
    paths = _write_plans(plan_root)
    package = _retained_package(tmp_path)
    source = package / "runtime-artifacts/linux-amd64/opa"
    alias = source.with_name("opa-alias")
    alias.hardlink_to(source)
    with pytest.raises(OpaRuntimeBinaryError, match="hard-linked"):
        materialize_retained_opa_runtime_binary(
            c0_package_root=package,
            image=_IMAGE,
            plan_paths=paths,
            output_path=plan_root / "opa",
            attestation_verifier=_AttestationVerifier(),
        )

    alias.unlink()
    output = plan_root / "opa"
    output.write_bytes(b"occupied")
    with pytest.raises(OpaRuntimeBinaryError, match="already exists"):
        materialize_retained_opa_runtime_binary(
            c0_package_root=package,
            image=_IMAGE,
            plan_paths=paths,
            output_path=output,
            attestation_verifier=_AttestationVerifier(),
        )
