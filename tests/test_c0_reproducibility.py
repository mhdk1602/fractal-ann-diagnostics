from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import stat
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

import fractal_ann_diagnostics.c0_reproducibility as reproducibility
from fractal_ann_diagnostics.c0_reproducibility import (
    C0ReproducibilityError,
    compare_c0_oci_archives,
    compare_tle_release_oci_archives,
    main,
)

BUILD_CONTEXT_TREE_SHA256 = "3" * 64
SOURCE_EPOCH = 1_700_000_000
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
OCI_LAYER_GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"
WHEEL_NAME = "hnswlib-0.8.0-cp312-cp312-linux_aarch64.whl"
EXTENSION_NAME = "hnswlib.cpython-312-aarch64-linux-gnu.so"
UV_LOCK = b"version = 1\n"
OPA_POLICY = b"package fractal.confirmatory\ndefault allow := false\n"
UV_LOCK_SHA256 = hashlib.sha256(UV_LOCK).hexdigest()
OPA_POLICY_SHA256 = hashlib.sha256(OPA_POLICY).hexdigest()
SQLITE_LIBRARY = b"synthetic-sqlite"
ZLIB_LIBRARY = b"synthetic-zlib"


def _elf(marker: bytes, *, machine: int = 183) -> bytes:
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    header[6] = 1
    header[16:18] = (3).to_bytes(2, "little")
    header[18:20] = machine.to_bytes(2, "little")
    return bytes(header) + marker


def _static_tle_elf(marker: bytes = b"synthetic-static-tle") -> bytes:
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    header[6] = 1
    header[16:18] = (2).to_bytes(2, "little")
    header[18:20] = (183).to_bytes(2, "little")
    header[32:40] = (64).to_bytes(8, "little")
    header[52:54] = (64).to_bytes(2, "little")
    header[54:56] = (56).to_bytes(2, "little")
    header[56:58] = (1).to_bytes(2, "little")
    program = bytearray(56)
    program[:4] = (1).to_bytes(4, "little")
    return bytes(header + program) + marker


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _descriptor(payload: bytes, media_type: str) -> dict[str, object]:
    return {
        "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "mediaType": media_type,
        "size": len(payload),
    }


def _wheel(*, member_name: str = EXTENSION_NAME) -> tuple[bytes, bytes]:
    extension = _elf(b"synthetic-arm64-hnsw-extension")
    output = io.BytesIO()
    timestamp = time.gmtime(SOURCE_EPOCH - SOURCE_EPOCH % 2)[:6]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        member = zipfile.ZipInfo(member_name, date_time=timestamp)
        member.compress_type = zipfile.ZIP_DEFLATED
        member.external_attr = (stat.S_IFREG | 0o444) << 16
        archive.writestr(member, extension)
    return output.getvalue(), extension


def _receipt(
    wheel: bytes,
    extension: bytes,
    *,
    canonical: bool = True,
    crossed: bool = False,
) -> bytes:
    value = {
        "extension_basename": EXTENSION_NAME,
        "extension_byte_count": len(extension),
        "extension_sha256": hashlib.sha256(extension).hexdigest(),
        "package": "hnswlib",
        "python_abi": "cp312",
        "schema_version": "fractal-hnswlib-runtime-artifact-v1",
        "sdist_sha256": "cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c",
        "version": "0.8.0",
        "wheel_basename": WHEEL_NAME,
        "wheel_byte_count": len(wheel),
        "wheel_sha256": "0" * 64 if crossed else hashlib.sha256(wheel).hexdigest(),
    }
    if canonical:
        return _canonical(value) + b"\n"
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def _native_receipt() -> bytes:
    value = {
        "debian_inrelease_sha256": (
            "77737fa4b34f2693e982cc9ee35736816c35a7778fc2d326cc1bbf5b301fe1aa"
        ),
        "debian_keyring_sha256": (
            "506b815cbb32d9b6066b4a2aa524071e071761e7e7f68c3ac74f3061ba852017"
        ),
        "debian_snapshot": "20260714T000000Z",
        "schema_version": "fractal-native-build-receipt-v1",
        "source_date_epoch": SOURCE_EPOCH,
        "sqlite_autoconf": "3530300",
        "sqlite_library_sha256": hashlib.sha256(SQLITE_LIBRARY).hexdigest(),
        "sqlite_sha256": ("c917d7db16648ec95f714974ace5e5dcf46b7dc70e26600a0a102a3141125db0"),
        "sqlite_sha3_256": ("98f2b3f3c11be6a03ea32346937b032c2472ebbd7a716bed36ca2f5693e7ce8b"),
        "sqlite_source_id": (
            "2026-06-26 20:14:12 d4c0e51e4aeb96955b99185ab9cde75c339e2c29c3f3f12428d364a10d782c62"
        ),
        "zlib_library_sha256": hashlib.sha256(ZLIB_LIBRARY).hexdigest(),
        "zlib_sha256": ("bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16"),
        "zlib_version": "1.3.2",
    }
    return _canonical(value) + b"\n"


def _opa_receipt(opa: bytes) -> bytes:
    value = {
        "dependency_delta_sha256": (
            "400699e81344ff2114fc5d2254734cb84a7015a68840505a7ab6a05df0dd39e0"
        ),
        "go_builder_image": (
            "docker.io/library/golang:1.26.5-bookworm@"
            "sha256:1ecb7edf62a0408027bd5729dfd6b1b8766e578e8df93995b225dfd0944eb651"
        ),
        "go_tarball_sha256": ("fe4789e92b1f33358680864bbe8704289e7bb5fc207d80623c308935bd696d49"),
        "go_version": "1.26.5",
        "module_versions": {
            "github.com/klauspost/compress": {"original": "1.18.5", "patched": "1.18.7"},
            "golang.org/x/crypto": {"original": "0.52.0", "patched": "0.53.0"},
            "golang.org/x/mod": {"original": "0.36.0", "patched": "0.37.0"},
            "golang.org/x/net": {"original": "0.55.0", "patched": "0.56.0"},
            "golang.org/x/sync": {"original": "0.21.0", "patched": "0.22.0"},
            "golang.org/x/sys": {"original": "0.45.0", "patched": "0.46.0"},
            "golang.org/x/text": {"original": "0.38.0", "patched": "0.40.0"},
            "golang.org/x/tools": {"original": "0.45.0", "patched": "0.47.0"},
            "google.golang.org/grpc": {"original": "1.81.1", "patched": "1.82.1"},
            "oras.land/oras-go/v2": {"original": "2.6.1", "patched": "2.6.2"},
        },
        "opa_commit": "e695c9ef8edb0f8b9f13d014d7bc8a7fbcc57297",
        "opa_release_timestamp": "2026-07-02T13:14:00Z",
        "opa_sha256": hashlib.sha256(opa).hexdigest(),
        "opa_source_sha256": ("a8b3ecdc925b75bdade52d315aa13efaa51c2de99acb78003ad353cce6e9e637"),
        "opa_version": "1.18.2",
        "original_go_mod_sha256": (
            "59b4beeea1af5d33ce1c22579e24ab4b0002a3638d0aadeb82a5a4500eb8a175"
        ),
        "original_go_sum_sha256": (
            "be7b973025c1a5588a822baed9513f7356e08a6794fa24db79b8fb832cee6b2f"
        ),
        "patched_go_mod_sha256": (
            "7a4e0b0a05ad266401896008bff46c6dd822e647c1ddcda26de44ccdb781fdb3"
        ),
        "patched_go_sum_sha256": (
            "6b6d66e548bce5eb3b4613daed39d87e563b99fcda36f286dabf1694b93195e1"
        ),
        "schema_version": "fractal-opa-build-receipt-v2",
        "source_date_epoch": SOURCE_EPOCH,
        "target_arch": "arm64",
    }
    return _canonical(value) + b"\n"


def _tle_receipt(tle: bytes) -> bytes:
    value = {
        "binary_byte_count": len(tle),
        "binary_image_path": "/usr/local/bin/tle",
        "binary_sha256": hashlib.sha256(tle).hexdigest(),
        "build_commands": [
            [
                "go",
                "build",
                "-trimpath",
                "-buildvcs=false",
                "-ldflags=-s -w -buildid=",
                "-o",
                output,
                "./cmd/tle",
            ]
            for output in ("/tmp/tle-a", "/tmp/tle-b")
        ],
        "build_environment": {
            "CGO_ENABLED": "0",
            "GOARCH": "arm64",
            "GOARM64": "v8.0",
            "GOOS": "linux",
            "GOTOOLCHAIN": "local",
        },
        "builder_image": reproducibility._GO_BUILDER_IMAGE,
        "dependency_delta_sha256": reproducibility._TLE_DEPENDENCY_DELTA_SHA256,
        "elf": {
            "class": "ELF64",
            "dynamic_program_header": False,
            "machine": "AArch64",
            "pt_interp": False,
            "type": "ET_EXEC",
        },
        "go_tarball_sha256": reproducibility._TLE_GO_TARBALL_SHA256,
        "go_tarball_url": "https://go.dev/dl/go1.26.5.linux-arm64.tar.gz",
        "go_tool_sha256": reproducibility._TLE_GO_TOOL_SHA256,
        "go_version": "1.26.5",
        "included": True,
        "independent_build_count": 2,
        "independent_builds_byte_identical": True,
        "offline_test_inventory_sha256": "8" * 64,
        "original_go_mod_sha256": reproducibility._TLE_ORIGINAL_GO_MOD_SHA256,
        "original_go_sum_sha256": reproducibility._TLE_ORIGINAL_GO_SUM_SHA256,
        "patched_go_mod_sha256": reproducibility._TLE_PATCHED_GO_MOD_SHA256,
        "patched_go_sum_sha256": reproducibility._TLE_PATCHED_GO_SUM_SHA256,
        "release_version": "1.2.0",
        "schema_version": "fractal-tle-source-build-receipt-v2",
        "source_archive_sha256": reproducibility._TLE_SOURCE_ARCHIVE_SHA256,
        "source_archive_url": (
            f"https://github.com/drand/tlock/archive/{reproducibility._TLE_SOURCE_COMMIT}.tar.gz"
        ),
        "source_commit_git_sha1": reproducibility._TLE_SOURCE_COMMIT,
        "source_date_epoch": SOURCE_EPOCH,
        "source_tree_manifest_sha256": (reproducibility._TLE_SOURCE_TREE_MANIFEST_SHA256),
        "tag_object_git_sha1": reproducibility._TLE_TAG_OBJECT,
        "target_arch": "arm64",
        "target_os": "linux",
    }
    return _canonical(value) + b"\n"


def _runtime_library_manifest() -> bytes:
    libraries = []
    for path, package, version, purl, payload in (
        (
            "/opt/native-libs/libsqlite3.so.0",
            "sqlite",
            "3.53.3",
            "pkg:generic/sqlite@3.53.3",
            SQLITE_LIBRARY,
        ),
        (
            "/opt/native-libs/libz.so.1",
            "zlib",
            "1.3.2",
            "pkg:generic/zlib@1.3.2",
            ZLIB_LIBRARY,
        ),
    ):
        libraries.append(
            {
                "byte_count": len(payload),
                "destination_path": path,
                "package": package,
                "package_version": version,
                "purl": purl,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_path": path,
            }
        )
    return (
        _canonical(
            {
                "libraries": libraries,
                "schema_version": "fractal-runtime-library-manifest-v1",
            }
        )
        + b"\n"
    )


def _layer(
    *,
    opa: bytes | None = None,
    opa_policy: bytes = OPA_POLICY,
    uv_lock: bytes = UV_LOCK,
    noncanonical_receipt: bool = False,
    crossed_receipt: bool = False,
    special: str | None = None,
    unrelated: bytes = b"first",
    tle: bytes | None = None,
    dot_prefix: bool = False,
    writable_runtime_target: str | None = None,
) -> bytes:
    if opa is None:
        opa = _elf(b"synthetic-arm64-opa")
    wheel_member_name = f"./{EXTENSION_NAME}" if special == "wheel-dot-prefix" else EXTENSION_NAME
    wheel, extension = _wheel(member_name=wheel_member_name)
    receipt = _receipt(
        wheel,
        extension,
        canonical=not noncanonical_receipt,
        crossed=crossed_receipt,
    )
    opa_build_info = (
        b"/usr/local/bin/opa: go1.26.5\n"
        b"\tdep\tgithub.com/klauspost/compress\tv1.18.7\n"
        b"\tdep\toras.land/oras-go/v2\tv2.6.2\n"
        b"\tdep\tgolang.org/x/crypto\tv0.53.0\n"
        b"\tdep\tgolang.org/x/net\tv0.56.0\n"
        b"\tdep\tgolang.org/x/sync\tv0.22.0\n"
        b"\tdep\tgolang.org/x/sys\tv0.46.0\n"
        b"\tdep\tgolang.org/x/text\tv0.40.0\n"
        b"\tdep\tgoogle.golang.org/grpc\tv1.82.1\n"
        b"\tbuild\tCGO_ENABLED=0\n"
    )
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        if dot_prefix:
            root = tarfile.TarInfo(".")
            root.type = tarfile.DIRTYPE
            root.mode = 0o755
            archive.addfile(root)
            directory = tarfile.TarInfo("./etc/")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            archive.addfile(directory)
        rows = [
            ("usr/local/bin/opa", opa, 0o555),
            ("opt/app/policy/opa_compiled_masks.rego", opa_policy, 0o444),
            ("opt/venv/bin/python", _elf(b"synthetic-arm64-python"), 0o555),
            ("opt/app/uv.lock", uv_lock, 0o444),
            ("opt/artifacts/hnswlib-runtime-receipt.json", receipt, 0o444),
            (
                "opt/artifacts/runtime-library-manifest.json",
                _runtime_library_manifest(),
                0o444,
            ),
            (
                "opt/artifacts/native-build/native-build-receipt.json",
                _native_receipt(),
                0o444,
            ),
            ("opt/artifacts/native-build/compiler-closure.tsv", b"gcc\t1\n", 0o444),
            ("opt/native-libs/libsqlite3.so.0", SQLITE_LIBRARY, 0o444),
            ("opt/native-libs/libz.so.1", ZLIB_LIBRARY, 0o444),
            (
                "opt/artifacts/opa-build/opa-build-receipt.json",
                _opa_receipt(opa),
                0o444,
            ),
            ("opt/artifacts/opa-build/opa-go-build-info.txt", opa_build_info, 0o444),
            ("var/lib/dpkg/status", b"Package: libgcc-s1\n", 0o444),
            (f"opt/artifacts/hnswlib/{WHEEL_NAME}", wheel, 0o444),
            ("opt/app/src/unrelated.txt", unrelated, 0o444),
        ]
        if tle is not None:
            rows.extend(
                [
                    ("usr/local/bin/tle", tle, 0o555),
                    (
                        "opt/artifacts/tle-build/tle-build-receipt.json",
                        _tle_receipt(tle),
                        0o444,
                    ),
                ]
            )
        if special in {"dot-prefix-symlink", "dot-prefix-hardlink"}:
            rows = [row for row in rows if row[0] != "usr/local/bin/opa"]
        for name, payload, mode in rows:
            member = tarfile.TarInfo(f"./{name}" if dot_prefix else name)
            member.size = len(payload)
            member.mode = 0o644 if name == writable_runtime_target else mode
            archive.addfile(member, io.BytesIO(payload))
        if special == "duplicate":
            payload = b"\x7fELF-duplicate"
            member = tarfile.TarInfo("usr/local/bin/opa")
            member.size = len(payload)
            member.mode = 0o555
            archive.addfile(member, io.BytesIO(payload))
        elif special in {"symlink", "hardlink", "dot-prefix-symlink", "dot-prefix-hardlink"}:
            dot_prefixed = special.startswith("dot-prefix-")
            member = tarfile.TarInfo("./usr/local/bin/opa" if dot_prefixed else "usr/local/bin/opa")
            member.type = tarfile.SYMTYPE if special.endswith("symlink") else tarfile.LNKTYPE
            member.linkname = "elsewhere"
            archive.addfile(member)
        elif special == "device":
            member = tarfile.TarInfo("dev/host-device")
            member.type = tarfile.CHRTYPE
            archive.addfile(member)
        elif special == "traversal":
            payload = b"escape"
            member = tarfile.TarInfo("../escape")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        elif special == "dot-prefix-alias":
            payload = b"\x7fELF-duplicate"
            member = tarfile.TarInfo("./usr/local/bin/opa")
            member.size = len(payload)
            member.mode = 0o555
            archive.addfile(member, io.BytesIO(payload))
        elif special == "dot-prefix-traversal":
            payload = b"escape"
            member = tarfile.TarInfo("./../escape")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        elif special == "repeated-dot-prefix":
            payload = b"escape"
            member = tarfile.TarInfo("././escape")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        elif special in {
            "absolute",
            "backslash",
            "dot-prefix-backslash",
            "collapsed-dot-prefix",
            "root-only-dot-prefix",
            "dot-prefix-root-alias",
        }:
            names = {
                "absolute": "/escape",
                "backslash": r"foo\bar",
                "dot-prefix-backslash": r"./foo\bar",
                "collapsed-dot-prefix": ".//escape",
                "root-only-dot-prefix": "./",
                "dot-prefix-root-alias": "./.",
            }
            payload = b"escape"
            member = tarfile.TarInfo(names[special])
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _replacement_layer(*, dot_prefix: bool = False) -> bytes:
    wheel, _extension = _wheel()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name in (
            "usr/local/bin/.wh.opa",
            "opt/artifacts/hnswlib/.wh..wh..opq",
        ):
            whiteout = tarfile.TarInfo(f"./{name}" if dot_prefix else name)
            whiteout.size = 0
            whiteout.mode = 0o000
            archive.addfile(whiteout, io.BytesIO())
        rows = [
            ("usr/local/bin/opa", _elf(b"synthetic-arm64-opa"), 0o555),
            (f"opt/artifacts/hnswlib/{WHEEL_NAME}", wheel, 0o444),
        ]
        for name, payload, mode in rows:
            member = tarfile.TarInfo(f"./{name}" if dot_prefix else name)
            member.size = len(payload)
            member.mode = mode
            archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _archive(
    path: Path,
    *,
    attestation_nonce: str,
    opa: bytes | None = None,
    opa_policy: bytes = OPA_POLICY,
    uv_lock: bytes = UV_LOCK,
    config_nonce: str | None = None,
    runtime_user: str = "65532:65532",
    extra_environment: tuple[str, str] | None = None,
    extra_label: tuple[str, str] | None = None,
    extra_runtime_config: tuple[str, object] | None = None,
    omit_runtime_config_field: str | None = None,
    unrelated: bytes = b"first",
    wrong_diff_id: bool = False,
    noncanonical_receipt: bool = False,
    crossed_receipt: bool = False,
    layer_special: str | None = None,
    unsupported_compression: bool = False,
    extra_platform: bool = False,
    attestation_count: int = 1,
    wrong_descriptor_size: bool = False,
    outer_traversal: bool = False,
    outer_dot_prefix: bool = False,
    malformed_index: bool = False,
    two_layers: bool = False,
    nested_index: bool = False,
    image_role: str = "scientific",
    tle: bytes | None = None,
    layer_dot_prefix: bool = False,
    writable_runtime_target: str | None = None,
) -> None:
    if image_role == "timelock-release" and tle is None:
        tle = _static_tle_elf()
    layer_tar = _layer(
        opa=opa,
        opa_policy=opa_policy,
        uv_lock=uv_lock,
        noncanonical_receipt=noncanonical_receipt,
        crossed_receipt=crossed_receipt,
        special=layer_special,
        unrelated=unrelated,
        tle=tle,
        dot_prefix=layer_dot_prefix,
        writable_runtime_target=writable_runtime_target,
    )
    layer_tars = [layer_tar]
    if two_layers:
        layer_tars.append(_replacement_layer(dot_prefix=layer_dot_prefix))
    layer_blobs = [gzip.compress(payload, mtime=0) for payload in layer_tars]
    layer_media_type = (
        "application/vnd.oci.image.layer.v1.tar+zstd" if unsupported_compression else OCI_LAYER_GZIP
    )
    layer_descriptors = [_descriptor(payload, layer_media_type) for payload in layer_blobs]
    diff_ids = ["sha256:" + hashlib.sha256(payload).hexdigest() for payload in layer_tars]
    if wrong_diff_id:
        diff_ids[0] = "sha256:" + "0" * 64
    labels = {
        "io.fractal-ann.confirmatory.build-context-tree-sha256": (BUILD_CONTEXT_TREE_SHA256),
        "io.fractal-ann.confirmatory.debian-inrelease-sha256": (
            "77737fa4b34f2693e982cc9ee35736816c35a7778fc2d326cc1bbf5b301fe1aa"
        ),
        "io.fractal-ann.confirmatory.debian-keyring-sha256": (
            "506b815cbb32d9b6066b4a2aa524071e071761e7e7f68c3ac74f3061ba852017"
        ),
        "io.fractal-ann.confirmatory.debian-snapshot": "20260714T000000Z",
        "io.fractal-ann.confirmatory.go-builder-image": (
            "docker.io/library/golang:1.26.5-bookworm@"
            "sha256:1ecb7edf62a0408027bd5729dfd6b1b8766e578e8df93995b225dfd0944eb651"
        ),
        "io.fractal-ann.confirmatory.opa-commit": ("e695c9ef8edb0f8b9f13d014d7bc8a7fbcc57297"),
        "io.fractal-ann.confirmatory.opa-dependency-delta-sha256": (
            "400699e81344ff2114fc5d2254734cb84a7015a68840505a7ab6a05df0dd39e0"
        ),
        "io.fractal-ann.confirmatory.opa-rego-sha256": OPA_POLICY_SHA256,
        "io.fractal-ann.confirmatory.opa-rego-test-sha256": (
            "67370adfcba1c5180bdc99ae2cab900785ec5cee6fd91a9a4a9058415a7d4f00"
        ),
        "io.fractal-ann.confirmatory.opa-source-sha256": (
            "a8b3ecdc925b75bdade52d315aa13efaa51c2de99acb78003ad353cce6e9e637"
        ),
        "io.fractal-ann.confirmatory.oras-version": "2.6.2",
        "io.fractal-ann.confirmatory.python-builder-image": (
            "python:3.12.13-slim-bookworm@"
            "sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b"
        ),
        "io.fractal-ann.confirmatory.runtime-role": "scientific",
        "io.fractal-ann.confirmatory.source-date-epoch": str(SOURCE_EPOCH),
        "io.fractal-ann.confirmatory.sqlite-sha256": (
            "c917d7db16648ec95f714974ace5e5dcf46b7dc70e26600a0a102a3141125db0"
        ),
        "io.fractal-ann.confirmatory.sqlite-sha3-256": (
            "98f2b3f3c11be6a03ea32346937b032c2472ebbd7a716bed36ca2f5693e7ce8b"
        ),
        "io.fractal-ann.confirmatory.tle-present": "false",
        "io.fractal-ann.confirmatory.uv-lock-sha256": UV_LOCK_SHA256,
        "io.fractal-ann.confirmatory.zlib-sha256": (
            "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16"
        ),
        "org.opencontainers.image.authors": "mhdk1602 <mhdk1602@users.noreply.github.com>",
        "org.opencontainers.image.base.name": (
            "gcr.io/distroless/base-nossl-debian12:nonroot@"
            "sha256:26cd77482910e221ff26cf7c480203ce97f8f01ad272e2dc8a9ae29c811e9efe"
        ),
        "org.opencontainers.image.description": (
            "C0-pinned execution image for the registered Fractal ANN confirmatory apparatus"
        ),
        "org.opencontainers.image.documentation": (
            "https://github.com/mhdk1602/fractal-ann-diagnostics/blob/master/"
            "research/runner-image.md"
        ),
        "org.opencontainers.image.licenses": "MIT",
        "org.opencontainers.image.revision": BUILD_CONTEXT_TREE_SHA256,
        "org.opencontainers.image.source": ("https://github.com/mhdk1602/fractal-ann-diagnostics"),
        "org.opencontainers.image.title": "Fractal ANN confirmatory runner",
        "org.opencontainers.image.url": ("https://github.com/mhdk1602/fractal-ann-diagnostics"),
        "org.opencontainers.image.vendor": "mhdk1602",
        "org.opencontainers.image.version": BUILD_CONTEXT_TREE_SHA256,
    }
    if image_role == "timelock-release":
        labels.update(
            {
                "io.fractal-ann.confirmatory.runtime-role": "timelock-release",
                "io.fractal-ann.confirmatory.tle-binary-sha256": (
                    hashlib.sha256(tle or b"").hexdigest()
                ),
                "io.fractal-ann.confirmatory.tle-dependency-delta-sha256": (
                    reproducibility._TLE_DEPENDENCY_DELTA_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-go-tarball-sha256": (
                    reproducibility._TLE_GO_TARBALL_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-go-tool-sha256": (
                    reproducibility._TLE_GO_TOOL_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-patched-go-mod-sha256": (
                    reproducibility._TLE_PATCHED_GO_MOD_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-patched-go-sum-sha256": (
                    reproducibility._TLE_PATCHED_GO_SUM_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-present": "true",
                "io.fractal-ann.confirmatory.tle-release-version": "1.2.0",
                "io.fractal-ann.confirmatory.tle-runtime-scope": "linux/arm64-only",
                "io.fractal-ann.confirmatory.tle-source-archive-sha256": (
                    reproducibility._TLE_SOURCE_ARCHIVE_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-source-commit": (
                    reproducibility._TLE_SOURCE_COMMIT
                ),
                "io.fractal-ann.confirmatory.tle-source-tree-manifest-sha256": (
                    reproducibility._TLE_SOURCE_TREE_MANIFEST_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-tag-object": (reproducibility._TLE_TAG_OBJECT),
                "org.opencontainers.image.description": (
                    "C0-pinned ARM64 timelock release image for the registered "
                    "Fractal ANN confirmatory apparatus"
                ),
                "org.opencontainers.image.title": ("Fractal ANN confirmatory release runner"),
            }
        )
    if extra_label is not None:
        labels[extra_label[0]] = extra_label[1]
    environment = [
        "HOME=/home/runner",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        "LD_LIBRARY_PATH=/opt/native-libs:/usr/local/lib",
        "LOGNAME=runner",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "PATH=/opt/venv/bin:/usr/local/bin",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        "PYTHONPATH=/opt/app/src",
        "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt",
        "TMPDIR=/tmp",
        "TZ=UTC",
        "USER=runner",
        "VECLIB_MAXIMUM_THREADS=1",
        "XDG_CACHE_HOME=/tmp/fractal-cache",
    ]
    if config_nonce is not None:
        environment = [
            f"LD_LIBRARY_PATH={config_nonce}" if row.startswith("LD_LIBRARY_PATH=") else row
            for row in environment
        ]
    if extra_environment is not None:
        environment.append(f"{extra_environment[0]}={extra_environment[1]}")
    runtime_config: dict[str, object] = {
        "ArgsEscaped": True,
        "Cmd": ["--help"],
        "Entrypoint": [
            "/opt/venv/bin/python",
            "-m",
            "fractal_ann_diagnostics.cli",
        ],
        "Env": environment,
        "Labels": labels,
        "User": runtime_user,
        "WorkingDir": "/workspace",
    }
    if extra_runtime_config is not None:
        runtime_config[extra_runtime_config[0]] = extra_runtime_config[1]
    if omit_runtime_config_field is not None:
        runtime_config.pop(omit_runtime_config_field)
    config = _canonical(
        {
            "architecture": "arm64",
            "config": runtime_config,
            "created": "2023-11-14T22:13:20Z",
            "os": "linux",
            "rootfs": {"diff_ids": diff_ids, "type": "layers"},
        }
    )
    config_descriptor = _descriptor(config, OCI_CONFIG)
    executable_manifest = _canonical(
        {
            "config": config_descriptor,
            "layers": layer_descriptors,
            "mediaType": OCI_MANIFEST,
            "schemaVersion": 2,
        }
    )
    executable_descriptor = _descriptor(executable_manifest, OCI_MANIFEST)
    executable_index_descriptor = {
        **executable_descriptor,
        "platform": {"architecture": "arm64", "os": "linux"},
    }
    if wrong_descriptor_size:
        executable_index_descriptor["size"] = len(executable_manifest) + 1

    attestation_config = _canonical({})
    attestation_config_descriptor = _descriptor(
        attestation_config, "application/vnd.oci.empty.v1+json"
    )
    attestation_payload = _canonical({"nonce": attestation_nonce})
    attestation_layer_descriptor = _descriptor(attestation_payload, "application/vnd.in-toto+json")
    attestation_manifest = _canonical(
        {
            "config": attestation_config_descriptor,
            "layers": [attestation_layer_descriptor],
            "mediaType": OCI_MANIFEST,
            "schemaVersion": 2,
        }
    )
    attestation_descriptor = {
        **_descriptor(attestation_manifest, OCI_MANIFEST),
        "annotations": {
            "vnd.docker.reference.digest": executable_descriptor["digest"],
            "vnd.docker.reference.type": "attestation-manifest",
        },
        "platform": {"architecture": "unknown", "os": "unknown"},
    }
    manifests: list[dict[str, object]] = [
        executable_index_descriptor,
        *([attestation_descriptor] * attestation_count),
    ]
    if extra_platform:
        manifests.append(
            {
                **executable_descriptor,
                "platform": {"architecture": "amd64", "os": "linux"},
            }
        )
    execution_index = _canonical(
        {
            "manifests": manifests,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "schemaVersion": 2,
        }
    )
    index = execution_index
    if nested_index:
        nested_descriptor = _descriptor(execution_index, "application/vnd.oci.image.index.v1+json")
        nested_descriptor["annotations"] = {
            "io.containerd.image.name": "example.invalid/fractal-ann:c0",
            "org.opencontainers.image.ref.name": "c0",
        }
        index = _canonical(
            {
                "manifests": [nested_descriptor],
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "schemaVersion": 2,
            }
        )
    if malformed_index:
        index = b'{"schemaVersion":2,"schemaVersion":2,"manifests":[]}'
    blobs = {
        **{hashlib.sha256(payload).hexdigest(): payload for payload in layer_blobs},
        hashlib.sha256(config).hexdigest(): config,
        hashlib.sha256(executable_manifest).hexdigest(): executable_manifest,
        hashlib.sha256(attestation_config).hexdigest(): attestation_config,
        hashlib.sha256(attestation_payload).hexdigest(): attestation_payload,
        hashlib.sha256(attestation_manifest).hexdigest(): attestation_manifest,
    }
    if nested_index:
        blobs[hashlib.sha256(execution_index).hexdigest()] = execution_index
    with tarfile.open(path, mode="w") as archive:
        files = {
            "oci-layout": _canonical({"imageLayoutVersion": "1.0.0"}),
            "index.json": index,
            **{f"blobs/sha256/{digest}": payload for digest, payload in blobs.items()},
        }
        if outer_traversal:
            files["../escape"] = b"escape"
        if outer_dot_prefix:
            files["./index.json"] = files.pop("index.json")
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o444
            archive.addfile(member, io.BytesIO(payload))


def _compare(tmp_path: Path, **second_options: object):
    first = tmp_path / "first.oci.tar"
    second = tmp_path / "second.oci.tar"
    output = tmp_path / "comparison.json"
    _archive(first, attestation_nonce="first")
    _archive(second, attestation_nonce="second", **second_options)
    receipt = compare_c0_oci_archives(
        archive_a_path=first,
        archive_b_path=second,
        expected_build_context_tree_sha256=BUILD_CONTEXT_TREE_SHA256,
        expected_source_date_epoch=SOURCE_EPOCH,
        expected_uv_lock_sha256=UV_LOCK_SHA256,
        expected_opa_policy_sha256=OPA_POLICY_SHA256,
        output_path=output,
    )
    return receipt, output


def test_attestation_and_outer_index_variation_are_recorded_but_accepted(tmp_path: Path) -> None:
    receipt, output = _compare(tmp_path)

    assert receipt.executable_equal is True
    assert receipt.schema_version == "fractal-c0-executable-reproducibility-v3"
    assert receipt.outer_index_equal is False
    assert receipt.attestation_metadata_equal is False
    assert receipt.archive_a.archive_sha256 != receipt.archive_b.archive_sha256
    assert receipt.archive_a.executable == receipt.archive_b.executable
    assert output.read_bytes() == receipt.canonical_file_bytes()
    assert output.read_bytes().endswith(b"\n")
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    parsed = json.loads(output.read_text())
    assert set(parsed) == {
        "archive_a",
        "archive_b",
        "attestation_metadata_equal",
        "executable_equal",
        "expected_build_context_tree_sha256",
        "expected_opa_policy_sha256",
        "expected_source_date_epoch",
        "expected_uv_lock_sha256",
        "outer_index_equal",
        "schema_version",
    }
    files = parsed["archive_a"]["executable_projection"]["runtime_files"]
    assert {item["image_path"] for item in files} == {
        "/opt/app/uv.lock",
        "/opt/app/policy/opa_compiled_masks.rego",
        "/opt/artifacts/hnswlib-runtime-receipt.json",
        "/opt/artifacts/native-build/compiler-closure.tsv",
        "/opt/artifacts/native-build/native-build-receipt.json",
        "/opt/artifacts/opa-build/opa-build-receipt.json",
        "/opt/artifacts/opa-build/opa-go-build-info.txt",
        "/opt/artifacts/runtime-library-manifest.json",
        f"/opt/artifacts/hnswlib/{WHEEL_NAME}",
        "/opt/native-libs/libsqlite3.so.0",
        "/opt/native-libs/libz.so.1",
        "/opt/venv/bin/python",
        "/usr/local/bin/opa",
        "/var/lib/dpkg/status",
    }


@pytest.mark.parametrize(
    "runtime_target",
    (
        "opt/artifacts/runtime-library-manifest.json",
        "var/lib/dpkg/status",
    ),
)
def test_generated_runtime_control_must_be_read_only(
    tmp_path: Path,
    runtime_target: str,
) -> None:
    with pytest.raises(
        C0ReproducibilityError,
        match=re.escape(f"runtime target {runtime_target!r} remains writable"),
    ):
        _compare(tmp_path, writable_runtime_target=runtime_target)


def test_whiteout_and_opaque_directory_replacement_preserve_final_targets(tmp_path: Path) -> None:
    first = tmp_path / "first.oci.tar"
    second = tmp_path / "second.oci.tar"
    output = tmp_path / "comparison.json"
    _archive(first, attestation_nonce="first", two_layers=True)
    _archive(second, attestation_nonce="second", two_layers=True)

    receipt = compare_c0_oci_archives(
        archive_a_path=first,
        archive_b_path=second,
        expected_build_context_tree_sha256=BUILD_CONTEXT_TREE_SHA256,
        expected_source_date_epoch=SOURCE_EPOCH,
        expected_uv_lock_sha256=UV_LOCK_SHA256,
        expected_opa_policy_sha256=OPA_POLICY_SHA256,
        output_path=output,
    )

    assert receipt.executable_equal is True
    assert len(receipt.archive_a.executable.ordered_layers) == 2


def test_distroless_single_dot_prefix_layer_members_are_normalized(tmp_path: Path) -> None:
    first = tmp_path / "first.oci.tar"
    second = tmp_path / "second.oci.tar"
    output = tmp_path / "comparison.json"
    _archive(first, attestation_nonce="first", layer_dot_prefix=True, two_layers=True)
    _archive(second, attestation_nonce="second", layer_dot_prefix=True, two_layers=True)

    receipt = compare_c0_oci_archives(
        archive_a_path=first,
        archive_b_path=second,
        expected_build_context_tree_sha256=BUILD_CONTEXT_TREE_SHA256,
        expected_source_date_epoch=SOURCE_EPOCH,
        expected_uv_lock_sha256=UV_LOCK_SHA256,
        expected_opa_policy_sha256=OPA_POLICY_SHA256,
        output_path=output,
    )

    assert receipt.executable_equal is True
    assert all(
        not runtime_file.image_path.startswith("/./")
        for runtime_file in receipt.archive_a.executable.runtime_files
    )


@pytest.mark.parametrize(
    ("special", "message"),
    [
        ("dot-prefix-alias", r"repeats member 'usr/local/bin/opa'"),
        ("dot-prefix-traversal", "noncanonical or traversal component"),
        ("repeated-dot-prefix", "noncanonical or traversal component"),
        ("absolute", "canonical relative POSIX path"),
        ("backslash", "canonical relative POSIX path"),
        ("dot-prefix-backslash", "canonical relative POSIX path"),
        ("collapsed-dot-prefix", "canonical relative POSIX path"),
        ("root-only-dot-prefix", "canonical relative POSIX path"),
        ("dot-prefix-root-alias", "noncanonical or traversal component"),
        ("dot-prefix-symlink", "crosses link member 'usr/local/bin/opa'"),
        ("dot-prefix-hardlink", "crosses link member 'usr/local/bin/opa'"),
        ("wheel-dot-prefix", "wheel member"),
    ],
)
def test_layer_dot_prefix_normalization_remains_fail_closed(
    tmp_path: Path,
    special: str,
    message: str,
) -> None:
    with pytest.raises(C0ReproducibilityError, match=message):
        _compare(tmp_path, layer_special=special)
    assert not (tmp_path / "comparison.json").exists()


def test_single_dot_prefix_remains_invalid_for_outer_oci_paths(tmp_path: Path) -> None:
    with pytest.raises(C0ReproducibilityError, match="OCI tar member path"):
        _compare(tmp_path, outer_dot_prefix=True)
    assert not (tmp_path / "comparison.json").exists()


def test_buildx_style_nested_execution_index_is_traversed_and_recorded(tmp_path: Path) -> None:
    first = tmp_path / "first.oci.tar"
    second = tmp_path / "second.oci.tar"
    output = tmp_path / "comparison.json"
    _archive(first, attestation_nonce="first", nested_index=True)
    _archive(second, attestation_nonce="second", nested_index=True)

    receipt = compare_c0_oci_archives(
        archive_a_path=first,
        archive_b_path=second,
        expected_build_context_tree_sha256=BUILD_CONTEXT_TREE_SHA256,
        expected_source_date_epoch=SOURCE_EPOCH,
        expected_uv_lock_sha256=UV_LOCK_SHA256,
        expected_opa_policy_sha256=OPA_POLICY_SHA256,
        output_path=output,
    )

    assert receipt.executable_equal is True
    assert receipt.archive_a.nested_index_descriptor is not None
    assert receipt.archive_b.nested_index_descriptor is not None
    assert (
        receipt.archive_a.nested_index_descriptor.digest
        != receipt.archive_b.nested_index_descriptor.digest
    )


@pytest.mark.parametrize(
    ("second_options", "message"),
    [
        ({"config_nonce": "b" * 64}, "image environment LD_LIBRARY_PATH differs"),
        ({"unrelated": b"different-layer-bytes"}, "executable projections differ"),
        ({"opa": _elf(b"different-runtime")}, "executable projections differ"),
    ],
)
def test_any_executable_config_layer_or_runtime_drift_fails_without_output(
    tmp_path: Path,
    second_options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(C0ReproducibilityError, match=message):
        _compare(tmp_path, **second_options)
    assert not (tmp_path / "comparison.json").exists()


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("wrong_diff_id", "diff_ids"),
        ("noncanonical_receipt", "not canonical JSON"),
        ("unsupported_compression", "unsupported compression"),
        ("extra_platform", "extra executable"),
        ("wrong_descriptor_size", "descriptor size"),
        ("outer_traversal", "traversal"),
        ("malformed_index", "duplicate key"),
    ],
)
def test_malformed_oci_graphs_are_rejected_before_output(
    tmp_path: Path, option: str, message: str
) -> None:
    with pytest.raises(C0ReproducibilityError, match=message):
        _compare(tmp_path, **{option: True})
    assert not (tmp_path / "comparison.json").exists()


@pytest.mark.parametrize("special", ["duplicate", "symlink", "hardlink", "device", "traversal"])
def test_ambiguous_or_unsafe_layer_target_states_are_rejected(tmp_path: Path, special: str) -> None:
    with pytest.raises(C0ReproducibilityError):
        _compare(tmp_path, layer_special=special)
    assert not (tmp_path / "comparison.json").exists()


def test_crossed_hnsw_receipt_and_wheel_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(C0ReproducibilityError, match="wheel digests disagree"):
        _compare(tmp_path, crossed_receipt=True)
    assert not (tmp_path / "comparison.json").exists()


@pytest.mark.parametrize(
    ("archive_options", "message"),
    [
        ({"runtime_user": "0:0"}, "runtime user"),
        ({"extra_environment": ("PYTHONINSPECT", "1")}, "closed C0 set"),
        ({"extra_label": ("io.attacker.injected", "true")}, "closed C0 label set"),
        (
            {"extra_runtime_config": ("StopSignal", "SIGTERM")},
            r"unexpected=\['StopSignal'\]",
        ),
        ({"extra_runtime_config": ("ArgsEscaped", False)}, "ArgsEscaped"),
        (
            {"omit_runtime_config_field": "ArgsEscaped"},
            r"missing=\['ArgsEscaped'\]",
        ),
        ({"opa": _elf(b"wrong-machine", machine=62)}, "AArch64"),
        ({"opa_policy": b"package substituted\n"}, "final OPA policy"),
        ({"uv_lock": b"version = 2\n"}, "final uv.lock"),
        ({"attestation_count": 0}, "exactly one arm64 attestation"),
        ({"attestation_count": 2}, "exactly one arm64 attestation"),
    ],
)
def test_two_matching_but_contract_wrong_archives_are_rejected(
    tmp_path: Path,
    archive_options: dict[str, object],
    message: str,
) -> None:
    first = tmp_path / "first.oci.tar"
    second = tmp_path / "second.oci.tar"
    output = tmp_path / "comparison.json"
    _archive(first, attestation_nonce="same", **archive_options)
    _archive(second, attestation_nonce="same", **archive_options)

    with pytest.raises(C0ReproducibilityError, match=message):
        compare_c0_oci_archives(
            archive_a_path=first,
            archive_b_path=second,
            expected_build_context_tree_sha256=BUILD_CONTEXT_TREE_SHA256,
            expected_source_date_epoch=SOURCE_EPOCH,
            expected_uv_lock_sha256=UV_LOCK_SHA256,
            expected_opa_policy_sha256=OPA_POLICY_SHA256,
            output_path=output,
        )
    assert not output.exists()


def test_cli_requires_distinct_archives_and_exclusive_output(tmp_path: Path, capsys) -> None:
    archive = tmp_path / "one.oci.tar"
    _archive(archive, attestation_nonce="only")
    output = tmp_path / "comparison.json"

    result = main(
        [
            "--archive-a",
            str(archive),
            "--archive-b",
            str(archive),
            "--expected-build-context-tree-sha256",
            BUILD_CONTEXT_TREE_SHA256,
            "--expected-source-date-epoch",
            str(SOURCE_EPOCH),
            "--expected-uv-lock-sha256",
            UV_LOCK_SHA256,
            "--expected-opa-policy-sha256",
            OPA_POLICY_SHA256,
            "--output",
            str(output),
        ]
    )

    assert result == 2
    assert "must be distinct files" in capsys.readouterr().err
    assert not output.exists()


def test_existing_output_is_never_replaced(tmp_path: Path) -> None:
    first = tmp_path / "first.oci.tar"
    second = tmp_path / "second.oci.tar"
    output = tmp_path / "comparison.json"
    _archive(first, attestation_nonce="first")
    _archive(second, attestation_nonce="second")
    output.write_bytes(b"custodied")

    with pytest.raises(C0ReproducibilityError, match="already exists"):
        compare_c0_oci_archives(
            archive_a_path=first,
            archive_b_path=second,
            expected_build_context_tree_sha256=BUILD_CONTEXT_TREE_SHA256,
            expected_source_date_epoch=SOURCE_EPOCH,
            expected_uv_lock_sha256=UV_LOCK_SHA256,
            expected_opa_policy_sha256=OPA_POLICY_SHA256,
            output_path=output,
        )

    assert output.read_bytes() == b"custodied"


def test_two_release_archives_require_equal_manifest_config_and_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tle = _static_tle_elf()
    monkeypatch.setattr(
        reproducibility,
        "_TLE_BINARY_SHA256",
        hashlib.sha256(tle).hexdigest(),
    )
    monkeypatch.setattr(reproducibility, "_TLE_BINARY_BYTE_COUNT", len(tle))
    first = tmp_path / "release-a.oci.tar"
    second = tmp_path / "release-b.oci.tar"
    output = tmp_path / "release-comparison.json"
    _archive(
        first,
        attestation_nonce="release-a",
        image_role="timelock-release",
        tle=tle,
    )
    _archive(
        second,
        attestation_nonce="release-b",
        image_role="timelock-release",
        tle=tle,
    )

    receipt = compare_tle_release_oci_archives(
        archive_a_path=first,
        archive_b_path=second,
        expected_build_context_tree_sha256=BUILD_CONTEXT_TREE_SHA256,
        expected_source_date_epoch=SOURCE_EPOCH,
        expected_uv_lock_sha256=UV_LOCK_SHA256,
        expected_opa_policy_sha256=OPA_POLICY_SHA256,
        output_path=output,
    )

    assert receipt.image_closure_equal is True
    assert receipt.tle_binary_sha256 == hashlib.sha256(tle).hexdigest()
    assert receipt.tle_binary_byte_count == len(tle)
    assert (
        receipt.archive_a.executable.manifest.digest == receipt.archive_b.executable.manifest.digest
    )
    assert output.read_bytes() == receipt.canonical_file_bytes()


def test_release_layer_drift_closes_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tle = _static_tle_elf()
    monkeypatch.setattr(
        reproducibility,
        "_TLE_BINARY_SHA256",
        hashlib.sha256(tle).hexdigest(),
    )
    monkeypatch.setattr(reproducibility, "_TLE_BINARY_BYTE_COUNT", len(tle))
    first = tmp_path / "release-a.oci.tar"
    second = tmp_path / "release-b.oci.tar"
    output = tmp_path / "release-comparison.json"
    _archive(
        first,
        attestation_nonce="release-a",
        image_role="timelock-release",
        tle=tle,
    )
    _archive(
        second,
        attestation_nonce="release-b",
        image_role="timelock-release",
        tle=tle,
        unrelated=b"changed-release-layer",
    )

    with pytest.raises(C0ReproducibilityError, match="manifest, config, or layer"):
        compare_tle_release_oci_archives(
            archive_a_path=first,
            archive_b_path=second,
            expected_build_context_tree_sha256=BUILD_CONTEXT_TREE_SHA256,
            expected_source_date_epoch=SOURCE_EPOCH,
            expected_uv_lock_sha256=UV_LOCK_SHA256,
            expected_opa_policy_sha256=OPA_POLICY_SHA256,
            output_path=output,
        )
    assert not output.exists()
