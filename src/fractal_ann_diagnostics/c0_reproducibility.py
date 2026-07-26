"""Offline reproducibility gate for two retained C0 OCI image archives.

The gate never extracts either archive.  It validates the OCI descriptor graph,
replays the executable layers only far enough to recover the registered runtime
targets, and compares the complete executable projection.  BuildKit attestations
and the enclosing OCI index are retained as evidence but are outside executable
equality.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import errno
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import unicodedata
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

C0_REPRODUCIBILITY_SCHEMA = "fractal-c0-executable-reproducibility-v3"
TLE_RELEASE_REPRODUCIBILITY_SCHEMA = "fractal-tle-release-oci-reproducibility-v2"

_OCI_LAYOUT_MEDIA_TYPE = "application/vnd.oci.image.index.v1+json"
_DOCKER_INDEX_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.list.v2+json"
_OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
_DOCKER_MANIFEST_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
_OCI_CONFIG_MEDIA_TYPE = "application/vnd.oci.image.config.v1+json"
_DOCKER_CONFIG_MEDIA_TYPE = "application/vnd.docker.container.image.v1+json"
_OCI_LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.v1.tar"
_OCI_GZIP_LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.v1.tar+gzip"
_DOCKER_GZIP_LAYER_MEDIA_TYPE = "application/vnd.docker.image.rootfs.diff.tar.gzip"
_ATTESTATION_REFERENCE_TYPE = "attestation-manifest"

_OPA_PATH = "usr/local/bin/opa"
_OPA_POLICY_PATH = "opt/app/policy/opa_compiled_masks.rego"
_PYTHON_PATH = "opt/venv/bin/python"
_UV_LOCK_PATH = "opt/app/uv.lock"
_HNSW_RECEIPT_PATH = "opt/artifacts/hnswlib-runtime-receipt.json"
_HNSW_WHEEL_ROOT = "opt/artifacts/hnswlib"
_NATIVE_LIBRARY_MANIFEST_PATH = "opt/artifacts/runtime-library-manifest.json"
_NATIVE_BUILD_RECEIPT_PATH = "opt/artifacts/native-build/native-build-receipt.json"
_NATIVE_COMPILER_CLOSURE_PATH = "opt/artifacts/native-build/compiler-closure.tsv"
_SQLITE_LIBRARY_PATH = "opt/native-libs/libsqlite3.so.0"
_ZLIB_LIBRARY_PATH = "opt/native-libs/libz.so.1"
_OPA_BUILD_RECEIPT_PATH = "opt/artifacts/opa-build/opa-build-receipt.json"
_OPA_BUILD_INFO_PATH = "opt/artifacts/opa-build/opa-go-build-info.txt"
_TLE_PATH = "usr/local/bin/tle"
_TLE_BUILD_RECEIPT_PATH = "opt/artifacts/tle-build/tle-build-receipt.json"
_DPKG_STATUS_PATH = "var/lib/dpkg/status"
_FIXED_TARGETS = (
    _OPA_PATH,
    _OPA_POLICY_PATH,
    _PYTHON_PATH,
    _UV_LOCK_PATH,
    _HNSW_RECEIPT_PATH,
    _NATIVE_LIBRARY_MANIFEST_PATH,
    _NATIVE_BUILD_RECEIPT_PATH,
    _NATIVE_COMPILER_CLOSURE_PATH,
    _SQLITE_LIBRARY_PATH,
    _ZLIB_LIBRARY_PATH,
    _OPA_BUILD_RECEIPT_PATH,
    _OPA_BUILD_INFO_PATH,
    _DPKG_STATUS_PATH,
)
_FORBIDDEN_TARGETS = (_TLE_PATH, _TLE_BUILD_RECEIPT_PATH)
_TRACKED_TARGETS = _FIXED_TARGETS + _FORBIDDEN_TARGETS

_DISTROLESS_BASE_IMAGE = (
    "gcr.io/distroless/base-nossl-debian12:nonroot@"
    "sha256:26cd77482910e221ff26cf7c480203ce97f8f01ad272e2dc8a9ae29c811e9efe"
)
_PYTHON_BUILDER_IMAGE = (
    "python:3.12.13-slim-bookworm@"
    "sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b"
)
_GO_BUILDER_IMAGE = (
    "docker.io/library/golang:1.26.5-bookworm@"
    "sha256:1ecb7edf62a0408027bd5729dfd6b1b8766e578e8df93995b225dfd0944eb651"
)
_OPA_REGO_TEST_SHA256 = "67370adfcba1c5180bdc99ae2cab900785ec5cee6fd91a9a4a9058415a7d4f00"
_DEBIAN_SNAPSHOT = "20260714T000000Z"
_DEBIAN_INRELEASE_SHA256 = "77737fa4b34f2693e982cc9ee35736816c35a7778fc2d326cc1bbf5b301fe1aa"
_DEBIAN_KEYRING_SHA256 = "506b815cbb32d9b6066b4a2aa524071e071761e7e7f68c3ac74f3061ba852017"
_OPA_COMMIT = "e695c9ef8edb0f8b9f13d014d7bc8a7fbcc57297"
_OPA_DEPENDENCY_DELTA_SHA256 = "2b66370c2620bea30ed5ed776a807ea9ac83ca7aef9b2214a2f444cbcf7a7524"
_OPA_SOURCE_SHA256 = "a8b3ecdc925b75bdade52d315aa13efaa51c2de99acb78003ad353cce6e9e637"
_SQLITE_SHA256 = "c917d7db16648ec95f714974ace5e5dcf46b7dc70e26600a0a102a3141125db0"
_SQLITE_SHA3_256 = "98f2b3f3c11be6a03ea32346937b032c2472ebbd7a716bed36ca2f5693e7ce8b"
_ZLIB_SHA256 = "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16"
_EXPECTED_RUNTIME_ENVIRONMENT = {
    "HOME": "/home/runner",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LD_LIBRARY_PATH": "/opt/native-libs:/usr/local/lib",
    "LOGNAME": "runner",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/opt/venv/bin:/usr/local/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONPATH": "/opt/app/src",
    "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
    "TMPDIR": "/tmp",
    "TZ": "UTC",
    "USER": "runner",
    "VECLIB_MAXIMUM_THREADS": "1",
    "XDG_CACHE_HOME": "/tmp/fractal-cache",
}
_RUNTIME_CONFIG_FIELDS = frozenset(
    {"ArgsEscaped", "Cmd", "Entrypoint", "Env", "Labels", "User", "WorkingDir"}
)

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_WHEEL_NAME = re.compile(r"^hnswlib-0\.8\.0-[A-Za-z0-9_.+-]+\.whl$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RFC3339_UTC = re.compile(
    r"^(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})T"
    r"(?P<time>[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(?P<fraction>\.[0-9]+)?Z$"
)

_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024
_MAX_LAYOUT_JSON_BYTES = 4 * 1024
_MAX_INDEX_JSON_BYTES = 32 * 1024 * 1024
_MAX_MANIFEST_JSON_BYTES = 32 * 1024 * 1024
_MAX_CONFIG_JSON_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 128 * 1024
_MAX_RUNTIME_BYTES = 512 * 1024 * 1024
_MAX_WHEEL_BYTES = 512 * 1024 * 1024
_MAX_UNCOMPRESSED_LAYER_BYTES = 32 * 1024 * 1024 * 1024
_MAX_OUTER_MEMBERS = 200_000
_MAX_LAYER_MEMBERS = 2_000_000

_INDEX_FIELDS = frozenset({"annotations", "manifests", "mediaType", "schemaVersion"})
_DESCRIPTOR_FIELDS = frozenset({"annotations", "digest", "mediaType", "platform", "size"})
_INNER_DESCRIPTOR_FIELDS = frozenset({"annotations", "digest", "mediaType", "size"})
_PLATFORM_FIELDS = frozenset({"architecture", "os", "os.features", "os.version", "variant"})
_MANIFEST_FIELDS = frozenset(
    {"annotations", "artifactType", "config", "layers", "mediaType", "schemaVersion", "subject"}
)
_HNSW_RECEIPT_FIELDS = frozenset(
    {
        "extension_basename",
        "extension_byte_count",
        "extension_sha256",
        "package",
        "python_abi",
        "schema_version",
        "sdist_sha256",
        "version",
        "wheel_basename",
        "wheel_byte_count",
        "wheel_sha256",
    }
)
_NATIVE_BUILD_RECEIPT_FIELDS = frozenset(
    {
        "debian_inrelease_sha256",
        "debian_keyring_sha256",
        "debian_snapshot",
        "schema_version",
        "source_date_epoch",
        "sqlite_autoconf",
        "sqlite_library_sha256",
        "sqlite_sha256",
        "sqlite_sha3_256",
        "sqlite_source_id",
        "zlib_library_sha256",
        "zlib_sha256",
        "zlib_version",
    }
)
_OPA_BUILD_RECEIPT_FIELDS = frozenset(
    {
        "dependency_delta_sha256",
        "go_builder_image",
        "go_tarball_sha256",
        "go_version",
        "opa_commit",
        "opa_release_timestamp",
        "opa_sha256",
        "opa_source_sha256",
        "opa_version",
        "oras_original_version",
        "oras_patched_version",
        "original_go_mod_sha256",
        "original_go_sum_sha256",
        "patched_go_mod_sha256",
        "patched_go_sum_sha256",
        "schema_version",
        "source_date_epoch",
        "target_arch",
        "x_sync_original_version",
        "x_sync_patched_version",
    }
)
_TLE_BUILD_RECEIPT_FIELDS = frozenset(
    {
        "binary_byte_count",
        "binary_image_path",
        "binary_sha256",
        "build_commands",
        "build_environment",
        "builder_image",
        "dependency_delta_sha256",
        "elf",
        "go_tarball_sha256",
        "go_tarball_url",
        "go_tool_sha256",
        "go_version",
        "included",
        "independent_build_count",
        "independent_builds_byte_identical",
        "offline_test_inventory_sha256",
        "original_go_mod_sha256",
        "original_go_sum_sha256",
        "patched_go_mod_sha256",
        "patched_go_sum_sha256",
        "release_version",
        "schema_version",
        "source_archive_sha256",
        "source_archive_url",
        "source_commit_git_sha1",
        "source_date_epoch",
        "source_tree_manifest_sha256",
        "tag_object_git_sha1",
        "target_arch",
        "target_os",
    }
)

_TLE_BINARY_SHA256 = "ca9d498b6a3c1ea8edff9ace7bf00eb0f90ce67166343161f9a53f21900a6ef5"
_TLE_BINARY_BYTE_COUNT = 13_303_934
_TLE_SOURCE_COMMIT = "7b54141a9733fd6fa207587a11148280e6fb020d"
_TLE_TAG_OBJECT = "6a94bf6b8200ab67f2b80af8000a55db64998d94"
_TLE_SOURCE_ARCHIVE_SHA256 = "98b5edb760cffbe6edd392f004d2d51fcc7a8e6ef7ed7672c32b1a9e1ce3e32d"
_TLE_SOURCE_TREE_MANIFEST_SHA256 = (
    "6fedff45430fc81e9dbf5b13b1a2dc90e9840ae91f03de307dac5c2f7475c94c"
)
_TLE_DEPENDENCY_DELTA_SHA256 = "1b15bd1dd497c5553806ea5c58c170d6580ccc6139199d4fa9028e0ef8b79c59"
_TLE_GO_TARBALL_SHA256 = "fe4789e92b1f33358680864bbe8704289e7bb5fc207d80623c308935bd696d49"
_TLE_GO_TOOL_SHA256 = "22201b57b855105df064a291863c3fc04f22a7431187a9205122aff42a0c825b"
_TLE_ORIGINAL_GO_MOD_SHA256 = "0ee3447d4c3149e657a2f63c2e0046c19c21dcc63f730402cb24b08399db7741"
_TLE_ORIGINAL_GO_SUM_SHA256 = "1cb67cce42d7cf12be184f0f6a820c1f8c2f105615d43cf0a176ee35741c523b"
_TLE_PATCHED_GO_MOD_SHA256 = "ca99d5021580cc77d05367b7356b542fa3d77bc9f286aaa7d236b2a95a350c08"
_TLE_PATCHED_GO_SUM_SHA256 = "baa8d4e184c2d516317ecaf984c9d7aa5ac9f7fbd0209058538200b0292c71e0"


class C0ReproducibilityError(ValueError):
    """One archive is unsafe, invalid, or executable-different from its peer."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_canonical_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise C0ReproducibilityError(f"{label} must be a string")
    if unicodedata.normalize("NFC", value) != value:
        raise C0ReproducibilityError(f"{label} must use NFC Unicode normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise C0ReproducibilityError(f"{label} contains a control character")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise C0ReproducibilityError(f"{label} is not valid UTF-8") from exc
    return value


def _require_json_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise C0ReproducibilityError(f"{label} must be a string")
    if unicodedata.normalize("NFC", value) != value:
        raise C0ReproducibilityError(f"{label} must use NFC Unicode normalization")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise C0ReproducibilityError(f"{label} is not valid UTF-8") from exc
    return value


def _validate_json_value(value: object, *, label: str) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        raise C0ReproducibilityError(f"{label} contains a noncanonical floating-point number")
    if isinstance(value, str):
        # OCI image history legitimately stores escaped newlines and tabs in
        # ``created_by``. The JSON encoding remains strict; semantic control
        # characters are narrowed later for fields that become identifiers.
        _require_json_string(value, label=label)
        return
    if isinstance(value, list):
        for position, item in enumerate(value):
            _validate_json_value(item, label=f"{label}[{position}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_canonical_string(key, label=f"{label} key")
            _validate_json_value(item, label=f"{label}.{key}")
        return
    raise C0ReproducibilityError(f"{label} contains an unsupported JSON value")


def _strict_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    if not payload or payload[:1] != b"{" or payload[-1:] != b"}":
        raise C0ReproducibilityError(
            f"{label} must be one UTF-8 JSON object without surrounding whitespace"
        )

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise C0ReproducibilityError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise C0ReproducibilityError(f"{label} contains non-finite number {value!r}")

    try:
        text = payload.decode("utf-8", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise C0ReproducibilityError(f"{label} is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise C0ReproducibilityError(f"{label} is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise C0ReproducibilityError(f"{label} must contain one JSON object")
    _validate_json_value(parsed, label=label)
    return parsed


def _closed_mapping(
    value: object,
    *,
    fields: frozenset[str],
    required: frozenset[str] | None = None,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise C0ReproducibilityError(f"{label} must be an object with string keys")
    observed = set(value)
    unknown = observed - fields
    missing = (fields if required is None else required) - observed
    if unknown:
        raise C0ReproducibilityError(f"{label} has unknown fields: {sorted(unknown)}")
    if missing:
        raise C0ReproducibilityError(f"{label} is missing fields: {sorted(missing)}")
    return value


def _array(value: object, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise C0ReproducibilityError(f"{label} must be an array")
    return value


def _positive_int(value: object, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise C0ReproducibilityError(f"{label} must be a {qualifier} integer")
    return value


def _sha256_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise C0ReproducibilityError(f"{label} must be a lowercase sha256 OCI digest")
    return value


def _sha256_hex(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
        raise C0ReproducibilityError(f"{label} must be one lowercase SHA-256 hex value")
    return value


def _string_array(value: object, *, label: str) -> tuple[str, ...]:
    rows = _array(value, label=label)
    return tuple(
        _require_canonical_string(item, label=f"{label}[{position}]")
        for position, item in enumerate(rows)
    )


def _validate_aarch64_elf(payload: bytes, *, label: str) -> None:
    if (
        len(payload) < 20
        or payload[:4] != b"\x7fELF"
        or payload[4] != 2
        or payload[5] != 1
        or payload[6] != 1
    ):
        raise C0ReproducibilityError(f"{label} is not a 64-bit little-endian ELF executable")
    object_type = int.from_bytes(payload[16:18], byteorder="little")
    machine = int.from_bytes(payload[18:20], byteorder="little")
    if object_type not in {2, 3} or machine != 183:
        raise C0ReproducibilityError(f"{label} is not an AArch64 executable or shared object")


def _annotations(value: object, *, label: str) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise C0ReproducibilityError(f"{label} must be an object")
    result: list[tuple[str, str]] = []
    for key, item in value.items():
        canonical_key = _require_canonical_string(key, label=f"{label} key")
        canonical_value = _require_canonical_string(item, label=f"{label}[{key!r}]")
        if not canonical_key:
            raise C0ReproducibilityError(f"{label} cannot contain an empty key")
        result.append((canonical_key, canonical_value))
    return tuple(sorted(result, key=lambda row: row[0].encode("utf-8")))


@dataclass(frozen=True)
class PlatformProjection:
    architecture: str
    operating_system: str
    variant: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "architecture": self.architecture,
            "os": self.operating_system,
        }
        if self.variant is not None:
            result["variant"] = self.variant
        return result


@dataclass(frozen=True)
class DescriptorProjection:
    media_type: str
    digest: str
    size: int
    annotations: tuple[tuple[str, str], ...] = ()
    platform: PlatformProjection | None = None

    def content_only(self) -> DescriptorProjection:
        return DescriptorProjection(
            media_type=self.media_type,
            digest=self.digest,
            size=self.size,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "digest": self.digest,
            "media_type": self.media_type,
            "size": self.size,
        }
        if self.annotations:
            result["annotations"] = {key: value for key, value in self.annotations}
        if self.platform is not None:
            result["platform"] = self.platform.to_dict()
        return result


@dataclass(frozen=True)
class FileProjection:
    image_path: str
    byte_count: int
    sha256: str
    mode: str

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "image_path": self.image_path,
            "mode": self.mode,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class HnswExtensionProjection:
    basename: str
    byte_count: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "basename": self.basename,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ExecutableProjection:
    platform: PlatformProjection
    manifest: DescriptorProjection
    config: DescriptorProjection
    ordered_layers: tuple[DescriptorProjection, ...]
    rootfs_diff_ids: tuple[str, ...]
    config_created: str | None
    c0_labels: tuple[tuple[str, str], ...]
    c0_environment: tuple[tuple[str, str], ...]
    runtime_files: tuple[FileProjection, ...]
    hnsw_imported_extension: HnswExtensionProjection

    def to_dict(self) -> dict[str, object]:
        return {
            "c0_environment": {key: value for key, value in self.c0_environment},
            "c0_labels": {key: value for key, value in self.c0_labels},
            "config_created": self.config_created,
            "config_descriptor": self.config.to_dict(),
            "hnsw_imported_extension": self.hnsw_imported_extension.to_dict(),
            "manifest_descriptor": self.manifest.to_dict(),
            "ordered_layer_descriptors": [item.to_dict() for item in self.ordered_layers],
            "platform": self.platform.to_dict(),
            "rootfs_diff_ids": list(self.rootfs_diff_ids),
            "runtime_files": [item.to_dict() for item in self.runtime_files],
        }


@dataclass(frozen=True)
class AttestationProjection:
    index_descriptor: DescriptorProjection
    referenced_executable_manifest: str
    config_descriptor: DescriptorProjection
    layer_descriptors: tuple[DescriptorProjection, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config_descriptor": self.config_descriptor.to_dict(),
            "index_descriptor": self.index_descriptor.to_dict(),
            "layer_descriptors": [item.to_dict() for item in self.layer_descriptors],
            "referenced_executable_manifest": self.referenced_executable_manifest,
        }


@dataclass(frozen=True)
class ArchiveProjection:
    archive_byte_count: int
    archive_sha256: str
    outer_index_byte_count: int
    outer_index_sha256: str
    nested_index_descriptor: DescriptorProjection | None
    executable: ExecutableProjection
    attestations: tuple[AttestationProjection, ...]

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "archive_byte_count": self.archive_byte_count,
            "archive_sha256": self.archive_sha256,
            "attestations": [item.to_dict() for item in self.attestations],
            "executable_projection": self.executable.to_dict(),
            "outer_index_byte_count": self.outer_index_byte_count,
            "outer_index_sha256": self.outer_index_sha256,
        }
        if self.nested_index_descriptor is not None:
            result["nested_index_descriptor"] = self.nested_index_descriptor.to_dict()
        else:
            result["nested_index_descriptor"] = None
        return result


@dataclass(frozen=True)
class C0ReproducibilityReceipt:
    expected_build_context_tree_sha256: str
    expected_source_date_epoch: int
    expected_uv_lock_sha256: str
    expected_opa_policy_sha256: str
    archive_a: ArchiveProjection
    archive_b: ArchiveProjection
    executable_equal: bool
    outer_index_equal: bool
    attestation_metadata_equal: bool
    schema_version: str = C0_REPRODUCIBILITY_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_a": self.archive_a.to_dict(),
            "archive_b": self.archive_b.to_dict(),
            "attestation_metadata_equal": self.attestation_metadata_equal,
            "executable_equal": self.executable_equal,
            "expected_build_context_tree_sha256": self.expected_build_context_tree_sha256,
            "expected_opa_policy_sha256": self.expected_opa_policy_sha256,
            "expected_source_date_epoch": self.expected_source_date_epoch,
            "expected_uv_lock_sha256": self.expected_uv_lock_sha256,
            "outer_index_equal": self.outer_index_equal,
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_json(self.to_dict()) + b"\n"


@dataclass(frozen=True)
class TleReleaseReproducibilityReceipt:
    expected_build_context_tree_sha256: str
    expected_source_date_epoch: int
    expected_uv_lock_sha256: str
    expected_opa_policy_sha256: str
    archive_a: ArchiveProjection
    archive_b: ArchiveProjection
    image_closure_equal: bool
    tle_binary_sha256: str
    tle_binary_byte_count: int
    tle_build_receipt_sha256: str
    schema_version: str = TLE_RELEASE_REPRODUCIBILITY_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_a": self.archive_a.to_dict(),
            "archive_b": self.archive_b.to_dict(),
            "expected_build_context_tree_sha256": self.expected_build_context_tree_sha256,
            "expected_opa_policy_sha256": self.expected_opa_policy_sha256,
            "expected_source_date_epoch": self.expected_source_date_epoch,
            "expected_uv_lock_sha256": self.expected_uv_lock_sha256,
            "image_closure_equal": self.image_closure_equal,
            "image_role": "timelock-release",
            "schema_version": self.schema_version,
            "tle_binary_byte_count": self.tle_binary_byte_count,
            "tle_binary_sha256": self.tle_binary_sha256,
            "tle_build_receipt_sha256": self.tle_build_receipt_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_json(self.to_dict()) + b"\n"


def _parse_platform(value: object, *, label: str) -> PlatformProjection:
    row = _closed_mapping(
        value,
        fields=_PLATFORM_FIELDS,
        required=frozenset({"architecture", "os"}),
        label=label,
    )
    architecture = _require_canonical_string(row["architecture"], label=f"{label}.architecture")
    operating_system = _require_canonical_string(row["os"], label=f"{label}.os")
    variant_value = row.get("variant")
    variant = (
        None
        if variant_value is None
        else _require_canonical_string(variant_value, label=f"{label}.variant")
    )
    for optional in ("os.version",):
        if optional in row:
            _require_canonical_string(row[optional], label=f"{label}.{optional}")
    for optional in ("os.features",):
        if optional in row:
            features = _array(row[optional], label=f"{label}.{optional}")
            for position, feature in enumerate(features):
                _require_canonical_string(feature, label=f"{label}.{optional}[{position}]")
    return PlatformProjection(architecture, operating_system, variant)


def _parse_descriptor(
    value: object,
    *,
    label: str,
    index_descriptor: bool,
    platform_required: bool = True,
) -> DescriptorProjection:
    fields = _DESCRIPTOR_FIELDS if index_descriptor else _INNER_DESCRIPTOR_FIELDS
    required = (
        frozenset({"digest", "mediaType", "size", "platform"})
        if index_descriptor and platform_required
        else frozenset({"digest", "mediaType", "size"})
    )
    row = _closed_mapping(value, fields=fields, required=required, label=label)
    media_type = _require_canonical_string(row["mediaType"], label=f"{label}.mediaType")
    if not media_type:
        raise C0ReproducibilityError(f"{label}.mediaType cannot be empty")
    digest = _sha256_digest(row["digest"], label=f"{label}.digest")
    size = _positive_int(row["size"], label=f"{label}.size")
    annotations = _annotations(row.get("annotations"), label=f"{label}.annotations")
    platform = None
    if index_descriptor and "platform" in row:
        platform = _parse_platform(row["platform"], label=f"{label}.platform")
    return DescriptorProjection(media_type, digest, size, annotations, platform)


def _canonical_tar_member_path(value: str, *, label: str) -> str:
    canonical = _require_canonical_string(value, label=label)
    if not canonical or "\\" in canonical or canonical.startswith("/"):
        raise C0ReproducibilityError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(canonical)
    if str(path) != canonical or any(part in {"", ".", ".."} for part in path.parts):
        raise C0ReproducibilityError(f"{label} contains a noncanonical or traversal component")
    return canonical


@dataclass
class _ArchiveHandle:
    path: Path
    descriptor: int
    stat_result: os.stat_result
    sha256: str

    def close(self) -> None:
        os.close(self.descriptor)


def _hash_fd(descriptor: int, *, maximum: int) -> tuple[int, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    count = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        count += len(block)
        if count > maximum:
            raise C0ReproducibilityError("OCI archive exceeds the admitted byte limit")
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return count, digest.hexdigest()


def _open_archive(path: str | Path, *, label: str) -> _ArchiveHandle:
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise C0ReproducibilityError(f"{label} must be an absolute path")
    try:
        resolved = target.resolve(strict=True)
    except OSError as exc:
        raise C0ReproducibilityError(f"{label} does not exist") from exc
    if resolved != target or target.is_symlink():
        raise C0ReproducibilityError(f"{label} must not cross a symlink")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise C0ReproducibilityError(f"cannot open {label}: {exc.strerror or exc}") from exc
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise C0ReproducibilityError(f"{label} must be a regular file")
        if observed.st_nlink != 1:
            raise C0ReproducibilityError(f"{label} must be singly linked")
        if observed.st_size <= 0 or observed.st_size > _MAX_ARCHIVE_BYTES:
            raise C0ReproducibilityError(f"{label} has an invalid byte count")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            raise C0ReproducibilityError(
                f"{label} cannot be locked against cooperative writers"
            ) from exc
        count, digest = _hash_fd(descriptor, maximum=_MAX_ARCHIVE_BYTES)
        if count != observed.st_size:
            raise C0ReproducibilityError(f"{label} changed while it was hashed")
        return _ArchiveHandle(target, descriptor, observed, digest)
    except BaseException:
        os.close(descriptor)
        raise


def _verify_unchanged(handle: _ArchiveHandle, *, label: str) -> None:
    current = os.fstat(handle.descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(current, name) != getattr(handle.stat_result, name) for name in stable_fields):
        raise C0ReproducibilityError(f"{label} changed during OCI inspection")
    count, digest = _hash_fd(handle.descriptor, maximum=_MAX_ARCHIVE_BYTES)
    if count != current.st_size or digest != handle.sha256:
        raise C0ReproducibilityError(f"{label} bytes changed during OCI inspection")


class _OciArchive:
    def __init__(self, archive: tarfile.TarFile) -> None:
        self.archive = archive
        self.members: dict[str, tarfile.TarInfo] = {}
        self.verified_blobs: set[str] = set()
        self.referenced_blobs: set[str] = set()
        self._index_members()

    def _index_members(self) -> None:
        allowed_directories = {"blobs", "blobs/sha256"}
        for position, member in enumerate(self.archive):
            if position >= _MAX_OUTER_MEMBERS:
                raise C0ReproducibilityError("OCI archive contains too many tar members")
            name = _canonical_tar_member_path(member.name, label="OCI tar member path")
            if name in self.members:
                raise C0ReproducibilityError(f"OCI archive repeats tar member {name!r}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise C0ReproducibilityError(f"OCI archive contains non-regular member {name!r}")
            if member.isdir():
                if name not in allowed_directories:
                    raise C0ReproducibilityError(
                        f"OCI archive contains unexpected directory {name!r}"
                    )
            elif member.isreg():
                admitted = name in {"index.json", "oci-layout"} or re.fullmatch(
                    r"blobs/sha256/[0-9a-f]{64}", name
                )
                if not admitted:
                    raise C0ReproducibilityError(f"OCI archive contains unexpected file {name!r}")
            else:
                raise C0ReproducibilityError(f"OCI archive contains unsupported member {name!r}")
            if member.sparse:
                raise C0ReproducibilityError(f"OCI archive contains sparse member {name!r}")
            self.members[name] = member
        if "index.json" not in self.members or "oci-layout" not in self.members:
            raise C0ReproducibilityError("OCI archive lacks index.json or oci-layout")

    def read_file(self, name: str, *, maximum: int, label: str) -> bytes:
        try:
            member = self.members[name]
        except KeyError as exc:
            raise C0ReproducibilityError(f"{label} is absent from the OCI archive") from exc
        if not member.isreg() or member.size < 0 or member.size > maximum:
            raise C0ReproducibilityError(f"{label} has an invalid byte count")
        source = self.archive.extractfile(member)
        if source is None:
            raise C0ReproducibilityError(f"cannot read {label}")
        payload = source.read(maximum + 1)
        if len(payload) != member.size or len(payload) > maximum or source.read(1):
            raise C0ReproducibilityError(f"{label} byte count disagrees with its tar header")
        return payload

    def open_blob(self, digest: str, *, label: str) -> BinaryIO:
        name = f"blobs/sha256/{digest.removeprefix('sha256:')}"
        try:
            member = self.members[name]
        except KeyError as exc:
            raise C0ReproducibilityError(f"{label} blob {digest} is absent") from exc
        source = self.archive.extractfile(member)
        if source is None:
            raise C0ReproducibilityError(f"cannot read {label} blob {digest}")
        return source

    def verify_descriptor(self, descriptor: DescriptorProjection, *, label: str) -> None:
        self.referenced_blobs.add(descriptor.digest)
        if descriptor.digest in self.verified_blobs:
            name = f"blobs/sha256/{descriptor.digest.removeprefix('sha256:')}"
            if self.members[name].size != descriptor.size:
                raise C0ReproducibilityError(f"{label} descriptor size disagrees with shared blob")
            return
        name = f"blobs/sha256/{descriptor.digest.removeprefix('sha256:')}"
        try:
            member = self.members[name]
        except KeyError as exc:
            raise C0ReproducibilityError(f"{label} descriptor blob is absent") from exc
        if not member.isreg() or member.size != descriptor.size:
            raise C0ReproducibilityError(f"{label} descriptor size does not match its blob")
        source = self.open_blob(descriptor.digest, label=label)
        digest = hashlib.sha256()
        count = 0
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            digest.update(block)
        if count != descriptor.size or f"sha256:{digest.hexdigest()}" != descriptor.digest:
            raise C0ReproducibilityError(f"{label} descriptor digest does not match its blob")
        self.verified_blobs.add(descriptor.digest)

    def read_descriptor_json(
        self,
        descriptor: DescriptorProjection,
        *,
        maximum: int,
        label: str,
    ) -> tuple[bytes, Mapping[str, Any]]:
        self.verify_descriptor(descriptor, label=label)
        if descriptor.size > maximum:
            raise C0ReproducibilityError(f"{label} exceeds the admitted JSON byte limit")
        source = self.open_blob(descriptor.digest, label=label)
        payload = source.read(maximum + 1)
        if len(payload) != descriptor.size or len(payload) > maximum or source.read(1):
            raise C0ReproducibilityError(f"{label} blob changed during JSON read")
        return payload, _strict_json_object(payload, label=label)

    def reject_unreferenced_blobs(self) -> None:
        observed = {
            f"sha256:{name.rsplit('/', 1)[1]}"
            for name, member in self.members.items()
            if name.startswith("blobs/sha256/") and member.isreg()
        }
        if observed != self.referenced_blobs:
            extra = sorted(observed - self.referenced_blobs)
            missing = sorted(self.referenced_blobs - observed)
            raise C0ReproducibilityError(
                f"OCI blob closure is not exact; extra={extra[:3]}, missing={missing[:3]}"
            )


@dataclass(frozen=True)
class _LayerFile:
    payload: bytes
    mode: int


class _DigestingRaw(io.RawIOBase):
    def __init__(self, source: BinaryIO, *, maximum: int) -> None:
        self.source = source
        self.maximum = maximum
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def readable(self) -> bool:
        return True

    def readinto(self, target: bytearray | memoryview) -> int:
        block = self.source.read(len(target))
        if not block:
            return 0
        self.byte_count += len(block)
        if self.byte_count > self.maximum:
            raise C0ReproducibilityError("uncompressed OCI layer exceeds the admitted byte limit")
        self.digest.update(block)
        target[: len(block)] = block
        return len(block)


def _path_intersects(first: str, second: str) -> bool:
    return first == second or first.startswith(f"{second}/") or second.startswith(f"{first}/")


def _victim_affects(target: str, victim: str) -> bool:
    return target == victim or target.startswith(f"{victim}/")


def _opaque_affects(target: str, directory: str) -> bool:
    return not directory or target.startswith(f"{directory}/")


def _read_layer_member(
    layer: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    maximum: int,
    label: str,
) -> bytes:
    if member.size < 0 or member.size > maximum or member.sparse:
        raise C0ReproducibilityError(f"{label} has an invalid byte count or sparse encoding")
    source = layer.extractfile(member)
    if source is None:
        raise C0ReproducibilityError(f"cannot read {label}")
    payload = source.read(maximum + 1)
    if len(payload) != member.size or len(payload) > maximum or source.read(1):
        raise C0ReproducibilityError(f"{label} byte count disagrees with its tar header")
    return payload


def _apply_layer(
    oci: _OciArchive,
    descriptor: DescriptorProjection,
    fixed_state: Mapping[str, _LayerFile],
    wheel_state: Mapping[str, _LayerFile],
    *,
    position: int,
) -> tuple[dict[str, _LayerFile], dict[str, _LayerFile], str]:
    if descriptor.media_type not in {
        _OCI_LAYER_MEDIA_TYPE,
        _OCI_GZIP_LAYER_MEDIA_TYPE,
        _DOCKER_GZIP_LAYER_MEDIA_TYPE,
    }:
        raise C0ReproducibilityError(
            f"executable layer {position} uses unsupported compression/media type "
            f"{descriptor.media_type!r}"
        )
    oci.verify_descriptor(descriptor, label=f"executable layer {position}")
    compressed = oci.open_blob(descriptor.digest, label=f"executable layer {position}")
    decoded_source: BinaryIO
    if descriptor.media_type == _OCI_LAYER_MEDIA_TYPE:
        decoded_source = compressed
    else:
        decoded_source = gzip.GzipFile(fileobj=compressed, mode="rb")
    digesting = _DigestingRaw(decoded_source, maximum=_MAX_UNCOMPRESSED_LAYER_BYTES)
    buffered = io.BufferedReader(digesting, buffer_size=1024 * 1024)

    additions: dict[str, _LayerFile] = {}
    wheel_additions: dict[str, _LayerFile] = {}
    whiteout_victims: set[str] = set()
    opaque_directories: set[str] = set()
    member_paths: set[str] = set()
    try:
        with tarfile.open(fileobj=buffered, mode="r|") as layer:
            for member_position, member in enumerate(layer):
                if member_position >= _MAX_LAYER_MEMBERS:
                    raise C0ReproducibilityError(
                        f"executable layer {position} contains too many members"
                    )
                path = _canonical_tar_member_path(
                    member.name,
                    label=f"executable layer {position} member path",
                )
                if path in member_paths:
                    raise C0ReproducibilityError(
                        f"executable layer {position} repeats member {path!r}"
                    )
                member_paths.add(path)
                if member.isdev() or member.isfifo():
                    raise C0ReproducibilityError(
                        f"executable layer {position} contains device/FIFO member {path!r}"
                    )

                parent, _, basename = path.rpartition("/")
                if basename == ".wh..wh..opq":
                    if not member.isreg() or member.size != 0:
                        raise C0ReproducibilityError("OCI opaque whiteout must be an empty file")
                    if parent in opaque_directories:
                        raise C0ReproducibilityError("OCI layer repeats one opaque whiteout")
                    opaque_directories.add(parent)
                    continue
                if basename.startswith(".wh."):
                    victim_basename = basename.removeprefix(".wh.")
                    if (
                        not victim_basename
                        or victim_basename.startswith(".wh.")
                        or not member.isreg()
                        or member.size != 0
                    ):
                        raise C0ReproducibilityError("OCI whiteout is malformed or reserved")
                    victim = f"{parent}/{victim_basename}" if parent else victim_basename
                    if victim in whiteout_victims:
                        raise C0ReproducibilityError("OCI layer repeats one whiteout victim")
                    whiteout_victims.add(victim)
                    continue

                relevant = any(_path_intersects(path, target) for target in _TRACKED_TARGETS)
                relevant = relevant or _path_intersects(path, _HNSW_WHEEL_ROOT)
                if (member.issym() or member.islnk()) and relevant:
                    raise C0ReproducibilityError(
                        f"runtime target state crosses link member {path!r}"
                    )

                for target in _TRACKED_TARGETS:
                    if path == target:
                        if not member.isreg():
                            raise C0ReproducibilityError(
                                f"runtime target {target!r} is not a regular file"
                            )
                        maximum = (
                            _MAX_RECEIPT_BYTES
                            if target == _HNSW_RECEIPT_PATH
                            else _MAX_RUNTIME_BYTES
                        )
                        additions[target] = _LayerFile(
                            _read_layer_member(
                                layer,
                                member,
                                maximum=maximum,
                                label=f"runtime target {target}",
                            ),
                            member.mode & 0o7777,
                        )
                    elif target.startswith(f"{path}/") and not member.isdir():
                        raise C0ReproducibilityError(
                            f"runtime target {target!r} has non-directory ancestor {path!r}"
                        )
                    elif path.startswith(f"{target}/"):
                        raise C0ReproducibilityError(
                            f"runtime file target {target!r} has descendant member {path!r}"
                        )

                if path == _HNSW_WHEEL_ROOT:
                    if not member.isdir():
                        raise C0ReproducibilityError("hnsw wheel root is not a directory")
                elif _HNSW_WHEEL_ROOT.startswith(f"{path}/") and not member.isdir():
                    raise C0ReproducibilityError(
                        f"hnsw wheel root has non-directory ancestor {path!r}"
                    )
                elif path.startswith(f"{_HNSW_WHEEL_ROOT}/"):
                    relative = path.removeprefix(f"{_HNSW_WHEEL_ROOT}/")
                    if (
                        "/" in relative
                        or not member.isreg()
                        or _WHEEL_NAME.fullmatch(relative) is None
                    ):
                        raise C0ReproducibilityError(
                            f"hnsw wheel directory contains ambiguous member {path!r}"
                        )
                    wheel_additions[path] = _LayerFile(
                        _read_layer_member(
                            layer,
                            member,
                            maximum=_MAX_WHEEL_BYTES,
                            label=f"hnsw wheel {relative}",
                        ),
                        member.mode & 0o7777,
                    )
        while buffered.read(1024 * 1024):
            pass
    except (tarfile.TarError, gzip.BadGzipFile, EOFError, OSError) as exc:
        raise C0ReproducibilityError(
            f"cannot decode executable layer {position} as its declared tar stream"
        ) from exc
    finally:
        try:
            buffered.close()
        except OSError:
            pass

    fixed = dict(fixed_state)
    for target in _TRACKED_TARGETS:
        if any(_victim_affects(target, victim) for victim in whiteout_victims) or any(
            _opaque_affects(target, directory) for directory in opaque_directories
        ):
            fixed.pop(target, None)
    fixed.update(additions)

    wheels = dict(wheel_state)
    if any(_victim_affects(_HNSW_WHEEL_ROOT, victim) for victim in whiteout_victims) or any(
        _opaque_affects(_HNSW_WHEEL_ROOT, directory) for directory in opaque_directories
    ):
        wheels.clear()
    else:
        for existing in tuple(wheels):
            if any(_victim_affects(existing, victim) for victim in whiteout_victims) or any(
                _opaque_affects(existing, directory) for directory in opaque_directories
            ):
                wheels.pop(existing, None)
    wheels.update(wheel_additions)
    return fixed, wheels, f"sha256:{digesting.digest.hexdigest()}"


def _parse_hnsw_receipt(payload: bytes) -> Mapping[str, Any]:
    if len(payload) > _MAX_RECEIPT_BYTES or not payload.endswith(b"\n"):
        raise C0ReproducibilityError("hnsw runtime receipt lacks its canonical terminal newline")
    row = _closed_mapping(
        _strict_json_object(payload[:-1], label="hnsw runtime receipt"),
        fields=_HNSW_RECEIPT_FIELDS,
        label="hnsw runtime receipt",
    )
    if _canonical_json(row) + b"\n" != payload:
        raise C0ReproducibilityError("hnsw runtime receipt is not canonical JSON")
    exact = {
        "package": "hnswlib",
        "python_abi": "cp312",
        "schema_version": "fractal-hnswlib-runtime-artifact-v1",
        "sdist_sha256": "cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c",
        "version": "0.8.0",
    }
    for field, expected in exact.items():
        if row[field] != expected:
            raise C0ReproducibilityError(f"hnsw runtime receipt has invalid {field}")
    for field in ("extension_sha256", "wheel_sha256"):
        if not isinstance(row[field], str) or _SHA256_HEX.fullmatch(row[field]) is None:
            raise C0ReproducibilityError(f"hnsw runtime receipt has invalid {field}")
    for field in ("extension_byte_count", "wheel_byte_count"):
        _positive_int(row[field], label=f"hnsw runtime receipt {field}")
    wheel_name = _require_canonical_string(row["wheel_basename"], label="hnsw wheel basename")
    if _WHEEL_NAME.fullmatch(wheel_name) is None:
        raise C0ReproducibilityError("hnsw runtime receipt has invalid wheel basename")
    extension_name = _require_canonical_string(
        row["extension_basename"], label="hnsw extension basename"
    )
    if (
        not extension_name.startswith("hnswlib.")
        or not extension_name.endswith(".so")
        or "/" in extension_name
        or "\\" in extension_name
    ):
        raise C0ReproducibilityError("hnsw runtime receipt has invalid extension basename")
    return row


def _parse_canonical_receipt(
    payload: bytes,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if len(payload) > _MAX_RECEIPT_BYTES or not payload.endswith(b"\n"):
        raise C0ReproducibilityError(f"{label} lacks its canonical terminal newline")
    row = _closed_mapping(
        _strict_json_object(payload[:-1], label=label),
        fields=fields,
        label=label,
    )
    if _canonical_json(row) + b"\n" != payload:
        raise C0ReproducibilityError(f"{label} is not canonical JSON")
    return row


def _verify_build_receipts(
    fixed: Mapping[str, _LayerFile],
    *,
    expected_source_epoch: int,
) -> None:
    native = _parse_canonical_receipt(
        fixed[_NATIVE_BUILD_RECEIPT_PATH].payload,
        fields=_NATIVE_BUILD_RECEIPT_FIELDS,
        label="native build receipt",
    )
    native_exact: dict[str, object] = {
        "debian_inrelease_sha256": _DEBIAN_INRELEASE_SHA256,
        "debian_keyring_sha256": _DEBIAN_KEYRING_SHA256,
        "debian_snapshot": _DEBIAN_SNAPSHOT,
        "schema_version": "fractal-native-build-receipt-v1",
        "source_date_epoch": expected_source_epoch,
        "sqlite_autoconf": "3530300",
        "sqlite_sha256": _SQLITE_SHA256,
        "sqlite_sha3_256": _SQLITE_SHA3_256,
        "sqlite_source_id": (
            "2026-06-26 20:14:12 d4c0e51e4aeb96955b99185ab9cde75c339e2c29c3f3f12428d364a10d782c62"
        ),
        "zlib_sha256": _ZLIB_SHA256,
        "zlib_version": "1.3.2",
    }
    for field, expected in native_exact.items():
        if native[field] != expected:
            raise C0ReproducibilityError(f"native build receipt has invalid {field}")
    for field, path in (
        ("sqlite_library_sha256", _SQLITE_LIBRARY_PATH),
        ("zlib_library_sha256", _ZLIB_LIBRARY_PATH),
    ):
        observed = hashlib.sha256(fixed[path].payload).hexdigest()
        if native[field] != observed:
            raise C0ReproducibilityError(
                f"native build receipt {field} disagrees with its runtime library"
            )

    opa = _parse_canonical_receipt(
        fixed[_OPA_BUILD_RECEIPT_PATH].payload,
        fields=_OPA_BUILD_RECEIPT_FIELDS,
        label="OPA build receipt",
    )
    opa_exact: dict[str, object] = {
        "dependency_delta_sha256": _OPA_DEPENDENCY_DELTA_SHA256,
        "go_builder_image": _GO_BUILDER_IMAGE,
        "go_version": "1.26.5",
        "opa_commit": _OPA_COMMIT,
        "opa_release_timestamp": "2026-07-02T13:14:00Z",
        "opa_source_sha256": _OPA_SOURCE_SHA256,
        "opa_version": "1.18.2",
        "oras_original_version": "2.6.1",
        "oras_patched_version": "2.6.2",
        "original_go_mod_sha256": (
            "59b4beeea1af5d33ce1c22579e24ab4b0002a3638d0aadeb82a5a4500eb8a175"
        ),
        "original_go_sum_sha256": (
            "be7b973025c1a5588a822baed9513f7356e08a6794fa24db79b8fb832cee6b2f"
        ),
        "patched_go_mod_sha256": (
            "049c4ae3f1d58e8dd8885249873f358b4abbab1ece8f910a3549521dffd026e5"
        ),
        "patched_go_sum_sha256": (
            "594c9098656b4b4b4a41f11093ff95babda2d0333077f8a7ad42528466da0903"
        ),
        "schema_version": "fractal-opa-build-receipt-v1",
        "source_date_epoch": expected_source_epoch,
        "target_arch": "arm64",
        "x_sync_original_version": "0.21.0",
        "x_sync_patched_version": "0.22.0",
    }
    for field, expected in opa_exact.items():
        if opa[field] != expected:
            raise C0ReproducibilityError(f"OPA build receipt has invalid {field}")
    if opa["go_tarball_sha256"] != (
        "fe4789e92b1f33358680864bbe8704289e7bb5fc207d80623c308935bd696d49"
    ):
        raise C0ReproducibilityError("OPA build receipt has invalid arm64 Go tarball digest")
    if opa["opa_sha256"] != hashlib.sha256(fixed[_OPA_PATH].payload).hexdigest():
        raise C0ReproducibilityError("OPA build receipt disagrees with the runtime OPA binary")
    build_info = fixed[_OPA_BUILD_INFO_PATH].payload
    for marker in (
        b"go1.26.5",
        b"oras.land/oras-go/v2\tv2.6.2",
        b"golang.org/x/sync\tv0.22.0",
        b"CGO_ENABLED=0",
    ):
        if marker not in build_info:
            raise C0ReproducibilityError("OPA Go build information lacks a fixed toolchain marker")

    manifest_payload = fixed[_NATIVE_LIBRARY_MANIFEST_PATH].payload
    if not manifest_payload.endswith(b"\n"):
        raise C0ReproducibilityError("runtime library manifest lacks its terminal newline")
    manifest = _strict_json_object(manifest_payload[:-1], label="runtime library manifest")
    if _canonical_json(manifest) + b"\n" != manifest_payload:
        raise C0ReproducibilityError("runtime library manifest is not canonical JSON")
    if (
        set(manifest) != {"libraries", "schema_version"}
        or manifest.get("schema_version") != "fractal-runtime-library-manifest-v1"
    ):
        raise C0ReproducibilityError("runtime library manifest has another schema")
    libraries = _array(manifest.get("libraries"), label="runtime library manifest libraries")
    custom = {
        row.get("destination_path"): row
        for row in libraries
        if isinstance(row, Mapping) and "purl" in row
    }
    for path, purl, version in (
        ("/opt/native-libs/libsqlite3.so.0", "pkg:generic/sqlite@3.53.3", "3.53.3"),
        ("/opt/native-libs/libz.so.1", "pkg:generic/zlib@1.3.2", "1.3.2"),
    ):
        row = custom.get(path)
        if not isinstance(row, Mapping):
            raise C0ReproducibilityError(f"runtime library manifest lacks {path}")
        fixed_path = path.removeprefix("/")
        payload = fixed[fixed_path].payload
        if (
            row.get("purl") != purl
            or row.get("package_version") != version
            or row.get("byte_count") != len(payload)
            or row.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise C0ReproducibilityError(f"runtime library manifest metadata disagrees with {path}")


def _validate_static_tle_elf(payload: bytes) -> None:
    _validate_aarch64_elf(payload, label="final tle target")
    if len(payload) < 64 or int.from_bytes(payload[16:18], "little") != 2:
        raise C0ReproducibilityError("final tle target is not one ET_EXEC binary")
    program_offset = int.from_bytes(payload[32:40], "little")
    program_entry_size = int.from_bytes(payload[54:56], "little")
    program_count = int.from_bytes(payload[56:58], "little")
    if (
        program_entry_size < 56
        or not 1 <= program_count <= 4096
        or program_offset + program_entry_size * program_count > len(payload)
    ):
        raise C0ReproducibilityError("final tle target has an invalid program-header table")
    program_types = {
        int.from_bytes(
            payload[
                program_offset + position * program_entry_size : program_offset
                + position * program_entry_size
                + 4
            ],
            "little",
        )
        for position in range(program_count)
    }
    if 2 in program_types or 3 in program_types:
        raise C0ReproducibilityError("final tle target contains PT_DYNAMIC or PT_INTERP")


def _verify_tle_build_receipt(
    fixed: Mapping[str, _LayerFile],
    *,
    expected_source_epoch: int,
) -> None:
    binary = fixed[_TLE_PATH].payload
    if len(binary) != _TLE_BINARY_BYTE_COUNT:
        raise C0ReproducibilityError("final tle target has another byte count")
    if hashlib.sha256(binary).hexdigest() != _TLE_BINARY_SHA256:
        raise C0ReproducibilityError("final tle target differs from its source-build pin")
    _validate_static_tle_elf(binary)

    receipt_payload = fixed[_TLE_BUILD_RECEIPT_PATH].payload
    receipt = _parse_canonical_receipt(
        receipt_payload,
        fields=_TLE_BUILD_RECEIPT_FIELDS,
        label="tle source-build receipt",
    )
    exact: dict[str, object] = {
        "binary_byte_count": _TLE_BINARY_BYTE_COUNT,
        "binary_image_path": "/usr/local/bin/tle",
        "binary_sha256": _TLE_BINARY_SHA256,
        "build_environment": {
            "CGO_ENABLED": "0",
            "GOARCH": "arm64",
            "GOARM64": "v8.0",
            "GOOS": "linux",
            "GOTOOLCHAIN": "local",
        },
        "builder_image": _GO_BUILDER_IMAGE,
        "dependency_delta_sha256": _TLE_DEPENDENCY_DELTA_SHA256,
        "elf": {
            "class": "ELF64",
            "dynamic_program_header": False,
            "machine": "AArch64",
            "pt_interp": False,
            "type": "ET_EXEC",
        },
        "go_tarball_sha256": _TLE_GO_TARBALL_SHA256,
        "go_tarball_url": "https://go.dev/dl/go1.26.5.linux-arm64.tar.gz",
        "go_tool_sha256": _TLE_GO_TOOL_SHA256,
        "go_version": "1.26.5",
        "included": True,
        "independent_build_count": 2,
        "independent_builds_byte_identical": True,
        "original_go_mod_sha256": _TLE_ORIGINAL_GO_MOD_SHA256,
        "original_go_sum_sha256": _TLE_ORIGINAL_GO_SUM_SHA256,
        "patched_go_mod_sha256": _TLE_PATCHED_GO_MOD_SHA256,
        "patched_go_sum_sha256": _TLE_PATCHED_GO_SUM_SHA256,
        "release_version": "1.2.0",
        "schema_version": "fractal-tle-source-build-receipt-v2",
        "source_archive_sha256": _TLE_SOURCE_ARCHIVE_SHA256,
        "source_archive_url": (
            f"https://github.com/drand/tlock/archive/{_TLE_SOURCE_COMMIT}.tar.gz"
        ),
        "source_commit_git_sha1": _TLE_SOURCE_COMMIT,
        "source_date_epoch": expected_source_epoch,
        "source_tree_manifest_sha256": _TLE_SOURCE_TREE_MANIFEST_SHA256,
        "tag_object_git_sha1": _TLE_TAG_OBJECT,
        "target_arch": "arm64",
        "target_os": "linux",
    }
    for field, expected in exact.items():
        if receipt[field] != expected:
            raise C0ReproducibilityError(f"tle source-build receipt has invalid {field}")
    if _SHA256_HEX.fullmatch(str(receipt["offline_test_inventory_sha256"])) is None:
        raise C0ReproducibilityError(
            "tle source-build receipt has invalid offline test inventory digest"
        )
    expected_commands = [
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
    ]
    if receipt["build_commands"] != expected_commands:
        raise C0ReproducibilityError("tle source-build receipt has another build command set")


def _verify_wheel_extension(
    wheel: bytes,
    receipt: Mapping[str, Any],
    *,
    source_epoch: int,
) -> HnswExtensionProjection:
    if len(wheel) != receipt["wheel_byte_count"]:
        raise C0ReproducibilityError("hnsw receipt and retained wheel byte counts disagree")
    if hashlib.sha256(wheel).hexdigest() != receipt["wheel_sha256"]:
        raise C0ReproducibilityError("hnsw receipt and retained wheel digests disagree")
    try:
        with zipfile.ZipFile(io.BytesIO(wheel), mode="r") as archive:
            members = archive.infolist()
            if len(members) > 200_000:
                raise C0ReproducibilityError("hnsw wheel contains too many ZIP members")
            names: set[str] = set()
            extension_members: list[zipfile.ZipInfo] = []
            expected_timestamp = dt.datetime.fromtimestamp(
                source_epoch - (source_epoch % 2), tz=dt.timezone.utc
            ).timetuple()[:6]
            for member in members:
                name = _canonical_tar_member_path(member.filename.rstrip("/"), label="wheel member")
                if name in names:
                    raise C0ReproducibilityError("hnsw wheel repeats a ZIP member")
                names.add(name)
                if member.flag_bits & 0x1:
                    raise C0ReproducibilityError("hnsw wheel contains encrypted content")
                mode = (member.external_attr >> 16) & 0o170000
                if mode and mode not in {stat.S_IFREG, stat.S_IFDIR}:
                    raise C0ReproducibilityError("hnsw wheel contains a link or special member")
                if not member.is_dir() and tuple(member.date_time) != tuple(expected_timestamp):
                    raise C0ReproducibilityError("hnsw wheel timestamp disagrees with source epoch")
                if PurePosixPath(name).name == receipt["extension_basename"]:
                    extension_members.append(member)
            if len(extension_members) != 1:
                raise C0ReproducibilityError(
                    "hnsw wheel does not contain exactly one named extension"
                )
            if (
                extension_members[0].file_size != receipt["extension_byte_count"]
                or extension_members[0].file_size > _MAX_RUNTIME_BYTES
            ):
                raise C0ReproducibilityError("hnsw extension ZIP size disagrees with its receipt")
            extension = archive.read(extension_members[0])
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise C0ReproducibilityError(
            "retained hnsw wheel is not a valid closed ZIP archive"
        ) from exc
    if len(extension) != receipt["extension_byte_count"]:
        raise C0ReproducibilityError("hnsw extension byte count disagrees with its receipt")
    digest = hashlib.sha256(extension).hexdigest()
    if digest != receipt["extension_sha256"]:
        raise C0ReproducibilityError("hnsw extension digest disagrees with its receipt")
    _validate_aarch64_elf(extension, label="hnsw imported extension")
    return HnswExtensionProjection(receipt["extension_basename"], len(extension), digest)


def _parse_env(value: object) -> dict[str, str]:
    if value is None:
        return {}
    rows = _array(value, label="image config Env")
    result: dict[str, str] = {}
    for position, row in enumerate(rows):
        text = _require_canonical_string(row, label=f"image config Env[{position}]")
        name, separator, item = text.partition("=")
        if not separator or _ENV_NAME.fullmatch(name) is None or name in result:
            raise C0ReproducibilityError("image config Env contains malformed or duplicate keys")
        result[name] = item
    return result


def _source_epoch_from_created(value: str) -> int:
    matched = _RFC3339_UTC.fullmatch(value)
    if matched is None:
        raise C0ReproducibilityError("image config created is not canonical UTC RFC3339")
    fraction = matched.group("fraction")
    if fraction is not None and any(character != "0" for character in fraction[1:]):
        raise C0ReproducibilityError("image config created has a fractional source epoch")
    try:
        parsed = dt.datetime.strptime(
            f"{matched.group('date')}T{matched.group('time')}", "%Y-%m-%dT%H:%M:%S"
        )
    except ValueError as exc:
        raise C0ReproducibilityError("image config created is not a real timestamp") from exc
    return calendar.timegm(parsed.timetuple())


def _manifest_parts(
    oci: _OciArchive,
    descriptor: DescriptorProjection,
    *,
    label: str,
) -> tuple[DescriptorProjection, tuple[DescriptorProjection, ...], Mapping[str, Any]]:
    if descriptor.media_type not in {_OCI_MANIFEST_MEDIA_TYPE, _DOCKER_MANIFEST_MEDIA_TYPE}:
        raise C0ReproducibilityError(f"{label} uses unsupported manifest media type")
    _raw, manifest = oci.read_descriptor_json(
        descriptor.content_only(), maximum=_MAX_MANIFEST_JSON_BYTES, label=label
    )
    row = _closed_mapping(
        manifest,
        fields=_MANIFEST_FIELDS,
        required=frozenset({"config", "layers", "schemaVersion"}),
        label=label,
    )
    if row["schemaVersion"] != 2:
        raise C0ReproducibilityError(f"{label} schemaVersion must equal 2")
    if "mediaType" in row and row["mediaType"] != descriptor.media_type:
        raise C0ReproducibilityError(f"{label} media type disagrees with its descriptor")
    _annotations(row.get("annotations"), label=f"{label}.annotations")
    if "artifactType" in row:
        _require_canonical_string(row["artifactType"], label=f"{label}.artifactType")
    if "subject" in row:
        _parse_descriptor(row["subject"], label=f"{label}.subject", index_descriptor=False)
    config = _parse_descriptor(row["config"], label=f"{label}.config", index_descriptor=False)
    layers = tuple(
        _parse_descriptor(value, label=f"{label}.layers[{position}]", index_descriptor=False)
        for position, value in enumerate(_array(row["layers"], label=f"{label}.layers"))
    )
    if not layers:
        raise C0ReproducibilityError(f"{label} must contain at least one layer")
    return config, layers, row


def _inspect_executable(
    oci: _OciArchive,
    descriptor: DescriptorProjection,
    *,
    expected_build_context_tree_sha256: str,
    expected_source_epoch: int,
    expected_uv_lock_sha256: str,
    expected_opa_policy_sha256: str,
    image_role: str,
) -> ExecutableProjection:
    if image_role not in {"scientific", "timelock-release"}:
        raise C0ReproducibilityError("image role is outside the closed C0 role set")
    platform = descriptor.platform
    if platform is None or platform.architecture != "arm64" or platform.operating_system != "linux":
        raise C0ReproducibilityError("selected executable descriptor is not linux/arm64")
    if platform.variant not in {None, "v8"}:
        raise C0ReproducibilityError("selected executable descriptor has another arm64 variant")
    config_descriptor, layer_descriptors, manifest = _manifest_parts(
        oci, descriptor, label="linux/arm64 executable manifest"
    )
    if "subject" in manifest or "artifactType" in manifest:
        raise C0ReproducibilityError("executable manifest cannot be an OCI artifact/referrer")
    if config_descriptor.media_type not in {_OCI_CONFIG_MEDIA_TYPE, _DOCKER_CONFIG_MEDIA_TYPE}:
        raise C0ReproducibilityError("executable manifest names another config media type")
    _raw_config, config = oci.read_descriptor_json(
        config_descriptor, maximum=_MAX_CONFIG_JSON_BYTES, label="linux/arm64 image config"
    )
    if config.get("architecture") != "arm64" or config.get("os") != "linux":
        raise C0ReproducibilityError("image config is not linux/arm64")
    config_variant = config.get("variant")
    if config_variant not in {None, "v8"} or (
        platform.variant is not None and config_variant not in {None, platform.variant}
    ):
        raise C0ReproducibilityError("image config and index arm64 variants disagree")
    rootfs = config.get("rootfs")
    if not isinstance(rootfs, Mapping) or set(rootfs) != {"diff_ids", "type"}:
        raise C0ReproducibilityError("image config rootfs must be one closed layers object")
    if rootfs["type"] != "layers":
        raise C0ReproducibilityError("image config rootfs type must equal 'layers'")
    diff_ids = tuple(
        _sha256_digest(value, label=f"image config rootfs.diff_ids[{position}]")
        for position, value in enumerate(
            _array(rootfs["diff_ids"], label="image config rootfs.diff_ids")
        )
    )
    if len(diff_ids) != len(layer_descriptors):
        raise C0ReproducibilityError("image config rootfs.diff_ids and manifest layers disagree")

    runtime_config = config.get("config")
    if not isinstance(runtime_config, Mapping):
        raise C0ReproducibilityError("image config lacks its runtime config object")
    labels_value = runtime_config.get("Labels")
    if not isinstance(labels_value, Mapping):
        raise C0ReproducibilityError("image config lacks its Labels object")
    labels = {
        _require_canonical_string(key, label="image label key"): _require_canonical_string(
            value, label=f"image label {key!r}"
        )
        for key, value in labels_value.items()
    }
    required_labels = {
        "io.fractal-ann.confirmatory.build-context-tree-sha256": (
            expected_build_context_tree_sha256
        ),
        "io.fractal-ann.confirmatory.debian-inrelease-sha256": (_DEBIAN_INRELEASE_SHA256),
        "io.fractal-ann.confirmatory.debian-keyring-sha256": _DEBIAN_KEYRING_SHA256,
        "io.fractal-ann.confirmatory.debian-snapshot": _DEBIAN_SNAPSHOT,
        "io.fractal-ann.confirmatory.go-builder-image": _GO_BUILDER_IMAGE,
        "io.fractal-ann.confirmatory.opa-commit": _OPA_COMMIT,
        "io.fractal-ann.confirmatory.opa-dependency-delta-sha256": (_OPA_DEPENDENCY_DELTA_SHA256),
        "io.fractal-ann.confirmatory.opa-rego-sha256": expected_opa_policy_sha256,
        "io.fractal-ann.confirmatory.opa-rego-test-sha256": _OPA_REGO_TEST_SHA256,
        "io.fractal-ann.confirmatory.opa-source-sha256": _OPA_SOURCE_SHA256,
        "io.fractal-ann.confirmatory.oras-version": "2.6.2",
        "io.fractal-ann.confirmatory.python-builder-image": _PYTHON_BUILDER_IMAGE,
        "io.fractal-ann.confirmatory.runtime-role": "scientific",
        "io.fractal-ann.confirmatory.source-date-epoch": str(expected_source_epoch),
        "io.fractal-ann.confirmatory.sqlite-sha256": _SQLITE_SHA256,
        "io.fractal-ann.confirmatory.sqlite-sha3-256": _SQLITE_SHA3_256,
        "io.fractal-ann.confirmatory.tle-present": "false",
        "io.fractal-ann.confirmatory.uv-lock-sha256": expected_uv_lock_sha256,
        "io.fractal-ann.confirmatory.zlib-sha256": _ZLIB_SHA256,
        "org.opencontainers.image.authors": "mhdk1602 <mhdk1602@users.noreply.github.com>",
        "org.opencontainers.image.base.name": _DISTROLESS_BASE_IMAGE,
        "org.opencontainers.image.description": (
            "C0-pinned execution image for the registered Fractal ANN confirmatory apparatus"
        ),
        "org.opencontainers.image.documentation": (
            "https://github.com/mhdk1602/fractal-ann-diagnostics/blob/master/"
            "research/runner-image.md"
        ),
        "org.opencontainers.image.licenses": "MIT",
        "org.opencontainers.image.revision": expected_build_context_tree_sha256,
        "org.opencontainers.image.source": "https://github.com/mhdk1602/fractal-ann-diagnostics",
        "org.opencontainers.image.title": "Fractal ANN confirmatory runner",
        "org.opencontainers.image.url": "https://github.com/mhdk1602/fractal-ann-diagnostics",
        "org.opencontainers.image.vendor": "mhdk1602",
        "org.opencontainers.image.version": expected_build_context_tree_sha256,
    }
    if image_role == "timelock-release":
        required_labels.update(
            {
                "io.fractal-ann.confirmatory.runtime-role": "timelock-release",
                "io.fractal-ann.confirmatory.tle-binary-sha256": _TLE_BINARY_SHA256,
                "io.fractal-ann.confirmatory.tle-dependency-delta-sha256": (
                    _TLE_DEPENDENCY_DELTA_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-go-tarball-sha256": (_TLE_GO_TARBALL_SHA256),
                "io.fractal-ann.confirmatory.tle-go-tool-sha256": _TLE_GO_TOOL_SHA256,
                "io.fractal-ann.confirmatory.tle-patched-go-mod-sha256": (
                    _TLE_PATCHED_GO_MOD_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-patched-go-sum-sha256": (
                    _TLE_PATCHED_GO_SUM_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-present": "true",
                "io.fractal-ann.confirmatory.tle-release-version": "1.2.0",
                "io.fractal-ann.confirmatory.tle-runtime-scope": "linux/arm64-only",
                "io.fractal-ann.confirmatory.tle-source-archive-sha256": (
                    _TLE_SOURCE_ARCHIVE_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-source-commit": _TLE_SOURCE_COMMIT,
                "io.fractal-ann.confirmatory.tle-source-tree-manifest-sha256": (
                    _TLE_SOURCE_TREE_MANIFEST_SHA256
                ),
                "io.fractal-ann.confirmatory.tle-tag-object": _TLE_TAG_OBJECT,
                "org.opencontainers.image.description": (
                    "C0-pinned ARM64 timelock release image for the registered "
                    "Fractal ANN confirmatory apparatus"
                ),
                "org.opencontainers.image.title": ("Fractal ANN confirmatory release runner"),
            }
        )
    if labels != required_labels:
        raise C0ReproducibilityError("image Labels differ from the closed C0 label set")
    runtime_config_fields = set(runtime_config)
    if runtime_config_fields != _RUNTIME_CONFIG_FIELDS:
        missing_fields = sorted(_RUNTIME_CONFIG_FIELDS - runtime_config_fields)
        unexpected_fields = sorted(runtime_config_fields - _RUNTIME_CONFIG_FIELDS)
        raise C0ReproducibilityError(
            "image runtime config fields differ from the closed C0 set: "
            f"missing={missing_fields!r}; unexpected={unexpected_fields!r}"
        )
    if runtime_config.get("ArgsEscaped") is not True:
        raise C0ReproducibilityError(
            "image runtime ArgsEscaped compatibility marker differs from true"
        )
    if runtime_config.get("User") != "65532:65532":
        raise C0ReproducibilityError("image runtime user differs from 65532:65532")
    if runtime_config.get("WorkingDir") != "/workspace":
        raise C0ReproducibilityError("image runtime working directory differs")
    if _string_array(runtime_config.get("Entrypoint"), label="image config Entrypoint") != (
        "/opt/venv/bin/python",
        "-m",
        "fractal_ann_diagnostics.cli",
    ):
        raise C0ReproducibilityError("image runtime entrypoint differs")
    if _string_array(runtime_config.get("Cmd"), label="image config Cmd") != ("--help",):
        raise C0ReproducibilityError("image runtime command differs")
    environment = _parse_env(runtime_config.get("Env"))
    expected_environment_keys = set(_EXPECTED_RUNTIME_ENVIRONMENT)
    if set(environment) != expected_environment_keys:
        raise C0ReproducibilityError("image Env keys differ from the closed C0 set")
    for key, expected in _EXPECTED_RUNTIME_ENVIRONMENT.items():
        if environment[key] != expected:
            raise C0ReproducibilityError(f"image environment {key} differs")
    environment_bindings = tuple(sorted(environment.items()))

    created_value = config.get("created")
    created: str | None
    if created_value is None:
        raise C0ReproducibilityError("image config lacks its source-epoch creation time")
    created = _require_canonical_string(created_value, label="image config created")
    if _source_epoch_from_created(created) != expected_source_epoch:
        raise C0ReproducibilityError("image config created disagrees with expected source epoch")

    fixed: dict[str, _LayerFile] = {}
    wheels: dict[str, _LayerFile] = {}
    observed_diff_ids: list[str] = []
    for position, layer_descriptor in enumerate(layer_descriptors):
        fixed, wheels, observed_diff_id = _apply_layer(
            oci,
            layer_descriptor,
            fixed,
            wheels,
            position=position,
        )
        observed_diff_ids.append(observed_diff_id)
    if tuple(observed_diff_ids) != diff_ids:
        raise C0ReproducibilityError(
            "computed executable layer diff_ids disagree with image config"
        )
    required_targets = (
        _FIXED_TARGETS if image_role == "scientific" else _FIXED_TARGETS + _FORBIDDEN_TARGETS
    )
    missing = [target for target in required_targets if target not in fixed]
    if missing:
        raise C0ReproducibilityError(f"executable image lacks runtime targets: {missing}")
    leaked_release_targets = [target for target in _FORBIDDEN_TARGETS if target in fixed]
    if image_role == "scientific" and leaked_release_targets:
        raise C0ReproducibilityError(
            f"scientific image contains release-only targets: {leaked_release_targets}"
        )
    if len(wheels) != 1:
        raise C0ReproducibilityError("executable image does not end with exactly one hnsw wheel")

    _verify_build_receipts(fixed, expected_source_epoch=expected_source_epoch)
    if image_role == "timelock-release":
        _verify_tle_build_receipt(fixed, expected_source_epoch=expected_source_epoch)
    receipt = _parse_hnsw_receipt(fixed[_HNSW_RECEIPT_PATH].payload)
    expected_wheel_path = f"{_HNSW_WHEEL_ROOT}/{receipt['wheel_basename']}"
    try:
        wheel_file = wheels[expected_wheel_path]
    except KeyError as exc:
        raise C0ReproducibilityError("hnsw receipt names another retained wheel") from exc
    extension = _verify_wheel_extension(
        wheel_file.payload,
        receipt,
        source_epoch=expected_source_epoch,
    )
    if hashlib.sha256(fixed[_UV_LOCK_PATH].payload).hexdigest() != expected_uv_lock_sha256:
        raise C0ReproducibilityError("final uv.lock differs from its C0 pin")
    if hashlib.sha256(fixed[_OPA_POLICY_PATH].payload).hexdigest() != expected_opa_policy_sha256:
        raise C0ReproducibilityError("final OPA policy differs from its C0 pin")
    _validate_aarch64_elf(fixed[_OPA_PATH].payload, label="final OPA target")
    _validate_aarch64_elf(fixed[_PYTHON_PATH].payload, label="final Python target")
    for executable_path in (_OPA_PATH, _PYTHON_PATH):
        mode = fixed[executable_path].mode
        if mode & 0o222 or not mode & 0o111:
            raise C0ReproducibilityError(f"runtime executable {executable_path!r} has unsafe mode")
    if image_role == "timelock-release":
        tle_mode = fixed[_TLE_PATH].mode
        if tle_mode & 0o222 or not tle_mode & 0o111:
            raise C0ReproducibilityError("runtime tle target has unsafe mode")
    for readonly in (
        target for target in required_targets if target not in {_OPA_PATH, _PYTHON_PATH, _TLE_PATH}
    ):
        if fixed[readonly].mode & 0o222:
            raise C0ReproducibilityError(f"runtime target {readonly!r} remains writable")
    if wheel_file.mode & 0o222:
        raise C0ReproducibilityError("retained hnsw wheel remains writable")

    runtime_rows: list[tuple[str, _LayerFile]] = [
        (target, fixed[target]) for target in required_targets
    ]
    runtime_rows.append((expected_wheel_path, wheel_file))
    runtime_files = tuple(
        FileProjection(
            image_path=f"/{path}",
            byte_count=len(item.payload),
            sha256=hashlib.sha256(item.payload).hexdigest(),
            mode=f"{item.mode:04o}",
        )
        for path, item in runtime_rows
    )
    return ExecutableProjection(
        platform=platform,
        manifest=descriptor.content_only(),
        config=config_descriptor,
        ordered_layers=layer_descriptors,
        rootfs_diff_ids=diff_ids,
        config_created=created,
        c0_labels=tuple(sorted(labels.items())),
        c0_environment=environment_bindings,
        runtime_files=runtime_files,
        hnsw_imported_extension=extension,
    )


def _inspect_attestation(
    oci: _OciArchive,
    descriptor: DescriptorProjection,
    *,
    executable_digest: str,
    position: int,
) -> AttestationProjection:
    annotations = dict(descriptor.annotations)
    if annotations.get("vnd.docker.reference.type") != _ATTESTATION_REFERENCE_TYPE:
        raise C0ReproducibilityError("unknown-platform descriptor is not an attestation manifest")
    if annotations.get("vnd.docker.reference.digest") != executable_digest:
        raise C0ReproducibilityError("attestation descriptor references another executable")
    config, layers, _manifest = _manifest_parts(
        oci, descriptor, label=f"attestation manifest {position}"
    )
    oci.verify_descriptor(config, label=f"attestation manifest {position} config")
    for layer_position, layer in enumerate(layers):
        oci.verify_descriptor(
            layer, label=f"attestation manifest {position} layer {layer_position}"
        )
    return AttestationProjection(descriptor, executable_digest, config, layers)


def _parse_index_descriptors(
    index: Mapping[str, Any],
    *,
    label: str,
    platform_required: bool,
    expected_media_type: str | None = None,
) -> tuple[DescriptorProjection, ...]:
    row = _closed_mapping(
        index,
        fields=_INDEX_FIELDS,
        required=frozenset({"manifests", "schemaVersion"}),
        label=label,
    )
    if row["schemaVersion"] != 2:
        raise C0ReproducibilityError(f"{label} schemaVersion must equal 2")
    media_type = row.get("mediaType")
    if media_type not in {None, _OCI_LAYOUT_MEDIA_TYPE, _DOCKER_INDEX_MEDIA_TYPE}:
        raise C0ReproducibilityError(f"{label} declares another media type")
    if expected_media_type is not None and media_type not in {None, expected_media_type}:
        raise C0ReproducibilityError(f"{label} media type disagrees with its descriptor")
    _annotations(row.get("annotations"), label=f"{label} annotations")
    descriptors = tuple(
        _parse_descriptor(
            value,
            label=f"{label} manifest {position}",
            index_descriptor=True,
            platform_required=platform_required,
        )
        for position, value in enumerate(_array(row["manifests"], label=f"{label} manifests"))
    )
    if not descriptors:
        raise C0ReproducibilityError(f"{label} contains no manifest descriptors")
    return descriptors


def _inspect_archive(
    handle: _ArchiveHandle,
    *,
    expected_build_context_tree_sha256: str,
    expected_source_epoch: int,
    expected_uv_lock_sha256: str,
    expected_opa_policy_sha256: str,
    image_role: str,
    label: str,
) -> ArchiveProjection:
    file_object = os.fdopen(os.dup(handle.descriptor), "rb", closefd=True)
    try:
        with tarfile.open(fileobj=file_object, mode="r:") as outer:
            oci = _OciArchive(outer)
            layout_bytes = oci.read_file(
                "oci-layout", maximum=_MAX_LAYOUT_JSON_BYTES, label="oci-layout"
            )
            layout = _closed_mapping(
                _strict_json_object(layout_bytes, label="oci-layout"),
                fields=frozenset({"imageLayoutVersion"}),
                label="oci-layout",
            )
            if layout["imageLayoutVersion"] != "1.0.0" or _canonical_json(layout) != layout_bytes:
                raise C0ReproducibilityError("oci-layout is not canonical OCI layout version 1.0.0")
            index_bytes = oci.read_file(
                "index.json", maximum=_MAX_INDEX_JSON_BYTES, label="OCI index"
            )
            outer_descriptors = _parse_index_descriptors(
                _strict_json_object(index_bytes, label="OCI index"),
                label="OCI index",
                platform_required=False,
            )
            nested_index_descriptor: DescriptorProjection | None = None
            unplatformed = [item for item in outer_descriptors if item.platform is None]
            if unplatformed:
                if (
                    len(outer_descriptors) != 1
                    or len(unplatformed) != 1
                    or unplatformed[0].media_type
                    not in {_OCI_LAYOUT_MEDIA_TYPE, _DOCKER_INDEX_MEDIA_TYPE}
                ):
                    raise C0ReproducibilityError(
                        "outer OCI index has an ambiguous unplatformed descriptor set"
                    )
                nested_index_descriptor = unplatformed[0]
                _nested_bytes, nested_index = oci.read_descriptor_json(
                    nested_index_descriptor.content_only(),
                    maximum=_MAX_INDEX_JSON_BYTES,
                    label="nested execution index",
                )
                descriptors = _parse_index_descriptors(
                    nested_index,
                    label="nested execution index",
                    platform_required=True,
                    expected_media_type=nested_index_descriptor.media_type,
                )
            else:
                descriptors = outer_descriptors
            executable_descriptors = [
                descriptor
                for descriptor in descriptors
                if descriptor.platform is not None
                and descriptor.platform.operating_system == "linux"
                and descriptor.platform.architecture == "arm64"
            ]
            if len(executable_descriptors) != 1:
                raise C0ReproducibilityError(
                    "OCI index must contain exactly one linux/arm64 executable manifest"
                )
            executable_descriptor = executable_descriptors[0]
            attestations: list[DescriptorProjection] = []
            for descriptor in descriptors:
                if descriptor is executable_descriptor:
                    continue
                platform = descriptor.platform
                if (
                    platform is None
                    or platform.operating_system != "unknown"
                    or platform.architecture != "unknown"
                    or platform.variant is not None
                ):
                    raise C0ReproducibilityError(
                        "OCI index contains an extra executable or unsupported platform manifest"
                    )
                attestations.append(descriptor)
            if len(attestations) != 1:
                raise C0ReproducibilityError(
                    "OCI index must contain exactly one arm64 attestation manifest"
                )
            executable = _inspect_executable(
                oci,
                executable_descriptor,
                expected_build_context_tree_sha256=(expected_build_context_tree_sha256),
                expected_source_epoch=expected_source_epoch,
                expected_uv_lock_sha256=expected_uv_lock_sha256,
                expected_opa_policy_sha256=expected_opa_policy_sha256,
                image_role=image_role,
            )
            attestation_rows = tuple(
                _inspect_attestation(
                    oci,
                    descriptor,
                    executable_digest=executable_descriptor.digest,
                    position=position,
                )
                for position, descriptor in enumerate(attestations)
            )
            oci.reject_unreferenced_blobs()
    except (tarfile.TarError, OSError) as exc:
        raise C0ReproducibilityError(f"{label} is not a valid uncompressed OCI-layout tar") from exc
    finally:
        file_object.close()
    _verify_unchanged(handle, label=label)
    return ArchiveProjection(
        archive_byte_count=handle.stat_result.st_size,
        archive_sha256=handle.sha256,
        outer_index_byte_count=len(index_bytes),
        outer_index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        nested_index_descriptor=nested_index_descriptor,
        executable=executable,
        attestations=attestation_rows,
    )


def _prepare_output(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise C0ReproducibilityError("comparison receipt output must be an absolute path")
    try:
        parent = target.parent.resolve(strict=True)
    except OSError as exc:
        raise C0ReproducibilityError("comparison receipt parent does not exist") from exc
    if parent != target.parent or target.parent.is_symlink() or not parent.is_dir():
        raise C0ReproducibilityError("comparison receipt parent must be one real directory")
    try:
        target.lstat()
    except FileNotFoundError:
        return target
    raise C0ReproducibilityError("comparison receipt output already exists")


def _write_exclusive(path: Path, payload: bytes) -> None:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        directory = os.open(path.parent, directory_flags)
    except OSError as exc:
        raise C0ReproducibilityError("cannot secure comparison receipt parent") from exc
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(path.name, flags, 0o444, dir_fd=directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise C0ReproducibilityError("comparison receipt output already exists") from exc
            raise C0ReproducibilityError("cannot create comparison receipt output") from exc
        try:
            os.fchmod(descriptor, 0o444)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("short write")
                offset += written
            os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = bytearray()
            while len(observed) <= len(payload):
                block = os.read(descriptor, min(1024 * 1024, len(payload) + 1 - len(observed)))
                if not block:
                    break
                observed.extend(block)
            if bytes(observed) != payload:
                raise C0ReproducibilityError("comparison receipt changed during exclusive creation")
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            if (
                descriptor_stat.st_dev,
                descriptor_stat.st_ino,
                descriptor_stat.st_size,
                stat.S_IMODE(descriptor_stat.st_mode),
            ) != (path_stat.st_dev, path_stat.st_ino, len(payload), 0o444):
                raise C0ReproducibilityError(
                    "comparison receipt path changed during exclusive creation"
                )
            os.fsync(directory)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def compare_c0_oci_archives(
    *,
    archive_a_path: str | Path,
    archive_b_path: str | Path,
    expected_build_context_tree_sha256: str,
    expected_source_date_epoch: int,
    expected_uv_lock_sha256: str,
    expected_opa_policy_sha256: str,
    output_path: str | Path,
) -> C0ReproducibilityReceipt:
    """Validate, compare, and exclusively publish one C0 executable receipt."""

    _sha256_hex(
        expected_build_context_tree_sha256,
        label="expected build-context tree SHA-256",
    )
    _positive_int(expected_source_date_epoch, label="expected source-date epoch")
    _sha256_hex(expected_uv_lock_sha256, label="expected uv.lock SHA-256")
    _sha256_hex(expected_opa_policy_sha256, label="expected OPA policy SHA-256")
    output = _prepare_output(output_path)
    first = _open_archive(archive_a_path, label="archive A")
    second: _ArchiveHandle | None = None
    try:
        second = _open_archive(archive_b_path, label="archive B")
        first_identity = (first.stat_result.st_dev, first.stat_result.st_ino)
        second_identity = (second.stat_result.st_dev, second.stat_result.st_ino)
        if first_identity == second_identity:
            raise C0ReproducibilityError("archive A and archive B must be distinct files")
        first_projection = _inspect_archive(
            first,
            expected_build_context_tree_sha256=expected_build_context_tree_sha256,
            expected_source_epoch=expected_source_date_epoch,
            expected_uv_lock_sha256=expected_uv_lock_sha256,
            expected_opa_policy_sha256=expected_opa_policy_sha256,
            image_role="scientific",
            label="archive A",
        )
        second_projection = _inspect_archive(
            second,
            expected_build_context_tree_sha256=expected_build_context_tree_sha256,
            expected_source_epoch=expected_source_date_epoch,
            expected_uv_lock_sha256=expected_uv_lock_sha256,
            expected_opa_policy_sha256=expected_opa_policy_sha256,
            image_role="scientific",
            label="archive B",
        )
        if first_projection.executable != second_projection.executable:
            raise C0ReproducibilityError(
                "C0 executable projections differ; no comparison receipt was written"
            )
        receipt = C0ReproducibilityReceipt(
            expected_build_context_tree_sha256=expected_build_context_tree_sha256,
            expected_source_date_epoch=expected_source_date_epoch,
            expected_uv_lock_sha256=expected_uv_lock_sha256,
            expected_opa_policy_sha256=expected_opa_policy_sha256,
            archive_a=first_projection,
            archive_b=second_projection,
            executable_equal=True,
            outer_index_equal=(
                first_projection.outer_index_sha256 == second_projection.outer_index_sha256
            ),
            attestation_metadata_equal=(
                first_projection.attestations == second_projection.attestations
            ),
        )
        _write_exclusive(output, receipt.canonical_file_bytes())
        return receipt
    finally:
        first.close()
        if second is not None:
            second.close()


def compare_tle_release_oci_archives(
    *,
    archive_a_path: str | Path,
    archive_b_path: str | Path,
    expected_build_context_tree_sha256: str,
    expected_source_date_epoch: int,
    expected_uv_lock_sha256: str,
    expected_opa_policy_sha256: str,
    output_path: str | Path,
) -> TleReleaseReproducibilityReceipt:
    """Validate and compare two complete ARM64 timelock-release image closures."""

    _sha256_hex(
        expected_build_context_tree_sha256,
        label="expected build-context tree SHA-256",
    )
    _positive_int(expected_source_date_epoch, label="expected source-date epoch")
    _sha256_hex(expected_uv_lock_sha256, label="expected uv.lock SHA-256")
    _sha256_hex(expected_opa_policy_sha256, label="expected OPA policy SHA-256")
    output = _prepare_output(output_path)
    first = _open_archive(archive_a_path, label="release archive A")
    second: _ArchiveHandle | None = None
    try:
        second = _open_archive(archive_b_path, label="release archive B")
        if (first.stat_result.st_dev, first.stat_result.st_ino) == (
            second.stat_result.st_dev,
            second.stat_result.st_ino,
        ):
            raise C0ReproducibilityError(
                "release archive A and release archive B must be distinct files"
            )
        first_projection = _inspect_archive(
            first,
            expected_build_context_tree_sha256=expected_build_context_tree_sha256,
            expected_source_epoch=expected_source_date_epoch,
            expected_uv_lock_sha256=expected_uv_lock_sha256,
            expected_opa_policy_sha256=expected_opa_policy_sha256,
            image_role="timelock-release",
            label="release archive A",
        )
        second_projection = _inspect_archive(
            second,
            expected_build_context_tree_sha256=expected_build_context_tree_sha256,
            expected_source_epoch=expected_source_date_epoch,
            expected_uv_lock_sha256=expected_uv_lock_sha256,
            expected_opa_policy_sha256=expected_opa_policy_sha256,
            image_role="timelock-release",
            label="release archive B",
        )
        if first_projection.executable != second_projection.executable:
            raise C0ReproducibilityError(
                "timelock-release manifest, config, or layer closure differs"
            )
        runtime_files = {row.image_path: row for row in first_projection.executable.runtime_files}
        tle_file = runtime_files.get("/usr/local/bin/tle")
        receipt_file = runtime_files.get("/opt/artifacts/tle-build/tle-build-receipt.json")
        if tle_file is None or receipt_file is None:
            raise C0ReproducibilityError("timelock-release projection lacks retained TLE files")
        receipt = TleReleaseReproducibilityReceipt(
            expected_build_context_tree_sha256=expected_build_context_tree_sha256,
            expected_source_date_epoch=expected_source_date_epoch,
            expected_uv_lock_sha256=expected_uv_lock_sha256,
            expected_opa_policy_sha256=expected_opa_policy_sha256,
            archive_a=first_projection,
            archive_b=second_projection,
            image_closure_equal=True,
            tle_binary_sha256=tle_file.sha256,
            tle_binary_byte_count=tle_file.byte_count,
            tle_build_receipt_sha256=receipt_file.sha256,
        )
        _write_exclusive(output, receipt.canonical_file_bytes())
        return receipt
    finally:
        first.close()
        if second is not None:
            second.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-c0-reproducibility",
        description="compare the linux/arm64 executable projection of two retained C0 OCI archives",
    )
    parser.add_argument("--archive-a", required=True, type=Path)
    parser.add_argument("--archive-b", required=True, type=Path)
    parser.add_argument("--expected-build-context-tree-sha256", required=True)
    parser.add_argument("--expected-source-date-epoch", required=True, type=int)
    parser.add_argument("--expected-uv-lock-sha256", required=True)
    parser.add_argument("--expected-opa-policy-sha256", required=True)
    parser.add_argument(
        "--image-role",
        choices=("scientific", "timelock-release"),
        default="scientific",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        comparator = (
            compare_c0_oci_archives
            if arguments.image_role == "scientific"
            else compare_tle_release_oci_archives
        )
        receipt = comparator(
            archive_a_path=arguments.archive_a,
            archive_b_path=arguments.archive_b,
            expected_build_context_tree_sha256=(arguments.expected_build_context_tree_sha256),
            expected_source_date_epoch=arguments.expected_source_date_epoch,
            expected_uv_lock_sha256=arguments.expected_uv_lock_sha256,
            expected_opa_policy_sha256=arguments.expected_opa_policy_sha256,
            output_path=arguments.output,
        )
    except (C0ReproducibilityError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(receipt.canonical_file_bytes().decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
