"""Materialize the C1-pinned OPA executable from the exact C0 OCI image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_regular_file,
    read_secure_regular_file,
)
from .runtime_attestation import (
    RUNTIME_ATTESTATION_MANIFEST_TOKEN,
    RuntimeAttestationError,
    RuntimeAttestationPlan,
    loads_runtime_attestation_plan,
    runtime_attestation_plan_template_file_bytes,
)
from .study import FIXED_CORPORA

OPA_RUNTIME_BINARY_PATH = "/usr/local/bin/opa"
PYTHON_RUNTIME_BINARY_PATH = "/opt/venv/bin/python"
UV_LOCK_RUNTIME_PATH = "/opt/app/uv.lock"
HNSWLIB_RUNTIME_RECEIPT_IMAGE_PATH = "/opt/artifacts/hnswlib-runtime-receipt.json"
NATIVE_BUILD_RECEIPT_IMAGE_PATH = "/opt/artifacts/native-build/native-build-receipt.json"
OPA_BUILD_RECEIPT_IMAGE_PATH = "/opt/artifacts/opa-build/opa-build-receipt.json"
RUNTIME_LIBRARY_MANIFEST_IMAGE_PATH = "/opt/artifacts/runtime-library-manifest.json"
SQLITE_RUNTIME_LIBRARY_IMAGE_PATH = "/opt/native-libs/libsqlite3.so.0"
ZLIB_RUNTIME_LIBRARY_IMAGE_PATH = "/opt/native-libs/libz.so.1"
OPA_RUNTIME_MOUNT_ROLE = "opa-runtime-binary"
OPA_RUNTIME_MATERIALIZATION_SCHEMA = "fractal-opa-runtime-materialization-v1"
C0_RUNTIME_EXTRACTION_SCHEMA = "fractal-c0-runtime-extraction-v3"
C0_ARTIFACT_CHECKSUMS_FILENAME = "C0-ARTIFACT-SHA256SUMS"
C0_ARTIFACT_ATTESTATION_BUNDLE_FILENAME = "c0-artifact-attestation-bundle.json"
RUNTIME_PLAN_TEMPLATE_FILENAME = "runtime-attestation-plan.template.json"

_OCI_DIGEST = re.compile(r"^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_WHEEL_BASENAME = re.compile(r"^hnswlib-0\.8\.0-[A-Za-z0-9_.+-]+\.whl$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{12,64}$")
_MAX_OPA_BYTES = 256 * 1024 * 1024
_MAX_PYTHON_BYTES = 256 * 1024 * 1024
_MAX_UV_LOCK_BYTES = 16 * 1024 * 1024
_MAX_HNSW_WHEEL_BYTES = 512 * 1024 * 1024
_MAX_NATIVE_LIBRARY_BYTES = 512 * 1024 * 1024
_MAX_RUNTIME_METADATA_BYTES = 4 * 1024 * 1024
_MAX_RECEIPT_BYTES = 64 * 1024
_MAX_CHECKSUM_BYTES = 4 * 1024 * 1024
_MAX_ATTESTATION_BUNDLE_BYTES = 32 * 1024 * 1024
_MAX_GH_OUTPUT_BYTES = 4 * 1024 * 1024
_TEMPLATE_TOKEN = f'"{RUNTIME_ATTESTATION_MANIFEST_TOKEN}"'.encode("ascii")
_ZERO_MANIFEST = b'"' + (b"0" * 64) + b'"'
_C0_REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
_C0_REF = "refs/tags/confirmatory-apparatus-c0"
_C0_WORKFLOW_PATH = ".github/workflows/confirmatory-image.yml"
_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
_PLATFORMS = {
    "aarch64": "linux/arm64",
    "arm64": "linux/arm64",
    "amd64": "linux/amd64",
    "x86_64": "linux/amd64",
}
_C0_RUNTIME_EXTRACTION_FIELDS = frozenset(
    {
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
        "native_build_receipt_image_path",
        "native_build_receipt_sha256",
        "opa_build_receipt_image_path",
        "opa_build_receipt_sha256",
        "opa_byte_count",
        "opa_image_path",
        "opa_sha256",
        "platform",
        "python_binary_byte_count",
        "python_binary_image_path",
        "python_binary_sha256",
        "runtime_library_manifest_image_path",
        "runtime_library_manifest_sha256",
        "schema_version",
        "source_date_epoch",
        "sqlite_library_byte_count",
        "sqlite_library_image_path",
        "sqlite_library_sha256",
        "uv_lock_byte_count",
        "uv_lock_image_path",
        "uv_lock_sha256",
        "zlib_library_byte_count",
        "zlib_library_image_path",
        "zlib_library_sha256",
    }
)


class OpaRuntimeBinaryError(ValueError):
    """The extracted executable or its five-plan binding is invalid."""


class OpaImageExtractor(Protocol):
    def extract(self, *, image: str, platform: str) -> bytes: ...


class C0ArtifactAttestationVerifier(Protocol):
    def verify(
        self,
        *,
        subject_path: Path,
        bundle_path: Path,
        c0_commit: str,
    ) -> bytes: ...


@dataclass(frozen=True)
class C0RuntimeExtractionReceipt:
    """Closed receipt for one platform's files copied from the C0 image."""

    c0_sha: str
    hnswlib_receipt_image_path: str
    hnswlib_receipt_sha256: str
    hnswlib_wheel_basename: str
    hnswlib_wheel_byte_count: int
    hnswlib_wheel_image_path: str
    hnswlib_wheel_sha256: str
    image_digest: str
    image_manifest_digest: str
    image_reference: str
    native_build_receipt_image_path: str
    native_build_receipt_sha256: str
    opa_build_receipt_image_path: str
    opa_build_receipt_sha256: str
    opa_byte_count: int
    opa_image_path: str
    opa_sha256: str
    platform: str
    python_binary_byte_count: int
    python_binary_image_path: str
    python_binary_sha256: str
    runtime_library_manifest_image_path: str
    runtime_library_manifest_sha256: str
    source_date_epoch: int
    sqlite_library_byte_count: int
    sqlite_library_image_path: str
    sqlite_library_sha256: str
    uv_lock_byte_count: int
    uv_lock_image_path: str
    uv_lock_sha256: str
    zlib_library_byte_count: int
    zlib_library_image_path: str
    zlib_library_sha256: str
    schema_version: str = C0_RUNTIME_EXTRACTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != C0_RUNTIME_EXTRACTION_SCHEMA:
            raise OpaRuntimeBinaryError("C0 extraction receipt has another schema version")
        if not isinstance(self.c0_sha, str) or _GIT_COMMIT.fullmatch(self.c0_sha) is None:
            raise OpaRuntimeBinaryError("C0 extraction receipt has a noncanonical C0 commit")
        if (
            not isinstance(self.image_digest, str)
            or _IMAGE_DIGEST.fullmatch(self.image_digest) is None
        ):
            raise OpaRuntimeBinaryError("C0 extraction receipt has an invalid OCI index digest")
        if (
            not isinstance(self.image_manifest_digest, str)
            or _IMAGE_DIGEST.fullmatch(self.image_manifest_digest) is None
        ):
            raise OpaRuntimeBinaryError(
                "C0 extraction receipt has an invalid platform manifest digest"
            )
        if _image(self.image_reference).rsplit("@", 1)[1] != self.image_digest:
            raise OpaRuntimeBinaryError(
                "C0 extraction receipt image reference and OCI index digest disagree"
            )
        if self.platform not in set(_PLATFORMS.values()):
            raise OpaRuntimeBinaryError("C0 extraction receipt platform is not admitted")
        if self.native_build_receipt_image_path != NATIVE_BUILD_RECEIPT_IMAGE_PATH:
            raise OpaRuntimeBinaryError(
                "C0 extraction receipt names another native-build receipt path"
            )
        if (
            not isinstance(self.native_build_receipt_sha256, str)
            or _SHA256.fullmatch(self.native_build_receipt_sha256) is None
        ):
            raise OpaRuntimeBinaryError(
                "C0 extraction receipt has an invalid native-build receipt digest"
            )
        if self.opa_build_receipt_image_path != OPA_BUILD_RECEIPT_IMAGE_PATH:
            raise OpaRuntimeBinaryError(
                "C0 extraction receipt names another OPA-build receipt path"
            )
        if (
            not isinstance(self.opa_build_receipt_sha256, str)
            or _SHA256.fullmatch(self.opa_build_receipt_sha256) is None
        ):
            raise OpaRuntimeBinaryError(
                "C0 extraction receipt has an invalid OPA-build receipt digest"
            )
        if self.opa_image_path != OPA_RUNTIME_BINARY_PATH:
            raise OpaRuntimeBinaryError("C0 extraction receipt names another OPA image path")
        if not isinstance(self.opa_sha256, str) or _SHA256.fullmatch(self.opa_sha256) is None:
            raise OpaRuntimeBinaryError("C0 extraction receipt has an invalid OPA digest")
        if (
            type(self.opa_byte_count) is not int
            or self.opa_byte_count <= 0
            or self.opa_byte_count > _MAX_OPA_BYTES
        ):
            raise OpaRuntimeBinaryError("C0 extraction receipt OPA byte count is invalid")
        if self.python_binary_image_path != PYTHON_RUNTIME_BINARY_PATH:
            raise OpaRuntimeBinaryError("C0 extraction receipt names another Python image path")
        if (
            not isinstance(self.python_binary_sha256, str)
            or _SHA256.fullmatch(self.python_binary_sha256) is None
        ):
            raise OpaRuntimeBinaryError("C0 extraction receipt has an invalid Python digest")
        if (
            type(self.python_binary_byte_count) is not int
            or self.python_binary_byte_count <= 0
            or self.python_binary_byte_count > _MAX_PYTHON_BYTES
        ):
            raise OpaRuntimeBinaryError("C0 extraction receipt Python byte count is invalid")
        if self.runtime_library_manifest_image_path != RUNTIME_LIBRARY_MANIFEST_IMAGE_PATH:
            raise OpaRuntimeBinaryError(
                "C0 extraction receipt names another runtime-library manifest path"
            )
        if (
            not isinstance(self.runtime_library_manifest_sha256, str)
            or _SHA256.fullmatch(self.runtime_library_manifest_sha256) is None
        ):
            raise OpaRuntimeBinaryError(
                "C0 extraction receipt has an invalid runtime-library manifest digest"
            )
        self._validate_native_library(
            label="SQLite",
            observed_path=self.sqlite_library_image_path,
            expected_path=SQLITE_RUNTIME_LIBRARY_IMAGE_PATH,
            observed_sha256=self.sqlite_library_sha256,
            observed_byte_count=self.sqlite_library_byte_count,
        )
        if self.uv_lock_image_path != UV_LOCK_RUNTIME_PATH:
            raise OpaRuntimeBinaryError("C0 extraction receipt names another uv lock image path")
        if (
            not isinstance(self.uv_lock_sha256, str)
            or _SHA256.fullmatch(self.uv_lock_sha256) is None
        ):
            raise OpaRuntimeBinaryError("C0 extraction receipt has an invalid uv lock digest")
        if (
            type(self.uv_lock_byte_count) is not int
            or self.uv_lock_byte_count <= 0
            or self.uv_lock_byte_count > _MAX_UV_LOCK_BYTES
        ):
            raise OpaRuntimeBinaryError("C0 extraction receipt uv lock byte count is invalid")
        if self.hnswlib_receipt_image_path != HNSWLIB_RUNTIME_RECEIPT_IMAGE_PATH:
            raise OpaRuntimeBinaryError("C0 extraction receipt names another hnsw receipt path")
        if (
            not isinstance(self.hnswlib_receipt_sha256, str)
            or _SHA256.fullmatch(self.hnswlib_receipt_sha256) is None
        ):
            raise OpaRuntimeBinaryError("C0 extraction receipt has an invalid hnsw receipt digest")
        if (
            not isinstance(self.hnswlib_wheel_basename, str)
            or _WHEEL_BASENAME.fullmatch(self.hnswlib_wheel_basename) is None
        ):
            raise OpaRuntimeBinaryError("C0 extraction receipt has an invalid hnsw wheel name")
        expected_wheel_path = f"/opt/artifacts/hnswlib/{self.hnswlib_wheel_basename}"
        if self.hnswlib_wheel_image_path != expected_wheel_path:
            raise OpaRuntimeBinaryError("C0 extraction receipt names another hnsw wheel path")
        if (
            not isinstance(self.hnswlib_wheel_sha256, str)
            or _SHA256.fullmatch(self.hnswlib_wheel_sha256) is None
        ):
            raise OpaRuntimeBinaryError("C0 extraction receipt has an invalid hnsw wheel digest")
        if (
            type(self.hnswlib_wheel_byte_count) is not int
            or self.hnswlib_wheel_byte_count <= 0
            or self.hnswlib_wheel_byte_count > _MAX_HNSW_WHEEL_BYTES
        ):
            raise OpaRuntimeBinaryError("C0 extraction receipt hnsw wheel byte count is invalid")
        self._validate_native_library(
            label="zlib",
            observed_path=self.zlib_library_image_path,
            expected_path=ZLIB_RUNTIME_LIBRARY_IMAGE_PATH,
            observed_sha256=self.zlib_library_sha256,
            observed_byte_count=self.zlib_library_byte_count,
        )
        if type(self.source_date_epoch) is not int or self.source_date_epoch <= 0:
            raise OpaRuntimeBinaryError("C0 extraction receipt source epoch is invalid")

    @staticmethod
    def _validate_native_library(
        *,
        label: str,
        observed_path: object,
        expected_path: str,
        observed_sha256: object,
        observed_byte_count: object,
    ) -> None:
        if observed_path != expected_path:
            raise OpaRuntimeBinaryError(f"C0 extraction receipt names another {label} library path")
        if not isinstance(observed_sha256, str) or _SHA256.fullmatch(observed_sha256) is None:
            raise OpaRuntimeBinaryError(
                f"C0 extraction receipt has an invalid {label} library digest"
            )
        if (
            type(observed_byte_count) is not int
            or observed_byte_count <= 0
            or observed_byte_count > _MAX_NATIVE_LIBRARY_BYTES
        ):
            raise OpaRuntimeBinaryError(
                f"C0 extraction receipt {label} library byte count is invalid"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "c0_sha": self.c0_sha,
            "hnswlib_receipt_image_path": self.hnswlib_receipt_image_path,
            "hnswlib_receipt_sha256": self.hnswlib_receipt_sha256,
            "hnswlib_wheel_basename": self.hnswlib_wheel_basename,
            "hnswlib_wheel_byte_count": self.hnswlib_wheel_byte_count,
            "hnswlib_wheel_image_path": self.hnswlib_wheel_image_path,
            "hnswlib_wheel_sha256": self.hnswlib_wheel_sha256,
            "image_digest": self.image_digest,
            "image_manifest_digest": self.image_manifest_digest,
            "image_reference": self.image_reference,
            "native_build_receipt_image_path": self.native_build_receipt_image_path,
            "native_build_receipt_sha256": self.native_build_receipt_sha256,
            "opa_build_receipt_image_path": self.opa_build_receipt_image_path,
            "opa_build_receipt_sha256": self.opa_build_receipt_sha256,
            "opa_byte_count": self.opa_byte_count,
            "opa_image_path": self.opa_image_path,
            "opa_sha256": self.opa_sha256,
            "platform": self.platform,
            "python_binary_byte_count": self.python_binary_byte_count,
            "python_binary_image_path": self.python_binary_image_path,
            "python_binary_sha256": self.python_binary_sha256,
            "runtime_library_manifest_image_path": (self.runtime_library_manifest_image_path),
            "runtime_library_manifest_sha256": self.runtime_library_manifest_sha256,
            "schema_version": self.schema_version,
            "source_date_epoch": self.source_date_epoch,
            "sqlite_library_byte_count": self.sqlite_library_byte_count,
            "sqlite_library_image_path": self.sqlite_library_image_path,
            "sqlite_library_sha256": self.sqlite_library_sha256,
            "uv_lock_byte_count": self.uv_lock_byte_count,
            "uv_lock_image_path": self.uv_lock_image_path,
            "uv_lock_sha256": self.uv_lock_sha256,
            "zlib_library_byte_count": self.zlib_library_byte_count,
            "zlib_library_image_path": self.zlib_library_image_path,
            "zlib_library_sha256": self.zlib_library_sha256,
        }


@dataclass(frozen=True)
class RetainedOpaMaterialization:
    """Verified link from retained C0 custody bytes to the C1 mount file."""

    extraction_receipt_sha256: str
    runtime_binding: OpaRuntimeBinaryVerification
    selected_manifest_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "extraction_receipt_sha256": self.extraction_receipt_sha256,
            "runtime_binding": self.runtime_binding.to_dict(),
            "selected_manifest_digest": self.selected_manifest_digest,
        }


@dataclass(frozen=True)
class OpaRuntimeBinaryVerification:
    """Typed evidence that one executable is shared by C0 and all five plans."""

    architecture: str
    binary_sha256: str
    byte_count: int
    code_commit: str
    image: str
    plan_template_sha256_by_corpus: tuple[tuple[str, str], ...]
    platform: str
    schema_version: str = OPA_RUNTIME_MATERIALIZATION_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "binary_sha256": self.binary_sha256,
            "byte_count": self.byte_count,
            "code_commit": self.code_commit,
            "image": self.image,
            "plan_template_sha256_by_corpus": {
                corpus_id: digest for corpus_id, digest in self.plan_template_sha256_by_corpus
            },
            "platform": self.platform,
            "schema_version": self.schema_version,
        }


def _image(value: str) -> str:
    if not isinstance(value, str) or _OCI_DIGEST.fullmatch(value) is None:
        raise OpaRuntimeBinaryError("C0 image must be one lowercase digest-qualified OCI reference")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_json_object(encoded: bytes, *, label: str) -> Mapping[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, value in pairs:
            if key in parsed:
                raise OpaRuntimeBinaryError(f"{label} contains duplicate key {key!r}")
            parsed[key] = value
        return parsed

    def reject_nonfinite(value: str) -> None:
        raise OpaRuntimeBinaryError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpaRuntimeBinaryError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise OpaRuntimeBinaryError(f"{label} must contain one JSON object")
    return value


def load_c0_runtime_extraction_receipt(
    path: str | Path,
) -> C0RuntimeExtractionReceipt:
    """Load one canonical, closed C0 platform-extraction receipt."""

    target = Path(path).expanduser()
    if not target.is_absolute():
        raise OpaRuntimeBinaryError("C0 extraction receipt path must be absolute")
    try:
        encoded = read_secure_regular_file(
            target,
            max_bytes=_MAX_RECEIPT_BYTES,
            label="C0 runtime extraction receipt",
        )
    except ArtifactIntegrityError as exc:
        raise OpaRuntimeBinaryError(f"cannot read C0 runtime extraction receipt: {exc}") from exc
    payload = _strict_json_object(encoded, label="C0 runtime extraction receipt")
    observed = set(payload)
    if observed != _C0_RUNTIME_EXTRACTION_FIELDS:
        raise OpaRuntimeBinaryError(
            "C0 runtime extraction receipt schema mismatch; "
            f"missing={sorted(_C0_RUNTIME_EXTRACTION_FIELDS - observed)}, "
            f"unknown={sorted(observed - _C0_RUNTIME_EXTRACTION_FIELDS)}"
        )
    receipt = C0RuntimeExtractionReceipt(
        c0_sha=payload["c0_sha"],  # type: ignore[arg-type]
        hnswlib_receipt_image_path=payload["hnswlib_receipt_image_path"],  # type: ignore[arg-type]
        hnswlib_receipt_sha256=payload["hnswlib_receipt_sha256"],  # type: ignore[arg-type]
        hnswlib_wheel_basename=payload["hnswlib_wheel_basename"],  # type: ignore[arg-type]
        hnswlib_wheel_byte_count=payload["hnswlib_wheel_byte_count"],  # type: ignore[arg-type]
        hnswlib_wheel_image_path=payload["hnswlib_wheel_image_path"],  # type: ignore[arg-type]
        hnswlib_wheel_sha256=payload["hnswlib_wheel_sha256"],  # type: ignore[arg-type]
        image_digest=payload["image_digest"],  # type: ignore[arg-type]
        image_manifest_digest=payload["image_manifest_digest"],  # type: ignore[arg-type]
        image_reference=payload["image_reference"],  # type: ignore[arg-type]
        native_build_receipt_image_path=payload[  # type: ignore[arg-type]
            "native_build_receipt_image_path"
        ],
        native_build_receipt_sha256=payload[  # type: ignore[arg-type]
            "native_build_receipt_sha256"
        ],
        opa_build_receipt_image_path=payload[  # type: ignore[arg-type]
            "opa_build_receipt_image_path"
        ],
        opa_build_receipt_sha256=payload["opa_build_receipt_sha256"],  # type: ignore[arg-type]
        opa_byte_count=payload["opa_byte_count"],  # type: ignore[arg-type]
        opa_image_path=payload["opa_image_path"],  # type: ignore[arg-type]
        opa_sha256=payload["opa_sha256"],  # type: ignore[arg-type]
        platform=payload["platform"],  # type: ignore[arg-type]
        python_binary_byte_count=payload["python_binary_byte_count"],  # type: ignore[arg-type]
        python_binary_image_path=payload["python_binary_image_path"],  # type: ignore[arg-type]
        python_binary_sha256=payload["python_binary_sha256"],  # type: ignore[arg-type]
        runtime_library_manifest_image_path=payload[  # type: ignore[arg-type]
            "runtime_library_manifest_image_path"
        ],
        runtime_library_manifest_sha256=payload[  # type: ignore[arg-type]
            "runtime_library_manifest_sha256"
        ],
        schema_version=payload["schema_version"],  # type: ignore[arg-type]
        source_date_epoch=payload["source_date_epoch"],  # type: ignore[arg-type]
        sqlite_library_byte_count=payload["sqlite_library_byte_count"],  # type: ignore[arg-type]
        sqlite_library_image_path=payload["sqlite_library_image_path"],  # type: ignore[arg-type]
        sqlite_library_sha256=payload["sqlite_library_sha256"],  # type: ignore[arg-type]
        uv_lock_byte_count=payload["uv_lock_byte_count"],  # type: ignore[arg-type]
        uv_lock_image_path=payload["uv_lock_image_path"],  # type: ignore[arg-type]
        uv_lock_sha256=payload["uv_lock_sha256"],  # type: ignore[arg-type]
        zlib_library_byte_count=payload["zlib_library_byte_count"],  # type: ignore[arg-type]
        zlib_library_image_path=payload["zlib_library_image_path"],  # type: ignore[arg-type]
        zlib_library_sha256=payload["zlib_library_sha256"],  # type: ignore[arg-type]
    )
    if encoded != _canonical_bytes(receipt.to_dict()) + b"\n":
        raise OpaRuntimeBinaryError("C0 runtime extraction receipt is not canonical JSON")
    return receipt


def _parse_c0_artifact_checksums(encoded: bytes) -> Mapping[str, str]:
    try:
        text = encoded.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise OpaRuntimeBinaryError("C0 artifact checksums must be ASCII") from exc
    if not text or not text.endswith("\n") or "\r" in text:
        raise OpaRuntimeBinaryError("C0 artifact checksums must end in one LF per row")
    checksums: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00-\x1f\x7f]+)", line)
        if match is None:
            raise OpaRuntimeBinaryError("C0 artifact checksums contain a malformed row")
        digest, relative_path = match.groups()
        if (
            relative_path.startswith("/")
            or "\\" in relative_path
            or unicodedata.normalize("NFC", relative_path) != relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise OpaRuntimeBinaryError("C0 artifact checksums contain a noncanonical path")
        if relative_path in checksums:
            raise OpaRuntimeBinaryError("C0 artifact checksums contain a duplicate path")
        checksums[relative_path] = digest
    if list(checksums) != sorted(checksums, key=lambda value: value.encode("utf-8")):
        raise OpaRuntimeBinaryError("C0 artifact checksum rows are not bytewise sorted")
    return checksums


@dataclass(frozen=True)
class GhC0ArtifactAttestationVerifier:
    """Verify one retained C0 file against the workflow's Sigstore bundle."""

    executable: str = "gh"
    timeout_seconds: int = 60

    def verify(
        self,
        *,
        subject_path: Path,
        bundle_path: Path,
        c0_commit: str,
    ) -> bytes:
        if not isinstance(c0_commit, str) or _GIT_COMMIT.fullmatch(c0_commit) is None:
            raise OpaRuntimeBinaryError("C0 attestation signer digest is invalid")
        identity = f"https://github.com/{_C0_REPOSITORY}/{_C0_WORKFLOW_PATH}@{_C0_REF}"
        command = [
            self.executable,
            "attestation",
            "verify",
            str(subject_path),
            "--bundle",
            str(bundle_path),
            "--hostname",
            "github.com",
            "--repo",
            _C0_REPOSITORY,
            "--cert-identity",
            identity,
            "--cert-oidc-issuer",
            _OIDC_ISSUER,
            "--signer-digest",
            c0_commit,
            "--source-digest",
            c0_commit,
            "--source-ref",
            _C0_REF,
            "--deny-self-hosted-runners",
            "--format",
            "json",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                env={**os.environ, "GH_PROMPT_DISABLED": "1", "NO_COLOR": "1"},
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OpaRuntimeBinaryError("cannot execute C0 gh attestation verification") from exc
        if result.returncode != 0:
            detail = result.stderr[:4096].decode("utf-8", errors="replace").strip()
            raise OpaRuntimeBinaryError(f"GitHub rejected the C0 artifact attestation: {detail}")
        if len(result.stdout) > _MAX_GH_OUTPUT_BYTES:
            raise OpaRuntimeBinaryError("C0 gh attestation output exceeds the limit")
        try:
            verified = json.loads(result.stdout.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpaRuntimeBinaryError("C0 gh attestation output is not valid JSON") from exc
        if (
            not isinstance(verified, list)
            or len(verified) != 1
            or not isinstance(verified[0], Mapping)
            or not isinstance(verified[0].get("verificationResult"), Mapping)
        ):
            raise OpaRuntimeBinaryError("gh must verify exactly one C0 artifact attestation")
        return result.stdout


def load_runtime_attestation_plan_template(path: str | Path) -> RuntimeAttestationPlan:
    """Load one canonical C1 template by substituting only its manifest token."""

    target = Path(path).expanduser()
    if not target.is_absolute():
        raise OpaRuntimeBinaryError("runtime plan template path must be absolute")
    try:
        encoded = read_secure_regular_file(
            target,
            max_bytes=4 * 1024 * 1024,
            label="runtime attestation plan template",
        )
    except ArtifactIntegrityError as exc:
        raise OpaRuntimeBinaryError(f"cannot read runtime plan template: {exc}") from exc
    if encoded.count(_TEMPLATE_TOKEN) != 1:
        raise OpaRuntimeBinaryError("runtime plan template must contain one manifest token")
    try:
        plan = loads_runtime_attestation_plan(encoded.replace(_TEMPLATE_TOKEN, _ZERO_MANIFEST))
    except RuntimeAttestationError as exc:
        raise OpaRuntimeBinaryError(f"runtime plan template is invalid: {exc}") from exc
    if runtime_attestation_plan_template_file_bytes(plan) != encoded:
        raise OpaRuntimeBinaryError("runtime plan template differs outside its manifest token")
    return plan


def plan_template_paths(plan_root: str | Path) -> dict[str, Path]:
    """Derive the exact five controlled plan-template paths."""

    root = Path(plan_root).expanduser()
    if not root.is_absolute():
        raise OpaRuntimeBinaryError("runtime plan root must be absolute")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise OpaRuntimeBinaryError("runtime plan root does not exist") from exc
    if root.is_symlink() or resolved != root or not root.is_dir():
        raise OpaRuntimeBinaryError("runtime plan root must be one absolute real directory")
    return {
        corpus_id: root / corpus_id / RUNTIME_PLAN_TEMPLATE_FILENAME for corpus_id in FIXED_CORPORA
    }


def _load_plan_set(
    plan_paths: Mapping[str, Path],
) -> tuple[tuple[str, RuntimeAttestationPlan, str], ...]:
    if set(plan_paths) != set(FIXED_CORPORA) or len(plan_paths) != len(FIXED_CORPORA):
        raise OpaRuntimeBinaryError("runtime plan set must cover every fixed corpus exactly once")
    rows: list[tuple[str, RuntimeAttestationPlan, str]] = []
    for corpus_id in FIXED_CORPORA:
        path = Path(plan_paths[corpus_id])
        plan = load_runtime_attestation_plan_template(path)
        template_digest = hashlib.sha256(
            runtime_attestation_plan_template_file_bytes(plan)
        ).hexdigest()
        rows.append((corpus_id, plan, template_digest))
    return tuple(rows)


def _verify_plan_contracts(
    rows: Sequence[tuple[str, RuntimeAttestationPlan, str]],
    *,
    image: str,
    binary_sha256: str,
    byte_count: int,
) -> OpaRuntimeBinaryVerification:
    expected_image = _image(image)
    if not re.fullmatch(r"[0-9a-f]{64}", binary_sha256):
        raise OpaRuntimeBinaryError("OPA binary digest must be one lowercase SHA-256")
    if type(byte_count) is not int or byte_count <= 0 or byte_count > _MAX_OPA_BYTES:
        raise OpaRuntimeBinaryError("OPA binary byte count is outside the admitted range")

    architectures: set[str] = set()
    code_commits: set[str] = set()
    for corpus_id, plan, _template_digest in rows:
        if plan.oci_image_digest != expected_image:
            raise OpaRuntimeBinaryError(f"{corpus_id} runtime plan names another C0 OCI image")
        if plan.opa_binary.path != OPA_RUNTIME_BINARY_PATH:
            raise OpaRuntimeBinaryError(f"{corpus_id} runtime plan names another OPA path")
        if plan.opa_binary.sha256 != binary_sha256:
            raise OpaRuntimeBinaryError(
                f"{corpus_id} runtime plan OPA digest differs from the extracted bytes"
            )
        mounts = [mount for mount in plan.mounts if mount.root == OPA_RUNTIME_BINARY_PATH]
        if len(mounts) != 1:
            raise OpaRuntimeBinaryError(f"{corpus_id} runtime plan lacks one exact OPA file mount")
        mount = mounts[0]
        if (
            mount.kind != "file"
            or not mount.read_only
            or mount.role != OPA_RUNTIME_MOUNT_ROLE
            or mount.artifact_sha256 != binary_sha256
        ):
            raise OpaRuntimeBinaryError(
                f"{corpus_id} runtime plan OPA mount differs from the pinned executable"
            )
        architectures.add(plan.architecture)
        code_commits.add(plan.code_commit)
    if len(architectures) != 1 or len(code_commits) != 1:
        raise OpaRuntimeBinaryError("five runtime plans disagree on architecture or C0 commit")
    architecture = architectures.pop()
    try:
        platform = _PLATFORMS[architecture]
    except KeyError as exc:
        raise OpaRuntimeBinaryError(
            f"runtime plan architecture {architecture!r} has no admitted OCI platform"
        ) from exc
    return OpaRuntimeBinaryVerification(
        architecture=architecture,
        binary_sha256=binary_sha256,
        byte_count=byte_count,
        code_commit=code_commits.pop(),
        image=expected_image,
        plan_template_sha256_by_corpus=tuple(
            (corpus_id, template_digest) for corpus_id, _plan, template_digest in rows
        ),
        platform=platform,
    )


def verify_opa_runtime_binary(
    binary_path: str | Path,
    *,
    image: str,
    plan_paths: Mapping[str, Path],
) -> OpaRuntimeBinaryVerification:
    """Verify one materialized executable against the image and all five templates."""

    target = Path(binary_path).expanduser()
    if not target.is_absolute():
        raise OpaRuntimeBinaryError("OPA runtime binary path must be absolute")
    try:
        metadata = target.stat(follow_symlinks=False)
    except OSError as exc:
        raise OpaRuntimeBinaryError("cannot inspect the OPA runtime binary") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or target.is_symlink()
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > _MAX_OPA_BYTES
        or metadata.st_mode & 0o222
        or not metadata.st_mode & 0o111
    ):
        raise OpaRuntimeBinaryError(
            "OPA runtime binary must be one bounded, executable, non-writable regular file"
        )
    try:
        digest = digest_regular_file(target, label="OPA runtime binary")
    except ArtifactIntegrityError as exc:
        raise OpaRuntimeBinaryError(f"cannot hash the OPA runtime binary: {exc}") from exc
    rows = _load_plan_set(plan_paths)
    return _verify_plan_contracts(
        rows,
        image=image,
        binary_sha256=digest,
        byte_count=metadata.st_size,
    )


@dataclass(frozen=True)
class DockerOpaImageExtractor:
    """Read `/usr/local/bin/opa` from a platform-specific exact OCI manifest."""

    executable: str = "docker"
    timeout_seconds: int = 600

    def _run(self, arguments: Sequence[str]) -> bytes:
        try:
            result = subprocess.run(
                [self.executable, *arguments],
                check=False,
                capture_output=True,
                env={**os.environ},
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OpaRuntimeBinaryError("cannot execute the Docker OPA extractor") from exc
        if result.returncode != 0:
            detail = result.stderr[:4096].decode("utf-8", errors="replace").strip()
            raise OpaRuntimeBinaryError(f"Docker OPA extraction failed: {detail}")
        return result.stdout

    def extract(self, *, image: str, platform: str) -> bytes:
        exact_image = _image(image)
        if platform not in set(_PLATFORMS.values()):
            raise OpaRuntimeBinaryError("OCI platform is not admitted for OPA extraction")
        self._run(("pull", "--platform", platform, exact_image))
        inspected_architecture = (
            self._run(("image", "inspect", "--format", "{{.Architecture}}", exact_image))
            .decode("ascii", errors="strict")
            .strip()
        )
        if f"linux/{inspected_architecture}" != platform:
            raise OpaRuntimeBinaryError("local OCI image architecture differs from the plans")
        repo_digests_bytes = self._run(
            ("image", "inspect", "--format", "{{json .RepoDigests}}", exact_image)
        )
        try:
            repo_digests = json.loads(repo_digests_bytes.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpaRuntimeBinaryError("Docker returned malformed RepoDigests") from exc
        expected_digest = exact_image.rsplit("@", 1)[1]
        if (
            not isinstance(repo_digests, list)
            or not repo_digests
            or any(type(value) is not str for value in repo_digests)
            or not any(value.endswith("@" + expected_digest) for value in repo_digests)
        ):
            raise OpaRuntimeBinaryError("local image does not retain the requested OCI digest")

        container_id = ""
        with tempfile.TemporaryDirectory(prefix="fractal-opa-extract-") as temporary:
            target = Path(temporary) / "opa"
            try:
                container_id = (
                    self._run(
                        (
                            "create",
                            "--platform",
                            platform,
                            "--network",
                            "none",
                            "--entrypoint",
                            "/bin/false",
                            exact_image,
                        )
                    )
                    .decode("ascii", errors="strict")
                    .strip()
                )
                if _CONTAINER_ID.fullmatch(container_id) is None:
                    raise OpaRuntimeBinaryError("Docker returned a malformed container ID")
                self._run(("cp", f"{container_id}:{OPA_RUNTIME_BINARY_PATH}", str(target)))
                try:
                    encoded = read_secure_regular_file(
                        target.resolve(strict=True),
                        max_bytes=_MAX_OPA_BYTES,
                        label="image-extracted OPA runtime binary",
                    )
                except (ArtifactIntegrityError, OSError) as exc:
                    raise OpaRuntimeBinaryError(
                        f"cannot read the image-extracted OPA binary: {exc}"
                    ) from exc
            finally:
                if container_id:
                    self._run(("rm", "--force", container_id))
        if not encoded.startswith(b"\x7fELF"):
            raise OpaRuntimeBinaryError("image-extracted OPA payload is not an ELF executable")
        return encoded


def _write_exclusive_executable(path: Path, encoded: bytes) -> None:
    if not path.is_absolute():
        raise OpaRuntimeBinaryError("OPA output path must be absolute")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise OpaRuntimeBinaryError("OPA output parent must already exist") from exc
    if path.parent.is_symlink() or parent != path.parent or not parent.is_dir():
        raise OpaRuntimeBinaryError("OPA output parent must be one absolute real directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o555)
    except OSError as exc:
        raise OpaRuntimeBinaryError("OPA output already exists or cannot be created") from exc
    try:
        os.fchmod(descriptor, 0o555)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.close(descriptor)


def materialize_opa_runtime_binary(
    *,
    image: str,
    plan_paths: Mapping[str, Path],
    output_path: str | Path,
    extractor: OpaImageExtractor | None = None,
) -> OpaRuntimeBinaryVerification:
    """Extract, cross-check, and exclusively publish the C1 OPA mount source."""

    exact_image = _image(image)
    rows = _load_plan_set(plan_paths)
    architectures = {plan.architecture for _corpus, plan, _digest in rows}
    if len(architectures) != 1:
        raise OpaRuntimeBinaryError("five runtime plans disagree on architecture")
    try:
        platform = _PLATFORMS[architectures.pop()]
    except KeyError as exc:
        raise OpaRuntimeBinaryError("runtime architecture has no admitted OCI platform") from exc
    active = extractor if extractor is not None else DockerOpaImageExtractor()
    encoded = active.extract(image=exact_image, platform=platform)
    if not isinstance(encoded, bytes) or not encoded.startswith(b"\x7fELF"):
        raise OpaRuntimeBinaryError("OPA image extractor did not return one ELF byte string")
    verification = _verify_plan_contracts(
        rows,
        image=exact_image,
        binary_sha256=hashlib.sha256(encoded).hexdigest(),
        byte_count=len(encoded),
    )
    output = Path(output_path).expanduser()
    _write_exclusive_executable(output, encoded)
    observed = verify_opa_runtime_binary(
        output,
        image=exact_image,
        plan_paths=plan_paths,
    )
    if observed != verification:
        raise OpaRuntimeBinaryError("materialized OPA bytes changed after exclusive creation")
    return observed


def _c0_package_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        raise OpaRuntimeBinaryError("retained C0 package root must be absolute")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise OpaRuntimeBinaryError("retained C0 package root does not exist") from exc
    if root.is_symlink() or resolved != root or not root.is_dir():
        raise OpaRuntimeBinaryError("retained C0 package root must be one absolute real directory")
    return root


def materialize_retained_opa_runtime_binary(
    *,
    c0_package_root: str | Path,
    image: str,
    plan_paths: Mapping[str, Path],
    output_path: str | Path,
    attestation_verifier: C0ArtifactAttestationVerifier | None = None,
) -> RetainedOpaMaterialization:
    """Copy one attested platform OPA payload from C0 custody into C1."""

    exact_image = _image(image)
    rows = _load_plan_set(plan_paths)
    prospective = _verify_plan_contracts(
        rows,
        image=exact_image,
        binary_sha256=rows[0][1].opa_binary.sha256,
        byte_count=1,
    )
    root = _c0_package_root(c0_package_root)
    slug = prospective.platform.replace("/", "-")
    runtime_relative = f"runtime-artifacts/{slug}"
    opa_relative = f"{runtime_relative}/opa"
    python_relative = f"{runtime_relative}/python"
    receipt_relative = f"{runtime_relative}/runtime-extraction.json"
    uv_lock_relative = f"{runtime_relative}/uv.lock"
    hnswlib_receipt_relative = f"{runtime_relative}/hnswlib-runtime-receipt.json"
    native_build_receipt_relative = f"{runtime_relative}/native-build/native-build-receipt.json"
    opa_build_receipt_relative = f"{runtime_relative}/opa-build/opa-build-receipt.json"
    runtime_library_manifest_relative = f"{runtime_relative}/runtime-library-manifest.json"
    sqlite_library_relative = f"{runtime_relative}/libsqlite3.so.0"
    zlib_library_relative = f"{runtime_relative}/libz.so.1"
    opa_path = root / "runtime-artifacts" / slug / "opa"
    python_path = root / "runtime-artifacts" / slug / "python"
    receipt_path = root / "runtime-artifacts" / slug / "runtime-extraction.json"
    uv_lock_path = root / "runtime-artifacts" / slug / "uv.lock"
    hnswlib_receipt_path = root / "runtime-artifacts" / slug / "hnswlib-runtime-receipt.json"
    native_build_receipt_path = (
        root / "runtime-artifacts" / slug / "native-build" / "native-build-receipt.json"
    )
    opa_build_receipt_path = (
        root / "runtime-artifacts" / slug / "opa-build" / "opa-build-receipt.json"
    )
    runtime_library_manifest_path = (
        root / "runtime-artifacts" / slug / "runtime-library-manifest.json"
    )
    sqlite_library_path = root / "runtime-artifacts" / slug / "libsqlite3.so.0"
    zlib_library_path = root / "runtime-artifacts" / slug / "libz.so.1"
    checksums_path = root / C0_ARTIFACT_CHECKSUMS_FILENAME
    bundle_path = root / C0_ARTIFACT_ATTESTATION_BUNDLE_FILENAME

    receipt = load_c0_runtime_extraction_receipt(receipt_path)
    hnswlib_wheel_relative = f"{runtime_relative}/hnswlib/{receipt.hnswlib_wheel_basename}"
    hnswlib_wheel_path = (
        root / "runtime-artifacts" / slug / "hnswlib" / receipt.hnswlib_wheel_basename
    )
    if receipt.image_reference != exact_image:
        raise OpaRuntimeBinaryError("C0 extraction receipt names another OCI index")
    if receipt.platform != prospective.platform:
        raise OpaRuntimeBinaryError("C0 extraction receipt names another OCI platform")
    if receipt.c0_sha != prospective.code_commit:
        raise OpaRuntimeBinaryError("C0 extraction receipt names another C0 commit")
    try:
        receipt_bytes = read_secure_regular_file(
            receipt_path,
            max_bytes=_MAX_RECEIPT_BYTES,
            label="C0 runtime extraction receipt",
        )
        opa_bytes = read_secure_regular_file(
            opa_path,
            max_bytes=_MAX_OPA_BYTES,
            label="retained C0 OPA binary",
        )
        python_bytes = read_secure_regular_file(
            python_path,
            max_bytes=_MAX_PYTHON_BYTES,
            label="retained C0 Python binary",
        )
        uv_lock_bytes = read_secure_regular_file(
            uv_lock_path,
            max_bytes=_MAX_UV_LOCK_BYTES,
            label="retained C0 uv lock",
        )
        hnswlib_receipt_bytes = read_secure_regular_file(
            hnswlib_receipt_path,
            max_bytes=_MAX_RUNTIME_METADATA_BYTES,
            label="retained C0 hnswlib receipt",
        )
        hnswlib_wheel_bytes = read_secure_regular_file(
            hnswlib_wheel_path,
            max_bytes=_MAX_HNSW_WHEEL_BYTES,
            label="retained C0 hnswlib wheel",
        )
        native_build_receipt_bytes = read_secure_regular_file(
            native_build_receipt_path,
            max_bytes=_MAX_RUNTIME_METADATA_BYTES,
            label="retained C0 native-build receipt",
        )
        opa_build_receipt_bytes = read_secure_regular_file(
            opa_build_receipt_path,
            max_bytes=_MAX_RUNTIME_METADATA_BYTES,
            label="retained C0 OPA-build receipt",
        )
        runtime_library_manifest_bytes = read_secure_regular_file(
            runtime_library_manifest_path,
            max_bytes=_MAX_RUNTIME_METADATA_BYTES,
            label="retained C0 runtime-library manifest",
        )
        sqlite_library_bytes = read_secure_regular_file(
            sqlite_library_path,
            max_bytes=_MAX_NATIVE_LIBRARY_BYTES,
            label="retained C0 SQLite library",
        )
        zlib_library_bytes = read_secure_regular_file(
            zlib_library_path,
            max_bytes=_MAX_NATIVE_LIBRARY_BYTES,
            label="retained C0 zlib library",
        )
        checksum_bytes = read_secure_regular_file(
            checksums_path,
            max_bytes=_MAX_CHECKSUM_BYTES,
            label="C0 artifact checksum manifest",
        )
        read_secure_regular_file(
            bundle_path,
            max_bytes=_MAX_ATTESTATION_BUNDLE_BYTES,
            label="C0 artifact attestation bundle",
        )
    except ArtifactIntegrityError as exc:
        raise OpaRuntimeBinaryError(f"cannot read retained C0 custody package: {exc}") from exc
    if receipt_bytes != _canonical_bytes(receipt.to_dict()) + b"\n":
        raise OpaRuntimeBinaryError("C0 extraction receipt changed after typed admission")
    opa_digest = hashlib.sha256(opa_bytes).hexdigest()
    python_digest = hashlib.sha256(python_bytes).hexdigest()
    receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
    uv_lock_digest = hashlib.sha256(uv_lock_bytes).hexdigest()
    hnswlib_receipt_digest = hashlib.sha256(hnswlib_receipt_bytes).hexdigest()
    hnswlib_wheel_digest = hashlib.sha256(hnswlib_wheel_bytes).hexdigest()
    native_build_receipt_digest = hashlib.sha256(native_build_receipt_bytes).hexdigest()
    opa_build_receipt_digest = hashlib.sha256(opa_build_receipt_bytes).hexdigest()
    runtime_library_manifest_digest = hashlib.sha256(runtime_library_manifest_bytes).hexdigest()
    sqlite_library_digest = hashlib.sha256(sqlite_library_bytes).hexdigest()
    zlib_library_digest = hashlib.sha256(zlib_library_bytes).hexdigest()
    if (
        not opa_bytes.startswith(b"\x7fELF")
        or opa_digest != receipt.opa_sha256
        or len(opa_bytes) != receipt.opa_byte_count
    ):
        raise OpaRuntimeBinaryError("retained OPA bytes differ from the C0 extraction receipt")
    if (
        not python_bytes.startswith(b"\x7fELF")
        or python_digest != receipt.python_binary_sha256
        or len(python_bytes) != receipt.python_binary_byte_count
    ):
        raise OpaRuntimeBinaryError("retained Python bytes differ from the C0 extraction receipt")
    if uv_lock_digest != receipt.uv_lock_sha256 or len(uv_lock_bytes) != receipt.uv_lock_byte_count:
        raise OpaRuntimeBinaryError("retained uv lock differs from the C0 extraction receipt")
    if hnswlib_receipt_digest != receipt.hnswlib_receipt_sha256:
        raise OpaRuntimeBinaryError(
            "retained hnswlib receipt differs from the C0 extraction receipt"
        )
    if (
        hnswlib_wheel_digest != receipt.hnswlib_wheel_sha256
        or len(hnswlib_wheel_bytes) != receipt.hnswlib_wheel_byte_count
    ):
        raise OpaRuntimeBinaryError("retained hnswlib wheel differs from the C0 extraction receipt")
    if native_build_receipt_digest != receipt.native_build_receipt_sha256:
        raise OpaRuntimeBinaryError(
            "retained native-build receipt differs from the C0 extraction receipt"
        )
    if opa_build_receipt_digest != receipt.opa_build_receipt_sha256:
        raise OpaRuntimeBinaryError(
            "retained OPA-build receipt differs from the C0 extraction receipt"
        )
    if runtime_library_manifest_digest != receipt.runtime_library_manifest_sha256:
        raise OpaRuntimeBinaryError(
            "retained runtime-library manifest differs from the C0 extraction receipt"
        )
    if (
        sqlite_library_digest != receipt.sqlite_library_sha256
        or len(sqlite_library_bytes) != receipt.sqlite_library_byte_count
    ):
        raise OpaRuntimeBinaryError(
            "retained SQLite library differs from the C0 extraction receipt"
        )
    if (
        zlib_library_digest != receipt.zlib_library_sha256
        or len(zlib_library_bytes) != receipt.zlib_library_byte_count
    ):
        raise OpaRuntimeBinaryError("retained zlib library differs from the C0 extraction receipt")
    checksums = _parse_c0_artifact_checksums(checksum_bytes)
    if checksums.get(opa_relative) != opa_digest:
        raise OpaRuntimeBinaryError("C0 artifact checksums do not bind the selected OPA bytes")
    if checksums.get(receipt_relative) != receipt_digest:
        raise OpaRuntimeBinaryError(
            "C0 artifact checksums do not bind the selected extraction receipt"
        )
    if checksums.get(python_relative) != python_digest:
        raise OpaRuntimeBinaryError("C0 artifact checksums do not bind the selected Python bytes")
    if checksums.get(uv_lock_relative) != uv_lock_digest:
        raise OpaRuntimeBinaryError("C0 artifact checksums do not bind the selected uv lock")
    supplemental_checksums = {
        hnswlib_receipt_relative: hnswlib_receipt_digest,
        hnswlib_wheel_relative: hnswlib_wheel_digest,
        native_build_receipt_relative: native_build_receipt_digest,
        opa_build_receipt_relative: opa_build_receipt_digest,
        runtime_library_manifest_relative: runtime_library_manifest_digest,
        sqlite_library_relative: sqlite_library_digest,
        zlib_library_relative: zlib_library_digest,
    }
    for relative_path, expected_digest in supplemental_checksums.items():
        if checksums.get(relative_path) != expected_digest:
            raise OpaRuntimeBinaryError(
                "C0 artifact checksums do not bind the selected "
                f"{relative_path.rsplit('/', 1)[-1]} bytes"
            )

    verification = _verify_plan_contracts(
        rows,
        image=exact_image,
        binary_sha256=opa_digest,
        byte_count=len(opa_bytes),
    )
    active = (
        attestation_verifier
        if attestation_verifier is not None
        else GhC0ArtifactAttestationVerifier()
    )
    for subject_path in (
        receipt_path,
        opa_path,
        python_path,
        uv_lock_path,
        hnswlib_receipt_path,
        hnswlib_wheel_path,
        native_build_receipt_path,
        opa_build_receipt_path,
        runtime_library_manifest_path,
        sqlite_library_path,
        zlib_library_path,
    ):
        active.verify(
            subject_path=subject_path,
            bundle_path=bundle_path,
            c0_commit=receipt.c0_sha,
        )
    try:
        if (
            read_secure_regular_file(
                receipt_path,
                max_bytes=_MAX_RECEIPT_BYTES,
                label="C0 runtime extraction receipt",
            )
            != receipt_bytes
            or read_secure_regular_file(
                opa_path,
                max_bytes=_MAX_OPA_BYTES,
                label="retained C0 OPA binary",
            )
            != opa_bytes
            or read_secure_regular_file(
                python_path,
                max_bytes=_MAX_PYTHON_BYTES,
                label="retained C0 Python binary",
            )
            != python_bytes
            or read_secure_regular_file(
                uv_lock_path,
                max_bytes=_MAX_UV_LOCK_BYTES,
                label="retained C0 uv lock",
            )
            != uv_lock_bytes
            or read_secure_regular_file(
                hnswlib_receipt_path,
                max_bytes=_MAX_RUNTIME_METADATA_BYTES,
                label="retained C0 hnswlib receipt",
            )
            != hnswlib_receipt_bytes
            or read_secure_regular_file(
                hnswlib_wheel_path,
                max_bytes=_MAX_HNSW_WHEEL_BYTES,
                label="retained C0 hnswlib wheel",
            )
            != hnswlib_wheel_bytes
            or read_secure_regular_file(
                native_build_receipt_path,
                max_bytes=_MAX_RUNTIME_METADATA_BYTES,
                label="retained C0 native-build receipt",
            )
            != native_build_receipt_bytes
            or read_secure_regular_file(
                opa_build_receipt_path,
                max_bytes=_MAX_RUNTIME_METADATA_BYTES,
                label="retained C0 OPA-build receipt",
            )
            != opa_build_receipt_bytes
            or read_secure_regular_file(
                runtime_library_manifest_path,
                max_bytes=_MAX_RUNTIME_METADATA_BYTES,
                label="retained C0 runtime-library manifest",
            )
            != runtime_library_manifest_bytes
            or read_secure_regular_file(
                sqlite_library_path,
                max_bytes=_MAX_NATIVE_LIBRARY_BYTES,
                label="retained C0 SQLite library",
            )
            != sqlite_library_bytes
            or read_secure_regular_file(
                zlib_library_path,
                max_bytes=_MAX_NATIVE_LIBRARY_BYTES,
                label="retained C0 zlib library",
            )
            != zlib_library_bytes
        ):
            raise OpaRuntimeBinaryError("retained C0 bytes changed during attestation review")
    except ArtifactIntegrityError as exc:
        raise OpaRuntimeBinaryError(f"retained C0 bytes changed during review: {exc}") from exc

    output = Path(output_path).expanduser()
    _write_exclusive_executable(output, opa_bytes)
    observed = verify_opa_runtime_binary(
        output,
        image=exact_image,
        plan_paths=plan_paths,
    )
    if observed != verification:
        raise OpaRuntimeBinaryError("retained OPA bytes changed after exclusive creation")
    return RetainedOpaMaterialization(
        extraction_receipt_sha256=receipt_digest,
        runtime_binding=observed,
        selected_manifest_digest=receipt.image_manifest_digest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-opa-runtime-binary",
        description="extract or verify the exact C0 OPA executable used by all five C1 plans",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    retained = commands.add_parser("materialize-retained")
    retained.add_argument("--c0-package", required=True, type=Path)
    retained.add_argument("--image", required=True)
    retained.add_argument("--plan-root", required=True, type=Path)
    retained.add_argument("--output", required=True, type=Path)
    image = commands.add_parser("materialize-image")
    image.add_argument("--image", required=True)
    image.add_argument("--plan-root", required=True, type=Path)
    image.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--image", required=True)
    verify.add_argument("--plan-root", required=True, type=Path)
    verify.add_argument("--opa", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        paths = plan_template_paths(arguments.plan_root)
        if arguments.command == "materialize-retained":
            result = materialize_retained_opa_runtime_binary(
                c0_package_root=arguments.c0_package,
                image=arguments.image,
                plan_paths=paths,
                output_path=arguments.output,
            )
        elif arguments.command == "materialize-image":
            result = materialize_opa_runtime_binary(
                image=arguments.image,
                plan_paths=paths,
                output_path=arguments.output,
            )
        else:
            result = verify_opa_runtime_binary(
                arguments.opa,
                image=arguments.image,
                plan_paths=paths,
            )
        print((_canonical_bytes(result.to_dict()) + b"\n").decode("ascii"), end="")
        return 0
    except (OpaRuntimeBinaryError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
