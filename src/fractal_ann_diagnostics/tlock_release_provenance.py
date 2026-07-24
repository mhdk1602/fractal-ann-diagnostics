"""Closed provenance contract for the release-only drand tlock executable.

The online execution container must not contain this tool. The Linux ARM64
release container admits it only after LABEL_RELEASE_CLAIMED. This module
records the exact release lineage, archive, executable, and drand chain. The
release round remains null in the pre-freeze record and may be set exactly once
when the custody package is frozen.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .artifact_integrity import (
    ArtifactIntegrityError,
    read_secure_control_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)

TLOCK_RELEASE_PROVENANCE_SCHEMA = "fractal-tlock-release-provenance-v2"

TLOCK_REPOSITORY = "https://github.com/drand/tlock"
TLOCK_RELEASE_VERSION = "v1.2.0"
TLOCK_RELEASE_TAG_OBJECT_GIT_SHA1 = "6a94bf6b8200ab67f2b80af8000a55db64998d94"
TLOCK_SOURCE_COMMIT_GIT_SHA1 = "7b54141a9733fd6fa207587a11148280e6fb020d"
TLOCK_SOURCE_ARCHIVE_URL = (
    "https://github.com/drand/tlock/archive/7b54141a9733fd6fa207587a11148280e6fb020d.tar.gz"
)
TLOCK_SOURCE_ARCHIVE_SHA256 = "98b5edb760cffbe6edd392f004d2d51fcc7a8e6ef7ed7672c32b1a9e1ce3e32d"
TLOCK_SOURCE_TREE_MANIFEST_SHA256 = (
    "6fedff45430fc81e9dbf5b13b1a2dc90e9840ae91f03de307dac5c2f7475c94c"
)
TLOCK_LINUX_ARM64_BINARY_SHA256 = "ca9d498b6a3c1ea8edff9ace7bf00eb0f90ce67166343161f9a53f21900a6ef5"
TLOCK_LINUX_ARM64_BINARY_BYTE_COUNT = 13_303_934
TLOCK_BUILDER_IMAGE = (
    "docker.io/library/golang:1.26.5-bookworm@"
    "sha256:1ecb7edf62a0408027bd5729dfd6b1b8766e578e8df93995b225dfd0944eb651"
)
TLOCK_GO_VERSION = "1.26.5"
TLOCK_GO_LINUX_ARM64_TARBALL_SHA256 = (
    "fe4789e92b1f33358680864bbe8704289e7bb5fc207d80623c308935bd696d49"
)
TLOCK_GO_LINUX_ARM64_TOOL_SHA256 = (
    "22201b57b855105df064a291863c3fc04f22a7431187a9205122aff42a0c825b"
)
TLOCK_ORIGINAL_GO_MOD_SHA256 = "0ee3447d4c3149e657a2f63c2e0046c19c21dcc63f730402cb24b08399db7741"
TLOCK_ORIGINAL_GO_SUM_SHA256 = "1cb67cce42d7cf12be184f0f6a820c1f8c2f105615d43cf0a176ee35741c523b"
TLOCK_PATCHED_GO_MOD_SHA256 = "ca99d5021580cc77d05367b7356b542fa3d77bc9f286aaa7d236b2a95a350c08"
TLOCK_PATCHED_GO_SUM_SHA256 = "baa8d4e184c2d516317ecaf984c9d7aa5ac9f7fbd0209058538200b0292c71e0"
TLOCK_DEPENDENCY_DELTA_SHA256 = "1b15bd1dd497c5553806ea5c58c170d6580ccc6139199d4fa9028e0ef8b79c59"

# This vulnerable upstream binary is never admitted to the release image. It is
# retained only as the fixed-side compatibility fixture in the Quicknet test.
TLOCK_OFFICIAL_INTEROP_FIXTURE_ARCHIVE_URL = (
    "https://github.com/drand/tlock/releases/download/v1.2.0/tlock_1.2.0_linux_arm64.tar.gz"
)
TLOCK_OFFICIAL_INTEROP_FIXTURE_ARCHIVE_SHA256 = (
    "3b724032620587c2551ee857c98dc02690076f4972a4fe4389b0f6e0911a6a92"
)
TLOCK_OFFICIAL_INTEROP_FIXTURE_BINARY_SHA256 = (
    "e153cfa8539e871f50143d1bde10fec7ec3fe82630f717c3c1bf166eb4975059"
)
TLOCK_OFFICIAL_INTEROP_FIXTURE_BINARY_BYTE_COUNT = 11_862_151

QUICKNET_NETWORK = "https://api.drand.sh/"
QUICKNET_CHAIN_HASH = "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
QUICKNET_SCHEME_ID = "bls-unchained-g1-rfc9380"
QUICKNET_PERIOD_SECONDS = 3
QUICKNET_GENESIS_UNIX_SECONDS = 1_692_803_367
QUICKNET_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_LOWER_HEX = re.compile(r"^[0-9a-f]+$")
_MAX_RECORD_BYTES = 64 * 1024
_MAX_TLOCK_BINARY_BYTES = 64 * 1024 * 1024

_FIELDS = frozenset(
    {
        "binary_byte_count",
        "binary_sha256",
        "builder_image",
        "chain_genesis_unix_seconds",
        "chain_hash",
        "chain_period_seconds",
        "chain_public_key",
        "chain_scheme_id",
        "drand_network",
        "drand_round",
        "dependency_delta_sha256",
        "go_linux_arm64_tarball_sha256",
        "go_linux_arm64_tool_sha256",
        "go_version",
        "official_interop_fixture_archive_sha256",
        "official_interop_fixture_archive_url",
        "official_interop_fixture_binary_byte_count",
        "official_interop_fixture_binary_sha256",
        "original_go_mod_sha256",
        "original_go_sum_sha256",
        "patched_go_mod_sha256",
        "patched_go_sum_sha256",
        "release_tag_object_git_sha1",
        "release_version",
        "schema_version",
        "source_archive_sha256",
        "source_archive_url",
        "source_commit_git_sha1",
        "source_tree_manifest_sha256",
        "target_architecture",
        "target_operating_system",
        "tool_repository",
    }
)


class TlockReleaseProvenanceError(ValueError):
    """Raised when tlock provenance is open, substituted, or internally inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TlockReleaseProvenanceError(
            "tlock release provenance must be finite canonical JSON"
        ) from exc


def _closed(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise TlockReleaseProvenanceError(
            "tlock release provenance must be a JSON object with string keys"
        )
    observed = set(value)
    if observed != _FIELDS:
        raise TlockReleaseProvenanceError(
            "tlock release provenance keys differ; "
            f"missing={sorted(_FIELDS - observed)}, unexpected={sorted(observed - _FIELDS)}"
        )
    return value


def _parse_json_object(encoded: bytes) -> Mapping[str, Any]:
    if type(encoded) is not bytes or not encoded or len(encoded) > _MAX_RECORD_BYTES:
        raise TlockReleaseProvenanceError(
            "tlock release provenance must be non-empty bounded bytes"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TlockReleaseProvenanceError(f"tlock release provenance repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise TlockReleaseProvenanceError(
            f"tlock release provenance contains non-finite value {value!r}"
        )

    try:
        decoded = json.loads(
            encoded.decode("ascii", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TlockReleaseProvenanceError(
            "tlock release provenance must contain valid ASCII JSON"
        ) from exc
    if type(decoded) is not dict:
        raise TlockReleaseProvenanceError("tlock release provenance must contain one JSON object")
    return decoded


def _require_exact(name: str, value: object, expected: object) -> None:
    if type(value) is not type(expected) or value != expected:
        raise TlockReleaseProvenanceError(f"{name} differs from the verified release pin")


def _require_sha256(name: str, value: object) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TlockReleaseProvenanceError(f"{name} must be a lowercase SHA-256 digest")


def _require_git_sha1(name: str, value: object) -> None:
    if type(value) is not str or _GIT_SHA1.fullmatch(value) is None:
        raise TlockReleaseProvenanceError(f"{name} must be one full lowercase Git SHA-1")


@dataclass(frozen=True)
class TlockReleaseProvenance:
    """Canonical lineage and chain record for one offline release executable."""

    tool_repository: str
    release_version: str
    release_tag_object_git_sha1: str
    source_commit_git_sha1: str
    target_operating_system: str
    target_architecture: str
    source_archive_url: str
    source_archive_sha256: str
    source_tree_manifest_sha256: str
    builder_image: str
    go_version: str
    go_linux_arm64_tarball_sha256: str
    go_linux_arm64_tool_sha256: str
    original_go_mod_sha256: str
    original_go_sum_sha256: str
    patched_go_mod_sha256: str
    patched_go_sum_sha256: str
    dependency_delta_sha256: str
    binary_sha256: str
    binary_byte_count: int
    official_interop_fixture_archive_url: str
    official_interop_fixture_archive_sha256: str
    official_interop_fixture_binary_sha256: str
    official_interop_fixture_binary_byte_count: int
    drand_network: str
    chain_hash: str
    chain_scheme_id: str
    chain_period_seconds: int
    chain_genesis_unix_seconds: int
    chain_public_key: str
    drand_round: int | None
    schema_version: str = TLOCK_RELEASE_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        _require_exact("schema_version", self.schema_version, TLOCK_RELEASE_PROVENANCE_SCHEMA)
        _require_git_sha1("release_tag_object_git_sha1", self.release_tag_object_git_sha1)
        _require_git_sha1("source_commit_git_sha1", self.source_commit_git_sha1)
        for name in (
            "source_archive_sha256",
            "source_tree_manifest_sha256",
            "go_linux_arm64_tarball_sha256",
            "go_linux_arm64_tool_sha256",
            "original_go_mod_sha256",
            "original_go_sum_sha256",
            "patched_go_mod_sha256",
            "patched_go_sum_sha256",
            "dependency_delta_sha256",
            "binary_sha256",
            "official_interop_fixture_archive_sha256",
            "official_interop_fixture_binary_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if type(self.binary_byte_count) is not int or self.binary_byte_count <= 0:
            raise TlockReleaseProvenanceError("binary_byte_count must be a positive integer")
        if (
            type(self.official_interop_fixture_binary_byte_count) is not int
            or self.official_interop_fixture_binary_byte_count <= 0
        ):
            raise TlockReleaseProvenanceError(
                "official_interop_fixture_binary_byte_count must be a positive integer"
            )
        if type(self.chain_period_seconds) is not int or self.chain_period_seconds <= 0:
            raise TlockReleaseProvenanceError("chain_period_seconds must be a positive integer")
        if type(self.chain_genesis_unix_seconds) is not int or self.chain_genesis_unix_seconds <= 0:
            raise TlockReleaseProvenanceError(
                "chain_genesis_unix_seconds must be a positive integer"
            )
        if type(self.chain_hash) is not str or _SHA256.fullmatch(self.chain_hash) is None:
            raise TlockReleaseProvenanceError(
                "chain_hash must be 64 lowercase hexadecimal characters"
            )
        if (
            type(self.chain_public_key) is not str
            or len(self.chain_public_key) % 2
            or _LOWER_HEX.fullmatch(self.chain_public_key) is None
        ):
            raise TlockReleaseProvenanceError(
                "chain_public_key must be even-length lowercase hexadecimal"
            )
        if self.drand_round is not None and (
            type(self.drand_round) is not int or self.drand_round <= 0
        ):
            raise TlockReleaseProvenanceError(
                "drand_round must be null before freeze or a positive integer at freeze"
            )

        expected = {
            "tool_repository": TLOCK_REPOSITORY,
            "release_version": TLOCK_RELEASE_VERSION,
            "release_tag_object_git_sha1": TLOCK_RELEASE_TAG_OBJECT_GIT_SHA1,
            "source_commit_git_sha1": TLOCK_SOURCE_COMMIT_GIT_SHA1,
            "target_operating_system": "linux",
            "target_architecture": "arm64",
            "source_archive_url": TLOCK_SOURCE_ARCHIVE_URL,
            "source_archive_sha256": TLOCK_SOURCE_ARCHIVE_SHA256,
            "source_tree_manifest_sha256": TLOCK_SOURCE_TREE_MANIFEST_SHA256,
            "builder_image": TLOCK_BUILDER_IMAGE,
            "go_version": TLOCK_GO_VERSION,
            "go_linux_arm64_tarball_sha256": TLOCK_GO_LINUX_ARM64_TARBALL_SHA256,
            "go_linux_arm64_tool_sha256": TLOCK_GO_LINUX_ARM64_TOOL_SHA256,
            "original_go_mod_sha256": TLOCK_ORIGINAL_GO_MOD_SHA256,
            "original_go_sum_sha256": TLOCK_ORIGINAL_GO_SUM_SHA256,
            "patched_go_mod_sha256": TLOCK_PATCHED_GO_MOD_SHA256,
            "patched_go_sum_sha256": TLOCK_PATCHED_GO_SUM_SHA256,
            "dependency_delta_sha256": TLOCK_DEPENDENCY_DELTA_SHA256,
            "binary_sha256": TLOCK_LINUX_ARM64_BINARY_SHA256,
            "binary_byte_count": TLOCK_LINUX_ARM64_BINARY_BYTE_COUNT,
            "official_interop_fixture_archive_url": (TLOCK_OFFICIAL_INTEROP_FIXTURE_ARCHIVE_URL),
            "official_interop_fixture_archive_sha256": (
                TLOCK_OFFICIAL_INTEROP_FIXTURE_ARCHIVE_SHA256
            ),
            "official_interop_fixture_binary_sha256": (
                TLOCK_OFFICIAL_INTEROP_FIXTURE_BINARY_SHA256
            ),
            "official_interop_fixture_binary_byte_count": (
                TLOCK_OFFICIAL_INTEROP_FIXTURE_BINARY_BYTE_COUNT
            ),
            "drand_network": QUICKNET_NETWORK,
            "chain_hash": QUICKNET_CHAIN_HASH,
            "chain_scheme_id": QUICKNET_SCHEME_ID,
            "chain_period_seconds": QUICKNET_PERIOD_SECONDS,
            "chain_genesis_unix_seconds": QUICKNET_GENESIS_UNIX_SECONDS,
            "chain_public_key": QUICKNET_PUBLIC_KEY,
        }
        for name, pinned in expected.items():
            _require_exact(name, getattr(self, name), pinned)

    @classmethod
    def prefreeze_quicknet_v1_2_0_linux_arm64(cls) -> TlockReleaseProvenance:
        """Construct the verified pre-freeze record without inventing a round."""

        return cls(
            tool_repository=TLOCK_REPOSITORY,
            release_version=TLOCK_RELEASE_VERSION,
            release_tag_object_git_sha1=TLOCK_RELEASE_TAG_OBJECT_GIT_SHA1,
            source_commit_git_sha1=TLOCK_SOURCE_COMMIT_GIT_SHA1,
            target_operating_system="linux",
            target_architecture="arm64",
            source_archive_url=TLOCK_SOURCE_ARCHIVE_URL,
            source_archive_sha256=TLOCK_SOURCE_ARCHIVE_SHA256,
            source_tree_manifest_sha256=TLOCK_SOURCE_TREE_MANIFEST_SHA256,
            builder_image=TLOCK_BUILDER_IMAGE,
            go_version=TLOCK_GO_VERSION,
            go_linux_arm64_tarball_sha256=TLOCK_GO_LINUX_ARM64_TARBALL_SHA256,
            go_linux_arm64_tool_sha256=TLOCK_GO_LINUX_ARM64_TOOL_SHA256,
            original_go_mod_sha256=TLOCK_ORIGINAL_GO_MOD_SHA256,
            original_go_sum_sha256=TLOCK_ORIGINAL_GO_SUM_SHA256,
            patched_go_mod_sha256=TLOCK_PATCHED_GO_MOD_SHA256,
            patched_go_sum_sha256=TLOCK_PATCHED_GO_SUM_SHA256,
            dependency_delta_sha256=TLOCK_DEPENDENCY_DELTA_SHA256,
            binary_sha256=TLOCK_LINUX_ARM64_BINARY_SHA256,
            binary_byte_count=TLOCK_LINUX_ARM64_BINARY_BYTE_COUNT,
            official_interop_fixture_archive_url=(TLOCK_OFFICIAL_INTEROP_FIXTURE_ARCHIVE_URL),
            official_interop_fixture_archive_sha256=(TLOCK_OFFICIAL_INTEROP_FIXTURE_ARCHIVE_SHA256),
            official_interop_fixture_binary_sha256=(TLOCK_OFFICIAL_INTEROP_FIXTURE_BINARY_SHA256),
            official_interop_fixture_binary_byte_count=(
                TLOCK_OFFICIAL_INTEROP_FIXTURE_BINARY_BYTE_COUNT
            ),
            drand_network=QUICKNET_NETWORK,
            chain_hash=QUICKNET_CHAIN_HASH,
            chain_scheme_id=QUICKNET_SCHEME_ID,
            chain_period_seconds=QUICKNET_PERIOD_SECONDS,
            chain_genesis_unix_seconds=QUICKNET_GENESIS_UNIX_SECONDS,
            chain_public_key=QUICKNET_PUBLIC_KEY,
            drand_round=None,
        )

    @property
    def is_frozen(self) -> bool:
        return self.drand_round is not None

    def require_frozen(self) -> None:
        if not self.is_frozen:
            raise TlockReleaseProvenanceError("tlock release provenance has no frozen drand round")

    def to_dict(self) -> dict[str, object]:
        return {
            "binary_byte_count": self.binary_byte_count,
            "binary_sha256": self.binary_sha256,
            "builder_image": self.builder_image,
            "chain_genesis_unix_seconds": self.chain_genesis_unix_seconds,
            "chain_hash": self.chain_hash,
            "chain_period_seconds": self.chain_period_seconds,
            "chain_public_key": self.chain_public_key,
            "chain_scheme_id": self.chain_scheme_id,
            "drand_network": self.drand_network,
            "drand_round": self.drand_round,
            "dependency_delta_sha256": self.dependency_delta_sha256,
            "go_linux_arm64_tarball_sha256": self.go_linux_arm64_tarball_sha256,
            "go_linux_arm64_tool_sha256": self.go_linux_arm64_tool_sha256,
            "go_version": self.go_version,
            "official_interop_fixture_archive_sha256": (
                self.official_interop_fixture_archive_sha256
            ),
            "official_interop_fixture_archive_url": (self.official_interop_fixture_archive_url),
            "official_interop_fixture_binary_byte_count": (
                self.official_interop_fixture_binary_byte_count
            ),
            "official_interop_fixture_binary_sha256": (self.official_interop_fixture_binary_sha256),
            "original_go_mod_sha256": self.original_go_mod_sha256,
            "original_go_sum_sha256": self.original_go_sum_sha256,
            "patched_go_mod_sha256": self.patched_go_mod_sha256,
            "patched_go_sum_sha256": self.patched_go_sum_sha256,
            "release_tag_object_git_sha1": self.release_tag_object_git_sha1,
            "release_version": self.release_version,
            "schema_version": self.schema_version,
            "source_archive_sha256": self.source_archive_sha256,
            "source_archive_url": self.source_archive_url,
            "source_commit_git_sha1": self.source_commit_git_sha1,
            "source_tree_manifest_sha256": self.source_tree_manifest_sha256,
            "target_architecture": self.target_architecture,
            "target_operating_system": self.target_operating_system,
            "tool_repository": self.tool_repository,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def record_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> TlockReleaseProvenance:
        row = _closed(value)
        return cls(**row)


def freeze_tlock_release_provenance(
    provenance: TlockReleaseProvenance,
    *,
    drand_round: int,
) -> TlockReleaseProvenance:
    """Bind the selected absolute round exactly once at the C1 custody freeze."""

    if not isinstance(provenance, TlockReleaseProvenance):
        raise TlockReleaseProvenanceError("provenance must be a TlockReleaseProvenance record")
    if provenance.is_frozen:
        raise TlockReleaseProvenanceError("drand_round is already frozen")
    return replace(provenance, drand_round=drand_round)


def loads_tlock_release_provenance(encoded: bytes) -> TlockReleaseProvenance:
    """Load one canonical newline-terminated provenance artifact."""

    provenance = TlockReleaseProvenance.from_dict(_parse_json_object(encoded))
    if encoded != provenance.canonical_file_bytes():
        raise TlockReleaseProvenanceError("tlock release provenance bytes are not canonical")
    return provenance


def load_tlock_release_provenance(path: str | Path) -> TlockReleaseProvenance:
    try:
        encoded = read_secure_control_file(path, label="tlock release provenance")
    except ArtifactIntegrityError as exc:
        raise TlockReleaseProvenanceError(f"cannot read tlock release provenance: {exc}") from exc
    return loads_tlock_release_provenance(encoded)


def write_tlock_release_provenance(
    provenance: TlockReleaseProvenance,
    target: str | Path,
) -> None:
    if not isinstance(provenance, TlockReleaseProvenance):
        raise TlockReleaseProvenanceError("provenance must be a TlockReleaseProvenance record")
    try:
        write_exclusive_receipt_bytes(provenance.canonical_file_bytes(), target)
    except ArtifactIntegrityError as exc:
        raise TlockReleaseProvenanceError(
            f"cannot publish tlock release provenance: {exc}"
        ) from exc


def verify_tlock_release_binary(
    provenance: TlockReleaseProvenance,
    binary_path: str | Path,
) -> None:
    """Verify exact executable bytes and count before a custody operation."""

    if not isinstance(provenance, TlockReleaseProvenance):
        raise TlockReleaseProvenanceError("provenance must be a TlockReleaseProvenance record")
    try:
        encoded = read_secure_regular_file(
            binary_path,
            max_bytes=_MAX_TLOCK_BINARY_BYTES,
            label="tlock release binary",
        )
    except ArtifactIntegrityError as exc:
        raise TlockReleaseProvenanceError(f"cannot read tlock release binary: {exc}") from exc
    if len(encoded) != provenance.binary_byte_count:
        raise TlockReleaseProvenanceError("tlock release binary byte count differs from provenance")
    if hashlib.sha256(encoded).hexdigest() != provenance.binary_sha256:
        raise TlockReleaseProvenanceError("tlock release binary digest differs from provenance")
