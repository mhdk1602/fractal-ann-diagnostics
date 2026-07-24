"""Typed artifact-pin inventory and fail-closed candidate-manifest assembly.

The candidate manifest is too consequential to accept operator-supplied hashes.  This
module derives its artifact table from the controlled freeze layout, records the
derivation class for every revision, and publishes the inventory and its receipt as
one exclusive directory transaction.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from .freeze_package import FreezePackageError, _inspect_target, layout_from_manifest
from .production_embedding_build import (
    QWEN_CURRENT_REVISION,
    QWEN_CURRENT_TREE_SHA256,
    QWEN_STALE_REVISION,
    QWEN_STALE_TREE_SHA256,
)
from .provider_rehearsal import CandidateImageClosure
from .study import (
    _ARTIFACT_ROLE_SPECS,
    C0_COMMIT_SENTINEL,
    FIXED_CORPORA,
    StudyManifestError,
    manifest_sha256,
    validate_candidate_rehearsal_manifest,
    validate_study_manifest,
)
from .study_data import StudyDataError, verify_staged_data

CANDIDATE_ARTIFACT_PIN_INVENTORY_SCHEMA = "fractal-candidate-artifact-pin-inventory-v1"
CANDIDATE_ARTIFACT_PIN_RECEIPT_SCHEMA = "fractal-candidate-artifact-pin-inventory-receipt-v1"
CANDIDATE_MANIFEST_ASSEMBLY_RECEIPT_SCHEMA = "fractal-candidate-manifest-assembly-v1"
INVENTORY_FILENAME = "candidate-artifact-pin-inventory.json"
INVENTORY_RECEIPT_FILENAME = "candidate-artifact-pin-inventory-receipt.json"
CANDIDATE_MANIFEST_FILENAME = "candidate-study-manifest.json"
ASSEMBLY_RECEIPT_FILENAME = "candidate-manifest-assembly-receipt.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CANDIDATE_VALIDATION_PROBE_COMMIT = "0" * 40
_PLACEHOLDERS = {"", "tbd", "todo", "latest", "main", "master", "unassigned"}
_LOCAL_CODE_ROLES = {
    "corpus-normalizer",
    "exact-authorized-oracle",
    "frozen-controller",
    "custody-builder",
}
_UPSTREAM_STAGED_ROLES = {
    "sealed-inputs",
    "sealed-labels",
    "study-data-package",
    "online-staging-package",
}
_QWEN_ROLES = {"primary-embedding", "stale-embedding"}
_TOOL_ROLES = {
    "strict-authorized-hnsw",
    "opa-runtime-binary",
    "timelock-tool",
}
_ROW_FIELDS = frozenset(
    {
        "artifact_id",
        "byte_count",
        "corpus_id",
        "directory_count",
        "evidence_class",
        "file_count",
        "kind",
        "license",
        "relative_path",
        "revision",
        "role",
        "sha256",
        "uri",
    }
)
_INVENTORY_FIELDS = frozenset(
    {
        "artifact_count",
        "artifacts",
        "fixed_corpora",
        "schema_version",
        "template_sha256",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "artifact_count",
        "artifact_root",
        "inventory_file_sha256",
        "repository_root",
        "schema_version",
        "template_sha256",
    }
)
_ASSEMBLY_FIELDS = frozenset(
    {
        "artifact_count",
        "artifact_inventory_file_sha256",
        "build_context_tree_sha256",
        "candidate_image_closure_file_sha256",
        "candidate_image_source_commit",
        "c0_sentinel_count",
        "manifest_file_sha256",
        "manifest_semantic_sha256",
        "provider_plan_template_closure_sha256",
        "release_image_index_digest",
        "schema_version",
        "scientific_image_index_digest",
    }
)


class CandidateManifestAssemblyError(ValueError):
    """Raised when evidence cannot close one candidate manifest exactly."""


@dataclass(frozen=True)
class CandidateManifestAssemblyReceipt:
    """Closed evidence for one candidate manifest package."""

    artifact_count: int
    artifact_inventory_file_sha256: str
    build_context_tree_sha256: str
    candidate_image_closure_file_sha256: str
    candidate_image_source_commit: str
    c0_sentinel_count: int
    manifest_file_sha256: str
    manifest_semantic_sha256: str
    provider_plan_template_closure_sha256: str
    release_image_index_digest: str
    scientific_image_index_digest: str
    schema_version: str = CANDIDATE_MANIFEST_ASSEMBLY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.artifact_count != 79:
            raise CandidateManifestAssemblyError("candidate assembly artifact_count must equal 79")
        if self.c0_sentinel_count != 13:
            raise CandidateManifestAssemblyError(
                "candidate assembly c0_sentinel_count must equal 13"
            )
        for name in (
            "artifact_inventory_file_sha256",
            "build_context_tree_sha256",
            "candidate_image_closure_file_sha256",
            "manifest_file_sha256",
            "manifest_semantic_sha256",
            "provider_plan_template_closure_sha256",
        ):
            _digest(getattr(self, name), label=f"candidate assembly {name}")
        if _GIT_COMMIT.fullmatch(self.candidate_image_source_commit) is None:
            raise CandidateManifestAssemblyError(
                "candidate assembly candidate_image_source_commit must be one full Git commit"
            )
        for name in ("release_image_index_digest", "scientific_image_index_digest"):
            value = getattr(self, name)
            if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise CandidateManifestAssemblyError(
                    f"candidate assembly {name} must be one OCI SHA-256 digest"
                )
        if self.schema_version != CANDIDATE_MANIFEST_ASSEMBLY_RECEIPT_SCHEMA:
            raise CandidateManifestAssemblyError("candidate assembly receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> CandidateManifestAssemblyReceipt:
        row = _closed(value, _ASSEMBLY_FIELDS, label="candidate assembly receipt")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ClosedCandidateManifestPackage:
    """Descriptor-bound readback of the exact two-member candidate package."""

    root: Path
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    receipt: CandidateManifestAssemblyReceipt
    receipt_bytes: bytes


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CandidateManifestAssemblyError("candidate evidence is not canonical JSON") from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise CandidateManifestAssemblyError(f"{label} must be an object with string keys")
    observed = frozenset(value)
    if observed != fields:
        raise CandidateManifestAssemblyError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise CandidateManifestAssemblyError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_json_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    maximum: int = 32 * 1024 * 1024,
    require_canonical: bool = True,
    required_mode: int | None = None,
    require_current_owner: bool = False,
) -> tuple[object, bytes]:
    if not name or "/" in name or name in {".", ".."}:
        raise CandidateManifestAssemblyError(f"{label} member name is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise CandidateManifestAssemblyError(f"{label} is not one bounded regular file")
        if required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode:
            raise CandidateManifestAssemblyError(f"{label} mode must equal {required_mode:04o}")
        if require_current_owner and hasattr(os, "geteuid") and before.st_uid != os.geteuid():
            raise CandidateManifestAssemblyError(f"{label} is not owned by the current operator")
        chunks: list[bytes] = []
        observed = 0
        while observed <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
        encoded = b"".join(chunks)
        if observed > maximum or _stat_signature(before) != _stat_signature(after):
            raise CandidateManifestAssemblyError(f"{label} changed while read")
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _stat_signature(current) != _stat_signature(before):
            raise CandidateManifestAssemblyError(f"{label} was replaced while read")
        value = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateManifestAssemblyError(f"cannot read {label}: {exc}") from exc
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    if require_canonical and encoded != _canonical_bytes(value):
        raise CandidateManifestAssemblyError(f"{label} bytes are not canonical")
    return value, encoded


def _open_real_directory(path: Path, *, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise CandidateManifestAssemblyError(f"cannot open {label}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise CandidateManifestAssemblyError(f"{label} is not a real directory")
    return descriptor


def _open_controlled_output_parent(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    """Open one owner-controlled output parent without following any component."""

    if not path.is_absolute():
        raise CandidateManifestAssemblyError(f"{label} must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open("/", flags)
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino)
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise CandidateManifestAssemblyError(
                f"{label} must be one owner-controlled non-writable-by-others directory"
            )
        return descriptor, metadata
    except CandidateManifestAssemblyError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise CandidateManifestAssemblyError(f"cannot open {label}: {exc}") from exc


def _load_template_secure(path: Path) -> dict[str, Any]:
    parent = path.parent.resolve(strict=True)
    descriptor = _open_real_directory(parent, label="manifest template parent")
    try:
        value, _ = _read_json_at(
            descriptor,
            path.name,
            label="manifest template",
            require_canonical=False,
        )
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise CandidateManifestAssemblyError("manifest template root is not an object")
    return value


def _load_canonical_mapping_secure(path: Path, *, label: str) -> dict[str, Any]:
    parent = path.parent.resolve(strict=True)
    descriptor = _open_real_directory(parent, label=f"{label} parent")
    try:
        value, _ = _read_json_at(descriptor, path.name, label=label)
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise CandidateManifestAssemblyError(f"{label} root is not an object")
    return value


def _load_candidate_image_closure_secure(path: Path) -> CandidateImageClosure:
    value = _load_canonical_mapping_secure(path, label="candidate image closure")
    expected = frozenset(CandidateImageClosure.__dataclass_fields__)
    row = _closed(value, expected, label="candidate image closure")
    try:
        closure = CandidateImageClosure(**row)
    except (TypeError, ValueError) as exc:
        raise CandidateManifestAssemblyError(f"candidate image closure is invalid: {exc}") from exc
    if _canonical_bytes(closure.to_dict()) != _canonical_bytes(value):
        raise CandidateManifestAssemblyError("candidate image closure changed after parsing")
    return closure


def _template_sha256(template: Mapping[str, Any]) -> str:
    return _sha256(_canonical_bytes(template))


def _is_placeholder(value: object) -> bool:
    return not isinstance(value, str) or value.strip().casefold() in _PLACEHOLDERS


def _revision_for(
    row: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    staged_inventory_revision: str,
) -> tuple[str, str]:
    digest = _digest(observed.get("sha256"), label=f"{row['artifact_id']} observed digest")
    typed_revision = observed.get("revision")
    role = str(row["role"])
    if role == "source-code":
        return C0_COMMIT_SENTINEL, "source-code-c0-sentinel"
    if isinstance(typed_revision, str) and typed_revision:
        return typed_revision, "typed-artifact-receipt"
    if role in _LOCAL_CODE_ROLES:
        source_digest = observed.get("source_sha256")
        if source_digest != digest:
            raise CandidateManifestAssemblyError(
                f"local code artifact {row['artifact_id']!r} differs from its repository source"
            )
        return f"sha256:{digest}", "repository-file-digest"
    if role in _UPSTREAM_STAGED_ROLES:
        return staged_inventory_revision, "staged-study-data-inventory-revision"
    if role in _QWEN_ROLES:
        expected = (
            (QWEN_CURRENT_REVISION, QWEN_CURRENT_TREE_SHA256)
            if role == "primary-embedding"
            else (QWEN_STALE_REVISION, QWEN_STALE_TREE_SHA256)
        )
        if digest != expected[1]:
            raise CandidateManifestAssemblyError(
                f"{role} tree differs from its admitted Qwen revision"
            )
        return expected[0], "admitted-upstream-model-revision"
    if role in _TOOL_ROLES:
        return f"sha256:{digest}", "typed-tool-or-lock-content"
    return f"sha256:{digest}", "controlled-generated-content"


def _inventory_row(
    template_row: Mapping[str, Any],
    layout_row: Any,
    observed: Mapping[str, Any],
    *,
    artifact_root: Path,
    staged_inventory_revision: str,
) -> dict[str, object]:
    if observed.get("state") != "present" or observed.get("sha256") is None:
        raise CandidateManifestAssemblyError(
            f"controlled artifact {layout_row.artifact_id!r} is not present"
        )
    if (
        observed.get("artifact_id") != layout_row.artifact_id
        or observed.get("role") != layout_row.role
        or observed.get("relative_path") != layout_row.relative_path
        or observed.get("kind") != layout_row.kind
    ):
        raise CandidateManifestAssemblyError(
            f"inspector identity differs for {layout_row.artifact_id!r}"
        )
    revision, evidence_class = _revision_for(
        template_row,
        observed,
        staged_inventory_revision=staged_inventory_revision,
    )
    uri = template_row.get("uri")
    if _is_placeholder(uri):
        target = artifact_root.joinpath(*PurePosixPath(layout_row.relative_path).parts)
        uri = target.resolve(strict=True).as_uri()
    license_value = template_row.get("license")
    if _is_placeholder(license_value):
        raise CandidateManifestAssemblyError(
            f"tracked schema lacks a license for {layout_row.artifact_id!r}"
        )
    return {
        "artifact_id": layout_row.artifact_id,
        "byte_count": observed["byte_count"],
        "corpus_id": template_row.get("corpus_id"),
        "directory_count": observed["directory_count"],
        "evidence_class": evidence_class,
        "file_count": observed["file_count"],
        "kind": layout_row.kind,
        "license": license_value,
        "relative_path": layout_row.relative_path,
        "revision": revision,
        "role": layout_row.role,
        "sha256": observed["sha256"],
        "uri": uri,
    }


@dataclass(frozen=True)
class CandidateArtifactPinInventory:
    template_sha256: str
    artifacts: tuple[Mapping[str, object], ...]
    schema_version: str = CANDIDATE_ARTIFACT_PIN_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        _digest(self.template_sha256, label="template_sha256")
        if self.schema_version != CANDIDATE_ARTIFACT_PIN_INVENTORY_SCHEMA:
            raise CandidateManifestAssemblyError("artifact inventory schema differs")
        if len(self.artifacts) != 79:
            raise CandidateManifestAssemblyError("artifact inventory must contain exactly 79 rows")
        ids: set[str] = set()
        coverage: list[tuple[str, str | None]] = []
        for position, value in enumerate(self.artifacts):
            row = _closed(value, _ROW_FIELDS, label=f"artifact inventory row {position}")
            artifact_id = row["artifact_id"]
            if type(artifact_id) is not str or not artifact_id or artifact_id in ids:
                raise CandidateManifestAssemblyError("artifact inventory IDs are not unique")
            ids.add(artifact_id)
            _digest(row["sha256"], label=f"artifact {artifact_id} sha256")
            coverage.append((str(row["role"]), row["corpus_id"]))
        corpus_roles = {
            "sealed-inputs",
            "sealed-labels",
            "sealed-label-ciphertext",
            "timelock-encryption-receipt",
            "online-execution",
            "corpus-normalizer",
            "policy-workload",
            "embedding-store",
            "authorized-index-store",
            "trial-runtime-package",
            "runtime-attestation-plan-template",
        }
        for role in corpus_roles:
            observed = tuple(corpus for item_role, corpus in coverage if item_role == role)
            if observed != FIXED_CORPORA:
                raise CandidateManifestAssemblyError(
                    f"artifact inventory {role!r} corpus order or coverage differs"
                )
        multiplicities = {
            role: sum(item_role == role for item_role, _ in coverage)
            for role in _ARTIFACT_ROLE_SPECS
        }
        expected_multiplicities = {role: count for role, (_, count) in _ARTIFACT_ROLE_SPECS.items()}
        if multiplicities != expected_multiplicities:
            raise CandidateManifestAssemblyError("artifact inventory role multiplicities differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_count": len(self.artifacts),
            "artifacts": [dict(row) for row in self.artifacts],
            "fixed_corpora": list(FIXED_CORPORA),
            "schema_version": self.schema_version,
            "template_sha256": self.template_sha256,
        }

    @property
    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes)

    @classmethod
    def from_dict(cls, value: object) -> CandidateArtifactPinInventory:
        row = _closed(value, _INVENTORY_FIELDS, label="artifact inventory")
        if row["artifact_count"] != 79 or row["fixed_corpora"] != list(FIXED_CORPORA):
            raise CandidateManifestAssemblyError("artifact inventory cardinality header differs")
        artifacts = row["artifacts"]
        if not isinstance(artifacts, list):
            raise CandidateManifestAssemblyError("artifact inventory artifacts must be an array")
        return cls(
            template_sha256=row["template_sha256"],
            artifacts=tuple(artifacts),
            schema_version=row["schema_version"],
        )


def _rename_noreplace_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        flag = 0x00000004
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        flag = 0x00000001
    else:  # pragma: no cover - supported production hosts are macOS and Linux
        function = None
        flag = 0
    if function is None:
        raise CandidateManifestAssemblyError("exclusive directory publication is unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    if (
        function(
            parent_fd,
            os.fsencode(source_name),
            parent_fd,
            os.fsencode(destination_name),
            flag,
        )
        != 0
    ):
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise CandidateManifestAssemblyError("candidate output already exists")
        raise CandidateManifestAssemblyError(
            f"cannot publish candidate output: {os.strerror(number)}"
        )


def _publish_directory_exclusive(
    work: Path,
    destination: Path,
    *,
    work_descriptor: int,
    expected_members: Mapping[str, bytes],
) -> None:
    parent = destination.parent.resolve(strict=True)
    if work.parent.resolve(strict=True) != parent or any(
        not path.name or path.name in {".", ".."} or "/" in path.name
        for path in (work, destination)
    ):
        raise CandidateManifestAssemblyError("candidate work and output must share one parent")
    if not expected_members or any(
        not name or name in {".", ".."} or "/" in name for name in expected_members
    ):
        raise CandidateManifestAssemblyError("candidate output member set is invalid")
    parent_fd, parent_metadata = _open_controlled_output_parent(
        parent,
        label="candidate output parent",
    )
    try:
        staged_metadata = os.fstat(work_descriptor)
        named_staging = os.stat(work.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(staged_metadata.st_mode)
            or stat.S_IMODE(staged_metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and staged_metadata.st_uid != os.geteuid())
            or (staged_metadata.st_dev, staged_metadata.st_ino)
            != (named_staging.st_dev, named_staging.st_ino)
        ):
            raise CandidateManifestAssemblyError(
                "candidate staging directory name changed before publication"
            )
        if destination.name in os.listdir(parent_fd):
            raise CandidateManifestAssemblyError("candidate output already exists")
        if set(os.listdir(work_descriptor)) != set(expected_members):
            raise CandidateManifestAssemblyError("candidate staging membership differs")
        for name, expected in expected_members.items():
            _, encoded = _read_json_at(
                work_descriptor,
                name,
                label=f"staged candidate member {name}",
            )
            if encoded != expected:
                raise CandidateManifestAssemblyError(
                    f"staged candidate member {name!r} bytes differ"
                )
        os.fsync(work_descriptor)
        _rename_noreplace_at(parent_fd, work.name, destination.name)
        os.fsync(parent_fd)
        after_parent = os.fstat(parent_fd)
        named_parent = destination.parent.lstat()
        published = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (parent_metadata.st_dev, parent_metadata.st_ino) != (
            after_parent.st_dev,
            after_parent.st_ino,
        ) or (parent_metadata.st_dev, parent_metadata.st_ino) != (
            named_parent.st_dev,
            named_parent.st_ino,
        ):
            raise CandidateManifestAssemblyError("candidate output parent changed")
        if (
            (published.st_dev, published.st_ino) != (staged_metadata.st_dev, staged_metadata.st_ino)
            or destination.name not in os.listdir(parent_fd)
            or work.name in os.listdir(parent_fd)
        ):
            raise CandidateManifestAssemblyError("candidate directory publication did not close")
        if set(os.listdir(work_descriptor)) != set(expected_members):
            raise CandidateManifestAssemblyError("published candidate membership differs")
        for name, expected in expected_members.items():
            _, encoded = _read_json_at(
                work_descriptor,
                name,
                label=f"published candidate member {name}",
            )
            if encoded != expected:
                raise CandidateManifestAssemblyError(
                    f"published candidate member {name!r} bytes differ"
                )
    finally:
        os.close(parent_fd)


def _write_private(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def build_candidate_artifact_pin_inventory(
    *,
    template_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    output_directory: str | Path,
) -> CandidateArtifactPinInventory:
    """Inspect the exact controlled layout and publish a closed 79-row inventory."""

    repository = Path(repository_root).expanduser().resolve(strict=True)
    root = Path(artifact_root).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise CandidateManifestAssemblyError("artifact root must be one real directory")
    template_file = Path(template_path).expanduser()
    if not template_file.is_absolute():
        template_file = Path.cwd() / template_file
    template = _load_template_secure(template_file)
    try:
        validate_study_manifest(template)
        layout = layout_from_manifest(template, repository)
    except (StudyManifestError, FreezePackageError) as exc:
        raise CandidateManifestAssemblyError(
            f"tracked manifest template is invalid: {exc}"
        ) from exc
    study_data_rows = tuple(row for row in layout if row.role == "study-data-package")
    if len(study_data_rows) != 1:
        raise CandidateManifestAssemblyError("controlled layout lacks one study-data package")
    staged_root = root.joinpath(*PurePosixPath(study_data_rows[0].relative_path).parts)
    try:
        staged_receipt = verify_staged_data(staged_root)
    except StudyDataError as exc:
        raise CandidateManifestAssemblyError(
            f"staged study-data revision cannot be admitted: {exc}"
        ) from exc
    staged_inventory_revision = f"sha256:{staged_receipt.inventory_sha256}"
    template_rows = {str(row["id"]): row for row in template["artifacts"]}
    rows: list[Mapping[str, object]] = []
    for layout_row in layout:
        try:
            observed = _inspect_target(layout_row, root, repository, template)
        except (FreezePackageError, OSError) as exc:
            raise CandidateManifestAssemblyError(
                f"cannot inspect {layout_row.artifact_id!r}: {exc}"
            ) from exc
        rows.append(
            _inventory_row(
                template_rows[layout_row.artifact_id],
                layout_row,
                observed,
                artifact_root=root,
                staged_inventory_revision=staged_inventory_revision,
            )
        )
    for layout_row, pinned in zip(layout, rows, strict=True):
        try:
            final = _inspect_target(layout_row, root, repository, template)
        except (FreezePackageError, OSError) as exc:
            raise CandidateManifestAssemblyError(
                f"cannot rehash {layout_row.artifact_id!r}: {exc}"
            ) from exc
        if (
            final.get("state") != "present"
            or final.get("sha256") != pinned["sha256"]
            or final.get("byte_count") != pinned["byte_count"]
            or final.get("file_count") != pinned["file_count"]
            or final.get("directory_count") != pinned["directory_count"]
        ):
            raise CandidateManifestAssemblyError(
                f"controlled artifact {layout_row.artifact_id!r} changed during assembly"
            )
    inventory = CandidateArtifactPinInventory(
        template_sha256=_template_sha256(template),
        artifacts=tuple(rows),
    )
    receipt = {
        "artifact_count": 79,
        "artifact_root": str(root),
        "inventory_file_sha256": inventory.file_sha256,
        "repository_root": str(repository),
        "schema_version": CANDIDATE_ARTIFACT_PIN_RECEIPT_SCHEMA,
        "template_sha256": inventory.template_sha256,
    }
    destination = Path(output_directory).expanduser().resolve(strict=False)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        os.chmod(work, 0o700)
        inventory_bytes = inventory.canonical_file_bytes
        receipt_bytes = _canonical_bytes(receipt)
        _write_private(work / INVENTORY_FILENAME, inventory_bytes)
        _write_private(work / INVENTORY_RECEIPT_FILENAME, receipt_bytes)
        loaded = load_candidate_artifact_pin_inventory(work)
        if loaded != inventory:
            raise CandidateManifestAssemblyError("artifact inventory changed during readback")
        work_descriptor = _open_real_directory(work, label="artifact inventory work directory")
        try:
            _publish_directory_exclusive(
                work,
                destination,
                work_descriptor=work_descriptor,
                expected_members={
                    INVENTORY_FILENAME: inventory_bytes,
                    INVENTORY_RECEIPT_FILENAME: receipt_bytes,
                },
            )
        finally:
            os.close(work_descriptor)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return inventory


def load_candidate_artifact_pin_inventory(
    directory: str | Path,
) -> CandidateArtifactPinInventory:
    root = Path(directory).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    descriptor = _open_real_directory(root, label="artifact inventory directory")
    try:
        observed_names = set(os.listdir(descriptor))
        if observed_names != {INVENTORY_FILENAME, INVENTORY_RECEIPT_FILENAME}:
            raise CandidateManifestAssemblyError("artifact inventory directory membership differs")
        value, _ = _read_json_at(
            descriptor,
            INVENTORY_FILENAME,
            label="artifact inventory",
        )
        receipt_value, _ = _read_json_at(
            descriptor,
            INVENTORY_RECEIPT_FILENAME,
            label="artifact inventory receipt",
        )
    finally:
        os.close(descriptor)
    inventory = CandidateArtifactPinInventory.from_dict(value)
    if _canonical_bytes(value) != inventory.canonical_file_bytes:
        raise CandidateManifestAssemblyError("artifact inventory semantic readback differs")
    receipt = _closed(receipt_value, _RECEIPT_FIELDS, label="artifact inventory receipt")
    if (
        receipt["schema_version"] != CANDIDATE_ARTIFACT_PIN_RECEIPT_SCHEMA
        or receipt["artifact_count"] != 79
        or receipt["inventory_file_sha256"] != inventory.file_sha256
        or receipt["template_sha256"] != inventory.template_sha256
    ):
        raise CandidateManifestAssemblyError("artifact inventory receipt binding differs")
    return inventory


def apply_candidate_artifact_inventory(
    template: Mapping[str, Any],
    inventory: CandidateArtifactPinInventory,
) -> dict[str, Any]:
    """Replace only artifact pins and locators in the tracked structural template."""

    if _template_sha256(template) != inventory.template_sha256:
        raise CandidateManifestAssemblyError("artifact inventory belongs to another template")
    candidate = copy.deepcopy(dict(template))
    rows = candidate.get("artifacts")
    if not isinstance(rows, list) or len(rows) != 79:
        raise CandidateManifestAssemblyError("template artifact cardinality differs")
    by_id = {str(row["artifact_id"]): row for row in inventory.artifacts}
    for template_row in rows:
        artifact_id = str(template_row.get("id"))
        pin = by_id.pop(artifact_id, None)
        if pin is None:
            raise CandidateManifestAssemblyError(f"artifact inventory omits {artifact_id!r}")
        if pin["role"] != template_row.get("role") or pin["corpus_id"] != template_row.get(
            "corpus_id"
        ):
            raise CandidateManifestAssemblyError(
                f"artifact inventory identity differs for {artifact_id!r}"
            )
        if _is_placeholder(template_row.get("uri")):
            parsed = urlsplit(str(pin["uri"]))
            expected_suffix = "/" + str(pin["relative_path"])
            if (
                parsed.scheme != "file"
                or parsed.netloc
                or parsed.query
                or parsed.fragment
                or not unquote(parsed.path).endswith(expected_suffix)
            ):
                raise CandidateManifestAssemblyError(
                    f"derived locator differs from the controlled layout for {artifact_id!r}"
                )
        template_row["uri"] = pin["uri"]
        template_row["revision"] = pin["revision"]
        template_row["sha256"] = pin["sha256"]
        template_row["license"] = pin["license"]
    if by_id:
        raise CandidateManifestAssemblyError("artifact inventory contains extra IDs")
    return candidate


def assert_candidate_manifest_closed(
    candidate: Mapping[str, Any],
    *,
    c0_commit: str,
) -> None:
    """Apply the final candidate gate, including the exact 13 C0 sentinels."""

    if _GIT_COMMIT.fullmatch(c0_commit) is None:
        raise CandidateManifestAssemblyError("C0 commit must be one full lowercase Git commit")
    try:
        validate_candidate_rehearsal_manifest(candidate, c0_commit=c0_commit)
    except StudyManifestError as exc:
        raise CandidateManifestAssemblyError(f"candidate manifest is not closed: {exc}") from exc


def _count_scalar(value: object, target: str) -> int:
    if isinstance(value, Mapping):
        return sum(_count_scalar(item, target) for item in value.values())
    if isinstance(value, list):
        return sum(_count_scalar(item, target) for item in value)
    return int(value == target)


def _cross_check_candidate_image_closure(
    candidate: Mapping[str, Any],
    closure: CandidateImageClosure,
) -> None:
    sealed = candidate.get("sealed_execution")
    if not isinstance(sealed, Mapping):
        raise CandidateManifestAssemblyError("candidate sealed_execution is absent")
    if sealed.get("runner_image") != closure.scientific_image_reference:
        raise CandidateManifestAssemblyError(
            "candidate scientific image differs from P/T/D closure"
        )
    plans = sealed.get("provider_phase_plans")
    if not isinstance(plans, Mapping):
        raise CandidateManifestAssemblyError("candidate provider plans are absent")
    expected = {
        "online": (
            closure.scientific_image_reference,
            closure.scientific_image_index_digest,
            closure.scientific_linux_arm64_manifest_digest,
        ),
        "label-release": (
            closure.release_image_reference,
            closure.release_image_index_digest,
            closure.release_linux_arm64_manifest_digest,
        ),
        "analysis": (
            closure.scientific_image_reference,
            closure.scientific_image_index_digest,
            closure.scientific_linux_amd64_manifest_digest,
        ),
    }
    for phase, (reference, index, platform) in expected.items():
        plan = plans.get(phase)
        if not isinstance(plan, Mapping) or (
            plan.get("runtime_image"),
            plan.get("oci_index_digest"),
            plan.get("oci_platform_manifest_digest"),
        ) != (reference, index, platform):
            raise CandidateManifestAssemblyError(
                f"candidate {phase} image binding differs from P/T/D closure"
            )
    workloads = candidate.get("production_workloads")
    if not isinstance(workloads, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("spec"), Mapping)
        or row["spec"].get("runner_image") != closure.scientific_image_reference
        for row in workloads
    ):
        raise CandidateManifestAssemblyError(
            "candidate production workload image differs from P/T/D closure"
        )


def _cross_check_candidate_artifact_inventory(
    candidate: Mapping[str, Any],
    inventory: CandidateArtifactPinInventory,
) -> None:
    values = candidate.get("artifacts")
    if not isinstance(values, list) or len(values) != 79:
        raise CandidateManifestAssemblyError("candidate artifact multiplicity differs")
    pins = {str(row["artifact_id"]): row for row in inventory.artifacts}
    for value in values:
        if not isinstance(value, Mapping):
            raise CandidateManifestAssemblyError("candidate artifact row is malformed")
        artifact_id = str(value.get("id"))
        pin = pins.pop(artifact_id, None)
        if pin is None:
            raise CandidateManifestAssemblyError(
                f"candidate artifact {artifact_id!r} is absent from its inventory"
            )
        expected = {
            "corpus_id": pin["corpus_id"],
            "license": pin["license"],
            "revision": pin["revision"],
            "role": pin["role"],
            "sha256": pin["sha256"],
            "uri": pin["uri"],
        }
        observed = {name: value.get(name) for name in expected}
        if observed != expected:
            raise CandidateManifestAssemblyError(
                f"candidate artifact {artifact_id!r} differs from its typed inventory"
            )
    if pins:
        raise CandidateManifestAssemblyError("candidate artifact inventory contains extra IDs")


def publish_closed_candidate_manifest(
    *,
    candidate: Mapping[str, Any],
    artifact_inventory: CandidateArtifactPinInventory,
    candidate_image_closure: CandidateImageClosure,
    output_directory: str | Path,
) -> Mapping[str, object]:
    """Publish one validated candidate manifest and its closed receipt atomically."""

    assert_candidate_manifest_closed(
        candidate,
        c0_commit=_CANDIDATE_VALIDATION_PROBE_COMMIT,
    )
    sentinel_count = _count_scalar(candidate, C0_COMMIT_SENTINEL)
    if sentinel_count != 13:
        raise CandidateManifestAssemblyError("candidate must contain exactly 13 C0 sentinels")
    if len(candidate.get("artifacts", ())) != 79:
        raise CandidateManifestAssemblyError("candidate artifact multiplicity differs")
    _cross_check_candidate_artifact_inventory(candidate, artifact_inventory)
    _cross_check_candidate_image_closure(candidate, candidate_image_closure)
    sealed = candidate["sealed_execution"]
    assert isinstance(sealed, Mapping)
    raw_plans = sealed["provider_phase_plans"]
    plan_closure = _sha256(_canonical_bytes(raw_plans)[:-1])
    encoded = _canonical_bytes(candidate)
    typed_receipt = CandidateManifestAssemblyReceipt(
        artifact_count=79,
        artifact_inventory_file_sha256=artifact_inventory.file_sha256,
        build_context_tree_sha256=candidate_image_closure.build_context_tree_sha256,
        candidate_image_closure_file_sha256=candidate_image_closure.file_sha256,
        candidate_image_source_commit=candidate_image_closure.github_sha,
        c0_sentinel_count=sentinel_count,
        manifest_file_sha256=_sha256(encoded),
        manifest_semantic_sha256=manifest_sha256(candidate),
        provider_plan_template_closure_sha256=plan_closure,
        release_image_index_digest=candidate_image_closure.release_image_index_digest,
        scientific_image_index_digest=candidate_image_closure.scientific_image_index_digest,
    )
    receipt = typed_receipt.to_dict()
    destination = Path(output_directory).expanduser().resolve(strict=False)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        os.chmod(work, 0o700)
        _write_private(work / CANDIDATE_MANIFEST_FILENAME, encoded)
        _write_private(work / ASSEMBLY_RECEIPT_FILENAME, _canonical_bytes(receipt))
        manifest_bytes = encoded
        receipt_bytes = _canonical_bytes(receipt)
        work_descriptor = _open_real_directory(work, label="candidate package work directory")
        try:
            observed_manifest, _ = _read_json_at(
                work_descriptor,
                CANDIDATE_MANIFEST_FILENAME,
                label="candidate manifest readback",
            )
            receipt_value, _ = _read_json_at(
                work_descriptor,
                ASSEMBLY_RECEIPT_FILENAME,
                label="candidate assembly receipt readback",
            )
            observed_receipt = _closed(
                receipt_value,
                _ASSEMBLY_FIELDS,
                label="candidate assembly receipt readback",
            )
            if observed_manifest != candidate or observed_receipt != receipt:
                raise CandidateManifestAssemblyError("candidate package changed during readback")
            assert_candidate_manifest_closed(
                observed_manifest,
                c0_commit=_CANDIDATE_VALIDATION_PROBE_COMMIT,
            )
            _publish_directory_exclusive(
                work,
                destination,
                work_descriptor=work_descriptor,
                expected_members={
                    CANDIDATE_MANIFEST_FILENAME: manifest_bytes,
                    ASSEMBLY_RECEIPT_FILENAME: receipt_bytes,
                },
            )
        finally:
            os.close(work_descriptor)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return receipt


def load_closed_candidate_manifest_package(
    directory: str | Path,
) -> ClosedCandidateManifestPackage:
    """Admit the exact private package emitted by ``publish-closed``."""

    root = Path(directory).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise CandidateManifestAssemblyError("candidate package is unavailable") from exc
    if resolved != root:
        raise CandidateManifestAssemblyError("candidate package path cannot contain links")
    descriptor = _open_real_directory(root, label="candidate manifest package")
    try:
        metadata = os.fstat(descriptor)
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise CandidateManifestAssemblyError("candidate package directory mode must equal 0700")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise CandidateManifestAssemblyError(
                "candidate package directory is not owned by the current operator"
            )
        observed_names = set(os.listdir(descriptor))
        expected_names = {CANDIDATE_MANIFEST_FILENAME, ASSEMBLY_RECEIPT_FILENAME}
        if observed_names != expected_names:
            raise CandidateManifestAssemblyError(
                "candidate package membership differs; "
                f"missing={sorted(expected_names - observed_names)}, "
                f"extra={sorted(observed_names - expected_names)}"
            )
        manifest_value, manifest_bytes = _read_json_at(
            descriptor,
            CANDIDATE_MANIFEST_FILENAME,
            label="candidate package manifest",
            required_mode=0o600,
            require_current_owner=True,
        )
        receipt_value, receipt_bytes = _read_json_at(
            descriptor,
            ASSEMBLY_RECEIPT_FILENAME,
            label="candidate package assembly receipt",
            maximum=128 * 1024,
            required_mode=0o600,
            require_current_owner=True,
        )
        named_root = root.lstat()
        if (
            not stat.S_ISDIR(named_root.st_mode)
            or (named_root.st_dev, named_root.st_ino) != (metadata.st_dev, metadata.st_ino)
            or stat.S_IMODE(named_root.st_mode) != 0o700
            or set(os.listdir(descriptor)) != expected_names
        ):
            raise CandidateManifestAssemblyError(
                "candidate package directory changed during admission"
            )
    finally:
        os.close(descriptor)
    if not isinstance(manifest_value, Mapping):
        raise CandidateManifestAssemblyError("candidate package manifest must be one object")
    candidate = dict(manifest_value)
    receipt = CandidateManifestAssemblyReceipt.from_dict(receipt_value)
    assert_candidate_manifest_closed(
        candidate,
        c0_commit=_CANDIDATE_VALIDATION_PROBE_COMMIT,
    )
    sealed = candidate.get("sealed_execution")
    if not isinstance(sealed, Mapping):  # pragma: no cover - closed validator owns this
        raise CandidateManifestAssemblyError("candidate package lacks sealed_execution")
    plans = sealed.get("provider_phase_plans")
    if not isinstance(plans, Mapping):  # pragma: no cover - closed validator owns this
        raise CandidateManifestAssemblyError("candidate package lacks provider plans")
    if (
        receipt_bytes != receipt.canonical_file_bytes()
        or receipt.manifest_file_sha256 != _sha256(manifest_bytes)
        or receipt.manifest_semantic_sha256 != manifest_sha256(candidate)
        or receipt.provider_plan_template_closure_sha256 != _sha256(_canonical_bytes(plans)[:-1])
    ):
        raise CandidateManifestAssemblyError(
            "candidate package manifest differs from its assembly receipt"
        )
    return ClosedCandidateManifestPackage(
        root=root,
        manifest=candidate,
        manifest_bytes=manifest_bytes,
        receipt=receipt,
        receipt_bytes=receipt_bytes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fractal-candidate-manifest-assembler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("artifact-inventory")
    inventory.add_argument("--template", type=Path, required=True)
    inventory.add_argument("--repository-root", type=Path, required=True)
    inventory.add_argument("--artifact-root", type=Path, required=True)
    inventory.add_argument("--output-directory", type=Path, required=True)
    verify = subparsers.add_parser("verify-artifact-inventory")
    verify.add_argument("--directory", type=Path, required=True)
    publish = subparsers.add_parser("publish-closed")
    publish.add_argument("--candidate", type=Path, required=True)
    publish.add_argument("--artifact-inventory", type=Path, required=True)
    publish.add_argument("--candidate-image-closure", type=Path, required=True)
    publish.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "artifact-inventory":
        inventory = build_candidate_artifact_pin_inventory(
            template_path=args.template,
            repository_root=args.repository_root,
            artifact_root=args.artifact_root,
            output_directory=args.output_directory,
        )
        print(f"artifact inventory: {inventory.file_sha256}")
        return 0
    if args.command == "verify-artifact-inventory":
        inventory = load_candidate_artifact_pin_inventory(args.directory)
        print(f"artifact inventory valid: {inventory.file_sha256}")
        return 0
    candidate_path = args.candidate.expanduser()
    if not candidate_path.is_absolute():
        candidate_path = Path.cwd() / candidate_path
    closure_path = args.candidate_image_closure.expanduser()
    if not closure_path.is_absolute():
        closure_path = Path.cwd() / closure_path
    receipt = publish_closed_candidate_manifest(
        candidate=_load_canonical_mapping_secure(candidate_path, label="candidate manifest"),
        artifact_inventory=load_candidate_artifact_pin_inventory(args.artifact_inventory),
        candidate_image_closure=_load_candidate_image_closure_secure(closure_path),
        output_directory=args.output_directory,
    )
    print(f"candidate manifest: {receipt['manifest_file_sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
