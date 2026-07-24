"""Typed, closed transition from the rehearsed candidate manifest to C1."""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .c0_evidence_release import (
    C0EvidenceReleaseError,
    validate_c0_evidence_release_binding,
)
from .candidate_manifest_assembler import (
    ASSEMBLY_RECEIPT_FILENAME,
    CANDIDATE_MANIFEST_ASSEMBLY_RECEIPT_SCHEMA,
    CANDIDATE_MANIFEST_FILENAME,
    CandidateManifestAssemblyError,
    load_closed_candidate_manifest_package,
)
from .execution_claim import (
    ExecutionClaimError,
    assert_normalized_provider_phase_plan_closure,
    provider_phase_plan_templates_sha256,
)
from .study import (
    StudyManifestError,
    manifest_sha256,
    resolve_candidate_provider_plan_commit_bindings,
    validate_candidate_rehearsal_manifest,
    validate_candidate_rehearsal_to_frozen_transition,
    validate_study_manifest,
)

C1_MANIFEST_TRANSITION_RECEIPT_SCHEMA = "fractal-c1-manifest-transition-receipt-v2"
C1_FROZEN_MANIFEST_FILENAME = "study-manifest.json"
C1_MANIFEST_TRANSITION_RECEIPT_FILENAME = "manifest-transition-receipt.json"
MAX_CANDIDATE_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_C0_EVIDENCE_RELEASE_BYTES = 1024 * 1024
MAX_C1_MANIFEST_TRANSITION_RECEIPT_BYTES = 128 * 1024

_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class C1ManifestTransitionError(ValueError):
    """The proposed C1 transition is not the registered closed transition."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise C1ManifestTransitionError(
            "transition evidence must be finite canonical JSON"
        ) from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _decode_object(encoded: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise C1ManifestTransitionError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise C1ManifestTransitionError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C1ManifestTransitionError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise C1ManifestTransitionError(f"{label} must contain one JSON object")
    return value


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise C1ManifestTransitionError(f"{label} must be one string-keyed object")
    observed = set(value)
    if observed != fields:
        raise C1ManifestTransitionError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise C1ManifestTransitionError(f"{name} must be one lowercase SHA-256")
    return value


def _require_commit(name: str, value: object) -> str:
    if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
        raise C1ManifestTransitionError(f"{name} must be one full lowercase Git commit")
    return value


def _canonical_file_uri(value: object, *, label: str) -> Path:
    if type(value) is not str or not value or value != value.strip():
        raise C1ManifestTransitionError(f"{label} must be a canonical file URI")
    parsed = urlsplit(value)
    if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
        raise C1ManifestTransitionError(f"{label} must be a local canonical file URI")
    path = Path(unquote(parsed.path))
    if not path.is_absolute() or path.as_uri() != value:
        raise C1ManifestTransitionError(f"{label} must be a local canonical file URI")
    return path


def _file_stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
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


def _read_regular_file_snapshot(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    """Read bytes and metadata from one descriptor-bound regular-file snapshot."""

    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise C1ManifestTransitionError(f"{label} must be an absolute file path")
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags |= getattr(os, "O_NONBLOCK", 0)
    parent: int | None = None
    descriptor: int | None = None
    try:
        parent = os.open("/", directory_flags)
        for component in path.parent.parts[1:]:
            child = os.open(component, directory_flags, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(path.name, file_flags, dir_fd=parent)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise C1ManifestTransitionError(f"{label} is not one bounded regular file")
        chunks: list[bytes] = []
        observed = 0
        while observed <= max_bytes:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
        if observed > max_bytes or _file_stat_signature(before) != _file_stat_signature(after):
            raise C1ManifestTransitionError(f"{label} changed during read")
        return b"".join(chunks), after
    except C1ManifestTransitionError:
        raise
    except OSError as exc:
        raise C1ManifestTransitionError(f"cannot read {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent is not None:
            os.close(parent)


def _read_canonical_object(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
    required_mode: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    target = Path(path)
    encoded, metadata = _read_regular_file_snapshot(
        target,
        label=label,
        max_bytes=max_bytes,
    )
    if required_mode is not None and stat.S_IMODE(metadata.st_mode) != required_mode:
        raise C1ManifestTransitionError(f"{label} mode must equal {required_mode:04o}")
    value = _decode_object(encoded, label=label)
    if encoded != _canonical_bytes(value) + b"\n":
        raise C1ManifestTransitionError(f"{label} bytes are not canonical")
    return value, encoded


@dataclass(frozen=True)
class C1ManifestTransitionReceipt:
    """Readback evidence for one exact candidate, C0 release, and frozen result."""

    c0_commit: str
    candidate_manifest_package_uri: str
    candidate_manifest_uri: str
    candidate_manifest_sha256: str
    candidate_manifest_file_sha256: str
    candidate_manifest_assembly_receipt_uri: str
    candidate_manifest_assembly_receipt_file_sha256: str
    candidate_manifest_assembly_receipt_schema: str
    c0_evidence_release_uri: str
    c0_evidence_release_sha256: str
    c0_evidence_release_file_sha256: str
    apparatus_evidence_sha256: str
    provider_phase_plan_closure_sha256: str
    frozen_manifest_uri: str
    frozen_manifest_sha256: str
    frozen_manifest_file_sha256: str
    frozen_manifest_byte_count: int
    frozen_manifest_mode: str
    schema_version: str = C1_MANIFEST_TRANSITION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_commit("c0_commit", self.c0_commit)
        for name in (
            "candidate_manifest_sha256",
            "candidate_manifest_file_sha256",
            "candidate_manifest_assembly_receipt_file_sha256",
            "c0_evidence_release_sha256",
            "c0_evidence_release_file_sha256",
            "apparatus_evidence_sha256",
            "provider_phase_plan_closure_sha256",
            "frozen_manifest_sha256",
            "frozen_manifest_file_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "candidate_manifest_package_uri",
            "candidate_manifest_uri",
            "candidate_manifest_assembly_receipt_uri",
            "c0_evidence_release_uri",
            "frozen_manifest_uri",
        ):
            _canonical_file_uri(getattr(self, name), label=name)
        package = _canonical_file_uri(
            self.candidate_manifest_package_uri,
            label="candidate_manifest_package_uri",
        )
        if (
            _canonical_file_uri(
                self.candidate_manifest_uri,
                label="candidate_manifest_uri",
            )
            != package / CANDIDATE_MANIFEST_FILENAME
            or _canonical_file_uri(
                self.candidate_manifest_assembly_receipt_uri,
                label="candidate_manifest_assembly_receipt_uri",
            )
            != package / ASSEMBLY_RECEIPT_FILENAME
        ):
            raise C1ManifestTransitionError(
                "candidate manifest package member paths differ from the closed package"
            )
        if (
            self.candidate_manifest_assembly_receipt_schema
            != CANDIDATE_MANIFEST_ASSEMBLY_RECEIPT_SCHEMA
        ):
            raise C1ManifestTransitionError("candidate assembly receipt schema differs")
        if type(self.frozen_manifest_byte_count) is not int or self.frozen_manifest_byte_count <= 0:
            raise C1ManifestTransitionError("frozen_manifest_byte_count must be a positive integer")
        if self.frozen_manifest_mode != "0600":
            raise C1ManifestTransitionError("frozen_manifest_mode must equal '0600'")
        if self.schema_version != C1_MANIFEST_TRANSITION_RECEIPT_SCHEMA:
            raise C1ManifestTransitionError("C1 manifest transition receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> C1ManifestTransitionReceipt:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="C1 manifest transition receipt",
        )
        return cls(**row)


@dataclass(frozen=True)
class C1ManifestTransitionResult:
    frozen_manifest: Mapping[str, Any]
    receipt: C1ManifestTransitionReceipt
    frozen_manifest_path: Path
    receipt_path: Path


def load_c1_manifest_transition_receipt(
    path: str | Path,
) -> C1ManifestTransitionReceipt:
    """Load one canonical private transition receipt without following links."""

    value, encoded = _read_canonical_object(
        path,
        label="C1 manifest transition receipt",
        max_bytes=MAX_C1_MANIFEST_TRANSITION_RECEIPT_BYTES,
        required_mode=0o600,
    )
    return loads_c1_manifest_transition_receipt(encoded)


def loads_c1_manifest_transition_receipt(encoded: bytes) -> C1ManifestTransitionReceipt:
    """Load canonical receipt bytes independently of their storage mode."""

    if (
        not isinstance(encoded, bytes)
        or len(encoded) > MAX_C1_MANIFEST_TRANSITION_RECEIPT_BYTES
        or not encoded.endswith(b"\n")
        or encoded.endswith(b"\n\n")
    ):
        raise C1ManifestTransitionError("C1 manifest transition receipt bytes are unbounded")
    receipt = C1ManifestTransitionReceipt.from_dict(
        _decode_object(encoded, label="C1 manifest transition receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise C1ManifestTransitionError("C1 manifest transition receipt changed after typing")
    return receipt


def verify_c1_manifest_transition_receipt_bindings(
    receipt: C1ManifestTransitionReceipt,
    *,
    frozen_manifest: Mapping[str, Any],
    frozen_manifest_bytes: bytes,
    c0_commit: str,
) -> None:
    """Bind a retained transition receipt to the exact frozen C1 manifest."""

    commit = _require_commit("c0_commit", c0_commit)
    if not isinstance(receipt, C1ManifestTransitionReceipt):
        raise C1ManifestTransitionError("transition evidence lacks a typed receipt")
    try:
        validate_study_manifest(frozen_manifest, require_frozen=True)
        plan_sha256 = provider_phase_plan_templates_sha256(frozen_manifest)
    except (ExecutionClaimError, StudyManifestError) as exc:
        raise C1ManifestTransitionError(
            f"frozen manifest cannot consume its transition receipt: {exc}"
        ) from exc
    canonical_manifest_bytes = _canonical_bytes(frozen_manifest) + b"\n"
    if frozen_manifest_bytes != canonical_manifest_bytes:
        raise C1ManifestTransitionError("frozen manifest bytes are not canonical")
    sealed = frozen_manifest.get("sealed_execution")
    evidence = sealed.get("c0_evidence_release") if isinstance(sealed, Mapping) else None
    if not isinstance(evidence, Mapping):
        raise C1ManifestTransitionError("frozen manifest lacks C0 evidence release")
    apparatus = evidence.get("apparatus_evidence")
    if not isinstance(apparatus, Mapping):
        raise C1ManifestTransitionError("frozen manifest lacks C0 apparatus evidence")
    if (
        receipt.c0_commit != commit
        or receipt.frozen_manifest_sha256 != manifest_sha256(frozen_manifest)
        or receipt.frozen_manifest_file_sha256 != _sha256(frozen_manifest_bytes)
        or receipt.frozen_manifest_byte_count != len(frozen_manifest_bytes)
        or receipt.candidate_manifest_sha256 != apparatus.get("rehearsal_manifest_sha256")
        or receipt.candidate_manifest_file_sha256 != apparatus.get("candidate_manifest_file_sha256")
        or receipt.candidate_manifest_assembly_receipt_file_sha256
        != apparatus.get("candidate_manifest_assembly_receipt_file_sha256")
        or receipt.c0_evidence_release_sha256 != _sha256(_canonical_bytes(evidence))
        or receipt.c0_evidence_release_file_sha256 != _sha256(_canonical_bytes(evidence) + b"\n")
        or receipt.apparatus_evidence_sha256 != evidence.get("apparatus_evidence_sha256")
        or receipt.provider_phase_plan_closure_sha256 != plan_sha256
        or receipt.provider_phase_plan_closure_sha256
        != apparatus.get("provider_phase_plan_closure_sha256")
    ):
        raise C1ManifestTransitionError("C1 transition receipt differs from the frozen manifest")


def derive_frozen_c1_manifest(
    candidate: Mapping[str, Any],
    *,
    c0_commit: str,
    c0_evidence_release: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Derive the sole registered C1 object without accepting replacement fields."""

    commit = _require_commit("c0_commit", c0_commit)
    try:
        validate_c0_evidence_release_binding(
            c0_evidence_release,
            frozen=True,
            code_commit=commit,
        )
        validate_candidate_rehearsal_manifest(candidate, c0_commit=commit)
        frozen = resolve_candidate_provider_plan_commit_bindings(
            candidate,
            c0_commit=commit,
        )
        frozen["status"] = "frozen"
        frozen["protocol_version"] = "0.3.0"
        frozen["freeze_blockers"] = []
        sealed = frozen.get("sealed_execution")
        if not isinstance(sealed, dict):  # pragma: no cover - candidate validation owns this
            raise C1ManifestTransitionError("candidate manifest lacks sealed_execution")
        sealed["c0_evidence_release"] = copy.deepcopy(dict(c0_evidence_release))
        validate_candidate_rehearsal_to_frozen_transition(
            candidate,
            frozen,
            c0_commit=commit,
        )
        plan_sha256 = assert_normalized_provider_phase_plan_closure(
            candidate,
            frozen,
            c0_commit=commit,
        )
        validate_study_manifest(frozen, require_frozen=True)
    except (
        C0EvidenceReleaseError,
        ExecutionClaimError,
        StudyManifestError,
    ) as exc:
        raise C1ManifestTransitionError(f"C1 manifest transition rejected: {exc}") from exc
    return frozen, plan_sha256


def _absolute_output_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise C1ManifestTransitionError(f"{label} must be an absolute file path")
    if Path(os.path.normpath(os.fspath(path))) != path:
        raise C1ManifestTransitionError(f"{label} must be lexically canonical")
    return path


def _open_private_parent(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open("/", flags)
        for component in path.parent.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        named = path.parent.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino)
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise C1ManifestTransitionError(
                f"{label} parent must be one runner-controlled private directory"
            )
        return descriptor, metadata
    except C1ManifestTransitionError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise C1ManifestTransitionError(f"cannot open {label} parent: {exc}") from exc


def _rename_noreplace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    *,
    label: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        flag = 0x00000004
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        flag = 0x00000001
    else:  # pragma: no cover - supported CI and operator hosts are macOS/Linux
        raise C1ManifestTransitionError(f"exclusive publication is unsupported on {sys.platform!r}")
    if function is None:  # pragma: no cover - registered hosts provide the primitive
        raise C1ManifestTransitionError("exclusive publication primitive is unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    if function(parent_descriptor, source, parent_descriptor, destination, flag) != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise C1ManifestTransitionError(f"{label} already exists")
        raise C1ManifestTransitionError(f"cannot publish {label}: {os.strerror(error_number)}")


def _write_staged_member(
    directory_descriptor: int,
    *,
    name: str,
    encoded: bytes,
    label: str,
) -> tuple[int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
    except OSError as exc:
        raise C1ManifestTransitionError(f"cannot allocate staged {label}: {exc}") from exc
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise C1ManifestTransitionError(f"cannot complete temporary {label}")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(encoded)
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise C1ManifestTransitionError(f"staged {label} is not one private exact file")
        identity = (metadata.st_dev, metadata.st_ino)
        return identity
    finally:
        os.close(descriptor)


def _remove_staging_directory(
    parent_descriptor: int,
    staging_name: str,
    member_names: Sequence[str],
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        staging_descriptor = os.open(staging_name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        return
    try:
        for name in member_names:
            try:
                os.unlink(name, dir_fd=staging_descriptor)
            except FileNotFoundError:
                pass
    finally:
        os.close(staging_descriptor)
    try:
        os.rmdir(staging_name, dir_fd=parent_descriptor)
    except FileNotFoundError:
        pass


def _read_published_member_at(
    directory_descriptor: int,
    *,
    name: str,
    maximum: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise C1ManifestTransitionError(f"cannot open published {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size <= 0
            or before.st_size > maximum
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise C1ManifestTransitionError(f"published {label} is not one private exact file")
        chunks: list[bytes] = []
        observed = 0
        while observed <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
        if observed > maximum or (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise C1ManifestTransitionError(f"published {label} changed during readback")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _atomic_publish_directory_noreplace(
    destination: Path,
    members: Mapping[str, bytes],
) -> Mapping[str, bytes]:
    expected_names = {
        C1_FROZEN_MANIFEST_FILENAME,
        C1_MANIFEST_TRANSITION_RECEIPT_FILENAME,
    }
    if set(members) != expected_names:
        raise C1ManifestTransitionError("C1 publication must contain exactly two fixed members")
    parent_descriptor, parent_metadata = _open_private_parent(
        destination,
        label="C1 transition output",
    )
    staging_name: str | None = None
    staging_descriptor: int | None = None
    published = False
    readback: dict[str, bytes] = {}
    try:
        for _attempt in range(16):
            candidate = f".{destination.name}.tmp-{secrets.token_hex(16)}"
            try:
                os.mkdir(candidate, 0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            staging_name = candidate
            break
        if staging_name is None:
            raise C1ManifestTransitionError("cannot allocate private C1 staging directory")
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        staging_descriptor = os.open(staging_name, directory_flags, dir_fd=parent_descriptor)
        staging_metadata = os.fstat(staging_descriptor)
        if (
            not stat.S_ISDIR(staging_metadata.st_mode)
            or stat.S_IMODE(staging_metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and staging_metadata.st_uid != os.geteuid())
        ):
            raise C1ManifestTransitionError("C1 staging directory is not private")
        identities = {
            name: _write_staged_member(
                staging_descriptor,
                name=name,
                encoded=members[name],
                label=(
                    "frozen C1 manifest"
                    if name == C1_FROZEN_MANIFEST_FILENAME
                    else "C1 manifest transition receipt"
                ),
            )
            for name in sorted(expected_names)
        }
        os.fsync(staging_descriptor)
        if set(os.listdir(staging_descriptor)) != expected_names:
            raise C1ManifestTransitionError("C1 staging directory membership differs")
        os.close(staging_descriptor)
        staging_descriptor = None
        _rename_noreplace_at(
            parent_descriptor,
            staging_name,
            destination.name,
            label="C1 transition directory",
        )
        published = True
        os.fsync(parent_descriptor)
        named_parent = destination.parent.lstat()
        if (parent_metadata.st_dev, parent_metadata.st_ino) != (
            named_parent.st_dev,
            named_parent.st_ino,
        ):
            raise C1ManifestTransitionError("C1 transition parent changed during publication")
        published_descriptor = os.open(
            destination.name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        try:
            published_metadata = os.fstat(published_descriptor)
            if (
                (published_metadata.st_dev, published_metadata.st_ino)
                != (staging_metadata.st_dev, staging_metadata.st_ino)
                or not stat.S_ISDIR(published_metadata.st_mode)
                or stat.S_IMODE(published_metadata.st_mode) != 0o700
                or set(os.listdir(published_descriptor)) != expected_names
            ):
                raise C1ManifestTransitionError("published C1 transition directory differs")
            for name in expected_names:
                metadata = os.stat(name, dir_fd=published_descriptor, follow_symlinks=False)
                if (
                    (metadata.st_dev, metadata.st_ino) != identities[name]
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_size != len(members[name])
                ):
                    raise C1ManifestTransitionError(
                        f"published C1 member {name!r} differs from staging"
                    )
                readback[name] = _read_published_member_at(
                    published_descriptor,
                    name=name,
                    maximum=(
                        MAX_CANDIDATE_MANIFEST_BYTES
                        if name == C1_FROZEN_MANIFEST_FILENAME
                        else MAX_C1_MANIFEST_TRANSITION_RECEIPT_BYTES
                    ),
                    label=name,
                )
                if readback[name] != members[name]:
                    raise C1ManifestTransitionError(
                        f"published C1 member {name!r} bytes differ from staging"
                    )
        finally:
            os.close(published_descriptor)
        final_named_metadata = os.stat(
            destination.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (final_named_metadata.st_dev, final_named_metadata.st_ino) != (
            staging_metadata.st_dev,
            staging_metadata.st_ino,
        ):
            raise C1ManifestTransitionError("published C1 directory changed during readback")
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if staging_name is not None and not published:
            _remove_staging_directory(
                parent_descriptor,
                staging_name,
                tuple(expected_names),
            )
        os.close(parent_descriptor)
    return readback


def write_c1_manifest_transition(
    *,
    candidate_manifest_package_path: str | Path,
    c0_commit: str,
    c0_evidence_release_path: str | Path,
    output_directory: str | Path,
) -> C1ManifestTransitionResult:
    """Validate, derive, exclusively publish, and read back one C1 transition."""

    try:
        candidate_package = load_closed_candidate_manifest_package(candidate_manifest_package_path)
    except CandidateManifestAssemblyError as exc:
        raise C1ManifestTransitionError(f"candidate manifest package is not closed: {exc}") from exc
    candidate_path = candidate_package.root / CANDIDATE_MANIFEST_FILENAME
    assembly_receipt_path = candidate_package.root / ASSEMBLY_RECEIPT_FILENAME
    evidence_path = Path(c0_evidence_release_path)
    destination = _absolute_output_path(output_directory, label="output directory")
    output_path = destination / C1_FROZEN_MANIFEST_FILENAME
    output_receipt_path = destination / C1_MANIFEST_TRANSITION_RECEIPT_FILENAME
    candidate = dict(candidate_package.manifest)
    candidate_bytes = candidate_package.manifest_bytes
    evidence, evidence_bytes = _read_canonical_object(
        evidence_path,
        label="C0 evidence release binding",
        max_bytes=MAX_C0_EVIDENCE_RELEASE_BYTES,
    )
    frozen, plan_sha256 = derive_frozen_c1_manifest(
        candidate,
        c0_commit=c0_commit,
        c0_evidence_release=evidence,
    )
    frozen_bytes = _canonical_bytes(frozen) + b"\n"
    apparatus = evidence["apparatus_evidence"]
    if not isinstance(apparatus, Mapping):  # pragma: no cover - binding validation owns this
        raise C1ManifestTransitionError("C0 evidence binding lacks apparatus evidence")
    receipt = C1ManifestTransitionReceipt(
        c0_commit=c0_commit,
        candidate_manifest_package_uri=candidate_package.root.as_uri(),
        candidate_manifest_uri=candidate_path.as_uri(),
        candidate_manifest_sha256=manifest_sha256(candidate),
        candidate_manifest_file_sha256=_sha256(candidate_bytes),
        candidate_manifest_assembly_receipt_uri=assembly_receipt_path.as_uri(),
        candidate_manifest_assembly_receipt_file_sha256=(candidate_package.receipt.file_sha256),
        candidate_manifest_assembly_receipt_schema=(candidate_package.receipt.schema_version),
        c0_evidence_release_uri=evidence_path.as_uri(),
        c0_evidence_release_sha256=_sha256(_canonical_bytes(evidence)),
        c0_evidence_release_file_sha256=_sha256(evidence_bytes),
        apparatus_evidence_sha256=str(evidence["apparatus_evidence_sha256"]),
        provider_phase_plan_closure_sha256=plan_sha256,
        frozen_manifest_uri=output_path.as_uri(),
        frozen_manifest_sha256=manifest_sha256(frozen),
        frozen_manifest_file_sha256=_sha256(frozen_bytes),
        frozen_manifest_byte_count=len(frozen_bytes),
        frozen_manifest_mode="0600",
    )
    receipt_bytes = receipt.canonical_file_bytes()
    try:
        final_candidate_package = load_closed_candidate_manifest_package(candidate_package.root)
    except CandidateManifestAssemblyError as exc:
        raise C1ManifestTransitionError(
            f"candidate manifest package changed before C1 publication: {exc}"
        ) from exc
    final_candidate = final_candidate_package.manifest
    final_candidate_bytes = final_candidate_package.manifest_bytes
    final_evidence, final_evidence_bytes = _read_canonical_object(
        evidence_path,
        label="C0 evidence release binding final snapshot",
        max_bytes=MAX_C0_EVIDENCE_RELEASE_BYTES,
    )
    if (
        final_candidate_bytes != candidate_bytes
        or final_candidate != candidate
        or final_candidate_package.receipt_bytes != candidate_package.receipt_bytes
        or final_candidate_package.receipt != candidate_package.receipt
    ):
        raise C1ManifestTransitionError("candidate manifest package changed before C1 publication")
    if final_evidence_bytes != evidence_bytes or final_evidence != evidence:
        raise C1ManifestTransitionError("C0 evidence release changed before C1 publication")
    published_readback = _atomic_publish_directory_noreplace(
        destination,
        {
            C1_FROZEN_MANIFEST_FILENAME: frozen_bytes,
            C1_MANIFEST_TRANSITION_RECEIPT_FILENAME: receipt_bytes,
        },
    )

    readback_manifest_bytes = published_readback[C1_FROZEN_MANIFEST_FILENAME]
    readback_receipt_bytes = published_readback[C1_MANIFEST_TRANSITION_RECEIPT_FILENAME]
    readback_manifest = _decode_object(
        readback_manifest_bytes,
        label="frozen C1 manifest descriptor readback",
    )
    if readback_manifest_bytes != _canonical_bytes(readback_manifest) + b"\n":
        raise C1ManifestTransitionError("frozen C1 manifest readback is not canonical")
    readback_receipt = C1ManifestTransitionReceipt.from_dict(
        _decode_object(
            readback_receipt_bytes,
            label="C1 manifest transition receipt descriptor readback",
        )
    )
    if readback_receipt_bytes != readback_receipt.canonical_file_bytes():
        raise C1ManifestTransitionError("C1 manifest transition receipt readback is not canonical")
    if readback_manifest_bytes != frozen_bytes or readback_manifest != frozen:
        raise C1ManifestTransitionError("frozen C1 manifest readback differs from derivation")
    if readback_receipt != receipt:
        raise C1ManifestTransitionError("C1 manifest transition receipt readback differs")
    try:
        validate_candidate_rehearsal_to_frozen_transition(
            candidate,
            readback_manifest,
            c0_commit=c0_commit,
        )
        if (
            provider_phase_plan_templates_sha256(readback_manifest)
            != receipt.provider_phase_plan_closure_sha256
        ):
            raise C1ManifestTransitionError(
                "frozen provider-plan closure differs from the transition receipt"
            )
    except (ExecutionClaimError, StudyManifestError) as exc:
        raise C1ManifestTransitionError(f"C1 readback rejected: {exc}") from exc
    return C1ManifestTransitionResult(
        frozen_manifest=readback_manifest,
        receipt=readback_receipt,
        frozen_manifest_path=output_path,
        receipt_path=output_receipt_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-c1-manifest-transition",
        description="Derive the sole registered frozen C1 manifest from canonical C0 evidence.",
    )
    parser.add_argument("--candidate-package", required=True, type=Path)
    parser.add_argument("--c0-commit", required=True)
    parser.add_argument("--c0-evidence-release", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = write_c1_manifest_transition(
            candidate_manifest_package_path=arguments.candidate_package,
            c0_commit=arguments.c0_commit,
            c0_evidence_release_path=arguments.c0_evidence_release,
            output_directory=arguments.output_directory,
        )
    except C1ManifestTransitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(result.receipt.canonical_file_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
