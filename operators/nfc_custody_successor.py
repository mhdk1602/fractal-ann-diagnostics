#!/usr/bin/env python3
"""Assemble a custody-complete NFC staging successor without opening label payloads.

The operator joins two already-sealed namespaces.  The NFC online projection supplies
the successor inventory, its checksum, and all 86 label-free artifacts.  The original
complete staging root supplies only the 24 qrel/evidence artifacts whose byte identities
are unchanged in the successor inventory.  Payload bytes are streamed as opaque bytes;
the operator computes only byte counts, newline counts, and SHA-256 digests.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import sys
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

RECEIPT_SCHEMA = "fractal-nfc-custody-successor-receipt-v1"
CLI_RESULT_SCHEMA = "fractal-nfc-custody-successor-cli-result-v1"
CUSTODY_CONTRACT = "fractal-exclusive-posix-advisory-custody-v1"
INVENTORY_SCHEMA = "fractal-study-data-inventory-v2"
PROJECTION_SCHEMA = "fractal-online-staging-projection-v1"
PROJECTION_POLICY = "corpus-query-assignment-controls-only-v1"
PROJECTION_RECEIPT_FILENAME = "projection-receipt.json"

EXPECTED_ARTIFACT_COUNT = 110
EXPECTED_PROJECTED_COUNT = 86
EXPECTED_CUSTODY_COUNT = 24
EXPECTED_QREL_COUNT = 15
EXPECTED_EVIDENCE_COUNT = 9

DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_CONTROL_BYTES = 256 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OUTCOME_ROLES = frozenset({"qrels", "evidence-bundles"})
_INVENTORY_FIELDS = frozenset(
    {
        "artifacts",
        "assignment_algorithm",
        "assignment_seed_sha256",
        "bright_document_identity",
        "bright_domains",
        "config_sha256",
        "counts",
        "hotpotqa_fullwiki_scope",
        "schema_version",
        "sources",
        "withhold_sealed_labels_from_online_process",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "byte_count",
        "dataset",
        "path",
        "record_count",
        "role",
        "sha256",
        "stage",
        "visibility",
    }
)
_PROJECTION_FIELDS = frozenset(
    {
        "projected_artifact_count",
        "projected_artifact_set_sha256",
        "projected_artifacts",
        "projection_policy",
        "schema_version",
        "source_artifact_count",
        "source_inventory_sha256",
    }
)
_ORIGIN_FIELDS = frozenset({"artifact", "source"})
_LIMIT_FIELDS = frozenset(
    {
        "copy_chunk_bytes",
        "max_control_bytes",
        "max_total_artifact_bytes",
    }
)
_CUSTODY_FIELDS = frozenset(
    {
        "contract",
        "noncooperating_same_uid_mutation_excluded",
        "producer_directories_and_files_leased_through_publication",
        "publication_parents_exclusively_leased",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "artifact_count",
        "artifacts",
        "custody",
        "custody_artifact_count",
        "custody_artifact_set_sha256",
        "limits",
        "original_inventory_sha256",
        "original_root",
        "output_artifact_set_sha256",
        "output_root",
        "projected_artifact_count",
        "projected_artifact_set_sha256",
        "projection_receipt_sha256",
        "projection_root",
        "receipt_output",
        "schema_version",
        "source_capture_set_sha256",
        "successor_inventory_sha256",
    }
)
_FILE_STABLE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_DIRECTORY_STABLE_FIELDS = _FILE_STABLE_FIELDS
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_DARWIN_ACL_TYPE_EXTENDED = 0x00000100
_DARWIN_ACL_FIRST_ENTRY = 0
_LINUX_POSIX_ACL_XATTRS = frozenset({b"system.posix_acl_access", b"system.posix_acl_default"})


class NfcCustodySuccessorError(RuntimeError):
    """Raised when custody-complete successor assembly cannot be proved."""


class NfcCustodyPublicationIndeterminate(NfcCustodySuccessorError):
    """Raised when an interrupted publication cannot be classified or rolled back."""


class NfcCustodyInterrupted(BaseException):
    """Transaction-visible process-control signal."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"NFC custody-successor assembly interrupted by signal {signum}")


class _SignalGuard:
    def __init__(self) -> None:
        self._previous: dict[int, Any] = {}

    @staticmethod
    def _interrupt(signum: int, _frame: object) -> None:
        raise NfcCustodyInterrupted(signum)

    def __enter__(self) -> _SignalGuard:
        if threading.current_thread() is not threading.main_thread():
            return self
        for signum in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if signum is None:
                continue
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._interrupt)
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> bool:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        return False


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8", errors="strict")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise NfcCustodySuccessorError("control data must be finite canonical JSON") from exc


def _canonical_value_bytes(value: object) -> bytes:
    return _canonical_bytes(value)[:-1]


def _digest(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _decode(encoded: bytes, *, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise NfcCustodySuccessorError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise NfcCustodySuccessorError(f"{label} contains non-finite value {value!r}")

    try:
        return json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise NfcCustodySuccessorError(f"{label} must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise NfcCustodySuccessorError(f"{label} must be JSON: {exc.msg}") from exc


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise NfcCustodySuccessorError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise NfcCustodySuccessorError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _require_sha256(label: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise NfcCustodySuccessorError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_integer(label: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NfcCustodySuccessorError(f"{label} must be an integer >= {minimum}")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise NfcCustodySuccessorError(f"{label} must be canonical non-empty text")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise NfcCustodySuccessorError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise NfcCustodySuccessorError(f"{label} must be a canonical relative POSIX path")
    return value


def _absolute_path(value: str | Path, *, label: str) -> Path:
    text = str(value)
    if not text or "\\" in text or "\x00" in text or text.startswith("//"):
        raise NfcCustodySuccessorError(f"{label} must be an absolute canonical POSIX path")
    path = Path(text)
    if (
        not path.is_absolute()
        or str(path) != text
        or text == "/"
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or unicodedata.normalize("NFC", text) != text
    ):
        raise NfcCustodySuccessorError(f"{label} must be an absolute canonical POSIX path")
    try:
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise NfcCustodySuccessorError(f"cannot resolve {label}: {exc}") from exc
    if resolved != path:
        raise NfcCustodySuccessorError(f"{label} crosses a symbolic-link alias")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _require_nonroot() -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise NfcCustodySuccessorError("NFC custody-successor operator refuses root execution")


def _same_metadata(
    before: os.stat_result,
    after: os.stat_result,
    *,
    fields: Sequence[str],
) -> bool:
    return all(getattr(before, field) == getattr(after, field) for field in fields)


def _require_stable_file(
    before: os.stat_result,
    after: os.stat_result,
    *,
    label: str,
) -> None:
    if not _same_metadata(before, after, fields=_FILE_STABLE_FIELDS):
        raise NfcCustodySuccessorError(f"{label} changed during the operation")


def _require_stable_directory(
    before: os.stat_result,
    after: os.stat_result,
    *,
    label: str,
) -> None:
    if not _same_metadata(before, after, fields=_DIRECTORY_STABLE_FIELDS):
        raise NfcCustodySuccessorError(f"{label} changed during the operation")


def _require_directory(
    metadata: os.stat_result,
    *,
    label: str,
    mode: int,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise NfcCustodySuccessorError(f"{label} must be a real directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise NfcCustodySuccessorError(f"{label} must be owned by the operator identity")
    observed = stat.S_IMODE(metadata.st_mode)
    if observed != mode:
        raise NfcCustodySuccessorError(f"{label} mode must be {mode:04o}, observed {observed:04o}")


def _require_regular(
    metadata: os.stat_result,
    *,
    label: str,
    mode: int,
) -> None:
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise NfcCustodySuccessorError(f"{label} must be a regular file")
    if metadata.st_nlink != 1:
        raise NfcCustodySuccessorError(f"{label} must have exactly one hard link")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise NfcCustodySuccessorError(f"{label} must be owned by the operator identity")
    observed = stat.S_IMODE(metadata.st_mode)
    if observed != mode:
        raise NfcCustodySuccessorError(f"{label} mode must be {mode:04o}, observed {observed:04o}")


def _directory_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _regular_flags(*, writable: bool = False, exclusive: bool = False) -> int:
    flags = (os.O_WRONLY if writable else os.O_RDONLY) | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if not writable and hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if exclusive:
        flags |= os.O_CREAT | os.O_EXCL
    return flags


def _open_absolute_directory(path: Path, *, label: str, mode: int) -> int:
    descriptor: int | None = None
    try:
        descriptor = os.open("/", _directory_flags())
        for component in path.parts[1:]:
            before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise NfcCustodySuccessorError(
                    f"{label} component {component!r} is not a real directory"
                )
            child = os.open(component, _directory_flags(), dir_fd=descriptor)
            opened = os.fstat(child)
            _require_stable_directory(
                before,
                opened,
                label=f"{label} component {component!r}",
            )
            os.close(descriptor)
            descriptor = child
        _require_directory(os.fstat(descriptor), label=label, mode=mode)
        _require_no_extended_acl(descriptor, label=label)
        result = descriptor
        descriptor = None
        return result
    except OSError as exc:
        raise NfcCustodySuccessorError(f"cannot open {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _lock(descriptor: int, *, exclusive: bool, label: str) -> None:
    operation = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
    try:
        fcntl.flock(descriptor, operation)
    except OSError as exc:
        kind = "exclusive" if exclusive else "shared"
        raise NfcCustodySuccessorError(
            f"{label} cannot acquire its cooperative {kind} lease"
        ) from exc


def _require_no_extended_acl(descriptor: int, *, label: str) -> None:
    """Reject descriptor-bound ACLs that make mode-bit custody incomplete."""

    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        acl_functions = ("acl_get_fd_np", "acl_get_entry", "acl_free")
        if not all(hasattr(library, name) for name in acl_functions):
            raise NfcCustodySuccessorError(
                f"{label} ACL inspection is unavailable on this macOS host"
            )
        get_acl = library.acl_get_fd_np
        get_acl.argtypes = [ctypes.c_int, ctypes.c_int]
        get_acl.restype = ctypes.c_void_p
        ctypes.set_errno(0)
        acl = get_acl(descriptor, _DARWIN_ACL_TYPE_EXTENDED)
        if not acl:
            error = ctypes.get_errno()
            if error in {0, errno.ENOENT}:
                return
            raise NfcCustodySuccessorError(
                f"cannot inspect {label} extended ACL: {os.strerror(error)}"
            )
        get_entry = library.acl_get_entry
        get_entry.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        get_entry.restype = ctypes.c_int
        free_acl = library.acl_free
        free_acl.argtypes = [ctypes.c_void_p]
        free_acl.restype = ctypes.c_int
        entry = ctypes.c_void_p()
        try:
            ctypes.set_errno(0)
            result = get_entry(
                ctypes.c_void_p(acl),
                _DARWIN_ACL_FIRST_ENTRY,
                ctypes.byref(entry),
            )
            error = ctypes.get_errno()
            if result == 0:
                raise NfcCustodySuccessorError(f"{label} must not carry an extended ACL")
            if error not in {0, errno.ENOENT}:
                raise NfcCustodySuccessorError(
                    f"cannot inspect {label} extended ACL entries: {os.strerror(error)}"
                )
            return
        finally:
            free_acl(ctypes.c_void_p(acl))

    if sys.platform.startswith("linux"):
        if not hasattr(library, "flistxattr"):
            raise NfcCustodySuccessorError(
                f"{label} POSIX ACL inspection is unavailable on this Linux host"
            )
        list_xattrs = library.flistxattr
        list_xattrs.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        list_xattrs.restype = ctypes.c_ssize_t
        ctypes.set_errno(0)
        size = list_xattrs(descriptor, None, 0)
        if size < 0:
            error = ctypes.get_errno()
            if error in {errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}:
                return
            raise NfcCustodySuccessorError(
                f"cannot inspect {label} extended attributes: {os.strerror(error)}"
            )
        if size == 0:
            return
        names_buffer = ctypes.create_string_buffer(size)
        ctypes.set_errno(0)
        observed_size = list_xattrs(descriptor, names_buffer, size)
        if observed_size < 0:
            error = ctypes.get_errno()
            raise NfcCustodySuccessorError(
                f"cannot read {label} extended attributes: {os.strerror(error)}"
            )
        names = frozenset(names_buffer.raw[:observed_size].split(b"\x00"))
        if names & _LINUX_POSIX_ACL_XATTRS:
            raise NfcCustodySuccessorError(f"{label} must not carry a POSIX ACL")
        return

    raise NfcCustodySuccessorError(
        f"{label} ACL inspection is unsupported on platform {sys.platform!r}"
    )


def _open_root_member(root_descriptor: int, relative: str, *, label: str, mode: int) -> int:
    parts = PurePosixPath(_relative_path(relative, label=label)).parts
    parent = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            _require_directory(before, label=f"{label} parent", mode=0o500)
            child = os.open(component, _directory_flags(), dir_fd=parent)
            _require_stable_directory(before, os.fstat(child), label=f"{label} parent")
            _require_no_extended_acl(child, label=f"{label} parent")
            os.close(parent)
            parent = child
        before = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        _require_regular(before, label=label, mode=mode)
        descriptor = os.open(parts[-1], _regular_flags(), dir_fd=parent)
        opened = os.fstat(descriptor)
        _require_stable_file(before, opened, label=label)
        _require_no_extended_acl(descriptor, label=label)
        _lock(descriptor, exclusive=False, label=label)
        return descriptor
    except OSError as exc:
        raise NfcCustodySuccessorError(f"cannot open {label}: {exc}") from exc
    finally:
        os.close(parent)


def _pread_bytes(descriptor: int, *, maximum: int, label: str) -> bytes:
    metadata = os.fstat(descriptor)
    if metadata.st_size > maximum:
        raise NfcCustodySuccessorError(f"{label} exceeds the registered byte limit")
    chunks: list[bytes] = []
    offset = 0
    while offset < metadata.st_size:
        try:
            chunk = os.pread(descriptor, min(COPY_CHUNK_BYTES, metadata.st_size - offset), offset)
        except OSError as exc:
            raise NfcCustodySuccessorError(f"cannot read {label}: {exc}") from exc
        if not chunk:
            raise NfcCustodySuccessorError(f"{label} ended before its admitted size")
        chunks.append(chunk)
        offset += len(chunk)
    _require_stable_file(metadata, os.fstat(descriptor), label=label)
    return b"".join(chunks)


@dataclass(frozen=True, order=True)
class Artifact:
    path: str
    sha256: str
    byte_count: int
    record_count: int
    dataset: str | None
    stage: str | None
    role: str
    visibility: str

    def __post_init__(self) -> None:
        _relative_path(self.path, label="artifact path")
        _require_sha256("artifact SHA-256", self.sha256)
        _require_integer("artifact byte_count", self.byte_count)
        _require_integer("artifact record_count", self.record_count)
        if self.dataset is not None:
            _require_text("artifact dataset", self.dataset)
        if self.stage is not None:
            _require_text("artifact stage", self.stage)
        _require_text("artifact role", self.role)
        _require_text("artifact visibility", self.visibility)

    @classmethod
    def from_dict(cls, value: object, *, label: str) -> Artifact:
        row = _closed(value, _ARTIFACT_FIELDS, label=label)
        return cls(
            path=row["path"],
            sha256=row["sha256"],
            byte_count=row["byte_count"],
            record_count=row["record_count"],
            dataset=row["dataset"],
            stage=row["stage"],
            role=row["role"],
            visibility=row["visibility"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "dataset": self.dataset,
            "path": self.path,
            "record_count": self.record_count,
            "role": self.role,
            "sha256": self.sha256,
            "stage": self.stage,
            "visibility": self.visibility,
        }

    @property
    def identity_contract(self) -> tuple[str, str | None, str | None, str, str]:
        return (self.path, self.dataset, self.stage, self.role, self.visibility)


def _artifact_rows(value: object, *, label: str) -> tuple[Artifact, ...]:
    if not isinstance(value, list):
        raise NfcCustodySuccessorError(f"{label} must be an array")
    rows = tuple(Artifact.from_dict(item, label=f"{label} row") for item in value)
    if rows != tuple(sorted(rows, key=lambda row: row.path.encode("utf-8"))):
        raise NfcCustodySuccessorError(f"{label} must be bytewise path-sorted")
    if len({row.path for row in rows}) != len(rows):
        raise NfcCustodySuccessorError(f"{label} repeats an artifact path")
    return rows


def _artifact_set_sha256(artifacts: Sequence[Artifact]) -> str:
    return _digest(_canonical_value_bytes([artifact.to_dict() for artifact in artifacts]))


@dataclass(frozen=True, order=True)
class ArtifactOrigin:
    artifact: Artifact
    source: str

    def __post_init__(self) -> None:
        if self.source not in {"nfc-projection", "original-custody"}:
            raise NfcCustodySuccessorError("artifact origin is not registered")
        expected = "original-custody" if self.artifact.role in _OUTCOME_ROLES else "nfc-projection"
        if self.source != expected:
            raise NfcCustodySuccessorError(
                f"artifact {self.artifact.path!r} names a forbidden source role"
            )

    @classmethod
    def from_dict(cls, value: object) -> ArtifactOrigin:
        row = _closed(value, _ORIGIN_FIELDS, label="artifact origin")
        return cls(
            artifact=Artifact.from_dict(row["artifact"], label="origin artifact"),
            source=row["source"],
        )

    def to_dict(self) -> dict[str, object]:
        return {"artifact": self.artifact.to_dict(), "source": self.source}


@dataclass(frozen=True)
class ResourceLimits:
    max_total_artifact_bytes: int
    max_control_bytes: int = MAX_CONTROL_BYTES
    copy_chunk_bytes: int = COPY_CHUNK_BYTES

    def __post_init__(self) -> None:
        _require_integer(
            "max_total_artifact_bytes",
            self.max_total_artifact_bytes,
            minimum=1,
        )
        if self.max_control_bytes != MAX_CONTROL_BYTES:
            raise NfcCustodySuccessorError("control-byte limit differs from the operator contract")
        if self.copy_chunk_bytes != COPY_CHUNK_BYTES:
            raise NfcCustodySuccessorError("copy chunk size differs from the operator contract")

    @classmethod
    def from_dict(cls, value: object) -> ResourceLimits:
        row = _closed(value, _LIMIT_FIELDS, label="resource limits")
        return cls(
            max_total_artifact_bytes=row["max_total_artifact_bytes"],
            max_control_bytes=row["max_control_bytes"],
            copy_chunk_bytes=row["copy_chunk_bytes"],
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "copy_chunk_bytes": self.copy_chunk_bytes,
            "max_control_bytes": self.max_control_bytes,
            "max_total_artifact_bytes": self.max_total_artifact_bytes,
        }


@dataclass(frozen=True)
class CustodyDeclaration:
    contract: str = CUSTODY_CONTRACT
    noncooperating_same_uid_mutation_excluded: bool = True
    producer_directories_and_files_leased_through_publication: bool = True
    publication_parents_exclusively_leased: bool = True

    def __post_init__(self) -> None:
        if self.contract != CUSTODY_CONTRACT:
            raise NfcCustodySuccessorError("custody contract differs")
        if (
            self.noncooperating_same_uid_mutation_excluded is not True
            or self.producer_directories_and_files_leased_through_publication is not True
            or self.publication_parents_exclusively_leased is not True
        ):
            raise NfcCustodySuccessorError("custody declaration weakens a required boundary")

    @classmethod
    def from_dict(cls, value: object) -> CustodyDeclaration:
        row = _closed(value, _CUSTODY_FIELDS, label="custody declaration")
        return cls(
            contract=row["contract"],
            noncooperating_same_uid_mutation_excluded=row[
                "noncooperating_same_uid_mutation_excluded"
            ],
            producer_directories_and_files_leased_through_publication=row[
                "producer_directories_and_files_leased_through_publication"
            ],
            publication_parents_exclusively_leased=row["publication_parents_exclusively_leased"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "noncooperating_same_uid_mutation_excluded": (
                self.noncooperating_same_uid_mutation_excluded
            ),
            "producer_directories_and_files_leased_through_publication": (
                self.producer_directories_and_files_leased_through_publication
            ),
            "publication_parents_exclusively_leased": (self.publication_parents_exclusively_leased),
        }


@dataclass(frozen=True)
class NfcCustodySuccessorReceipt:
    projection_root: Path
    original_root: Path
    output_root: Path
    receipt_output: Path
    successor_inventory_sha256: str
    original_inventory_sha256: str
    projection_receipt_sha256: str
    projected_artifact_set_sha256: str
    custody_artifact_set_sha256: str
    output_artifact_set_sha256: str
    source_capture_set_sha256: str
    artifacts: tuple[ArtifactOrigin, ...]
    limits: ResourceLimits
    custody: CustodyDeclaration = CustodyDeclaration()
    artifact_count: int = EXPECTED_ARTIFACT_COUNT
    projected_artifact_count: int = EXPECTED_PROJECTED_COUNT
    custody_artifact_count: int = EXPECTED_CUSTODY_COUNT
    schema_version: str = RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("projection_root", "original_root", "output_root", "receipt_output"):
            object.__setattr__(self, name, _absolute_path(getattr(self, name), label=name))
        for name in (
            "successor_inventory_sha256",
            "original_inventory_sha256",
            "projection_receipt_sha256",
            "projected_artifact_set_sha256",
            "custody_artifact_set_sha256",
            "output_artifact_set_sha256",
            "source_capture_set_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.schema_version != RECEIPT_SCHEMA:
            raise NfcCustodySuccessorError("receipt schema differs")
        if (
            self.artifact_count != EXPECTED_ARTIFACT_COUNT
            or self.projected_artifact_count != EXPECTED_PROJECTED_COUNT
            or self.custody_artifact_count != EXPECTED_CUSTODY_COUNT
        ):
            raise NfcCustodySuccessorError("receipt artifact cardinalities differ")
        artifacts = tuple(self.artifacts)
        if len(artifacts) != EXPECTED_ARTIFACT_COUNT:
            raise NfcCustodySuccessorError("receipt does not contain 110 artifact origins")
        if artifacts != tuple(sorted(artifacts, key=lambda row: row.artifact.path.encode("utf-8"))):
            raise NfcCustodySuccessorError("receipt artifact origins are not bytewise path-sorted")
        if len({origin.artifact.path for origin in artifacts}) != len(artifacts):
            raise NfcCustodySuccessorError("receipt repeats an artifact path")
        projected = tuple(
            origin.artifact for origin in artifacts if origin.source == "nfc-projection"
        )
        custody = tuple(
            origin.artifact for origin in artifacts if origin.source == "original-custody"
        )
        if len(projected) != EXPECTED_PROJECTED_COUNT or len(custody) != EXPECTED_CUSTODY_COUNT:
            raise NfcCustodySuccessorError("receipt source split differs")
        if _artifact_set_sha256(projected) != self.projected_artifact_set_sha256:
            raise NfcCustodySuccessorError("receipt projected artifact-set digest differs")
        if _artifact_set_sha256(custody) != self.custody_artifact_set_sha256:
            raise NfcCustodySuccessorError("receipt custody artifact-set digest differs")
        all_artifacts = tuple(origin.artifact for origin in artifacts)
        if _artifact_set_sha256(all_artifacts) != self.output_artifact_set_sha256:
            raise NfcCustodySuccessorError("receipt output artifact-set digest differs")
        if not isinstance(self.limits, ResourceLimits):
            raise NfcCustodySuccessorError("receipt limits must be typed")
        if not isinstance(self.custody, CustodyDeclaration):
            raise NfcCustodySuccessorError("receipt custody declaration must be typed")
        if any(
            _paths_overlap(self.receipt_output, root)
            for root in (self.projection_root, self.original_root, self.output_root)
        ):
            raise NfcCustodySuccessorError("receipt must remain outside every staged root")
        object.__setattr__(self, "artifacts", artifacts)

    @classmethod
    def from_dict(cls, value: object) -> NfcCustodySuccessorReceipt:
        row = _closed(value, _RECEIPT_FIELDS, label="NFC custody-successor receipt")
        values = row["artifacts"]
        if not isinstance(values, list):
            raise NfcCustodySuccessorError("receipt artifacts must be an array")
        return cls(
            projection_root=Path(row["projection_root"]),
            original_root=Path(row["original_root"]),
            output_root=Path(row["output_root"]),
            receipt_output=Path(row["receipt_output"]),
            successor_inventory_sha256=row["successor_inventory_sha256"],
            original_inventory_sha256=row["original_inventory_sha256"],
            projection_receipt_sha256=row["projection_receipt_sha256"],
            projected_artifact_set_sha256=row["projected_artifact_set_sha256"],
            custody_artifact_set_sha256=row["custody_artifact_set_sha256"],
            output_artifact_set_sha256=row["output_artifact_set_sha256"],
            source_capture_set_sha256=row["source_capture_set_sha256"],
            artifacts=tuple(ArtifactOrigin.from_dict(item) for item in values),
            limits=ResourceLimits.from_dict(row["limits"]),
            custody=CustodyDeclaration.from_dict(row["custody"]),
            artifact_count=row["artifact_count"],
            projected_artifact_count=row["projected_artifact_count"],
            custody_artifact_count=row["custody_artifact_count"],
            schema_version=row["schema_version"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_count": self.artifact_count,
            "artifacts": [origin.to_dict() for origin in self.artifacts],
            "custody": self.custody.to_dict(),
            "custody_artifact_count": self.custody_artifact_count,
            "custody_artifact_set_sha256": self.custody_artifact_set_sha256,
            "limits": self.limits.to_dict(),
            "original_inventory_sha256": self.original_inventory_sha256,
            "original_root": str(self.original_root),
            "output_artifact_set_sha256": self.output_artifact_set_sha256,
            "output_root": str(self.output_root),
            "projected_artifact_count": self.projected_artifact_count,
            "projected_artifact_set_sha256": self.projected_artifact_set_sha256,
            "projection_receipt_sha256": self.projection_receipt_sha256,
            "projection_root": str(self.projection_root),
            "receipt_output": str(self.receipt_output),
            "schema_version": self.schema_version,
            "source_capture_set_sha256": self.source_capture_set_sha256,
            "successor_inventory_sha256": self.successor_inventory_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return _digest(self.canonical_file_bytes())


@dataclass
class _CapturedTree:
    root: Path
    root_descriptor: int
    root_before: os.stat_result
    files: dict[str, int]
    directories: list[int]
    file_metadata: dict[str, os.stat_result]
    directory_metadata: dict[int, os.stat_result]

    def close(self) -> None:
        first_error: BaseException | None = None
        for descriptor in [*self.files.values(), *self.directories, self.root_descriptor]:
            try:
                os.close(descriptor)
            except BaseException as exc:  # pragma: no cover - close failure is OS-specific
                if first_error is None:
                    first_error = exc
        self.files.clear()
        self.directories.clear()
        if first_error is not None:
            raise NfcCustodySuccessorError(
                "cannot release captured-tree descriptors"
            ) from first_error


def _expected_directories(files: set[str]) -> set[str]:
    return {
        parent.as_posix()
        for path in files
        for parent in PurePosixPath(path).parents
        if parent.as_posix() != "."
    }


def _capture_exact_tree(
    *,
    root: Path,
    root_descriptor: int,
    root_before: os.stat_result,
    expected_files: set[str],
    preopened: Mapping[str, int],
    label: str,
    directory_mode: int = 0o500,
    file_mode: int = 0o400,
) -> _CapturedTree:
    expected_directories = _expected_directories(expected_files)
    files: dict[str, int] = {}
    directories: list[int] = []
    file_metadata: dict[str, os.stat_result] = {}
    directory_metadata: dict[int, os.stat_result] = {}
    preopened_remaining = dict(preopened)
    observed_directories: set[str] = set()
    observed_inodes: set[tuple[int, int]] = set()

    def scan(descriptor: int, prefix: str) -> None:
        before_directory = os.fstat(descriptor)
        try:
            names = os.listdir(descriptor)
        except OSError as exc:
            raise NfcCustodySuccessorError(f"cannot enumerate {label}: {exc}") from exc
        if not all(
            isinstance(name, str)
            and name not in {"", ".", ".."}
            and "/" not in name
            and "\\" not in name
            and unicodedata.normalize("NFC", name) == name
            for name in names
        ):
            raise NfcCustodySuccessorError(f"{label} contains a noncanonical entry name")
        for name in sorted(names, key=lambda item: item.encode("utf-8")):
            relative = f"{prefix}/{name}" if prefix else name
            _relative_path(relative, label=f"{label} member")
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                if relative not in expected_directories:
                    raise NfcCustodySuccessorError(
                        f"{label} contains unexpected directory {relative!r}"
                    )
                _require_directory(
                    metadata,
                    label=f"{label} directory {relative!r}",
                    mode=directory_mode,
                )
                child = os.open(name, _directory_flags(), dir_fd=descriptor)
                try:
                    opened = os.fstat(child)
                    _require_stable_directory(
                        metadata,
                        opened,
                        label=f"{label} directory {relative!r}",
                    )
                    _require_no_extended_acl(
                        child,
                        label=f"{label} directory {relative!r}",
                    )
                    _lock(child, exclusive=False, label=f"{label} directory {relative!r}")
                    scan(child, relative)
                    _require_stable_directory(
                        opened,
                        os.fstat(child),
                        label=f"{label} directory {relative!r}",
                    )
                    directories.append(child)
                    directory_metadata[child] = opened
                    child = -1
                finally:
                    if child >= 0:
                        os.close(child)
                observed_directories.add(relative)
                continue
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise NfcCustodySuccessorError(
                    f"{label} contains linked or special member {relative!r}"
                )
            if relative not in expected_files:
                raise NfcCustodySuccessorError(f"{label} contains unexpected file {relative!r}")
            _require_regular(
                metadata,
                label=f"{label} file {relative!r}",
                mode=file_mode,
            )
            opened_descriptor = preopened_remaining.pop(relative, None)
            if opened_descriptor is None:
                opened_descriptor = os.open(name, _regular_flags(), dir_fd=descriptor)
                _lock(
                    opened_descriptor,
                    exclusive=False,
                    label=f"{label} file {relative!r}",
                )
            opened = os.fstat(opened_descriptor)
            _require_stable_file(
                metadata,
                opened,
                label=f"{label} file {relative!r}",
            )
            _require_no_extended_acl(
                opened_descriptor,
                label=f"{label} file {relative!r}",
            )
            inode = (opened.st_dev, opened.st_ino)
            if inode in observed_inodes:
                raise NfcCustodySuccessorError(f"{label} aliases an artifact inode")
            observed_inodes.add(inode)
            files[relative] = opened_descriptor
            file_metadata[relative] = opened
        _require_stable_directory(before_directory, os.fstat(descriptor), label=label)

    try:
        scan(root_descriptor, "")
        if preopened_remaining:
            raise NfcCustodySuccessorError(
                f"{label} omitted pre-opened controls {sorted(preopened_remaining)}"
            )
        if set(files) != expected_files or observed_directories != expected_directories:
            raise NfcCustodySuccessorError(
                f"{label} membership differs; missing_files={sorted(expected_files - set(files))}, "
                f"missing_directories={sorted(expected_directories - observed_directories)}"
            )
        _require_stable_directory(root_before, os.fstat(root_descriptor), label=f"{label} root")
        bound = os.stat(root, follow_symlinks=False)
        _require_stable_directory(root_before, bound, label=f"{label} root path")
        return _CapturedTree(
            root=root,
            root_descriptor=root_descriptor,
            root_before=root_before,
            files=files,
            directories=directories,
            file_metadata=file_metadata,
            directory_metadata=directory_metadata,
        )
    except BaseException:
        for descriptor in {*files.values(), *preopened_remaining.values(), *directories}:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.close(root_descriptor)
        except OSError:
            pass
        raise


def _require_capture_stable(capture: _CapturedTree, *, label: str) -> None:
    _require_stable_directory(
        capture.root_before,
        os.fstat(capture.root_descriptor),
        label=f"{label} root",
    )
    _require_stable_directory(
        capture.root_before,
        os.stat(capture.root, follow_symlinks=False),
        label=f"{label} root path",
    )
    for path, descriptor in capture.files.items():
        _require_stable_file(
            capture.file_metadata[path],
            os.fstat(descriptor),
            label=f"{label} file {path!r}",
        )
    for descriptor in capture.directories:
        _require_stable_directory(
            capture.directory_metadata[descriptor],
            os.fstat(descriptor),
            label=f"{label} directory",
        )


def _fingerprint_descriptor(
    descriptor: int,
    *,
    label: str,
    expected_size: int | None = None,
) -> tuple[str, int, int]:
    before = os.fstat(descriptor)
    if expected_size is not None and before.st_size != expected_size:
        raise NfcCustodySuccessorError(f"{label} differs from its inventory before reading")
    hasher = hashlib.sha256()
    byte_count = 0
    record_count = 0
    while byte_count < before.st_size:
        try:
            chunk = os.pread(
                descriptor,
                min(COPY_CHUNK_BYTES, before.st_size - byte_count),
                byte_count,
            )
        except OSError as exc:
            raise NfcCustodySuccessorError(f"cannot fingerprint {label}: {exc}") from exc
        if not chunk:
            raise NfcCustodySuccessorError(f"{label} ended before its admitted size")
        hasher.update(chunk)
        byte_count += len(chunk)
        record_count += chunk.count(b"\n")
    _require_stable_file(before, os.fstat(descriptor), label=label)
    return hasher.hexdigest(), byte_count, record_count


def _validate_fingerprints(
    capture: _CapturedTree,
    artifacts: Sequence[Artifact],
    *,
    label: str,
    max_total_bytes: int,
) -> None:
    total = sum(artifact.byte_count for artifact in artifacts)
    if total > max_total_bytes:
        raise NfcCustodySuccessorError(f"{label} exceeds max_total_artifact_bytes")
    for artifact in artifacts:
        observed = _fingerprint_descriptor(
            capture.files[artifact.path],
            label=f"{label} artifact {artifact.path!r}",
            expected_size=artifact.byte_count,
        )
        if observed != (artifact.sha256, artifact.byte_count, artifact.record_count):
            raise NfcCustodySuccessorError(
                f"{label} artifact {artifact.path!r} differs from its inventory"
            )


@dataclass
class _AdmittedSources:
    projection: _CapturedTree
    original: _CapturedTree
    successor_inventory_bytes: bytes
    successor_checksum_bytes: bytes
    successor_artifacts: tuple[Artifact, ...]
    projected_artifacts: tuple[Artifact, ...]
    custody_artifacts: tuple[Artifact, ...]
    projected_artifact_set_sha256: str
    projection_receipt_sha256: str

    def close(self) -> None:
        first_error: BaseException | None = None
        for capture in (self.original, self.projection):
            try:
                capture.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _open_source_root(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    descriptor = _open_absolute_directory(path, label=label, mode=0o500)
    try:
        _lock(descriptor, exclusive=False, label=label)
        return descriptor, os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def _parse_inventory(
    encoded: bytes,
    *,
    expected_sha256: str,
    checksum: bytes,
    label: str,
) -> tuple[Mapping[str, Any], tuple[Artifact, ...]]:
    if _digest(encoded) != expected_sha256:
        raise NfcCustodySuccessorError(f"{label} differs from its caller pin")
    if checksum != f"{expected_sha256}  inventory.json\n".encode("ascii"):
        raise NfcCustodySuccessorError(f"{label} checksum differs")
    value = _decode(encoded, label=label)
    inventory = _closed(value, _INVENTORY_FIELDS, label=label)
    if (
        inventory["schema_version"] != INVENTORY_SCHEMA
        or inventory["withhold_sealed_labels_from_online_process"] is not True
        or encoded != _canonical_bytes(value)
    ):
        raise NfcCustodySuccessorError(f"{label} protocol or canonical bytes differ")
    artifacts = _artifact_rows(inventory["artifacts"], label=f"{label} artifacts")
    if len(artifacts) != EXPECTED_ARTIFACT_COUNT:
        raise NfcCustodySuccessorError(f"{label} must name exactly 110 artifacts")
    return inventory, artifacts


def _read_preopened_control(
    root_descriptor: int,
    name: str,
    *,
    label: str,
    maximum: int,
) -> tuple[int, bytes]:
    descriptor = _open_root_member(root_descriptor, name, label=label, mode=0o400)
    try:
        return descriptor, _pread_bytes(descriptor, maximum=maximum, label=label)
    except BaseException:
        os.close(descriptor)
        raise


def _admit_sources(
    *,
    projection_root: Path,
    successor_inventory_sha256: str,
    projection_receipt_sha256: str,
    original_root: Path,
    original_inventory_sha256: str,
    max_total_bytes: int,
) -> _AdmittedSources:
    projection_root_descriptor, projection_root_before = _open_source_root(
        projection_root,
        label="NFC projection root",
    )
    projection_controls: dict[str, int] = {}
    projection_capture: _CapturedTree | None = None
    original_capture: _CapturedTree | None = None
    original_root_descriptor: int | None = None
    original_controls: dict[str, int] = {}
    try:
        projection_inventory_fd, successor_inventory_bytes = _read_preopened_control(
            projection_root_descriptor,
            "inventory.json",
            label="successor inventory",
            maximum=MAX_CONTROL_BYTES,
        )
        projection_controls["inventory.json"] = projection_inventory_fd
        projection_checksum_fd, successor_checksum_bytes = _read_preopened_control(
            projection_root_descriptor,
            "inventory.sha256",
            label="successor inventory checksum",
            maximum=1024,
        )
        projection_controls["inventory.sha256"] = projection_checksum_fd
        projection_receipt_fd, projection_receipt_bytes = _read_preopened_control(
            projection_root_descriptor,
            PROJECTION_RECEIPT_FILENAME,
            label="projection receipt",
            maximum=MAX_CONTROL_BYTES,
        )
        projection_controls[PROJECTION_RECEIPT_FILENAME] = projection_receipt_fd
        _, successor_artifacts = _parse_inventory(
            successor_inventory_bytes,
            expected_sha256=successor_inventory_sha256,
            checksum=successor_checksum_bytes,
            label="successor inventory",
        )

        if _digest(projection_receipt_bytes) != projection_receipt_sha256:
            raise NfcCustodySuccessorError("projection receipt differs from its caller pin")
        projection_value = _decode(projection_receipt_bytes, label="projection receipt")
        projection = _closed(
            projection_value,
            _PROJECTION_FIELDS,
            label="projection receipt",
        )
        projected_artifacts = _artifact_rows(
            projection["projected_artifacts"],
            label="projection receipt artifacts",
        )
        successor_by_path = {artifact.path: artifact for artifact in successor_artifacts}
        expected_projected = tuple(
            artifact for artifact in successor_artifacts if artifact.role not in _OUTCOME_ROLES
        )
        custody_artifacts = tuple(
            artifact for artifact in successor_artifacts if artifact.role in _OUTCOME_ROLES
        )
        outcome_counts = {
            role: sum(artifact.role == role for artifact in custody_artifacts)
            for role in _OUTCOME_ROLES
        }
        projected_set_sha256 = _artifact_set_sha256(projected_artifacts)
        if (
            projection["schema_version"] != PROJECTION_SCHEMA
            or projection["projection_policy"] != PROJECTION_POLICY
            or projection["source_inventory_sha256"] != successor_inventory_sha256
            or projection["source_artifact_count"] != EXPECTED_ARTIFACT_COUNT
            or projection["projected_artifact_count"] != EXPECTED_PROJECTED_COUNT
            or len(projected_artifacts) != EXPECTED_PROJECTED_COUNT
            or len(custody_artifacts) != EXPECTED_CUSTODY_COUNT
            or outcome_counts
            != {
                "qrels": EXPECTED_QREL_COUNT,
                "evidence-bundles": EXPECTED_EVIDENCE_COUNT,
            }
            or projected_artifacts != expected_projected
            or projected_set_sha256 != projection["projected_artifact_set_sha256"]
            or projection_receipt_bytes != _canonical_bytes(projection_value)
            or any(
                successor_by_path.get(artifact.path) != artifact for artifact in projected_artifacts
            )
        ):
            raise NfcCustodySuccessorError("projection receipt or 86/24 source split differs")

        projection_expected_files = {
            "inventory.json",
            "inventory.sha256",
            PROJECTION_RECEIPT_FILENAME,
            *(artifact.path for artifact in projected_artifacts),
        }
        capture_root_descriptor = projection_root_descriptor
        capture_controls = projection_controls
        projection_root_descriptor = -1
        projection_controls = {}
        projection_capture = _capture_exact_tree(
            root=projection_root,
            root_descriptor=capture_root_descriptor,
            root_before=projection_root_before,
            expected_files=projection_expected_files,
            preopened=capture_controls,
            label="NFC projection root",
        )
        _validate_fingerprints(
            projection_capture,
            projected_artifacts,
            label="NFC projection",
            max_total_bytes=max_total_bytes,
        )

        original_root_descriptor, original_root_before = _open_source_root(
            original_root,
            label="original complete root",
        )
        original_inventory_fd, original_inventory_bytes = _read_preopened_control(
            original_root_descriptor,
            "inventory.json",
            label="original inventory",
            maximum=MAX_CONTROL_BYTES,
        )
        original_controls["inventory.json"] = original_inventory_fd
        original_checksum_fd, original_checksum_bytes = _read_preopened_control(
            original_root_descriptor,
            "inventory.sha256",
            label="original inventory checksum",
            maximum=1024,
        )
        original_controls["inventory.sha256"] = original_checksum_fd
        _, original_artifacts = _parse_inventory(
            original_inventory_bytes,
            expected_sha256=original_inventory_sha256,
            checksum=original_checksum_bytes,
            label="original inventory",
        )
        original_by_path = {artifact.path: artifact for artifact in original_artifacts}
        if set(original_by_path) != set(successor_by_path):
            raise NfcCustodySuccessorError("original and successor artifact paths differ")
        for path, successor in successor_by_path.items():
            original = original_by_path[path]
            if original.identity_contract != successor.identity_contract:
                raise NfcCustodySuccessorError(f"original artifact contract differs at {path!r}")
            if successor.role in _OUTCOME_ROLES and original != successor:
                raise NfcCustodySuccessorError(
                    f"original custody artifact differs from successor pin at {path!r}"
                )

        original_expected_files = {
            "inventory.json",
            "inventory.sha256",
            *(artifact.path for artifact in original_artifacts),
        }
        capture_root_descriptor = original_root_descriptor
        capture_controls = original_controls
        original_root_descriptor = None
        original_controls = {}
        original_capture = _capture_exact_tree(
            root=original_root,
            root_descriptor=capture_root_descriptor,
            root_before=original_root_before,
            expected_files=original_expected_files,
            preopened=capture_controls,
            label="original complete root",
        )
        _validate_fingerprints(
            original_capture,
            original_artifacts,
            label="original complete root",
            max_total_bytes=max_total_bytes,
        )
        return _AdmittedSources(
            projection=projection_capture,
            original=original_capture,
            successor_inventory_bytes=successor_inventory_bytes,
            successor_checksum_bytes=successor_checksum_bytes,
            successor_artifacts=successor_artifacts,
            projected_artifacts=projected_artifacts,
            custody_artifacts=custody_artifacts,
            projected_artifact_set_sha256=projected_set_sha256,
            projection_receipt_sha256=projection_receipt_sha256,
        )
    except BaseException:
        if original_capture is not None:
            original_capture.close()
        else:
            for descriptor in original_controls.values():
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if original_root_descriptor is not None:
                os.close(original_root_descriptor)
        if projection_capture is not None:
            projection_capture.close()
        else:
            for descriptor in projection_controls.values():
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if projection_root_descriptor >= 0:
                os.close(projection_root_descriptor)
        raise


def _source_capture_set_sha256(
    *,
    origins: Sequence[ArtifactOrigin],
    successor_inventory_sha256: str,
    original_inventory_sha256: str,
    projection_receipt_sha256: str,
) -> str:
    return _digest(
        _canonical_value_bytes(
            {
                "artifact_origins": [origin.to_dict() for origin in origins],
                "original_inventory_sha256": original_inventory_sha256,
                "projection_receipt_sha256": projection_receipt_sha256,
                "successor_inventory_sha256": successor_inventory_sha256,
            }
        )
    )


def _origins(admitted: _AdmittedSources) -> tuple[ArtifactOrigin, ...]:
    return tuple(
        ArtifactOrigin(
            artifact=artifact,
            source=("original-custody" if artifact.role in _OUTCOME_ROLES else "nfc-projection"),
        )
        for artifact in admitted.successor_artifacts
    )


def _open_writable_parent(path: Path, *, label: str) -> int:
    descriptor = _open_absolute_directory(path, label=label, mode=0o700)
    _lock(descriptor, exclusive=True, label=label)
    return descriptor


def _entry_metadata(parent_descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _classify_no_replace_move(
    *,
    source_parent: int,
    source_name: str,
    destination_parent: int,
    destination_name: str,
    expected: os.stat_result,
    label: str,
) -> bool:
    """Return True only when the expected inode is solely at the destination."""

    observations: list[tuple[os.stat_result | None, os.stat_result | None]] = []
    for _pass in range(2):
        observations.append(
            (
                _entry_metadata(source_parent, source_name),
                _entry_metadata(destination_parent, destination_name),
            )
        )
    first, second = observations

    def equivalent(
        left: os.stat_result | None,
        right: os.stat_result | None,
    ) -> bool:
        if left is None or right is None:
            return left is right
        return _same_metadata(left, right, fields=_FILE_STABLE_FIELDS)

    if not equivalent(first[0], second[0]) or not equivalent(first[1], second[1]):
        raise NfcCustodyPublicationIndeterminate(
            f"{label} names changed while publication state was observed"
        )
    source, destination = second
    at_source = source is not None and _same_inode(source, expected)
    at_destination = destination is not None and _same_inode(destination, expected)
    if at_source and destination is None:
        return False
    if source is None and at_destination:
        return True
    raise NfcCustodyPublicationIndeterminate(
        f"{label} left an unclassified source/destination name state"
    )


def _create_temporary_directory(parent_descriptor: int, *, stem: str) -> tuple[str, int]:
    for _attempt in range(64):
        name = f".{stem}.nfc-custody-{secrets.token_hex(16)}.tmp"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        descriptor: int | None = None
        try:
            os.chmod(name, 0o700, dir_fd=parent_descriptor, follow_symlinks=False)
            descriptor = os.open(name, _directory_flags(), dir_fd=parent_descriptor)
            _require_directory(os.fstat(descriptor), label="temporary output root", mode=0o700)
            _require_no_extended_acl(descriptor, label="temporary output root")
            return name, descriptor
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.rmdir(name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise
    raise NfcCustodySuccessorError("cannot allocate a unique temporary output directory")


def _open_or_create_target_parent(root_descriptor: int, relative: str) -> tuple[int, str]:
    parts = PurePosixPath(_relative_path(relative, label="output artifact path")).parts
    descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            _require_directory(metadata, label="temporary output directory", mode=0o700)
            child = os.open(component, _directory_flags(), dir_fd=descriptor)
            _require_stable_directory(metadata, os.fstat(child), label="temporary output directory")
            _require_no_extended_acl(child, label="temporary output directory")
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, encoded: bytes, *, label: str) -> None:
    view = memoryview(encoded)
    while view:
        try:
            written = os.write(descriptor, view)
        except OSError as exc:
            raise NfcCustodySuccessorError(f"cannot write {label}: {exc}") from exc
        if written <= 0:
            raise NfcCustodySuccessorError(f"short write while publishing {label}")
        view = view[written:]


def _write_control(root_descriptor: int, relative: str, encoded: bytes) -> None:
    parent, name = _open_or_create_target_parent(root_descriptor, relative)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            _regular_flags(writable=True, exclusive=True),
            0o600,
            dir_fd=parent,
        )
        os.fchmod(descriptor, 0o600)
        _require_no_extended_acl(descriptor, label=f"temporary control {relative!r}")
        _write_all(descriptor, encoded, label=relative)
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise NfcCustodySuccessorError(f"temporary output repeats {relative!r}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _copy_pinned_file(
    *,
    source_descriptor: int,
    target_root_descriptor: int,
    artifact: Artifact,
) -> None:
    source_before = os.fstat(source_descriptor)
    if source_before.st_size != artifact.byte_count:
        raise NfcCustodySuccessorError(f"source artifact {artifact.path!r} changed before copy")
    parent, name = _open_or_create_target_parent(target_root_descriptor, artifact.path)
    target: int | None = None
    hasher = hashlib.sha256()
    byte_count = 0
    record_count = 0
    try:
        target = os.open(
            name,
            _regular_flags(writable=True, exclusive=True),
            0o600,
            dir_fd=parent,
        )
        os.fchmod(target, 0o600)
        _require_no_extended_acl(target, label=f"temporary artifact {artifact.path!r}")
        while byte_count < source_before.st_size:
            try:
                chunk = os.pread(
                    source_descriptor,
                    min(COPY_CHUNK_BYTES, source_before.st_size - byte_count),
                    byte_count,
                )
            except OSError as exc:
                raise NfcCustodySuccessorError(
                    f"cannot read source artifact {artifact.path!r}"
                ) from exc
            if not chunk:
                raise NfcCustodySuccessorError(
                    f"source artifact {artifact.path!r} ended during copy"
                )
            _write_all(target, chunk, label=artifact.path)
            hasher.update(chunk)
            byte_count += len(chunk)
            record_count += chunk.count(b"\n")
        os.fsync(target)
        _require_stable_file(
            source_before,
            os.fstat(source_descriptor),
            label=f"source artifact {artifact.path!r}",
        )
        if (hasher.hexdigest(), byte_count, record_count) != (
            artifact.sha256,
            artifact.byte_count,
            artifact.record_count,
        ):
            raise NfcCustodySuccessorError(
                f"source artifact {artifact.path!r} differs while copied"
            )
    except FileExistsError as exc:
        raise NfcCustodySuccessorError(
            f"temporary output repeats artifact {artifact.path!r}"
        ) from exc
    finally:
        if target is not None:
            os.close(target)
        os.close(parent)


def _fsync_tree(root_descriptor: int, expected_files: set[str]) -> None:
    directories = sorted(
        _expected_directories(expected_files),
        key=lambda path: (-len(PurePosixPath(path).parts), path.encode("utf-8")),
    )
    for relative in directories:
        parts = PurePosixPath(relative).parts
        descriptor = os.dup(root_descriptor)
        try:
            for component in parts:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.fsync(root_descriptor)


def _seal_tree(root_descriptor: int, expected_files: set[str]) -> None:
    for relative in sorted(expected_files, key=lambda path: path.encode("utf-8")):
        parts = PurePosixPath(relative).parts
        parent = os.dup(root_descriptor)
        descriptor: int | None = None
        try:
            for component in parts[:-1]:
                child = os.open(component, _directory_flags(), dir_fd=parent)
                os.close(parent)
                parent = child
            descriptor = os.open(parts[-1], _regular_flags(), dir_fd=parent)
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            _require_regular(os.fstat(descriptor), label=f"sealed file {relative!r}", mode=0o400)
            _require_no_extended_acl(descriptor, label=f"sealed file {relative!r}")
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)
    directories = sorted(
        _expected_directories(expected_files),
        key=lambda path: (-len(PurePosixPath(path).parts), path.encode("utf-8")),
    )
    for relative in directories:
        descriptor = os.dup(root_descriptor)
        try:
            for component in PurePosixPath(relative).parts:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            os.fchmod(descriptor, 0o500)
            os.fsync(descriptor)
            _require_directory(
                os.fstat(descriptor),
                label=f"sealed directory {relative!r}",
                mode=0o500,
            )
            _require_no_extended_acl(descriptor, label=f"sealed directory {relative!r}")
        finally:
            os.close(descriptor)
    os.fchmod(root_descriptor, 0o500)
    os.fsync(root_descriptor)
    _require_directory(os.fstat(root_descriptor), label="sealed output root", mode=0o500)
    _require_no_extended_acl(root_descriptor, label="sealed output root")


def _rename_no_replace(
    source_parent: int,
    source_name: str,
    destination_parent: int,
    destination_name: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        function = library.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_parent,
            source,
            destination_parent,
            destination,
            _RENAME_EXCL,
        )
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_parent,
            source,
            destination_parent,
            destination,
            _RENAME_NOREPLACE,
        )
    else:
        raise NfcCustodySuccessorError(
            "atomic no-replace publication requires renameatx_np or renameat2"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise NfcCustodySuccessorError("publication destination already exists")
        raise NfcCustodySuccessorError(
            f"atomic no-replace publication failed: {os.strerror(error)}"
        )


def _rename_sealed_directory(
    parent: int,
    source_name: str,
    destination_name: str,
    descriptor: int,
) -> None:
    error: BaseException | None = None
    try:
        if sys.platform == "darwin":
            os.fchmod(descriptor, 0o700)
        _rename_no_replace(parent, source_name, parent, destination_name)
    except BaseException as exc:
        error = exc
    try:
        if sys.platform == "darwin":
            os.fchmod(descriptor, 0o500)
        os.fsync(descriptor)
        _require_directory(os.fstat(descriptor), label="published output root", mode=0o500)
    except BaseException as seal_error:
        raise NfcCustodyPublicationIndeterminate(
            "output root could not be resealed across publication"
        ) from seal_error
    if error is not None:
        raise error


def _write_temporary_receipt(
    parent_descriptor: int,
    *,
    stem: str,
    encoded: bytes,
) -> tuple[str, int]:
    for _attempt in range(64):
        name = f".{stem}.nfc-custody-{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                name,
                _regular_flags(writable=True, exclusive=True),
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            continue
        try:
            os.fchmod(descriptor, 0o600)
            _require_no_extended_acl(descriptor, label="temporary receipt")
            _write_all(descriptor, encoded, label="external receipt")
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            _require_regular(os.fstat(descriptor), label="temporary receipt", mode=0o400)
            return name, descriptor
        except BaseException:
            os.close(descriptor)
            try:
                os.unlink(name, dir_fd=parent_descriptor)
            except OSError:
                pass
            raise
    raise NfcCustodySuccessorError("cannot allocate a unique temporary receipt")


def _remove_tree(descriptor: int) -> None:
    os.fchmod(descriptor, 0o700)
    for name in os.listdir(descriptor):
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            child = os.open(name, _directory_flags(), dir_fd=descriptor)
            try:
                _remove_tree(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        elif stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            os.unlink(name, dir_fd=descriptor)
        else:
            raise NfcCustodyPublicationIndeterminate(
                "temporary output gained a linked or special member during cleanup"
            )


def _verify_output(
    *,
    root: Path,
    inventory_bytes: bytes,
    checksum_bytes: bytes,
    artifacts: Sequence[Artifact],
    max_total_bytes: int,
) -> os.stat_result:
    root_descriptor, root_before = _open_source_root(root, label="published successor root")
    preopened: dict[str, int] = {}
    capture: _CapturedTree | None = None
    try:
        inventory_fd, observed_inventory = _read_preopened_control(
            root_descriptor,
            "inventory.json",
            label="published successor inventory",
            maximum=MAX_CONTROL_BYTES,
        )
        preopened["inventory.json"] = inventory_fd
        checksum_fd, observed_checksum = _read_preopened_control(
            root_descriptor,
            "inventory.sha256",
            label="published successor checksum",
            maximum=1024,
        )
        preopened["inventory.sha256"] = checksum_fd
        if observed_inventory != inventory_bytes or observed_checksum != checksum_bytes:
            raise NfcCustodySuccessorError("published successor controls differ")
        expected_files = {
            "inventory.json",
            "inventory.sha256",
            *(artifact.path for artifact in artifacts),
        }
        capture = _capture_exact_tree(
            root=root,
            root_descriptor=root_descriptor,
            root_before=root_before,
            expected_files=expected_files,
            preopened=preopened,
            label="published successor root",
        )
        root_descriptor = -1
        preopened = {}
        _validate_fingerprints(
            capture,
            artifacts,
            label="published successor",
            max_total_bytes=max_total_bytes,
        )
        _require_capture_stable(capture, label="published successor")
        return capture.root_before
    finally:
        if capture is not None:
            capture.close()
        else:
            for descriptor in preopened.values():
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if root_descriptor >= 0:
                os.close(root_descriptor)


def _rebind_published_directory(
    parent_descriptor: int,
    name: str,
    *,
    expected: os.stat_result,
    label: str,
) -> int:
    """Reopen and retain the final name only if it still names the proved inode."""

    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require_directory(before, label=label, mode=0o500)
        _require_stable_directory(expected, before, label=f"{label} path after proof")
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        _require_stable_directory(before, opened, label=label)
        _require_no_extended_acl(descriptor, label=label)
        _lock(descriptor, exclusive=False, label=label)
        rebound = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        _require_stable_directory(opened, rebound, label=f"{label} rebound path")
        result = descriptor
        descriptor = None
        return result
    except FileNotFoundError as exc:
        raise NfcCustodySuccessorError(f"{label} disappeared after proof") from exc
    except OSError as exc:
        raise NfcCustodySuccessorError(f"cannot rebind {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_external_receipt(
    path: Path,
    *,
    expected_sha256: str,
) -> NfcCustodySuccessorReceipt:
    parent = _open_absolute_directory(path.parent, label="receipt parent", mode=0o700)
    descriptor: int | None = None
    try:
        descriptor = _open_root_member(parent, path.name, label="external receipt", mode=0o400)
        encoded = _pread_bytes(descriptor, maximum=MAX_CONTROL_BYTES, label="external receipt")
        if _digest(encoded) != expected_sha256:
            raise NfcCustodySuccessorError("external receipt differs from its caller pin")
        value = _decode(encoded, label="external receipt")
        receipt = NfcCustodySuccessorReceipt.from_dict(value)
        if encoded != receipt.canonical_file_bytes() or receipt.receipt_output != path:
            raise NfcCustodySuccessorError(
                "external receipt canonical bytes or path binding differ"
            )
        return receipt
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _validate_path_contract(
    *,
    projection_root: Path,
    original_root: Path,
    output_root: Path,
    receipt_output: Path,
) -> None:
    roots = (projection_root, original_root, output_root)
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if _paths_overlap(left, right):
                raise NfcCustodySuccessorError("staged roots must be pairwise disjoint")
    if any(_paths_overlap(receipt_output, root) for root in roots):
        raise NfcCustodySuccessorError("external receipt must remain outside every staged root")
    if receipt_output.name in {"", ".", ".."}:
        raise NfcCustodySuccessorError("external receipt requires a file name")


def _open_publication_parents(
    output_parent: Path,
    receipt_parent: Path,
) -> tuple[dict[Path, int], list[int]]:
    descriptors: dict[Path, int] = {}
    ordered: list[int] = []
    try:
        for path in sorted({output_parent, receipt_parent}, key=lambda item: str(item).encode()):
            descriptor = _open_writable_parent(path, label=f"publication parent {path}")
            descriptors[path] = descriptor
            ordered.append(descriptor)
        return descriptors, ordered
    except BaseException:
        for descriptor in reversed(ordered):
            os.close(descriptor)
        raise


def build_nfc_custody_successor(
    *,
    projection_root: str | Path,
    successor_inventory_sha256: str,
    projection_receipt_sha256: str,
    original_root: str | Path,
    original_inventory_sha256: str,
    output_root: str | Path,
    receipt_output: str | Path,
    max_total_artifact_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> NfcCustodySuccessorReceipt:
    """Build and publish one byte-exact complete NFC custody successor."""

    with _SignalGuard():
        _require_nonroot()
        projection = _absolute_path(projection_root, label="projection_root")
        original = _absolute_path(original_root, label="original_root")
        output = _absolute_path(output_root, label="output_root")
        receipt_path = _absolute_path(receipt_output, label="receipt_output")
        successor_pin = _require_sha256(
            "successor_inventory_sha256",
            successor_inventory_sha256,
        )
        original_pin = _require_sha256("original_inventory_sha256", original_inventory_sha256)
        projection_pin = _require_sha256(
            "projection_receipt_sha256",
            projection_receipt_sha256,
        )
        max_bytes = _require_integer(
            "max_total_artifact_bytes",
            max_total_artifact_bytes,
            minimum=1,
        )
        _validate_path_contract(
            projection_root=projection,
            original_root=original,
            output_root=output,
            receipt_output=receipt_path,
        )

        admitted = _admit_sources(
            projection_root=projection,
            successor_inventory_sha256=successor_pin,
            projection_receipt_sha256=projection_pin,
            original_root=original,
            original_inventory_sha256=original_pin,
            max_total_bytes=max_bytes,
        )
        parents: list[int] = []
        temporary_output_name: str | None = None
        temporary_output_descriptor: int | None = None
        temporary_receipt_name: str | None = None
        temporary_receipt_descriptor: int | None = None
        output_publication_state = "absent"
        receipt_publication_state = "absent"
        parent_map: dict[Path, int] = {}
        try:
            origins = _origins(admitted)
            projected = tuple(
                origin.artifact for origin in origins if origin.source == "nfc-projection"
            )
            custody = tuple(
                origin.artifact for origin in origins if origin.source == "original-custody"
            )
            receipt = NfcCustodySuccessorReceipt(
                projection_root=projection,
                original_root=original,
                output_root=output,
                receipt_output=receipt_path,
                successor_inventory_sha256=successor_pin,
                original_inventory_sha256=original_pin,
                projection_receipt_sha256=projection_pin,
                projected_artifact_set_sha256=_artifact_set_sha256(projected),
                custody_artifact_set_sha256=_artifact_set_sha256(custody),
                output_artifact_set_sha256=_artifact_set_sha256(admitted.successor_artifacts),
                source_capture_set_sha256=_source_capture_set_sha256(
                    origins=origins,
                    successor_inventory_sha256=successor_pin,
                    original_inventory_sha256=original_pin,
                    projection_receipt_sha256=projection_pin,
                ),
                artifacts=origins,
                limits=ResourceLimits(max_total_artifact_bytes=max_bytes),
            )

            parent_map, parents = _open_publication_parents(output.parent, receipt_path.parent)
            output_parent = parent_map[output.parent]
            receipt_parent = parent_map[receipt_path.parent]
            if _entry_metadata(output_parent, output.name) is not None:
                raise NfcCustodySuccessorError("output root already exists")
            if _entry_metadata(receipt_parent, receipt_path.name) is not None:
                raise NfcCustodySuccessorError("external receipt already exists")
            temporary_output_name, temporary_output_descriptor = _create_temporary_directory(
                output_parent,
                stem=output.name,
            )
            output_publication_state = "temporary"
            _write_control(
                temporary_output_descriptor,
                "inventory.json",
                admitted.successor_inventory_bytes,
            )
            _write_control(
                temporary_output_descriptor,
                "inventory.sha256",
                admitted.successor_checksum_bytes,
            )
            projected_by_path = {
                artifact.path: artifact for artifact in admitted.projected_artifacts
            }
            custody_by_path = {artifact.path: artifact for artifact in admitted.custody_artifacts}
            for artifact in admitted.successor_artifacts:
                if artifact.role in _OUTCOME_ROLES:
                    if custody_by_path.get(artifact.path) != artifact:
                        raise NfcCustodySuccessorError(
                            f"custody source contract missing {artifact.path!r}"
                        )
                    source = admitted.original.files[artifact.path]
                else:
                    if projected_by_path.get(artifact.path) != artifact:
                        raise NfcCustodySuccessorError(
                            f"projection source contract missing {artifact.path!r}"
                        )
                    source = admitted.projection.files[artifact.path]
                _copy_pinned_file(
                    source_descriptor=source,
                    target_root_descriptor=temporary_output_descriptor,
                    artifact=artifact,
                )
            _require_capture_stable(admitted.projection, label="NFC projection")
            _require_capture_stable(admitted.original, label="original complete root")
            output_files = {
                "inventory.json",
                "inventory.sha256",
                *(artifact.path for artifact in admitted.successor_artifacts),
            }
            _fsync_tree(temporary_output_descriptor, output_files)
            _seal_tree(temporary_output_descriptor, output_files)
            _verify_output(
                root=output.parent / temporary_output_name,
                inventory_bytes=admitted.successor_inventory_bytes,
                checksum_bytes=admitted.successor_checksum_bytes,
                artifacts=admitted.successor_artifacts,
                max_total_bytes=max_bytes,
            )

            temporary_receipt_name, temporary_receipt_descriptor = _write_temporary_receipt(
                receipt_parent,
                stem=receipt_path.name,
                encoded=receipt.canonical_file_bytes(),
            )
            receipt_publication_state = "temporary"
            expected_output = os.fstat(temporary_output_descriptor)
            output_publication_state = "indeterminate"
            try:
                _rename_sealed_directory(
                    output_parent,
                    temporary_output_name,
                    output.name,
                    temporary_output_descriptor,
                )
            except BaseException:
                moved = _classify_no_replace_move(
                    source_parent=output_parent,
                    source_name=temporary_output_name,
                    destination_parent=output_parent,
                    destination_name=output.name,
                    expected=expected_output,
                    label="output publication",
                )
                output_publication_state = "published" if moved else "temporary"
                raise
            moved = _classify_no_replace_move(
                source_parent=output_parent,
                source_name=temporary_output_name,
                destination_parent=output_parent,
                destination_name=output.name,
                expected=expected_output,
                label="output publication",
            )
            output_publication_state = "published" if moved else "temporary"
            if output_publication_state != "published":
                raise NfcCustodyPublicationIndeterminate(
                    "output publication returned without moving the pinned tree"
                )
            os.fsync(output_parent)
            expected_receipt = os.fstat(temporary_receipt_descriptor)
            receipt_publication_state = "indeterminate"
            try:
                _rename_no_replace(
                    receipt_parent,
                    temporary_receipt_name,
                    receipt_parent,
                    receipt_path.name,
                )
            except BaseException:
                moved = _classify_no_replace_move(
                    source_parent=receipt_parent,
                    source_name=temporary_receipt_name,
                    destination_parent=receipt_parent,
                    destination_name=receipt_path.name,
                    expected=expected_receipt,
                    label="receipt publication",
                )
                receipt_publication_state = "published" if moved else "temporary"
                raise
            moved = _classify_no_replace_move(
                source_parent=receipt_parent,
                source_name=temporary_receipt_name,
                destination_parent=receipt_parent,
                destination_name=receipt_path.name,
                expected=expected_receipt,
                label="receipt publication",
            )
            receipt_publication_state = "published" if moved else "temporary"
            if receipt_publication_state != "published":
                raise NfcCustodyPublicationIndeterminate(
                    "receipt publication returned without moving the pinned file"
                )
            os.fsync(receipt_parent)
            _verify_output(
                root=output,
                inventory_bytes=admitted.successor_inventory_bytes,
                checksum_bytes=admitted.successor_checksum_bytes,
                artifacts=admitted.successor_artifacts,
                max_total_bytes=max_bytes,
            )
            loaded = _load_external_receipt(
                receipt_path,
                expected_sha256=receipt.artifact_sha256,
            )
            if loaded != receipt:
                raise NfcCustodyPublicationIndeterminate(
                    "published external receipt differs from the transaction receipt"
                )
            _require_capture_stable(admitted.projection, label="NFC projection")
            _require_capture_stable(admitted.original, label="original complete root")
            temporary_output_name = None
            temporary_receipt_name = None
            return receipt
        except BaseException as publication_error:
            rollback_error: BaseException | None = None
            try:
                if receipt_publication_state == "indeterminate":
                    moved = _classify_no_replace_move(
                        source_parent=receipt_parent,
                        source_name=temporary_receipt_name,
                        destination_parent=receipt_parent,
                        destination_name=receipt_path.name,
                        expected=os.fstat(temporary_receipt_descriptor),
                        label="receipt publication recovery",
                    )
                    receipt_publication_state = "published" if moved else "temporary"
                if receipt_publication_state == "published" and temporary_receipt_name is not None:
                    receipt_publication_state = "indeterminate"
                    try:
                        _rename_no_replace(
                            receipt_parent,
                            receipt_path.name,
                            receipt_parent,
                            temporary_receipt_name,
                        )
                    except BaseException:
                        rolled_back = _classify_no_replace_move(
                            source_parent=receipt_parent,
                            source_name=receipt_path.name,
                            destination_parent=receipt_parent,
                            destination_name=temporary_receipt_name,
                            expected=os.fstat(temporary_receipt_descriptor),
                            label="receipt rollback",
                        )
                        receipt_publication_state = "temporary" if rolled_back else "published"
                        if not rolled_back:
                            raise
                    else:
                        rolled_back = _classify_no_replace_move(
                            source_parent=receipt_parent,
                            source_name=receipt_path.name,
                            destination_parent=receipt_parent,
                            destination_name=temporary_receipt_name,
                            expected=os.fstat(temporary_receipt_descriptor),
                            label="receipt rollback",
                        )
                        receipt_publication_state = "temporary" if rolled_back else "published"
                        if not rolled_back:
                            raise NfcCustodyPublicationIndeterminate(
                                "receipt rollback returned without restoring its temporary name"
                            )
                    os.fsync(receipt_parent)
                if output_publication_state == "indeterminate":
                    moved = _classify_no_replace_move(
                        source_parent=output_parent,
                        source_name=temporary_output_name,
                        destination_parent=output_parent,
                        destination_name=output.name,
                        expected=os.fstat(temporary_output_descriptor),
                        label="output publication recovery",
                    )
                    output_publication_state = "published" if moved else "temporary"
                if output_publication_state == "published" and temporary_output_name is not None:
                    if temporary_output_descriptor is None:
                        raise NfcCustodyPublicationIndeterminate(
                            "published output lost its pinned descriptor"
                        )
                    output_publication_state = "indeterminate"
                    try:
                        _rename_sealed_directory(
                            output_parent,
                            output.name,
                            temporary_output_name,
                            temporary_output_descriptor,
                        )
                    except BaseException:
                        rolled_back = _classify_no_replace_move(
                            source_parent=output_parent,
                            source_name=output.name,
                            destination_parent=output_parent,
                            destination_name=temporary_output_name,
                            expected=os.fstat(temporary_output_descriptor),
                            label="output rollback",
                        )
                        output_publication_state = "temporary" if rolled_back else "published"
                        if not rolled_back:
                            raise
                    else:
                        rolled_back = _classify_no_replace_move(
                            source_parent=output_parent,
                            source_name=output.name,
                            destination_parent=output_parent,
                            destination_name=temporary_output_name,
                            expected=os.fstat(temporary_output_descriptor),
                            label="output rollback",
                        )
                        output_publication_state = "temporary" if rolled_back else "published"
                        if not rolled_back:
                            raise NfcCustodyPublicationIndeterminate(
                                "output rollback returned without restoring its temporary name"
                            )
                    os.fsync(output_parent)
            except BaseException as exc:
                rollback_error = exc
            if rollback_error is not None:
                raise NfcCustodyPublicationIndeterminate(
                    "publication failed and rollback could not be proved"
                ) from rollback_error
            raise publication_error
        finally:
            if temporary_receipt_descriptor is not None:
                os.close(temporary_receipt_descriptor)
            if (
                temporary_receipt_name is not None
                and receipt_publication_state == "temporary"
                and parents
            ):
                try:
                    os.unlink(temporary_receipt_name, dir_fd=parent_map[receipt_path.parent])
                    os.fsync(parent_map[receipt_path.parent])
                except FileNotFoundError:
                    pass
            if temporary_output_descriptor is not None:
                if temporary_output_name is not None and output_publication_state == "temporary":
                    try:
                        _remove_tree(temporary_output_descriptor)
                        os.rmdir(temporary_output_name, dir_fd=parent_map[output.parent])
                        os.fsync(parent_map[output.parent])
                    except FileNotFoundError:
                        pass
                os.close(temporary_output_descriptor)
            for descriptor in reversed(parents):
                os.close(descriptor)
            admitted.close()


def verify_nfc_custody_successor(
    *,
    projection_root: str | Path,
    successor_inventory_sha256: str,
    projection_receipt_sha256: str,
    original_root: str | Path,
    original_inventory_sha256: str,
    output_root: str | Path,
    receipt_output: str | Path,
    receipt_sha256: str,
    max_total_artifact_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> NfcCustodySuccessorReceipt:
    """Re-admit both sources and verify the published successor and external receipt."""

    _require_nonroot()
    projection = _absolute_path(projection_root, label="projection_root")
    original = _absolute_path(original_root, label="original_root")
    output = _absolute_path(output_root, label="output_root")
    receipt_path = _absolute_path(receipt_output, label="receipt_output")
    successor_pin = _require_sha256("successor_inventory_sha256", successor_inventory_sha256)
    original_pin = _require_sha256("original_inventory_sha256", original_inventory_sha256)
    projection_pin = _require_sha256("projection_receipt_sha256", projection_receipt_sha256)
    receipt_pin = _require_sha256("receipt_sha256", receipt_sha256)
    max_bytes = _require_integer(
        "max_total_artifact_bytes",
        max_total_artifact_bytes,
        minimum=1,
    )
    _validate_path_contract(
        projection_root=projection,
        original_root=original,
        output_root=output,
        receipt_output=receipt_path,
    )
    admitted = _admit_sources(
        projection_root=projection,
        successor_inventory_sha256=successor_pin,
        projection_receipt_sha256=projection_pin,
        original_root=original,
        original_inventory_sha256=original_pin,
        max_total_bytes=max_bytes,
    )
    parents: list[int] = []
    rebound_output_descriptor: int | None = None
    try:
        parent_map, parents = _open_publication_parents(output.parent, receipt_path.parent)
        output_parent = parent_map[output.parent]
        receipt = _load_external_receipt(receipt_path, expected_sha256=receipt_pin)
        if (
            receipt.projection_root != projection
            or receipt.original_root != original
            or receipt.output_root != output
            or receipt.successor_inventory_sha256 != successor_pin
            or receipt.original_inventory_sha256 != original_pin
            or receipt.projection_receipt_sha256 != projection_pin
            or receipt.limits.max_total_artifact_bytes != max_bytes
        ):
            raise NfcCustodySuccessorError("receipt differs from explicit verifier bindings")
        origins = _origins(admitted)
        expected = NfcCustodySuccessorReceipt(
            projection_root=projection,
            original_root=original,
            output_root=output,
            receipt_output=receipt_path,
            successor_inventory_sha256=successor_pin,
            original_inventory_sha256=original_pin,
            projection_receipt_sha256=projection_pin,
            projected_artifact_set_sha256=_artifact_set_sha256(admitted.projected_artifacts),
            custody_artifact_set_sha256=_artifact_set_sha256(admitted.custody_artifacts),
            output_artifact_set_sha256=_artifact_set_sha256(admitted.successor_artifacts),
            source_capture_set_sha256=_source_capture_set_sha256(
                origins=origins,
                successor_inventory_sha256=successor_pin,
                original_inventory_sha256=original_pin,
                projection_receipt_sha256=projection_pin,
            ),
            artifacts=origins,
            limits=ResourceLimits(max_total_artifact_bytes=max_bytes),
        )
        if receipt != expected:
            raise NfcCustodySuccessorError("external receipt differs from admitted sources")
        proved_output = _verify_output(
            root=output,
            inventory_bytes=admitted.successor_inventory_bytes,
            checksum_bytes=admitted.successor_checksum_bytes,
            artifacts=admitted.successor_artifacts,
            max_total_bytes=max_bytes,
        )
        rebound_output_descriptor = _rebind_published_directory(
            output_parent,
            output.name,
            expected=proved_output,
            label="published successor root",
        )
        _require_capture_stable(admitted.projection, label="NFC projection")
        _require_capture_stable(admitted.original, label="original complete root")
        if _load_external_receipt(receipt_path, expected_sha256=receipt_pin) != receipt:
            raise NfcCustodySuccessorError("external receipt changed during verification")
        _require_stable_directory(
            proved_output,
            os.fstat(rebound_output_descriptor),
            label="published successor root after verification",
        )
        return receipt
    finally:
        if rebound_output_descriptor is not None:
            os.close(rebound_output_descriptor)
        for descriptor in reversed(parents):
            os.close(descriptor)
        admitted.close()


def _result(receipt: NfcCustodySuccessorReceipt) -> dict[str, object]:
    return {
        "artifact_count": receipt.artifact_count,
        "custody_artifact_count": receipt.custody_artifact_count,
        "output_root": str(receipt.output_root),
        "projected_artifact_count": receipt.projected_artifact_count,
        "receipt_output": str(receipt.receipt_output),
        "receipt_sha256": receipt.artifact_sha256,
        "schema_version": CLI_RESULT_SCHEMA,
        "successor_inventory_sha256": receipt.successor_inventory_sha256,
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--projection-root", required=True)
    parser.add_argument("--successor-inventory-sha256", required=True)
    parser.add_argument("--projection-receipt-sha256", required=True)
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--original-inventory-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument(
        "--max-total-artifact-bytes",
        type=int,
        default=DEFAULT_MAX_TOTAL_BYTES,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify a custody-complete NFC staging successor.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build one new successor", allow_abbrev=False)
    _add_common_arguments(build)
    verify = commands.add_parser("verify", help="verify one successor", allow_abbrev=False)
    _add_common_arguments(verify)
    verify.add_argument("--receipt-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    common = {
        "projection_root": arguments.projection_root,
        "successor_inventory_sha256": arguments.successor_inventory_sha256,
        "projection_receipt_sha256": arguments.projection_receipt_sha256,
        "original_root": arguments.original_root,
        "original_inventory_sha256": arguments.original_inventory_sha256,
        "output_root": arguments.output_root,
        "receipt_output": arguments.receipt_output,
        "max_total_artifact_bytes": arguments.max_total_artifact_bytes,
    }
    try:
        if arguments.command == "build":
            receipt = build_nfc_custody_successor(**common)
        else:
            receipt = verify_nfc_custody_successor(
                **common,
                receipt_sha256=arguments.receipt_sha256,
            )
    except NfcCustodyInterrupted as exc:
        return 128 + exc.signum
    except NfcCustodySuccessorError as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(_canonical_bytes(_result(receipt)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
