"""Materialize the sole suite OPENED genesis from already-frozen controls.

The request is a custody envelope, not a second scientific configuration.  It
binds the absolute location and exact file bytes for every explicit input.  All
scientific values are then derived by the existing typed C1, finalization,
launcher, and suite-attempt apparatus.
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
import select
import signal
import stat
import struct
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fractal_ann_diagnostics.production_controls import (
    load_production_control_finalization_receipt,
    load_production_control_finalization_request,
    verify_production_run_closure_authority,
)
from fractal_ann_diagnostics.runtime_attestation import load_runtime_preflight_receipt
from fractal_ann_diagnostics.sealed_container_launcher import (
    load_preflight_launch_contract,
    load_registered_plan_instantiation,
    load_runtime_plan_transition,
    load_sealed_launch_contract,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA
from fractal_ann_diagnostics.suite_attempt import (
    SuiteAttestationDescriptor,
    load_suite_state_record,
    open_suite_attempt,
)
from fractal_ann_diagnostics.zenodo_publication import (
    PACKAGE_FILE_NAMES,
    verify_production_protocol_registration,
)

OPENED_GENESIS_REQUEST_SCHEMA = "fractal-opened-genesis-request-v1"
_ATTEMPT_MARKER_SCHEMA = "fractal-opened-genesis-attempt-marker-v1"
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_PREFIX = "c1-package/"
_GLOBAL_ROLES = (
    "production-finalization-receipt",
    "production-finalization-request",
    "protocol-registration-receipt",
    "protocol-registry-record",
    "sealed-run-receipt",
    "suite-attestation-descriptor",
)
_CORPUS_ROLE_SUFFIXES = (
    "preflight-launch-contract",
    "registered-plan-instantiation",
    "runtime-plan-transition",
    "runtime-preflight-receipt",
    "sealed-launch-contract",
)
_OUTPUT_MEMBERS = frozenset({"000.state.json", "attestation-descriptor.json", "online"})


class OpenedGenesisOperatorError(ValueError):
    """Raised when the OPENED genesis cannot be published unambiguously."""


class _ArgumentError(OpenedGenesisOperatorError):
    pass


class _TerminationSignal(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"received signal {signum}")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise OpenedGenesisOperatorError("request must be finite canonical JSON") from exc


def _strict_json(encoded: bytes, *, label: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OpenedGenesisOperatorError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise OpenedGenesisOperatorError(f"{label} contains non-finite number {value!r}")

    try:
        return json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise OpenedGenesisOperatorError(f"{label} must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise OpenedGenesisOperatorError(f"{label} must be valid JSON") from exc


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise OpenedGenesisOperatorError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _canonical_path(name: str, value: object) -> Path:
    if type(value) is not str or not value or value != value.strip():
        raise OpenedGenesisOperatorError(f"{name} must be a canonical absolute path string")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or str(path) != value
        or value == "/"
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise OpenedGenesisOperatorError(f"{name} must be a canonical absolute POSIX file path")
    return path


def _closed_object(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise OpenedGenesisOperatorError(f"{label} must be one JSON object")
    observed = set(value)
    if observed != fields:
        raise OpenedGenesisOperatorError(
            f"{label} schema differs; "
            f"missing={sorted(fields - observed)!r}, "
            f"unknown={sorted(observed - fields)!r}"
        )
    return value


def _expected_roles() -> tuple[str, ...]:
    roles = [f"{_PACKAGE_PREFIX}{name}" for name in PACKAGE_FILE_NAMES]
    roles.extend(_GLOBAL_ROLES)
    roles.extend(
        f"{corpus_id}/{suffix}" for corpus_id in FIXED_CORPORA for suffix in _CORPUS_ROLE_SUFFIXES
    )
    return tuple(sorted(roles, key=lambda value: value.encode("utf-8")))


@dataclass(frozen=True)
class InputBinding:
    role: str
    path: Path
    file_sha256: str

    @classmethod
    def from_dict(cls, value: object, *, position: int) -> InputBinding:
        row = _closed_object(
            value,
            frozenset({"file_sha256", "path", "role"}),
            label=f"request inputs[{position}]",
        )
        role = row["role"]
        if type(role) is not str or not role or role != role.strip():
            raise OpenedGenesisOperatorError(
                f"request inputs[{position}].role must be canonical text"
            )
        return cls(
            role=role,
            path=_canonical_path(f"request input {role!r} path", row["path"]),
            file_sha256=_sha256(
                f"request input {role!r} file_sha256",
                row["file_sha256"],
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "file_sha256": self.file_sha256,
            "path": str(self.path),
            "role": self.role,
        }


@dataclass(frozen=True)
class OpenedGenesisRequest:
    inputs: tuple[InputBinding, ...]
    schema_version: str = OPENED_GENESIS_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != OPENED_GENESIS_REQUEST_SCHEMA:
            raise OpenedGenesisOperatorError("OPENED-genesis request schema differs")
        expected = _expected_roles()
        roles = tuple(binding.role for binding in self.inputs)
        if roles != expected:
            raise OpenedGenesisOperatorError(
                "OPENED-genesis request role set or bytewise order differs"
            )
        paths = tuple(binding.path for binding in self.inputs)
        if len(paths) != len(set(paths)):
            raise OpenedGenesisOperatorError("OPENED-genesis request cannot reuse an input path")

    @classmethod
    def from_bytes(cls, encoded: bytes) -> OpenedGenesisRequest:
        if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
            raise OpenedGenesisOperatorError(
                "OPENED-genesis request must end with exactly one newline"
            )
        row = _closed_object(
            _strict_json(encoded, label="OPENED-genesis request"),
            frozenset({"inputs", "schema_version"}),
            label="OPENED-genesis request",
        )
        values = row["inputs"]
        if type(values) is not list:
            raise OpenedGenesisOperatorError("OPENED-genesis request inputs must be an array")
        request = cls(
            inputs=tuple(
                InputBinding.from_dict(value, position=position)
                for position, value in enumerate(values)
            ),
            schema_version=row["schema_version"],
        )
        if encoded != request.canonical_file_bytes():
            raise OpenedGenesisOperatorError("OPENED-genesis request bytes are not canonical")
        return request

    def canonical_file_bytes(self) -> bytes:
        return (
            _canonical_json(
                {
                    "inputs": [binding.to_dict() for binding in self.inputs],
                    "schema_version": self.schema_version,
                }
            )
            + b"\n"
        )

    @property
    def by_role(self) -> dict[str, InputBinding]:
        return {binding.role: binding for binding in self.inputs}


def _directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required):
        raise OpenedGenesisOperatorError("host lacks required no-follow directory primitives")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _file_flags() -> int:
    required = ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required):
        raise OpenedGenesisOperatorError("host lacks required no-follow file primitives")
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def _open_absolute_directory(path: Path, *, label: str) -> int:
    if not path.is_absolute() or path.anchor != "/":
        raise OpenedGenesisOperatorError(f"{label} must be absolute")
    try:
        descriptor = os.open("/", _directory_flags())
    except OSError as exc:
        raise OpenedGenesisOperatorError(f"cannot open {label}") from exc
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise OpenedGenesisOperatorError(
                    f"cannot open {label} without following links"
                ) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OpenedGenesisOperatorError(f"{label} must be a real directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _rename_no_replace_at(
    source_parent: int,
    source_name: str,
    target_parent: int,
    target_name: str,
) -> None:
    """Atomically rename one directory entry only if the destination is absent."""

    if (
        not source_name
        or not target_name
        or "/" in source_name
        or "/" in target_name
        or "\x00" in source_name
        or "\x00" in target_name
    ):
        raise OpenedGenesisOperatorError("atomic rename names must be single path components")
    try:
        library = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "darwin":
            rename = library.renameatx_np
            flag = 0x00000004  # RENAME_EXCL
        elif sys.platform.startswith("linux"):
            rename = library.renameat2
            flag = 0x00000001  # RENAME_NOREPLACE
        else:
            raise OpenedGenesisOperatorError("host lacks a required no-replace rename primitive")
    except (AttributeError, OSError) as exc:
        raise OpenedGenesisOperatorError(
            "host lacks a required no-replace rename primitive"
        ) from exc
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    result = rename(
        source_parent,
        os.fsencode(source_name),
        target_parent,
        os.fsencode(target_name),
        flag,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), target_name)
    if error == errno.ENOENT:
        raise FileNotFoundError(error, os.strerror(error), source_name)
    raise OSError(error, os.strerror(error), source_name, target_name)


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _stable_signature(metadata: os.stat_result) -> tuple[int, ...]:
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


def _pread_all(descriptor: int, *, size: int, max_bytes: int, label: str) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        try:
            chunk = os.pread(
                descriptor,
                min(_READ_CHUNK_BYTES, size - offset),
                offset,
            )
        except OSError as exc:
            raise OpenedGenesisOperatorError(f"cannot read {label}") from exc
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
        if offset > max_bytes:
            raise OpenedGenesisOperatorError(f"{label} exceeds its byte limit")
    if offset != size:
        raise OpenedGenesisOperatorError(f"{label} changed length while read")
    return b"".join(chunks)


def _read_member_at(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                name,
                _file_flags(),
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise OpenedGenesisOperatorError(
                f"cannot open {label} without following links"
            ) from exc
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OpenedGenesisOperatorError(f"{label} must be a regular file")
        if before.st_nlink != 1:
            raise OpenedGenesisOperatorError(f"{label} must be singly linked")
        if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
            raise OpenedGenesisOperatorError(f"{label} must be owned by the current identity")
        if before.st_size > max_bytes:
            raise OpenedGenesisOperatorError(f"{label} exceeds its byte limit")
        data = _pread_all(
            descriptor,
            size=before.st_size,
            max_bytes=max_bytes,
            label=label,
        )
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _stable_signature(before) != _stable_signature(after) or _stable_signature(
            before
        ) != _stable_signature(current):
            raise OpenedGenesisOperatorError(f"{label} changed while it was read")
        return data, before
    except OSError as exc:
        raise OpenedGenesisOperatorError(f"cannot read {label}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass
class _PinnedFile:
    path: Path
    label: str
    expected_sha256: str
    max_bytes: int
    parent_descriptor: int
    descriptor: int
    parent_identity: tuple[int, int]
    signature: tuple[int, ...]
    data: bytes

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        label: str,
        expected_sha256: str,
        max_bytes: int,
    ) -> _PinnedFile:
        _sha256(f"{label} expected SHA-256", expected_sha256)
        if type(max_bytes) is not int or max_bytes <= 0:
            raise OpenedGenesisOperatorError("input byte limit must be positive")
        parent = _open_absolute_directory(path.parent, label=f"{label} parent")
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(
                    path.name,
                    _file_flags(),
                    dir_fd=parent,
                )
            except OSError as exc:
                raise OpenedGenesisOperatorError(
                    f"cannot open {label} without following links"
                ) from exc
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise OpenedGenesisOperatorError(f"{label} must be a regular file")
            if before.st_nlink != 1:
                raise OpenedGenesisOperatorError(f"{label} must be singly linked")
            if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
                raise OpenedGenesisOperatorError(f"{label} must be owned by the current identity")
            if stat.S_IMODE(before.st_mode) & 0o022:
                raise OpenedGenesisOperatorError(
                    f"{label} cannot be writable by group or other users"
                )
            if before.st_size > max_bytes:
                raise OpenedGenesisOperatorError(f"{label} exceeds its byte limit")
            data = _pread_all(
                descriptor,
                size=before.st_size,
                max_bytes=max_bytes,
                label=label,
            )
            after = os.fstat(descriptor)
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (
                _stable_signature(before) != _stable_signature(after)
                or _identity(before) != _identity(current)
                or _stable_signature(before) != _stable_signature(current)
            ):
                raise OpenedGenesisOperatorError(f"{label} changed while it was read")
            if _digest(data) != expected_sha256:
                raise OpenedGenesisOperatorError(f"{label} differs from its request SHA-256")
            return cls(
                path=path,
                label=label,
                expected_sha256=expected_sha256,
                max_bytes=max_bytes,
                parent_descriptor=parent,
                descriptor=descriptor,
                parent_identity=_identity(os.fstat(parent)),
                signature=_stable_signature(before),
                data=data,
            )
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)
            raise

    @property
    def file_identity(self) -> tuple[int, int]:
        return self.signature[0], self.signature[1]

    @property
    def mode(self) -> int:
        return stat.S_IMODE(self.signature[2])

    def assert_current(self) -> None:
        reopened_parent = _open_absolute_directory(
            self.path.parent,
            label=f"{self.label} parent",
        )
        try:
            if _identity(os.fstat(reopened_parent)) != self.parent_identity:
                raise OpenedGenesisOperatorError(f"{self.label} parent path was replaced")
            descriptor_stat = os.fstat(self.descriptor)
            path_stat = os.stat(
                self.path.name,
                dir_fd=reopened_parent,
                follow_symlinks=False,
            )
            if (
                _stable_signature(descriptor_stat) != self.signature
                or _stable_signature(path_stat) != self.signature
            ):
                raise OpenedGenesisOperatorError(f"{self.label} changed after admission")
            data = _pread_all(
                self.descriptor,
                size=descriptor_stat.st_size,
                max_bytes=self.max_bytes,
                label=self.label,
            )
            final = os.fstat(self.descriptor)
            if _stable_signature(final) != self.signature or _digest(data) != self.expected_sha256:
                raise OpenedGenesisOperatorError(f"{self.label} changed after admission")
        except OSError as exc:
            raise OpenedGenesisOperatorError(f"cannot revalidate {self.label}") from exc
        finally:
            os.close(reopened_parent)

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        finally:
            os.close(self.parent_descriptor)


class _PinnedInputs:
    def __init__(self, pins: Mapping[str, _PinnedFile]) -> None:
        self._pins = dict(pins)

    @classmethod
    def open(cls, request: OpenedGenesisRequest) -> _PinnedInputs:
        pins: dict[str, _PinnedFile] = {}
        try:
            for binding in request.inputs:
                pins[binding.role] = _PinnedFile.open(
                    binding.path,
                    label=f"request input {binding.role!r}",
                    expected_sha256=binding.file_sha256,
                    max_bytes=_MAX_INPUT_BYTES,
                )
            identities = [pin.file_identity for pin in pins.values()]
            if len(identities) != len(set(identities)):
                raise OpenedGenesisOperatorError(
                    "OPENED-genesis inputs must be pairwise-distinct files"
                )
            result = cls(pins)
            result.assert_current()
            return result
        except BaseException:
            for pin in reversed(tuple(pins.values())):
                pin.close()
            raise

    def __getitem__(self, role: str) -> _PinnedFile:
        return self._pins[role]

    def assert_current(self) -> None:
        for role in sorted(self._pins, key=lambda value: value.encode("utf-8")):
            self._pins[role].assert_current()

    def pins(self) -> tuple[_PinnedFile, ...]:
        return tuple(
            self._pins[role] for role in sorted(self._pins, key=lambda value: value.encode("utf-8"))
        )

    def close(self) -> None:
        for pin in reversed(tuple(self._pins.values())):
            pin.close()


@dataclass(frozen=True)
class _MutationWatch:
    descriptor: int
    include_child_changes: bool
    owned: bool


class _PathMutationGuard:
    """Retain kernel watches across every independent path-based typed read."""

    def __init__(
        self,
        watches: Sequence[_MutationWatch],
        *,
        backend: str,
        backend_descriptor: object,
    ) -> None:
        self._watches = tuple(watches)
        self._backend = backend
        self._backend_descriptor = backend_descriptor

    @classmethod
    def open(cls, pins: Sequence[_PinnedFile]) -> _PathMutationGuard:
        watches: dict[tuple[int, int], _MutationWatch] = {}

        def admit(descriptor: int, *, include_child_changes: bool, owned: bool) -> None:
            identity = _identity(os.fstat(descriptor))
            prior = watches.get(identity)
            if prior is None:
                watches[identity] = _MutationWatch(
                    descriptor=descriptor,
                    include_child_changes=include_child_changes,
                    owned=owned,
                )
                return
            if include_child_changes and not prior.include_child_changes:
                watches[identity] = _MutationWatch(
                    descriptor=prior.descriptor,
                    include_child_changes=True,
                    owned=prior.owned,
                )
            if owned:
                os.close(descriptor)

        try:
            for pin in pins:
                admit(pin.descriptor, include_child_changes=True, owned=False)
                admit(pin.parent_descriptor, include_child_changes=False, owned=False)

            ancestor_paths: set[Path] = set()
            for pin in pins:
                ancestor = pin.path.parent.parent
                while ancestor != Path("/"):
                    ancestor_paths.add(ancestor)
                    ancestor = ancestor.parent
            for ancestor in sorted(
                ancestor_paths,
                key=lambda value: str(value).encode("utf-8"),
            ):
                descriptor = _open_absolute_directory(
                    ancestor,
                    label="request-bound input ancestor",
                )
                admit(descriptor, include_child_changes=False, owned=True)

            ordered = tuple(watches.values())
            if hasattr(select, "kqueue") and hasattr(select, "kevent"):
                return cls._open_kqueue(ordered)
            if sys.platform.startswith("linux"):
                return cls._open_inotify(ordered)
            raise OpenedGenesisOperatorError(
                "host lacks a required path-mutation notification primitive"
            )
        except BaseException:
            for watch in watches.values():
                if watch.owned:
                    os.close(watch.descriptor)
            raise

    @classmethod
    def _open_kqueue(
        cls,
        watches: Sequence[_MutationWatch],
    ) -> _PathMutationGuard:
        queue = select.kqueue()
        base_flags = (
            select.KQ_NOTE_DELETE
            | select.KQ_NOTE_ATTRIB
            | select.KQ_NOTE_RENAME
            | select.KQ_NOTE_REVOKE
        )
        changes = []
        for watch in watches:
            flags = base_flags
            if watch.include_child_changes:
                flags |= select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND | select.KQ_NOTE_LINK
            changes.append(
                select.kevent(
                    watch.descriptor,
                    filter=select.KQ_FILTER_VNODE,
                    flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                    fflags=flags,
                )
            )
        try:
            queue.control(changes, 0, 0)
        except BaseException:
            queue.close()
            raise
        return cls(watches, backend="kqueue", backend_descriptor=queue)

    @classmethod
    def _open_inotify(
        cls,
        watches: Sequence[_MutationWatch],
    ) -> _PathMutationGuard:
        try:
            library = ctypes.CDLL(None, use_errno=True)
            initialize = library.inotify_init1
            add_watch = library.inotify_add_watch
        except (AttributeError, OSError) as exc:
            raise OpenedGenesisOperatorError("host lacks required inotify entry points") from exc
        initialize.argtypes = [ctypes.c_int]
        initialize.restype = ctypes.c_int
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int

        descriptor = initialize(os.O_NONBLOCK | os.O_CLOEXEC)
        if descriptor < 0:
            error = ctypes.get_errno()
            raise OpenedGenesisOperatorError(
                "cannot initialize request-bound path-mutation notification"
            ) from OSError(error, os.strerror(error))

        self_changes = 0x00000004 | 0x00000400 | 0x00000800 | 0x00002000 | 0x00008000
        child_changes = 0x00000002 | 0x00000008 | 0x00000040 | 0x00000080 | 0x00000100 | 0x00000200
        try:
            for watch in watches:
                mask = self_changes
                if watch.include_child_changes:
                    mask |= child_changes
                watched = add_watch(
                    descriptor,
                    os.fsencode(f"/proc/self/fd/{watch.descriptor}"),
                    mask,
                )
                if watched < 0:
                    error = ctypes.get_errno()
                    raise OpenedGenesisOperatorError(
                        "cannot arm request-bound path-mutation notification"
                    ) from OSError(error, os.strerror(error))
        except BaseException:
            os.close(descriptor)
            raise
        return cls(watches, backend="inotify", backend_descriptor=descriptor)

    def checkpoint(self) -> None:
        if self._backend == "kqueue":
            queue = self._backend_descriptor
            assert isinstance(queue, select.kqueue)
            events = queue.control(
                None,
                max(16, len(self._watches) * 2),
                0,
            )
            if events:
                raise OpenedGenesisOperatorError(
                    "request-bound input path changed after admission during typed consumption"
                )
            return

        descriptor = self._backend_descriptor
        assert isinstance(descriptor, int)
        header = struct.Struct("iIII")
        observed = False
        while True:
            try:
                encoded = os.read(descriptor, 64 * 1024)
            except BlockingIOError:
                break
            except OSError as exc:
                raise OpenedGenesisOperatorError(
                    "cannot read request-bound path-mutation notification"
                ) from exc
            if not encoded:
                break
            offset = 0
            while offset + header.size <= len(encoded):
                _, _, _, name_length = header.unpack_from(encoded, offset)
                offset += header.size + name_length
                observed = True
            if offset != len(encoded):
                raise OpenedGenesisOperatorError(
                    "request-bound path-mutation notification is malformed"
                )
        if observed:
            raise OpenedGenesisOperatorError(
                "request-bound input path changed after admission during typed consumption"
            )

    def close(self) -> None:
        if self._backend == "kqueue":
            queue = self._backend_descriptor
            assert isinstance(queue, select.kqueue)
            queue.close()
        else:
            descriptor = self._backend_descriptor
            assert isinstance(descriptor, int)
            os.close(descriptor)
        for watch in reversed(self._watches):
            if watch.owned:
                os.close(watch.descriptor)


def _package_root(inputs: _PinnedInputs) -> Path:
    package_paths = {name: inputs[f"{_PACKAGE_PREFIX}{name}"].path for name in PACKAGE_FILE_NAMES}
    roots = {path.parent for path in package_paths.values()}
    if len(roots) != 1:
        raise OpenedGenesisOperatorError(
            "all C1 package files must share one request-bound directory"
        )
    root = roots.pop()
    for name, path in package_paths.items():
        if path != root / name:
            raise OpenedGenesisOperatorError("C1 package paths do not use their fixed filenames")
    return root


def _manifest_from_package(inputs: _PinnedInputs) -> dict[str, Any]:
    value = _strict_json(
        inputs[f"{_PACKAGE_PREFIX}study-manifest.json"].data,
        label="frozen C1 study manifest",
    )
    if not isinstance(value, dict):
        raise OpenedGenesisOperatorError("frozen C1 study manifest must contain one object")
    return value


def _descriptor(inputs: _PinnedInputs) -> SuiteAttestationDescriptor:
    pin = inputs["suite-attestation-descriptor"]
    value = _strict_json(pin.data, label="suite attestation descriptor")
    try:
        descriptor = SuiteAttestationDescriptor.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise OpenedGenesisOperatorError(f"suite attestation descriptor is invalid: {exc}") from exc
    if pin.data != descriptor.canonical_bytes() + b"\n":
        raise OpenedGenesisOperatorError("suite attestation descriptor bytes are not canonical")
    return descriptor


def _execution_artifacts(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Derive the caller-required map solely from frozen workload registrations."""

    values = manifest.get("production_workloads")
    if type(values) is not list:
        raise OpenedGenesisOperatorError("frozen C1 manifest production_workloads are malformed")
    result: dict[str, str] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise OpenedGenesisOperatorError(
                "frozen C1 manifest production_workloads are malformed"
            )
        corpus_id = value.get("corpus_id")
        spec = value.get("spec")
        if (
            type(corpus_id) is not str
            or corpus_id not in FIXED_CORPORA
            or corpus_id in result
            or not isinstance(spec, Mapping)
        ):
            raise OpenedGenesisOperatorError(
                "frozen C1 manifest production_workloads are malformed"
            )
        result[corpus_id] = _sha256(
            f"{corpus_id} frozen online execution plan",
            spec.get("online_execution_plan_sha256"),
        )
    if set(result) != set(FIXED_CORPORA):
        raise OpenedGenesisOperatorError(
            "frozen C1 manifest must register each fixed corpus exactly once"
        )
    return {corpus_id: result[corpus_id] for corpus_id in FIXED_CORPORA}


def _corpus_paths(
    inputs: _PinnedInputs,
    suffix: str,
) -> dict[str, Path]:
    return {corpus_id: inputs[f"{corpus_id}/{suffix}"].path for corpus_id in FIXED_CORPORA}


def _typed_open_arguments(
    inputs: _PinnedInputs,
    *,
    package_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[
    object,
    object,
    dict[str, object],
    dict[str, dict[str, Path]],
    SuiteAttestationDescriptor,
]:
    registration_record_path = inputs["protocol-registry-record"].path
    registration_receipt_path = inputs["protocol-registration-receipt"].path
    finalization_request_path = inputs["production-finalization-request"].path
    finalization_receipt_path = inputs["production-finalization-receipt"].path

    registration = verify_production_protocol_registration(
        package_root,
        registration_record_path=registration_record_path,
        registration_receipt_path=registration_receipt_path,
    )
    inputs.assert_current()

    finalization_receipt = load_production_control_finalization_receipt(
        finalization_receipt_path,
        expected_sha256=inputs["production-finalization-receipt"].expected_sha256,
    )
    finalization_request = load_production_control_finalization_request(
        finalization_request_path,
        expected_sha256=inputs["production-finalization-request"].expected_sha256,
    )
    if finalization_receipt.finalization_request_sha256 != (
        inputs["production-finalization-request"].expected_sha256
    ):
        raise OpenedGenesisOperatorError("finalization receipt names another finalization request")
    expected_request_bindings = {
        "c1_package_root": package_root,
        "protocol_registry_record_path": registration_record_path,
        "protocol_registration_receipt_path": registration_receipt_path,
        "sealed_run_receipt_path": inputs["sealed-run-receipt"].path,
    }
    if any(
        getattr(finalization_request, name) != expected
        for name, expected in expected_request_bindings.items()
    ):
        raise OpenedGenesisOperatorError(
            "finalization request paths differ from the OPENED-genesis custody request"
        )

    path_sets = {
        "preflight_contract_paths": _corpus_paths(inputs, "preflight-launch-contract"),
        "runtime_preflight_receipt_paths": _corpus_paths(inputs, "runtime-preflight-receipt"),
        "runtime_plan_transition_paths": _corpus_paths(inputs, "runtime-plan-transition"),
        "registered_plan_instantiation_paths": _corpus_paths(
            inputs, "registered-plan-instantiation"
        ),
        "sealed_launch_contract_paths": _corpus_paths(inputs, "sealed-launch-contract"),
    }
    closures: dict[str, object] = {}
    for corpus_id in FIXED_CORPORA:
        preflight = load_preflight_launch_contract(path_sets["preflight_contract_paths"][corpus_id])
        load_runtime_preflight_receipt(path_sets["runtime_preflight_receipt_paths"][corpus_id])
        transition = load_runtime_plan_transition(
            path_sets["runtime_plan_transition_paths"][corpus_id]
        )
        instantiation = load_registered_plan_instantiation(
            path_sets["registered_plan_instantiation_paths"][corpus_id]
        )
        sealed = load_sealed_launch_contract(path_sets["sealed_launch_contract_paths"][corpus_id])
        observed_corpora = (
            preflight.geometry.corpus_id,
            transition.corpus_id,
            instantiation.corpus_id,
            sealed.geometry.corpus_id,
        )
        if any(value != corpus_id for value in observed_corpora):
            raise OpenedGenesisOperatorError(f"{corpus_id} runtime controls name another corpus")
        closures[corpus_id] = verify_production_run_closure_authority(
            finalization_request_path=finalization_request_path,
            finalization_receipt_path=finalization_receipt_path,
            preflight=preflight,
            transition=transition,
        )
    inputs.assert_current()
    return (
        registration,
        finalization_receipt,
        closures,
        path_sets,
        _descriptor(inputs),
    )


class _Publication:
    def __init__(
        self,
        namespace: Path,
        suite_attempt_id: str,
        *,
        descriptor_file_sha256: str,
        request_sha256: str,
    ) -> None:
        self.namespace = namespace
        self.suite_attempt_id = _sha256("suite attempt ID", suite_attempt_id)
        self.descriptor_file_sha256 = _sha256(
            "attestation descriptor file SHA-256",
            descriptor_file_sha256,
        )
        self.request_sha256 = _sha256("request SHA-256", request_sha256)
        self.parent_descriptor: int | None = None
        self.parent_identity: tuple[int, int] | None = None
        self.lock_descriptor: int | None = None
        self.attempt_marker_descriptor: int | None = None
        self.namespace_identity: tuple[int, int] | None = None
        self.output_member_signatures: dict[str, tuple[int, ...]] = {}

    @property
    def _attempt_marker_name(self) -> str:
        return f".opened-genesis-{self.suite_attempt_id}.attempted"

    @property
    def _quarantine_name(self) -> str:
        return f".opened-genesis-{self.suite_attempt_id}.quarantine"

    @property
    def _attempt_marker_bytes(self) -> bytes:
        return (
            _canonical_json(
                {
                    "attestation_descriptor_file_sha256": self.descriptor_file_sha256,
                    "request_sha256": self.request_sha256,
                    "schema_version": _ATTEMPT_MARKER_SCHEMA,
                    "state": "ATTEMPTED",
                    "suite_attempt_id": self.suite_attempt_id,
                }
            )
            + b"\n"
        )

    def __enter__(self) -> _Publication:
        parent = _open_absolute_directory(
            self.namespace.parent,
            label="suite namespace parent",
        )
        self.parent_descriptor = parent
        metadata = os.fstat(parent)
        self.parent_identity = _identity(metadata)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            self.__exit__(None, None, None)
            raise OpenedGenesisOperatorError(
                "suite namespace parent must be private and owned by the current identity"
            )
        lock_name = f".opened-genesis-{self.suite_attempt_id}.lock"
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        created = False
        try:
            try:
                lock = os.open(
                    lock_name,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=parent,
                )
                created = True
            except FileExistsError:
                lock = os.open(lock_name, flags, dir_fd=parent)
        except OSError as exc:
            self.__exit__(None, None, None)
            raise OpenedGenesisOperatorError(
                "cannot open the OPENED-genesis publication lock"
            ) from exc
        self.lock_descriptor = lock
        try:
            if created:
                os.fchmod(lock, 0o600)
                os.fsync(lock)
                os.fsync(parent)
            lock_stat = os.fstat(lock)
            lock_path_stat = os.stat(
                lock_name,
                dir_fd=parent,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(lock_stat.st_mode)
                or lock_stat.st_nlink != 1
                or stat.S_IMODE(lock_stat.st_mode) != 0o600
                or lock_stat.st_size != 0
                or (hasattr(os, "geteuid") and lock_stat.st_uid != os.geteuid())
                or _stable_signature(lock_stat) != _stable_signature(lock_path_stat)
            ):
                raise OpenedGenesisOperatorError(
                    "OPENED-genesis publication lock is not one private empty file"
                )
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise OpenedGenesisOperatorError(
                    "another OPENED-genesis publisher holds the suite lock"
                ) from exc
            self._assert_attempt_marker_absent()
            self._assert_quarantine_absent()
            try:
                os.stat(
                    self.namespace.name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return self
            except OSError as exc:
                raise OpenedGenesisOperatorError(
                    "cannot establish that the suite namespace is absent"
                ) from exc
            raise OpenedGenesisOperatorError(
                "suite namespace already exists; OPENED replay is forbidden"
            )
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _assert_attempt_marker_absent(self) -> None:
        assert self.parent_descriptor is not None
        try:
            marker_metadata = os.stat(
                self._attempt_marker_name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            raise OpenedGenesisOperatorError(
                "cannot establish whether prior OPENED-attempt evidence exists"
            ) from exc

        valid_metadata = (
            stat.S_ISREG(marker_metadata.st_mode)
            and marker_metadata.st_nlink == 1
            and stat.S_IMODE(marker_metadata.st_mode) == 0o600
            and marker_metadata.st_size == len(self._attempt_marker_bytes)
            and (not hasattr(os, "geteuid") or marker_metadata.st_uid == os.geteuid())
        )
        valid_bytes = False
        if valid_metadata:
            try:
                marker_bytes, opened_metadata = _read_member_at(
                    self.parent_descriptor,
                    self._attempt_marker_name,
                    label="OPENED-attempt evidence",
                    max_bytes=len(self._attempt_marker_bytes),
                )
                valid_bytes = marker_bytes == self._attempt_marker_bytes and _stable_signature(
                    opened_metadata
                ) == _stable_signature(marker_metadata)
            except OpenedGenesisOperatorError:
                valid_bytes = False
        if valid_metadata and valid_bytes:
            raise OpenedGenesisOperatorError(
                "prior OPENED-attempt evidence exists; OPENED replay is forbidden"
            )
        raise OpenedGenesisOperatorError(
            "OPENED-attempt evidence is invalid; OPENED replay is forbidden"
        )

    def _assert_quarantine_absent(self) -> None:
        assert self.parent_descriptor is not None
        try:
            os.stat(
                self._quarantine_name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as exc:
            raise OpenedGenesisOperatorError(
                "cannot establish whether an OPENED cleanup quarantine exists"
            ) from exc
        raise OpenedGenesisOperatorError(
            "OPENED cleanup quarantine already exists; publication is forbidden"
        )

    def arm_attempt(self) -> None:
        """Durably reserve this suite ID before any publication primitive runs."""

        if self.parent_descriptor is None or self.lock_descriptor is None:
            raise OpenedGenesisOperatorError("OPENED publication is not locked")
        self.assert_parent_current()
        self._assert_attempt_marker_absent()
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        marker: int | None = None
        created = False
        try:
            try:
                marker = os.open(
                    self._attempt_marker_name,
                    flags,
                    0o600,
                    dir_fd=self.parent_descriptor,
                )
                created = True
            except FileExistsError:
                self._assert_attempt_marker_absent()
                raise AssertionError("existing OPENED-attempt evidence was not rejected")
            except OSError as exc:
                raise OpenedGenesisOperatorError("cannot create OPENED-attempt evidence") from exc

            os.fchmod(marker, 0o600)
            os.fsync(marker)
            os.fsync(self.parent_descriptor)
            encoded = self._attempt_marker_bytes
            offset = 0
            while offset < len(encoded):
                written = os.write(marker, encoded[offset:])
                if written <= 0:
                    raise OSError("OPENED-attempt evidence write made no progress")
                offset += written
            os.fsync(marker)
            marker_metadata = os.fstat(marker)
            marker_path_metadata = os.stat(
                self._attempt_marker_name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            observed = _pread_all(
                marker,
                size=marker_metadata.st_size,
                max_bytes=len(encoded),
                label="OPENED-attempt evidence",
            )
            if (
                observed != encoded
                or not stat.S_ISREG(marker_metadata.st_mode)
                or marker_metadata.st_nlink != 1
                or stat.S_IMODE(marker_metadata.st_mode) != 0o600
                or (hasattr(os, "geteuid") and marker_metadata.st_uid != os.geteuid())
                or _stable_signature(marker_metadata) != _stable_signature(marker_path_metadata)
            ):
                raise OpenedGenesisOperatorError(
                    "OPENED-attempt evidence differs after durable creation"
                )
            os.fsync(self.parent_descriptor)
            self.assert_parent_current()
            self.attempt_marker_descriptor = marker
            marker = None
        except BaseException as exc:
            if created:
                if isinstance(exc, OpenedGenesisOperatorError) and str(exc).startswith(
                    "OPENED publication is indeterminate:"
                ):
                    raise
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    f"attempt evidence persistence failed: {_single_line(exc)}"
                ) from exc
            raise
        finally:
            if marker is not None:
                os.close(marker)

    def assert_attempt_marker_current(self) -> None:
        marker = self.attempt_marker_descriptor
        if marker is None or self.parent_descriptor is None:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: attempt evidence is not pinned"
            )
        try:
            marker_metadata = os.fstat(marker)
            marker_path_metadata = os.stat(
                self._attempt_marker_name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            observed = _pread_all(
                marker,
                size=marker_metadata.st_size,
                max_bytes=len(self._attempt_marker_bytes),
                label="OPENED-attempt evidence",
            )
        except (OpenedGenesisOperatorError, OSError) as exc:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: attempt evidence cannot be revalidated"
            ) from exc
        if (
            observed != self._attempt_marker_bytes
            or not stat.S_ISREG(marker_metadata.st_mode)
            or marker_metadata.st_nlink != 1
            or stat.S_IMODE(marker_metadata.st_mode) != 0o600
            or (hasattr(os, "geteuid") and marker_metadata.st_uid != os.geteuid())
            or _stable_signature(marker_metadata) != _stable_signature(marker_path_metadata)
        ):
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: attempt evidence changed"
            )

    def assert_parent_current(self) -> None:
        if self.parent_descriptor is None or self.parent_identity is None:
            raise OpenedGenesisOperatorError("suite namespace parent is not pinned")
        reopened = _open_absolute_directory(
            self.namespace.parent,
            label="suite namespace parent",
        )
        try:
            if (
                _identity(os.fstat(self.parent_descriptor)) != self.parent_identity
                or _identity(os.fstat(reopened)) != self.parent_identity
            ):
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: suite namespace parent path was replaced"
                )
        finally:
            os.close(reopened)

    def _open_namespace(self) -> int | None:
        assert self.parent_descriptor is not None
        try:
            return os.open(
                self.namespace.name,
                _directory_flags(),
                dir_fd=self.parent_descriptor,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: namespace is not a real directory"
            ) from exc

    @staticmethod
    def _require_private_directory(
        descriptor: int,
        *,
        label: str,
    ) -> os.stat_result:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise OpenedGenesisOperatorError(
                f"OPENED publication is indeterminate: {label} mode or owner differs"
            )
        return metadata

    def _restore_quarantined_entry(
        self,
        *,
        moved_identity: tuple[int, int],
    ) -> None:
        assert self.parent_descriptor is not None
        try:
            _rename_no_replace_at(
                self.parent_descriptor,
                self._quarantine_name,
                self.parent_descriptor,
                self.namespace.name,
            )
        except (OpenedGenesisOperatorError, OSError) as exc:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: "
                "a replacement namespace entry was quarantined and could not be restored"
            ) from exc
        try:
            restored = os.stat(
                self.namespace.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            try:
                os.stat(
                    self._quarantine_name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    "cleanup quarantine remained after replacement restoration"
                )
            if _identity(restored) != moved_identity:
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: restored namespace entry identity differs"
                )
            os.fsync(self.parent_descriptor)
        except OpenedGenesisOperatorError:
            raise
        except OSError as exc:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: "
                "replacement namespace restoration could not be synchronized"
            ) from exc
        raise OpenedGenesisOperatorError(
            "OPENED publication is indeterminate: "
            "namespace changed at the quarantine boundary; replacement entry restored"
        )

    def _quarantine_partial_namespace(
        self,
        namespace_descriptor: int,
        namespace_metadata: os.stat_result,
    ) -> None:
        assert self.parent_descriptor is not None
        admitted_identity = _identity(namespace_metadata)
        try:
            _rename_no_replace_at(
                self.parent_descriptor,
                self.namespace.name,
                self.parent_descriptor,
                self._quarantine_name,
            )
        except FileExistsError as exc:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: cleanup quarantine already exists"
            ) from exc
        except FileNotFoundError as exc:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: namespace disappeared before quarantine"
            ) from exc
        except (OpenedGenesisOperatorError, OSError) as exc:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: namespace quarantine failed"
            ) from exc

        try:
            moved = os.stat(
                self._quarantine_name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: "
                "quarantined namespace entry cannot be inspected"
            ) from exc
        moved_identity = _identity(moved)
        if moved_identity != admitted_identity:
            self._restore_quarantined_entry(moved_identity=moved_identity)

        try:
            descriptor_metadata = os.fstat(namespace_descriptor)
            if (
                _identity(descriptor_metadata) != admitted_identity
                or not stat.S_ISDIR(moved.st_mode)
                or stat.S_IMODE(moved.st_mode) != 0o700
                or (hasattr(os, "geteuid") and moved.st_uid != os.geteuid())
            ):
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    "quarantined namespace differs from its retained descriptor"
                )
            try:
                os.stat(
                    self.namespace.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    "canonical namespace was repopulated after quarantine"
                )
            os.fsync(namespace_descriptor)
            os.fsync(self.parent_descriptor)
            final_quarantine = os.stat(
                self._quarantine_name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            if _identity(final_quarantine) != admitted_identity:
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    "cleanup quarantine identity changed during synchronization"
                )
            try:
                os.stat(
                    self.namespace.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    "canonical namespace reappeared during quarantine synchronization"
                )
            self.assert_parent_current()
            if self.attempt_marker_descriptor is not None:
                self.assert_attempt_marker_current()
        except OpenedGenesisOperatorError:
            raise
        except OSError as exc:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: namespace quarantine synchronization failed"
            ) from exc

    def cleanup_partial(self) -> None:
        assert self.parent_descriptor is not None
        self.assert_parent_current()
        if self.attempt_marker_descriptor is not None:
            self.assert_attempt_marker_current()
        namespace_descriptor = self._open_namespace()
        if namespace_descriptor is None:
            return
        try:
            namespace_metadata = self._require_private_directory(
                namespace_descriptor,
                label="suite namespace",
            )
            if (
                self.namespace_identity is not None
                and _identity(namespace_metadata) != self.namespace_identity
            ):
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    "namespace identity differs from the admitted output"
                )
            entries = set(os.listdir(namespace_descriptor))
            unknown = entries - _OUTPUT_MEMBERS
            if unknown:
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    f"unexpected namespace members {sorted(unknown)!r}"
                )
            if "attestation-descriptor.json" not in entries:
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    "partial namespace lacks its request-bound provenance file"
                )
            descriptor_bytes, descriptor_metadata = _read_member_at(
                namespace_descriptor,
                "attestation-descriptor.json",
                label="partial OPENED attestation descriptor",
                max_bytes=_MAX_INPUT_BYTES,
            )
            if _digest(descriptor_bytes) != self.descriptor_file_sha256:
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    "partial namespace provenance differs from the custody request"
                )
            member_metadata = {
                "attestation-descriptor.json": descriptor_metadata,
            }
            if "000.state.json" in entries:
                _, state_metadata = _read_member_at(
                    namespace_descriptor,
                    "000.state.json",
                    label="partial OPENED state",
                    max_bytes=_MAX_INPUT_BYTES,
                )
                member_metadata["000.state.json"] = state_metadata
            online_metadata: os.stat_result | None = None
            if "online" in entries:
                try:
                    online = os.open(
                        "online",
                        _directory_flags(),
                        dir_fd=namespace_descriptor,
                    )
                except OSError as exc:
                    raise OpenedGenesisOperatorError(
                        "OPENED publication is indeterminate: online is not a real directory"
                    ) from exc
                try:
                    online_metadata = self._require_private_directory(
                        online,
                        label="online directory",
                    )
                    if os.listdir(online):
                        raise OpenedGenesisOperatorError(
                            "OPENED publication is indeterminate: online is not empty"
                        )
                finally:
                    os.close(online)

            current_namespace = os.stat(
                self.namespace.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            if _identity(current_namespace) != _identity(namespace_metadata):
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    "namespace path changed before fail-clean removal"
                )
            for name, admitted_metadata in member_metadata.items():
                current_metadata = os.stat(
                    name,
                    dir_fd=namespace_descriptor,
                    follow_symlinks=False,
                )
                if _stable_signature(current_metadata) != _stable_signature(admitted_metadata):
                    raise OpenedGenesisOperatorError(
                        "OPENED publication is indeterminate: "
                        f"{name} changed before fail-clean removal"
                    )
            if online_metadata is not None:
                current_online = os.stat(
                    "online",
                    dir_fd=namespace_descriptor,
                    follow_symlinks=False,
                )
                if _stable_signature(current_online) != _stable_signature(online_metadata):
                    raise OpenedGenesisOperatorError(
                        "OPENED publication is indeterminate: "
                        "online changed before fail-clean removal"
                    )
            observed_signatures = {
                name: _stable_signature(metadata) for name, metadata in member_metadata.items()
            }
            if online_metadata is not None:
                observed_signatures["online"] = _stable_signature(online_metadata)
            for name, admitted_signature in self.output_member_signatures.items():
                if observed_signatures.get(name) != admitted_signature:
                    raise OpenedGenesisOperatorError(
                        "OPENED publication is indeterminate: "
                        f"{name} differs from the admitted output"
                    )
            self._quarantine_partial_namespace(
                namespace_descriptor,
                namespace_metadata,
            )
        except OpenedGenesisOperatorError:
            raise
        except OSError as exc:
            raise OpenedGenesisOperatorError(
                "OPENED publication is indeterminate: fail-clean quarantine failed"
            ) from exc
        finally:
            os.close(namespace_descriptor)

    def admit_namespace(self) -> None:
        """Bind cleanup to the exact directory returned by the state primitive."""

        self.assert_parent_current()
        self.assert_attempt_marker_current()
        namespace_descriptor = self._open_namespace()
        if namespace_descriptor is None:
            raise OpenedGenesisOperatorError(
                "OPENED publisher returned without its suite namespace"
            )
        try:
            metadata = self._require_private_directory(
                namespace_descriptor,
                label="suite namespace",
            )
            descriptor_bytes, descriptor_metadata = _read_member_at(
                namespace_descriptor,
                "attestation-descriptor.json",
                label="published attestation descriptor",
                max_bytes=_MAX_INPUT_BYTES,
            )
            if _digest(descriptor_bytes) != self.descriptor_file_sha256:
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: "
                    "published namespace provenance differs from the custody request"
                )
            self.namespace_identity = _identity(metadata)
            self.output_member_signatures = {
                "attestation-descriptor.json": _stable_signature(descriptor_metadata)
            }
        finally:
            os.close(namespace_descriptor)

    def verify_complete(
        self,
        *,
        descriptor_file_sha256: str,
        manifest_sha256: str,
    ) -> object:
        assert self.parent_descriptor is not None
        self.assert_parent_current()
        self.assert_attempt_marker_current()
        namespace_descriptor = self._open_namespace()
        if namespace_descriptor is None:
            raise OpenedGenesisOperatorError(
                "OPENED publisher returned without its suite namespace"
            )
        try:
            namespace_metadata = self._require_private_directory(
                namespace_descriptor,
                label="suite namespace",
            )
            observed_namespace_identity = _identity(namespace_metadata)
            if (
                self.namespace_identity is not None
                and observed_namespace_identity != self.namespace_identity
            ):
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: namespace was replaced before readback"
                )
            self.namespace_identity = observed_namespace_identity
            entries = set(os.listdir(namespace_descriptor))
            if entries != _OUTPUT_MEMBERS:
                raise OpenedGenesisOperatorError("OPENED suite namespace inventory differs")
            online = os.open(
                "online",
                _directory_flags(),
                dir_fd=namespace_descriptor,
            )
            try:
                online_metadata = self._require_private_directory(
                    online,
                    label="online directory",
                )
                if os.listdir(online):
                    raise OpenedGenesisOperatorError("OPENED online directory must be empty")
                os.fsync(online)
            finally:
                os.close(online)
            descriptor_bytes, descriptor_metadata = _read_member_at(
                namespace_descriptor,
                "attestation-descriptor.json",
                label="published attestation descriptor",
                max_bytes=_MAX_INPUT_BYTES,
            )
            state_bytes, state_metadata = _read_member_at(
                namespace_descriptor,
                "000.state.json",
                label="published OPENED state",
                max_bytes=_MAX_INPUT_BYTES,
            )
            if (
                stat.S_IMODE(descriptor_metadata.st_mode) != 0o600
                or _digest(descriptor_bytes) != descriptor_file_sha256
            ):
                raise OpenedGenesisOperatorError(
                    "published attestation descriptor bytes or mode differ"
                )
            if stat.S_IMODE(state_metadata.st_mode) != 0o600:
                raise OpenedGenesisOperatorError("published OPENED state mode must equal 0600")
            self.output_member_signatures = {
                "000.state.json": _stable_signature(state_metadata),
                "attestation-descriptor.json": _stable_signature(descriptor_metadata),
                "online": _stable_signature(online_metadata),
            }
            os.fsync(namespace_descriptor)
            os.fsync(self.parent_descriptor)

            state_path = self.namespace / "000.state.json"
            try:
                record = load_suite_state_record(state_path)
            except (TypeError, ValueError) as exc:
                raise OpenedGenesisOperatorError(
                    f"published OPENED state is invalid: {exc}"
                ) from exc
            if state_bytes != record.canonical_bytes() + b"\n":
                raise OpenedGenesisOperatorError(
                    "published OPENED state readback differs from pinned bytes"
                )
            self.assert_parent_current()
            reopened_namespace = self._open_namespace()
            if reopened_namespace is None:
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: namespace disappeared during readback"
                )
            try:
                if _identity(os.fstat(reopened_namespace)) != _identity(namespace_metadata):
                    raise OpenedGenesisOperatorError(
                        "OPENED publication is indeterminate: "
                        "namespace was replaced during readback"
                    )
            finally:
                os.close(reopened_namespace)
            final_descriptor_bytes, final_descriptor_metadata = _read_member_at(
                namespace_descriptor,
                "attestation-descriptor.json",
                label="published attestation descriptor after typed readback",
                max_bytes=_MAX_INPUT_BYTES,
            )
            final_state_bytes, final_state_metadata = _read_member_at(
                namespace_descriptor,
                "000.state.json",
                label="published OPENED state after typed readback",
                max_bytes=_MAX_INPUT_BYTES,
            )
            if (
                final_descriptor_bytes != descriptor_bytes
                or _stable_signature(final_descriptor_metadata)
                != _stable_signature(descriptor_metadata)
                or final_state_bytes != state_bytes
                or _stable_signature(final_state_metadata) != _stable_signature(state_metadata)
            ):
                raise OpenedGenesisOperatorError(
                    "OPENED publication is indeterminate: output changed during typed readback"
                )
            os.fsync(namespace_descriptor)
            os.fsync(self.parent_descriptor)
            if (
                record.state != "OPENED"
                or record.sequence != 0
                or record.previous_state_record_sha256 is not None
                or record.suite_attempt_id != self.suite_attempt_id
                or record.manifest_sha256 != manifest_sha256
                or record.namespace_uri != self.namespace.as_uri()
            ):
                raise OpenedGenesisOperatorError(
                    "published suite state is not the expected OPENED genesis"
                )
            self.assert_attempt_marker_current()
            return record
        finally:
            os.close(namespace_descriptor)

    def __exit__(self, *_: object) -> None:
        if self.attempt_marker_descriptor is not None:
            os.close(self.attempt_marker_descriptor)
            self.attempt_marker_descriptor = None
        if self.lock_descriptor is not None:
            try:
                fcntl.flock(self.lock_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self.lock_descriptor)
            self.lock_descriptor = None
        if self.parent_descriptor is not None:
            os.close(self.parent_descriptor)
            self.parent_descriptor = None


@contextmanager
def _private_creation_umask() -> Iterator[None]:
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


@contextmanager
def _publication_signal_guard() -> Iterator[None]:
    watched = tuple(
        value
        for value in (
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGTERM", None),
        )
        if isinstance(value, signal.Signals)
    )
    previous: dict[signal.Signals, object] = {}

    def terminate(signum: int, _frame: object) -> None:
        raise _TerminationSignal(signum)

    try:
        for signum in watched:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, terminate)
    except ValueError:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except ValueError:
                pass
        previous.clear()
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def execute(
    request_path: str | Path,
    *,
    expected_request_sha256: str,
) -> dict[str, object]:
    """Verify all inputs and exclusively publish the manifest-derived OPENED state."""

    request_sha256 = _sha256("request_sha256", expected_request_sha256)
    request = _canonical_path("request path", str(request_path))
    request_pin = _PinnedFile.open(
        request,
        label="OPENED-genesis request",
        expected_sha256=request_sha256,
        max_bytes=_MAX_REQUEST_BYTES,
    )
    inputs: _PinnedInputs | None = None
    mutation_guard: _PathMutationGuard | None = None
    try:
        parsed = OpenedGenesisRequest.from_bytes(request_pin.data)
        inputs = _PinnedInputs.open(parsed)
        if request_pin.file_identity in {inputs[role].file_identity for role in _expected_roles()}:
            raise OpenedGenesisOperatorError("OPENED-genesis request cannot also serve as an input")
        mutation_guard = _PathMutationGuard.open((request_pin, *inputs.pins()))
        request_pin.assert_current()
        inputs.assert_current()
        mutation_guard.checkpoint()
        package_root = _package_root(inputs)
        manifest = _manifest_from_package(inputs)
        (
            registration,
            finalization_receipt,
            closures,
            path_sets,
            descriptor,
        ) = _typed_open_arguments(
            inputs,
            package_root=package_root,
            manifest=manifest,
        )
        mutation_guard.checkpoint()
        namespace = Path(finalization_receipt.canonical_suite_namespace)
        if (
            not namespace.is_absolute()
            or str(namespace) != finalization_receipt.canonical_suite_namespace
        ):
            raise OpenedGenesisOperatorError(
                "finalization receipt suite namespace is not canonical absolute"
            )
        publication = _Publication(
            namespace,
            finalization_receipt.suite_attempt_id,
            descriptor_file_sha256=inputs["suite-attestation-descriptor"].expected_sha256,
            request_sha256=request_sha256,
        )
        with publication:
            try:
                with _publication_signal_guard():
                    request_pin.assert_current()
                    inputs.assert_current()
                    mutation_guard.checkpoint()
                    publication.assert_parent_current()
                    publication.arm_attempt()
                    mutation_guard.checkpoint()
                    with _private_creation_umask():
                        opened = open_suite_attempt(
                            manifest,
                            verified_protocol_registration=registration,
                            production_finalization_receipt_path=(
                                inputs["production-finalization-receipt"].path
                            ),
                            verified_production_closures=closures,
                            run_receipt_path=inputs["sealed-run-receipt"].path,
                            preflight_contract_paths=path_sets["preflight_contract_paths"],
                            runtime_preflight_receipt_paths=path_sets[
                                "runtime_preflight_receipt_paths"
                            ],
                            runtime_plan_transition_paths=path_sets[
                                "runtime_plan_transition_paths"
                            ],
                            registered_plan_instantiation_paths=path_sets[
                                "registered_plan_instantiation_paths"
                            ],
                            sealed_launch_contract_paths=path_sets["sealed_launch_contract_paths"],
                            execution_artifacts=_execution_artifacts(manifest),
                            attestation_descriptor=descriptor,
                        )
                    mutation_guard.checkpoint()
                    publication.assert_parent_current()
                    if opened != namespace:
                        raise OpenedGenesisOperatorError("suite opener returned another namespace")
                    publication.admit_namespace()
                    request_pin.assert_current()
                    inputs.assert_current()
                    state = publication.verify_complete(
                        descriptor_file_sha256=inputs[
                            "suite-attestation-descriptor"
                        ].expected_sha256,
                        manifest_sha256=registration.manifest_sha256,
                    )
                    request_pin.assert_current()
                    inputs.assert_current()
                    mutation_guard.checkpoint()
            except BaseException as exc:
                try:
                    publication.cleanup_partial()
                except OpenedGenesisOperatorError:
                    raise
                if isinstance(exc, OpenedGenesisOperatorError):
                    raise
                if isinstance(exc, Exception):
                    raise OpenedGenesisOperatorError(
                        f"OPENED genesis was not published: {_single_line(exc)}"
                    ) from exc
                raise
        return {
            "manifest_sha256": state.manifest_sha256,
            "namespace": str(namespace),
            "request_sha256": request_sha256,
            "schema_version": "fractal-opened-genesis-result-v1",
            "state": "OPENED",
            "state_record_sha256": state.record_sha256,
            "suite_attempt_id": state.suite_attempt_id,
        }
    finally:
        if mutation_guard is not None:
            mutation_guard.close()
        if inputs is not None:
            inputs.close()
        request_pin.close()


def _single_line(value: object) -> str:
    text = " ".join(str(value).splitlines()).strip()
    return text or type(value).__name__


class _OneLineParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _OneLineParser(
        prog="opened-genesis",
        description="Publish the sole manifest-derived OPENED suite genesis.",
        allow_abbrev=False,
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = execute(
            args.request,
            expected_request_sha256=args.request_sha256,
        )
    except _ArgumentError as exc:
        print(f"error: {_single_line(exc)}", file=sys.stderr)
        return 2
    except _TerminationSignal as exc:
        print(f"error: {_single_line(exc)}", file=sys.stderr)
        return 128 + exc.signum
    except (OpenedGenesisOperatorError, OSError, ValueError) as exc:
        print(f"error: {_single_line(exc)}", file=sys.stderr)
        return 1
    print(_canonical_json(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
