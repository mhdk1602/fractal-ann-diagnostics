"""Local artifact verification for sealed study execution.

The verifier has no network code. Callers must download remote artifacts into an
isolated local root before constructing :class:`LocalArtifactSpec` records.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

ArtifactKind = Literal["file", "directory"]
EntryKind = Literal["file", "directory"]

ARTIFACT_RECEIPT_SCHEMA = "fractal-artifact-verification-v1"
LOCAL_ARTIFACT_MAP_SCHEMA = "fractal-local-artifact-map-v1"
DIRECTORY_DIGEST_SCHEMA = "fractal-directory-tree-sha256-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_READ_CHUNK_BYTES = 1024 * 1024
_MAX_CONTROL_FILE_BYTES = 16 * 1024 * 1024

_VERIFIED_ARTIFACT_FIELDS = {
    "artifact_id",
    "byte_count",
    "directory_count",
    "exact",
    "expected_sha256",
    "file_count",
    "kind",
    "observed_byte_count",
    "observed_directory_count",
    "observed_file_count",
    "relative_path",
    "verified_sha256",
}
_ARTIFACT_RECEIPT_FIELDS = {"artifacts", "manifest_sha256", "schema_version"}
_LOCAL_ARTIFACT_MAP_FIELDS = {"artifacts", "schema_version"}
_LOCAL_ARTIFACT_ENTRY_FIELDS = {
    "artifact_id",
    "expected_entries",
    "kind",
    "relative_path",
}
_LOCAL_ARTIFACT_ENTRY_REQUIRED_FIELDS = {
    "artifact_id",
    "kind",
    "relative_path",
}


class ArtifactIntegrityError(ValueError):
    """Raised when a local artifact cannot be verified exactly as declared."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _closed_mapping(
    value: object,
    *,
    fields: set[str],
    required: set[str] | None = None,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ArtifactIntegrityError(f"{label} keys must be strings")
    observed = set(value)
    unknown = observed - fields
    if unknown:
        raise ArtifactIntegrityError(f"{label} has unknown fields: {sorted(unknown)}")
    missing = (fields if required is None else required) - observed
    if missing:
        raise ArtifactIntegrityError(f"{label} is missing fields: {sorted(missing)}")
    return value


def _parse_json_object(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ArtifactIntegrityError(f"{label} contains duplicate key {key!r}")
            parsed[key] = value
        return parsed

    def reject_nonfinite(value: str) -> None:
        raise ArtifactIntegrityError(f"{label} contains non-finite number {value!r}")

    try:
        text = payload.decode("utf-8", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise ArtifactIntegrityError(f"{label} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(f"{label} must be valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise ArtifactIntegrityError(f"{label} must contain one JSON object")
    return parsed


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactIntegrityError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_identifier(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ArtifactIntegrityError("artifact_id must be a non-empty canonical string")
    if unicodedata.normalize("NFC", value) != value:
        raise ArtifactIntegrityError("artifact_id must use NFC Unicode normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ArtifactIntegrityError("artifact_id cannot contain control characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ArtifactIntegrityError("artifact_id must be valid UTF-8") from exc
    return value


def _relative_parts(value: str, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise ArtifactIntegrityError(f"{name} must be a non-empty local path")
    if _URI_SCHEME.match(value):
        raise ArtifactIntegrityError(f"{name} must be local; URI schemes are not accepted")
    if "\\" in value:
        raise ArtifactIntegrityError(f"{name} must use POSIX separators")
    if unicodedata.normalize("NFC", value) != value:
        raise ArtifactIntegrityError(f"{name} must use NFC Unicode normalization")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ArtifactIntegrityError(f"{name} must be valid UTF-8") from exc
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value:
        raise ArtifactIntegrityError(f"{name} must be a canonical relative POSIX path")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ArtifactIntegrityError(f"{name} cannot contain dot or parent traversal")
    for part in parts:
        if any(ord(character) < 32 or ord(character) == 127 for character in part):
            raise ArtifactIntegrityError(f"{name} cannot contain control characters")
    return tuple(parts)


def _path_sort_key(value: str) -> bytes:
    return value.encode("utf-8", errors="strict")


def _normalize_expected_entries(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ArtifactIntegrityError(f"{name} must be a sequence of relative paths")
    normalized: list[str] = []
    for position, value in enumerate(values):
        parts = _relative_parts(value, name=f"{name}[{position}]")
        normalized.append("/".join(parts))
    if len(normalized) != len(set(normalized)):
        raise ArtifactIntegrityError(f"{name} contains duplicate paths")
    return tuple(sorted(normalized, key=_path_sort_key))


@dataclass(frozen=True)
class LocalArtifactSpec:
    """One manifest-derived local artifact declaration.

    ``expected_entries`` contains every file and directory path relative to a
    directory artifact. With ``exact=True``, any undeclared path is rejected.
    With ``exact=False``, the digest covers only these entries while the full
    tree is still scanned for symlinks and non-regular objects.
    """

    artifact_id: str
    relative_path: str
    kind: ArtifactKind
    expected_sha256: str
    exact: bool = True
    expected_entries: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.artifact_id)
        parts = _relative_parts(self.relative_path, name="relative_path")
        object.__setattr__(self, "relative_path", "/".join(parts))
        if self.kind not in {"file", "directory"}:
            raise ArtifactIntegrityError("kind must be 'file' or 'directory'")
        _require_sha256("expected_sha256", self.expected_sha256)
        if not isinstance(self.exact, bool):
            raise ArtifactIntegrityError("exact must be boolean")

        entries = self.expected_entries
        if entries is not None:
            entries = _normalize_expected_entries(entries, name="expected_entries")
            object.__setattr__(self, "expected_entries", entries)
        if self.kind == "file":
            if not self.exact:
                raise ArtifactIntegrityError("file artifacts must be exact")
            if entries is not None:
                raise ArtifactIntegrityError("file artifacts cannot declare expected_entries")
        elif not self.exact and not entries:
            raise ArtifactIntegrityError(
                "non-exact directory artifacts need at least one expected entry"
            )


@dataclass(frozen=True)
class DirectoryDigest:
    """A deterministic digest plus selected and observed tree accounting."""

    sha256: str
    entries: tuple[str, ...]
    file_count: int
    directory_count: int
    byte_count: int
    observed_file_count: int
    observed_directory_count: int
    observed_byte_count: int

    def __post_init__(self) -> None:
        _require_sha256("sha256", self.sha256)
        entries = _normalize_expected_entries(self.entries, name="entries")
        object.__setattr__(self, "entries", entries)
        for name in (
            "file_count",
            "directory_count",
            "byte_count",
            "observed_file_count",
            "observed_directory_count",
            "observed_byte_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ArtifactIntegrityError(f"{name} must be a non-negative integer")
        if self.file_count > self.observed_file_count:
            raise ArtifactIntegrityError("file_count cannot exceed observed_file_count")
        if self.directory_count > self.observed_directory_count:
            raise ArtifactIntegrityError("directory_count cannot exceed observed_directory_count")
        if self.byte_count > self.observed_byte_count:
            raise ArtifactIntegrityError("byte_count cannot exceed observed_byte_count")
        if len(entries) != self.file_count + self.directory_count:
            raise ArtifactIntegrityError(
                "entries must account for every selected file and directory"
            )


@dataclass(frozen=True)
class VerifiedArtifact:
    """Receipt row for one successfully verified artifact."""

    artifact_id: str
    relative_path: str
    kind: ArtifactKind
    exact: bool
    expected_sha256: str
    verified_sha256: str
    file_count: int
    directory_count: int
    byte_count: int
    observed_file_count: int
    observed_directory_count: int
    observed_byte_count: int

    def __post_init__(self) -> None:
        _require_identifier(self.artifact_id)
        _relative_parts(self.relative_path, name="relative_path")
        if self.kind not in {"file", "directory"}:
            raise ArtifactIntegrityError("kind must be 'file' or 'directory'")
        if not isinstance(self.exact, bool):
            raise ArtifactIntegrityError("exact must be boolean")
        _require_sha256("expected_sha256", self.expected_sha256)
        _require_sha256("verified_sha256", self.verified_sha256)
        if self.expected_sha256 != self.verified_sha256:
            raise ArtifactIntegrityError("a verified artifact must match its expected SHA-256")
        for name in (
            "file_count",
            "directory_count",
            "byte_count",
            "observed_file_count",
            "observed_directory_count",
            "observed_byte_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ArtifactIntegrityError(f"{name} must be a non-negative integer")
        if self.exact and (
            self.file_count != self.observed_file_count
            or self.directory_count != self.observed_directory_count
            or self.byte_count != self.observed_byte_count
        ):
            raise ArtifactIntegrityError("exact artifact accounting must cover the observed tree")
        if self.file_count > self.observed_file_count:
            raise ArtifactIntegrityError("file_count cannot exceed observed_file_count")
        if self.directory_count > self.observed_directory_count:
            raise ArtifactIntegrityError("directory_count cannot exceed observed_directory_count")
        if self.byte_count > self.observed_byte_count:
            raise ArtifactIntegrityError("byte_count cannot exceed observed_byte_count")
        if self.kind == "file" and (
            not self.exact
            or self.file_count != 1
            or self.directory_count != 0
            or self.observed_file_count != 1
            or self.observed_directory_count != 0
        ):
            raise ArtifactIntegrityError("file artifact accounting must describe one exact file")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "byte_count": self.byte_count,
            "directory_count": self.directory_count,
            "exact": self.exact,
            "expected_sha256": self.expected_sha256,
            "file_count": self.file_count,
            "kind": self.kind,
            "observed_byte_count": self.observed_byte_count,
            "observed_directory_count": self.observed_directory_count,
            "observed_file_count": self.observed_file_count,
            "relative_path": self.relative_path,
            "verified_sha256": self.verified_sha256,
        }

    @classmethod
    def from_dict(cls, payload: object) -> VerifiedArtifact:
        """Parse one receipt row through the same closed validation contract."""

        row = _closed_mapping(
            payload,
            fields=_VERIFIED_ARTIFACT_FIELDS,
            label="verified artifact",
        )
        return cls(
            artifact_id=row["artifact_id"],
            relative_path=row["relative_path"],
            kind=row["kind"],
            exact=row["exact"],
            expected_sha256=row["expected_sha256"],
            verified_sha256=row["verified_sha256"],
            file_count=row["file_count"],
            directory_count=row["directory_count"],
            byte_count=row["byte_count"],
            observed_file_count=row["observed_file_count"],
            observed_directory_count=row["observed_directory_count"],
            observed_byte_count=row["observed_byte_count"],
        )


@dataclass(frozen=True)
class ArtifactVerificationReceipt:
    """Canonical verification evidence bound to one study-manifest digest."""

    manifest_sha256: str
    artifacts: tuple[VerifiedArtifact, ...]
    schema_version: str = ARTIFACT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("manifest_sha256", self.manifest_sha256)
        if self.schema_version != ARTIFACT_RECEIPT_SCHEMA:
            raise ArtifactIntegrityError(f"schema_version must equal {ARTIFACT_RECEIPT_SCHEMA!r}")
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ArtifactIntegrityError("a verification receipt needs at least one artifact")
        if not all(isinstance(artifact, VerifiedArtifact) for artifact in artifacts):
            raise ArtifactIntegrityError("artifacts must contain VerifiedArtifact records")
        artifacts = tuple(
            sorted(
                artifacts,
                key=lambda item: (
                    _path_sort_key(item.artifact_id),
                    _path_sort_key(item.relative_path),
                ),
            )
        )
        identifiers = [artifact.artifact_id for artifact in artifacts]
        paths = [artifact.relative_path for artifact in artifacts]
        if len(identifiers) != len(set(identifiers)):
            raise ArtifactIntegrityError("verification receipt contains duplicate artifact IDs")
        if len(paths) != len(set(paths)):
            raise ArtifactIntegrityError("verification receipt contains duplicate artifact paths")
        for position, first in enumerate(artifacts):
            first_parts = PurePosixPath(first.relative_path).parts
            for second in artifacts[position + 1 :]:
                second_parts = PurePosixPath(second.relative_path).parts
                common = min(len(first_parts), len(second_parts))
                if first_parts[:common] == second_parts[:common]:
                    raise ArtifactIntegrityError(
                        "verification receipt contains overlapping artifact paths"
                    )
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "manifest_sha256": self.manifest_sha256,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        """Return stable UTF-8 JSON with no machine path or wall-clock field."""
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> ArtifactVerificationReceipt:
        """Parse a closed receipt object and restore all dataclass invariants."""

        root = _closed_mapping(
            payload,
            fields=_ARTIFACT_RECEIPT_FIELDS,
            label="artifact verification receipt",
        )
        artifact_values = root["artifacts"]
        if isinstance(artifact_values, (str, bytes)) or not isinstance(artifact_values, Sequence):
            raise ArtifactIntegrityError("artifact verification receipt artifacts must be an array")
        return cls(
            manifest_sha256=root["manifest_sha256"],
            artifacts=tuple(VerifiedArtifact.from_dict(row) for row in artifact_values),
            schema_version=root["schema_version"],
        )

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class _TreeEntry:
    relative_path: str
    kind: EntryKind
    size_bytes: int = 0
    sha256: str | None = None

    def to_digest_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "path": self.relative_path,
        }
        if self.kind == "file":
            payload["sha256"] = self.sha256
            payload["size_bytes"] = self.size_bytes
        return payload


def _require_secure_filesystem_primitives() -> None:
    missing_flags = [
        name
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        if not hasattr(os, name)
    ]
    if missing_flags or os.open not in os.supports_dir_fd:
        missing = ", ".join(missing_flags) or "dir_fd support"
        raise ArtifactIntegrityError(
            f"sealed artifact verification is unavailable on this platform: {missing}"
        )
    if os.stat not in os.supports_dir_fd or os.stat not in os.supports_follow_symlinks:
        raise ArtifactIntegrityError(
            "sealed artifact verification needs non-following dir_fd stat support"
        )


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _file_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def _open_error(label: str, exc: OSError) -> ArtifactIntegrityError:
    if exc.errno == errno.ENOENT:
        return ArtifactIntegrityError(f"{label} is missing")
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        return ArtifactIntegrityError(
            f"{label} crosses a symlink or contains a non-directory ancestor"
        )
    return ArtifactIntegrityError(f"cannot open {label}: {exc.strerror or exc}")


def _open_absolute_directory(path: str | Path, *, label: str) -> int:
    _require_secure_filesystem_primitives()
    target = Path(path)
    if not target.is_absolute():
        raise ArtifactIntegrityError(f"{label} must be an absolute path")
    if any(part in {".", ".."} for part in target.parts):
        raise ArtifactIntegrityError(f"{label} cannot contain dot or parent traversal")
    if target.anchor != "/":
        raise ArtifactIntegrityError(f"{label} must be rooted on a POSIX filesystem")

    try:
        descriptor = os.open("/", _directory_flags())
    except OSError as exc:
        raise _open_error(label, exc) from exc
    try:
        for component in target.parts[1:]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise _open_error(label, exc) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ArtifactIntegrityError(f"{label} must be a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_secure_control_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int = _MAX_CONTROL_FILE_BYTES,
) -> bytes:
    """Read a small absolute file without following any component or hard link."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    target = Path(path)
    if not target.is_absolute() or target.name in {"", ".", ".."}:
        raise ArtifactIntegrityError(f"{label} must be an absolute file path")
    parent = _open_absolute_directory(target.parent, label=f"{label} parent")
    try:
        try:
            descriptor = os.open(target.name, _file_flags(), dir_fd=parent)
        except OSError as exc:
            raise _open_error(label, exc) from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactIntegrityError(f"{label} must be a regular file")
            if before.st_nlink != 1:
                raise ArtifactIntegrityError(f"hard-linked {label} is forbidden")
            if before.st_size > max_bytes:
                raise ArtifactIntegrityError(f"{label} exceeds {max_bytes} bytes")
            chunks: list[bytes] = []
            byte_count = 0
            while True:
                chunk = os.read(descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > max_bytes:
                    raise ArtifactIntegrityError(f"{label} exceeds {max_bytes} bytes")
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                _stable_stat_signature(before) != _stable_stat_signature(after)
                or byte_count != before.st_size
            ):
                raise ArtifactIntegrityError(f"{label} changed during read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def read_secure_control_file(path: str | Path, *, label: str = "control file") -> bytes:
    """Read one small absolute control file without links or concurrent mutation."""

    return _read_secure_control_file(path, label=label)


def read_secure_regular_file(
    path: str | Path,
    *,
    max_bytes: int,
    label: str = "file",
) -> bytes:
    """Read an absolute regular file under an explicit size bound without links."""

    return _read_secure_control_file(path, label=label, max_bytes=max_bytes)


def _open_directory_at(root_descriptor: int, parts: Sequence[str], *, label: str) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for component in parts:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise _open_error(label, exc) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ArtifactIntegrityError(f"{label} must be a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _stable_stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _hash_regular_file_at(
    parent_descriptor: int,
    filename: str,
    *,
    relative_path: str,
    expected_identity: tuple[int, int] | None = None,
) -> _TreeEntry:
    try:
        descriptor = os.open(filename, _file_flags(), dir_fd=parent_descriptor)
    except OSError as exc:
        raise _open_error(relative_path, exc) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactIntegrityError(
                f"{relative_path} is a symlink or non-regular filesystem entry"
            )
        if before.st_nlink != 1:
            raise ArtifactIntegrityError(
                f"hard-linked file is forbidden in sealed artifacts: {relative_path}"
            )
        if expected_identity is not None and (before.st_dev, before.st_ino) != expected_identity:
            raise ArtifactIntegrityError(f"{relative_path} changed during verification")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        if (
            _stable_stat_signature(before) != _stable_stat_signature(after)
            or byte_count != before.st_size
        ):
            raise ArtifactIntegrityError(f"{relative_path} changed during verification")
        return _TreeEntry(
            relative_path=relative_path,
            kind="file",
            size_bytes=byte_count,
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(descriptor)


def digest_regular_file(path: str | Path, *, label: str = "file") -> str:
    """Hash one absolute regular file without following links or accepting hard links."""

    target = Path(path)
    if not target.is_absolute() or target.name in {"", ".", ".."}:
        raise ArtifactIntegrityError(f"{label} must be an absolute file path")
    parent = _open_absolute_directory(target.parent, label=f"{label} parent")
    try:
        return _hash_regular_file_at(
            parent,
            target.name,
            relative_path=label,
        ).sha256
    finally:
        os.close(parent)


def _validate_filesystem_name(name: str, *, relative_path: str) -> None:
    if name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ArtifactIntegrityError(f"non-canonical filesystem entry: {relative_path!r}")
    if unicodedata.normalize("NFC", name) != name:
        raise ArtifactIntegrityError(
            f"filesystem entry must use NFC Unicode normalization: {relative_path!r}"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ArtifactIntegrityError(
            f"filesystem entry contains a control character: {relative_path!r}"
        )
    try:
        name.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ArtifactIntegrityError(
            f"filesystem entry is not valid UTF-8: {relative_path!r}"
        ) from exc


def _scan_directory(
    descriptor: int,
    *,
    prefix: tuple[str, ...] = (),
) -> tuple[_TreeEntry, ...]:
    before = os.fstat(descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise ArtifactIntegrityError("tree scan target must be a directory")
    try:
        with os.scandir(descriptor) as iterator:
            names = [entry.name for entry in iterator]
    except OSError as exc:
        raise ArtifactIntegrityError(f"cannot scan artifact directory: {exc}") from exc

    try:
        names.sort(key=_path_sort_key)
    except UnicodeEncodeError as exc:
        raise ArtifactIntegrityError("artifact tree contains a non-UTF-8 path") from exc
    if len(names) != len(set(names)):
        raise ArtifactIntegrityError("artifact tree contains duplicate directory entries")

    records: list[_TreeEntry] = []
    for name in names:
        relative_parts = (*prefix, name)
        relative_path = "/".join(relative_parts)
        _validate_filesystem_name(name, relative_path=relative_path)
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise _open_error(relative_path, exc) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactIntegrityError(f"symlink is forbidden in artifact tree: {relative_path}")
        if stat.S_ISDIR(metadata.st_mode):
            try:
                child = os.open(name, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise _open_error(relative_path, exc) from exc
            try:
                opened = os.fstat(child)
                if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                    raise ArtifactIntegrityError(f"{relative_path} changed during verification")
                records.append(_TreeEntry(relative_path=relative_path, kind="directory"))
                records.extend(_scan_directory(child, prefix=relative_parts))
            finally:
                os.close(child)
        elif stat.S_ISREG(metadata.st_mode):
            records.append(
                _hash_regular_file_at(
                    descriptor,
                    name,
                    relative_path=relative_path,
                    expected_identity=(metadata.st_dev, metadata.st_ino),
                )
            )
        else:
            raise ArtifactIntegrityError(
                f"non-regular filesystem entry is forbidden: {relative_path}"
            )

    after = os.fstat(descriptor)
    if _stable_stat_signature(before) != _stable_stat_signature(after):
        raise ArtifactIntegrityError("artifact directory changed during verification")
    return tuple(records)


def _format_path_sample(paths: set[str]) -> str:
    ordered = sorted(paths, key=_path_sort_key)
    sample = ordered[:5]
    suffix = "" if len(ordered) <= 5 else f" (+{len(ordered) - 5} more)"
    return f"{sample}{suffix}"


def _directory_digest_from_snapshot(
    snapshot: Sequence[_TreeEntry],
    *,
    included_entries: tuple[str, ...] | None,
    label: str,
) -> DirectoryDigest:
    by_path = {entry.relative_path: entry for entry in snapshot}
    if len(by_path) != len(snapshot):
        raise ArtifactIntegrityError(f"{label} contains duplicate paths")
    observed_paths = set(by_path)
    if included_entries is None:
        selected_paths = tuple(sorted(observed_paths, key=_path_sort_key))
    else:
        selected_set = set(included_entries)
        missing = selected_set - observed_paths
        if missing:
            raise ArtifactIntegrityError(
                f"{label} is missing declared entries: {_format_path_sample(missing)}"
            )
        selected_paths = included_entries
    selected = tuple(by_path[path] for path in selected_paths)
    digest_payload = {
        "entries": [entry.to_digest_dict() for entry in selected],
        "schema_version": DIRECTORY_DIGEST_SCHEMA,
    }
    file_count = sum(entry.kind == "file" for entry in selected)
    directory_count = sum(entry.kind == "directory" for entry in selected)
    byte_count = sum(entry.size_bytes for entry in selected if entry.kind == "file")
    observed_file_count = sum(entry.kind == "file" for entry in snapshot)
    observed_directory_count = sum(entry.kind == "directory" for entry in snapshot)
    observed_byte_count = sum(entry.size_bytes for entry in snapshot if entry.kind == "file")
    return DirectoryDigest(
        sha256=hashlib.sha256(_canonical_bytes(digest_payload)).hexdigest(),
        entries=selected_paths,
        file_count=file_count,
        directory_count=directory_count,
        byte_count=byte_count,
        observed_file_count=observed_file_count,
        observed_directory_count=observed_directory_count,
        observed_byte_count=observed_byte_count,
    )


def digest_directory_tree(
    directory: str | Path,
    *,
    included_entries: Sequence[str] | None = None,
) -> DirectoryDigest:
    """Hash a local directory as ordered paths, entry types, sizes, and file hashes.

    The root directory name is not part of the digest. Empty directories are
    represented, paths use NFC-normalized POSIX notation, and symlinks or other
    non-regular objects are rejected.
    """

    normalized = (
        None
        if included_entries is None
        else _normalize_expected_entries(included_entries, name="included_entries")
    )
    descriptor = _open_absolute_directory(directory, label="directory")
    try:
        snapshot = _scan_directory(descriptor)
    finally:
        os.close(descriptor)
    return _directory_digest_from_snapshot(
        snapshot,
        included_entries=normalized,
        label="directory",
    )


def _specs_overlap(first: LocalArtifactSpec, second: LocalArtifactSpec) -> bool:
    first_parts = PurePosixPath(first.relative_path).parts
    second_parts = PurePosixPath(second.relative_path).parts
    common = min(len(first_parts), len(second_parts))
    return first_parts[:common] == second_parts[:common]


def _validate_specs(specs: Iterable[LocalArtifactSpec]) -> tuple[LocalArtifactSpec, ...]:
    try:
        normalized = tuple(specs)
    except TypeError as exc:
        raise ArtifactIntegrityError("artifacts must be an iterable of LocalArtifactSpec") from exc
    if not normalized:
        raise ArtifactIntegrityError("at least one local artifact must be declared")
    if not all(isinstance(spec, LocalArtifactSpec) for spec in normalized):
        raise ArtifactIntegrityError("artifacts must contain LocalArtifactSpec records")
    identifiers = [spec.artifact_id for spec in normalized]
    paths = [spec.relative_path for spec in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise ArtifactIntegrityError("duplicate artifact ID")
    if len(paths) != len(set(paths)):
        raise ArtifactIntegrityError("duplicate artifact path")
    for position, first in enumerate(normalized):
        for second in normalized[position + 1 :]:
            if _specs_overlap(first, second):
                raise ArtifactIntegrityError(
                    "artifact declarations cannot overlap: "
                    f"{first.relative_path!r} and {second.relative_path!r}"
                )
    return tuple(
        sorted(
            normalized,
            key=lambda item: (
                _path_sort_key(item.artifact_id),
                _path_sort_key(item.relative_path),
            ),
        )
    )


def _verify_file(root_descriptor: int, spec: LocalArtifactSpec) -> VerifiedArtifact:
    parts = _relative_parts(spec.relative_path, name="relative_path")
    parent = _open_directory_at(
        root_descriptor,
        parts[:-1],
        label=f"artifact {spec.artifact_id!r}",
    )
    try:
        entry = _hash_regular_file_at(
            parent,
            parts[-1],
            relative_path=spec.relative_path,
        )
    finally:
        os.close(parent)
    if entry.sha256 != spec.expected_sha256:
        raise ArtifactIntegrityError(
            f"artifact {spec.artifact_id!r} SHA-256 mismatch: "
            f"expected {spec.expected_sha256}, observed {entry.sha256}"
        )
    return VerifiedArtifact(
        artifact_id=spec.artifact_id,
        relative_path=spec.relative_path,
        kind="file",
        exact=True,
        expected_sha256=spec.expected_sha256,
        verified_sha256=entry.sha256,
        file_count=1,
        directory_count=0,
        byte_count=entry.size_bytes,
        observed_file_count=1,
        observed_directory_count=0,
        observed_byte_count=entry.size_bytes,
    )


def _verify_directory(root_descriptor: int, spec: LocalArtifactSpec) -> VerifiedArtifact:
    parts = _relative_parts(spec.relative_path, name="relative_path")
    descriptor = _open_directory_at(
        root_descriptor,
        parts,
        label=f"artifact {spec.artifact_id!r}",
    )
    try:
        snapshot = _scan_directory(descriptor)
    finally:
        os.close(descriptor)
    observed_paths = {entry.relative_path for entry in snapshot}
    expected_entries = spec.expected_entries
    if expected_entries is not None and spec.exact:
        expected_paths = set(expected_entries)
        missing = expected_paths - observed_paths
        extra = observed_paths - expected_paths
        if missing:
            raise ArtifactIntegrityError(
                f"artifact {spec.artifact_id!r} is missing declared entries: "
                f"{_format_path_sample(missing)}"
            )
        if extra:
            raise ArtifactIntegrityError(
                f"artifact {spec.artifact_id!r} has unexpected entries: "
                f"{_format_path_sample(extra)}"
            )
    digest = _directory_digest_from_snapshot(
        snapshot,
        included_entries=expected_entries,
        label=f"artifact {spec.artifact_id!r}",
    )
    if digest.sha256 != spec.expected_sha256:
        raise ArtifactIntegrityError(
            f"artifact {spec.artifact_id!r} SHA-256 mismatch: "
            f"expected {spec.expected_sha256}, observed {digest.sha256}"
        )
    return VerifiedArtifact(
        artifact_id=spec.artifact_id,
        relative_path=spec.relative_path,
        kind="directory",
        exact=spec.exact,
        expected_sha256=spec.expected_sha256,
        verified_sha256=digest.sha256,
        file_count=digest.file_count,
        directory_count=digest.directory_count,
        byte_count=digest.byte_count,
        observed_file_count=digest.observed_file_count,
        observed_directory_count=digest.observed_directory_count,
        observed_byte_count=digest.observed_byte_count,
    )


def verify_local_artifacts(
    artifact_root: str | Path,
    *,
    manifest_sha256: str,
    artifacts: Iterable[LocalArtifactSpec],
) -> ArtifactVerificationReceipt:
    """Verify local declarations and return canonical manifest-bound evidence.

    The verifier opens every path relative to an already-local artifact root and
    never resolves a URI. Directory descriptors and ``O_NOFOLLOW`` prevent a
    declared path from leaving that root through a symlink.
    """

    _require_sha256("manifest_sha256", manifest_sha256)
    specs = _validate_specs(artifacts)
    root_descriptor = _open_absolute_directory(artifact_root, label="artifact_root")
    try:
        verified = tuple(
            _verify_file(root_descriptor, spec)
            if spec.kind == "file"
            else _verify_directory(root_descriptor, spec)
            for spec in specs
        )
    finally:
        os.close(root_descriptor)
    return ArtifactVerificationReceipt(
        manifest_sha256=manifest_sha256,
        artifacts=verified,
    )


def artifact_specs_from_local_map(
    payload: object,
    *,
    expected_sha256_by_id: Mapping[str, str],
) -> tuple[LocalArtifactSpec, ...]:
    """Bind an explicit local path map to every expected manifest artifact.

    The map does not repeat digests. Each expected SHA-256 comes from the
    already-validated study manifest, so the local map can only assign storage
    locations and file-system kinds. Study artifacts are always exact.
    """

    if not isinstance(expected_sha256_by_id, Mapping) or not expected_sha256_by_id:
        raise ArtifactIntegrityError("expected manifest artifact pins must be non-empty")
    expected: dict[str, str] = {}
    for artifact_id, sha256 in expected_sha256_by_id.items():
        _require_identifier(artifact_id)
        _require_sha256(f"expected digest for artifact {artifact_id!r}", sha256)
        expected[artifact_id] = sha256

    root = _closed_mapping(
        payload,
        fields=_LOCAL_ARTIFACT_MAP_FIELDS,
        label="local artifact map",
    )
    if root["schema_version"] != LOCAL_ARTIFACT_MAP_SCHEMA:
        raise ArtifactIntegrityError(
            f"local artifact map schema_version must equal {LOCAL_ARTIFACT_MAP_SCHEMA!r}"
        )
    values = root["artifacts"]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ArtifactIntegrityError("local artifact map artifacts must be a non-empty array")

    specs: list[LocalArtifactSpec] = []
    for position, value in enumerate(values):
        row = _closed_mapping(
            value,
            fields=_LOCAL_ARTIFACT_ENTRY_FIELDS,
            required=_LOCAL_ARTIFACT_ENTRY_REQUIRED_FIELDS,
            label=f"local artifact map artifacts[{position}]",
        )
        artifact_id = row["artifact_id"]
        _require_identifier(artifact_id)
        if artifact_id not in expected:
            raise ArtifactIntegrityError(
                f"local artifact map contains unexpected artifact ID {artifact_id!r}"
            )
        entries_value = row.get("expected_entries")
        if entries_value is None:
            expected_entries = None
        elif isinstance(entries_value, (str, bytes)) or not isinstance(entries_value, Sequence):
            raise ArtifactIntegrityError(
                f"local artifact map artifacts[{position}].expected_entries must be an array"
            )
        else:
            expected_entries = tuple(entries_value)
        specs.append(
            LocalArtifactSpec(
                artifact_id=artifact_id,
                relative_path=row["relative_path"],
                kind=row["kind"],
                expected_sha256=expected[artifact_id],
                exact=True,
                expected_entries=expected_entries,
            )
        )

    normalized = _validate_specs(specs)
    observed_ids = {spec.artifact_id for spec in normalized}
    missing = set(expected) - observed_ids
    if missing:
        raise ArtifactIntegrityError(
            "local artifact map is missing manifest artifact IDs: "
            f"{sorted(missing, key=_path_sort_key)}"
        )
    return normalized


def load_local_artifact_map(
    path: str | Path,
    *,
    expected_sha256_by_id: Mapping[str, str],
) -> tuple[LocalArtifactSpec, ...]:
    """Securely load a closed local map and bind it to manifest digests."""

    encoded = _read_secure_control_file(path, label="local artifact map")
    payload = _parse_json_object(encoded, label="local artifact map")
    return artifact_specs_from_local_map(
        payload,
        expected_sha256_by_id=expected_sha256_by_id,
    )


def load_verification_receipt(path: str | Path) -> ArtifactVerificationReceipt:
    """Load canonical verification evidence without following filesystem links."""

    encoded = _read_secure_control_file(path, label="artifact verification receipt")
    payload = _parse_json_object(encoded, label="artifact verification receipt")
    receipt = ArtifactVerificationReceipt.from_dict(payload)
    if encoded != receipt.canonical_bytes() + b"\n":
        raise ArtifactIntegrityError("artifact verification receipt bytes are not canonical")
    return receipt


def write_verification_receipt(
    receipt: ArtifactVerificationReceipt,
    target: str | Path,
) -> None:
    """Write one canonical receipt without following links or replacing a file."""

    if not isinstance(receipt, ArtifactVerificationReceipt):
        raise ArtifactIntegrityError("receipt must be an ArtifactVerificationReceipt")
    write_exclusive_receipt_bytes(receipt.canonical_bytes() + b"\n", target)


def write_exclusive_receipt_bytes(payload: bytes, target: str | Path) -> None:
    """Write pre-serialized receipt bytes through a no-follow directory handle."""

    if not isinstance(payload, bytes) or not payload:
        raise ArtifactIntegrityError("receipt payload must be non-empty immutable bytes")
    path = Path(target)
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ArtifactIntegrityError("receipt target must be an absolute file path")
    parent = _open_absolute_directory(path.parent, label="receipt parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        parent_metadata = os.fstat(parent)
        if hasattr(os, "geteuid") and parent_metadata.st_uid != os.geteuid():
            raise ArtifactIntegrityError("receipt parent must be owned by the runner identity")
        if stat.S_IMODE(parent_metadata.st_mode) & 0o022:
            raise ArtifactIntegrityError(
                "receipt parent cannot be writable by group or other identities"
            )
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent)
        except FileExistsError as exc:
            raise ArtifactIntegrityError(f"verification receipt already exists: {path}") from exc
        except OSError as exc:
            raise _open_error(f"verification receipt {path}", exc) from exc
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise ArtifactIntegrityError("verification receipt write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent)
    finally:
        os.close(parent)
