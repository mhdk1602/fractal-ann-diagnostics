"""Offline closure of the public GitHub evidence for the frozen C0 release.

The command in this module performs no network access.  It admits retained raw
outputs produced by the C0 publication workflow, verifies them against either
the frozen study manifest or its exact embedded release binding, and writes one
canonical no-replace receipt.
"""

from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .c0_evidence_release import (
    C0_EVIDENCE_RELEASE_TAG,
    C0EvidenceReleaseError,
    validate_c0_evidence_release_binding,
)
from .provider_contract import OFFICIAL_GH_VERSION
from .study import StudyManifestError, validate_study_manifest

C0_PUBLIC_VERIFICATION_SCHEMA = "fractal-c0-public-verification-v1"
C0_RELEASE_TITLE = "Confirmatory apparatus C0 evidence"

MAX_SOURCE_JSON_BYTES = 16 * 1024 * 1024
MAX_RELEASE_API_BYTES = 2 * 1024 * 1024
MAX_GH_VERIFICATION_BYTES = 2 * 1024 * 1024
MAX_TAG_INPUT_BYTES = 4096
MAX_CHECKSUM_INPUT_BYTES = 4096
MAX_GH_VERSION_BYTES = 4096
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024 * 1024
MAX_RECEIPT_BYTES = 16 * 1024 * 1024

_READ_CHUNK_BYTES = 1024 * 1024
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TAG_ROW = re.compile(rb"^([0-9a-f]{40})\t([^\r\n\t ]+)$")
_GH_VERSION = re.compile(
    rf"^gh version ({re.escape(OFFICIAL_GH_VERSION)}) \(([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})\)\n"
    rf"https://github\.com/cli/cli/releases/tag/v{re.escape(OFFICIAL_GH_VERSION)}\n$"
)


class C0PublicVerificationError(ValueError):
    """One retained C0 public-evidence input is unsafe or inconsistent."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise C0PublicVerificationError("evidence is not finite ASCII-canonical JSON") from exc


def _canonical_file_bytes(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise C0PublicVerificationError(f"{label} must be one string-keyed object")
    observed = set(value)
    if observed != fields:
        raise C0PublicVerificationError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _decode_json(encoded: bytes, *, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise C0PublicVerificationError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise C0PublicVerificationError(f"{label} contains non-finite value {value!r}")

    try:
        return json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C0PublicVerificationError(f"cannot decode {label}: {exc}") from exc


def _decode_json_file(
    encoded: bytes,
    *,
    label: str,
    canonical: bool,
) -> object:
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n") or b"\r" in encoded:
        raise C0PublicVerificationError(f"{label} must end in exactly one LF")
    body = encoded[:-1]
    if body != body.strip():
        raise C0PublicVerificationError(f"{label} has whitespace outside its JSON value")
    value = _decode_json(body, label=label)
    if canonical and encoded != _canonical_file_bytes(value):
        raise C0PublicVerificationError(f"{label} bytes are not canonical JSON plus LF")
    return value


def _require_digest(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise C0PublicVerificationError(f"{label} must be one lowercase SHA-256")
    return value


def _require_commit(value: object, *, label: str) -> str:
    if type(value) is not str or _GIT_OBJECT.fullmatch(value) is None:
        raise C0PublicVerificationError(f"{label} must be one full lowercase Git object ID")
    return value


def _require_positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise C0PublicVerificationError(f"{label} must be a positive integer")
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


def _require_secure_primitives() -> None:
    missing = [
        name
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        if not hasattr(os, name)
    ]
    if missing or os.open not in os.supports_dir_fd:
        detail = ", ".join(missing) or "openat support"
        raise C0PublicVerificationError(
            f"secure C0 verification is unavailable on this platform: {detail}"
        )
    if os.stat not in os.supports_dir_fd or os.stat not in os.supports_follow_symlinks:
        raise C0PublicVerificationError(
            "secure C0 verification needs non-following dir_fd stat support"
        )


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _file_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def _open_error(label: str, exc: OSError) -> C0PublicVerificationError:
    if exc.errno == errno.ENOENT:
        return C0PublicVerificationError(f"{label} is missing")
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        return C0PublicVerificationError(
            f"{label} crosses a symlink or contains a non-directory ancestor"
        )
    return C0PublicVerificationError(f"cannot open {label}: {exc.strerror or exc}")


def _open_absolute_directory(path: Path, *, label: str) -> int:
    _require_secure_primitives()
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise C0PublicVerificationError(f"{label} must be one canonical absolute directory")
    try:
        descriptor = os.open("/", _directory_flags())
    except OSError as exc:
        raise _open_error(label, exc) from exc
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except OSError as exc:
                raise _open_error(label, exc) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@dataclass
class _OpenSnapshot:
    path: Path
    label: str
    parent_descriptor: int
    descriptor: int
    max_bytes: int
    retain_bytes: bool
    signature: tuple[int, ...]
    sha256: str
    size: int
    encoded: bytes | None

    @classmethod
    def open(
        cls,
        value: str | Path,
        *,
        label: str,
        max_bytes: int,
        retain_bytes: bool = True,
    ) -> _OpenSnapshot:
        path = Path(value)
        if not path.is_absolute() or path.name in {"", ".", ".."}:
            raise C0PublicVerificationError(f"{label} must be one absolute file path")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise C0PublicVerificationError(f"{label} maximum size is invalid")
        parent = _open_absolute_directory(path.parent, label=f"{label} parent")
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(path.name, _file_flags(), dir_fd=parent)
            except OSError as exc:
                raise _open_error(label, exc) from exc
            metadata = os.fstat(descriptor)
            mode = stat.S_IMODE(metadata.st_mode)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise C0PublicVerificationError(f"{label} must be one singly linked regular file")
            if metadata.st_size <= 0 or metadata.st_size > max_bytes:
                raise C0PublicVerificationError(f"{label} must contain 1..{max_bytes} bytes")
            if mode & 0o022:
                raise C0PublicVerificationError(
                    f"{label} cannot be writable by group or other identities"
                )
            if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
                raise C0PublicVerificationError(f"{label} must be owned by the operator")
            snapshot = cls(
                path=path,
                label=label,
                parent_descriptor=parent,
                descriptor=descriptor,
                max_bytes=max_bytes,
                retain_bytes=retain_bytes,
                signature=_stat_signature(metadata),
                sha256="",
                size=0,
                encoded=None,
            )
            snapshot.sha256, snapshot.size, snapshot.encoded = snapshot._scan()
            return snapshot
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)
            raise

    @property
    def identity(self) -> tuple[int, int]:
        return (self.signature[0], self.signature[1])

    def _scan(self) -> tuple[str, int, bytes | None]:
        try:
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            before = os.fstat(self.descriptor)
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            observed = 0
            while observed <= self.max_bytes:
                chunk = os.read(
                    self.descriptor,
                    min(_READ_CHUNK_BYTES, self.max_bytes + 1 - observed),
                )
                if not chunk:
                    break
                digest.update(chunk)
                observed += len(chunk)
                if self.retain_bytes:
                    chunks.append(chunk)
            after = os.fstat(self.descriptor)
            named = os.stat(
                self.path.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise C0PublicVerificationError(f"cannot read {self.label}: {exc}") from exc
        signature = _stat_signature(before)
        if (
            observed > self.max_bytes
            or signature != _stat_signature(after)
            or signature != _stat_signature(named)
            or signature != self.signature
            or observed != before.st_size
        ):
            raise C0PublicVerificationError(f"{self.label} changed while admitted")
        encoded = b"".join(chunks) if self.retain_bytes else None
        return digest.hexdigest(), observed, encoded

    def revalidate(self) -> None:
        digest, size, encoded = self._scan()
        if digest != self.sha256 or size != self.size or encoded != self.encoded:
            raise C0PublicVerificationError(f"{self.label} changed before receipt publication")

    def close(self) -> None:
        os.close(self.descriptor)
        os.close(self.parent_descriptor)


@dataclass(frozen=True)
class GitTagRow:
    """One closed row from the canonical ``git ls-remote`` readback."""

    object_id: str
    ref: str

    def __post_init__(self) -> None:
        _require_commit(self.object_id, label="tag row object_id")
        base = f"refs/tags/{C0_EVIDENCE_RELEASE_TAG}"
        if self.ref not in {base, f"{base}^{{}}"}:
            raise C0PublicVerificationError("tag row ref differs from the C0 release tag")

    def to_dict(self) -> dict[str, str]:
        return {"object_id": self.object_id, "ref": self.ref}

    @classmethod
    def from_dict(cls, value: object) -> GitTagRow:
        row = _closed(value, frozenset({"object_id", "ref"}), label="tag row")
        return cls(object_id=row["object_id"], ref=row["ref"])


def _parse_tag_rows(encoded: bytes, *, target_commit: str) -> tuple[tuple[GitTagRow, ...], str]:
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n") or b"\r" in encoded:
        raise C0PublicVerificationError("tag ls-remote bytes must end in exactly one LF")
    raw_rows = encoded[:-1].split(b"\n")
    if raw_rows != sorted(raw_rows):
        raise C0PublicVerificationError("tag ls-remote rows are not bytewise sorted")
    if len(raw_rows) not in {1, 2}:
        raise C0PublicVerificationError("tag ls-remote must contain one or two rows")
    rows: list[GitTagRow] = []
    for raw in raw_rows:
        match = _TAG_ROW.fullmatch(raw)
        if match is None:
            raise C0PublicVerificationError("tag ls-remote contains a malformed row")
        rows.append(
            GitTagRow(
                object_id=match.group(1).decode("ascii"),
                ref=match.group(2).decode("ascii"),
            )
        )
    base_ref = f"refs/tags/{C0_EVIDENCE_RELEASE_TAG}"
    peeled_ref = f"{base_ref}^{{}}"
    base = [row for row in rows if row.ref == base_ref]
    peeled = [row for row in rows if row.ref == peeled_ref]
    if len(base) != 1 or len(peeled) > 1:
        raise C0PublicVerificationError("tag ls-remote row cardinality differs from Git")
    if peeled:
        if len(rows) != 2 or peeled[0].object_id != target_commit:
            raise C0PublicVerificationError("annotated tag does not peel to the frozen C0 commit")
        if base[0].object_id == target_commit:
            raise C0PublicVerificationError("annotated tag object cannot equal its peeled commit")
        kind = "annotated"
    else:
        if len(rows) != 1 or base[0].object_id != target_commit:
            raise C0PublicVerificationError("lightweight tag does not equal the frozen C0 commit")
        kind = "lightweight"
    return tuple(rows), kind


def _published_at(value: object) -> str:
    if type(value) is not str or not value.endswith("Z") or value != value.strip():
        raise C0PublicVerificationError("release published_at must be canonical UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise C0PublicVerificationError("release published_at is not valid RFC3339") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise C0PublicVerificationError("release published_at must be UTC")
    return value


def _validate_gh_version(encoded: bytes) -> tuple[str, str]:
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise C0PublicVerificationError("gh version output is not UTF-8") from exc
    match = _GH_VERSION.fullmatch(text)
    if match is None:
        raise C0PublicVerificationError(
            f"gh version output must identify the pinned gh {OFFICIAL_GH_VERSION} release"
        )
    try:
        date.fromisoformat(match.group(2))
    except ValueError as exc:
        raise C0PublicVerificationError("gh version build date is invalid") from exc
    return text, match.group(1)


def _utf8_bytes(value: object, *, label: str) -> bytes:
    if type(value) is not str:
        raise C0PublicVerificationError(f"{label} must be a string")
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise C0PublicVerificationError(f"{label} is not valid UTF-8 text") from exc


def _validate_release_api(value: object, *, binding: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise C0PublicVerificationError("release API output must contain one object")
    release_id = _require_positive_integer(value.get("id"), label="release API id")
    exact = {
        "tag_name": binding["release_tag"],
        "target_commitish": binding["target_commit"],
        "html_url": binding["release_url"],
        "name": C0_RELEASE_TITLE,
        "draft": False,
        "prerelease": False,
        "immutable": True,
    }
    for field, expected in exact.items():
        if value.get(field) != expected or type(value.get(field)) is not type(expected):
            raise C0PublicVerificationError(f"release API {field} differs from C0")
    _published_at(value.get("published_at"))
    api_url = f"https://api.github.com/repos/{binding['repository']}/releases/{release_id}"
    if value.get("url") != api_url or value.get("assets_url") != f"{api_url}/assets":
        raise C0PublicVerificationError("release API identity URLs differ from the release ID")
    assets = value.get("assets")
    if isinstance(assets, (str, bytes)) or not isinstance(assets, Sequence) or len(assets) != 2:
        raise C0PublicVerificationError("release API must contain exactly two assets")
    expected_assets = {
        binding["asset_name"]: (
            binding["asset_sha256"],
            binding["asset_size"],
            binding["asset_url"],
        ),
        binding["checksum_asset_name"]: (
            binding["checksum_asset_sha256"],
            binding["checksum_asset_size"],
            binding["checksum_asset_url"],
        ),
    }
    observed_names: set[str] = set()
    observed_ids: set[int] = set()
    for position, asset in enumerate(assets):
        if not isinstance(asset, Mapping) or any(type(key) is not str for key in asset):
            raise C0PublicVerificationError(f"release API asset {position} must be an object")
        name = asset.get("name")
        if type(name) is not str or name not in expected_assets or name in observed_names:
            raise C0PublicVerificationError("release API has an unknown or duplicate asset")
        observed_names.add(name)
        asset_id = _require_positive_integer(asset.get("id"), label=f"release asset {name} id")
        if asset_id in observed_ids:
            raise C0PublicVerificationError("release API repeats an asset ID")
        observed_ids.add(asset_id)
        digest, size, browser_url = expected_assets[name]
        if (
            asset.get("size") != size
            or type(asset.get("size")) is not int
            or asset.get("digest") != f"sha256:{digest}"
            or asset.get("browser_download_url") != browser_url
            or asset.get("state") != "uploaded"
            or asset.get("url")
            != f"https://api.github.com/repos/{binding['repository']}/releases/assets/{asset_id}"
        ):
            raise C0PublicVerificationError(f"release API asset {name} differs from C0")
    if observed_names != set(expected_assets):
        raise C0PublicVerificationError("release API asset set differs from C0")
    return value


def _validated_subjects(value: object, *, label: str) -> dict[str, str]:
    root = _closed(
        value,
        frozenset({"attestation", "verificationResult"}),
        label=label,
    )
    if not isinstance(root["attestation"], Mapping) or not root["attestation"]:
        raise C0PublicVerificationError(f"{label}.attestation must be a non-empty object")
    result = root["verificationResult"]
    if not isinstance(result, Mapping) or not result:
        raise C0PublicVerificationError(f"{label}.verificationResult must be a non-empty object")
    media_type = result.get("mediaType")
    if type(media_type) is not str or not media_type.startswith(
        "application/vnd.dev.sigstore.verificationresult+json;version="
    ):
        raise C0PublicVerificationError(f"{label} lacks a Sigstore verification result media type")
    statement = result.get("statement")
    if not isinstance(statement, Mapping):
        raise C0PublicVerificationError(f"{label} lacks the verified in-toto statement")
    subjects = statement.get("subject")
    if isinstance(subjects, (str, bytes)) or not isinstance(subjects, Sequence) or not subjects:
        raise C0PublicVerificationError(f"{label} verified statement has no subjects")
    named: dict[str, str] = {}
    for position, subject in enumerate(subjects):
        if not isinstance(subject, Mapping):
            raise C0PublicVerificationError(f"{label} subject {position} must be an object")
        name = subject.get("name")
        digests = subject.get("digest")
        if type(name) is not str or not isinstance(digests, Mapping):
            raise C0PublicVerificationError(f"{label} subject {position} is malformed")
        digest = digests.get("sha256")
        if name and digest is not None:
            digest = _require_digest(digest, label=f"{label} subject {name!r} SHA-256")
            if name in named:
                raise C0PublicVerificationError(f"{label} repeats subject {name!r}")
            named[name] = digest
    return named


def _validate_gh_outputs(
    release_value: object,
    asset_value: object,
    *,
    binding: Mapping[str, Any],
) -> None:
    release_subjects = _validated_subjects(
        release_value,
        label="gh release verification output",
    )
    expected = {
        binding["asset_name"]: binding["asset_sha256"],
        binding["checksum_asset_name"]: binding["checksum_asset_sha256"],
    }
    if release_subjects != expected:
        raise C0PublicVerificationError(
            "gh release verification subjects differ from the exact C0 asset set"
        )
    asset_subjects = _validated_subjects(
        asset_value,
        label="gh asset verification output",
    )
    if asset_subjects.get(binding["asset_name"]) != binding["asset_sha256"]:
        raise C0PublicVerificationError(
            "gh asset verification does not bind the downloaded archive"
        )


@dataclass(frozen=True)
class C0PublicVerificationReceipt:
    """Canonical, closed record for one offline verification of public C0 evidence."""

    binding_source_kind: str
    binding_source_file_sha256: str
    binding_sha256: str
    c0_evidence_release_binding: Mapping[str, Any]
    repository: str
    release_tag: str
    target_commit: str
    release_id: int
    gh_version: str
    gh_version_file_sha256: str
    gh_version_text: str
    release_api: Mapping[str, Any]
    release_api_file_sha256: str
    tag_rows: tuple[GitTagRow, ...]
    tag_ls_remote_file_sha256: str
    tag_kind: str
    release_verification: Mapping[str, Any]
    release_verification_file_sha256: str
    release_verification_text: str
    asset_verification: Mapping[str, Any]
    asset_verification_file_sha256: str
    asset_verification_text: str
    archive_name: str
    archive_sha256: str
    archive_size: int
    checksum_name: str
    checksum_sha256: str
    checksum_size: int
    checksum_text: str
    schema_version: str = C0_PUBLIC_VERIFICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.binding_source_kind not in {"c0-binding", "frozen-manifest"}:
            raise C0PublicVerificationError("binding_source_kind is not registered")
        for name in (
            "binding_source_file_sha256",
            "binding_sha256",
            "gh_version_file_sha256",
            "release_api_file_sha256",
            "tag_ls_remote_file_sha256",
            "release_verification_file_sha256",
            "asset_verification_file_sha256",
            "archive_sha256",
            "checksum_sha256",
        ):
            _require_digest(getattr(self, name), label=name)
        commit = _require_commit(self.target_commit, label="target_commit")
        try:
            validate_c0_evidence_release_binding(
                self.c0_evidence_release_binding,
                frozen=True,
                code_commit=commit,
            )
        except C0EvidenceReleaseError as exc:
            raise C0PublicVerificationError(f"invalid embedded C0 binding: {exc}") from exc
        binding = self.c0_evidence_release_binding
        exact = {
            "repository": binding["repository"],
            "release_tag": binding["release_tag"],
            "archive_name": binding["asset_name"],
            "archive_sha256": binding["asset_sha256"],
            "archive_size": binding["asset_size"],
            "checksum_name": binding["checksum_asset_name"],
            "checksum_sha256": binding["checksum_asset_sha256"],
            "checksum_size": binding["checksum_asset_size"],
        }
        for field, expected in exact.items():
            if getattr(self, field) != expected:
                raise C0PublicVerificationError(f"receipt {field} differs from its C0 binding")
        if self.binding_sha256 != _sha256(_canonical_file_bytes(binding)):
            raise C0PublicVerificationError("binding_sha256 differs from the embedded binding")
        if (
            self.binding_source_kind == "c0-binding"
            and self.binding_source_file_sha256 != self.binding_sha256
        ):
            raise C0PublicVerificationError(
                "direct binding source hash differs from the embedded binding"
            )
        if self.tag_kind not in {"annotated", "lightweight"}:
            raise C0PublicVerificationError("tag_kind is not registered")
        gh_version_bytes = _utf8_bytes(self.gh_version_text, label="gh version text")
        gh_version_text, gh_version = _validate_gh_version(gh_version_bytes)
        if gh_version_text != self.gh_version_text or gh_version != self.gh_version:
            raise C0PublicVerificationError("receipt gh version fields are inconsistent")
        if _sha256(gh_version_bytes) != self.gh_version_file_sha256:
            raise C0PublicVerificationError(
                "gh_version_file_sha256 differs from the embedded raw transcript"
            )
        _require_positive_integer(self.release_id, label="release_id")
        _require_positive_integer(self.archive_size, label="archive_size")
        _require_positive_integer(self.checksum_size, label="checksum_size")
        release_api_bytes = _canonical_file_bytes(self.release_api)
        if _sha256(release_api_bytes) != self.release_api_file_sha256:
            raise C0PublicVerificationError(
                "release_api_file_sha256 differs from the embedded fresh API object"
            )
        release_api = _validate_release_api(self.release_api, binding=binding)
        if release_api["id"] != self.release_id:
            raise C0PublicVerificationError("release_id differs from the embedded API object")
        tag_bytes = b"".join(
            f"{row.object_id}\t{row.ref}\n".encode("ascii") for row in self.tag_rows
        )
        if _sha256(tag_bytes) != self.tag_ls_remote_file_sha256:
            raise C0PublicVerificationError(
                "tag_ls_remote_file_sha256 differs from the embedded tag rows"
            )
        rows, kind = _parse_tag_rows(
            tag_bytes,
            target_commit=commit,
        )
        if rows != self.tag_rows or kind != self.tag_kind:
            raise C0PublicVerificationError("receipt tag rows are internally inconsistent")
        fresh_gh: list[tuple[str, str, str, Mapping[str, Any]]] = [
            (
                "release",
                self.release_verification_text,
                self.release_verification_file_sha256,
                self.release_verification,
            ),
            (
                "asset",
                self.asset_verification_text,
                self.asset_verification_file_sha256,
                self.asset_verification,
            ),
        ]
        for label, text, expected_sha256, expected_object in fresh_gh:
            encoded = _utf8_bytes(text, label=f"{label} verification text")
            if _sha256(encoded) != expected_sha256:
                raise C0PublicVerificationError(
                    f"{label} verification raw hash differs from its embedded text"
                )
            observed = _decode_json_file(
                encoded,
                label=f"gh {label} verification output",
                canonical=False,
            )
            if _canonical_json(observed) != _canonical_json(expected_object):
                raise C0PublicVerificationError(
                    f"{label} verification object differs from its embedded raw text"
                )
        _validate_gh_outputs(self.release_verification, self.asset_verification, binding=binding)
        expected_checksum = f"{self.archive_sha256}  {self.archive_name}\n"
        if self.checksum_text != expected_checksum:
            raise C0PublicVerificationError("receipt checksum text differs from the archive")
        if self.schema_version != C0_PUBLIC_VERIFICATION_SCHEMA:
            raise C0PublicVerificationError("C0 public-verification schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_name": self.archive_name,
            "archive_sha256": self.archive_sha256,
            "archive_size": self.archive_size,
            "asset_verification": copy.deepcopy(dict(self.asset_verification)),
            "asset_verification_file_sha256": self.asset_verification_file_sha256,
            "asset_verification_text": self.asset_verification_text,
            "binding_sha256": self.binding_sha256,
            "binding_source_file_sha256": self.binding_source_file_sha256,
            "binding_source_kind": self.binding_source_kind,
            "c0_evidence_release_binding": copy.deepcopy(dict(self.c0_evidence_release_binding)),
            "checksum_name": self.checksum_name,
            "checksum_sha256": self.checksum_sha256,
            "checksum_size": self.checksum_size,
            "checksum_text": self.checksum_text,
            "gh_version": self.gh_version,
            "gh_version_file_sha256": self.gh_version_file_sha256,
            "gh_version_text": self.gh_version_text,
            "release_api": copy.deepcopy(dict(self.release_api)),
            "release_api_file_sha256": self.release_api_file_sha256,
            "release_id": self.release_id,
            "release_tag": self.release_tag,
            "release_verification": copy.deepcopy(dict(self.release_verification)),
            "release_verification_file_sha256": self.release_verification_file_sha256,
            "release_verification_text": self.release_verification_text,
            "repository": self.repository,
            "schema_version": self.schema_version,
            "tag_kind": self.tag_kind,
            "tag_ls_remote_file_sha256": self.tag_ls_remote_file_sha256,
            "tag_rows": [row.to_dict() for row in self.tag_rows],
            "target_commit": self.target_commit,
        }

    def canonical_file_bytes(self) -> bytes:
        encoded = _canonical_file_bytes(self.to_dict())
        if len(encoded) > MAX_RECEIPT_BYTES:
            raise C0PublicVerificationError("C0 public-verification receipt exceeds its limit")
        return encoded

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> C0PublicVerificationReceipt:
        fields = frozenset(cls.__dataclass_fields__)
        row = _closed(value, fields, label="C0 public-verification receipt")
        tag_values = row["tag_rows"]
        if (
            isinstance(tag_values, (str, bytes))
            or not isinstance(tag_values, Sequence)
            or not tag_values
        ):
            raise C0PublicVerificationError("receipt tag_rows must be a non-empty array")
        values = dict(row)
        values["tag_rows"] = tuple(GitTagRow.from_dict(item) for item in tag_values)
        return cls(**values)


def _binding_from_source(
    source: _OpenSnapshot,
    *,
    source_kind: str,
    c0_commit: str | None,
) -> tuple[Mapping[str, Any], str]:
    assert source.encoded is not None
    value = _decode_json_file(source.encoded, label=source.label, canonical=True)
    if not isinstance(value, Mapping):
        raise C0PublicVerificationError(f"{source.label} must contain one JSON object")
    if source_kind == "frozen-manifest":
        if c0_commit is not None:
            raise C0PublicVerificationError("--c0-commit cannot accompany --manifest")
        try:
            validate_study_manifest(value, require_frozen=True)
        except StudyManifestError as exc:
            raise C0PublicVerificationError(f"invalid frozen study manifest: {exc}") from exc
        sealed = value.get("sealed_execution")
        if not isinstance(sealed, Mapping):
            raise C0PublicVerificationError("frozen manifest lacks sealed_execution")
        commit = _require_commit(sealed.get("code_commit"), label="sealed_execution.code_commit")
        binding = sealed.get("c0_evidence_release")
    else:
        commit = _require_commit(c0_commit, label="--c0-commit")
        binding = value
    try:
        validate_c0_evidence_release_binding(binding, frozen=True, code_commit=commit)
    except C0EvidenceReleaseError as exc:
        raise C0PublicVerificationError(f"invalid C0 release binding: {exc}") from exc
    assert isinstance(binding, Mapping)
    return copy.deepcopy(dict(binding)), commit


def _assert_unique_inputs(snapshots: Sequence[_OpenSnapshot]) -> None:
    identities: set[tuple[int, int]] = set()
    paths: set[Path] = set()
    for snapshot in snapshots:
        if snapshot.identity in identities or snapshot.path in paths:
            raise C0PublicVerificationError("C0 verification inputs must be distinct files")
        identities.add(snapshot.identity)
        paths.add(snapshot.path)


def build_c0_public_verification_receipt(
    *,
    manifest_path: str | Path | None = None,
    binding_path: str | Path | None = None,
    c0_commit: str | None = None,
    gh_version_path: str | Path,
    release_api_path: str | Path,
    tag_ls_remote_path: str | Path,
    release_verification_path: str | Path,
    asset_verification_path: str | Path,
    archive_path: str | Path,
    checksum_path: str | Path,
) -> C0PublicVerificationReceipt:
    """Verify retained C0 public evidence without resolving any URI."""

    if (manifest_path is None) == (binding_path is None):
        raise C0PublicVerificationError("provide exactly one of manifest_path or binding_path")
    source_kind = "frozen-manifest" if manifest_path is not None else "c0-binding"
    source_path = manifest_path if manifest_path is not None else binding_path
    assert source_path is not None
    snapshots: list[_OpenSnapshot] = []
    try:
        source = _OpenSnapshot.open(
            source_path,
            label="frozen manifest" if manifest_path is not None else "C0 release binding",
            max_bytes=MAX_SOURCE_JSON_BYTES,
        )
        snapshots.append(source)
        binding, commit = _binding_from_source(
            source,
            source_kind=source_kind,
            c0_commit=c0_commit,
        )
        archive_size = _require_positive_integer(binding["asset_size"], label="asset_size")
        if archive_size > MAX_ARCHIVE_BYTES:
            raise C0PublicVerificationError("C0 archive exceeds the offline verifier limit")
        specs = (
            (gh_version_path, "gh version output", MAX_GH_VERSION_BYTES, True),
            (release_api_path, "release API output", MAX_RELEASE_API_BYTES, True),
            (tag_ls_remote_path, "tag ls-remote output", MAX_TAG_INPUT_BYTES, True),
            (
                release_verification_path,
                "gh release verification output",
                MAX_GH_VERIFICATION_BYTES,
                True,
            ),
            (
                asset_verification_path,
                "gh asset verification output",
                MAX_GH_VERIFICATION_BYTES,
                True,
            ),
            (archive_path, "anonymous archive", archive_size, False),
            (checksum_path, "anonymous checksum", MAX_CHECKSUM_INPUT_BYTES, True),
        )
        for path, label, maximum, retain in specs:
            snapshots.append(
                _OpenSnapshot.open(
                    path,
                    label=label,
                    max_bytes=maximum,
                    retain_bytes=retain,
                )
            )
        _assert_unique_inputs(snapshots)
        gh_version, release_api, tag_readback, release_gh, asset_gh, archive, checksum = snapshots[
            1:
        ]
        if archive.path.name != binding["asset_name"]:
            raise C0PublicVerificationError("anonymous archive basename differs from the binding")
        if checksum.path.name != binding["checksum_asset_name"]:
            raise C0PublicVerificationError("anonymous checksum basename differs from the binding")
        if archive.sha256 != binding["asset_sha256"] or archive.size != binding["asset_size"]:
            raise C0PublicVerificationError("anonymous archive bytes differ from C0")
        if (
            checksum.sha256 != binding["checksum_asset_sha256"]
            or checksum.size != binding["checksum_asset_size"]
        ):
            raise C0PublicVerificationError("anonymous checksum bytes differ from C0")
        assert gh_version.encoded is not None
        gh_version_text, gh_version_number = _validate_gh_version(gh_version.encoded)
        assert release_api.encoded is not None
        release_api_value = _decode_json_file(
            release_api.encoded,
            label="release API output",
            canonical=True,
        )
        release_api_object = _validate_release_api(release_api_value, binding=binding)
        assert tag_readback.encoded is not None
        tag_rows, tag_kind = _parse_tag_rows(tag_readback.encoded, target_commit=commit)
        assert release_gh.encoded is not None and asset_gh.encoded is not None
        release_gh_value = _decode_json_file(
            release_gh.encoded,
            label="gh release verification output",
            canonical=False,
        )
        asset_gh_value = _decode_json_file(
            asset_gh.encoded,
            label="gh asset verification output",
            canonical=False,
        )
        _validate_gh_outputs(release_gh_value, asset_gh_value, binding=binding)
        assert isinstance(release_gh_value, Mapping) and isinstance(asset_gh_value, Mapping)
        try:
            release_gh_text = release_gh.encoded.decode("utf-8", errors="strict")
            asset_gh_text = asset_gh.encoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise C0PublicVerificationError("gh verification output is not UTF-8") from exc
        assert checksum.encoded is not None
        try:
            checksum_text = checksum.encoded.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise C0PublicVerificationError("anonymous checksum is not ASCII") from exc
        expected_checksum = f"{binding['asset_sha256']}  {binding['asset_name']}\n"
        if checksum_text != expected_checksum:
            raise C0PublicVerificationError("anonymous checksum text differs from C0")
        receipt = C0PublicVerificationReceipt(
            binding_source_kind=source_kind,
            binding_source_file_sha256=source.sha256,
            binding_sha256=_sha256(_canonical_file_bytes(binding)),
            c0_evidence_release_binding=binding,
            repository=binding["repository"],
            release_tag=binding["release_tag"],
            target_commit=commit,
            release_id=release_api_object["id"],
            gh_version=gh_version_number,
            gh_version_file_sha256=gh_version.sha256,
            gh_version_text=gh_version_text,
            release_api=copy.deepcopy(dict(release_api_object)),
            release_api_file_sha256=release_api.sha256,
            tag_rows=tag_rows,
            tag_ls_remote_file_sha256=tag_readback.sha256,
            tag_kind=tag_kind,
            release_verification=copy.deepcopy(dict(release_gh_value)),
            release_verification_file_sha256=release_gh.sha256,
            release_verification_text=release_gh_text,
            asset_verification=copy.deepcopy(dict(asset_gh_value)),
            asset_verification_file_sha256=asset_gh.sha256,
            asset_verification_text=asset_gh_text,
            archive_name=binding["asset_name"],
            archive_sha256=archive.sha256,
            archive_size=archive.size,
            checksum_name=binding["checksum_asset_name"],
            checksum_sha256=checksum.sha256,
            checksum_size=checksum.size,
            checksum_text=checksum_text,
        )
        for snapshot in snapshots:
            snapshot.revalidate()
        return receipt
    finally:
        for snapshot in reversed(snapshots):
            snapshot.close()


def write_c0_public_verification_receipt(
    receipt: C0PublicVerificationReceipt,
    output_path: str | Path,
) -> Path:
    """Write one canonical private receipt through an exclusive descriptor."""

    if not isinstance(receipt, C0PublicVerificationReceipt):
        raise C0PublicVerificationError("receipt must be a C0PublicVerificationReceipt")
    path = Path(output_path)
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise C0PublicVerificationError("output path must be one absolute file path")
    parent = _open_absolute_directory(path.parent, label="receipt parent")
    descriptor: int | None = None
    encoded = receipt.canonical_file_bytes()
    try:
        metadata = os.fstat(parent)
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise C0PublicVerificationError("receipt parent mode must equal 0700")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise C0PublicVerificationError("receipt parent must be owned by the operator")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent)
        except FileExistsError as exc:
            raise C0PublicVerificationError(
                "C0 public-verification receipt already exists"
            ) from exc
        except OSError as exc:
            raise _open_error("C0 public-verification receipt", exc) from exc
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise C0PublicVerificationError("receipt write made no progress")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_size != len(encoded):
            raise C0PublicVerificationError("receipt mode or size changed during publication")
        os.lseek(descriptor, 0, os.SEEK_SET)
        readback = bytearray()
        while len(readback) < len(encoded):
            chunk = os.read(descriptor, len(encoded) - len(readback))
            if not chunk:
                break
            readback.extend(chunk)
        if bytes(readback) != encoded:
            raise C0PublicVerificationError("receipt readback differs from staged bytes")
        os.fsync(parent)
        return path
    except OSError as exc:
        raise C0PublicVerificationError(f"cannot publish C0 receipt: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def verify_and_write_c0_public_receipt(
    *,
    output_path: str | Path,
    **inputs: Any,
) -> C0PublicVerificationReceipt:
    """Build and publish one no-replace C0 public-verification receipt."""

    receipt = build_c0_public_verification_receipt(**inputs)
    write_c0_public_verification_receipt(receipt, output_path)
    return receipt


def load_c0_public_verification_receipt(
    path: str | Path,
) -> C0PublicVerificationReceipt:
    """Load one canonical, mode-0600 receipt through the secure input path."""

    snapshot = _OpenSnapshot.open(
        path,
        label="C0 public-verification receipt",
        max_bytes=MAX_RECEIPT_BYTES,
    )
    try:
        metadata = os.fstat(snapshot.descriptor)
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise C0PublicVerificationError("C0 public-verification receipt mode must equal 0600")
        assert snapshot.encoded is not None
        value = _decode_json_file(
            snapshot.encoded,
            label="C0 public-verification receipt",
            canonical=True,
        )
        receipt = C0PublicVerificationReceipt.from_dict(value)
        if snapshot.encoded != receipt.canonical_file_bytes():
            raise C0PublicVerificationError("receipt changed after typed reconstruction")
        snapshot.revalidate()
        return receipt
    finally:
        snapshot.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify retained public C0 release evidence without network access.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="canonical frozen study manifest")
    source.add_argument("--binding", type=Path, help="canonical C0 release binding")
    parser.add_argument(
        "--c0-commit",
        help="full C0 commit; required with --binding and forbidden with --manifest",
    )
    parser.add_argument("--gh-version", type=Path, required=True)
    parser.add_argument("--release-api", type=Path, required=True)
    parser.add_argument("--tag-ls-remote", type=Path, required=True)
    parser.add_argument("--release-verification", type=Path, required=True)
    parser.add_argument("--asset-verification", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.binding is not None and args.c0_commit is None:
        parser.error("--c0-commit is required with --binding")
    if args.manifest is not None and args.c0_commit is not None:
        parser.error("--c0-commit is forbidden with --manifest")
    try:
        receipt = verify_and_write_c0_public_receipt(
            manifest_path=args.manifest,
            binding_path=args.binding,
            c0_commit=args.c0_commit,
            gh_version_path=args.gh_version,
            release_api_path=args.release_api,
            tag_ls_remote_path=args.tag_ls_remote,
            release_verification_path=args.release_verification,
            asset_verification_path=args.asset_verification,
            archive_path=args.archive,
            checksum_path=args.checksum,
            output_path=args.output,
        )
    except C0PublicVerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "receipt_sha256": receipt.file_sha256,
                "schema_version": receipt.schema_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
