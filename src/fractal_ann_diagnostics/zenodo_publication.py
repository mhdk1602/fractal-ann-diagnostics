"""Fail-closed publication of the fixed C1 registration package to Zenodo.

The CLI has no token-valued argument.  Authenticated operations read one bearer
token from stdin or an already-open caller-supplied file descriptor.  Staging
and publication are separate commands; publication never repairs a draft.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import ssl
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote, unquote, urlsplit

from .c0_public_verification import (
    C0_PUBLIC_VERIFICATION_SCHEMA,
    C0PublicVerificationError,
    load_c0_public_verification_receipt,
)
from .c1_manifest_transition import (
    C1_MANIFEST_TRANSITION_RECEIPT_SCHEMA,
    C1ManifestTransitionError,
    loads_c1_manifest_transition_receipt,
    verify_c1_manifest_transition_receipt_bindings,
)
from .github_state_attestation import (
    C1_LOCK_PATH,
    C1_MANIFEST_PATH,
    C1_REF,
    C1_TRANSITION_RECEIPT_PATH,
    COMMON_CONTROL_LIMITATION,
    GIT_IDENTITY_EMAIL,
    GIT_IDENTITY_NAME,
    REGISTRATION_PREDICATE_TYPE,
    REGISTRATION_RECEIPT_SCHEMA,
    REGISTRATION_WORKFLOW_PATH,
    REGISTRY_ATTESTATION_RECEIPT_SCHEMA,
    REGISTRY_MATERIALIZATION_SCHEMA,
    REGISTRY_RECORD_PREDICATE_TYPE,
    REGISTRY_RECORD_SUBJECT_PATH,
    REPOSITORY,
    ZENODO_RECORD_ID,
    ZENODO_REGISTRY_IDENTITY,
    ZENODO_REGISTRY_URI,
    ZENODO_RESERVED_DOI,
    C1AttestationVerifier,
    GhC1AttestationVerifier,
    _canonical_bytes,
    _load_closed_c1_predicate,
    _load_registry_record_predicate,
    _load_zenodo_reservation,
    _strict_json,
    _utc_datetime,
    _validated_gh_output,
    _verify_closed_c1_statement,
    parse_sigstore_bundle,
)
from .study import (
    ProtocolRegistrationReceipt,
    ProtocolRegistryRecord,
    VerifiedC1ProtocolRegistration,
    _mint_verified_c1_protocol_registration,
    load_protocol_registration_receipt,
    load_protocol_registry_record,
    manifest_sha256,
    validate_study_manifest,
)
from .suite_attempt import SuiteAttemptError

ZENODO_API_ORIGIN = "https://zenodo.org"
ZENODO_DRAFT_API_URI = f"{ZENODO_API_ORIGIN}/api/deposit/depositions/{ZENODO_RECORD_ID}"
ZENODO_PUBLISH_API_URI = f"{ZENODO_DRAFT_API_URI}/actions/publish"
ZENODO_PUBLIC_API_URI = f"{ZENODO_API_ORIGIN}/api/records/{ZENODO_RECORD_ID}"
ZENODO_TITLE = (
    "Prospective confirmatory protocol: Adaptive authorization-first vector retrieval "
    "under drift (v0.3.0)"
)
ZENODO_CREATOR = "mhdk1602"
ZENODO_CREATOR_ORCID = "0009-0003-1036-9477"
ZENODO_UPLOAD_TYPE = "publication"
ZENODO_PUBLICATION_TYPE = "other"
ZENODO_LICENSE_ID = "cc-by-4.0"
ZENODO_ACCESS_RIGHT = "open"
ZENODO_PUBLICATION_DATE = "2026-07-14"
ZENODO_DESCRIPTION = (
    "Prospective protocol registration for a five-corpus confirmatory study of "
    "authorization-first, geometry-aware vector retrieval. The deposit binds the frozen "
    "C1 study manifest, executable protocol, artifact lock, sealed-run contract, external "
    "timestamp evidence, and predeclared analysis rules before any admissible "
    "provider-claimed online execution or apparatus label release. Public benchmark labels "
    "remain accessible outside the apparatus; the study does not claim human outcome "
    "blindness or independent organizational custody. This record reports no confirmatory "
    "finding."
)
ZENODO_NOTES = (
    "Sole creator: mhdk1602. The public record establishes the sole admissible "
    "provider-claimed execution lineage; it does not prove the physical absence of "
    "off-apparatus runs."
)
ZENODO_KEYWORDS = (
    "authorization-first retrieval",
    "retrieval-augmented generation",
    "approximate nearest neighbors",
    "local intrinsic dimensionality",
    "AI governance",
    "preregistration",
    "confirmatory research",
    "sealed execution",
)


def _fixed_protocol_metadata() -> dict[str, object]:
    """Return the sole mutable-draft metadata payload for the protocol record."""

    return {
        "access_right": ZENODO_ACCESS_RIGHT,
        "creators": [
            {
                "name": ZENODO_CREATOR,
                "orcid": ZENODO_CREATOR_ORCID,
            }
        ],
        "description": ZENODO_DESCRIPTION,
        "keywords": list(ZENODO_KEYWORDS),
        "license": ZENODO_LICENSE_ID,
        "notes": ZENODO_NOTES,
        "publication_date": ZENODO_PUBLICATION_DATE,
        "publication_type": ZENODO_PUBLICATION_TYPE,
        "title": ZENODO_TITLE,
        "upload_type": ZENODO_UPLOAD_TYPE,
    }


def _fixed_draft_metadata() -> dict[str, object]:
    metadata = _fixed_protocol_metadata()
    metadata["prereserve_doi"] = {
        "doi": ZENODO_RESERVED_DOI,
        "recid": ZENODO_RECORD_ID,
    }
    return metadata


PACKAGE_FILE_NAMES = (
    "c0-commit.txt",
    "c0-public-verification.json",
    "c1-commit-object.txt",
    "c1-commit.txt",
    "c1-tag-object-record.txt",
    "c1-tag-object.txt",
    "gh-version.txt",
    "manifest-gh-verification.json",
    "manifest-github-attestation-id.txt",
    "manifest-github-attestation-url.txt",
    "manifest-transition-receipt.json",
    "protocol-registry-record.json",
    "protocol-registry-record.sigstore.bundle.json",
    "registration-predicate.json",
    "registration-validation.json",
    "registry-attestation-validation.json",
    "registry-gh-verification.json",
    "registry-materialization.json",
    "registry-record-github-attestation-id.txt",
    "registry-record-github-attestation-url.txt",
    "registry-record-predicate.json",
    "study-manifest.json",
    "study-manifest.sha256",
    "study-manifest.sigstore.bundle.json",
    "workflow-run.txt",
    "zenodo-reservation.json",
    "SHA256SUMS",
)

_PACKAGE_NAMES = frozenset(PACKAGE_FILE_NAMES)
if len(PACKAGE_FILE_NAMES) != 27 or len(_PACKAGE_NAMES) != 27:  # pragma: no cover
    raise RuntimeError("the C1 registration package contract must contain 27 unique files")
_CHECKSUM_NAMES = tuple(
    sorted(_PACKAGE_NAMES - {"SHA256SUMS"}, key=lambda value: value.encode("utf-8"))
)
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_ATTESTATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_WORKFLOW_RUN = re.compile(
    r"^https://github\.com/mhdk1602/fractal-ann-diagnostics/actions/runs/"
    r"[1-9][0-9]*/attempts/[1-9][0-9]*$"
)
_GIT_IDENTITY_HEADER = re.compile(
    rb"^"
    + re.escape(f"{GIT_IDENTITY_NAME} <{GIT_IDENTITY_EMAIL}>".encode("ascii"))
    + rb" [0-9]+ [+-][0-9]{4}$"
)
_COAUTHOR_TRAILER = re.compile(rb"(?im)^co-authored-by\s*:")
_BUCKET_PATH = re.compile(
    r"^/api/files/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}(?:/[A-Za-z0-9._-]+)?$"
)
_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_PACKAGE_BYTES = 128 * 1024 * 1024
_MAX_JSON_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_DIRECT_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_TOKEN_BYTES = 4096
_TOKEN = re.compile(rb"^[A-Za-z0-9._~+-]{16,4096}$")
_TIMEOUT_SECONDS = 30.0
_PUBLICATION_POLL_ATTEMPTS = 60
_PUBLICATION_POLL_SECONDS = 5.0


def _public_file_uri(name: str) -> str:
    if name not in _PACKAGE_NAMES:
        raise ValueError("public Zenodo file name is outside the closed C1 package")
    return f"{ZENODO_PUBLIC_API_URI}/files/{quote(name, safe='')}/content"


_PUBLIC_FILE_PATHS = frozenset(urlsplit(_public_file_uri(name)).path for name in PACKAGE_FILE_NAMES)


class ZenodoPublicationError(SuiteAttemptError):
    """The local package or remote Zenodo state violates the fixed boundary."""


class _ZenodoHttpStatusError(ZenodoPublicationError):
    """A bounded HTTP failure whose status may identify publish integration lag."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"Zenodo returned HTTP status {status}")


@dataclass(frozen=True)
class RegistrationPackageFile:
    """One immutable file admitted from the validated package directory."""

    name: str
    size: int
    sha256: str
    md5: str
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class ValidatedRegistrationPackage:
    """Closed C1 package whose bytes and cross-file bindings have been checked."""

    root: Path
    c0_commit: str
    c1_commit: str
    manifest_sha256: str
    registry_record_sha256: str
    registry_record_bytes: bytes = field(repr=False)
    files: tuple[RegistrationPackageFile, ...] = field(repr=False)

    @property
    def inventory(self) -> Mapping[str, RegistrationPackageFile]:
        return {item.name: item for item in self.files}


def _fail(message: str) -> ZenodoPublicationError:
    return ZenodoPublicationError(message)


def _canonical_object(data: bytes, *, label: str) -> Mapping[str, Any]:
    value = _strict_json(data, label=label)
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise _fail(f"{label} must be a JSON object")
    if data != _canonical_bytes(value) + b"\n":
        raise _fail(f"{label} must be canonical JSON plus one LF")
    return value


def _one_line(data: bytes, *, label: str) -> str:
    if not data.endswith(b"\n") or data.count(b"\n") != 1 or b"\r" in data:
        raise _fail(f"{label} must contain exactly one UTF-8 line plus LF")
    try:
        value = data[:-1].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _fail(f"{label} must be UTF-8") from exc
    if not value or value != value.strip():
        raise _fail(f"{label} must contain one canonical non-empty value")
    return value


def _write_private_snapshot(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _oid_line(data: bytes, *, label: str) -> str:
    value = _one_line(data, label=label)
    if _SHA1.fullmatch(value) is None:
        raise _fail(f"{label} must contain one lowercase SHA-1 Git object ID")
    return value


def _read_package_file_at(root_descriptor: int, *, name: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=root_descriptor)
    except OSError as exc:
        raise _fail(f"cannot open C1 package file {name!r} without following links") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _MAX_FILE_BYTES
        ):
            raise _fail(f"C1 package file {name!r} must be one bounded regular file")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_FILE_BYTES + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > _MAX_FILE_BYTES:
                raise _fail(f"C1 package file {name!r} exceeds the byte limit")
        after = os.fstat(descriptor)
    except OSError as exc:
        raise _fail(f"cannot read C1 package file {name!r}") from exc
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise _fail(f"C1 package file {name!r} changed while it was read")
    try:
        current = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise _fail(f"C1 package file {name!r} disappeared after it was read") from exc
    if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
        before.st_dev,
        before.st_ino,
    ):
        raise _fail(f"C1 package file {name!r} was replaced while it was read")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise _fail(f"C1 package file {name!r} byte count changed while it was read")
    return data


def _read_closed_package(root: Path) -> dict[str, RegistrationPackageFile]:
    if not root.is_absolute():
        raise _fail("C1 registration package path must be absolute")
    try:
        resolved = root.resolve(strict=True)
        root_stat = root.stat(follow_symlinks=False)
    except OSError as exc:
        raise _fail("cannot inspect the C1 registration package directory") from exc
    if root.is_symlink() or root != resolved or not stat.S_ISDIR(root_stat.st_mode):
        raise _fail("C1 registration package must be one absolute real directory")
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_descriptor = os.open(resolved, root_flags)
    except OSError as exc:
        raise _fail("cannot pin the C1 registration package directory") from exc
    try:
        pinned = os.fstat(root_descriptor)
        if (pinned.st_dev, pinned.st_ino) != (root_stat.st_dev, root_stat.st_ino):
            raise _fail("C1 registration package directory changed while it was opened")
        try:
            entries = tuple(os.listdir(root_descriptor))
        except OSError as exc:
            raise _fail("cannot enumerate the C1 registration package") from exc
        observed = set(entries)
        if len(entries) != len(observed) or observed != _PACKAGE_NAMES:
            missing = sorted(_PACKAGE_NAMES - observed)
            extra = sorted(observed - _PACKAGE_NAMES)
            raise _fail(f"C1 package file set differs; missing={missing!r}, extra={extra!r}")

        files: dict[str, RegistrationPackageFile] = {}
        total = 0
        for name in PACKAGE_FILE_NAMES:
            data = _read_package_file_at(root_descriptor, name=name)
            total += len(data)
            if total > _MAX_PACKAGE_BYTES:
                raise _fail("C1 registration package exceeds the total byte limit")
            files[name] = RegistrationPackageFile(
                name=name,
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                md5=hashlib.md5(data, usedforsecurity=False).hexdigest(),
                data=data,
            )
        try:
            final_entries = tuple(os.listdir(root_descriptor))
        except OSError as exc:
            raise _fail("cannot re-enumerate the C1 registration package") from exc
        if len(final_entries) != len(_PACKAGE_NAMES) or set(final_entries) != _PACKAGE_NAMES:
            raise _fail("C1 package file set changed while it was read")
        try:
            current_root = root.stat(follow_symlinks=False)
        except OSError as exc:
            raise _fail("C1 registration package root disappeared while it was read") from exc
        if not stat.S_ISDIR(current_root.st_mode) or (
            current_root.st_dev,
            current_root.st_ino,
        ) != (pinned.st_dev, pinned.st_ino):
            raise _fail("C1 registration package root was replaced while it was read")
    finally:
        os.close(root_descriptor)
    return files


def _verify_checksum_manifest(files: Mapping[str, RegistrationPackageFile]) -> None:
    expected = b"".join(
        f"{files[name].sha256}  ./{name}\n".encode("ascii") for name in _CHECKSUM_NAMES
    )
    if files["SHA256SUMS"].data != expected:
        raise _fail("SHA256SUMS is not the canonical closed package checksum manifest")


def _git_object_oid(kind: str, data: bytes) -> str:
    header = f"{kind} {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _git_headers(data: bytes, *, label: str) -> list[bytes]:
    if not data.endswith(b"\n") or b"\r" in data or b"\x00" in data:
        raise _fail(f"{label} is not canonical Git object content")
    header, separator, _message = data.partition(b"\n\n")
    if not separator:
        raise _fail(f"{label} lacks the Git header/message boundary")
    return [line for line in header.splitlines() if line and not line.startswith(b" ")]


def _header_values(headers: Sequence[bytes], key: bytes) -> list[bytes]:
    prefix = key + b" "
    return [line[len(prefix) :] for line in headers if line.startswith(prefix)]


def _verify_git_boundary(
    files: Mapping[str, RegistrationPackageFile],
    *,
    c0_commit: str,
    c1_commit: str,
    predicate: Mapping[str, Any],
) -> None:
    commit_data = files["c1-commit-object.txt"].data
    if _git_object_oid("commit", commit_data) != c1_commit:
        raise _fail("C1 commit object bytes do not hash to the fixed C1 commit")
    headers = _git_headers(commit_data, label="C1 commit object")
    for key in (b"tree", b"author", b"committer"):
        if len(_header_values(headers, key)) != 1:
            raise _fail(f"C1 commit object must contain exactly one {key.decode()} header")
    if (
        _GIT_IDENTITY_HEADER.fullmatch(_header_values(headers, b"author")[0]) is None
        or _GIT_IDENTITY_HEADER.fullmatch(_header_values(headers, b"committer")[0]) is None
        or _COAUTHOR_TRAILER.search(commit_data.partition(b"\n\n")[2]) is not None
    ):
        raise _fail("C1 commit must use only the fixed mhdk1602 author and committer identity")
    parents = _header_values(headers, b"parent")
    if parents != [c0_commit.encode("ascii")]:
        raise _fail("C1 commit must be the direct single-parent child of C0")

    tag_line = _one_line(files["c1-tag-object.txt"].data, label="C1 tag-object binding")
    try:
        tag_oid, tag_type = tag_line.split(" ")
    except ValueError as exc:
        raise _fail("C1 tag-object binding must contain one object ID and type") from exc
    freeze = predicate.get("freeze")
    if not isinstance(freeze, Mapping):
        raise _fail("C1 predicate lacks its freeze object")
    if (
        _SHA1.fullmatch(tag_oid) is None
        or tag_type not in {"commit", "tag"}
        or freeze.get("tag_object_id") != tag_oid
        or freeze.get("tag_object_type") != tag_type
    ):
        raise _fail("C1 tag object differs from the signed freeze predicate")
    tag_data = files["c1-tag-object-record.txt"].data
    if tag_type == "commit":
        if tag_oid != c1_commit or tag_data != commit_data:
            raise _fail("lightweight C1 tag does not resolve to the exact C1 commit object")
        return
    if _git_object_oid("tag", tag_data) != tag_oid:
        raise _fail("annotated C1 tag bytes do not hash to the signed tag object")
    tag_headers = _git_headers(tag_data, label="annotated C1 tag object")
    if (
        _header_values(tag_headers, b"object") != [c1_commit.encode("ascii")]
        or _header_values(tag_headers, b"type") != [b"commit"]
        or _header_values(tag_headers, b"tag") != [C1_REF.removeprefix("refs/tags/").encode()]
        or len(_header_values(tag_headers, b"tagger")) != 1
        or _GIT_IDENTITY_HEADER.fullmatch(_header_values(tag_headers, b"tagger")[0]) is None
    ):
        raise _fail("annotated C1 tag differs from the fixed commit, ref, or mhdk1602 tagger")


def _verify_attestation_pointer(
    files: Mapping[str, RegistrationPackageFile],
    *,
    prefix: str,
) -> str:
    identifier = _one_line(
        files[f"{prefix}-github-attestation-id.txt"].data,
        label=f"{prefix} GitHub attestation ID",
    )
    if _ATTESTATION_ID.fullmatch(identifier) is None:
        raise _fail(f"{prefix} GitHub attestation ID has an unsafe form")
    url = _one_line(
        files[f"{prefix}-github-attestation-url.txt"].data,
        label=f"{prefix} GitHub attestation URL",
    )
    expected = f"https://github.com/{REPOSITORY}/attestations/{identifier}"
    if url != expected:
        raise _fail(f"{prefix} GitHub attestation URL does not bind its exact ID")
    return identifier


def validate_registration_package(package_dir: Path) -> ValidatedRegistrationPackage:
    """Validate the exact retained C1 package without making a network request."""

    files = _read_closed_package(package_dir)
    _verify_checksum_manifest(files)
    root = package_dir.resolve(strict=True)

    c0_commit = _oid_line(files["c0-commit.txt"].data, label="C0 commit")
    c1_commit = _oid_line(files["c1-commit.txt"].data, label="C1 commit")
    if c0_commit == c1_commit:
        raise _fail("C0 and C1 commits must differ")

    snapshot_registration_predicate = _canonical_object(
        files["registration-predicate.json"].data,
        label="C1 registration predicate",
    )
    try:
        with tempfile.TemporaryDirectory(prefix="fractal-c1-package-json-") as directory:
            snapshot_root = Path(directory)
            predicate_path = snapshot_root / "registration-predicate.json"
            _write_private_snapshot(
                predicate_path,
                files["registration-predicate.json"].data,
            )
            registration_predicate, predicate_c1 = _load_closed_c1_predicate(predicate_path)
    except SuiteAttemptError as exc:
        raise _fail(f"invalid C1 registration predicate: {exc}") from exc
    if registration_predicate != snapshot_registration_predicate:
        raise _fail("C1 registration predicate changed after package admission")
    if predicate_c1 != c1_commit:
        raise _fail("C1 commit file differs from the signed registration predicate")
    freeze = registration_predicate["freeze"]
    if freeze.get("c0_commit") != c0_commit or freeze.get("c1_ref") != C1_REF:
        raise _fail("C0/C1 commit and ref files differ from the signed freeze predicate")

    manifest_data = files["study-manifest.json"].data
    manifest_value = _strict_json(manifest_data, label="frozen C1 study manifest")
    if not isinstance(manifest_value, Mapping):
        raise _fail("frozen C1 study manifest must be one JSON object")
    try:
        validate_study_manifest(manifest_value, require_frozen=True)
        semantic_digest = manifest_sha256(manifest_value)
    except (SuiteAttemptError, ValueError, TypeError) as exc:
        raise _fail(f"frozen C1 study manifest is invalid: {exc}") from exc
    manifest_row = registration_predicate["manifest"]
    lock_row = registration_predicate["lock"]
    if (
        manifest_row.get("path") != C1_MANIFEST_PATH
        or manifest_row.get("file_sha256") != files["study-manifest.json"].sha256
        or manifest_row.get("manifest_sha256") != semantic_digest
    ):
        raise _fail("frozen manifest bytes or meaning differ from the signed predicate")
    lock_data = files["study-manifest.sha256"].data
    if (
        lock_row.get("path") != C1_LOCK_PATH
        or lock_data != f"{semantic_digest}\n".encode("ascii")
        or lock_row.get("file_sha256") != files["study-manifest.sha256"].sha256
        or lock_row.get("manifest_sha256") != semantic_digest
    ):
        raise _fail("manifest lock differs from the frozen manifest and signed predicate")

    c0_public_bytes = files["c0-public-verification.json"].data
    try:
        with tempfile.TemporaryDirectory(prefix="fractal-c0-public-package-") as directory:
            receipt_path = Path(directory) / "c0-public-verification.json"
            _write_private_snapshot(receipt_path, c0_public_bytes)
            c0_public = load_c0_public_verification_receipt(receipt_path)
    except C0PublicVerificationError as exc:
        raise _fail(f"invalid retained C0 public verification: {exc}") from exc
    sealed_execution = manifest_value.get("sealed_execution")
    if not isinstance(sealed_execution, Mapping):
        raise _fail("frozen manifest lacks its sealed_execution object")
    manifest_c0_binding = sealed_execution.get("c0_evidence_release")
    expected_c0_public_row = {
        "binding_sha256": c0_public.binding_sha256,
        "file_sha256": files["c0-public-verification.json"].sha256,
        "path": "c0-public-verification.json",
        "release_tag": c0_public.release_tag,
        "schema_version": C0_PUBLIC_VERIFICATION_SCHEMA,
        "target_commit": c0_public.target_commit,
    }
    if (
        c0_public.canonical_file_bytes() != c0_public_bytes
        or c0_public.file_sha256 != files["c0-public-verification.json"].sha256
        or c0_public.binding_source_kind != "frozen-manifest"
        or c0_public.binding_source_file_sha256 != files["study-manifest.json"].sha256
        or c0_public.target_commit != c0_commit
        or c0_public.c0_evidence_release_binding != manifest_c0_binding
        or registration_predicate.get("c0_public_verification") != expected_c0_public_row
        or files["gh-version.txt"].data
        != c0_public.gh_version_text.encode("utf-8", errors="strict")
    ):
        raise _fail(
            "retained C0 public verification differs from the manifest, "
            "signed predicate, or verifier transcript"
        )

    transition_bytes = files["manifest-transition-receipt.json"].data
    try:
        transition_receipt = loads_c1_manifest_transition_receipt(transition_bytes)
        verify_c1_manifest_transition_receipt_bindings(
            transition_receipt,
            frozen_manifest=manifest_value,
            frozen_manifest_bytes=manifest_data,
            c0_commit=c0_commit,
        )
    except C1ManifestTransitionError as exc:
        raise _fail(f"invalid C1 manifest transition receipt: {exc}") from exc
    transition_row = registration_predicate.get("manifest_transition")
    expected_transition_row = {
        "candidate_manifest_assembly_receipt_file_sha256": (
            transition_receipt.candidate_manifest_assembly_receipt_file_sha256
        ),
        "candidate_manifest_file_sha256": transition_receipt.candidate_manifest_file_sha256,
        "candidate_manifest_sha256": transition_receipt.candidate_manifest_sha256,
        "file_sha256": files["manifest-transition-receipt.json"].sha256,
        "path": C1_TRANSITION_RECEIPT_PATH,
        "schema_version": C1_MANIFEST_TRANSITION_RECEIPT_SCHEMA,
    }
    if transition_row != expected_transition_row:
        raise _fail("manifest transition receipt differs from the signed C1 predicate")

    try:
        with tempfile.TemporaryDirectory(prefix="fractal-c1-package-json-") as directory:
            reservation_path = Path(directory) / "zenodo-reservation.json"
            _write_private_snapshot(reservation_path, files["zenodo-reservation.json"].data)
            reservation, reservation_digest = _load_zenodo_reservation(reservation_path)
    except SuiteAttemptError as exc:
        raise _fail(f"invalid fixed Zenodo reservation: {exc}") from exc
    snapshot_reservation = _canonical_object(
        files["zenodo-reservation.json"].data,
        label="fixed Zenodo reservation",
    )
    reservation_row = registration_predicate["registry_reservation"]
    if (
        reservation != snapshot_reservation
        or reservation_digest != files["zenodo-reservation.json"].sha256
        or reservation_row.get("file_sha256") != reservation_digest
    ):
        raise _fail("Zenodo reservation bytes differ from the signed predicate")

    registration_receipt = _canonical_object(
        files["registration-validation.json"].data,
        label="C1 registration verification receipt",
    )
    expected_registration_receipt = {
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "predicate": registration_predicate,
        "predicate_type": REGISTRATION_PREDICATE_TYPE,
        "repository": REPOSITORY,
        "schema_version": REGISTRATION_RECEIPT_SCHEMA,
        "workflow_ref": f"{REPOSITORY}/{REGISTRATION_WORKFLOW_PATH}@{C1_REF}",
        "workflow_sha": c1_commit,
    }
    if registration_receipt != expected_registration_receipt:
        raise _fail("C1 registration verification receipt differs from fixed workflow policy")

    record_value = _canonical_object(
        files["protocol-registry-record.json"].data,
        label="protocol registry record",
    )
    try:
        record = ProtocolRegistryRecord.from_dict(record_value)
    except (ValueError, TypeError) as exc:
        raise _fail(f"protocol registry record is invalid: {exc}") from exc
    if (
        record.canonical_bytes() + b"\n" != files["protocol-registry-record.json"].data
        or record.manifest_sha256 != semantic_digest
        or record.registry_identity != ZENODO_REGISTRY_IDENTITY
        or record.registry_uri != ZENODO_REGISTRY_URI
    ):
        raise _fail("protocol registry record differs from the fixed manifest or Zenodo record")

    try:
        first_predicate = registration_predicate
        first_bundle = files["study-manifest.sigstore.bundle.json"].data
        first_observation = parse_sigstore_bundle(first_bundle)
        _verify_closed_c1_statement(
            first_observation,
            predicate_type=REGISTRATION_PREDICATE_TYPE,
            predicate=first_predicate,
            subject_name=C1_MANIFEST_PATH,
            subject_digest=files["study-manifest.json"].sha256,
        )
        with tempfile.TemporaryDirectory(prefix="fractal-c1-package-json-") as directory:
            registry_predicate_path = Path(directory) / "registry-record-predicate.json"
            _write_private_snapshot(
                registry_predicate_path,
                files["registry-record-predicate.json"].data,
            )
            registry_predicate, registry_c1, recorded_first = _load_registry_record_predicate(
                registry_predicate_path,
                record=record,
            )
    except SuiteAttemptError as exc:
        raise _fail(f"invalid C1 attestation package: {exc}") from exc
    snapshot_registry_predicate = _canonical_object(
        files["registry-record-predicate.json"].data,
        label="registry-record predicate",
    )
    if registry_predicate != snapshot_registry_predicate:
        raise _fail("registry-record predicate changed after package admission")
    if registry_c1 != c1_commit:
        raise _fail("registry-record predicate uses another C1 commit")
    first_fields = (
        "log_key_sha256",
        "log_index",
        "entry_id",
        "integrated_at_utc",
        "timestamp_token_sha256",
    )
    if any(getattr(first_observation, key) != getattr(recorded_first, key) for key in first_fields):
        raise _fail("registry predicate does not bind the actual manifest Rekor observation")
    if (
        registry_predicate["manifest_attestation"].get("bundle_sha256")
        != files["study-manifest.sigstore.bundle.json"].sha256
        or record.registered_at_utc != first_observation.integrated_at_utc
    ):
        raise _fail("registry record time or manifest bundle digest differs from Rekor evidence")

    try:
        second_bundle = files["protocol-registry-record.sigstore.bundle.json"].data
        second_observation = parse_sigstore_bundle(second_bundle)
        _verify_closed_c1_statement(
            second_observation,
            predicate_type=REGISTRY_RECORD_PREDICATE_TYPE,
            predicate=registry_predicate,
            subject_name=REGISTRY_RECORD_SUBJECT_PATH,
            subject_digest=record.record_sha256,
        )
    except SuiteAttemptError as exc:
        raise _fail(f"invalid registry-record attestation: {exc}") from exc
    if (
        _utc_datetime(second_observation.integrated_at_utc, label="registry Rekor time")
        < _utc_datetime(first_observation.integrated_at_utc, label="manifest Rekor time")
        or second_observation.entry_id == first_observation.entry_id
        or second_observation.timestamp_token_sha256 == first_observation.timestamp_token_sha256
    ):
        raise _fail("registry-record Rekor evidence does not follow the manifest evidence")

    manifest_verification = files["manifest-gh-verification.json"].data
    registry_verification = files["registry-gh-verification.json"].data
    try:
        _validated_gh_output(manifest_verification)
        _validated_gh_output(registry_verification)
    except SuiteAttemptError as exc:
        raise _fail(f"invalid typed GitHub attestation verification output: {exc}") from exc
    materialization = _canonical_object(
        files["registry-materialization.json"].data,
        label="registry materialization receipt",
    )
    expected_materialization = {
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "manifest_attestation_verification_sha256": hashlib.sha256(
            manifest_verification
        ).hexdigest(),
        "predicate": registry_predicate,
        "predicate_type": REGISTRY_RECORD_PREDICATE_TYPE,
        "registry_record_sha256": record.record_sha256,
        "schema_version": REGISTRY_MATERIALIZATION_SCHEMA,
    }
    if materialization != expected_materialization:
        raise _fail("registry materialization receipt differs from the attested package")
    final_receipt = _canonical_object(
        files["registry-attestation-validation.json"].data,
        label="registry attestation verification receipt",
    )
    expected_final_receipt = {
        "c1_commit": c1_commit,
        "control_boundary": COMMON_CONTROL_LIMITATION,
        "manifest_rekor_entry_id": first_observation.entry_id,
        "manifest_rekor_integrated_at_utc": first_observation.integrated_at_utc,
        "registry_record_bundle_sha256": files[
            "protocol-registry-record.sigstore.bundle.json"
        ].sha256,
        "registry_record_rekor_entry_id": second_observation.entry_id,
        "registry_record_rekor_integrated_at_utc": second_observation.integrated_at_utc,
        "registry_record_sha256": record.record_sha256,
        "registry_record_verification_sha256": hashlib.sha256(registry_verification).hexdigest(),
        "schema_version": REGISTRY_ATTESTATION_RECEIPT_SCHEMA,
    }
    if final_receipt != expected_final_receipt:
        raise _fail("registry attestation receipt differs from the actual signed evidence")

    _verify_git_boundary(
        files,
        c0_commit=c0_commit,
        c1_commit=c1_commit,
        predicate=registration_predicate,
    )
    manifest_attestation_id = _verify_attestation_pointer(files, prefix="manifest")
    registry_attestation_id = _verify_attestation_pointer(files, prefix="registry-record")
    if manifest_attestation_id == registry_attestation_id:
        raise _fail("manifest and registry-record GitHub attestations must have distinct IDs")
    workflow_run = _one_line(files["workflow-run.txt"].data, label="workflow run URL")
    if _WORKFLOW_RUN.fullmatch(workflow_run) is None:
        raise _fail("workflow run URL does not name the fixed GitHub repository")
    try:
        gh_version = files["gh-version.txt"].data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _fail("gh version evidence must be UTF-8") from exc
    if (
        not gh_version.startswith("gh version ")
        or not gh_version.endswith("\n")
        or any(ord(character) < 32 and character not in "\n\t" for character in gh_version)
    ):
        raise _fail("gh version evidence is malformed")

    ordered = tuple(files[name] for name in PACKAGE_FILE_NAMES)
    return ValidatedRegistrationPackage(
        root=root,
        c0_commit=c0_commit,
        c1_commit=c1_commit,
        manifest_sha256=semantic_digest,
        registry_record_sha256=record.record_sha256,
        registry_record_bytes=files["protocol-registry-record.json"].data,
        files=ordered,
    )


def verify_registration_package_attestations(
    package: ValidatedRegistrationPackage,
    *,
    verifier: C1AttestationVerifier | None = None,
) -> Mapping[str, Any]:
    """Freshly verify both retained bundles under the exact C1 workflow identity."""

    active = verifier if verifier is not None else GhC1AttestationVerifier()
    inventory = package.inventory
    checks = (
        (
            inventory["study-manifest.json"],
            inventory["study-manifest.sigstore.bundle.json"],
            REGISTRATION_PREDICATE_TYPE,
        ),
        (
            inventory["protocol-registry-record.json"],
            inventory["protocol-registry-record.sigstore.bundle.json"],
            REGISTRY_RECORD_PREDICATE_TYPE,
        ),
    )
    verification_sha256: list[str] = []
    with tempfile.TemporaryDirectory(prefix="fractal-c1-gh-preflight-") as directory:
        snapshot_root = Path(directory)
        for subject, bundle, predicate_type in checks:
            subject_path = snapshot_root / subject.name
            bundle_path = snapshot_root / bundle.name
            _write_private_snapshot(subject_path, subject.data)
            _write_private_snapshot(bundle_path, bundle.data)
            try:
                encoded = active.verify(
                    subject_path=subject_path,
                    bundle_path=bundle_path,
                    c1_commit=package.c1_commit,
                    predicate_type=predicate_type,
                )
                _validated_gh_output(encoded)
                root_descriptor = os.open(
                    snapshot_root,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    if (
                        _read_package_file_at(root_descriptor, name=subject.name) != subject.data
                        or _read_package_file_at(root_descriptor, name=bundle.name) != bundle.data
                    ):
                        raise _fail("C1 attestation snapshot changed during verification")
                finally:
                    os.close(root_descriptor)
            except SuiteAttemptError as exc:
                raise _fail(f"fresh C1 attestation verification failed: {exc}") from exc
            verification_sha256.append(hashlib.sha256(encoded).hexdigest())
    refreshed = validate_registration_package(package.root)
    if refreshed != package:
        raise _fail("C1 registration package changed during attestation verification")
    return {
        "c1_commit": package.c1_commit,
        "predicate_types": [REGISTRATION_PREDICATE_TYPE, REGISTRY_RECORD_PREDICATE_TYPE],
        "schema_version": "fractal-zenodo-package-attestation-preflight-v2",
        "verification_sha256": verification_sha256,
        "verified": True,
    }


def _require_package_current(
    package: ValidatedRegistrationPackage,
    *,
    phase: str,
) -> None:
    refreshed = validate_registration_package(package.root)
    if refreshed != package:
        raise _fail(f"C1 registration package changed during {phase}")


@dataclass(frozen=True)
class _RemoteFile:
    name: str
    size: int
    md5: str


def _nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is int and value >= 0:
        return value
    if type(value) is str and value.isascii() and value.isdigit():
        return int(value)
    raise _fail(f"{label} must be a non-negative integer")


def _remote_file(row: object, *, label: str) -> _RemoteFile:
    if not isinstance(row, Mapping):
        raise _fail(f"{label} must be a JSON object")
    names = [row[key] for key in ("key", "filename") if key in row]
    sizes = [row[key] for key in ("size", "filesize") if key in row]
    if not names or any(value != names[0] for value in names):
        raise _fail(f"{label} has no single canonical file name")
    if type(names[0]) is not str or names[0] not in _PACKAGE_NAMES:
        raise _fail(f"{label} names an unexpected file")
    if not sizes:
        raise _fail(f"{label} lacks a byte count")
    parsed_sizes = [_nonnegative_integer(value, label=f"{label} size") for value in sizes]
    if any(value != parsed_sizes[0] for value in parsed_sizes):
        raise _fail(f"{label} has conflicting byte counts")
    checksum = row.get("checksum")
    if type(checksum) is not str:
        raise _fail(f"{label} lacks an MD5 checksum")
    md5 = checksum.removeprefix("md5:")
    if _MD5.fullmatch(md5) is None:
        raise _fail(f"{label} checksum is not one lowercase MD5")
    return _RemoteFile(name=names[0], size=parsed_sizes[0], md5=md5)


def _inventory(
    value: object,
    *,
    label: str,
    require_public_content_links: bool = False,
) -> dict[str, _RemoteFile]:
    if not isinstance(value, list):
        raise _fail(f"{label} must be a JSON array")
    result: dict[str, _RemoteFile] = {}
    for position, row in enumerate(value):
        item = _remote_file(row, label=f"{label}[{position}]")
        if item.name in result:
            raise _fail(f"{label} repeats file {item.name!r}")
        if require_public_content_links:
            assert isinstance(row, Mapping)
            links = row.get("links")
            expected = _public_file_uri(item.name)
            if not isinstance(links, Mapping) or links.get("self") != expected:
                raise _fail(f"{label}[{position}] lacks its exact public content URI")
            _validate_transport_url(expected)
        result[item.name] = item
    return result


def _check_inventory(
    package: ValidatedRegistrationPackage,
    remote: Mapping[str, _RemoteFile],
    *,
    require_complete: bool,
) -> tuple[str, ...]:
    expected = package.inventory
    extra = sorted(set(remote) - set(expected))
    if extra:
        raise _fail(f"Zenodo inventory contains unexpected files: {extra!r}")
    for name, item in remote.items():
        local = expected[name]
        if item.size != local.size or item.md5 != local.md5:
            raise _fail(f"Zenodo file {name!r} differs in size or MD5")
    missing = tuple(sorted(set(expected) - set(remote), key=lambda value: value.encode()))
    if require_complete and missing:
        raise _fail(f"Zenodo inventory is incomplete: {list(missing)!r}")
    return missing


def _creator_orcid(value: object) -> str:
    if type(value) is not str:
        raise _fail("Zenodo creator ORCID must be a string")
    return value.removeprefix("https://orcid.org/")


def _verify_creator(metadata: Mapping[str, Any]) -> None:
    creators = metadata.get("creators")
    if not isinstance(creators, list) or len(creators) != 1:
        raise _fail("Zenodo metadata must contain exactly one creator")
    creator = creators[0]
    if not isinstance(creator, Mapping):
        raise _fail("Zenodo creator must be a JSON object")
    if not {"name", "orcid"}.issubset(creator) or not set(creator).issubset(
        {"name", "orcid", "affiliation"}
    ):
        raise _fail("Zenodo creator fields differ from the fixed identity")
    if (
        creator.get("name") != ZENODO_CREATOR
        or _creator_orcid(creator.get("orcid")) != ZENODO_CREATOR_ORCID
        or creator.get("affiliation") not in {None, ""}
    ):
        raise _fail("Zenodo creator name, ORCID, or affiliation differs from the fixed identity")
    contributors = metadata.get("contributors")
    if contributors is not None and contributors != []:
        raise _fail("Zenodo metadata must not name a contributor or co-author")


def _license_id(value: object, *, label: str) -> str:
    if type(value) is str:
        result = value
    elif isinstance(value, Mapping):
        result = value.get("id")
    else:
        raise _fail(f"{label} is not a Zenodo license identifier")
    if result != ZENODO_LICENSE_ID:
        raise _fail(f"{label} differs from the fixed CC BY 4.0 license")
    return result


def _verify_fixed_metadata(
    metadata: object,
    *,
    public: bool,
    require_reserved_doi: bool = True,
) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise _fail("Zenodo metadata must be a JSON object")
    if metadata.get("title") != ZENODO_TITLE:
        raise _fail("Zenodo title differs from the fixed C1 title")
    _verify_creator(metadata)

    if metadata.get("access_right") != ZENODO_ACCESS_RIGHT:
        raise _fail("Zenodo access right differs from the fixed open record")
    if metadata.get("description") != ZENODO_DESCRIPTION:
        raise _fail("Zenodo description differs from the fixed prospective protocol text")
    keywords = metadata.get("keywords")
    if type(keywords) is not list or tuple(keywords) != ZENODO_KEYWORDS:
        raise _fail("Zenodo keywords differ from the fixed ordered protocol terms")
    if "notes" in metadata and metadata.get("notes") != ZENODO_NOTES:
        raise _fail("Zenodo notes differ from the fixed publication boundary")
    if metadata.get("publication_date") != ZENODO_PUBLICATION_DATE:
        raise _fail("Zenodo publication date differs from the fixed reservation date")

    if public:
        resource_type = metadata.get("resource_type")
        if not isinstance(resource_type, Mapping):
            raise _fail("Zenodo public resource_type must be a JSON object")
        if (
            resource_type.get("type") != ZENODO_UPLOAD_TYPE
            or resource_type.get("subtype") != ZENODO_PUBLICATION_TYPE
        ):
            raise _fail("Zenodo public resource type differs from publication-other")
    elif (
        metadata.get("upload_type") != ZENODO_UPLOAD_TYPE
        or metadata.get("publication_type") != ZENODO_PUBLICATION_TYPE
    ):
        raise _fail("Zenodo draft resource type differs from publication-other")

    observed_licenses: list[str] = []
    if "license" in metadata:
        observed_licenses.append(_license_id(metadata["license"], label="Zenodo license"))
    if "rights" in metadata:
        rights = metadata["rights"]
        if not isinstance(rights, list) or len(rights) != 1:
            raise _fail("Zenodo rights must contain exactly one license")
        observed_licenses.append(_license_id(rights[0], label="Zenodo right"))
    if not observed_licenses or any(value != ZENODO_LICENSE_ID for value in observed_licenses):
        raise _fail("Zenodo metadata lacks the fixed CC BY 4.0 license")

    if not public and require_reserved_doi:
        reserved = metadata.get("prereserve_doi")
        if not isinstance(reserved, Mapping) or set(reserved) != {"doi", "recid"}:
            raise _fail("Zenodo draft lacks the exact reserved DOI object")
        if (
            reserved.get("doi") != ZENODO_RESERVED_DOI
            or _nonnegative_integer(reserved.get("recid"), label="reserved DOI record ID")
            != ZENODO_RECORD_ID
        ):
            raise _fail("Zenodo draft reserved DOI differs from the fixed reservation")
    return metadata


def _verify_exact_draft_metadata(metadata: object) -> Mapping[str, Any]:
    verified = _verify_fixed_metadata(metadata, public=False)
    if verified != _fixed_draft_metadata():
        raise _fail("Zenodo draft metadata differs from the exact protocol-record payload")
    return verified


def _exact_record_id(value: object, *, label: str) -> int:
    result = _nonnegative_integer(value, label=label)
    if result != ZENODO_RECORD_ID:
        raise _fail(f"{label} differs from fixed Zenodo record {ZENODO_RECORD_ID}")
    return result


def _validate_api_url(value: object, *, expected: str, label: str) -> str:
    if type(value) is not str or value != expected:
        raise _fail(f"{label} differs from the fixed Zenodo endpoint")
    _validate_transport_url(value)
    return value


def _validate_bucket_url(value: object) -> str:
    if type(value) is not str:
        raise _fail("Zenodo draft bucket link must be a string")
    _validate_transport_url(value)
    parsed = urlsplit(value)
    if parsed.path.endswith(tuple(f"/{name}" for name in PACKAGE_FILE_NAMES)):
        raise _fail("Zenodo draft bucket link must name the bucket, not one file")
    if _BUCKET_PATH.fullmatch(parsed.path) is None or parsed.path.count("/") != 3:
        raise _fail("Zenodo draft bucket link has another path")
    return value


def _verify_draft(
    package: ValidatedRegistrationPackage,
    payload: object,
    *,
    require_complete: bool,
) -> tuple[str, tuple[str, ...]]:
    if not isinstance(payload, Mapping):
        raise _fail("Zenodo draft response must be a JSON object")
    _exact_record_id(payload.get("id"), label="Zenodo draft ID")
    _exact_record_id(payload.get("record_id"), label="Zenodo draft record ID")
    if payload.get("state") != "unsubmitted" or payload.get("submitted") is not False:
        raise _fail("Zenodo record is not the fixed unsubmitted draft")
    if "doi" in payload and payload.get("doi") not in {"", None, ZENODO_RESERVED_DOI}:
        raise _fail("Zenodo draft DOI differs from the reserved DOI")
    _verify_exact_draft_metadata(payload.get("metadata"))
    links = payload.get("links")
    if not isinstance(links, Mapping):
        raise _fail("Zenodo draft lacks API links")
    _validate_api_url(links.get("self"), expected=ZENODO_DRAFT_API_URI, label="draft self link")
    _validate_api_url(
        links.get("publish"), expected=ZENODO_PUBLISH_API_URI, label="draft publish link"
    )
    bucket = _validate_bucket_url(links.get("bucket"))
    remote = _inventory(payload.get("files"), label="Zenodo draft files")
    missing = _check_inventory(package, remote, require_complete=require_complete)
    return bucket, missing


def _verify_public_payload(
    package: ValidatedRegistrationPackage,
    payload: object,
) -> None:
    if not isinstance(payload, Mapping):
        raise _fail("Zenodo public response must be a JSON object")
    _exact_record_id(payload.get("id"), label="Zenodo public ID")
    if "recid" in payload:
        _exact_record_id(payload.get("recid"), label="Zenodo public recid")
    if payload.get("doi") != ZENODO_RESERVED_DOI:
        raise _fail("Zenodo public DOI differs from the fixed reserved DOI")
    if (
        payload.get("submitted") is not True
        or payload.get("state") != "done"
        or payload.get("status") != "published"
    ):
        raise _fail("Zenodo public record is not in the submitted published state")
    _verify_fixed_metadata(payload.get("metadata"), public=True)
    remote = _inventory(
        payload.get("files"),
        label="Zenodo public files",
        require_public_content_links=True,
    )
    _check_inventory(package, remote, require_complete=True)


def _verify_publish_response(
    package: ValidatedRegistrationPackage,
    payload: object,
) -> None:
    """Validate the deposition API acknowledgement before the anonymous readback."""

    if not isinstance(payload, Mapping):
        raise _fail("Zenodo publish response must be a JSON object")
    _exact_record_id(payload.get("id"), label="Zenodo published deposition ID")
    if "record_id" in payload:
        _exact_record_id(payload.get("record_id"), label="Zenodo published record ID")
    if payload.get("submitted") is not True or payload.get("state") != "done":
        raise _fail("Zenodo publish response is not in the submitted done state")
    if "status" in payload and payload.get("status") != "published":
        raise _fail("Zenodo publish response has another publication status")
    if payload.get("doi") != ZENODO_RESERVED_DOI:
        raise _fail("Zenodo publish response has another DOI")
    metadata = payload.get("metadata")
    public_metadata = isinstance(metadata, Mapping) and "resource_type" in metadata
    _verify_fixed_metadata(
        metadata,
        public=public_metadata,
        require_reserved_doi=False,
    )
    remote = _inventory(payload.get("files"), label="Zenodo published files")
    _check_inventory(package, remote, require_complete=True)


class _Transport(Protocol):
    def get_json(self, url: str, *, authenticated: bool) -> Mapping[str, Any]: ...

    def get_bytes(self, url: str, *, authenticated: bool) -> bytes: ...

    def put_bytes(self, url: str, data: bytes) -> Mapping[str, Any]: ...

    def put_json(self, url: str, value: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def post_json(self, url: str) -> Mapping[str, Any]: ...


class _NoRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib_request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _verified_tls_context() -> ssl.SSLContext:
    """Build a verified context that remains usable behind managed enterprise CAs.

    Python 3.14 enables OpenSSL's strict X.509 mode by default. Some managed
    inspection roots accepted by the operating-system trust store predate that
    profile. Removing only the strict-profile flag retains certificate-chain
    validation, hostname verification, and the TLS 1.2 floor.
    """

    context = ssl.create_default_context()
    strict = getattr(ssl, "VERIFY_X509_STRICT", 0)
    if strict:
        context.verify_flags &= ~strict
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise _fail("Zenodo TLS context lacks certificate or hostname verification")
    return context


def _validate_transport_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise _fail("Zenodo transport URL is malformed") from exc
    fixed_paths = {
        urlsplit(ZENODO_DRAFT_API_URI).path,
        urlsplit(ZENODO_PUBLISH_API_URI).path,
        urlsplit(ZENODO_PUBLIC_API_URI).path,
        urlsplit(ZENODO_REGISTRY_URI).path,
        *_PUBLIC_FILE_PATHS,
    }
    if (
        parsed.scheme != "https"
        or parsed.netloc != "zenodo.org"
        or parsed.hostname != "zenodo.org"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or (parsed.path not in fixed_paths and _BUCKET_PATH.fullmatch(parsed.path) is None)
        or unquote(parsed.path) != parsed.path
    ):
        raise _fail("transport permits only fixed query-free HTTPS paths at zenodo.org")
    return url


class _ZenodoHttpsTransport:
    """Verified-TLS, no-redirect transport used only by the production CLI."""

    def __init__(self, token: bytearray | None) -> None:
        self._token = token
        context = _verified_tls_context()
        self._opener = urllib_request.build_opener(
            _NoRedirectHandler(), urllib_request.HTTPSHandler(context=context)
        )

    def __repr__(self) -> str:
        return "_ZenodoHttpsTransport(token=<redacted>)"

    def close(self) -> None:
        if self._token is not None:
            for position in range(len(self._token)):
                self._token[position] = 0
            self._token = None

    def _request(
        self,
        url: str,
        *,
        method: str,
        authenticated: bool,
        data: bytes | None,
        expected_status: int,
        max_bytes: int,
        request_content_type: str | None = None,
        expected_response_content_type: str | None = None,
    ) -> bytes:
        _validate_transport_url(url)
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "fractal-ann-diagnostics/0.3.0",
        }
        if authenticated:
            if self._token is None:
                raise _fail("authenticated Zenodo request has no token")
            headers["Authorization"] = "Bearer " + self._token.decode("ascii", errors="strict")
        if data is not None:
            if request_content_type not in {"application/json", "application/octet-stream"}:
                raise _fail("Zenodo request body lacks an admitted content type")
            headers["Content-Type"] = request_content_type
            headers["Content-Length"] = str(len(data))
        request = urllib_request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            response = self._opener.open(request, timeout=_TIMEOUT_SECONDS)
        except urllib_error.HTTPError as exc:
            raise _ZenodoHttpStatusError(exc.code) from exc
        except (OSError, TimeoutError, urllib_error.URLError, ValueError) as exc:
            raise _fail("Zenodo HTTPS request failed before a verified response") from exc
        with response:
            status = getattr(response, "status", response.getcode())
            if status != expected_status:
                raise _fail(f"Zenodo returned HTTP status {status}, expected {expected_status}")
            if response.geturl() != url:
                raise _fail("Zenodo response URL changed; redirects are forbidden")
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if encoding not in {"", "identity"}:
                raise _fail("Zenodo response used a content encoding")
            if expected_response_content_type is not None:
                raw_content_type = response.headers.get("Content-Type")
                if type(raw_content_type) is not str:
                    raise _fail("Zenodo JSON response lacks Content-Type")
                parts = [part.strip().lower() for part in raw_content_type.split(";")]
                if (
                    not parts
                    or parts[0] != expected_response_content_type
                    or any(part != "charset=utf-8" for part in parts[1:])
                ):
                    raise _fail("Zenodo JSON response has another Content-Type")
            length = response.headers.get("Content-Length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError as exc:
                    raise _fail("Zenodo response has an invalid Content-Length") from exc
                if declared < 0 or declared > max_bytes:
                    raise _fail("Zenodo response exceeds the byte limit")
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise _fail("Zenodo response exceeds the byte limit")
            if length is not None and len(body) != declared:
                raise _fail("Zenodo response byte count differs from Content-Length")
            return body

    def _json(
        self,
        url: str,
        *,
        method: str,
        authenticated: bool,
        data: bytes | None = None,
        expected_status: int = 200,
        request_content_type: str | None = None,
    ) -> Mapping[str, Any]:
        body = self._request(
            url,
            method=method,
            authenticated=authenticated,
            data=data,
            expected_status=expected_status,
            max_bytes=_MAX_JSON_RESPONSE_BYTES,
            request_content_type=request_content_type,
            expected_response_content_type="application/json",
        )
        value = _strict_json(body, label="Zenodo API response")
        if not isinstance(value, Mapping):
            raise _fail("Zenodo API response must be a JSON object")
        return value

    def get_json(self, url: str, *, authenticated: bool) -> Mapping[str, Any]:
        return self._json(url, method="GET", authenticated=authenticated)

    def get_bytes(self, url: str, *, authenticated: bool) -> bytes:
        return self._request(
            url,
            method="GET",
            authenticated=authenticated,
            data=None,
            expected_status=200,
            max_bytes=_MAX_DIRECT_RESPONSE_BYTES,
        )

    def put_bytes(self, url: str, data: bytes) -> Mapping[str, Any]:
        return self._json(
            url,
            method="PUT",
            authenticated=True,
            data=data,
            request_content_type="application/octet-stream",
        )

    def put_json(self, url: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):  # pragma: no cover - protocol type owns this
            raise _fail("Zenodo metadata PUT must contain one JSON object")
        return self._json(
            url,
            method="PUT",
            authenticated=True,
            data=_canonical_bytes(value),
            request_content_type="application/json",
        )

    def post_json(self, url: str) -> Mapping[str, Any]:
        return self._json(
            url,
            method="POST",
            authenticated=True,
            data=None,
            expected_status=202,
        )


def _stage_package(
    package: ValidatedRegistrationPackage,
    transport: _Transport,
) -> Mapping[str, Any]:
    _require_package_current(package, phase="Zenodo staging")
    metadata_request = {"metadata": _fixed_protocol_metadata()}
    metadata_response = transport.put_json(ZENODO_DRAFT_API_URI, metadata_request)
    _verify_draft(package, metadata_response, require_complete=False)
    draft = transport.get_json(ZENODO_DRAFT_API_URI, authenticated=True)
    bucket, missing = _verify_draft(package, draft, require_complete=False)
    uploaded: list[str] = []
    inventory = package.inventory
    for name in missing:
        item = inventory[name]
        response = transport.put_bytes(f"{bucket}/{quote(name, safe='')}", item.data)
        observed = _remote_file(response, label=f"Zenodo upload response for {name!r}")
        if observed != _RemoteFile(name=name, size=item.size, md5=item.md5):
            raise _fail(f"Zenodo upload response differs for {name!r}")
        uploaded.append(name)
    final_draft = transport.get_json(ZENODO_DRAFT_API_URI, authenticated=True)
    final_bucket, _ = _verify_draft(package, final_draft, require_complete=True)
    _verify_draft_bytes(package, transport, bucket=final_bucket)
    _require_package_current(package, phase="Zenodo staging")
    return {
        "c1_commit": package.c1_commit,
        "deposition_id": ZENODO_RECORD_ID,
        "file_count": len(package.files),
        "ready_to_publish": True,
        "schema_version": "fractal-zenodo-stage-v2",
        "uploaded": uploaded,
    }


def _verify_public(
    package: ValidatedRegistrationPackage,
    transport: _Transport,
) -> Mapping[str, Any]:
    _require_package_current(package, phase="anonymous Zenodo verification")
    public = transport.get_json(ZENODO_PUBLIC_API_URI, authenticated=False)
    _verify_public_payload(package, public)
    for item in package.files:
        content_uri = _public_file_uri(item.name)
        remote_bytes = transport.get_bytes(content_uri, authenticated=False)
        if (
            remote_bytes != item.data
            or len(remote_bytes) != item.size
            or hashlib.sha256(remote_bytes).hexdigest() != item.sha256
        ):
            raise _fail(f"public Zenodo file {item.name!r} differs from the C1 package bytes")
    _require_package_current(package, phase="anonymous Zenodo verification")
    return {
        "byte_verified_file_count": len(package.files),
        "deposition_id": ZENODO_RECORD_ID,
        "doi": ZENODO_RESERVED_DOI,
        "file_count": len(package.files),
        "registry_record_sha256": package.registry_record_sha256,
        "schema_version": "fractal-zenodo-public-verification-v2",
        "submitted": True,
    }


def _verify_draft_bytes(
    package: ValidatedRegistrationPackage,
    transport: _Transport,
    *,
    bucket: str,
) -> None:
    """Read every unpublished byte; the draft inventory exposes only MD5."""

    for item in package.files:
        remote_bytes = transport.get_bytes(
            f"{bucket}/{quote(item.name, safe='')}",
            authenticated=True,
        )
        if (
            remote_bytes != item.data
            or len(remote_bytes) != item.size
            or hashlib.sha256(remote_bytes).hexdigest() != item.sha256
        ):
            raise _fail(f"Zenodo draft file {item.name!r} differs from the C1 package bytes")


def _load_fixed_local_registration(
    package: ValidatedRegistrationPackage,
    *,
    registration_record_path: Path,
    registration_receipt_path: Path,
) -> tuple[ProtocolRegistryRecord, ProtocolRegistrationReceipt]:
    try:
        record = load_protocol_registry_record(registration_record_path)
        receipt = load_protocol_registration_receipt(registration_receipt_path)
    except ValueError as exc:
        raise _fail(f"local C1 registration evidence is invalid: {exc}") from exc
    if record.canonical_bytes() + b"\n" != package.registry_record_bytes:
        raise _fail("local registry-record bytes differ from the verified C1 package")
    if (
        record.registry_identity != ZENODO_REGISTRY_IDENTITY
        or record.registry_uri != ZENODO_REGISTRY_URI
        or record.record_sha256 != package.registry_record_sha256
        or record.manifest_sha256 != package.manifest_sha256
    ):
        raise _fail("local registry record differs from the fixed C1 Zenodo registration")
    shared = (
        "manifest_sha256",
        "protocol_version",
        "registered_at_utc",
        "registry_identity",
        "registry_uri",
    )
    if any(getattr(receipt, name) != getattr(record, name) for name in shared):
        raise _fail("local registration receipt differs from the fixed C1 registry record")
    if receipt.registry_record_sha256 != record.record_sha256:
        raise _fail("local registration receipt binds another registry-record byte string")
    return record, receipt


def _verify_production_protocol_registration(
    package_dir: str | Path,
    *,
    registration_record_path: str | Path,
    registration_receipt_path: str | Path,
    verifier: C1AttestationVerifier | None = None,
    transport: _Transport | None = None,
) -> VerifiedC1ProtocolRegistration:
    """Private deterministic bridge for the fixed production verifier."""

    package_path = Path(package_dir)
    record_path = Path(registration_record_path)
    receipt_path = Path(registration_receipt_path)
    package = validate_registration_package(package_path)

    def fresh_revalidator() -> None:
        refreshed = validate_registration_package(package.root)
        if refreshed != package:
            raise _fail("C1 registration package changed after capability creation")
        verify_registration_package_attestations(refreshed, verifier=verifier)
        if transport is None:
            public_transport = _ZenodoHttpsTransport(None)
            try:
                _verify_public(refreshed, public_transport)
            finally:
                public_transport.close()
        else:
            _verify_public(refreshed, transport)
        _load_fixed_local_registration(
            refreshed,
            registration_record_path=record_path,
            registration_receipt_path=receipt_path,
        )

    fresh_revalidator()
    record, receipt = _load_fixed_local_registration(
        package,
        registration_record_path=record_path,
        registration_receipt_path=receipt_path,
    )
    inventory = tuple(
        sorted(
            ((item.name, item.sha256) for item in package.files),
            key=lambda row: row[0].encode("utf-8"),
        )
    )
    return _mint_verified_c1_protocol_registration(
        record=record,
        receipt=receipt,
        package_root=package.root,
        registration_record_path=record_path,
        registration_receipt_path=receipt_path,
        c0_commit=package.c0_commit,
        c1_commit=package.c1_commit,
        package_file_sha256s=inventory,
        fresh_revalidator=fresh_revalidator,
    )


def verify_production_protocol_registration(
    package_dir: str | Path,
    *,
    registration_record_path: str | Path,
    registration_receipt_path: str | Path,
) -> VerifiedC1ProtocolRegistration:
    """Mint admission using real GitHub verification and anonymous Zenodo HTTPS.

    The returned capability retains the same verifier, so :func:`begin_sealed_run`
    repeats package, attestation, and public-byte checks immediately before it
    creates the one-shot receipt. Production callers cannot replace either trust
    boundary through this function.
    """

    return _verify_production_protocol_registration(
        package_dir,
        registration_record_path=registration_record_path,
        registration_receipt_path=registration_receipt_path,
        verifier=None,
        transport=None,
    )


def _publish_package(
    package: ValidatedRegistrationPackage,
    transport: _Transport,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    _require_package_current(package, phase="Zenodo publication")
    try:
        return _verify_public(package, transport)
    except _ZenodoHttpStatusError as exc:
        if exc.status != 404:
            raise

    draft = transport.get_json(ZENODO_DRAFT_API_URI, authenticated=True)
    bucket, _ = _verify_draft(package, draft, require_complete=True)
    _verify_draft_bytes(package, transport, bucket=bucket)
    post_error: ZenodoPublicationError | None = None
    try:
        published = transport.post_json(ZENODO_PUBLISH_API_URI)
        _verify_publish_response(package, published)
    except ZenodoPublicationError as exc:
        post_error = exc
    for attempt in range(_PUBLICATION_POLL_ATTEMPTS):
        try:
            return _verify_public(package, transport)
        except _ZenodoHttpStatusError as exc:
            if exc.status not in {404, 409, 503}:
                raise
            if attempt + 1 == _PUBLICATION_POLL_ATTEMPTS:
                if post_error is not None:
                    raise post_error from exc
                raise
            sleep(_PUBLICATION_POLL_SECONDS)
    if post_error is not None:  # pragma: no cover - bounded loop returns or raises
        raise post_error
    raise _fail("Zenodo publication polling exhausted without a public record")


def _read_token_fd(descriptor: int) -> bytearray:
    if type(descriptor) is not int or descriptor < 0:
        raise _fail("token file descriptor must be a non-negative integer")
    chunks: list[bytes] = []
    observed = 0
    while True:
        try:
            chunk = os.read(descriptor, min(1024, _MAX_TOKEN_BYTES + 2 - observed))
        except OSError as exc:
            raise _fail("cannot read the Zenodo token file descriptor") from exc
        if not chunk:
            break
        chunks.append(chunk)
        observed += len(chunk)
        if observed > _MAX_TOKEN_BYTES + 1:
            raise _fail("Zenodo token exceeds the byte limit")
    encoded = b"".join(chunks)
    if encoded.endswith(b"\n"):
        encoded = encoded[:-1]
    if _TOKEN.fullmatch(encoded) is None:
        raise _fail("Zenodo token has an invalid bounded bearer-token form")
    return bytearray(encoded)


def _summary(package: ValidatedRegistrationPackage) -> Mapping[str, Any]:
    return {
        "c0_commit": package.c0_commit,
        "c1_commit": package.c1_commit,
        "deposition_id": ZENODO_RECORD_ID,
        "file_count": len(package.files),
        "manifest_sha256": package.manifest_sha256,
        "registry_record_sha256": package.registry_record_sha256,
        "schema_version": "fractal-zenodo-package-validation-v2",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-zenodo-publication",
        description="Validate, stage, publish, or verify the fixed C1 Zenodo package.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate the closed package offline")
    validate.add_argument("--package", required=True, type=Path)
    preflight = commands.add_parser(
        "preflight",
        help="validate the package and freshly verify both GitHub attestations",
    )
    preflight.add_argument("--package", required=True, type=Path)
    stage = commands.add_parser("stage", help="upload missing files without publishing")
    stage.add_argument("--package", required=True, type=Path)
    stage.add_argument("--token-fd", type=int, default=0)
    publish = commands.add_parser("publish", help="publish an already complete fixed draft")
    publish.add_argument("--package", required=True, type=Path)
    publish.add_argument("--token-fd", type=int, default=0)
    publish.add_argument("--confirm-record", required=True, type=int)
    verify = commands.add_parser("verify-public", help="verify the public record anonymously")
    verify.add_argument("--package", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    transport: _ZenodoHttpsTransport | None = None
    token: bytearray | None = None
    try:
        package = validate_registration_package(arguments.package)
        if arguments.command == "validate":
            result = _summary(package)
        else:
            if arguments.command == "publish" and arguments.confirm_record != ZENODO_RECORD_ID:
                raise _fail(
                    f"publish requires --confirm-record {ZENODO_RECORD_ID} for the fixed draft"
                )
            attestation_result = verify_registration_package_attestations(package)
            if arguments.command == "preflight":
                result = attestation_result
            elif arguments.command == "verify-public":
                transport = _ZenodoHttpsTransport(None)
                result = _verify_public(package, transport)
            else:
                token = _read_token_fd(arguments.token_fd)
                transport = _ZenodoHttpsTransport(token)
                token = None
                if arguments.command == "stage":
                    result = _stage_package(package, transport)
                else:
                    result = _publish_package(package, transport)
        sys.stdout.buffer.write(_canonical_bytes(result) + b"\n")
        return 0
    except (SuiteAttemptError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if transport is not None:
            transport.close()
        if token is not None:
            for position in range(len(token)):
                token[position] = 0


if __name__ == "__main__":
    raise SystemExit(main())
