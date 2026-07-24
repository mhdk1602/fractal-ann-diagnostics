"""Fail-closed transport for GitHub Actions artifact evidence.

GitHub's artifact endpoint is deliberately treated as an untrusted transport.
The verifier reads the repository, workflow, run, run-artifact inventory, and
individual artifact separately, binds all of them to an already closed claim,
then verifies the ``upload-artifact`` archive digest before materializing a
closed archive inventory.  It performs no GitHub mutation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

GITHUB_REST_API_VERSION = "2026-03-10"
GITHUB_API_URL = "https://api.github.com/"
REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
REPOSITORY_ID = 1_239_189_910
REPOSITORY_NODE_ID = "R_kgDOSdyJlg"
OWNER_LOGIN = "mhdk1602"
OWNER_ID = 9_646_005
OWNER_NODE_ID = "MDQ6VXNlcjk2NDYwMDU="
OWNER_EMAIL = "mhdk1602@users.noreply.github.com"
C0_REF = "refs/tags/confirmatory-apparatus-c0"
C0_HEAD_BRANCH = "confirmatory-apparatus-c0"
ADMITTED_WORKFLOW_PATHS = frozenset(
    {
        ".github/workflows/confirmatory-image.yml",
        ".github/workflows/confirmatory-online-execution.yml",
        ".github/workflows/confirmatory-label-release.yml",
        ".github/workflows/confirmatory-analysis.yml",
    }
)
EXECUTION_ARTIFACT_CLAIM_SCHEMA = "fractal-github-artifact-execution-claim-v1"
EXECUTION_ARTIFACT_RECEIPT_SCHEMA = "fractal-github-artifact-receipt-v1"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
MAX_ARCHIVE_MEMBERS = 4096
MAX_INVENTORY_BYTES = 8 * 1024 * 1024
CLAIM_PACKAGE_INVENTORY_PATH = "claim-package.SHA256SUMS"

_PROVIDER_CLAIM_WORKFLOWS = {
    "online": ".github/workflows/confirmatory-online-execution.yml",
    "label-release": ".github/workflows/confirmatory-label-release.yml",
    "analysis": ".github/workflows/confirmatory-analysis.yml",
}

_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA256SUM_LINE = re.compile(rb"([0-9a-f]{64})  ([^\r\n]+)\n")
_CAPABILITY = object()


class GitHubArtifactTransportError(ValueError):
    """GitHub artifact evidence is absent, malformed, or differs from the claim."""


def _timestamp(value: object, *, label: str) -> datetime:
    text = _text(value, label=label)
    if not text.endswith("Z"):
        raise GitHubArtifactTransportError(f"{label} must be a UTC RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GitHubArtifactTransportError(f"{label} is not an RFC 3339 timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise GitHubArtifactTransportError(f"{label} must use UTC Z")
    return parsed


@dataclass(frozen=True)
class GitHubHttpResponse:
    """The only response shape the verifier accepts from its injected transport."""

    status: int
    headers: Mapping[str, str]
    body: bytes

    def __post_init__(self) -> None:
        if type(self.status) is not int or not 100 <= self.status <= 599:
            raise GitHubArtifactTransportError("HTTP response status is invalid")
        if not isinstance(self.headers, Mapping) or any(
            type(key) is not str or type(value) is not str for key, value in self.headers.items()
        ):
            raise GitHubArtifactTransportError("HTTP response headers are malformed")
        if not isinstance(self.body, bytes):
            raise GitHubArtifactTransportError("HTTP response body must be bytes")


class GitHubArtifactReadApi(Protocol):
    """Read-only transport; ``location`` may be a relative API path or HTTPS URL."""

    def get(self, location: str, *, accept: str) -> GitHubHttpResponse: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: object, code: int, msg: str, headers: object, newurl: str
    ) -> None:  # type: ignore[override]
        return None


class UrllibGitHubArtifactReadApi:
    """Small read-only GitHub REST client with the current REST API version pinned."""

    def __init__(self, token: str, *, api_url: str = GITHUB_API_URL) -> None:
        if type(token) is not str or not token.strip():
            raise GitHubArtifactTransportError("GitHub token must be a non-empty string")
        parsed = urlsplit(api_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
            raise GitHubArtifactTransportError("GitHub API URL must be an HTTPS origin")
        self._token = token
        self._api_url = api_url.rstrip("/") + "/"
        self._api_origin = urlsplit(self._api_url).netloc
        self._opener = build_opener(_NoRedirect())

    def get(self, location: str, *, accept: str) -> GitHubHttpResponse:
        if not isinstance(location, str) or not location:
            raise GitHubArtifactTransportError("GitHub request location is invalid")
        parsed = urlsplit(location)
        if parsed.scheme:
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise GitHubArtifactTransportError(
                    "artifact redirect is not a credential-free HTTPS URL"
                )
            url = location
        else:
            if location.startswith("/"):
                raise GitHubArtifactTransportError("GitHub API path must be relative")
            url = urljoin(self._api_url, location)
        headers = {
            "Accept": accept,
            "User-Agent": "fractal-ann-diagnostics-artifact-verifier/1",
            "X-GitHub-Api-Version": GITHUB_REST_API_VERSION,
        }
        # Never send the GitHub bearer token to a signed artifact-download host.
        if urlsplit(url).netloc == self._api_origin:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(url, headers=headers, method="GET")
        try:
            response = self._opener.open(request, timeout=30)
        except HTTPError as exc:
            return self._bounded_response(exc.code, dict(exc.headers.items()), exc)
        except OSError as exc:
            raise GitHubArtifactTransportError("GitHub artifact transport failed") from exc
        with response:
            return self._bounded_response(response.status, dict(response.headers.items()), response)

    @staticmethod
    def _bounded_response(
        status: int, headers: Mapping[str, str], source: Any
    ) -> GitHubHttpResponse:
        content_length = headers.get("Content-Length") or headers.get("content-length")
        if content_length is not None:
            if not content_length.isascii() or not content_length.isdigit():
                raise GitHubArtifactTransportError("GitHub response Content-Length is malformed")
            if int(content_length) > MAX_ARCHIVE_BYTES:
                raise GitHubArtifactTransportError("GitHub response exceeds the byte limit")
        body = source.read(MAX_ARCHIVE_BYTES + 1)
        if len(body) > MAX_ARCHIVE_BYTES:
            raise GitHubArtifactTransportError("GitHub response exceeds the byte limit")
        return GitHubHttpResponse(status, headers, body)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise GitHubArtifactTransportError("receipt cannot be rendered as canonical JSON") from exc


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise GitHubArtifactTransportError(f"{label} must be a canonical non-empty string")
    return value


def _positive(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise GitHubArtifactTransportError(f"{label} must be a positive integer")
    return value


def _sha1(value: object, *, label: str) -> str:
    value = _text(value, label=label)
    if _SHA1.fullmatch(value) is None:
        raise GitHubArtifactTransportError(f"{label} must be a lowercase Git SHA-1")
    return value


def _digest(value: object, *, label: str) -> str:
    value = _text(value, label=label)
    if _SHA256.fullmatch(value) is None:
        raise GitHubArtifactTransportError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _artifact_digest(value: object, *, label: str) -> str:
    value = _text(value, label=label)
    if not value.startswith("sha256:") or _SHA256.fullmatch(value[7:]) is None:
        raise GitHubArtifactTransportError(f"{label} must be an upload-artifact SHA-256 digest")
    return value


def _json_object(response: GitHubHttpResponse, *, label: str) -> Mapping[str, Any]:
    if response.status != 200:
        raise GitHubArtifactTransportError(f"{label} returned HTTP {response.status}")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in rows:
            if key in result:
                raise GitHubArtifactTransportError(f"{label} repeats JSON key {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise GitHubArtifactTransportError(f"{label} contains non-finite JSON number {value!r}")

    try:
        value = json.loads(
            response.body.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitHubArtifactTransportError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise GitHubArtifactTransportError(f"{label} must be a JSON object")
    return value


def _canonical_member_path(value: object, *, label: str) -> str:
    name = _text(value, label=label)
    if name != unicodedata.normalize("NFC", name) or "\\" in name or "\x00" in name:
        raise GitHubArtifactTransportError(f"{label} is not a canonical archive path")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or str(path) != name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise GitHubArtifactTransportError(f"{label} escapes the archive root")
    if name.endswith("/") or any(
        ord(character) < 32 or ord(character) == 127 for character in name
    ):
        raise GitHubArtifactTransportError(f"{label} is not a regular-file archive path")
    return name


@dataclass(frozen=True)
class ArchiveMember:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path", _canonical_member_path(self.path, label="archive member path")
        )
        object.__setattr__(self, "sha256", _digest(self.sha256, label="archive member SHA-256"))
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= MAX_MEMBER_BYTES:
            raise GitHubArtifactTransportError("archive member size is out of bounds")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class _FixedClaimArtifactContext:
    phase: str
    suite_attempt_id: str
    workflow_path: str
    run_id: int
    head_sha: str
    artifact_name: str


def _fixed_claim_artifact_context(
    workflow_context: object, suite_attempt_id: object
) -> _FixedClaimArtifactContext:
    """Close the subset of the live provider context that GitHub can re-prove.

    ``ProviderWorkflowContext`` lives in the orchestration layer and cannot be
    imported here without creating a dependency cycle.  This function accepts
    that typed object structurally, then rejects every non-production field.
    The returned value is an expectation, not an authority: the REST evidence
    is re-read before a receipt capability is minted.
    """

    def field(name: str) -> object:
        try:
            return getattr(workflow_context, name)
        except (AttributeError, TypeError) as exc:
            raise GitHubArtifactTransportError(
                f"fixed GitHub workflow context omits {name}"
            ) from exc

    phase = field("phase")
    if type(phase) is not str or phase not in _PROVIDER_CLAIM_WORKFLOWS:
        raise GitHubArtifactTransportError("fixed GitHub workflow phase is not admitted")
    suite = _digest(suite_attempt_id, label="suite_attempt_id")
    workflow_path = _PROVIDER_CLAIM_WORKFLOWS[phase]
    workflow_ref = f"{REPOSITORY}/{workflow_path}@{C0_REF}"
    exact = {
        "job": "execute",
        "repository": REPOSITORY,
        "repository_id": REPOSITORY_ID,
        "repository_owner": OWNER_LOGIN,
        "repository_owner_id": OWNER_ID,
        "actor": OWNER_LOGIN,
        "actor_id": OWNER_ID,
        "triggering_actor": OWNER_LOGIN,
        "workflow_path": workflow_path,
        "workflow_ref": workflow_ref,
        "github_ref": C0_REF,
        "github_ref_name": C0_HEAD_BRANCH,
        "github_ref_type": "tag",
        "github_ref_protected": True,
        "run_attempt": 1,
        "event_name": "workflow_dispatch",
        "runner_environment": "self-hosted",
        "runner_os": "macOS",
        "runner_arch": "ARM64",
    }
    for name, expected in exact.items():
        observed = field(name)
        if observed != expected or type(observed) is not type(expected):
            raise GitHubArtifactTransportError(
                f"fixed GitHub workflow context {name} differs from production"
            )
    run_id = _positive(field("run_id"), label="run_id")
    workflow_sha = _sha1(field("workflow_sha"), label="workflow_sha")
    github_sha = _sha1(field("github_sha"), label="github_sha")
    if workflow_sha != github_sha:
        raise GitHubArtifactTransportError(
            "fixed GitHub workflow source and dispatch commits differ"
        )
    return _FixedClaimArtifactContext(
        phase=phase,
        suite_attempt_id=suite,
        workflow_path=workflow_path,
        run_id=run_id,
        head_sha=github_sha,
        artifact_name=f"confirmatory-{phase}-claim-{suite}-{run_id}",
    )


@dataclass(frozen=True)
class ExecutionArtifactClaim:
    """Closed expected GitHub identity and closed expected ZIP inventory."""

    repository: str
    repository_id: int
    repository_node_id: str
    owner_id: int
    owner_login: str
    workflow_path: str
    workflow_id: int
    workflow_ref: str
    run_id: int
    head_sha: str
    head_branch: str | None
    actor_id: int
    actor_login: str
    conclusion: str
    artifact_id: int
    artifact_node_id: str
    artifact_name: str
    artifact_digest: str
    artifact_size_bytes: int
    artifact_created_at: str
    artifact_expires_at: str
    inventory: tuple[ArchiveMember, ...]
    run_attempt: int = 1
    event: str = "workflow_dispatch"
    status: str = "completed"
    schema_version: str = EXECUTION_ARTIFACT_CLAIM_SCHEMA

    def __post_init__(self) -> None:
        if self.repository != REPOSITORY:
            raise GitHubArtifactTransportError(
                "repository differs from the fixed production repository"
            )
        if self.repository_id != REPOSITORY_ID or self.repository_node_id != REPOSITORY_NODE_ID:
            raise GitHubArtifactTransportError(
                "repository identity differs from the fixed production repository"
            )
        if self.owner_id != OWNER_ID or self.owner_login != OWNER_LOGIN:
            raise GitHubArtifactTransportError(
                "repository owner differs from the fixed production owner"
            )
        if self.actor_id != OWNER_ID or self.actor_login != OWNER_LOGIN:
            raise GitHubArtifactTransportError("run actor differs from the fixed production owner")
        for name in ("repository_id", "owner_id", "workflow_id", "run_id", "artifact_id"):
            _positive(getattr(self, name), label=name)
        for name in (
            "repository_node_id",
            "owner_login",
            "workflow_path",
            "workflow_ref",
            "actor_login",
            "artifact_node_id",
            "artifact_name",
            "artifact_created_at",
            "artifact_expires_at",
            "conclusion",
        ):
            _text(getattr(self, name), label=name)
        if self.workflow_path not in ADMITTED_WORKFLOW_PATHS:
            raise GitHubArtifactTransportError(
                "workflow_path is not admitted for production evidence"
            )
        if self.workflow_ref != C0_REF:
            raise GitHubArtifactTransportError("workflow_ref differs from the fixed C0 tag")
        if self.head_branch != C0_HEAD_BRANCH:
            raise GitHubArtifactTransportError("head_branch differs from the fixed C0 branch")
        _sha1(self.head_sha, label="head_sha")
        _positive(self.actor_id, label="actor_id")
        if self.head_branch is not None:
            _text(self.head_branch, label="head_branch")
        _artifact_digest(self.artifact_digest, label="artifact_digest")
        if (
            type(self.artifact_size_bytes) is not int
            or not 0 <= self.artifact_size_bytes <= MAX_ARCHIVE_BYTES
        ):
            raise GitHubArtifactTransportError("artifact_size_bytes is out of bounds")
        created = _timestamp(self.artifact_created_at, label="artifact_created_at")
        expires = _timestamp(self.artifact_expires_at, label="artifact_expires_at")
        if created >= expires:
            raise GitHubArtifactTransportError("artifact expires_at must follow created_at")
        if (
            self.run_attempt != 1
            or self.event != "workflow_dispatch"
            or self.status != "completed"
            or self.conclusion != "success"
        ):
            raise GitHubArtifactTransportError(
                "claim admits only successful completed dispatch attempt 1"
            )
        if self.schema_version != EXECUTION_ARTIFACT_CLAIM_SCHEMA:
            raise GitHubArtifactTransportError("execution artifact claim schema differs")
        rows = tuple(self.inventory)
        if not rows or not all(isinstance(row, ArchiveMember) for row in rows):
            raise GitHubArtifactTransportError("artifact inventory must be non-empty typed members")
        names = [row.path for row in rows]
        aliases = [unicodedata.normalize("NFC", name).casefold() for name in names]
        if names != sorted(names, key=lambda name: name.encode("utf-8")) or len(names) != len(
            set(names)
        ):
            raise GitHubArtifactTransportError(
                "artifact inventory paths must be unique UTF-8-byte sorted"
            )
        if len(aliases) != len(set(aliases)):
            raise GitHubArtifactTransportError("artifact inventory has case or Unicode aliases")
        object.__setattr__(self, "inventory", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "inventory"
            },
            "inventory": [row.to_dict() for row in self.inventory],
        }

    @property
    def claim_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @property
    def head_ref(self) -> str:
        """The ref encoded by GitHub's ``path`` field for this workflow run."""

        return self.workflow_ref


@dataclass(frozen=True)
class VerifiedExecutionArtifactReceipt:
    """Capability minted only after all GitHub and ZIP evidence has been cross-bound."""

    claim_sha256: str
    repository: str
    repository_id: int
    workflow_path: str
    workflow_id: int
    workflow_ref: str
    run_id: int
    run_attempt: int
    event: str
    head_sha: str
    head_branch: str | None
    actor_id: int
    actor_login: str
    status: str
    conclusion: str
    artifact_id: int
    artifact_node_id: str
    artifact_name: str
    artifact_digest: str
    artifact_size_bytes: int
    inventory: tuple[ArchiveMember, ...]
    inventory_sha256: str
    archive_sha256: str
    materialized_root: str
    schema_version: str = EXECUTION_ARTIFACT_RECEIPT_SCHEMA
    _capability: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _CAPABILITY:
            raise GitHubArtifactTransportError(
                "artifact receipt capability was not minted by verifier"
            )
        _digest(self.claim_sha256, label="claim_sha256")
        _digest(self.inventory_sha256, label="inventory_sha256")
        _digest(self.archive_sha256, label="archive_sha256")
        _artifact_digest(self.artifact_digest, label="artifact_digest")
        _positive(self.actor_id, label="actor_id")
        _text(self.actor_login, label="actor_login")
        _text(self.artifact_node_id, label="artifact_node_id")
        if type(self.materialized_root) is not str or not self.materialized_root:
            raise GitHubArtifactTransportError("materialized_root is invalid")
        if self.schema_version != EXECUTION_ARTIFACT_RECEIPT_SCHEMA:
            raise GitHubArtifactTransportError("artifact receipt schema differs")
        object.__setattr__(self, "inventory", tuple(self.inventory))

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"inventory", "_capability"}
            },
            "inventory": [row.to_dict() for row in self.inventory],
        }

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @property
    def head_ref(self) -> str:
        """The ref encoded by GitHub's ``path`` field for this workflow run."""

        return self.workflow_ref


def require_execution_artifact_capability(value: object) -> VerifiedExecutionArtifactReceipt:
    if (
        not isinstance(value, VerifiedExecutionArtifactReceipt)
        or value._capability is not _CAPABILITY
    ):
        raise GitHubArtifactTransportError("a verified execution artifact capability is required")
    return value


def _response(api: GitHubArtifactReadApi, location: str, *, accept: str) -> GitHubHttpResponse:
    try:
        result = api.get(location, accept=accept)
    except Exception as exc:
        raise GitHubArtifactTransportError(f"cannot read GitHub evidence at {location!r}") from exc
    if not isinstance(result, GitHubHttpResponse):
        raise GitHubArtifactTransportError("GitHub transport returned an untyped response")
    return result


def _exact(row: Mapping[str, Any], expected: Mapping[str, object], *, label: str) -> None:
    for field_name, value in expected.items():
        observed = row.get(field_name)
        if observed != value or type(observed) is not type(value):
            raise GitHubArtifactTransportError(f"{label} {field_name} differs from the claim")


def _artifact_fields(row: Mapping[str, Any], claim: ExecutionArtifactClaim, *, label: str) -> None:
    _exact(
        row,
        {
            "id": claim.artifact_id,
            "node_id": claim.artifact_node_id,
            "name": claim.artifact_name,
            "digest": claim.artifact_digest,
            "size_in_bytes": claim.artifact_size_bytes,
            "created_at": claim.artifact_created_at,
            "expires_at": claim.artifact_expires_at,
            "expired": False,
        },
        label=label,
    )
    workflow_run = row.get("workflow_run")
    if not isinstance(workflow_run, Mapping):
        raise GitHubArtifactTransportError(f"{label} workflow_run is required and malformed")
    _exact(
        workflow_run,
        {
            "id": claim.run_id,
            "head_sha": claim.head_sha,
            "head_branch": claim.head_branch,
            "repository_id": claim.repository_id,
            "head_repository_id": claim.repository_id,
        },
        label=f"{label} workflow_run",
    )
    expected_download_url = (
        f"https://api.github.com/repos/{claim.repository}/actions/artifacts/{claim.artifact_id}/zip"
    )
    if row.get("archive_download_url") != expected_download_url:
        raise GitHubArtifactTransportError(
            f"{label} archive download URL differs from the artifact ID"
        )


def _verify_c0_tag(api: GitHubArtifactReadApi, claim: ExecutionArtifactClaim) -> None:
    tag_name = C0_REF.removeprefix("refs/tags/")
    tag_ref = _json_object(
        _response(
            api,
            f"repos/{claim.repository}/git/ref/tags/{tag_name}",
            accept="application/vnd.github+json",
        ),
        label="C0 tag reference",
    )
    _exact(tag_ref, {"ref": C0_REF}, label="C0 tag reference")
    target = tag_ref.get("object")
    if not isinstance(target, Mapping):
        raise GitHubArtifactTransportError("C0 tag reference object is malformed")
    if target.get("type") != "tag":
        raise GitHubArtifactTransportError("C0 reference must resolve to an annotated Git tag")
    tag_oid = _sha1(target.get("sha"), label="C0 annotated tag object ID")
    tag = _json_object(
        _response(
            api,
            f"repos/{claim.repository}/git/tags/{tag_oid}",
            accept="application/vnd.github+json",
        ),
        label="C0 annotated tag",
    )
    _exact(
        tag,
        {
            "tag": tag_name,
            "object": {"type": "commit", "sha": claim.head_sha},
            "tagger": {"name": OWNER_LOGIN, "email": OWNER_EMAIL},
        },
        label="C0 annotated tag",
    )
    readback = _json_object(
        _response(
            api,
            f"repos/{claim.repository}/git/ref/tags/{tag_name}",
            accept="application/vnd.github+json",
        ),
        label="C0 tag reference readback",
    )
    _exact(
        readback,
        {"ref": C0_REF, "object": {"type": "tag", "sha": tag_oid}},
        label="C0 tag reference readback",
    )


def _download_archive_exact(
    api: GitHubArtifactReadApi,
    *,
    artifact_id: int,
    artifact_digest: str,
    artifact_size_bytes: int,
) -> bytes:
    artifact_id = _positive(artifact_id, label="artifact_id")
    artifact_digest = _artifact_digest(artifact_digest, label="artifact_digest")
    if type(artifact_size_bytes) is not int or not 0 <= artifact_size_bytes <= MAX_ARCHIVE_BYTES:
        raise GitHubArtifactTransportError("artifact_size_bytes is out of bounds")
    location = f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip"
    response = _response(api, location, accept="application/vnd.github+json")
    if response.status == 410:
        raise GitHubArtifactTransportError("GitHub artifact was deleted or expired before download")
    if response.status == 302:
        target = response.headers.get("Location") or response.headers.get("location")
        parsed = urlsplit(target or "")
        if (
            not target
            or parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise GitHubArtifactTransportError("artifact redirect is malformed")
        response = _response(api, target, accept="application/octet-stream")
    if response.status == 410:
        raise GitHubArtifactTransportError("GitHub artifact expired during download")
    if response.status != 200:
        raise GitHubArtifactTransportError(f"artifact download returned HTTP {response.status}")
    if len(response.body) > MAX_ARCHIVE_BYTES:
        raise GitHubArtifactTransportError("downloaded archive exceeds the byte limit")
    observed = _sha256(response.body)
    if observed != artifact_digest[7:]:
        raise GitHubArtifactTransportError(
            "downloaded archive differs from immutable upload-artifact digest"
        )
    if len(response.body) != artifact_size_bytes:
        raise GitHubArtifactTransportError("downloaded archive size differs from artifact metadata")
    return response.body


def _download_archive(api: GitHubArtifactReadApi, claim: ExecutionArtifactClaim) -> bytes:
    return _download_archive_exact(
        api,
        artifact_id=claim.artifact_id,
        artifact_digest=claim.artifact_digest,
        artifact_size_bytes=claim.artifact_size_bytes,
    )


def _safe_zip_infos(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    infos = tuple(archive.infolist())
    if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
        raise GitHubArtifactTransportError("ZIP member count is out of bounds")
    actual_names: set[str] = set()
    aliases: set[str] = set()
    total = 0
    for info in infos:
        name = _canonical_member_path(info.filename, label="ZIP member name")
        if info.flag_bits & 0x1:
            raise GitHubArtifactTransportError("encrypted ZIP members are forbidden")
        mode = (info.external_attr >> 16) & 0o177777
        file_type = stat.S_IFMT(mode)
        if stat.S_ISLNK(mode) or (file_type and file_type != stat.S_IFREG):
            raise GitHubArtifactTransportError("ZIP member is not a regular file")
        if info.file_size < 0 or info.compress_size < 0 or info.file_size > MAX_MEMBER_BYTES:
            raise GitHubArtifactTransportError("ZIP member size is out of bounds")
        if info.file_size and not info.compress_size:
            raise GitHubArtifactTransportError("ZIP member has an invalid compression size")
        if info.compress_size and info.file_size > info.compress_size * MAX_COMPRESSION_RATIO:
            raise GitHubArtifactTransportError("ZIP member compression ratio is unsafe")
        total += info.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise GitHubArtifactTransportError("ZIP uncompressed size exceeds the byte limit")
        alias = unicodedata.normalize("NFC", name).casefold()
        if name in actual_names or alias in aliases:
            raise GitHubArtifactTransportError(
                "ZIP has duplicate, case, or Unicode-alias member names"
            )
        actual_names.add(name)
        aliases.add(alias)
    return infos


def _validate_zip_infos(
    archive: zipfile.ZipFile, claim: ExecutionArtifactClaim
) -> tuple[zipfile.ZipInfo, ...]:
    infos = _safe_zip_infos(archive)
    expected = {row.path: row for row in claim.inventory}
    actual_names: set[str] = set()
    for info in infos:
        name = info.filename
        actual_names.add(name)
        if name not in expected:
            raise GitHubArtifactTransportError("ZIP contains an unclaimed extra member")
        if info.file_size != expected[name].size_bytes:
            raise GitHubArtifactTransportError("ZIP member size differs from closed inventory")
    if actual_names != set(expected):
        raise GitHubArtifactTransportError("ZIP inventory omits or substitutes a claimed member")
    return infos


def _read_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    byte_limit: int,
    retain: bool,
) -> tuple[str, int, bytes | None]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    observed = 0
    try:
        with archive.open(info, "r") as source:
            while chunk := source.read(min(1024 * 1024, byte_limit + 1 - observed)):
                observed += len(chunk)
                if observed > byte_limit or observed > info.file_size:
                    raise GitHubArtifactTransportError(
                        f"ZIP member {info.filename!r} exceeds its declared bound"
                    )
                digest.update(chunk)
                if retain:
                    chunks.append(chunk)
    except (RuntimeError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise GitHubArtifactTransportError(
            f"cannot read ZIP member {info.filename!r} exactly"
        ) from exc
    if observed != info.file_size:
        raise GitHubArtifactTransportError(
            f"ZIP member {info.filename!r} differs from its declared size"
        )
    return digest.hexdigest(), observed, b"".join(chunks) if retain else None


def _parse_claim_package_inventory(encoded: bytes) -> tuple[tuple[str, str], ...]:
    if not encoded or len(encoded) > MAX_INVENTORY_BYTES or not encoded.endswith(b"\n"):
        raise GitHubArtifactTransportError(
            "claim-package.SHA256SUMS is empty, oversized, or not newline terminated"
        )
    rows: list[tuple[str, str]] = []
    offset = 0
    for match in _SHA256SUM_LINE.finditer(encoded):
        if match.start() != offset:
            raise GitHubArtifactTransportError(
                "claim-package.SHA256SUMS is not strict GNU SHA256SUMS syntax"
            )
        offset = match.end()
        try:
            path_text = match.group(2).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise GitHubArtifactTransportError(
                "claim-package.SHA256SUMS path is not UTF-8"
            ) from exc
        path = _canonical_member_path(path_text, label="claim package inventory path")
        if path == CLAIM_PACKAGE_INVENTORY_PATH:
            raise GitHubArtifactTransportError(
                "claim-package.SHA256SUMS cannot contain a self-referential row"
            )
        rows.append((path, match.group(1).decode("ascii")))
    if offset != len(encoded) or not rows:
        raise GitHubArtifactTransportError(
            "claim-package.SHA256SUMS is malformed or has no member rows"
        )
    paths = [path for path, _ in rows]
    aliases = [unicodedata.normalize("NFC", path).casefold() for path in paths]
    if paths != sorted(paths, key=lambda path: path.encode("utf-8")):
        raise GitHubArtifactTransportError("claim-package.SHA256SUMS paths are not bytewise sorted")
    if len(paths) != len(set(paths)) or len(aliases) != len(set(aliases)):
        raise GitHubArtifactTransportError(
            "claim-package.SHA256SUMS has duplicate, case, or Unicode-alias paths"
        )
    return tuple(rows)


def _derive_closed_inventory(
    archive_bytes: bytes, *, expected_inventory_sha256: str
) -> tuple[ArchiveMember, ...]:
    expected_inventory_sha256 = _digest(
        expected_inventory_sha256,
        label="expected claim package inventory SHA-256",
    )
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(archive_bytes), "r") as archive:
            infos = _safe_zip_infos(archive)
            by_name = {info.filename: info for info in infos}
            inventory_info = by_name.get(CLAIM_PACKAGE_INVENTORY_PATH)
            if inventory_info is None:
                raise GitHubArtifactTransportError("ZIP omits claim-package.SHA256SUMS")
            if inventory_info.file_size > MAX_INVENTORY_BYTES:
                raise GitHubArtifactTransportError(
                    "claim-package.SHA256SUMS exceeds the byte limit"
                )
            inventory_digest, inventory_size, retained = _read_zip_member(
                archive,
                inventory_info,
                byte_limit=MAX_INVENTORY_BYTES,
                retain=True,
            )
            if retained is None:  # pragma: no cover - guarded by retain=True
                raise GitHubArtifactTransportError("claim package inventory was not retained")
            if inventory_digest != expected_inventory_sha256:
                raise GitHubArtifactTransportError(
                    "claim-package.SHA256SUMS differs from the trusted inventory digest"
                )
            declared_rows = _parse_claim_package_inventory(retained)
            declared = dict(declared_rows)
            expected_names = set(declared) | {CLAIM_PACKAGE_INVENTORY_PATH}
            if set(by_name) != expected_names:
                extras = sorted(set(by_name) - expected_names)
                missing = sorted(expected_names - set(by_name))
                raise GitHubArtifactTransportError(
                    "ZIP members differ from claim-package.SHA256SUMS; "
                    f"missing={missing}, unexpected={extras}"
                )
            members: list[ArchiveMember] = [
                ArchiveMember(
                    path=CLAIM_PACKAGE_INVENTORY_PATH,
                    sha256=inventory_digest,
                    size_bytes=inventory_size,
                )
            ]
            for path, declared_digest in declared_rows:
                observed_digest, observed_size, _ = _read_zip_member(
                    archive,
                    by_name[path],
                    byte_limit=MAX_MEMBER_BYTES,
                    retain=False,
                )
                if observed_digest != declared_digest:
                    raise GitHubArtifactTransportError(
                        f"ZIP member {path!r} differs from claim-package.SHA256SUMS"
                    )
                members.append(
                    ArchiveMember(
                        path=path,
                        sha256=observed_digest,
                        size_bytes=observed_size,
                    )
                )
    except zipfile.BadZipFile as exc:
        raise GitHubArtifactTransportError(
            "downloaded artifact is not a valid ZIP archive"
        ) from exc
    return tuple(sorted(members, key=lambda row: row.path.encode("utf-8")))


def _materialize_archive(
    archive_bytes: bytes, claim: ExecutionArtifactClaim, destination: Path
) -> tuple[ArchiveMember, ...]:
    if not destination.is_absolute():
        raise GitHubArtifactTransportError(
            "artifact destination must be an absolute controlled path"
        )
    if destination.exists():
        raise GitHubArtifactTransportError("artifact destination must not already exist")
    try:
        parent_status = destination.parent.lstat()
    except OSError as exc:
        raise GitHubArtifactTransportError("artifact destination parent does not exist") from exc
    if (
        stat.S_ISLNK(parent_status.st_mode)
        or not stat.S_ISDIR(parent_status.st_mode)
        or parent_status.st_uid != os.getuid()
        or parent_status.st_mode & 0o022
    ):
        raise GitHubArtifactTransportError(
            "artifact destination parent is not a controlled directory"
        )
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(archive_bytes), "r") as archive:
            infos = _validate_zip_infos(archive, claim)
            staging = Path(tempfile.mkdtemp(prefix=".artifact-", dir=destination.parent))
            try:
                expected = {row.path: row for row in claim.inventory}
                verified: list[ArchiveMember] = []
                for info in infos:
                    item = expected[info.filename]
                    output = staging.joinpath(*PurePosixPath(info.filename).parts)
                    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    os.chmod(output.parent, 0o700)
                    digest = hashlib.sha256()
                    observed = 0
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                    descriptor = os.open(output, flags, 0o600)
                    with archive.open(info, "r") as source, os.fdopen(descriptor, "wb") as target:
                        while chunk := source.read(1024 * 1024):
                            observed += len(chunk)
                            if observed > item.size_bytes:
                                raise GitHubArtifactTransportError(
                                    "ZIP member exceeds its declared size"
                                )
                            digest.update(chunk)
                            target.write(chunk)
                    if observed != item.size_bytes or digest.hexdigest() != item.sha256:
                        raise GitHubArtifactTransportError(
                            "ZIP member digest differs from closed inventory"
                        )
                    os.chmod(output, 0o600)
                    verified.append(item)
                os.replace(staging, destination)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
    except zipfile.BadZipFile as exc:
        raise GitHubArtifactTransportError(
            "downloaded artifact is not a valid ZIP archive"
        ) from exc
    return tuple(sorted(verified, key=lambda row: row.path.encode("utf-8")))


class GitHubArtifactTransport:
    """Read, cross-bind, and safely materialize one claimed immutable Actions artifact."""

    def __init__(self, api: GitHubArtifactReadApi) -> None:
        self._api = api

    def _verify_remote_claim(self, claim: ExecutionArtifactClaim) -> None:
        if not isinstance(claim, ExecutionArtifactClaim):
            raise GitHubArtifactTransportError(
                "execution artifact verification requires a typed claim"
            )
        if _timestamp(claim.artifact_expires_at, label="artifact_expires_at") <= datetime.now(
            timezone.utc
        ):
            raise GitHubArtifactTransportError("artifact evidence was expired at verification")
        repository = _json_object(
            _response(self._api, f"repos/{claim.repository}", accept="application/vnd.github+json"),
            label="repository",
        )
        _exact(
            repository,
            {
                "full_name": claim.repository,
                "id": claim.repository_id,
                "node_id": claim.repository_node_id,
                "private": False,
                "fork": False,
            },
            label="repository",
        )
        owner = repository.get("owner")
        if not isinstance(owner, Mapping):
            raise GitHubArtifactTransportError("repository owner is malformed")
        _exact(
            owner,
            {"id": OWNER_ID, "login": OWNER_LOGIN, "node_id": OWNER_NODE_ID},
            label="repository owner",
        )
        workflow = _json_object(
            _response(
                self._api,
                f"repos/{claim.repository}/actions/workflows/{claim.workflow_id}",
                accept="application/vnd.github+json",
            ),
            label="workflow",
        )
        _exact(workflow, {"id": claim.workflow_id, "path": claim.workflow_path}, label="workflow")
        run_endpoint = (
            f"repos/{claim.repository}/actions/runs/{claim.run_id}/attempts/{claim.run_attempt}"
        )
        run = _json_object(
            _response(self._api, run_endpoint, accept="application/vnd.github+json"),
            label="workflow run",
        )
        _exact(
            run,
            {
                "id": claim.run_id,
                "run_attempt": 1,
                "event": "workflow_dispatch",
                "workflow_id": claim.workflow_id,
                "head_sha": claim.head_sha,
                "head_branch": claim.head_branch,
                "actor": {"id": claim.actor_id, "login": claim.actor_login},
                "status": "completed",
                "conclusion": claim.conclusion,
            },
            label="workflow run",
        )
        if run.get("path") != claim.workflow_path:
            raise GitHubArtifactTransportError("workflow run path differs from the claim")
        for actor_field in ("actor", "triggering_actor"):
            actor = run.get(actor_field)
            if not isinstance(actor, Mapping):
                raise GitHubArtifactTransportError(f"workflow run {actor_field} is malformed")
            _exact(
                actor,
                {"id": OWNER_ID, "login": OWNER_LOGIN},
                label=f"workflow run {actor_field}",
            )
        run_repository = run.get("repository")
        if not isinstance(run_repository, Mapping):
            raise GitHubArtifactTransportError("workflow run repository is malformed")
        _exact(
            run_repository,
            {
                "id": claim.repository_id,
                "node_id": claim.repository_node_id,
                "full_name": claim.repository,
            },
            label="workflow run repository",
        )
        head_repository = run.get("head_repository")
        if not isinstance(head_repository, Mapping):
            raise GitHubArtifactTransportError("workflow run head_repository is malformed")
        _exact(
            head_repository,
            {
                "id": claim.repository_id,
                "node_id": claim.repository_node_id,
                "full_name": claim.repository,
            },
            label="workflow run head_repository",
        )
        _verify_c0_tag(self._api, claim)
        inventory_response = _json_object(
            _response(
                self._api,
                f"repos/{claim.repository}/actions/runs/{claim.run_id}/artifacts?per_page=100",
                accept="application/vnd.github+json",
            ),
            label="run artifact inventory",
        )
        artifacts = inventory_response.get("artifacts")
        total_count = inventory_response.get("total_count")
        if (
            not isinstance(artifacts, list)
            or type(total_count) is not int
            or total_count != len(artifacts)
            or total_count > 100
            or any(not isinstance(row, Mapping) for row in artifacts)
        ):
            raise GitHubArtifactTransportError("run artifact inventory is incomplete or malformed")
        candidates = [row for row in artifacts if row.get("name") == claim.artifact_name]
        if len(candidates) != 1:
            raise GitHubArtifactTransportError(
                "claimed artifact name is not a singleton in the complete run inventory"
            )
        _artifact_fields(candidates[0], claim, label="run artifact")
        id_candidates = [row for row in artifacts if row.get("id") == claim.artifact_id]
        if len(id_candidates) != 1 or id_candidates[0] is not candidates[0]:
            raise GitHubArtifactTransportError(
                "claimed artifact ID is not a singleton matching the claimed name"
            )
        artifact = _json_object(
            _response(
                self._api,
                f"repos/{claim.repository}/actions/artifacts/{claim.artifact_id}",
                accept="application/vnd.github+json",
            ),
            label="artifact",
        )
        _artifact_fields(artifact, claim, label="artifact")

    def _finish_execution_claim(
        self,
        claim: ExecutionArtifactClaim,
        archive: bytes,
        *,
        destination: Path,
    ) -> VerifiedExecutionArtifactReceipt:
        _artifact_fields(
            _json_object(
                _response(
                    self._api,
                    f"repos/{claim.repository}/actions/artifacts/{claim.artifact_id}",
                    accept="application/vnd.github+json",
                ),
                label="post-download artifact",
            ),
            claim,
            label="post-download artifact",
        )
        members = _materialize_archive(archive, claim, destination)
        inventory_sha256 = _sha256(_canonical_bytes([row.to_dict() for row in members]))
        return VerifiedExecutionArtifactReceipt(
            claim_sha256=claim.claim_sha256,
            repository=claim.repository,
            repository_id=claim.repository_id,
            workflow_path=claim.workflow_path,
            workflow_id=claim.workflow_id,
            workflow_ref=claim.workflow_ref,
            run_id=claim.run_id,
            run_attempt=claim.run_attempt,
            event=claim.event,
            head_sha=claim.head_sha,
            head_branch=claim.head_branch,
            actor_id=claim.actor_id,
            actor_login=claim.actor_login,
            status=claim.status,
            conclusion=claim.conclusion,
            artifact_id=claim.artifact_id,
            artifact_node_id=claim.artifact_node_id,
            artifact_name=claim.artifact_name,
            artifact_digest=claim.artifact_digest,
            artifact_size_bytes=claim.artifact_size_bytes,
            inventory=members,
            inventory_sha256=inventory_sha256,
            archive_sha256=_sha256(archive),
            materialized_root=str(destination.resolve(strict=True)),
            _capability=_CAPABILITY,
        )

    def verify_execution_claim(
        self, claim: ExecutionArtifactClaim, *, destination: Path
    ) -> VerifiedExecutionArtifactReceipt:
        self._verify_remote_claim(claim)
        archive = _download_archive(self._api, claim)
        return self._finish_execution_claim(claim, archive, destination=destination)

    def derive_and_verify_fixed_claim_artifact(
        self,
        workflow_context: object,
        *,
        suite_attempt_id: str,
        artifact_id: int,
        artifact_digest: str,
        expected_inventory_sha256: str,
        destination: Path,
    ) -> VerifiedExecutionArtifactReceipt:
        """Derive a closed claim from one fixed run and three trusted outputs.

        GitHub supplies the remaining artifact and workflow metadata.  The ZIP
        is downloaded once.  Its internal ``claim-package.SHA256SUMS`` is
        authenticated by ``expected_inventory_sha256`` and then used to close
        the exact member inventory before extraction.
        """

        context = _fixed_claim_artifact_context(workflow_context, suite_attempt_id)
        artifact_id = _positive(artifact_id, label="artifact_id")
        artifact_digest = _artifact_digest(
            artifact_digest,
            label="trusted upload-artifact digest",
        )
        expected_inventory_sha256 = _digest(
            expected_inventory_sha256,
            label="expected claim package inventory SHA-256",
        )
        run_endpoint = f"repos/{REPOSITORY}/actions/runs/{context.run_id}/attempts/1"
        run = _json_object(
            _response(self._api, run_endpoint, accept="application/vnd.github+json"),
            label="fixed workflow run",
        )
        workflow_id = _positive(run.get("workflow_id"), label="workflow_id")
        _exact(
            run,
            {
                "id": context.run_id,
                "run_attempt": 1,
                "event": "workflow_dispatch",
                "workflow_id": workflow_id,
                "head_sha": context.head_sha,
                "head_branch": C0_HEAD_BRANCH,
                "status": "completed",
                "conclusion": "success",
                "path": context.workflow_path,
            },
            label="fixed workflow run",
        )
        for actor_field in ("actor", "triggering_actor"):
            actor = run.get(actor_field)
            if not isinstance(actor, Mapping):
                raise GitHubArtifactTransportError(f"fixed workflow run {actor_field} is malformed")
            _exact(
                actor,
                {"id": OWNER_ID, "login": OWNER_LOGIN},
                label=f"fixed workflow run {actor_field}",
            )
        for repository_field in ("repository", "head_repository"):
            repository = run.get(repository_field)
            if not isinstance(repository, Mapping):
                raise GitHubArtifactTransportError(
                    f"fixed workflow run {repository_field} is malformed"
                )
            _exact(
                repository,
                {
                    "id": REPOSITORY_ID,
                    "node_id": REPOSITORY_NODE_ID,
                    "full_name": REPOSITORY,
                },
                label=f"fixed workflow run {repository_field}",
            )
        workflow = _json_object(
            _response(
                self._api,
                f"repos/{REPOSITORY}/actions/workflows/{workflow_id}",
                accept="application/vnd.github+json",
            ),
            label="fixed workflow",
        )
        _exact(
            workflow,
            {"id": workflow_id, "path": context.workflow_path},
            label="fixed workflow",
        )
        artifact = _json_object(
            _response(
                self._api,
                f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}",
                accept="application/vnd.github+json",
            ),
            label="fixed claim artifact",
        )
        _exact(
            artifact,
            {
                "id": artifact_id,
                "name": context.artifact_name,
                "digest": artifact_digest,
                "expired": False,
            },
            label="fixed claim artifact",
        )
        artifact_node_id = _text(
            artifact.get("node_id"),
            label="fixed claim artifact node_id",
        )
        artifact_size = artifact.get("size_in_bytes")
        if type(artifact_size) is not int or not 0 <= artifact_size <= MAX_ARCHIVE_BYTES:
            raise GitHubArtifactTransportError(
                "fixed claim artifact size_in_bytes is out of bounds"
            )
        created_at = _text(
            artifact.get("created_at"),
            label="fixed claim artifact created_at",
        )
        expires_at = _text(
            artifact.get("expires_at"),
            label="fixed claim artifact expires_at",
        )
        created = _timestamp(created_at, label="fixed claim artifact created_at")
        expires = _timestamp(expires_at, label="fixed claim artifact expires_at")
        if created >= expires or expires <= datetime.now(timezone.utc):
            raise GitHubArtifactTransportError(
                "fixed claim artifact is expired or has an invalid retention interval"
            )
        workflow_run = artifact.get("workflow_run")
        if not isinstance(workflow_run, Mapping):
            raise GitHubArtifactTransportError(
                "fixed claim artifact workflow_run is required and malformed"
            )
        _exact(
            workflow_run,
            {
                "id": context.run_id,
                "head_sha": context.head_sha,
                "head_branch": C0_HEAD_BRANCH,
                "repository_id": REPOSITORY_ID,
                "head_repository_id": REPOSITORY_ID,
            },
            label="fixed claim artifact workflow_run",
        )
        expected_download_url = (
            f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip"
        )
        if artifact.get("archive_download_url") != expected_download_url:
            raise GitHubArtifactTransportError(
                "fixed claim artifact archive download URL differs from its ID"
            )
        archive = _download_archive_exact(
            self._api,
            artifact_id=artifact_id,
            artifact_digest=artifact_digest,
            artifact_size_bytes=artifact_size,
        )
        inventory = _derive_closed_inventory(
            archive,
            expected_inventory_sha256=expected_inventory_sha256,
        )
        claim = ExecutionArtifactClaim(
            repository=REPOSITORY,
            repository_id=REPOSITORY_ID,
            repository_node_id=REPOSITORY_NODE_ID,
            owner_id=OWNER_ID,
            owner_login=OWNER_LOGIN,
            workflow_path=context.workflow_path,
            workflow_id=workflow_id,
            workflow_ref=C0_REF,
            run_id=context.run_id,
            head_sha=context.head_sha,
            head_branch=C0_HEAD_BRANCH,
            actor_id=OWNER_ID,
            actor_login=OWNER_LOGIN,
            conclusion="success",
            artifact_id=artifact_id,
            artifact_node_id=artifact_node_id,
            artifact_name=context.artifact_name,
            artifact_digest=artifact_digest,
            artifact_size_bytes=artifact_size,
            artifact_created_at=created_at,
            artifact_expires_at=expires_at,
            inventory=inventory,
        )
        self._verify_remote_claim(claim)
        return self._finish_execution_claim(claim, archive, destination=destination)


def verify_execution_claim(
    claim: ExecutionArtifactClaim, api: GitHubArtifactReadApi, *, destination: Path
) -> VerifiedExecutionArtifactReceipt:
    """Convenience entry point for execution-claim code and hostile fake transports."""

    return GitHubArtifactTransport(api).verify_execution_claim(claim, destination=destination)


def derive_and_verify_fixed_claim_artifact(
    workflow_context: object,
    api: GitHubArtifactReadApi,
    *,
    suite_attempt_id: str,
    artifact_id: int,
    artifact_digest: str,
    expected_inventory_sha256: str,
    destination: Path,
) -> VerifiedExecutionArtifactReceipt:
    """Convenience entry point for the provider execute-job activation gate."""

    return GitHubArtifactTransport(api).derive_and_verify_fixed_claim_artifact(
        workflow_context,
        suite_attempt_id=suite_attempt_id,
        artifact_id=artifact_id,
        artifact_digest=artifact_digest,
        expected_inventory_sha256=expected_inventory_sha256,
        destination=destination,
    )
