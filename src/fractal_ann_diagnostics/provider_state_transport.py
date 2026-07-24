"""Read-only transport from protected GitHub state to provider authority.

The protected ledger is the state authority.  Actions artifacts are treated as
bounded, untrusted carriers for the state attestation bytes that are absent
from the ledger.  A provider predecessor is minted only after every state in
the current prefix has an exact successful attempt-1 artifact, a verified
Sigstore state attestation, and byte equality with the protected ledger.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol
from urllib.parse import quote

from .execution_claim import ANALYSIS_PHASE, LABEL_RELEASE_PHASE, ONLINE_PHASE, ProviderPhase
from .github_artifact_transport import (
    C0_HEAD_BRANCH,
    C0_REF,
    MAX_ARCHIVE_MEMBERS,
    MAX_INVENTORY_BYTES,
    MAX_MEMBER_BYTES,
    MAX_UNCOMPRESSED_BYTES,
    OWNER_ID,
    OWNER_LOGIN,
    REPOSITORY,
    REPOSITORY_ID,
    REPOSITORY_NODE_ID,
    GitHubArtifactReadApi,
    GitHubArtifactTransportError,
    GitHubHttpResponse,
    UrllibGitHubArtifactReadApi,
    _download_archive_exact,
)
from .github_state_attestation import (
    LEDGER_CONTROL_PREFIX,
    AttestationVerifier,
    GhApiClient,
    GitHubApi,
    GitHubSuiteEvidenceVerifier,
    LedgerSnapshot,
    LedgerTransition,
    load_ledger_snapshot,
)
from .suite_attempt import (
    PhaseClaimBindings,
    RunClaimBindings,
    SuiteAttemptError,
    SuiteAttestationDescriptor,
    SuiteAttestationEvidence,
    SuiteState,
    VerifiedProviderPredecessor,
    _mint_verified_provider_predecessor,
    _verify_one_attestation,
)

STATE_ATTESTATION_WORKFLOW = ".github/workflows/confirmatory-state-attestation.yml"
PROVIDER_STATE_TRANSPORT_SCHEMA = "fractal-provider-state-transport-v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_ARTIFACTS = 7

_PHASE_PREDECESSOR: Mapping[ProviderPhase, SuiteState] = {
    ONLINE_PHASE: "OPENED",
    LABEL_RELEASE_PHASE: "ONLINE_COMPLETE",
    ANALYSIS_PHASE: "LABELS_RELEASED",
}
_PHASE_PREDECESSOR_SEQUENCE: Mapping[ProviderPhase, int] = {
    ONLINE_PHASE: 0,
    LABEL_RELEASE_PHASE: 2,
    ANALYSIS_PHASE: 4,
}
_PHASE_CLAIM: Mapping[ProviderPhase, SuiteState] = {
    ONLINE_PHASE: "RUN_CLAIMED",
    LABEL_RELEASE_PHASE: "LABEL_RELEASE_CLAIMED",
    ANALYSIS_PHASE: "ANALYSIS_CLAIMED",
}
_PHASE_CLAIM_SEQUENCE: Mapping[ProviderPhase, int] = {
    ONLINE_PHASE: 1,
    LABEL_RELEASE_PHASE: 3,
    ANALYSIS_PHASE: 5,
}
_CLAIM_STATE_PHASE: Mapping[SuiteState, ProviderPhase] = {
    "RUN_CLAIMED": ONLINE_PHASE,
    "LABEL_RELEASE_CLAIMED": LABEL_RELEASE_PHASE,
    "ANALYSIS_CLAIMED": ANALYSIS_PHASE,
}
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA256SUM_LINE = re.compile(rb"([0-9a-f]{64})  ([^\r\n]+)\n")
_CLAIM_RECOVERY_BASENAMES = (
    "claim-receipt.json",
    "provider-plan.materialized.json",
    "study-manifest.json",
)


class _ProviderAuthorityTarget(Enum):
    PREDECESSOR = "predecessor"
    CLAIM = "claim"


class ProviderStateTransportError(SuiteAttemptError):
    """Protected state or its retained artifact evidence is inadmissible."""


class ProviderStateArtifactReadApi(GitHubArtifactReadApi, Protocol):
    """Dedicated read-only API used for hostile transport tests and production."""


@dataclass(frozen=True)
class ProviderStateArtifactAuthority:
    sequence: int
    state: SuiteState
    ledger_commit: str
    workflow_path: str
    workflow_sha: str
    run_id: int
    workflow_id: int
    artifact_id: int
    artifact_node_id: str
    artifact_name: str
    artifact_digest: str
    artifact_size_bytes: int
    artifact_created_at: str
    artifact_expires_at: str
    inventory_name: Literal["SHA256SUMS", "claim-package.SHA256SUMS"]
    inventory_sha256: str
    archive_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ProviderStateTransportReceipt:
    phase: ProviderPhase
    authority_kind: Literal["predecessor", "claim"]
    suite_attempt_id: str
    predecessor_state: SuiteState
    predecessor_sequence: int
    predecessor_state_record_sha256: str
    predecessor_ledger_commit: str
    control_inventory_sha256: str
    artifacts: tuple[ProviderStateArtifactAuthority, ...]
    materialized_root: str
    schema_version: str = PROVIDER_STATE_TRANSPORT_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifacts"
            },
            "artifacts": [row.to_dict() for row in self.artifacts],
        }

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))


@dataclass(frozen=True)
class MaterializedProviderPredecessor:
    """Verified provider capability plus its deterministic transport receipt."""

    predecessor: VerifiedProviderPredecessor
    receipt: ProviderStateTransportReceipt


@dataclass(frozen=True)
class _ArtifactExpectation:
    transition: LedgerTransition
    workflow_path: str
    workflow_sha: str
    run_id: int | None
    artifact_name: str
    inventory_name: Literal["SHA256SUMS", "claim-package.SHA256SUMS"]


@dataclass(frozen=True)
class _RemoteArtifact:
    expectation: _ArtifactExpectation
    run_id: int
    workflow_id: int
    artifact_id: int
    artifact_node_id: str
    artifact_name: str
    artifact_digest: str
    artifact_size_bytes: int
    artifact_created_at: str
    artifact_expires_at: str


@dataclass(frozen=True)
class _VerifiedArchive:
    authority: ProviderStateArtifactAuthority
    retained: Mapping[str, bytes]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ProviderStateTransportError("provider state value is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProviderStateTransportError(f"{label} must be lowercase SHA-256")
    return value


def _commit(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA1.fullmatch(value) is None:
        raise ProviderStateTransportError(f"{label} must be a full Git object ID")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ProviderStateTransportError(f"{label} must be non-empty text")
    return value


def _positive(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ProviderStateTransportError(f"{label} must be a positive integer")
    return value


def _timestamp(value: object, *, label: str) -> datetime:
    text = _text(value, label=label)
    if not text.endswith("Z"):
        raise ProviderStateTransportError(f"{label} must use UTC Z")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProviderStateTransportError(f"{label} is not RFC 3339") from exc
    if parsed.tzinfo != timezone.utc:
        raise ProviderStateTransportError(f"{label} must use UTC")
    return parsed


def _response(api: ProviderStateArtifactReadApi, location: str) -> GitHubHttpResponse:
    try:
        response = api.get(location, accept="application/vnd.github+json")
    except Exception as exc:
        raise ProviderStateTransportError(
            f"cannot read GitHub artifact authority at {location!r}"
        ) from exc
    if not isinstance(response, GitHubHttpResponse):
        raise ProviderStateTransportError("artifact API returned an untyped response")
    return response


def _json_response(
    api: ProviderStateArtifactReadApi, location: str, *, label: str
) -> Mapping[str, Any]:
    response = _response(api, location)
    if response.status != 200:
        raise ProviderStateTransportError(f"{label} returned HTTP {response.status}")
    if not response.body or len(response.body) > MAX_JSON_BYTES:
        raise ProviderStateTransportError(f"{label} response is empty or oversized")
    try:
        value = json.loads(response.body.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderStateTransportError(f"{label} response is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ProviderStateTransportError(f"{label} response is not an object")
    return value


def _exact(row: Mapping[str, Any], expected: Mapping[str, object], *, label: str) -> None:
    for name, value in expected.items():
        if row.get(name) != value or type(row.get(name)) is not type(value):
            raise ProviderStateTransportError(f"{label} {name} differs from protected state")


def _artifact_rows(value: Mapping[str, Any], *, label: str) -> tuple[Mapping[str, Any], ...]:
    rows = value.get("artifacts")
    count = value.get("total_count")
    if (
        not isinstance(rows, list)
        or type(count) is not int
        or count != len(rows)
        or count > 100
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise ProviderStateTransportError(f"{label} is incomplete or malformed")
    return tuple(rows)


def _claim_identity(transition: LedgerTransition) -> tuple[ProviderPhase, int, str, str]:
    payload = transition.state.payload
    if isinstance(payload, RunClaimBindings):
        phase = ONLINE_PHASE
        identity = payload.provider_identity
        contract = payload.execution_claim
        return phase, identity.run_id, contract.claim_workflow_path, contract.claim_workflow_sha
    if isinstance(payload, PhaseClaimBindings):
        phase = payload.phase_claim.phase
        identity = payload.provider_identity
        return (
            phase,
            identity.run_id,
            payload.phase_claim.claim_workflow_path,
            payload.phase_claim.claim_workflow_sha,
        )
    raise ProviderStateTransportError("claim state lacks typed provider identity")


def _expectation(
    transition: LedgerTransition,
    *,
    descriptor: SuiteAttestationDescriptor,
) -> _ArtifactExpectation:
    suite = transition.state.suite_attempt_id
    if transition.state.state in _CLAIM_STATE_PHASE:
        phase, run_id, workflow_path, workflow_sha = _claim_identity(transition)
        if phase != _CLAIM_STATE_PHASE[transition.state.state]:
            raise ProviderStateTransportError("claim state phase differs from its transition")
        return _ArtifactExpectation(
            transition=transition,
            workflow_path=workflow_path,
            workflow_sha=workflow_sha,
            run_id=run_id,
            artifact_name=f"confirmatory-{phase}-claim-{suite}-{run_id}",
            inventory_name="claim-package.SHA256SUMS",
        )
    return _ArtifactExpectation(
        transition=transition,
        workflow_path=STATE_ATTESTATION_WORKFLOW,
        workflow_sha=descriptor.expected_signer_digest,
        run_id=None,
        artifact_name=(
            f"confirmatory-state-{suite}-{transition.state.sequence}-{transition.commit_oid}"
        ),
        inventory_name="SHA256SUMS",
    )


def _verify_repository_and_tag(
    api: ProviderStateArtifactReadApi,
    *,
    workflow_sha: str,
) -> None:
    repository = _json_response(api, f"repos/{REPOSITORY}", label="repository")
    _exact(
        repository,
        {
            "full_name": REPOSITORY,
            "id": REPOSITORY_ID,
            "node_id": REPOSITORY_NODE_ID,
            "private": False,
            "fork": False,
        },
        label="repository",
    )
    owner = repository.get("owner")
    if not isinstance(owner, Mapping):
        raise ProviderStateTransportError("repository owner is malformed")
    _exact(owner, {"id": OWNER_ID, "login": OWNER_LOGIN}, label="repository owner")
    ref = _json_response(
        api,
        f"repos/{REPOSITORY}/git/ref/tags/{C0_HEAD_BRANCH}",
        label="C0 tag ref",
    )
    if ref.get("ref") != C0_REF or not isinstance(ref.get("object"), Mapping):
        raise ProviderStateTransportError("C0 tag ref differs")
    ref_object = ref["object"]
    if ref_object.get("type") != "tag":
        raise ProviderStateTransportError("C0 tag must be annotated")
    tag_oid = _commit(ref_object.get("sha"), label="C0 tag object")
    tag = _json_response(api, f"repos/{REPOSITORY}/git/tags/{tag_oid}", label="C0 tag")
    target = tag.get("object")
    tagger = tag.get("tagger")
    if not isinstance(target, Mapping) or not isinstance(tagger, Mapping):
        raise ProviderStateTransportError("C0 annotated tag is malformed")
    _exact(
        tag,
        {"tag": C0_HEAD_BRANCH},
        label="C0 annotated tag",
    )
    _exact(
        target,
        {"type": "commit", "sha": workflow_sha},
        label="C0 annotated tag target",
    )
    _exact(
        tagger,
        {"name": OWNER_LOGIN, "email": f"{OWNER_LOGIN}@users.noreply.github.com"},
        label="C0 tagger",
    )


def _artifact_candidate(
    api: ProviderStateArtifactReadApi,
    expectation: _ArtifactExpectation,
) -> tuple[Mapping[str, Any], int]:
    if expectation.run_id is None:
        endpoint = (
            f"repos/{REPOSITORY}/actions/artifacts?name="
            f"{quote(expectation.artifact_name, safe='')}&per_page=100"
        )
        rows = _artifact_rows(
            _json_response(api, endpoint, label="named artifact inventory"),
            label="named artifact inventory",
        )
        matches = [row for row in rows if row.get("name") == expectation.artifact_name]
        if len(matches) != 1:
            raise ProviderStateTransportError("state artifact name is not a singleton")
        workflow_run = matches[0].get("workflow_run")
        if not isinstance(workflow_run, Mapping):
            raise ProviderStateTransportError("state artifact lacks workflow_run authority")
        return matches[0], _positive(workflow_run.get("id"), label="state artifact run ID")
    endpoint = f"repos/{REPOSITORY}/actions/runs/{expectation.run_id}/artifacts?per_page=100"
    rows = _artifact_rows(
        _json_response(api, endpoint, label="run artifact inventory"),
        label="run artifact inventory",
    )
    matches = [row for row in rows if row.get("name") == expectation.artifact_name]
    if len(matches) != 1:
        raise ProviderStateTransportError("claim artifact name is not a singleton")
    return matches[0], expectation.run_id


def _remote_artifact(
    api: ProviderStateArtifactReadApi,
    expectation: _ArtifactExpectation,
) -> _RemoteArtifact:
    candidate, run_id = _artifact_candidate(api, expectation)
    run = _json_response(
        api,
        f"repos/{REPOSITORY}/actions/runs/{run_id}/attempts/1",
        label="artifact workflow run",
    )
    workflow_id = _positive(run.get("workflow_id"), label="workflow ID")
    _exact(
        run,
        {
            "id": run_id,
            "run_attempt": 1,
            "event": "workflow_dispatch",
            "workflow_id": workflow_id,
            "head_sha": expectation.workflow_sha,
            "head_branch": C0_HEAD_BRANCH,
            "status": "completed",
            "conclusion": "success",
            "path": expectation.workflow_path,
        },
        label="artifact workflow run",
    )
    for field in ("actor", "triggering_actor"):
        actor = run.get(field)
        if not isinstance(actor, Mapping):
            raise ProviderStateTransportError(f"artifact workflow run {field} is malformed")
        _exact(actor, {"id": OWNER_ID, "login": OWNER_LOGIN}, label=f"run {field}")
    for field in ("repository", "head_repository"):
        repository = run.get(field)
        if not isinstance(repository, Mapping):
            raise ProviderStateTransportError(f"artifact run {field} is malformed")
        _exact(
            repository,
            {
                "id": REPOSITORY_ID,
                "node_id": REPOSITORY_NODE_ID,
                "full_name": REPOSITORY,
            },
            label=f"run {field}",
        )
    workflow = _json_response(
        api,
        f"repos/{REPOSITORY}/actions/workflows/{workflow_id}",
        label="artifact workflow",
    )
    _exact(
        workflow,
        {"id": workflow_id, "path": expectation.workflow_path},
        label="artifact workflow",
    )
    artifact_id = _positive(candidate.get("id"), label="artifact ID")
    detail = _json_response(
        api,
        f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}",
        label="artifact",
    )
    if detail != candidate:
        raise ProviderStateTransportError("artifact detail differs from complete inventory")
    digest = _text(detail.get("digest"), label="artifact digest")
    if not digest.startswith("sha256:") or _SHA256.fullmatch(digest[7:]) is None:
        raise ProviderStateTransportError("artifact digest is not upload-artifact SHA-256")
    size = detail.get("size_in_bytes")
    if type(size) is not int or not 0 <= size <= 512 * 1024 * 1024:
        raise ProviderStateTransportError("artifact size is out of bounds")
    created = _text(detail.get("created_at"), label="artifact created_at")
    expires = _text(detail.get("expires_at"), label="artifact expires_at")
    if (
        detail.get("expired") is not False
        or _timestamp(created, label="artifact created_at")
        >= _timestamp(expires, label="artifact expires_at")
        or _timestamp(expires, label="artifact expires_at") <= datetime.now(timezone.utc)
    ):
        raise ProviderStateTransportError("artifact is expired or has invalid retention")
    workflow_run = detail.get("workflow_run")
    if not isinstance(workflow_run, Mapping):
        raise ProviderStateTransportError("artifact workflow_run is malformed")
    _exact(
        workflow_run,
        {
            "id": run_id,
            "head_sha": expectation.workflow_sha,
            "head_branch": C0_HEAD_BRANCH,
            "repository_id": REPOSITORY_ID,
            "head_repository_id": REPOSITORY_ID,
        },
        label="artifact workflow_run",
    )
    expected_url = f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip"
    if detail.get("archive_download_url") != expected_url:
        raise ProviderStateTransportError("artifact archive URL differs from its ID")
    _verify_repository_and_tag(api, workflow_sha=expectation.workflow_sha)
    return _RemoteArtifact(
        expectation=expectation,
        run_id=run_id,
        workflow_id=workflow_id,
        artifact_id=artifact_id,
        artifact_node_id=_text(detail.get("node_id"), label="artifact node ID"),
        artifact_name=expectation.artifact_name,
        artifact_digest=digest,
        artifact_size_bytes=size,
        artifact_created_at=created,
        artifact_expires_at=expires,
    )


def _member_path(value: bytes) -> str:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProviderStateTransportError("SHA256SUMS path is not UTF-8") from exc
    if text.startswith("./"):
        text = text[2:]
    path = PurePosixPath(text)
    if (
        not text
        or text.startswith("/")
        or "\\" in text
        or text != unicodedata.normalize("NFC", text)
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != text
    ):
        raise ProviderStateTransportError("SHA256SUMS contains a non-canonical path")
    return text


def _inventory(encoded: bytes, *, inventory_name: str) -> tuple[tuple[str, str], ...]:
    if not encoded or len(encoded) > MAX_INVENTORY_BYTES or not encoded.endswith(b"\n"):
        raise ProviderStateTransportError("SHA256SUMS is empty, oversized, or unterminated")
    rows: list[tuple[str, str]] = []
    offset = 0
    for match in _SHA256SUM_LINE.finditer(encoded):
        if match.start() != offset:
            raise ProviderStateTransportError("SHA256SUMS syntax is not strict GNU form")
        offset = match.end()
        path = _member_path(match.group(2))
        if path == inventory_name:
            raise ProviderStateTransportError("SHA256SUMS cannot include itself")
        rows.append((path, match.group(1).decode("ascii")))
    if offset != len(encoded) or not rows:
        raise ProviderStateTransportError("SHA256SUMS is malformed or empty")
    names = [name for name, _ in rows]
    aliases = [unicodedata.normalize("NFC", name).casefold() for name in names]
    if names != sorted(names, key=lambda item: item.encode("utf-8")):
        raise ProviderStateTransportError("SHA256SUMS paths are not bytewise sorted")
    if len(names) != len(set(names)) or len(aliases) != len(set(aliases)):
        raise ProviderStateTransportError("SHA256SUMS contains aliased paths")
    return tuple(rows)


def _safe_infos(archive: zipfile.ZipFile) -> Mapping[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
        raise ProviderStateTransportError("artifact ZIP member count is invalid")
    by_name: dict[str, zipfile.ZipInfo] = {}
    aliases: set[str] = set()
    total = 0
    for info in infos:
        try:
            raw_name = info.filename.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ProviderStateTransportError("artifact ZIP member name is not UTF-8") from exc
        path = _member_path(raw_name)
        mode = (info.external_attr >> 16) & 0o177777
        file_type = stat.S_IFMT(mode)
        alias = unicodedata.normalize("NFC", path).casefold()
        total += info.file_size
        if (
            path != info.filename
            or info.is_dir()
            or info.flag_bits & 0x1
            or info.file_size < 0
            or info.file_size > MAX_MEMBER_BYTES
            or info.compress_size < 0
            or stat.S_ISLNK(mode)
            or (file_type and file_type != stat.S_IFREG)
            or path in by_name
            or alias in aliases
        ):
            raise ProviderStateTransportError("artifact ZIP has an unsafe member")
        if info.compress_size == 0 and info.file_size:
            raise ProviderStateTransportError("artifact ZIP has an invalid compression ratio")
        if info.compress_size and info.file_size > info.compress_size * 200:
            raise ProviderStateTransportError("artifact ZIP exceeds the compression ratio")
        by_name[path] = info
        aliases.add(alias)
    if total > MAX_UNCOMPRESSED_BYTES:
        raise ProviderStateTransportError("artifact ZIP expands beyond the byte limit")
    return by_name


def _read_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    retain: bool,
) -> tuple[str, int, bytes | None]:
    digest = hashlib.sha256()
    total = 0
    chunks: list[bytes] = []
    with archive.open(info, "r") as source:
        while chunk := source.read(1024 * 1024):
            total += len(chunk)
            if total > info.file_size or total > MAX_MEMBER_BYTES:
                raise ProviderStateTransportError("artifact ZIP member exceeds its bound")
            digest.update(chunk)
            if retain:
                chunks.append(chunk)
    if total != info.file_size:
        raise ProviderStateTransportError("artifact ZIP member size differs")
    return digest.hexdigest(), total, b"".join(chunks) if retain else None


def _critical_paths(expectation: _ArtifactExpectation) -> set[str]:
    sequence = expectation.transition.state.sequence
    return {
        f"{sequence:03d}.state.json",
        f"{sequence:03d}.attestation.json",
        f"{sequence:03d}.sigstore.bundle.json",
    }


def _claim_recovery_paths(declared: Mapping[str, str]) -> tuple[str, ...]:
    selected: list[str] = []
    names = tuple(declared)
    for expected in _CLAIM_RECOVERY_BASENAMES:
        expected_alias = unicodedata.normalize("NFC", expected).casefold()
        matches = tuple(
            name
            for name in names
            if unicodedata.normalize("NFC", PurePosixPath(name).name).casefold() == expected_alias
        )
        if len(matches) != 1 or PurePosixPath(matches[0]).name != expected:
            raise ProviderStateTransportError(
                f"claim artifact requires one canonical {expected} member"
            )
        selected.append(matches[0])
    if len(selected) != len(set(selected)):
        raise ProviderStateTransportError("claim recovery members alias one path")
    return tuple(selected)


def _write_retained(root: Path, retained: Mapping[str, bytes]) -> None:
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name in sorted(retained, key=lambda item: item.encode("utf-8")):
        target = root.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(target.parent, 0o700)
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(retained[name])


def _verify_archive(
    api: ProviderStateArtifactReadApi,
    remote: _RemoteArtifact,
    *,
    destination: Path,
    retain_claim_recovery: bool = False,
) -> _VerifiedArchive:
    try:
        encoded = _download_archive_exact(
            api,
            artifact_id=remote.artifact_id,
            artifact_digest=remote.artifact_digest,
            artifact_size_bytes=remote.artifact_size_bytes,
        )
    except GitHubArtifactTransportError as exc:
        raise ProviderStateTransportError("artifact archive download failed closed") from exc
    required = _critical_paths(remote.expectation)
    retain_names = set(required)
    retain_names.add(remote.expectation.inventory_name)
    retained: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(encoded), "r") as archive:
            by_name = _safe_infos(archive)
            inventory_info = by_name.get(remote.expectation.inventory_name)
            if inventory_info is None:
                raise ProviderStateTransportError("artifact omits its fixed SHA256SUMS")
            inventory_digest, _, inventory_bytes = _read_member(
                archive,
                inventory_info,
                retain=True,
            )
            if inventory_bytes is None:  # pragma: no cover
                raise ProviderStateTransportError("artifact inventory was not retained")
            rows = _inventory(
                inventory_bytes,
                inventory_name=remote.expectation.inventory_name,
            )
            declared = dict(rows)
            claim_recovery_paths = _claim_recovery_paths(declared) if retain_claim_recovery else ()
            retain_names.update(claim_recovery_paths)
            expected_names = set(declared) | {remote.expectation.inventory_name}
            if set(by_name) != expected_names:
                raise ProviderStateTransportError("artifact members differ from closed SHA256SUMS")
            if not required.issubset(declared):
                raise ProviderStateTransportError("artifact omits required state evidence")
            for path, expected_digest in rows:
                retain = path in retain_names or path.startswith("ledger-controls/")
                observed, _, value = _read_member(archive, by_name[path], retain=retain)
                if observed != expected_digest:
                    raise ProviderStateTransportError(
                        f"artifact member {path!r} differs from SHA256SUMS"
                    )
                if value is not None:
                    retained[path] = value
            retained[remote.expectation.inventory_name] = inventory_bytes
    except zipfile.BadZipFile as exc:
        raise ProviderStateTransportError("artifact is not a valid ZIP") from exc
    _write_retained(destination, retained)
    transition = remote.expectation.transition
    authority = ProviderStateArtifactAuthority(
        sequence=transition.state.sequence,
        state=transition.state.state,
        ledger_commit=transition.commit_oid,
        workflow_path=remote.expectation.workflow_path,
        workflow_sha=remote.expectation.workflow_sha,
        run_id=remote.run_id,
        workflow_id=remote.workflow_id,
        artifact_id=remote.artifact_id,
        artifact_node_id=remote.artifact_node_id,
        artifact_name=remote.artifact_name,
        artifact_digest=remote.artifact_digest,
        artifact_size_bytes=remote.artifact_size_bytes,
        artifact_created_at=remote.artifact_created_at,
        artifact_expires_at=remote.artifact_expires_at,
        inventory_name=remote.expectation.inventory_name,
        inventory_sha256=inventory_digest,
        archive_sha256=_sha256(encoded),
    )
    return _VerifiedArchive(authority=authority, retained=retained)


def _descriptor(snapshot: LedgerSnapshot) -> tuple[SuiteAttestationDescriptor, bytes]:
    rows = [control for control in snapshot.controls if control.role == "attestation-descriptor"]
    if len(rows) != 1:
        raise ProviderStateTransportError("protected ledger lacks one attestation descriptor")
    encoded = rows[0].encoded
    try:
        value = json.loads(encoded.decode("utf-8", errors="strict"))
        descriptor = SuiteAttestationDescriptor.from_dict(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProviderStateTransportError("protected descriptor is malformed") from exc
    if encoded != descriptor.canonical_bytes() + b"\n":
        raise ProviderStateTransportError("protected descriptor bytes are not canonical")
    return descriptor, encoded


def _control_paths(snapshot: LedgerSnapshot) -> Mapping[str, bytes]:
    prefix = f"{LEDGER_CONTROL_PREFIX}/{snapshot.tip.state.suite_attempt_id}/"
    rows: dict[str, bytes] = {"ledger-controls/inventory.json": snapshot.control_inventory_bytes}
    for control in snapshot.controls:
        if not control.ledger_path.startswith(prefix):
            raise ProviderStateTransportError("ledger control path escapes its suite prefix")
        relative = control.ledger_path[len(prefix) :]
        rows[f"ledger-controls/{relative}"] = control.encoded
    return rows


def _snapshot_fingerprint(snapshot: LedgerSnapshot) -> str:
    return _sha256(
        _canonical_bytes(
            {
                "repository": snapshot.repository,
                "state_key": snapshot.state_key,
                "transitions": [
                    {
                        "commit_oid": row.commit_oid,
                        "previous_commit_oid": row.previous_commit_oid,
                        "state_sha256": _sha256(row.state_bytes),
                        "tree_oid": row.tree_oid,
                    }
                    for row in snapshot.transitions
                ],
                "control_inventory_sha256": _sha256(snapshot.control_inventory_bytes),
                "controls": [
                    {
                        "ledger_path": row.ledger_path,
                        "sha256": _sha256(row.encoded),
                    }
                    for row in snapshot.controls
                ],
            }
        )
    )


def _controlled_parent(path: Path) -> Path:
    if not path.is_absolute():
        raise ProviderStateTransportError("materialization parent must be absolute")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ProviderStateTransportError("cannot inspect materialization parent") from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ProviderStateTransportError("materialization parent is not controlled")
    return path


def _materialize_provider_authority(
    phase: ProviderPhase,
    suite_attempt_id: str,
    materialization_parent: Path,
    *,
    target: _ProviderAuthorityTarget,
    ledger_api: GitHubApi,
    artifact_api: ProviderStateArtifactReadApi,
    attestation_verifier: AttestationVerifier | None,
) -> MaterializedProviderPredecessor:
    if phase not in _PHASE_PREDECESSOR:
        raise ProviderStateTransportError("provider phase is not admitted")
    suite = _digest(suite_attempt_id, label="suite_attempt_id")
    parent = _controlled_parent(materialization_parent)
    try:
        snapshot = load_ledger_snapshot(
            repository=REPOSITORY,
            suite_attempt_id=suite,
            api=ledger_api,
        )
    except SuiteAttemptError:
        raise
    except Exception as exc:
        raise ProviderStateTransportError("cannot reconstruct protected ledger") from exc
    expected_state = {
        _ProviderAuthorityTarget.PREDECESSOR: _PHASE_PREDECESSOR,
        _ProviderAuthorityTarget.CLAIM: _PHASE_CLAIM,
    }[target][phase]
    expected_sequence = {
        _ProviderAuthorityTarget.PREDECESSOR: _PHASE_PREDECESSOR_SEQUENCE,
        _ProviderAuthorityTarget.CLAIM: _PHASE_CLAIM_SEQUENCE,
    }[target][phase]
    if (
        snapshot.state_key != f"refs/heads/confirmatory-ledger/{suite}"
        or snapshot.tip.state.suite_attempt_id != suite
        or snapshot.tip.state.state != expected_state
        or snapshot.tip.state.sequence != expected_sequence
        or len(snapshot.transitions) != expected_sequence + 1
    ):
        raise ProviderStateTransportError(
            f"protected ledger tip differs from the exact phase {target.value}"
        )
    if len(snapshot.transitions) > MAX_ARTIFACTS:
        raise ProviderStateTransportError("protected ledger exceeds the closed state machine")
    descriptor, descriptor_bytes = _descriptor(snapshot)
    expectations = tuple(
        _expectation(transition, descriptor=descriptor) for transition in snapshot.transitions
    )
    final_root = parent / f"provider-state-{suite}-{snapshot.tip.commit_oid}"
    if final_root.exists():
        raise ProviderStateTransportError("provider state materialization already exists")
    staging = Path(tempfile.mkdtemp(prefix=".provider-state-", dir=parent))
    os.chmod(staging, 0o700)
    authorities: list[ProviderStateArtifactAuthority] = []
    remotes: list[_RemoteArtifact] = []
    critical_digests: dict[str, str] = {}
    try:
        artifact_root = staging / "artifacts"
        artifact_root.mkdir(mode=0o700)
        controls = _control_paths(snapshot)
        control_proven = False
        for expectation in expectations:
            remote = _remote_artifact(artifact_api, expectation)
            remotes.append(remote)
            verified = _verify_archive(
                artifact_api,
                remote,
                destination=artifact_root
                / f"{expectation.transition.state.sequence:03d}-{remote.artifact_id}",
                retain_claim_recovery=(
                    target is _ProviderAuthorityTarget.CLAIM
                    and expectation.transition.state.sequence == expected_sequence
                ),
            )
            authorities.append(verified.authority)
            artifact_relative_root = (
                f"artifacts/{expectation.transition.state.sequence:03d}-{remote.artifact_id}"
            )
            for relative, encoded in verified.retained.items():
                critical_digests[f"{artifact_relative_root}/{relative}"] = _sha256(encoded)
            if expectation.inventory_name == "SHA256SUMS":
                if any(verified.retained.get(name) != value for name, value in controls.items()):
                    raise ProviderStateTransportError(
                        "state-attestation artifact controls differ from protected ledger"
                    )
                control_proven = True
            sequence = expectation.transition.state.sequence
            names = {
                f"{sequence:03d}.state.json": expectation.transition.state_bytes,
                f"{sequence:03d}.attestation.json": verified.retained[
                    f"{sequence:03d}.attestation.json"
                ],
                f"{sequence:03d}.sigstore.bundle.json": verified.retained[
                    f"{sequence:03d}.sigstore.bundle.json"
                ],
            }
            if (
                verified.retained[f"{sequence:03d}.state.json"]
                != expectation.transition.state_bytes
            ):
                raise ProviderStateTransportError(
                    "artifact state bytes differ from protected ledger"
                )
            for name, encoded in names.items():
                state_target = staging / name
                descriptor_fd = os.open(
                    state_target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                with os.fdopen(descriptor_fd, "wb") as stream:
                    stream.write(encoded)
                critical_digests[name] = _sha256(encoded)
        if not control_proven:
            raise ProviderStateTransportError("state prefix has no artifact-proven ledger controls")
        descriptor_path = staging / "attestation-descriptor.json"
        descriptor_path.write_bytes(descriptor_bytes)
        os.chmod(descriptor_path, 0o600)
        critical_digests[descriptor_path.name] = _sha256(descriptor_bytes)

        evidence_verifier = GitHubSuiteEvidenceVerifier(
            staging,
            api=ledger_api,
            attestation_verifier=attestation_verifier,
        )
        evidences: list[SuiteAttestationEvidence] = []
        for transition in snapshot.transitions:
            evidence = _verify_one_attestation(
                namespace=staging,
                state=transition.state,
                descriptor=descriptor,
                verifier=evidence_verifier,
                previous_evidence=None if not evidences else evidences[-1],
            )
            if evidence.transition_id != transition.commit_oid:
                raise ProviderStateTransportError("state evidence differs from ledger commit")
            evidences.append(evidence)
        os.replace(staging, final_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    control_digest = _sha256(snapshot.control_inventory_bytes)
    receipt = ProviderStateTransportReceipt(
        phase=phase,
        authority_kind=target.value,
        suite_attempt_id=suite,
        predecessor_state=snapshot.tip.state.state,
        predecessor_sequence=snapshot.tip.state.sequence,
        predecessor_state_record_sha256=snapshot.tip.state.record_sha256,
        predecessor_ledger_commit=snapshot.tip.commit_oid,
        control_inventory_sha256=control_digest,
        artifacts=tuple(authorities),
        materialized_root=str(final_root),
    )
    fingerprint = _snapshot_fingerprint(snapshot)
    authority_rows = tuple(zip(remotes, authorities, strict=True))

    def fresh_revalidator() -> None:
        current = load_ledger_snapshot(
            repository=REPOSITORY,
            suite_attempt_id=suite,
            api=ledger_api,
        )
        if _snapshot_fingerprint(current) != fingerprint:
            raise ProviderStateTransportError("protected ledger changed after verification")
        for remote, authority in authority_rows:
            refreshed = _remote_artifact(artifact_api, remote.expectation)
            comparable = {
                "run_id": refreshed.run_id,
                "workflow_id": refreshed.workflow_id,
                "artifact_id": refreshed.artifact_id,
                "artifact_node_id": refreshed.artifact_node_id,
                "artifact_name": refreshed.artifact_name,
                "artifact_digest": refreshed.artifact_digest,
                "artifact_size_bytes": refreshed.artifact_size_bytes,
                "artifact_created_at": refreshed.artifact_created_at,
                "artifact_expires_at": refreshed.artifact_expires_at,
            }
            if any(comparable[name] != getattr(authority, name) for name in comparable):
                raise ProviderStateTransportError("artifact authority changed after verification")
        for relative, expected in critical_digests.items():
            try:
                encoded = (final_root / relative).read_bytes()
            except OSError as exc:
                raise ProviderStateTransportError(
                    "materialized state evidence disappeared"
                ) from exc
            if _sha256(encoded) != expected:
                raise ProviderStateTransportError("materialized state evidence changed")

    predecessor = _mint_verified_provider_predecessor(
        records=tuple(transition.state for transition in snapshot.transitions),
        evidences=tuple(evidences),
        control_inventory_sha256=control_digest,
        artifact_receipt_sha256=receipt.receipt_sha256,
        fresh_revalidator=fresh_revalidator,
    )
    return MaterializedProviderPredecessor(predecessor=predecessor, receipt=receipt)


def _materialize_provider_predecessor(
    phase: ProviderPhase,
    suite_attempt_id: str,
    materialization_parent: Path,
    *,
    ledger_api: GitHubApi,
    artifact_api: ProviderStateArtifactReadApi,
    attestation_verifier: AttestationVerifier | None,
) -> MaterializedProviderPredecessor:
    return _materialize_provider_authority(
        phase,
        suite_attempt_id,
        materialization_parent,
        target=_ProviderAuthorityTarget.PREDECESSOR,
        ledger_api=ledger_api,
        artifact_api=artifact_api,
        attestation_verifier=attestation_verifier,
    )


def materialize_provider_predecessor(
    phase: ProviderPhase,
    suite_attempt_id: str,
    materialization_parent: Path,
    *,
    ledger_api: GitHubApi,
    artifact_api: ProviderStateArtifactReadApi,
) -> MaterializedProviderPredecessor:
    """Verify and materialize the exact current predecessor for one phase.

    Production callers choose only the phase, manifest-derived suite ID, and a
    controlled parent directory.  Artifact names, run IDs, workflow identities,
    member paths, predecessor state, and trust policy are ledger-derived.
    """

    return _materialize_provider_predecessor(
        phase,
        suite_attempt_id,
        materialization_parent,
        ledger_api=ledger_api,
        artifact_api=artifact_api,
        attestation_verifier=None,
    )


def materialize_provider_claim(
    phase: ProviderPhase,
    suite_attempt_id: str,
    materialization_parent: Path,
    *,
    ledger_api: GitHubApi,
    artifact_api: ProviderStateArtifactReadApi,
) -> MaterializedProviderPredecessor:
    """Verify the exact winning claim for one fixed provider phase.

    The phase alone determines RUN_CLAIMED, LABEL_RELEASE_CLAIMED, or
    ANALYSIS_CLAIMED and its fixed sequence. Claim artifacts, workflow runs,
    commits, archive members, and attestation policy remain ledger-derived.
    """

    return _materialize_provider_authority(
        phase,
        suite_attempt_id,
        materialization_parent,
        target=_ProviderAuthorityTarget.CLAIM,
        ledger_api=ledger_api,
        artifact_api=artifact_api,
        attestation_verifier=None,
    )


def materialize_provider_predecessor_with_token(
    phase: ProviderPhase,
    suite_attempt_id: str,
    materialization_parent: Path,
    *,
    github_token: str,
) -> MaterializedProviderPredecessor:
    """Production entry point using the fixed GitHub CLI and HTTPS transports."""

    return materialize_provider_predecessor(
        phase,
        suite_attempt_id,
        materialization_parent,
        ledger_api=GhApiClient(),
        artifact_api=UrllibGitHubArtifactReadApi(github_token),
    )


def materialize_provider_claim_with_token(
    phase: ProviderPhase,
    suite_attempt_id: str,
    materialization_parent: Path,
    *,
    github_token: str,
) -> MaterializedProviderPredecessor:
    """Production claim transport using fixed GitHub CLI and HTTPS readers."""

    return materialize_provider_claim(
        phase,
        suite_attempt_id,
        materialization_parent,
        ledger_api=GhApiClient(),
        artifact_api=UrllibGitHubArtifactReadApi(github_token),
    )
