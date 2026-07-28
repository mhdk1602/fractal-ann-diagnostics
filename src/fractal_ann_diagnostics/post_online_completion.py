"""Closed post-online completion receipts and Zenodo anchor publication.

The operator accepts only a freshly verified provider predecessor.  It derives
every local path and scientific input from that authority, publishes the five
canonical anchor records once, verifies anonymous byte readback, and closes the
local ``completion`` directory with one aggregate receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import stat
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote, unquote, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    load_verification_receipt,
    read_secure_control_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .confirmatory_input_operator import _assert_online_directory
from .execution_claim import (
    LABEL_RELEASE_PHASE,
    ExecutionBeaconContract,
    ExecutionClaimError,
    VerifiedPhaseClaimCapability,
)
from .external_anchors import (
    MAX_EXTERNAL_ANCHOR_RECORD_BYTES,
    PredictionCompletionAnchorReceipt,
    PredictionCompletionAnchorRecord,
    VerifiedPredictionCompletionAnchor,
    load_prediction_completion_anchor_receipt,
    load_prediction_completion_anchor_record,
    write_prediction_completion_anchor_receipt,
    write_prediction_completion_anchor_record,
)
from .label_separation import (
    PredictionCompletionReceipt,
    create_prediction_completion_receipt,
    load_prediction_completion_receipt,
    sealed_run_receipt_sha256,
    write_prediction_completion_receipt,
)
from .production_controls import (
    load_production_control_finalization_receipt,
    load_production_control_finalization_request,
)
from .scalable_execution import (
    ONLINE_EXECUTION_PLAN_FILENAME,
    ShardedOnlineExecutionPlan,
    load_sharded_online_execution_plan,
)
from .study import (
    FIXED_CORPORA,
    SealedRunReceipt,
    load_sealed_run_receipt,
    manifest_sha256,
    revision_sha256,
    validate_study_manifest,
)
from .suite_attempt import (
    OnlineCorpusClosure,
    OnlineSuiteClosure,
    PhaseClaimBindings,
    SuiteOpenBindings,
    SuiteStateRecord,
    VerifiedProviderPredecessor,
)

POST_ONLINE_COMPLETION_AGGREGATE_SCHEMA = "fractal-post-online-completion-aggregate-v1"
POST_ONLINE_COMPLETION_BINDING_SCHEMA = "fractal-post-online-completion-binding-v1"
POST_ONLINE_COMPLETION_AGGREGATE_FILENAME = "post-online-completion-anchor-receipt.json"
COMPLETION_RECEIPT_TIMESTAMP_SEMANTICS = "zenodo-deposition-created-at-utc"
ZENODO_API_ORIGIN = "https://zenodo.org"
ZENODO_CREATOR_NAME = "mhdk1602"
ZENODO_CREATOR_ORCID = "0009-0003-1036-9477"

_CANONICAL_CORPORA = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TOKEN = re.compile(rb"^[A-Za-z0-9._~+\-]{16,4096}$")
_ZENODO_BUCKET_PATH = re.compile(
    r"^/api/files/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_ONLINE_TRANSFER_FILE_BYTES = 1024 * 1024 * 1024
_MAX_TOKEN_BYTES = 4097
_HTTP_TIMEOUT_SECONDS = 15.0
_PUBLICATION_READBACK_ATTEMPTS = 12
_PUBLICATION_READBACK_SECONDS = 5.0


class PostOnlineCompletionError(ValueError):
    """Raised when completion evidence cannot be closed without ambiguity."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PostOnlineCompletionError("completion evidence is not canonical JSON") from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PostOnlineCompletionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise PostOnlineCompletionError(f"{name} must be a canonical non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise PostOnlineCompletionError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise PostOnlineCompletionError(f"{name} cannot contain control characters")
    return value


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise PostOnlineCompletionError(f"{name} must be a positive integer")
    return value


def _utc(name: str, value: object) -> datetime:
    text = _require_text(name, value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PostOnlineCompletionError(f"{name} must be ISO 8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise PostOnlineCompletionError(f"{name} must use UTC")
    return parsed.astimezone(timezone.utc)


def _canonical_utc(name: str, value: object) -> str:
    return _utc(name, value).isoformat()


def _closed_json(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PostOnlineCompletionError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PostOnlineCompletionError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostOnlineCompletionError(f"cannot decode {label}") from exc
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PostOnlineCompletionError(f"{label} must contain one JSON object")
    return value


def _closed(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PostOnlineCompletionError(f"{label} must be a JSON object")
    observed = set(value)
    if observed != fields:
        raise PostOnlineCompletionError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _file_uri(value: str, *, label: str) -> Path:
    parsed = urlsplit(_require_text(label, value))
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not Path(unquote(parsed.path)).is_absolute()
    ):
        raise PostOnlineCompletionError(f"{label} must be a canonical absolute file URI")
    path = Path(unquote(parsed.path))
    if path.as_uri() != value:
        raise PostOnlineCompletionError(f"{label} must use canonical URI encoding")
    return path


def _https_uri(value: object, *, label: str, zenodo_only: bool = False) -> str:
    text = _require_text(label, value)
    parsed = urlsplit(text)
    try:
        port = parsed.port
    except ValueError as exc:
        raise PostOnlineCompletionError(f"{label} has an invalid port") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or parsed.path in {"", "/"}
    ):
        raise PostOnlineCompletionError(f"{label} must be a fixed HTTPS URI")
    if zenodo_only and (
        parsed.hostname != "zenodo.org"
        or parsed.path != unquote(parsed.path)
        or not parsed.path.startswith("/api/")
    ):
        raise PostOnlineCompletionError(f"{label} must remain under the Zenodo API origin")
    return text


def _state_record(
    predecessor: VerifiedProviderPredecessor,
    state: str,
    *,
    payload_type: type,
) -> SuiteStateRecord:
    matches = [row for row in predecessor.records if row.state == state]
    if len(matches) != 1 or not isinstance(matches[0].payload, payload_type):
        raise PostOnlineCompletionError(
            f"provider predecessor must contain one typed {state} record"
        )
    return matches[0]


def _assert_predecessor_current(
    predecessor: VerifiedProviderPredecessor,
) -> None:
    try:
        predecessor.assert_current()
    except PostOnlineCompletionError:
        raise
    except Exception as exc:
        raise PostOnlineCompletionError("provider predecessor failed fresh revalidation") from exc


@dataclass(frozen=True)
class AdmittedCompletionSource:
    """Typed source objects for one state-bound completion receipt."""

    corpus_id: str
    predictions: object
    action_panel: object
    online_result: object
    execution: ShardedOnlineExecutionPlan

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise PostOnlineCompletionError("completion source names an unregistered corpus")
        if not isinstance(self.execution, ShardedOnlineExecutionPlan):
            raise PostOnlineCompletionError("completion source execution must be sharded")
        for name in ("predictions", "action_panel"):
            if getattr(getattr(self, name), "corpus", None) != self.corpus_id:
                raise PostOnlineCompletionError(
                    f"completion source {name} belongs to another corpus"
                )
        if (
            getattr(self.online_result, "execution_artifact_sha256", None)
            != self.execution.artifact_sha256
        ):
            raise PostOnlineCompletionError(
                "completion source online result belongs to another execution"
            )


@dataclass(frozen=True)
class AdmittedPostOnlineSuite:
    """Closed source set admitted from one fresh provider predecessor."""

    predecessor: VerifiedProviderPredecessor
    online_record: SuiteStateRecord
    claim_record: SuiteStateRecord
    manifest_digest: str
    sealed_run: SealedRunReceipt
    beacon: ExecutionBeaconContract
    completion_root: Path
    sources: tuple[AdmittedCompletionSource, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.predecessor, VerifiedProviderPredecessor):
            raise PostOnlineCompletionError("suite requires a verified provider predecessor")
        _require_sha256("manifest_digest", self.manifest_digest)
        if not isinstance(self.sealed_run, SealedRunReceipt):
            raise PostOnlineCompletionError("suite requires one sealed-run receipt")
        if not isinstance(self.beacon, ExecutionBeaconContract):
            raise PostOnlineCompletionError("suite requires one execution beacon contract")
        root = Path(self.completion_root)
        if not root.is_absolute() or root != self.predecessor.namespace / "completion":
            raise PostOnlineCompletionError("completion root is not provider-authoritative")
        rows = tuple(self.sources)
        if tuple(row.corpus_id for row in rows) != _CANONICAL_CORPORA:
            raise PostOnlineCompletionError(
                "completion sources must contain each fixed corpus once in UTF-8 byte order"
            )
        object.__setattr__(self, "completion_root", root)
        object.__setattr__(self, "sources", rows)


def _manifest_execution(
    manifest: Mapping[str, Any],
    corpus_id: str,
) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not list:
        raise PostOnlineCompletionError("frozen manifest artifact table is malformed")
    rows = [
        row
        for row in artifacts
        if isinstance(row, Mapping)
        and row.get("role") == "online-execution"
        and row.get("corpus_id") == corpus_id
    ]
    if len(rows) != 1:
        raise PostOnlineCompletionError(
            f"frozen manifest lacks one online execution for {corpus_id}"
        )
    return rows[0]


def _verify_online_transfer_bytes(
    closure: OnlineCorpusClosure,
    *,
    corpus_id: str,
) -> None:
    root = _file_uri(closure.output_uri, label=f"{corpus_id} online output URI")
    bindings = tuple(closure.transfer_files)
    expected_names = {row.relative_path for row in bindings}
    try:
        observed_names = {path.name for path in root.iterdir()}
    except OSError as exc:
        raise PostOnlineCompletionError(
            f"cannot enumerate canonical online outputs for {corpus_id}"
        ) from exc
    if observed_names != expected_names:
        raise PostOnlineCompletionError(
            f"canonical online output inventory changed for {corpus_id}"
        )
    for binding in bindings:
        if binding.byte_count <= 0 or binding.byte_count > _MAX_ONLINE_TRANSFER_FILE_BYTES:
            raise PostOnlineCompletionError(
                f"canonical online output {binding.role} has an invalid byte count"
            )
        try:
            encoded = read_secure_regular_file(
                root / binding.relative_path,
                max_bytes=binding.byte_count,
                label=f"{corpus_id} canonical online output {binding.role}",
            )
        except ArtifactIntegrityError as exc:
            raise PostOnlineCompletionError(
                f"cannot read canonical online output {binding.role} for {corpus_id}"
            ) from exc
        if len(encoded) != binding.byte_count or _sha256(encoded) != binding.file_sha256:
            raise PostOnlineCompletionError(
                f"canonical online output {binding.role} changed for {corpus_id}"
            )


def _secure_manifest(path: Path) -> Mapping[str, Any]:
    try:
        encoded = read_secure_control_file(path, label="frozen study manifest")
    except ArtifactIntegrityError as exc:
        raise PostOnlineCompletionError("cannot read frozen manifest safely") from exc
    manifest = _closed_json(encoded, label="frozen study manifest")
    try:
        validate_study_manifest(manifest, require_frozen=True)
    except ValueError as exc:
        raise PostOnlineCompletionError("frozen study manifest is invalid") from exc
    if encoded != _canonical_bytes(manifest) + b"\n":
        raise PostOnlineCompletionError("frozen study manifest bytes are not canonical")
    return manifest


def _admit_post_online_suite(
    predecessor: VerifiedProviderPredecessor,
) -> AdmittedPostOnlineSuite:
    if not isinstance(predecessor, VerifiedProviderPredecessor):
        raise PostOnlineCompletionError(
            "post-online completion requires a VerifiedProviderPredecessor"
        )
    _assert_predecessor_current(predecessor)
    if predecessor.state.state != "LABEL_RELEASE_CLAIMED":
        raise PostOnlineCompletionError(
            "post-online completion requires LABEL_RELEASE_CLAIMED as the current state"
        )
    opened = _state_record(predecessor, "OPENED", payload_type=SuiteOpenBindings)
    online = _state_record(predecessor, "ONLINE_COMPLETE", payload_type=OnlineSuiteClosure)
    claim = _state_record(predecessor, "LABEL_RELEASE_CLAIMED", payload_type=PhaseClaimBindings)
    if claim is not predecessor.state:
        raise PostOnlineCompletionError("label-release claim is not the current state")
    if claim.sequence != online.sequence + 1:
        raise PostOnlineCompletionError("label-release claim is not the online successor")
    if claim.previous_state_record_sha256 != online.record_sha256:
        raise PostOnlineCompletionError("label-release claim changes its online predecessor")

    claim_payload = claim.payload
    assert isinstance(claim_payload, PhaseClaimBindings)
    phase = claim_payload.phase_claim
    beacon = phase.label_release_beacon
    if (
        phase.phase != "label-release"
        or not isinstance(beacon, ExecutionBeaconContract)
        or phase.predecessor_state_sha256 != online.record_sha256
        or claim_payload.predecessor_state_sha256 != online.record_sha256
        or phase.predecessor_ledger_commit != predecessor.evidences[online.sequence].transition_id
        or phase.manifest_sha256 != claim.manifest_sha256
        or phase.run_receipt_sha256 != claim.run_receipt_sha256
    ):
        raise PostOnlineCompletionError("label-release claim does not bind ONLINE_COMPLETE")

    opened_payload = opened.payload
    assert isinstance(opened_payload, SuiteOpenBindings)
    finalization_path = _file_uri(
        opened_payload.production_finalization_receipt_uri,
        label="production finalization receipt URI",
    )
    try:
        finalization = load_production_control_finalization_receipt(
            finalization_path,
            expected_sha256=opened_payload.production_finalization_receipt_file_sha256,
        )
    except Exception as exc:
        raise PostOnlineCompletionError("cannot admit production finalization receipt") from exc
    if (
        finalization.finalization_request_sha256
        != opened_payload.production_finalization_request_sha256
        or finalization.manifest_sha256 != claim.manifest_sha256
        or finalization.suite_attempt_id != claim.suite_attempt_id
        or Path(finalization.canonical_suite_namespace) != predecessor.namespace
        or finalization.provisional_closure_tree_sha256
        != opened_payload.provisional_closure_tree_sha256
        or finalization.instantiated_closure_tree_sha256
        != opened_payload.instantiated_closure_tree_sha256
        or finalization.sealed_run_receipt_file_sha256 != opened_payload.run_receipt_file_sha256
    ):
        raise PostOnlineCompletionError("production finalization authority differs from state")
    request_path = finalization_path.with_name("finalization-request.json")
    try:
        request = load_production_control_finalization_request(
            request_path,
            expected_sha256=finalization.finalization_request_sha256,
        )
    except Exception as exc:
        raise PostOnlineCompletionError("cannot admit production finalization request") from exc

    manifest = _secure_manifest(request.frozen_manifest_path)
    digest = manifest_sha256(manifest)
    if digest != claim.manifest_sha256:
        raise PostOnlineCompletionError("frozen manifest differs from provider state")
    try:
        sealed_run = load_sealed_run_receipt(request.sealed_run_receipt_path)
        run_file_digest = digest_regular_file(
            request.sealed_run_receipt_path,
            label="sealed run receipt",
        )
        verification = load_verification_receipt(request.artifact_verification_receipt_path)
    except Exception as exc:
        raise PostOnlineCompletionError("cannot admit sealed-run evidence") from exc
    run_digest = sealed_run_receipt_sha256(sealed_run)
    if (
        sealed_run.manifest_sha256 != digest
        or run_digest != claim.run_receipt_sha256
        or run_file_digest != opened_payload.run_receipt_file_sha256
        or sealed_run.code_commit != opened_payload.code_commit
        or sealed_run.code_commit != finalization.c0_commit
        or sealed_run.runner_image != opened_payload.runner_image
        or sealed_run.started_at_utc != opened_payload.run_started_at_utc
        or _file_uri(
            sealed_run.verification_receipt_uri,
            label="sealed-run verification receipt URI",
        )
        != request.artifact_verification_receipt_path
        or verification.manifest_sha256 != digest
        or verification.receipt_sha256 != sealed_run.verification_receipt_sha256
        or verification.receipt_sha256 != predecessor.artifact_receipt_sha256
    ):
        raise PostOnlineCompletionError("sealed-run evidence differs from provider state")

    executions = {row.corpus_id: row.sha256 for row in opened_payload.execution_artifacts}
    namespaces = {row.corpus_id: row.output_uri for row in opened_payload.output_namespaces}
    verification_rows = {row.artifact_id: row for row in verification.artifacts}
    closure_rows = {row.corpus_id: row for row in online.payload.corpora}
    sources: list[AdmittedCompletionSource] = []
    for corpus_id in _CANONICAL_CORPORA:
        closure = closure_rows.get(corpus_id)
        if closure is None or closure.output_uri != namespaces.get(corpus_id):
            raise PostOnlineCompletionError(
                f"ONLINE_COMPLETE changes the canonical output for {corpus_id}"
            )
        _verify_online_transfer_bytes(closure, corpus_id=corpus_id)
        try:
            result, predictions, panel, _admission, _members = _assert_online_directory(
                closure,
                manifest_digest=digest,
            )
        except Exception as exc:
            raise PostOnlineCompletionError(
                f"cannot admit canonical online outputs for {corpus_id}"
            ) from exc
        artifact = _manifest_execution(manifest, corpus_id)
        artifact_id = artifact.get("id")
        if type(artifact_id) is not str or artifact_id not in verification_rows:
            raise PostOnlineCompletionError(
                f"artifact verification lacks {corpus_id} online execution"
            )
        verified = verification_rows[artifact_id]
        execution_root = request.artifact_root / verified.relative_path
        try:
            observed_tree = digest_directory_tree(execution_root)
        except Exception as exc:
            raise PostOnlineCompletionError(f"cannot rehash {corpus_id} execution package") from exc
        if (
            verified.kind != "directory"
            or verified.exact is not True
            or verified.verified_sha256 != artifact.get("sha256")
            or observed_tree.sha256 != verified.verified_sha256
        ):
            raise PostOnlineCompletionError(
                f"{corpus_id} execution package differs from verification evidence"
            )
        try:
            execution = load_sharded_online_execution_plan(
                execution_root / ONLINE_EXECUTION_PLAN_FILENAME
            )
        except Exception as exc:
            raise PostOnlineCompletionError(f"cannot load {corpus_id} execution plan") from exc
        logical_digest = revision_sha256(
            artifact.get("revision"),
            field=f"{corpus_id} online execution revision",
        )
        if (
            execution.corpus != corpus_id
            or execution.stage != "sealed"
            or execution.artifact_sha256 != logical_digest
            or execution.artifact_sha256 != executions.get(corpus_id)
            or execution.artifact_sha256 != closure.execution_artifact_sha256
        ):
            raise PostOnlineCompletionError(
                f"{corpus_id} execution identity differs across custody records"
            )
        sources.append(
            AdmittedCompletionSource(
                corpus_id=corpus_id,
                predictions=predictions,
                action_panel=panel,
                online_result=result,
                execution=execution,
            )
        )
    _assert_predecessor_current(predecessor)
    return AdmittedPostOnlineSuite(
        predecessor=predecessor,
        online_record=online,
        claim_record=claim,
        manifest_digest=digest,
        sealed_run=sealed_run,
        beacon=beacon,
        completion_root=predecessor.namespace / "completion",
        sources=tuple(sources),
    )


@dataclass(frozen=True)
class ZenodoDeposition:
    record_id: int
    created_at_utc: str
    self_uri: str
    bucket_uri: str
    publish_uri: str

    def __post_init__(self) -> None:
        _positive_integer("record_id", self.record_id)
        object.__setattr__(
            self,
            "created_at_utc",
            _canonical_utc("Zenodo created timestamp", self.created_at_utc),
        )
        expected_self = f"{ZENODO_API_ORIGIN}/api/deposit/depositions/{self.record_id}"
        observed_self = _https_uri(
            self.self_uri,
            label="Zenodo deposition URI",
            zenodo_only=True,
        )
        if observed_self != expected_self:
            raise PostOnlineCompletionError("Zenodo deposition URI changes the record ID")
        expected_publish = f"{expected_self}/actions/publish"
        if (
            _https_uri(self.publish_uri, label="Zenodo publish URI", zenodo_only=True)
            != expected_publish
        ):
            raise PostOnlineCompletionError("Zenodo publish URI changes the record ID")
        bucket = _https_uri(
            self.bucket_uri,
            label="Zenodo bucket URI",
            zenodo_only=True,
        )
        if _ZENODO_BUCKET_PATH.fullmatch(urlsplit(bucket).path) is None:
            raise PostOnlineCompletionError("Zenodo bucket URI has an invalid path")

    @property
    def identity(self) -> str:
        return f"zenodo-record:{self.record_id}"

    def content_uri(self, filename: str) -> str:
        name = _require_text("anchor filename", filename)
        if "/" in name or quote(name, safe="") != name:
            raise PostOnlineCompletionError("anchor filename must be one safe URI segment")
        return f"{ZENODO_API_ORIGIN}/api/records/{self.record_id}/files/{name}/content"


@dataclass(frozen=True)
class ZenodoRemoteFile:
    filename: str
    byte_count: int
    checksum: str | None = None

    def __post_init__(self) -> None:
        _require_text("remote filename", self.filename)
        if "/" in self.filename or quote(self.filename, safe="") != self.filename:
            raise PostOnlineCompletionError("remote filename must be one safe segment")
        _positive_integer("remote byte_count", self.byte_count)
        if self.checksum is not None:
            _require_text("remote checksum", self.checksum)


@dataclass(frozen=True)
class ZenodoPublishedRecord:
    """Anonymous public-record identity and server-side latest-update bound."""

    record_id: int
    created_at_utc: str
    updated_at_utc: str
    files: tuple[ZenodoRemoteFile, ...]

    def __post_init__(self) -> None:
        _positive_integer("public Zenodo record_id", self.record_id)
        created = _utc("public Zenodo created timestamp", self.created_at_utc)
        updated = _utc("public Zenodo updated timestamp", self.updated_at_utc)
        if updated <= created:
            raise PostOnlineCompletionError(
                "public Zenodo updated timestamp must follow deposition creation"
            )
        rows = tuple(self.files)
        if (
            not rows
            or not all(isinstance(row, ZenodoRemoteFile) for row in rows)
            or len({row.filename for row in rows}) != len(rows)
        ):
            raise PostOnlineCompletionError("public Zenodo record needs unique typed files")
        object.__setattr__(self, "created_at_utc", created.isoformat())
        object.__setattr__(self, "updated_at_utc", updated.isoformat())
        object.__setattr__(
            self,
            "files",
            tuple(sorted(rows, key=lambda row: row.filename.encode("utf-8"))),
        )


class CompletionAnchorPublisher(Protocol):
    """Transport surface used by the closed operator."""

    def create_deposition(self) -> ZenodoDeposition: ...

    def set_metadata(
        self,
        deposition: ZenodoDeposition,
        metadata: Mapping[str, object],
    ) -> None: ...

    def upload(
        self,
        deposition: ZenodoDeposition,
        filename: str,
        payload: bytes,
    ) -> None: ...

    def draft_files(
        self,
        deposition: ZenodoDeposition,
    ) -> tuple[ZenodoRemoteFile, ...]: ...

    def publish_once(self, deposition: ZenodoDeposition) -> None: ...

    def public_record(
        self,
        deposition: ZenodoDeposition,
    ) -> ZenodoPublishedRecord | None: ...

    def anonymous_read(self, uri: str, *, max_bytes: int) -> bytes: ...


class AnonymousCompletionAnchorReader(Protocol):
    """Tokenless transport used for fresh release-time revalidation."""

    def public_record(
        self,
        *,
        record_id: int,
        expected_created_at_utc: str,
    ) -> ZenodoPublishedRecord | None: ...

    def anonymous_read(self, uri: str, *, max_bytes: int) -> bytes: ...


class RoundPublicationGuard(Protocol):
    """Fresh drand publication check used at each irreversible boundary."""

    def assert_not_public(self, beacon: ExecutionBeaconContract) -> None: ...


class _NoRedirects(urllib_request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, msg, headers, newurl
        raise PostOnlineCompletionError(f"HTTPS redirect status {code} was refused")


class _NotFound(Exception):
    pass


def _verified_opener() -> Any:
    context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return urllib_request.build_opener(
        _NoRedirects(),
        urllib_request.HTTPSHandler(context=context),
    )


def _read_token_fd(token_fd: int) -> bytearray:
    if type(token_fd) is not int or token_fd < 0:
        raise PostOnlineCompletionError("token_fd must be a non-negative file descriptor")
    try:
        metadata = os.fstat(token_fd)
        if stat.S_ISDIR(metadata.st_mode):
            raise PostOnlineCompletionError("token_fd cannot refer to a directory")
    except PostOnlineCompletionError:
        raise
    except OSError as exc:
        raise PostOnlineCompletionError("cannot read Zenodo token file descriptor") from exc
    chunks: list[bytes] = []
    observed = 0
    while True:
        try:
            chunk = os.read(token_fd, min(1024, _MAX_TOKEN_BYTES + 2 - observed))
        except OSError as exc:
            raise PostOnlineCompletionError("cannot read Zenodo token file descriptor") from exc
        if not chunk:
            break
        chunks.append(chunk)
        observed += len(chunk)
        if observed > _MAX_TOKEN_BYTES + 1:
            raise PostOnlineCompletionError("Zenodo token exceeds the byte limit")
    encoded = b"".join(chunks)
    if encoded.endswith(b"\n"):
        encoded = encoded[:-1]
    if b"\n" in encoded or b"\r" in encoded or _TOKEN.fullmatch(encoded) is None:
        raise PostOnlineCompletionError("Zenodo token file descriptor has invalid bytes")
    return bytearray(encoded)


class ZenodoCompletionAnchorPublisher:
    """Certificate-validated, no-redirect Zenodo publisher with a redacted token."""

    def __init__(
        self,
        *,
        token_fd: int,
        opener: Any | None = None,
        timeout_seconds: float = _HTTP_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise PostOnlineCompletionError("HTTPS timeout must be positive")
        active_opener = _verified_opener() if opener is None else opener
        self._token = _read_token_fd(token_fd)
        self._opener = active_opener
        self._timeout_seconds = float(timeout_seconds)
        self._closed = False

    @classmethod
    def from_token_fd(
        cls,
        token_fd: int,
        *,
        opener: Any | None = None,
        timeout_seconds: float = _HTTP_TIMEOUT_SECONDS,
    ) -> ZenodoCompletionAnchorPublisher:
        return cls(
            token_fd=token_fd,
            opener=opener,
            timeout_seconds=timeout_seconds,
        )

    def __repr__(self) -> str:
        return (
            "ZenodoCompletionAnchorPublisher("
            f"closed={self._closed}, timeout_seconds={self._timeout_seconds})"
        )

    def close(self) -> None:
        for index in range(len(self._token)):
            self._token[index] = 0
        self._closed = True

    def __enter__(self) -> ZenodoCompletionAnchorPublisher:
        if self._closed:
            raise PostOnlineCompletionError("Zenodo publisher is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def _request(
        self,
        method: str,
        uri: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        authenticated: bool,
        expected_statuses: frozenset[int],
        max_bytes: int,
    ) -> bytes:
        if self._closed:
            raise PostOnlineCompletionError("Zenodo publisher is closed")
        fixed_uri = _https_uri(uri, label="Zenodo request URI", zenodo_only=True)
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "fractal-ann-diagnostics/0.3 completion-anchor",
        }
        if authenticated:
            headers["Authorization"] = f"Bearer {bytes(self._token).decode('ascii')}"
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib_request.Request(fixed_uri, data=body, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self._timeout_seconds) as response:
                status = response.getcode()
                if status not in expected_statuses:
                    raise PostOnlineCompletionError(
                        f"Zenodo returned unexpected HTTP status {status}"
                    )
                if response.geturl() != fixed_uri:
                    raise PostOnlineCompletionError("Zenodo response URL changed")
                content_encoding = response.headers.get("Content-Encoding")
                if content_encoding not in {None, "", "identity"}:
                    raise PostOnlineCompletionError("Zenodo response used content encoding")
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdecimal() or int(length) > max_bytes):
                    raise PostOnlineCompletionError(
                        "Zenodo response Content-Length is invalid or excessive"
                    )
                encoded = response.read(max_bytes + 1)
        except PostOnlineCompletionError:
            raise
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                raise _NotFound from None
            if 300 <= exc.code < 400:
                raise PostOnlineCompletionError("Zenodo redirect was refused") from None
            raise PostOnlineCompletionError(
                f"Zenodo returned unexpected HTTP status {exc.code}"
            ) from None
        except (urllib_error.URLError, TimeoutError, ssl.SSLError, OSError):
            raise PostOnlineCompletionError("Zenodo HTTPS request failed") from None
        if len(encoded) > max_bytes:
            raise PostOnlineCompletionError("Zenodo response exceeds the byte limit")
        if length is not None and len(encoded) != int(length):
            raise PostOnlineCompletionError("Zenodo response differs from its Content-Length")
        return encoded

    def _json(
        self,
        method: str,
        uri: str,
        *,
        payload: Mapping[str, object] | None = None,
        expected_statuses: frozenset[int],
        authenticated: bool = True,
    ) -> Mapping[str, Any]:
        encoded_payload = None if payload is None else _canonical_bytes(payload)
        encoded = self._request(
            method,
            uri,
            body=encoded_payload,
            content_type=None if payload is None else "application/json",
            authenticated=authenticated,
            expected_statuses=expected_statuses,
            max_bytes=_MAX_JSON_BYTES,
        )
        return _closed_json(encoded, label="Zenodo response")

    @staticmethod
    def _deposition(payload: Mapping[str, Any]) -> ZenodoDeposition:
        record_id = payload.get("id")
        submitted = payload.get("submitted")
        state = payload.get("state")
        links = payload.get("links")
        if (
            type(record_id) is not int
            or record_id <= 0
            or submitted is not False
            or state != "unsubmitted"
            or not isinstance(links, Mapping)
        ):
            raise PostOnlineCompletionError("Zenodo created an invalid deposition")
        return ZenodoDeposition(
            record_id=record_id,
            created_at_utc=_canonical_utc(
                "Zenodo created timestamp",
                payload.get("created"),
            ),
            self_uri=links.get("self"),
            bucket_uri=links.get("bucket"),
            publish_uri=links.get("publish"),
        )

    @staticmethod
    def _files(payload: Mapping[str, Any]) -> tuple[ZenodoRemoteFile, ...]:
        values = payload.get("files")
        if type(values) is not list:
            raise PostOnlineCompletionError("Zenodo response lacks a file inventory")
        result: list[ZenodoRemoteFile] = []
        for value in values:
            if not isinstance(value, Mapping):
                raise PostOnlineCompletionError("Zenodo file inventory is malformed")
            names = [value[key] for key in ("filename", "key") if key in value]
            sizes = [value[key] for key in ("filesize", "size") if key in value]
            if not names or any(name != names[0] for name in names):
                raise PostOnlineCompletionError("Zenodo file inventory has conflicting filenames")
            if (
                not sizes
                or any(type(size) is not int for size in sizes)
                or any(size != sizes[0] for size in sizes)
            ):
                raise PostOnlineCompletionError("Zenodo file inventory has conflicting byte counts")
            checksum = value.get("checksum")
            result.append(
                ZenodoRemoteFile(
                    filename=names[0],
                    byte_count=sizes[0],
                    checksum=checksum if isinstance(checksum, str) else None,
                )
            )
        if len({row.filename for row in result}) != len(result):
            raise PostOnlineCompletionError("Zenodo file inventory repeats a filename")
        return tuple(sorted(result, key=lambda row: row.filename.encode("utf-8")))

    def create_deposition(self) -> ZenodoDeposition:
        payload = self._json(
            "POST",
            f"{ZENODO_API_ORIGIN}/api/deposit/depositions",
            payload={},
            expected_statuses=frozenset({201}),
        )
        return self._deposition(payload)

    def set_metadata(
        self,
        deposition: ZenodoDeposition,
        metadata: Mapping[str, object],
    ) -> None:
        self._json(
            "PUT",
            deposition.self_uri,
            payload={"metadata": dict(metadata)},
            expected_statuses=frozenset({200}),
        )

    def upload(
        self,
        deposition: ZenodoDeposition,
        filename: str,
        payload: bytes,
    ) -> None:
        name = ZenodoRemoteFile(filename=filename, byte_count=len(payload)).filename
        self._request(
            "PUT",
            f"{deposition.bucket_uri}/{name}",
            body=payload,
            content_type="application/octet-stream",
            authenticated=True,
            expected_statuses=frozenset({200, 201}),
            max_bytes=_MAX_JSON_BYTES,
        )

    def draft_files(
        self,
        deposition: ZenodoDeposition,
    ) -> tuple[ZenodoRemoteFile, ...]:
        payload = self._json(
            "GET",
            deposition.self_uri,
            expected_statuses=frozenset({200}),
        )
        if (
            payload.get("id") != deposition.record_id
            or payload.get("submitted") is not False
            or payload.get("state") != "unsubmitted"
            or _canonical_utc("Zenodo draft created timestamp", payload.get("created"))
            != deposition.created_at_utc
        ):
            raise PostOnlineCompletionError("Zenodo draft changed its identity or creation time")
        return self._files(payload)

    def publish_once(self, deposition: ZenodoDeposition) -> None:
        self._json(
            "POST",
            deposition.publish_uri,
            payload={},
            expected_statuses=frozenset({200, 201, 202}),
        )

    def public_record(
        self,
        deposition: ZenodoDeposition,
    ) -> ZenodoPublishedRecord | None:
        try:
            payload = self._json(
                "GET",
                f"{ZENODO_API_ORIGIN}/api/records/{deposition.record_id}",
                expected_statuses=frozenset({200}),
                authenticated=False,
            )
        except _NotFound:
            return None
        return _published_record_from_payload(
            payload,
            record_id=deposition.record_id,
            expected_created_at_utc=deposition.created_at_utc,
        )

    def anonymous_read(self, uri: str, *, max_bytes: int) -> bytes:
        if (
            type(max_bytes) is not int
            or max_bytes <= 0
            or max_bytes > MAX_EXTERNAL_ANCHOR_RECORD_BYTES
        ):
            raise PostOnlineCompletionError("anonymous readback byte limit is invalid")
        return self._request(
            "GET",
            uri,
            authenticated=False,
            expected_statuses=frozenset({200}),
            max_bytes=max_bytes,
        )


def _published_record_from_payload(
    payload: Mapping[str, Any],
    *,
    record_id: int,
    expected_created_at_utc: str,
) -> ZenodoPublishedRecord:
    if payload.get("id") != record_id:
        raise PostOnlineCompletionError("public Zenodo record changed its ID")
    created_at_utc = _canonical_utc(
        "public Zenodo created timestamp",
        payload.get("created"),
    )
    if created_at_utc != _canonical_utc(
        "expected Zenodo deposition creation",
        expected_created_at_utc,
    ):
        raise PostOnlineCompletionError("public Zenodo record changed its creation time")
    files = ZenodoCompletionAnchorPublisher._files(payload)
    raw_files = payload.get("files")
    assert isinstance(raw_files, list)
    by_name = {
        value.get("filename", value.get("key")): value
        for value in raw_files
        if isinstance(value, Mapping)
    }
    for row in files:
        raw = by_name[row.filename]
        links = raw.get("links")
        expected_uri = f"{ZENODO_API_ORIGIN}/api/records/{record_id}/files/{row.filename}/content"
        if not isinstance(links, Mapping) or links.get("self") != expected_uri:
            raise PostOnlineCompletionError(
                f"public Zenodo file {row.filename} changes its content URI"
            )
    return ZenodoPublishedRecord(
        record_id=record_id,
        created_at_utc=created_at_utc,
        updated_at_utc=_canonical_utc(
            "public Zenodo updated timestamp",
            payload.get("updated"),
        ),
        files=files,
    )


class ZenodoAnonymousCompletionAnchorReader:
    """Certificate-validated tokenless reader for one existing public record."""

    def __init__(
        self,
        *,
        opener: Any | None = None,
        timeout_seconds: float = _HTTP_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise PostOnlineCompletionError("anonymous Zenodo timeout must be positive")
        self._opener = _verified_opener() if opener is None else opener
        self._timeout_seconds = float(timeout_seconds)

    def __repr__(self) -> str:
        return f"ZenodoAnonymousCompletionAnchorReader(timeout_seconds={self._timeout_seconds})"

    def _get(self, uri: str, *, max_bytes: int, accept: str) -> bytes:
        fixed_uri = _https_uri(
            uri,
            label="anonymous Zenodo request URI",
            zenodo_only=True,
        )
        if type(max_bytes) is not int or max_bytes <= 0:
            raise PostOnlineCompletionError("anonymous Zenodo byte limit must be positive")
        request = urllib_request.Request(
            fixed_uri,
            headers={
                "Accept": accept,
                "Accept-Encoding": "identity",
                "User-Agent": ("fractal-ann-diagnostics/0.3 completion-anchor-revalidation"),
            },
            method="GET",
        )
        try:
            with self._opener.open(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                if response.getcode() != 200:
                    raise PostOnlineCompletionError(
                        "anonymous Zenodo read returned an unexpected HTTP status"
                    )
                if response.geturl() != fixed_uri:
                    raise PostOnlineCompletionError("anonymous Zenodo response URL changed")
                content_encoding = response.headers.get("Content-Encoding")
                if content_encoding not in {None, "", "identity"}:
                    raise PostOnlineCompletionError(
                        "anonymous Zenodo response used content encoding"
                    )
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdecimal() or int(length) > max_bytes):
                    raise PostOnlineCompletionError(
                        "anonymous Zenodo Content-Length is invalid or excessive"
                    )
                encoded = response.read(max_bytes + 1)
        except PostOnlineCompletionError:
            raise
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                raise _NotFound from None
            if 300 <= exc.code < 400:
                raise PostOnlineCompletionError("anonymous Zenodo redirect was refused") from None
            raise PostOnlineCompletionError(
                f"anonymous Zenodo returned HTTP status {exc.code}"
            ) from None
        except (urllib_error.URLError, TimeoutError, ssl.SSLError, OSError):
            raise PostOnlineCompletionError("anonymous Zenodo HTTPS request failed") from None
        if len(encoded) > max_bytes:
            raise PostOnlineCompletionError("anonymous Zenodo response exceeds the byte limit")
        if length is not None and len(encoded) != int(length):
            raise PostOnlineCompletionError(
                "anonymous Zenodo response differs from its Content-Length"
            )
        return encoded

    def public_record(
        self,
        *,
        record_id: int,
        expected_created_at_utc: str,
    ) -> ZenodoPublishedRecord | None:
        identifier = _positive_integer("Zenodo record_id", record_id)
        uri = f"{ZENODO_API_ORIGIN}/api/records/{identifier}"
        try:
            encoded = self._get(
                uri,
                max_bytes=_MAX_JSON_BYTES,
                accept="application/json",
            )
        except _NotFound:
            return None
        return _published_record_from_payload(
            _closed_json(encoded, label="anonymous public Zenodo record"),
            record_id=identifier,
            expected_created_at_utc=expected_created_at_utc,
        )

    def anonymous_read(self, uri: str, *, max_bytes: int) -> bytes:
        if (
            type(max_bytes) is not int
            or max_bytes <= 0
            or max_bytes > MAX_EXTERNAL_ANCHOR_RECORD_BYTES
        ):
            raise PostOnlineCompletionError("anonymous anchor readback byte limit is invalid")
        return self._get(
            uri,
            max_bytes=max_bytes,
            accept="application/octet-stream",
        )


class DrandRoundPublicationGuard:
    """Fail-closed HTTPS check for the registered drand release round."""

    def __init__(
        self,
        *,
        opener: Any | None = None,
        timeout_seconds: float = _HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self._opener = _verified_opener() if opener is None else opener
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise PostOnlineCompletionError("drand timeout must be positive")
        self._timeout_seconds = float(timeout_seconds)

    def assert_not_public(self, beacon: ExecutionBeaconContract) -> None:
        if not isinstance(beacon, ExecutionBeaconContract):
            raise PostOnlineCompletionError("drand guard requires an execution beacon")
        network = urlsplit(beacon.drand_network)
        if (
            network.scheme != "https"
            or not network.hostname
            or network.username is not None
            or network.password is not None
            or network.query
            or network.fragment
            or network.path not in {"", "/"}
        ):
            raise PostOnlineCompletionError("registered drand network URI is invalid")
        uri = (
            f"{beacon.drand_network.rstrip('/')}/{beacon.chain_hash}"
            f"/public/{beacon.label_release_round}"
        )
        request = urllib_request.Request(
            uri,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "fractal-ann-diagnostics/0.3 round-guard",
            },
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=self._timeout_seconds) as response:
                if response.geturl() != uri:
                    raise PostOnlineCompletionError("drand response URL changed")
                if response.getcode() != 200:
                    raise PostOnlineCompletionError(
                        f"drand returned unexpected HTTP status {response.getcode()}"
                    )
                content_encoding = response.headers.get("Content-Encoding")
                if content_encoding not in {None, "", "identity"}:
                    raise PostOnlineCompletionError("drand response used content encoding")
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdecimal() or int(length) > _MAX_JSON_BYTES):
                    raise PostOnlineCompletionError(
                        "drand response Content-Length is invalid or excessive"
                    )
                encoded = response.read(_MAX_JSON_BYTES + 1)
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                return
            if 300 <= exc.code < 400:
                raise PostOnlineCompletionError("drand redirect was refused") from None
            raise PostOnlineCompletionError(
                f"drand returned unexpected HTTP status {exc.code}"
            ) from None
        except PostOnlineCompletionError:
            raise
        except (urllib_error.URLError, TimeoutError, ssl.SSLError, OSError):
            raise PostOnlineCompletionError("drand HTTPS request failed") from None
        if len(encoded) > _MAX_JSON_BYTES:
            raise PostOnlineCompletionError("drand response exceeds the byte limit")
        if length is not None and len(encoded) != int(length):
            raise PostOnlineCompletionError("drand response differs from its Content-Length")
        payload = _closed_json(encoded, label="drand response")
        if payload.get("round") != beacon.label_release_round:
            raise PostOnlineCompletionError("drand response changed the requested round")
        raise PostOnlineCompletionError("registered label-release drand round is already public")


@dataclass(frozen=True)
class PostOnlineCompletionBinding:
    corpus: str
    prediction_completion_filename: str
    prediction_completion_receipt_sha256: str
    prediction_completion_file_sha256: str
    anchor_record_filename: str
    anchor_record_sha256: str
    anchor_receipt_filename: str
    anchor_receipt_sha256: str
    anchor_receipt_file_sha256: str
    external_anchor_uri: str
    schema_version: str = POST_ONLINE_COMPLETION_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.corpus not in FIXED_CORPORA:
            raise PostOnlineCompletionError("aggregate binding names another corpus")
        for name in (
            "prediction_completion_receipt_sha256",
            "prediction_completion_file_sha256",
            "anchor_record_sha256",
            "anchor_receipt_sha256",
            "anchor_receipt_file_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "prediction_completion_filename",
            "anchor_record_filename",
            "anchor_receipt_filename",
        ):
            value = _require_text(name, getattr(self, name))
            if "/" in value or quote(value, safe="") != value:
                raise PostOnlineCompletionError(f"{name} must be one safe segment")
        _https_uri(self.external_anchor_uri, label="external_anchor_uri", zenodo_only=True)
        if self.schema_version != POST_ONLINE_COMPLETION_BINDING_SCHEMA:
            raise PostOnlineCompletionError("completion binding schema differs")

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> PostOnlineCompletionBinding:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="post-online completion binding",
            )
        )


@dataclass(frozen=True)
class PostOnlineCompletionAggregateReceipt:
    suite_attempt_id: str
    manifest_sha256: str
    run_receipt_sha256: str
    online_complete_state_sha256: str
    online_output_aggregate_sha256: str
    online_attestation_descriptor_sha256: str
    online_attestation_bundle_sha256: str
    label_release_claim_state_sha256: str
    label_release_claim_ledger_commit: str
    zenodo_record_id: int
    zenodo_deposition_created_at_utc: str
    zenodo_public_record_updated_at_utc: str
    zenodo_record_uri: str
    label_release_round: int
    label_release_beacon_contract_sha256: str
    label_release_publication_time_utc: str
    bindings: tuple[PostOnlineCompletionBinding, ...]
    completion_receipt_timestamp_semantics: str = COMPLETION_RECEIPT_TIMESTAMP_SEMANTICS
    schema_version: str = POST_ONLINE_COMPLETION_AGGREGATE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "suite_attempt_id",
            "manifest_sha256",
            "run_receipt_sha256",
            "online_complete_state_sha256",
            "online_output_aggregate_sha256",
            "online_attestation_descriptor_sha256",
            "online_attestation_bundle_sha256",
            "label_release_claim_state_sha256",
            "label_release_beacon_contract_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_text(
            "label_release_claim_ledger_commit",
            self.label_release_claim_ledger_commit,
        )
        if _GIT_COMMIT.fullmatch(self.label_release_claim_ledger_commit) is None:
            raise PostOnlineCompletionError(
                "label_release_claim_ledger_commit must be one full Git commit"
            )
        _positive_integer("zenodo_record_id", self.zenodo_record_id)
        _positive_integer("label_release_round", self.label_release_round)
        created = _canonical_utc(
            "zenodo_deposition_created_at_utc",
            self.zenodo_deposition_created_at_utc,
        )
        updated = _canonical_utc(
            "zenodo_public_record_updated_at_utc",
            self.zenodo_public_record_updated_at_utc,
        )
        release = _canonical_utc(
            "label_release_publication_time_utc",
            self.label_release_publication_time_utc,
        )
        object.__setattr__(
            self,
            "zenodo_deposition_created_at_utc",
            created,
        )
        object.__setattr__(
            self,
            "zenodo_public_record_updated_at_utc",
            updated,
        )
        object.__setattr__(self, "label_release_publication_time_utc", release)
        if not (
            _utc("zenodo deposition creation", created)
            < _utc("Zenodo public record update", updated)
            < _utc("label release publication", release)
        ):
            raise PostOnlineCompletionError(
                "Zenodo public record update must follow deposition creation "
                "and strictly precede label release"
            )
        expected_record_uri = f"{ZENODO_API_ORIGIN}/api/records/{self.zenodo_record_id}"
        if (
            _https_uri(
                self.zenodo_record_uri,
                label="zenodo_record_uri",
                zenodo_only=True,
            )
            != expected_record_uri
        ):
            raise PostOnlineCompletionError("aggregate Zenodo URI changes its record ID")
        rows = tuple(self.bindings)
        if tuple(row.corpus for row in rows) != _CANONICAL_CORPORA:
            raise PostOnlineCompletionError(
                "aggregate must bind each fixed corpus once in UTF-8 byte order"
            )
        for row in rows:
            expected_completion = f"{row.corpus}-prediction-completion.json"
            expected_record = _anchor_filename(row.corpus)
            expected_receipt = f"{row.corpus}-prediction-completion-anchor-receipt.json"
            expected_uri = (
                f"{ZENODO_API_ORIGIN}/api/records/{self.zenodo_record_id}"
                f"/files/{expected_record}/content"
            )
            if (
                row.prediction_completion_filename != expected_completion
                or row.anchor_record_filename != expected_record
                or row.anchor_receipt_filename != expected_receipt
                or row.external_anchor_uri != expected_uri
            ):
                raise PostOnlineCompletionError(
                    f"aggregate changes the fixed completion paths for {row.corpus}"
                )
        if self.schema_version != POST_ONLINE_COMPLETION_AGGREGATE_SCHEMA:
            raise PostOnlineCompletionError("aggregate receipt schema differs")
        if self.completion_receipt_timestamp_semantics != COMPLETION_RECEIPT_TIMESTAMP_SEMANTICS:
            raise PostOnlineCompletionError(
                "aggregate misstates the per-corpus completion timestamp"
            )
        object.__setattr__(self, "bindings", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "bindings"
            },
            "bindings": [row.to_dict() for row in self.bindings],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(
        cls,
        value: object,
    ) -> PostOnlineCompletionAggregateReceipt:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="post-online completion aggregate receipt",
        )
        bindings = row["bindings"]
        if type(bindings) is not list:
            raise PostOnlineCompletionError(
                "post-online completion aggregate bindings must be an array"
            )
        return cls(
            **{key: item for key, item in row.items() if key != "bindings"},
            bindings=tuple(PostOnlineCompletionBinding.from_dict(item) for item in bindings),
        )


def load_post_online_completion_aggregate_receipt(
    path: str | Path,
) -> PostOnlineCompletionAggregateReceipt:
    """Load and revalidate the closed completion aggregate through secure I/O."""

    try:
        encoded = read_secure_control_file(
            path,
            label="post-online completion aggregate receipt",
        )
    except ArtifactIntegrityError as exc:
        raise PostOnlineCompletionError(
            "cannot read post-online completion aggregate safely"
        ) from exc
    receipt = PostOnlineCompletionAggregateReceipt.from_dict(
        _closed_json(encoded, label="post-online completion aggregate receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise PostOnlineCompletionError("post-online completion aggregate bytes are not canonical")
    return receipt


def verify_post_online_completion_directory(
    root: str | Path,
) -> PostOnlineCompletionAggregateReceipt:
    """Verify the exact sixteen-file local completion closure."""

    directory = Path(root)
    if not directory.is_absolute():
        raise PostOnlineCompletionError("completion directory must be absolute")
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise PostOnlineCompletionError("cannot inspect completion directory") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise PostOnlineCompletionError("completion directory is not private")
    aggregate = load_post_online_completion_aggregate_receipt(
        directory / POST_ONLINE_COMPLETION_AGGREGATE_FILENAME
    )
    expected_names = {
        POST_ONLINE_COMPLETION_AGGREGATE_FILENAME,
        *(
            name
            for row in aggregate.bindings
            for name in (
                row.prediction_completion_filename,
                row.anchor_record_filename,
                row.anchor_receipt_filename,
            )
        ),
    }
    if {path.name for path in directory.iterdir()} != expected_names:
        raise PostOnlineCompletionError(
            "completion directory does not contain its exact sixteen files"
        )
    for binding in aggregate.bindings:
        try:
            completion = load_prediction_completion_receipt(
                directory / binding.prediction_completion_filename
            )
            record = load_prediction_completion_anchor_record(
                directory / binding.anchor_record_filename
            )
            receipt = load_prediction_completion_anchor_receipt(
                directory / binding.anchor_receipt_filename
            )
            completion_file_sha256 = digest_regular_file(
                directory / binding.prediction_completion_filename,
                label=f"{binding.corpus} prediction completion receipt",
            )
            receipt_file_sha256 = digest_regular_file(
                directory / binding.anchor_receipt_filename,
                label=f"{binding.corpus} prediction completion anchor receipt",
            )
        except Exception as exc:
            raise PostOnlineCompletionError(
                f"cannot verify local completion evidence for {binding.corpus}"
            ) from exc
        expected_record = PredictionCompletionAnchorRecord.from_completion_receipt(completion)
        expected_receipt = PredictionCompletionAnchorReceipt.from_record(record)
        if (
            completion.corpus != binding.corpus
            or completion.manifest_sha256 != aggregate.manifest_sha256
            or completion.run_receipt_sha256 != aggregate.run_receipt_sha256
            or completion.external_anchor_identity != f"zenodo-record:{aggregate.zenodo_record_id}"
            or completion.external_anchor_uri != binding.external_anchor_uri
            or completion.anchored_at_utc != aggregate.zenodo_deposition_created_at_utc
            or completion.receipt_sha256 != binding.prediction_completion_receipt_sha256
            or completion_file_sha256 != binding.prediction_completion_file_sha256
            or record != expected_record
            or record.record_sha256 != binding.anchor_record_sha256
            or receipt != expected_receipt
            or receipt.receipt_sha256 != binding.anchor_receipt_sha256
            or receipt_file_sha256 != binding.anchor_receipt_file_sha256
        ):
            raise PostOnlineCompletionError(
                f"local completion evidence differs for {binding.corpus}"
            )
    if {path.name for path in directory.iterdir()} != expected_names:
        raise PostOnlineCompletionError("completion directory changed during verification")
    return aggregate


_VERIFIED_POST_ONLINE_ANCHORS_CAPABILITY = object()


@dataclass(frozen=True)
class VerifiedPostOnlineCompletionAnchors:
    """Capability returned only after fresh anonymous public byte revalidation."""

    completion_root: Path
    aggregate: PostOnlineCompletionAggregateReceipt
    anchors: tuple[VerifiedPredictionCompletionAnchor, ...]
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _VERIFIED_POST_ONLINE_ANCHORS_CAPABILITY:
            raise PostOnlineCompletionError(
                "verified completion anchors can only come from anonymous revalidation"
            )
        root = Path(self.completion_root)
        if not root.is_absolute() or root.name != "completion":
            raise PostOnlineCompletionError(
                "verified completion root must be one absolute completion directory"
            )
        if not isinstance(
            self.aggregate,
            PostOnlineCompletionAggregateReceipt,
        ):
            raise PostOnlineCompletionError("verified completion anchors require a typed aggregate")
        anchors = tuple(self.anchors)
        if not all(isinstance(anchor, VerifiedPredictionCompletionAnchor) for anchor in anchors):
            raise PostOnlineCompletionError(
                "verified completion anchors must contain typed anchor pairs"
            )
        rows = tuple(anchor.record for anchor in anchors)
        if tuple(row.corpus for row in rows) != _CANONICAL_CORPORA or not all(
            isinstance(row, PredictionCompletionAnchorRecord) for row in rows
        ):
            raise PostOnlineCompletionError(
                "verified completion anchors must contain each fixed corpus once"
            )
        bindings = {row.corpus: row for row in self.aggregate.bindings}
        if any(
            row.record_sha256 != bindings[row.corpus].anchor_record_sha256
            or row.external_anchor_uri != bindings[row.corpus].external_anchor_uri
            for row in rows
        ):
            raise PostOnlineCompletionError("verified completion records differ from the aggregate")
        object.__setattr__(self, "completion_root", root)
        object.__setattr__(self, "anchors", anchors)

    @property
    def records(self) -> tuple[PredictionCompletionAnchorRecord, ...]:
        return tuple(anchor.record for anchor in self.anchors)

    def anchor_for(self, corpus_id: str) -> VerifiedPredictionCompletionAnchor:
        if corpus_id not in FIXED_CORPORA:
            raise PostOnlineCompletionError("anchor lookup names another corpus")
        matches = [anchor for anchor in self.anchors if anchor.record.corpus == corpus_id]
        if len(matches) != 1:
            raise PostOnlineCompletionError(f"verified anchor set lacks one record for {corpus_id}")
        return matches[0]


_VERIFIED_POST_ONLINE_AUTHORITY_CAPABILITY = object()


@dataclass(frozen=True)
class VerifiedPostOnlineCompletionAuthority:
    """Provider-bound release authority minted after anonymous public readback."""

    completion: VerifiedPostOnlineCompletionAnchors
    provider_namespace: Path
    phase_claim_state_sha256: str
    phase_claim_ledger_commit: str
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _VERIFIED_POST_ONLINE_AUTHORITY_CAPABILITY:
            raise PostOnlineCompletionError(
                "post-online release authority can only come from provider verification"
            )
        if not isinstance(self.completion, VerifiedPostOnlineCompletionAnchors):
            raise PostOnlineCompletionError(
                "post-online release authority lacks anonymous completion evidence"
            )
        namespace = Path(self.provider_namespace)
        if (
            not namespace.is_absolute()
            or self.completion.completion_root != namespace / "completion"
        ):
            raise PostOnlineCompletionError(
                "post-online release authority changes the provider namespace"
            )
        _require_sha256(
            "phase_claim_state_sha256",
            self.phase_claim_state_sha256,
        )
        if _GIT_COMMIT.fullmatch(self.phase_claim_ledger_commit) is None:
            raise PostOnlineCompletionError("phase_claim_ledger_commit must be one full Git commit")
        if (
            self.completion.aggregate.label_release_claim_state_sha256
            != self.phase_claim_state_sha256
            or self.completion.aggregate.label_release_claim_ledger_commit
            != self.phase_claim_ledger_commit
        ):
            raise PostOnlineCompletionError("post-online release authority changes the phase claim")
        object.__setattr__(self, "provider_namespace", namespace)

    @property
    def completion_root(self) -> Path:
        return self.completion.completion_root

    @property
    def aggregate(self) -> PostOnlineCompletionAggregateReceipt:
        return self.completion.aggregate

    def anchor_for(self, corpus_id: str) -> VerifiedPredictionCompletionAnchor:
        return self.completion.anchor_for(corpus_id)


def revalidate_post_online_completion_anchors(
    completion_root: str | Path,
    *,
    reader: AnonymousCompletionAnchorReader | None = None,
) -> VerifiedPostOnlineCompletionAnchors:
    """Freshly verify the public Zenodo record and all five local record bytes."""

    root = Path(completion_root)
    aggregate = verify_post_online_completion_directory(root)
    active_reader = ZenodoAnonymousCompletionAnchorReader() if reader is None else reader
    try:
        public = active_reader.public_record(
            record_id=aggregate.zenodo_record_id,
            expected_created_at_utc=(aggregate.zenodo_deposition_created_at_utc),
        )
    except PostOnlineCompletionError:
        raise
    except Exception as exc:
        raise PostOnlineCompletionError(
            "anonymous Zenodo public-record revalidation failed"
        ) from exc
    if public is None:
        raise PostOnlineCompletionError("anchored Zenodo record is not anonymously public")
    if (
        public.record_id != aggregate.zenodo_record_id
        or public.created_at_utc != aggregate.zenodo_deposition_created_at_utc
        or public.updated_at_utc != aggregate.zenodo_public_record_updated_at_utc
    ):
        raise PostOnlineCompletionError(
            "live Zenodo identity or updated timestamp differs from the aggregate"
        )

    expected_bytes: dict[str, bytes] = {}
    anchors: list[VerifiedPredictionCompletionAnchor] = []
    for binding in aggregate.bindings:
        path = root / binding.anchor_record_filename
        try:
            encoded = read_secure_regular_file(
                path,
                max_bytes=MAX_EXTERNAL_ANCHOR_RECORD_BYTES,
                label=f"{binding.corpus} local completion anchor record",
            )
            record = load_prediction_completion_anchor_record(path)
            receipt = load_prediction_completion_anchor_receipt(
                root / binding.anchor_receipt_filename
            )
        except Exception as exc:
            raise PostOnlineCompletionError(
                f"cannot re-read local anchor record for {binding.corpus}"
            ) from exc
        if (
            _sha256(encoded) != binding.anchor_record_sha256
            or record.record_sha256 != binding.anchor_record_sha256
        ):
            raise PostOnlineCompletionError(f"local anchor record changed for {binding.corpus}")
        expected_bytes[binding.anchor_record_filename] = encoded
        anchors.append(
            VerifiedPredictionCompletionAnchor(
                record=record,
                receipt=receipt,
            )
        )
    if len(expected_bytes) != len(FIXED_CORPORA):
        raise PostOnlineCompletionError(
            "local completion closure does not contain exactly five anchor records"
        )
    if any(row.checksum is None for row in public.files):
        raise PostOnlineCompletionError("public Zenodo inventory lacks a checksum")
    _assert_remote_inventory(
        public.files,
        expected_bytes,
        label="live public Zenodo inventory",
    )
    for binding in aggregate.bindings:
        expected = expected_bytes[binding.anchor_record_filename]
        try:
            observed = active_reader.anonymous_read(
                binding.external_anchor_uri,
                max_bytes=MAX_EXTERNAL_ANCHOR_RECORD_BYTES,
            )
        except PostOnlineCompletionError:
            raise
        except Exception as exc:
            raise PostOnlineCompletionError(
                f"anonymous anchor byte read failed for {binding.corpus}"
            ) from exc
        if observed != expected:
            raise PostOnlineCompletionError(f"live public anchor bytes differ for {binding.corpus}")
    try:
        stable_public = active_reader.public_record(
            record_id=aggregate.zenodo_record_id,
            expected_created_at_utc=(aggregate.zenodo_deposition_created_at_utc),
        )
    except PostOnlineCompletionError:
        raise
    except Exception as exc:
        raise PostOnlineCompletionError(
            "anonymous Zenodo public-record stability check failed"
        ) from exc
    if stable_public != public:
        raise PostOnlineCompletionError("live public Zenodo record changed during anchor readback")
    if verify_post_online_completion_directory(root) != aggregate:
        raise PostOnlineCompletionError(
            "local completion closure changed during anonymous revalidation"
        )
    return VerifiedPostOnlineCompletionAnchors(
        completion_root=root,
        aggregate=aggregate,
        anchors=tuple(anchors),
        _capability=_VERIFIED_POST_ONLINE_ANCHORS_CAPABILITY,
    )


def revalidate_post_online_completion_authority(
    predecessor: VerifiedProviderPredecessor,
    phase_claim: VerifiedPhaseClaimCapability,
    *,
    reader: AnonymousCompletionAnchorReader | None = None,
) -> VerifiedPostOnlineCompletionAuthority:
    """Bind anonymous public anchors to the live label-release authority."""

    if not isinstance(predecessor, VerifiedProviderPredecessor):
        raise PostOnlineCompletionError(
            "post-online release requires a verified provider predecessor"
        )
    if not isinstance(phase_claim, VerifiedPhaseClaimCapability):
        raise PostOnlineCompletionError("post-online release requires a verified phase capability")
    _assert_predecessor_current(predecessor)
    try:
        phase_claim.assert_current()
    except ExecutionClaimError as exc:
        raise PostOnlineCompletionError(
            "label-release phase capability is no longer current"
        ) from exc
    suite = _admit_post_online_suite(predecessor)
    claim = suite.claim_record
    online = suite.online_record
    claim_payload = claim.payload
    online_payload = online.payload
    if not isinstance(claim_payload, PhaseClaimBindings) or not isinstance(
        online_payload,
        OnlineSuiteClosure,
    ):
        raise PostOnlineCompletionError("provider completion states carry untyped payloads")
    beacon = phase_claim.contract.label_release_beacon
    online_evidence = predecessor.evidences[online.sequence]
    if (
        phase_claim.contract.phase != LABEL_RELEASE_PHASE
        or not isinstance(beacon, ExecutionBeaconContract)
        or claim_payload.phase_claim != phase_claim.contract
        or claim_payload.provider_identity != phase_claim.provider_identity
        or claim.record_sha256 != phase_claim.phase_claim_state_sha256
        or predecessor.ledger_commit != phase_claim.phase_claim_ledger_commit
    ):
        raise PostOnlineCompletionError(
            "post-online completion belongs to another label-release claim"
        )

    verified = revalidate_post_online_completion_anchors(
        suite.completion_root,
        reader=reader,
    )
    aggregate = verified.aggregate
    expected = (
        claim.suite_attempt_id,
        claim.manifest_sha256,
        claim.run_receipt_sha256,
        online.record_sha256,
        online_payload.run_output_aggregate.aggregate_sha256,
        online_evidence.descriptor_sha256,
        online_evidence.bundle_sha256,
        claim.record_sha256,
        predecessor.ledger_commit,
        beacon.label_release_round,
        beacon.contract_sha256,
        beacon.label_release_publication_time.isoformat(),
    )
    observed = (
        aggregate.suite_attempt_id,
        aggregate.manifest_sha256,
        aggregate.run_receipt_sha256,
        aggregate.online_complete_state_sha256,
        aggregate.online_output_aggregate_sha256,
        aggregate.online_attestation_descriptor_sha256,
        aggregate.online_attestation_bundle_sha256,
        aggregate.label_release_claim_state_sha256,
        aggregate.label_release_claim_ledger_commit,
        aggregate.label_release_round,
        aggregate.label_release_beacon_contract_sha256,
        aggregate.label_release_publication_time_utc,
    )
    if observed != expected:
        raise PostOnlineCompletionError(
            "post-online completion aggregate differs from provider authority"
        )
    if verified.completion_root != predecessor.namespace / "completion":
        raise PostOnlineCompletionError(
            "post-online completion root differs from provider authority"
        )
    _assert_predecessor_current(predecessor)
    try:
        phase_claim.assert_current()
    except ExecutionClaimError as exc:
        raise PostOnlineCompletionError(
            "label-release phase capability changed during anchor revalidation"
        ) from exc
    return VerifiedPostOnlineCompletionAuthority(
        completion=verified,
        provider_namespace=predecessor.namespace,
        phase_claim_state_sha256=phase_claim.phase_claim_state_sha256,
        phase_claim_ledger_commit=phase_claim.phase_claim_ledger_commit,
        _capability=_VERIFIED_POST_ONLINE_AUTHORITY_CAPABILITY,
    )


def _anchor_filename(corpus_id: str) -> str:
    return f"{corpus_id}-prediction-completion-anchor.json"


def _metadata(
    suite: AdmittedPostOnlineSuite,
    *,
    created_at_utc: datetime,
) -> dict[str, object]:
    return {
        "access_right": "open",
        "creators": [
            {
                "name": ZENODO_CREATOR_NAME,
                "orcid": ZENODO_CREATOR_ORCID,
            }
        ],
        "description": (
            "Exact prediction-completion anchor records for suite attempt "
            f"{suite.claim_record.suite_attempt_id}."
        ),
        "keywords": [
            "approximate-nearest-neighbor",
            "prediction-completion",
            "registered-analysis",
        ],
        "license": "cc-by-4.0",
        "publication_date": created_at_utc.date().isoformat(),
        "title": (
            f"Fractal ANN prediction-completion anchors {suite.claim_record.suite_attempt_id}"
        ),
        "upload_type": "dataset",
    }


def _private_completion_directory(path: Path) -> bool:
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        metadata = path.lstat()
        created = True
    except FileExistsError:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise PostOnlineCompletionError("cannot inspect completion directory") from exc
        created = False
    except OSError as exc:
        raise PostOnlineCompletionError("cannot create completion directory") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise PostOnlineCompletionError("completion directory is not private")
    return created


def _assert_remote_inventory(
    observed: Sequence[ZenodoRemoteFile],
    expected: Mapping[str, bytes],
    *,
    label: str,
) -> None:
    rows = tuple(observed)
    by_name = {row.filename: row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != set(expected):
        raise PostOnlineCompletionError(f"{label} does not contain the exact five records")
    for filename, encoded in expected.items():
        remote = by_name[filename]
        if remote.byte_count != len(encoded):
            raise PostOnlineCompletionError(f"{label} changed byte count for {filename}")
        if remote.checksum is not None:
            expected_checksums = {
                hashlib.md5(encoded, usedforsecurity=False).hexdigest(),
                f"md5:{hashlib.md5(encoded, usedforsecurity=False).hexdigest()}",
                f"sha256:{_sha256(encoded)}",
            }
            if remote.checksum not in expected_checksums:
                raise PostOnlineCompletionError(f"{label} changed checksum for {filename}")


@dataclass(frozen=True)
class _CompletionMaterials:
    record_id: int
    created_at_utc: str
    completions: Mapping[str, PredictionCompletionReceipt]
    records: Mapping[str, PredictionCompletionAnchorRecord]
    record_bytes: Mapping[str, bytes]


def _completion_input_names() -> frozenset[str]:
    return frozenset(
        name
        for corpus_id in _CANONICAL_CORPORA
        for name in (
            f"{corpus_id}-prediction-completion.json",
            _anchor_filename(corpus_id),
        )
    )


def _completion_receipt_names() -> frozenset[str]:
    return frozenset(
        f"{corpus_id}-prediction-completion-anchor-receipt.json" for corpus_id in _CANONICAL_CORPORA
    )


def _create_completion_materials(
    suite: AdmittedPostOnlineSuite,
    *,
    record_id: int,
    created_at_utc: str,
    require_existing: bool,
) -> _CompletionMaterials:
    identifier = _positive_integer("Zenodo recovery record ID", record_id)
    anchored_at = _canonical_utc("Zenodo recovery creation time", created_at_utc)
    completions: dict[str, PredictionCompletionReceipt] = {}
    records: dict[str, PredictionCompletionAnchorRecord] = {}
    record_bytes: dict[str, bytes] = {}
    for source in suite.sources:
        corpus_id = source.corpus_id
        record_filename = _anchor_filename(corpus_id)
        content_uri = (
            f"{ZENODO_API_ORIGIN}/api/records/{identifier}/files/{record_filename}/content"
        )
        try:
            panel_binding = source.action_panel.completion_binding()
            completion = create_prediction_completion_receipt(
                source.predictions,
                execution=source.execution,
                receipt=suite.sealed_run,
                manifest_sha256=suite.manifest_digest,
                action_panel_binding=panel_binding,
                online_execution_result_receipt=source.online_result,
                external_anchor_identity=f"zenodo-record:{identifier}",
                external_anchor_uri=content_uri,
                anchored_at_utc=anchored_at,
            )
            record = PredictionCompletionAnchorRecord.from_completion_receipt(completion)
        except Exception as exc:
            raise PostOnlineCompletionError(
                f"cannot create prediction completion for {corpus_id}"
            ) from exc
        if completion.corpus != corpus_id or record.corpus != corpus_id:
            raise PostOnlineCompletionError("completion factory crossed corpus identities")
        completion_path = suite.completion_root / f"{corpus_id}-prediction-completion.json"
        record_path = suite.completion_root / record_filename
        try:
            if require_existing:
                existing_completion = load_prediction_completion_receipt(completion_path)
                existing_record = load_prediction_completion_anchor_record(record_path)
                if existing_completion != completion or existing_record != record:
                    raise PostOnlineCompletionError(
                        f"recoverable completion inputs differ for {corpus_id}"
                    )
            else:
                write_prediction_completion_receipt(completion, completion_path)
                write_prediction_completion_anchor_record(record, record_path)
        except PostOnlineCompletionError:
            raise
        except Exception as exc:
            action = "verify" if require_existing else "persist"
            raise PostOnlineCompletionError(
                f"cannot {action} completion anchor inputs for {corpus_id}"
            ) from exc
        completions[corpus_id] = completion
        records[corpus_id] = record
        record_bytes[record_filename] = record.canonical_bytes() + b"\n"
    if (
        tuple(completions) != _CANONICAL_CORPORA
        or tuple(records) != _CANONICAL_CORPORA
        or len(record_bytes) != len(_CANONICAL_CORPORA)
    ):
        raise PostOnlineCompletionError(
            "completion materials do not bind the exact five canonical corpora"
        )
    return _CompletionMaterials(
        record_id=identifier,
        created_at_utc=anchored_at,
        completions=completions,
        records=records,
        record_bytes=record_bytes,
    )


def _infer_recovery_identity(
    suite: AdmittedPostOnlineSuite,
) -> tuple[int, str, frozenset[str]]:
    try:
        observed_names = frozenset(path.name for path in suite.completion_root.iterdir())
    except OSError as exc:
        raise PostOnlineCompletionError(
            "cannot inspect the interrupted completion directory"
        ) from exc
    required = _completion_input_names()
    receipt_names = _completion_receipt_names()
    if (
        POST_ONLINE_COMPLETION_AGGREGATE_FILENAME in observed_names
        or not required.issubset(observed_names)
        or not observed_names.issubset(required | receipt_names)
    ):
        raise PostOnlineCompletionError(
            "existing completion directory is not an exact recoverable publication closure"
        )

    identities: set[tuple[int, str]] = set()
    for corpus_id in _CANONICAL_CORPORA:
        try:
            completion = load_prediction_completion_receipt(
                suite.completion_root / f"{corpus_id}-prediction-completion.json"
            )
        except Exception as exc:
            raise PostOnlineCompletionError(
                f"cannot inspect interrupted completion input for {corpus_id}"
            ) from exc
        prefix = "zenodo-record:"
        raw_identity = completion.external_anchor_identity
        raw_identifier = raw_identity.removeprefix(prefix)
        if (
            not raw_identity.startswith(prefix)
            or not raw_identifier.isdecimal()
            or raw_identifier.startswith("0")
        ):
            raise PostOnlineCompletionError(
                f"interrupted completion changes the Zenodo identity for {corpus_id}"
            )
        identifier = int(raw_identifier)
        if identifier <= 0:
            raise PostOnlineCompletionError(
                f"interrupted completion changes the Zenodo identity for {corpus_id}"
            )
        anchored_at = _canonical_utc(
            f"{corpus_id} interrupted anchor time",
            completion.anchored_at_utc,
        )
        expected_uri = (
            f"{ZENODO_API_ORIGIN}/api/records/{identifier}/files/"
            f"{_anchor_filename(corpus_id)}/content"
        )
        if completion.corpus != corpus_id or completion.external_anchor_uri != expected_uri:
            raise PostOnlineCompletionError(
                f"interrupted completion crosses the five-anchor binding for {corpus_id}"
            )
        identities.add((identifier, anchored_at))
    if len(identities) != 1:
        raise PostOnlineCompletionError(
            "interrupted completion inputs name different Zenodo depositions"
        )
    identifier, anchored_at = identities.pop()
    if _utc("interrupted Zenodo creation", anchored_at) >= (
        suite.beacon.label_release_publication_time
    ):
        raise PostOnlineCompletionError(
            "interrupted Zenodo deposition was created at or after label release"
        )
    return identifier, anchored_at, observed_names & receipt_names


def _assert_public_snapshot(
    public: ZenodoPublishedRecord,
    materials: _CompletionMaterials,
    suite: AdmittedPostOnlineSuite,
) -> None:
    if public.record_id != materials.record_id or public.created_at_utc != materials.created_at_utc:
        raise PostOnlineCompletionError("public Zenodo record changed its deposition identity")
    if (
        _utc("public Zenodo updated timestamp", public.updated_at_utc)
        >= suite.beacon.label_release_publication_time
    ):
        raise PostOnlineCompletionError(
            "public Zenodo record was updated at or after label release"
        )
    if any(row.checksum is None for row in public.files):
        raise PostOnlineCompletionError("public Zenodo inventory lacks a checksum")
    _assert_remote_inventory(
        public.files,
        materials.record_bytes,
        label="public Zenodo inventory",
    )


def _assert_anonymous_bytes(
    reader: CompletionAnchorPublisher | AnonymousCompletionAnchorReader,
    materials: _CompletionMaterials,
) -> None:
    for filename in sorted(materials.record_bytes, key=lambda item: item.encode("utf-8")):
        expected = materials.record_bytes[filename]
        uri = f"{ZENODO_API_ORIGIN}/api/records/{materials.record_id}/files/{filename}/content"
        try:
            observed = reader.anonymous_read(
                uri,
                max_bytes=MAX_EXTERNAL_ANCHOR_RECORD_BYTES,
            )
        except PostOnlineCompletionError:
            raise
        except Exception as exc:
            raise PostOnlineCompletionError(
                f"anonymous anchor byte read failed for {filename}"
            ) from exc
        if observed != expected:
            raise PostOnlineCompletionError(
                f"anonymous Zenodo readback changed bytes for {filename}"
            )


def _aggregate_authority_values(
    suite: AdmittedPostOnlineSuite,
) -> tuple[object, ...]:
    return (
        suite.claim_record.suite_attempt_id,
        suite.manifest_digest,
        suite.claim_record.run_receipt_sha256,
        suite.online_record.record_sha256,
        suite.online_record.payload.run_output_aggregate.aggregate_sha256,
        suite.predecessor.evidences[suite.online_record.sequence].descriptor_sha256,
        suite.predecessor.evidences[suite.online_record.sequence].bundle_sha256,
        suite.claim_record.record_sha256,
        suite.predecessor.ledger_commit,
        suite.beacon.label_release_round,
        suite.beacon.contract_sha256,
        suite.beacon.label_release_publication_time.isoformat(),
    )


def _assert_aggregate_authority(
    aggregate: PostOnlineCompletionAggregateReceipt,
    suite: AdmittedPostOnlineSuite,
) -> None:
    observed = (
        aggregate.suite_attempt_id,
        aggregate.manifest_sha256,
        aggregate.run_receipt_sha256,
        aggregate.online_complete_state_sha256,
        aggregate.online_output_aggregate_sha256,
        aggregate.online_attestation_descriptor_sha256,
        aggregate.online_attestation_bundle_sha256,
        aggregate.label_release_claim_state_sha256,
        aggregate.label_release_claim_ledger_commit,
        aggregate.label_release_round,
        aggregate.label_release_beacon_contract_sha256,
        aggregate.label_release_publication_time_utc,
    )
    if observed != _aggregate_authority_values(suite):
        raise PostOnlineCompletionError(
            "existing completion aggregate differs from provider authority"
        )


def _close_completion_directory(
    suite: AdmittedPostOnlineSuite,
    materials: _CompletionMaterials,
    public_record: ZenodoPublishedRecord,
    *,
    round_guard: RoundPublicationGuard,
    existing_receipt_names: frozenset[str] = frozenset(),
) -> PostOnlineCompletionAggregateReceipt:
    bindings: list[PostOnlineCompletionBinding] = []
    for source in suite.sources:
        corpus_id = source.corpus_id
        completion = materials.completions[corpus_id]
        record = materials.records[corpus_id]
        try:
            receipt = PredictionCompletionAnchorReceipt.from_record(record)
        except Exception as exc:
            raise PostOnlineCompletionError(
                f"cannot create verified anchor receipt for {corpus_id}"
            ) from exc
        completion_filename = f"{corpus_id}-prediction-completion.json"
        record_filename = _anchor_filename(corpus_id)
        receipt_filename = f"{corpus_id}-prediction-completion-anchor-receipt.json"
        receipt_path = suite.completion_root / receipt_filename
        try:
            if receipt_filename in existing_receipt_names:
                if load_prediction_completion_anchor_receipt(receipt_path) != receipt:
                    raise PostOnlineCompletionError(
                        f"interrupted anchor receipt differs for {corpus_id}"
                    )
            else:
                _assert_predecessor_current(suite.predecessor)
                round_guard.assert_not_public(suite.beacon)
                write_prediction_completion_anchor_receipt(receipt, receipt_path)
        except PostOnlineCompletionError:
            raise
        except Exception as exc:
            raise PostOnlineCompletionError(
                f"cannot persist verified anchor receipt for {corpus_id}"
            ) from exc
        bindings.append(
            PostOnlineCompletionBinding(
                corpus=corpus_id,
                prediction_completion_filename=completion_filename,
                prediction_completion_receipt_sha256=completion.receipt_sha256,
                prediction_completion_file_sha256=_sha256(completion.canonical_bytes() + b"\n"),
                anchor_record_filename=record_filename,
                anchor_record_sha256=record.record_sha256,
                anchor_receipt_filename=receipt_filename,
                anchor_receipt_sha256=receipt.receipt_sha256,
                anchor_receipt_file_sha256=_sha256(receipt.canonical_bytes() + b"\n"),
                external_anchor_uri=record.external_anchor_uri,
            )
        )

    _assert_predecessor_current(suite.predecessor)
    round_guard.assert_not_public(suite.beacon)
    aggregate = PostOnlineCompletionAggregateReceipt(
        suite_attempt_id=suite.claim_record.suite_attempt_id,
        manifest_sha256=suite.manifest_digest,
        run_receipt_sha256=suite.claim_record.run_receipt_sha256,
        online_complete_state_sha256=suite.online_record.record_sha256,
        online_output_aggregate_sha256=(
            suite.online_record.payload.run_output_aggregate.aggregate_sha256
        ),
        online_attestation_descriptor_sha256=(
            suite.predecessor.evidences[suite.online_record.sequence].descriptor_sha256
        ),
        online_attestation_bundle_sha256=(
            suite.predecessor.evidences[suite.online_record.sequence].bundle_sha256
        ),
        label_release_claim_state_sha256=suite.claim_record.record_sha256,
        label_release_claim_ledger_commit=suite.predecessor.ledger_commit,
        zenodo_record_id=materials.record_id,
        zenodo_deposition_created_at_utc=materials.created_at_utc,
        zenodo_public_record_updated_at_utc=public_record.updated_at_utc,
        zenodo_record_uri=f"{ZENODO_API_ORIGIN}/api/records/{materials.record_id}",
        label_release_round=suite.beacon.label_release_round,
        label_release_beacon_contract_sha256=suite.beacon.contract_sha256,
        label_release_publication_time_utc=(
            suite.beacon.label_release_publication_time.isoformat()
        ),
        bindings=tuple(bindings),
    )
    expected_names = {
        POST_ONLINE_COMPLETION_AGGREGATE_FILENAME,
        *(
            name
            for row in bindings
            for name in (
                row.prediction_completion_filename,
                row.anchor_record_filename,
                row.anchor_receipt_filename,
            )
        ),
    }
    observed_before_aggregate = {path.name for path in suite.completion_root.iterdir()}
    if observed_before_aggregate != expected_names - {POST_ONLINE_COMPLETION_AGGREGATE_FILENAME}:
        raise PostOnlineCompletionError("completion directory membership changed")
    try:
        write_exclusive_receipt_bytes(
            aggregate.canonical_file_bytes(),
            suite.completion_root / POST_ONLINE_COMPLETION_AGGREGATE_FILENAME,
        )
    except ArtifactIntegrityError as exc:
        raise PostOnlineCompletionError("cannot close completion aggregate") from exc
    if verify_post_online_completion_directory(suite.completion_root) != aggregate:
        raise PostOnlineCompletionError("completion aggregate readback changed")
    if {path.name for path in suite.completion_root.iterdir()} != expected_names:
        raise PostOnlineCompletionError("closed completion directory membership changed")
    return aggregate


def execute_post_online_completion(
    predecessor: VerifiedProviderPredecessor,
    *,
    publisher: CompletionAnchorPublisher,
    round_guard: RoundPublicationGuard,
    recovery_reader: AnonymousCompletionAnchorReader | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> PostOnlineCompletionAggregateReceipt:
    """Run the closed operator using explicit transports.

    This entry point exists for provider integration and local fake-transport
    tests.  Production callers should use
    :func:`publish_post_online_completion_anchors`, which admits credentials
    only through a file descriptor.
    """

    suite = _admit_post_online_suite(predecessor)
    sleep = (lambda _seconds: None) if sleeper is None else sleeper
    if not callable(sleep):
        raise PostOnlineCompletionError("sleeper must be callable")
    _assert_predecessor_current(suite.predecessor)
    if not os.path.lexists(suite.completion_root):
        round_guard.assert_not_public(suite.beacon)
    created_directory = _private_completion_directory(suite.completion_root)

    if not created_directory:
        aggregate_path = suite.completion_root / POST_ONLINE_COMPLETION_AGGREGATE_FILENAME
        if aggregate_path.exists():
            aggregate = verify_post_online_completion_directory(suite.completion_root)
            _assert_aggregate_authority(aggregate, suite)
            return aggregate
    round_guard.assert_not_public(suite.beacon)
    if not created_directory:
        record_id, created_at_utc, existing_receipts = _infer_recovery_identity(suite)
        materials = _create_completion_materials(
            suite,
            record_id=record_id,
            created_at_utc=created_at_utc,
            require_existing=True,
        )
        active_reader = (
            ZenodoAnonymousCompletionAnchorReader() if recovery_reader is None else recovery_reader
        )
        public_record: ZenodoPublishedRecord | None = None
        for attempt in range(_PUBLICATION_READBACK_ATTEMPTS):
            _assert_predecessor_current(suite.predecessor)
            round_guard.assert_not_public(suite.beacon)
            try:
                public_record = active_reader.public_record(
                    record_id=record_id,
                    expected_created_at_utc=created_at_utc,
                )
            except PostOnlineCompletionError:
                raise
            except Exception as exc:
                raise PostOnlineCompletionError(
                    "interrupted Zenodo publication revalidation failed"
                ) from exc
            if public_record is not None:
                break
            if attempt + 1 < _PUBLICATION_READBACK_ATTEMPTS:
                sleep(_PUBLICATION_READBACK_SECONDS)
        if public_record is None:
            raise PostOnlineCompletionError(
                "interrupted completion has no exact anonymous public record; "
                "the prior attempt is terminal"
            )
        _assert_public_snapshot(public_record, materials, suite)
        _assert_anonymous_bytes(active_reader, materials)
        try:
            stable_public_record = active_reader.public_record(
                record_id=record_id,
                expected_created_at_utc=created_at_utc,
            )
        except PostOnlineCompletionError:
            raise
        except Exception as exc:
            raise PostOnlineCompletionError(
                "interrupted Zenodo publication stability check failed"
            ) from exc
        if stable_public_record != public_record:
            raise PostOnlineCompletionError(
                "public Zenodo record changed during anonymous readback"
            )
        _assert_predecessor_current(suite.predecessor)
        round_guard.assert_not_public(suite.beacon)
        return _close_completion_directory(
            suite,
            materials,
            public_record,
            round_guard=round_guard,
            existing_receipt_names=existing_receipts,
        )

    deposition = publisher.create_deposition()
    created = _utc("Zenodo created timestamp", deposition.created_at_utc)
    if created >= suite.beacon.label_release_publication_time:
        raise PostOnlineCompletionError(
            "Zenodo deposition was created at or after label-release publication"
        )
    _assert_predecessor_current(suite.predecessor)
    round_guard.assert_not_public(suite.beacon)
    publisher.set_metadata(
        deposition,
        _metadata(suite, created_at_utc=created),
    )

    materials = _create_completion_materials(
        suite,
        record_id=deposition.record_id,
        created_at_utc=deposition.created_at_utc,
        require_existing=False,
    )

    _assert_predecessor_current(suite.predecessor)
    round_guard.assert_not_public(suite.beacon)
    for filename in sorted(
        materials.record_bytes,
        key=lambda item: item.encode("utf-8"),
    ):
        publisher.upload(deposition, filename, materials.record_bytes[filename])
    _assert_remote_inventory(
        publisher.draft_files(deposition),
        materials.record_bytes,
        label="Zenodo draft inventory",
    )

    _assert_predecessor_current(suite.predecessor)
    round_guard.assert_not_public(suite.beacon)
    publication_error: Exception | None = None
    try:
        publisher.publish_once(deposition)
    except Exception as exc:
        publication_error = exc

    public_record: ZenodoPublishedRecord | None = None
    for attempt in range(_PUBLICATION_READBACK_ATTEMPTS):
        _assert_predecessor_current(suite.predecessor)
        round_guard.assert_not_public(suite.beacon)
        try:
            public_record = publisher.public_record(deposition)
        except Exception:
            public_record = None
        if public_record is not None:
            break
        if attempt + 1 < _PUBLICATION_READBACK_ATTEMPTS:
            sleep(_PUBLICATION_READBACK_SECONDS)
    if public_record is None:
        if publication_error is not None:
            raise PostOnlineCompletionError(
                "Zenodo publication outcome is ambiguous; the local attempt is terminal"
            ) from publication_error
        raise PostOnlineCompletionError(
            "published Zenodo record did not become anonymously readable"
        )
    _assert_public_snapshot(public_record, materials, suite)
    _assert_anonymous_bytes(publisher, materials)
    try:
        stable_public_record = publisher.public_record(deposition)
    except Exception as exc:
        raise PostOnlineCompletionError("public Zenodo publication stability check failed") from exc
    if stable_public_record != public_record:
        raise PostOnlineCompletionError("public Zenodo record changed during anonymous readback")

    _assert_predecessor_current(suite.predecessor)
    round_guard.assert_not_public(suite.beacon)
    return _close_completion_directory(
        suite,
        materials,
        public_record,
        round_guard=round_guard,
    )


def publish_post_online_completion_anchors(
    predecessor: VerifiedProviderPredecessor,
    *,
    token_fd: int,
    round_guard: RoundPublicationGuard | None = None,
) -> PostOnlineCompletionAggregateReceipt:
    """Production integration API with token-fd-only Zenodo authentication."""

    guard = DrandRoundPublicationGuard() if round_guard is None else round_guard
    with ZenodoCompletionAnchorPublisher.from_token_fd(token_fd) as publisher:
        return execute_post_online_completion(
            predecessor,
            publisher=publisher,
            round_guard=guard,
            sleeper=time.sleep,
        )


__all__ = [
    "AdmittedCompletionSource",
    "AdmittedPostOnlineSuite",
    "AnonymousCompletionAnchorReader",
    "CompletionAnchorPublisher",
    "DrandRoundPublicationGuard",
    "POST_ONLINE_COMPLETION_AGGREGATE_FILENAME",
    "PostOnlineCompletionAggregateReceipt",
    "PostOnlineCompletionBinding",
    "PostOnlineCompletionError",
    "RoundPublicationGuard",
    "VerifiedPostOnlineCompletionAnchors",
    "VerifiedPostOnlineCompletionAuthority",
    "ZenodoAnonymousCompletionAnchorReader",
    "ZenodoCompletionAnchorPublisher",
    "ZenodoDeposition",
    "ZenodoPublishedRecord",
    "ZenodoRemoteFile",
    "execute_post_online_completion",
    "load_post_online_completion_aggregate_receipt",
    "verify_post_online_completion_directory",
    "publish_post_online_completion_anchors",
    "revalidate_post_online_completion_anchors",
    "revalidate_post_online_completion_authority",
]
