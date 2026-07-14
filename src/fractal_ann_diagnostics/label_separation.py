"""Custodian, online-serving, and post-receipt label boundaries.

The custodian converts a sealed :class:`NormalizedCorpus` into two artifacts.
Online code receives the label-separated execution artifact. Offline scoring
receives sealed labels only through an exact join after predictions have been
emitted and their completion receipt has been externally anchored. Public
benchmark questions remain reidentifiable; this module removes serialized label
fields, not outside knowledge of a public dataset.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .corpora import EvidenceQuery, NormalizedCorpus
from .study import SealedRunReceipt

ONLINE_EXECUTION_SCHEMA = "fractal-online-execution-v2"
SEALED_LABEL_SCHEMA = "fractal-sealed-labels-v2"
PREDICTION_ARTIFACT_SCHEMA = "fractal-online-predictions-v1"
ACTION_PANEL_BINDING_SCHEMA = "fractal-action-panel-binding-v1"
PREDICTION_COMPLETION_SCHEMA = "fractal-prediction-completion-v3"
OFFLINE_JOIN_SCHEMA = "fractal-offline-label-join-v1"
RECEIPT_BINDING_SCHEMA = "fractal-sealed-run-receipt-binding-v1"

_MAX_CUSTODY_ARTIFACT_BYTES = 1024 * 1024 * 1024
_ONLINE_DOCUMENT_FIELDS = frozenset(
    {"content_hash", "document_id", "external_id", "source_uri", "text", "title"}
)
_ONLINE_TRIAL_FIELDS = frozenset({"corpus", "family_key", "stage", "text", "trial_key"})
_ONLINE_EXECUTION_FIELDS = frozenset(
    {
        "corpus",
        "documents",
        "key_id",
        "schema_version",
        "stage",
        "trials",
    }
)
_SEALED_EVIDENCE_LOCATION_FIELDS = frozenset(
    {"content_hash", "document_id", "locator", "source_uri"}
)
_SEALED_EVIDENCE_BUNDLE_FIELDS = frozenset({"bundle_id", "locations"})
_SEALED_TRIAL_LABEL_FIELDS = frozenset(
    {
        "answer",
        "evidence_bundles",
        "family_key",
        "label_metadata",
        "relevant_document_ids",
        "trial_key",
    }
)
_SEALED_LABEL_ARTIFACT_FIELDS = frozenset(
    {
        "corpus",
        "document_count",
        "execution_artifact_sha256",
        "key_id",
        "labels",
        "schema_version",
        "stage",
    }
)
_ONLINE_PREDICTION_FIELDS = frozenset(
    {"emitted_answer", "family_key", "returned_document_ids", "trial_key"}
)
_PREDICTION_ARTIFACT_FIELDS = frozenset(
    {
        "corpus",
        "document_count",
        "execution_artifact_sha256",
        "key_id",
        "manifest_sha256",
        "predictions",
        "run_receipt_sha256",
        "schema_version",
        "stage",
    }
)
_PREDICTION_COMPLETION_FIELDS = frozenset(
    {
        "anchored_at_utc",
        "action_panel_binding",
        "corpus",
        "execution_artifact_sha256",
        "external_anchor_identity",
        "external_anchor_uri",
        "manifest_sha256",
        "prediction_artifact_sha256",
        "prediction_count",
        "run_receipt_sha256",
        "schema_version",
        "stage",
    }
)
_ACTION_PANEL_BINDING_FIELDS = frozenset(
    {
        "action_panel_artifact_sha256",
        "corpus",
        "execution_artifact_sha256",
        "manifest_sha256",
        "run_receipt_sha256",
        "schema_version",
        "stage",
    }
)
_JOINED_EVALUATION_TRIAL_FIELDS = frozenset({"labels", "prediction"})
_OFFLINE_EVALUATION_FIELDS = frozenset(
    {
        "corpus",
        "execution_artifact_sha256",
        "manifest_sha256",
        "prediction_artifact_sha256",
        "prediction_completion_receipt_sha256",
        "run_receipt_sha256",
        "schema_version",
        "sealed_label_artifact_sha256",
        "stage",
        "trials",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_KEY = re.compile(r"^[0-9a-f]{64}$")
_LABEL_METADATA_TOKENS = (
    "answer",
    "correct",
    "evidence",
    "gold",
    "label",
    "relevance",
    "target",
)
_FORBIDDEN_EXECUTION_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "correct",
        "correctness",
        "evidence_labels",
        "gold",
        "gold_evidence",
        "label",
        "labels",
        "metadata",
        "original_query_id",
        "query_family",
        "query_id",
        "relevance",
        "relevant_document_ids",
        "target",
        "targets",
    }
)
_FORBIDDEN_EXECUTION_TOKENS = (
    "answer",
    "correct",
    "evidence",
    "gold",
    "label",
    "query_id",
    "relevance",
    "target",
)


class LabelSeparationError(ValueError):
    """Raised when a label boundary or post-receipt binding is invalid."""


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
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LabelSeparationError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise LabelSeparationError(f"{label} keys must be strings")
    observed = set(value)
    unknown = observed - fields
    if unknown:
        raise LabelSeparationError(f"{label} has unknown fields: {sorted(unknown)}")
    missing = fields - observed
    if missing:
        raise LabelSeparationError(f"{label} is missing fields: {sorted(missing)}")
    return value


def _require_array(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise LabelSeparationError(f"{name} must be a JSON array")
    return value


def _require_json_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise LabelSeparationError(f"{name} must be a JSON string")
    return value


def _require_nullable_json_string(name: str, value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise LabelSeparationError(f"{name} must be a JSON string or null")
    return value


def _parse_json_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise LabelSeparationError(f"{label} contains duplicate key {key!r}")
            parsed[key] = value
        return parsed

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise LabelSeparationError(
                f"{label} contains non-finite number {value!r}"
            )
        return parsed

    def reject_nonfinite(value: str) -> None:
        raise LabelSeparationError(f"{label} contains non-finite number {value!r}")

    try:
        text = encoded.decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
            parse_float=parse_float,
        )
    except UnicodeDecodeError as exc:
        raise LabelSeparationError(f"{label} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise LabelSeparationError(f"{label} must be valid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise LabelSeparationError(f"{label} must contain one JSON object")
    return payload


def _read_artifact_bytes(path: str | Path, *, label: str) -> bytes:
    try:
        return read_secure_regular_file(
            path,
            max_bytes=_MAX_CUSTODY_ARTIFACT_BYTES,
            label=label,
        )
    except ArtifactIntegrityError as exc:
        raise LabelSeparationError(f"cannot read {label} safely: {exc}") from exc


def _require_canonical_file_bytes(
    encoded: bytes,
    canonical_bytes: bytes,
    *,
    label: str,
) -> None:
    if encoded != canonical_bytes + b"\n":
        raise LabelSeparationError(
            f"{label} bytes are not canonical; exactly one trailing newline is required"
        )


def _write_artifact_bytes(
    canonical_bytes: bytes,
    target: str | Path,
    *,
    label: str,
) -> None:
    try:
        write_exclusive_receipt_bytes(canonical_bytes + b"\n", target)
    except ArtifactIntegrityError as exc:
        raise LabelSeparationError(f"cannot write {label} safely: {exc}") from exc


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LabelSeparationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_opaque_key(name: str, value: str) -> str:
    if not isinstance(value, str) or _OPAQUE_KEY.fullmatch(value) is None:
        raise LabelSeparationError(f"{name} must be an opaque HMAC-SHA-256 key")
    return value


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise LabelSeparationError(f"{name} must be a non-empty canonical string")
    if unicodedata.normalize("NFC", value) != value:
        raise LabelSeparationError(f"{name} must use NFC Unicode normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise LabelSeparationError(f"{name} cannot contain control characters")
    return value


def _require_utc_timestamp(name: str, value: str) -> datetime:
    _require_text(name, value)
    parse_value = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        instant = datetime.fromisoformat(parse_value)
    except ValueError as exc:
        raise LabelSeparationError(f"{name} must be an ISO 8601 timestamp") from exc
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise LabelSeparationError(f"{name} must use the UTC +00:00 offset")
    if instant.isoformat() != value:
        raise LabelSeparationError(f"{name} must use canonical ISO 8601 form")
    return instant


def _require_external_anchor_uri(name: str, value: str) -> str:
    _require_text(name, value)
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise LabelSeparationError(f"{name} must be an absolute HTTPS URI")
    if parsed.username is not None or parsed.password is not None:
        raise LabelSeparationError(f"{name} cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise LabelSeparationError(f"{name} cannot contain a query or fragment")
    if parsed.path in {"", "/"}:
        raise LabelSeparationError(f"{name} must identify a specific external anchor")
    return value


def _validate_hmac_secret(secret: bytes) -> bytes:
    if not isinstance(secret, bytes):
        raise LabelSeparationError("HMAC key must be immutable bytes")
    if len(secret) < 32:
        raise LabelSeparationError("HMAC key must contain at least 32 bytes")
    if len(set(secret)) < 8:
        raise LabelSeparationError("HMAC key has insufficient byte diversity")
    return secret


def _derive_opaque_key(
    secret: bytes,
    *,
    domain: str,
    key_id: str,
    corpus: str,
    stage: str,
    source_value: str,
) -> str:
    digest = hmac.new(secret, digestmod=hashlib.sha256)
    for value in (
        "fractal-label-separation-v2",
        domain,
        key_id,
        corpus,
        stage,
        source_value,
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _assert_no_label_fields(value: Any, *, path: str = "execution") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_EXECUTION_FIELDS or any(
                token in normalized for token in _FORBIDDEN_EXECUTION_TOKENS
            ):
                raise LabelSeparationError(f"label-bearing field leaked into {path}: {key!r}")
            _assert_no_label_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, nested in enumerate(value):
            _assert_no_label_fields(nested, path=f"{path}[{position}]")


def _path_sample(values: set[str]) -> str:
    ordered = sorted(values)
    shown = ordered[:5]
    suffix = "" if len(ordered) <= 5 else f" (+{len(ordered) - 5} more)"
    return f"{shown}{suffix}"


@dataclass(frozen=True)
class OnlineDocument:
    """A retrieval document with no query-side judgment fields."""

    document_id: int
    external_id: str
    title: str
    text: str
    source_uri: str
    content_hash: str

    def __post_init__(self) -> None:
        if type(self.document_id) is not int or self.document_id < 0:
            raise LabelSeparationError("document_id must be a non-negative integer")
        for name in ("external_id", "title", "text", "source_uri"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise LabelSeparationError(f"{name} must be non-empty")
        if (
            not isinstance(self.content_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.content_hash) is None
        ):
            raise LabelSeparationError("content_hash must use sha256:<lowercase hex>")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "document_id": self.document_id,
            "external_id": self.external_id,
            "source_uri": self.source_uri,
            "text": self.text,
            "title": self.title,
        }


@dataclass(frozen=True)
class OnlineTrial:
    """The complete online query surface for one sealed trial."""

    trial_key: str
    family_key: str
    text: str
    corpus: str
    stage: str

    def __post_init__(self) -> None:
        _require_opaque_key("trial_key", self.trial_key)
        _require_opaque_key("family_key", self.family_key)
        for name in ("text", "corpus", "stage"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise LabelSeparationError(f"{name} must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "corpus": self.corpus,
            "family_key": self.family_key,
            "stage": self.stage,
            "text": self.text,
            "trial_key": self.trial_key,
        }


def _online_document_from_dict(payload: object) -> OnlineDocument:
    row = _closed_mapping(
        payload,
        fields=_ONLINE_DOCUMENT_FIELDS,
        label="online document",
    )
    return OnlineDocument(
        document_id=row["document_id"],
        external_id=_require_json_string("online document external_id", row["external_id"]),
        title=_require_json_string("online document title", row["title"]),
        text=_require_json_string("online document text", row["text"]),
        source_uri=_require_json_string("online document source_uri", row["source_uri"]),
        content_hash=_require_json_string(
            "online document content_hash", row["content_hash"]
        ),
    )


def _online_trial_from_dict(payload: object) -> OnlineTrial:
    row = _closed_mapping(payload, fields=_ONLINE_TRIAL_FIELDS, label="online trial")
    return OnlineTrial(
        trial_key=_require_json_string("online trial trial_key", row["trial_key"]),
        family_key=_require_json_string("online trial family_key", row["family_key"]),
        text=_require_json_string("online trial text", row["text"]),
        corpus=_require_json_string("online trial corpus", row["corpus"]),
        stage=_require_json_string("online trial stage", row["stage"]),
    )


@dataclass(frozen=True)
class OnlineExecutionArtifact:
    """Pre-freeze custodian output pinned by the outer study manifest."""

    key_id: str
    corpus: str
    stage: str
    documents: tuple[OnlineDocument, ...]
    trials: tuple[OnlineTrial, ...]
    schema_version: str = ONLINE_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        _require_text("key_id", self.key_id)
        _require_text("corpus", self.corpus)
        _require_text("stage", self.stage)
        if self.schema_version != ONLINE_EXECUTION_SCHEMA:
            raise LabelSeparationError(
                f"schema_version must equal {ONLINE_EXECUTION_SCHEMA!r}"
            )
        documents = tuple(self.documents)
        trials = tuple(sorted(tuple(self.trials), key=lambda trial: trial.trial_key))
        if not documents or not all(isinstance(row, OnlineDocument) for row in documents):
            raise LabelSeparationError("documents must contain online documents")
        if not trials or not all(isinstance(row, OnlineTrial) for row in trials):
            raise LabelSeparationError("trials must contain online trials")
        if [row.document_id for row in documents] != list(range(len(documents))):
            raise LabelSeparationError("document IDs must be contiguous and ordered")
        trial_keys = [trial.trial_key for trial in trials]
        if len(trial_keys) != len(set(trial_keys)):
            raise LabelSeparationError("online execution contains duplicate trial keys")
        if any(trial.corpus != self.corpus or trial.stage != self.stage for trial in trials):
            raise LabelSeparationError("every trial must match the artifact corpus and stage")
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "trials", trials)
        _assert_no_label_fields(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "documents": [document.to_dict() for document in self.documents],
            "key_id": self.key_id,
            "schema_version": self.schema_version,
            "stage": self.stage,
            "trials": [trial.to_dict() for trial in self.trials],
        }

    def canonical_bytes(self) -> bytes:
        payload = self.to_dict()
        _assert_no_label_fields(payload)
        return _canonical_bytes(payload)

    @classmethod
    def from_dict(cls, payload: object) -> OnlineExecutionArtifact:
        """Parse one closed-schema execution artifact and all nested rows."""

        row = _closed_mapping(
            payload,
            fields=_ONLINE_EXECUTION_FIELDS,
            label="online execution artifact",
        )
        documents = _require_array("online execution documents", row["documents"])
        trials = _require_array("online execution trials", row["trials"])
        return cls(
            key_id=_require_json_string("online execution key_id", row["key_id"]),
            corpus=_require_json_string("online execution corpus", row["corpus"]),
            stage=_require_json_string("online execution stage", row["stage"]),
            documents=tuple(_online_document_from_dict(item) for item in documents),
            trials=tuple(_online_trial_from_dict(item) for item in trials),
            schema_version=_require_json_string(
                "online execution schema_version", row["schema_version"]
            ),
        )

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class SealedEvidenceLocation:
    document_id: int
    source_uri: str
    locator: str
    content_hash: str | None

    def __post_init__(self) -> None:
        if type(self.document_id) is not int or self.document_id < 0:
            raise LabelSeparationError("evidence document_id must be non-negative")
        if not self.source_uri or not self.locator:
            raise LabelSeparationError("evidence source_uri and locator must be non-empty")
        if self.content_hash is not None and not self.content_hash:
            raise LabelSeparationError("evidence content_hash cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "document_id": self.document_id,
            "locator": self.locator,
            "source_uri": self.source_uri,
        }


@dataclass(frozen=True)
class SealedEvidenceBundle:
    bundle_id: str
    locations: tuple[SealedEvidenceLocation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_id, str) or not self.bundle_id:
            raise LabelSeparationError("bundle_id must be non-empty")
        locations = tuple(
            sorted(
                tuple(self.locations),
                key=lambda row: (
                    row.document_id,
                    row.source_uri,
                    row.locator,
                    row.content_hash or "",
                ),
            )
        )
        if not locations or not all(
            isinstance(row, SealedEvidenceLocation) for row in locations
        ):
            raise LabelSeparationError("evidence bundles need typed locations")
        if len(locations) != len(set(locations)):
            raise LabelSeparationError("evidence bundles cannot repeat a location")
        object.__setattr__(self, "locations", locations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "locations": [location.to_dict() for location in self.locations],
        }


@dataclass(frozen=True)
class SealedTrialLabels:
    """Labels retained by the custodian until the run receipt exists."""

    trial_key: str
    family_key: str
    answer: str | None
    relevant_document_ids: tuple[int, ...]
    evidence_bundles: tuple[SealedEvidenceBundle, ...]
    label_metadata: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_opaque_key("trial_key", self.trial_key)
        _require_opaque_key("family_key", self.family_key)
        if self.answer is not None and not isinstance(self.answer, str):
            raise LabelSeparationError("answer must be a string or None")
        relevant = tuple(sorted(tuple(self.relevant_document_ids)))
        if any(type(value) is not int or value < 0 for value in relevant):
            raise LabelSeparationError(
                "relevant_document_ids must contain non-negative integers"
            )
        if len(relevant) != len(set(relevant)):
            raise LabelSeparationError("relevant_document_ids cannot contain duplicates")
        bundles = tuple(
            sorted(tuple(self.evidence_bundles), key=lambda bundle: bundle.bundle_id)
        )
        if not all(isinstance(bundle, SealedEvidenceBundle) for bundle in bundles):
            raise LabelSeparationError("evidence_bundles must contain typed bundles")
        bundle_ids = [bundle.bundle_id for bundle in bundles]
        if len(bundle_ids) != len(set(bundle_ids)):
            raise LabelSeparationError("evidence bundle IDs must be unique")
        metadata = tuple(sorted(tuple(self.label_metadata)))
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or not value
            for key, value in metadata
        ):
            raise LabelSeparationError("label_metadata needs non-empty string pairs")
        metadata_keys = [key for key, _ in metadata]
        if len(metadata_keys) != len(set(metadata_keys)):
            raise LabelSeparationError("label_metadata keys must be unique")
        object.__setattr__(self, "relevant_document_ids", relevant)
        object.__setattr__(self, "evidence_bundles", bundles)
        object.__setattr__(self, "label_metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence_bundles": [bundle.to_dict() for bundle in self.evidence_bundles],
            "family_key": self.family_key,
            "label_metadata": [list(item) for item in self.label_metadata],
            "relevant_document_ids": list(self.relevant_document_ids),
            "trial_key": self.trial_key,
        }


def _sealed_evidence_location_from_dict(payload: object) -> SealedEvidenceLocation:
    row = _closed_mapping(
        payload,
        fields=_SEALED_EVIDENCE_LOCATION_FIELDS,
        label="sealed evidence location",
    )
    return SealedEvidenceLocation(
        document_id=row["document_id"],
        source_uri=_require_json_string(
            "sealed evidence location source_uri", row["source_uri"]
        ),
        locator=_require_json_string(
            "sealed evidence location locator", row["locator"]
        ),
        content_hash=_require_nullable_json_string(
            "sealed evidence location content_hash", row["content_hash"]
        ),
    )


def _sealed_evidence_bundle_from_dict(payload: object) -> SealedEvidenceBundle:
    row = _closed_mapping(
        payload,
        fields=_SEALED_EVIDENCE_BUNDLE_FIELDS,
        label="sealed evidence bundle",
    )
    locations = _require_array("sealed evidence bundle locations", row["locations"])
    return SealedEvidenceBundle(
        bundle_id=_require_json_string(
            "sealed evidence bundle bundle_id", row["bundle_id"]
        ),
        locations=tuple(_sealed_evidence_location_from_dict(item) for item in locations),
    )


def _sealed_trial_labels_from_dict(payload: object) -> SealedTrialLabels:
    row = _closed_mapping(
        payload,
        fields=_SEALED_TRIAL_LABEL_FIELDS,
        label="sealed trial labels",
    )
    relevant = _require_array(
        "sealed trial labels relevant_document_ids", row["relevant_document_ids"]
    )
    bundles = _require_array(
        "sealed trial labels evidence_bundles", row["evidence_bundles"]
    )
    metadata_items = _require_array(
        "sealed trial labels label_metadata", row["label_metadata"]
    )
    metadata: list[tuple[str, str]] = []
    for position, item in enumerate(metadata_items):
        pair = _require_array(
            f"sealed trial labels label_metadata[{position}]",
            item,
        )
        if len(pair) != 2:
            raise LabelSeparationError(
                f"sealed trial labels label_metadata[{position}] must contain two strings"
            )
        metadata.append(
            (
                _require_json_string(
                    f"sealed trial labels label_metadata[{position}][0]", pair[0]
                ),
                _require_json_string(
                    f"sealed trial labels label_metadata[{position}][1]", pair[1]
                ),
            )
        )
    return SealedTrialLabels(
        trial_key=_require_json_string("sealed trial labels trial_key", row["trial_key"]),
        family_key=_require_json_string(
            "sealed trial labels family_key", row["family_key"]
        ),
        answer=_require_nullable_json_string("sealed trial labels answer", row["answer"]),
        relevant_document_ids=tuple(relevant),
        evidence_bundles=tuple(
            _sealed_evidence_bundle_from_dict(item) for item in bundles
        ),
        label_metadata=tuple(metadata),
    )


@dataclass(frozen=True)
class SealedLabelArtifact:
    """Custodian-only labels bound to an execution and pinned by an outer manifest."""

    execution_artifact_sha256: str
    key_id: str
    corpus: str
    stage: str
    document_count: int
    labels: tuple[SealedTrialLabels, ...]
    schema_version: str = SEALED_LABEL_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("execution_artifact_sha256", self.execution_artifact_sha256)
        _require_text("key_id", self.key_id)
        _require_text("corpus", self.corpus)
        _require_text("stage", self.stage)
        if self.schema_version != SEALED_LABEL_SCHEMA:
            raise LabelSeparationError(f"schema_version must equal {SEALED_LABEL_SCHEMA!r}")
        if type(self.document_count) is not int or self.document_count <= 0:
            raise LabelSeparationError("document_count must be a positive integer")
        labels = tuple(sorted(tuple(self.labels), key=lambda row: row.trial_key))
        if not labels or not all(isinstance(row, SealedTrialLabels) for row in labels):
            raise LabelSeparationError("labels must contain sealed trial labels")
        trial_keys = [row.trial_key for row in labels]
        if len(trial_keys) != len(set(trial_keys)):
            raise LabelSeparationError("sealed labels contain duplicate trial keys")
        for row in labels:
            if any(value >= self.document_count for value in row.relevant_document_ids):
                raise LabelSeparationError("relevance label names an unknown document")
            for bundle in row.evidence_bundles:
                if any(
                    location.document_id >= self.document_count
                    for location in bundle.locations
                ):
                    raise LabelSeparationError("gold evidence names an unknown document")
        object.__setattr__(self, "labels", labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "document_count": self.document_count,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "key_id": self.key_id,
            "labels": [row.to_dict() for row in self.labels],
            "schema_version": self.schema_version,
            "stage": self.stage,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> SealedLabelArtifact:
        """Parse one closed-schema sealed-label artifact and all nested rows."""

        row = _closed_mapping(
            payload,
            fields=_SEALED_LABEL_ARTIFACT_FIELDS,
            label="sealed label artifact",
        )
        labels = _require_array("sealed label artifact labels", row["labels"])
        return cls(
            execution_artifact_sha256=_require_json_string(
                "sealed label artifact execution_artifact_sha256",
                row["execution_artifact_sha256"],
            ),
            key_id=_require_json_string("sealed label artifact key_id", row["key_id"]),
            corpus=_require_json_string("sealed label artifact corpus", row["corpus"]),
            stage=_require_json_string("sealed label artifact stage", row["stage"]),
            document_count=row["document_count"],
            labels=tuple(_sealed_trial_labels_from_dict(item) for item in labels),
            schema_version=_require_json_string(
                "sealed label artifact schema_version", row["schema_version"]
            ),
        )

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class CustodianSplit:
    execution: OnlineExecutionArtifact
    sealed_labels: SealedLabelArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.execution, OnlineExecutionArtifact) or not isinstance(
            self.sealed_labels, SealedLabelArtifact
        ):
            raise LabelSeparationError("custodian split needs typed execution and label artifacts")
        if self.sealed_labels.execution_artifact_sha256 != self.execution.artifact_sha256:
            raise LabelSeparationError("sealed labels are not bound to the execution artifact")
        for name in ("key_id", "corpus", "stage"):
            if getattr(self.execution, name) != getattr(self.sealed_labels, name):
                raise LabelSeparationError(f"custodian split has mismatched {name}")


def _sealed_bundles(query: EvidenceQuery) -> tuple[SealedEvidenceBundle, ...]:
    if query.gold_evidence is None:
        return ()
    return tuple(
        SealedEvidenceBundle(
            bundle_id=bundle.bundle_id,
            locations=tuple(
                SealedEvidenceLocation(
                    document_id=location.document_id,
                    source_uri=location.source_uri,
                    locator=location.locator,
                    content_hash=location.content_hash,
                )
                for location in bundle.locations
            ),
        )
        for bundle in query.gold_evidence.alternatives
    )


def _sealed_metadata(metadata: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (str(key), str(value))
            for key, value in metadata.items()
            if value
            and any(token in str(key).casefold() for token in _LABEL_METADATA_TOKENS)
        )
    )


def split_custodian_corpus(
    corpus: NormalizedCorpus,
    *,
    hmac_key: bytes,
    key_id: str,
) -> CustodianSplit:
    """Create disjoint online and custodian artifacts from a sealed corpus."""

    if not isinstance(corpus, NormalizedCorpus):
        raise LabelSeparationError("corpus must be a NormalizedCorpus")
    if corpus.stage != "sealed":
        raise LabelSeparationError("label separation accepts only stage='sealed' corpora")
    secret = _validate_hmac_secret(hmac_key)
    _require_text("key_id", key_id)
    documents = tuple(
        OnlineDocument(
            document_id=document.document_id,
            external_id=document.external_id,
            title=document.title,
            text=document.text,
            source_uri=document.source_uri,
            content_hash=document.content_hash,
        )
        for document in corpus.documents
    )
    online_trials: list[OnlineTrial] = []
    sealed_labels: list[SealedTrialLabels] = []
    source_by_family_key: dict[str, str] = {}
    for query in corpus.queries:
        trial_key = _derive_opaque_key(
            secret,
            domain="trial",
            key_id=key_id,
            corpus=corpus.name,
            stage=corpus.stage,
            source_value=query.query_id,
        )
        family_key = _derive_opaque_key(
            secret,
            domain="family",
            key_id=key_id,
            corpus=corpus.name,
            stage=corpus.stage,
            source_value=query.query_family,
        )
        prior_family = source_by_family_key.setdefault(family_key, query.query_family)
        if prior_family != query.query_family:
            raise LabelSeparationError("HMAC family-key collision")
        online_trials.append(
            OnlineTrial(
                trial_key=trial_key,
                family_key=family_key,
                text=query.text,
                corpus=corpus.name,
                stage=corpus.stage,
            )
        )
        sealed_labels.append(
            SealedTrialLabels(
                trial_key=trial_key,
                family_key=family_key,
                answer=query.answer,
                relevant_document_ids=query.relevant_document_ids,
                evidence_bundles=_sealed_bundles(query),
                label_metadata=_sealed_metadata(query.metadata),
            )
        )
    execution = OnlineExecutionArtifact(
        key_id=key_id,
        corpus=corpus.name,
        stage=corpus.stage,
        documents=documents,
        trials=tuple(online_trials),
    )
    labels = SealedLabelArtifact(
        execution_artifact_sha256=execution.artifact_sha256,
        key_id=key_id,
        corpus=corpus.name,
        stage=corpus.stage,
        document_count=len(documents),
        labels=tuple(sealed_labels),
    )
    return CustodianSplit(execution=execution, sealed_labels=labels)


def sealed_run_receipt_sha256(receipt: SealedRunReceipt) -> str:
    """Return the canonical digest used to bind immutable prediction emission."""

    if not isinstance(receipt, SealedRunReceipt):
        raise LabelSeparationError("receipt must be a SealedRunReceipt")
    _require_sha256("receipt.manifest_sha256", receipt.manifest_sha256)
    for name in (
        "protocol_version",
        "started_at_utc",
        "runner_identity",
        "code_commit",
        "runner_image",
        "receipt_uri",
        "protocol_registration_receipt_uri",
        "protocol_registration_receipt_sha256",
        "protocol_registration_record_uri",
        "verification_receipt_uri",
        "verification_receipt_sha256",
    ):
        value = getattr(receipt, name)
        if not isinstance(value, str) or not value:
            raise LabelSeparationError(f"receipt.{name} must be non-empty")
    payload = {
        "code_commit": receipt.code_commit,
        "manifest_sha256": receipt.manifest_sha256,
        "protocol_version": receipt.protocol_version,
        "protocol_registration_receipt_sha256": (
            receipt.protocol_registration_receipt_sha256
        ),
        "protocol_registration_receipt_uri": (
            receipt.protocol_registration_receipt_uri
        ),
        "protocol_registration_record_uri": receipt.protocol_registration_record_uri,
        "receipt_uri": receipt.receipt_uri,
        "runner_identity": receipt.runner_identity,
        "runner_image": receipt.runner_image,
        "schema_version": RECEIPT_BINDING_SCHEMA,
        "started_at_utc": receipt.started_at_utc,
        "verification_receipt_sha256": receipt.verification_receipt_sha256,
        "verification_receipt_uri": receipt.verification_receipt_uri,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class OnlinePrediction:
    """One online output; the type has no ground-truth label fields."""

    trial_key: str
    family_key: str
    returned_document_ids: tuple[int, ...]
    emitted_answer: str | None = None

    def __post_init__(self) -> None:
        _require_opaque_key("trial_key", self.trial_key)
        _require_opaque_key("family_key", self.family_key)
        returned = tuple(self.returned_document_ids)
        if any(type(value) is not int or value < 0 for value in returned):
            raise LabelSeparationError(
                "returned_document_ids must contain non-negative integers"
            )
        if len(returned) != len(set(returned)):
            raise LabelSeparationError("returned_document_ids cannot contain duplicates")
        if self.emitted_answer is not None and (
            not isinstance(self.emitted_answer, str) or not self.emitted_answer
        ):
            raise LabelSeparationError("emitted_answer must be non-empty or None")
        object.__setattr__(self, "returned_document_ids", returned)

    def to_dict(self) -> dict[str, Any]:
        return {
            "emitted_answer": self.emitted_answer,
            "family_key": self.family_key,
            "returned_document_ids": list(self.returned_document_ids),
            "trial_key": self.trial_key,
        }


def _online_prediction_from_dict(payload: object) -> OnlinePrediction:
    row = _closed_mapping(
        payload,
        fields=_ONLINE_PREDICTION_FIELDS,
        label="online prediction",
    )
    returned = _require_array(
        "online prediction returned_document_ids", row["returned_document_ids"]
    )
    return OnlinePrediction(
        trial_key=_require_json_string("online prediction trial_key", row["trial_key"]),
        family_key=_require_json_string(
            "online prediction family_key", row["family_key"]
        ),
        returned_document_ids=tuple(returned),
        emitted_answer=_require_nullable_json_string(
            "online prediction emitted_answer", row["emitted_answer"]
        ),
    )


@dataclass(frozen=True)
class PredictionArtifact:
    """Immutable, receipt-bound output of the online serving role."""

    manifest_sha256: str
    run_receipt_sha256: str
    execution_artifact_sha256: str
    key_id: str
    corpus: str
    stage: str
    document_count: int
    predictions: tuple[OnlinePrediction, ...]
    schema_version: str = PREDICTION_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("manifest_sha256", self.manifest_sha256)
        _require_sha256("run_receipt_sha256", self.run_receipt_sha256)
        _require_sha256("execution_artifact_sha256", self.execution_artifact_sha256)
        _require_text("key_id", self.key_id)
        _require_text("corpus", self.corpus)
        _require_text("stage", self.stage)
        if self.schema_version != PREDICTION_ARTIFACT_SCHEMA:
            raise LabelSeparationError(
                f"schema_version must equal {PREDICTION_ARTIFACT_SCHEMA!r}"
            )
        if type(self.document_count) is not int or self.document_count <= 0:
            raise LabelSeparationError("document_count must be positive")
        predictions = tuple(
            sorted(tuple(self.predictions), key=lambda row: row.trial_key)
        )
        if not predictions or not all(
            isinstance(row, OnlinePrediction) for row in predictions
        ):
            raise LabelSeparationError("predictions must contain online prediction records")
        keys = [row.trial_key for row in predictions]
        if len(keys) != len(set(keys)):
            raise LabelSeparationError("prediction artifact contains duplicate trial keys")
        if any(
            document_id >= self.document_count
            for row in predictions
            for document_id in row.returned_document_ids
        ):
            raise LabelSeparationError("prediction names an unknown document")
        object.__setattr__(self, "predictions", predictions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "document_count": self.document_count,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "key_id": self.key_id,
            "manifest_sha256": self.manifest_sha256,
            "predictions": [row.to_dict() for row in self.predictions],
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
            "stage": self.stage,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> PredictionArtifact:
        """Parse one closed-schema prediction artifact and all nested rows."""

        row = _closed_mapping(
            payload,
            fields=_PREDICTION_ARTIFACT_FIELDS,
            label="prediction artifact",
        )
        predictions = _require_array("prediction artifact predictions", row["predictions"])
        return cls(
            manifest_sha256=_require_json_string(
                "prediction artifact manifest_sha256", row["manifest_sha256"]
            ),
            run_receipt_sha256=_require_json_string(
                "prediction artifact run_receipt_sha256", row["run_receipt_sha256"]
            ),
            execution_artifact_sha256=_require_json_string(
                "prediction artifact execution_artifact_sha256",
                row["execution_artifact_sha256"],
            ),
            key_id=_require_json_string("prediction artifact key_id", row["key_id"]),
            corpus=_require_json_string("prediction artifact corpus", row["corpus"]),
            stage=_require_json_string("prediction artifact stage", row["stage"]),
            document_count=row["document_count"],
            predictions=tuple(
                _online_prediction_from_dict(item) for item in predictions
            ),
            schema_version=_require_json_string(
                "prediction artifact schema_version", row["schema_version"]
            ),
        )

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class ActionPanelBinding:
    """Closed, non-circular identity of one completed pre-label action panel."""

    manifest_sha256: str
    run_receipt_sha256: str
    execution_artifact_sha256: str
    corpus: str
    stage: str
    action_panel_artifact_sha256: str
    schema_version: str = ACTION_PANEL_BINDING_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
            "action_panel_artifact_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_text("corpus", self.corpus)
        if self.stage != "sealed":
            raise LabelSeparationError("action panel binding stage must equal 'sealed'")
        if self.schema_version != ACTION_PANEL_BINDING_SCHEMA:
            raise LabelSeparationError(
                f"schema_version must equal {ACTION_PANEL_BINDING_SCHEMA!r}"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "action_panel_artifact_sha256": self.action_panel_artifact_sha256,
            "corpus": self.corpus,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ActionPanelBinding:
        row = _closed_mapping(
            payload,
            fields=_ACTION_PANEL_BINDING_FIELDS,
            label="action panel binding",
        )
        return cls(
            manifest_sha256=_require_json_string(
                "action panel binding manifest_sha256", row["manifest_sha256"]
            ),
            run_receipt_sha256=_require_json_string(
                "action panel binding run_receipt_sha256",
                row["run_receipt_sha256"],
            ),
            execution_artifact_sha256=_require_json_string(
                "action panel binding execution_artifact_sha256",
                row["execution_artifact_sha256"],
            ),
            corpus=_require_json_string("action panel binding corpus", row["corpus"]),
            stage=_require_json_string("action panel binding stage", row["stage"]),
            action_panel_artifact_sha256=_require_json_string(
                "action panel binding action_panel_artifact_sha256",
                row["action_panel_artifact_sha256"],
            ),
            schema_version=_require_json_string(
                "action panel binding schema_version", row["schema_version"]
            ),
        )


@dataclass(frozen=True)
class PredictionCompletionReceipt:
    """Custodian release token created only after prediction completion."""

    manifest_sha256: str
    run_receipt_sha256: str
    execution_artifact_sha256: str
    prediction_artifact_sha256: str
    action_panel_binding: ActionPanelBinding
    prediction_count: int
    corpus: str
    stage: str
    external_anchor_identity: str
    external_anchor_uri: str
    anchored_at_utc: str
    schema_version: str = PREDICTION_COMPLETION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
            "prediction_artifact_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.action_panel_binding, ActionPanelBinding):
            raise LabelSeparationError(
                "action_panel_binding must be an ActionPanelBinding"
            )
        if type(self.prediction_count) is not int or self.prediction_count <= 0:
            raise LabelSeparationError("prediction_count must be a positive integer")
        _require_text("corpus", self.corpus)
        _require_text("stage", self.stage)
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
            "corpus",
            "stage",
        ):
            if getattr(self.action_panel_binding, name) != getattr(self, name):
                raise LabelSeparationError(
                    f"action panel binding has mismatched {name}"
                )
        _require_text("external_anchor_identity", self.external_anchor_identity)
        _require_external_anchor_uri("external_anchor_uri", self.external_anchor_uri)
        _require_utc_timestamp("anchored_at_utc", self.anchored_at_utc)
        if self.schema_version != PREDICTION_COMPLETION_SCHEMA:
            raise LabelSeparationError(
                f"schema_version must equal {PREDICTION_COMPLETION_SCHEMA!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchored_at_utc": self.anchored_at_utc,
            "action_panel_binding": self.action_panel_binding.to_dict(),
            "corpus": self.corpus,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "external_anchor_identity": self.external_anchor_identity,
            "external_anchor_uri": self.external_anchor_uri,
            "manifest_sha256": self.manifest_sha256,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "prediction_count": self.prediction_count,
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
            "stage": self.stage,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> PredictionCompletionReceipt:
        """Parse one closed-schema version-three completion receipt."""

        row = _closed_mapping(
            payload,
            fields=_PREDICTION_COMPLETION_FIELDS,
            label="prediction completion receipt",
        )
        return cls(
            manifest_sha256=_require_json_string(
                "prediction completion manifest_sha256", row["manifest_sha256"]
            ),
            run_receipt_sha256=_require_json_string(
                "prediction completion run_receipt_sha256",
                row["run_receipt_sha256"],
            ),
            execution_artifact_sha256=_require_json_string(
                "prediction completion execution_artifact_sha256",
                row["execution_artifact_sha256"],
            ),
            prediction_artifact_sha256=_require_json_string(
                "prediction completion prediction_artifact_sha256",
                row["prediction_artifact_sha256"],
            ),
            action_panel_binding=ActionPanelBinding.from_dict(
                row["action_panel_binding"]
            ),
            prediction_count=row["prediction_count"],
            corpus=_require_json_string("prediction completion corpus", row["corpus"]),
            stage=_require_json_string("prediction completion stage", row["stage"]),
            external_anchor_identity=_require_json_string(
                "prediction completion external_anchor_identity",
                row["external_anchor_identity"],
            ),
            external_anchor_uri=_require_json_string(
                "prediction completion external_anchor_uri",
                row["external_anchor_uri"],
            ),
            anchored_at_utc=_require_json_string(
                "prediction completion anchored_at_utc", row["anchored_at_utc"]
            ),
            schema_version=_require_json_string(
                "prediction completion schema_version", row["schema_version"]
            ),
        )

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def create_prediction_completion_receipt(
    predictions: PredictionArtifact,
    *,
    execution: OnlineExecutionArtifact,
    receipt: SealedRunReceipt,
    manifest_sha256: str,
    action_panel_binding: ActionPanelBinding,
    external_anchor_identity: str,
    external_anchor_uri: str,
    anchored_at_utc: str,
) -> PredictionCompletionReceipt:
    """Bind an external completion anchor to one exact prediction artifact."""

    if not isinstance(predictions, PredictionArtifact):
        raise LabelSeparationError("predictions must be a PredictionArtifact")
    if not isinstance(execution, OnlineExecutionArtifact):
        raise LabelSeparationError("execution must be an OnlineExecutionArtifact")
    if not isinstance(action_panel_binding, ActionPanelBinding):
        raise LabelSeparationError(
            "action_panel_binding must be an ActionPanelBinding"
        )
    _require_sha256("manifest_sha256", manifest_sha256)
    run_receipt_sha256 = sealed_run_receipt_sha256(receipt)
    if receipt.manifest_sha256 != manifest_sha256:
        raise LabelSeparationError("sealed-run receipt belongs to another manifest")
    if predictions.manifest_sha256 != manifest_sha256:
        raise LabelSeparationError("prediction artifact belongs to another manifest")
    if predictions.run_receipt_sha256 != run_receipt_sha256:
        raise LabelSeparationError("prediction artifact belongs to another sealed run")
    if predictions.execution_artifact_sha256 != execution.artifact_sha256:
        raise LabelSeparationError("prediction artifact binds another execution artifact")
    for name, expected in (
        ("manifest_sha256", manifest_sha256),
        ("run_receipt_sha256", run_receipt_sha256),
        ("execution_artifact_sha256", execution.artifact_sha256),
        ("corpus", predictions.corpus),
        ("stage", predictions.stage),
    ):
        if getattr(action_panel_binding, name) != expected:
            raise LabelSeparationError(
                f"action panel binding has mismatched {name}"
            )
    for name, prediction_value, execution_value in (
        ("key_id", predictions.key_id, execution.key_id),
        ("corpus", predictions.corpus, execution.corpus),
        ("stage", predictions.stage, execution.stage),
        ("document_count", predictions.document_count, len(execution.documents)),
    ):
        if prediction_value != execution_value:
            raise LabelSeparationError(
                f"prediction and execution artifacts have mismatched {name}"
            )
    anchor_time = _require_utc_timestamp("anchored_at_utc", anchored_at_utc)
    try:
        run_started_at = datetime.fromisoformat(receipt.started_at_utc.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise LabelSeparationError(
            "sealed-run receipt started_at_utc must be ISO 8601"
        ) from exc
    if anchor_time <= run_started_at:
        raise LabelSeparationError("completion anchor must postdate the sealed run")
    return PredictionCompletionReceipt(
        manifest_sha256=manifest_sha256,
        run_receipt_sha256=run_receipt_sha256,
        execution_artifact_sha256=execution.artifact_sha256,
        prediction_artifact_sha256=predictions.artifact_sha256,
        action_panel_binding=action_panel_binding,
        prediction_count=len(predictions.predictions),
        corpus=predictions.corpus,
        stage=predictions.stage,
        external_anchor_identity=external_anchor_identity,
        external_anchor_uri=external_anchor_uri,
        anchored_at_utc=anchored_at_utc,
    )


def write_prediction_completion_receipt(
    completion_receipt: PredictionCompletionReceipt,
    target: str | Path,
) -> None:
    """Write one completion receipt exclusively without following links."""

    if not isinstance(completion_receipt, PredictionCompletionReceipt):
        raise LabelSeparationError(
            "completion_receipt must be a PredictionCompletionReceipt"
        )
    _write_artifact_bytes(
        completion_receipt.canonical_bytes(),
        target,
        label="prediction completion receipt",
    )


def load_prediction_completion_receipt(
    path: str | Path,
) -> PredictionCompletionReceipt:
    """Load one canonical v3 completion receipt through secure file handles.

    The on-disk representation is canonical JSON followed by exactly one newline.
    """

    label = "prediction completion receipt"
    encoded = _read_artifact_bytes(path, label=label)
    receipt = PredictionCompletionReceipt.from_dict(
        _parse_json_object(encoded, label=label)
    )
    _require_canonical_file_bytes(
        encoded,
        receipt.canonical_bytes(),
        label=label,
    )
    return receipt


def emit_online_predictions(
    execution: OnlineExecutionArtifact,
    predictions: Iterable[OnlinePrediction],
    *,
    receipt: SealedRunReceipt,
    manifest_sha256: str,
) -> PredictionArtifact:
    """Freeze an exact online prediction set without accepting any labels."""

    if not isinstance(execution, OnlineExecutionArtifact):
        raise LabelSeparationError("execution must be an OnlineExecutionArtifact")
    _require_sha256("manifest_sha256", manifest_sha256)
    receipt_sha256 = sealed_run_receipt_sha256(receipt)
    if receipt.manifest_sha256 != manifest_sha256:
        raise LabelSeparationError("sealed-run receipt belongs to another manifest")
    try:
        rows = tuple(predictions)
    except TypeError as exc:
        raise LabelSeparationError("predictions must be an iterable") from exc
    if not all(isinstance(row, OnlinePrediction) for row in rows):
        raise LabelSeparationError("online emission accepts only OnlinePrediction records")
    keys = [row.trial_key for row in rows]
    if len(keys) != len(set(keys)):
        raise LabelSeparationError("online emission contains duplicate trial keys")
    expected = {trial.trial_key: trial for trial in execution.trials}
    observed = {row.trial_key: row for row in rows}
    missing = set(expected) - set(observed)
    extra = set(observed) - set(expected)
    if missing:
        raise LabelSeparationError(f"predictions are missing trial keys: {_path_sample(missing)}")
    if extra:
        raise LabelSeparationError(f"predictions have extra trial keys: {_path_sample(extra)}")
    for trial_key, row in observed.items():
        if row.family_key != expected[trial_key].family_key:
            raise LabelSeparationError("prediction family key does not match its trial")
    return PredictionArtifact(
        manifest_sha256=manifest_sha256,
        run_receipt_sha256=receipt_sha256,
        execution_artifact_sha256=execution.artifact_sha256,
        key_id=execution.key_id,
        corpus=execution.corpus,
        stage=execution.stage,
        document_count=len(execution.documents),
        predictions=rows,
    )


@dataclass(frozen=True)
class JoinedEvaluationTrial:
    """Offline-only prediction and label row after receipt verification."""

    prediction: OnlinePrediction
    labels: SealedTrialLabels

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, OnlinePrediction) or not isinstance(
            self.labels, SealedTrialLabels
        ):
            raise LabelSeparationError("joined trials need typed prediction and label rows")
        if self.prediction.trial_key != self.labels.trial_key:
            raise LabelSeparationError("joined trial keys do not match")
        if self.prediction.family_key != self.labels.family_key:
            raise LabelSeparationError("joined family keys do not match")

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels.to_dict(),
            "prediction": self.prediction.to_dict(),
        }


def _joined_evaluation_trial_from_dict(payload: object) -> JoinedEvaluationTrial:
    row = _closed_mapping(
        payload,
        fields=_JOINED_EVALUATION_TRIAL_FIELDS,
        label="joined evaluation trial",
    )
    return JoinedEvaluationTrial(
        prediction=_online_prediction_from_dict(row["prediction"]),
        labels=_sealed_trial_labels_from_dict(row["labels"]),
    )


@dataclass(frozen=True)
class OfflineEvaluationArtifact:
    """Exact offline join with digests for both immutable input artifacts."""

    manifest_sha256: str
    run_receipt_sha256: str
    execution_artifact_sha256: str
    prediction_artifact_sha256: str
    prediction_completion_receipt_sha256: str
    sealed_label_artifact_sha256: str
    corpus: str
    stage: str
    trials: tuple[JoinedEvaluationTrial, ...]
    schema_version: str = OFFLINE_JOIN_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
            "prediction_artifact_sha256",
            "prediction_completion_receipt_sha256",
            "sealed_label_artifact_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_text("corpus", self.corpus)
        _require_text("stage", self.stage)
        if self.schema_version != OFFLINE_JOIN_SCHEMA:
            raise LabelSeparationError(f"schema_version must equal {OFFLINE_JOIN_SCHEMA!r}")
        trials = tuple(sorted(tuple(self.trials), key=lambda row: row.prediction.trial_key))
        if not trials or not all(isinstance(row, JoinedEvaluationTrial) for row in trials):
            raise LabelSeparationError("offline artifact needs joined evaluation trials")
        keys = [row.prediction.trial_key for row in trials]
        if len(keys) != len(set(keys)):
            raise LabelSeparationError("offline artifact contains duplicate trial keys")
        object.__setattr__(self, "trials", trials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "prediction_completion_receipt_sha256": (
                self.prediction_completion_receipt_sha256
            ),
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
            "sealed_label_artifact_sha256": self.sealed_label_artifact_sha256,
            "stage": self.stage,
            "trials": [trial.to_dict() for trial in self.trials],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> OfflineEvaluationArtifact:
        """Parse one closed-schema offline join and all nested rows."""

        row = _closed_mapping(
            payload,
            fields=_OFFLINE_EVALUATION_FIELDS,
            label="offline evaluation artifact",
        )
        trials = _require_array("offline evaluation artifact trials", row["trials"])
        return cls(
            manifest_sha256=_require_json_string(
                "offline evaluation manifest_sha256", row["manifest_sha256"]
            ),
            run_receipt_sha256=_require_json_string(
                "offline evaluation run_receipt_sha256", row["run_receipt_sha256"]
            ),
            execution_artifact_sha256=_require_json_string(
                "offline evaluation execution_artifact_sha256",
                row["execution_artifact_sha256"],
            ),
            prediction_artifact_sha256=_require_json_string(
                "offline evaluation prediction_artifact_sha256",
                row["prediction_artifact_sha256"],
            ),
            prediction_completion_receipt_sha256=_require_json_string(
                "offline evaluation prediction_completion_receipt_sha256",
                row["prediction_completion_receipt_sha256"],
            ),
            sealed_label_artifact_sha256=_require_json_string(
                "offline evaluation sealed_label_artifact_sha256",
                row["sealed_label_artifact_sha256"],
            ),
            corpus=_require_json_string("offline evaluation corpus", row["corpus"]),
            stage=_require_json_string("offline evaluation stage", row["stage"]),
            trials=tuple(_joined_evaluation_trial_from_dict(item) for item in trials),
            schema_version=_require_json_string(
                "offline evaluation schema_version", row["schema_version"]
            ),
        )

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def join_predictions_after_receipt(
    predictions: PredictionArtifact,
    sealed_labels: SealedLabelArtifact,
    *,
    execution: OnlineExecutionArtifact,
    receipt: SealedRunReceipt,
    completion_receipt: PredictionCompletionReceipt,
    action_panel_binding: ActionPanelBinding,
    manifest_sha256: str,
) -> OfflineEvaluationArtifact:
    """Join labels only after run and prediction-completion receipt checks."""

    if not isinstance(predictions, PredictionArtifact):
        raise LabelSeparationError("predictions must be a PredictionArtifact")
    if not isinstance(sealed_labels, SealedLabelArtifact):
        raise LabelSeparationError("sealed_labels must be a SealedLabelArtifact")
    if not isinstance(execution, OnlineExecutionArtifact):
        raise LabelSeparationError("execution must be an OnlineExecutionArtifact")
    if not isinstance(completion_receipt, PredictionCompletionReceipt):
        raise LabelSeparationError(
            "completion_receipt must be a PredictionCompletionReceipt"
        )
    if not isinstance(action_panel_binding, ActionPanelBinding):
        raise LabelSeparationError(
            "action_panel_binding must be an ActionPanelBinding"
        )
    if completion_receipt.action_panel_binding != action_panel_binding:
        raise LabelSeparationError(
            "prediction completion receipt binds a different action panel"
        )
    _require_sha256("manifest_sha256", manifest_sha256)
    receipt_sha256 = sealed_run_receipt_sha256(receipt)
    if receipt.manifest_sha256 != manifest_sha256:
        raise LabelSeparationError("sealed-run receipt belongs to another manifest")
    if predictions.manifest_sha256 != manifest_sha256:
        raise LabelSeparationError("prediction artifact belongs to another manifest")
    if predictions.run_receipt_sha256 != receipt_sha256:
        raise LabelSeparationError("prediction artifact belongs to another sealed run")
    expected_completion_bindings: tuple[tuple[str, Any, Any], ...] = (
        ("manifest_sha256", completion_receipt.manifest_sha256, manifest_sha256),
        ("run_receipt_sha256", completion_receipt.run_receipt_sha256, receipt_sha256),
        (
            "execution_artifact_sha256",
            completion_receipt.execution_artifact_sha256,
            execution.artifact_sha256,
        ),
        (
            "prediction_artifact_sha256",
            completion_receipt.prediction_artifact_sha256,
            predictions.artifact_sha256,
        ),
        (
            "prediction_count",
            completion_receipt.prediction_count,
            len(predictions.predictions),
        ),
        ("corpus", completion_receipt.corpus, predictions.corpus),
        ("stage", completion_receipt.stage, predictions.stage),
        (
            "action_panel_binding.manifest_sha256",
            action_panel_binding.manifest_sha256,
            manifest_sha256,
        ),
        (
            "action_panel_binding.run_receipt_sha256",
            action_panel_binding.run_receipt_sha256,
            receipt_sha256,
        ),
        (
            "action_panel_binding.execution_artifact_sha256",
            action_panel_binding.execution_artifact_sha256,
            execution.artifact_sha256,
        ),
        (
            "action_panel_binding.corpus",
            action_panel_binding.corpus,
            predictions.corpus,
        ),
        (
            "action_panel_binding.stage",
            action_panel_binding.stage,
            predictions.stage,
        ),
    )
    for name, observed, expected in expected_completion_bindings:
        if observed != expected:
            raise LabelSeparationError(
                f"prediction completion receipt has mismatched {name}"
            )
    completion_time = _require_utc_timestamp(
        "completion_receipt.anchored_at_utc",
        completion_receipt.anchored_at_utc,
    )
    run_started_at = datetime.fromisoformat(receipt.started_at_utc.replace("Z", "+00:00"))
    if completion_time <= run_started_at:
        raise LabelSeparationError(
            "prediction completion receipt must postdate the sealed run"
        )
    for name in ("key_id", "corpus", "stage", "document_count"):
        prediction_value = getattr(predictions, name)
        label_value = getattr(sealed_labels, name)
        execution_value = (
            len(execution.documents) if name == "document_count" else getattr(execution, name)
        )
        if prediction_value != label_value or prediction_value != execution_value:
            raise LabelSeparationError(f"prediction and label artifacts have mismatched {name}")
    execution_sha256 = execution.artifact_sha256
    if (
        predictions.execution_artifact_sha256 != sealed_labels.execution_artifact_sha256
        or predictions.execution_artifact_sha256 != execution_sha256
    ):
        raise LabelSeparationError("prediction and labels bind different execution artifacts")

    prediction_rows = {row.trial_key: row for row in predictions.predictions}
    label_rows = {row.trial_key: row for row in sealed_labels.labels}
    execution_rows = {row.trial_key: row for row in execution.trials}
    expected_keys = set(execution_rows)
    for role, observed_rows in (
        ("predictions", prediction_rows),
        ("sealed labels", label_rows),
    ):
        missing = expected_keys - set(observed_rows)
        extra = set(observed_rows) - expected_keys
        if missing:
            raise LabelSeparationError(
                f"{role} are missing trial keys: {_path_sample(missing)}"
            )
        if extra:
            raise LabelSeparationError(f"{role} have extra trial keys: {_path_sample(extra)}")
    joined: list[JoinedEvaluationTrial] = []
    for trial_key in sorted(prediction_rows):
        prediction = prediction_rows[trial_key]
        labels = label_rows[trial_key]
        expected_family = execution_rows[trial_key].family_key
        if prediction.family_key != labels.family_key or prediction.family_key != expected_family:
            raise LabelSeparationError("prediction and labels have mismatched family keys")
        joined.append(JoinedEvaluationTrial(prediction=prediction, labels=labels))
    return OfflineEvaluationArtifact(
        manifest_sha256=manifest_sha256,
        run_receipt_sha256=receipt_sha256,
        execution_artifact_sha256=execution_sha256,
        prediction_artifact_sha256=predictions.artifact_sha256,
        prediction_completion_receipt_sha256=completion_receipt.receipt_sha256,
        sealed_label_artifact_sha256=sealed_labels.artifact_sha256,
        corpus=predictions.corpus,
        stage=predictions.stage,
        trials=tuple(joined),
    )


def write_online_execution_artifact(
    execution: OnlineExecutionArtifact,
    target: str | Path,
) -> None:
    """Write one canonical execution artifact without replacing an existing path."""

    if not isinstance(execution, OnlineExecutionArtifact):
        raise LabelSeparationError("execution must be an OnlineExecutionArtifact")
    _write_artifact_bytes(
        execution.canonical_bytes(),
        target,
        label="online execution artifact",
    )


def load_online_execution_artifact(path: str | Path) -> OnlineExecutionArtifact:
    """Load canonical execution JSON with exactly one trailing newline."""

    label = "online execution artifact"
    encoded = _read_artifact_bytes(path, label=label)
    artifact = OnlineExecutionArtifact.from_dict(_parse_json_object(encoded, label=label))
    _require_canonical_file_bytes(encoded, artifact.canonical_bytes(), label=label)
    return artifact


def write_sealed_label_artifact(
    sealed_labels: SealedLabelArtifact,
    target: str | Path,
) -> None:
    """Write one canonical sealed-label artifact without replacing a path."""

    if not isinstance(sealed_labels, SealedLabelArtifact):
        raise LabelSeparationError("sealed_labels must be a SealedLabelArtifact")
    _write_artifact_bytes(
        sealed_labels.canonical_bytes(),
        target,
        label="sealed label artifact",
    )


def load_sealed_label_artifact(path: str | Path) -> SealedLabelArtifact:
    """Load canonical sealed-label JSON with exactly one trailing newline."""

    label = "sealed label artifact"
    encoded = _read_artifact_bytes(path, label=label)
    artifact = SealedLabelArtifact.from_dict(_parse_json_object(encoded, label=label))
    _require_canonical_file_bytes(encoded, artifact.canonical_bytes(), label=label)
    return artifact


def write_prediction_artifact(
    predictions: PredictionArtifact,
    target: str | Path,
) -> None:
    """Write one canonical prediction artifact without replacing a path."""

    if not isinstance(predictions, PredictionArtifact):
        raise LabelSeparationError("predictions must be a PredictionArtifact")
    _write_artifact_bytes(
        predictions.canonical_bytes(),
        target,
        label="prediction artifact",
    )


def load_prediction_artifact(path: str | Path) -> PredictionArtifact:
    """Load canonical prediction JSON with exactly one trailing newline."""

    label = "prediction artifact"
    encoded = _read_artifact_bytes(path, label=label)
    artifact = PredictionArtifact.from_dict(_parse_json_object(encoded, label=label))
    _require_canonical_file_bytes(encoded, artifact.canonical_bytes(), label=label)
    return artifact


def write_offline_evaluation_artifact(
    evaluation: OfflineEvaluationArtifact,
    target: str | Path,
) -> None:
    """Write one canonical offline evaluation artifact without replacing a path."""

    if not isinstance(evaluation, OfflineEvaluationArtifact):
        raise LabelSeparationError("evaluation must be an OfflineEvaluationArtifact")
    _write_artifact_bytes(
        evaluation.canonical_bytes(),
        target,
        label="offline evaluation artifact",
    )


def load_offline_evaluation_artifact(path: str | Path) -> OfflineEvaluationArtifact:
    """Load canonical offline-evaluation JSON with one trailing newline."""

    label = "offline evaluation artifact"
    encoded = _read_artifact_bytes(path, label=label)
    artifact = OfflineEvaluationArtifact.from_dict(
        _parse_json_object(encoded, label=label)
    )
    _require_canonical_file_bytes(encoded, artifact.canonical_bytes(), label=label)
    return artifact
