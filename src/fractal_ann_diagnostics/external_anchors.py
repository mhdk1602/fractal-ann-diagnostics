"""Exact external-anchor records, receipts, writers, and verification.

The protocol registry and prediction-completion controls deliberately use two
objects.  A canonical record is published by an independent HTTPS service.  A
local canonical receipt binds that record's exact bytes.  Production
verification performs one fresh, certificate-validated, no-redirect GET and
requires byte equality before returning a verified token.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import ssl
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    read_secure_control_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .label_separation import (
    ActionPanelBinding,
    PredictionCompletionReceipt,
    load_prediction_completion_receipt,
)
from .study import (
    PROTOCOL_REGISTRATION_RECEIPT_SCHEMA,
    PROTOCOL_REGISTRY_RECORD_SCHEMA,
    ProtocolRegistrationReceipt,
    ProtocolRegistryRecord,
    load_protocol_registry_record,
    manifest_sha256,
    validate_study_manifest,
)

PREDICTION_COMPLETION_ANCHOR_RECORD_SCHEMA = "fractal-prediction-completion-anchor-record-v2"
PREDICTION_COMPLETION_ANCHOR_RECEIPT_SCHEMA = "fractal-prediction-completion-anchor-receipt-v2"
MAX_EXTERNAL_ANCHOR_RECORD_BYTES = 64 * 1024
_EXTERNAL_FETCH_TIMEOUT_SECONDS = 10.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_PREDICTION_ANCHOR_RECORD_FIELDS = frozenset(
    {
        "anchored_at_utc",
        "action_panel_binding",
        "corpus",
        "execution_artifact_sha256",
        "external_anchor_identity",
        "external_anchor_uri",
        "manifest_sha256",
        "online_execution_result_receipt_sha256",
        "prediction_artifact_sha256",
        "prediction_completion_receipt_sha256",
        "prediction_count",
        "run_receipt_sha256",
        "schema_version",
        "stage",
    }
)
_PREDICTION_ANCHOR_RECEIPT_FIELDS = frozenset(
    {
        "anchor_record_sha256",
        "anchored_at_utc",
        "action_panel_artifact_sha256",
        "corpus",
        "execution_artifact_sha256",
        "external_anchor_identity",
        "external_anchor_uri",
        "manifest_sha256",
        "online_execution_result_receipt_sha256",
        "prediction_artifact_sha256",
        "prediction_completion_receipt_sha256",
        "run_receipt_sha256",
        "schema_version",
        "stage",
    }
)


class ExternalAnchorError(ValueError):
    """Raised when an external anchor or its custody evidence is invalid."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _parse_json_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ExternalAnchorError(f"{label} contains duplicate key {key!r}")
            parsed[key] = value
        return parsed

    def reject_nonfinite(value: str) -> None:
        raise ExternalAnchorError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalAnchorError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ExternalAnchorError(f"{label} must contain one JSON object")
    return value


def _closed_mapping(
    payload: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ExternalAnchorError(f"{label} must be an object")
    observed = set(payload)
    if observed != fields:
        raise ExternalAnchorError(
            f"{label} schema mismatch; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return payload


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ExternalAnchorError(f"{name} must be a canonical non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise ExternalAnchorError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ExternalAnchorError(f"{name} cannot contain control characters")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ExternalAnchorError(f"{name} must be a lowercase SHA-256")
    return value


def _require_positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ExternalAnchorError(f"{name} must be a positive integer")
    return value


def _require_utc_timestamp(name: str, value: object) -> str:
    text = _require_text(name, value)
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ExternalAnchorError(f"{name} must be ISO 8601") from exc
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise ExternalAnchorError(f"{name} must use UTC")
    if instant.isoformat() != text:
        raise ExternalAnchorError(f"{name} must use canonical ISO 8601 form")
    return text


def _require_external_uri(name: str, value: object) -> str:
    text = _require_text(name, value)
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path in {"", "/"}
    ):
        raise ExternalAnchorError(
            f"{name} must be a specific HTTPS URI without credentials, query, or fragment"
        )
    return text


def _read_canonical_file(
    path: str | Path,
    *,
    label: str,
    parser: Callable[[Mapping[str, Any]], Any],
) -> tuple[Any, bytes]:
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=MAX_EXTERNAL_ANCHOR_RECORD_BYTES,
            label=label,
        )
    except ArtifactIntegrityError as exc:
        raise ExternalAnchorError(f"cannot read {label} safely: {exc}") from exc
    value = parser(_parse_json_object(encoded, label=label))
    if encoded != value.canonical_bytes() + b"\n":
        raise ExternalAnchorError(f"{label} bytes must equal canonical JSON plus one newline")
    return value, encoded


def _write_canonical_file(value: object, target: str | Path, *, label: str) -> None:
    canonical = getattr(value, "canonical_bytes", None)
    if not callable(canonical):
        raise ExternalAnchorError(f"{label} must expose canonical_bytes()")
    try:
        write_exclusive_receipt_bytes(canonical() + b"\n", target)
    except ArtifactIntegrityError as exc:
        raise ExternalAnchorError(f"cannot write {label} safely: {exc}") from exc


def _load_frozen_manifest(path: str | Path) -> Mapping[str, Any]:
    try:
        encoded = read_secure_control_file(path, label="frozen study manifest")
    except ArtifactIntegrityError as exc:
        raise ExternalAnchorError(f"cannot read frozen manifest safely: {exc}") from exc
    payload = _parse_json_object(encoded, label="frozen study manifest")
    try:
        validate_study_manifest(payload, require_frozen=True)
    except ValueError as exc:
        raise ExternalAnchorError(f"invalid frozen study manifest: {exc}") from exc
    return payload


def create_protocol_registry_record(
    manifest_path: str | Path,
    *,
    registered_at_utc: str,
    registry_identity: str,
    registry_uri: str,
) -> ProtocolRegistryRecord:
    """Create a development/integration record; production consumes the fixed C1 package."""

    manifest = _load_frozen_manifest(manifest_path)
    return ProtocolRegistryRecord(
        manifest_sha256=manifest_sha256(manifest),
        protocol_version=str(manifest["protocol_version"]),
        registered_at_utc=registered_at_utc,
        registry_identity=registry_identity,
        registry_uri=registry_uri,
        schema_version=PROTOCOL_REGISTRY_RECORD_SCHEMA,
    )


def write_protocol_registry_record(
    record: ProtocolRegistryRecord,
    target: str | Path,
) -> None:
    """Write one canonical protocol record exclusively through a no-follow path."""

    if not isinstance(record, ProtocolRegistryRecord):
        raise ExternalAnchorError("record must be a ProtocolRegistryRecord")
    _write_canonical_file(record, target, label="protocol registry record")


def create_protocol_registration_receipt(
    manifest_path: str | Path,
    registry_record_path: str | Path,
) -> ProtocolRegistrationReceipt:
    """Bind a local record copy; this receipt alone cannot authorize production."""

    manifest = _load_frozen_manifest(manifest_path)
    try:
        record = load_protocol_registry_record(registry_record_path)
    except ValueError as exc:
        raise ExternalAnchorError(f"invalid protocol registry record: {exc}") from exc
    digest = manifest_sha256(manifest)
    if record.manifest_sha256 != digest:
        raise ExternalAnchorError("protocol registry record belongs to another frozen manifest")
    if record.protocol_version != manifest["protocol_version"]:
        raise ExternalAnchorError("protocol registry record has another protocol version")
    return ProtocolRegistrationReceipt(
        manifest_sha256=record.manifest_sha256,
        protocol_version=record.protocol_version,
        registered_at_utc=record.registered_at_utc,
        registry_identity=record.registry_identity,
        registry_uri=record.registry_uri,
        registry_record_sha256=record.record_sha256,
        schema_version=PROTOCOL_REGISTRATION_RECEIPT_SCHEMA,
    )


def write_protocol_registration_receipt(
    receipt: ProtocolRegistrationReceipt,
    target: str | Path,
) -> None:
    """Write one canonical protocol-registration receipt exclusively."""

    if not isinstance(receipt, ProtocolRegistrationReceipt):
        raise ExternalAnchorError("receipt must be a ProtocolRegistrationReceipt")
    _write_canonical_file(receipt, target, label="protocol registration receipt")


@dataclass(frozen=True)
class PredictionCompletionAnchorRecord:
    """Exact prediction and action-panel completion record published externally."""

    prediction_completion_receipt_sha256: str
    manifest_sha256: str
    run_receipt_sha256: str
    execution_artifact_sha256: str
    prediction_artifact_sha256: str
    online_execution_result_receipt_sha256: str
    action_panel_binding: ActionPanelBinding
    prediction_count: int
    corpus: str
    stage: str
    external_anchor_identity: str
    external_anchor_uri: str
    anchored_at_utc: str
    schema_version: str = PREDICTION_COMPLETION_ANCHOR_RECORD_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "prediction_completion_receipt_sha256",
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
            "prediction_artifact_sha256",
            "online_execution_result_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.action_panel_binding, ActionPanelBinding):
            raise ExternalAnchorError("action_panel_binding must be an ActionPanelBinding")
        _require_positive_integer("prediction_count", self.prediction_count)
        _require_text("corpus", self.corpus)
        _require_text("stage", self.stage)
        _require_text("external_anchor_identity", self.external_anchor_identity)
        _require_external_uri("external_anchor_uri", self.external_anchor_uri)
        _require_utc_timestamp("anchored_at_utc", self.anchored_at_utc)
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
            "corpus",
            "stage",
        ):
            if getattr(self.action_panel_binding, name) != getattr(self, name):
                raise ExternalAnchorError(f"action panel binding has mismatched {name}")
        if self.schema_version != PREDICTION_COMPLETION_ANCHOR_RECORD_SCHEMA:
            raise ExternalAnchorError(
                "anchor record schema_version must equal "
                f"{PREDICTION_COMPLETION_ANCHOR_RECORD_SCHEMA!r}"
            )

    @classmethod
    def from_completion_receipt(
        cls,
        receipt: PredictionCompletionReceipt,
    ) -> PredictionCompletionAnchorRecord:
        if not isinstance(receipt, PredictionCompletionReceipt):
            raise ExternalAnchorError("completion receipt must be a PredictionCompletionReceipt")
        return cls(
            prediction_completion_receipt_sha256=receipt.receipt_sha256,
            manifest_sha256=receipt.manifest_sha256,
            run_receipt_sha256=receipt.run_receipt_sha256,
            execution_artifact_sha256=receipt.execution_artifact_sha256,
            prediction_artifact_sha256=receipt.prediction_artifact_sha256,
            online_execution_result_receipt_sha256=(receipt.online_execution_result_receipt_sha256),
            action_panel_binding=receipt.action_panel_binding,
            prediction_count=receipt.prediction_count,
            corpus=receipt.corpus,
            stage=receipt.stage,
            external_anchor_identity=receipt.external_anchor_identity,
            external_anchor_uri=receipt.external_anchor_uri,
            anchored_at_utc=receipt.anchored_at_utc,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> PredictionCompletionAnchorRecord:
        row = _closed_mapping(
            payload,
            fields=_PREDICTION_ANCHOR_RECORD_FIELDS,
            label="prediction completion anchor record",
        )
        return cls(
            prediction_completion_receipt_sha256=row["prediction_completion_receipt_sha256"],
            manifest_sha256=row["manifest_sha256"],
            run_receipt_sha256=row["run_receipt_sha256"],
            execution_artifact_sha256=row["execution_artifact_sha256"],
            prediction_artifact_sha256=row["prediction_artifact_sha256"],
            online_execution_result_receipt_sha256=(row["online_execution_result_receipt_sha256"]),
            action_panel_binding=ActionPanelBinding.from_dict(row["action_panel_binding"]),
            prediction_count=row["prediction_count"],
            corpus=row["corpus"],
            stage=row["stage"],
            external_anchor_identity=row["external_anchor_identity"],
            external_anchor_uri=row["external_anchor_uri"],
            anchored_at_utc=row["anchored_at_utc"],
            schema_version=row["schema_version"],
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
            "online_execution_result_receipt_sha256": (self.online_execution_result_receipt_sha256),
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "prediction_completion_receipt_sha256": (self.prediction_completion_receipt_sha256),
            "prediction_count": self.prediction_count,
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
            "stage": self.stage,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def record_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes() + b"\n").hexdigest()


@dataclass(frozen=True)
class PredictionCompletionAnchorReceipt:
    """Local custody receipt for one exact external completion record."""

    anchor_record_sha256: str
    prediction_completion_receipt_sha256: str
    manifest_sha256: str
    run_receipt_sha256: str
    execution_artifact_sha256: str
    prediction_artifact_sha256: str
    online_execution_result_receipt_sha256: str
    action_panel_artifact_sha256: str
    corpus: str
    stage: str
    external_anchor_identity: str
    external_anchor_uri: str
    anchored_at_utc: str
    schema_version: str = PREDICTION_COMPLETION_ANCHOR_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "anchor_record_sha256",
            "prediction_completion_receipt_sha256",
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
            "prediction_artifact_sha256",
            "online_execution_result_receipt_sha256",
            "action_panel_artifact_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_text("corpus", self.corpus)
        _require_text("stage", self.stage)
        _require_text("external_anchor_identity", self.external_anchor_identity)
        _require_external_uri("external_anchor_uri", self.external_anchor_uri)
        _require_utc_timestamp("anchored_at_utc", self.anchored_at_utc)
        if self.schema_version != PREDICTION_COMPLETION_ANCHOR_RECEIPT_SCHEMA:
            raise ExternalAnchorError(
                "anchor receipt schema_version must equal "
                f"{PREDICTION_COMPLETION_ANCHOR_RECEIPT_SCHEMA!r}"
            )

    @classmethod
    def from_record(
        cls,
        record: PredictionCompletionAnchorRecord,
    ) -> PredictionCompletionAnchorReceipt:
        if not isinstance(record, PredictionCompletionAnchorRecord):
            raise ExternalAnchorError("record must be a PredictionCompletionAnchorRecord")
        return cls(
            anchor_record_sha256=record.record_sha256,
            prediction_completion_receipt_sha256=(record.prediction_completion_receipt_sha256),
            manifest_sha256=record.manifest_sha256,
            run_receipt_sha256=record.run_receipt_sha256,
            execution_artifact_sha256=record.execution_artifact_sha256,
            prediction_artifact_sha256=record.prediction_artifact_sha256,
            online_execution_result_receipt_sha256=(record.online_execution_result_receipt_sha256),
            action_panel_artifact_sha256=(record.action_panel_binding.action_panel_artifact_sha256),
            corpus=record.corpus,
            stage=record.stage,
            external_anchor_identity=record.external_anchor_identity,
            external_anchor_uri=record.external_anchor_uri,
            anchored_at_utc=record.anchored_at_utc,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> PredictionCompletionAnchorReceipt:
        row = _closed_mapping(
            payload,
            fields=_PREDICTION_ANCHOR_RECEIPT_FIELDS,
            label="prediction completion anchor receipt",
        )
        return cls(**row)

    def to_dict(self) -> dict[str, str]:
        return {
            "anchor_record_sha256": self.anchor_record_sha256,
            "anchored_at_utc": self.anchored_at_utc,
            "action_panel_artifact_sha256": self.action_panel_artifact_sha256,
            "corpus": self.corpus,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "external_anchor_identity": self.external_anchor_identity,
            "external_anchor_uri": self.external_anchor_uri,
            "manifest_sha256": self.manifest_sha256,
            "online_execution_result_receipt_sha256": (self.online_execution_result_receipt_sha256),
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "prediction_completion_receipt_sha256": (self.prediction_completion_receipt_sha256),
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
            "stage": self.stage,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class VerifiedPredictionCompletionAnchor:
    """Evidence returned only after local and remote byte verification."""

    record: PredictionCompletionAnchorRecord
    receipt: PredictionCompletionAnchorReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.record, PredictionCompletionAnchorRecord):
            raise ExternalAnchorError("record must be a PredictionCompletionAnchorRecord")
        if not isinstance(self.receipt, PredictionCompletionAnchorReceipt):
            raise ExternalAnchorError("receipt must be a PredictionCompletionAnchorReceipt")
        if self.record.record_sha256 != self.receipt.anchor_record_sha256:
            raise ExternalAnchorError("verified anchor has mismatched record digest")


def write_prediction_completion_anchor_record(
    record: PredictionCompletionAnchorRecord,
    target: str | Path,
) -> None:
    if not isinstance(record, PredictionCompletionAnchorRecord):
        raise ExternalAnchorError("record must be a PredictionCompletionAnchorRecord")
    _write_canonical_file(
        record,
        target,
        label="prediction completion anchor record",
    )


def load_prediction_completion_anchor_record(
    path: str | Path,
) -> PredictionCompletionAnchorRecord:
    record, _ = _read_canonical_file(
        path,
        label="prediction completion anchor record",
        parser=PredictionCompletionAnchorRecord.from_dict,
    )
    return record


def write_prediction_completion_anchor_receipt(
    receipt: PredictionCompletionAnchorReceipt,
    target: str | Path,
) -> None:
    if not isinstance(receipt, PredictionCompletionAnchorReceipt):
        raise ExternalAnchorError("receipt must be a PredictionCompletionAnchorReceipt")
    _write_canonical_file(
        receipt,
        target,
        label="prediction completion anchor receipt",
    )


def load_prediction_completion_anchor_receipt(
    path: str | Path,
) -> PredictionCompletionAnchorReceipt:
    receipt, _ = _read_canonical_file(
        path,
        label="prediction completion anchor receipt",
        parser=PredictionCompletionAnchorReceipt.from_dict,
    )
    return receipt


class _NoExternalAnchorRedirects(urllib_request.HTTPRedirectHandler):
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
        raise ExternalAnchorError(
            f"external anchor verification refused HTTP redirect status {code}"
        )


def _fetch_external_anchor_record(uri: str, max_bytes: int) -> bytes:
    _require_external_uri("external_anchor_uri", uri)
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes <= 0
        or max_bytes > MAX_EXTERNAL_ANCHOR_RECORD_BYTES
    ):
        raise ExternalAnchorError("external anchor fetch exceeds the safety limit")
    context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    opener = urllib_request.build_opener(
        _NoExternalAnchorRedirects(),
        urllib_request.HTTPSHandler(context=context),
    )
    request = urllib_request.Request(
        uri,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "fractal-ann-diagnostics/0.3 prediction-anchor-revalidation",
        },
        method="GET",
    )
    try:
        with opener.open(
            request,
            timeout=_EXTERNAL_FETCH_TIMEOUT_SECONDS,
        ) as response:
            status = response.getcode()
            if status != 200:
                if isinstance(status, int) and 300 <= status < 400:
                    raise ExternalAnchorError(
                        "external anchor verification refused an HTTP redirect"
                    )
                raise ExternalAnchorError(
                    f"external anchor verification returned HTTP status {status}"
                )
            if response.geturl() != uri:
                raise ExternalAnchorError("external anchor verification response URL changed")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                if not content_length.isdecimal():
                    raise ExternalAnchorError(
                        "external anchor response has an invalid Content-Length"
                    )
                if int(content_length) > max_bytes:
                    raise ExternalAnchorError(
                        "external anchor record exceeds the maximum byte limit"
                    )
            encoded = response.read(max_bytes + 1)
    except ExternalAnchorError:
        raise
    except urllib_error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise ExternalAnchorError(
                "external anchor verification refused an HTTP redirect"
            ) from exc
        raise ExternalAnchorError(
            f"external anchor verification returned HTTP status {exc.code}"
        ) from exc
    except (OSError, TimeoutError, urllib_error.URLError, ValueError) as exc:
        raise ExternalAnchorError(
            "external anchor record could not be fetched over verified HTTPS"
        ) from exc
    if not isinstance(encoded, bytes):
        raise ExternalAnchorError("external anchor fetcher must return bytes")
    if len(encoded) > max_bytes:
        raise ExternalAnchorError("external anchor record exceeds the maximum byte limit")
    return encoded


def _assert_anchor_bindings(
    completion: PredictionCompletionReceipt,
    record: PredictionCompletionAnchorRecord,
    receipt: PredictionCompletionAnchorReceipt,
) -> None:
    expected: tuple[tuple[str, object, object], ...] = (
        (
            "prediction_completion_receipt_sha256",
            record.prediction_completion_receipt_sha256,
            completion.receipt_sha256,
        ),
        ("manifest_sha256", record.manifest_sha256, completion.manifest_sha256),
        (
            "run_receipt_sha256",
            record.run_receipt_sha256,
            completion.run_receipt_sha256,
        ),
        (
            "execution_artifact_sha256",
            record.execution_artifact_sha256,
            completion.execution_artifact_sha256,
        ),
        (
            "prediction_artifact_sha256",
            record.prediction_artifact_sha256,
            completion.prediction_artifact_sha256,
        ),
        (
            "online_execution_result_receipt_sha256",
            record.online_execution_result_receipt_sha256,
            completion.online_execution_result_receipt_sha256,
        ),
        ("prediction_count", record.prediction_count, completion.prediction_count),
        ("corpus", record.corpus, completion.corpus),
        ("stage", record.stage, completion.stage),
        (
            "external_anchor_identity",
            record.external_anchor_identity,
            completion.external_anchor_identity,
        ),
        (
            "external_anchor_uri",
            record.external_anchor_uri,
            completion.external_anchor_uri,
        ),
        ("anchored_at_utc", record.anchored_at_utc, completion.anchored_at_utc),
        (
            "action_panel_binding",
            record.action_panel_binding,
            completion.action_panel_binding,
        ),
        ("anchor_record_sha256", receipt.anchor_record_sha256, record.record_sha256),
        (
            "receipt.prediction_completion_receipt_sha256",
            receipt.prediction_completion_receipt_sha256,
            completion.receipt_sha256,
        ),
        ("receipt.manifest_sha256", receipt.manifest_sha256, record.manifest_sha256),
        (
            "receipt.run_receipt_sha256",
            receipt.run_receipt_sha256,
            record.run_receipt_sha256,
        ),
        (
            "receipt.execution_artifact_sha256",
            receipt.execution_artifact_sha256,
            record.execution_artifact_sha256,
        ),
        (
            "receipt.prediction_artifact_sha256",
            receipt.prediction_artifact_sha256,
            record.prediction_artifact_sha256,
        ),
        (
            "receipt.online_execution_result_receipt_sha256",
            receipt.online_execution_result_receipt_sha256,
            record.online_execution_result_receipt_sha256,
        ),
        (
            "receipt.action_panel_artifact_sha256",
            receipt.action_panel_artifact_sha256,
            record.action_panel_binding.action_panel_artifact_sha256,
        ),
        ("receipt.corpus", receipt.corpus, record.corpus),
        ("receipt.stage", receipt.stage, record.stage),
        (
            "receipt.external_anchor_identity",
            receipt.external_anchor_identity,
            record.external_anchor_identity,
        ),
        (
            "receipt.external_anchor_uri",
            receipt.external_anchor_uri,
            record.external_anchor_uri,
        ),
        (
            "receipt.anchored_at_utc",
            receipt.anchored_at_utc,
            record.anchored_at_utc,
        ),
    )
    for name, observed, wanted in expected:
        if observed != wanted:
            raise ExternalAnchorError(f"prediction completion anchor has mismatched {name}")


def verify_prediction_completion_anchor(
    completion_receipt: PredictionCompletionReceipt,
    *,
    anchor_record_path: str | Path,
    anchor_receipt_path: str | Path,
    trusted_anchor_record_fetcher: Callable[[str, int], bytes] | None = None,
) -> VerifiedPredictionCompletionAnchor:
    """Revalidate an exact external completion record before label release.

    The default path performs one certificate-validated HTTPS GET and refuses
    redirects.  The injectable fetcher is an explicit test/integration seam;
    callers that inject it assume responsibility for transport authentication.
    Exact digest and byte comparisons still apply.
    """

    if not isinstance(completion_receipt, PredictionCompletionReceipt):
        raise ExternalAnchorError("completion_receipt must be a PredictionCompletionReceipt")
    if trusted_anchor_record_fetcher is not None and not callable(trusted_anchor_record_fetcher):
        raise ExternalAnchorError("trusted_anchor_record_fetcher must be callable or None")
    record, local_bytes = _read_canonical_file(
        anchor_record_path,
        label="prediction completion anchor record",
        parser=PredictionCompletionAnchorRecord.from_dict,
    )
    receipt = load_prediction_completion_anchor_receipt(anchor_receipt_path)
    _assert_anchor_bindings(completion_receipt, record, receipt)
    fetcher = (
        _fetch_external_anchor_record
        if trusted_anchor_record_fetcher is None
        else trusted_anchor_record_fetcher
    )
    try:
        fetched = fetcher(
            record.external_anchor_uri,
            MAX_EXTERNAL_ANCHOR_RECORD_BYTES,
        )
    except ExternalAnchorError:
        raise
    except Exception as exc:
        raise ExternalAnchorError(
            "trusted external anchor fetcher failed during revalidation"
        ) from exc
    if not isinstance(fetched, bytes):
        raise ExternalAnchorError("external anchor fetcher must return bytes")
    if len(fetched) > MAX_EXTERNAL_ANCHOR_RECORD_BYTES:
        raise ExternalAnchorError("external anchor record exceeds the maximum byte limit")
    fetched_digest = hashlib.sha256(fetched).hexdigest()
    if not hmac.compare_digest(fetched_digest, receipt.anchor_record_sha256):
        raise ExternalAnchorError(
            "fetched external anchor record digest does not match its receipt"
        )
    if not hmac.compare_digest(fetched, local_bytes):
        raise ExternalAnchorError(
            "fetched external anchor bytes do not match the secure local record"
        )
    return VerifiedPredictionCompletionAnchor(record=record, receipt=receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fractal_ann_diagnostics.external_anchors")
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser("write-protocol-registry-record")
    record.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--registered-at-utc", required=True)
    record.add_argument("--registry-identity", required=True)
    record.add_argument("--registry-uri", required=True)
    record.add_argument("--output", type=Path, required=True)

    registration = commands.add_parser("write-protocol-registration-receipt")
    registration.add_argument("--manifest", type=Path, required=True)
    registration.add_argument("--registry-record", type=Path, required=True)
    registration.add_argument("--output", type=Path, required=True)

    prediction_record = commands.add_parser("write-prediction-completion-anchor-record")
    prediction_record.add_argument("--completion-receipt", type=Path, required=True)
    prediction_record.add_argument("--output", type=Path, required=True)

    prediction_receipt = commands.add_parser("write-prediction-completion-anchor-receipt")
    prediction_receipt.add_argument("--anchor-record", type=Path, required=True)
    prediction_receipt.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-prediction-completion-anchor")
    verify.add_argument("--completion-receipt", type=Path, required=True)
    verify.add_argument("--anchor-record", type=Path, required=True)
    verify.add_argument("--anchor-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write-protocol-registry-record":
        record = create_protocol_registry_record(
            args.manifest,
            registered_at_utc=args.registered_at_utc,
            registry_identity=args.registry_identity,
            registry_uri=args.registry_uri,
        )
        write_protocol_registry_record(record, args.output)
        print(f"protocol registry record sha256: {record.record_sha256}")
        print(f"record: {args.output}")
        return 0
    if args.command == "write-protocol-registration-receipt":
        receipt = create_protocol_registration_receipt(
            args.manifest,
            args.registry_record,
        )
        write_protocol_registration_receipt(receipt, args.output)
        print(f"protocol registration receipt sha256: {receipt.receipt_sha256}")
        print(f"receipt: {args.output}")
        return 0
    if args.command == "write-prediction-completion-anchor-record":
        completion = load_prediction_completion_receipt(args.completion_receipt)
        record = PredictionCompletionAnchorRecord.from_completion_receipt(completion)
        write_prediction_completion_anchor_record(record, args.output)
        print(f"prediction anchor record sha256: {record.record_sha256}")
        print(f"record: {args.output}")
        return 0
    if args.command == "write-prediction-completion-anchor-receipt":
        record = load_prediction_completion_anchor_record(args.anchor_record)
        receipt = PredictionCompletionAnchorReceipt.from_record(record)
        write_prediction_completion_anchor_receipt(receipt, args.output)
        print(f"prediction anchor receipt sha256: {receipt.receipt_sha256}")
        print(f"receipt: {args.output}")
        return 0
    if args.command == "verify-prediction-completion-anchor":
        completion = load_prediction_completion_receipt(args.completion_receipt)
        verified = verify_prediction_completion_anchor(
            completion,
            anchor_record_path=args.anchor_record,
            anchor_receipt_path=args.anchor_receipt,
        )
        print(f"verified prediction completion anchor: {verified.record.record_sha256}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
