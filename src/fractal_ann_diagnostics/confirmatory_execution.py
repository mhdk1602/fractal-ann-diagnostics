"""Custody controls for the single admitted confirmatory analysis attempt."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    read_secure_control_file,
    write_exclusive_receipt_bytes,
)
from .confirmatory_analysis import (
    ConfirmatoryAnalysisError,
    ConfirmatoryInputArtifact,
    ConfirmatoryResultArtifact,
    run_confirmatory_analysis,
)
from .confirmatory_modeling import FrozenModelSuite

CONFIRMATORY_ANALYSIS_ATTEMPT_SCHEMA = "fractal-confirmatory-analysis-attempt-v1"
CONFIRMATORY_ANALYSIS_RESULT_RECEIPT_SCHEMA = (
    "fractal-confirmatory-analysis-result-receipt-v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_SUFFIX = ".confirmatory-analysis-attempt.json"
_RESULT_SUFFIX = ".confirmatory-result.json"
_RESULT_RECEIPT_SUFFIX = ".confirmatory-result-receipt.json"


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfirmatoryAnalysisError(
            "confirmatory execution evidence must be finite canonical JSON"
        ) from exc


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ConfirmatoryAnalysisError(
            f"{name} must be a lowercase SHA-256 hex digest"
        )
    return value


def _require_identity(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfirmatoryAnalysisError(
            "runner_identity must be a non-empty canonical string"
        )
    if unicodedata.normalize("NFC", value) != value:
        raise ConfirmatoryAnalysisError("runner_identity must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfirmatoryAnalysisError(
            "runner_identity cannot contain control characters"
        )
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ConfirmatoryAnalysisError(
            "runner_identity must contain valid Unicode"
        ) from exc
    return value


def _closed_mapping(
    payload: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or not all(
        isinstance(key, str) for key in payload
    ):
        raise ConfirmatoryAnalysisError(f"{label} must be a JSON object")
    missing = fields - set(payload)
    unexpected = set(payload) - fields
    if missing or unexpected:
        raise ConfirmatoryAnalysisError(
            f"{label} keys do not match the closed schema; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return payload


def _decode_canonical_object(
    encoded: bytes,
    *,
    label: str,
) -> Mapping[str, Any]:
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ConfirmatoryAnalysisError(f"{label} must be valid UTF-8") from exc

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise ConfirmatoryAnalysisError(
                    f"{label} contains duplicate key {key!r}"
                )
            decoded[key] = value
        return decoded

    def reject_nonfinite(value: str) -> None:
        raise ConfirmatoryAnalysisError(
            f"{label} contains non-finite number {value!r}"
        )

    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ConfirmatoryAnalysisError(
            f"{label} must contain valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ConfirmatoryAnalysisError(f"{label} must contain one JSON object")
    return payload


def _canonical_file_uri_path(uri: object, *, label: str) -> Path:
    if not isinstance(uri, str) or not uri or "\x00" in uri:
        raise ConfirmatoryAnalysisError(f"{label} must be a canonical file URI")
    if any(ord(character) < 32 or ord(character) == 127 for character in uri):
        raise ConfirmatoryAnalysisError(f"{label} cannot contain control characters")
    try:
        parsed = urlsplit(uri)
    except ValueError as exc:
        raise ConfirmatoryAnalysisError(f"{label} is not a valid URI") from exc
    if parsed.scheme in {"s3", "gs"}:
        raise ConfirmatoryAnalysisError(
            f"{label} uses an unsupported remote store; the built-in runner requires "
            "a canonical file URI"
        )
    if parsed.scheme != "file":
        raise ConfirmatoryAnalysisError(f"{label} must use the file URI scheme")
    if parsed.netloc:
        raise ConfirmatoryAnalysisError(f"{label} cannot contain an authority")
    if parsed.query or parsed.fragment:
        raise ConfirmatoryAnalysisError(f"{label} cannot contain a query or fragment")
    if not parsed.path.startswith("/"):
        raise ConfirmatoryAnalysisError(f"{label} must contain an absolute POSIX path")
    try:
        decoded = unquote_to_bytes(parsed.path).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ConfirmatoryAnalysisError(
            f"{label} path must be valid UTF-8"
        ) from exc
    if (
        not decoded.startswith("/")
        or "\\" in decoded
        or unicodedata.normalize("NFC", decoded) != decoded
    ):
        raise ConfirmatoryAnalysisError(f"{label} path is not canonical")
    components = decoded.split("/")[1:]
    if any(component in {"", ".", ".."} for component in components):
        if decoded != "/":
            raise ConfirmatoryAnalysisError(
                f"{label} path cannot contain empty, dot, or parent components"
            )
    if any(
        ord(character) < 32 or ord(character) == 127 for character in decoded
    ):
        raise ConfirmatoryAnalysisError(f"{label} path contains a control character")
    path = Path(decoded)
    if not path.is_absolute() or path.anchor != "/" or path.as_uri() != uri:
        raise ConfirmatoryAnalysisError(f"{label} must use canonical file URI encoding")
    return path


def _results_store_directory(inputs: ConfirmatoryInputArtifact) -> Path:
    if not isinstance(inputs, ConfirmatoryInputArtifact):
        raise ConfirmatoryAnalysisError("inputs must be a ConfirmatoryInputArtifact")
    sealed = inputs.frozen_manifest.get("sealed_execution")
    if not isinstance(sealed, Mapping):
        raise ConfirmatoryAnalysisError("frozen manifest has no sealed_execution object")
    if "results_store" not in sealed:
        raise ConfirmatoryAnalysisError("frozen manifest has no pinned results_store")
    return _canonical_file_uri_path(
        sealed["results_store"],
        label="sealed_execution.results_store",
    )


def _manifest_digest(inputs: ConfirmatoryInputArtifact) -> str:
    return _require_sha256("manifest_sha256", inputs.manifest_sha256)


def confirmatory_attempt_path(inputs: ConfirmatoryInputArtifact) -> Path:
    """Return the manifest-derived exclusive attempt-receipt path."""

    return _results_store_directory(inputs) / f"{_manifest_digest(inputs)}{_ATTEMPT_SUFFIX}"


def confirmatory_result_path(inputs: ConfirmatoryInputArtifact) -> Path:
    """Return the manifest-derived sole result-artifact path."""

    return _results_store_directory(inputs) / f"{_manifest_digest(inputs)}{_RESULT_SUFFIX}"


def confirmatory_result_receipt_path(inputs: ConfirmatoryInputArtifact) -> Path:
    """Return the manifest-derived detached result-receipt path."""

    return _results_store_directory(inputs) / (
        f"{_manifest_digest(inputs)}{_RESULT_RECEIPT_SUFFIX}"
    )


@dataclass(frozen=True)
class ConfirmatoryAnalysisAttemptReceipt:
    """Canonical admission evidence written before any outcome computation."""

    manifest_sha256: str
    run_receipt_sha256: str
    confirmatory_input_artifact_sha256: str
    model_suite_sha256: str
    runner_identity: str
    result_uri: str
    schema_version: str = CONFIRMATORY_ANALYSIS_ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFIRMATORY_ANALYSIS_ATTEMPT_SCHEMA:
            raise ConfirmatoryAnalysisError(
                f"schema_version must equal {CONFIRMATORY_ANALYSIS_ATTEMPT_SCHEMA!r}"
            )
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "confirmatory_input_artifact_sha256",
            "model_suite_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_identity(self.runner_identity)
        result_path = _canonical_file_uri_path(self.result_uri, label="result_uri")
        if result_path.name != f"{self.manifest_sha256}{_RESULT_SUFFIX}":
            raise ConfirmatoryAnalysisError(
                "result_uri must name the manifest-derived confirmatory result"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "confirmatory_input_artifact_sha256": (
                self.confirmatory_input_artifact_sha256
            ),
            "manifest_sha256": self.manifest_sha256,
            "model_suite_sha256": self.model_suite_sha256,
            "result_uri": self.result_uri,
            "run_receipt_sha256": self.run_receipt_sha256,
            "runner_identity": self.runner_identity,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: object) -> ConfirmatoryAnalysisAttemptReceipt:
        row = _closed_mapping(
            payload,
            fields={
                "confirmatory_input_artifact_sha256",
                "manifest_sha256",
                "model_suite_sha256",
                "result_uri",
                "run_receipt_sha256",
                "runner_identity",
                "schema_version",
            },
            label="confirmatory analysis attempt receipt",
        )
        return cls(
            manifest_sha256=row["manifest_sha256"],
            run_receipt_sha256=row["run_receipt_sha256"],
            confirmatory_input_artifact_sha256=row[
                "confirmatory_input_artifact_sha256"
            ],
            model_suite_sha256=row["model_suite_sha256"],
            runner_identity=row["runner_identity"],
            result_uri=row["result_uri"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class ConfirmatoryAnalysisResultReceipt:
    """Detached evidence binding the immutable result to its admitted attempt."""

    manifest_sha256: str
    attempt_receipt_sha256: str
    result_artifact_sha256: str
    result_uri: str
    schema_version: str = CONFIRMATORY_ANALYSIS_RESULT_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFIRMATORY_ANALYSIS_RESULT_RECEIPT_SCHEMA:
            raise ConfirmatoryAnalysisError(
                "schema_version must equal "
                f"{CONFIRMATORY_ANALYSIS_RESULT_RECEIPT_SCHEMA!r}"
            )
        for name in (
            "manifest_sha256",
            "attempt_receipt_sha256",
            "result_artifact_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        result_path = _canonical_file_uri_path(self.result_uri, label="result_uri")
        if result_path.name != f"{self.manifest_sha256}{_RESULT_SUFFIX}":
            raise ConfirmatoryAnalysisError(
                "result_uri must name the manifest-derived confirmatory result"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "attempt_receipt_sha256": self.attempt_receipt_sha256,
            "manifest_sha256": self.manifest_sha256,
            "result_artifact_sha256": self.result_artifact_sha256,
            "result_uri": self.result_uri,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: object) -> ConfirmatoryAnalysisResultReceipt:
        row = _closed_mapping(
            payload,
            fields={
                "attempt_receipt_sha256",
                "manifest_sha256",
                "result_artifact_sha256",
                "result_uri",
                "schema_version",
            },
            label="confirmatory analysis result receipt",
        )
        return cls(
            manifest_sha256=row["manifest_sha256"],
            attempt_receipt_sha256=row["attempt_receipt_sha256"],
            result_artifact_sha256=row["result_artifact_sha256"],
            result_uri=row["result_uri"],
            schema_version=row["schema_version"],
        )


def _load_canonical_receipt(
    path: str | Path,
    *,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    try:
        encoded = read_secure_control_file(path, label=label)
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryAnalysisError(f"cannot load {label}: {exc}") from exc
    payload = _decode_canonical_object(encoded, label=label)
    return payload, encoded


def _assert_receipt_location(
    path: str | Path,
    *,
    manifest_sha256: str,
    result_uri: str,
    suffix: str,
    label: str,
) -> None:
    target = Path(path)
    result_path = _canonical_file_uri_path(result_uri, label="result_uri")
    if (
        not target.is_absolute()
        or target.name != f"{manifest_sha256}{suffix}"
        or target.parent != result_path.parent
    ):
        raise ConfirmatoryAnalysisError(
            f"{label} is not at its manifest-derived results-store path"
        )


def load_confirmatory_analysis_attempt_receipt(
    path: str | Path,
) -> ConfirmatoryAnalysisAttemptReceipt:
    """Securely load and verify canonical attempt-admission evidence."""

    payload, encoded = _load_canonical_receipt(
        path,
        label="confirmatory analysis attempt receipt",
    )
    receipt = ConfirmatoryAnalysisAttemptReceipt.from_dict(payload)
    _assert_receipt_location(
        path,
        manifest_sha256=receipt.manifest_sha256,
        result_uri=receipt.result_uri,
        suffix=_ATTEMPT_SUFFIX,
        label="confirmatory analysis attempt receipt",
    )
    if encoded != receipt.canonical_bytes() + b"\n":
        raise ConfirmatoryAnalysisError(
            "confirmatory analysis attempt receipt bytes are not canonical"
        )
    return receipt


def load_confirmatory_analysis_result_receipt(
    path: str | Path,
) -> ConfirmatoryAnalysisResultReceipt:
    """Securely load the canonical detached result receipt."""

    payload, encoded = _load_canonical_receipt(
        path,
        label="confirmatory analysis result receipt",
    )
    receipt = ConfirmatoryAnalysisResultReceipt.from_dict(payload)
    _assert_receipt_location(
        path,
        manifest_sha256=receipt.manifest_sha256,
        result_uri=receipt.result_uri,
        suffix=_RESULT_RECEIPT_SUFFIX,
        label="confirmatory analysis result receipt",
    )
    if encoded != receipt.canonical_bytes() + b"\n":
        raise ConfirmatoryAnalysisError(
            "confirmatory analysis result receipt bytes are not canonical"
        )
    return receipt


def load_confirmatory_result_artifact_bytes(
    path: str | Path,
    *,
    result_receipt_path: str | Path,
    attempt_receipt_path: str | Path,
) -> bytes:
    """Load canonical result bytes and verify the full attempt-to-result chain."""

    receipt = load_confirmatory_analysis_result_receipt(result_receipt_path)
    attempt = load_confirmatory_analysis_attempt_receipt(attempt_receipt_path)
    expected_receipt_bindings = {
        "manifest_sha256": attempt.manifest_sha256,
        "attempt_receipt_sha256": attempt.receipt_sha256,
        "result_uri": attempt.result_uri,
    }
    for name, expected in expected_receipt_bindings.items():
        if getattr(receipt, name) != expected:
            raise ConfirmatoryAnalysisError(
                f"confirmatory result receipt {name} does not match the admitted attempt"
            )
    target = Path(path)
    if not target.is_absolute() or target.as_uri() != receipt.result_uri:
        raise ConfirmatoryAnalysisError("result path does not match receipt result_uri")
    try:
        encoded = read_secure_control_file(target, label="confirmatory result artifact")
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryAnalysisError(
            f"cannot load confirmatory result artifact: {exc}"
        ) from exc
    payload = _decode_canonical_object(encoded, label="confirmatory result artifact")
    canonical = _canonical_bytes(payload)
    if encoded != canonical + b"\n":
        raise ConfirmatoryAnalysisError(
            "confirmatory result artifact bytes are not canonical"
        )
    if hashlib.sha256(canonical).hexdigest() != receipt.result_artifact_sha256:
        raise ConfirmatoryAnalysisError(
            "confirmatory result artifact does not match its detached receipt"
        )
    expected_result_bindings = {
        "manifest_sha256": attempt.manifest_sha256,
        "run_receipt_sha256": attempt.run_receipt_sha256,
        "confirmatory_input_artifact_sha256": (
            attempt.confirmatory_input_artifact_sha256
        ),
        "model_suite_sha256": attempt.model_suite_sha256,
    }
    for name, expected in expected_result_bindings.items():
        if payload.get(name) != expected:
            raise ConfirmatoryAnalysisError(
                f"confirmatory result {name} does not match the admitted attempt"
            )
    return canonical


def _attempt_receipt(
    inputs: ConfirmatoryInputArtifact,
    *,
    suite: FrozenModelSuite,
) -> ConfirmatoryAnalysisAttemptReceipt:
    return ConfirmatoryAnalysisAttemptReceipt(
        manifest_sha256=_manifest_digest(inputs),
        run_receipt_sha256=_require_sha256(
            "run_receipt_sha256", inputs.run_receipt_sha256
        ),
        confirmatory_input_artifact_sha256=_require_sha256(
            "confirmatory_input_artifact_sha256", inputs.artifact_sha256
        ),
        model_suite_sha256=_require_sha256("model_suite_sha256", suite.suite_digest),
        runner_identity=inputs.run_receipt.runner_identity,
        result_uri=confirmatory_result_path(inputs).as_uri(),
    )


def _assert_result_binding(
    result: ConfirmatoryResultArtifact,
    *,
    attempt: ConfirmatoryAnalysisAttemptReceipt,
) -> None:
    if not isinstance(result, ConfirmatoryResultArtifact):
        raise ConfirmatoryAnalysisError(
            "analysis runner must return a ConfirmatoryResultArtifact"
        )
    expected = {
        "manifest_sha256": attempt.manifest_sha256,
        "run_receipt_sha256": attempt.run_receipt_sha256,
        "confirmatory_input_artifact_sha256": (
            attempt.confirmatory_input_artifact_sha256
        ),
        "model_suite_sha256": attempt.model_suite_sha256,
    }
    for name, value in expected.items():
        if getattr(result, name) != value:
            raise ConfirmatoryAnalysisError(
                f"confirmatory result {name} does not match the admitted attempt"
            )


def run_confirmatory_analysis_once(
    inputs: ConfirmatoryInputArtifact,
    *,
    suite: FrozenModelSuite,
) -> ConfirmatoryResultArtifact:
    """Admit one attempt before computation, then persist its bound result."""

    if not isinstance(inputs, ConfirmatoryInputArtifact):
        raise ConfirmatoryAnalysisError("inputs must be a ConfirmatoryInputArtifact")
    if not isinstance(suite, FrozenModelSuite):
        raise ConfirmatoryAnalysisError("suite must be a FrozenModelSuite")

    # Admission validation may inspect frozen bytes, but outcome analysis must not start
    # until the exclusive attempt receipt is durable.
    inputs.assert_model_suite_admitted(suite)
    attempt = _attempt_receipt(inputs, suite=suite)
    attempt_target = confirmatory_attempt_path(inputs)
    try:
        write_exclusive_receipt_bytes(
            attempt.canonical_bytes() + b"\n",
            attempt_target,
        )
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryAnalysisError(
            "confirmatory analysis attempt was not admitted exclusively at "
            f"{attempt_target}: {exc}"
        ) from exc

    result = run_confirmatory_analysis(inputs, suite=suite)
    _assert_result_binding(result, attempt=attempt)

    result_target = confirmatory_result_path(inputs)
    result_receipt = ConfirmatoryAnalysisResultReceipt(
        manifest_sha256=attempt.manifest_sha256,
        attempt_receipt_sha256=attempt.receipt_sha256,
        result_artifact_sha256=result.artifact_sha256,
        result_uri=result_target.as_uri(),
    )
    receipt_target = confirmatory_result_receipt_path(inputs)
    try:
        # Reserve and bind the detached custody evidence before exposing the result.
        write_exclusive_receipt_bytes(
            result_receipt.canonical_bytes() + b"\n",
            receipt_target,
        )
        write_exclusive_receipt_bytes(result.canonical_bytes() + b"\n", result_target)
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryAnalysisError(
            f"confirmatory result custody write failed: {exc}"
        ) from exc
    return result
