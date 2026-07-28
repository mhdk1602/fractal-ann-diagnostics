"""Custody controls for the single admitted confirmatory analysis attempt."""

from __future__ import annotations

import hashlib
import json
import math
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
    ConfirmatoryAnalysisConfig,
    ConfirmatoryAnalysisError,
    ConfirmatoryInputArtifact,
    ConfirmatoryResultArtifact,
    CorpusGeometryResult,
    CorpusInputDigests,
    DirectionalGate,
    EntitlementResult,
    H1Result,
    H2Result,
    H3Result,
    PositionAdjustedSensitivityResult,
    run_confirmatory_analysis,
)
from .confirmatory_modeling import FrozenModelSuite, GeometryGainThresholds

CONFIRMATORY_ANALYSIS_ATTEMPT_SCHEMA = "fractal-confirmatory-analysis-attempt-v1"
CONFIRMATORY_ANALYSIS_RESULT_RECEIPT_SCHEMA = "fractal-confirmatory-analysis-result-receipt-v1"

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
        raise ConfirmatoryAnalysisError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_identity(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfirmatoryAnalysisError("runner_identity must be a non-empty canonical string")
    if unicodedata.normalize("NFC", value) != value:
        raise ConfirmatoryAnalysisError("runner_identity must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfirmatoryAnalysisError("runner_identity cannot contain control characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ConfirmatoryAnalysisError("runner_identity must contain valid Unicode") from exc
    return value


def _closed_mapping(
    payload: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
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
                raise ConfirmatoryAnalysisError(f"{label} contains duplicate key {key!r}")
            decoded[key] = value
        return decoded

    def reject_nonfinite(value: str) -> None:
        raise ConfirmatoryAnalysisError(f"{label} contains non-finite number {value!r}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ConfirmatoryAnalysisError(f"{label} must contain valid JSON: {exc.msg}") from exc
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
        raise ConfirmatoryAnalysisError(f"{label} path must be valid UTF-8") from exc
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
    if any(ord(character) < 32 or ord(character) == 127 for character in decoded):
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


def confirmatory_output_filenames(manifest_sha256: str) -> tuple[str, str, str]:
    """Return the three registered outcome filenames in bytewise order."""

    digest = _require_sha256("manifest_sha256", manifest_sha256)
    return tuple(
        sorted(
            (
                f"{digest}{_ATTEMPT_SUFFIX}",
                f"{digest}{_RESULT_RECEIPT_SUFFIX}",
                f"{digest}{_RESULT_SUFFIX}",
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )  # type: ignore[return-value]


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
            "confirmatory_input_artifact_sha256": (self.confirmatory_input_artifact_sha256),
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
            confirmatory_input_artifact_sha256=row["confirmatory_input_artifact_sha256"],
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
                f"schema_version must equal {CONFIRMATORY_ANALYSIS_RESULT_RECEIPT_SCHEMA!r}"
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


def _typed_array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfirmatoryAnalysisError(f"{label} must be an array")
    return value


def _typed_boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ConfirmatoryAnalysisError(f"{label} must be boolean")
    return value


def _typed_integer(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ConfirmatoryAnalysisError(f"{label} must be an integer no smaller than {minimum}")
    return value


def _typed_number(
    value: object,
    *,
    label: str,
    nullable: bool = False,
) -> float | None:
    if nullable and value is None:
        return None
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ConfirmatoryAnalysisError(f"{label} must be a finite number")
    return float(value)


def _typed_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ConfirmatoryAnalysisError(f"{label} must be canonical non-empty text")
    return value


def _typed_string_array(value: object, *, label: str) -> tuple[str, ...]:
    rows = _typed_array(value, label=label)
    return tuple(_typed_text(item, label=f"{label} item") for item in rows)


def _typed_number_mapping(
    value: object,
    *,
    label: str,
) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ConfirmatoryAnalysisError(f"{label} must be a string-number object")
    return tuple(
        sorted(
            (
                (
                    _typed_text(key, label=f"{label} key"),
                    float(_typed_number(item, label=f"{label}.{key}")),
                )
                for key, item in value.items()
            ),
            key=lambda row: row[0].encode("utf-8"),
        )
    )


def _typed_analysis_config(value: object) -> ConfirmatoryAnalysisConfig:
    row = _closed_mapping(
        value,
        fields=set(ConfirmatoryAnalysisConfig.__dataclass_fields__),
        label="confirmatory result frozen_config",
    )
    thresholds = _closed_mapping(
        row["geometry_gain_thresholds"],
        fields={
            "auprc_gain",
            "brier_score_reduction",
            "log_loss_reduction",
        },
        label="confirmatory result geometry thresholds",
    )
    try:
        config = ConfirmatoryAnalysisConfig(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "action_set",
                    "evidence_corpora",
                    "fixed_corpora",
                    "geometry_gain_thresholds",
                    "high_geometry",
                    "low_geometry",
                }
            },
            action_set=_typed_string_array(
                row["action_set"],
                label="confirmatory result action_set",
            ),
            evidence_corpora=_typed_string_array(
                row["evidence_corpora"],
                label="confirmatory result evidence_corpora",
            ),
            fixed_corpora=_typed_string_array(
                row["fixed_corpora"],
                label="confirmatory result fixed_corpora",
            ),
            geometry_gain_thresholds=GeometryGainThresholds(
                auprc_gain=float(
                    _typed_number(
                        thresholds["auprc_gain"],
                        label="confirmatory result auprc threshold",
                    )
                ),
                brier_score_reduction=float(
                    _typed_number(
                        thresholds["brier_score_reduction"],
                        label="confirmatory result brier threshold",
                    )
                ),
                log_loss_reduction=float(
                    _typed_number(
                        thresholds["log_loss_reduction"],
                        label="confirmatory result log-loss threshold",
                    )
                ),
            ),
            high_geometry=_typed_number_mapping(
                row["high_geometry"],
                label="confirmatory result high_geometry",
            ),
            low_geometry=_typed_number_mapping(
                row["low_geometry"],
                label="confirmatory result low_geometry",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfirmatoryAnalysisError(
            f"confirmatory result frozen_config is invalid: {exc}"
        ) from exc
    return config


def _typed_directional_gate(value: object, *, label: str) -> DirectionalGate:
    row = _closed_mapping(
        value,
        fields=set(DirectionalGate.__dataclass_fields__),
        label=label,
    )
    gate = DirectionalGate(
        name=_typed_text(row["name"], label=f"{label} name"),
        estimate=_typed_number(
            row["estimate"],
            label=f"{label} estimate",
            nullable=True,
        ),
        lower=_typed_number(
            row["lower"],
            label=f"{label} lower",
            nullable=True,
        ),
        upper=_typed_number(
            row["upper"],
            label=f"{label} upper",
            nullable=True,
        ),
        threshold=float(_typed_number(row["threshold"], label=f"{label} threshold")),
        rule=_typed_text(row["rule"], label=f"{label} rule"),
        confidence=float(_typed_number(row["confidence"], label=f"{label} confidence")),
        n_corpora=_typed_integer(
            row["n_corpora"],
            label=f"{label} n_corpora",
            minimum=1,
        ),
        n_families=_typed_integer(
            row["n_families"],
            label=f"{label} n_families",
            minimum=1,
        ),
        bootstrap_replicates=_typed_integer(
            row["bootstrap_replicates"],
            label=f"{label} bootstrap_replicates",
            minimum=0,
        ),
        bootstrap_seed=_typed_integer(
            row["bootstrap_seed"],
            label=f"{label} bootstrap_seed",
        ),
        passed=_typed_boolean(row["passed"], label=f"{label} passed"),
    )
    interval = (gate.estimate, gate.lower, gate.upper)
    if (
        not 0.0 < gate.confidence < 1.0
        or (
            any(value is None for value in interval)
            and any(value is not None for value in interval)
        )
        or (gate.lower is not None and gate.upper is not None and gate.lower > gate.upper)
    ):
        raise ConfirmatoryAnalysisError(f"{label} interval is invalid")
    return gate


def _require_registered_gate(
    gate: DirectionalGate,
    *,
    name: str,
    threshold: float,
    rule: str,
    confidence: float,
    n_corpora: int,
    n_families: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    direction: str,
    undefined: bool = False,
) -> None:
    if (
        gate.name != name
        or gate.threshold != threshold
        or gate.rule != rule
        or gate.confidence != confidence
        or gate.n_corpora != n_corpora
        or gate.n_families != n_families
        or gate.bootstrap_replicates != bootstrap_replicates
        or gate.bootstrap_seed != bootstrap_seed
    ):
        raise ConfirmatoryAnalysisError(f"{name} differs from the registered gate")
    if undefined:
        if (
            (gate.estimate, gate.lower, gate.upper) != (None, None, None)
            or gate.passed is not False
            or bootstrap_replicates != 0
        ):
            raise ConfirmatoryAnalysisError(f"{name} undefined gate is inconsistent")
        return
    if gate.estimate is None or gate.lower is None or gate.upper is None:
        raise ConfirmatoryAnalysisError(f"{name} lacks its registered interval")
    expected_passed = gate.lower > threshold if direction == "greater" else gate.upper < threshold
    if gate.passed is not expected_passed:
        raise ConfirmatoryAnalysisError(f"{name} interval and decision differ")


def _typed_h1(
    value: object,
    *,
    config: ConfirmatoryAnalysisConfig,
) -> H1Result:
    row = _closed_mapping(
        value,
        fields=set(H1Result.__dataclass_fields__),
        label="confirmatory result H1",
    )
    gate = _typed_directional_gate(row["gate"], label="confirmatory result H1 gate")
    _require_registered_gate(
        gate,
        name="h1_high_minus_low_predictive_risk",
        threshold=config.h1_minimum_risk_increase,
        rule="directional-lower-greater-than",
        confidence=config.confidence,
        n_corpora=len(config.fixed_corpora),
        n_families=len(config.fixed_corpora) * config.selected_families_per_corpus,
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_seed=config.bootstrap_seed + 11,
        direction="greater",
    )
    passed = _typed_boolean(row["passed"], label="confirmatory result H1 passed")
    if passed is not gate.passed:
        raise ConfirmatoryAnalysisError("confirmatory result H1 gate and decision differ")
    return H1Result(
        gate=gate,
        model_digest=_require_sha256("h1.model_digest", row["model_digest"]),
        passed=passed,
    )


def _typed_h2(
    value: object,
    *,
    config: ConfirmatoryAnalysisConfig,
) -> H2Result:
    row = _closed_mapping(
        value,
        fields=set(H2Result.__dataclass_fields__),
        label="confirmatory result H2",
    )
    gates = tuple(
        _typed_directional_gate(item, label="confirmatory result H2 metric gate")
        for item in _typed_array(
            row["metric_gates"],
            label="confirmatory result H2 metric_gates",
        )
    )
    expected_gate_contracts = (
        (
            "h2_log_loss_reduction",
            config.geometry_gain_thresholds.log_loss_reduction,
            config.bootstrap_seed + 21,
        ),
        (
            "h2_brier_score_reduction",
            config.geometry_gain_thresholds.brier_score_reduction,
            config.bootstrap_seed + 22,
        ),
        (
            "h2_auprc_gain",
            config.geometry_gain_thresholds.auprc_gain,
            config.bootstrap_seed + 23,
        ),
    )
    if len(gates) != len(expected_gate_contracts):
        raise ConfirmatoryAnalysisError("confirmatory result H2 metric gates differ")
    n_families = len(config.fixed_corpora) * config.selected_families_per_corpus
    for gate, (name, threshold, seed) in zip(
        gates,
        expected_gate_contracts,
        strict=True,
    ):
        undefined = name == "h2_auprc_gain" and gate.estimate is None
        _require_registered_gate(
            gate,
            name=name,
            threshold=threshold,
            rule=(
                "undefined-one-class-corpus_conservative-fail"
                if undefined
                else "directional-lower-greater-than"
            ),
            confidence=config.confidence,
            n_corpora=len(config.fixed_corpora),
            n_families=n_families,
            bootstrap_replicates=(0 if undefined else config.bootstrap_replicates),
            bootstrap_seed=seed,
            direction="greater",
            undefined=undefined,
        )
    corpus_results: list[CorpusGeometryResult] = []
    for item in _typed_array(
        row["corpus_results"],
        label="confirmatory result H2 corpus_results",
    ):
        result = _closed_mapping(
            item,
            fields=set(CorpusGeometryResult.__dataclass_fields__),
            label="confirmatory result H2 corpus result",
        )
        corpus_results.append(
            CorpusGeometryResult(
                corpus_id=_typed_text(
                    result["corpus_id"],
                    label="confirmatory result H2 corpus_id",
                ),
                log_loss_reduction=float(
                    _typed_number(
                        result["log_loss_reduction"],
                        label="confirmatory result H2 log_loss_reduction",
                    )
                ),
                brier_score_reduction=float(
                    _typed_number(
                        result["brier_score_reduction"],
                        label="confirmatory result H2 brier_score_reduction",
                    )
                ),
                auprc_gain=_typed_number(
                    result["auprc_gain"],
                    label="confirmatory result H2 auprc_gain",
                    nullable=True,
                ),
                passed=_typed_boolean(
                    result["passed"],
                    label="confirmatory result H2 corpus passed",
                ),
            )
        )
    passing = _typed_string_array(
        row["passing_corpora"],
        label="confirmatory result H2 passing_corpora",
    )
    minimum = _typed_integer(
        row["minimum_corpora"],
        label="confirmatory result H2 minimum_corpora",
        minimum=1,
    )
    passed = _typed_boolean(row["passed"], label="confirmatory result H2 passed")
    for result in corpus_results:
        expected_corpus_passed = (
            result.log_loss_reduction > config.geometry_gain_thresholds.log_loss_reduction
            and result.brier_score_reduction > config.geometry_gain_thresholds.brier_score_reduction
            and result.auprc_gain is not None
            and result.auprc_gain > config.geometry_gain_thresholds.auprc_gain
        )
        if result.passed is not expected_corpus_passed:
            raise ConfirmatoryAnalysisError(
                f"confirmatory result H2 {result.corpus_id} decision differs"
            )
    expected_passing = tuple(result.corpus_id for result in corpus_results if result.passed)
    expected_passed = len(expected_passing) >= minimum and all(gate.passed for gate in gates)
    if (
        not gates
        or tuple(result.corpus_id for result in corpus_results) != config.fixed_corpora
        or passing != expected_passing
        or minimum != config.minimum_corpora_with_geometry_gain
        or passed is not expected_passed
    ):
        raise ConfirmatoryAnalysisError("confirmatory result H2 decisions are inconsistent")
    return H2Result(
        metric_gates=gates,
        corpus_results=tuple(corpus_results),
        passing_corpora=passing,
        minimum_corpora=minimum,
        row_identity_digest=_require_sha256(
            "h2.row_identity_digest",
            row["row_identity_digest"],
        ),
        passed=passed,
    )


def _typed_h3(
    value: object,
    *,
    config: ConfirmatoryAnalysisConfig,
) -> H3Result:
    row = _closed_mapping(
        value,
        fields=set(H3Result.__dataclass_fields__),
        label="confirmatory result H3",
    )
    gates = tuple(
        _typed_directional_gate(item, label="confirmatory result H3 gate")
        for item in _typed_array(
            row["gates"],
            label="confirmatory result H3 gates",
        )
    )
    gate_contracts = (
        (
            "h3_family_latency_relative_reduction",
            config.minimum_cost_reduction,
            "directional-lower-greater-than",
            config.bootstrap_seed + 31,
            len(config.fixed_corpora),
            "greater",
        ),
        (
            "h3_retrieval_target_difference",
            -config.retrieval_target_noninferiority_margin,
            "directional-lower-greater-than-negative-margin",
            config.bootstrap_seed + 33,
            len(config.fixed_corpora),
            "greater",
        ),
        (
            "h3_evidence_sufficiency_difference",
            -config.evidence_sufficiency_noninferiority_margin,
            "directional-lower-greater-than-negative-margin",
            config.bootstrap_seed + 34,
            len(config.evidence_corpora),
            "greater",
        ),
        (
            "h3_p95_family_latency_ratio",
            config.maximum_p95_latency_ratio,
            "directional-upper-less-than",
            config.bootstrap_seed + 32,
            len(config.fixed_corpora),
            "less",
        ),
    )
    if len(gates) != len(gate_contracts):
        raise ConfirmatoryAnalysisError("confirmatory result H3 gates differ")
    for gate, (name, threshold, rule, seed, n_corpora, direction) in zip(
        gates,
        gate_contracts,
        strict=True,
    ):
        _require_registered_gate(
            gate,
            name=name,
            threshold=threshold,
            rule=rule,
            confidence=config.confidence,
            n_corpora=n_corpora,
            n_families=n_corpora * config.selected_families_per_corpus,
            bootstrap_replicates=config.bootstrap_replicates,
            bootstrap_seed=seed,
            direction=direction,
        )
    entitlement_row = _closed_mapping(
        row["entitlement"],
        fields=set(EntitlementResult.__dataclass_fields__),
        label="confirmatory result H3 entitlement",
    )
    entitlement = EntitlementResult(
        observed_events=_typed_integer(
            entitlement_row["observed_events"],
            label="confirmatory result H3 entitlement observed_events",
        ),
        families_with_events=_typed_integer(
            entitlement_row["families_with_events"],
            label="confirmatory result H3 entitlement families_with_events",
        ),
        n_families=_typed_integer(
            entitlement_row["n_families"],
            label="confirmatory result H3 entitlement n_families",
            minimum=1,
        ),
        exact_upper_bound=float(
            _typed_number(
                entitlement_row["exact_upper_bound"],
                label="confirmatory result H3 entitlement exact_upper_bound",
            )
        ),
        confidence=float(
            _typed_number(
                entitlement_row["confidence"],
                label="confirmatory result H3 entitlement confidence",
            )
        ),
        passed=_typed_boolean(
            entitlement_row["passed"],
            label="confirmatory result H3 entitlement passed",
        ),
    )
    if (
        entitlement.observed_events < entitlement.families_with_events
        or entitlement.families_with_events > entitlement.n_families
        or entitlement.n_families != len(config.fixed_corpora) * config.selected_families_per_corpus
        or not 0.0 <= entitlement.exact_upper_bound <= 1.0
        or entitlement.confidence != config.confidence
        or entitlement.passed
        is not (entitlement.observed_events <= config.maximum_entitlement_violations)
    ):
        raise ConfirmatoryAnalysisError("confirmatory result H3 entitlement is inconsistent")
    sensitivity_row = _closed_mapping(
        row["position_adjusted_sensitivity"],
        fields=set(PositionAdjustedSensitivityResult.__dataclass_fields__),
        label="confirmatory result H3 position sensitivity",
    )
    sensitivity = PositionAdjustedSensitivityResult(
        gate=_typed_directional_gate(
            sensitivity_row["gate"],
            label="confirmatory result H3 position sensitivity gate",
        ),
        position_trend_log_ratio_per_position=float(
            _typed_number(
                sensitivity_row["position_trend_log_ratio_per_position"],
                label="confirmatory result H3 position trend",
            )
        ),
        method=_typed_text(
            sensitivity_row["method"],
            label="confirmatory result H3 position sensitivity method",
        ),
        affects_primary_claim=_typed_boolean(
            sensitivity_row["affects_primary_claim"],
            label="confirmatory result H3 position sensitivity gating flag",
        ),
    )
    _require_registered_gate(
        sensitivity.gate,
        name="h3_position_adjusted_log_latency_ratio_sensitivity",
        threshold=math.log(1.0 - config.minimum_cost_reduction),
        rule=("sensitivity-directional-upper-less-than-log-one-minus-minimum-reduction"),
        confidence=config.confidence,
        n_corpora=len(config.fixed_corpora),
        n_families=len(config.fixed_corpora) * config.selected_families_per_corpus,
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_seed=config.bootstrap_seed + 35,
        direction="less",
    )
    counts = row["execution_state_counts"]
    if (
        not isinstance(counts, Mapping)
        or set(counts) != {"abstained", "completed", "failed"}
        or any(type(key) is not str for key in counts)
    ):
        raise ConfirmatoryAnalysisError("confirmatory result H3 execution_state_counts differ")
    execution_state_counts = tuple(
        (
            state,
            _typed_integer(
                counts[state],
                label=f"confirmatory result H3 {state} count",
            ),
        )
        for state in ("completed", "failed", "abstained")
    )
    passed = _typed_boolean(row["passed"], label="confirmatory result H3 passed")
    if len(gates) != 4 or passed is not (all(gate.passed for gate in gates) and entitlement.passed):
        raise ConfirmatoryAnalysisError("confirmatory result H3 decisions are inconsistent")
    return H3Result(
        gates=gates,
        entitlement=entitlement,
        position_adjusted_sensitivity=sensitivity,
        execution_state_counts=execution_state_counts,
        passed=passed,
    )


def _typed_confirmatory_result(
    payload: object,
) -> ConfirmatoryResultArtifact:
    fields = {
        "confirmatory_input_artifact_sha256",
        "corpus_inputs",
        "frozen_config",
        "frozen_config_sha256",
        "h1",
        "h2",
        "h3",
        "input_row_count",
        "input_rows_sha256",
        "manifest_sha256",
        "model_suite_sha256",
        "primary_claim_passed",
        "run_receipt_sha256",
        "schema_version",
        "trial_count",
    }
    row = _closed_mapping(
        payload,
        fields=fields,
        label="confirmatory result artifact",
    )
    config = _typed_analysis_config(row["frozen_config"])
    if (
        _require_sha256(
            "frozen_config_sha256",
            row["frozen_config_sha256"],
        )
        != config.config_sha256
    ):
        raise ConfirmatoryAnalysisError("confirmatory result frozen_config digest differs")
    corpus_inputs: list[CorpusInputDigests] = []
    for item in _typed_array(
        row["corpus_inputs"],
        label="confirmatory result corpus_inputs",
    ):
        binding = _closed_mapping(
            item,
            fields=set(CorpusInputDigests.__dataclass_fields__),
            label="confirmatory result corpus input",
        )
        try:
            corpus_inputs.append(CorpusInputDigests(**binding))
        except (TypeError, ValueError) as exc:
            raise ConfirmatoryAnalysisError(
                f"confirmatory result corpus input is invalid: {exc}"
            ) from exc
    h1 = _typed_h1(row["h1"], config=config)
    h2 = _typed_h2(row["h2"], config=config)
    h3 = _typed_h3(row["h3"], config=config)
    artifact = ConfirmatoryResultArtifact(
        manifest_sha256=row["manifest_sha256"],
        run_receipt_sha256=row["run_receipt_sha256"],
        confirmatory_input_artifact_sha256=(row["confirmatory_input_artifact_sha256"]),
        corpus_input_digests=tuple(corpus_inputs),
        frozen_config=config,
        model_suite_sha256=row["model_suite_sha256"],
        input_rows_sha256=row["input_rows_sha256"],
        input_row_count=_typed_integer(
            row["input_row_count"],
            label="confirmatory result input_row_count",
            minimum=1,
        ),
        trial_count=_typed_integer(
            row["trial_count"],
            label="confirmatory result trial_count",
            minimum=1,
        ),
        h1=h1,
        h2=h2,
        h3=h3,
        primary_claim_passed=_typed_boolean(
            row["primary_claim_passed"],
            label="confirmatory result primary_claim_passed",
        ),
        schema_version=row["schema_version"],
    )
    if (
        tuple(item.corpus_id for item in artifact.corpus_input_digests) != config.fixed_corpora
        or artifact.input_row_count != artifact.trial_count * len(config.action_set)
        or sum(count for _, count in artifact.h3.execution_state_counts) != artifact.input_row_count
    ):
        raise ConfirmatoryAnalysisError("confirmatory result typed row and corpus closure differs")
    return artifact


def load_confirmatory_result_artifact(
    path: str | Path,
    *,
    result_receipt_path: str | Path,
    attempt_receipt_path: str | Path,
) -> ConfirmatoryResultArtifact:
    """Load and reconstruct the typed result behind the detached receipt chain."""

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
        encoded = read_secure_control_file(
            target,
            label="confirmatory result artifact",
        )
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryAnalysisError(f"cannot load confirmatory result artifact: {exc}") from exc
    payload = _decode_canonical_object(encoded, label="confirmatory result artifact")
    canonical_payload = _canonical_bytes(payload)
    if encoded != canonical_payload + b"\n":
        raise ConfirmatoryAnalysisError("confirmatory result artifact bytes are not canonical")
    if hashlib.sha256(canonical_payload).hexdigest() != receipt.result_artifact_sha256:
        raise ConfirmatoryAnalysisError(
            "confirmatory result artifact does not match its detached receipt"
        )
    expected_result_bindings = {
        "manifest_sha256": attempt.manifest_sha256,
        "run_receipt_sha256": attempt.run_receipt_sha256,
        "confirmatory_input_artifact_sha256": (attempt.confirmatory_input_artifact_sha256),
        "model_suite_sha256": attempt.model_suite_sha256,
    }
    for name, expected in expected_result_bindings.items():
        if payload.get(name) != expected:
            raise ConfirmatoryAnalysisError(
                f"confirmatory result {name} does not match the admitted attempt"
            )
    result = _typed_confirmatory_result(payload)
    if result.canonical_bytes() != canonical_payload:
        raise ConfirmatoryAnalysisError(
            "confirmatory result typed reconstruction changed canonical bytes"
        )
    return result


def load_confirmatory_result_artifact_bytes(
    path: str | Path,
    *,
    result_receipt_path: str | Path,
    attempt_receipt_path: str | Path,
) -> bytes:
    """Return canonical bytes only after typed result reconstruction succeeds."""

    return load_confirmatory_result_artifact(
        path,
        result_receipt_path=result_receipt_path,
        attempt_receipt_path=attempt_receipt_path,
    ).canonical_bytes()


def _attempt_receipt(
    inputs: ConfirmatoryInputArtifact,
    *,
    suite: FrozenModelSuite,
) -> ConfirmatoryAnalysisAttemptReceipt:
    return ConfirmatoryAnalysisAttemptReceipt(
        manifest_sha256=_manifest_digest(inputs),
        run_receipt_sha256=_require_sha256("run_receipt_sha256", inputs.run_receipt_sha256),
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
        raise ConfirmatoryAnalysisError("analysis runner must return a ConfirmatoryResultArtifact")
    expected = {
        "manifest_sha256": attempt.manifest_sha256,
        "run_receipt_sha256": attempt.run_receipt_sha256,
        "confirmatory_input_artifact_sha256": (attempt.confirmatory_input_artifact_sha256),
        "model_suite_sha256": attempt.model_suite_sha256,
    }
    for name, value in expected.items():
        if getattr(result, name) != value:
            raise ConfirmatoryAnalysisError(
                f"confirmatory result {name} does not match the admitted attempt"
            )


def _require_suite_labels_released(
    token: object,
    *,
    manifest_digest: str,
    run_receipt_sha256: str,
) -> None:
    """Admit analysis only from the file-backed all-five release capability."""

    try:
        from .suite_attempt import SuiteAttemptError, require_verified_labels_released
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise ConfirmatoryAnalysisError("suite attempt verifier is unavailable") from exc
    try:
        require_verified_labels_released(
            token,
            manifest_digest=manifest_digest,
            run_receipt_sha256=run_receipt_sha256,
        )
    except SuiteAttemptError as exc:
        raise ConfirmatoryAnalysisError(f"suite label-release gate failed: {exc}") from exc


def run_confirmatory_analysis_once(
    inputs: ConfirmatoryInputArtifact,
    *,
    suite: FrozenModelSuite,
    verified_labels_released: object | None = None,
) -> ConfirmatoryResultArtifact:
    """After all-five release attestation, admit and persist one analysis."""

    if not isinstance(inputs, ConfirmatoryInputArtifact):
        raise ConfirmatoryAnalysisError("inputs must be a ConfirmatoryInputArtifact")
    if not isinstance(suite, FrozenModelSuite):
        raise ConfirmatoryAnalysisError("suite must be a FrozenModelSuite")

    _require_suite_labels_released(
        verified_labels_released,
        manifest_digest=_manifest_digest(inputs),
        run_receipt_sha256=_require_sha256(
            "run_receipt_sha256",
            inputs.run_receipt_sha256,
        ),
    )

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
            f"confirmatory analysis attempt was not admitted exclusively at {attempt_target}: {exc}"
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
        raise ConfirmatoryAnalysisError(f"confirmatory result custody write failed: {exc}") from exc
    return result
