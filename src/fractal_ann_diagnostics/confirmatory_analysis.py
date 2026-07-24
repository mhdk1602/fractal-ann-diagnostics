"""One-shot confirmatory analysis over a closed paired-action panel.

The runner owns validation of the complete analysis matrix.  No outcome row is
silently discarded: every registered trial must have exactly one row for every
registered action, and a non-completed action must carry an explicit state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import beta as beta_distribution

from .artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactVerificationReceipt,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .audit import GENESIS_RECORD_SHA256
from .confirmatory_modeling import (
    FeatureBatch,
    FrozenModelSuite,
    GeometryGainThresholds,
    LabeledFeatureBatch,
    canonical_h1_model_artifact_bytes,
    canonical_h2_model_suite_artifact_bytes,
    evaluate_geometry_gain_gate,
    evaluate_h2_by_corpus,
    predictive_geometry_risk_contrast,
)
from .confirmatory_stats import (
    ClusterBootstrapConfig,
    ClusterBootstrapResult,
    ConfidenceInterval,
    noninferiority_decision,
    paired_stratified_family_bootstrap,
    paired_stratified_metric_bootstrap,
    superiority_decision,
    upper_limit_decision,
)
from .label_separation import (
    ActionPanelBinding,
    OfflineEvaluationArtifact,
    PredictionCompletionReceipt,
    SealedLabelArtifact,
    sealed_run_receipt_sha256,
)
from .study import (
    SealedRunReceipt,
    StudyManifestError,
    manifest_sha256,
    revision_sha256,
    sealed_receipt_uri,
    validate_study_manifest,
)

ACTION_PANEL_ROW_SCHEMA = "fractal-prelabel-action-row-v3"
ACTION_PANEL_ARTIFACT_SCHEMA = "fractal-prelabel-action-panel-v3"
ACTION_PANEL_ADMISSION_RECORD_SCHEMA = "fractal-action-panel-admission-record-v2"
ACTION_PANEL_ADMISSION_RECEIPT_SCHEMA = "fractal-action-panel-admission-receipt-v2"
ACTION_PANEL_FACTORY_VERSION = "fractal-action-panel-factory-v1"
CONFIRMATORY_INPUT_SCHEMA = "fractal-confirmatory-input-v6"
CONFIRMATORY_ROW_SCHEMA = "fractal-confirmatory-row-v2"
CONFIRMATORY_RESULT_SCHEMA = "fractal-confirmatory-result-v5"
ExecutionState = Literal["completed", "failed", "abstained"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ACTION_PANEL_BYTES = 256 * 1024 * 1024
_PRELABEL_ACTION_ROW_FIELDS = {
    "action",
    "action_order",
    "audit_record_sha256",
    "controller_selected",
    "entitlement_violations",
    "execution_position",
    "execution_state",
    "failure_state",
    "family_key",
    "feature_values",
    "request_latency_ms",
    "returned_document_ids",
    "schema_version",
    "trial_key",
}
_ACTION_PANEL_ARTIFACT_FIELDS = {
    "action_set",
    "corpus",
    "document_count",
    "execution_artifact_sha256",
    "manifest_sha256",
    "rows",
    "run_receipt_sha256",
    "schema_version",
    "stage",
}
_ACTION_PANEL_ADMISSION_RECORD_FIELDS = {
    "action",
    "action_order",
    "audit_previous_record_sha256",
    "audit_record_sha256",
    "audit_sequence",
    "authorization_decision_id",
    "authorization_decision_sha256",
    "authorization_mask_sha256",
    "authorization_mask_size",
    "authorization_request_sha256",
    "controller_decision_sha256",
    "controller_policy_version",
    "controller_reasons",
    "controller_risk_score",
    "controller_selected",
    "document_universe_sha256",
    "environment_sha256",
    "execution_position",
    "execution_state",
    "failure_code",
    "failure_finished_monotonic_ns",
    "failure_runner_identity",
    "failure_started_monotonic_ns",
    "failure_timing_receipt_sha256",
    "family_key",
    "policy_available",
    "schema_version",
    "trial_key",
}
_ACTION_PANEL_ADMISSION_RECEIPT_FIELDS = {
    "action_panel_artifact_sha256",
    "audit_chain_length",
    "audit_head_sha256",
    "audit_record_sha256s",
    "corpus",
    "execution_artifact_sha256",
    "factory_version",
    "manifest_sha256",
    "partition_label",
    "query_partition_audit_sha256",
    "records",
    "run_receipt_sha256",
    "schema_version",
}
_ROW_FIELDS = {
    "action",
    "action_order",
    "controller_selected",
    "corpus_id",
    "entitlement_violations",
    "execution_position",
    "evidence_sufficient",
    "execution_state",
    "failure_state",
    "family_id",
    "feature_values",
    "recall_at_k",
    "request_latency_ms",
    "schema_version",
    "trial_id",
}


class ConfirmatoryAnalysisError(ValueError):
    """Raised when a frozen input or analysis invariant is violated."""


def _canonical_bytes(payload: object) -> bytes:
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
            "analysis artifacts must be canonical finite JSON values"
        ) from exc


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _closed_mapping(
    payload: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ConfirmatoryAnalysisError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in payload):
        raise ConfirmatoryAnalysisError(f"{label} keys must be strings")
    observed = set(payload)
    missing = fields - observed
    unexpected = observed - fields
    if missing or unexpected:
        raise ConfirmatoryAnalysisError(
            f"{label} keys do not match the closed schema; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return payload


def _decode_canonical_json_object(
    payload: str | bytes,
    *,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    if type(payload) not in {str, bytes}:
        raise TypeError("payload must be str or bytes")
    if isinstance(payload, bytes):
        supplied_bytes = payload
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ConfirmatoryAnalysisError(f"{label} must be valid UTF-8") from exc
    else:
        text = payload
        try:
            supplied_bytes = payload.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ConfirmatoryAnalysisError(f"{label} must be valid UTF-8") from exc
    if len(supplied_bytes) > _MAX_ACTION_PANEL_BYTES:
        raise ConfirmatoryAnalysisError(f"{label} exceeds the {_MAX_ACTION_PANEL_BYTES}-byte limit")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfirmatoryAnalysisError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ConfirmatoryAnalysisError(f"{label} contains non-finite number {value!r}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ConfirmatoryAnalysisError(f"{label} must be valid JSON: {exc.msg}") from exc
    except RecursionError as exc:
        raise ConfirmatoryAnalysisError(f"{label} exceeds the JSON nesting limit") from exc
    if not isinstance(decoded, Mapping):
        raise ConfirmatoryAnalysisError(f"{label} must contain one JSON object")
    return decoded, supplied_bytes


def _read_action_panel_file(path: str | Path, *, label: str) -> bytes:
    try:
        return read_secure_regular_file(
            Path(path),
            max_bytes=_MAX_ACTION_PANEL_BYTES,
            label=label,
        )
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryAnalysisError(f"cannot safely load {label}: {exc}") from exc


def _write_action_panel_file(
    payload: bytes,
    target: str | Path,
    *,
    label: str,
) -> None:
    try:
        write_exclusive_receipt_bytes(payload, Path(target))
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryAnalysisError(f"cannot safely write {label}: {exc}") from exc


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ConfirmatoryAnalysisError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_identifier(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ConfirmatoryAnalysisError(f"{name} must be a canonical non-empty string")
    return value


def _finite_number(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ConfirmatoryAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfirmatoryAnalysisError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ConfirmatoryAnalysisError(f"{name} must be at least {minimum}")
    return result


def _json_feature(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise ConfirmatoryAnalysisError("feature values cannot contain infinity")
        return number
    if isinstance(value, str):
        return value
    raise ConfirmatoryAnalysisError(
        "feature values must be strings, finite numbers, or missing numeric values"
    )


def _assert_balanced_execution_positions(
    rows: Sequence[object],
    *,
    action_set: Sequence[str],
    trial_attribute: str,
    family_attribute: str,
    corpus_attribute: str | None = None,
) -> None:
    """Validate complete Latin rows and floor/ceiling position balance."""

    actions = tuple(action_set)
    if len(actions) != 4 or len(set(actions)) != 4:
        raise ConfirmatoryAnalysisError("position balance requires the registered four actions")
    by_trial: dict[tuple[str, str], list[object]] = {}
    balance_groups: dict[tuple[str, ...], list[object]] = {}
    for row in rows:
        trial = getattr(row, trial_attribute)
        family = getattr(row, family_attribute)
        corpus = "single-corpus" if corpus_attribute is None else getattr(row, corpus_attribute)
        by_trial.setdefault((corpus, trial), []).append(row)
        balance_groups.setdefault(("corpus", corpus), []).append(row)
        balance_groups.setdefault(("query-family", corpus, family), []).append(row)
    expected_positions = set(range(len(actions)))
    for key, trial_rows in by_trial.items():
        positions = [getattr(row, "execution_position") for row in trial_rows]
        if len(positions) != len(actions) or set(positions) != expected_positions:
            raise ConfirmatoryAnalysisError(
                f"trial {key!r} has missing, duplicate, or impossible execution positions"
            )
    for key, group_rows in balance_groups.items():
        trials = {getattr(row, trial_attribute) for row in group_rows}
        for action in actions:
            counts = [
                sum(
                    getattr(row, "action") == action
                    and getattr(row, "execution_position") == position
                    for row in group_rows
                )
                for position in range(len(actions))
            ]
            if sum(counts) != len(trials) or max(counts) - min(counts) > 1:
                raise ConfirmatoryAnalysisError(
                    f"action execution positions are not floor/ceiling balanced in {key!r}"
                )


@dataclass(frozen=True)
class PreLabelActionRow:
    """One label-free online action outcome anchored before label release."""

    trial_key: str
    family_key: str
    action: str
    action_order: int
    execution_position: int
    audit_record_sha256: str | None
    execution_state: ExecutionState
    failure_state: str | None
    controller_selected: bool
    request_latency_ms: float
    entitlement_violations: int
    returned_document_ids: tuple[int, ...]
    feature_values: tuple[object, ...] | None
    schema_version: str = ACTION_PANEL_ROW_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("trial_key", self.trial_key)
        _require_sha256("family_key", self.family_key)
        _require_identifier("action", self.action)
        if self.schema_version != ACTION_PANEL_ROW_SCHEMA:
            raise ConfirmatoryAnalysisError(
                f"schema_version must equal {ACTION_PANEL_ROW_SCHEMA!r}"
            )
        if (
            isinstance(self.action_order, bool)
            or not isinstance(self.action_order, (int, np.integer))
            or int(self.action_order) < 0
        ):
            raise ConfirmatoryAnalysisError("action_order must be a non-negative integer")
        object.__setattr__(self, "action_order", int(self.action_order))
        if (
            isinstance(self.execution_position, bool)
            or not isinstance(self.execution_position, (int, np.integer))
            or not 0 <= int(self.execution_position) < 4
        ):
            raise ConfirmatoryAnalysisError(
                "execution_position must be an integer from zero through three"
            )
        object.__setattr__(self, "execution_position", int(self.execution_position))
        if self.audit_record_sha256 is not None:
            _require_sha256("audit_record_sha256", self.audit_record_sha256)
        if not isinstance(self.execution_state, str) or self.execution_state not in {
            "completed",
            "failed",
            "abstained",
        }:
            raise ConfirmatoryAnalysisError("execution_state is not registered")
        if type(self.controller_selected) is not bool:
            raise ConfirmatoryAnalysisError("controller_selected must be boolean")
        latency = _finite_number("request_latency_ms", self.request_latency_ms, minimum=0.0)
        if latency == 0.0:
            raise ConfirmatoryAnalysisError("request_latency_ms must be positive")
        object.__setattr__(self, "request_latency_ms", latency)
        if (
            isinstance(self.entitlement_violations, bool)
            or not isinstance(self.entitlement_violations, (int, np.integer))
            or int(self.entitlement_violations) < 0
        ):
            raise ConfirmatoryAnalysisError("entitlement_violations must be a non-negative integer")
        object.__setattr__(self, "entitlement_violations", int(self.entitlement_violations))
        returned = tuple(self.returned_document_ids)
        if any(type(value) is not int or value < 0 for value in returned):
            raise ConfirmatoryAnalysisError(
                "returned_document_ids must contain non-negative integers"
            )
        if len(returned) != len(set(returned)):
            raise ConfirmatoryAnalysisError("returned_document_ids cannot contain duplicates")
        object.__setattr__(self, "returned_document_ids", returned)

        if self.execution_state == "completed":
            if self.failure_state is not None:
                raise ConfirmatoryAnalysisError("completed rows cannot carry a failure_state")
            if self.audit_record_sha256 is None:
                raise ConfirmatoryAnalysisError("completed rows require an audit_record_sha256")
        else:
            if self.failure_state is None:
                raise ConfirmatoryAnalysisError(
                    "failed and abstained rows need an explicit failure_state"
                )
            _require_identifier("failure_state", self.failure_state)
            if returned:
                raise ConfirmatoryAnalysisError(
                    "failed and abstained rows cannot emit document IDs"
                )
            if self.execution_state == "abstained" and self.audit_record_sha256 is None:
                raise ConfirmatoryAnalysisError("abstained rows require an audit_record_sha256")
            if self.execution_state == "failed" and self.audit_record_sha256 is not None:
                raise ConfirmatoryAnalysisError("failed rows cannot claim a governed audit record")
        if self.feature_values is not None:
            values = tuple(np.nan if value is None else value for value in self.feature_values)
            for value in values:
                _json_feature(value)
            object.__setattr__(self, "feature_values", values)

    @classmethod
    def from_dict(cls, payload: object) -> PreLabelActionRow:
        """Restore one action row from its exact closed JSON object schema."""

        row = _closed_mapping(
            payload,
            fields=_PRELABEL_ACTION_ROW_FIELDS,
            label="pre-label action row",
        )
        returned = row["returned_document_ids"]
        if not isinstance(returned, list):
            raise ConfirmatoryAnalysisError(
                "pre-label action row returned_document_ids must be an array"
            )
        raw_features = row["feature_values"]
        if raw_features is not None and not isinstance(raw_features, list):
            raise ConfirmatoryAnalysisError(
                "pre-label action row feature_values must be an array or null"
            )
        if raw_features is not None and any(
            isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
            for value in raw_features
        ):
            raise ConfirmatoryAnalysisError(
                "pre-label action row feature_values cannot contain non-finite numbers"
            )
        features = (
            None
            if raw_features is None
            else tuple(np.nan if value is None else value for value in raw_features)
        )
        return cls(
            trial_key=row["trial_key"],
            family_key=row["family_key"],
            action=row["action"],
            action_order=row["action_order"],
            execution_position=row["execution_position"],
            audit_record_sha256=row["audit_record_sha256"],
            execution_state=row["execution_state"],
            failure_state=row["failure_state"],
            controller_selected=row["controller_selected"],
            request_latency_ms=row["request_latency_ms"],
            entitlement_violations=row["entitlement_violations"],
            returned_document_ids=tuple(returned),
            feature_values=features,
            schema_version=row["schema_version"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "action_order": self.action_order,
            "audit_record_sha256": self.audit_record_sha256,
            "controller_selected": self.controller_selected,
            "entitlement_violations": self.entitlement_violations,
            "execution_position": self.execution_position,
            "execution_state": self.execution_state,
            "failure_state": self.failure_state,
            "family_key": self.family_key,
            "feature_values": (
                None
                if self.feature_values is None
                else [_json_feature(value) for value in self.feature_values]
            ),
            "request_latency_ms": self.request_latency_ms,
            "returned_document_ids": list(self.returned_document_ids),
            "schema_version": self.schema_version,
            "trial_key": self.trial_key,
        }

    def canonical_bytes(self) -> bytes:
        """Return canonical JSON bytes without the file-format newline."""

        return _canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class ActionPanelArtifact:
    """Canonical all-action online record anchored before labels are released."""

    manifest_sha256: str
    run_receipt_sha256: str
    execution_artifact_sha256: str
    corpus: str
    stage: str
    document_count: int
    action_set: tuple[str, ...]
    rows: tuple[PreLabelActionRow, ...]
    schema_version: str = ACTION_PANEL_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_identifier("corpus", self.corpus)
        if self.stage != "sealed":
            raise ConfirmatoryAnalysisError("action panel stage must equal 'sealed'")
        if self.schema_version != ACTION_PANEL_ARTIFACT_SCHEMA:
            raise ConfirmatoryAnalysisError(
                f"schema_version must equal {ACTION_PANEL_ARTIFACT_SCHEMA!r}"
            )
        if (
            isinstance(self.document_count, bool)
            or not isinstance(self.document_count, int)
            or self.document_count <= 0
        ):
            raise ConfirmatoryAnalysisError("document_count must be a positive integer")
        actions = tuple(self.action_set)
        if not actions or len(actions) != len(set(actions)):
            raise ConfirmatoryAnalysisError("action_set must be non-empty and unique")
        for action in actions:
            _require_identifier("action_set", action)
        object.__setattr__(self, "action_set", actions)
        rows = tuple(self.rows)
        if not rows or not all(isinstance(row, PreLabelActionRow) for row in rows):
            raise ConfirmatoryAnalysisError(
                "action panel rows must contain PreLabelActionRow values"
            )
        keys = [(row.trial_key, row.action) for row in rows]
        if len(keys) != len(set(keys)):
            raise ConfirmatoryAnalysisError("action panel contains duplicate trial-action rows")
        by_trial: dict[str, list[PreLabelActionRow]] = {}
        for row in rows:
            if row.action not in actions:
                raise ConfirmatoryAnalysisError("action panel contains an unregistered action")
            if row.action_order != actions.index(row.action):
                raise ConfirmatoryAnalysisError(
                    "action_order does not match the panel action-set order"
                )
            if any(document_id >= self.document_count for document_id in row.returned_document_ids):
                raise ConfirmatoryAnalysisError(
                    "action panel row names a document outside document_count"
                )
            if row.action == "hnsw-low":
                if row.feature_values is None:
                    raise ConfirmatoryAnalysisError(
                        "each hnsw-low row needs a frozen feature vector"
                    )
            elif row.feature_values is not None:
                raise ConfirmatoryAnalysisError(
                    "only hnsw-low rows may carry predictive feature values"
                )
            if row.action == "abstain":
                if row.execution_state != "abstained":
                    raise ConfirmatoryAnalysisError(
                        "the registered abstain action must have abstained state"
                    )
            elif row.execution_state == "abstained":
                raise ConfirmatoryAnalysisError(
                    "only the registered abstain action may have abstained state"
                )
            by_trial.setdefault(row.trial_key, []).append(row)
        for trial_key, trial_rows in by_trial.items():
            if {row.action for row in trial_rows} != set(actions):
                raise ConfirmatoryAnalysisError(
                    f"trial {trial_key!r} does not contain the complete action set"
                )
            if len({row.family_key for row in trial_rows}) != 1:
                raise ConfirmatoryAnalysisError("one trial cannot be rebound across query families")
            if sum(row.controller_selected for row in trial_rows) != 1:
                raise ConfirmatoryAnalysisError(
                    "each trial needs exactly one controller-selected action"
                )
            exact = [row for row in trial_rows if row.action == "exact-authorized"]
            if len(exact) != 1 or exact[0].execution_state != "completed":
                raise ConfirmatoryAnalysisError(
                    "each trial needs one completed exact-authorized oracle action"
                )
        _assert_balanced_execution_positions(
            rows,
            action_set=actions,
            trial_attribute="trial_key",
            family_attribute="family_key",
        )
        ordered = tuple(
            sorted(
                rows,
                key=lambda row: (row.family_key, row.trial_key, row.action_order),
            )
        )
        object.__setattr__(self, "rows", ordered)

    @classmethod
    def from_dict(cls, payload: object) -> ActionPanelArtifact:
        """Restore one complete panel from its exact closed JSON object schema."""

        artifact = _closed_mapping(
            payload,
            fields=_ACTION_PANEL_ARTIFACT_FIELDS,
            label="action panel artifact",
        )
        action_set = artifact["action_set"]
        if not isinstance(action_set, list):
            raise ConfirmatoryAnalysisError("action panel artifact action_set must be an array")
        if not all(isinstance(action, str) for action in action_set):
            raise ConfirmatoryAnalysisError("action panel artifact action_set must contain strings")
        rows = artifact["rows"]
        if not isinstance(rows, list):
            raise ConfirmatoryAnalysisError("action panel artifact rows must be an array")
        return cls(
            manifest_sha256=artifact["manifest_sha256"],
            run_receipt_sha256=artifact["run_receipt_sha256"],
            execution_artifact_sha256=artifact["execution_artifact_sha256"],
            corpus=artifact["corpus"],
            stage=artifact["stage"],
            document_count=artifact["document_count"],
            action_set=tuple(action_set),
            rows=tuple(PreLabelActionRow.from_dict(row) for row in rows),
            schema_version=artifact["schema_version"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action_set": list(self.action_set),
            "corpus": self.corpus,
            "document_count": self.document_count,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "rows": [row.to_dict() for row in self.rows],
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
            "stage": self.stage,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def completion_binding(self) -> ActionPanelBinding:
        """Return the closed panel identity admitted before label release."""

        return ActionPanelBinding(
            manifest_sha256=self.manifest_sha256,
            run_receipt_sha256=self.run_receipt_sha256,
            execution_artifact_sha256=self.execution_artifact_sha256,
            corpus=self.corpus,
            stage=self.stage,
            action_panel_artifact_sha256=self.artifact_sha256,
        )


@dataclass(frozen=True)
class ActionPanelAdmissionRecord:
    """One factory-admitted action with decision, policy, audit, and timing evidence."""

    trial_key: str
    family_key: str
    action: str
    action_order: int
    execution_position: int
    controller_selected: bool
    execution_state: ExecutionState
    controller_risk_score: float
    controller_reasons: tuple[str, ...]
    controller_policy_version: str
    controller_decision_sha256: str
    authorization_decision_id: str
    authorization_request_sha256: str
    authorization_mask_sha256: str
    authorization_mask_size: int
    authorization_decision_sha256: str
    policy_available: bool
    environment_sha256: str
    document_universe_sha256: str
    audit_sequence: int | None
    audit_previous_record_sha256: str | None
    audit_record_sha256: str | None
    failure_code: str | None
    failure_started_monotonic_ns: int | None
    failure_finished_monotonic_ns: int | None
    failure_runner_identity: str | None
    failure_timing_receipt_sha256: str | None
    schema_version: str = ACTION_PANEL_ADMISSION_RECORD_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("trial_key", self.trial_key)
        _require_sha256("family_key", self.family_key)
        _require_identifier("action", self.action)
        if self.schema_version != ACTION_PANEL_ADMISSION_RECORD_SCHEMA:
            raise ConfirmatoryAnalysisError(
                f"schema_version must equal {ACTION_PANEL_ADMISSION_RECORD_SCHEMA!r}"
            )
        if (
            isinstance(self.action_order, bool)
            or not isinstance(self.action_order, int)
            or self.action_order < 0
        ):
            raise ConfirmatoryAnalysisError("action_order must be a non-negative integer")
        if (
            isinstance(self.execution_position, bool)
            or not isinstance(self.execution_position, int)
            or not 0 <= self.execution_position < 4
        ):
            raise ConfirmatoryAnalysisError(
                "execution_position must be an integer from zero through three"
            )
        if type(self.controller_selected) is not bool:
            raise ConfirmatoryAnalysisError("controller_selected must be boolean")
        if self.execution_state not in {"completed", "failed", "abstained"}:
            raise ConfirmatoryAnalysisError("execution_state is not registered")

        risk_score = _finite_number("controller_risk_score", self.controller_risk_score)
        object.__setattr__(self, "controller_risk_score", risk_score)
        reasons = tuple(self.controller_reasons)
        if not reasons:
            raise ConfirmatoryAnalysisError("controller_reasons must not be empty")
        for reason in reasons:
            _require_identifier("controller reason", reason)
        object.__setattr__(self, "controller_reasons", reasons)
        _require_identifier("controller_policy_version", self.controller_policy_version)
        _require_sha256("controller_decision_sha256", self.controller_decision_sha256)
        expected_controller_digest = _sha256(
            {
                "action": self.action,
                "policy_version": self.controller_policy_version,
                "reasons": list(reasons),
                "risk_score": risk_score,
            }
        )
        if self.controller_decision_sha256 != expected_controller_digest:
            raise ConfirmatoryAnalysisError(
                "controller_decision_sha256 does not bind the controller decision"
            )

        _require_identifier("authorization_decision_id", self.authorization_decision_id)
        for name in (
            "authorization_request_sha256",
            "authorization_mask_sha256",
            "authorization_decision_sha256",
            "environment_sha256",
            "document_universe_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            isinstance(self.authorization_mask_size, bool)
            or not isinstance(self.authorization_mask_size, int)
            or self.authorization_mask_size <= 0
        ):
            raise ConfirmatoryAnalysisError("authorization_mask_size must be a positive integer")
        if type(self.policy_available) is not bool:
            raise ConfirmatoryAnalysisError("policy_available must be boolean")
        expected_authorization_digest = _sha256(
            {
                "available": self.policy_available,
                "decision_id": self.authorization_decision_id,
                "document_universe_sha256": self.document_universe_sha256,
                "environment_sha256": self.environment_sha256,
                "mask_sha256": self.authorization_mask_sha256,
                "mask_size": self.authorization_mask_size,
                "policy_version": self.controller_policy_version,
                "request_sha256": self.authorization_request_sha256,
            }
        )
        if self.authorization_decision_sha256 != expected_authorization_digest:
            raise ConfirmatoryAnalysisError(
                "authorization_decision_sha256 does not bind the policy decision"
            )

        if self.execution_state == "failed":
            if any(
                value is not None
                for value in (
                    self.audit_sequence,
                    self.audit_previous_record_sha256,
                    self.audit_record_sha256,
                )
            ):
                raise ConfirmatoryAnalysisError(
                    "failed admissions cannot claim governed audit records"
                )
            _require_identifier("failure_code", self.failure_code)
            _require_identifier(
                "failure_runner_identity",
                self.failure_runner_identity,
            )
            for name, value in (
                ("failure_started_monotonic_ns", self.failure_started_monotonic_ns),
                ("failure_finished_monotonic_ns", self.failure_finished_monotonic_ns),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ConfirmatoryAnalysisError(f"{name} must be a non-negative integer")
            if self.failure_finished_monotonic_ns <= self.failure_started_monotonic_ns:
                raise ConfirmatoryAnalysisError(
                    "failure timing must have positive monotonic duration"
                )
            _require_sha256(
                "failure_timing_receipt_sha256",
                self.failure_timing_receipt_sha256,
            )
            if self.failure_timing_receipt_sha256 != self.computed_failure_timing_sha256():
                raise ConfirmatoryAnalysisError(
                    "failure timing receipt digest does not bind the timing window"
                )
        else:
            if (
                isinstance(self.audit_sequence, bool)
                or not isinstance(self.audit_sequence, int)
                or self.audit_sequence < 0
            ):
                raise ConfirmatoryAnalysisError(
                    "governed admissions need a non-negative audit sequence"
                )
            _require_sha256(
                "audit_previous_record_sha256",
                self.audit_previous_record_sha256,
            )
            _require_sha256("audit_record_sha256", self.audit_record_sha256)
            if any(
                value is not None
                for value in (
                    self.failure_code,
                    self.failure_started_monotonic_ns,
                    self.failure_finished_monotonic_ns,
                    self.failure_runner_identity,
                    self.failure_timing_receipt_sha256,
                )
            ):
                raise ConfirmatoryAnalysisError(
                    "governed admissions cannot claim a runner failure receipt"
                )

    def computed_failure_timing_sha256(self) -> str:
        if self.execution_state != "failed":
            raise ConfirmatoryAnalysisError("only failed admissions have a failure timing digest")
        return _sha256(
            {
                "action": self.action,
                "authorization_decision_sha256": self.authorization_decision_sha256,
                "controller_decision_sha256": self.controller_decision_sha256,
                "failure_code": self.failure_code,
                "family_key": self.family_key,
                "finished_monotonic_ns": self.failure_finished_monotonic_ns,
                "runner_identity": self.failure_runner_identity,
                "schema_version": "fractal-runner-failure-timing-v1",
                "started_monotonic_ns": self.failure_started_monotonic_ns,
                "trial_key": self.trial_key,
            }
        )

    @property
    def failure_latency_ms(self) -> float | None:
        if self.execution_state != "failed":
            return None
        return (
            self.failure_finished_monotonic_ns - self.failure_started_monotonic_ns
        ) / 1_000_000.0

    @classmethod
    def from_dict(cls, payload: object) -> ActionPanelAdmissionRecord:
        row = _closed_mapping(
            payload,
            fields=_ACTION_PANEL_ADMISSION_RECORD_FIELDS,
            label="action panel admission record",
        )
        reasons = row["controller_reasons"]
        if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
            raise ConfirmatoryAnalysisError(
                "action panel admission controller_reasons must be a string array"
            )
        return cls(
            trial_key=row["trial_key"],
            family_key=row["family_key"],
            action=row["action"],
            action_order=row["action_order"],
            execution_position=row["execution_position"],
            controller_selected=row["controller_selected"],
            execution_state=row["execution_state"],
            controller_risk_score=row["controller_risk_score"],
            controller_reasons=tuple(reasons),
            controller_policy_version=row["controller_policy_version"],
            controller_decision_sha256=row["controller_decision_sha256"],
            authorization_decision_id=row["authorization_decision_id"],
            authorization_request_sha256=row["authorization_request_sha256"],
            authorization_mask_sha256=row["authorization_mask_sha256"],
            authorization_mask_size=row["authorization_mask_size"],
            authorization_decision_sha256=row["authorization_decision_sha256"],
            policy_available=row["policy_available"],
            environment_sha256=row["environment_sha256"],
            document_universe_sha256=row["document_universe_sha256"],
            audit_sequence=row["audit_sequence"],
            audit_previous_record_sha256=row["audit_previous_record_sha256"],
            audit_record_sha256=row["audit_record_sha256"],
            failure_code=row["failure_code"],
            failure_started_monotonic_ns=row["failure_started_monotonic_ns"],
            failure_finished_monotonic_ns=row["failure_finished_monotonic_ns"],
            failure_runner_identity=row["failure_runner_identity"],
            failure_timing_receipt_sha256=row["failure_timing_receipt_sha256"],
            schema_version=row["schema_version"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "action_order": self.action_order,
            "audit_previous_record_sha256": self.audit_previous_record_sha256,
            "audit_record_sha256": self.audit_record_sha256,
            "audit_sequence": self.audit_sequence,
            "authorization_decision_id": self.authorization_decision_id,
            "authorization_decision_sha256": self.authorization_decision_sha256,
            "authorization_mask_sha256": self.authorization_mask_sha256,
            "authorization_mask_size": self.authorization_mask_size,
            "authorization_request_sha256": self.authorization_request_sha256,
            "controller_decision_sha256": self.controller_decision_sha256,
            "controller_policy_version": self.controller_policy_version,
            "controller_reasons": list(self.controller_reasons),
            "controller_risk_score": self.controller_risk_score,
            "controller_selected": self.controller_selected,
            "document_universe_sha256": self.document_universe_sha256,
            "environment_sha256": self.environment_sha256,
            "execution_position": self.execution_position,
            "execution_state": self.execution_state,
            "failure_code": self.failure_code,
            "failure_finished_monotonic_ns": self.failure_finished_monotonic_ns,
            "failure_runner_identity": self.failure_runner_identity,
            "failure_started_monotonic_ns": self.failure_started_monotonic_ns,
            "failure_timing_receipt_sha256": self.failure_timing_receipt_sha256,
            "family_key": self.family_key,
            "policy_available": self.policy_available,
            "schema_version": self.schema_version,
            "trial_key": self.trial_key,
        }


@dataclass(frozen=True)
class ActionPanelAdmissionReceipt:
    """Detached proof that the trusted factory admitted one exact panel."""

    manifest_sha256: str
    run_receipt_sha256: str
    execution_artifact_sha256: str
    action_panel_artifact_sha256: str
    corpus: str
    query_partition_audit_sha256: str
    partition_label: Literal["primary", "reserve"]
    audit_head_sha256: str
    audit_chain_length: int
    audit_record_sha256s: tuple[str, ...]
    records: tuple[ActionPanelAdmissionRecord, ...]
    factory_version: str = ACTION_PANEL_FACTORY_VERSION
    schema_version: str = ACTION_PANEL_ADMISSION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
            "action_panel_artifact_sha256",
            "query_partition_audit_sha256",
            "audit_head_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_identifier("corpus", self.corpus)
        if self.partition_label not in {"primary", "reserve"}:
            raise ConfirmatoryAnalysisError("partition_label must be 'primary' or 'reserve'")
        if self.factory_version != ACTION_PANEL_FACTORY_VERSION:
            raise ConfirmatoryAnalysisError(
                f"factory_version must equal {ACTION_PANEL_FACTORY_VERSION!r}"
            )
        if self.schema_version != ACTION_PANEL_ADMISSION_RECEIPT_SCHEMA:
            raise ConfirmatoryAnalysisError(
                f"schema_version must equal {ACTION_PANEL_ADMISSION_RECEIPT_SCHEMA!r}"
            )
        if (
            isinstance(self.audit_chain_length, bool)
            or not isinstance(self.audit_chain_length, int)
            or self.audit_chain_length <= 0
        ):
            raise ConfirmatoryAnalysisError("audit_chain_length must be positive")
        audit_hashes = tuple(self.audit_record_sha256s)
        if len(audit_hashes) != self.audit_chain_length:
            raise ConfirmatoryAnalysisError("audit_record_sha256s must match audit_chain_length")
        if len(audit_hashes) != len(set(audit_hashes)):
            raise ConfirmatoryAnalysisError("audit record digests must be unique")
        for digest in audit_hashes:
            _require_sha256("audit_record_sha256", digest)
        if audit_hashes[-1] != self.audit_head_sha256:
            raise ConfirmatoryAnalysisError("audit head does not match the final record")
        object.__setattr__(self, "audit_record_sha256s", audit_hashes)

        records = tuple(self.records)
        if not records or not all(
            isinstance(record, ActionPanelAdmissionRecord) for record in records
        ):
            raise ConfirmatoryAnalysisError("admission receipt records must contain typed records")
        keys = [(record.trial_key, record.action) for record in records]
        if len(keys) != len(set(keys)):
            raise ConfirmatoryAnalysisError(
                "admission receipt contains duplicate trial-action records"
            )
        ordered = tuple(
            sorted(
                records,
                key=lambda record: (
                    record.family_key,
                    record.trial_key,
                    record.action_order,
                ),
            )
        )
        object.__setattr__(self, "records", ordered)

        governed = sorted(
            (record for record in records if record.audit_record_sha256 is not None),
            key=lambda record: record.audit_sequence,
        )
        if len(governed) != self.audit_chain_length:
            raise ConfirmatoryAnalysisError(
                "admitted governed records must match audit_chain_length"
            )
        previous = GENESIS_RECORD_SHA256
        observed_hashes: list[str] = []
        for sequence, record in enumerate(governed):
            if record.audit_sequence != sequence:
                raise ConfirmatoryAnalysisError(
                    "admission receipt audit sequences are not contiguous"
                )
            if record.audit_previous_record_sha256 != previous:
                raise ConfirmatoryAnalysisError("admission receipt audit predecessor mismatch")
            observed_hashes.append(record.audit_record_sha256)
            previous = record.audit_record_sha256
        if tuple(observed_hashes) != audit_hashes or previous != self.audit_head_sha256:
            raise ConfirmatoryAnalysisError(
                "admission receipt audit records do not match the trusted chain anchor"
            )

        by_trial: dict[str, list[ActionPanelAdmissionRecord]] = {}
        for record in records:
            by_trial.setdefault(record.trial_key, []).append(record)
        for trial_records in by_trial.values():
            if sum(record.controller_selected for record in trial_records) != 1:
                raise ConfirmatoryAnalysisError(
                    "each admitted trial needs one controller-selected decision"
                )
            contexts = {
                (
                    record.controller_policy_version,
                    record.environment_sha256,
                    record.document_universe_sha256,
                    record.authorization_mask_sha256,
                    record.authorization_mask_size,
                    record.policy_available,
                )
                for record in trial_records
            }
            if len(contexts) != 1:
                raise ConfirmatoryAnalysisError(
                    "paired admissions do not share one trusted policy universe"
                )
        action_sets = {frozenset(record.action for record in rows) for rows in by_trial.values()}
        if len(action_sets) != 1:
            raise ConfirmatoryAnalysisError("admission trials do not share one action set")
        _assert_balanced_execution_positions(
            records,
            action_set=tuple(sorted(next(iter(action_sets)))),
            trial_attribute="trial_key",
            family_attribute="family_key",
        )

    def validate_panel(self, panel: ActionPanelArtifact) -> None:
        """Require the receipt to bind every byte and every action row in ``panel``."""

        if not isinstance(panel, ActionPanelArtifact):
            raise ConfirmatoryAnalysisError("panel must be an ActionPanelArtifact")
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "execution_artifact_sha256",
            "corpus",
        ):
            if getattr(self, name) != getattr(panel, name):
                raise ConfirmatoryAnalysisError(
                    f"action-panel admission receipt has mismatched {name}"
                )
        if self.action_panel_artifact_sha256 != panel.artifact_sha256:
            raise ConfirmatoryAnalysisError(
                "action-panel admission receipt does not bind the panel digest"
            )
        admitted_by_key = {(record.trial_key, record.action): record for record in self.records}
        panel_by_key = {(row.trial_key, row.action): row for row in panel.rows}
        if set(admitted_by_key) != set(panel_by_key):
            raise ConfirmatoryAnalysisError(
                "action-panel admission receipt does not cover the exact panel rows"
            )
        for key, row in panel_by_key.items():
            admitted = admitted_by_key[key]
            for name in (
                "family_key",
                "action_order",
                "execution_position",
                "controller_selected",
                "execution_state",
                "audit_record_sha256",
            ):
                if getattr(admitted, name) != getattr(row, name):
                    raise ConfirmatoryAnalysisError(
                        f"action-panel admission record has mismatched {name}"
                    )
            if row.execution_state == "failed":
                if admitted.failure_code != row.failure_state:
                    raise ConfirmatoryAnalysisError("action-panel failure code is not admitted")
                if not math.isclose(
                    row.request_latency_ms,
                    admitted.failure_latency_ms,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ConfirmatoryAnalysisError(
                        "action-panel failure latency is not derived from runner timing"
                    )

    @classmethod
    def from_dict(cls, payload: object) -> ActionPanelAdmissionReceipt:
        receipt = _closed_mapping(
            payload,
            fields=_ACTION_PANEL_ADMISSION_RECEIPT_FIELDS,
            label="action panel admission receipt",
        )
        audit_hashes = receipt["audit_record_sha256s"]
        records = receipt["records"]
        if not isinstance(audit_hashes, list) or not all(
            isinstance(value, str) for value in audit_hashes
        ):
            raise ConfirmatoryAnalysisError(
                "admission receipt audit_record_sha256s must be a string array"
            )
        if not isinstance(records, list):
            raise ConfirmatoryAnalysisError("admission receipt records must be an array")
        return cls(
            manifest_sha256=receipt["manifest_sha256"],
            run_receipt_sha256=receipt["run_receipt_sha256"],
            execution_artifact_sha256=receipt["execution_artifact_sha256"],
            action_panel_artifact_sha256=receipt["action_panel_artifact_sha256"],
            corpus=receipt["corpus"],
            query_partition_audit_sha256=receipt["query_partition_audit_sha256"],
            partition_label=receipt["partition_label"],
            audit_head_sha256=receipt["audit_head_sha256"],
            audit_chain_length=receipt["audit_chain_length"],
            audit_record_sha256s=tuple(audit_hashes),
            records=tuple(ActionPanelAdmissionRecord.from_dict(row) for row in records),
            factory_version=receipt["factory_version"],
            schema_version=receipt["schema_version"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action_panel_artifact_sha256": self.action_panel_artifact_sha256,
            "audit_chain_length": self.audit_chain_length,
            "audit_head_sha256": self.audit_head_sha256,
            "audit_record_sha256s": list(self.audit_record_sha256s),
            "corpus": self.corpus,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "factory_version": self.factory_version,
            "manifest_sha256": self.manifest_sha256,
            "partition_label": self.partition_label,
            "query_partition_audit_sha256": self.query_partition_audit_sha256,
            "records": [record.to_dict() for record in self.records],
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def loads_action_panel_admission_receipt(
    payload: str | bytes,
) -> ActionPanelAdmissionReceipt:
    """Decode one canonical detached action-panel admission receipt."""

    decoded, supplied_bytes = _decode_canonical_json_object(
        payload,
        label="action panel admission receipt",
    )
    receipt = ActionPanelAdmissionReceipt.from_dict(decoded)
    if supplied_bytes != receipt.canonical_bytes() + b"\n":
        raise ConfirmatoryAnalysisError(
            "action panel admission receipt bytes are not canonical newline-terminated JSON"
        )
    return receipt


def load_action_panel_admission_receipt(
    path: str | Path,
) -> ActionPanelAdmissionReceipt:
    """Load one bounded admission receipt without following links or hard links."""

    return loads_action_panel_admission_receipt(
        _read_action_panel_file(path, label="action panel admission receipt")
    )


def write_action_panel_admission_receipt(
    receipt: ActionPanelAdmissionReceipt,
    target: str | Path,
) -> None:
    """Create one canonical detached admission receipt without replacement."""

    if not isinstance(receipt, ActionPanelAdmissionReceipt):
        raise ConfirmatoryAnalysisError("receipt must be an ActionPanelAdmissionReceipt")
    _write_action_panel_file(
        receipt.canonical_bytes() + b"\n",
        target,
        label="action panel admission receipt",
    )


def loads_prelabel_action_row(payload: str | bytes) -> PreLabelActionRow:
    """Decode one newline-terminated canonical action-row JSON document."""

    decoded, supplied_bytes = _decode_canonical_json_object(
        payload,
        label="pre-label action row",
    )
    row = PreLabelActionRow.from_dict(decoded)
    if supplied_bytes != row.canonical_bytes() + b"\n":
        raise ConfirmatoryAnalysisError(
            "pre-label action row bytes are not canonical newline-terminated JSON"
        )
    return row


def load_prelabel_action_row(path: str | Path) -> PreLabelActionRow:
    """Load one bounded action row without following links or hard links."""

    return loads_prelabel_action_row(_read_action_panel_file(path, label="pre-label action row"))


def write_prelabel_action_row(
    row: PreLabelActionRow,
    target: str | Path,
) -> None:
    """Create one canonical action-row file without replacing any path."""

    if not isinstance(row, PreLabelActionRow):
        raise ConfirmatoryAnalysisError("row must be a PreLabelActionRow")
    _write_action_panel_file(
        row.canonical_bytes() + b"\n",
        target,
        label="pre-label action row",
    )


def loads_action_panel_artifact(payload: str | bytes) -> ActionPanelArtifact:
    """Decode one newline-terminated canonical action-panel JSON document."""

    decoded, supplied_bytes = _decode_canonical_json_object(
        payload,
        label="action panel artifact",
    )
    artifact = ActionPanelArtifact.from_dict(decoded)
    if supplied_bytes != artifact.canonical_bytes() + b"\n":
        raise ConfirmatoryAnalysisError(
            "action panel artifact bytes are not canonical newline-terminated JSON"
        )
    return artifact


def load_action_panel_artifact(path: str | Path) -> ActionPanelArtifact:
    """Load one bounded action panel without following links or hard links."""

    return loads_action_panel_artifact(_read_action_panel_file(path, label="action panel artifact"))


def write_action_panel_artifact(
    artifact: ActionPanelArtifact,
    target: str | Path,
) -> None:
    """Create one canonical action-panel file without replacing any path."""

    if not isinstance(artifact, ActionPanelArtifact):
        raise ConfirmatoryAnalysisError("artifact must be an ActionPanelArtifact")
    _write_action_panel_file(
        artifact.canonical_bytes() + b"\n",
        target,
        label="action panel artifact",
    )


@dataclass(frozen=True)
class ConfirmatoryTrialRow:
    """One action outcome under the closed confirmatory row schema."""

    corpus_id: str
    family_id: str
    trial_id: str
    action: str
    action_order: int
    execution_position: int
    execution_state: ExecutionState
    failure_state: str | None
    controller_selected: bool
    request_latency_ms: float
    recall_at_k: float | None
    evidence_sufficient: bool | None
    entitlement_violations: int
    feature_values: tuple[object, ...] | None
    schema_version: str = CONFIRMATORY_ROW_SCHEMA

    def __post_init__(self) -> None:
        for name in ("corpus_id", "family_id", "trial_id", "action"):
            _require_identifier(name, getattr(self, name))
        if self.schema_version != CONFIRMATORY_ROW_SCHEMA:
            raise ConfirmatoryAnalysisError(
                f"schema_version must equal {CONFIRMATORY_ROW_SCHEMA!r}"
            )
        if not isinstance(self.execution_state, str) or self.execution_state not in {
            "completed",
            "failed",
            "abstained",
        }:
            raise ConfirmatoryAnalysisError("execution_state is not registered")
        if (
            isinstance(self.action_order, bool)
            or not isinstance(self.action_order, (int, np.integer))
            or int(self.action_order) < 0
        ):
            raise ConfirmatoryAnalysisError("action_order must be a non-negative integer")
        object.__setattr__(self, "action_order", int(self.action_order))
        if (
            isinstance(self.execution_position, bool)
            or not isinstance(self.execution_position, (int, np.integer))
            or not 0 <= int(self.execution_position) < 4
        ):
            raise ConfirmatoryAnalysisError(
                "execution_position must be an integer from zero through three"
            )
        object.__setattr__(self, "execution_position", int(self.execution_position))
        if type(self.controller_selected) is not bool:
            raise ConfirmatoryAnalysisError("controller_selected must be boolean")
        latency = _finite_number("request_latency_ms", self.request_latency_ms, minimum=0.0)
        if latency == 0.0:
            raise ConfirmatoryAnalysisError("request_latency_ms must be positive")
        object.__setattr__(self, "request_latency_ms", latency)
        if (
            isinstance(self.entitlement_violations, bool)
            or not isinstance(self.entitlement_violations, (int, np.integer))
            or int(self.entitlement_violations) < 0
        ):
            raise ConfirmatoryAnalysisError("entitlement_violations must be a non-negative integer")
        object.__setattr__(self, "entitlement_violations", int(self.entitlement_violations))

        if self.execution_state == "completed":
            if self.failure_state is not None:
                raise ConfirmatoryAnalysisError("completed rows cannot carry a failure_state")
            recall = _finite_number("recall_at_k", self.recall_at_k)
            if not 0.0 <= recall <= 1.0:
                raise ConfirmatoryAnalysisError("recall_at_k must be between zero and one")
            object.__setattr__(self, "recall_at_k", recall)
        else:
            if self.failure_state is None:
                raise ConfirmatoryAnalysisError(
                    "failed and abstained rows need an explicit failure_state"
                )
            _require_identifier("failure_state", self.failure_state)
            if self.recall_at_k is not None:
                raise ConfirmatoryAnalysisError(
                    "non-completed rows must encode recall_at_k as null"
                )

        if self.evidence_sufficient is not None and type(self.evidence_sufficient) is not bool:
            raise ConfirmatoryAnalysisError("evidence_sufficient must be boolean or null")
        if self.feature_values is not None:
            values = tuple(np.nan if value is None else value for value in self.feature_values)
            for value in values:
                _json_feature(value)
            object.__setattr__(self, "feature_values", values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConfirmatoryTrialRow:
        if not isinstance(payload, Mapping) or set(payload) != _ROW_FIELDS:
            observed = set(payload) if isinstance(payload, Mapping) else set()
            raise ConfirmatoryAnalysisError(
                "confirmatory row keys do not match the closed schema; "
                f"missing={sorted(_ROW_FIELDS - observed)}, "
                f"unexpected={sorted(observed - _ROW_FIELDS)}"
            )
        raw_features = payload["feature_values"]
        if raw_features is not None and not isinstance(raw_features, list):
            raise ConfirmatoryAnalysisError("feature_values must be an array or null")
        features = (
            None
            if raw_features is None
            else tuple(np.nan if value is None else value for value in raw_features)
        )
        return cls(
            corpus_id=payload["corpus_id"],
            family_id=payload["family_id"],
            trial_id=payload["trial_id"],
            action=payload["action"],
            action_order=payload["action_order"],
            execution_position=payload["execution_position"],
            execution_state=payload["execution_state"],
            failure_state=payload["failure_state"],
            controller_selected=payload["controller_selected"],
            request_latency_ms=payload["request_latency_ms"],
            recall_at_k=payload["recall_at_k"],
            evidence_sufficient=payload["evidence_sufficient"],
            entitlement_violations=payload["entitlement_violations"],
            feature_values=features,
            schema_version=payload["schema_version"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "action_order": self.action_order,
            "controller_selected": self.controller_selected,
            "corpus_id": self.corpus_id,
            "entitlement_violations": self.entitlement_violations,
            "execution_position": self.execution_position,
            "evidence_sufficient": self.evidence_sufficient,
            "execution_state": self.execution_state,
            "failure_state": self.failure_state,
            "family_id": self.family_id,
            "feature_values": (
                None
                if self.feature_values is None
                else [_json_feature(value) for value in self.feature_values]
            ),
            "recall_at_k": self.recall_at_k,
            "request_latency_ms": self.request_latency_ms,
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
        }


@dataclass(frozen=True)
class ConfirmatoryAnalysisConfig:
    """Every frozen analysis choice that can alter a primary decision."""

    fixed_corpora: tuple[str, ...]
    evidence_corpora: tuple[str, ...]
    action_set: tuple[str, ...]
    static_comparator_action: str
    low_geometry: tuple[tuple[str, float], ...]
    high_geometry: tuple[tuple[str, float], ...]
    geometry_gain_thresholds: GeometryGainThresholds
    selected_families_per_corpus: int
    nested_rows_per_family: int
    k: int = 10
    minimum_corpora_with_geometry_gain: int = 4
    failure_recall_threshold: float = 0.90
    alpha: float = 0.05
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 20260713
    h1_minimum_risk_increase: float = 0.0
    retrieval_target_noninferiority_margin: float = 0.01
    evidence_sufficiency_noninferiority_margin: float = 0.01
    minimum_cost_reduction: float = 0.10
    maximum_p95_latency_ratio: float = 1.25
    maximum_entitlement_violations: int = 0

    def __post_init__(self) -> None:
        if self.k != 10 or isinstance(self.k, bool):
            raise ConfirmatoryAnalysisError("k must equal the registered 10")
        for name in ("fixed_corpora", "evidence_corpora", "action_set"):
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise ConfirmatoryAnalysisError(f"{name} must be non-empty and unique")
            for value in values:
                _require_identifier(name, value)
            object.__setattr__(self, name, values)
        if not set(self.evidence_corpora).issubset(self.fixed_corpora):
            raise ConfirmatoryAnalysisError("evidence_corpora must be a fixed-corpus subset")
        if self.static_comparator_action not in self.action_set:
            raise ConfirmatoryAnalysisError("static comparator is outside the action set")
        if self.static_comparator_action == "abstain":
            raise ConfirmatoryAnalysisError("static comparator cannot be abstain")
        if not {"hnsw-low", "exact-authorized", "abstain"}.issubset(self.action_set):
            raise ConfirmatoryAnalysisError(
                "action_set must contain hnsw-low, exact-authorized, and abstain"
            )
        _require_identifier("static_comparator_action", self.static_comparator_action)
        if not isinstance(self.geometry_gain_thresholds, GeometryGainThresholds):
            raise ConfirmatoryAnalysisError(
                "geometry_gain_thresholds must be GeometryGainThresholds"
            )
        if any(
            value > 1.0
            for value in (
                self.geometry_gain_thresholds.log_loss_reduction,
                self.geometry_gain_thresholds.brier_score_reduction,
                self.geometry_gain_thresholds.auprc_gain,
            )
        ):
            raise ConfirmatoryAnalysisError("geometry gain thresholds cannot exceed one")
        if (
            isinstance(self.minimum_corpora_with_geometry_gain, bool)
            or not isinstance(self.minimum_corpora_with_geometry_gain, int)
            or not 1 <= self.minimum_corpora_with_geometry_gain <= len(self.fixed_corpora)
        ):
            raise ConfirmatoryAnalysisError(
                "minimum_corpora_with_geometry_gain is outside the fixed suite"
            )
        for name in ("selected_families_per_corpus", "nested_rows_per_family"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfirmatoryAnalysisError(f"{name} must be a positive integer")
        if self.selected_families_per_corpus < 2:
            raise ConfirmatoryAnalysisError(
                "selected_families_per_corpus must be at least two for family bootstrap"
            )
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or self.bootstrap_replicates < 10_000
        ):
            raise ConfirmatoryAnalysisError("bootstrap_replicates must be at least 10000")
        if (
            isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, int)
            or self.bootstrap_seed < 0
        ):
            raise ConfirmatoryAnalysisError("bootstrap_seed must be non-negative")
        if self.alpha != 0.05:
            raise ConfirmatoryAnalysisError("alpha must equal the registered 0.05")
        bounded = {
            "failure_recall_threshold": self.failure_recall_threshold,
            "retrieval_target_noninferiority_margin": (self.retrieval_target_noninferiority_margin),
            "evidence_sufficiency_noninferiority_margin": (
                self.evidence_sufficiency_noninferiority_margin
            ),
            "minimum_cost_reduction": self.minimum_cost_reduction,
        }
        for name, value in bounded.items():
            number = _finite_number(name, value, minimum=0.0)
            if number > 1.0:
                raise ConfirmatoryAnalysisError(f"{name} cannot exceed one")
        _finite_number(
            "h1_minimum_risk_increase",
            self.h1_minimum_risk_increase,
            minimum=0.0,
        )
        if self.h1_minimum_risk_increase > 1.0:
            raise ConfirmatoryAnalysisError("h1_minimum_risk_increase cannot exceed one")
        _finite_number(
            "maximum_p95_latency_ratio",
            self.maximum_p95_latency_ratio,
            minimum=0.0,
        )
        if self.maximum_p95_latency_ratio == 0.0:
            raise ConfirmatoryAnalysisError("maximum_p95_latency_ratio must be positive")
        if self.maximum_entitlement_violations != 0 or isinstance(
            self.maximum_entitlement_violations, bool
        ):
            raise ConfirmatoryAnalysisError("maximum_entitlement_violations must equal zero")
        low = self._validated_geometry(self.low_geometry, name="low_geometry")
        high = self._validated_geometry(self.high_geometry, name="high_geometry")
        if tuple(name for name, _ in low) != tuple(name for name, _ in high):
            raise ConfirmatoryAnalysisError(
                "low and high geometry profiles must name identical features"
            )
        object.__setattr__(self, "low_geometry", low)
        object.__setattr__(self, "high_geometry", high)

    @classmethod
    def from_frozen_manifest(
        cls,
        payload: Mapping[str, Any],
    ) -> ConfirmatoryAnalysisConfig:
        """Derive every primary-analysis choice from one valid frozen manifest."""

        try:
            validate_study_manifest(payload, require_frozen=True)
        except StudyManifestError as exc:
            raise ConfirmatoryAnalysisError(
                f"frozen manifest is not admissible for analysis: {exc}"
            ) from exc
        analysis = payload["analysis"]
        if not isinstance(analysis, Mapping):  # defensive after schema validation
            raise ConfirmatoryAnalysisError("frozen manifest analysis must be an object")
        gain = analysis["geometry_gain_thresholds"]
        power = analysis["power"]
        low = analysis["low_geometry"]
        high = analysis["high_geometry"]
        if not all(isinstance(value, Mapping) for value in (gain, low, high, power)):
            raise ConfirmatoryAnalysisError(
                "frozen geometry thresholds and profiles must be objects"
            )
        return cls(
            fixed_corpora=tuple(analysis["fixed_corpora"]),
            evidence_corpora=tuple(analysis["evidence_corpora"]),
            action_set=tuple(analysis["action_set"]),
            static_comparator_action=analysis["static_comparator_action"],
            low_geometry=tuple(low.items()),
            high_geometry=tuple(high.items()),
            geometry_gain_thresholds=GeometryGainThresholds(
                log_loss_reduction=gain["log_loss_reduction"],
                brier_score_reduction=gain["brier_score_reduction"],
                auprc_gain=gain["auprc_gain"],
            ),
            selected_families_per_corpus=power["selected_families_per_corpus"],
            nested_rows_per_family=analysis["nested_rows_per_family"],
            k=analysis["k"],
            minimum_corpora_with_geometry_gain=(analysis["minimum_corpora_with_geometry_gain"]),
            failure_recall_threshold=analysis["failure_recall_threshold"],
            alpha=analysis["alpha"],
            bootstrap_replicates=analysis["bootstrap_replicates"],
            bootstrap_seed=analysis["bootstrap_seed"],
            h1_minimum_risk_increase=analysis["h1_minimum_risk_increase"],
            retrieval_target_noninferiority_margin=(
                analysis["retrieval_target_noninferiority_margin"]
            ),
            evidence_sufficiency_noninferiority_margin=(
                analysis["evidence_sufficiency_noninferiority_margin"]
            ),
            minimum_cost_reduction=analysis["minimum_cost_reduction"],
            maximum_p95_latency_ratio=analysis["maximum_p95_latency_ratio"],
            maximum_entitlement_violations=(analysis["maximum_entitlement_violations"]),
        )

    @staticmethod
    def _validated_geometry(
        values: Sequence[tuple[str, float]], *, name: str
    ) -> tuple[tuple[str, float], ...]:
        pairs = tuple(values)
        if not pairs:
            raise ConfirmatoryAnalysisError(f"{name} cannot be empty")
        names: list[str] = []
        normalized: list[tuple[str, float]] = []
        for feature, value in pairs:
            _require_identifier(f"{name} feature", feature)
            names.append(feature)
            normalized.append((feature, _finite_number(f"{name}.{feature}", value)))
        if len(names) != len(set(names)):
            raise ConfirmatoryAnalysisError(f"{name} contains duplicate features")
        return tuple(sorted(normalized))

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    def low_geometry_mapping(self) -> dict[str, float]:
        return dict(self.low_geometry)

    def high_geometry_mapping(self) -> dict[str, float]:
        return dict(self.high_geometry)

    def to_dict(self) -> dict[str, object]:
        return {
            "action_set": list(self.action_set),
            "alpha": self.alpha,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "evidence_corpora": list(self.evidence_corpora),
            "evidence_sufficiency_noninferiority_margin": (
                self.evidence_sufficiency_noninferiority_margin
            ),
            "failure_recall_threshold": self.failure_recall_threshold,
            "fixed_corpora": list(self.fixed_corpora),
            "geometry_gain_thresholds": {
                "auprc_gain": self.geometry_gain_thresholds.auprc_gain,
                "brier_score_reduction": (self.geometry_gain_thresholds.brier_score_reduction),
                "log_loss_reduction": (self.geometry_gain_thresholds.log_loss_reduction),
            },
            "h1_minimum_risk_increase": self.h1_minimum_risk_increase,
            "high_geometry": dict(self.high_geometry),
            "k": self.k,
            "low_geometry": dict(self.low_geometry),
            "maximum_entitlement_violations": self.maximum_entitlement_violations,
            "maximum_p95_latency_ratio": self.maximum_p95_latency_ratio,
            "minimum_corpora_with_geometry_gain": (self.minimum_corpora_with_geometry_gain),
            "minimum_cost_reduction": self.minimum_cost_reduction,
            "nested_rows_per_family": self.nested_rows_per_family,
            "retrieval_target_noninferiority_margin": (self.retrieval_target_noninferiority_margin),
            "selected_families_per_corpus": self.selected_families_per_corpus,
            "static_comparator_action": self.static_comparator_action,
        }

    @property
    def config_sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True)
class CorpusInputDigests:
    corpus_id: str
    prediction_completion_receipt_sha256: str
    online_execution_result_receipt_sha256: str
    timelock_decryption_receipt_sha256: str
    offline_evaluation_artifact_sha256: str
    action_panel_artifact_sha256: str
    action_panel_admission_receipt_sha256: str
    online_execution_artifact_id: str
    online_execution_artifact_sha256: str
    sealed_input_artifact_id: str
    sealed_input_artifact_sha256: str
    sealed_label_artifact_id: str
    sealed_label_artifact_sha256: str

    def __post_init__(self) -> None:
        _require_identifier("corpus_id", self.corpus_id)
        for name in (
            "online_execution_artifact_id",
            "sealed_input_artifact_id",
            "sealed_label_artifact_id",
        ):
            _require_identifier(name, getattr(self, name))
        for name in (
            "prediction_completion_receipt_sha256",
            "online_execution_result_receipt_sha256",
            "timelock_decryption_receipt_sha256",
            "offline_evaluation_artifact_sha256",
            "action_panel_artifact_sha256",
            "action_panel_admission_receipt_sha256",
            "online_execution_artifact_sha256",
            "sealed_input_artifact_sha256",
            "sealed_label_artifact_sha256",
        ):
            _require_sha256(name, getattr(self, name))

    def to_dict(self) -> dict[str, str]:
        return {
            "action_panel_admission_receipt_sha256": (self.action_panel_admission_receipt_sha256),
            "action_panel_artifact_sha256": self.action_panel_artifact_sha256,
            "corpus_id": self.corpus_id,
            "offline_evaluation_artifact_sha256": (self.offline_evaluation_artifact_sha256),
            "online_execution_artifact_id": self.online_execution_artifact_id,
            "online_execution_artifact_sha256": (self.online_execution_artifact_sha256),
            "online_execution_result_receipt_sha256": (self.online_execution_result_receipt_sha256),
            "prediction_completion_receipt_sha256": (self.prediction_completion_receipt_sha256),
            "sealed_input_artifact_id": self.sealed_input_artifact_id,
            "sealed_input_artifact_sha256": self.sealed_input_artifact_sha256,
            "sealed_label_artifact_id": self.sealed_label_artifact_id,
            "sealed_label_artifact_sha256": self.sealed_label_artifact_sha256,
            "timelock_decryption_receipt_sha256": (self.timelock_decryption_receipt_sha256),
        }


@dataclass(frozen=True)
class ConfirmatoryInputArtifact:
    """Typed post-release join of receipts, labels, and anchored action panels."""

    run_receipt: SealedRunReceipt
    frozen_manifest: Mapping[str, Any]
    artifact_verification_receipt: ArtifactVerificationReceipt
    completion_receipts: tuple[PredictionCompletionReceipt, ...]
    offline_evaluations: tuple[OfflineEvaluationArtifact, ...]
    sealed_label_artifacts: tuple[SealedLabelArtifact, ...]
    action_panels: tuple[ActionPanelArtifact, ...]
    action_panel_admission_receipts: tuple[ActionPanelAdmissionReceipt, ...]
    frozen_config: ConfirmatoryAnalysisConfig = field(init=False)
    _sealed_input_bindings: tuple[tuple[str, str, str], ...] = field(
        init=False,
        repr=False,
    )
    _online_execution_bindings: tuple[tuple[str, str, str], ...] = field(
        init=False,
        repr=False,
    )
    _sealed_label_bindings: tuple[tuple[str, str, str], ...] = field(
        init=False,
        repr=False,
    )
    _h1_model_artifact_sha256: str = field(init=False, repr=False)
    _h2_model_suite_artifact_sha256: str = field(init=False, repr=False)
    schema_version: str = CONFIRMATORY_INPUT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.run_receipt, SealedRunReceipt):
            raise ConfirmatoryAnalysisError("run_receipt must be a SealedRunReceipt")
        if not isinstance(self.frozen_manifest, Mapping):
            raise ConfirmatoryAnalysisError("frozen_manifest must be the frozen manifest object")
        try:
            canonical_manifest = _canonical_bytes(self.frozen_manifest)
            manifest_payload = json.loads(canonical_manifest)
        except (ConfirmatoryAnalysisError, json.JSONDecodeError) as exc:
            raise ConfirmatoryAnalysisError(
                f"frozen_manifest cannot be canonicalized: {exc}"
            ) from exc
        if not isinstance(manifest_payload, Mapping):
            raise ConfirmatoryAnalysisError("frozen_manifest must contain one object")
        frozen_config = ConfirmatoryAnalysisConfig.from_frozen_manifest(manifest_payload)
        manifest_digest = manifest_sha256(manifest_payload)
        if manifest_digest != self.run_receipt.manifest_sha256:
            raise ConfirmatoryAnalysisError("frozen manifest does not match the sealed run receipt")
        self._validate_run_receipt(manifest_payload)
        if not isinstance(
            self.artifact_verification_receipt,
            ArtifactVerificationReceipt,
        ):
            raise ConfirmatoryAnalysisError(
                "artifact_verification_receipt must be an ArtifactVerificationReceipt"
            )
        manifest_artifacts = self._validate_verification_receipt(
            manifest_payload,
            manifest_digest=manifest_digest,
        )
        input_bindings = self._artifact_bindings_by_corpus(
            manifest_artifacts,
            role="sealed-inputs",
        )
        execution_bindings = self._artifact_bindings_by_corpus(
            manifest_artifacts,
            role="online-execution",
        )
        label_bindings = self._artifact_bindings_by_corpus(
            manifest_artifacts,
            role="sealed-labels",
        )
        h1_model = self._sole_artifact_for_role(
            manifest_artifacts,
            role="h1-predictive-model",
        )
        h2_suite = self._sole_artifact_for_role(
            manifest_artifacts,
            role="h2-model-suite",
        )
        query_partition_audit = self._sole_artifact_for_role(
            manifest_artifacts,
            role="query-partition-audit",
        )
        object.__setattr__(self, "frozen_manifest", manifest_payload)
        object.__setattr__(self, "frozen_config", frozen_config)
        object.__setattr__(
            self,
            "_sealed_input_bindings",
            tuple(
                (
                    corpus,
                    str(input_bindings[corpus]["id"]),
                    str(input_bindings[corpus]["sha256"]),
                )
                for corpus in frozen_config.fixed_corpora
            ),
        )
        object.__setattr__(
            self,
            "_sealed_label_bindings",
            tuple(
                (corpus, str(label_bindings[corpus]["id"]), str(label_bindings[corpus]["sha256"]))
                for corpus in frozen_config.fixed_corpora
            ),
        )
        object.__setattr__(
            self,
            "_online_execution_bindings",
            tuple(
                (
                    corpus,
                    str(execution_bindings[corpus]["id"]),
                    revision_sha256(
                        execution_bindings[corpus]["revision"],
                        field=(
                            "online-execution logical revision for "
                            f"{execution_bindings[corpus]['id']!r}"
                        ),
                    ),
                )
                for corpus in frozen_config.fixed_corpora
            ),
        )
        object.__setattr__(
            self,
            "_h1_model_artifact_sha256",
            str(h1_model["sha256"]),
        )
        object.__setattr__(
            self,
            "_h2_model_suite_artifact_sha256",
            str(h2_suite["sha256"]),
        )
        if self.schema_version != CONFIRMATORY_INPUT_SCHEMA:
            raise ConfirmatoryAnalysisError(
                f"schema_version must equal {CONFIRMATORY_INPUT_SCHEMA!r}"
            )
        completions = self._by_corpus(
            self.completion_receipts,
            PredictionCompletionReceipt,
            name="completion receipts",
        )
        evaluations = self._by_corpus(
            self.offline_evaluations,
            OfflineEvaluationArtifact,
            name="offline evaluations",
        )
        sealed_labels = self._by_corpus(
            self.sealed_label_artifacts,
            SealedLabelArtifact,
            name="sealed label artifacts",
        )
        panels = self._by_corpus(
            self.action_panels,
            ActionPanelArtifact,
            name="action panels",
        )
        admissions = self._by_corpus(
            self.action_panel_admission_receipts,
            ActionPanelAdmissionReceipt,
            name="action-panel admission receipts",
        )
        expected = set(frozen_config.fixed_corpora)
        for name, observed in (
            ("completion receipts", set(completions)),
            ("offline evaluations", set(evaluations)),
            ("sealed label artifacts", set(sealed_labels)),
            ("action panels", set(panels)),
            ("action-panel admission receipts", set(admissions)),
        ):
            if observed != expected:
                raise ConfirmatoryAnalysisError(
                    f"{name} do not match the frozen corpus suite; "
                    f"missing={sorted(expected - observed)}, "
                    f"unexpected={sorted(observed - expected)}"
                )
        run_digest = sealed_run_receipt_sha256(self.run_receipt)
        run_started_at = datetime.fromisoformat(
            self.run_receipt.started_at_utc.replace("Z", "+00:00")
        )
        global_trial_owners: dict[str, str] = {}
        label_binding_lookup = {
            corpus: (artifact_id, artifact_sha256)
            for corpus, artifact_id, artifact_sha256 in self._sealed_label_bindings
        }
        execution_binding_lookup = {
            corpus: (artifact_id, artifact_sha256)
            for corpus, artifact_id, artifact_sha256 in self._online_execution_bindings
        }
        for corpus_id in frozen_config.fixed_corpora:
            completion = completions[corpus_id]
            evaluation = evaluations[corpus_id]
            label_artifact = sealed_labels[corpus_id]
            panel = panels[corpus_id]
            admission = admissions[corpus_id]
            for artifact_name, stage in (
                ("completion receipt", completion.stage),
                ("offline evaluation", evaluation.stage),
                ("sealed label artifact", label_artifact.stage),
                ("action panel", panel.stage),
            ):
                if stage != "sealed":
                    raise ConfirmatoryAnalysisError(
                        f"{artifact_name} for {corpus_id!r} must have sealed stage"
                    )
            completion_time = datetime.fromisoformat(
                completion.anchored_at_utc.replace("Z", "+00:00")
            )
            if completion_time <= run_started_at:
                raise ConfirmatoryAnalysisError(
                    f"completion receipt for {corpus_id!r} must postdate the sealed run"
                )
            for artifact_name, observed_manifest in (
                ("completion receipt", completion.manifest_sha256),
                ("offline evaluation", evaluation.manifest_sha256),
                ("action panel", panel.manifest_sha256),
            ):
                if observed_manifest != manifest_digest:
                    raise ConfirmatoryAnalysisError(
                        f"{artifact_name} for {corpus_id!r} belongs to another manifest"
                    )
            for artifact_name, observed_run in (
                ("completion receipt", completion.run_receipt_sha256),
                ("offline evaluation", evaluation.run_receipt_sha256),
                ("action panel", panel.run_receipt_sha256),
            ):
                if observed_run != run_digest:
                    raise ConfirmatoryAnalysisError(
                        f"{artifact_name} for {corpus_id!r} belongs to another run"
                    )
            if panel.action_set != frozen_config.action_set:
                raise ConfirmatoryAnalysisError(
                    f"action panel for {corpus_id!r} has action-set drift"
                )
            self._validate_frozen_panel_design(panel, frozen_config)
            admission.validate_panel(panel)
            if any(
                record.execution_state == "failed"
                and record.failure_runner_identity != self.run_receipt.runner_identity
                for record in admission.records
            ):
                raise ConfirmatoryAnalysisError(
                    f"action-panel failure timing for {corpus_id!r} belongs to another runner"
                )
            if admission.partition_label != "primary":
                raise ConfirmatoryAnalysisError(
                    f"action panel for {corpus_id!r} is not bound to the primary partition"
                )
            if admission.query_partition_audit_sha256 != str(query_partition_audit["sha256"]):
                raise ConfirmatoryAnalysisError(
                    "action-panel admission receipt does not bind the frozen "
                    f"query-partition audit for {corpus_id!r}"
                )
            if not (
                completion.execution_artifact_sha256
                == evaluation.execution_artifact_sha256
                == label_artifact.execution_artifact_sha256
                == panel.execution_artifact_sha256
            ):
                raise ConfirmatoryAnalysisError(
                    f"execution artifact binding mismatch for {corpus_id!r}"
                )
            execution_artifact_id, execution_artifact_sha256 = execution_binding_lookup[corpus_id]
            if panel.execution_artifact_sha256 != execution_artifact_sha256:
                raise ConfirmatoryAnalysisError(
                    "online execution logical digest does not match manifest "
                    f"revision for artifact {execution_artifact_id!r} and corpus {corpus_id!r}"
                )
            if completion.action_panel_binding != panel.completion_binding():
                raise ConfirmatoryAnalysisError(
                    f"completion receipt does not bind the action panel for {corpus_id!r}"
                )
            if evaluation.prediction_completion_receipt_sha256 != completion.receipt_sha256:
                raise ConfirmatoryAnalysisError(
                    f"offline evaluation does not bind the completion receipt for {corpus_id!r}"
                )
            if evaluation.prediction_artifact_sha256 != completion.prediction_artifact_sha256:
                raise ConfirmatoryAnalysisError(
                    f"prediction artifact binding mismatch for {corpus_id!r}"
                )
            if (
                evaluation.online_execution_result_receipt_sha256
                != completion.online_execution_result_receipt_sha256
            ):
                raise ConfirmatoryAnalysisError(
                    f"sealed online result receipt binding mismatch for {corpus_id!r}"
                )
            label_artifact_id, label_artifact_sha256 = label_binding_lookup[corpus_id]
            observed_label_file_sha256 = hashlib.sha256(
                label_artifact.canonical_bytes() + b"\n"
            ).hexdigest()
            if observed_label_file_sha256 != label_artifact_sha256:
                raise ConfirmatoryAnalysisError(
                    "sealed label artifact file digest does not match manifest "
                    f"artifact {label_artifact_id!r} for {corpus_id!r}"
                )
            if evaluation.sealed_label_artifact_sha256 != label_artifact.artifact_sha256:
                raise ConfirmatoryAnalysisError(
                    "offline evaluation sealed-label digest does not match admitted "
                    f"artifact {label_artifact_id!r} for {corpus_id!r}; "
                    "the offline field carries the semantic identity"
                )
            if label_artifact.document_count != panel.document_count:
                raise ConfirmatoryAnalysisError(
                    f"sealed label artifact document count differs for {corpus_id!r}"
                )
            if completion.prediction_count != len(evaluation.trials):
                raise ConfirmatoryAnalysisError(f"prediction count mismatch for {corpus_id!r}")
            self._validate_trial_join(panel, evaluation, label_artifact)
            for trial in evaluation.trials:
                trial_key = trial.prediction.trial_key
                previous = global_trial_owners.setdefault(trial_key, corpus_id)
                if previous != corpus_id:
                    raise ConfirmatoryAnalysisError(
                        "one trial key cannot be reused across fixed corpora"
                    )
        object.__setattr__(
            self,
            "completion_receipts",
            tuple(completions[corpus] for corpus in frozen_config.fixed_corpora),
        )
        object.__setattr__(
            self,
            "offline_evaluations",
            tuple(evaluations[corpus] for corpus in frozen_config.fixed_corpora),
        )
        object.__setattr__(
            self,
            "sealed_label_artifacts",
            tuple(sealed_labels[corpus] for corpus in frozen_config.fixed_corpora),
        )
        object.__setattr__(
            self,
            "action_panels",
            tuple(panels[corpus] for corpus in frozen_config.fixed_corpora),
        )
        object.__setattr__(
            self,
            "action_panel_admission_receipts",
            tuple(admissions[corpus] for corpus in frozen_config.fixed_corpora),
        )

    def _validate_run_receipt(self, manifest_payload: Mapping[str, Any]) -> None:
        sealed = manifest_payload["sealed_execution"]
        expected_fields = {
            "protocol_version": manifest_payload["protocol_version"],
            "runner_identity": sealed["runner_identity"],
            "code_commit": sealed["code_commit"],
            "runner_image": sealed["runner_image"],
            "receipt_uri": sealed_receipt_uri(manifest_payload),
        }
        for field_name, expected in expected_fields.items():
            if getattr(self.run_receipt, field_name) != expected:
                raise ConfirmatoryAnalysisError(
                    f"sealed run receipt {field_name} differs from the frozen manifest"
                )

    def _validate_verification_receipt(
        self,
        manifest_payload: Mapping[str, Any],
        *,
        manifest_digest: str,
    ) -> tuple[Mapping[str, Any], ...]:
        receipt = self.artifact_verification_receipt
        if receipt.manifest_sha256 != manifest_digest:
            raise ConfirmatoryAnalysisError(
                "artifact verification receipt belongs to another manifest"
            )
        if receipt.receipt_sha256 != self.run_receipt.verification_receipt_sha256:
            raise ConfirmatoryAnalysisError(
                "artifact verification receipt does not match the sealed run receipt"
            )
        manifest_artifacts = tuple(manifest_payload["artifacts"])
        pinned_by_id = {str(artifact["id"]): artifact for artifact in manifest_artifacts}
        verified_by_id = {artifact.artifact_id: artifact for artifact in receipt.artifacts}
        if set(pinned_by_id) != set(verified_by_id):
            missing = sorted(set(pinned_by_id) - set(verified_by_id))
            unexpected = sorted(set(verified_by_id) - set(pinned_by_id))
            raise ConfirmatoryAnalysisError(
                "artifact verification receipt does not cover the frozen manifest; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for artifact_id, pin in pinned_by_id.items():
            verified = verified_by_id[artifact_id]
            expected_sha256 = str(pin["sha256"])
            if not verified.exact or (
                verified.expected_sha256 != expected_sha256
                or verified.verified_sha256 != expected_sha256
            ):
                raise ConfirmatoryAnalysisError(
                    f"artifact verification mismatch for {artifact_id!r}"
                )
            if pin["role"] in {"h1-predictive-model", "h2-model-suite"} and (
                verified.kind != "file"
            ):
                raise ConfirmatoryAnalysisError(
                    f"canonical model artifact {artifact_id!r} must be one exact file"
                )
        return manifest_artifacts

    @staticmethod
    def _artifact_bindings_by_corpus(
        artifacts: Sequence[Mapping[str, Any]],
        *,
        role: str,
    ) -> dict[str, Mapping[str, Any]]:
        return {
            str(artifact["corpus_id"]): artifact
            for artifact in artifacts
            if artifact["role"] == role
        }

    @staticmethod
    def _sole_artifact_for_role(
        artifacts: Sequence[Mapping[str, Any]],
        *,
        role: str,
    ) -> Mapping[str, Any]:
        matches = tuple(artifact for artifact in artifacts if artifact["role"] == role)
        if len(matches) != 1:  # defensive after frozen-manifest validation
            raise ConfirmatoryAnalysisError(f"frozen manifest must contain one {role!r} artifact")
        return matches[0]

    @staticmethod
    def _by_corpus(
        values: Iterable[object],
        expected_type: type,
        *,
        name: str,
    ) -> dict[str, Any]:
        rows = tuple(values)
        if not rows or not all(isinstance(row, expected_type) for row in rows):
            raise ConfirmatoryAnalysisError(
                f"{name} must contain only {expected_type.__name__} values"
            )
        corpora = [row.corpus for row in rows]
        if len(corpora) != len(set(corpora)):
            raise ConfirmatoryAnalysisError(f"{name} contain duplicate corpus entries")
        return {row.corpus: row for row in rows}

    @staticmethod
    def _validate_trial_join(
        panel: ActionPanelArtifact,
        evaluation: OfflineEvaluationArtifact,
        sealed_labels: SealedLabelArtifact,
    ) -> None:
        panel_by_trial: dict[str, list[PreLabelActionRow]] = {}
        for row in panel.rows:
            panel_by_trial.setdefault(row.trial_key, []).append(row)
        evaluation_by_trial = {trial.prediction.trial_key: trial for trial in evaluation.trials}
        labels_by_trial = {row.trial_key: row for row in sealed_labels.labels}
        if set(panel_by_trial) != set(evaluation_by_trial):
            raise ConfirmatoryAnalysisError(
                f"action panel and offline trial coverage differ for {panel.corpus!r}"
            )
        if set(evaluation_by_trial) != set(labels_by_trial):
            raise ConfirmatoryAnalysisError(
                f"offline and sealed-label trial coverage differ for {panel.corpus!r}"
            )
        for trial_key, action_rows in panel_by_trial.items():
            joined = evaluation_by_trial[trial_key]
            if joined.labels != labels_by_trial[trial_key]:
                raise ConfirmatoryAnalysisError(
                    f"joined label payload differs from the sealed artifact for {trial_key!r}"
                )
            families = {row.family_key for row in action_rows}
            if families != {joined.prediction.family_key} or (
                joined.labels.family_key != joined.prediction.family_key
            ):
                raise ConfirmatoryAnalysisError(f"family binding mismatch for trial {trial_key!r}")
            selected = [row for row in action_rows if row.controller_selected]
            if len(selected) != 1 or (
                selected[0].returned_document_ids != joined.prediction.returned_document_ids
            ):
                raise ConfirmatoryAnalysisError(
                    f"selected prediction mismatch for trial {trial_key!r}"
                )

    @staticmethod
    def _validate_frozen_panel_design(
        panel: ActionPanelArtifact,
        config: ConfirmatoryAnalysisConfig,
    ) -> None:
        trials_by_family: dict[str, set[str]] = {}
        for row in panel.rows:
            trials_by_family.setdefault(row.family_key, set()).add(row.trial_key)
        if len(trials_by_family) != config.selected_families_per_corpus:
            raise ConfirmatoryAnalysisError(
                f"corpus {panel.corpus!r} must contain exactly "
                f"{config.selected_families_per_corpus} registered query families"
            )
        wrong_nested = {
            family_key: len(trial_keys)
            for family_key, trial_keys in trials_by_family.items()
            if len(trial_keys) != config.nested_rows_per_family
        }
        if wrong_nested:
            sample = sorted(wrong_nested.items())[:5]
            raise ConfirmatoryAnalysisError(
                f"corpus {panel.corpus!r} must contain exactly "
                f"{config.nested_rows_per_family} nested trials per family; "
                f"observed={sample}"
            )

    @property
    def manifest_sha256(self) -> str:
        return self.run_receipt.manifest_sha256

    @property
    def run_receipt_sha256(self) -> str:
        return sealed_run_receipt_sha256(self.run_receipt)

    @property
    def corpus_input_digests(self) -> tuple[CorpusInputDigests, ...]:
        inputs = {
            corpus: (artifact_id, artifact_sha256)
            for corpus, artifact_id, artifact_sha256 in self._sealed_input_bindings
        }
        executions = {
            corpus: (artifact_id, artifact_sha256)
            for corpus, artifact_id, artifact_sha256 in self._online_execution_bindings
        }
        labels = {
            corpus: (artifact_id, artifact_sha256)
            for corpus, artifact_id, artifact_sha256 in self._sealed_label_bindings
        }
        return tuple(
            CorpusInputDigests(
                corpus_id=corpus_id,
                prediction_completion_receipt_sha256=completion.receipt_sha256,
                online_execution_result_receipt_sha256=(
                    completion.online_execution_result_receipt_sha256
                ),
                timelock_decryption_receipt_sha256=(evaluation.timelock_decryption_receipt_sha256),
                offline_evaluation_artifact_sha256=evaluation.artifact_sha256,
                action_panel_artifact_sha256=panel.artifact_sha256,
                action_panel_admission_receipt_sha256=admission.receipt_sha256,
                online_execution_artifact_id=executions[corpus_id][0],
                online_execution_artifact_sha256=executions[corpus_id][1],
                sealed_input_artifact_id=inputs[corpus_id][0],
                sealed_input_artifact_sha256=inputs[corpus_id][1],
                sealed_label_artifact_id=labels[corpus_id][0],
                sealed_label_artifact_sha256=labels[corpus_id][1],
            )
            for corpus_id, completion, evaluation, panel, admission in zip(
                self.frozen_config.fixed_corpora,
                self.completion_receipts,
                self.offline_evaluations,
                self.action_panels,
                self.action_panel_admission_receipts,
                strict=True,
            )
        )

    def analysis_rows(self) -> tuple[ConfirmatoryTrialRow, ...]:
        rows: list[ConfirmatoryTrialRow] = []
        for corpus_id, label_artifact, panel in zip(
            self.frozen_config.fixed_corpora,
            self.sealed_label_artifacts,
            self.action_panels,
            strict=True,
        ):
            labels = {row.trial_key: row for row in label_artifact.labels}
            exact_truth = {
                row.trial_key: row.returned_document_ids[: self.frozen_config.k]
                for row in panel.rows
                if row.action == "exact-authorized"
            }
            for action_row in panel.rows:
                trial_labels = labels[action_row.trial_key]
                if action_row.execution_state == "completed":
                    relevant = exact_truth[action_row.trial_key]
                    returned = action_row.returned_document_ids[: self.frozen_config.k]
                    if not relevant:
                        recall = 1.0 if not returned else 0.0
                    else:
                        recall = len(set(returned).intersection(relevant)) / len(relevant)
                else:
                    recall = None
                if corpus_id in self.frozen_config.evidence_corpora:
                    returned_set = set(action_row.returned_document_ids)
                    evidence_sufficient = (
                        action_row.execution_state == "completed"
                        and action_row.entitlement_violations == 0
                        and any(
                            all(
                                location.document_id in returned_set
                                for location in bundle.locations
                            )
                            for bundle in trial_labels.evidence_bundles
                        )
                    )
                else:
                    evidence_sufficient = None
                rows.append(
                    ConfirmatoryTrialRow(
                        corpus_id=corpus_id,
                        family_id=action_row.family_key,
                        trial_id=action_row.trial_key,
                        action=action_row.action,
                        action_order=action_row.action_order,
                        execution_position=action_row.execution_position,
                        execution_state=action_row.execution_state,
                        failure_state=action_row.failure_state,
                        controller_selected=action_row.controller_selected,
                        request_latency_ms=action_row.request_latency_ms,
                        recall_at_k=recall,
                        evidence_sufficient=evidence_sufficient,
                        entitlement_violations=action_row.entitlement_violations,
                        feature_values=action_row.feature_values,
                    )
                )
        return tuple(rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_verification_receipt_sha256": (
                self.artifact_verification_receipt.receipt_sha256
            ),
            "corpus_inputs": [row.to_dict() for row in self.corpus_input_digests],
            "frozen_config_sha256": self.frozen_config.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def assert_model_suite_admitted(self, suite: FrozenModelSuite) -> None:
        """Reject model bytes that differ from either verified manifest model pin."""

        if not isinstance(suite, FrozenModelSuite):
            raise ConfirmatoryAnalysisError("suite must be a FrozenModelSuite")
        h1_artifact_sha256 = hashlib.sha256(canonical_h1_model_artifact_bytes(suite)).hexdigest()
        h2_artifact_sha256 = hashlib.sha256(
            canonical_h2_model_suite_artifact_bytes(suite)
        ).hexdigest()
        if h1_artifact_sha256 != self._h1_model_artifact_sha256:
            raise ConfirmatoryAnalysisError(
                "full predictive model bytes do not match the verified h1-predictive-model artifact"
            )
        if h2_artifact_sha256 != self._h2_model_suite_artifact_sha256:
            raise ConfirmatoryAnalysisError(
                "model suite bytes do not match the verified h2-model-suite artifact"
            )


@dataclass(frozen=True)
class DirectionalGate:
    name: str
    estimate: float | None
    lower: float | None
    upper: float | None
    threshold: float
    rule: str
    confidence: float
    n_corpora: int
    n_families: int
    bootstrap_replicates: int
    bootstrap_seed: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence": self.confidence,
            "estimate": self.estimate,
            "lower": self.lower,
            "name": self.name,
            "n_corpora": self.n_corpora,
            "n_families": self.n_families,
            "passed": self.passed,
            "rule": self.rule,
            "threshold": self.threshold,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class H1Result:
    """Descriptive orientation result; it is not a primary success gate."""

    gate: DirectionalGate
    model_digest: str
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate.to_dict(),
            "model_digest": self.model_digest,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class CorpusGeometryResult:
    corpus_id: str
    log_loss_reduction: float
    brier_score_reduction: float
    auprc_gain: float | None
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "auprc_gain": self.auprc_gain,
            "brier_score_reduction": self.brier_score_reduction,
            "corpus_id": self.corpus_id,
            "log_loss_reduction": self.log_loss_reduction,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class H2Result:
    metric_gates: tuple[DirectionalGate, ...]
    corpus_results: tuple[CorpusGeometryResult, ...]
    passing_corpora: tuple[str, ...]
    minimum_corpora: int
    row_identity_digest: str
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_results": [result.to_dict() for result in self.corpus_results],
            "metric_gates": [gate.to_dict() for gate in self.metric_gates],
            "minimum_corpora": self.minimum_corpora,
            "passed": self.passed,
            "passing_corpora": list(self.passing_corpora),
            "row_identity_digest": self.row_identity_digest,
        }


@dataclass(frozen=True)
class EntitlementResult:
    observed_events: int
    families_with_events: int
    n_families: int
    exact_upper_bound: float
    confidence: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence": self.confidence,
            "exact_upper_bound": self.exact_upper_bound,
            "families_with_events": self.families_with_events,
            "n_families": self.n_families,
            "observed_events": self.observed_events,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class PositionAdjustedSensitivityResult:
    """Non-gating carryover check based on the observed execution positions."""

    gate: DirectionalGate
    position_trend_log_ratio_per_position: float
    method: str = "paired-log-ratio-corpus-linear-position-adjustment-v1"
    affects_primary_claim: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.gate, DirectionalGate):
            raise ConfirmatoryAnalysisError("position sensitivity gate must be typed")
        if not math.isfinite(self.position_trend_log_ratio_per_position):
            raise ConfirmatoryAnalysisError("position sensitivity trend must be finite")
        if self.method != "paired-log-ratio-corpus-linear-position-adjustment-v1":
            raise ConfirmatoryAnalysisError("unsupported position sensitivity method")
        if self.affects_primary_claim:
            raise ConfirmatoryAnalysisError(
                "position-adjusted sensitivity cannot alter the registered primary claim"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "affects_primary_claim": self.affects_primary_claim,
            "gate": self.gate.to_dict(),
            "method": self.method,
            "position_trend_log_ratio_per_position": (self.position_trend_log_ratio_per_position),
        }


@dataclass(frozen=True)
class H3Result:
    gates: tuple[DirectionalGate, ...]
    entitlement: EntitlementResult
    position_adjusted_sensitivity: PositionAdjustedSensitivityResult
    execution_state_counts: tuple[tuple[str, int], ...]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "entitlement": self.entitlement.to_dict(),
            "execution_state_counts": dict(self.execution_state_counts),
            "gates": [gate.to_dict() for gate in self.gates],
            "passed": self.passed,
            "position_adjusted_sensitivity": self.position_adjusted_sensitivity.to_dict(),
        }


@dataclass(frozen=True)
class ConfirmatoryResultArtifact:
    """Canonical result bound to the sealed run and all frozen choices."""

    manifest_sha256: str
    run_receipt_sha256: str
    confirmatory_input_artifact_sha256: str
    corpus_input_digests: tuple[CorpusInputDigests, ...]
    frozen_config: ConfirmatoryAnalysisConfig
    model_suite_sha256: str
    input_rows_sha256: str
    input_row_count: int
    trial_count: int
    h1: H1Result
    h2: H2Result
    h3: H3Result
    primary_claim_passed: bool
    schema_version: str = CONFIRMATORY_RESULT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "confirmatory_input_artifact_sha256",
            "model_suite_sha256",
            "input_rows_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.schema_version != CONFIRMATORY_RESULT_SCHEMA:
            raise ConfirmatoryAnalysisError(
                f"schema_version must equal {CONFIRMATORY_RESULT_SCHEMA!r}"
            )
        if self.input_row_count <= 0 or self.trial_count <= 0:
            raise ConfirmatoryAnalysisError("result row and trial counts must be positive")
        expected = self.h2.passed and self.h3.passed
        if self.primary_claim_passed is not expected:
            raise ConfirmatoryAnalysisError(
                "primary_claim_passed must equal the H2/H3 intersection; H1 is descriptive"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "frozen_config": self.frozen_config.to_dict(),
            "frozen_config_sha256": self.frozen_config.config_sha256,
            "h1": self.h1.to_dict(),
            "h2": self.h2.to_dict(),
            "h3": self.h3.to_dict(),
            "confirmatory_input_artifact_sha256": (self.confirmatory_input_artifact_sha256),
            "corpus_inputs": [row.to_dict() for row in self.corpus_input_digests],
            "input_rows_sha256": self.input_rows_sha256,
            "input_row_count": self.input_row_count,
            "manifest_sha256": self.manifest_sha256,
            "model_suite_sha256": self.model_suite_sha256,
            "primary_claim_passed": self.primary_claim_passed,
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
            "trial_count": self.trial_count,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class _ValidatedPanel:
    rows: tuple[ConfirmatoryTrialRow, ...]
    low_effort_rows: tuple[ConfirmatoryTrialRow, ...]
    adaptive_rows: tuple[ConfirmatoryTrialRow, ...]
    comparator_rows: tuple[ConfirmatoryTrialRow, ...]


def _validate_panel(
    rows: Iterable[ConfirmatoryTrialRow],
    *,
    config: ConfirmatoryAnalysisConfig,
    feature_count: int,
) -> _ValidatedPanel:
    try:
        observed = tuple(rows)
    except TypeError as exc:
        raise ConfirmatoryAnalysisError("rows must be iterable") from exc
    if not observed or not all(isinstance(row, ConfirmatoryTrialRow) for row in observed):
        raise ConfirmatoryAnalysisError("rows must contain ConfirmatoryTrialRow values")
    ordered = tuple(
        sorted(
            observed,
            key=lambda row: (
                row.corpus_id,
                row.family_id,
                row.trial_id,
                row.action_order,
            ),
        )
    )
    observed_corpora = {row.corpus_id for row in ordered}
    if observed_corpora != set(config.fixed_corpora):
        raise ConfirmatoryAnalysisError(
            "corpus set drift; "
            f"missing={sorted(set(config.fixed_corpora) - observed_corpora)}, "
            f"unexpected={sorted(observed_corpora - set(config.fixed_corpora))}"
        )
    unexpected_actions = sorted({row.action for row in ordered} - set(config.action_set))
    if unexpected_actions:
        raise ConfirmatoryAnalysisError(f"unregistered actions: {unexpected_actions}")

    keys = [(row.corpus_id, row.family_id, row.trial_id, row.action) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ConfirmatoryAnalysisError("duplicate corpus-family-trial-action row")
    trial_owners: dict[str, tuple[str, str]] = {}
    by_trial: dict[tuple[str, str, str], list[ConfirmatoryTrialRow]] = {}
    for row in ordered:
        owner = (row.corpus_id, row.family_id)
        previous = trial_owners.setdefault(row.trial_id, owner)
        if previous != owner:
            raise ConfirmatoryAnalysisError("trial_id is reused across corpus or family")
        by_trial.setdefault((*owner, row.trial_id), []).append(row)
        expected_order = config.action_set.index(row.action)
        if row.action_order != expected_order:
            raise ConfirmatoryAnalysisError(
                "action_order does not match the frozen action-set order"
            )
        needs_evidence = row.corpus_id in config.evidence_corpora
        if needs_evidence != (row.evidence_sufficient is not None):
            raise ConfirmatoryAnalysisError(
                "evidence_sufficient must be boolean exactly on evidence corpora"
            )
        if row.execution_state != "completed" and row.evidence_sufficient is True:
            raise ConfirmatoryAnalysisError("a non-completed action cannot be evidence sufficient")
        if row.action == "abstain":
            if row.execution_state != "abstained":
                raise ConfirmatoryAnalysisError(
                    "the registered abstain action must have abstained state"
                )
        elif row.execution_state == "abstained":
            raise ConfirmatoryAnalysisError(
                "only the registered abstain action may have abstained state"
            )
        if row.action == "hnsw-low":
            if row.feature_values is None or len(row.feature_values) != feature_count:
                raise ConfirmatoryAnalysisError(
                    "each hnsw-low row needs the exact frozen feature vector"
                )
        elif row.feature_values is not None:
            raise ConfirmatoryAnalysisError(
                "only hnsw-low rows may carry predictive feature values"
            )

    expected_actions = set(config.action_set)
    for trial_key, trial_rows in by_trial.items():
        trial_actions = {row.action for row in trial_rows}
        if trial_actions != expected_actions:
            raise ConfirmatoryAnalysisError(
                "missing expected corpus-family-action pairs for trial "
                f"{trial_key!r}; missing={sorted(expected_actions - trial_actions)}"
            )
        if sum(row.controller_selected for row in trial_rows) != 1:
            raise ConfirmatoryAnalysisError(
                "each trial needs exactly one controller-selected action"
            )

    _assert_balanced_execution_positions(
        ordered,
        action_set=config.action_set,
        trial_attribute="trial_id",
        family_attribute="family_id",
        corpus_attribute="corpus_id",
    )

    for corpus_id in config.fixed_corpora:
        corpus_rows = tuple(row for row in ordered if row.corpus_id == corpus_id)
        families = {row.family_id for row in corpus_rows}
        if len(families) != config.selected_families_per_corpus:
            raise ConfirmatoryAnalysisError(
                f"corpus {corpus_id!r} must contain exactly "
                f"{config.selected_families_per_corpus} registered query families"
            )
        trials_by_family: dict[str, set[str]] = {}
        for row in corpus_rows:
            trials_by_family.setdefault(row.family_id, set()).add(row.trial_id)
        wrong_nested = {
            family_id: len(trial_ids)
            for family_id, trial_ids in trials_by_family.items()
            if len(trial_ids) != config.nested_rows_per_family
        }
        if wrong_nested:
            sample = sorted(wrong_nested.items())[:5]
            raise ConfirmatoryAnalysisError(
                f"corpus {corpus_id!r} must contain exactly "
                f"{config.nested_rows_per_family} nested trials per family; "
                f"observed={sample}"
            )

    low_effort = tuple(row for row in ordered if row.action == "hnsw-low")
    adaptive = tuple(
        sorted(
            (row for row in ordered if row.controller_selected),
            key=lambda row: (row.corpus_id, row.family_id, row.trial_id),
        )
    )
    comparator = tuple(
        sorted(
            (row for row in ordered if row.action == config.static_comparator_action),
            key=lambda row: (row.corpus_id, row.family_id, row.trial_id),
        )
    )
    adaptive_ids = [(row.corpus_id, row.family_id, row.trial_id) for row in adaptive]
    comparator_ids = [(row.corpus_id, row.family_id, row.trial_id) for row in comparator]
    if adaptive_ids != comparator_ids:
        raise ConfirmatoryAnalysisError(
            "adaptive and comparator pair IDs do not match in exact order"
        )
    return _ValidatedPanel(ordered, low_effort, adaptive, comparator)


def _feature_batch(
    rows: Sequence[ConfirmatoryTrialRow],
    suite: FrozenModelSuite,
    *,
    include_action_failure_labels: bool,
    failure_recall_threshold: float,
) -> FeatureBatch | LabeledFeatureBatch:
    common = {
        "partition": "sealed",
        "feature_names": suite.schema.input_features,
        "features": np.asarray([row.feature_values for row in rows], dtype=object),
        "corpus_ids": tuple(row.corpus_id for row in rows),
        "family_ids": tuple(row.family_id for row in rows),
        "row_ids": tuple(row.trial_id for row in rows),
    }
    if not include_action_failure_labels:
        return FeatureBatch(**common)
    labels = tuple(int(_low_effort_action_failed(row, failure_recall_threshold)) for row in rows)
    return LabeledFeatureBatch(**common, labels=labels)


def _low_effort_action_failed(
    row: ConfirmatoryTrialRow,
    failure_recall_threshold: float,
) -> bool:
    """Return the intent-to-treat low-effort action-failure composite.

    A failed or otherwise non-completed low-effort action is a failure. A
    completed action is a failure only when recall against the authorized exact
    result is below the frozen threshold. Empty authorized truth is therefore a
    success only when the completed action also returns no documents.
    """

    if row.action != "hnsw-low":
        raise ConfirmatoryAnalysisError("action-failure labels are defined only for hnsw-low rows")
    return (
        row.execution_state != "completed"
        or row.recall_at_k is None
        or row.recall_at_k < failure_recall_threshold
    )


def _bootstrap_config(
    config: ConfirmatoryAnalysisConfig, *, seed_offset: int
) -> ClusterBootstrapConfig:
    return ClusterBootstrapConfig(
        n_resamples=config.bootstrap_replicates,
        confidence=config.confidence,
        seed=config.bootstrap_seed + seed_offset,
        interval_construction="directional-one-sided",
    )


def _gate(
    name: str,
    bootstrap: ClusterBootstrapResult,
    *,
    threshold: float,
    rule: str,
    passed: bool,
) -> DirectionalGate:
    interval = bootstrap.interval
    return DirectionalGate(
        name=name,
        estimate=interval.estimate,
        lower=interval.lower,
        upper=interval.upper,
        threshold=threshold,
        rule=rule,
        confidence=interval.confidence,
        n_corpora=bootstrap.n_corpora,
        n_families=bootstrap.n_families,
        bootstrap_replicates=len(bootstrap.replicates),
        bootstrap_seed=bootstrap.seed,
        passed=passed,
    )


def _weighted_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> float:
    positive_weight = float(np.sum(weights * labels))
    if positive_weight == 0.0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_weights = weights[order]
    sorted_scores = scores[order]
    weighted_positive = sorted_weights * sorted_labels
    cumulative_positive = np.cumsum(weighted_positive)
    cumulative_weight = np.cumsum(sorted_weights)
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    group_starts = np.r_[0, group_ends[:-1] + 1]
    group_positive = np.asarray(
        [
            weighted_positive[start : end + 1].sum()
            for start, end in zip(group_starts, group_ends, strict=True)
        ]
    )
    precision_at_threshold = cumulative_positive[group_ends] / cumulative_weight[group_ends]
    return float(np.sum(group_positive * precision_at_threshold) / positive_weight)


def _auprc_bootstrap(
    suite: FrozenModelSuite,
    batch: LabeledFeatureBatch,
    *,
    fixed_corpora: Sequence[str],
    estimate: float,
    config: ClusterBootstrapConfig,
) -> ClusterBootstrapResult:
    labels = np.asarray(batch.labels, dtype=np.int8)
    reference = suite.predict_proba(batch, model_name="system-policy")
    full = suite.predict_proba(batch, model_name="full")
    corpus_ids = tuple(batch.corpus_ids)
    family_ids = tuple(batch.family_ids)
    groups: dict[str, dict[str, np.ndarray]] = {}
    for corpus_id in fixed_corpora:
        corpus_groups: dict[str, np.ndarray] = {}
        for family_id in sorted(
            {
                family
                for corpus, family in zip(corpus_ids, family_ids, strict=True)
                if corpus == corpus_id
            }
        ):
            corpus_groups[family_id] = np.asarray(
                [
                    index
                    for index, (corpus, family) in enumerate(
                        zip(corpus_ids, family_ids, strict=True)
                    )
                    if corpus == corpus_id and family == family_id
                ],
                dtype=np.int64,
            )
        if len(corpus_groups) < 2:
            raise ConfirmatoryAnalysisError(
                f"corpus {corpus_id!r} needs two families for H2 bootstrap"
            )
        groups[corpus_id] = corpus_groups

    point_gains: list[float] = []
    for corpus_id in fixed_corpora:
        family_groups = tuple(groups[corpus_id].values())
        indices = np.concatenate(family_groups)
        weights = np.concatenate(
            [np.full(len(group), 1.0 / len(group), dtype=np.float64) for group in family_groups]
        )
        point_gains.append(
            _weighted_average_precision(labels[indices], full[indices], weights)
            - _weighted_average_precision(labels[indices], reference[indices], weights)
        )
    if not math.isclose(float(np.mean(point_gains)), estimate, abs_tol=1e-12):
        raise ConfirmatoryAnalysisError(
            "H2 AUPRC point and family-weighted bootstrap estimands disagree"
        )

    rng = np.random.default_rng(config.seed)
    replicates = np.zeros(config.n_resamples, dtype=np.float64)
    for replicate in range(config.n_resamples):
        for corpus_id in fixed_corpora:
            family_groups = tuple(groups[corpus_id].values())
            draws = rng.integers(0, len(family_groups), size=len(family_groups))
            sampled = [family_groups[index] for index in draws]
            indices = np.concatenate(sampled)
            weights = np.concatenate(
                [np.full(len(group), 1.0 / len(group), dtype=np.float64) for group in sampled]
            )
            reference_ap = _weighted_average_precision(labels[indices], reference[indices], weights)
            full_ap = _weighted_average_precision(labels[indices], full[indices], weights)
            replicates[replicate] += (full_ap - reference_ap) / len(fixed_corpora)
    lower, upper = np.quantile(replicates, (1.0 - config.confidence, config.confidence))
    return ClusterBootstrapResult(
        interval=ConfidenceInterval(
            estimate=estimate,
            lower=float(lower),
            upper=float(upper),
            confidence=config.confidence,
            construction="directional-one-sided",
        ),
        replicates=replicates,
        n_corpora=len(fixed_corpora),
        n_families=sum(len(value) for value in groups.values()),
        seed=config.seed,
    )


def _h1(
    suite: FrozenModelSuite,
    batch: FeatureBatch,
    config: ConfirmatoryAnalysisConfig,
) -> H1Result:
    contrast = predictive_geometry_risk_contrast(
        suite,
        batch,
        low_geometry=config.low_geometry_mapping(),
        high_geometry=config.high_geometry_mapping(),
        fixed_corpora=config.fixed_corpora,
    )
    zeros = np.zeros(len(contrast.per_row_differences), dtype=np.float64)
    bootstrap = paired_stratified_family_bootstrap(
        contrast.per_row_differences,
        zeros,
        batch.corpus_ids,
        batch.family_ids,
        proposed_pair_ids=batch.row_ids,
        comparator_pair_ids=batch.row_ids,
        config=_bootstrap_config(config, seed_offset=11),
    )
    if not math.isclose(bootstrap.interval.estimate, contrast.estimate, abs_tol=1e-12):
        raise ConfirmatoryAnalysisError("H1 point and clustered estimands disagree")
    decision = superiority_decision(
        bootstrap.interval,
        minimum_effect=config.h1_minimum_risk_increase,
        direction="greater",
    )
    gate = _gate(
        "h1_high_minus_low_predictive_risk",
        bootstrap,
        threshold=decision.threshold,
        rule="directional-lower-greater-than",
        passed=decision.passed,
    )
    return H1Result(gate=gate, model_digest=contrast.model_digest, passed=gate.passed)


def _h2(
    suite: FrozenModelSuite,
    batch: LabeledFeatureBatch,
    config: ConfirmatoryAnalysisConfig,
) -> H2Result:
    evaluation = evaluate_h2_by_corpus(
        suite,
        batch,
        fixed_corpora=config.fixed_corpora,
    )
    point_gate = evaluate_geometry_gain_gate(
        evaluation,
        thresholds=config.geometry_gain_thresholds,
        minimum_corpora=config.minimum_corpora_with_geometry_gain,
    )
    corpus_ids: list[str] = []
    family_ids: list[str] = []
    pair_ids: list[str] = []
    reference_log: list[float] = []
    full_log: list[float] = []
    reference_brier: list[float] = []
    full_brier: list[float] = []
    for corpus in evaluation.corpus_metrics:
        for family in corpus.family_paired_losses:
            corpus_ids.append(family.corpus_id)
            family_ids.append(family.family_id)
            pair_ids.append(f"{family.corpus_id}\x1f{family.family_id}")
            reference_log.append(family.system_policy_log_loss)
            full_log.append(family.full_log_loss)
            reference_brier.append(family.system_policy_brier_score)
            full_brier.append(family.full_brier_score)
    log_bootstrap = paired_stratified_family_bootstrap(
        reference_log,
        full_log,
        corpus_ids,
        family_ids,
        proposed_pair_ids=pair_ids,
        comparator_pair_ids=pair_ids,
        config=_bootstrap_config(config, seed_offset=21),
    )
    brier_bootstrap = paired_stratified_family_bootstrap(
        reference_brier,
        full_brier,
        corpus_ids,
        family_ids,
        proposed_pair_ids=pair_ids,
        comparator_pair_ids=pair_ids,
        config=_bootstrap_config(config, seed_offset=22),
    )
    bootstrap_metrics = (
        (
            "h2_log_loss_reduction",
            log_bootstrap,
            config.geometry_gain_thresholds.log_loss_reduction,
        ),
        (
            "h2_brier_score_reduction",
            brier_bootstrap,
            config.geometry_gain_thresholds.brier_score_reduction,
        ),
    )
    metric_gates: list[DirectionalGate] = []
    for name, bootstrap, threshold in bootstrap_metrics:
        decision = superiority_decision(
            bootstrap.interval,
            minimum_effect=threshold,
            direction="greater",
        )
        metric_gates.append(
            _gate(
                name,
                bootstrap,
                threshold=decision.threshold,
                rule="directional-lower-greater-than",
                passed=decision.passed,
            )
        )
    auprc_gain = evaluation.equal_corpus_system_policy_to_full.auprc_gain
    if auprc_gain is None:
        metric_gates.append(
            DirectionalGate(
                name="h2_auprc_gain",
                estimate=None,
                lower=None,
                upper=None,
                threshold=config.geometry_gain_thresholds.auprc_gain,
                rule="undefined-one-class-corpus_conservative-fail",
                confidence=config.confidence,
                n_corpora=len(config.fixed_corpora),
                n_families=len(set(zip(batch.corpus_ids, batch.family_ids, strict=True))),
                bootstrap_replicates=0,
                bootstrap_seed=config.bootstrap_seed + 23,
                passed=False,
            )
        )
    else:
        auprc_bootstrap = _auprc_bootstrap(
            suite,
            batch,
            fixed_corpora=config.fixed_corpora,
            estimate=auprc_gain,
            config=_bootstrap_config(config, seed_offset=23),
        )
        decision = superiority_decision(
            auprc_bootstrap.interval,
            minimum_effect=config.geometry_gain_thresholds.auprc_gain,
            direction="greater",
        )
        metric_gates.append(
            _gate(
                "h2_auprc_gain",
                auprc_bootstrap,
                threshold=decision.threshold,
                rule="directional-lower-greater-than",
                passed=decision.passed,
            )
        )
    point_lookup = {decision.corpus_id: decision for decision in point_gate.corpus_decisions}
    corpus_results = tuple(
        CorpusGeometryResult(
            corpus_id=corpus.corpus_id,
            log_loss_reduction=corpus.system_policy_to_full.log_loss_reduction,
            brier_score_reduction=corpus.system_policy_to_full.brier_score_reduction,
            auprc_gain=corpus.system_policy_to_full.auprc_gain,
            passed=point_lookup[corpus.corpus_id].passed,
        )
        for corpus in evaluation.corpus_metrics
    )
    passed = point_gate.passed and all(gate.passed for gate in metric_gates)
    return H2Result(
        metric_gates=tuple(metric_gates),
        corpus_results=corpus_results,
        passing_corpora=point_gate.passing_corpora,
        minimum_corpora=point_gate.minimum_corpora,
        row_identity_digest=evaluation.row_identity_digest,
        passed=passed,
    )


def _exact_entitlement_result(
    rows: Sequence[ConfirmatoryTrialRow],
    *,
    confidence: float,
    maximum_events: int,
) -> EntitlementResult:
    observed_events = sum(row.entitlement_violations for row in rows)
    by_family: dict[tuple[str, str], bool] = {}
    for row in rows:
        key = (row.corpus_id, row.family_id)
        by_family[key] = by_family.get(key, False) or row.entitlement_violations > 0
    event_families = sum(by_family.values())
    n_families = len(by_family)
    if event_families == n_families:
        upper = 1.0
    else:
        upper = float(
            beta_distribution.ppf(
                confidence,
                event_families + 1,
                n_families - event_families,
            )
        )
    return EntitlementResult(
        observed_events=observed_events,
        families_with_events=event_families,
        n_families=n_families,
        exact_upper_bound=upper,
        confidence=confidence,
        passed=observed_events <= maximum_events,
    )


def _position_adjusted_log_ratio_fit(
    proposed_latency: np.ndarray,
    comparator_latency: np.ndarray,
    proposed_position: np.ndarray,
    comparator_position: np.ndarray,
) -> tuple[float, float]:
    """Fit the preregistered linear position trend and its zero-delta contrast."""

    y = np.log(proposed_latency / comparator_latency)
    x = proposed_position.astype(np.float64) - comparator_position.astype(np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered = x - x_mean
    denominator = float(centered @ centered)
    slope = 0.0 if denominator == 0.0 else float(centered @ (y - y_mean) / denominator)
    return y_mean - slope * x_mean, slope


def _position_adjusted_sensitivity(
    adaptive: Sequence[ConfirmatoryTrialRow],
    comparator: Sequence[ConfirmatoryTrialRow],
    config: ConfirmatoryAnalysisConfig,
) -> PositionAdjustedSensitivityResult:
    grouped: dict[str, dict[str, list[int]]] = {}
    for index, row in enumerate(adaptive):
        grouped.setdefault(row.corpus_id, {}).setdefault(row.family_id, []).append(index)
    if set(grouped) != set(config.fixed_corpora):
        raise ConfirmatoryAnalysisError("position sensitivity lacks the fixed corpus suite")

    proposed_latency = np.asarray([row.request_latency_ms for row in adaptive], dtype=np.float64)
    comparator_latency = np.asarray(
        [row.request_latency_ms for row in comparator], dtype=np.float64
    )
    proposed_position = np.asarray([row.execution_position for row in adaptive], dtype=np.int8)
    comparator_position = np.asarray([row.execution_position for row in comparator], dtype=np.int8)
    estimate = 0.0
    slope = 0.0
    for corpus in config.fixed_corpora:
        indices = np.asarray(
            [index for values in grouped[corpus].values() for index in values],
            dtype=np.int64,
        )
        corpus_estimate, corpus_slope = _position_adjusted_log_ratio_fit(
            proposed_latency[indices],
            comparator_latency[indices],
            proposed_position[indices],
            comparator_position[indices],
        )
        estimate += corpus_estimate / len(config.fixed_corpora)
        slope += corpus_slope / len(config.fixed_corpora)

    bootstrap = _bootstrap_config(config, seed_offset=35)
    rng = np.random.default_rng(bootstrap.seed)
    replicates = np.zeros(bootstrap.n_resamples, dtype=np.float64)
    for corpus in config.fixed_corpora:
        families = tuple(sorted(grouped[corpus]))
        for replicate in range(bootstrap.n_resamples):
            draws = rng.integers(0, len(families), size=len(families))
            indices = np.asarray(
                [index for draw in draws for index in grouped[corpus][families[int(draw)]]],
                dtype=np.int64,
            )
            corpus_estimate, _ = _position_adjusted_log_ratio_fit(
                proposed_latency[indices],
                comparator_latency[indices],
                proposed_position[indices],
                comparator_position[indices],
            )
            replicates[replicate] += corpus_estimate / len(config.fixed_corpora)
    alpha = 1.0 - bootstrap.confidence
    lower, upper = np.quantile(replicates, (alpha, 1.0 - alpha))
    result = ClusterBootstrapResult(
        interval=ConfidenceInterval(
            estimate=estimate,
            lower=float(lower),
            upper=float(upper),
            confidence=bootstrap.confidence,
            construction="directional-one-sided",
        ),
        replicates=replicates,
        n_corpora=len(config.fixed_corpora),
        n_families=sum(len(families) for families in grouped.values()),
        seed=bootstrap.seed,
    )
    threshold = math.log(1.0 - config.minimum_cost_reduction)
    gate = _gate(
        "h3_position_adjusted_log_latency_ratio_sensitivity",
        result,
        threshold=threshold,
        rule="sensitivity-directional-upper-less-than-log-one-minus-minimum-reduction",
        passed=result.interval.upper < threshold,
    )
    return PositionAdjustedSensitivityResult(
        gate=gate,
        position_trend_log_ratio_per_position=slope,
    )


def _h3(panel: _ValidatedPanel, config: ConfirmatoryAnalysisConfig) -> H3Result:
    adaptive = panel.adaptive_rows
    comparator = panel.comparator_rows
    corpus_ids = tuple(row.corpus_id for row in adaptive)
    family_ids = tuple(row.family_id for row in adaptive)
    pair_ids = tuple(row.trial_id for row in adaptive)
    common = {
        "corpus_ids": corpus_ids,
        "family_ids": family_ids,
        "proposed_pair_ids": pair_ids,
        "comparator_pair_ids": pair_ids,
    }
    cost = paired_stratified_metric_bootstrap(
        (row.request_latency_ms for row in adaptive),
        (row.request_latency_ms for row in comparator),
        metric="relative-reduction",
        config=_bootstrap_config(config, seed_offset=31),
        **common,
    )
    tail = paired_stratified_metric_bootstrap(
        (row.request_latency_ms for row in adaptive),
        (row.request_latency_ms for row in comparator),
        metric="p95-ratio",
        config=_bootstrap_config(config, seed_offset=32),
        **common,
    )
    proposed_retrieval = tuple(
        float(
            row.execution_state == "completed"
            and row.recall_at_k is not None
            and row.recall_at_k >= config.failure_recall_threshold
        )
        for row in adaptive
    )
    comparator_retrieval = tuple(
        float(
            row.execution_state == "completed"
            and row.recall_at_k is not None
            and row.recall_at_k >= config.failure_recall_threshold
        )
        for row in comparator
    )
    retrieval = paired_stratified_family_bootstrap(
        proposed_retrieval,
        comparator_retrieval,
        config=_bootstrap_config(config, seed_offset=33),
        **common,
    )
    evidence_indices = tuple(
        index for index, row in enumerate(adaptive) if row.corpus_id in config.evidence_corpora
    )
    evidence_pair_ids = tuple(pair_ids[index] for index in evidence_indices)
    evidence = paired_stratified_family_bootstrap(
        (float(adaptive[index].evidence_sufficient is True) for index in evidence_indices),
        (float(comparator[index].evidence_sufficient is True) for index in evidence_indices),
        (corpus_ids[index] for index in evidence_indices),
        (family_ids[index] for index in evidence_indices),
        proposed_pair_ids=evidence_pair_ids,
        comparator_pair_ids=evidence_pair_ids,
        config=_bootstrap_config(config, seed_offset=34),
    )

    cost_decision = superiority_decision(
        cost.interval,
        minimum_effect=config.minimum_cost_reduction,
        direction="greater",
    )
    retrieval_decision = noninferiority_decision(
        retrieval.interval,
        margin=config.retrieval_target_noninferiority_margin,
        direction="greater",
    )
    evidence_decision = noninferiority_decision(
        evidence.interval,
        margin=config.evidence_sufficiency_noninferiority_margin,
        direction="greater",
    )
    tail_decision = upper_limit_decision(
        tail.interval,
        maximum=config.maximum_p95_latency_ratio,
    )
    gates = (
        _gate(
            "h3_family_latency_relative_reduction",
            cost,
            threshold=cost_decision.threshold,
            rule="directional-lower-greater-than",
            passed=cost_decision.passed,
        ),
        _gate(
            "h3_retrieval_target_difference",
            retrieval,
            threshold=retrieval_decision.threshold,
            rule="directional-lower-greater-than-negative-margin",
            passed=retrieval_decision.passed,
        ),
        _gate(
            "h3_evidence_sufficiency_difference",
            evidence,
            threshold=evidence_decision.threshold,
            rule="directional-lower-greater-than-negative-margin",
            passed=evidence_decision.passed,
        ),
        _gate(
            "h3_p95_family_latency_ratio",
            tail,
            threshold=tail_decision.threshold,
            rule="directional-upper-less-than",
            passed=tail_decision.passed,
        ),
    )
    entitlement = _exact_entitlement_result(
        panel.rows,
        confidence=config.confidence,
        maximum_events=config.maximum_entitlement_violations,
    )
    execution_state_counts = tuple(
        (state, sum(row.execution_state == state for row in panel.rows))
        for state in ("completed", "failed", "abstained")
    )
    position_sensitivity = _position_adjusted_sensitivity(adaptive, comparator, config)
    return H3Result(
        gates=gates,
        entitlement=entitlement,
        position_adjusted_sensitivity=position_sensitivity,
        execution_state_counts=execution_state_counts,
        passed=all(gate.passed for gate in gates) and entitlement.passed,
    )


def run_confirmatory_analysis(
    inputs: ConfirmatoryInputArtifact,
    *,
    suite: FrozenModelSuite,
) -> ConfirmatoryResultArtifact:
    """Analyze only a receipt-verified typed input and emit canonical results."""

    if not isinstance(inputs, ConfirmatoryInputArtifact):
        raise ConfirmatoryAnalysisError("inputs must be a ConfirmatoryInputArtifact")
    if not isinstance(suite, FrozenModelSuite):
        raise ConfirmatoryAnalysisError("suite must be a FrozenModelSuite")
    inputs.assert_model_suite_admitted(suite)
    config = inputs.frozen_config
    rows = inputs.analysis_rows()
    panel = _validate_panel(
        rows,
        config=config,
        feature_count=len(suite.schema.input_features),
    )
    feature_batch = _feature_batch(
        panel.low_effort_rows,
        suite,
        include_action_failure_labels=False,
        failure_recall_threshold=config.failure_recall_threshold,
    )
    labeled_batch = _feature_batch(
        panel.low_effort_rows,
        suite,
        include_action_failure_labels=True,
        failure_recall_threshold=config.failure_recall_threshold,
    )
    if not isinstance(feature_batch, FeatureBatch) or isinstance(
        feature_batch, LabeledFeatureBatch
    ):
        raise ConfirmatoryAnalysisError("internal H1 feature batch construction failed")
    if not isinstance(labeled_batch, LabeledFeatureBatch):
        raise ConfirmatoryAnalysisError("internal H2 feature batch construction failed")
    h1 = _h1(suite, feature_batch, config)
    h2 = _h2(suite, labeled_batch, config)
    h3 = _h3(panel, config)
    return ConfirmatoryResultArtifact(
        manifest_sha256=inputs.manifest_sha256,
        run_receipt_sha256=inputs.run_receipt_sha256,
        confirmatory_input_artifact_sha256=inputs.artifact_sha256,
        corpus_input_digests=inputs.corpus_input_digests,
        frozen_config=config,
        model_suite_sha256=suite.suite_digest,
        input_rows_sha256=_sha256([row.to_dict() for row in panel.rows]),
        input_row_count=len(panel.rows),
        trial_count=len(panel.adaptive_rows),
        h1=h1,
        h2=h2,
        h3=h3,
        primary_claim_passed=h2.passed and h3.passed,
    )
