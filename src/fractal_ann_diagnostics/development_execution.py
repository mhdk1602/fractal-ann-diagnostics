"""Outcome-blind paired-action execution for the fixed development cohort.

The runner reads only the materialized query and execution-plan artifacts.  It
never opens qrels or evidence bundles.  Each of the ten development strata is
executed against receipt-bound dual-epoch embeddings, compiled policy masks,
and authorized HNSW indexes.  The emitted action rows are the sole response-side
input accepted by :mod:`fractal_ann_diagnostics.development_freeze`.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterator, Literal, Protocol

import numpy as np

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    read_secure_regular_file,
)
from .authorized_index_store import (
    HnswlibBackend,
    VerifiedAuthorizedIndexProvider,
    load_authorized_index_store_receipt,
    open_verified_document_matrices,
)
from .compiled_policy import (
    CompiledPolicyMaskStore,
    OpenPolicyAgentMaskDecisionPoint,
)
from .confirmatory_modeling import REGISTERED_FEATURE_SCHEMA
from .controller import ControllerConfig, ControllerDecision, GovernedRetriever, RuleController
from .development_cohort import (
    CALIBRATION_FAMILY_COUNT,
    FIT_FAMILY_COUNT,
    DevelopmentCohortMaterializationReceipt,
    DevelopmentEmbeddingBinding,
    DevelopmentExecutionPlan,
    verify_materialized_development_cohort,
)
from .development_freeze import (
    PAIRED_ACTION_ROW_SCHEMA,
    REGISTERED_ACTIONS,
    REGISTERED_K,
    DevelopmentCorpusSources,
    DevelopmentFreezeConfig,
    PinnedDevelopmentFile,
    PinnedDevelopmentSelectionReceipt,
    PinnedEmbeddingStore,
    canonical_development_freeze_config_bytes,
)
from .embedding_store import (
    EmbeddingStoreReceipt,
    load_embedding_store_receipt,
    verify_embedding_store,
)
from .joint_power_design import EVIDENCE_CORPORA, FIXED_CORPORA
from .online_runner import portable_balanced_action_orders
from .policy_adapters import OPAHTTPResponse
from .policy_intervention import (
    CATALOG_FILENAME,
    OPA_DATA_FILENAME,
    CanonicalTrialSchedule,
    OPACompiledMaskData,
    PolicyInterventionConfig,
    PolicyInterventionReceipt,
    derive_policy_transition_evidence,
    load_canonical_trial_schedule,
    load_opa_compiled_mask_data,
    load_policy_intervention_config,
    load_policy_intervention_receipt,
    verify_policy_intervention_package,
)
from .policy_intervention import (
    CONFIG_FILENAME as POLICY_CONFIG_FILENAME,
)
from .policy_intervention import (
    RECEIPT_FILENAME as POLICY_RECEIPT_FILENAME,
)
from .policy_intervention import (
    SCHEDULE_FILENAME as POLICY_SCHEDULE_FILENAME,
)
from .retrieval import PolicyTransitionEvidence, packed_policy_mask_sha256

DEVELOPMENT_PAIRED_EXECUTION_CONFIG_SCHEMA = "fractal-development-paired-execution-config-v1"
DEVELOPMENT_PAIRED_EXECUTION_RECEIPT_SCHEMA = "fractal-development-paired-execution-receipt-v1"
DEVELOPMENT_PAIRED_EXECUTION_STRATUM_SCHEMA = "fractal-development-paired-execution-stratum-v1"
DEVELOPMENT_EXECUTION_ORDER_ROW_SCHEMA = "fractal-development-execution-order-row-v1"
DEVELOPMENT_EXECUTION_ORDER_SCHEMA = "fractal-development-execution-order-v1"
DEVELOPMENT_OUTPUT_ARTIFACT_SCHEMA = "fractal-development-output-artifact-v1"
DEVELOPMENT_EXECUTION_CLI_RESULT_SCHEMA = "fractal-development-execution-cli-result-v1"

DEVELOPMENT_ACTION_PERMUTATION_SEED = 20260714
DEVELOPMENT_ACTION_ORDER_ALGORITHM = "sha256-ranked-family-latin-square-v1"
DEVELOPMENT_POLICY_COMPLEXITY = 1.0
DEVELOPMENT_VERSION_LAG = 1.0
DEVELOPMENT_DRIFT_FAMILY = "qwen-terminal-token-revision-lag-one"
DEVELOPMENT_CONFIG_FILENAME = "execution-config.json"
DEVELOPMENT_RECEIPT_FILENAME = "execution-receipt.json"
DEVELOPMENT_ACTION_FILENAME = "paired-actions.jsonl"
DEVELOPMENT_ORDER_FILENAME = "execution-order.json"

PAIRING_CONTROLLER_CONFIG = ControllerConfig(
    low_ef=128,
    high_ef=512,
    probe_k=101,
    exact_scan_threshold=256,
    high_effort_threshold=0.24,
    exact_threshold=0.36,
)

DevelopmentStage = Literal["development-fit", "development-calibration"]

_DEVELOPMENT_STAGES = ("development-fit", "development-calibration")
_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_MAX_QUERY_VECTOR_BYTES = 1024 * 1024 * 1024
_FORBIDDEN_PATH_TOKENS = frozenset(
    {"sealed", "custody", "holdout", "heldout", "reserve", "reserved"}
)
_QUERY_FIELDS = frozenset({"id", "text"})
_ACTION_ROW_FIELDS = frozenset(
    {
        "action",
        "entitlement_violations",
        "execution_position",
        "execution_state",
        "failure_state",
        "family_id",
        "feature_values",
        "query_id",
        "request_latency_ms",
        "returned_document_rows",
        "schedule_order",
        "schema_version",
        "trial_key",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "action_set",
        "controller",
        "inputs",
        "k",
        "materialization_receipt_sha256",
        "materialization_root",
        "output_root",
        "permutation_seed",
        "schema_version",
    }
)
_INPUT_FIELDS = frozenset(
    {
        "authorized_index_receipt_sha256",
        "authorized_index_root",
        "corpus",
        "policy_intervention_receipt_sha256",
        "policy_intervention_root",
        "stage",
    }
)
_CONTROLLER_FIELDS = frozenset(
    {
        "exact_scan_threshold",
        "exact_threshold",
        "high_ef",
        "high_effort_threshold",
        "low_ef",
        "probe_k",
    }
)
_ORDER_ROW_FIELDS = frozenset(
    {
        "actions",
        "family_id",
        "query_id",
        "schedule_order",
        "schema_version",
        "trial_key",
    }
)
_ORDER_FIELDS = frozenset(
    {
        "action_set",
        "algorithm",
        "execution_plan_sha256",
        "permutation_seed",
        "rows",
        "schema_version",
    }
)
_OUTPUT_FIELDS = frozenset(
    {
        "byte_count",
        "path",
        "record_count",
        "role",
        "schema_version",
        "sha256",
    }
)
_STRATUM_FIELDS = frozenset(
    {
        "authorized_index_receipt_sha256",
        "corpus",
        "embedding_receipt_sha256",
        "execution_plan_sha256",
        "outputs",
        "policy_catalog_sha256",
        "policy_config_sha256",
        "policy_intervention_receipt_sha256",
        "policy_schedule_sha256",
        "selected_family_count",
        "stage",
        "trial_count",
        "schema_version",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "action_order_algorithm",
        "action_set",
        "config_sha256",
        "k",
        "materialization_receipt_sha256",
        "permutation_seed",
        "schema_version",
        "selection_receipt_sha256",
        "strata",
    }
)


class DevelopmentExecutionError(RuntimeError):
    """Raised when development execution cannot preserve its frozen boundary."""


def _canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DevelopmentExecutionError(
            "development execution artifacts require finite canonical JSON"
        ) from exc
    return encoded + (b"\n" if newline else b"")


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DevelopmentExecutionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise DevelopmentExecutionError(f"{name} must be canonical non-empty text")
    return value


def _require_integer(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DevelopmentExecutionError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise DevelopmentExecutionError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise DevelopmentExecutionError(f"{name} must be finite and >= {minimum}")
    return result


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DevelopmentExecutionError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise DevelopmentExecutionError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _decode(encoded: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DevelopmentExecutionError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise DevelopmentExecutionError(f"{label} contains non-finite value {value!r}")

    try:
        return json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentExecutionError(f"cannot decode {label}: {exc}") from exc


def _canonical_absolute_path(name: str, value: str | Path) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path.anchor != "/"
        or path.name in {"", ".", ".."}
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise DevelopmentExecutionError(f"{name} must be an absolute canonical path")
    return path


def _reject_forbidden_path(name: str, path: Path) -> None:
    tokens = {
        token
        for part in path.parts
        for token in __import__("re").split(r"[^a-z0-9]+", part.casefold())
        if token
    }
    forbidden = sorted(tokens.intersection(_FORBIDDEN_PATH_TOKENS))
    if forbidden:
        raise DevelopmentExecutionError(f"{name} crosses a non-development boundary: {forbidden}")


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _relative_path(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise DevelopmentExecutionError(f"{name} must be a relative path")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DevelopmentExecutionError(f"{name} must be a canonical relative path")
    return value


def _controller_dict(config: ControllerConfig) -> dict[str, object]:
    return {
        "exact_scan_threshold": config.exact_scan_threshold,
        "exact_threshold": config.exact_threshold,
        "high_ef": config.high_ef,
        "high_effort_threshold": config.high_effort_threshold,
        "low_ef": config.low_ef,
        "probe_k": config.probe_k,
    }


def _controller_from_dict(value: object) -> ControllerConfig:
    row = _closed(value, _CONTROLLER_FIELDS, label="pairing controller")
    try:
        return ControllerConfig(**row)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DevelopmentExecutionError(f"invalid pairing controller: {exc}") from exc


@dataclass(frozen=True, order=True)
class DevelopmentExecutionInput:
    """Exact policy and index packages for one materialized stratum."""

    corpus: str
    stage: DevelopmentStage | str
    policy_intervention_root: Path
    policy_intervention_receipt_sha256: str
    authorized_index_root: Path
    authorized_index_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.corpus not in FIXED_CORPORA:
            raise DevelopmentExecutionError("execution input corpus is outside the fixed suite")
        if self.stage not in _DEVELOPMENT_STAGES:
            raise DevelopmentExecutionError("execution input stage is not development-only")
        for name in ("policy_intervention_root", "authorized_index_root"):
            path = _canonical_absolute_path(name, getattr(self, name))
            _reject_forbidden_path(name, path)
            object.__setattr__(self, name, path)
        for name in (
            "policy_intervention_receipt_sha256",
            "authorized_index_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if _paths_overlap(self.policy_intervention_root, self.authorized_index_root):
            raise DevelopmentExecutionError("policy and index roots cannot overlap")

    def to_dict(self) -> dict[str, object]:
        return {
            "authorized_index_receipt_sha256": self.authorized_index_receipt_sha256,
            "authorized_index_root": str(self.authorized_index_root),
            "corpus": self.corpus,
            "policy_intervention_receipt_sha256": (self.policy_intervention_receipt_sha256),
            "policy_intervention_root": str(self.policy_intervention_root),
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentExecutionInput:
        row = _closed(value, _INPUT_FIELDS, label="development execution input")
        return cls(
            corpus=row["corpus"],
            stage=row["stage"],
            policy_intervention_root=Path(row["policy_intervention_root"]),
            policy_intervention_receipt_sha256=row["policy_intervention_receipt_sha256"],
            authorized_index_root=Path(row["authorized_index_root"]),
            authorized_index_receipt_sha256=row["authorized_index_receipt_sha256"],
        )


@dataclass(frozen=True)
class DevelopmentPairedExecutionConfig:
    """Closed, path-explicit configuration for all ten development strata."""

    materialization_root: Path
    materialization_receipt_sha256: str
    inputs: tuple[DevelopmentExecutionInput, ...]
    output_root: Path
    controller: ControllerConfig = PAIRING_CONTROLLER_CONFIG
    k: int = REGISTERED_K
    permutation_seed: int = DEVELOPMENT_ACTION_PERMUTATION_SEED
    action_set: tuple[str, ...] = REGISTERED_ACTIONS
    schema_version: str = DEVELOPMENT_PAIRED_EXECUTION_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in ("materialization_root", "output_root"):
            path = _canonical_absolute_path(name, getattr(self, name))
            _reject_forbidden_path(name, path)
            object.__setattr__(self, name, path)
        _require_sha256("materialization_receipt_sha256", self.materialization_receipt_sha256)
        if self.controller != PAIRING_CONTROLLER_CONFIG:
            raise DevelopmentExecutionError("pairing controller differs from the frozen probe")
        if self.k != REGISTERED_K:
            raise DevelopmentExecutionError(f"development k must equal {REGISTERED_K}")
        if self.permutation_seed != DEVELOPMENT_ACTION_PERMUTATION_SEED:
            raise DevelopmentExecutionError("development action-order seed differs")
        if tuple(self.action_set) != REGISTERED_ACTIONS:
            raise DevelopmentExecutionError("development action set differs")
        if self.schema_version != DEVELOPMENT_PAIRED_EXECUTION_CONFIG_SCHEMA:
            raise DevelopmentExecutionError("development execution config schema differs")
        inputs = tuple(sorted(self.inputs, key=lambda row: (row.stage, row.corpus)))
        expected = {(stage, corpus) for stage in _DEVELOPMENT_STAGES for corpus in FIXED_CORPORA}
        if len(inputs) != len(expected) or {(row.stage, row.corpus) for row in inputs} != expected:
            raise DevelopmentExecutionError("execution inputs do not cover the fixed ten strata")
        roots = [
            path
            for item in inputs
            for path in (item.policy_intervention_root, item.authorized_index_root)
        ]
        roots.extend((self.materialization_root, self.output_root))
        for position, left in enumerate(roots):
            for right in roots[position + 1 :]:
                if _paths_overlap(left, right):
                    raise DevelopmentExecutionError("development execution roots overlap")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "action_set", tuple(self.action_set))

    def to_dict(self) -> dict[str, object]:
        return {
            "action_set": list(self.action_set),
            "controller": _controller_dict(self.controller),
            "inputs": [row.to_dict() for row in self.inputs],
            "k": self.k,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "materialization_root": str(self.materialization_root),
            "output_root": str(self.output_root),
            "permutation_seed": self.permutation_seed,
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def config_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentPairedExecutionConfig:
        row = _closed(value, _CONFIG_FIELDS, label="development execution config")
        inputs = row["inputs"]
        actions = row["action_set"]
        if not isinstance(inputs, list) or not isinstance(actions, list):
            raise DevelopmentExecutionError("execution config arrays differ")
        return cls(
            materialization_root=Path(row["materialization_root"]),
            materialization_receipt_sha256=row["materialization_receipt_sha256"],
            inputs=tuple(DevelopmentExecutionInput.from_dict(item) for item in inputs),
            output_root=Path(row["output_root"]),
            controller=_controller_from_dict(row["controller"]),
            k=row["k"],
            permutation_seed=row["permutation_seed"],
            action_set=tuple(actions),
            schema_version=row["schema_version"],
        )


def load_development_paired_execution_config(
    path: str | Path,
) -> DevelopmentPairedExecutionConfig:
    config_path = _canonical_absolute_path("execution config path", path)
    encoded = read_secure_regular_file(
        config_path,
        max_bytes=_MAX_CONTROL_BYTES,
        label="development paired-execution config",
    )
    config = DevelopmentPairedExecutionConfig.from_dict(
        _decode(encoded, label="development paired-execution config")
    )
    if encoded != config.canonical_file_bytes():
        raise DevelopmentExecutionError("development execution config is not canonical")
    return config


def _freeze_feature_values(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(
        REGISTERED_FEATURE_SCHEMA.input_features
    ):
        raise DevelopmentExecutionError("low-action feature fields differ from the schema")
    frozen: dict[str, object] = {}
    categorical = set(REGISTERED_FEATURE_SCHEMA.system_categorical)
    missing_allowed = set(REGISTERED_FEATURE_SCHEMA.geometry_numeric) | {"probe_work"}
    for name in REGISTERED_FEATURE_SCHEMA.input_features:
        item = value[name]
        if name in categorical:
            frozen[name] = _require_text(f"feature {name}", item)
        elif item is None and name in missing_allowed:
            frozen[name] = None
        else:
            frozen[name] = _finite(f"feature {name}", item)
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class DevelopmentPairedActionRow:
    schedule_order: int
    trial_key: str
    family_id: str
    query_id: str
    action: str
    execution_position: int
    execution_state: str
    failure_state: str | None
    request_latency_ms: float
    entitlement_violations: int
    returned_document_rows: tuple[int, ...]
    feature_values: Mapping[str, object] | None
    schema_version: str = PAIRED_ACTION_ROW_SCHEMA

    def __post_init__(self) -> None:
        _require_integer("schedule_order", self.schedule_order)
        _require_sha256("trial_key", self.trial_key)
        _require_sha256("family_id", self.family_id)
        _require_text("query_id", self.query_id)
        if self.action not in REGISTERED_ACTIONS:
            raise DevelopmentExecutionError("paired action is outside the registered set")
        _require_integer("execution_position", self.execution_position)
        if not 0 <= self.execution_position < len(REGISTERED_ACTIONS):
            raise DevelopmentExecutionError("execution_position must be from zero to three")
        if self.execution_state not in {"completed", "failed", "abstained"}:
            raise DevelopmentExecutionError("paired action state is not registered")
        if self.execution_state == "completed":
            if self.failure_state is not None:
                raise DevelopmentExecutionError("completed action cannot name a failure")
        else:
            _require_text("failure_state", self.failure_state)
        latency = _finite("request_latency_ms", self.request_latency_ms, minimum=0.0)
        if latency == 0.0:
            raise DevelopmentExecutionError("request_latency_ms must be positive")
        object.__setattr__(self, "request_latency_ms", latency)
        _require_integer("entitlement_violations", self.entitlement_violations)
        if self.entitlement_violations != 0:
            raise DevelopmentExecutionError("development output contains an entitlement violation")
        returned = tuple(self.returned_document_rows)
        if (
            any(type(row) is not int or row < 0 for row in returned)
            or len(returned) != len(set(returned))
            or len(returned) > REGISTERED_K
        ):
            raise DevelopmentExecutionError(
                "returned document rows must be unique integers within registered k"
            )
        if self.execution_state != "completed" and returned:
            raise DevelopmentExecutionError("non-completed action cannot return documents")
        features = _freeze_feature_values(self.feature_values)
        if (self.action == "hnsw-low") != (features is not None):
            raise DevelopmentExecutionError("only hnsw-low must carry the feature row")
        if self.schema_version != PAIRED_ACTION_ROW_SCHEMA:
            raise DevelopmentExecutionError("paired-action row schema differs")
        object.__setattr__(self, "returned_document_rows", returned)
        object.__setattr__(self, "feature_values", features)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "entitlement_violations": self.entitlement_violations,
            "execution_position": self.execution_position,
            "execution_state": self.execution_state,
            "failure_state": self.failure_state,
            "family_id": self.family_id,
            "feature_values": (None if self.feature_values is None else dict(self.feature_values)),
            "query_id": self.query_id,
            "request_latency_ms": self.request_latency_ms,
            "returned_document_rows": list(self.returned_document_rows),
            "schedule_order": self.schedule_order,
            "schema_version": self.schema_version,
            "trial_key": self.trial_key,
        }

    def canonical_line_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentPairedActionRow:
        row = _closed(value, _ACTION_ROW_FIELDS, label="development paired-action row")
        returned = row["returned_document_rows"]
        if not isinstance(returned, list):
            raise DevelopmentExecutionError("returned_document_rows must be an array")
        return cls(
            schedule_order=row["schedule_order"],
            trial_key=row["trial_key"],
            family_id=row["family_id"],
            query_id=row["query_id"],
            action=row["action"],
            execution_position=row["execution_position"],
            execution_state=row["execution_state"],
            failure_state=row["failure_state"],
            request_latency_ms=row["request_latency_ms"],
            entitlement_violations=row["entitlement_violations"],
            returned_document_rows=tuple(returned),
            feature_values=row["feature_values"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True, order=True)
class DevelopmentExecutionOrderRow:
    schedule_order: int
    trial_key: str
    family_id: str
    query_id: str
    actions: tuple[str, ...]
    schema_version: str = DEVELOPMENT_EXECUTION_ORDER_ROW_SCHEMA

    def __post_init__(self) -> None:
        _require_integer("execution-order schedule_order", self.schedule_order)
        _require_sha256("execution-order trial_key", self.trial_key)
        _require_sha256("execution-order family_id", self.family_id)
        _require_text("execution-order query_id", self.query_id)
        actions = tuple(self.actions)
        if len(actions) != len(REGISTERED_ACTIONS) or set(actions) != set(REGISTERED_ACTIONS):
            raise DevelopmentExecutionError("execution order is not an action permutation")
        if self.schema_version != DEVELOPMENT_EXECUTION_ORDER_ROW_SCHEMA:
            raise DevelopmentExecutionError("execution-order row schema differs")
        object.__setattr__(self, "actions", actions)

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": list(self.actions),
            "family_id": self.family_id,
            "query_id": self.query_id,
            "schedule_order": self.schedule_order,
            "schema_version": self.schema_version,
            "trial_key": self.trial_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentExecutionOrderRow:
        row = _closed(value, _ORDER_ROW_FIELDS, label="development execution-order row")
        actions = row["actions"]
        if not isinstance(actions, list):
            raise DevelopmentExecutionError("execution-order actions must be an array")
        return cls(
            schedule_order=row["schedule_order"],
            trial_key=row["trial_key"],
            family_id=row["family_id"],
            query_id=row["query_id"],
            actions=tuple(actions),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class DevelopmentExecutionOrder:
    execution_plan_sha256: str
    permutation_seed: int
    rows: tuple[DevelopmentExecutionOrderRow, ...]
    action_set: tuple[str, ...] = REGISTERED_ACTIONS
    algorithm: str = DEVELOPMENT_ACTION_ORDER_ALGORITHM
    schema_version: str = DEVELOPMENT_EXECUTION_ORDER_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("execution_plan_sha256", self.execution_plan_sha256)
        if self.permutation_seed != DEVELOPMENT_ACTION_PERMUTATION_SEED:
            raise DevelopmentExecutionError("execution-order seed differs")
        if tuple(self.action_set) != REGISTERED_ACTIONS:
            raise DevelopmentExecutionError("execution-order action set differs")
        if self.algorithm != DEVELOPMENT_ACTION_ORDER_ALGORITHM:
            raise DevelopmentExecutionError("execution-order algorithm differs")
        if self.schema_version != DEVELOPMENT_EXECUTION_ORDER_SCHEMA:
            raise DevelopmentExecutionError("execution-order schema differs")
        rows = tuple(self.rows)
        if (
            not rows
            or rows != tuple(sorted(rows, key=lambda row: row.schedule_order))
            or tuple(row.schedule_order for row in rows) != tuple(range(len(rows)))
            or len({row.trial_key for row in rows}) != len(rows)
        ):
            raise DevelopmentExecutionError("execution-order rows are incomplete or unordered")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "action_set", tuple(self.action_set))

    def to_dict(self) -> dict[str, object]:
        return {
            "action_set": list(self.action_set),
            "algorithm": self.algorithm,
            "execution_plan_sha256": self.execution_plan_sha256,
            "permutation_seed": self.permutation_seed,
            "rows": [row.to_dict() for row in self.rows],
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentExecutionOrder:
        row = _closed(value, _ORDER_FIELDS, label="development execution order")
        rows = row["rows"]
        actions = row["action_set"]
        if not isinstance(rows, list) or not isinstance(actions, list):
            raise DevelopmentExecutionError("execution-order arrays differ")
        return cls(
            execution_plan_sha256=row["execution_plan_sha256"],
            permutation_seed=row["permutation_seed"],
            rows=tuple(DevelopmentExecutionOrderRow.from_dict(item) for item in rows),
            action_set=tuple(actions),
            algorithm=row["algorithm"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class DevelopmentStratumExecution:
    """Typed in-memory result from one admitted corpus-stage executor."""

    action_rows: tuple[DevelopmentPairedActionRow, ...]
    execution_order: DevelopmentExecutionOrder
    embedding_receipt_sha256: str
    policy_config_sha256: str
    policy_catalog_sha256: str
    policy_schedule_sha256: str
    policy_intervention_receipt_sha256: str
    authorized_index_receipt_sha256: str

    def __post_init__(self) -> None:
        rows = tuple(self.action_rows)
        if not rows:
            raise DevelopmentExecutionError("stratum action rows cannot be empty")
        expected_order = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.schedule_order,
                    REGISTERED_ACTIONS.index(row.action),
                ),
            )
        )
        if rows != expected_order:
            raise DevelopmentExecutionError("stratum action rows are not canonically ordered")
        for name in (
            "embedding_receipt_sha256",
            "policy_config_sha256",
            "policy_catalog_sha256",
            "policy_schedule_sha256",
            "policy_intervention_receipt_sha256",
            "authorized_index_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        object.__setattr__(self, "action_rows", rows)


@dataclass(frozen=True, order=True)
class DevelopmentOutputArtifact:
    path: str
    role: str
    sha256: str
    byte_count: int
    record_count: int
    schema_version: str = DEVELOPMENT_OUTPUT_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path("output path", self.path))
        _require_text("output role", self.role)
        _require_sha256("output SHA-256", self.sha256)
        _require_integer("output byte_count", self.byte_count, minimum=1)
        _require_integer("output record_count", self.record_count, minimum=1)
        if self.schema_version != DEVELOPMENT_OUTPUT_ARTIFACT_SCHEMA:
            raise DevelopmentExecutionError("output artifact schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "path": self.path,
            "record_count": self.record_count,
            "role": self.role,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentOutputArtifact:
        row = _closed(value, _OUTPUT_FIELDS, label="development output artifact")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True, order=True)
class DevelopmentStratumReceipt:
    corpus: str
    stage: str
    execution_plan_sha256: str
    selected_family_count: int
    trial_count: int
    embedding_receipt_sha256: str
    policy_config_sha256: str
    policy_catalog_sha256: str
    policy_schedule_sha256: str
    policy_intervention_receipt_sha256: str
    authorized_index_receipt_sha256: str
    outputs: tuple[DevelopmentOutputArtifact, ...]
    schema_version: str = DEVELOPMENT_PAIRED_EXECUTION_STRATUM_SCHEMA

    def __post_init__(self) -> None:
        if self.corpus not in FIXED_CORPORA or self.stage not in _DEVELOPMENT_STAGES:
            raise DevelopmentExecutionError("stratum receipt corpus or stage differs")
        for name in (
            "execution_plan_sha256",
            "embedding_receipt_sha256",
            "policy_config_sha256",
            "policy_catalog_sha256",
            "policy_schedule_sha256",
            "policy_intervention_receipt_sha256",
            "authorized_index_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_integer("selected_family_count", self.selected_family_count, minimum=1)
        _require_integer("trial_count", self.trial_count, minimum=1)
        expected_families = (
            FIT_FAMILY_COUNT if self.stage == "development-fit" else CALIBRATION_FAMILY_COUNT
        )
        if self.selected_family_count != expected_families:
            raise DevelopmentExecutionError("stratum family count differs from registration")
        if self.trial_count != self.selected_family_count * 3:
            raise DevelopmentExecutionError("stratum trial count is not three per family")
        outputs = tuple(sorted(self.outputs, key=lambda row: row.path.encode("utf-8")))
        if len(outputs) != 2 or {row.role for row in outputs} != {
            "execution-order",
            "paired-actions",
        }:
            raise DevelopmentExecutionError("stratum output roles differ")
        by_role = {row.role: row for row in outputs}
        expected_paths = {
            "execution-order": f"{self.stage}/{self.corpus}/{DEVELOPMENT_ORDER_FILENAME}",
            "paired-actions": f"{self.stage}/{self.corpus}/{DEVELOPMENT_ACTION_FILENAME}",
        }
        expected_records = {
            "execution-order": self.trial_count,
            "paired-actions": self.trial_count * len(REGISTERED_ACTIONS),
        }
        if any(
            by_role[role].path != expected_paths[role]
            or by_role[role].record_count != expected_records[role]
            for role in expected_paths
        ):
            raise DevelopmentExecutionError("stratum output path or record count differs")
        if self.schema_version != DEVELOPMENT_PAIRED_EXECUTION_STRATUM_SCHEMA:
            raise DevelopmentExecutionError("stratum receipt schema differs")
        object.__setattr__(self, "outputs", outputs)

    def to_dict(self) -> dict[str, object]:
        return {
            "authorized_index_receipt_sha256": self.authorized_index_receipt_sha256,
            "corpus": self.corpus,
            "embedding_receipt_sha256": self.embedding_receipt_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "outputs": [row.to_dict() for row in self.outputs],
            "policy_catalog_sha256": self.policy_catalog_sha256,
            "policy_config_sha256": self.policy_config_sha256,
            "policy_intervention_receipt_sha256": (self.policy_intervention_receipt_sha256),
            "policy_schedule_sha256": self.policy_schedule_sha256,
            "schema_version": self.schema_version,
            "selected_family_count": self.selected_family_count,
            "stage": self.stage,
            "trial_count": self.trial_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentStratumReceipt:
        row = _closed(value, _STRATUM_FIELDS, label="development stratum receipt")
        outputs = row["outputs"]
        if not isinstance(outputs, list):
            raise DevelopmentExecutionError("stratum outputs must be an array")
        return cls(
            corpus=row["corpus"],
            stage=row["stage"],
            execution_plan_sha256=row["execution_plan_sha256"],
            selected_family_count=row["selected_family_count"],
            trial_count=row["trial_count"],
            embedding_receipt_sha256=row["embedding_receipt_sha256"],
            policy_config_sha256=row["policy_config_sha256"],
            policy_catalog_sha256=row["policy_catalog_sha256"],
            policy_schedule_sha256=row["policy_schedule_sha256"],
            policy_intervention_receipt_sha256=row["policy_intervention_receipt_sha256"],
            authorized_index_receipt_sha256=row["authorized_index_receipt_sha256"],
            outputs=tuple(DevelopmentOutputArtifact.from_dict(item) for item in outputs),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class DevelopmentPairedExecutionReceipt:
    config_sha256: str
    materialization_receipt_sha256: str
    selection_receipt_sha256: str
    strata: tuple[DevelopmentStratumReceipt, ...]
    k: int = REGISTERED_K
    permutation_seed: int = DEVELOPMENT_ACTION_PERMUTATION_SEED
    action_set: tuple[str, ...] = REGISTERED_ACTIONS
    action_order_algorithm: str = DEVELOPMENT_ACTION_ORDER_ALGORITHM
    schema_version: str = DEVELOPMENT_PAIRED_EXECUTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "config_sha256",
            "materialization_receipt_sha256",
            "selection_receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.k != REGISTERED_K:
            raise DevelopmentExecutionError("execution receipt k differs")
        if self.permutation_seed != DEVELOPMENT_ACTION_PERMUTATION_SEED:
            raise DevelopmentExecutionError("execution receipt seed differs")
        if tuple(self.action_set) != REGISTERED_ACTIONS:
            raise DevelopmentExecutionError("execution receipt action set differs")
        if self.action_order_algorithm != DEVELOPMENT_ACTION_ORDER_ALGORITHM:
            raise DevelopmentExecutionError("execution receipt order algorithm differs")
        if self.schema_version != DEVELOPMENT_PAIRED_EXECUTION_RECEIPT_SCHEMA:
            raise DevelopmentExecutionError("execution receipt schema differs")
        strata = tuple(sorted(self.strata, key=lambda row: (row.stage, row.corpus)))
        expected = {(stage, corpus) for stage in _DEVELOPMENT_STAGES for corpus in FIXED_CORPORA}
        if {(row.stage, row.corpus) for row in strata} != expected or len(strata) != len(expected):
            raise DevelopmentExecutionError("execution receipt omits a development stratum")
        paths = [artifact.path for row in strata for artifact in row.outputs]
        if len(paths) != len(set(paths)):
            raise DevelopmentExecutionError("execution receipt repeats an output path")
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "action_set", tuple(self.action_set))

    def to_dict(self) -> dict[str, object]:
        return {
            "action_order_algorithm": self.action_order_algorithm,
            "action_set": list(self.action_set),
            "config_sha256": self.config_sha256,
            "k": self.k,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "permutation_seed": self.permutation_seed,
            "schema_version": self.schema_version,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "strata": [row.to_dict() for row in self.strata],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DevelopmentPairedExecutionReceipt:
        row = _closed(value, _RECEIPT_FIELDS, label="development execution receipt")
        strata = row["strata"]
        actions = row["action_set"]
        if not isinstance(strata, list) or not isinstance(actions, list):
            raise DevelopmentExecutionError("execution receipt arrays differ")
        return cls(
            config_sha256=row["config_sha256"],
            materialization_receipt_sha256=row["materialization_receipt_sha256"],
            selection_receipt_sha256=row["selection_receipt_sha256"],
            strata=tuple(DevelopmentStratumReceipt.from_dict(item) for item in strata),
            k=row["k"],
            permutation_seed=row["permutation_seed"],
            action_set=tuple(actions),
            action_order_algorithm=row["action_order_algorithm"],
            schema_version=row["schema_version"],
        )


class DevelopmentStratumExecutor(Protocol):
    def __call__(
        self,
        source: DevelopmentExecutionInput,
        plan: DevelopmentExecutionPlan,
        queries: Mapping[str, str],
        embedding: DevelopmentEmbeddingBinding,
        *,
        controller: ControllerConfig,
        k: int,
        permutation_seed: int,
    ) -> DevelopmentStratumExecution: ...


def _materialized_artifact(
    receipt: DevelopmentCohortMaterializationReceipt,
    relative_path: str,
) -> object:
    matches = [artifact for artifact in receipt.artifacts if artifact.path == relative_path]
    if len(matches) != 1:
        raise DevelopmentExecutionError(
            f"materialization receipt does not bind exactly one {relative_path!r}"
        )
    return matches[0]


def _read_materialized_file(
    root: Path,
    receipt: DevelopmentCohortMaterializationReceipt,
    relative_path: str,
    *,
    label: str,
) -> bytes:
    artifact = _materialized_artifact(receipt, relative_path)
    try:
        byte_count = artifact.byte_count  # type: ignore[attr-defined]
        expected_sha256 = artifact.sha256  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise DevelopmentExecutionError("materialized artifact lacks its byte binding") from exc
    encoded = read_secure_regular_file(
        root.joinpath(*PurePosixPath(relative_path).parts),
        max_bytes=byte_count,
        label=label,
    )
    if len(encoded) != byte_count or _sha256(encoded) != expected_sha256:
        raise DevelopmentExecutionError(f"{label} differs from its materialization receipt")
    return encoded


def _load_materialized_plan(
    root: Path,
    receipt: DevelopmentCohortMaterializationReceipt,
    *,
    corpus: str,
    stage: str,
) -> DevelopmentExecutionPlan:
    relative = f"{stage}/{corpus}/execution-plan.json"
    encoded = _read_materialized_file(
        root,
        receipt,
        relative,
        label=f"development execution plan {stage}:{corpus}",
    )
    plan = DevelopmentExecutionPlan.from_dict(
        _decode(encoded, label=f"development execution plan {stage}:{corpus}")
    )
    if encoded != plan.canonical_file_bytes():
        raise DevelopmentExecutionError("development execution plan is not canonical")
    if plan.corpus != corpus or plan.stage != stage:
        raise DevelopmentExecutionError("development execution plan crosses its stratum")
    return plan


def _load_materialized_queries(
    root: Path,
    receipt: DevelopmentCohortMaterializationReceipt,
    *,
    corpus: str,
    stage: str,
) -> Mapping[str, str]:
    relative = f"{stage}/{corpus}/queries.jsonl"
    encoded = _read_materialized_file(
        root,
        receipt,
        relative,
        label=f"development queries {stage}:{corpus}",
    )
    if not encoded or not encoded.endswith(b"\n"):
        raise DevelopmentExecutionError("development query JSONL needs a terminal newline")
    queries: dict[str, str] = {}
    previous: bytes | None = None
    for line_number, line in enumerate(encoded.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n"):
            raise DevelopmentExecutionError("development query row lacks a newline")
        value = _decode(line[:-1], label=f"development query line {line_number}")
        row = _closed(value, _QUERY_FIELDS, label=f"development query line {line_number}")
        if line != _canonical_bytes(value):
            raise DevelopmentExecutionError("development query JSONL is not canonical")
        query_id = _require_text("development query ID", row["id"])
        text = row["text"]
        if not isinstance(text, str) or unicodedata.normalize("NFC", text) != text:
            raise DevelopmentExecutionError("development query text must be NFC text")
        encoded_id = query_id.encode("utf-8")
        if previous is not None and encoded_id <= previous:
            raise DevelopmentExecutionError("development query IDs are repeated or unordered")
        previous = encoded_id
        queries[query_id] = text
    if not queries:
        raise DevelopmentExecutionError("development query source is empty")
    return MappingProxyType(queries)


def _embedding_binding(
    receipt: DevelopmentCohortMaterializationReceipt,
    *,
    corpus: str,
    stage: str,
) -> DevelopmentEmbeddingBinding:
    matches = [
        row
        for row in receipt.embedding_bindings
        if row.corpus == corpus and row.development_stage == stage
    ]
    if len(matches) != 1:
        raise DevelopmentExecutionError("materialization lacks one embedding binding")
    return matches[0]


def _validate_stratum_execution(
    result: DevelopmentStratumExecution,
    plan: DevelopmentExecutionPlan,
    queries: Mapping[str, str],
    source: DevelopmentExecutionInput,
    embedding: DevelopmentEmbeddingBinding,
    *,
    permutation_seed: int,
) -> None:
    if not isinstance(result, DevelopmentStratumExecution):
        raise DevelopmentExecutionError("stratum executor returned an untyped result")
    if result.execution_order.execution_plan_sha256 != plan.artifact_sha256:
        raise DevelopmentExecutionError("execution order binds another plan")
    if result.embedding_receipt_sha256 != embedding.receipt_sha256:
        raise DevelopmentExecutionError("stratum result binds another embedding store")
    if (
        result.policy_intervention_receipt_sha256 != source.policy_intervention_receipt_sha256
        or result.authorized_index_receipt_sha256 != source.authorized_index_receipt_sha256
    ):
        raise DevelopmentExecutionError("stratum result differs from its input receipts")

    plan_by_key = {trial.trial_key: trial for trial in plan.trials}
    if len(plan_by_key) != len(plan.trials):
        raise DevelopmentExecutionError("development plan repeats a trial key")
    if {trial.query_id for trial in plan.trials} != set(queries):
        raise DevelopmentExecutionError("query text source and execution plan differ")
    order_rows = result.execution_order.rows
    if {row.trial_key for row in order_rows} != set(plan_by_key):
        raise DevelopmentExecutionError("execution order does not cover the exact plan")
    expected_orders = portable_balanced_action_orders(
        permutation_seed=permutation_seed,
        execution_artifact_sha256=plan.artifact_sha256,
        trial_families=tuple((trial.trial_key, trial.family_key) for trial in plan.trials),
    )
    for row in order_rows:
        trial = plan_by_key[row.trial_key]
        if (
            row.family_id != trial.family_key
            or row.query_id != trial.query_id
            or row.actions != expected_orders[row.trial_key]
        ):
            raise DevelopmentExecutionError("execution order differs from the frozen plan")

    grouped: dict[int, list[DevelopmentPairedActionRow]] = defaultdict(list)
    for row in result.action_rows:
        grouped[row.schedule_order].append(row)
    if set(grouped) != set(range(len(plan.trials))):
        raise DevelopmentExecutionError("paired-action rows omit a schedule position")
    orders_by_position = {row.schedule_order: row for row in order_rows}
    for position in range(len(plan.trials)):
        order = orders_by_position[position]
        rows = grouped[position]
        if tuple(row.action for row in rows) != REGISTERED_ACTIONS:
            raise DevelopmentExecutionError("paired-action output lacks the complete action set")
        if any(row.execution_position != order.actions.index(row.action) for row in rows):
            raise DevelopmentExecutionError(
                "paired-action execution positions differ from the order"
            )
        if any(
            row.trial_key != order.trial_key
            or row.family_id != order.family_id
            or row.query_id != order.query_id
            for row in rows
        ):
            raise DevelopmentExecutionError("paired-action identities differ from the order row")
        exact = rows[REGISTERED_ACTIONS.index("exact-authorized")]
        abstain = rows[REGISTERED_ACTIONS.index("abstain")]
        if exact.execution_state != "completed" or abstain.execution_state != "abstained":
            raise DevelopmentExecutionError("development exact or abstain control is incomplete")
        if abstain.failure_state != "registered-abstention":
            raise DevelopmentExecutionError("development abstention reason differs")


def _validate_source_bindings(
    result: DevelopmentStratumExecution,
    plan: DevelopmentExecutionPlan,
    source: DevelopmentExecutionInput,
    embedding: DevelopmentEmbeddingBinding,
) -> None:
    """Bind an executor result back to every source-side canonical receipt."""

    try:
        policy_config = load_policy_intervention_config(
            source.policy_intervention_root / POLICY_CONFIG_FILENAME
        )
        policy_schedule = load_canonical_trial_schedule(
            source.policy_intervention_root / POLICY_SCHEDULE_FILENAME
        )
        policy_receipt = load_policy_intervention_receipt(
            source.policy_intervention_root / POLICY_RECEIPT_FILENAME
        )
        mask_store = CompiledPolicyMaskStore(source.policy_intervention_root / CATALOG_FILENAME)
        index_receipt = load_authorized_index_store_receipt(source.authorized_index_root)
        embedding_receipt = load_embedding_store_receipt(embedding.root)
    except Exception as exc:
        raise DevelopmentExecutionError(
            f"development source receipt admission failed: {exc}"
        ) from exc
    if (
        result.embedding_receipt_sha256 != embedding.receipt_sha256
        or embedding_receipt.receipt_sha256 != embedding.receipt_sha256
        or result.policy_config_sha256 != policy_config.config_sha256
        or result.policy_catalog_sha256 != mask_store.catalog_sha256
        or result.policy_schedule_sha256 != policy_schedule.artifact_sha256
        or result.policy_intervention_receipt_sha256 != policy_receipt.artifact_sha256
        or result.authorized_index_receipt_sha256 != index_receipt.artifact_sha256
        or policy_receipt.artifact_sha256 != source.policy_intervention_receipt_sha256
        or index_receipt.artifact_sha256 != source.authorized_index_receipt_sha256
        or policy_receipt.config_sha256 != policy_config.config_sha256
        or policy_schedule.config_sha256 != policy_config.config_sha256
        or policy_schedule.mask_catalog_sha256 != mask_store.catalog_sha256
        or index_receipt.policy_catalog_sha256 != mask_store.catalog_sha256
        or index_receipt.policy_receipt_sha256 != policy_receipt.artifact_sha256
        or index_receipt.embedding_receipt_sha256 != embedding.receipt_sha256
        or policy_receipt.execution_artifact_sha256 != plan.artifact_sha256
        or policy_schedule.execution_artifact_sha256 != plan.artifact_sha256
        or index_receipt.policy_execution_artifact_sha256 != plan.artifact_sha256
        or policy_schedule.corpus != plan.corpus
        or policy_schedule.stage != plan.stage
        or policy_schedule.document_count != plan.document_count
        or policy_schedule.document_universe_sha256 != plan.document_universe_sha256
    ):
        raise DevelopmentExecutionError(
            "execution result differs from policy, embedding, or index receipts"
        )
    expected_order = tuple(
        (
            row.schedule_order,
            row.trial_key,
            row.family_key,
        )
        for row in policy_schedule.rows
    )
    observed_order = tuple(
        (row.schedule_order, row.trial_key, row.family_id) for row in result.execution_order.rows
    )
    if observed_order != expected_order:
        raise DevelopmentExecutionError(
            "execution order differs from the canonical policy schedule"
        )


def _action_bytes(rows: Sequence[DevelopmentPairedActionRow]) -> bytes:
    return b"".join(row.canonical_line_bytes() for row in rows)


def _output_artifact(
    *,
    path: str,
    role: str,
    encoded: bytes,
    record_count: int,
) -> DevelopmentOutputArtifact:
    return DevelopmentOutputArtifact(
        path=path,
        role=role,
        sha256=_sha256(encoded),
        byte_count=len(encoded),
        record_count=record_count,
    )


def _write_exclusive(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        raise DevelopmentExecutionError(f"cannot create {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_publish(work: Path, output: Path) -> None:
    if os.path.lexists(output):
        raise DevelopmentExecutionError("development execution output already exists")
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(work)
    destination = os.fsencode(output)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise DevelopmentExecutionError("exclusive directory rename is unavailable on macOS")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-2, source, -2, destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise DevelopmentExecutionError("exclusive directory rename is unavailable on Linux")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source, -100, destination, 0x00000001)
    else:
        raise DevelopmentExecutionError(
            f"exclusive directory rename is unsupported on {sys.platform!r}"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise DevelopmentExecutionError("development execution output already exists")
        raise DevelopmentExecutionError(
            f"cannot publish development execution: {os.strerror(error_number)}"
        )
    _fsync_directory(output.parent)


def _expected_tree_entries(receipt: DevelopmentPairedExecutionReceipt) -> set[str]:
    entries = {DEVELOPMENT_CONFIG_FILENAME, DEVELOPMENT_RECEIPT_FILENAME}
    for stratum in receipt.strata:
        for artifact in stratum.outputs:
            path = PurePosixPath(artifact.path)
            entries.add(str(path))
            for position in range(1, len(path.parts)):
                entries.add(str(PurePosixPath(*path.parts[:position])))
    return entries


def _verify_output_tree(
    root: Path,
    receipt: DevelopmentPairedExecutionReceipt,
) -> None:
    try:
        tree = digest_directory_tree(root)
    except ArtifactIntegrityError as exc:
        raise DevelopmentExecutionError(f"cannot verify development output tree: {exc}") from exc
    expected = _expected_tree_entries(receipt)
    if set(tree.entries) != expected:
        raise DevelopmentExecutionError(
            "development output tree differs; "
            f"missing={sorted(expected - set(tree.entries))}, "
            f"extra={sorted(set(tree.entries) - expected)}"
        )


def _read_bound_output(
    root: Path,
    artifact: DevelopmentOutputArtifact,
) -> bytes:
    path = root.joinpath(*PurePosixPath(artifact.path).parts)
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=artifact.byte_count,
            label=f"development output {artifact.role}",
        )
    except ArtifactIntegrityError as exc:
        raise DevelopmentExecutionError(
            f"cannot read development output {artifact.path!r}: {exc}"
        ) from exc
    if len(encoded) != artifact.byte_count or _sha256(encoded) != artifact.sha256:
        raise DevelopmentExecutionError(
            f"development output {artifact.path!r} differs from its receipt"
        )
    return encoded


def _load_action_rows(
    encoded: bytes,
    *,
    expected_count: int,
) -> tuple[DevelopmentPairedActionRow, ...]:
    if not encoded or not encoded.endswith(b"\n"):
        raise DevelopmentExecutionError("paired-action output must be canonical JSONL")
    rows: list[DevelopmentPairedActionRow] = []
    for line_number, line in enumerate(encoded.splitlines(keepends=True), start=1):
        value = _decode(line[:-1], label=f"paired-action line {line_number}")
        row = DevelopmentPairedActionRow.from_dict(value)
        if line != row.canonical_line_bytes():
            raise DevelopmentExecutionError("paired-action output is not canonical")
        rows.append(row)
    if len(rows) != expected_count:
        raise DevelopmentExecutionError("paired-action record count differs")
    return tuple(rows)


def _load_execution_order(encoded: bytes) -> DevelopmentExecutionOrder:
    value = _decode(encoded, label="development execution order")
    order = DevelopmentExecutionOrder.from_dict(value)
    if encoded != order.canonical_file_bytes():
        raise DevelopmentExecutionError("development execution order is not canonical")
    return order


def load_development_paired_execution_receipt(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
) -> DevelopmentPairedExecutionReceipt:
    """Load one exact, canonical paired-development execution receipt."""

    receipt_path = _canonical_absolute_path("execution receipt path", path)
    encoded = read_secure_regular_file(
        receipt_path,
        max_bytes=_MAX_CONTROL_BYTES,
        label="development paired-execution receipt",
    )
    receipt = DevelopmentPairedExecutionReceipt.from_dict(
        _decode(encoded, label="development paired-execution receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise DevelopmentExecutionError("development execution receipt is not canonical")
    if expected_artifact_sha256 is not None and (
        receipt.artifact_sha256
        != _require_sha256("expected execution receipt SHA-256", expected_artifact_sha256)
    ):
        raise DevelopmentExecutionError("development execution receipt digest differs")
    return receipt


def _load_policy_objects(
    source: DevelopmentExecutionInput,
    plan: DevelopmentExecutionPlan,
) -> tuple[
    PolicyInterventionConfig,
    CanonicalTrialSchedule,
    PolicyInterventionReceipt,
    OPACompiledMaskData,
    CompiledPolicyMaskStore,
]:
    try:
        config = load_policy_intervention_config(
            source.policy_intervention_root / POLICY_CONFIG_FILENAME
        )
        verification = verify_policy_intervention_package(
            source.policy_intervention_root,
            plan,
            config,
        )
        schedule = load_canonical_trial_schedule(
            source.policy_intervention_root / POLICY_SCHEDULE_FILENAME
        )
        receipt = load_policy_intervention_receipt(
            source.policy_intervention_root / POLICY_RECEIPT_FILENAME
        )
        opa_data = load_opa_compiled_mask_data(source.policy_intervention_root / OPA_DATA_FILENAME)
        mask_store = CompiledPolicyMaskStore(source.policy_intervention_root / CATALOG_FILENAME)
        mask_store.verify_all()
    except Exception as exc:
        raise DevelopmentExecutionError(
            f"policy package admission failed for {source.stage}:{source.corpus}: {exc}"
        ) from exc
    if (
        verification.receipt_sha256 != source.policy_intervention_receipt_sha256
        or receipt.artifact_sha256 != source.policy_intervention_receipt_sha256
        or verification.catalog_sha256 != mask_store.catalog_sha256
        or verification.schedule_sha256 != schedule.artifact_sha256
        or receipt.execution_artifact_sha256 != plan.artifact_sha256
        or schedule.execution_artifact_sha256 != plan.artifact_sha256
        or schedule.corpus != plan.corpus
        or schedule.stage != plan.stage
        or schedule.document_count != plan.document_count
        or schedule.document_universe_sha256 != plan.document_universe_sha256
        or schedule.mask_catalog_sha256 != mask_store.catalog_sha256
        or opa_data.mask_catalog_sha256 != mask_store.catalog_sha256
        or opa_data.policy_revision != receipt.policy_bundle_revision
    ):
        raise DevelopmentExecutionError(
            "policy config, schedule, catalog, receipt, and development plan are not closed"
        )
    return config, schedule, receipt, opa_data, mask_store


class _PinnedOPADataTransport:
    """In-process evaluation of the exact receipt-bound OPA data value.

    Development execution does not need a network service. The normal OPA
    adapter still validates the request and response contract, while this
    transport performs the single table lookup implemented by the frozen Rego
    rule. Production keeps the independently pinned loopback OPA sidecar.
    """

    def __init__(self, data: OPACompiledMaskData) -> None:
        self._assignments = {(row.subject, row.policy_state): row for row in data.assignments}

    def __call__(self, endpoint: str, body: bytes, timeout: float) -> OPAHTTPResponse:
        if endpoint != "http://127.0.0.1:8181/v1/data/fractal/retrieval/mask":
            raise DevelopmentExecutionError("development OPA endpoint changed")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise DevelopmentExecutionError("development OPA timeout changed")
        payload = _closed(
            _decode(body, label="development OPA request"),
            frozenset({"input"}),
            label="development OPA request",
        )
        policy_input = payload["input"]
        if not isinstance(policy_input, Mapping):
            raise DevelopmentExecutionError("development OPA input must be an object")
        environment = policy_input.get("environment")
        if not isinstance(environment, Mapping) or set(environment) != {
            "assignment_repetition",
            "policy_state",
        }:
            raise DevelopmentExecutionError("development OPA environment differs")
        subject = policy_input.get("subject")
        state = environment.get("policy_state")
        if not isinstance(subject, str) or not isinstance(state, str):
            raise DevelopmentExecutionError("development OPA lookup key differs")
        assignment = self._assignments.get((subject, state))
        if assignment is None:
            return OPAHTTPResponse(status=404, body=b"{}")
        echoed_fields = (
            "action",
            "catalog_request_sha256",
            "document_count",
            "document_universe_sha256",
            "environment_sha256",
            "mask_catalog_sha256",
            "policy_revision",
            "request_nonce",
            "request_sha256",
            "subject",
        )
        try:
            result = {name: policy_input[name] for name in echoed_fields}
        except KeyError as exc:
            raise DevelopmentExecutionError("development OPA request fields differ") from exc
        result.update(assignment.decision_dict())
        decision_id = "development-opa-" + _sha256(body)
        return OPAHTTPResponse(
            status=200,
            body=_canonical_bytes(
                {"decision_id": decision_id, "result": result},
                newline=False,
            ),
        )


def _policy_transitions(
    source: DevelopmentExecutionInput,
    schedule: CanonicalTrialSchedule,
    receipt: PolicyInterventionReceipt,
    mask_store: CompiledPolicyMaskStore,
) -> Mapping[str, PolicyTransitionEvidence]:
    bindings = {row.policy_state: row for row in receipt.transitions}
    transitions: dict[str, PolicyTransitionEvidence] = {}
    for row in schedule.rows:
        if row.environment_sha256 in transitions:
            continue
        try:
            current = mask_store.mask(
                row.mask_id,
                expected_sha256=row.mask_sha256,
                expected_authorized_count=row.authorized_count,
            )
            evidence = derive_policy_transition_evidence(
                source.policy_intervention_root,
                row,
                document_count=schedule.document_count,
                current_mask=current,
            )
        except Exception as exc:
            raise DevelopmentExecutionError(
                f"cannot derive development policy transition: {exc}"
            ) from exc
        binding = bindings.get(row.policy_state)
        if binding is None or (
            binding.baseline_policy_revision != evidence.baseline_policy_revision
            or binding.current_policy_revision != evidence.current_policy_revision
            or binding.baseline_mask_sha256 != evidence.baseline_mask_sha256
            or binding.current_mask_sha256 != evidence.current_mask_sha256
            or binding.baseline_authorized_count != evidence.baseline_authorized_count
            or binding.current_authorized_count != evidence.current_authorized_count
            or binding.policy_churn != evidence.policy_churn
        ):
            raise DevelopmentExecutionError(
                "policy transition evidence differs from its intervention receipt"
            )
        transitions[evidence.environment_sha256] = evidence
    expected = {row.environment_sha256 for row in schedule.rows}
    if set(transitions) != expected:
        raise DevelopmentExecutionError("policy transitions do not cover the schedule")
    return MappingProxyType(transitions)


@contextmanager
def _open_query_epochs(
    root: Path,
    receipt: EmbeddingStoreReceipt,
) -> Iterator[tuple[np.memmap, np.memmap]]:
    arrays: list[np.memmap] = []
    signatures: list[tuple[Path, tuple[int, int, int, int]]] = []
    try:
        for name in ("old_queries", "current_queries"):
            descriptor = receipt.vectors[name]
            path = root / descriptor.relative_path
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != descriptor.byte_count
            ):
                raise DevelopmentExecutionError("query epoch is linked or non-regular")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                not isinstance(value, np.memmap)
                or value.shape != descriptor.shape
                or value.dtype != np.dtype(np.float32)
                or value.flags.writeable
                or not value.flags.c_contiguous
            ):
                raise DevelopmentExecutionError("query epoch geometry differs")
            arrays.append(value)
            signatures.append(
                (
                    path,
                    (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns),
                )
            )
        yield arrays[0], arrays[1]
        for path, before in signatures:
            metadata = path.lstat()
            after = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            if after != before or metadata.st_nlink != 1:
                raise DevelopmentExecutionError("query epoch changed during execution")
    finally:
        for value in arrays:
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()


class _ForcedDevelopmentController:
    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self._delegate = RuleController(config)
        self.action: str | None = None

    def decide(
        self,
        features: object,
        *,
        n_authorized: int,
        policy_version: str,
        policy_available: bool = True,
        expected_policy_version: str | None = None,
    ) -> ControllerDecision:
        if self.action not in REGISTERED_ACTIONS:
            raise DevelopmentExecutionError("development action was not assigned")
        selected = self._delegate.decide(
            features,  # type: ignore[arg-type]
            n_authorized=n_authorized,
            policy_version=policy_version,
            policy_available=policy_available,
            expected_policy_version=expected_policy_version,
        )
        return ControllerDecision(
            action=self.action,  # type: ignore[arg-type]
            risk_score=selected.risk_score,
            reasons=("registered development paired counterfactual",),
            policy_version=selected.policy_version,
        )


def _development_feature_values(
    *,
    geometry: object,
    authorized_count: int,
    document_count: int,
    dimension: int,
    corpus: str,
    backend: str,
    group_order: int,
) -> Mapping[str, object]:
    try:
        work = geometry.distance_evaluations  # type: ignore[attr-defined]
        if work is None:
            work = geometry.visited_candidates  # type: ignore[attr-defined]
        values: dict[str, object] = {
            "corpus_size": float(document_count),
            "authorized_universe_size": float(authorized_count),
            "embedding_dimension": float(dimension),
            "version_lag": DEVELOPMENT_VERSION_LAG,
            "drift_severity": float(geometry.embedding_drift),  # type: ignore[attr-defined]
            "probe_latency_ms": float(geometry.search_latency_ms),  # type: ignore[attr-defined]
            "probe_work": None if work is None else float(work),
            "corpus_stratum": corpus,
            "backend": backend,
            "drift_family": DEVELOPMENT_DRIFT_FAMILY,
            "allow_rate": (0.25, 0.50, 0.75)[group_order],
            "policy_complexity": DEVELOPMENT_POLICY_COMPLEXITY,
            "policy_churn": float(geometry.policy_churn),  # type: ignore[attr-defined]
            "lid_k50": float(geometry.lid),  # type: ignore[attr-defined]
            "lid_cv": float(geometry.lid_scale_instability),  # type: ignore[attr-defined]
            "relative_contrast": float(geometry.relative_contrast),  # type: ignore[attr-defined]
            "radius_expansion": float(geometry.radius_expansion),  # type: ignore[attr-defined]
        }
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        raise DevelopmentExecutionError("development geometry is incomplete") from exc
    for name in REGISTERED_FEATURE_SCHEMA.geometry_numeric:
        value = values[name]
        if isinstance(value, float) and not math.isfinite(value):
            values[name] = None
    if tuple(values) != REGISTERED_FEATURE_SCHEMA.input_features:
        raise DevelopmentExecutionError("development feature order differs")
    frozen = _freeze_feature_values(values)
    if frozen is None:  # pragma: no cover - mapping input cannot produce None
        raise DevelopmentExecutionError("development feature row is absent")
    return frozen


def _default_stratum_executor(
    source: DevelopmentExecutionInput,
    plan: DevelopmentExecutionPlan,
    queries: Mapping[str, str],
    embedding: DevelopmentEmbeddingBinding,
    *,
    controller: ControllerConfig,
    k: int,
    permutation_seed: int,
) -> DevelopmentStratumExecution:
    if (
        source.corpus != plan.corpus
        or source.stage != plan.stage
        or embedding.corpus != plan.corpus
        or embedding.development_stage != plan.stage
        or embedding.receipt_sha256 != plan.embedding_receipt_sha256
    ):
        raise DevelopmentExecutionError("development executor inputs cross a stratum")
    try:
        embedding_receipt = verify_embedding_store(embedding.root)
    except Exception as exc:
        raise DevelopmentExecutionError(f"embedding store admission failed: {exc}") from exc
    if (
        embedding_receipt.receipt_sha256 != embedding.receipt_sha256
        or embedding_receipt.old_model is None
        or set(embedding_receipt.vectors)
        != {"old_documents", "current_documents", "old_queries", "current_queries"}
        or embedding_receipt.document_count != plan.document_count
        or embedding_receipt.row_orders["documents"].file_sha256 != plan.document_row_order_sha256
        or embedding_receipt.row_orders["queries"].file_sha256 != plan.query_row_order_sha256
    ):
        raise DevelopmentExecutionError("embedding store differs from the development plan")

    policy_config, schedule, policy_receipt, opa_data, mask_store = _load_policy_objects(
        source, plan
    )
    try:
        index_receipt = load_authorized_index_store_receipt(source.authorized_index_root)
    except Exception as exc:
        raise DevelopmentExecutionError(f"authorized index receipt failed: {exc}") from exc
    if (
        index_receipt.artifact_sha256 != source.authorized_index_receipt_sha256
        or index_receipt.embedding_receipt_sha256 != embedding.receipt_sha256
        or index_receipt.policy_receipt_sha256 != source.policy_intervention_receipt_sha256
        or index_receipt.policy_catalog_sha256 != mask_store.catalog_sha256
        or index_receipt.policy_execution_artifact_sha256 != plan.artifact_sha256
        or index_receipt.policy_revision != policy_receipt.policy_bundle_revision
        or index_receipt.document_count != plan.document_count
        or index_receipt.document_universe_sha256 != plan.document_universe_sha256
        or index_receipt.document_row_order_sha256 != plan.document_row_order_sha256
    ):
        raise DevelopmentExecutionError("authorized index receipt differs from its source plan")
    try:
        provider = VerifiedAuthorizedIndexProvider(
            source.authorized_index_root,
            embedding_store_root=embedding.root,
            policy_intervention_root=source.policy_intervention_root,
            expected_embedding_receipt_sha256=embedding.receipt_sha256,
            expected_policy_receipt_sha256=source.policy_intervention_receipt_sha256,
            expected_store_receipt_sha256=source.authorized_index_receipt_sha256,
            backend=HnswlibBackend(),
        )
    except Exception as exc:
        raise DevelopmentExecutionError(f"authorized index admission failed: {exc}") from exc
    transitions = _policy_transitions(source, schedule, policy_receipt, mask_store)
    policy = OpenPolicyAgentMaskDecisionPoint(
        "http://127.0.0.1:8181/v1/data/fractal/retrieval/mask",
        mask_store,
        transport=_PinnedOPADataTransport(opa_data),
    )
    subjects = {row.subject for row in schedule.rows}
    if len(subjects) != 1:
        raise DevelopmentExecutionError("development schedule must use one subject")
    subject = subjects.pop()
    trials = {row.trial_key: row for row in plan.trials}
    if {row.trial_key for row in schedule.rows} != set(trials):
        raise DevelopmentExecutionError("policy schedule and development plan differ")
    action_orders = portable_balanced_action_orders(
        permutation_seed=permutation_seed,
        execution_artifact_sha256=plan.artifact_sha256,
        trial_families=tuple((row.trial_key, row.family_key) for row in plan.trials),
    )

    forced = _ForcedDevelopmentController(controller)
    action_rows: list[DevelopmentPairedActionRow] = []
    order_rows: list[DevelopmentExecutionOrderRow] = []
    with (
        open_verified_document_matrices(
            embedding.root,
            index_receipt=index_receipt,
            expected_embedding_receipt_sha256=embedding.receipt_sha256,
        ) as documents,
        _open_query_epochs(embedding.root, embedding_receipt) as query_epochs,
    ):
        retriever = GovernedRetriever(
            documents.old_active,
            policy,
            subject,
            expected_document_universe_sha256=plan.document_universe_sha256,
            exact_truth_vectors=documents.current_truth,
            metric=provider.retrieval_metric,
            controller=forced,  # type: ignore[arg-type]
            policy_transitions=transitions,
            require_policy_transition=True,
            trusted_readonly_vectors=True,
            authorized_hnsw_provider=provider,
        )
        prepared_environments: set[str] = set()
        for schedule_row in schedule.rows:
            if schedule_row.environment_sha256 in prepared_environments:
                continue
            decision = retriever.prepare_authorization(
                action="retrieve",
                environment=dict(schedule_row.environment),
                expected_policy_version=schedule.policy_bundle_revision,
            )
            if (
                decision.authorized_count != schedule_row.authorized_count
                or packed_policy_mask_sha256(decision.authorized_mask) != schedule_row.mask_sha256
            ):
                raise DevelopmentExecutionError(
                    "prepared authorization differs from the development schedule"
                )
            prepared_environments.add(schedule_row.environment_sha256)
        retriever.seal_authorized_index_cache()

        old_queries, current_queries = query_epochs
        for schedule_row in schedule.rows:
            trial = trials[schedule_row.trial_key]
            if trial.query_id not in queries or not 0 <= trial.query_row < len(old_queries):
                raise DevelopmentExecutionError("development query binding is incomplete")
            action_order = action_orders[trial.trial_key]
            order_rows.append(
                DevelopmentExecutionOrderRow(
                    schedule_order=schedule_row.schedule_order,
                    trial_key=trial.trial_key,
                    family_id=trial.family_key,
                    query_id=trial.query_id,
                    actions=action_order,
                )
            )
            emitted: dict[str, DevelopmentPairedActionRow] = {}
            for execution_position, action in enumerate(action_order):
                forced.action = action
                result = retriever.query(
                    old_queries[trial.query_row],
                    current_truth_query=current_queries[trial.query_row],
                    k=k,
                    expected_policy_version=schedule.policy_bundle_revision,
                    action="retrieve",
                    environment=dict(schedule_row.environment),
                )
                if result.geometry is None:
                    raise DevelopmentExecutionError(
                        "development action ended before registered features were measured"
                    )
                features = (
                    _development_feature_values(
                        geometry=result.geometry,
                        authorized_count=schedule_row.authorized_count,
                        document_count=plan.document_count,
                        dimension=documents.old_active.shape[1],
                        corpus=plan.corpus,
                        backend=index_receipt.backend_id,
                        group_order=schedule_row.group_order,
                    )
                    if action == "hnsw-low"
                    else None
                )
                latency = result.total_online_latency_ms
                if not math.isfinite(latency) or latency <= 0.0:
                    raise DevelopmentExecutionError("development action latency is not positive")
                if action == "abstain":
                    if result.decision.action != "abstain" or result.search is not None:
                        raise DevelopmentExecutionError("registered abstention did not abstain")
                    state = "abstained"
                    failure = "registered-abstention"
                    returned: tuple[int, ...] = ()
                elif result.decision.action != action or result.search is None:
                    state = "failed"
                    failure = "action-did-not-complete"
                    returned = ()
                else:
                    state = "completed"
                    failure = None
                    returned = tuple(int(value) for value in result.search.ids.tolist())
                authorization = result.final_authorization or result.initial_authorization
                violations = 0
                if authorization is not None:
                    violations = sum(
                        not bool(authorization.authorized_mask[row]) for row in returned
                    )
                if result.search is not None:
                    violations += (
                        result.search.unauthorized_candidates + result.search.unauthorized_context
                    )
                if violations:
                    raise DevelopmentExecutionError(
                        "development retrieval crossed the authorization boundary"
                    )
                emitted[action] = DevelopmentPairedActionRow(
                    schedule_order=schedule_row.schedule_order,
                    trial_key=trial.trial_key,
                    family_id=trial.family_key,
                    query_id=trial.query_id,
                    action=action,
                    execution_position=execution_position,
                    execution_state=state,
                    failure_state=failure,
                    request_latency_ms=latency,
                    entitlement_violations=violations,
                    returned_document_rows=returned,
                    feature_values=features,
                )
            action_rows.extend(emitted[action] for action in REGISTERED_ACTIONS)

    return DevelopmentStratumExecution(
        action_rows=tuple(action_rows),
        execution_order=DevelopmentExecutionOrder(
            execution_plan_sha256=plan.artifact_sha256,
            permutation_seed=permutation_seed,
            rows=tuple(order_rows),
        ),
        embedding_receipt_sha256=embedding.receipt_sha256,
        policy_config_sha256=policy_config.config_sha256,
        policy_catalog_sha256=mask_store.catalog_sha256,
        policy_schedule_sha256=schedule.artifact_sha256,
        policy_intervention_receipt_sha256=policy_receipt.artifact_sha256,
        authorized_index_receipt_sha256=index_receipt.artifact_sha256,
    )


def _execute_to_work_tree(
    config: DevelopmentPairedExecutionConfig,
    work: Path,
    *,
    executor: DevelopmentStratumExecutor,
) -> DevelopmentPairedExecutionReceipt:
    try:
        materialization = verify_materialized_development_cohort(
            config.materialization_root,
            expected_receipt_sha256=config.materialization_receipt_sha256,
            verify_label_payloads=False,
        )
    except Exception as exc:
        raise DevelopmentExecutionError(
            f"label-free development materialization admission failed: {exc}"
        ) from exc
    if materialization.artifact_sha256 != config.materialization_receipt_sha256:
        raise DevelopmentExecutionError("development materialization receipt changed")

    _write_exclusive(work / DEVELOPMENT_CONFIG_FILENAME, config.canonical_file_bytes())
    strata: list[DevelopmentStratumReceipt] = []
    for source in config.inputs:
        plan = _load_materialized_plan(
            config.materialization_root,
            materialization,
            corpus=source.corpus,
            stage=str(source.stage),
        )
        queries = _load_materialized_queries(
            config.materialization_root,
            materialization,
            corpus=source.corpus,
            stage=str(source.stage),
        )
        embedding = _embedding_binding(
            materialization,
            corpus=source.corpus,
            stage=str(source.stage),
        )
        result = executor(
            source,
            plan,
            queries,
            embedding,
            controller=config.controller,
            k=config.k,
            permutation_seed=config.permutation_seed,
        )
        _validate_stratum_execution(
            result,
            plan,
            queries,
            source,
            embedding,
            permutation_seed=config.permutation_seed,
        )
        _validate_source_bindings(result, plan, source, embedding)
        prefix = f"{source.stage}/{source.corpus}"
        action_path = f"{prefix}/{DEVELOPMENT_ACTION_FILENAME}"
        order_path = f"{prefix}/{DEVELOPMENT_ORDER_FILENAME}"
        action_bytes = _action_bytes(result.action_rows)
        order_bytes = result.execution_order.canonical_file_bytes()
        _write_exclusive(work.joinpath(*PurePosixPath(action_path).parts), action_bytes)
        _write_exclusive(work.joinpath(*PurePosixPath(order_path).parts), order_bytes)
        strata.append(
            DevelopmentStratumReceipt(
                corpus=source.corpus,
                stage=str(source.stage),
                execution_plan_sha256=plan.artifact_sha256,
                selected_family_count=plan.selected_family_count,
                trial_count=len(plan.trials),
                embedding_receipt_sha256=result.embedding_receipt_sha256,
                policy_config_sha256=result.policy_config_sha256,
                policy_catalog_sha256=result.policy_catalog_sha256,
                policy_schedule_sha256=result.policy_schedule_sha256,
                policy_intervention_receipt_sha256=(result.policy_intervention_receipt_sha256),
                authorized_index_receipt_sha256=(result.authorized_index_receipt_sha256),
                outputs=(
                    _output_artifact(
                        path=action_path,
                        role="paired-actions",
                        encoded=action_bytes,
                        record_count=len(result.action_rows),
                    ),
                    _output_artifact(
                        path=order_path,
                        role="execution-order",
                        encoded=order_bytes,
                        record_count=len(result.execution_order.rows),
                    ),
                ),
            )
        )
    receipt = DevelopmentPairedExecutionReceipt(
        config_sha256=config.config_sha256,
        materialization_receipt_sha256=materialization.artifact_sha256,
        selection_receipt_sha256=materialization.selection_receipt_sha256,
        strata=tuple(strata),
    )
    _write_exclusive(work / DEVELOPMENT_RECEIPT_FILENAME, receipt.canonical_file_bytes())
    _verify_output_tree(work, receipt)
    return receipt


def run_development_paired_execution(
    config_path: str | Path,
    *,
    executor: DevelopmentStratumExecutor | None = None,
) -> DevelopmentPairedExecutionReceipt:
    """Run and exclusively publish all ten frozen development strata."""

    config = load_development_paired_execution_config(config_path)
    if os.path.lexists(config.output_root):
        raise DevelopmentExecutionError("development execution output already exists")
    config.output_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if stat.S_IMODE(config.output_root.parent.stat().st_mode) & 0o022:
        raise DevelopmentExecutionError("development output parent is group- or world-writable")
    work = Path(
        tempfile.mkdtemp(
            prefix=f".{config.output_root.name}.work-",
            dir=config.output_root.parent,
        )
    )
    try:
        receipt = _execute_to_work_tree(
            config,
            work,
            executor=_default_stratum_executor if executor is None else executor,
        )
        _exclusive_publish(work, config.output_root)
        work = Path()
        verified = verify_development_paired_execution(
            config.output_root,
            expected_receipt_sha256=receipt.artifact_sha256,
        )
        if verified != receipt:
            raise DevelopmentExecutionError("published development execution differs")
        return receipt
    finally:
        if work != Path() and work.exists():
            shutil.rmtree(work)


def verify_development_paired_execution(
    root: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
) -> DevelopmentPairedExecutionReceipt:
    """Verify closed membership, source bindings, and every paired-action row."""

    package = _canonical_absolute_path("development execution root", root)
    _reject_forbidden_path("development execution root", package)
    receipt = load_development_paired_execution_receipt(
        package / DEVELOPMENT_RECEIPT_FILENAME,
        expected_artifact_sha256=expected_receipt_sha256,
    )
    config = load_development_paired_execution_config(package / DEVELOPMENT_CONFIG_FILENAME)
    if config.output_root != package or config.config_sha256 != receipt.config_sha256:
        raise DevelopmentExecutionError("execution config and published root differ")
    try:
        materialization = verify_materialized_development_cohort(
            config.materialization_root,
            expected_receipt_sha256=receipt.materialization_receipt_sha256,
            verify_label_payloads=False,
        )
    except Exception as exc:
        raise DevelopmentExecutionError(
            f"cannot reverify label-free materialization: {exc}"
        ) from exc
    if (
        materialization.artifact_sha256 != config.materialization_receipt_sha256
        or materialization.selection_receipt_sha256 != receipt.selection_receipt_sha256
    ):
        raise DevelopmentExecutionError("execution receipt names another materialization")
    _verify_output_tree(package, receipt)
    inputs = {(row.stage, row.corpus): row for row in config.inputs}
    for stratum in receipt.strata:
        source = inputs[(stratum.stage, stratum.corpus)]
        plan = _load_materialized_plan(
            config.materialization_root,
            materialization,
            corpus=stratum.corpus,
            stage=stratum.stage,
        )
        queries = _load_materialized_queries(
            config.materialization_root,
            materialization,
            corpus=stratum.corpus,
            stage=stratum.stage,
        )
        embedding = _embedding_binding(
            materialization,
            corpus=stratum.corpus,
            stage=stratum.stage,
        )
        outputs = {row.role: row for row in stratum.outputs}
        action_rows = _load_action_rows(
            _read_bound_output(package, outputs["paired-actions"]),
            expected_count=outputs["paired-actions"].record_count,
        )
        order = _load_execution_order(_read_bound_output(package, outputs["execution-order"]))
        if outputs["execution-order"].record_count != len(order.rows):
            raise DevelopmentExecutionError("execution-order record count differs")
        result = DevelopmentStratumExecution(
            action_rows=action_rows,
            execution_order=order,
            embedding_receipt_sha256=stratum.embedding_receipt_sha256,
            policy_config_sha256=stratum.policy_config_sha256,
            policy_catalog_sha256=stratum.policy_catalog_sha256,
            policy_schedule_sha256=stratum.policy_schedule_sha256,
            policy_intervention_receipt_sha256=(stratum.policy_intervention_receipt_sha256),
            authorized_index_receipt_sha256=(stratum.authorized_index_receipt_sha256),
        )
        if (
            plan.artifact_sha256 != stratum.execution_plan_sha256
            or plan.selected_family_count != stratum.selected_family_count
            or len(plan.trials) != stratum.trial_count
        ):
            raise DevelopmentExecutionError("stratum receipt differs from its plan")
        _validate_stratum_execution(
            result,
            plan,
            queries,
            source,
            embedding,
            permutation_seed=config.permutation_seed,
        )
        _validate_source_bindings(result, plan, source, embedding)
    return receipt


def _materialized_pin(
    materialization_root: Path,
    receipt: DevelopmentCohortMaterializationReceipt,
    *,
    corpus: str,
    stage: str,
    role: str,
) -> PinnedDevelopmentFile:
    matches = [
        row
        for row in receipt.artifacts
        if row.corpus == corpus and row.stage == stage and row.role == role
    ]
    if len(matches) != 1:
        raise DevelopmentExecutionError(
            f"materialization does not bind one {stage}:{corpus}:{role} artifact"
        )
    artifact = matches[0]
    return PinnedDevelopmentFile(
        path=materialization_root.joinpath(*PurePosixPath(artifact.path).parts),
        sha256=artifact.sha256,
        byte_count=artifact.byte_count,
        corpus_id=corpus,
        stage=stage,
        role=role,
    )


def build_development_freeze_config(
    execution_root: str | Path,
    *,
    output_root: str | Path,
) -> DevelopmentFreezeConfig:
    """Build the only development-freeze handoff admitted by this execution."""

    package = _canonical_absolute_path("development execution root", execution_root)
    receipt = verify_development_paired_execution(package)
    config = load_development_paired_execution_config(package / DEVELOPMENT_CONFIG_FILENAME)
    materialization = verify_materialized_development_cohort(
        config.materialization_root,
        expected_receipt_sha256=receipt.materialization_receipt_sha256,
        verify_label_payloads=False,
    )
    selection_matches = [
        row for row in materialization.artifacts if row.role == "development-cohort-selection"
    ]
    if len(selection_matches) != 1:
        raise DevelopmentExecutionError("materialization selection pin differs")
    selection = selection_matches[0]
    if selection.sha256 != receipt.selection_receipt_sha256:
        raise DevelopmentExecutionError("execution and materialization selections differ")
    input_by_key = {(row.stage, row.corpus): row for row in config.inputs}
    stratum_by_key = {(row.stage, row.corpus): row for row in receipt.strata}
    embeddings = {
        (row.development_stage, row.corpus): row for row in materialization.embedding_bindings
    }
    sources: list[DevelopmentCorpusSources] = []
    for stage in _DEVELOPMENT_STAGES:
        for corpus in FIXED_CORPORA:
            key = (stage, corpus)
            source = input_by_key[key]
            stratum = stratum_by_key[key]
            paired = next(row for row in stratum.outputs if row.role == "paired-actions")
            schedule_path = source.policy_intervention_root / POLICY_SCHEDULE_FILENAME
            schedule_bytes = read_secure_regular_file(
                schedule_path,
                max_bytes=_MAX_CONTROL_BYTES,
                label=f"development policy schedule {stage}:{corpus}",
            )
            if _sha256(schedule_bytes) != stratum.policy_schedule_sha256:
                raise DevelopmentExecutionError("freeze handoff schedule changed")
            embedding = embeddings[key]
            evidence = (
                _materialized_pin(
                    config.materialization_root,
                    materialization,
                    corpus=corpus,
                    stage=stage,
                    role="evidence-bundles",
                )
                if corpus in EVIDENCE_CORPORA
                else None
            )
            sources.append(
                DevelopmentCorpusSources(
                    corpus_id=corpus,
                    stage=stage,
                    queries=_materialized_pin(
                        config.materialization_root,
                        materialization,
                        corpus=corpus,
                        stage=stage,
                        role="queries",
                    ),
                    qrels=_materialized_pin(
                        config.materialization_root,
                        materialization,
                        corpus=corpus,
                        stage=stage,
                        role="qrels",
                    ),
                    evidence_bundles=evidence,
                    policy_schedule=PinnedDevelopmentFile(
                        path=schedule_path,
                        sha256=stratum.policy_schedule_sha256,
                        byte_count=len(schedule_bytes),
                        corpus_id=corpus,
                        stage=stage,
                        role="policy-schedule",
                    ),
                    paired_actions=PinnedDevelopmentFile(
                        path=package.joinpath(*PurePosixPath(paired.path).parts),
                        sha256=paired.sha256,
                        byte_count=paired.byte_count,
                        corpus_id=corpus,
                        stage=stage,
                        role="paired-actions",
                    ),
                    embedding_store=PinnedEmbeddingStore(
                        root=embedding.root,
                        receipt_sha256=embedding.receipt_sha256,
                        corpus_id=corpus,
                        stage=stage,
                    ),
                )
            )
    return DevelopmentFreezeConfig(
        sources=tuple(sources),
        selection_receipt=PinnedDevelopmentSelectionReceipt(
            path=config.materialization_root.joinpath(*PurePosixPath(selection.path).parts),
            sha256=selection.sha256,
            byte_count=selection.byte_count,
        ),
        output_root=_canonical_absolute_path("development freeze output", output_root),
    )


def write_bound_development_freeze_config(
    execution_root: str | Path,
    config_output: str | Path,
    *,
    output_root: str | Path,
) -> DevelopmentFreezeConfig:
    """Exclusively write the label-authorized compiler handoff."""

    target = _canonical_absolute_path("development freeze config output", config_output)
    config = build_development_freeze_config(
        execution_root,
        output_root=output_root,
    )
    _write_exclusive(target, canonical_development_freeze_config_bytes(config))
    return config


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-development-execution",
        description="Run or verify the outcome-blind paired development cohort.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", required=True)
    verify.add_argument("--receipt-sha256")
    freeze = commands.add_parser("write-freeze-config")
    freeze.add_argument("--execution-root", required=True)
    freeze.add_argument("--config-output", required=True)
    freeze.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    if args.command == "run":
        receipt = run_development_paired_execution(args.config)
        result = {
            "command": "run",
            "receipt_sha256": receipt.artifact_sha256,
            "schema_version": DEVELOPMENT_EXECUTION_CLI_RESULT_SCHEMA,
        }
    elif args.command == "verify":
        receipt = verify_development_paired_execution(
            args.root,
            expected_receipt_sha256=args.receipt_sha256,
        )
        result = {
            "command": "verify",
            "receipt_sha256": receipt.artifact_sha256,
            "schema_version": DEVELOPMENT_EXECUTION_CLI_RESULT_SCHEMA,
        }
    else:
        config = write_bound_development_freeze_config(
            args.execution_root,
            args.config_output,
            output_root=args.output_root,
        )
        result = {
            "command": "write-freeze-config",
            "config_sha256": _sha256(canonical_development_freeze_config_bytes(config)),
            "schema_version": DEVELOPMENT_EXECUTION_CLI_RESULT_SCHEMA,
        }
    sys.stdout.buffer.write(_canonical_bytes(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
