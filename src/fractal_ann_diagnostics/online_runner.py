"""Label-free paired-action execution over one admitted online artifact.

The runner produces a complete four-action panel, a linked governed-audit
chain, and detached evidence of the actual seeded action order. It consumes no
outcome data and has no path for opening custody artifacts.

This is an execution scaffold. Acquisition, embedding, environment isolation,
artifact writes, external anchoring, and the one-shot production invocation
remain responsibilities of the pinned C0 runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np

from .action_panel_admission import (
    REGISTERED_FAILURE_CODES,
    AdmittedActionPanel,
    FailedActionExecution,
    FailureCode,
    GovernedActionExecution,
    action_panel_from_governed_executions,
)
from .artifact_integrity import read_secure_control_file, write_exclusive_receipt_bytes
from .audit import (
    AdmittedProvenanceRegistry,
    AuditRecord,
    audit_record_from_governed_result,
    verify_audit_chain,
)
from .confirmatory_analysis import ConfirmatoryAnalysisError
from .confirmatory_modeling import REGISTERED_FEATURE_SCHEMA
from .controller import (
    ControllerDecision,
    GovernedRetriever,
    RuleController,
)
from .geometry import QueryGeometry
from .policy import (
    PolicyDecision,
    PolicyDecisionPoint,
    policy_document_universe_sha256,
    policy_environment_sha256,
)
from .scalable_execution import execution_artifact_sha256, execution_document_count
from .study import SealedRunReceipt

REGISTERED_ACTION_SET = (
    "hnsw-low",
    "hnsw-high",
    "exact-authorized",
    "abstain",
)
CACHE_PREPARATION_ROW_SCHEMA = "fractal-online-cache-preparation-row-v1"
CACHE_PREPARATION_RECEIPT_SCHEMA = "fractal-online-cache-preparation-v1"
CACHE_PREPARATION_ALGORITHM = "preload-all-trial-authorizations-fail-closed-v1"
EXECUTION_ORDER_ROW_SCHEMA = "fractal-online-execution-order-row-v3"
EXECUTION_ORDER_RECEIPT_SCHEMA = "fractal-online-execution-order-v4"
PERMUTATION_ALGORITHM = "sha256-ranked-family-latin-square-v1"
POSITION_BALANCE_UNITS = ("corpus", "query-family")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ORDER_ROW_FIELDS = {
    "active_query_vector_sha256",
    "actions",
    "current_truth_query_vector_sha256",
    "environment_sha256",
    "family_key",
    "query_binding_sha256",
    "query_text_sha256",
    "schema_version",
    "trial_key",
}
_ORDER_RECEIPT_FIELDS = {
    "action_set",
    "position_balance_units",
    "cache_preparation_receipt_sha256",
    "execution_artifact_sha256",
    "permutation_algorithm",
    "permutation_seed",
    "rows",
    "run_receipt_sha256",
    "schema_version",
}
_CACHE_PREPARATION_ROW_FIELDS = {
    "authorized_count",
    "environment_sha256",
    "mask_sha256",
    "policy_version",
    "schema_version",
}
_CACHE_PREPARATION_RECEIPT_FIELDS = {
    "algorithm",
    "document_universe_sha256",
    "execution_artifact_sha256",
    "expected_policy_version",
    "policy_action",
    "role",
    "rows",
    "run_receipt_sha256",
    "schema_version",
}


class OnlineRunnerError(RuntimeError):
    """Raised when an online matrix cannot be admitted without ambiguity."""


class _TrialView(Protocol):
    trial_key: str
    family_key: str
    text: str
    corpus: str
    stage: str


def _canonical_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OnlineRunnerError("runner evidence must be finite canonical JSON") from exc


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OnlineRunnerError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise OnlineRunnerError(f"{name} must be a canonical non-empty string")
    return value


def _closed_mapping(
    payload: object,
    *,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise OnlineRunnerError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in payload):
        raise OnlineRunnerError(f"{name} keys must be strings")
    observed = set(payload)
    if observed != fields:
        raise OnlineRunnerError(
            f"{name} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return payload


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise OnlineRunnerError("permutation_seed must be an unsigned 64-bit integer")
    return seed


def _permutation_score(
    *,
    permutation_seed: int,
    execution_artifact_sha256: str,
    role: str,
    value: str,
) -> bytes:
    seed = _validate_seed(permutation_seed)
    execution_digest = _require_sha256("execution_artifact_sha256", execution_artifact_sha256)
    _require_text("permutation role", role)
    _require_text("permutation value", value)
    return hashlib.sha256(
        _canonical_bytes(
            {
                "algorithm": PERMUTATION_ALGORITHM,
                "execution_artifact_sha256": execution_digest,
                "permutation_seed": seed,
                "role": role,
                "value": value,
            }
        )
    ).digest()


def portable_balanced_action_orders(
    *,
    permutation_seed: int,
    execution_artifact_sha256: str,
    trial_families: Sequence[tuple[str, str]],
) -> Mapping[str, tuple[str, ...]]:
    """Return a portable Latin schedule balanced by corpus and query family.

    Families and their trials are SHA-256 ranked. The resulting contiguous
    trial stream receives successive cyclic rows of one SHA-256-ranked Latin
    square. Every action therefore occupies every execution position either
    ``floor(n/4)`` or ``ceil(n/4)`` times both over the complete corpus and
    inside each query-family block.
    """

    seed = _validate_seed(permutation_seed)
    execution_digest = _require_sha256("execution_artifact_sha256", execution_artifact_sha256)
    pairs = tuple(trial_families)
    if not pairs:
        raise OnlineRunnerError("trial_families must not be empty")
    grouped: dict[str, list[str]] = {}
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise OnlineRunnerError("trial_families must contain trial/family pairs")
        trial_key, family_key = pair
        _require_sha256("trial_key", trial_key)
        _require_sha256("family_key", family_key)
        grouped.setdefault(family_key, []).append(trial_key)
    trial_keys = [trial_key for trial_key, _ in pairs]
    if len(trial_keys) != len(set(trial_keys)):
        raise OnlineRunnerError("trial_families contains duplicate trials")

    base = tuple(
        action
        for _, action in sorted(
            (
                _permutation_score(
                    permutation_seed=seed,
                    execution_artifact_sha256=execution_digest,
                    role="base-action",
                    value=action,
                ),
                action,
            )
            for action in REGISTERED_ACTION_SET
        )
    )
    family_order = sorted(
        grouped,
        key=lambda family_key: (
            _permutation_score(
                permutation_seed=seed,
                execution_artifact_sha256=execution_digest,
                role="family",
                value=family_key,
            ),
            family_key,
        ),
    )
    schedule: dict[str, tuple[str, ...]] = {}
    stream_position = 0
    for family_key in family_order:
        family_trials = sorted(
            grouped[family_key],
            key=lambda trial_key: (
                _permutation_score(
                    permutation_seed=seed,
                    execution_artifact_sha256=execution_digest,
                    role=f"trial-in-family:{family_key}",
                    value=trial_key,
                ),
                trial_key,
            ),
        )
        for trial_key in family_trials:
            rotation = stream_position % len(base)
            schedule[trial_key] = base[rotation:] + base[:rotation]
            stream_position += 1
    _assert_action_position_balance(schedule, {trial: family for trial, family in pairs})
    return MappingProxyType(schedule)


def _assert_action_position_balance(
    schedule: Mapping[str, tuple[str, ...]],
    trial_families: Mapping[str, str],
) -> None:
    if set(schedule) != set(trial_families):
        raise OnlineRunnerError("action schedule does not cover the exact trial set")

    groups: dict[str, list[tuple[str, ...]]] = {"corpus": []}
    for trial_key, actions in schedule.items():
        if len(actions) != len(REGISTERED_ACTION_SET) or set(actions) != set(REGISTERED_ACTION_SET):
            raise OnlineRunnerError("each trial action schedule must be one full permutation")
        groups["corpus"].append(actions)
        groups.setdefault(f"family:{trial_families[trial_key]}", []).append(actions)
    for group, orders in groups.items():
        for action in REGISTERED_ACTION_SET:
            counts = [sum(order[position] == action for order in orders) for position in range(4)]
            if max(counts) - min(counts) > 1:
                raise OnlineRunnerError(f"action execution positions are imbalanced in {group}")


def _query_vector_sha256(vector: np.ndarray) -> str:
    little_endian = np.asarray(vector, dtype="<f4", order="C")
    digest = hashlib.sha256()
    digest.update(b"fractal-online-query-vector-v1\x00")
    digest.update(little_endian.size.to_bytes(8, "big"))
    digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def _query_binding_sha256(
    *,
    execution_artifact_sha256: str,
    trial_key: str,
    family_key: str,
    query_text_sha256: str,
    active_query_vector_sha256: str,
    current_truth_query_vector_sha256: str,
    environment_sha256: str,
) -> str:
    return _sha256(
        {
            "active_query_vector_sha256": active_query_vector_sha256,
            "current_truth_query_vector_sha256": (current_truth_query_vector_sha256),
            "environment_sha256": environment_sha256,
            "execution_artifact_sha256": execution_artifact_sha256,
            "family_key": family_key,
            "query_text_sha256": query_text_sha256,
            "schema": "fractal-online-query-binding-v2",
            "trial_key": trial_key,
        }
    )


def _authorization_mask_sha256(mask: np.ndarray) -> str:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 1:
        raise OnlineRunnerError("authorization mask must be one-dimensional")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _sealed_run_receipt_binding_sha256(receipt: SealedRunReceipt) -> str:
    return receipt.binding_sha256


@dataclass(frozen=True)
class FrozenFeatureContext:
    """Registered trial covariates not derived by the governed retriever."""

    version_lag: float
    backend: str
    drift_family: str
    policy_complexity: float

    def __post_init__(self) -> None:
        for name in ("version_lag", "policy_complexity"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise OnlineRunnerError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        _require_text("backend", self.backend)
        _require_text("drift_family", self.drift_family)


@dataclass(frozen=True, init=False)
class OnlineTrialRuntime:
    """Immutable active and current-truth queries plus finite policy context."""

    active_query_vector: np.ndarray
    current_truth_query_vector: np.ndarray
    active_query_vector_sha256: str
    current_truth_query_vector_sha256: str
    feature_context: FrozenFeatureContext
    environment_sha256: str
    _environment_json: str

    def __init__(
        self,
        *,
        active_query_vector: np.ndarray,
        current_truth_query_vector: np.ndarray,
        feature_context: FrozenFeatureContext,
        environment: Mapping[str, object] | None = None,
    ) -> None:
        vectors: dict[str, np.ndarray] = {}
        for name, value in (
            ("active_query_vector", active_query_vector),
            ("current_truth_query_vector", current_truth_query_vector),
        ):
            vector = np.array(value, dtype=np.float32, copy=True, order="C")
            if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
                raise OnlineRunnerError(f"{name} must be one finite non-empty vector")
            vector.setflags(write=False)
            vectors[name] = vector
        active = vectors["active_query_vector"]
        current = vectors["current_truth_query_vector"]
        if active.shape != current.shape:
            raise OnlineRunnerError(
                "active and current-truth query vectors must have the same width"
            )
        if np.shares_memory(active, current):
            raise OnlineRunnerError("query epochs must own separate memory")
        if not isinstance(feature_context, FrozenFeatureContext):
            raise OnlineRunnerError("feature_context must be FrozenFeatureContext")
        if environment is not None and not isinstance(environment, Mapping):
            raise OnlineRunnerError("environment must be a mapping or None")
        try:
            encoded = json.dumps(
                dict(environment or {}),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise OnlineRunnerError("environment must contain finite JSON values") from exc
        object.__setattr__(self, "active_query_vector", active)
        object.__setattr__(self, "current_truth_query_vector", current)
        object.__setattr__(
            self,
            "active_query_vector_sha256",
            _query_vector_sha256(active),
        )
        object.__setattr__(
            self,
            "current_truth_query_vector_sha256",
            _query_vector_sha256(current),
        )
        object.__setattr__(self, "feature_context", feature_context)
        object.__setattr__(
            self,
            "environment_sha256",
            policy_environment_sha256(json.loads(encoded)),
        )
        object.__setattr__(self, "_environment_json", encoded)

    @property
    def environment(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self._environment_json))


@dataclass(frozen=True)
class PreparedAuthorization:
    """Stable authorization identity loaded before any action timer starts."""

    environment_sha256: str
    policy_version: str
    mask_sha256: str
    authorized_count: int
    schema_version: str = CACHE_PREPARATION_ROW_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("environment_sha256", self.environment_sha256)
        _require_text("policy_version", self.policy_version)
        _require_sha256("mask_sha256", self.mask_sha256)
        if (
            isinstance(self.authorized_count, bool)
            or not isinstance(self.authorized_count, int)
            or self.authorized_count <= 0
        ):
            raise OnlineRunnerError("authorized_count must be a positive integer")
        if self.schema_version != CACHE_PREPARATION_ROW_SCHEMA:
            raise OnlineRunnerError(f"schema_version must equal {CACHE_PREPARATION_ROW_SCHEMA!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "authorized_count": self.authorized_count,
            "environment_sha256": self.environment_sha256,
            "mask_sha256": self.mask_sha256,
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: object) -> PreparedAuthorization:
        row = _closed_mapping(
            payload,
            fields=_CACHE_PREPARATION_ROW_FIELDS,
            name="cache-preparation row",
        )
        return cls(
            environment_sha256=row["environment_sha256"],
            policy_version=row["policy_version"],
            mask_sha256=row["mask_sha256"],
            authorized_count=row["authorized_count"],
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class CachePreparationReceipt:
    """Closed proof that every admitted policy universe was loaded pre-timing."""

    execution_artifact_sha256: str
    run_receipt_sha256: str
    document_universe_sha256: str
    role: str
    policy_action: str
    expected_policy_version: str
    rows: tuple[PreparedAuthorization, ...]
    algorithm: str = CACHE_PREPARATION_ALGORITHM
    schema_version: str = CACHE_PREPARATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("execution_artifact_sha256", self.execution_artifact_sha256)
        _require_sha256("run_receipt_sha256", self.run_receipt_sha256)
        _require_sha256("document_universe_sha256", self.document_universe_sha256)
        _require_text("role", self.role)
        _require_text("policy_action", self.policy_action)
        _require_text("expected_policy_version", self.expected_policy_version)
        if self.algorithm != CACHE_PREPARATION_ALGORITHM:
            raise OnlineRunnerError(f"algorithm must equal {CACHE_PREPARATION_ALGORITHM!r}")
        if self.schema_version != CACHE_PREPARATION_RECEIPT_SCHEMA:
            raise OnlineRunnerError(
                f"schema_version must equal {CACHE_PREPARATION_RECEIPT_SCHEMA!r}"
            )
        rows = tuple(self.rows)
        if not rows or not all(isinstance(row, PreparedAuthorization) for row in rows):
            raise OnlineRunnerError("cache-preparation rows must be non-empty and typed")
        environments = [row.environment_sha256 for row in rows]
        if len(environments) != len(set(environments)):
            raise OnlineRunnerError("cache-preparation rows repeat an environment")
        if any(row.policy_version != self.expected_policy_version for row in rows):
            raise OnlineRunnerError("prepared policy version differs from the expected revision")
        object.__setattr__(
            self,
            "rows",
            tuple(sorted(rows, key=lambda row: row.environment_sha256)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "document_universe_sha256": self.document_universe_sha256,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "expected_policy_version": self.expected_policy_version,
            "policy_action": self.policy_action,
            "role": self.role,
            "rows": [row.to_dict() for row in self.rows],
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: object) -> CachePreparationReceipt:
        receipt = _closed_mapping(
            payload,
            fields=_CACHE_PREPARATION_RECEIPT_FIELDS,
            name="cache-preparation receipt",
        )
        rows = receipt["rows"]
        if not isinstance(rows, list):
            raise OnlineRunnerError("cache-preparation rows must be an array")
        return cls(
            execution_artifact_sha256=receipt["execution_artifact_sha256"],
            run_receipt_sha256=receipt["run_receipt_sha256"],
            document_universe_sha256=receipt["document_universe_sha256"],
            role=receipt["role"],
            policy_action=receipt["policy_action"],
            expected_policy_version=receipt["expected_policy_version"],
            rows=tuple(PreparedAuthorization.from_dict(row) for row in rows),
            algorithm=receipt["algorithm"],
            schema_version=receipt["schema_version"],
        )


def _validate_runtime_query_vectors(
    runtime: OnlineTrialRuntime,
    *,
    dimension: int,
) -> None:
    bindings = (
        (
            "active_query_vector",
            runtime.active_query_vector,
            runtime.active_query_vector_sha256,
        ),
        (
            "current_truth_query_vector",
            runtime.current_truth_query_vector,
            runtime.current_truth_query_vector_sha256,
        ),
    )
    for name, vector, expected_sha256 in bindings:
        if (
            not isinstance(vector, np.ndarray)
            or vector.dtype != np.dtype(np.float32)
            or vector.shape != (dimension,)
            or not vector.flags.c_contiguous
            or vector.flags.writeable
            or not np.all(np.isfinite(vector))
        ):
            raise OnlineRunnerError(
                f"{name} width, finiteness, or immutable storage contract changed"
            )
        if _query_vector_sha256(vector) != expected_sha256:
            raise OnlineRunnerError(f"{name} mutability drift changed its frozen digest")
    if np.shares_memory(runtime.active_query_vector, runtime.current_truth_query_vector):
        raise OnlineRunnerError("active and current-truth query vectors share mutable storage")


@dataclass(frozen=True)
class TrialExecutionOrder:
    """Detached evidence of one query binding and its actual action order."""

    trial_key: str
    family_key: str
    query_text_sha256: str
    active_query_vector_sha256: str
    current_truth_query_vector_sha256: str
    environment_sha256: str
    query_binding_sha256: str
    actions: tuple[str, ...]
    schema_version: str = EXECUTION_ORDER_ROW_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "trial_key",
            "family_key",
            "query_text_sha256",
            "active_query_vector_sha256",
            "current_truth_query_vector_sha256",
            "environment_sha256",
            "query_binding_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.schema_version != EXECUTION_ORDER_ROW_SCHEMA:
            raise OnlineRunnerError(f"schema_version must equal {EXECUTION_ORDER_ROW_SCHEMA!r}")
        actions = tuple(self.actions)
        if len(actions) != len(REGISTERED_ACTION_SET) or set(actions) != set(REGISTERED_ACTION_SET):
            raise OnlineRunnerError("actions must be one permutation of the registered set")
        object.__setattr__(self, "actions", actions)

    def to_dict(self) -> dict[str, object]:
        return {
            "active_query_vector_sha256": self.active_query_vector_sha256,
            "actions": list(self.actions),
            "current_truth_query_vector_sha256": (self.current_truth_query_vector_sha256),
            "environment_sha256": self.environment_sha256,
            "family_key": self.family_key,
            "query_binding_sha256": self.query_binding_sha256,
            "query_text_sha256": self.query_text_sha256,
            "schema_version": self.schema_version,
            "trial_key": self.trial_key,
        }

    @classmethod
    def from_dict(cls, payload: object) -> TrialExecutionOrder:
        row = _closed_mapping(payload, fields=_ORDER_ROW_FIELDS, name="execution-order row")
        actions = row["actions"]
        if not isinstance(actions, list) or not all(isinstance(action, str) for action in actions):
            raise OnlineRunnerError("execution-order row actions must be a string array")
        return cls(
            trial_key=row["trial_key"],
            family_key=row["family_key"],
            query_text_sha256=row["query_text_sha256"],
            active_query_vector_sha256=row["active_query_vector_sha256"],
            current_truth_query_vector_sha256=row["current_truth_query_vector_sha256"],
            environment_sha256=row["environment_sha256"],
            query_binding_sha256=row["query_binding_sha256"],
            actions=tuple(actions),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class ExecutionOrderReceipt:
    """Closed receipt binding portable action permutations to one admitted run."""

    execution_artifact_sha256: str
    run_receipt_sha256: str
    cache_preparation_receipt_sha256: str
    permutation_seed: int
    rows: tuple[TrialExecutionOrder, ...]
    action_set: tuple[str, ...] = REGISTERED_ACTION_SET
    position_balance_units: tuple[str, ...] = POSITION_BALANCE_UNITS
    permutation_algorithm: str = PERMUTATION_ALGORITHM
    schema_version: str = EXECUTION_ORDER_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("execution_artifact_sha256", self.execution_artifact_sha256)
        _require_sha256("run_receipt_sha256", self.run_receipt_sha256)
        _require_sha256(
            "cache_preparation_receipt_sha256",
            self.cache_preparation_receipt_sha256,
        )
        seed = _validate_seed(self.permutation_seed)
        object.__setattr__(self, "permutation_seed", seed)
        actions = tuple(self.action_set)
        if actions != REGISTERED_ACTION_SET:
            raise OnlineRunnerError("action_set must equal the registered four-action set")
        object.__setattr__(self, "action_set", actions)
        if self.permutation_algorithm != PERMUTATION_ALGORITHM:
            raise OnlineRunnerError(f"permutation_algorithm must equal {PERMUTATION_ALGORITHM!r}")
        balance_units = tuple(self.position_balance_units)
        if balance_units != POSITION_BALANCE_UNITS:
            raise OnlineRunnerError(f"position_balance_units must equal {POSITION_BALANCE_UNITS!r}")
        object.__setattr__(self, "position_balance_units", balance_units)
        if self.schema_version != EXECUTION_ORDER_RECEIPT_SCHEMA:
            raise OnlineRunnerError(f"schema_version must equal {EXECUTION_ORDER_RECEIPT_SCHEMA!r}")
        rows = tuple(self.rows)
        if not rows or not all(isinstance(row, TrialExecutionOrder) for row in rows):
            raise OnlineRunnerError("rows must contain execution-order rows")
        keys = [row.trial_key for row in rows]
        if len(keys) != len(set(keys)):
            raise OnlineRunnerError("execution-order receipt contains duplicate trials")
        for row in rows:
            expected_binding = _query_binding_sha256(
                execution_artifact_sha256=self.execution_artifact_sha256,
                trial_key=row.trial_key,
                family_key=row.family_key,
                query_text_sha256=row.query_text_sha256,
                active_query_vector_sha256=row.active_query_vector_sha256,
                current_truth_query_vector_sha256=(row.current_truth_query_vector_sha256),
                environment_sha256=row.environment_sha256,
            )
            if row.query_binding_sha256 != expected_binding:
                raise OnlineRunnerError("execution-order row has a changed query binding")
        expected_orders = portable_balanced_action_orders(
            permutation_seed=seed,
            execution_artifact_sha256=self.execution_artifact_sha256,
            trial_families=tuple((row.trial_key, row.family_key) for row in rows),
        )
        for row in rows:
            if row.actions != expected_orders[row.trial_key]:
                raise OnlineRunnerError("execution-order row does not match the balanced schedule")
        object.__setattr__(
            self,
            "rows",
            tuple(sorted(rows, key=lambda row: (row.family_key, row.trial_key))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action_set": list(self.action_set),
            "cache_preparation_receipt_sha256": (self.cache_preparation_receipt_sha256),
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "permutation_algorithm": self.permutation_algorithm,
            "permutation_seed": self.permutation_seed,
            "position_balance_units": list(self.position_balance_units),
            "rows": [row.to_dict() for row in self.rows],
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: object) -> ExecutionOrderReceipt:
        receipt = _closed_mapping(
            payload,
            fields=_ORDER_RECEIPT_FIELDS,
            name="execution-order receipt",
        )
        actions = receipt["action_set"]
        balance_units = receipt["position_balance_units"]
        rows = receipt["rows"]
        if not isinstance(actions, list) or not all(isinstance(action, str) for action in actions):
            raise OnlineRunnerError("execution-order action_set must be a string array")
        if not isinstance(balance_units, list) or not all(
            isinstance(unit, str) for unit in balance_units
        ):
            raise OnlineRunnerError("position_balance_units must be a string array")
        if not isinstance(rows, list):
            raise OnlineRunnerError("execution-order rows must be an array")
        return cls(
            execution_artifact_sha256=receipt["execution_artifact_sha256"],
            run_receipt_sha256=receipt["run_receipt_sha256"],
            cache_preparation_receipt_sha256=(receipt["cache_preparation_receipt_sha256"]),
            permutation_seed=receipt["permutation_seed"],
            rows=tuple(TrialExecutionOrder.from_dict(row) for row in rows),
            action_set=tuple(actions),
            position_balance_units=tuple(balance_units),
            permutation_algorithm=receipt["permutation_algorithm"],
            schema_version=receipt["schema_version"],
        )


def _decode_json_object(payload: str | bytes, *, name: str) -> tuple[object, bytes]:
    encoded = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OnlineRunnerError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise OnlineRunnerError(f"{name} contains non-finite number {value!r}")

    try:
        decoded = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OnlineRunnerError(f"{name} must be valid UTF-8 JSON") from exc
    return decoded, encoded


def loads_cache_preparation_receipt(payload: str | bytes) -> CachePreparationReceipt:
    decoded, encoded = _decode_json_object(payload, name="cache-preparation receipt")
    receipt = CachePreparationReceipt.from_dict(decoded)
    if encoded != receipt.canonical_bytes():
        raise OnlineRunnerError("cache-preparation receipt bytes are not canonical")
    return receipt


def load_cache_preparation_receipt(path: str | Path) -> CachePreparationReceipt:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    encoded = read_secure_control_file(
        target,
        label="cache-preparation receipt",
    )
    if not encoded.endswith(b"\n"):
        raise OnlineRunnerError("cache-preparation receipt file needs one terminal newline")
    return loads_cache_preparation_receipt(encoded[:-1])


def write_cache_preparation_receipt(
    receipt: CachePreparationReceipt,
    target: str | Path,
) -> None:
    if not isinstance(receipt, CachePreparationReceipt):
        raise OnlineRunnerError("receipt must be CachePreparationReceipt")
    write_exclusive_receipt_bytes(receipt.canonical_bytes() + b"\n", target)


def loads_execution_order_receipt(payload: str | bytes) -> ExecutionOrderReceipt:
    decoded, encoded = _decode_json_object(payload, name="execution-order receipt")
    receipt = ExecutionOrderReceipt.from_dict(decoded)
    if encoded != receipt.canonical_bytes():
        raise OnlineRunnerError("execution-order receipt bytes are not canonical")
    return receipt


def load_execution_order_receipt(path: str | Path) -> ExecutionOrderReceipt:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    encoded = read_secure_control_file(
        target,
        label="execution-order receipt",
    )
    if not encoded.endswith(b"\n"):
        raise OnlineRunnerError("execution-order receipt file needs one terminal newline")
    return loads_execution_order_receipt(encoded[:-1])


def write_execution_order_receipt(
    receipt: ExecutionOrderReceipt,
    target: str | Path,
) -> None:
    if not isinstance(receipt, ExecutionOrderReceipt):
        raise OnlineRunnerError("receipt must be ExecutionOrderReceipt")
    write_exclusive_receipt_bytes(receipt.canonical_bytes() + b"\n", target)


class _CapturingPolicy:
    def __init__(
        self,
        delegate: PolicyDecisionPoint,
        cache_preparation_receipt: CachePreparationReceipt,
    ) -> None:
        self.delegate = delegate
        self.prepared = {row.environment_sha256: row for row in cache_preparation_receipt.rows}
        self.expected_role = cache_preparation_receipt.role
        self.expected_action = cache_preparation_receipt.policy_action
        self.expected_document_universe_sha256 = cache_preparation_receipt.document_universe_sha256
        self.reference_context: tuple[object, ...] | None = None
        self.last_decision: PolicyDecision | None = None
        self.context_changed = False
        self.decision_replayed = False
        self._seen_decision_ids: set[str] = set()
        self._seen_request_nonces: set[str] = set()
        self._seen_request_sha256s: set[str] = set()

    @property
    def n_documents(self) -> int:
        return self.delegate.n_documents

    @property
    def document_universe_sha256(self) -> str:
        return self.delegate.document_universe_sha256

    def start_trial(self) -> None:
        self.reference_context = None
        self.last_decision = None
        self.context_changed = False

    def start_action(self) -> None:
        self.last_decision = None

    @staticmethod
    def _context(decision: PolicyDecision) -> tuple[object, ...]:
        mask_sha256 = hashlib.sha256(decision.authorized_mask.tobytes(order="C")).hexdigest()
        return (
            decision.subject,
            decision.action,
            decision.policy_version,
            decision.available,
            decision.environment_sha256,
            decision.document_universe_sha256,
            mask_sha256,
            int(decision.authorized_mask.size),
        )

    def decide(
        self,
        subject: str,
        *,
        action: str = "retrieve",
        environment: Mapping[str, object] | None = None,
    ) -> PolicyDecision:
        decision = self.delegate.decide(
            subject,
            action=action,
            environment=environment,
        )
        self.last_decision = decision
        prepared = self.prepared.get(decision.environment_sha256)
        if (
            prepared is None
            or decision.subject != self.expected_role
            or decision.action != self.expected_action
            or decision.document_universe_sha256 != self.expected_document_universe_sha256
            or decision.policy_version != prepared.policy_version
            or decision.authorized_count != prepared.authorized_count
            or _authorization_mask_sha256(decision.authorized_mask) != prepared.mask_sha256
        ):
            self.context_changed = True
        if (
            decision.decision_id in self._seen_decision_ids
            or decision.request_nonce in self._seen_request_nonces
            or decision.request_sha256 in self._seen_request_sha256s
        ):
            self.decision_replayed = True
        self._seen_decision_ids.add(decision.decision_id)
        self._seen_request_nonces.add(decision.request_nonce)
        self._seen_request_sha256s.add(decision.request_sha256)
        context = self._context(decision)
        if self.reference_context is None:
            self.reference_context = context
        elif context != self.reference_context:
            self.context_changed = True
        return decision


class _PairedActionController:
    def __init__(self, delegate: RuleController) -> None:
        self.delegate = delegate
        self.config = delegate.config
        self.forced_action: str | None = None
        self.selected_decision: ControllerDecision | None = None
        self.last_decision: ControllerDecision | None = None
        self.frozen_geometry: QueryGeometry | None = None
        self.frozen_authorized_count: int | None = None

    def start_trial(self) -> None:
        self.forced_action = None
        self.selected_decision = None
        self.last_decision = None
        self.frozen_geometry = None
        self.frozen_authorized_count = None

    def start_action(self, action: str) -> None:
        if action not in REGISTERED_ACTION_SET:
            raise OnlineRunnerError("forced action is not registered")
        self.forced_action = action
        self.last_decision = (
            None if self.selected_decision is None else self._forced_decision(action)
        )

    def _forced_decision(self, action: str) -> ControllerDecision:
        selected = self.selected_decision
        if selected is None:
            raise OnlineRunnerError("controller selection has not been frozen")
        if action == selected.action:
            return selected
        return ControllerDecision(
            action=action,  # type: ignore[arg-type]
            risk_score=selected.risk_score,
            reasons=(
                "registered paired counterfactual",
                f"frozen controller selected {selected.action}",
            ),
            policy_version=selected.policy_version,
        )

    def decide(
        self,
        features: QueryGeometry,
        *,
        n_authorized: int,
        policy_version: str,
        policy_available: bool = True,
        expected_policy_version: str | None = None,
    ) -> ControllerDecision:
        if self.forced_action is None:
            raise OnlineRunnerError("no action was assigned before controller invocation")
        if self.selected_decision is None:
            selected = self.delegate.decide(
                features,
                n_authorized=n_authorized,
                policy_version=policy_version,
                policy_available=policy_available,
                expected_policy_version=expected_policy_version,
            )
            if selected.action not in REGISTERED_ACTION_SET:
                raise OnlineRunnerError("frozen controller selected an unregistered action")
            self.selected_decision = selected
            self.frozen_geometry = features
            self.frozen_authorized_count = n_authorized
        else:
            if (
                policy_version != self.selected_decision.policy_version
                or n_authorized != self.frozen_authorized_count
            ):
                raise OnlineRunnerError(
                    "policy revision or authorized universe changed across paired actions"
                )
        decision = self._forced_decision(self.forced_action)
        self.last_decision = decision
        return decision


def _execution_view(
    execution: object,
) -> tuple[str, str, int, str, tuple[_TrialView, ...]]:
    try:
        corpus = execution.corpus  # type: ignore[attr-defined]
        stage = execution.stage  # type: ignore[attr-defined]
        trials = tuple(execution.trials)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise OnlineRunnerError("execution lacks the admitted online artifact interface") from exc
    try:
        artifact_sha256 = execution_artifact_sha256(execution)
        document_count = execution_document_count(execution)
    except ValueError as exc:
        raise OnlineRunnerError("execution control bindings are invalid") from exc
    _require_text("execution.corpus", corpus)
    if stage != "sealed":
        raise OnlineRunnerError("execution stage must equal 'sealed'")
    if not trials:
        raise OnlineRunnerError("execution must contain trials")
    try:
        documents = tuple(execution.documents)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        documents = ()
    if documents:
        if len(documents) != document_count:
            raise OnlineRunnerError("execution document count differs from inline documents")
        document_universe_sha256 = _execution_document_universe_sha256(documents)
    else:
        try:
            document_universe_sha256 = execution.document_universe_sha256  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise OnlineRunnerError(
                "sharded execution lacks an ordered document-universe digest"
            ) from exc
        _require_sha256("execution.document_universe_sha256", document_universe_sha256)
    keys: list[str] = []
    typed_trials: list[_TrialView] = []
    for trial in trials:
        try:
            trial_key = trial.trial_key
            family_key = trial.family_key
            text = trial.text
            trial_corpus = trial.corpus
            trial_stage = trial.stage
        except AttributeError as exc:
            raise OnlineRunnerError("execution trial lacks required online fields") from exc
        _require_sha256("trial_key", trial_key)
        _require_sha256("family_key", family_key)
        _require_text("trial.text", text)
        if trial_corpus != corpus or trial_stage != stage:
            raise OnlineRunnerError("trial corpus or stage differs from its execution")
        keys.append(trial_key)
        typed_trials.append(trial)
    if len(keys) != len(set(keys)):
        raise OnlineRunnerError("execution contains duplicate trial keys")
    return (
        artifact_sha256,
        corpus,
        document_count,
        document_universe_sha256,
        tuple(typed_trials),
    )


def _execution_document_universe_sha256(documents: Sequence[object]) -> str:
    identities: list[str] = []
    for position, document in enumerate(documents):
        try:
            document_id = document.document_id
            payload = {
                "content_hash": document.content_hash,
                "document_id": document_id,
                "external_id": document.external_id,
                "source_uri": document.source_uri,
            }
        except AttributeError as exc:
            raise OnlineRunnerError("execution document lacks provenance fields") from exc
        if type(document_id) is not int or document_id != position:
            raise OnlineRunnerError("execution document IDs must be contiguous and ordered")
        identities.append(_canonical_bytes(payload).decode("utf-8"))
    return policy_document_universe_sha256(identities)


def _failure_code(exception: Exception) -> FailureCode:
    if isinstance(exception, TimeoutError):
        code: FailureCode = "backend-timeout"
    elif isinstance(exception, MemoryError):
        code = "resource-exhausted"
    elif isinstance(exception, InterruptedError):
        code = "runner-interruption"
    elif isinstance(exception, (AssertionError, TypeError, ValueError)):
        code = "invalid-result"
    else:
        code = "backend-error"
    if code not in REGISTERED_FAILURE_CODES:
        raise OnlineRunnerError("runner produced an unregistered failure code")
    return code


def _probe_work(geometry: QueryGeometry) -> float:
    if geometry.distance_evaluations is not None:
        return float(geometry.distance_evaluations)
    if geometry.visited_candidates is not None:
        return float(geometry.visited_candidates)
    return float("nan")


def _registered_features(
    *,
    geometry: QueryGeometry,
    n_authorized: int,
    retriever: GovernedRetriever,
    corpus: str,
    context: FrozenFeatureContext,
) -> tuple[object, ...]:
    values = {
        "corpus_size": float(len(retriever.vectors)),
        "authorized_universe_size": float(n_authorized),
        "embedding_dimension": float(retriever.vectors.shape[1]),
        "version_lag": context.version_lag,
        "drift_severity": float(geometry.embedding_drift),
        "probe_latency_ms": float(geometry.search_latency_ms),
        "probe_work": _probe_work(geometry),
        "corpus_stratum": corpus,
        "backend": context.backend,
        "drift_family": context.drift_family,
        "allow_rate": float(geometry.authorized_selectivity),
        "policy_complexity": context.policy_complexity,
        "policy_churn": float(geometry.policy_churn),
        "lid_k50": float(geometry.lid),
        "lid_cv": float(geometry.lid_scale_instability),
        "relative_contrast": float(geometry.relative_contrast),
        "radius_expansion": float(geometry.radius_expansion),
    }
    if tuple(values) != REGISTERED_FEATURE_SCHEMA.input_features:
        raise OnlineRunnerError("runner feature order differs from REGISTERED_FEATURE_SCHEMA")
    return tuple(values[name] for name in REGISTERED_FEATURE_SCHEMA.input_features)


def _request_identifiers(
    *,
    execution_artifact_sha256: str,
    query_binding_sha256: str,
    trial_key: str,
    action: str,
    actual_order: int,
) -> tuple[str, str]:
    trace = _sha256(
        {
            "execution_artifact_sha256": execution_artifact_sha256,
            "query_binding_sha256": query_binding_sha256,
            "trial_key": trial_key,
        }
    )
    request = _sha256(
        {
            "action": action,
            "actual_order": actual_order,
            "trace_sha256": trace,
        }
    )
    return f"request-{request}", f"trace-{trace}"


def _failed_execution(
    *,
    trial: _TrialView,
    action: str,
    paired_controller: _PairedActionController,
    policy_capture: _CapturingPolicy,
    failure_code: FailureCode,
    started_monotonic_ns: int,
    finished_monotonic_ns: int,
    runner_identity: str,
    feature_values: tuple[object, ...] | None,
) -> FailedActionExecution:
    if action == "exact-authorized":
        raise OnlineRunnerError("exact-authorized must complete; the matrix was aborted")
    if policy_capture.context_changed or policy_capture.decision_replayed:
        raise OnlineRunnerError(
            "policy universe changed or a decision was replayed during paired execution"
        )
    decision = paired_controller.last_decision
    authorization = policy_capture.last_decision
    if decision is None or authorization is None or not authorization.available:
        raise OnlineRunnerError("failed action lacks an available bound authorization")
    return FailedActionExecution(
        trial=trial,  # type: ignore[arg-type]
        decision=decision,
        authorization=authorization,
        failure_code=failure_code,
        started_monotonic_ns=started_monotonic_ns,
        finished_monotonic_ns=max(finished_monotonic_ns, started_monotonic_ns + 1),
        runner_identity=runner_identity,
        feature_values=feature_values,
    )


@dataclass(frozen=True)
class OnlineRunArtifacts:
    """In-memory outputs that can be written and externally anchored by C0."""

    admitted_panel: AdmittedActionPanel
    cache_preparation_receipt: CachePreparationReceipt
    execution_order_receipt: ExecutionOrderReceipt
    audit_records: tuple[AuditRecord, ...]
    selected_decisions: tuple[tuple[str, ControllerDecision], ...]
    failed_executions: tuple[FailedActionExecution, ...]

    def __post_init__(self) -> None:
        panel = self.admitted_panel.panel
        preparation = self.cache_preparation_receipt
        receipt = self.execution_order_receipt
        if not isinstance(preparation, CachePreparationReceipt):
            raise OnlineRunnerError("cache preparation evidence must be a typed receipt")
        if panel.execution_artifact_sha256 != preparation.execution_artifact_sha256:
            raise OnlineRunnerError("panel and cache preparation bind different executions")
        if panel.run_receipt_sha256 != preparation.run_receipt_sha256:
            raise OnlineRunnerError("panel and cache preparation bind different run receipts")
        if panel.execution_artifact_sha256 != receipt.execution_artifact_sha256:
            raise OnlineRunnerError("panel and order receipt bind different executions")
        if panel.run_receipt_sha256 != receipt.run_receipt_sha256:
            raise OnlineRunnerError("panel and order receipt bind different run receipts")
        if receipt.cache_preparation_receipt_sha256 != preparation.receipt_sha256:
            raise OnlineRunnerError("execution order does not bind its cache preparation")
        orders = {row.trial_key: row.actions for row in receipt.rows}
        for row in panel.rows:
            order = orders.get(row.trial_key)
            if order is None or order[row.execution_position] != row.action:
                raise OnlineRunnerError(
                    "panel execution_position differs from the detached balanced schedule"
                )
        records = tuple(self.audit_records)
        if not records:
            raise OnlineRunnerError("online outputs require a governed audit chain")
        verification = verify_audit_chain(
            records,
            expected_head_sha256=records[-1].record_sha256,
            expected_length=len(records),
        )
        if not verification.valid:
            raise OnlineRunnerError("online outputs contain an invalid audit chain")
        object.__setattr__(self, "audit_records", records)
        selections = tuple(self.selected_decisions)
        if not all(
            isinstance(key, str) and isinstance(value, ControllerDecision)
            for key, value in selections
        ):
            raise OnlineRunnerError("selected_decisions must contain typed pairs")
        if len({key for key, _ in selections}) != len(selections):
            raise OnlineRunnerError("selected_decisions contain duplicate trials")
        selections = tuple(sorted(selections, key=lambda item: item[0]))
        if len(selections) != len(receipt.rows):
            raise OnlineRunnerError("selected decisions do not cover every order row")
        if {key for key, _ in selections} != {row.trial_key for row in receipt.rows}:
            raise OnlineRunnerError("selected decisions and order rows bind different trials")
        object.__setattr__(self, "selected_decisions", selections)
        object.__setattr__(self, "failed_executions", tuple(self.failed_executions))

    @property
    def anchoring_digests(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "action_panel_admission_receipt_sha256": (
                    self.admitted_panel.admission_receipt.receipt_sha256
                ),
                "action_panel_artifact_sha256": self.admitted_panel.panel.artifact_sha256,
                "audit_head_sha256": self.audit_records[-1].record_sha256,
                "cache_preparation_receipt_sha256": (self.cache_preparation_receipt.receipt_sha256),
                "execution_order_receipt_sha256": (self.execution_order_receipt.receipt_sha256),
            }
        )


def _prepare_authorization_cache(
    *,
    execution_artifact_sha256: str,
    run_receipt_sha256: str,
    document_universe_sha256: str,
    trials: Sequence[_TrialView],
    trial_runtimes: Mapping[str, OnlineTrialRuntime],
    retriever: GovernedRetriever,
    policy_action: str,
    expected_policy_version: str,
) -> CachePreparationReceipt:
    """Preload each distinct trial authorization, then close the cache."""

    rows: dict[str, PreparedAuthorization] = {}
    for trial in trials:
        runtime = trial_runtimes[trial.trial_key]
        environment_sha256 = runtime.environment_sha256
        if environment_sha256 in rows:
            continue
        try:
            decision = retriever.prepare_authorization(
                action=policy_action,
                environment=runtime.environment,
                expected_policy_version=expected_policy_version,
            )
        except Exception as exc:
            raise OnlineRunnerError(
                "authorization cache preparation failed before request timing"
            ) from exc
        row = PreparedAuthorization(
            environment_sha256=environment_sha256,
            policy_version=decision.policy_version,
            mask_sha256=_authorization_mask_sha256(decision.authorized_mask),
            authorized_count=decision.authorized_count,
        )
        rows[environment_sha256] = row
    try:
        retriever.seal_authorized_index_cache()
    except Exception as exc:
        raise OnlineRunnerError("authorization cache could not be sealed") from exc
    return CachePreparationReceipt(
        execution_artifact_sha256=execution_artifact_sha256,
        run_receipt_sha256=run_receipt_sha256,
        document_universe_sha256=document_universe_sha256,
        role=retriever.role,
        policy_action=policy_action,
        expected_policy_version=expected_policy_version,
        rows=tuple(rows.values()),
    )


def run_online_action_matrix(
    *,
    execution: object,
    run_receipt: SealedRunReceipt,
    retriever: GovernedRetriever,
    provenance_registry: AdmittedProvenanceRegistry,
    trial_runtimes: Mapping[str, OnlineTrialRuntime],
    permutation_seed: int,
    expected_policy_version: str,
    query_partition_audit_sha256: str,
    pseudonym_key: bytes,
    pseudonym_key_id: str,
    k: int = 10,
    policy_action: str = "retrieve",
    partition_label: Literal["primary", "reserve"] = "primary",
    occurred_at_factory: Callable[[str, str, int], str | None] | None = None,
) -> OnlineRunArtifacts:
    """Execute one complete, label-free, paired four-action matrix.

    The caller must give this function exclusive ownership of ``retriever`` for
    the call. Its original controller and policy objects are restored on every
    exit path.
    """

    if not isinstance(run_receipt, SealedRunReceipt):
        raise OnlineRunnerError("run_receipt must be SealedRunReceipt")
    if not isinstance(retriever, GovernedRetriever):
        raise OnlineRunnerError("retriever must be GovernedRetriever")
    if not isinstance(provenance_registry, AdmittedProvenanceRegistry):
        raise OnlineRunnerError("provenance_registry lacks the admitted digest-only interface")
    if not isinstance(retriever.controller, RuleController):
        raise OnlineRunnerError("retriever controller must be a frozen RuleController")
    _require_text("expected_policy_version", expected_policy_version)
    _require_sha256("query_partition_audit_sha256", query_partition_audit_sha256)
    _require_text("pseudonym_key_id", pseudonym_key_id)
    _require_text("policy_action", policy_action)
    seed = _validate_seed(permutation_seed)
    if type(k) is not int or k <= 0 or k > retriever.controller.config.probe_k:
        raise OnlineRunnerError("k must be within the frozen probe bound")

    execution_sha256, corpus, document_count, universe_sha256, trials = _execution_view(execution)
    if provenance_registry.corpus_name != corpus:
        raise OnlineRunnerError("provenance registry belongs to another corpus")
    if provenance_registry.corpus_stage != "sealed":
        raise OnlineRunnerError("provenance registry stage must equal 'sealed'")
    if provenance_registry.document_count != document_count:
        raise OnlineRunnerError("provenance registry document count differs")
    if (
        provenance_registry.document_universe_sha256 != universe_sha256
        or retriever.document_universe_sha256 != universe_sha256
    ):
        raise OnlineRunnerError("execution, provenance, and retriever universes differ")
    if len(retriever.vectors) != document_count:
        raise OnlineRunnerError("retriever vector count differs from execution documents")
    if retriever.role == "" or retriever.role != retriever.role.strip():
        raise OnlineRunnerError("retriever role must be a canonical subject")

    runtimes = dict(trial_runtimes)
    expected_keys = {trial.trial_key for trial in trials}
    if set(runtimes) != expected_keys:
        raise OnlineRunnerError("trial_runtimes must cover the exact execution trial set")
    if not all(isinstance(value, OnlineTrialRuntime) for value in runtimes.values()):
        raise OnlineRunnerError("trial_runtimes must contain OnlineTrialRuntime values")
    for runtime in runtimes.values():
        _validate_runtime_query_vectors(
            runtime,
            dimension=retriever.vectors.shape[1],
        )

    original_controller = retriever.controller
    original_policy = retriever.policy
    cache_preparation_receipt = _prepare_authorization_cache(
        execution_artifact_sha256=execution_sha256,
        run_receipt_sha256=_sealed_run_receipt_binding_sha256(run_receipt),
        document_universe_sha256=universe_sha256,
        trials=trials,
        trial_runtimes=runtimes,
        retriever=retriever,
        policy_action=policy_action,
        expected_policy_version=expected_policy_version,
    )
    paired_controller = _PairedActionController(original_controller)
    policy_capture = _CapturingPolicy(
        original_policy,
        cache_preparation_receipt,
    )
    governed: list[GovernedActionExecution] = []
    failed: list[FailedActionExecution] = []
    audit_records: list[AuditRecord] = []
    selected: dict[str, ControllerDecision] = {}
    order_rows: list[TrialExecutionOrder] = []
    previous_record: AuditRecord | None = None
    action_orders = portable_balanced_action_orders(
        permutation_seed=seed,
        execution_artifact_sha256=execution_sha256,
        trial_families=tuple((trial.trial_key, trial.family_key) for trial in trials),
    )

    retriever.controller = paired_controller  # type: ignore[assignment]
    retriever.policy = policy_capture
    try:
        for trial in trials:
            runtime = runtimes[trial.trial_key]
            _validate_runtime_query_vectors(
                runtime,
                dimension=retriever.vectors.shape[1],
            )
            query_text_sha256 = hashlib.sha256(trial.text.encode("utf-8")).hexdigest()
            query_binding_sha256 = _query_binding_sha256(
                execution_artifact_sha256=execution_sha256,
                trial_key=trial.trial_key,
                family_key=trial.family_key,
                query_text_sha256=query_text_sha256,
                active_query_vector_sha256=runtime.active_query_vector_sha256,
                current_truth_query_vector_sha256=(runtime.current_truth_query_vector_sha256),
                environment_sha256=runtime.environment_sha256,
            )
            action_order = action_orders[trial.trial_key]
            order_rows.append(
                TrialExecutionOrder(
                    trial_key=trial.trial_key,
                    family_key=trial.family_key,
                    query_text_sha256=query_text_sha256,
                    active_query_vector_sha256=(runtime.active_query_vector_sha256),
                    current_truth_query_vector_sha256=(runtime.current_truth_query_vector_sha256),
                    environment_sha256=runtime.environment_sha256,
                    query_binding_sha256=query_binding_sha256,
                    actions=action_order,
                )
            )
            paired_controller.start_trial()
            policy_capture.start_trial()
            trial_governed: list[GovernedActionExecution] = []
            trial_failed: list[FailedActionExecution] = []

            for actual_order, action in enumerate(action_order):
                _validate_runtime_query_vectors(
                    runtime,
                    dimension=retriever.vectors.shape[1],
                )
                paired_controller.start_action(action)
                policy_capture.start_action()
                started = perf_counter_ns()
                try:
                    result = retriever.query(
                        runtime.active_query_vector,
                        current_truth_query=runtime.current_truth_query_vector,
                        k=k,
                        expected_policy_version=expected_policy_version,
                        action=policy_action,
                        environment=runtime.environment,
                    )
                except OnlineRunnerError:
                    raise
                except Exception as exc:
                    _validate_runtime_query_vectors(
                        runtime,
                        dimension=retriever.vectors.shape[1],
                    )
                    finished = perf_counter_ns()
                    if paired_controller.frozen_geometry is None:
                        raise OnlineRunnerError(
                            "action failed before the controller selection was frozen"
                        ) from exc
                    features = _registered_features(
                        geometry=paired_controller.frozen_geometry,
                        n_authorized=paired_controller.frozen_authorized_count or 0,
                        retriever=retriever,
                        corpus=corpus,
                        context=runtime.feature_context,
                    )
                    failure = _failed_execution(
                        trial=trial,
                        action=action,
                        paired_controller=paired_controller,
                        policy_capture=policy_capture,
                        failure_code=_failure_code(exc),
                        started_monotonic_ns=started,
                        finished_monotonic_ns=finished,
                        runner_identity=run_receipt.runner_identity,
                        feature_values=features if action == "hnsw-low" else None,
                    )
                    trial_failed.append(failure)
                    continue
                _validate_runtime_query_vectors(
                    runtime,
                    dimension=retriever.vectors.shape[1],
                )
                finished = perf_counter_ns()

                if policy_capture.context_changed or policy_capture.decision_replayed:
                    raise OnlineRunnerError(
                        "policy universe changed or a decision was replayed during paired execution"
                    )
                if paired_controller.selected_decision is None:
                    raise OnlineRunnerError("controller selection was not frozen")
                if result.decision.action != action:
                    raise OnlineRunnerError(
                        "governed action abstained or changed outside its assigned matrix cell"
                    )
                if result.decision != paired_controller.last_decision:
                    raise OnlineRunnerError("governed result differs from its forced decision")
                if action != "abstain" and (
                    result.search is None or result.final_authorization is None
                ):
                    raise OnlineRunnerError("governed retrieval did not complete its action")
                if action == "exact-authorized" and result.search is None:
                    raise OnlineRunnerError("exact-authorized must complete")
                if result.index_refresh is None or result.index_refresh.rebuilt:
                    raise OnlineRunnerError(
                        "timed action used an index that was not prepared before timing"
                    )

                geometry = paired_controller.frozen_geometry
                n_authorized = paired_controller.frozen_authorized_count
                if geometry is None or n_authorized is None:
                    raise OnlineRunnerError("registered features were not frozen")
                features = _registered_features(
                    geometry=geometry,
                    n_authorized=n_authorized,
                    retriever=retriever,
                    corpus=corpus,
                    context=runtime.feature_context,
                )
                request_id, trace_id = _request_identifiers(
                    execution_artifact_sha256=execution_sha256,
                    query_binding_sha256=query_binding_sha256,
                    trial_key=trial.trial_key,
                    action=action,
                    actual_order=actual_order,
                )
                occurred_at = (
                    None
                    if occurred_at_factory is None
                    else occurred_at_factory(
                        trial.trial_key,
                        action,
                        len(audit_records),
                    )
                )
                try:
                    record = audit_record_from_governed_result(
                        result,
                        request_id=request_id,
                        trace_id=trace_id,
                        trial_sha256=trial.trial_key,
                        subject=retriever.role,
                        pseudonym_key=pseudonym_key,
                        pseudonym_key_id=pseudonym_key_id,
                        provenance_registry=provenance_registry,
                        occurred_at=occurred_at,
                        previous_record=previous_record,
                    )
                    admitted = GovernedActionExecution(
                        trial=trial,  # type: ignore[arg-type]
                        result=result,
                        audit_record=record,
                        feature_values=features if action == "hnsw-low" else None,
                    )
                except (ConfirmatoryAnalysisError, TypeError, ValueError):
                    failure = _failed_execution(
                        trial=trial,
                        action=action,
                        paired_controller=paired_controller,
                        policy_capture=policy_capture,
                        failure_code="invalid-result",
                        started_monotonic_ns=started,
                        finished_monotonic_ns=finished,
                        runner_identity=run_receipt.runner_identity,
                        feature_values=features if action == "hnsw-low" else None,
                    )
                    trial_failed.append(failure)
                    continue
                trial_governed.append(admitted)
                audit_records.append(record)
                previous_record = record

            if paired_controller.selected_decision is None:
                raise OnlineRunnerError("trial ended without a frozen controller decision")
            _validate_runtime_query_vectors(
                runtime,
                dimension=retriever.vectors.shape[1],
            )
            selected[trial.trial_key] = paired_controller.selected_decision
            governed.extend(trial_governed)
            failed.extend(trial_failed)
    finally:
        retriever.controller = original_controller
        retriever.policy = original_policy

    if previous_record is None:
        raise OnlineRunnerError("matrix produced no governed audit records")
    try:
        admitted_panel = action_panel_from_governed_executions(
            execution=execution,  # type: ignore[arg-type]
            run_receipt=run_receipt,
            governed_executions=tuple(governed),
            failed_executions=tuple(failed),
            selected_decisions=selected,
            action_set=REGISTERED_ACTION_SET,
            execution_orders=action_orders,
            expected_audit_head_sha256=previous_record.record_sha256,
            query_partition_audit_sha256=query_partition_audit_sha256,
            partition_label=partition_label,
        )
    except ConfirmatoryAnalysisError as exc:
        raise OnlineRunnerError(f"paired action panel was not admissible: {exc}") from exc
    order_receipt = ExecutionOrderReceipt(
        execution_artifact_sha256=execution_sha256,
        run_receipt_sha256=admitted_panel.panel.run_receipt_sha256,
        cache_preparation_receipt_sha256=(cache_preparation_receipt.receipt_sha256),
        permutation_seed=seed,
        rows=tuple(order_rows),
    )
    return OnlineRunArtifacts(
        admitted_panel=admitted_panel,
        cache_preparation_receipt=cache_preparation_receipt,
        execution_order_receipt=order_receipt,
        audit_records=tuple(audit_records),
        selected_decisions=tuple(selected.items()),
        failed_executions=tuple(failed),
    )
