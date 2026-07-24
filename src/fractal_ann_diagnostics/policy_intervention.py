"""Compile a label-independent policy intervention into sealed local artifacts."""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .compiled_policy import (
    CompiledMaskDescriptor,
    CompiledPolicyCatalog,
    CompiledPolicyError,
    CompiledPolicyMaskStore,
    load_compiled_policy_catalog,
    write_compiled_mask,
    write_compiled_policy_catalog,
)
from .policy import policy_environment_sha256
from .retrieval import PolicyTransitionEvidence, policy_mask_churn

POLICY_INTERVENTION_CONFIG_SCHEMA = "fractal-policy-intervention-config-v2"
TRIAL_SCHEDULE_SCHEMA = "fractal-policy-trial-schedule-v3"
INTERVENTION_RECEIPT_SCHEMA = "fractal-policy-intervention-receipt-v2"

NESTED_TRIALS_PER_FAMILY = 3
TRIAL_STATE_ASSIGNMENT_ALGORITHM = "sha256-config-seed-family-trial-rank-v1"

DEFAULT_ALLOW_RATE_STRATA = (0.25, 0.50, 0.75)
DEFAULT_POLICY_STATE_IDS = ("low", "medium", "high")
CONFIG_FILENAME = "intervention-config.json"
CATALOG_FILENAME = "compiled-policy-catalog.json"
OPA_DATA_FILENAME = "opa-data.json"
SCHEDULE_FILENAME = "trial-schedule.json"
RECEIPT_FILENAME = "intervention-receipt.json"

_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_OPA_DATA_BYTES = 8 * 1024 * 1024
_MAX_SCHEDULE_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
_FORBIDDEN_TOKENS = (
    "answer",
    "evidence",
    "gold",
    "judgment",
    "label",
    "outcome",
    "qrel",
    "relevance",
    "retrieved",
)

_CONFIG_FIELDS = frozenset(
    {
        "allow_rate_strata",
        "assignment_repetitions",
        "baseline_policy_revision",
        "baseline_seed_sha256",
        "grouped_execution_order",
        "policy_bundle_revision",
        "policy_state_ids",
        "schema_version",
        "seed_sha256",
        "subject_ids",
    }
)
_OPA_DATA_FIELDS = frozenset(
    {
        "assignments",
        "document_count",
        "document_universe_sha256",
        "mask_catalog_sha256",
        "policy_revision",
    }
)
_OPA_ASSIGNMENT_FIELDS = frozenset({"authorized_count", "mask_id", "mask_sha256"})
_SCHEDULE_ROW_FIELDS = frozenset(
    {
        "authorized_count",
        "baseline_authorized_count",
        "baseline_mask_byte_count",
        "baseline_mask_id",
        "baseline_mask_path",
        "baseline_mask_sha256",
        "baseline_policy_revision",
        "environment",
        "environment_sha256",
        "expected_policy_revision",
        "family_key",
        "group_order",
        "mask_id",
        "mask_sha256",
        "policy_churn",
        "realized_allow_rate",
        "repetition",
        "schedule_order",
        "subject",
        "trial_key",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "assignment_algorithm",
        "assignment_map_sha256",
        "assignment_seed_sha256",
        "baseline_policy_revision",
        "baseline_seed_sha256",
        "config_sha256",
        "corpus",
        "document_count",
        "document_universe_sha256",
        "execution_artifact_sha256",
        "grouped_execution_order",
        "mask_catalog_sha256",
        "policy_bundle_revision",
        "rows",
        "schema_version",
        "stage",
    }
)
_BOUND_ARTIFACT_FIELDS = frozenset({"byte_count", "path", "role", "sha256"})
_RECEIPT_FIELDS = frozenset(
    {
        "artifacts",
        "baseline_policy_revision",
        "baseline_seed_sha256",
        "config_sha256",
        "corpus",
        "document_count",
        "document_universe_sha256",
        "execution_artifact_sha256",
        "policy_bundle_revision",
        "seed_sha256",
        "schema_version",
        "stage",
        "transitions",
    }
)
_TRANSITION_FIELDS = frozenset(
    {
        "baseline_authorized_count",
        "baseline_mask_sha256",
        "baseline_policy_revision",
        "current_authorized_count",
        "current_mask_sha256",
        "current_policy_revision",
        "policy_churn",
        "policy_state",
        "schema_version",
    }
)


class PolicyInterventionError(ValueError):
    """Raised when intervention compilation or package admission fails."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolicyInterventionError(
            "policy intervention objects must be finite canonical JSON"
        ) from exc


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PolicyInterventionError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PolicyInterventionError(f"{label} contains non-finite value {value!r}")

    try:
        parsed = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as exc:
        raise PolicyInterventionError(f"{label} must be valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise PolicyInterventionError(f"{label} must be valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise PolicyInterventionError(f"{label} must contain one JSON object")
    return parsed


def _closed_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PolicyInterventionError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise PolicyInterventionError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PolicyInterventionError(f"{name} must be a non-empty canonical string")
    if unicodedata.normalize("NFC", value) != value:
        raise PolicyInterventionError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise PolicyInterventionError(f"{name} cannot contain control characters")
    return value


def _require_identifier(name: str, value: object) -> str:
    text = _require_text(name, value)
    if _IDENTIFIER.fullmatch(text) is None:
        raise PolicyInterventionError(f"{name} must be a lowercase filesystem-safe identifier")
    return text


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PolicyInterventionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_revision(name: str, value: object) -> str:
    if not isinstance(value, str) or _IMMUTABLE_REVISION.fullmatch(value) is None:
        raise PolicyInterventionError(
            f"{name} must be an immutable sha256:<lowercase digest> revision"
        )
    return value


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyInterventionError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyInterventionError(f"{name} must be a non-negative integer")
    return value


def _require_rate(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise PolicyInterventionError(f"{name} must be finite and strictly between 0 and 1")
    return float(value)


def _relative_path(name: str, value: object) -> str:
    text = _require_text(name, value)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or chr(92) in text
    ):
        raise PolicyInterventionError(f"{name} must be a canonical relative POSIX path")
    _reject_forbidden_path(text)
    return text


def _reject_forbidden_path(path: str) -> None:
    normalized = path.casefold()
    if any(token in normalized for token in _FORBIDDEN_TOKENS):
        raise PolicyInterventionError(f"forbidden response-bearing token appears in path {path!r}")


def _assert_no_forbidden_fields(value: object, *, path: str = "output") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold()
            if any(token in normalized for token in _FORBIDDEN_TOKENS):
                raise PolicyInterventionError(f"forbidden response-bearing field at {path}.{key}")
            _assert_no_forbidden_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, nested in enumerate(value):
            _assert_no_forbidden_fields(nested, path=f"{path}[{position}]")


def _require_array(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise PolicyInterventionError(f"{name} must be a JSON array")
    return value


@dataclass(frozen=True)
class PolicyInterventionConfig:
    """Frozen assignment and grouping contract for one intervention."""

    seed_sha256: str
    baseline_seed_sha256: str
    policy_bundle_revision: str
    baseline_policy_revision: str
    subject_ids: tuple[str, ...]
    assignment_repetitions: int = 1
    allow_rate_strata: tuple[float, ...] = DEFAULT_ALLOW_RATE_STRATA
    policy_state_ids: tuple[str, ...] = DEFAULT_POLICY_STATE_IDS
    grouped_execution_order: tuple[str, ...] = DEFAULT_POLICY_STATE_IDS
    schema_version: str = POLICY_INTERVENTION_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256("seed_sha256", self.seed_sha256)
        _require_sha256("baseline_seed_sha256", self.baseline_seed_sha256)
        _require_revision("policy_bundle_revision", self.policy_bundle_revision)
        _require_revision("baseline_policy_revision", self.baseline_policy_revision)
        if self.seed_sha256 == self.baseline_seed_sha256:
            raise PolicyInterventionError(
                "baseline_seed_sha256 must differ from the current policy seed"
            )
        if self.policy_bundle_revision == self.baseline_policy_revision:
            raise PolicyInterventionError(
                "baseline_policy_revision must differ from policy_bundle_revision"
            )
        if self.schema_version != POLICY_INTERVENTION_CONFIG_SCHEMA:
            raise PolicyInterventionError(
                f"schema_version must equal {POLICY_INTERVENTION_CONFIG_SCHEMA!r}"
            )
        subjects = tuple(self.subject_ids)
        states = tuple(self.policy_state_ids)
        group_order = tuple(self.grouped_execution_order)
        rates = tuple(
            _require_rate(f"allow_rate_strata[{position}]", value)
            for position, value in enumerate(self.allow_rate_strata)
        )
        if not subjects:
            raise PolicyInterventionError("subject_ids cannot be empty")
        for position, value in enumerate(subjects):
            _require_identifier(f"subject_ids[{position}]", value)
        if len(subjects) != len(set(subjects)):
            raise PolicyInterventionError("subject_ids must be unique")
        if not states:
            raise PolicyInterventionError("policy_state_ids cannot be empty")
        for position, value in enumerate(states):
            _require_identifier(f"policy_state_ids[{position}]", value)
        if len(states) != len(set(states)):
            raise PolicyInterventionError("policy_state_ids must be unique")
        if len(rates) != len(states):
            raise PolicyInterventionError(
                "allow_rate_strata and policy_state_ids must have equal length"
            )
        if rates != tuple(sorted(rates)) or len(rates) != len(set(rates)):
            raise PolicyInterventionError(
                "allow_rate_strata must be unique and strictly increasing"
            )
        if len(group_order) != len(states) or set(group_order) != set(states):
            raise PolicyInterventionError(
                "grouped_execution_order must be a permutation of policy_state_ids"
            )
        if len(group_order) != len(set(group_order)):
            raise PolicyInterventionError("grouped_execution_order cannot repeat a policy state")
        _require_positive_integer(
            "assignment_repetitions",
            self.assignment_repetitions,
        )
        if len(states) != NESTED_TRIALS_PER_FAMILY:
            raise PolicyInterventionError(
                "policy_state_ids must contain exactly three nested policy states"
            )
        block_count = len(states) * len(subjects) * self.assignment_repetitions
        if block_count != NESTED_TRIALS_PER_FAMILY:
            raise PolicyInterventionError(
                "subject_ids, policy_state_ids, and assignment_repetitions must "
                "define exactly three schedule blocks"
            )
        object.__setattr__(self, "subject_ids", subjects)
        object.__setattr__(self, "policy_state_ids", states)
        object.__setattr__(self, "grouped_execution_order", group_order)
        object.__setattr__(self, "allow_rate_strata", rates)
        _assert_no_forbidden_fields(self.to_dict(), path="config")

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_rate_strata": list(self.allow_rate_strata),
            "assignment_repetitions": self.assignment_repetitions,
            "baseline_policy_revision": self.baseline_policy_revision,
            "baseline_seed_sha256": self.baseline_seed_sha256,
            "grouped_execution_order": list(self.grouped_execution_order),
            "policy_bundle_revision": self.policy_bundle_revision,
            "policy_state_ids": list(self.policy_state_ids),
            "schema_version": self.schema_version,
            "seed_sha256": self.seed_sha256,
            "subject_ids": list(self.subject_ids),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> PolicyInterventionConfig:
        row = _closed_mapping(
            value,
            fields=_CONFIG_FIELDS,
            label="policy intervention config",
        )
        rates = _require_array("allow_rate_strata", row["allow_rate_strata"])
        subjects = _require_array("subject_ids", row["subject_ids"])
        states = _require_array("policy_state_ids", row["policy_state_ids"])
        group_order = _require_array(
            "grouped_execution_order",
            row["grouped_execution_order"],
        )
        return cls(
            seed_sha256=row["seed_sha256"],
            baseline_seed_sha256=row["baseline_seed_sha256"],
            policy_bundle_revision=row["policy_bundle_revision"],
            baseline_policy_revision=row["baseline_policy_revision"],
            subject_ids=tuple(subjects),
            assignment_repetitions=row["assignment_repetitions"],
            allow_rate_strata=tuple(rates),
            policy_state_ids=tuple(states),
            grouped_execution_order=tuple(group_order),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class _ExecutionTrial:
    trial_key: str
    family_key: str


@dataclass(frozen=True)
class _ExecutionView:
    corpus: str
    stage: str
    document_count: int
    document_universe_sha256: str
    trials: tuple[_ExecutionTrial, ...]
    artifact_sha256: str

    @property
    def trial_keys(self) -> tuple[str, ...]:
        return tuple(row.trial_key for row in self.trials)


def _admitted_execution_view(execution: object) -> _ExecutionView:
    try:
        corpus = execution.corpus  # type: ignore[attr-defined]
        stage = execution.stage  # type: ignore[attr-defined]
        document_count = execution.document_count  # type: ignore[attr-defined]
        universe = execution.document_universe_sha256  # type: ignore[attr-defined]
        trials = execution.trials  # type: ignore[attr-defined]
        artifact_sha256 = execution.artifact_sha256  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise PolicyInterventionError(
            "execution lacks the admitted inline-or-sharded interface"
        ) from exc
    _require_text("execution corpus", corpus)
    _require_text("execution stage", stage)
    _require_positive_integer("execution document_count", document_count)
    _require_sha256("execution document_universe_sha256", universe)
    _require_sha256("execution artifact_sha256", artifact_sha256)
    try:
        admitted_trials = tuple(
            _ExecutionTrial(
                trial_key=row.trial_key,
                family_key=row.family_key,
            )
            for row in trials
        )
    except (AttributeError, TypeError) as exc:
        raise PolicyInterventionError(
            "execution trials must expose opaque trial_key and family_key values"
        ) from exc
    if not admitted_trials:
        raise PolicyInterventionError("execution must contain at least one trial")
    for position, row in enumerate(admitted_trials):
        _require_sha256(f"execution trial_keys[{position}]", row.trial_key)
        _require_sha256(f"execution family_keys[{position}]", row.family_key)
    trial_keys = tuple(row.trial_key for row in admitted_trials)
    if len(trial_keys) != len(set(trial_keys)):
        raise PolicyInterventionError("execution contains duplicate trial keys")
    family_counts: dict[str, int] = {}
    for row in admitted_trials:
        family_counts[row.family_key] = family_counts.get(row.family_key, 0) + 1
    if any(count != NESTED_TRIALS_PER_FAMILY for count in family_counts.values()):
        raise PolicyInterventionError(
            "every execution family must contain exactly three nested trials"
        )
    admitted_trials = tuple(sorted(admitted_trials, key=lambda row: row.trial_key.encode("ascii")))
    return _ExecutionView(
        corpus=corpus,
        stage=stage,
        document_count=document_count,
        document_universe_sha256=universe,
        trials=admitted_trials,
        artifact_sha256=artifact_sha256,
    )


@dataclass(frozen=True, order=True)
class OPAMaskAssignment:
    """One subject and policy-state lookup value under data.fractal."""

    subject: str
    policy_state: str
    mask_id: str
    mask_sha256: str
    authorized_count: int

    def __post_init__(self) -> None:
        _require_identifier("assignment subject", self.subject)
        _require_identifier("assignment policy_state", self.policy_state)
        _require_identifier("assignment mask_id", self.mask_id)
        _require_sha256("assignment mask_sha256", self.mask_sha256)
        _require_positive_integer(
            "assignment authorized_count",
            self.authorized_count,
        )

    def decision_dict(self) -> dict[str, object]:
        return {
            "authorized_count": self.authorized_count,
            "mask_id": self.mask_id,
            "mask_sha256": self.mask_sha256,
        }


@dataclass(frozen=True)
class OPACompiledMaskData:
    """Exact JSON value mounted at data.fractal for opa_compiled_masks.rego."""

    document_count: int
    document_universe_sha256: str
    mask_catalog_sha256: str
    policy_revision: str
    assignments: tuple[OPAMaskAssignment, ...]

    def __post_init__(self) -> None:
        _require_positive_integer("OPA document_count", self.document_count)
        _require_sha256(
            "OPA document_universe_sha256",
            self.document_universe_sha256,
        )
        _require_sha256("OPA mask_catalog_sha256", self.mask_catalog_sha256)
        _require_revision("OPA policy_revision", self.policy_revision)
        assignments = tuple(self.assignments)
        if not assignments or not all(isinstance(row, OPAMaskAssignment) for row in assignments):
            raise PolicyInterventionError("OPA assignments must contain typed assignment rows")
        assignments = tuple(
            sorted(
                assignments,
                key=lambda row: (
                    row.subject.encode("ascii"),
                    row.policy_state.encode("ascii"),
                ),
            )
        )
        keys = [(row.subject, row.policy_state) for row in assignments]
        if len(keys) != len(set(keys)):
            raise PolicyInterventionError("OPA assignments repeat a subject and policy-state pair")
        if any(row.authorized_count >= self.document_count for row in assignments):
            raise PolicyInterventionError("OPA assignment mask must be neither empty nor full")
        object.__setattr__(self, "assignments", assignments)
        _assert_no_forbidden_fields(self.to_dict(), path="opa_data")

    def to_dict(self) -> dict[str, object]:
        assignments: dict[str, dict[str, dict[str, object]]] = {}
        for row in self.assignments:
            assignments.setdefault(row.subject, {})[row.policy_state] = row.decision_dict()
        return {
            "assignments": assignments,
            "document_count": self.document_count,
            "document_universe_sha256": self.document_universe_sha256,
            "mask_catalog_sha256": self.mask_catalog_sha256,
            "policy_revision": self.policy_revision,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> OPACompiledMaskData:
        row = _closed_mapping(
            value,
            fields=_OPA_DATA_FIELDS,
            label="OPA compiled-mask data",
        )
        assignment_object = row["assignments"]
        if not isinstance(assignment_object, Mapping) or not all(
            isinstance(key, str) for key in assignment_object
        ):
            raise PolicyInterventionError("OPA assignments must be an object keyed by subject")
        assignments: list[OPAMaskAssignment] = []
        for subject, state_object in assignment_object.items():
            if not isinstance(state_object, Mapping) or not all(
                isinstance(key, str) for key in state_object
            ):
                raise PolicyInterventionError(f"OPA assignments[{subject!r}] must be an object")
            for policy_state, assignment_value in state_object.items():
                assignment = _closed_mapping(
                    assignment_value,
                    fields=_OPA_ASSIGNMENT_FIELDS,
                    label=f"OPA assignment {subject}/{policy_state}",
                )
                assignments.append(
                    OPAMaskAssignment(
                        subject=subject,
                        policy_state=policy_state,
                        mask_id=assignment["mask_id"],
                        mask_sha256=assignment["mask_sha256"],
                        authorized_count=assignment["authorized_count"],
                    )
                )
        return cls(
            document_count=row["document_count"],
            document_universe_sha256=row["document_universe_sha256"],
            mask_catalog_sha256=row["mask_catalog_sha256"],
            policy_revision=row["policy_revision"],
            assignments=tuple(assignments),
        )


def _trial_state_rank_sha256(
    *,
    seed_sha256: str,
    family_key: str,
    trial_key: str,
) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "algorithm": TRIAL_STATE_ASSIGNMENT_ALGORITHM,
                "family_key": family_key,
                "seed_sha256": seed_sha256,
                "trial_key": trial_key,
            }
        )
    ).hexdigest()


def _trial_assignment_map_sha256(
    rows: tuple[TrialScheduleRow, ...],
) -> str:
    payload = b"".join(
        _canonical_bytes(
            {
                "family_key": row.family_key,
                "group_order": row.group_order,
                "policy_state": row.policy_state,
                "trial_key": row.trial_key,
            }
        )
        + b"\n"
        for row in sorted(
            rows,
            key=lambda item: item.trial_key.encode("ascii"),
        )
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TrialScheduleRow:
    """One policy assignment for one opaque trial repetition."""

    schedule_order: int
    group_order: int
    trial_key: str
    family_key: str
    repetition: int
    subject: str
    environment: tuple[tuple[str, object], ...]
    environment_sha256: str
    baseline_policy_revision: str
    baseline_mask_id: str
    baseline_mask_path: str
    baseline_mask_sha256: str
    baseline_mask_byte_count: int
    baseline_authorized_count: int
    mask_id: str
    mask_sha256: str
    authorized_count: int
    realized_allow_rate: float
    policy_churn: float
    expected_policy_revision: str

    def __post_init__(self) -> None:
        _require_nonnegative_integer("schedule_order", self.schedule_order)
        _require_nonnegative_integer("group_order", self.group_order)
        _require_sha256("schedule trial_key", self.trial_key)
        _require_sha256("schedule family_key", self.family_key)
        _require_nonnegative_integer("schedule repetition", self.repetition)
        _require_identifier("schedule subject", self.subject)
        if not isinstance(self.environment, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in self.environment
        ):
            raise PolicyInterventionError(
                "schedule environment must be an immutable key/value tuple"
            )
        environment = dict(self.environment)
        if len(environment) != len(self.environment) or set(environment) != {
            "assignment_repetition",
            "policy_state",
        }:
            raise PolicyInterventionError(
                "schedule environment must contain policy_state and assignment_repetition"
            )
        if environment["assignment_repetition"] != self.repetition:
            raise PolicyInterventionError("schedule environment repetition differs from its row")
        _require_identifier(
            "schedule environment policy_state",
            environment["policy_state"],
        )
        try:
            observed_environment_sha256 = policy_environment_sha256(environment)
        except (TypeError, ValueError) as exc:
            raise PolicyInterventionError(
                "schedule environment must be finite canonical JSON"
            ) from exc
        _require_sha256("schedule environment_sha256", self.environment_sha256)
        if observed_environment_sha256 != self.environment_sha256:
            raise PolicyInterventionError(
                "schedule environment digest differs from its finite environment"
            )
        _require_revision(
            "schedule baseline_policy_revision",
            self.baseline_policy_revision,
        )
        _require_identifier("schedule baseline_mask_id", self.baseline_mask_id)
        baseline_path = _relative_path(
            "schedule baseline_mask_path",
            self.baseline_mask_path,
        )
        if not baseline_path.startswith("baseline-masks/"):
            raise PolicyInterventionError("schedule baseline mask must remain in baseline-masks")
        _require_sha256("schedule baseline_mask_sha256", self.baseline_mask_sha256)
        _require_positive_integer(
            "schedule baseline_mask_byte_count",
            self.baseline_mask_byte_count,
        )
        _require_positive_integer(
            "schedule baseline_authorized_count",
            self.baseline_authorized_count,
        )
        _require_identifier("schedule mask_id", self.mask_id)
        _require_sha256("schedule mask_sha256", self.mask_sha256)
        _require_positive_integer(
            "schedule authorized_count",
            self.authorized_count,
        )
        rate = _require_rate(
            "schedule realized_allow_rate",
            self.realized_allow_rate,
        )
        object.__setattr__(self, "realized_allow_rate", rate)
        if (
            isinstance(self.policy_churn, bool)
            or not isinstance(self.policy_churn, (int, float))
            or not math.isfinite(float(self.policy_churn))
            or not 0.0 < float(self.policy_churn) <= 1.0
        ):
            raise PolicyInterventionError(
                "schedule policy_churn must be finite and non-zero in (0, 1]"
            )
        object.__setattr__(self, "policy_churn", float(self.policy_churn))
        _require_revision(
            "schedule expected_policy_revision",
            self.expected_policy_revision,
        )
        if self.baseline_policy_revision == self.expected_policy_revision:
            raise PolicyInterventionError(
                "schedule baseline and current policy revisions must differ"
            )
        if self.baseline_mask_sha256 == self.mask_sha256:
            raise PolicyInterventionError("schedule baseline and current policy masks must differ")

    @property
    def policy_state(self) -> str:
        return str(dict(self.environment)["policy_state"])

    def to_dict(self) -> dict[str, object]:
        return {
            "authorized_count": self.authorized_count,
            "baseline_authorized_count": self.baseline_authorized_count,
            "baseline_mask_byte_count": self.baseline_mask_byte_count,
            "baseline_mask_id": self.baseline_mask_id,
            "baseline_mask_path": self.baseline_mask_path,
            "baseline_mask_sha256": self.baseline_mask_sha256,
            "baseline_policy_revision": self.baseline_policy_revision,
            "environment": dict(self.environment),
            "environment_sha256": self.environment_sha256,
            "expected_policy_revision": self.expected_policy_revision,
            "family_key": self.family_key,
            "group_order": self.group_order,
            "mask_id": self.mask_id,
            "mask_sha256": self.mask_sha256,
            "policy_churn": self.policy_churn,
            "realized_allow_rate": self.realized_allow_rate,
            "repetition": self.repetition,
            "schedule_order": self.schedule_order,
            "subject": self.subject,
            "trial_key": self.trial_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> TrialScheduleRow:
        row = _closed_mapping(
            value,
            fields=_SCHEDULE_ROW_FIELDS,
            label="trial schedule row",
        )
        environment = row["environment"]
        if not isinstance(environment, Mapping) or not all(
            isinstance(key, str) for key in environment
        ):
            raise PolicyInterventionError("trial schedule environment must be an object")
        return cls(
            schedule_order=row["schedule_order"],
            group_order=row["group_order"],
            trial_key=row["trial_key"],
            family_key=row["family_key"],
            repetition=row["repetition"],
            subject=row["subject"],
            environment=tuple(
                sorted(environment.items(), key=lambda item: item[0].encode("utf-8"))
            ),
            environment_sha256=row["environment_sha256"],
            baseline_policy_revision=row["baseline_policy_revision"],
            baseline_mask_id=row["baseline_mask_id"],
            baseline_mask_path=row["baseline_mask_path"],
            baseline_mask_sha256=row["baseline_mask_sha256"],
            baseline_mask_byte_count=row["baseline_mask_byte_count"],
            baseline_authorized_count=row["baseline_authorized_count"],
            mask_id=row["mask_id"],
            mask_sha256=row["mask_sha256"],
            authorized_count=row["authorized_count"],
            realized_allow_rate=row["realized_allow_rate"],
            policy_churn=row["policy_churn"],
            expected_policy_revision=row["expected_policy_revision"],
        )


@dataclass(frozen=True)
class CanonicalTrialSchedule:
    """Grouped policy schedule with no action-order or response-side field."""

    execution_artifact_sha256: str
    corpus: str
    stage: str
    document_count: int
    document_universe_sha256: str
    config_sha256: str
    policy_bundle_revision: str
    mask_catalog_sha256: str
    grouped_execution_order: tuple[str, ...]
    assignment_seed_sha256: str
    baseline_seed_sha256: str
    baseline_policy_revision: str
    assignment_algorithm: str
    assignment_map_sha256: str
    rows: tuple[TrialScheduleRow, ...]
    schema_version: str = TRIAL_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256(
            "schedule execution_artifact_sha256",
            self.execution_artifact_sha256,
        )
        _require_text("schedule corpus", self.corpus)
        _require_text("schedule stage", self.stage)
        _require_positive_integer("schedule document_count", self.document_count)
        _require_sha256(
            "schedule document_universe_sha256",
            self.document_universe_sha256,
        )
        _require_sha256("schedule config_sha256", self.config_sha256)
        _require_revision(
            "schedule policy_bundle_revision",
            self.policy_bundle_revision,
        )
        _require_sha256(
            "schedule mask_catalog_sha256",
            self.mask_catalog_sha256,
        )
        _require_sha256(
            "schedule assignment_seed_sha256",
            self.assignment_seed_sha256,
        )
        _require_sha256(
            "schedule baseline_seed_sha256",
            self.baseline_seed_sha256,
        )
        _require_revision(
            "schedule baseline_policy_revision",
            self.baseline_policy_revision,
        )
        if self.assignment_seed_sha256 == self.baseline_seed_sha256:
            raise PolicyInterventionError("schedule baseline and current policy seeds must differ")
        if self.policy_bundle_revision == self.baseline_policy_revision:
            raise PolicyInterventionError(
                "schedule baseline and current policy revisions must differ"
            )
        _require_sha256(
            "schedule assignment_map_sha256",
            self.assignment_map_sha256,
        )
        if self.assignment_algorithm != TRIAL_STATE_ASSIGNMENT_ALGORITHM:
            raise PolicyInterventionError(
                "schedule assignment_algorithm differs from the frozen algorithm"
            )
        if self.schema_version != TRIAL_SCHEDULE_SCHEMA:
            raise PolicyInterventionError(
                f"schedule schema_version must equal {TRIAL_SCHEDULE_SCHEMA!r}"
            )
        groups = tuple(self.grouped_execution_order)
        if not groups:
            raise PolicyInterventionError("schedule grouped_execution_order cannot be empty")
        for position, group in enumerate(groups):
            _require_identifier(f"schedule group {position}", group)
        if len(groups) != len(set(groups)):
            raise PolicyInterventionError("schedule grouped_execution_order repeats a state")
        if len(groups) != NESTED_TRIALS_PER_FAMILY:
            raise PolicyInterventionError(
                "schedule grouped_execution_order must contain exactly three states"
            )
        rows = tuple(self.rows)
        if not rows or not all(isinstance(row, TrialScheduleRow) for row in rows):
            raise PolicyInterventionError("schedule rows must contain typed trial schedule rows")
        if [row.schedule_order for row in rows] != list(range(len(rows))):
            raise PolicyInterventionError("schedule_order must be contiguous and canonical")
        if len({row.subject for row in rows}) != 1 or {row.repetition for row in rows} != {0}:
            raise PolicyInterventionError("trial schedule must use one subject and one repetition")
        trial_keys = [row.trial_key for row in rows]
        if len(trial_keys) != len(set(trial_keys)):
            raise PolicyInterventionError("trial schedule assigns one trial more than once")
        prior_group = -1
        group_masks: dict[int, str] = {}
        group_transitions: dict[int, tuple[object, ...]] = {}
        for row in rows:
            if row.group_order < prior_group:
                raise PolicyInterventionError(
                    "trial schedule rows must be contiguous by mask group"
                )
            prior_group = row.group_order
            if not 0 <= row.group_order < len(groups):
                raise PolicyInterventionError(
                    "trial schedule group_order is outside grouped_execution_order"
                )
            if row.policy_state != groups[row.group_order]:
                raise PolicyInterventionError(
                    "trial schedule policy state differs from its group order"
                )
            prior_mask = group_masks.setdefault(row.group_order, row.mask_id)
            if prior_mask != row.mask_id:
                raise PolicyInterventionError("one grouped schedule block names more than one mask")
            transition = (
                row.baseline_mask_id,
                row.baseline_mask_path,
                row.baseline_mask_sha256,
                row.baseline_mask_byte_count,
                row.baseline_authorized_count,
                row.baseline_policy_revision,
                row.mask_sha256,
                row.authorized_count,
                row.expected_policy_revision,
                row.policy_churn,
            )
            prior_transition = group_transitions.setdefault(row.group_order, transition)
            if prior_transition != transition:
                raise PolicyInterventionError(
                    "one grouped schedule block names more than one policy transition"
                )
            if (
                row.expected_policy_revision != self.policy_bundle_revision
                or row.baseline_policy_revision != self.baseline_policy_revision
                or row.authorized_count >= self.document_count
                or row.baseline_authorized_count >= self.document_count
                or row.baseline_mask_byte_count != (self.document_count + 7) // 8
                or not math.isclose(
                    row.realized_allow_rate,
                    row.authorized_count / self.document_count,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise PolicyInterventionError("schedule row binding differs from its schedule")
        if set(group_masks) != set(range(len(groups))):
            raise PolicyInterventionError("trial schedule omits one or more mask groups")
        by_family: dict[str, list[TrialScheduleRow]] = {}
        for row in rows:
            by_family.setdefault(row.family_key, []).append(row)
        for family_key, family_rows in by_family.items():
            if len(family_rows) != NESTED_TRIALS_PER_FAMILY:
                raise PolicyInterventionError(
                    "every schedule family must contain exactly three nested trials"
                )
            ranked = sorted(
                family_rows,
                key=lambda row: (
                    _trial_state_rank_sha256(
                        seed_sha256=self.assignment_seed_sha256,
                        family_key=family_key,
                        trial_key=row.trial_key,
                    ),
                    row.trial_key,
                ),
            )
            if tuple(row.policy_state for row in ranked) != groups:
                raise PolicyInterventionError(
                    "schedule trial-to-state mapping differs from the frozen ranking"
                )
            if {row.group_order for row in family_rows} != set(range(len(groups))):
                raise PolicyInterventionError(
                    "one schedule family does not cover every policy state once"
                )
        observed_map_sha256 = _trial_assignment_map_sha256(rows)
        if observed_map_sha256 != self.assignment_map_sha256:
            raise PolicyInterventionError("schedule assignment map digest differs from its rows")
        object.__setattr__(self, "grouped_execution_order", groups)
        object.__setattr__(self, "rows", rows)
        _assert_no_forbidden_fields(self.to_dict(), path="schedule")

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_algorithm": self.assignment_algorithm,
            "assignment_map_sha256": self.assignment_map_sha256,
            "assignment_seed_sha256": self.assignment_seed_sha256,
            "baseline_policy_revision": self.baseline_policy_revision,
            "baseline_seed_sha256": self.baseline_seed_sha256,
            "config_sha256": self.config_sha256,
            "corpus": self.corpus,
            "document_count": self.document_count,
            "document_universe_sha256": self.document_universe_sha256,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "grouped_execution_order": list(self.grouped_execution_order),
            "mask_catalog_sha256": self.mask_catalog_sha256,
            "policy_bundle_revision": self.policy_bundle_revision,
            "rows": [row.to_dict() for row in self.rows],
            "schema_version": self.schema_version,
            "stage": self.stage,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> CanonicalTrialSchedule:
        row = _closed_mapping(
            value,
            fields=_SCHEDULE_FIELDS,
            label="canonical trial schedule",
        )
        groups = _require_array(
            "schedule grouped_execution_order",
            row["grouped_execution_order"],
        )
        rows = _require_array("schedule rows", row["rows"])
        return cls(
            execution_artifact_sha256=row["execution_artifact_sha256"],
            corpus=row["corpus"],
            stage=row["stage"],
            document_count=row["document_count"],
            document_universe_sha256=row["document_universe_sha256"],
            config_sha256=row["config_sha256"],
            policy_bundle_revision=row["policy_bundle_revision"],
            mask_catalog_sha256=row["mask_catalog_sha256"],
            grouped_execution_order=tuple(groups),
            assignment_seed_sha256=row["assignment_seed_sha256"],
            baseline_seed_sha256=row["baseline_seed_sha256"],
            baseline_policy_revision=row["baseline_policy_revision"],
            assignment_algorithm=row["assignment_algorithm"],
            assignment_map_sha256=row["assignment_map_sha256"],
            rows=tuple(TrialScheduleRow.from_dict(item) for item in rows),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True, order=True)
class BoundOutputArtifact:
    """Digest and size for one package payload."""

    path: str
    role: str
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path("output path", self.path))
        _require_identifier("output role", self.role)
        _require_sha256("output sha256", self.sha256)
        _require_positive_integer("output byte_count", self.byte_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> BoundOutputArtifact:
        row = _closed_mapping(
            value,
            fields=_BOUND_ARTIFACT_FIELDS,
            label="bound output artifact",
        )
        return cls(
            path=row["path"],
            role=row["role"],
            sha256=row["sha256"],
            byte_count=row["byte_count"],
        )


@dataclass(frozen=True, order=True)
class PolicyTransitionBinding:
    """Receipt row for one synthetic baseline-to-current policy mutation."""

    policy_state: str
    baseline_policy_revision: str
    current_policy_revision: str
    baseline_mask_sha256: str
    current_mask_sha256: str
    baseline_authorized_count: int
    current_authorized_count: int
    policy_churn: float
    schema_version: str = "fractal-policy-transition-binding-v1"

    def __post_init__(self) -> None:
        _require_identifier("transition policy_state", self.policy_state)
        _require_revision(
            "transition baseline_policy_revision",
            self.baseline_policy_revision,
        )
        _require_revision(
            "transition current_policy_revision",
            self.current_policy_revision,
        )
        if self.baseline_policy_revision == self.current_policy_revision:
            raise PolicyInterventionError(
                "transition baseline and current policy revisions must differ"
            )
        _require_sha256("transition baseline_mask_sha256", self.baseline_mask_sha256)
        _require_sha256("transition current_mask_sha256", self.current_mask_sha256)
        if self.baseline_mask_sha256 == self.current_mask_sha256:
            raise PolicyInterventionError(
                "transition baseline and current policy masks must differ"
            )
        _require_positive_integer(
            "transition baseline_authorized_count",
            self.baseline_authorized_count,
        )
        _require_positive_integer(
            "transition current_authorized_count",
            self.current_authorized_count,
        )
        if (
            isinstance(self.policy_churn, bool)
            or not isinstance(self.policy_churn, (int, float))
            or not math.isfinite(float(self.policy_churn))
            or not 0.0 < float(self.policy_churn) <= 1.0
        ):
            raise PolicyInterventionError(
                "transition policy_churn must be finite and non-zero in (0, 1]"
            )
        object.__setattr__(self, "policy_churn", float(self.policy_churn))
        if self.schema_version != "fractal-policy-transition-binding-v1":
            raise PolicyInterventionError("policy transition binding schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_authorized_count": self.baseline_authorized_count,
            "baseline_mask_sha256": self.baseline_mask_sha256,
            "baseline_policy_revision": self.baseline_policy_revision,
            "current_authorized_count": self.current_authorized_count,
            "current_mask_sha256": self.current_mask_sha256,
            "current_policy_revision": self.current_policy_revision,
            "policy_churn": self.policy_churn,
            "policy_state": self.policy_state,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> PolicyTransitionBinding:
        row = _closed_mapping(
            value,
            fields=_TRANSITION_FIELDS,
            label="policy transition binding",
        )
        return cls(**row)


@dataclass(frozen=True)
class PolicyInterventionReceipt:
    """Binding from source execution and config to every emitted payload."""

    execution_artifact_sha256: str
    corpus: str
    stage: str
    document_count: int
    document_universe_sha256: str
    config_sha256: str
    seed_sha256: str
    baseline_seed_sha256: str
    policy_bundle_revision: str
    baseline_policy_revision: str
    transitions: tuple[PolicyTransitionBinding, ...]
    artifacts: tuple[BoundOutputArtifact, ...]
    schema_version: str = INTERVENTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256(
            "receipt execution_artifact_sha256",
            self.execution_artifact_sha256,
        )
        _require_text("receipt corpus", self.corpus)
        _require_text("receipt stage", self.stage)
        _require_positive_integer("receipt document_count", self.document_count)
        _require_sha256(
            "receipt document_universe_sha256",
            self.document_universe_sha256,
        )
        _require_sha256("receipt config_sha256", self.config_sha256)
        _require_sha256("receipt seed_sha256", self.seed_sha256)
        _require_sha256("receipt baseline_seed_sha256", self.baseline_seed_sha256)
        _require_revision(
            "receipt policy_bundle_revision",
            self.policy_bundle_revision,
        )
        _require_revision(
            "receipt baseline_policy_revision",
            self.baseline_policy_revision,
        )
        if self.seed_sha256 == self.baseline_seed_sha256:
            raise PolicyInterventionError("receipt baseline and current policy seeds must differ")
        if self.policy_bundle_revision == self.baseline_policy_revision:
            raise PolicyInterventionError(
                "receipt baseline and current policy revisions must differ"
            )
        if self.schema_version != INTERVENTION_RECEIPT_SCHEMA:
            raise PolicyInterventionError(
                f"receipt schema_version must equal {INTERVENTION_RECEIPT_SCHEMA!r}"
            )
        artifacts = tuple(self.artifacts)
        if not artifacts or not all(isinstance(row, BoundOutputArtifact) for row in artifacts):
            raise PolicyInterventionError("receipt artifacts must contain bound output records")
        artifacts = tuple(sorted(artifacts, key=lambda row: row.path.encode("utf-8")))
        paths = [row.path for row in artifacts]
        roles = [row.role for row in artifacts]
        if len(paths) != len(set(paths)):
            raise PolicyInterventionError("receipt repeats an output path")
        if len(roles) != len(set(roles)):
            raise PolicyInterventionError("receipt repeats an output role")
        transition_values = tuple(self.transitions)
        if not all(isinstance(row, PolicyTransitionBinding) for row in transition_values):
            raise PolicyInterventionError("receipt transitions must contain typed policy mutations")
        transitions = tuple(
            sorted(transition_values, key=lambda row: row.policy_state.encode("ascii"))
        )
        if (
            len(transitions) != NESTED_TRIALS_PER_FAMILY
            or len({row.policy_state for row in transitions}) != len(transitions)
            or any(
                row.baseline_policy_revision != self.baseline_policy_revision
                or row.current_policy_revision != self.policy_bundle_revision
                or row.baseline_authorized_count >= self.document_count
                or row.current_authorized_count >= self.document_count
                for row in transitions
            )
        ):
            raise PolicyInterventionError(
                "receipt transitions do not define three closed policy mutations"
            )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "transitions", transitions)
        _assert_no_forbidden_fields(self.to_dict(), path="receipt")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifacts": [row.to_dict() for row in self.artifacts],
            "baseline_policy_revision": self.baseline_policy_revision,
            "baseline_seed_sha256": self.baseline_seed_sha256,
            "config_sha256": self.config_sha256,
            "corpus": self.corpus,
            "document_count": self.document_count,
            "document_universe_sha256": self.document_universe_sha256,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "policy_bundle_revision": self.policy_bundle_revision,
            "seed_sha256": self.seed_sha256,
            "schema_version": self.schema_version,
            "stage": self.stage,
            "transitions": [row.to_dict() for row in self.transitions],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> PolicyInterventionReceipt:
        row = _closed_mapping(
            value,
            fields=_RECEIPT_FIELDS,
            label="policy intervention receipt",
        )
        artifacts = _require_array("receipt artifacts", row["artifacts"])
        transitions = _require_array("receipt transitions", row["transitions"])
        return cls(
            execution_artifact_sha256=row["execution_artifact_sha256"],
            corpus=row["corpus"],
            stage=row["stage"],
            document_count=row["document_count"],
            document_universe_sha256=row["document_universe_sha256"],
            config_sha256=row["config_sha256"],
            seed_sha256=row["seed_sha256"],
            baseline_seed_sha256=row["baseline_seed_sha256"],
            policy_bundle_revision=row["policy_bundle_revision"],
            baseline_policy_revision=row["baseline_policy_revision"],
            transitions=tuple(PolicyTransitionBinding.from_dict(item) for item in transitions),
            artifacts=tuple(BoundOutputArtifact.from_dict(item) for item in artifacts),
            schema_version=row["schema_version"],
        )


@dataclass(frozen=True)
class GeneratedPolicyMask:
    policy_state: str
    allow_rate: float
    realized_allow_rate: float
    descriptor: CompiledMaskDescriptor
    encoded: bytes

    def __post_init__(self) -> None:
        _require_identifier("generated mask policy_state", self.policy_state)
        _require_rate("generated mask allow_rate", self.allow_rate)
        _require_rate(
            "generated mask realized_allow_rate",
            self.realized_allow_rate,
        )
        if not isinstance(self.descriptor, CompiledMaskDescriptor):
            raise PolicyInterventionError("generated mask descriptor has the wrong type")
        if not isinstance(self.encoded, bytes) or not self.encoded:
            raise PolicyInterventionError("generated mask bytes must be non-empty")
        if (
            len(self.encoded) != self.descriptor.byte_count
            or hashlib.sha256(self.encoded).hexdigest() != self.descriptor.sha256
        ):
            raise PolicyInterventionError("generated mask bytes differ from their descriptor")


@dataclass(frozen=True)
class CompiledPolicyIntervention:
    """Deterministic in-memory payloads ready for exclusive package writing."""

    config: PolicyInterventionConfig
    masks: tuple[GeneratedPolicyMask, ...]
    baseline_masks: tuple[GeneratedPolicyMask, ...]
    catalog: CompiledPolicyCatalog
    opa_data: OPACompiledMaskData
    schedule: CanonicalTrialSchedule
    receipt: PolicyInterventionReceipt

    def payloads(self) -> dict[str, bytes]:
        values = {
            CONFIG_FILENAME: self.config.canonical_file_bytes(),
            CATALOG_FILENAME: self.catalog.canonical_bytes() + b"\n",
            OPA_DATA_FILENAME: self.opa_data.canonical_file_bytes(),
            SCHEDULE_FILENAME: self.schedule.canonical_file_bytes(),
            RECEIPT_FILENAME: self.receipt.canonical_file_bytes(),
        }
        values.update({mask.descriptor.path: mask.encoded for mask in self.masks})
        values.update({mask.descriptor.path: mask.encoded for mask in self.baseline_masks})
        return values


def _rank_prefix(view: _ExecutionView) -> bytes:
    payload = _canonical_bytes(
        {
            "corpus": view.corpus,
            "document_universe_sha256": view.document_universe_sha256,
            "scheme": "hmac-sha256-row-threshold-v1",
        }
    )
    return len(payload).to_bytes(8, "big") + payload


def _rate_threshold(rate: float) -> int:
    return int(rate * (1 << 64))


def _compile_masks(
    view: _ExecutionView,
    config: PolicyInterventionConfig,
    *,
    seed_sha256: str,
    policy_revision: str,
    baseline: bool,
) -> tuple[GeneratedPolicyMask, ...]:
    byte_count = (view.document_count + 7) // 8
    encoded_masks = [bytearray(byte_count) for _ in config.allow_rate_strata]
    counts = [0 for _ in config.allow_rate_strata]
    thresholds = [_rate_threshold(rate) for rate in config.allow_rate_strata]
    template = hmac.new(
        bytes.fromhex(seed_sha256),
        _rank_prefix(view),
        digestmod=hashlib.sha256,
    )
    for document_id in range(view.document_count):
        digest = template.copy()
        digest.update(document_id.to_bytes(8, "big"))
        rank = int.from_bytes(digest.digest()[:8], "big")
        byte_position = document_id // 8
        bit = 1 << (document_id % 8)
        for position, threshold in enumerate(thresholds):
            if rank < threshold:
                encoded_masks[position][byte_position] |= bit
                counts[position] += 1

    masks: list[GeneratedPolicyMask] = []
    for position, (state, allow_rate, encoded_value, count) in enumerate(
        zip(
            config.policy_state_ids,
            config.allow_rate_strata,
            encoded_masks,
            counts,
        )
    ):
        if count <= 0 or count >= view.document_count:
            raise PolicyInterventionError(
                f"policy state {state!r} produced a degenerate empty or full mask"
            )
        binding_sha256 = hashlib.sha256(
            _canonical_bytes(
                {
                    "allow_rate": allow_rate,
                    "config_sha256": config.config_sha256,
                    "document_universe_sha256": view.document_universe_sha256,
                    "execution_artifact_sha256": view.artifact_sha256,
                    "policy_bundle_revision": policy_revision,
                    "policy_state": state,
                    "policy_epoch": "baseline" if baseline else "current",
                    "seed_sha256": seed_sha256,
                }
            )
        ).hexdigest()
        prefix = "baseline-allow" if baseline else "allow"
        mask_id = f"{prefix}-{position:02d}-{binding_sha256}"
        directory = "baseline-masks" if baseline else "masks"
        relative_path = f"{directory}/{mask_id}.bin"
        encoded = bytes(encoded_value)
        descriptor = CompiledMaskDescriptor(
            mask_id=mask_id,
            path=relative_path,
            sha256=hashlib.sha256(encoded).hexdigest(),
            byte_count=len(encoded),
            authorized_count=count,
        )
        masks.append(
            GeneratedPolicyMask(
                policy_state=state,
                allow_rate=allow_rate,
                realized_allow_rate=count / view.document_count,
                descriptor=descriptor,
                encoded=encoded,
            )
        )
    return tuple(masks)


def _opa_data(
    view: _ExecutionView,
    config: PolicyInterventionConfig,
    masks: tuple[GeneratedPolicyMask, ...],
    catalog: CompiledPolicyCatalog,
) -> OPACompiledMaskData:
    by_state = {row.policy_state: row for row in masks}
    assignments = tuple(
        OPAMaskAssignment(
            subject=subject,
            policy_state=state,
            mask_id=by_state[state].descriptor.mask_id,
            mask_sha256=by_state[state].descriptor.sha256,
            authorized_count=by_state[state].descriptor.authorized_count,
        )
        for subject in config.subject_ids
        for state in config.policy_state_ids
    )
    return OPACompiledMaskData(
        document_count=view.document_count,
        document_universe_sha256=view.document_universe_sha256,
        mask_catalog_sha256=catalog.artifact_sha256,
        policy_revision=config.policy_bundle_revision,
        assignments=assignments,
    )


def _trial_schedule(
    view: _ExecutionView,
    config: PolicyInterventionConfig,
    masks: tuple[GeneratedPolicyMask, ...],
    baseline_masks: tuple[GeneratedPolicyMask, ...],
    catalog: CompiledPolicyCatalog,
) -> CanonicalTrialSchedule:
    by_state = {row.policy_state: row for row in masks}
    baseline_by_state = {row.policy_state: row for row in baseline_masks}
    by_family: dict[str, list[_ExecutionTrial]] = {}
    for trial in view.trials:
        by_family.setdefault(trial.family_key, []).append(trial)
    assignments: dict[str, tuple[str, str]] = {}
    for family_key, family_trials in by_family.items():
        if len(family_trials) != NESTED_TRIALS_PER_FAMILY:
            raise PolicyInterventionError(
                "every execution family must contain exactly three nested trials"
            )
        ranked = sorted(
            family_trials,
            key=lambda row: (
                _trial_state_rank_sha256(
                    seed_sha256=config.seed_sha256,
                    family_key=family_key,
                    trial_key=row.trial_key,
                ),
                row.trial_key,
            ),
        )
        for group_order, trial in enumerate(ranked):
            assignments[trial.trial_key] = (
                config.grouped_execution_order[group_order],
                family_key,
            )
    if len(assignments) != len(view.trials):
        raise PolicyInterventionError(
            "trial-state assignment does not cover the exact execution trials"
        )
    rows: list[TrialScheduleRow] = []
    for group_order, state in enumerate(config.grouped_execution_order):
        mask = by_state[state]
        baseline_mask = baseline_by_state[state]
        descriptor = mask.descriptor
        baseline_descriptor = baseline_mask.descriptor
        churn = policy_mask_churn(
            np.unpackbits(
                np.frombuffer(baseline_mask.encoded, dtype=np.uint8),
                bitorder="little",
            )[: view.document_count].astype(bool),
            np.unpackbits(
                np.frombuffer(mask.encoded, dtype=np.uint8),
                bitorder="little",
            )[: view.document_count].astype(bool),
        )
        if churn == 0.0:
            raise PolicyInterventionError(
                f"policy state {state!r} produced a zero synthetic mutation; "
                "choose a different baseline seed before freeze"
            )
        subject = config.subject_ids[0]
        repetition = 0
        environment = {
            "assignment_repetition": repetition,
            "policy_state": state,
        }
        environment_sha256 = policy_environment_sha256(environment)
        frozen_environment = tuple(
            sorted(
                environment.items(),
                key=lambda item: item[0].encode("utf-8"),
            )
        )
        assigned_trials = sorted(
            (
                (trial_key, family_key)
                for trial_key, (assigned_state, family_key) in assignments.items()
                if assigned_state == state
            ),
            key=lambda item: item[0].encode("ascii"),
        )
        for trial_key, family_key in assigned_trials:
            rows.append(
                TrialScheduleRow(
                    schedule_order=len(rows),
                    group_order=group_order,
                    trial_key=trial_key,
                    family_key=family_key,
                    repetition=repetition,
                    subject=subject,
                    environment=frozen_environment,
                    environment_sha256=environment_sha256,
                    baseline_policy_revision=config.baseline_policy_revision,
                    baseline_mask_id=baseline_descriptor.mask_id,
                    baseline_mask_path=baseline_descriptor.path,
                    baseline_mask_sha256=baseline_descriptor.sha256,
                    baseline_mask_byte_count=baseline_descriptor.byte_count,
                    baseline_authorized_count=(baseline_descriptor.authorized_count),
                    mask_id=descriptor.mask_id,
                    mask_sha256=descriptor.sha256,
                    authorized_count=descriptor.authorized_count,
                    realized_allow_rate=mask.realized_allow_rate,
                    policy_churn=churn,
                    expected_policy_revision=config.policy_bundle_revision,
                )
            )
    frozen_rows = tuple(rows)
    return CanonicalTrialSchedule(
        execution_artifact_sha256=view.artifact_sha256,
        corpus=view.corpus,
        stage=view.stage,
        document_count=view.document_count,
        document_universe_sha256=view.document_universe_sha256,
        config_sha256=config.config_sha256,
        policy_bundle_revision=config.policy_bundle_revision,
        mask_catalog_sha256=catalog.artifact_sha256,
        grouped_execution_order=config.grouped_execution_order,
        assignment_seed_sha256=config.seed_sha256,
        baseline_seed_sha256=config.baseline_seed_sha256,
        baseline_policy_revision=config.baseline_policy_revision,
        assignment_algorithm=TRIAL_STATE_ASSIGNMENT_ALGORITHM,
        assignment_map_sha256=_trial_assignment_map_sha256(frozen_rows),
        rows=frozen_rows,
    )


def _bound_artifact(path: str, role: str, payload: bytes) -> BoundOutputArtifact:
    return BoundOutputArtifact(
        path=path,
        role=role,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
    )


def compile_policy_intervention(
    execution: object,
    config: PolicyInterventionConfig,
) -> CompiledPolicyIntervention:
    """Compile deterministic masks, OPA data, and a grouped trial schedule."""

    if not isinstance(config, PolicyInterventionConfig):
        raise PolicyInterventionError("config must be a frozen PolicyInterventionConfig")
    initial_view = _admitted_execution_view(execution)
    masks = _compile_masks(
        initial_view,
        config,
        seed_sha256=config.seed_sha256,
        policy_revision=config.policy_bundle_revision,
        baseline=False,
    )
    baseline_masks = _compile_masks(
        initial_view,
        config,
        seed_sha256=config.baseline_seed_sha256,
        policy_revision=config.baseline_policy_revision,
        baseline=True,
    )
    catalog = CompiledPolicyCatalog(
        document_count=initial_view.document_count,
        document_universe_sha256=initial_view.document_universe_sha256,
        policy_revision=config.policy_bundle_revision,
        masks=tuple(row.descriptor for row in masks),
    )
    opa_data = _opa_data(initial_view, config, masks, catalog)
    schedule = _trial_schedule(
        initial_view,
        config,
        masks,
        baseline_masks,
        catalog,
    )
    config_payload = config.canonical_file_bytes()
    catalog_payload = catalog.canonical_bytes() + b"\n"
    opa_payload = opa_data.canonical_file_bytes()
    schedule_payload = schedule.canonical_file_bytes()
    for label, payload, limit in (
        ("policy intervention config", config_payload, _MAX_CONFIG_BYTES),
        ("OPA compiled-mask data", opa_payload, _MAX_OPA_DATA_BYTES),
        ("trial schedule", schedule_payload, _MAX_SCHEDULE_BYTES),
    ):
        if len(payload) > limit:
            raise PolicyInterventionError(f"{label} exceeds its {limit}-byte control bound")
    artifacts = [
        _bound_artifact(CONFIG_FILENAME, "config", config_payload),
        _bound_artifact(CATALOG_FILENAME, "mask-catalog", catalog_payload),
        _bound_artifact(OPA_DATA_FILENAME, "opa-data", opa_payload),
        _bound_artifact(SCHEDULE_FILENAME, "trial-schedule", schedule_payload),
    ]
    artifacts.extend(
        _bound_artifact(
            row.descriptor.path,
            f"mask-{position:02d}",
            row.encoded,
        )
        for position, row in enumerate(masks)
    )
    artifacts.extend(
        _bound_artifact(
            row.descriptor.path,
            f"baseline-mask-{position:02d}",
            row.encoded,
        )
        for position, row in enumerate(baseline_masks)
    )
    current_by_state = {row.policy_state: row for row in masks}
    baseline_by_state = {row.policy_state: row for row in baseline_masks}
    churn_by_state = {row.policy_state: row.policy_churn for row in schedule.rows}
    transitions = tuple(
        PolicyTransitionBinding(
            policy_state=state,
            baseline_policy_revision=config.baseline_policy_revision,
            current_policy_revision=config.policy_bundle_revision,
            baseline_mask_sha256=baseline_by_state[state].descriptor.sha256,
            current_mask_sha256=current_by_state[state].descriptor.sha256,
            baseline_authorized_count=(baseline_by_state[state].descriptor.authorized_count),
            current_authorized_count=(current_by_state[state].descriptor.authorized_count),
            policy_churn=churn_by_state[state],
        )
        for state in config.policy_state_ids
    )
    receipt = PolicyInterventionReceipt(
        execution_artifact_sha256=initial_view.artifact_sha256,
        corpus=initial_view.corpus,
        stage=initial_view.stage,
        document_count=initial_view.document_count,
        document_universe_sha256=initial_view.document_universe_sha256,
        config_sha256=config.config_sha256,
        seed_sha256=config.seed_sha256,
        baseline_seed_sha256=config.baseline_seed_sha256,
        policy_bundle_revision=config.policy_bundle_revision,
        baseline_policy_revision=config.baseline_policy_revision,
        transitions=transitions,
        artifacts=tuple(artifacts),
    )
    if len(receipt.canonical_file_bytes()) > _MAX_RECEIPT_BYTES:
        raise PolicyInterventionError(f"intervention receipt exceeds {_MAX_RECEIPT_BYTES} bytes")
    final_view = _admitted_execution_view(execution)
    if final_view != initial_view:
        raise PolicyInterventionError("admitted execution source changed during policy compilation")
    return CompiledPolicyIntervention(
        config=config,
        masks=masks,
        baseline_masks=baseline_masks,
        catalog=catalog,
        opa_data=opa_data,
        schedule=schedule,
        receipt=receipt,
    )


def _loads_canonical(
    value: bytes | str,
    *,
    label: str,
    max_bytes: int,
    parser: Any,
) -> Any:
    if isinstance(value, str):
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise PolicyInterventionError(f"{label} must be valid UTF-8") from exc
    elif isinstance(value, bytes):
        encoded = value
    else:
        raise PolicyInterventionError(f"{label} must be bytes or text")
    if not encoded or len(encoded) > max_bytes:
        raise PolicyInterventionError(f"{label} is empty or exceeds {max_bytes} bytes")
    parsed = parser(_decode_object(encoded, label=label))
    if encoded != parsed.canonical_file_bytes():
        raise PolicyInterventionError(f"{label} bytes must be canonical JSON plus one newline")
    return parsed


def loads_policy_intervention_config(
    value: bytes | str,
) -> PolicyInterventionConfig:
    return _loads_canonical(
        value,
        label="policy intervention config",
        max_bytes=_MAX_CONFIG_BYTES,
        parser=PolicyInterventionConfig.from_dict,
    )


def loads_opa_compiled_mask_data(value: bytes | str) -> OPACompiledMaskData:
    return _loads_canonical(
        value,
        label="OPA compiled-mask data",
        max_bytes=_MAX_OPA_DATA_BYTES,
        parser=OPACompiledMaskData.from_dict,
    )


def loads_canonical_trial_schedule(
    value: bytes | str,
) -> CanonicalTrialSchedule:
    return _loads_canonical(
        value,
        label="canonical trial schedule",
        max_bytes=_MAX_SCHEDULE_BYTES,
        parser=CanonicalTrialSchedule.from_dict,
    )


def loads_policy_intervention_receipt(
    value: bytes | str,
) -> PolicyInterventionReceipt:
    return _loads_canonical(
        value,
        label="policy intervention receipt",
        max_bytes=_MAX_RECEIPT_BYTES,
        parser=PolicyInterventionReceipt.from_dict,
    )


def _load_control(
    path: str | Path,
    *,
    label: str,
    max_bytes: int,
    loader: Any,
) -> Any:
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=max_bytes,
            label=label,
        )
    except ArtifactIntegrityError as exc:
        raise PolicyInterventionError(f"cannot read {label} safely: {exc}") from exc
    return loader(encoded)


def load_policy_intervention_config(
    path: str | Path,
) -> PolicyInterventionConfig:
    return _load_control(
        path,
        label="policy intervention config",
        max_bytes=_MAX_CONFIG_BYTES,
        loader=loads_policy_intervention_config,
    )


def load_opa_compiled_mask_data(path: str | Path) -> OPACompiledMaskData:
    return _load_control(
        path,
        label="OPA compiled-mask data",
        max_bytes=_MAX_OPA_DATA_BYTES,
        loader=loads_opa_compiled_mask_data,
    )


def load_canonical_trial_schedule(
    path: str | Path,
) -> CanonicalTrialSchedule:
    return _load_control(
        path,
        label="canonical trial schedule",
        max_bytes=_MAX_SCHEDULE_BYTES,
        loader=loads_canonical_trial_schedule,
    )


def load_policy_intervention_receipt(
    path: str | Path,
) -> PolicyInterventionReceipt:
    return _load_control(
        path,
        label="policy intervention receipt",
        max_bytes=_MAX_RECEIPT_BYTES,
        loader=loads_policy_intervention_receipt,
    )


def load_baseline_policy_mask(
    package_root: str | Path,
    row: TrialScheduleRow,
    *,
    document_count: int,
) -> np.ndarray:
    """Load one receipt-bound baseline mask that is never served by OPA."""

    if not isinstance(row, TrialScheduleRow):
        raise PolicyInterventionError("baseline mask row must be a trial schedule row")
    if type(document_count) is not int or document_count <= 0:
        raise PolicyInterventionError("baseline mask document_count must be positive")
    if row.baseline_mask_byte_count != (document_count + 7) // 8:
        raise PolicyInterventionError("baseline mask byte count differs from the universe")
    root = Path(package_root)
    if not root.is_absolute():
        raise PolicyInterventionError("baseline mask package root must be absolute")
    path = root.joinpath(*PurePosixPath(row.baseline_mask_path).parts)
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=row.baseline_mask_byte_count,
            label="compiled baseline policy mask",
        )
    except ArtifactIntegrityError as exc:
        raise PolicyInterventionError(f"cannot read baseline mask safely: {exc}") from exc
    if (
        len(encoded) != row.baseline_mask_byte_count
        or hashlib.sha256(encoded).hexdigest() != row.baseline_mask_sha256
    ):
        raise PolicyInterventionError("baseline mask differs from its frozen row")
    unpacked = np.unpackbits(
        np.frombuffer(encoded, dtype=np.uint8),
        bitorder="little",
    )
    if np.any(unpacked[document_count:]):
        raise PolicyInterventionError("baseline mask has non-zero padding bits")
    mask = np.asarray(unpacked[:document_count], dtype=bool)
    if int(np.count_nonzero(mask)) != row.baseline_authorized_count:
        raise PolicyInterventionError("baseline authorized count differs from its frozen row")
    mask.setflags(write=False)
    return mask


def derive_policy_transition_evidence(
    package_root: str | Path,
    row: TrialScheduleRow,
    *,
    document_count: int,
    current_mask: np.ndarray,
) -> PolicyTransitionEvidence:
    """Recompute one schedule transition from both complete decision vectors."""

    baseline = load_baseline_policy_mask(
        package_root,
        row,
        document_count=document_count,
    )
    try:
        evidence = PolicyTransitionEvidence.derive(
            environment_sha256=row.environment_sha256,
            baseline_policy_revision=row.baseline_policy_revision,
            current_policy_revision=row.expected_policy_revision,
            baseline_mask=baseline,
            current_mask=current_mask,
            expected_baseline_mask_sha256=row.baseline_mask_sha256,
            expected_current_mask_sha256=row.mask_sha256,
            expected_baseline_authorized_count=row.baseline_authorized_count,
            expected_current_authorized_count=row.authorized_count,
        )
    except ValueError as exc:
        raise PolicyInterventionError(f"cannot derive policy transition evidence: {exc}") from exc
    if evidence.policy_churn != row.policy_churn:
        raise PolicyInterventionError(
            "schedule policy churn differs from the complete baseline/current masks"
        )
    return evidence


def _assert_schedule_coverage(
    view: _ExecutionView,
    config: PolicyInterventionConfig,
    schedule: CanonicalTrialSchedule,
) -> None:
    if (
        schedule.execution_artifact_sha256 != view.artifact_sha256
        or schedule.corpus != view.corpus
        or schedule.stage != view.stage
        or schedule.document_count != view.document_count
        or schedule.document_universe_sha256 != view.document_universe_sha256
        or schedule.config_sha256 != config.config_sha256
        or schedule.policy_bundle_revision != config.policy_bundle_revision
        or schedule.baseline_policy_revision != config.baseline_policy_revision
        or schedule.grouped_execution_order != config.grouped_execution_order
        or schedule.assignment_seed_sha256 != config.seed_sha256
        or schedule.baseline_seed_sha256 != config.baseline_seed_sha256
        or schedule.assignment_algorithm != TRIAL_STATE_ASSIGNMENT_ALGORITHM
    ):
        raise PolicyInterventionError("trial schedule source or config binding differs")
    expected_families = {row.trial_key: row.family_key for row in view.trials}
    observed_families = {row.trial_key: row.family_key for row in schedule.rows}
    if (
        observed_families != expected_families
        or len(schedule.rows) != len(view.trials)
        or {row.subject for row in schedule.rows} != set(config.subject_ids)
        or {row.repetition for row in schedule.rows} != {0}
    ):
        raise PolicyInterventionError(
            "trial schedule does not partition the exact registered trials"
        )


def _assert_opa_assignments(
    config: PolicyInterventionConfig,
    catalog: CompiledPolicyCatalog,
    opa_data: OPACompiledMaskData,
    masks: tuple[GeneratedPolicyMask, ...],
) -> None:
    if (
        opa_data.document_count != catalog.document_count
        or opa_data.document_universe_sha256 != catalog.document_universe_sha256
        or opa_data.mask_catalog_sha256 != catalog.artifact_sha256
        or opa_data.policy_revision != config.policy_bundle_revision
    ):
        raise PolicyInterventionError("OPA data binding differs from its mask catalog")
    by_state = {row.policy_state: row.descriptor for row in masks}
    expected = {
        (
            subject,
            state,
            by_state[state].mask_id,
            by_state[state].sha256,
            by_state[state].authorized_count,
        )
        for subject in config.subject_ids
        for state in config.policy_state_ids
    }
    observed = {
        (
            row.subject,
            row.policy_state,
            row.mask_id,
            row.mask_sha256,
            row.authorized_count,
        )
        for row in opa_data.assignments
    }
    if observed != expected:
        raise PolicyInterventionError("OPA assignments differ from the frozen subject/state matrix")


@dataclass(frozen=True)
class PolicyInterventionVerification:
    root: Path
    receipt_sha256: str
    catalog_sha256: str
    schedule_sha256: str
    mask_ids: tuple[str, ...]


def _verify_expected_package(
    package_root: str | Path,
    execution: object,
    expected: CompiledPolicyIntervention,
) -> PolicyInterventionVerification:
    root = Path(package_root)
    if not root.is_absolute():
        raise PolicyInterventionError("package_root must be an absolute path")
    payloads = expected.payloads()
    expected_entries = set(payloads)
    expected_entries.update({"baseline-masks", "masks"})
    try:
        tree = digest_directory_tree(root)
    except ArtifactIntegrityError as exc:
        raise PolicyInterventionError(f"cannot verify intervention package tree: {exc}") from exc
    if set(tree.entries) != expected_entries:
        missing = sorted(expected_entries - set(tree.entries))
        extra = sorted(set(tree.entries) - expected_entries)
        raise PolicyInterventionError(
            f"intervention package tree differs; missing={missing}, extra={extra}"
        )

    for relative_path, expected_payload in payloads.items():
        path = root.joinpath(*PurePosixPath(relative_path).parts)
        try:
            observed = read_secure_regular_file(
                path,
                max_bytes=len(expected_payload),
                label=f"intervention output {relative_path}",
            )
        except ArtifactIntegrityError as exc:
            raise PolicyInterventionError(
                f"cannot verify intervention output {relative_path!r}: {exc}"
            ) from exc
        if observed != expected_payload:
            raise PolicyInterventionError(
                f"intervention output {relative_path!r} differs from compilation"
            )

    actual_config = load_policy_intervention_config(root / CONFIG_FILENAME)
    actual_catalog = load_compiled_policy_catalog(root / CATALOG_FILENAME)
    actual_opa = load_opa_compiled_mask_data(root / OPA_DATA_FILENAME)
    actual_schedule = load_canonical_trial_schedule(root / SCHEDULE_FILENAME)
    actual_receipt = load_policy_intervention_receipt(root / RECEIPT_FILENAME)
    if (
        actual_config != expected.config
        or actual_catalog != expected.catalog
        or actual_opa != expected.opa_data
        or actual_schedule != expected.schedule
        or actual_receipt != expected.receipt
    ):
        raise PolicyInterventionError(
            "parsed intervention outputs differ from deterministic compilation"
        )
    store = CompiledPolicyMaskStore(root / CATALOG_FILENAME)
    try:
        verified_masks = store.verify_all()
    except CompiledPolicyError as exc:
        raise PolicyInterventionError(f"compiled mask self-verification failed: {exc}") from exc
    expected_mask_ids = tuple(row.descriptor.mask_id for row in expected.masks)
    if verified_masks != expected_mask_ids:
        raise PolicyInterventionError("compiled mask store coverage differs from the intervention")
    current_view = _admitted_execution_view(execution)
    _assert_schedule_coverage(current_view, expected.config, actual_schedule)
    _assert_opa_assignments(
        expected.config,
        actual_catalog,
        actual_opa,
        expected.masks,
    )
    receipt_rows = {row.path: (row.sha256, row.byte_count) for row in actual_receipt.artifacts}
    for path, payload in payloads.items():
        if path == RECEIPT_FILENAME:
            continue
        if receipt_rows.get(path) != (
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        ):
            raise PolicyInterventionError(f"receipt does not bind intervention output {path!r}")
    if set(receipt_rows) != set(payloads) - {RECEIPT_FILENAME}:
        raise PolicyInterventionError("receipt artifact set differs from emitted package outputs")
    return PolicyInterventionVerification(
        root=root,
        receipt_sha256=actual_receipt.artifact_sha256,
        catalog_sha256=actual_catalog.artifact_sha256,
        schedule_sha256=actual_schedule.artifact_sha256,
        mask_ids=verified_masks,
    )


def verify_policy_intervention_package(
    package_root: str | Path,
    execution: object,
    config: PolicyInterventionConfig,
) -> PolicyInterventionVerification:
    """Recompile expected bytes and admit one exact finalized package."""

    expected = compile_policy_intervention(execution, config)
    return _verify_expected_package(package_root, execution, expected)


def _directory_flags() -> int:
    missing = [name for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW") if not hasattr(os, name)]
    if missing or os.open not in os.supports_dir_fd:
        raise PolicyInterventionError(
            "atomic package finalization needs POSIX no-follow directory handles"
        )
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_error(label: str, exc: OSError) -> PolicyInterventionError:
    if exc.errno == errno.ENOENT:
        return PolicyInterventionError(f"{label} is missing")
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        return PolicyInterventionError(f"{label} crosses a symlink or non-directory ancestor")
    return PolicyInterventionError(f"cannot open {label}: {exc.strerror or str(exc)}")


def _open_private_directory(path: Path, *, label: str) -> int:
    if not path.is_absolute() or path.anchor != "/":
        raise PolicyInterventionError(f"{label} must be an absolute POSIX path")
    if any(part in {".", ".."} for part in path.parts):
        raise PolicyInterventionError(f"{label} cannot contain dot or parent traversal")
    flags = _directory_flags()
    try:
        descriptor = os.open("/", flags)
    except OSError as exc:
        raise _open_error(label, exc) from exc
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise _open_error(label, exc) from exc
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise PolicyInterventionError(f"{label} must be a directory")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise PolicyInterventionError(f"{label} must be owned by the runner identity")
        if stat.S_IMODE(metadata.st_mode) & 0o022:
            raise PolicyInterventionError(
                f"{label} cannot be writable by group or other identities"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _exists_at(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise _open_error(f"package target {name!r}", exc) from exc
    return True


def _write_control(payload: bytes, target: Path, *, label: str) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise PolicyInterventionError(f"{label} payload must be non-empty bytes")
    try:
        write_exclusive_receipt_bytes(payload, target)
    except ArtifactIntegrityError as exc:
        raise PolicyInterventionError(f"cannot write {label}: {exc}") from exc


def write_policy_intervention_package(
    execution: object,
    config: PolicyInterventionConfig,
    target: str | Path,
) -> PolicyInterventionVerification:
    """Compile, self-verify, then atomically rename one immutable package."""

    package = Path(target)
    if (
        not package.is_absolute()
        or package.name in {"", ".", ".."}
        or any(part in {".", ".."} for part in package.parts)
    ):
        raise PolicyInterventionError("package target must be an absolute canonical directory path")
    _reject_forbidden_path(package.name)
    parent_descriptor = _open_private_directory(
        package.parent,
        label="package target parent",
    )
    lock_name = f".{package.name}.policy-intervention.lock"
    lock_descriptor: int | None = None
    staging_name: str | None = None
    staging_path: Path | None = None
    finalized = False
    try:
        if _exists_at(parent_descriptor, package.name):
            raise PolicyInterventionError(f"package target already exists: {package}")
        lock_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            lock_descriptor = os.open(
                lock_name,
                lock_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError as exc:
            raise PolicyInterventionError(
                "another policy intervention compiler holds the package lock"
            ) from exc
        except OSError as exc:
            raise _open_error("policy intervention lock", exc) from exc

        compiled = compile_policy_intervention(execution, config)
        for _ in range(16):
            candidate = f".{package.name}.staging-{secrets.token_hex(12)}"
            try:
                os.mkdir(
                    candidate,
                    mode=0o700,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                raise _open_error(
                    "policy intervention staging directory",
                    exc,
                ) from exc
            staging_name = candidate
            break
        if staging_name is None:
            raise PolicyInterventionError(
                "cannot allocate a unique policy intervention staging directory"
            )
        staging_path = package.parent / staging_name
        masks_path = staging_path / "masks"
        masks_path.mkdir(mode=0o700)
        baseline_masks_path = staging_path / "baseline-masks"
        baseline_masks_path.mkdir(mode=0o700)

        _write_control(
            compiled.config.canonical_file_bytes(),
            staging_path / CONFIG_FILENAME,
            label="policy intervention config",
        )
        for mask in compiled.masks:
            try:
                write_compiled_mask(
                    mask.encoded,
                    staging_path.joinpath(*PurePosixPath(mask.descriptor.path).parts),
                )
            except CompiledPolicyError as exc:
                raise PolicyInterventionError(f"cannot write compiled mask: {exc}") from exc
        for mask in compiled.baseline_masks:
            try:
                write_compiled_mask(
                    mask.encoded,
                    staging_path.joinpath(*PurePosixPath(mask.descriptor.path).parts),
                )
            except CompiledPolicyError as exc:
                raise PolicyInterventionError(
                    f"cannot write compiled baseline mask: {exc}"
                ) from exc
        try:
            write_compiled_policy_catalog(
                compiled.catalog,
                staging_path / CATALOG_FILENAME,
            )
        except CompiledPolicyError as exc:
            raise PolicyInterventionError(f"cannot write compiled policy catalog: {exc}") from exc
        _write_control(
            compiled.opa_data.canonical_file_bytes(),
            staging_path / OPA_DATA_FILENAME,
            label="OPA compiled-mask data",
        )
        _write_control(
            compiled.schedule.canonical_file_bytes(),
            staging_path / SCHEDULE_FILENAME,
            label="canonical trial schedule",
        )
        _write_control(
            compiled.receipt.canonical_file_bytes(),
            staging_path / RECEIPT_FILENAME,
            label="policy intervention receipt",
        )
        verification = _verify_expected_package(
            staging_path,
            execution,
            compiled,
        )
        if _exists_at(parent_descriptor, package.name):
            raise PolicyInterventionError("package target appeared before atomic finalization")
        try:
            os.rename(
                staging_name,
                package.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise PolicyInterventionError(f"atomic package finalization failed: {exc}") from exc
        finalized = True
        os.fsync(parent_descriptor)
        return PolicyInterventionVerification(
            root=package,
            receipt_sha256=verification.receipt_sha256,
            catalog_sha256=verification.catalog_sha256,
            schedule_sha256=verification.schedule_sha256,
            mask_ids=verification.mask_ids,
        )
    finally:
        if not finalized and staging_path is not None and staging_path.exists():
            shutil.rmtree(staging_path)
        if lock_descriptor is not None:
            os.close(lock_descriptor)
            try:
                os.unlink(lock_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)
