"""Closed policy-workload artifacts for confirmatory retrieval experiments.

The workload contains only policy inputs.  Retrieval labels and measured outcomes
are rejected recursively, including when hidden inside nested attribute objects.
One canonical digest binds the complete workload to its ``policy-data`` manifest
artifact before sealed execution starts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, Union, cast

from .artifact_integrity import ArtifactIntegrityError, read_secure_regular_file

POLICY_WORKLOAD_SCHEMA = "fractal-policy-workload-v1"
POLICY_WORKLOAD_ARTIFACT_KIND = "policy-data"
SEEDED_TRIAL_ORDER_ALGORITHM = "sha256-seeded-permutation-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_MAX_PORTABLE_INTEGER = 2**53 - 1
_MAX_WORKLOAD_BYTES = 64 * 1024 * 1024

_ROOT_FIELDS = {
    "artifact_id",
    "artifact_kind",
    "corpus_id",
    "document_universe_sha256",
    "documents",
    "environments",
    "execution_schedule",
    "mutation_schedule",
    "mutations",
    "schema_version",
    "subjects",
    "trials",
}
_ENTITY_FIELDS = {
    "document": {"attributes", "document_id"},
    "environment": {"attributes", "environment_id"},
    "subject": {"attributes", "subject_id"},
}
_TRIAL_FIELDS = {
    "environment_id",
    "query_id",
    "subject_id",
    "trial_id",
}
_MUTATION_FIELDS = {
    "attributes",
    "mutation_id",
    "operation",
    "target_id",
    "target_kind",
}
_MUTATION_SCHEDULE_FIELDS = {"before_trial_id", "mutation_ids"}
_SEEDED_SCHEDULE_FIELDS = {"algorithm", "kind", "seed"}
_EXPLICIT_SCHEDULE_FIELDS = {"kind", "trial_order"}

_FORBIDDEN_FIELD_PREFIXES = (
    "answer",
    "correct",
    "evidence",
    "gold",
    "label",
    "outcome",
    "relev",
    "target",
)
_FORBIDDEN_FIELD_TOKENS = {
    "correctness",
    "groundtruth",
    "judgement",
    "judgment",
}

TargetKind: TypeAlias = Literal["document", "environment", "subject"]
JsonScalar: TypeAlias = None | bool | int | float | str


class PolicyWorkloadError(ValueError):
    """Raised when a policy workload is not closed, canonical, or label-free."""


def _canonical_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PolicyWorkloadError(f"cannot encode canonical policy workload: {exc}") from exc


def _closed_object(value: object, fields: set[str], *, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise PolicyWorkloadError(f"{path} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise PolicyWorkloadError(f"{path} field names must be JSON strings")
    observed = set(value)
    if observed != fields:
        missing = sorted(fields - observed)
        unknown = sorted(observed - fields)
        raise PolicyWorkloadError(f"{path} schema mismatch; missing={missing}, unknown={unknown}")
    return value


def _json_array(value: object, *, path: str) -> list[Any]:
    if type(value) is not list:
        raise PolicyWorkloadError(f"{path} must be a JSON array")
    return value


def _canonical_string(value: object, *, path: str, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise PolicyWorkloadError(f"{path} must be a JSON string")
    if (not allow_empty and not value) or value != value.strip():
        qualification = "canonical" if allow_empty else "non-empty canonical"
        raise PolicyWorkloadError(f"{path} must be a {qualification} string")
    if unicodedata.normalize("NFC", value) != value:
        raise PolicyWorkloadError(f"{path} must use NFC Unicode normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise PolicyWorkloadError(f"{path} cannot contain control characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise PolicyWorkloadError(f"{path} must be valid UTF-8") from exc
    return value


def _stable_identifier(value: object, *, path: str) -> str:
    identifier = _canonical_string(value, path=path)
    if _STABLE_ID.fullmatch(identifier) is None:
        raise PolicyWorkloadError(f"{path} must be an ASCII stable ID of at most 256 characters")
    return identifier


def _sha256(value: object, *, path: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PolicyWorkloadError(f"{path} must be a lowercase SHA-256 hex digest")
    return value


def _forbidden_attribute_field(name: str) -> bool:
    with_camel_boundaries = _CAMEL_BOUNDARY.sub("_", name)
    compact = _NON_ALPHANUMERIC.sub("", with_camel_boundaries.casefold())
    tokens = tuple(
        token for token in _NON_ALPHANUMERIC.split(with_camel_boundaries.casefold()) if token
    )
    if compact in _FORBIDDEN_FIELD_TOKENS:
        return True
    return any(token.startswith(prefix) for token in tokens for prefix in _FORBIDDEN_FIELD_PREFIXES)


FrozenJsonValue: TypeAlias = Union[JsonScalar, tuple["FrozenJsonValue", ...], "FrozenJsonObject"]


@dataclass(frozen=True)
class FrozenJsonObject(Mapping[str, FrozenJsonValue]):
    """An immutable JSON object with recursively frozen values."""

    entries: tuple[tuple[str, FrozenJsonValue], ...]

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise TypeError("FrozenJsonObject entries must be a tuple")
        keys: list[str] = []
        for position, entry in enumerate(self.entries):
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError(f"FrozenJsonObject entries[{position}] must be a key-value tuple")
            key, value = entry
            canonical_key = _canonical_string(key, path=f"FrozenJsonObject entries[{position}] key")
            if _forbidden_attribute_field(canonical_key):
                raise PolicyWorkloadError(
                    f"FrozenJsonObject.{canonical_key} is a forbidden label or outcome field"
                )
            _validate_frozen_json_value(value, path=f"FrozenJsonObject.{canonical_key}")
            keys.append(canonical_key)
        key_tuple = tuple(keys)
        if key_tuple != tuple(sorted(key_tuple)) or len(key_tuple) != len(set(key_tuple)):
            raise PolicyWorkloadError(
                "FrozenJsonObject keys must be unique and lexicographically sorted"
            )

    def __getitem__(self, key: str) -> FrozenJsonValue:
        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable JSON representation for serialization."""
        return {key: _thaw_json(value) for key, value in self.entries}


def _validate_frozen_json_value(value: object, *, path: str) -> None:
    if value is None or type(value) is bool or type(value) is str:
        if type(value) is str:
            _canonical_string(value, path=path, allow_empty=True)
        return
    if type(value) is int:
        if not -_MAX_PORTABLE_INTEGER <= value <= _MAX_PORTABLE_INTEGER:
            raise PolicyWorkloadError(f"{path} integer must be within the portable JSON range")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise PolicyWorkloadError(f"{path} number must be finite")
        if value == 0.0 and math.copysign(1.0, value) < 0:
            raise PolicyWorkloadError(f"{path} cannot use negative zero")
        return
    if type(value) is tuple:
        for position, item in enumerate(value):
            _validate_frozen_json_value(item, path=f"{path}[{position}]")
        return
    if isinstance(value, FrozenJsonObject):
        return
    raise TypeError(f"{path} must contain only recursively frozen JSON values")


def _freeze_json(value: object, *, path: str) -> FrozenJsonValue:
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if not -_MAX_PORTABLE_INTEGER <= value <= _MAX_PORTABLE_INTEGER:
            raise PolicyWorkloadError(f"{path} integer must be within the portable JSON range")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PolicyWorkloadError(f"{path} number must be finite")
        if value == 0.0 and math.copysign(1.0, value) < 0:
            raise PolicyWorkloadError(f"{path} cannot use negative zero")
        return value
    if type(value) is str:
        return _canonical_string(value, path=path, allow_empty=True)
    if type(value) is list:
        return tuple(
            _freeze_json(item, path=f"{path}[{position}]") for position, item in enumerate(value)
        )
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise PolicyWorkloadError(f"{path} field names must be JSON strings")
        entries: list[tuple[str, FrozenJsonValue]] = []
        for key in sorted(value):
            canonical_key = _canonical_string(key, path=f"{path} field name")
            if _forbidden_attribute_field(canonical_key):
                raise PolicyWorkloadError(
                    f"{path}.{canonical_key} is a forbidden label or outcome field"
                )
            entries.append(
                (
                    canonical_key,
                    _freeze_json(value[key], path=f"{path}.{canonical_key}"),
                )
            )
        return FrozenJsonObject(tuple(entries))
    raise PolicyWorkloadError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _freeze_attributes(value: object, *, path: str) -> FrozenJsonObject:
    if type(value) is not dict:
        raise PolicyWorkloadError(f"{path} must be a JSON object")
    frozen = _freeze_json(value, path=path)
    assert isinstance(frozen, FrozenJsonObject)
    return frozen


def _thaw_json(value: FrozenJsonValue) -> Any:
    if isinstance(value, FrozenJsonObject):
        return value.to_dict()
    if type(value) is tuple:
        return [_thaw_json(item) for item in value]
    return value


def _validate_frozen_attributes(value: object, *, path: str) -> FrozenJsonObject:
    if not isinstance(value, FrozenJsonObject):
        raise TypeError(f"{path} must be a FrozenJsonObject")
    return value


@dataclass(frozen=True, order=True)
class PolicySubject:
    """One stable subject identity and its initial policy attributes."""

    subject_id: str
    attributes: FrozenJsonObject

    def __post_init__(self) -> None:
        _stable_identifier(self.subject_id, path="subject_id")
        _validate_frozen_attributes(self.attributes, path="subject attributes")

    def to_dict(self) -> dict[str, Any]:
        return {"attributes": self.attributes.to_dict(), "subject_id": self.subject_id}


@dataclass(frozen=True, order=True)
class PolicyEnvironment:
    """One stable environment identity and its initial policy attributes."""

    environment_id: str
    attributes: FrozenJsonObject

    def __post_init__(self) -> None:
        _stable_identifier(self.environment_id, path="environment_id")
        _validate_frozen_attributes(self.attributes, path="environment attributes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributes": self.attributes.to_dict(),
            "environment_id": self.environment_id,
        }


@dataclass(frozen=True, order=True)
class PolicyDocument:
    """One stable document identity and its initial policy attributes."""

    document_id: str
    attributes: FrozenJsonObject

    def __post_init__(self) -> None:
        _stable_identifier(self.document_id, path="document_id")
        _validate_frozen_attributes(self.attributes, path="document attributes")

    def to_dict(self) -> dict[str, Any]:
        return {"attributes": self.attributes.to_dict(), "document_id": self.document_id}


@dataclass(frozen=True, order=True)
class PolicyTrial:
    """A fixed query, subject, and environment assignment for one trial."""

    trial_id: str
    query_id: str
    subject_id: str
    environment_id: str

    def __post_init__(self) -> None:
        for name in ("trial_id", "query_id", "subject_id", "environment_id"):
            _stable_identifier(getattr(self, name), path=name)

    def to_dict(self) -> dict[str, str]:
        return {
            "environment_id": self.environment_id,
            "query_id": self.query_id,
            "subject_id": self.subject_id,
            "trial_id": self.trial_id,
        }


@dataclass(frozen=True, order=True)
class PolicyMutation:
    """A deterministic attribute update applied at a scheduled trial boundary."""

    mutation_id: str
    target_kind: TargetKind
    target_id: str
    attributes: FrozenJsonObject
    operation: str = "set-attributes"

    def __post_init__(self) -> None:
        _stable_identifier(self.mutation_id, path="mutation_id")
        if self.target_kind not in {"document", "environment", "subject"}:
            raise PolicyWorkloadError("target_kind must be 'document', 'environment', or 'subject'")
        _stable_identifier(self.target_id, path="target_id")
        if self.operation != "set-attributes":
            raise PolicyWorkloadError("operation must equal 'set-attributes'")
        attributes = _validate_frozen_attributes(self.attributes, path="mutation attributes")
        if not attributes:
            raise PolicyWorkloadError("mutation attributes cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributes": self.attributes.to_dict(),
            "mutation_id": self.mutation_id,
            "operation": self.operation,
            "target_id": self.target_id,
            "target_kind": self.target_kind,
        }


@dataclass(frozen=True, order=True)
class MutationScheduleEntry:
    """Ordered mutations applied immediately before one trial."""

    before_trial_id: str
    mutation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _stable_identifier(self.before_trial_id, path="before_trial_id")
        mutation_ids = tuple(self.mutation_ids)
        object.__setattr__(self, "mutation_ids", mutation_ids)
        if not mutation_ids:
            raise PolicyWorkloadError("mutation_ids cannot be empty")
        for position, mutation_id in enumerate(mutation_ids):
            _stable_identifier(mutation_id, path=f"mutation_ids[{position}]")
        if len(mutation_ids) != len(set(mutation_ids)):
            raise PolicyWorkloadError("mutation_ids must be unique within an entry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_trial_id": self.before_trial_id,
            "mutation_ids": list(self.mutation_ids),
        }


@dataclass(frozen=True)
class SeededTrialSchedule:
    """Trial order derived from a portable SHA-256 seeded permutation."""

    seed: int
    kind: str = "seeded"
    algorithm: str = SEEDED_TRIAL_ORDER_ALGORITHM

    def __post_init__(self) -> None:
        if self.kind != "seeded":
            raise PolicyWorkloadError("seeded schedule kind must equal 'seeded'")
        if self.algorithm != SEEDED_TRIAL_ORDER_ALGORITHM:
            raise PolicyWorkloadError(
                f"seeded schedule algorithm must equal {SEEDED_TRIAL_ORDER_ALGORITHM!r}"
            )
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_PORTABLE_INTEGER:
            raise PolicyWorkloadError("seed must be a non-negative portable JSON integer")

    def ordered_trial_ids(self, trial_ids: Iterable[str]) -> tuple[str, ...]:
        seed_bytes = str(self.seed).encode("ascii")

        def key(trial_id: str) -> tuple[bytes, bytes]:
            digest = hashlib.sha256(
                b"fractal-policy-trial-order-v1\x00"
                + seed_bytes
                + b"\x00"
                + trial_id.encode("utf-8", errors="strict")
            ).digest()
            return digest, trial_id.encode("utf-8", errors="strict")

        return tuple(sorted(trial_ids, key=key))

    def to_dict(self) -> dict[str, str | int]:
        return {"algorithm": self.algorithm, "kind": self.kind, "seed": self.seed}


@dataclass(frozen=True)
class ExplicitTrialSchedule:
    """An explicitly enumerated deterministic trial order."""

    trial_order: tuple[str, ...]
    kind: str = "explicit"

    def __post_init__(self) -> None:
        if self.kind != "explicit":
            raise PolicyWorkloadError("explicit schedule kind must equal 'explicit'")
        trial_order = tuple(self.trial_order)
        object.__setattr__(self, "trial_order", trial_order)
        if not trial_order:
            raise PolicyWorkloadError("trial_order cannot be empty")
        for position, trial_id in enumerate(trial_order):
            _stable_identifier(trial_id, path=f"trial_order[{position}]")
        if len(trial_order) != len(set(trial_order)):
            raise PolicyWorkloadError("trial_order must contain unique trial IDs")

    def ordered_trial_ids(self, trial_ids: Iterable[str]) -> tuple[str, ...]:
        expected = set(trial_ids)
        observed = set(self.trial_order)
        if observed != expected:
            raise PolicyWorkloadError(
                "explicit trial_order must contain every declared trial exactly once"
            )
        return self.trial_order

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "trial_order": list(self.trial_order)}


TrialSchedule: TypeAlias = SeededTrialSchedule | ExplicitTrialSchedule


@dataclass(frozen=True)
class PolicyWorkload:
    """One corpus-bound, label-independent policy workload artifact."""

    artifact_id: str
    corpus_id: str
    document_universe_sha256: str
    subjects: tuple[PolicySubject, ...]
    environments: tuple[PolicyEnvironment, ...]
    documents: tuple[PolicyDocument, ...]
    mutations: tuple[PolicyMutation, ...]
    trials: tuple[PolicyTrial, ...]
    execution_schedule: TrialSchedule
    mutation_schedule: tuple[MutationScheduleEntry, ...]
    schema_version: str = POLICY_WORKLOAD_SCHEMA
    artifact_kind: str = POLICY_WORKLOAD_ARTIFACT_KIND

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_WORKLOAD_SCHEMA:
            raise PolicyWorkloadError(f"schema_version must equal {POLICY_WORKLOAD_SCHEMA!r}")
        if self.artifact_kind != POLICY_WORKLOAD_ARTIFACT_KIND:
            raise PolicyWorkloadError("artifact_kind must equal 'policy-data'")
        _stable_identifier(self.artifact_id, path="artifact_id")
        _stable_identifier(self.corpus_id, path="corpus_id")
        _sha256(self.document_universe_sha256, path="document_universe_sha256")

        for name in (
            "subjects",
            "environments",
            "documents",
            "mutations",
            "trials",
            "mutation_schedule",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        record_types = {
            "subjects": PolicySubject,
            "environments": PolicyEnvironment,
            "documents": PolicyDocument,
            "mutations": PolicyMutation,
            "trials": PolicyTrial,
            "mutation_schedule": MutationScheduleEntry,
        }
        for name, record_type in record_types.items():
            if any(not isinstance(value, record_type) for value in getattr(self, name)):
                raise TypeError(f"{name} must contain {record_type.__name__} records")
        if not isinstance(self.execution_schedule, (SeededTrialSchedule, ExplicitTrialSchedule)):
            raise TypeError(
                "execution_schedule must be SeededTrialSchedule or ExplicitTrialSchedule"
            )
        if not self.subjects or not self.environments or not self.documents or not self.trials:
            raise PolicyWorkloadError(
                "subjects, environments, documents, and trials must be non-empty"
            )

        self._require_sorted_unique(self.subjects, "subject_id", "subjects")
        self._require_sorted_unique(self.environments, "environment_id", "environments")
        self._require_sorted_unique(self.documents, "document_id", "documents")
        self._require_sorted_unique(self.mutations, "mutation_id", "mutations")
        self._require_sorted_unique(self.trials, "trial_id", "trials")

        subject_ids = {subject.subject_id for subject in self.subjects}
        environment_ids = {environment.environment_id for environment in self.environments}
        document_ids = {document.document_id for document in self.documents}
        trial_ids = tuple(trial.trial_id for trial in self.trials)
        for trial in self.trials:
            if trial.subject_id not in subject_ids:
                raise PolicyWorkloadError(
                    f"trial {trial.trial_id!r} references unknown subject {trial.subject_id!r}"
                )
            if trial.environment_id not in environment_ids:
                raise PolicyWorkloadError(
                    f"trial {trial.trial_id!r} references unknown environment "
                    f"{trial.environment_id!r}"
                )

        target_ids = {
            "document": document_ids,
            "environment": environment_ids,
            "subject": subject_ids,
        }
        for mutation in self.mutations:
            if mutation.target_id not in target_ids[mutation.target_kind]:
                raise PolicyWorkloadError(
                    f"mutation {mutation.mutation_id!r} references unknown "
                    f"{mutation.target_kind} {mutation.target_id!r}"
                )

        ordered_trial_ids = self.execution_schedule.ordered_trial_ids(trial_ids)
        trial_rank = {trial_id: position for position, trial_id in enumerate(ordered_trial_ids)}
        scheduled_trials = [entry.before_trial_id for entry in self.mutation_schedule]
        if len(scheduled_trials) != len(set(scheduled_trials)):
            raise PolicyWorkloadError("mutation_schedule can contain at most one entry per trial")
        unknown_schedule_trials = set(scheduled_trials) - set(trial_ids)
        if unknown_schedule_trials:
            raise PolicyWorkloadError(
                f"mutation_schedule references unknown trials: {sorted(unknown_schedule_trials)}"
            )
        schedule_ranks = tuple(trial_rank[trial_id] for trial_id in scheduled_trials)
        if schedule_ranks != tuple(sorted(schedule_ranks)):
            raise PolicyWorkloadError(
                "mutation_schedule entries must follow deterministic execution order"
            )

        declared_mutations = {mutation.mutation_id for mutation in self.mutations}
        scheduled_mutations = tuple(
            mutation_id for entry in self.mutation_schedule for mutation_id in entry.mutation_ids
        )
        if len(scheduled_mutations) != len(set(scheduled_mutations)):
            raise PolicyWorkloadError("each mutation must appear exactly once in mutation_schedule")
        if set(scheduled_mutations) != declared_mutations:
            missing = sorted(declared_mutations - set(scheduled_mutations))
            unknown = sorted(set(scheduled_mutations) - declared_mutations)
            raise PolicyWorkloadError(
                "mutation_schedule does not exactly cover mutations; "
                f"missing={missing}, unknown={unknown}"
            )

    @staticmethod
    def _require_sorted_unique(
        values: Sequence[object], id_field: str, collection_name: str
    ) -> None:
        identifiers = tuple(getattr(value, id_field) for value in values)
        if len(identifiers) != len(set(identifiers)):
            raise PolicyWorkloadError(f"{collection_name} contains duplicate stable IDs")
        if identifiers != tuple(sorted(identifiers)):
            raise PolicyWorkloadError(f"{collection_name} must be sorted by {id_field}")

    @property
    def ordered_trial_ids(self) -> tuple[str, ...]:
        """Return the fully determined execution order."""
        return self.execution_schedule.ordered_trial_ids(trial.trial_id for trial in self.trials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "corpus_id": self.corpus_id,
            "document_universe_sha256": self.document_universe_sha256,
            "documents": [document.to_dict() for document in self.documents],
            "environments": [environment.to_dict() for environment in self.environments],
            "execution_schedule": self.execution_schedule.to_dict(),
            "mutation_schedule": [entry.to_dict() for entry in self.mutation_schedule],
            "mutations": [mutation.to_dict() for mutation in self.mutations],
            "schema_version": self.schema_version,
            "subjects": [subject.to_dict() for subject in self.subjects],
            "trials": [trial.to_dict() for trial in self.trials],
        }

    def canonical_bytes(self) -> bytes:
        """Return the only byte representation used for workload identity."""
        return _canonical_bytes(self.to_dict())

    @property
    def canonical_sha256(self) -> str:
        """Return the lowercase SHA-256 digest of :meth:`canonical_bytes`."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, order=True)
class RequiredPolicyTrial:
    """The preregistered trial-to-query pairing used during validation."""

    trial_id: str
    query_id: str

    def __post_init__(self) -> None:
        _stable_identifier(self.trial_id, path="required trial_id")
        _stable_identifier(self.query_id, path="required query_id")


@dataclass(frozen=True)
class PolicyDataArtifactBinding:
    """Expected manifest identity for one canonical policy-data artifact."""

    artifact_id: str
    canonical_sha256: str
    artifact_kind: str = POLICY_WORKLOAD_ARTIFACT_KIND

    def __post_init__(self) -> None:
        _stable_identifier(self.artifact_id, path="expected artifact_id")
        _sha256(self.canonical_sha256, path="expected canonical_sha256")
        if self.artifact_kind != POLICY_WORKLOAD_ARTIFACT_KIND:
            raise PolicyWorkloadError("expected artifact_kind must equal 'policy-data'")


def _parse_entity(
    value: object, *, kind: str, position: int
) -> PolicySubject | PolicyEnvironment | PolicyDocument:
    path = f"{kind}s[{position}]"
    payload = _closed_object(value, _ENTITY_FIELDS[kind], path=path)
    identifier_field = f"{kind}_id"
    identifier = _stable_identifier(payload[identifier_field], path=f"{path}.{identifier_field}")
    attributes = _freeze_attributes(payload["attributes"], path=f"{path}.attributes")
    if kind == "subject":
        return PolicySubject(identifier, attributes)
    if kind == "environment":
        return PolicyEnvironment(identifier, attributes)
    return PolicyDocument(identifier, attributes)


def _parse_trial(value: object, position: int) -> PolicyTrial:
    path = f"trials[{position}]"
    payload = _closed_object(value, _TRIAL_FIELDS, path=path)
    return PolicyTrial(
        trial_id=_stable_identifier(payload["trial_id"], path=f"{path}.trial_id"),
        query_id=_stable_identifier(payload["query_id"], path=f"{path}.query_id"),
        subject_id=_stable_identifier(payload["subject_id"], path=f"{path}.subject_id"),
        environment_id=_stable_identifier(payload["environment_id"], path=f"{path}.environment_id"),
    )


def _parse_mutation(value: object, position: int) -> PolicyMutation:
    path = f"mutations[{position}]"
    payload = _closed_object(value, _MUTATION_FIELDS, path=path)
    target_kind = _canonical_string(payload["target_kind"], path=f"{path}.target_kind")
    if target_kind not in {"document", "environment", "subject"}:
        raise PolicyWorkloadError(
            f"{path}.target_kind must be 'document', 'environment', or 'subject'"
        )
    return PolicyMutation(
        mutation_id=_stable_identifier(payload["mutation_id"], path=f"{path}.mutation_id"),
        target_kind=cast(TargetKind, target_kind),
        target_id=_stable_identifier(payload["target_id"], path=f"{path}.target_id"),
        operation=_canonical_string(payload["operation"], path=f"{path}.operation"),
        attributes=_freeze_attributes(payload["attributes"], path=f"{path}.attributes"),
    )


def _parse_execution_schedule(value: object) -> TrialSchedule:
    if type(value) is not dict:
        raise PolicyWorkloadError("execution_schedule must be a JSON object")
    kind = _canonical_string(value.get("kind"), path="execution_schedule.kind")
    if kind == "seeded":
        payload = _closed_object(value, _SEEDED_SCHEDULE_FIELDS, path="execution_schedule")
        algorithm = _canonical_string(payload["algorithm"], path="execution_schedule.algorithm")
        seed = payload["seed"]
        if type(seed) is not int:
            raise PolicyWorkloadError("execution_schedule.seed must be a JSON integer")
        return SeededTrialSchedule(seed=seed, algorithm=algorithm)
    if kind == "explicit":
        payload = _closed_object(value, _EXPLICIT_SCHEDULE_FIELDS, path="execution_schedule")
        trial_order = tuple(
            _stable_identifier(trial_id, path=f"execution_schedule.trial_order[{position}]")
            for position, trial_id in enumerate(
                _json_array(payload["trial_order"], path="execution_schedule.trial_order")
            )
        )
        return ExplicitTrialSchedule(trial_order=trial_order)
    raise PolicyWorkloadError("execution_schedule.kind must be 'seeded' or 'explicit'")


def _parse_mutation_schedule_entry(value: object, position: int) -> MutationScheduleEntry:
    path = f"mutation_schedule[{position}]"
    payload = _closed_object(value, _MUTATION_SCHEDULE_FIELDS, path=path)
    mutation_ids = tuple(
        _stable_identifier(mutation_id, path=f"{path}.mutation_ids[{index}]")
        for index, mutation_id in enumerate(
            _json_array(payload["mutation_ids"], path=f"{path}.mutation_ids")
        )
    )
    return MutationScheduleEntry(
        before_trial_id=_stable_identifier(
            payload["before_trial_id"], path=f"{path}.before_trial_id"
        ),
        mutation_ids=mutation_ids,
    )


def parse_policy_workload(payload: object) -> PolicyWorkload:
    """Parse and validate a decoded JSON workload using the closed schema."""
    root = _closed_object(payload, _ROOT_FIELDS, path="policy workload")
    subjects = tuple(
        _parse_entity(value, kind="subject", position=position)
        for position, value in enumerate(_json_array(root["subjects"], path="subjects"))
    )
    environments = tuple(
        _parse_entity(value, kind="environment", position=position)
        for position, value in enumerate(_json_array(root["environments"], path="environments"))
    )
    documents = tuple(
        _parse_entity(value, kind="document", position=position)
        for position, value in enumerate(_json_array(root["documents"], path="documents"))
    )
    mutations = tuple(
        _parse_mutation(value, position)
        for position, value in enumerate(_json_array(root["mutations"], path="mutations"))
    )
    trials = tuple(
        _parse_trial(value, position)
        for position, value in enumerate(_json_array(root["trials"], path="trials"))
    )
    mutation_schedule = tuple(
        _parse_mutation_schedule_entry(value, position)
        for position, value in enumerate(
            _json_array(root["mutation_schedule"], path="mutation_schedule")
        )
    )
    return PolicyWorkload(
        artifact_id=_stable_identifier(root["artifact_id"], path="artifact_id"),
        artifact_kind=_canonical_string(root["artifact_kind"], path="artifact_kind"),
        corpus_id=_stable_identifier(root["corpus_id"], path="corpus_id"),
        document_universe_sha256=_sha256(
            root["document_universe_sha256"], path="document_universe_sha256"
        ),
        subjects=cast(tuple[PolicySubject, ...], subjects),
        environments=cast(tuple[PolicyEnvironment, ...], environments),
        documents=cast(tuple[PolicyDocument, ...], documents),
        mutations=mutations,
        trials=trials,
        execution_schedule=_parse_execution_schedule(root["execution_schedule"]),
        mutation_schedule=mutation_schedule,
        schema_version=_canonical_string(root["schema_version"], path="schema_version"),
    )


def loads_policy_workload(payload: str | bytes) -> PolicyWorkload:
    """Decode JSON and require its bytes to equal the canonical representation."""
    if type(payload) not in {str, bytes}:
        raise TypeError("payload must be str or bytes")
    if isinstance(payload, bytes):
        if len(payload) > _MAX_WORKLOAD_BYTES:
            raise PolicyWorkloadError("policy workload exceeds the 64 MiB limit")
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PolicyWorkloadError("policy workload must be valid UTF-8") from exc
    else:
        if len(payload.encode("utf-8", errors="strict")) > _MAX_WORKLOAD_BYTES:
            raise PolicyWorkloadError("policy workload exceeds the 64 MiB limit")
        text = payload

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PolicyWorkloadError(f"policy workload contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PolicyWorkloadError(f"policy workload contains non-finite number {value!r}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise PolicyWorkloadError(f"policy workload is not valid JSON: {exc.msg}") from exc
    workload = parse_policy_workload(decoded)
    try:
        supplied_bytes = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise PolicyWorkloadError("policy workload must be valid UTF-8") from exc
    if supplied_bytes != workload.canonical_bytes():
        raise PolicyWorkloadError(
            "policy workload bytes are not canonical; write canonical_bytes() exactly"
        )
    return workload


def load_policy_workload(path: str | Path) -> PolicyWorkload:
    """Load one bounded policy workload without following links or hard links."""
    workload_path = Path(path)
    try:
        payload = read_secure_regular_file(
            workload_path,
            max_bytes=_MAX_WORKLOAD_BYTES,
            label="policy workload",
        )
    except ArtifactIntegrityError as exc:
        raise PolicyWorkloadError(f"cannot load policy workload {workload_path}: {exc}") from exc
    return loads_policy_workload(payload)


def _normalize_expected_ids(values: Iterable[str], *, path: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{path} must be an iterable of stable IDs")
    identifiers = tuple(
        _stable_identifier(value, path=f"{path}[{position}]")
        for position, value in enumerate(values)
    )
    if len(identifiers) != len(set(identifiers)):
        raise PolicyWorkloadError(f"{path} contains duplicate stable IDs")
    return frozenset(identifiers)


def validate_policy_workload(
    workload: PolicyWorkload,
    *,
    expected_corpus_id: str,
    expected_document_universe_sha256: str,
    expected_document_ids: Iterable[str],
    expected_query_ids: Iterable[str],
    required_trials: Iterable[RequiredPolicyTrial],
    expected_artifact_binding: PolicyDataArtifactBinding,
) -> PolicyWorkload:
    """Validate one workload against preregistered identities and trial pairing."""
    if not isinstance(workload, PolicyWorkload):
        raise TypeError("workload must be a PolicyWorkload")
    corpus_id = _stable_identifier(expected_corpus_id, path="expected_corpus_id")
    universe_sha256 = _sha256(
        expected_document_universe_sha256,
        path="expected_document_universe_sha256",
    )
    if not isinstance(expected_artifact_binding, PolicyDataArtifactBinding):
        raise TypeError("expected_artifact_binding must be a PolicyDataArtifactBinding")
    document_ids = _normalize_expected_ids(expected_document_ids, path="expected_document_ids")
    if not document_ids:
        raise PolicyWorkloadError("expected_document_ids cannot be empty")
    query_ids = _normalize_expected_ids(expected_query_ids, path="expected_query_ids")
    if isinstance(required_trials, (str, bytes)):
        raise TypeError("required_trials must contain RequiredPolicyTrial records")
    trial_requirements = tuple(required_trials)
    if not trial_requirements:
        raise PolicyWorkloadError("required_trials cannot be empty")
    if any(not isinstance(trial, RequiredPolicyTrial) for trial in trial_requirements):
        raise TypeError("required_trials must contain RequiredPolicyTrial records")
    required_pairs = {trial.trial_id: trial.query_id for trial in trial_requirements}
    if len(required_pairs) != len(trial_requirements):
        raise PolicyWorkloadError("required_trials contains duplicate trial IDs")

    if workload.corpus_id != corpus_id:
        raise PolicyWorkloadError(
            f"corpus mismatch: observed={workload.corpus_id!r}, expected={corpus_id!r}"
        )
    if workload.document_universe_sha256 != universe_sha256:
        raise PolicyWorkloadError("document universe digest does not match sealed corpus")
    if workload.artifact_kind != expected_artifact_binding.artifact_kind:
        raise PolicyWorkloadError("policy-data artifact kind does not match manifest")
    if workload.artifact_id != expected_artifact_binding.artifact_id:
        raise PolicyWorkloadError("policy-data artifact ID does not match manifest")
    if workload.canonical_sha256 != expected_artifact_binding.canonical_sha256:
        raise PolicyWorkloadError("policy-data artifact digest does not match manifest")

    observed_query_ids = {trial.query_id for trial in workload.trials}
    if observed_query_ids != query_ids:
        missing = sorted(query_ids - observed_query_ids)
        unknown = sorted(observed_query_ids - query_ids)
        raise PolicyWorkloadError(f"query set mismatch; missing={missing}, unknown={unknown}")
    observed_document_ids = {document.document_id for document in workload.documents}
    if observed_document_ids != document_ids:
        missing = sorted(document_ids - observed_document_ids)
        unknown = sorted(observed_document_ids - document_ids)
        raise PolicyWorkloadError(
            f"document attribute coverage mismatch; missing={missing}, unknown={unknown}"
        )
    if not set(required_pairs.values()) <= query_ids:
        raise PolicyWorkloadError("required_trials contains a query outside expected_query_ids")
    observed_pairs = {trial.trial_id: trial.query_id for trial in workload.trials}
    if observed_pairs != required_pairs:
        missing = sorted(set(required_pairs) - set(observed_pairs))
        unknown = sorted(set(observed_pairs) - set(required_pairs))
        changed = sorted(
            trial_id
            for trial_id in set(observed_pairs) & set(required_pairs)
            if observed_pairs[trial_id] != required_pairs[trial_id]
        )
        raise PolicyWorkloadError(
            "trial-to-query pairing mismatch; "
            f"missing={missing}, unknown={unknown}, changed={changed}"
        )
    return workload


def _expected_mapping(value: object, *, corpora: frozenset[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping keyed by corpus ID")
    if any(type(key) is not str for key in value):
        raise PolicyWorkloadError(f"{path} keys must be corpus ID strings")
    observed = set(value)
    if observed != corpora:
        missing = sorted(corpora - observed)
        unknown = sorted(observed - corpora)
        raise PolicyWorkloadError(
            f"{path} corpus coverage mismatch; missing={missing}, unknown={unknown}"
        )
    return value


def validate_policy_workload_suite(
    workloads: Sequence[PolicyWorkload],
    *,
    expected_corpus_ids: Iterable[str],
    expected_document_universe_sha256: Mapping[str, str],
    expected_document_ids: Mapping[str, Iterable[str]],
    expected_query_ids: Mapping[str, Iterable[str]],
    required_trials: Mapping[str, Iterable[RequiredPolicyTrial]],
    expected_artifact_bindings: Mapping[str, PolicyDataArtifactBinding],
) -> tuple[PolicyWorkload, ...]:
    """Validate exact corpus coverage and all externally sealed bindings."""
    if isinstance(workloads, (str, bytes)):
        raise TypeError("workloads must be a sequence of PolicyWorkload records")
    workload_tuple = tuple(workloads)
    if any(not isinstance(workload, PolicyWorkload) for workload in workload_tuple):
        raise TypeError("workloads must contain PolicyWorkload records")
    corpora = _normalize_expected_ids(expected_corpus_ids, path="expected_corpus_ids")
    if not corpora:
        raise PolicyWorkloadError("expected_corpus_ids cannot be empty")
    workload_by_corpus = {workload.corpus_id: workload for workload in workload_tuple}
    if len(workload_by_corpus) != len(workload_tuple):
        raise PolicyWorkloadError("workloads contains duplicate corpus IDs")
    if set(workload_by_corpus) != corpora:
        missing = sorted(corpora - set(workload_by_corpus))
        unknown = sorted(set(workload_by_corpus) - corpora)
        raise PolicyWorkloadError(
            f"workload corpus coverage mismatch; missing={missing}, unknown={unknown}"
        )

    universe_map = _expected_mapping(
        expected_document_universe_sha256,
        corpora=corpora,
        path="expected_document_universe_sha256",
    )
    document_map = _expected_mapping(
        expected_document_ids, corpora=corpora, path="expected_document_ids"
    )
    query_map = _expected_mapping(expected_query_ids, corpora=corpora, path="expected_query_ids")
    trial_map = _expected_mapping(required_trials, corpora=corpora, path="required_trials")
    artifact_map = _expected_mapping(
        expected_artifact_bindings,
        corpora=corpora,
        path="expected_artifact_bindings",
    )
    validated = []
    for corpus_id in sorted(corpora):
        validated.append(
            validate_policy_workload(
                workload_by_corpus[corpus_id],
                expected_corpus_id=corpus_id,
                expected_document_universe_sha256=universe_map[corpus_id],
                expected_document_ids=document_map[corpus_id],
                expected_query_ids=query_map[corpus_id],
                required_trials=trial_map[corpus_id],
                expected_artifact_binding=artifact_map[corpus_id],
            )
        )
    return tuple(validated)
