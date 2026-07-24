"""Closed post-label input materialization and provider-claimed analysis operator.

The command-line path reconstructs ``LABELS_RELEASED`` through the registered
GitHub verifier.  Local configuration can locate custody evidence, but it
cannot select corpora, online outputs, label bytes, models, estimands, or code.
Those identities come from the verified suite state and frozen manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote_to_bytes, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_regular_file,
    load_verification_receipt,
    read_secure_control_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .confirmatory_analysis import (
    ActionPanelAdmissionReceipt,
    ActionPanelArtifact,
    ConfirmatoryInputArtifact,
    load_action_panel_admission_receipt,
    load_action_panel_artifact,
)
from .confirmatory_execution import (
    confirmatory_attempt_path,
    confirmatory_result_path,
    confirmatory_result_receipt_path,
    run_confirmatory_analysis_once,
)
from .confirmatory_modeling import (
    FrozenModelSuite,
    canonical_h1_model_artifact_bytes,
    canonical_h2_model_suite_artifact_bytes,
)
from .external_anchors import (
    VerifiedPredictionCompletionAnchor,
    verify_prediction_completion_anchor,
)
from .github_state_attestation import GitHubSuiteEvidenceVerifier
from .label_separation import (
    JoinedEvaluationTrial,
    OfflineEvaluationArtifact,
    PredictionArtifact,
    PredictionCompletionReceipt,
    SealedLabelArtifact,
    load_prediction_artifact,
    load_prediction_completion_receipt,
    load_sealed_label_artifact,
    sealed_run_receipt_sha256,
)
from .production_corpus_run import (
    PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
    RUNTIME_ATTESTATION_RECEIPT_FILENAME,
    RUNTIME_INVOCATION_MARKER_FILENAME,
    load_production_corpus_command_attempt,
)
from .runtime_attestation import load_runtime_attestation_receipt
from .sealed_online_execution import (
    SealedOnlineResultReceipt,
    load_sealed_online_attempt_receipt,
    load_sealed_online_result_receipt,
    sealed_online_attempt_path,
    sealed_online_result_path,
    verify_sealed_online_outputs,
)
from .study import (
    FIXED_CORPORA,
    load_sealed_run_receipt,
    load_study_manifest,
    manifest_sha256,
    validate_study_manifest,
)
from .suite_attempt import (
    LabelCorpusClosure,
    OnlineCorpusClosure,
    OnlineSuiteClosure,
    SuiteStateRecord,
    VerifiedPhaseClaimCapability,
    VerifiedProviderPredecessor,
    VerifiedSuiteLabelsReleased,
    complete_confirmatory_analysis,
    require_verified_labels_released,
    verify_suite_state,
)
from .timelock_release import (
    TimelockDecryptionReceipt,
    load_timelock_decryption_receipt,
)

CONFIRMATORY_INPUT_OPERATOR_CONFIG_SCHEMA = "fractal-confirmatory-input-operator-config-v1"
CONFIRMATORY_INPUT_MATERIALIZATION_RECEIPT_SCHEMA = (
    "fractal-confirmatory-input-materialization-receipt-v1"
)
CONFIRMATORY_INPUT_MEMBER_SCHEMA = "fractal-confirmatory-input-member-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INPUT_SUFFIX = ".confirmatory-input.json"
_INPUT_RECEIPT_SUFFIX = ".confirmatory-input-receipt.json"
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_MODEL_BYTES = 64 * 1024 * 1024


class ConfirmatoryInputOperatorError(ValueError):
    """Raised when the closed post-label operator cannot prove its inputs."""


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
        raise ConfirmatoryInputOperatorError(
            "confirmatory input evidence must be finite canonical JSON"
        ) from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ConfirmatoryInputOperatorError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ConfirmatoryInputOperatorError(f"{name} must be a canonical non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise ConfirmatoryInputOperatorError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfirmatoryInputOperatorError(f"{name} cannot contain control characters")
    return value


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ConfirmatoryInputOperatorError(f"{name} must be a positive integer")
    return value


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ConfirmatoryInputOperatorError(f"{label} must be a JSON object")
    observed = set(value)
    if observed != fields:
        raise ConfirmatoryInputOperatorError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _decode_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ConfirmatoryInputOperatorError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ConfirmatoryInputOperatorError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfirmatoryInputOperatorError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ConfirmatoryInputOperatorError(f"{label} must contain one JSON object")
    return value


def _file_uri(value: object, *, label: str, directory: bool | None = None) -> Path:
    uri = _require_text(label, value)
    try:
        parsed = urlsplit(uri)
    except ValueError as exc:
        raise ConfirmatoryInputOperatorError(f"{label} is not a valid URI") from exc
    if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
        raise ConfirmatoryInputOperatorError(f"{label} must be a canonical local file URI")
    if not parsed.path.startswith("/"):
        raise ConfirmatoryInputOperatorError(f"{label} must contain an absolute path")
    try:
        decoded = unquote_to_bytes(parsed.path).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ConfirmatoryInputOperatorError(f"{label} path must be valid UTF-8") from exc
    if (
        not decoded.startswith("/")
        or "\\" in decoded
        or unicodedata.normalize("NFC", decoded) != decoded
        or any(part in {"", ".", ".."} for part in decoded.split("/")[1:])
    ):
        raise ConfirmatoryInputOperatorError(f"{label} path is not canonical")
    path = Path(decoded)
    if not path.is_absolute() or path.anchor != "/" or path.as_uri() != uri:
        raise ConfirmatoryInputOperatorError(f"{label} must use canonical file URI encoding")
    if directory is not None:
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise ConfirmatoryInputOperatorError(f"cannot inspect {label}: {exc}") from exc
        expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
        if not expected or path.is_symlink():
            kind = "directory" if directory else "regular file"
            raise ConfirmatoryInputOperatorError(f"{label} must name a real {kind}")
    return path


def _ordered_corpora() -> tuple[str, ...]:
    return tuple(FIXED_CORPORA)


@dataclass(frozen=True)
class CorpusEvidenceLocation:
    """Locations for evidence whose identity is fixed by release receipts."""

    corpus_id: str
    prediction_completion_receipt_uri: str
    prediction_completion_anchor_record_uri: str
    prediction_completion_anchor_receipt_uri: str
    timelock_decryption_receipt_uri: str

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ConfirmatoryInputOperatorError("corpus evidence names an unregistered corpus")
        for name in (
            "prediction_completion_receipt_uri",
            "prediction_completion_anchor_record_uri",
            "prediction_completion_anchor_receipt_uri",
            "timelock_decryption_receipt_uri",
        ):
            _file_uri(getattr(self, name), label=name)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> CorpusEvidenceLocation:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="corpus evidence location",
        )
        return cls(**row)


@dataclass(frozen=True)
class ConfirmatoryInputOperatorConfig:
    """Closed locator set; scientific and corpus choices are deliberately absent."""

    suite_namespace_uri: str
    manifest_uri: str
    sealed_run_receipt_uri: str
    artifact_verification_receipt_uri: str
    artifact_root_uri: str
    corpus_evidence: tuple[CorpusEvidenceLocation, ...]
    schema_version: str = CONFIRMATORY_INPUT_OPERATOR_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFIRMATORY_INPUT_OPERATOR_CONFIG_SCHEMA:
            raise ConfirmatoryInputOperatorError(
                "confirmatory input operator config schema differs"
            )
        _file_uri(self.suite_namespace_uri, label="suite_namespace_uri", directory=True)
        _file_uri(self.manifest_uri, label="manifest_uri")
        _file_uri(self.sealed_run_receipt_uri, label="sealed_run_receipt_uri")
        _file_uri(
            self.artifact_verification_receipt_uri,
            label="artifact_verification_receipt_uri",
        )
        _file_uri(self.artifact_root_uri, label="artifact_root_uri", directory=True)
        rows = tuple(self.corpus_evidence)
        if not all(isinstance(row, CorpusEvidenceLocation) for row in rows):
            raise ConfirmatoryInputOperatorError(
                "corpus_evidence must contain typed evidence locations"
            )
        by_corpus = {row.corpus_id: row for row in rows}
        if len(by_corpus) != len(rows) or set(by_corpus) != set(FIXED_CORPORA):
            raise ConfirmatoryInputOperatorError(
                "corpus_evidence must cover each fixed corpus exactly once"
            )
        uris = [
            getattr(row, field)
            for row in rows
            for field in (
                "prediction_completion_receipt_uri",
                "prediction_completion_anchor_record_uri",
                "prediction_completion_anchor_receipt_uri",
                "timelock_decryption_receipt_uri",
            )
        ]
        if len(uris) != len(set(uris)):
            raise ConfirmatoryInputOperatorError("corpus evidence cannot reuse one file URI")
        object.__setattr__(
            self,
            "corpus_evidence",
            tuple(by_corpus[corpus_id] for corpus_id in _ordered_corpora()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_root_uri": self.artifact_root_uri,
            "artifact_verification_receipt_uri": self.artifact_verification_receipt_uri,
            "corpus_evidence": [row.to_dict() for row in self.corpus_evidence],
            "manifest_uri": self.manifest_uri,
            "schema_version": self.schema_version,
            "sealed_run_receipt_uri": self.sealed_run_receipt_uri,
            "suite_namespace_uri": self.suite_namespace_uri,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ConfirmatoryInputOperatorConfig:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="confirmatory input operator config",
        )
        evidence = row["corpus_evidence"]
        if not isinstance(evidence, list):
            raise ConfirmatoryInputOperatorError("corpus_evidence must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "corpus_evidence"},
            corpus_evidence=tuple(CorpusEvidenceLocation.from_dict(item) for item in evidence),
        )


def load_confirmatory_input_operator_config(
    path: str | Path,
) -> ConfirmatoryInputOperatorConfig:
    try:
        encoded = read_secure_control_file(path, label="confirmatory input operator config")
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryInputOperatorError(f"cannot load operator config: {exc}") from exc
    if len(encoded) > _MAX_CONFIG_BYTES:
        raise ConfirmatoryInputOperatorError("confirmatory input operator config is too large")
    config = ConfirmatoryInputOperatorConfig.from_dict(
        _decode_object(encoded, label="confirmatory input operator config")
    )
    if encoded != config.canonical_bytes() + b"\n":
        raise ConfirmatoryInputOperatorError("operator config bytes are not canonical")
    return config


@dataclass(frozen=True)
class ConfirmatoryInputMember:
    """One exact persisted source, with distinct semantic and file identities."""

    role: str
    corpus_id: str | None
    uri: str
    semantic_sha256: str
    file_sha256: str
    byte_count: int
    schema_version: str = CONFIRMATORY_INPUT_MEMBER_SCHEMA

    def __post_init__(self) -> None:
        _require_text("role", self.role)
        if self.corpus_id is not None and self.corpus_id not in FIXED_CORPORA:
            raise ConfirmatoryInputOperatorError("input member names an unregistered corpus")
        _file_uri(self.uri, label="input member URI")
        _require_sha256("semantic_sha256", self.semantic_sha256)
        _require_sha256("file_sha256", self.file_sha256)
        _positive_integer("byte_count", self.byte_count)
        if self.schema_version != CONFIRMATORY_INPUT_MEMBER_SCHEMA:
            raise ConfirmatoryInputOperatorError("input member schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> ConfirmatoryInputMember:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="input member")
        return cls(**row)


@dataclass(frozen=True)
class ConfirmatoryInputMaterializationReceipt:
    """Detached closure over the typed input and every admitted source file."""

    suite_attempt_id: str
    suite_state_record_sha256: str
    suite_descriptor_sha256: str
    manifest_sha256: str
    run_receipt_sha256: str
    artifact_uri: str
    artifact_sha256: str
    artifact_file_sha256: str
    artifact_byte_count: int
    members: tuple[ConfirmatoryInputMember, ...]
    schema_version: str = CONFIRMATORY_INPUT_MATERIALIZATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "suite_attempt_id",
            "suite_state_record_sha256",
            "suite_descriptor_sha256",
            "manifest_sha256",
            "run_receipt_sha256",
            "artifact_sha256",
            "artifact_file_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        artifact_path = _file_uri(self.artifact_uri, label="artifact_uri")
        if artifact_path.name != f"{self.manifest_sha256}{_INPUT_SUFFIX}":
            raise ConfirmatoryInputOperatorError("artifact_uri is not manifest-derived")
        _positive_integer("artifact_byte_count", self.artifact_byte_count)
        rows = tuple(self.members)
        if not rows or not all(isinstance(row, ConfirmatoryInputMember) for row in rows):
            raise ConfirmatoryInputOperatorError("members must contain typed source members")
        canonical = tuple(
            sorted(
                rows,
                key=lambda row: (
                    (row.corpus_id or "").encode("utf-8"),
                    row.role.encode("utf-8"),
                    row.uri.encode("utf-8"),
                ),
            )
        )
        if rows != canonical:
            raise ConfirmatoryInputOperatorError("input members are not canonically ordered")
        if len({row.uri for row in rows}) != len(rows):
            raise ConfirmatoryInputOperatorError("input members repeat a source URI")
        per_corpus = {corpus_id: set() for corpus_id in FIXED_CORPORA}
        for row in rows:
            if row.corpus_id is not None:
                per_corpus[row.corpus_id].add(row.role)
        required = {
            "action-panel",
            "action-panel-admission",
            "prediction-completion-anchor-receipt",
            "prediction-completion-anchor-record",
            "prediction-completion-receipt",
            "predictions",
            "released-sealed-labels",
            "sealed-online-result",
            "timelock-decryption-receipt",
        }
        if any(not required.issubset(roles) for roles in per_corpus.values()):
            raise ConfirmatoryInputOperatorError(
                "input members omit a required source for one or more fixed corpora"
            )
        if self.schema_version != CONFIRMATORY_INPUT_MATERIALIZATION_RECEIPT_SCHEMA:
            raise ConfirmatoryInputOperatorError("input materialization receipt schema differs")
        object.__setattr__(self, "members", canonical)

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_byte_count": self.artifact_byte_count,
            "artifact_file_sha256": self.artifact_file_sha256,
            "artifact_sha256": self.artifact_sha256,
            "artifact_uri": self.artifact_uri,
            "manifest_sha256": self.manifest_sha256,
            "members": [row.to_dict() for row in self.members],
            "run_receipt_sha256": self.run_receipt_sha256,
            "schema_version": self.schema_version,
            "suite_attempt_id": self.suite_attempt_id,
            "suite_descriptor_sha256": self.suite_descriptor_sha256,
            "suite_state_record_sha256": self.suite_state_record_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ConfirmatoryInputMaterializationReceipt:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="confirmatory input materialization receipt",
        )
        members = row["members"]
        if not isinstance(members, list):
            raise ConfirmatoryInputOperatorError("receipt members must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "members"},
            members=tuple(ConfirmatoryInputMember.from_dict(item) for item in members),
        )


@dataclass(frozen=True)
class MaterializedConfirmatoryInput:
    inputs: ConfirmatoryInputArtifact
    receipt: ConfirmatoryInputMaterializationReceipt
    artifact_path: Path
    receipt_path: Path


def confirmatory_input_path(inputs: ConfirmatoryInputArtifact) -> Path:
    """Return the fixed persisted-input path beside the attempt receipt."""

    attempt = confirmatory_attempt_path(inputs)
    return attempt.parent / f"{inputs.manifest_sha256}{_INPUT_SUFFIX}"


def confirmatory_input_receipt_path(inputs: ConfirmatoryInputArtifact) -> Path:
    """Return the fixed detached-input-receipt path."""

    attempt = confirmatory_attempt_path(inputs)
    return attempt.parent / f"{inputs.manifest_sha256}{_INPUT_RECEIPT_SUFFIX}"


def confirmatory_store_closure_filenames(manifest_sha256: str) -> tuple[str, ...]:
    """Return the exact two input and three outcome files admitted in the store."""

    digest = _require_sha256("manifest_sha256", manifest_sha256)
    from .confirmatory_execution import confirmatory_output_filenames

    return tuple(
        sorted(
            (
                f"{digest}{_INPUT_SUFFIX}",
                f"{digest}{_INPUT_RECEIPT_SUFFIX}",
                *confirmatory_output_filenames(digest),
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )


def _file_member(
    path: Path,
    *,
    role: str,
    semantic_sha256: str,
    corpus_id: str | None = None,
    expected_file_sha256: str | None = None,
    expected_byte_count: int | None = None,
) -> ConfirmatoryInputMember:
    try:
        metadata = path.stat(follow_symlinks=False)
        file_digest = digest_regular_file(path, label=f"confirmatory input member {role}")
    except (OSError, ArtifactIntegrityError) as exc:
        raise ConfirmatoryInputOperatorError(f"cannot admit input member {role}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or metadata.st_nlink != 1:
        raise ConfirmatoryInputOperatorError(f"input member {role} must be a singly linked file")
    if metadata.st_size <= 0:
        raise ConfirmatoryInputOperatorError(f"input member {role} cannot be empty")
    if expected_file_sha256 is not None and file_digest != expected_file_sha256:
        raise ConfirmatoryInputOperatorError(f"input member {role} file digest differs")
    if expected_byte_count is not None and metadata.st_size != expected_byte_count:
        raise ConfirmatoryInputOperatorError(f"input member {role} byte count differs")
    return ConfirmatoryInputMember(
        role=role,
        corpus_id=corpus_id,
        uri=path.as_uri(),
        semantic_sha256=semantic_sha256,
        file_sha256=file_digest,
        byte_count=metadata.st_size,
    )


def _manifest_artifact(manifest: Mapping[str, Any], *, role: str, corpus_id: str | None = None):
    matches = [
        row
        for row in manifest["artifacts"]
        if row["role"] == role and (corpus_id is None or row.get("corpus_id") == corpus_id)
    ]
    if len(matches) != 1:
        raise ConfirmatoryInputOperatorError(f"frozen manifest lacks one exact {role!r} artifact")
    return matches[0]


LabelsAuthority = VerifiedSuiteLabelsReleased | VerifiedProviderPredecessor


def _labels_state(token: LabelsAuthority) -> SuiteStateRecord:
    if isinstance(token, VerifiedSuiteLabelsReleased):
        return token.state
    if isinstance(token, VerifiedProviderPredecessor) and token.state.state == "ANALYSIS_CLAIMED":
        rows = [record for record in token.records if record.state == "LABELS_RELEASED"]
        if len(rows) == 1:
            return rows[0]
    if hasattr(token, "state") and hasattr(token, "descriptor_sha256"):
        return token.state  # type: ignore[return-value]
    raise ConfirmatoryInputOperatorError(
        "input materialization requires verified LABELS_RELEASED ancestry"
    )


def _labels_descriptor_sha256(token: LabelsAuthority) -> str:
    state = _labels_state(token)
    if isinstance(token, VerifiedSuiteLabelsReleased):
        return token.descriptor_sha256
    if isinstance(token, VerifiedProviderPredecessor):
        return token.evidences[state.sequence].descriptor_sha256
    return token.descriptor_sha256  # type: ignore[union-attr]


def _online_and_label_records(
    token: LabelsAuthority,
) -> tuple[SuiteStateRecord, OnlineSuiteClosure, tuple[LabelCorpusClosure, ...]]:
    online_records = [record for record in token.records if record.state == "ONLINE_COMPLETE"]
    if len(online_records) != 1 or not isinstance(online_records[0].payload, OnlineSuiteClosure):
        raise ConfirmatoryInputOperatorError("verified suite lacks one ONLINE_COMPLETE closure")
    labels = _labels_state(token).payload
    if not isinstance(labels, tuple) or not all(
        isinstance(row, LabelCorpusClosure) for row in labels
    ):
        raise ConfirmatoryInputOperatorError("verified LABELS_RELEASED payload is malformed")
    return online_records[0], online_records[0].payload, labels


def _assert_online_directory(
    closure: OnlineCorpusClosure,
    *,
    manifest_digest: str,
) -> tuple[
    SealedOnlineResultReceipt,
    PredictionArtifact,
    ActionPanelArtifact,
    ActionPanelAdmissionReceipt,
    tuple[ConfirmatoryInputMember, ...],
]:
    root = _file_uri(closure.output_uri, label="online output URI", directory=True)
    attempt_path = sealed_online_attempt_path(root, manifest_digest)
    result_path = sealed_online_result_path(root, manifest_digest)
    attempt = load_sealed_online_attempt_receipt(attempt_path)
    result = load_sealed_online_result_receipt(result_path)
    verify_sealed_online_outputs(result, output_root=root)
    if (
        attempt.receipt_sha256 != closure.attempt_receipt_sha256
        or result.receipt_sha256 != closure.result_receipt_sha256
        or result.attempt_receipt_sha256 != attempt.receipt_sha256
        or result.manifest_sha256 != manifest_digest
        or result.execution_artifact_sha256 != closure.execution_artifact_sha256
    ):
        raise ConfirmatoryInputOperatorError(
            "sealed online receipt bindings differ from suite state"
        )
    pins = {row.role: row for row in result.outputs}
    expected_names = {
        attempt_path.name,
        result_path.name,
        RUNTIME_ATTESTATION_RECEIPT_FILENAME,
        RUNTIME_INVOCATION_MARKER_FILENAME,
        PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME,
        *(row.filename for row in result.outputs),
    }
    if {path.name for path in root.iterdir()} != expected_names:
        raise ConfirmatoryInputOperatorError(
            "online output directory membership changed after ONLINE_COMPLETE"
        )
    runtime_path = root / RUNTIME_ATTESTATION_RECEIPT_FILENAME
    marker_path = root / RUNTIME_INVOCATION_MARKER_FILENAME
    command_path = root / PRODUCTION_CORPUS_COMMAND_ATTEMPT_FILENAME
    runtime = load_runtime_attestation_receipt(runtime_path)
    command = load_production_corpus_command_attempt(command_path)
    if (
        runtime.receipt_sha256 != closure.runtime_attestation_receipt_sha256
        or command.receipt_sha256 != closure.production_command_attempt_sha256
        or command.manifest_sha256 != manifest_digest
    ):
        raise ConfirmatoryInputOperatorError("online runtime evidence differs from suite state")
    prediction_path = root / pins["predictions"].filename
    panel_path = root / pins["action-panel"].filename
    admission_path = root / pins["action-panel-admission"].filename
    predictions = load_prediction_artifact(prediction_path)
    panel = load_action_panel_artifact(panel_path)
    admission = load_action_panel_admission_receipt(admission_path)
    expected_semantic = {
        "predictions": closure.prediction_artifact_sha256,
        "action-panel": closure.action_panel_artifact_sha256,
        "action-panel-admission": closure.action_panel_admission_receipt_sha256,
        "audit-chain": closure.audit_head_sha256,
        "cache-preparation": closure.cache_preparation_receipt_sha256,
        "execution-order": closure.execution_order_receipt_sha256,
    }
    expected_files = {
        "predictions": closure.prediction_file_sha256,
        "action-panel": closure.action_panel_file_sha256,
        "action-panel-admission": closure.action_panel_admission_file_sha256,
        "audit-chain": closure.audit_file_sha256,
        "cache-preparation": closure.cache_preparation_file_sha256,
        "execution-order": closure.execution_order_file_sha256,
    }
    members: list[ConfirmatoryInputMember] = [
        _file_member(
            attempt_path,
            role="sealed-online-attempt",
            corpus_id=closure.corpus_id,
            semantic_sha256=attempt.receipt_sha256,
            expected_file_sha256=closure.attempt_file_sha256,
        ),
        _file_member(
            result_path,
            role="sealed-online-result",
            corpus_id=closure.corpus_id,
            semantic_sha256=result.receipt_sha256,
            expected_file_sha256=closure.result_file_sha256,
        ),
        _file_member(
            runtime_path,
            role="runtime-attestation-receipt",
            corpus_id=closure.corpus_id,
            semantic_sha256=runtime.receipt_sha256,
            expected_file_sha256=closure.runtime_attestation_receipt_file_sha256,
        ),
        _file_member(
            marker_path,
            role="runtime-invocation-marker",
            corpus_id=closure.corpus_id,
            semantic_sha256=closure.runtime_invocation_marker_sha256,
            expected_file_sha256=closure.runtime_invocation_marker_file_sha256,
        ),
        _file_member(
            command_path,
            role="production-command-attempt",
            corpus_id=closure.corpus_id,
            semantic_sha256=command.receipt_sha256,
            expected_file_sha256=closure.production_command_attempt_file_sha256,
        ),
    ]
    for role, pin in pins.items():
        if (
            pin.semantic_sha256 != expected_semantic[role]
            or pin.file_sha256 != expected_files[role]
        ):
            raise ConfirmatoryInputOperatorError(f"online {role} pin differs from suite state")
        members.append(
            _file_member(
                root / pin.filename,
                role=role,
                corpus_id=closure.corpus_id,
                semantic_sha256=pin.semantic_sha256,
                expected_file_sha256=pin.file_sha256,
                expected_byte_count=pin.byte_count,
            )
        )
    if (
        predictions.artifact_sha256 != closure.prediction_artifact_sha256
        or panel.artifact_sha256 != closure.action_panel_artifact_sha256
        or admission.receipt_sha256 != closure.action_panel_admission_receipt_sha256
    ):
        raise ConfirmatoryInputOperatorError("typed online artifacts differ from suite state")
    return result, predictions, panel, admission, tuple(members)


def _assert_release_chain(
    location: CorpusEvidenceLocation,
    *,
    label_closure: LabelCorpusClosure,
    online_closure: OnlineCorpusClosure,
    result: SealedOnlineResultReceipt,
    predictions: PredictionArtifact,
    panel: ActionPanelArtifact,
    manifest_digest: str,
    run_receipt_digest: str,
    trusted_anchor_record_fetcher: Callable[[str, int], bytes] | None,
) -> tuple[
    PredictionCompletionReceipt,
    VerifiedPredictionCompletionAnchor,
    TimelockDecryptionReceipt,
    SealedLabelArtifact,
    OfflineEvaluationArtifact,
    tuple[ConfirmatoryInputMember, ...],
]:
    completion_path = _file_uri(
        location.prediction_completion_receipt_uri,
        label="prediction completion receipt URI",
    )
    anchor_record_path = _file_uri(
        location.prediction_completion_anchor_record_uri,
        label="prediction completion anchor record URI",
    )
    anchor_receipt_path = _file_uri(
        location.prediction_completion_anchor_receipt_uri,
        label="prediction completion anchor receipt URI",
    )
    decryption_path = _file_uri(
        location.timelock_decryption_receipt_uri,
        label="timelock decryption receipt URI",
    )
    completion = load_prediction_completion_receipt(completion_path)
    verified_anchor = verify_prediction_completion_anchor(
        completion,
        anchor_record_path=anchor_record_path,
        anchor_receipt_path=anchor_receipt_path,
        trusted_anchor_record_fetcher=trusted_anchor_record_fetcher,
    )
    decryption = load_timelock_decryption_receipt(decryption_path)
    plaintext_path = _file_uri(label_closure.plaintext_uri, label="released plaintext URI")
    labels = load_sealed_label_artifact(plaintext_path)
    anchor_record = verified_anchor.record
    anchor_receipt = verified_anchor.receipt
    expected = (
        ("completion corpus", completion.corpus, location.corpus_id),
        ("completion manifest", completion.manifest_sha256, manifest_digest),
        ("completion run", completion.run_receipt_sha256, run_receipt_digest),
        (
            "completion online result",
            completion.online_execution_result_receipt_sha256,
            result.receipt_sha256,
        ),
        (
            "completion prediction",
            completion.prediction_artifact_sha256,
            predictions.artifact_sha256,
        ),
        (
            "completion panel",
            completion.action_panel_binding.action_panel_artifact_sha256,
            panel.artifact_sha256,
        ),
        ("decryption corpus", decryption.corpus_id, location.corpus_id),
        ("decryption manifest", decryption.manifest_sha256, manifest_digest),
        (
            "decryption online result",
            decryption.online_execution_result_receipt_sha256,
            result.receipt_sha256,
        ),
        (
            "decryption anchor record",
            decryption.prediction_completion_anchor_record_sha256,
            anchor_record.record_sha256,
        ),
        (
            "decryption anchor receipt",
            decryption.prediction_completion_anchor_receipt_sha256,
            anchor_receipt.receipt_sha256,
        ),
        ("released label corpus", labels.corpus, location.corpus_id),
        (
            "released label execution",
            labels.execution_artifact_sha256,
            online_closure.execution_artifact_sha256,
        ),
    )
    for name, observed, wanted in expected:
        if observed != wanted:
            raise ConfirmatoryInputOperatorError(f"{name} binding differs")
    plaintext_member = _file_member(
        plaintext_path,
        role="released-sealed-labels",
        corpus_id=location.corpus_id,
        semantic_sha256=labels.artifact_sha256,
        expected_file_sha256=label_closure.plaintext_sha256,
        expected_byte_count=label_closure.plaintext_byte_count,
    )
    if (
        decryption.receipt_sha256 != label_closure.decryption_receipt_sha256
        or decryption.plaintext_sha256 != label_closure.plaintext_sha256
        or decryption.plaintext_byte_count != label_closure.plaintext_byte_count
    ):
        raise ConfirmatoryInputOperatorError("released plaintext differs from LABELS_RELEASED")
    prediction_rows = {row.trial_key: row for row in predictions.predictions}
    label_rows = {row.trial_key: row for row in labels.labels}
    if set(prediction_rows) != set(label_rows):
        raise ConfirmatoryInputOperatorError("prediction and released-label membership differ")
    joined = tuple(
        JoinedEvaluationTrial(
            prediction=prediction_rows[trial_key],
            labels=label_rows[trial_key],
        )
        for trial_key in sorted(prediction_rows)
    )
    evaluation = OfflineEvaluationArtifact(
        manifest_sha256=manifest_digest,
        run_receipt_sha256=run_receipt_digest,
        execution_artifact_sha256=online_closure.execution_artifact_sha256,
        prediction_artifact_sha256=predictions.artifact_sha256,
        prediction_completion_receipt_sha256=completion.receipt_sha256,
        online_execution_result_receipt_sha256=result.receipt_sha256,
        timelock_decryption_receipt_sha256=decryption.receipt_sha256,
        sealed_label_artifact_sha256=labels.artifact_sha256,
        corpus=location.corpus_id,
        stage="sealed",
        trials=joined,
    )
    members = (
        _file_member(
            completion_path,
            role="prediction-completion-receipt",
            corpus_id=location.corpus_id,
            semantic_sha256=completion.receipt_sha256,
        ),
        _file_member(
            anchor_record_path,
            role="prediction-completion-anchor-record",
            corpus_id=location.corpus_id,
            semantic_sha256=anchor_record.record_sha256,
            expected_file_sha256=anchor_record.record_sha256,
        ),
        _file_member(
            anchor_receipt_path,
            role="prediction-completion-anchor-receipt",
            corpus_id=location.corpus_id,
            semantic_sha256=anchor_receipt.receipt_sha256,
        ),
        _file_member(
            decryption_path,
            role="timelock-decryption-receipt",
            corpus_id=location.corpus_id,
            semantic_sha256=decryption.receipt_sha256,
            expected_file_sha256=label_closure.decryption_receipt_file_sha256,
        ),
        plaintext_member,
    )
    return completion, verified_anchor, decryption, labels, evaluation, members


def _source_member(
    path: Path,
    *,
    role: str,
    semantic_sha256: str,
) -> ConfirmatoryInputMember:
    return _file_member(path, role=role, semantic_sha256=semantic_sha256)


def _assemble_input(
    config: ConfirmatoryInputOperatorConfig,
    verified_labels: LabelsAuthority,
    *,
    trusted_anchor_record_fetcher: Callable[[str, int], bytes] | None = None,
) -> tuple[ConfirmatoryInputArtifact, tuple[ConfirmatoryInputMember, ...]]:
    labels_state = _labels_state(verified_labels)
    verified_labels.assert_current()
    namespace = _file_uri(config.suite_namespace_uri, label="suite_namespace_uri", directory=True)
    if namespace != verified_labels.namespace:
        raise ConfirmatoryInputOperatorError("operator config names another suite namespace")
    manifest_path = _file_uri(config.manifest_uri, label="manifest_uri")
    run_path = _file_uri(config.sealed_run_receipt_uri, label="sealed_run_receipt_uri")
    verification_path = _file_uri(
        config.artifact_verification_receipt_uri,
        label="artifact_verification_receipt_uri",
    )
    manifest = load_study_manifest(manifest_path)
    validate_study_manifest(manifest, require_frozen=True)
    manifest_digest = manifest_sha256(manifest)
    run_receipt = load_sealed_run_receipt(run_path)
    verification_receipt = load_verification_receipt(verification_path)
    if (
        labels_state.manifest_sha256 != manifest_digest
        or run_receipt.manifest_sha256 != manifest_digest
        or verification_receipt.manifest_sha256 != manifest_digest
    ):
        raise ConfirmatoryInputOperatorError("manifest identity differs across admitted evidence")
    if run_receipt.receipt_uri != run_path.as_uri():
        raise ConfirmatoryInputOperatorError("sealed run receipt is not at its self-declared URI")
    if run_receipt.verification_receipt_uri != verification_path.as_uri():
        raise ConfirmatoryInputOperatorError("verification receipt URI differs from sealed run")
    run_digest = sealed_run_receipt_sha256(run_receipt)
    if (
        run_digest != labels_state.run_receipt_sha256
        or verification_receipt.receipt_sha256 != run_receipt.verification_receipt_sha256
    ):
        raise ConfirmatoryInputOperatorError("run or verification receipt digest differs")
    require_verified_labels_released(
        verified_labels,
        manifest_digest=manifest_digest,
        run_receipt_sha256=run_digest,
    )
    _, online_payload, label_payload = _online_and_label_records(verified_labels)
    online_by_corpus = {row.corpus_id: row for row in online_payload.corpora}
    labels_by_corpus = {row.corpus_id: row for row in label_payload}
    locations = {row.corpus_id: row for row in config.corpus_evidence}
    members: list[ConfirmatoryInputMember] = [
        _source_member(
            manifest_path,
            role="frozen-study-manifest",
            semantic_sha256=manifest_digest,
        ),
        _source_member(
            run_path,
            role="sealed-run-receipt",
            semantic_sha256=run_digest,
        ),
        _source_member(
            verification_path,
            role="artifact-verification-receipt",
            semantic_sha256=verification_receipt.receipt_sha256,
        ),
    ]
    completions: list[PredictionCompletionReceipt] = []
    evaluations: list[OfflineEvaluationArtifact] = []
    sealed_labels: list[SealedLabelArtifact] = []
    panels: list[ActionPanelArtifact] = []
    admissions: list[ActionPanelAdmissionReceipt] = []
    for corpus_id in _ordered_corpora():
        online = online_by_corpus[corpus_id]
        label = labels_by_corpus[corpus_id]
        result, predictions, panel, admission, online_members = _assert_online_directory(
            online,
            manifest_digest=manifest_digest,
        )
        completion, _, _, label_artifact, evaluation, release_members = _assert_release_chain(
            locations[corpus_id],
            label_closure=label,
            online_closure=online,
            result=result,
            predictions=predictions,
            panel=panel,
            manifest_digest=manifest_digest,
            run_receipt_digest=run_digest,
            trusted_anchor_record_fetcher=trusted_anchor_record_fetcher,
        )
        manifest_label_pin = str(
            _manifest_artifact(manifest, role="sealed-labels", corpus_id=corpus_id)["sha256"]
        )
        if manifest_label_pin != label.plaintext_sha256:
            raise ConfirmatoryInputOperatorError(
                "sealed-label manifest pin must equal the exact newline-terminated file digest"
            )
        completions.append(completion)
        evaluations.append(evaluation)
        sealed_labels.append(label_artifact)
        panels.append(panel)
        admissions.append(admission)
        members.extend(online_members)
        members.extend(release_members)
    inputs = ConfirmatoryInputArtifact(
        run_receipt=run_receipt,
        frozen_manifest=manifest,
        artifact_verification_receipt=verification_receipt,
        completion_receipts=tuple(completions),
        offline_evaluations=tuple(evaluations),
        sealed_label_artifacts=tuple(sealed_labels),
        action_panels=tuple(panels),
        action_panel_admission_receipts=tuple(admissions),
    )
    verified_labels.assert_current()
    return inputs, tuple(
        sorted(
            members,
            key=lambda row: (
                (row.corpus_id or "").encode("utf-8"),
                row.role.encode("utf-8"),
                row.uri.encode("utf-8"),
            ),
        )
    )


def _materialization_receipt(
    inputs: ConfirmatoryInputArtifact,
    token: LabelsAuthority,
    members: tuple[ConfirmatoryInputMember, ...],
) -> ConfirmatoryInputMaterializationReceipt:
    encoded = inputs.canonical_bytes() + b"\n"
    return ConfirmatoryInputMaterializationReceipt(
        suite_attempt_id=_labels_state(token).suite_attempt_id,
        suite_state_record_sha256=_labels_state(token).record_sha256,
        suite_descriptor_sha256=_labels_descriptor_sha256(token),
        manifest_sha256=inputs.manifest_sha256,
        run_receipt_sha256=inputs.run_receipt_sha256,
        artifact_uri=confirmatory_input_path(inputs).as_uri(),
        artifact_sha256=inputs.artifact_sha256,
        artifact_file_sha256=_sha256(encoded),
        artifact_byte_count=len(encoded),
        members=members,
    )


def _verify_member_files(members: Sequence[ConfirmatoryInputMember]) -> None:
    for member in members:
        path = _file_uri(member.uri, label=f"{member.role} member URI")
        try:
            metadata = path.stat(follow_symlinks=False)
            observed = digest_regular_file(path, label=f"confirmatory input member {member.role}")
        except (OSError, ArtifactIntegrityError) as exc:
            raise ConfirmatoryInputOperatorError(
                f"cannot revalidate input member {member.role}: {exc}"
            ) from exc
        if metadata.st_size != member.byte_count or observed != member.file_sha256:
            raise ConfirmatoryInputOperatorError(
                f"input member {member.role} changed during materialization"
            )


def materialize_confirmatory_input(
    config: ConfirmatoryInputOperatorConfig,
    verified_labels: LabelsAuthority,
    *,
    trusted_anchor_record_fetcher: Callable[[str, int], bytes] | None = None,
) -> MaterializedConfirmatoryInput:
    """Build, close, and create the one persisted analysis input exclusively."""

    inputs, members = _assemble_input(
        config,
        verified_labels,
        trusted_anchor_record_fetcher=trusted_anchor_record_fetcher,
    )
    receipt = _materialization_receipt(inputs, verified_labels, members)
    artifact_path = confirmatory_input_path(inputs)
    receipt_path = confirmatory_input_receipt_path(inputs)
    _verify_member_files(receipt.members)
    verified_labels.assert_current()
    try:
        # Reserve the detached closure before exposing the persisted input bytes.
        write_exclusive_receipt_bytes(receipt.canonical_bytes() + b"\n", receipt_path)
        write_exclusive_receipt_bytes(inputs.canonical_bytes() + b"\n", artifact_path)
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryInputOperatorError(f"cannot persist confirmatory input: {exc}") from exc
    return MaterializedConfirmatoryInput(inputs, receipt, artifact_path, receipt_path)


def load_confirmatory_input_materialization_receipt(
    path: str | Path,
) -> ConfirmatoryInputMaterializationReceipt:
    try:
        encoded = read_secure_control_file(path, label="confirmatory input materialization receipt")
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryInputOperatorError(f"cannot load input receipt: {exc}") from exc
    receipt = ConfirmatoryInputMaterializationReceipt.from_dict(
        _decode_object(encoded, label="confirmatory input materialization receipt")
    )
    if encoded != receipt.canonical_bytes() + b"\n":
        raise ConfirmatoryInputOperatorError(
            "input materialization receipt bytes are not canonical"
        )
    return receipt


def load_materialized_confirmatory_input(
    config: ConfirmatoryInputOperatorConfig,
    verified_labels: LabelsAuthority,
    *,
    trusted_anchor_record_fetcher: Callable[[str, int], bytes] | None = None,
) -> MaterializedConfirmatoryInput:
    """Reconstruct all sources and verify the persisted input and detached receipt."""

    inputs, members = _assemble_input(
        config,
        verified_labels,
        trusted_anchor_record_fetcher=trusted_anchor_record_fetcher,
    )
    expected = _materialization_receipt(inputs, verified_labels, members)
    artifact_path = confirmatory_input_path(inputs)
    receipt_path = confirmatory_input_receipt_path(inputs)
    receipt = load_confirmatory_input_materialization_receipt(receipt_path)
    if receipt != expected:
        raise ConfirmatoryInputOperatorError(
            "persisted input receipt differs from the current verified source closure"
        )
    try:
        encoded = read_secure_control_file(artifact_path, label="persisted confirmatory input")
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryInputOperatorError(f"cannot load persisted input: {exc}") from exc
    if (
        encoded != inputs.canonical_bytes() + b"\n"
        or len(encoded) != receipt.artifact_byte_count
        or _sha256(encoded) != receipt.artifact_file_sha256
        or inputs.artifact_sha256 != receipt.artifact_sha256
    ):
        raise ConfirmatoryInputOperatorError("persisted confirmatory input bytes differ")
    _verify_member_files(receipt.members)
    verified_labels.assert_current()
    return MaterializedConfirmatoryInput(inputs, receipt, artifact_path, receipt_path)


def _verified_model_file(
    config: ConfirmatoryInputOperatorConfig,
    inputs: ConfirmatoryInputArtifact,
    *,
    role: str,
) -> bytes:
    artifact = _manifest_artifact(inputs.frozen_manifest, role=role)
    artifact_id = str(artifact["id"])
    rows = [
        row
        for row in inputs.artifact_verification_receipt.artifacts
        if row.artifact_id == artifact_id
    ]
    if len(rows) != 1 or rows[0].kind != "file" or not rows[0].exact:
        raise ConfirmatoryInputOperatorError(f"{role} must be one exact verified file")
    relative = PurePosixPath(rows[0].relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ConfirmatoryInputOperatorError(f"{role} verification path is not canonical")
    root = _file_uri(config.artifact_root_uri, label="artifact_root_uri", directory=True)
    path = root.joinpath(*relative.parts)
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=_MAX_MODEL_BYTES,
            label=f"{role} artifact",
        )
    except ArtifactIntegrityError as exc:
        raise ConfirmatoryInputOperatorError(f"cannot load {role}: {exc}") from exc
    expected = str(artifact["sha256"])
    if (
        _sha256(encoded) != expected
        or rows[0].expected_sha256 != expected
        or rows[0].verified_sha256 != expected
    ):
        raise ConfirmatoryInputOperatorError(f"{role} bytes differ from frozen verification")
    return encoded


def load_admitted_model_suite(
    config: ConfirmatoryInputOperatorConfig,
    inputs: ConfirmatoryInputArtifact,
) -> FrozenModelSuite:
    """Load only the two manifest-selected model files and prove exact canonical bytes."""

    h1_bytes = _verified_model_file(config, inputs, role="h1-predictive-model")
    h2_bytes = _verified_model_file(config, inputs, role="h2-model-suite")
    try:
        suite = FrozenModelSuite.from_json(h2_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ConfirmatoryInputOperatorError("frozen H2 suite cannot be decoded") from exc
    if h1_bytes != canonical_h1_model_artifact_bytes(
        suite
    ) or h2_bytes != canonical_h2_model_suite_artifact_bytes(suite):
        raise ConfirmatoryInputOperatorError("model files are not the canonical registered pair")
    inputs.assert_model_suite_admitted(suite)
    return suite


def run_materialized_confirmatory_analysis_once(
    config: ConfirmatoryInputOperatorConfig,
    verified_labels: VerifiedSuiteLabelsReleased,
    *,
    trusted_anchor_record_fetcher: Callable[[str, int], bytes] | None = None,
) -> SuiteStateRecord:
    """Consume the sole analysis attempt and create the local ANALYSIS_COMPLETE state."""

    materialized = load_materialized_confirmatory_input(
        config,
        verified_labels,
        trusted_anchor_record_fetcher=trusted_anchor_record_fetcher,
    )
    suite = load_admitted_model_suite(config, materialized.inputs)
    verified_labels.assert_current()
    result = run_confirmatory_analysis_once(
        materialized.inputs,
        suite=suite,
        verified_labels_released=verified_labels,
    )
    if result.confirmatory_input_artifact_sha256 != materialized.receipt.artifact_sha256:
        raise ConfirmatoryInputOperatorError("analysis result binds another persisted input")
    verified_labels.assert_current()
    return complete_confirmatory_analysis(
        verified_labels,
        phase_claim=None,  # type: ignore[arg-type]
        confirmatory_input_artifact_sha256=materialized.receipt.artifact_sha256,
        attempt_receipt_path=confirmatory_attempt_path(materialized.inputs),
        result_receipt_path=confirmatory_result_receipt_path(materialized.inputs),
        final_result_path=confirmatory_result_path(materialized.inputs),
    )


def run_provider_claimed_confirmatory_analysis_once(
    config: ConfirmatoryInputOperatorConfig,
    verified_claimed: VerifiedProviderPredecessor,
    phase_claim: VerifiedPhaseClaimCapability,
    *,
    trusted_anchor_record_fetcher: Callable[[str, int], bytes] | None = None,
    fresh_claim_supplier: Callable[
        [], tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]
    ]
    | None = None,
) -> SuiteStateRecord:
    """Persist and analyze once under the winning ANALYSIS_CLAIMED capability."""

    if (
        not isinstance(verified_claimed, VerifiedProviderPredecessor)
        or verified_claimed.state.state != "ANALYSIS_CLAIMED"
    ):
        raise ConfirmatoryInputOperatorError("provider analysis requires verified ANALYSIS_CLAIMED")
    if not isinstance(phase_claim, VerifiedPhaseClaimCapability):
        raise ConfirmatoryInputOperatorError("provider analysis lacks phase authority")
    phase_claim.assert_current()
    materialized = materialize_confirmatory_input(
        config,
        verified_claimed,
        trusted_anchor_record_fetcher=trusted_anchor_record_fetcher,
    )
    suite = load_admitted_model_suite(config, materialized.inputs)
    verified_claimed.assert_current()
    result = run_confirmatory_analysis_once(
        materialized.inputs,
        suite=suite,
        verified_labels_released=verified_claimed,
    )
    if result.confirmatory_input_artifact_sha256 != materialized.receipt.artifact_sha256:
        raise ConfirmatoryInputOperatorError("analysis result binds another persisted input")
    verified_claimed.assert_current()
    completion_claimed = verified_claimed
    completion_phase_claim = phase_claim
    if fresh_claim_supplier is not None:
        try:
            completion_claimed, completion_phase_claim = fresh_claim_supplier()
        except Exception as exc:
            raise ConfirmatoryInputOperatorError(
                "cannot refresh ANALYSIS_CLAIMED completion authority"
            ) from exc
        if (
            not isinstance(completion_claimed, VerifiedProviderPredecessor)
            or completion_claimed.state.state != "ANALYSIS_CLAIMED"
            or not isinstance(completion_phase_claim, VerifiedPhaseClaimCapability)
            or completion_claimed.state.record_sha256 != verified_claimed.state.record_sha256
            or completion_claimed.ledger_commit != verified_claimed.ledger_commit
            or completion_claimed.control_inventory_sha256
            != verified_claimed.control_inventory_sha256
            or completion_claimed.artifact_receipt_sha256
            != verified_claimed.artifact_receipt_sha256
        ):
            raise ConfirmatoryInputOperatorError(
                "refreshed ANALYSIS_CLAIMED authority differs from the input claim"
            )
    completion_phase_claim.assert_current()
    return complete_confirmatory_analysis(
        completion_claimed,
        phase_claim=completion_phase_claim,
        confirmatory_input_artifact_sha256=materialized.receipt.artifact_sha256,
        attempt_receipt_path=confirmatory_attempt_path(materialized.inputs),
        result_receipt_path=confirmatory_result_receipt_path(materialized.inputs),
        final_result_path=confirmatory_result_path(materialized.inputs),
    )


def _github_verified_labels(
    config: ConfirmatoryInputOperatorConfig,
) -> VerifiedSuiteLabelsReleased:
    namespace = _file_uri(config.suite_namespace_uri, label="suite_namespace_uri", directory=True)
    verifier = GitHubSuiteEvidenceVerifier(namespace)
    token = verify_suite_state(
        namespace,
        verifier=verifier,
        expected_state="LABELS_RELEASED",
    )
    if not isinstance(token, VerifiedSuiteLabelsReleased):
        raise ConfirmatoryInputOperatorError("GitHub verifier did not return LABELS_RELEASED")
    return token


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed post-label confirmatory input and analysis operator"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("materialize", "create the sole persisted confirmatory input and receipt"),
        ("verify", "reconstruct and verify the persisted confirmatory input"),
        ("analyze", "consume the sole analysis attempt and create ANALYSIS_COMPLETE"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_confirmatory_input_operator_config(args.config)
    verified = _github_verified_labels(config)
    if args.command == "materialize":
        output = materialize_confirmatory_input(config, verified)
        payload = {
            "artifact_sha256": output.receipt.artifact_sha256,
            "artifact_uri": output.receipt.artifact_uri,
            "receipt_sha256": output.receipt.receipt_sha256,
            "receipt_uri": output.receipt_path.as_uri(),
        }
    elif args.command == "verify":
        output = load_materialized_confirmatory_input(config, verified)
        payload = {
            "artifact_sha256": output.receipt.artifact_sha256,
            "receipt_sha256": output.receipt.receipt_sha256,
            "verified": True,
        }
    elif args.command == "analyze":
        state = run_materialized_confirmatory_analysis_once(config, verified)
        payload = {
            "state": state.state,
            "state_record_sha256": state.record_sha256,
            "suite_attempt_id": state.suite_attempt_id,
        }
    else:  # pragma: no cover
        raise ConfirmatoryInputOperatorError("unknown operator command")
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
