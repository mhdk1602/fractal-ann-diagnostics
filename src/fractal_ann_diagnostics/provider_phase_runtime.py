"""Closed provider-phase driver selected by the frozen C1 phase plan.

The public command accepts only the five bindings named by the plan.  It never
accepts a scientific command, container argument, environment override, corpus
subset, image name, tool path, or retry flag.  Lower-level driver requests are
canonical records below the admitted phase-input root and bind the exact plan,
claim receipt, controls, portable runtime receipts, and output namespaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    write_exclusive_receipt_bytes,
)
from .execution_claim import (
    ANALYSIS_PHASE,
    LABEL_RELEASE_PHASE,
    ONLINE_PHASE,
    PHASE_RUNTIME_CLAIM_RECEIPT_SCHEMA,
    PROVIDER_PHASE_PLAN_SCHEMA,
    RUNTIME_CLAIM_RECEIPT_SCHEMA,
    ExecutionClaimError,
    PhaseRuntimeClaimReceipt,
    ProviderPhase,
    ProviderPhasePlan,
    VerifiedPhaseClaimCapability,
    VerifiedRunClaimCapability,
    load_materialized_provider_phase_plan,
    load_provider_runner_bootstrap,
    loads_runtime_claim_receipt,
)
from .study import FIXED_CORPORA, PROVIDER_PHASE_COMMAND_IDS
from .suite_attempt import VerifiedProviderPredecessor

PROVIDER_DRIVER_REQUEST_SCHEMA = "fractal-provider-driver-request-v1"
PROVIDER_PHASE_RUNTIME_REQUEST_SCHEMA = "fractal-provider-phase-runtime-request-v1"
PROVIDER_DRIVER_OUTPUT_SCHEMA = "fractal-provider-driver-output-v1"
PROVIDER_PHASE_EXECUTION_RECEIPT_SCHEMA = "fractal-provider-phase-execution-v1"
LABEL_RELEASE_DRIVER_CONTROL_SCHEMA = "fractal-provider-label-release-driver-v1"
ONLINE_SEALED_LAUNCH_DRIVER_CONTROL_SCHEMA = "fractal-provider-online-sealed-launch-driver-v1"
ANALYSIS_RUNTIME_CLAIM_BUNDLE_SCHEMA = "fractal-analysis-runtime-claim-bundle-v1"

PROVIDER_RUNTIME_REQUEST_FILENAME = "provider-runtime-request.json"
PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME = "provider-phase-execution-receipt.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 8 * 1024 * 1024
_ALL_FIVE = "all-five"
_DRIVER_IDS: Mapping[ProviderPhase, str] = {
    ONLINE_PHASE: "sealed-online-corpus-v1",
    LABEL_RELEASE_PHASE: "timelock-label-release-v1",
    ANALYSIS_PHASE: "confirmatory-analysis-v1",
}


class ProviderPhaseRuntimeError(ValueError):
    """A phase request differs from the C1-selected driver contract."""


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
        raise ProviderPhaseRuntimeError("provider runtime record is not canonical JSON") from exc


def _digest(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProviderPhaseRuntimeError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderPhaseRuntimeError(f"{name} must be canonical non-empty text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ProviderPhaseRuntimeError(f"{name} contains a control character")
    return value


def _absolute_path(name: str, value: object) -> Path:
    path = Path(_text(name, value))
    if not path.is_absolute() or path.anchor != "/":
        raise ProviderPhaseRuntimeError(f"{name} must be an absolute POSIX path")
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ProviderPhaseRuntimeError(f"{name} contains an aliasing component")
    return path


def _below(name: str, value: object, root: Path) -> Path:
    path = _absolute_path(name, value)
    if path == root or not path.is_relative_to(root):
        raise ProviderPhaseRuntimeError(f"{name} must be a strict descendant of its root")
    return path


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProviderPhaseRuntimeError(f"{label} must be one JSON object")
    observed = set(value)
    if observed != fields:
        raise ProviderPhaseRuntimeError(
            f"{label} schema differs; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _strict_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ProviderPhaseRuntimeError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ProviderPhaseRuntimeError(f"{label} contains non-finite number {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderPhaseRuntimeError(f"cannot decode {label}") from exc
    if not isinstance(value, Mapping):
        raise ProviderPhaseRuntimeError(f"{label} must contain one object")
    return value


def _secure_file_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProviderPhaseRuntimeError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_CONTROL_BYTES
        ):
            raise ProviderPhaseRuntimeError(
                f"{label} must be one bounded singly linked regular file"
            )
        chunks: list[bytes] = []
        observed = 0
        while observed <= _MAX_CONTROL_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_CONTROL_BYTES + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ProviderPhaseRuntimeError(f"cannot read {label}") from exc
    finally:
        os.close(descriptor)
    if observed > _MAX_CONTROL_BYTES or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ProviderPhaseRuntimeError(f"{label} changed while read")
    return b"".join(chunks)


def _verified_file(path: Path, expected_sha256: str, *, label: str) -> bytes:
    encoded = _secure_file_bytes(path, label=label)
    if hashlib.sha256(encoded).hexdigest() != _digest(f"{label} SHA-256", expected_sha256):
        raise ProviderPhaseRuntimeError(f"{label} differs from the request digest")
    return encoded


def _controlled_directory_entries(path: Path, *, label: str) -> tuple[str, ...]:
    try:
        metadata = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
        entries = tuple(
            sorted(
                (entry.name for entry in path.iterdir()),
                key=lambda value: value.encode("utf-8"),
            )
        )
    except OSError as exc:
        raise ProviderPhaseRuntimeError(f"cannot admit {label}") from exc
    if (
        resolved != path
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise ProviderPhaseRuntimeError(f"{label} is not a controlled directory")
    return entries


@dataclass(frozen=True)
class ProviderDriverRequest:
    corpus_id: str
    driver_id: str
    control_path: str
    control_file_sha256: str
    runtime_claim_receipt_path: str
    runtime_claim_receipt_file_sha256: str
    output_root: str
    schema_version: str = PROVIDER_DRIVER_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        _text("corpus_id", self.corpus_id)
        _text("driver_id", self.driver_id)
        _absolute_path("control_path", self.control_path)
        _digest("control_file_sha256", self.control_file_sha256)
        _absolute_path("runtime_claim_receipt_path", self.runtime_claim_receipt_path)
        _digest(
            "runtime_claim_receipt_file_sha256",
            self.runtime_claim_receipt_file_sha256,
        )
        _absolute_path("output_root", self.output_root)
        if self.schema_version != PROVIDER_DRIVER_REQUEST_SCHEMA:
            raise ProviderPhaseRuntimeError("provider driver request schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> ProviderDriverRequest:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="provider driver request",
            )
        )


AnalysisClaimSupplier = Callable[
    [], tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]
]


@dataclass(frozen=True)
class FreshOnlineClaimAuthority:
    capability: VerifiedRunClaimCapability
    claim_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.capability, VerifiedRunClaimCapability):
            raise ProviderPhaseRuntimeError("fresh online authority is untyped")
        if type(self.claim_bytes) is not bytes:
            raise ProviderPhaseRuntimeError("fresh online claim bytes are untyped")


@dataclass(frozen=True)
class FreshLabelClaimAuthority:
    capability: VerifiedPhaseClaimCapability
    claim_bytes: bytes
    admission_marker_path: str
    admission_marker_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.capability, VerifiedPhaseClaimCapability):
            raise ProviderPhaseRuntimeError("fresh label authority is untyped")
        if type(self.claim_bytes) is not bytes:
            raise ProviderPhaseRuntimeError("fresh label claim bytes are untyped")
        _absolute_path("admission_marker_path", self.admission_marker_path)
        _digest("admission_marker_sha256", self.admission_marker_sha256)


OnlineRunClaimSupplier = Callable[[ProviderDriverRequest], FreshOnlineClaimAuthority]
LabelPhaseClaimSupplier = Callable[[ProviderDriverRequest], FreshLabelClaimAuthority]


@dataclass(frozen=True)
class ProviderPhaseRuntimeRequest:
    phase: ProviderPhase
    activation_command_id: str
    suite_attempt_id: str
    provider_plan_path: str
    provider_plan_sha256: str
    provider_plan_file_sha256: str
    claim_receipt_path: str
    claim_receipt_file_sha256: str
    phase_input_root: str
    phase_output_root: str
    drivers: tuple[ProviderDriverRequest, ...]
    schema_version: str = PROVIDER_PHASE_RUNTIME_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in _DRIVER_IDS:
            raise ProviderPhaseRuntimeError("provider runtime request has another phase")
        if self.activation_command_id != PROVIDER_PHASE_COMMAND_IDS[self.phase]:
            raise ProviderPhaseRuntimeError("provider runtime command differs from C1")
        _digest("suite_attempt_id", self.suite_attempt_id)
        _absolute_path("provider_plan_path", self.provider_plan_path)
        _digest("provider_plan_sha256", self.provider_plan_sha256)
        _digest("provider_plan_file_sha256", self.provider_plan_file_sha256)
        _absolute_path("claim_receipt_path", self.claim_receipt_path)
        _digest("claim_receipt_file_sha256", self.claim_receipt_file_sha256)
        input_root = _absolute_path("phase_input_root", self.phase_input_root)
        output_root = _absolute_path("phase_output_root", self.phase_output_root)
        if (
            input_root == output_root
            or input_root in output_root.parents
            or output_root in input_root.parents
        ):
            raise ProviderPhaseRuntimeError("provider phase input and output roots overlap")
        rows = tuple(self.drivers)
        expected_ids = set(FIXED_CORPORA) if self.phase != ANALYSIS_PHASE else {_ALL_FIVE}
        if (
            len(rows) != len(expected_ids)
            or not all(isinstance(row, ProviderDriverRequest) for row in rows)
            or {row.corpus_id for row in rows} != expected_ids
            or [row.corpus_id for row in rows]
            != sorted(expected_ids, key=lambda value: value.encode("utf-8"))
        ):
            raise ProviderPhaseRuntimeError("provider driver rows do not cover the fixed phase")
        if any(row.driver_id != _DRIVER_IDS[self.phase] for row in rows):
            raise ProviderPhaseRuntimeError("provider driver ID differs from the phase")
        if len({row.control_path for row in rows}) != len(rows):
            raise ProviderPhaseRuntimeError("provider driver rows reuse one control file")
        if len({row.runtime_claim_receipt_path for row in rows}) != len(rows):
            raise ProviderPhaseRuntimeError("provider driver rows reuse one runtime receipt")
        if self.phase != ANALYSIS_PHASE:
            expected_output_roots = {str(output_root / row.corpus_id) for row in rows}
            if {row.output_root for row in rows} != expected_output_roots:
                raise ProviderPhaseRuntimeError("provider driver output namespaces differ")
        for row in rows:
            _below("driver control_path", row.control_path, input_root)
            _below(
                "driver runtime_claim_receipt_path",
                row.runtime_claim_receipt_path,
                input_root,
            )
            if self.phase != ANALYSIS_PHASE:
                _below("driver output_root", row.output_root, output_root)
        if self.schema_version != PROVIDER_PHASE_RUNTIME_REQUEST_SCHEMA:
            raise ProviderPhaseRuntimeError("provider phase runtime request schema differs")
        object.__setattr__(self, "drivers", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name) for name in self.__dataclass_fields__ if name != "drivers"
            },
            "drivers": [row.to_dict() for row in self.drivers],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> ProviderPhaseRuntimeRequest:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="provider phase runtime request",
        )
        drivers = row["drivers"]
        if not isinstance(drivers, list):
            raise ProviderPhaseRuntimeError("provider phase driver requests must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "drivers"},
            drivers=tuple(ProviderDriverRequest.from_dict(item) for item in drivers),
        )


def load_provider_phase_runtime_request(
    path: str | Path,
) -> ProviderPhaseRuntimeRequest:
    candidate = Path(path)
    encoded = _secure_file_bytes(candidate, label="provider phase runtime request")
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ProviderPhaseRuntimeError("provider runtime request needs one terminal newline")
    request = ProviderPhaseRuntimeRequest.from_dict(
        _strict_object(encoded[:-1], label="provider phase runtime request")
    )
    if request.canonical_file_bytes() != encoded:
        raise ProviderPhaseRuntimeError("provider runtime request bytes are not canonical")
    expected_path = Path(request.phase_input_root) / PROVIDER_RUNTIME_REQUEST_FILENAME
    if candidate != expected_path:
        raise ProviderPhaseRuntimeError("provider runtime request is outside its fixed path")
    return request


def write_provider_phase_runtime_request(
    request: ProviderPhaseRuntimeRequest,
) -> Path:
    if not isinstance(request, ProviderPhaseRuntimeRequest):
        raise ProviderPhaseRuntimeError("runtime request must use the closed typed schema")
    path = Path(request.phase_input_root) / PROVIDER_RUNTIME_REQUEST_FILENAME
    try:
        write_exclusive_receipt_bytes(request.canonical_file_bytes(), path)
    except ArtifactIntegrityError as exc:
        raise ProviderPhaseRuntimeError("cannot write provider runtime request once") from exc
    return path


@dataclass(frozen=True)
class OnlineSealedLaunchDriverControl:
    preflight_contract_path: str
    preflight_receipt_path: str
    transition_receipt_path: str
    instantiation_receipt_path: str
    finalization_request_path: str
    finalization_receipt_path: str
    sealed_contract_path: str
    volume_receipt_path: str
    audit_root: str
    schema_version: str = ONLINE_SEALED_LAUNCH_DRIVER_CONTROL_SCHEMA

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "schema_version":
                _absolute_path(name, getattr(self, name))
        if self.schema_version != ONLINE_SEALED_LAUNCH_DRIVER_CONTROL_SCHEMA:
            raise ProviderPhaseRuntimeError("online sealed-launch control schema differs")

    def canonical_file_bytes(self) -> bytes:
        return (
            _canonical_bytes({name: getattr(self, name) for name in self.__dataclass_fields__})
            + b"\n"
        )

    @classmethod
    def from_bytes(cls, encoded: bytes) -> OnlineSealedLaunchDriverControl:
        if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
            raise ProviderPhaseRuntimeError("online control needs one terminal newline")
        control = cls(
            **_closed(
                _strict_object(encoded[:-1], label="online sealed-launch control"),
                frozenset(cls.__dataclass_fields__),
                label="online sealed-launch control",
            )
        )
        if encoded != control.canonical_file_bytes():
            raise ProviderPhaseRuntimeError("online control bytes are not canonical")
        return control


@dataclass(frozen=True)
class LabelReleaseDriverControl:
    manifest_path: str
    custody_seal_path: str
    encryption_receipt_path: str
    completion_receipt_path: str
    completion_anchor_record_path: str
    completion_anchor_receipt_path: str
    suite_namespace: str
    ciphertext_path: str
    tle_binary_path: str
    plaintext_output_path: str
    decryption_receipt_path: str
    schema_version: str = LABEL_RELEASE_DRIVER_CONTROL_SCHEMA

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "schema_version":
                _absolute_path(name, getattr(self, name))
        if self.schema_version != LABEL_RELEASE_DRIVER_CONTROL_SCHEMA:
            raise ProviderPhaseRuntimeError("label-release driver control schema differs")

    @classmethod
    def from_bytes(cls, encoded: bytes) -> LabelReleaseDriverControl:
        if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
            raise ProviderPhaseRuntimeError("label-release control needs one terminal newline")
        value = _strict_object(encoded[:-1], label="label-release driver control")
        control = cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="label-release driver control",
            )
        )
        expected = (
            _canonical_bytes(
                {name: getattr(control, name) for name in control.__dataclass_fields__}
            )
            + b"\n"
        )
        if encoded != expected:
            raise ProviderPhaseRuntimeError("label-release control bytes are not canonical")
        return control


@dataclass(frozen=True)
class AnalysisRuntimeClaimBundle:
    phase: Literal["analysis"]
    receipts: tuple[PhaseRuntimeClaimReceipt, ...]
    schema_version: str = ANALYSIS_RUNTIME_CLAIM_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.phase != ANALYSIS_PHASE:
            raise ProviderPhaseRuntimeError("analysis runtime bundle has another phase")
        rows = tuple(self.receipts)
        ordered = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
        if (
            len(rows) != len(ordered)
            or not all(isinstance(row, PhaseRuntimeClaimReceipt) for row in rows)
            or tuple(row.corpus_id for row in rows) != ordered
            or any(row.phase != ANALYSIS_PHASE for row in rows)
        ):
            raise ProviderPhaseRuntimeError(
                "analysis runtime bundle must contain the ordered five corpus receipts"
            )
        shared = (
            "manifest_sha256",
            "run_receipt_sha256",
            "c1_commit",
            "phase_claim_contract_sha256",
            "phase_claim_state_sha256",
            "phase_claim_ledger_commit",
            "provider_identity_sha256",
            "live_execute_job_receipt_sha256",
            "execute_job_id",
            "phase_input_aggregate_sha256",
            "phase_output_identity",
        )
        first = rows[0]
        if any(getattr(row, name) != getattr(first, name) for row in rows[1:] for name in shared):
            raise ProviderPhaseRuntimeError("analysis runtime receipts cross provider claims")
        if self.schema_version != ANALYSIS_RUNTIME_CLAIM_BUNDLE_SCHEMA:
            raise ProviderPhaseRuntimeError("analysis runtime bundle schema differs")
        object.__setattr__(self, "receipts", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "receipts": [row.to_dict() for row in self.receipts],
            "schema_version": self.schema_version,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_bytes(cls, encoded: bytes) -> AnalysisRuntimeClaimBundle:
        if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
            raise ProviderPhaseRuntimeError("analysis runtime bundle needs one terminal newline")
        row = _closed(
            _strict_object(encoded[:-1], label="analysis runtime claim bundle"),
            frozenset({"phase", "receipts", "schema_version"}),
            label="analysis runtime claim bundle",
        )
        receipts = row["receipts"]
        if not isinstance(receipts, list):
            raise ProviderPhaseRuntimeError("analysis runtime receipts must be an array")
        bundle = cls(
            phase=row["phase"],
            receipts=tuple(PhaseRuntimeClaimReceipt.from_dict(item) for item in receipts),
            schema_version=row["schema_version"],
        )
        if bundle.canonical_file_bytes() != encoded:
            raise ProviderPhaseRuntimeError("analysis runtime bundle bytes are not canonical")
        return bundle


def _load_phase_runtime_claim(
    encoded: bytes,
    *,
    phase: Literal["label-release", "analysis"],
    corpus_id: str,
) -> PhaseRuntimeClaimReceipt:
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ProviderPhaseRuntimeError("phase runtime receipt needs one terminal newline")
    receipt = PhaseRuntimeClaimReceipt.from_dict(
        _strict_object(encoded[:-1], label="phase runtime claim receipt")
    )
    if (
        receipt.canonical_file_bytes() != encoded
        or receipt.schema_version != PHASE_RUNTIME_CLAIM_RECEIPT_SCHEMA
        or receipt.phase != phase
        or (corpus_id != _ALL_FIVE and receipt.corpus_id != corpus_id)
    ):
        raise ProviderPhaseRuntimeError("phase runtime receipt differs from the driver request")
    return receipt


@dataclass(frozen=True)
class ProviderDriverOutput:
    corpus_id: str
    driver_id: str
    output_root: str
    output_tree_sha256: str
    output_entries: tuple[str, ...]
    schema_version: str = PROVIDER_DRIVER_OUTPUT_SCHEMA

    def __post_init__(self) -> None:
        _text("corpus_id", self.corpus_id)
        _text("driver_id", self.driver_id)
        _absolute_path("output_root", self.output_root)
        _digest("output_tree_sha256", self.output_tree_sha256)
        entries = tuple(self.output_entries)
        if (
            not entries
            or not all(type(item) is str and item for item in entries)
            or list(entries) != sorted(entries, key=lambda value: value.encode("utf-8"))
        ):
            raise ProviderPhaseRuntimeError("provider driver output inventory differs")
        if self.schema_version != PROVIDER_DRIVER_OUTPUT_SCHEMA:
            raise ProviderPhaseRuntimeError("provider driver output schema differs")
        object.__setattr__(self, "output_entries", entries)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "output_entries"
            },
            "output_entries": list(self.output_entries),
        }


@dataclass(frozen=True)
class ProviderPhaseExecutionReceipt:
    phase: ProviderPhase
    suite_attempt_id: str
    provider_plan_sha256: str
    provider_plan_file_sha256: str
    claim_receipt_file_sha256: str
    runtime_request_sha256: str
    runtime_request_file_sha256: str
    outputs: tuple[ProviderDriverOutput, ...]
    schema_version: str = PROVIDER_PHASE_EXECUTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in _DRIVER_IDS:
            raise ProviderPhaseRuntimeError("provider execution receipt has another phase")
        for name in (
            "suite_attempt_id",
            "provider_plan_sha256",
            "provider_plan_file_sha256",
            "claim_receipt_file_sha256",
            "runtime_request_sha256",
            "runtime_request_file_sha256",
        ):
            _digest(name, getattr(self, name))
        rows = tuple(self.outputs)
        if not rows or not all(isinstance(row, ProviderDriverOutput) for row in rows):
            raise ProviderPhaseRuntimeError("provider execution receipt lacks typed outputs")
        if self.schema_version != PROVIDER_PHASE_EXECUTION_RECEIPT_SCHEMA:
            raise ProviderPhaseRuntimeError("provider execution receipt schema differs")
        object.__setattr__(self, "outputs", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name) for name in self.__dataclass_fields__ if name != "outputs"
            },
            "outputs": [row.to_dict() for row in self.outputs],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.canonical_file_bytes()).hexdigest()


def _output_receipt(row: ProviderDriverRequest) -> ProviderDriverOutput:
    try:
        inventory = digest_directory_tree(Path(row.output_root))
    except ArtifactIntegrityError as exc:
        raise ProviderPhaseRuntimeError("cannot close provider driver output tree") from exc
    if not inventory.entries:
        raise ProviderPhaseRuntimeError("provider driver produced an empty output tree")
    return ProviderDriverOutput(
        corpus_id=row.corpus_id,
        driver_id=row.driver_id,
        output_root=row.output_root,
        output_tree_sha256=inventory.sha256,
        output_entries=tuple(inventory.entries),
    )


def _run_online(
    row: ProviderDriverRequest,
    claim_bytes: bytes,
    run_claim: VerifiedRunClaimCapability | None,
) -> None:
    try:
        receipt = loads_runtime_claim_receipt(claim_bytes)
    except ExecutionClaimError as exc:
        raise ProviderPhaseRuntimeError("online runtime receipt is invalid") from exc
    if receipt.schema_version != RUNTIME_CLAIM_RECEIPT_SCHEMA:
        raise ProviderPhaseRuntimeError("online runtime receipt schema differs")
    if not isinstance(run_claim, VerifiedRunClaimCapability):
        raise ProviderPhaseRuntimeError("online sealed launch lacks in-memory claim authority")
    control = OnlineSealedLaunchDriverControl.from_bytes(
        _verified_file(Path(row.control_path), row.control_file_sha256, label="online control")
    )
    from .production_controls import verify_production_run_closure_authority
    from .runtime_attestation import loads_runtime_preflight_receipt
    from .sealed_container_launcher import (
        launch_sealed_once,
        load_preflight_launch_contract,
        load_registered_plan_instantiation,
        load_runtime_plan_transition,
        load_sealed_launch_contract,
        load_volume_initialization_receipt,
    )

    preflight = load_preflight_launch_contract(control.preflight_contract_path)
    try:
        preflight_receipt = loads_runtime_preflight_receipt(
            _secure_file_bytes(
                Path(control.preflight_receipt_path), label="runtime preflight receipt"
            )
        )
    except Exception as exc:
        raise ProviderPhaseRuntimeError("runtime preflight receipt is invalid") from exc
    transition = load_runtime_plan_transition(control.transition_receipt_path)
    verified_closure = verify_production_run_closure_authority(
        finalization_request_path=control.finalization_request_path,
        finalization_receipt_path=control.finalization_receipt_path,
        preflight=preflight,
        transition=transition,
    )
    sealed = load_sealed_launch_contract(control.sealed_contract_path)
    if sealed.geometry.corpus_id != row.corpus_id or Path(sealed.geometry.copy_output_root) != Path(
        row.output_root
    ):
        raise ProviderPhaseRuntimeError("online sealed launch changes corpus or output root")
    launch_sealed_once(
        sealed,
        preflight,
        preflight_receipt,
        transition,
        load_registered_plan_instantiation(control.instantiation_receipt_path),
        verified_closure,
        load_volume_initialization_receipt(control.volume_receipt_path),
        run_claim,
        secret=claim_bytes,
        audit_root=control.audit_root,
    )


def _run_label_release(
    row: ProviderDriverRequest,
    claim_bytes: bytes,
    phase_claim: VerifiedPhaseClaimCapability,
) -> None:
    if not isinstance(phase_claim, VerifiedPhaseClaimCapability):
        raise ProviderPhaseRuntimeError("label release lacks in-memory claim authority")
    receipt = _load_phase_runtime_claim(
        claim_bytes,
        phase=LABEL_RELEASE_PHASE,
        corpus_id=row.corpus_id,
    )
    expected = phase_claim.require_input(
        corpus_id=receipt.corpus_id,
        input_uri=receipt.input_uri,
        input_sha256=receipt.input_sha256,
        supporting_input_uri=receipt.supporting_input_uri,
        supporting_input_sha256=receipt.supporting_input_sha256,
    )
    if expected.canonical_file_bytes() != claim_bytes:
        raise ProviderPhaseRuntimeError(
            "label runtime receipt differs from freshly admitted authority"
        )
    control_bytes = _verified_file(
        Path(row.control_path), row.control_file_sha256, label="label-release driver control"
    )
    control = LabelReleaseDriverControl.from_bytes(control_bytes)
    if Path(control.plaintext_output_path).parent != Path(row.output_root):
        raise ProviderPhaseRuntimeError("label plaintext output differs from its driver root")
    if Path(control.decryption_receipt_path).parent != Path(row.output_root):
        raise ProviderPhaseRuntimeError("label receipt output differs from its driver root")
    from .cli import main as core_main

    status = core_main(
        [
            "release-timelock-label",
            "--manifest",
            control.manifest_path,
            "--corpus-id",
            row.corpus_id,
            "--custody-seal",
            control.custody_seal_path,
            "--encryption-receipt",
            control.encryption_receipt_path,
            "--completion-receipt",
            control.completion_receipt_path,
            "--completion-anchor-record",
            control.completion_anchor_record_path,
            "--completion-anchor-receipt",
            control.completion_anchor_receipt_path,
            "--suite-namespace",
            control.suite_namespace,
            "--ciphertext",
            control.ciphertext_path,
            "--tle-binary",
            control.tle_binary_path,
            "--plaintext-output",
            control.plaintext_output_path,
            "--receipt",
            control.decryption_receipt_path,
        ]
    )
    if status != 0:
        raise ProviderPhaseRuntimeError("label-release core returned a failure status")


def _verify_pre_decryption_marker(
    row: ProviderDriverRequest,
    authority: FreshLabelClaimAuthority,
    *,
    suite_attempt_id: str,
) -> None:
    encoded = _verified_file(
        Path(authority.admission_marker_path),
        authority.admission_marker_sha256,
        label=f"{row.corpus_id} pre-decryption admission marker",
    )
    marker = _closed(
        _strict_object(encoded, label="pre-decryption admission marker"),
        frozenset(
            {
                "admitted_at_utc",
                "beacon_receipt_sha256",
                "corpus_id",
                "input_sha256",
                "input_uri",
                "live_execute_job_receipt_sha256",
                "output_identity_sha256",
                "output_uri",
                "phase",
                "phase_claim_contract_sha256",
                "phase_claim_ledger_commit",
                "phase_claim_state_sha256",
                "provider_identity_sha256",
                "runtime_claim_receipt_sha256",
                "schema_version",
                "suite_attempt_id",
                "supporting_input_sha256",
                "supporting_input_uri",
            }
        ),
        label="pre-decryption admission marker",
    )
    if encoded != _canonical_bytes(marker) + b"\n":
        raise ProviderPhaseRuntimeError("pre-decryption marker bytes are not canonical")
    receipt = _load_phase_runtime_claim(
        authority.claim_bytes,
        phase=LABEL_RELEASE_PHASE,
        corpus_id=row.corpus_id,
    )
    matches = [
        binding
        for binding in authority.capability.contract.corpora
        if binding.corpus_id == row.corpus_id
    ]
    if len(matches) != 1:
        raise ProviderPhaseRuntimeError("label marker lacks one claimed corpus binding")
    binding = matches[0]
    expected = {
        "beacon_receipt_sha256": receipt.phase_beacon_receipt_sha256,
        "corpus_id": row.corpus_id,
        "input_sha256": receipt.input_sha256,
        "input_uri": receipt.input_uri,
        "live_execute_job_receipt_sha256": receipt.live_execute_job_receipt_sha256,
        "output_identity_sha256": receipt.phase_output_identity,
        "output_uri": binding.output_uri,
        "phase": LABEL_RELEASE_PHASE,
        "phase_claim_contract_sha256": receipt.phase_claim_contract_sha256,
        "phase_claim_ledger_commit": receipt.phase_claim_ledger_commit,
        "phase_claim_state_sha256": receipt.phase_claim_state_sha256,
        "provider_identity_sha256": receipt.provider_identity_sha256,
        "runtime_claim_receipt_sha256": hashlib.sha256(authority.claim_bytes).hexdigest(),
        "schema_version": "fractal-pre-decryption-admission-v1",
        "suite_attempt_id": suite_attempt_id,
        "supporting_input_sha256": receipt.supporting_input_sha256,
        "supporting_input_uri": receipt.supporting_input_uri,
    }
    if any(marker.get(name) != value for name, value in expected.items()):
        raise ProviderPhaseRuntimeError("pre-decryption marker differs from fresh claim authority")
    _text("admitted_at_utc", marker.get("admitted_at_utc"))


def _run_analysis(
    row: ProviderDriverRequest,
    claim_bytes: bytes,
    provider_claimed: VerifiedProviderPredecessor | None,
    phase_claim: VerifiedPhaseClaimCapability | None,
    fresh_claim_supplier: AnalysisClaimSupplier | None = None,
) -> None:
    bundle = AnalysisRuntimeClaimBundle.from_bytes(claim_bytes)
    if not isinstance(provider_claimed, VerifiedProviderPredecessor) or not isinstance(
        phase_claim, VerifiedPhaseClaimCapability
    ):
        raise ProviderPhaseRuntimeError("analysis driver lacks provider claim authority")
    for receipt in bundle.receipts:
        fresh_receipt = phase_claim.require_input(
            corpus_id=receipt.corpus_id,
            input_uri=receipt.input_uri,
            input_sha256=receipt.input_sha256,
            supporting_input_uri=receipt.supporting_input_uri,
            supporting_input_sha256=receipt.supporting_input_sha256,
        )
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "c1_commit",
            "phase_claim_contract_sha256",
            "phase_claim_state_sha256",
            "phase_claim_ledger_commit",
            "provider_identity_sha256",
            "phase_input_aggregate_sha256",
            "phase_output_identity",
            "corpus_id",
            "input_uri",
            "input_sha256",
            "supporting_input_uri",
            "supporting_input_sha256",
        ):
            if getattr(fresh_receipt, name) != getattr(receipt, name):
                raise ProviderPhaseRuntimeError(
                    "analysis start authority differs from the fixed input intent"
                )
    from .confirmatory_input_operator import (
        load_confirmatory_input_operator_config,
        run_provider_claimed_confirmatory_analysis_once,
    )

    candidate = run_provider_claimed_confirmatory_analysis_once(
        load_confirmatory_input_operator_config(row.control_path),
        provider_claimed,
        phase_claim,
        fresh_claim_supplier=fresh_claim_supplier,
    )
    if candidate.state != "ANALYSIS_COMPLETE":
        raise ProviderPhaseRuntimeError("analysis driver did not reach candidate closure")


_DRIVERS: dict[ProviderPhase, Callable[[ProviderDriverRequest, bytes], None]] = {}


def execute_provider_phase_request(
    *,
    plan: ProviderPhasePlan,
    request: ProviderPhaseRuntimeRequest,
    online_run_claim: VerifiedRunClaimCapability | None = None,
    online_run_claim_supplier: OnlineRunClaimSupplier | None = None,
    label_phase_claim_supplier: LabelPhaseClaimSupplier | None = None,
    provider_claimed: VerifiedProviderPredecessor | None = None,
    analysis_phase_claim: VerifiedPhaseClaimCapability | None = None,
    analysis_claim_supplier: AnalysisClaimSupplier | None = None,
) -> ProviderPhaseExecutionReceipt:
    """Execute the exact canonical request without caller-selected driver data."""

    if not isinstance(plan, ProviderPhasePlan) or plan.schema_version != PROVIDER_PHASE_PLAN_SCHEMA:
        raise ProviderPhaseRuntimeError("provider runtime requires a typed resolved C1 plan")
    if not isinstance(request, ProviderPhaseRuntimeRequest):
        raise ProviderPhaseRuntimeError("provider runtime requires a typed request")
    if (
        request.phase != plan.phase
        or request.activation_command_id != plan.activation_command_id
        or request.suite_attempt_id != plan.suite_attempt_id
        or request.provider_plan_path != plan.provider_plan_path
        or request.provider_plan_sha256 != plan.plan_sha256
        or request.provider_plan_file_sha256 != plan.file_sha256
        or request.claim_receipt_path != plan.claim_receipt_path(request.suite_attempt_id)
        or request.phase_output_root != plan.phase_evidence_root(request.suite_attempt_id)
    ):
        raise ProviderPhaseRuntimeError("runtime request differs from the resolved C1 plan")
    plan_bytes = _verified_file(
        Path(request.provider_plan_path),
        request.provider_plan_file_sha256,
        label="resolved provider plan",
    )
    if plan_bytes != plan.canonical_file_bytes():
        raise ProviderPhaseRuntimeError("runtime plan bytes differ after typed admission")
    _verified_file(
        Path(request.claim_receipt_path),
        request.claim_receipt_file_sha256,
        label="provider claim receipt",
    )
    request_path = Path(request.phase_input_root) / PROVIDER_RUNTIME_REQUEST_FILENAME
    request_bytes = _verified_file(
        request_path,
        request.file_sha256,
        label="provider runtime request",
    )
    if request_bytes != request.canonical_file_bytes():
        raise ProviderPhaseRuntimeError("runtime request changed after typed admission")
    if online_run_claim is not None and online_run_claim_supplier is not None:
        raise ProviderPhaseRuntimeError("online runtime received two claim authorities")
    if request.phase == LABEL_RELEASE_PHASE and label_phase_claim_supplier is None:
        raise ProviderPhaseRuntimeError("label runtime lacks a fresh claim supplier")
    phase_output_root = Path(request.phase_output_root)
    if online_run_claim_supplier is not None:
        if _controlled_directory_entries(phase_output_root, label="online phase output root"):
            raise ProviderPhaseRuntimeError("online phase output root is not empty")
    if label_phase_claim_supplier is not None:
        expected = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
        if (
            _controlled_directory_entries(phase_output_root, label="label phase output root")
            != expected
        ):
            raise ProviderPhaseRuntimeError("label phase output roots differ from fixed corpora")
        for corpus_id in expected:
            if _controlled_directory_entries(
                phase_output_root / corpus_id,
                label=f"{corpus_id} label output root",
            ):
                raise ProviderPhaseRuntimeError(f"{corpus_id} label output root is not empty")
    if request.phase == ANALYSIS_PHASE:
        if not isinstance(analysis_phase_claim, VerifiedPhaseClaimCapability):
            raise ProviderPhaseRuntimeError("analysis runtime lacks phase claim authority")
        output_uris = {row.output_uri for row in analysis_phase_claim.contract.corpora}
        if len(output_uris) != 1:
            raise ProviderPhaseRuntimeError("analysis claim has no sole output namespace")
        parsed = urlsplit(next(iter(output_uris)))
        authorized_output = Path(unquote(parsed.path))
        if (
            parsed.scheme != "file"
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or authorized_output.as_uri() != next(iter(output_uris))
            or request.drivers[0].output_root != str(authorized_output)
        ):
            raise ProviderPhaseRuntimeError(
                "analysis driver output differs from the claimed results store"
            )
        if _controlled_directory_entries(phase_output_root, label="analysis phase evidence root"):
            raise ProviderPhaseRuntimeError("analysis phase evidence root is not empty")
        if _controlled_directory_entries(authorized_output, label="analysis results store"):
            raise ProviderPhaseRuntimeError(
                "analysis results store is not empty before input materialization"
            )

    outputs: list[ProviderDriverOutput] = []
    for row in request.drivers:
        _verified_file(
            Path(row.control_path),
            row.control_file_sha256,
            label=f"{row.corpus_id} driver control",
        )
        claim_bytes = _verified_file(
            Path(row.runtime_claim_receipt_path),
            row.runtime_claim_receipt_file_sha256,
            label=f"{row.corpus_id} runtime claim receipt",
        )
        if request.phase == ONLINE_PHASE and online_run_claim_supplier is not None:
            fresh_online = online_run_claim_supplier(row)
            if not isinstance(fresh_online, FreshOnlineClaimAuthority):
                raise ProviderPhaseRuntimeError("online supplier returned untyped authority")
            _run_online(row, fresh_online.claim_bytes, fresh_online.capability)
        elif (
            request.phase == ONLINE_PHASE and online_run_claim is None and ONLINE_PHASE in _DRIVERS
        ):
            _DRIVERS[ONLINE_PHASE](row, claim_bytes)
        elif request.phase == ONLINE_PHASE:
            _run_online(row, claim_bytes, online_run_claim)
        elif request.phase == ANALYSIS_PHASE:
            if analysis_claim_supplier is None:
                raise ProviderPhaseRuntimeError("analysis runtime lacks a fresh claim supplier")
            start_claimed, start_phase_claim = analysis_claim_supplier()
            _run_analysis(
                row,
                claim_bytes,
                start_claimed,
                start_phase_claim,
                analysis_claim_supplier,
            )
        elif request.phase == LABEL_RELEASE_PHASE and label_phase_claim_supplier is not None:
            fresh_label = label_phase_claim_supplier(row)
            if not isinstance(fresh_label, FreshLabelClaimAuthority):
                raise ProviderPhaseRuntimeError("label supplier returned untyped authority")
            _verify_pre_decryption_marker(
                row,
                fresh_label,
                suite_attempt_id=request.suite_attempt_id,
            )
            _run_label_release(
                row,
                fresh_label.claim_bytes,
                fresh_label.capability,
            )
        elif request.phase in _DRIVERS:
            _DRIVERS[request.phase](row, claim_bytes)
        else:
            raise ProviderPhaseRuntimeError("provider phase lacks an in-memory execution authority")
        output = _output_receipt(row)
        if request.phase == ANALYSIS_PHASE:
            from .confirmatory_input_operator import confirmatory_store_closure_filenames

            expected_entries = confirmatory_store_closure_filenames(
                analysis_phase_claim.contract.manifest_sha256
            )
            if output.output_entries != expected_entries:
                raise ProviderPhaseRuntimeError(
                    "analysis output tree differs from the exact five-file closure"
                )
        outputs.append(output)

    receipt = ProviderPhaseExecutionReceipt(
        phase=request.phase,
        suite_attempt_id=request.suite_attempt_id,
        provider_plan_sha256=request.provider_plan_sha256,
        provider_plan_file_sha256=request.provider_plan_file_sha256,
        claim_receipt_file_sha256=request.claim_receipt_file_sha256,
        runtime_request_sha256=request.request_sha256,
        runtime_request_file_sha256=request.file_sha256,
        outputs=tuple(outputs),
    )
    target = Path(request.phase_output_root) / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME
    try:
        write_exclusive_receipt_bytes(receipt.canonical_file_bytes(), target)
    except ArtifactIntegrityError as exc:
        raise ProviderPhaseRuntimeError("provider phase execution receipt already exists") from exc
    if digest_regular_file(target, label="provider phase execution receipt") != receipt.file_sha256:
        raise ProviderPhaseRuntimeError("provider phase execution receipt failed readback")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.provider_phase_runtime",
        description="Execute one C1-bound provider phase request.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in PROVIDER_PHASE_COMMAND_IDS.values():
        item = commands.add_parser(command)
        item.add_argument("--provider-plan", type=Path, required=True)
        item.add_argument("--suite-attempt-id", required=True)
        item.add_argument("--claim-receipt", type=Path, required=True)
        item.add_argument("--phase-input-root", type=Path, required=True)
        item.add_argument("--phase-output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        plan = load_materialized_provider_phase_plan(arguments.provider_plan)
        # This fixed local receipt is admitted before the request root is even opened.
        # The hosted claim job validates the embedded C1 copy; the self-hosted runner
        # must additionally prove byte equality with its own bootstrap record.
        load_provider_runner_bootstrap(plan)
        phase = next(
            phase
            for phase, command in PROVIDER_PHASE_COMMAND_IDS.items()
            if command == arguments.command
        )
        request_path = arguments.phase_input_root / PROVIDER_RUNTIME_REQUEST_FILENAME
        request = load_provider_phase_runtime_request(request_path)
        if (
            plan.phase != phase
            or arguments.suite_attempt_id != plan.suite_attempt_id
            or arguments.claim_receipt != Path(plan.claim_receipt_path(plan.suite_attempt_id))
            or arguments.phase_output_root != Path(plan.phase_evidence_root(plan.suite_attempt_id))
            or request.phase_input_root != str(arguments.phase_input_root)
        ):
            raise ProviderPhaseRuntimeError("provider runtime argv differs from the C1 bindings")
        receipt = execute_provider_phase_request(plan=plan, request=request)
    except (ExecutionClaimError, ProviderPhaseRuntimeError) as exc:
        print(f"provider-phase-runtime error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "execution_receipt_file_sha256": receipt.file_sha256,
                "execution_receipt_path": str(
                    Path(request.phase_output_root) / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME
                ),
                "execution_receipt_sha256": receipt.receipt_sha256,
                "phase": receipt.phase,
                "suite_attempt_id": receipt.suite_attempt_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
