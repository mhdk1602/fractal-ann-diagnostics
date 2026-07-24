"""Typed production-runner registration and post-C0 activation.

Before A exists, the registration command writes the sole P-bound runner receipt
accepted by provider-plan finalization.  After A and C1 exist, the activation
command replaces that candidate binding with the fixed A-bound bootstrap
receipt.  Both transitions derive their output paths, read GitHub twice, and
publish closed owner-private bundles without caller-supplied identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifact_integrity import (
    ArtifactIntegrityError,
    _open_absolute_directory,
    digest_regular_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .execution_claim import (
    ANALYSIS_PHASE,
    LABEL_RELEASE_PHASE,
    ONLINE_PHASE,
    PROVIDER_APPROVAL_ENVIRONMENT,
    PROVIDER_RUNNER_IDENTITY,
    ExecutionClaimError,
    ProviderPhase,
    ProviderPhasePlan,
    ProviderRunnerBootstrapReceipt,
    derive_phase_runner_label,
    load_provider_phase_plans,
    required_execute_runner_labels,
)
from .production_controls import (
    ProductionControlC0InstantiationReceipt,
    ProductionControlError,
)
from .provider_plan_operator import (
    PROVIDER_PLAN_BLUEPRINT_FILENAME,
    ProviderPlanBlueprint,
    ProviderPlanBlueprintWriteReceipt,
    ProviderPlanOperatorError,
    ProviderRunnerExpectation,
    _publish_private_bundle,
    _read_closed_bundle,
    _revalidate_blueprint_sources,
    load_provider_plan_blueprint_bundle,
)
from .provider_rehearsal import (
    CandidateImageClosure,
    GitHubBytesApi,
    GitHubCliBytesApi,
    ProviderRehearsalError,
    RepositoryRunnerInventoryReceipt,
    RepositoryRunnerSnapshot,
    capture_repository_runner_inventory,
)

PROVIDER_RUNNER_ACTIVATION_SCHEMA = "fractal-provider-runner-activation-v2"
PROVIDER_RUNNER_REGISTRATION_SCHEMA = "fractal-provider-runner-registration-v2"
PROVIDER_RUNNER_REGISTRATION_BUNDLE_DERIVATION = (
    "sha256-fractal-provider-runner-registration-bundle-v1"
)
BOOTSTRAP_RECEIPT_FILENAME = "bootstrap-receipt.json"
REGISTRATION_RECEIPT_FILENAME = "registration-receipt.json"
INVENTORY_RECEIPT_FILENAME = "repository-runner-inventory.json"
RAW_INVENTORY_FILENAME = "repository-runners-api.raw.json"
ACTIVATION_RECEIPT_FILENAME = "provider-runner-activation-receipt.json"
REGISTRATION_EVIDENCE_FILENAME = "provider-runner-registration-receipt.json"
REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
PHASES: tuple[ProviderPhase, ...] = (
    ONLINE_PHASE,
    LABEL_RELEASE_PHASE,
    ANALYSIS_PHASE,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_OCI_REFERENCE = re.compile(r"^[^\s/@]+(?:/[^\s/@]+)+@sha256:[0-9a-f]{64}$")
_MAX_JSON_BYTES = 16 * 1024 * 1024


class ProviderRunnerActivationError(RuntimeError):
    """Raised when production runner activation cannot be authenticated."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ProviderRunnerActivationError("activation value is not canonical JSON") from exc


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderRunnerActivationError(f"{name} must be nonempty canonical text")
    return value


def _digest(name: str, value: object) -> str:
    text = _text(name, value)
    if _SHA256.fullmatch(text) is None:
        raise ProviderRunnerActivationError(f"{name} must be a lowercase SHA-256")
    return text


def _commit(name: str, value: object) -> str:
    text = _text(name, value)
    if _GIT_COMMIT.fullmatch(text) is None:
        raise ProviderRunnerActivationError(f"{name} must be one full Git commit")
    return text


def _absolute_path(name: str, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ProviderRunnerActivationError(f"{name} must be one absolute path")
    return path


def _timestamp(name: str, value: object) -> str:
    text = _text(name, value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProviderRunnerActivationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ProviderRunnerActivationError(f"{name} must be UTC")
    return text


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProviderRunnerActivationError(f"{label} must be one JSON object")
    if set(value) != fields:
        raise ProviderRunnerActivationError(f"{label} fields differ")
    return value


def _parse_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    if (
        not encoded
        or len(encoded) > _MAX_JSON_BYTES
        or not encoded.endswith(b"\n")
        or encoded.endswith(b"\n\n")
    ):
        raise ProviderRunnerActivationError(f"{label} is not bounded canonical JSON")
    try:
        value = json.loads(encoded[:-1])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderRunnerActivationError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping) or _canonical_bytes(value) + b"\n" != encoded:
        raise ProviderRunnerActivationError(f"{label} bytes are not canonical")
    return value


def _read(path: Path, *, label: str) -> bytes:
    try:
        return read_secure_regular_file(path, max_bytes=_MAX_JSON_BYTES, label=label)
    except ArtifactIntegrityError as exc:
        raise ProviderRunnerActivationError(f"cannot read {label}: {exc}") from exc


def _candidate_closure_from_bytes(encoded: bytes) -> CandidateImageClosure:
    row = _closed(
        _parse_object(encoded, label="candidate image closure"),
        frozenset(CandidateImageClosure.__dataclass_fields__),
        label="candidate image closure",
    )
    try:
        closure = CandidateImageClosure(**row)  # type: ignore[arg-type]
    except (ProviderRehearsalError, TypeError, ValueError) as exc:
        raise ProviderRunnerActivationError("candidate image closure is invalid") from exc
    if _canonical_bytes(closure.to_dict()) + b"\n" != encoded:
        raise ProviderRunnerActivationError("candidate image closure typed bytes differ")
    return closure


def _c0_instantiation_from_bytes(
    encoded: bytes,
) -> ProductionControlC0InstantiationReceipt:
    row = _parse_object(encoded, label="C0 control instantiation receipt")
    try:
        receipt = ProductionControlC0InstantiationReceipt.from_dict(row)
    except (ProductionControlError, TypeError, ValueError) as exc:
        raise ProviderRunnerActivationError("C0 control instantiation receipt is invalid") from exc
    if receipt.canonical_file_bytes() != encoded:
        raise ProviderRunnerActivationError("C0 control instantiation typed bytes differ")
    return receipt


def _inventory_from_bytes(encoded: bytes) -> RepositoryRunnerInventoryReceipt:
    row = _closed(
        _parse_object(encoded, label="repository runner inventory"),
        frozenset(RepositoryRunnerInventoryReceipt.__dataclass_fields__),
        label="repository runner inventory",
    )
    raw_runners = row["runners"]
    if not isinstance(raw_runners, list):
        raise ProviderRunnerActivationError("repository runner inventory runners differ")
    try:
        inventory = RepositoryRunnerInventoryReceipt(
            **{key: item for key, item in row.items() if key != "runners"},
            runners=tuple(RepositoryRunnerSnapshot.from_dict(item) for item in raw_runners),
        )
    except (ProviderRehearsalError, TypeError, ValueError) as exc:
        raise ProviderRunnerActivationError("repository runner inventory is invalid") from exc
    if _canonical_bytes(inventory.to_dict()) + b"\n" != encoded:
        raise ProviderRunnerActivationError("repository runner inventory typed bytes differ")
    return inventory


def _load_fixed_provider_plan(path: Path) -> ProviderPhasePlan:
    """Read one owner-private plan through bound parent and file descriptors."""

    try:
        parent = _open_absolute_directory(path.parent, label="fixed provider plan parent")
    except ArtifactIntegrityError as exc:
        raise ProviderRunnerActivationError("cannot open fixed provider plan parent") from exc
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path.name, flags, dir_fd=parent)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_JSON_BYTES
            or stat.S_IMODE(before.st_mode) != 0o600
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
        ):
            raise ProviderRunnerActivationError(
                "fixed provider plan must be one owner mode-0600 file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OSError("short fixed provider plan read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise OSError("fixed provider plan grew during read")
        after = os.fstat(descriptor)
    except ProviderRunnerActivationError:
        raise
    except OSError as exc:
        raise ProviderRunnerActivationError("cannot read fixed provider plan") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ProviderRunnerActivationError("fixed provider plan changed during read")
    encoded = b"".join(chunks)
    try:
        plan = ProviderPhasePlan.from_dict(
            _parse_object(encoded, label="fixed materialized provider plan")
        )
    except ExecutionClaimError as exc:
        raise ProviderRunnerActivationError("fixed materialized provider plan is invalid") from exc
    if (
        plan.canonical_file_bytes() != encoded
        or Path(plan.provider_plan_path) != path
        or plan.file_sha256 != _sha256(encoded)
    ):
        raise ProviderRunnerActivationError("fixed materialized provider plan bytes differ")
    return plan


@dataclass(frozen=True)
class ProviderRunnerRegistrationReceipt:
    """P-bound registration evidence created before apparatus commit A exists."""

    phase: ProviderPhase
    repository: str
    approval_environment: str
    runner_identity: str
    blueprint_directory: str
    blueprint_file_sha256: str
    blueprint_write_receipt_file_sha256: str
    candidate_manifest_file_sha256: str
    candidate_image_source_commit: str
    build_context_tree_sha256: str
    candidate_bootstrap_closure_sha256: str
    scientific_image_index_digest: str
    release_image_index_digest: str
    host_tool_contract_sha256: str
    claim_nonce: str
    runner_label: str
    registration_receipt_path: str
    registration_receipt_file_sha256: str
    registration_inventory_path: str
    registration_inventory_file_sha256: str
    registration_inventory_response_sha256: str
    raw_inventory_path: str
    raw_inventory_sha256: str
    runner_id: int
    runner_name: str
    runner_group_id: int | None
    runner_version: str
    runner_archive_sha256: str
    runner_operating_system: str
    runner_status: str
    runner_busy: bool
    runner_labels: tuple[str, ...]
    captured_at_utc: str
    schema_version: str = PROVIDER_RUNNER_REGISTRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in PHASES or self.repository != REPOSITORY:
            raise ProviderRunnerActivationError("registration phase or repository differs")
        if (
            self.approval_environment != PROVIDER_APPROVAL_ENVIRONMENT
            or self.runner_identity != PROVIDER_RUNNER_IDENTITY
            or self.runner_identity != f"github-actions:environment:{self.approval_environment}"
        ):
            raise ProviderRunnerActivationError("registration approval environment differs")
        for name in (
            "blueprint_file_sha256",
            "blueprint_write_receipt_file_sha256",
            "candidate_manifest_file_sha256",
            "build_context_tree_sha256",
            "candidate_bootstrap_closure_sha256",
            "host_tool_contract_sha256",
            "claim_nonce",
            "registration_receipt_file_sha256",
            "registration_inventory_file_sha256",
            "registration_inventory_response_sha256",
            "raw_inventory_sha256",
            "runner_archive_sha256",
        ):
            _digest(name, getattr(self, name))
        _commit("candidate_image_source_commit", self.candidate_image_source_commit)
        for name in ("scientific_image_index_digest", "release_image_index_digest"):
            value = _text(name, getattr(self, name))
            if _OCI_DIGEST.fullmatch(value) is None:
                raise ProviderRunnerActivationError(f"{name} must be one OCI digest")
        for name in (
            "blueprint_directory",
            "registration_receipt_path",
            "registration_inventory_path",
            "raw_inventory_path",
        ):
            _absolute_path(name, getattr(self, name))
        if type(self.runner_id) is not int or self.runner_id <= 0:
            raise ProviderRunnerActivationError("registration runner_id must be positive")
        _text("runner_name", self.runner_name)
        _text("runner_version", self.runner_version)
        if self.runner_group_id is not None:
            raise ProviderRunnerActivationError("personal repository runner_group_id must be null")
        if self.runner_operating_system != "macOS":
            raise ProviderRunnerActivationError("registration runner operating system differs")
        if self.runner_status != "offline" or self.runner_busy is not False:
            raise ProviderRunnerActivationError(
                "registered runner must be stopped before publication"
            )
        expected_label = derive_phase_runner_label(self.claim_nonce, self.phase)
        if self.runner_label != expected_label:
            raise ProviderRunnerActivationError("registration label differs from claim nonce")
        labels = tuple(self.runner_labels)
        expected_labels = tuple(
            sorted(required_execute_runner_labels(expected_label), key=lambda item: item.encode())
        )
        if labels != expected_labels:
            raise ProviderRunnerActivationError("registered runner labels differ")
        object.__setattr__(self, "runner_labels", labels)
        if self.registration_inventory_response_sha256 != self.raw_inventory_sha256:
            raise ProviderRunnerActivationError("registration inventory raw hash differs")
        root = Path(self.registration_receipt_path).parent
        if (
            Path(self.registration_inventory_path) != root / INVENTORY_RECEIPT_FILENAME
            or Path(self.raw_inventory_path) != root / RAW_INVENTORY_FILENAME
            or Path(self.registration_receipt_path).name != REGISTRATION_RECEIPT_FILENAME
        ):
            raise ProviderRunnerActivationError("registration bundle paths differ")
        _timestamp("captured_at_utc", self.captured_at_utc)
        if self.schema_version != PROVIDER_RUNNER_REGISTRATION_SCHEMA:
            raise ProviderRunnerActivationError("registration receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "runner_labels"
            },
            "runner_labels": list(self.runner_labels),
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProviderRunnerRegistrationReceipt:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="provider runner registration receipt",
        )
        labels = row["runner_labels"]
        if not isinstance(labels, list):
            raise ProviderRunnerActivationError("registration runner_labels must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "runner_labels"},
            runner_labels=tuple(labels),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProviderRunnerActivationReceipt:
    phase: ProviderPhase
    repository: str
    approval_environment: str
    runner_identity: str
    apparatus_commit: str
    c1_commit: str
    manifest_path: str
    manifest_sha256: str
    manifest_file_sha256: str
    c0_instantiation_receipt_path: str
    c0_instantiation_receipt_file_sha256: str
    candidate_image_closure_path: str
    candidate_image_closure_file_sha256: str
    candidate_image_source_commit: str
    build_context_tree_sha256: str
    candidate_bootstrap_closure_sha256: str
    scientific_image_index_digest: str
    release_image_index_digest: str
    provider_plan_path: str
    provider_plan_sha256: str
    provider_plan_file_sha256: str
    claim_nonce: str
    runner_label: str
    bootstrap_receipt_path: str
    bootstrap_receipt_file_sha256: str
    registration_inventory_file_sha256: str
    activation_inventory_path: str
    activation_inventory_file_sha256: str
    activation_inventory_response_sha256: str
    raw_inventory_path: str
    raw_inventory_sha256: str
    runtime_image: str
    runtime_index_digest: str
    runtime_platform_manifest_digest: str
    runtime_probe_receipt_sha256: str
    runner_id: int
    runner_name: str
    runner_group_id: int | None
    runner_version: str
    runner_archive_sha256: str
    runner_operating_system: str
    runner_status: str
    runner_busy: bool
    runner_labels: tuple[str, ...]
    captured_at_utc: str
    schema_version: str = PROVIDER_RUNNER_ACTIVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in PHASES or self.repository != REPOSITORY:
            raise ProviderRunnerActivationError("activation phase or repository differs")
        if (
            self.approval_environment != PROVIDER_APPROVAL_ENVIRONMENT
            or self.runner_identity != PROVIDER_RUNNER_IDENTITY
            or self.runner_identity != f"github-actions:environment:{self.approval_environment}"
        ):
            raise ProviderRunnerActivationError("activation approval environment differs")
        _commit("apparatus_commit", self.apparatus_commit)
        _commit("c1_commit", self.c1_commit)
        for name in (
            "manifest_sha256",
            "manifest_file_sha256",
            "c0_instantiation_receipt_file_sha256",
            "candidate_image_closure_file_sha256",
            "build_context_tree_sha256",
            "candidate_bootstrap_closure_sha256",
            "provider_plan_sha256",
            "provider_plan_file_sha256",
            "claim_nonce",
            "bootstrap_receipt_file_sha256",
            "registration_inventory_file_sha256",
            "activation_inventory_file_sha256",
            "activation_inventory_response_sha256",
            "raw_inventory_sha256",
            "runtime_probe_receipt_sha256",
            "runner_archive_sha256",
        ):
            _digest(name, getattr(self, name))
        _commit("candidate_image_source_commit", self.candidate_image_source_commit)
        for name in ("scientific_image_index_digest", "release_image_index_digest"):
            value = _text(name, getattr(self, name))
            if _OCI_DIGEST.fullmatch(value) is None:
                raise ProviderRunnerActivationError(f"{name} must be one OCI digest")
        if (
            _OCI_DIGEST.fullmatch(self.runtime_index_digest) is None
            or _OCI_DIGEST.fullmatch(self.runtime_platform_manifest_digest) is None
        ):
            raise ProviderRunnerActivationError("activation runtime OCI digests differ")
        if _OCI_REFERENCE.fullmatch(self.runtime_image) is None:
            raise ProviderRunnerActivationError("activation runtime image is not immutable")
        for name in (
            "manifest_path",
            "c0_instantiation_receipt_path",
            "candidate_image_closure_path",
            "provider_plan_path",
            "bootstrap_receipt_path",
            "activation_inventory_path",
            "raw_inventory_path",
        ):
            _absolute_path(name, getattr(self, name))
        _text("runner_name", self.runner_name)
        _text("runner_version", self.runner_version)
        if type(self.runner_id) is not int or self.runner_id <= 0:
            raise ProviderRunnerActivationError("runner_id must be positive")
        if self.runner_group_id is not None:
            raise ProviderRunnerActivationError("personal repository runner_group_id must be null")
        if self.runner_operating_system != "macOS":
            raise ProviderRunnerActivationError("production runner operating system differs")
        if self.runner_status != "offline" or self.runner_busy is not False:
            raise ProviderRunnerActivationError(
                "production runner must be stopped before activation"
            )
        expected_label = derive_phase_runner_label(self.claim_nonce, self.phase)
        if self.runner_label != expected_label:
            raise ProviderRunnerActivationError("production runner label differs from claim nonce")
        labels = tuple(self.runner_labels)
        expected_labels = tuple(
            sorted(required_execute_runner_labels(expected_label), key=lambda item: item.encode())
        )
        if labels != expected_labels:
            raise ProviderRunnerActivationError("production runner labels differ")
        object.__setattr__(self, "runner_labels", labels)
        if self.activation_inventory_response_sha256 != self.raw_inventory_sha256:
            raise ProviderRunnerActivationError("activation inventory raw-response hash differs")
        root = Path(self.bootstrap_receipt_path).parent
        if (
            Path(self.activation_inventory_path) != root / INVENTORY_RECEIPT_FILENAME
            or Path(self.raw_inventory_path) != root / RAW_INVENTORY_FILENAME
            or Path(self.bootstrap_receipt_path).name != BOOTSTRAP_RECEIPT_FILENAME
        ):
            raise ProviderRunnerActivationError("activation bundle paths differ")
        _timestamp("captured_at_utc", self.captured_at_utc)
        if self.schema_version != PROVIDER_RUNNER_ACTIVATION_SCHEMA:
            raise ProviderRunnerActivationError("activation receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "runner_labels"
            },
            "runner_labels": list(self.runner_labels),
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProviderRunnerActivationReceipt:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="provider runner activation receipt",
        )
        labels = row["runner_labels"]
        if not isinstance(labels, list):
            raise ProviderRunnerActivationError("activation runner_labels must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "runner_labels"},
            runner_labels=tuple(labels),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class _RegistrationContext:
    blueprint_directory: Path
    blueprint: ProviderPlanBlueprint
    blueprint_write_receipt: ProviderPlanBlueprintWriteReceipt
    expectation: ProviderRunnerExpectation
    candidate_closure: CandidateImageClosure


@dataclass(frozen=True)
class _ActivationContext:
    manifest_path: Path
    manifest_file_sha256: str
    c0_instantiation_path: Path
    c0_instantiation_file_sha256: str
    c0_instantiation: ProductionControlC0InstantiationReceipt
    candidate_closure_path: Path
    candidate_closure_file_sha256: str
    candidate_closure: CandidateImageClosure
    plan: ProviderPhasePlan


def _admit_registration_sources(
    *,
    blueprint_directory: str | Path,
    phase: ProviderPhase,
) -> _RegistrationContext:
    """Reauthenticate the closed pre-A blueprint and select one runner slot."""

    directory = _absolute_path("blueprint_directory", blueprint_directory)
    if phase not in PHASES:
        raise ProviderRunnerActivationError("registration phase differs")
    try:
        blueprint, write_receipt = load_provider_plan_blueprint_bundle(directory)
        _manifest, closure, _config, host_tools = _revalidate_blueprint_sources(blueprint)
    except ProviderPlanOperatorError as exc:
        raise ProviderRunnerActivationError(
            "provider-plan blueprint cannot authorize runner registration"
        ) from exc
    if (
        write_receipt.blueprint_path != directory / PROVIDER_PLAN_BLUEPRINT_FILENAME
        or write_receipt.blueprint_file_sha256 != blueprint.file_sha256
        or write_receipt.candidate_manifest_file_sha256 != blueprint.candidate_manifest_file_sha256
        or write_receipt.candidate_image_closure_file_sha256
        != blueprint.candidate_image_closure_file_sha256
        or write_receipt.host_tool_contract_sha256 != blueprint.host_tools.contract_sha256
        or blueprint.approval_environment != PROVIDER_APPROVAL_ENVIRONMENT
        or blueprint.runner_identity != PROVIDER_RUNNER_IDENTITY
        or blueprint.runner_identity
        != f"github-actions:environment:{blueprint.approval_environment}"
        or host_tools != blueprint.host_tools
        or closure.github_sha != blueprint.candidate_image_source_commit
        or closure.build_context_tree_sha256 != blueprint.build_context_tree_sha256
        or closure.bootstrap_closure_sha256 != blueprint.candidate_bootstrap_closure_sha256
        or not blueprint.scientific_candidate_reference.endswith(
            f"@{closure.scientific_image_index_digest}"
        )
        or not blueprint.scientific_production_reference.endswith(
            f"@{closure.scientific_image_index_digest}"
        )
        or not blueprint.release_candidate_reference.endswith(
            f"@{closure.release_image_index_digest}"
        )
        or not blueprint.release_production_reference.endswith(
            f"@{closure.release_image_index_digest}"
        )
    ):
        raise ProviderRunnerActivationError("registration blueprint P/T/D closure differs")
    expectations = tuple(row for row in blueprint.runner_expectations if row.phase == phase)
    if len(expectations) != 1:
        raise ProviderRunnerActivationError("registration runner expectation is not unique")
    expectation = expectations[0]
    if expectation.runner_label != derive_phase_runner_label(expectation.claim_nonce, phase):
        raise ProviderRunnerActivationError("registration runner label derivation differs")
    return _RegistrationContext(
        blueprint_directory=directory,
        blueprint=blueprint,
        blueprint_write_receipt=write_receipt,
        expectation=expectation,
        candidate_closure=closure,
    )


def _select_stopped_registration_runner(
    context: _RegistrationContext,
    inventory: RepositoryRunnerInventoryReceipt,
) -> RepositoryRunnerSnapshot:
    expectation = context.expectation
    matches = [row for row in inventory.runners if row.runner_name == expectation.runner_name]
    if len(matches) != 1:
        raise ProviderRunnerActivationError(
            "expected registration runner is not a live inventory singleton"
        )
    runner = matches[0]
    labels = tuple(
        sorted(
            required_execute_runner_labels(expectation.runner_label),
            key=lambda item: item.encode(),
        )
    )
    if (
        runner.operating_system != "macOS"
        or runner.status != "offline"
        or runner.busy is not False
        or runner.labels != labels
    ):
        raise ProviderRunnerActivationError(
            "registration runner must be stopped with the exact derived label"
        )
    return runner


def _registration_output(context: _RegistrationContext) -> Path:
    return (
        context.blueprint.host_tool_sources.controlled_root
        / "production"
        / "runner-registrations"
        / context.expectation.phase
        / context.expectation.runner_label
    )


def _registration_receipts(
    context: _RegistrationContext,
    inventory: RepositoryRunnerInventoryReceipt,
    raw_inventory: bytes,
    runner: RepositoryRunnerSnapshot,
) -> tuple[ProviderRunnerBootstrapReceipt, ProviderRunnerRegistrationReceipt]:
    blueprint = context.blueprint
    expectation = context.expectation
    output = _registration_output(context)
    bootstrap = ProviderRunnerBootstrapReceipt(
        phase=expectation.phase,
        repository=REPOSITORY,
        approval_environment=blueprint.approval_environment,
        runner_identity=blueprint.runner_identity,
        workflow_sha=blueprint.candidate_image_source_commit,
        runner_label=expectation.runner_label,
        runner_id=runner.runner_id,
        runner_name=runner.runner_name,
        runner_group_id=expectation.runner_group_id,
        runner_version=blueprint.host_tools.runner_version,
        runner_archive_sha256=blueprint.host_tools.runner_archive_sha256,
        repository_runner_inventory_sha256=inventory.file_sha256,
        ephemeral=blueprint.host_tools.runner_ephemeral,
        disable_update=blueprint.host_tools.runner_disable_update,
        unattended=blueprint.host_tools.runner_unattended,
        registered_at_utc=inventory.captured_at_utc,
    )
    receipt = ProviderRunnerRegistrationReceipt(
        phase=expectation.phase,
        repository=REPOSITORY,
        approval_environment=blueprint.approval_environment,
        runner_identity=blueprint.runner_identity,
        blueprint_directory=str(context.blueprint_directory),
        blueprint_file_sha256=blueprint.file_sha256,
        blueprint_write_receipt_file_sha256=context.blueprint_write_receipt.file_sha256,
        candidate_manifest_file_sha256=blueprint.candidate_manifest_file_sha256,
        candidate_image_source_commit=blueprint.candidate_image_source_commit,
        build_context_tree_sha256=blueprint.build_context_tree_sha256,
        candidate_bootstrap_closure_sha256=blueprint.candidate_bootstrap_closure_sha256,
        scientific_image_index_digest=context.candidate_closure.scientific_image_index_digest,
        release_image_index_digest=context.candidate_closure.release_image_index_digest,
        host_tool_contract_sha256=blueprint.host_tools.contract_sha256,
        claim_nonce=expectation.claim_nonce,
        runner_label=expectation.runner_label,
        registration_receipt_path=str(output / REGISTRATION_RECEIPT_FILENAME),
        registration_receipt_file_sha256=bootstrap.file_sha256,
        registration_inventory_path=str(output / INVENTORY_RECEIPT_FILENAME),
        registration_inventory_file_sha256=inventory.file_sha256,
        registration_inventory_response_sha256=inventory.response_sha256,
        raw_inventory_path=str(output / RAW_INVENTORY_FILENAME),
        raw_inventory_sha256=_sha256(raw_inventory),
        runner_id=runner.runner_id,
        runner_name=runner.runner_name,
        runner_group_id=expectation.runner_group_id,
        runner_version=blueprint.host_tools.runner_version,
        runner_archive_sha256=blueprint.host_tools.runner_archive_sha256,
        runner_operating_system=runner.operating_system,
        runner_status=runner.status,
        runner_busy=runner.busy,
        runner_labels=runner.labels,
        captured_at_utc=inventory.captured_at_utc,
    )
    return bootstrap, receipt


def load_provider_runner_registration_bundle(
    directory: str | Path,
) -> tuple[
    ProviderRunnerBootstrapReceipt,
    RepositoryRunnerInventoryReceipt,
    bytes,
    ProviderRunnerRegistrationReceipt,
]:
    """Load one closed pre-A registration bundle through a bound root fd."""

    try:
        root, members = _read_closed_bundle(
            directory,
            frozenset(
                {
                    REGISTRATION_RECEIPT_FILENAME,
                    INVENTORY_RECEIPT_FILENAME,
                    RAW_INVENTORY_FILENAME,
                    REGISTRATION_EVIDENCE_FILENAME,
                }
            ),
            label="provider runner registration bundle",
        )
    except ProviderPlanOperatorError as exc:
        raise ProviderRunnerActivationError(
            "provider runner registration bundle is not one closed private directory"
        ) from exc
    try:
        bootstrap = ProviderRunnerBootstrapReceipt.from_dict(
            _parse_object(
                members[REGISTRATION_RECEIPT_FILENAME],
                label="provider runner registration receipt",
            )
        )
    except ExecutionClaimError as exc:
        raise ProviderRunnerActivationError("provider runner registration is invalid") from exc
    if bootstrap.canonical_file_bytes() != members[REGISTRATION_RECEIPT_FILENAME]:
        raise ProviderRunnerActivationError("provider runner registration typed bytes differ")
    inventory = _inventory_from_bytes(members[INVENTORY_RECEIPT_FILENAME])
    raw = members[RAW_INVENTORY_FILENAME]
    receipt = ProviderRunnerRegistrationReceipt.from_dict(
        _parse_object(
            members[REGISTRATION_EVIDENCE_FILENAME],
            label="provider runner registration evidence",
        )
    )
    if receipt.canonical_file_bytes() != members[REGISTRATION_EVIDENCE_FILENAME]:
        raise ProviderRunnerActivationError("provider runner registration evidence bytes differ")
    if (
        receipt.registration_receipt_path != str(root / REGISTRATION_RECEIPT_FILENAME)
        or receipt.registration_inventory_path != str(root / INVENTORY_RECEIPT_FILENAME)
        or receipt.raw_inventory_path != str(root / RAW_INVENTORY_FILENAME)
        or bootstrap.file_sha256 != receipt.registration_receipt_file_sha256
        or inventory.file_sha256 != receipt.registration_inventory_file_sha256
        or inventory.response_sha256 != receipt.registration_inventory_response_sha256
        or _sha256(raw) != receipt.raw_inventory_sha256
        or inventory.captured_at_utc != receipt.captured_at_utc
        or bootstrap.repository_runner_inventory_sha256 != inventory.file_sha256
    ):
        raise ProviderRunnerActivationError("provider runner registration bundle hashes differ")
    matches = [
        row
        for row in inventory.runners
        if row.runner_id == receipt.runner_id and row.runner_name == receipt.runner_name
    ]
    if len(matches) != 1:
        raise ProviderRunnerActivationError("registration inventory lacks its expected runner")
    row = matches[0]
    if (
        row.operating_system != receipt.runner_operating_system
        or row.status != receipt.runner_status
        or row.busy != receipt.runner_busy
        or row.labels != receipt.runner_labels
        or bootstrap.phase != receipt.phase
        or bootstrap.repository != receipt.repository
        or bootstrap.approval_environment != receipt.approval_environment
        or bootstrap.runner_identity != receipt.runner_identity
        or bootstrap.workflow_sha != receipt.candidate_image_source_commit
        or bootstrap.runner_label != receipt.runner_label
        or bootstrap.runner_id != receipt.runner_id
        or bootstrap.runner_name != receipt.runner_name
        or bootstrap.runner_group_id != receipt.runner_group_id
        or bootstrap.runner_version != receipt.runner_version
        or bootstrap.runner_archive_sha256 != receipt.runner_archive_sha256
        or (bootstrap.ephemeral, bootstrap.disable_update, bootstrap.unattended)
        != (True, True, True)
        or bootstrap.registered_at_utc != receipt.captured_at_utc
    ):
        raise ProviderRunnerActivationError("registration inventory/bootstrap identity differs")
    return bootstrap, inventory, raw, receipt


def _registration_bundle_sha256(
    bootstrap: ProviderRunnerBootstrapReceipt,
    inventory: RepositoryRunnerInventoryReceipt,
    raw_inventory: bytes,
    receipt: ProviderRunnerRegistrationReceipt,
) -> str:
    return _sha256(
        _canonical_bytes(
            {
                "derivation": PROVIDER_RUNNER_REGISTRATION_BUNDLE_DERIVATION,
                "members": {
                    INVENTORY_RECEIPT_FILENAME: inventory.file_sha256,
                    RAW_INVENTORY_FILENAME: _sha256(raw_inventory),
                    REGISTRATION_EVIDENCE_FILENAME: receipt.file_sha256,
                    REGISTRATION_RECEIPT_FILENAME: bootstrap.file_sha256,
                },
            }
        )
    )


def _verify_registration_bundle_against_context(
    context: _RegistrationContext,
) -> tuple[
    ProviderRunnerBootstrapReceipt,
    ProviderRunnerRegistrationReceipt,
    str,
]:
    blueprint = context.blueprint
    expectation = context.expectation
    output = _registration_output(context)
    bootstrap, inventory, raw, receipt = load_provider_runner_registration_bundle(output)
    expected = {
        "phase": expectation.phase,
        "repository": REPOSITORY,
        "approval_environment": blueprint.approval_environment,
        "runner_identity": blueprint.runner_identity,
        "blueprint_directory": str(context.blueprint_directory),
        "blueprint_file_sha256": blueprint.file_sha256,
        "blueprint_write_receipt_file_sha256": context.blueprint_write_receipt.file_sha256,
        "candidate_manifest_file_sha256": blueprint.candidate_manifest_file_sha256,
        "candidate_image_source_commit": blueprint.candidate_image_source_commit,
        "build_context_tree_sha256": blueprint.build_context_tree_sha256,
        "candidate_bootstrap_closure_sha256": blueprint.candidate_bootstrap_closure_sha256,
        "scientific_image_index_digest": context.candidate_closure.scientific_image_index_digest,
        "release_image_index_digest": context.candidate_closure.release_image_index_digest,
        "host_tool_contract_sha256": blueprint.host_tools.contract_sha256,
        "claim_nonce": expectation.claim_nonce,
        "runner_label": expectation.runner_label,
        "runner_name": expectation.runner_name,
        "runner_group_id": expectation.runner_group_id,
        "runner_version": blueprint.host_tools.runner_version,
        "runner_archive_sha256": blueprint.host_tools.runner_archive_sha256,
    }
    if any(getattr(receipt, name) != value for name, value in expected.items()):
        raise ProviderRunnerActivationError(
            "runner registration bundle differs from the pre-A blueprint"
        )
    if (
        bootstrap.approval_environment != blueprint.approval_environment
        or bootstrap.runner_identity != blueprint.runner_identity
        or bootstrap.workflow_sha != blueprint.candidate_image_source_commit
        or bootstrap.runner_label != expectation.runner_label
        or bootstrap.runner_name != expectation.runner_name
        or bootstrap.runner_group_id != expectation.runner_group_id
        or bootstrap.runner_version != blueprint.host_tools.runner_version
        or bootstrap.runner_archive_sha256 != blueprint.host_tools.runner_archive_sha256
    ):
        raise ProviderRunnerActivationError(
            "runner registration bootstrap differs from the pre-A blueprint"
        )
    return (
        bootstrap,
        receipt,
        _registration_bundle_sha256(
            bootstrap,
            inventory,
            raw,
            receipt,
        ),
    )


def admit_provider_runner_registration(
    *,
    blueprint_directory: str | Path,
    phase: ProviderPhase,
) -> tuple[
    ProviderRunnerBootstrapReceipt,
    ProviderRunnerRegistrationReceipt,
    str,
]:
    """Admit one closed P-bound bundle for provider-plan finalization."""

    context = _admit_registration_sources(
        blueprint_directory=blueprint_directory,
        phase=phase,
    )
    return _verify_registration_bundle_against_context(context)


def verify_provider_runner_registration(
    *,
    blueprint_directory: str | Path,
    phase: ProviderPhase,
) -> ProviderRunnerRegistrationReceipt:
    """Revalidate a retained P-bound registration against its source blueprint."""

    _bootstrap, receipt, _bundle_sha256 = admit_provider_runner_registration(
        blueprint_directory=blueprint_directory,
        phase=phase,
    )
    return receipt


def write_provider_runner_registration(
    *,
    blueprint_directory: str | Path,
    phase: ProviderPhase,
    api: GitHubBytesApi | None,
    captured_at_utc: str,
) -> ProviderRunnerRegistrationReceipt:
    """Publish the sole P-bound runner registration consumed by plan finalization."""

    captured = _timestamp("captured_at_utc", captured_at_utc)
    context = _admit_registration_sources(
        blueprint_directory=blueprint_directory,
        phase=phase,
    )
    blueprint = context.blueprint
    live_api = api or GitHubCliBytesApi(blueprint.host_tools.gh_executable, os.environ)
    output = _registration_output(context)
    if os.path.lexists(output):
        raise ProviderRunnerActivationError("provider runner registration output already exists")
    try:
        inventory, raw = capture_repository_runner_inventory(
            api=live_api,
            captured_at_utc=captured,
        )
    except ProviderRehearsalError as exc:
        raise ProviderRunnerActivationError("cannot capture registration runner inventory") from exc
    runner = _select_stopped_registration_runner(context, inventory)
    bootstrap, receipt = _registration_receipts(context, inventory, raw, runner)

    def revalidate_before_publish() -> None:
        current = _admit_registration_sources(
            blueprint_directory=blueprint_directory,
            phase=phase,
        )
        if current != context:
            raise ProviderRunnerActivationError("registration sources changed before publish")
        try:
            current_inventory, current_raw = capture_repository_runner_inventory(
                api=live_api,
                captured_at_utc=captured,
            )
        except ProviderRehearsalError as exc:
            raise ProviderRunnerActivationError(
                "cannot revalidate registration runner inventory"
            ) from exc
        _select_stopped_registration_runner(context, current_inventory)
        if current_inventory != inventory or current_raw != raw:
            raise ProviderRunnerActivationError(
                "registration runner inventory changed before publication"
            )

    try:
        _publish_private_bundle(
            output,
            {
                REGISTRATION_RECEIPT_FILENAME: bootstrap.canonical_file_bytes(),
                INVENTORY_RECEIPT_FILENAME: _canonical_bytes(inventory.to_dict()) + b"\n",
                RAW_INVENTORY_FILENAME: raw,
                REGISTRATION_EVIDENCE_FILENAME: receipt.canonical_file_bytes(),
            },
            label="provider runner registration bundle",
            pre_publish=revalidate_before_publish,
        )
    except ProviderPlanOperatorError as exc:
        raise ProviderRunnerActivationError("cannot publish runner registration bundle") from exc
    _readback_bootstrap, readback, _bundle_sha256 = _verify_registration_bundle_against_context(
        context
    )
    if readback != receipt:
        raise ProviderRunnerActivationError("runner registration readback differs")
    return receipt


def _phase_runtime_binding(
    phase: ProviderPhase,
    closure: CandidateImageClosure,
    c0: ProductionControlC0InstantiationReceipt,
) -> tuple[str, str, str, str]:
    if phase == ONLINE_PHASE:
        return (
            c0.scientific_production_reference,
            closure.scientific_image_index_digest,
            closure.scientific_linux_arm64_manifest_digest,
            closure.scientific_linux_arm64_runtime_extraction_sha256,
        )
    if phase == ANALYSIS_PHASE:
        return (
            c0.scientific_production_reference,
            closure.scientific_image_index_digest,
            closure.scientific_linux_amd64_manifest_digest,
            closure.scientific_linux_amd64_runtime_extraction_sha256,
        )
    return (
        "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-release@"
        f"{closure.release_image_index_digest}",
        closure.release_image_index_digest,
        closure.release_linux_arm64_manifest_digest,
        closure.release_reproducibility_receipt_sha256,
    )


def _validate_lineage(
    plan: ProviderPhasePlan,
    c0: ProductionControlC0InstantiationReceipt,
    closure: CandidateImageClosure,
) -> None:
    if (
        plan.approval_environment != c0.approval_environment
        or plan.approval_environment != PROVIDER_APPROVAL_ENVIRONMENT
        or plan.runner_identity != PROVIDER_RUNNER_IDENTITY
        or plan.runner_identity != f"github-actions:environment:{plan.approval_environment}"
        or plan.workflow_sha != c0.apparatus_commit
        or closure.github_sha != c0.candidate_image_source_commit
        or closure.build_context_tree_sha256 != c0.build_context_tree_sha256
        or closure.file_sha256 != c0.candidate_image_closure_file_sha256
        or closure.bootstrap_closure_sha256 != c0.candidate_bootstrap_closure_sha256
        or closure.scientific_image_index_digest != c0.scientific_index_digest
        or closure.release_image_index_digest != c0.release_image_index_digest
    ):
        raise ProviderRunnerActivationError("activation A/P/T/D lineage differs")
    expected_plan_path = (
        Path(plan.host_tools.controlled_root)
        / "production"
        / "provider-plans"
        / plan.phase
        / "provider-plan.json"
    )
    if Path(plan.provider_plan_path) != expected_plan_path:
        raise ProviderRunnerActivationError("activation provider-plan path differs")
    runtime = _phase_runtime_binding(plan.phase, closure, c0)
    if (
        plan.runtime_image,
        plan.oci_index_digest,
        plan.oci_platform_manifest_digest,
        plan.runtime_probe_receipt_sha256,
    ) != runtime:
        raise ProviderRunnerActivationError("activation runtime P/T/D binding differs")
    expected_label = derive_phase_runner_label(plan.claim_nonce, plan.phase)
    bootstrap = plan.runner_bootstrap_receipt
    if (
        bootstrap.workflow_sha != c0.apparatus_commit
        or bootstrap.runner_label != expected_label
        or bootstrap.phase != plan.phase
        or bootstrap.runner_id != plan.runner_id
        or bootstrap.runner_name != plan.runner_name
        or bootstrap.runner_group_id != plan.runner_group_id
        or bootstrap.runner_version != plan.runner_version
        or bootstrap.runner_archive_sha256 != plan.runner_archive_sha256
        or bootstrap.file_sha256 != plan.runner_bootstrap_receipt_file_sha256
    ):
        raise ProviderRunnerActivationError("activation plan/claim/bootstrap binding differs")


def _plans_from_exact_manifest(
    path: Path,
    *,
    c1_commit: str,
) -> tuple[Mapping[ProviderPhase, ProviderPhasePlan], bytes]:
    encoded = _read(path, label="frozen C1 manifest")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise ProviderRunnerActivationError("frozen C1 manifest parent is unavailable") from exc
    if parent != path.parent:
        raise ProviderRunnerActivationError("frozen C1 manifest parent contains a symlink")
    try:
        with tempfile.TemporaryDirectory(prefix=f".{path.name}.activation-", dir=parent) as root:
            copy = Path(root) / path.name
            write_exclusive_receipt_bytes(encoded, copy)
            plans = load_provider_phase_plans(copy, c1_commit=c1_commit)
    except (ArtifactIntegrityError, ExecutionClaimError, OSError, ValueError) as exc:
        raise ProviderRunnerActivationError("frozen C1 provider plans are invalid") from exc
    return plans, encoded


def _admit_activation_sources(
    *,
    manifest_path: str | Path,
    c1_commit: str,
    phase: ProviderPhase,
    c0_instantiation_receipt_path: str | Path,
    candidate_image_closure_path: str | Path,
) -> _ActivationContext:
    manifest = _absolute_path("manifest_path", manifest_path)
    c0_path = _absolute_path("c0_instantiation_receipt_path", c0_instantiation_receipt_path)
    closure_path = _absolute_path("candidate_image_closure_path", candidate_image_closure_path)
    commit = _commit("c1_commit", c1_commit)
    if phase not in PHASES:
        raise ProviderRunnerActivationError("activation phase differs")
    plans, manifest_bytes = _plans_from_exact_manifest(manifest, c1_commit=commit)
    if set(plans) != set(PHASES):
        raise ProviderRunnerActivationError("frozen C1 plans do not contain three phases")
    plan = plans[phase]
    materialized = _load_fixed_provider_plan(Path(plan.provider_plan_path))
    if materialized != plan:
        raise ProviderRunnerActivationError("fixed provider plan differs from frozen C1")

    c0_bytes = _read(c0_path, label="C0 control instantiation receipt")
    c0 = _c0_instantiation_from_bytes(c0_bytes)
    closure_bytes = _read(closure_path, label="candidate image closure")
    closure = _candidate_closure_from_bytes(closure_bytes)
    _validate_lineage(plan, c0, closure)
    try:
        gh_sha256 = digest_regular_file(
            plan.host_tools.gh_executable,
            label="controlled GitHub CLI executable",
        )
    except ArtifactIntegrityError as exc:
        raise ProviderRunnerActivationError("cannot authenticate controlled GitHub CLI") from exc
    if gh_sha256 != plan.host_tools.gh_executable_sha256:
        raise ProviderRunnerActivationError("controlled GitHub CLI differs from provider plan")
    return _ActivationContext(
        manifest_path=manifest,
        manifest_file_sha256=_sha256(manifest_bytes),
        c0_instantiation_path=c0_path,
        c0_instantiation_file_sha256=_sha256(c0_bytes),
        c0_instantiation=c0,
        candidate_closure_path=closure_path,
        candidate_closure_file_sha256=_sha256(closure_bytes),
        candidate_closure=closure,
        plan=plan,
    )


def _select_stopped_runner(
    plan: ProviderPhasePlan,
    inventory: RepositoryRunnerInventoryReceipt,
) -> RepositoryRunnerSnapshot:
    matches = [
        row
        for row in inventory.runners
        if row.runner_id == plan.runner_id and row.runner_name == plan.runner_name
    ]
    if len(matches) != 1:
        raise ProviderRunnerActivationError("production runner is not a live inventory singleton")
    runner = matches[0]
    runner_label = derive_phase_runner_label(plan.claim_nonce, plan.phase)
    labels = tuple(
        sorted(required_execute_runner_labels(runner_label), key=lambda item: item.encode())
    )
    if (
        runner.operating_system != "macOS"
        or runner.status != "offline"
        or runner.busy is not False
        or runner.labels != labels
    ):
        raise ProviderRunnerActivationError(
            "production runner must be stopped with the exact claim label"
        )
    return runner


def _activation_receipt(
    context: _ActivationContext,
    inventory: RepositoryRunnerInventoryReceipt,
    raw_inventory: bytes,
    runner: RepositoryRunnerSnapshot,
) -> ProviderRunnerActivationReceipt:
    plan = context.plan
    root = Path(plan.runner_bootstrap_receipt_path).parent
    return ProviderRunnerActivationReceipt(
        phase=plan.phase,
        repository=plan.repository,
        approval_environment=plan.approval_environment,
        runner_identity=plan.runner_identity,
        apparatus_commit=context.c0_instantiation.apparatus_commit,
        c1_commit=plan.c1_commit,
        manifest_path=str(context.manifest_path),
        manifest_sha256=plan.manifest_sha256,
        manifest_file_sha256=context.manifest_file_sha256,
        c0_instantiation_receipt_path=str(context.c0_instantiation_path),
        c0_instantiation_receipt_file_sha256=(context.c0_instantiation_file_sha256),
        candidate_image_closure_path=str(context.candidate_closure_path),
        candidate_image_closure_file_sha256=context.candidate_closure_file_sha256,
        candidate_image_source_commit=context.candidate_closure.github_sha,
        build_context_tree_sha256=context.candidate_closure.build_context_tree_sha256,
        candidate_bootstrap_closure_sha256=(context.candidate_closure.bootstrap_closure_sha256),
        scientific_image_index_digest=(context.candidate_closure.scientific_image_index_digest),
        release_image_index_digest=context.candidate_closure.release_image_index_digest,
        provider_plan_path=plan.provider_plan_path,
        provider_plan_sha256=plan.plan_sha256,
        provider_plan_file_sha256=plan.file_sha256,
        claim_nonce=plan.claim_nonce,
        runner_label=derive_phase_runner_label(plan.claim_nonce, plan.phase),
        bootstrap_receipt_path=plan.runner_bootstrap_receipt_path,
        bootstrap_receipt_file_sha256=plan.runner_bootstrap_receipt.file_sha256,
        registration_inventory_file_sha256=(
            plan.runner_bootstrap_receipt.repository_runner_inventory_sha256
        ),
        activation_inventory_path=str(root / INVENTORY_RECEIPT_FILENAME),
        activation_inventory_file_sha256=inventory.file_sha256,
        activation_inventory_response_sha256=inventory.response_sha256,
        raw_inventory_path=str(root / RAW_INVENTORY_FILENAME),
        raw_inventory_sha256=_sha256(raw_inventory),
        runtime_image=plan.runtime_image,
        runtime_index_digest=plan.oci_index_digest,
        runtime_platform_manifest_digest=plan.oci_platform_manifest_digest,
        runtime_probe_receipt_sha256=plan.runtime_probe_receipt_sha256,
        runner_id=runner.runner_id,
        runner_name=runner.runner_name,
        runner_group_id=plan.runner_group_id,
        runner_version=plan.runner_version,
        runner_archive_sha256=plan.runner_archive_sha256,
        runner_operating_system=runner.operating_system,
        runner_status=runner.status,
        runner_busy=runner.busy,
        runner_labels=runner.labels,
        captured_at_utc=inventory.captured_at_utc,
    )


def load_provider_runner_activation_bundle(
    directory: str | Path,
) -> tuple[
    ProviderRunnerBootstrapReceipt,
    RepositoryRunnerInventoryReceipt,
    bytes,
    ProviderRunnerActivationReceipt,
]:
    """Load one closed four-member activation bundle through a bound root fd."""

    try:
        root, members = _read_closed_bundle(
            directory,
            frozenset(
                {
                    BOOTSTRAP_RECEIPT_FILENAME,
                    INVENTORY_RECEIPT_FILENAME,
                    RAW_INVENTORY_FILENAME,
                    ACTIVATION_RECEIPT_FILENAME,
                }
            ),
            label="provider runner activation bundle",
        )
    except ProviderPlanOperatorError as exc:
        raise ProviderRunnerActivationError(
            "provider runner activation bundle is not one closed private directory"
        ) from exc
    try:
        bootstrap = ProviderRunnerBootstrapReceipt.from_dict(
            _parse_object(
                members[BOOTSTRAP_RECEIPT_FILENAME],
                label="provider runner bootstrap receipt",
            )
        )
    except ExecutionClaimError as exc:
        raise ProviderRunnerActivationError("provider runner bootstrap receipt is invalid") from exc
    if bootstrap.canonical_file_bytes() != members[BOOTSTRAP_RECEIPT_FILENAME]:
        raise ProviderRunnerActivationError("provider runner bootstrap typed bytes differ")
    inventory = _inventory_from_bytes(members[INVENTORY_RECEIPT_FILENAME])
    raw = members[RAW_INVENTORY_FILENAME]
    receipt = ProviderRunnerActivationReceipt.from_dict(
        _parse_object(
            members[ACTIVATION_RECEIPT_FILENAME],
            label="provider runner activation receipt",
        )
    )
    if receipt.canonical_file_bytes() != members[ACTIVATION_RECEIPT_FILENAME]:
        raise ProviderRunnerActivationError("provider runner activation typed bytes differ")
    if (
        receipt.bootstrap_receipt_path != str(root / BOOTSTRAP_RECEIPT_FILENAME)
        or receipt.activation_inventory_path != str(root / INVENTORY_RECEIPT_FILENAME)
        or receipt.raw_inventory_path != str(root / RAW_INVENTORY_FILENAME)
        or bootstrap.file_sha256 != receipt.bootstrap_receipt_file_sha256
        or bootstrap.repository_runner_inventory_sha256
        != receipt.registration_inventory_file_sha256
        or inventory.file_sha256 != receipt.activation_inventory_file_sha256
        or inventory.response_sha256 != receipt.activation_inventory_response_sha256
        or _sha256(raw) != receipt.raw_inventory_sha256
        or inventory.captured_at_utc != receipt.captured_at_utc
    ):
        raise ProviderRunnerActivationError("provider runner activation bundle hashes differ")
    matches = [
        row
        for row in inventory.runners
        if row.runner_id == receipt.runner_id and row.runner_name == receipt.runner_name
    ]
    if len(matches) != 1:
        raise ProviderRunnerActivationError("activation inventory lacks its production runner")
    row = matches[0]
    if (
        row.operating_system != receipt.runner_operating_system
        or row.status != receipt.runner_status
        or row.busy != receipt.runner_busy
        or row.labels != receipt.runner_labels
        or bootstrap.phase != receipt.phase
        or bootstrap.repository != receipt.repository
        or bootstrap.approval_environment != receipt.approval_environment
        or bootstrap.runner_identity != receipt.runner_identity
        or bootstrap.workflow_sha != receipt.apparatus_commit
        or bootstrap.runner_label != receipt.runner_label
        or bootstrap.runner_id != receipt.runner_id
        or bootstrap.runner_name != receipt.runner_name
        or bootstrap.runner_group_id != receipt.runner_group_id
        or bootstrap.runner_version != receipt.runner_version
        or bootstrap.runner_archive_sha256 != receipt.runner_archive_sha256
    ):
        raise ProviderRunnerActivationError("activation inventory/bootstrap identity differs")
    return bootstrap, inventory, raw, receipt


def _verify_bundle_against_context(
    context: _ActivationContext,
) -> ProviderRunnerActivationReceipt:
    plan = context.plan
    root = Path(plan.runner_bootstrap_receipt_path).parent
    bootstrap, _inventory, _raw, receipt = load_provider_runner_activation_bundle(root)
    expected = {
        "phase": plan.phase,
        "repository": plan.repository,
        "approval_environment": plan.approval_environment,
        "runner_identity": plan.runner_identity,
        "apparatus_commit": context.c0_instantiation.apparatus_commit,
        "c1_commit": plan.c1_commit,
        "manifest_path": str(context.manifest_path),
        "manifest_sha256": plan.manifest_sha256,
        "manifest_file_sha256": context.manifest_file_sha256,
        "c0_instantiation_receipt_path": str(context.c0_instantiation_path),
        "c0_instantiation_receipt_file_sha256": context.c0_instantiation_file_sha256,
        "candidate_image_closure_path": str(context.candidate_closure_path),
        "candidate_image_closure_file_sha256": context.candidate_closure_file_sha256,
        "candidate_image_source_commit": context.candidate_closure.github_sha,
        "build_context_tree_sha256": context.candidate_closure.build_context_tree_sha256,
        "candidate_bootstrap_closure_sha256": (context.candidate_closure.bootstrap_closure_sha256),
        "scientific_image_index_digest": (context.candidate_closure.scientific_image_index_digest),
        "release_image_index_digest": context.candidate_closure.release_image_index_digest,
        "provider_plan_path": plan.provider_plan_path,
        "provider_plan_sha256": plan.plan_sha256,
        "provider_plan_file_sha256": plan.file_sha256,
        "claim_nonce": plan.claim_nonce,
        "runner_label": derive_phase_runner_label(plan.claim_nonce, plan.phase),
        "bootstrap_receipt_path": plan.runner_bootstrap_receipt_path,
        "bootstrap_receipt_file_sha256": plan.runner_bootstrap_receipt.file_sha256,
        "registration_inventory_file_sha256": (
            plan.runner_bootstrap_receipt.repository_runner_inventory_sha256
        ),
        "runtime_image": plan.runtime_image,
        "runtime_index_digest": plan.oci_index_digest,
        "runtime_platform_manifest_digest": plan.oci_platform_manifest_digest,
        "runtime_probe_receipt_sha256": plan.runtime_probe_receipt_sha256,
        "runner_id": plan.runner_id,
        "runner_name": plan.runner_name,
        "runner_group_id": plan.runner_group_id,
        "runner_version": plan.runner_version,
        "runner_archive_sha256": plan.runner_archive_sha256,
    }
    if bootstrap != plan.runner_bootstrap_receipt or any(
        getattr(receipt, name) != value for name, value in expected.items()
    ):
        raise ProviderRunnerActivationError("activation bundle differs from frozen C1 context")
    return receipt


def verify_provider_runner_activation(
    *,
    manifest_path: str | Path,
    c1_commit: str,
    phase: ProviderPhase,
    c0_instantiation_receipt_path: str | Path,
    candidate_image_closure_path: str | Path,
) -> ProviderRunnerActivationReceipt:
    """Revalidate retained activation evidence against current immutable sources."""

    context = _admit_activation_sources(
        manifest_path=manifest_path,
        c1_commit=c1_commit,
        phase=phase,
        c0_instantiation_receipt_path=c0_instantiation_receipt_path,
        candidate_image_closure_path=candidate_image_closure_path,
    )
    return _verify_bundle_against_context(context)


def write_provider_runner_activation(
    *,
    manifest_path: str | Path,
    c1_commit: str,
    phase: ProviderPhase,
    c0_instantiation_receipt_path: str | Path,
    candidate_image_closure_path: str | Path,
    api: GitHubBytesApi | None,
    captured_at_utc: str,
) -> ProviderRunnerActivationReceipt:
    """Publish the sole A-bound production bootstrap receipt before listener start."""

    captured = _timestamp("captured_at_utc", captured_at_utc)
    context = _admit_activation_sources(
        manifest_path=manifest_path,
        c1_commit=c1_commit,
        phase=phase,
        c0_instantiation_receipt_path=c0_instantiation_receipt_path,
        candidate_image_closure_path=candidate_image_closure_path,
    )
    plan = context.plan
    live_api = api or GitHubCliBytesApi(plan.host_tools.gh_executable, os.environ)
    output = Path(plan.runner_bootstrap_receipt_path).parent
    if os.path.lexists(output):
        raise ProviderRunnerActivationError("production runner activation output already exists")
    try:
        inventory, raw = capture_repository_runner_inventory(
            api=live_api,
            captured_at_utc=captured,
        )
    except ProviderRehearsalError as exc:
        raise ProviderRunnerActivationError("cannot capture production runner inventory") from exc
    runner = _select_stopped_runner(plan, inventory)
    receipt = _activation_receipt(context, inventory, raw, runner)

    def revalidate_before_publish() -> None:
        current = _admit_activation_sources(
            manifest_path=manifest_path,
            c1_commit=c1_commit,
            phase=phase,
            c0_instantiation_receipt_path=c0_instantiation_receipt_path,
            candidate_image_closure_path=candidate_image_closure_path,
        )
        if current != context:
            raise ProviderRunnerActivationError("activation source set changed before publish")
        try:
            current_inventory, current_raw = capture_repository_runner_inventory(
                api=live_api,
                captured_at_utc=captured,
            )
        except ProviderRehearsalError as exc:
            raise ProviderRunnerActivationError(
                "cannot revalidate production runner inventory"
            ) from exc
        _select_stopped_runner(plan, current_inventory)
        if current_inventory != inventory or current_raw != raw:
            raise ProviderRunnerActivationError(
                "production runner inventory changed before publication"
            )

    try:
        _publish_private_bundle(
            output,
            {
                BOOTSTRAP_RECEIPT_FILENAME: plan.runner_bootstrap_receipt.canonical_file_bytes(),
                INVENTORY_RECEIPT_FILENAME: _canonical_bytes(inventory.to_dict()) + b"\n",
                RAW_INVENTORY_FILENAME: raw,
                ACTIVATION_RECEIPT_FILENAME: receipt.canonical_file_bytes(),
            },
            label="provider runner activation bundle",
            pre_publish=revalidate_before_publish,
        )
    except ProviderPlanOperatorError as exc:
        raise ProviderRunnerActivationError("cannot publish runner activation bundle") from exc
    readback = _verify_bundle_against_context(context)
    if readback != receipt:
        raise ProviderRunnerActivationError("runner activation readback differs")
    return receipt


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-provider-runner-activation",
        description=(
            "Publish or verify P-bound runner registration and A-bound activation bundles."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "verify-registration"):
        command = commands.add_parser(name)
        command.add_argument("--blueprint-directory", type=Path, required=True)
        command.add_argument("--phase", choices=PHASES, required=True)
    for name in ("write", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--c1-commit", required=True)
        command.add_argument("--phase", choices=PHASES, required=True)
        command.add_argument("--c0-instantiation-receipt", type=Path, required=True)
        command.add_argument("--candidate-image-closure", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "register":
            receipt = write_provider_runner_registration(
                blueprint_directory=arguments.blueprint_directory,
                phase=arguments.phase,
                api=None,
                captured_at_utc=_utc_now(),
            )
        elif arguments.command == "verify-registration":
            receipt = verify_provider_runner_registration(
                blueprint_directory=arguments.blueprint_directory,
                phase=arguments.phase,
            )
        elif arguments.command == "write":
            receipt = write_provider_runner_activation(
                manifest_path=arguments.manifest,
                c1_commit=arguments.c1_commit,
                phase=arguments.phase,
                c0_instantiation_receipt_path=arguments.c0_instantiation_receipt,
                candidate_image_closure_path=arguments.candidate_image_closure,
                api=None,
                captured_at_utc=_utc_now(),
            )
        else:
            receipt = verify_provider_runner_activation(
                manifest_path=arguments.manifest,
                c1_commit=arguments.c1_commit,
                phase=arguments.phase,
                c0_instantiation_receipt_path=arguments.c0_instantiation_receipt,
                candidate_image_closure_path=arguments.candidate_image_closure,
            )
    except ProviderRunnerActivationError as exc:
        print(f"provider runner activation failed: {exc}", file=sys.stderr)
        return 2
    if isinstance(receipt, ProviderRunnerRegistrationReceipt):
        payload = {
            "phase": receipt.phase,
            "registration_evidence_file_sha256": receipt.file_sha256,
            "registration_evidence_path": str(
                Path(receipt.registration_receipt_path).parent / REGISTRATION_EVIDENCE_FILENAME
            ),
            "registration_receipt_file_sha256": receipt.registration_receipt_file_sha256,
            "registration_receipt_path": receipt.registration_receipt_path,
            "runner_id": receipt.runner_id,
            "runner_label": receipt.runner_label,
            "runner_name": receipt.runner_name,
        }
    else:
        payload = {
            "activation_receipt_file_sha256": receipt.file_sha256,
            "activation_receipt_path": str(
                Path(receipt.bootstrap_receipt_path).parent / ACTIVATION_RECEIPT_FILENAME
            ),
            "bootstrap_receipt_file_sha256": receipt.bootstrap_receipt_file_sha256,
            "bootstrap_receipt_path": receipt.bootstrap_receipt_path,
            "phase": receipt.phase,
            "runner_id": receipt.runner_id,
            "runner_label": receipt.runner_label,
            "runner_name": receipt.runner_name,
        }
    print(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
