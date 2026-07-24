"""Provider-claimed execution capabilities for the confirmatory apparatus.

The records in this module are data, not authority.  A launch capability is
minted only after a provider-backed suite-state CAS and an exact future drand
beacon have both been verified.  Its revalidator must re-read the provider
ledger tip before every host launch.  Local exclusive files remain a secondary
crash/replay control and never select the admissible provider run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from .provider_contract import (
    DOCKER_SERVER_PROBE_FIELDS as SHARED_DOCKER_SERVER_PROBE_FIELDS,
)
from .provider_contract import (
    DOCKER_SERVER_PROBE_SCHEMA as SHARED_DOCKER_SERVER_PROBE_SCHEMA,
)
from .provider_contract import (
    PHASE_HOST_PROBE_FIELDS as SHARED_PHASE_HOST_PROBE_FIELDS,
)
from .provider_contract import (
    PHASE_HOST_PROBE_SCHEMA as SHARED_PHASE_HOST_PROBE_SCHEMA,
)
from .provider_contract import (
    PHASE_HOST_TOOL_CONTRACT_FIELDS as SHARED_PHASE_HOST_TOOL_CONTRACT_FIELDS,
)
from .provider_contract import (
    PHASE_HOST_TOOL_CONTRACT_SCHEMA as SHARED_PHASE_HOST_TOOL_CONTRACT_SCHEMA,
)
from .study import (
    FIXED_CORPORA,
    PROVIDER_APPROVAL_ENVIRONMENT,
    PROVIDER_PHASE_COMMAND_IDS,
    PROVIDER_PHASE_PLAN_TEMPLATE_FIELDS,
    PROVIDER_PHASE_PLAN_TEMPLATE_SCHEMA,
    PROVIDER_PHASE_RUNTIME_CEILINGS,
    PROVIDER_PHASE_WORKFLOWS,
    PROVIDER_PLAN_C1_COMMIT_BINDING,
    PROVIDER_PLAN_CLAIM_RECEIPT_BINDING,
    PROVIDER_PLAN_MANIFEST_BINDING,
    PROVIDER_PLAN_PHASE_INPUT_BINDING,
    PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
    PROVIDER_PLAN_PREDECESSOR_BINDING,
    PROVIDER_PLAN_SUITE_BINDING,
    PROVIDER_RUNNER_IDENTITY,
    load_study_manifest,
    manifest_sha256,
    resolve_candidate_provider_plan_commit_bindings,
    validate_candidate_rehearsal_manifest,
    validate_study_manifest,
)

EXECUTION_CLAIM_CONTRACT_SCHEMA = "fractal-execution-claim-contract-v1"
PROVIDER_EXECUTION_IDENTITY_SCHEMA = "fractal-provider-execution-identity-v1"
EXECUTION_BEACON_CONTRACT_SCHEMA = "fractal-execution-beacon-contract-v1"
EXECUTION_BEACON_RECEIPT_SCHEMA = "fractal-execution-beacon-receipt-v1"
RUN_OUTPUT_AGGREGATE_SCHEMA = "fractal-run-output-aggregate-v1"
C1_REGISTRATION_PACKAGE_FILE_COUNT = 27
ZENODO_ADMISSION_SCHEMA = "fractal-zenodo-anonymous-admission-v2"
PHASE_FAILURE_SCHEMA = "fractal-provider-phase-failure-v1"
RUNTIME_CLAIM_RECEIPT_SCHEMA = "fractal-runtime-claim-capability-v1"
PHASE_HOST_TOOL_CONTRACT_SCHEMA = "fractal-phase-host-tool-contract-v1"
PHASE_HOST_TOOL_RECEIPT_SCHEMA = "fractal-phase-host-tool-receipt-v1"
PHASE_HOST_PROBE_SCHEMA = "fractal-phase-host-probe-v1"
DOCKER_SERVER_PROBE_SCHEMA = "fractal-docker-server-probe-v1"
PHASE_HOST_PROBE_FILENAME = "phase-host-probe.json"
DOCKER_SERVER_PROBE_FILENAME = "docker-server-probe.json"
PHASE_CLAIM_CONTRACT_SCHEMA = "fractal-provider-phase-claim-contract-v1"
PHASE_BEACON_RECEIPT_SCHEMA = "fractal-provider-phase-beacon-receipt-v1"
PHASE_RUNTIME_CLAIM_RECEIPT_SCHEMA = "fractal-provider-phase-runtime-claim-v1"
PROVIDER_PHASE_PLAN_SCHEMA = "fractal-provider-phase-plan-v2"
PROVIDER_RUNNER_BOOTSTRAP_SCHEMA = "fractal-provider-runner-bootstrap-v2"
LIVE_EXECUTE_JOB_RECEIPT_SCHEMA = "fractal-live-execute-job-identity-v1"
FAILED_EXECUTE_JOB_RECEIPT_SCHEMA = "fractal-failed-execute-job-identity-v1"
PROVIDER_RUNNER_READINESS_SCHEMA = "fractal-provider-runner-readiness-v1"
HOST_TOOL_TREE_DERIVATION = "sha256-fractal-host-tool-tree-v1"

OFFICIAL_GH_VERSION = "2.96.0"
OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI = (
    "https://github.com/cli/cli/releases/download/v2.96.0/gh_2.96.0_macOS_arm64.zip"
)
OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256 = (
    "f23a0c37d963aacc3bed703ccbd59b41c5ca22101fab7f00eb2b7cad23aba463"
)
OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT = 13_950_131
OFFICIAL_GH_OSX_ARM64_BINARY_SHA256 = (
    "b1d6c442fde99ca27c04e1e74d624895abe37785f4a3e9e9b684bf7586ce4bc8"
)
OFFICIAL_ACTIONS_RUNNER_VERSION = "2.335.1"
OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI = (
    "https://github.com/actions/runner/releases/download/v2.335.1/"
    "actions-runner-osx-arm64-2.335.1.tar.gz"
)
OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256 = (
    "e1a9bc7a3661e06fa0b129d15c2064fe65dc81a431001d8958a9db1409b73769"
)
OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT = 127_138_003
OFFICIAL_ACTIONS_RUNNER_LISTENER_SHA256 = (
    "57a04bccf7e22e6e9e0cf92c691a5a8b87c8cfa86535548f131f422d53a0a4df"
)
OFFICIAL_ACTIONS_RUNNER_LISTENER_DLL_SHA256 = (
    "a969651efdf3b35e905968f6434dad4adcd5fd07d3f20e43595840f075cd1b15"
)
OFFICIAL_ACTIONS_RUNNER_CONFIG_SHA256 = (
    "4ad01727c3f29a0b6473d625412af6bdefc6c077763a6410f359c764fc0b3ae8"
)
OFFICIAL_ACTIONS_RUNNER_RUN_SHA256 = (
    "b39d7e0ca921a3189f7fe4e0a2f686b46719d4ccc2647f156f14407ec4517e8f"
)
REGISTERED_DOCKER_CLIENT_VERSION = "28.3.2"
REGISTERED_DOCKER_CLIENT_BUILD = "578ccf6"
REGISTERED_DOCKER_CLIENT_SHA256 = "9614e706a1bd7a56eaf739e7cd8da760df5ea536f062f1ffef306920d199f63f"
SOURCE_BUILT_LINUX_ARM64_TLE_SHA256 = (
    "ca9d498b6a3c1ea8edff9ace7bf00eb0f90ce67166343161f9a53f21900a6ef5"
)
SOURCE_BUILT_LINUX_ARM64_TLE_BYTE_COUNT = 13_303_934
SOURCE_BUILT_LINUX_ARM64_TLE_SOURCE_COMMIT = "7b54141a9733fd6fa207587a11148280e6fb020d"
OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION = "3.12.13"
OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260510/"
    "cpython-3.12.13%2B20260510-aarch64-apple-darwin-install_only.tar.gz"
)
OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256 = (
    "5a30271f8d345a5b02b0c9e4e31e0f1e1455a8e4a04fba95cd9762472abc3b17"
)
OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT = 25_102_827
OFFICIAL_PYTHON_BUILD_STANDALONE_BINARY_SHA256 = (
    "14b79bc842a2c806fc8dc6ab16b3b13fabb6b3043a6868ccee1b1170a19388b3"
)
MAXIMUM_REGISTERED_ONLINE_RUNTIME_SECONDS = 20 * 60 * 60

EXECUTION_SEED_DERIVATION = "sha256-fractal-execution-seed-v1-u64be"
OUTPUT_AGGREGATE_DERIVATION = "sha256-five-canonical-output-trees-v1"
RUNNER_LABEL_DERIVATION = "sha256-fractal-phase-runner-label-v1"
ONLINE_PHASE: Literal["online"] = "online"
LABEL_RELEASE_PHASE: Literal["label-release"] = "label-release"
ANALYSIS_PHASE: Literal["analysis"] = "analysis"
ProviderPhase = Literal["online", "label-release", "analysis"]

PHASE_JOB_NAMES: Mapping[ProviderPhase, tuple[str, str]] = {
    ONLINE_PHASE: ("claim-online", "execute-online"),
    LABEL_RELEASE_PHASE: ("claim-label-release", "release-labels"),
    ANALYSIS_PHASE: ("claim-analysis", "run-analysis"),
}
BASE_EXECUTE_RUNNER_LABELS = ("self-hosted", "macOS", "ARM64")
PHASE_RUNTIME_BINDINGS: Mapping[ProviderPhase, tuple[str, str, str]] = {
    ONLINE_PHASE: ("linux/arm64", "scientific", "main"),
    LABEL_RELEASE_PHASE: ("linux/arm64", "timelock-release", "release"),
    ANALYSIS_PHASE: ("linux/amd64", "scientific", "main"),
}
PHASE_STATE_TRANSITIONS: Mapping[ProviderPhase, tuple[str, str, str]] = {
    ONLINE_PHASE: ("RUN_CLAIMED", "ONLINE_COMPLETE", "FAILED"),
    LABEL_RELEASE_PHASE: ("LABEL_RELEASE_CLAIMED", "LABELS_RELEASED", "FAILED"),
    ANALYSIS_PHASE: ("ANALYSIS_CLAIMED", "ANALYSIS_COMPLETE", "FAILED"),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_WORKFLOW = re.compile(r"^\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml$")
_RUNNER_LABEL = re.compile(r"^fractal-ann-confirmatory-[a-z0-9][a-z0-9-]{15,95}$")
_HEX = re.compile(r"^[0-9a-f]+$")
_CAPABILITY = object()
_MAX_CAPABILITY_AGE_NS = 5 * 60 * 1_000_000_000

PREREQUISITE_OUTPUT_KEYS = frozenset(
    {
        "docker_client_build",
        "docker_client_version",
        "docker_file_sha256",
        "docker_path",
        "docker_resolved_path",
        "gh_file_sha256",
        "gh_path",
        "gh_version",
        "host_python_file_sha256",
        "host_python_path",
        "oci_index_digest",
        "oci_platform_manifest_digest",
        "phase_evidence_root",
        "prerequisite_receipt_path",
        "prerequisite_receipt_sha256",
        "provider_plan_file_sha256",
        "provider_plan_materialization_path",
        "provider_plan_path",
        "runner_bootstrap_receipt_file_sha256",
        "runner_bootstrap_receipt_path",
        "runner_listener_file_sha256",
        "runner_listener_path",
        "runtime_image",
        "runtime_image_role",
        "runtime_index_role",
        "runtime_platform",
        "tle_binary_sha256",
    }
)
ACTIVATION_COMMON_OUTPUT_KEYS = frozenset(
    {
        "execute_job_id",
        "fixed_corpora_completed",
        "live_execute_job_receipt_path",
        "live_execute_job_receipt_sha256",
        "phase_execution_receipt_path",
        "phase_execution_receipt_sha256",
        "runtime_claim_receipt_path",
        "runtime_claim_receipt_sha256",
    }
)
ACTIVATION_PHASE_OUTPUT_KEYS: Mapping[ProviderPhase, frozenset[str]] = {
    ONLINE_PHASE: frozenset(
        {
            "five_corpora_executed",
            "launch_receipt_inventory_path",
            "launch_receipt_inventory_sha256",
        }
    ),
    LABEL_RELEASE_PHASE: frozenset(
        {
            "five_label_payloads_decrypted",
            "label_release_inventory_path",
            "label_release_inventory_sha256",
        }
    ),
    ANALYSIS_PHASE: frozenset(
        {
            "analysis_result_path",
            "analysis_result_sha256",
            "five_corpora_analyzed",
        }
    ),
}
CLAIM_OUTPUT_KEYS = frozenset(
    {
        "claim_ledger_commit",
        "claim_predicate_path",
        "claim_predicate_sha256",
        "claim_receipt_path",
        "claim_receipt_sha256",
        "claim_state_sha256",
        "claim_subject_path",
        "claim_subject_sha256",
        "expected_execute_job_name",
        "provider_identity_sha256",
        "runner_label",
        "suite_namespace",
    }
)
PREPARE_COMMON_OUTPUT_KEYS = frozenset(
    {
        "preparation_receipt_path",
        "preparation_receipt_sha256",
        "prepared_subject_path",
        "prepared_subject_sha256",
    }
)
PUBLISH_OUTPUT_KEYS = frozenset(
    {
        "ledger_commit",
        "publication_receipt_path",
        "publication_receipt_sha256",
        "state_record_sha256",
    }
)


class ExecutionClaimError(ValueError):
    """Raised when a provider phase cannot claim or consume authority."""


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
        raise ExecutionClaimError("execution-claim evidence must be canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ExecutionClaimError(f"{name} must be a canonical non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise ExecutionClaimError(f"{name} must use NFC normalization")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ExecutionClaimError(f"{name} cannot contain control characters")
    return value


def _digest(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ExecutionClaimError(f"{name} must be a lowercase SHA-256")
    return value


def _git_commit(name: str, value: object) -> str:
    if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
        raise ExecutionClaimError(f"{name} must be one full lowercase Git commit")
    return value


def _positive(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ExecutionClaimError(f"{name} must be a positive integer")
    return value


def _nonnegative(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ExecutionClaimError(f"{name} must be a non-negative integer")
    return value


def _runner_group(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ExecutionClaimError("runner_group_id must be null or a non-negative integer")
    return value


def _api_head_branch(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _text(name, value)


def _timestamp(name: str, value: object) -> datetime:
    text = _text(name, value)
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ExecutionClaimError(f"{name} must use canonical ISO 8601") from exc
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise ExecutionClaimError(f"{name} must use UTC")
    if instant.isoformat() != text:
        raise ExecutionClaimError(f"{name} must use canonical ISO 8601")
    return instant


def _https(name: str, value: object) -> str:
    text = _text(name, value)
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ExecutionClaimError(f"{name} must be a credential-free HTTPS URI")
    return text


def _absolute_path(name: str, value: object) -> str:
    text = _text(name, value)
    path = Path(text)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExecutionClaimError(f"{name} must be one canonical absolute path")
    if str(path) != text:
        raise ExecutionClaimError(f"{name} must use its canonical spelling")
    return text


def _path_below(name: str, value: object, root: str) -> str:
    text = _absolute_path(name, value)
    try:
        Path(text).relative_to(Path(root))
    except ValueError as exc:
        raise ExecutionClaimError(f"{name} must be below controlled_root") from exc
    if text == root:
        raise ExecutionClaimError(f"{name} cannot equal controlled_root")
    return text


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ExecutionClaimError(f"{label} must be an object with string keys")
    observed = set(value)
    if observed != fields:
        raise ExecutionClaimError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _strict_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ExecutionClaimError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ExecutionClaimError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionClaimError(f"cannot decode {label}") from exc
    if not isinstance(value, Mapping):
        raise ExecutionClaimError(f"{label} must contain one object")
    return value


def _fixed_rows(name: str, rows: Sequence[Any], row_type: type) -> tuple[Any, ...]:
    values = tuple(rows)
    expected = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8")))
    if (
        len(values) != len(expected)
        or not all(isinstance(row, row_type) for row in values)
        or tuple(row.corpus_id for row in values) != expected
    ):
        raise ExecutionClaimError(f"{name} must bind every fixed corpus once in byte order")
    return values


def derive_phase_runner_label(claim_nonce: str, phase: ProviderPhase) -> str:
    nonce = _digest("claim_nonce", claim_nonce)
    if phase not in {ONLINE_PHASE, LABEL_RELEASE_PHASE, ANALYSIS_PHASE}:
        raise ExecutionClaimError("phase runner label has an unknown phase")
    suffix = _sha256(
        RUNNER_LABEL_DERIVATION.encode("ascii")
        + b"\0"
        + phase.encode("ascii")
        + b"\0"
        + bytes.fromhex(nonce)
    )[:24]
    return f"fractal-ann-confirmatory-{phase}-{suffix}"


def required_execute_runner_labels(runner_label: str) -> tuple[str, ...]:
    label = _text("runner_label", runner_label)
    if _RUNNER_LABEL.fullmatch(label) is None:
        raise ExecutionClaimError("execute runner label is invalid")
    return (*BASE_EXECUTE_RUNNER_LABELS, label)


@dataclass(frozen=True)
class ClaimCorpusBinding:
    """C1 identity of one corpus plan and its sole output namespace."""

    corpus_id: str
    staging_namespace_uri: str
    canonical_namespace_uri: str
    runtime_plan_sha256: str
    runtime_plan_file_sha256: str

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ExecutionClaimError("claim corpus is outside the fixed suite")
        for name in ("staging_namespace_uri", "canonical_namespace_uri"):
            parsed = urlsplit(_text(name, getattr(self, name)))
            if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
                raise ExecutionClaimError(f"{name} must be a canonical file URI")
        if self.staging_namespace_uri == self.canonical_namespace_uri:
            raise ExecutionClaimError("claim staging and canonical namespaces must differ")
        for name in ("runtime_plan_sha256", "runtime_plan_file_sha256"):
            _digest(name, getattr(self, name))

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> ClaimCorpusBinding:
        return cls(**_closed(value, frozenset(cls.__dataclass_fields__), label="claim corpus"))


@dataclass(frozen=True)
class PhaseHostProbe:
    operating_system: str
    operating_system_version: str
    kernel_release: str
    architecture: str
    logical_cpu_count: int
    physical_memory_bytes: int
    schema_version: str = PHASE_HOST_PROBE_SCHEMA

    def __post_init__(self) -> None:
        if self.operating_system != "macOS" or self.architecture != "ARM64":
            raise ExecutionClaimError("host probe must describe macOS ARM64")
        for name in ("operating_system_version", "kernel_release"):
            _text(name, getattr(self, name))
        _positive("logical_cpu_count", self.logical_cpu_count)
        _positive("physical_memory_bytes", self.physical_memory_bytes)
        if self.schema_version != PHASE_HOST_PROBE_SCHEMA:
            raise ExecutionClaimError("phase host probe schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> PhaseHostProbe:
        return cls(**_closed(value, frozenset(cls.__dataclass_fields__), label="phase host probe"))


@dataclass(frozen=True)
class DockerServerProbe:
    engine_version: str
    engine_build: str
    kernel_version: str
    operating_system: str
    architecture: str
    cpu_count: int
    memory_bytes: int
    schema_version: str = DOCKER_SERVER_PROBE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "engine_version",
            "engine_build",
            "kernel_version",
            "operating_system",
        ):
            _text(name, getattr(self, name))
        if self.operating_system != "linux" or self.architecture != "arm64":
            raise ExecutionClaimError("Docker server probe must describe linux/arm64")
        _positive("cpu_count", self.cpu_count)
        _positive("memory_bytes", self.memory_bytes)
        if self.schema_version != DOCKER_SERVER_PROBE_SCHEMA:
            raise ExecutionClaimError("Docker server probe schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> DockerServerProbe:
        return cls(
            **_closed(value, frozenset(cls.__dataclass_fields__), label="Docker server probe")
        )


@dataclass(frozen=True)
class PhaseHostToolContract:
    """C1-fixed host executables and probes used by every provider phase."""

    controlled_root: str
    python_archive_uri: str
    python_archive_sha256: str
    python_archive_byte_count: int
    python_executable: str
    python_version: str
    python_executable_sha256: str
    venv_root: str
    venv_tree_sha256: str
    venv_symlink_inventory_sha256: str
    gh_archive_uri: str
    gh_archive_sha256: str
    gh_archive_byte_count: int
    gh_executable: str
    gh_executable_sha256: str
    gh_version: str
    runner_archive_uri: str
    runner_archive_sha256: str
    runner_archive_byte_count: int
    runner_listener_executable: str
    runner_listener_sha256: str
    runner_listener_dll: str
    runner_listener_dll_sha256: str
    runner_config_executable: str
    runner_config_sha256: str
    runner_run_executable: str
    runner_run_sha256: str
    runner_version: str
    runner_ephemeral: bool
    runner_disable_update: bool
    runner_unattended: bool
    docker_executable: str
    docker_resolved_executable: str
    docker_executable_sha256: str
    docker_client_version: str
    docker_client_build: str
    host_probe: PhaseHostProbe
    docker_server_probe: DockerServerProbe
    host_probe_receipt_sha256: str
    docker_server_probe_receipt_sha256: str
    host_operating_system: str
    host_architecture: str
    schema_version: str = PHASE_HOST_TOOL_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        root = _absolute_path("controlled_root", self.controlled_root)
        if root == "/":
            raise ExecutionClaimError("controlled_root cannot be the filesystem root")
        for name in (
            "python_executable",
            "venv_root",
            "gh_executable",
            "runner_listener_executable",
            "runner_listener_dll",
            "runner_config_executable",
            "runner_run_executable",
        ):
            _path_below(name, getattr(self, name), root)
        _absolute_path("docker_executable", self.docker_executable)
        _absolute_path("docker_resolved_executable", self.docker_resolved_executable)
        if self.docker_executable == self.docker_resolved_executable:
            raise ExecutionClaimError("docker executable must bind its resolved symlink target")
        for name in (
            "python_executable_sha256",
            "python_archive_sha256",
            "venv_tree_sha256",
            "venv_symlink_inventory_sha256",
            "gh_archive_sha256",
            "gh_executable_sha256",
            "runner_archive_sha256",
            "runner_listener_sha256",
            "runner_listener_dll_sha256",
            "runner_config_sha256",
            "runner_run_sha256",
            "docker_executable_sha256",
            "host_probe_receipt_sha256",
            "docker_server_probe_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in (
            "python_version",
            "docker_client_version",
            "docker_client_build",
            "host_operating_system",
            "host_architecture",
        ):
            _text(name, getattr(self, name))
        if self.python_archive_uri != OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI:
            raise ExecutionClaimError("Python archive URI is not the registered official release")
        if self.python_archive_sha256 != OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256:
            raise ExecutionClaimError("Python archive SHA-256 differs from the official release")
        if self.python_archive_byte_count != OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT:
            raise ExecutionClaimError("Python archive byte count differs from the official release")
        if self.python_version != OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION:
            raise ExecutionClaimError("Python version differs from the registered release")
        if self.python_executable_sha256 != OFFICIAL_PYTHON_BUILD_STANDALONE_BINARY_SHA256:
            raise ExecutionClaimError("Python executable SHA-256 differs from the official archive")
        if self.gh_archive_uri != OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI:
            raise ExecutionClaimError("gh archive URI is not the registered official release")
        if self.gh_archive_sha256 != OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256:
            raise ExecutionClaimError("gh archive SHA-256 differs from the official release")
        if self.gh_archive_byte_count != OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT:
            raise ExecutionClaimError("gh archive byte count differs from the official release")
        if self.gh_executable_sha256 != OFFICIAL_GH_OSX_ARM64_BINARY_SHA256:
            raise ExecutionClaimError("gh executable SHA-256 differs from the official release")
        if self.gh_version != OFFICIAL_GH_VERSION:
            raise ExecutionClaimError("gh version differs from the registered release")
        if self.runner_archive_uri != OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI:
            raise ExecutionClaimError("runner archive URI is not the registered official release")
        if self.runner_archive_sha256 != OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256:
            raise ExecutionClaimError("runner archive SHA-256 differs from the official release")
        if self.runner_archive_byte_count != OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT:
            raise ExecutionClaimError("runner archive byte count differs from the official release")
        if self.runner_version != OFFICIAL_ACTIONS_RUNNER_VERSION:
            raise ExecutionClaimError("runner version differs from the registered release")
        if (
            self.runner_listener_sha256 != OFFICIAL_ACTIONS_RUNNER_LISTENER_SHA256
            or self.runner_listener_dll_sha256 != OFFICIAL_ACTIONS_RUNNER_LISTENER_DLL_SHA256
            or self.runner_config_sha256 != OFFICIAL_ACTIONS_RUNNER_CONFIG_SHA256
            or self.runner_run_sha256 != OFFICIAL_ACTIONS_RUNNER_RUN_SHA256
        ):
            raise ExecutionClaimError("runner member digest differs from the official archive")
        if (
            self.runner_ephemeral is not True
            or self.runner_disable_update is not True
            or self.runner_unattended is not True
        ):
            raise ExecutionClaimError("runner must be ephemeral, update-disabled, and unattended")
        if self.host_operating_system != "macOS" or self.host_architecture != "ARM64":
            raise ExecutionClaimError("phase host must be the registered macOS ARM64 runner")
        if (
            self.docker_client_version != REGISTERED_DOCKER_CLIENT_VERSION
            or self.docker_client_build != REGISTERED_DOCKER_CLIENT_BUILD
            or self.docker_executable_sha256 != REGISTERED_DOCKER_CLIENT_SHA256
        ):
            raise ExecutionClaimError("Docker client identity differs from the registered binary")
        if not isinstance(self.host_probe, PhaseHostProbe) or not isinstance(
            self.docker_server_probe, DockerServerProbe
        ):
            raise ExecutionClaimError("host and Docker server probes must be typed")
        if (
            self.host_probe.file_sha256 != self.host_probe_receipt_sha256
            or self.docker_server_probe.file_sha256 != self.docker_server_probe_receipt_sha256
        ):
            raise ExecutionClaimError("probe receipt digest differs from canonical probe bytes")
        if self.schema_version != PHASE_HOST_TOOL_CONTRACT_SCHEMA:
            raise ExecutionClaimError("phase host-tool contract schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"host_probe", "docker_server_probe"}
            },
            "docker_server_probe": self.docker_server_probe.to_dict(),
            "host_probe": self.host_probe.to_dict(),
        }

    @property
    def contract_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> PhaseHostToolContract:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="phase host-tool contract",
        )
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key not in {"host_probe", "docker_server_probe"}
            },
            host_probe=PhaseHostProbe.from_dict(row["host_probe"]),
            docker_server_probe=DockerServerProbe.from_dict(row["docker_server_probe"]),
        )


if (
    PHASE_HOST_TOOL_CONTRACT_SCHEMA != SHARED_PHASE_HOST_TOOL_CONTRACT_SCHEMA
    or PHASE_HOST_PROBE_SCHEMA != SHARED_PHASE_HOST_PROBE_SCHEMA
    or DOCKER_SERVER_PROBE_SCHEMA != SHARED_DOCKER_SERVER_PROBE_SCHEMA
    or frozenset(PhaseHostToolContract.__dataclass_fields__)
    != SHARED_PHASE_HOST_TOOL_CONTRACT_FIELDS
    or frozenset(PhaseHostProbe.__dataclass_fields__) != SHARED_PHASE_HOST_PROBE_FIELDS
    or frozenset(DockerServerProbe.__dataclass_fields__) != SHARED_DOCKER_SERVER_PROBE_FIELDS
):  # pragma: no cover - import-time contract assertion
    raise RuntimeError("provider host-tool schema drifted between manifest and runtime")


@dataclass(frozen=True)
class ExecutionClaimInputs:
    """The only online claim values not already present elsewhere in C1.

    The runtime budget is prespecified from development-only capacity planning. It is
    not a timing measurement over sealed confirmatory inputs.
    """

    design_seed_sha256: str
    registered_online_runtime_budget_seconds: int
    beacon: ExecutionBeaconContract

    def __post_init__(self) -> None:
        _digest("design_seed_sha256", self.design_seed_sha256)
        budget = _positive(
            "registered_online_runtime_budget_seconds",
            self.registered_online_runtime_budget_seconds,
        )
        if budget > MAXIMUM_REGISTERED_ONLINE_RUNTIME_SECONDS:
            raise ExecutionClaimError(
                "registered online runtime budget exceeds the 20-hour ceiling"
            )
        if not isinstance(self.beacon, ExecutionBeaconContract):
            raise ExecutionClaimError("execution claim inputs require a typed beacon")

    def to_dict(self) -> dict[str, object]:
        return {
            "beacon": self.beacon.to_dict(),
            "design_seed_sha256": self.design_seed_sha256,
            "registered_online_runtime_budget_seconds": (
                self.registered_online_runtime_budget_seconds
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> ExecutionClaimInputs:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="execution claim inputs",
        )
        return cls(
            design_seed_sha256=row["design_seed_sha256"],
            registered_online_runtime_budget_seconds=row[
                "registered_online_runtime_budget_seconds"
            ],
            beacon=ExecutionBeaconContract.from_dict(row["beacon"]),
        )


@dataclass(frozen=True)
class ProviderRunnerBootstrapReceipt:
    phase: ProviderPhase
    repository: str
    approval_environment: str
    runner_identity: str
    workflow_sha: str
    runner_label: str
    runner_id: int
    runner_name: str
    runner_group_id: int | None
    runner_version: str
    runner_archive_sha256: str
    repository_runner_inventory_sha256: str
    ephemeral: bool
    disable_update: bool
    unattended: bool
    registered_at_utc: str
    schema_version: str = PROVIDER_RUNNER_BOOTSTRAP_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.phase not in PHASE_JOB_NAMES
            or self.repository != "mhdk1602/fractal-ann-diagnostics"
        ):
            raise ExecutionClaimError("runner bootstrap phase or repository differs")
        if (
            self.approval_environment != PROVIDER_APPROVAL_ENVIRONMENT
            or self.runner_identity != PROVIDER_RUNNER_IDENTITY
            or self.runner_identity != f"github-actions:environment:{self.approval_environment}"
        ):
            raise ExecutionClaimError("runner bootstrap approval environment differs")
        _git_commit("workflow_sha", self.workflow_sha)
        if _RUNNER_LABEL.fullmatch(self.runner_label) is None:
            raise ExecutionClaimError("runner bootstrap label is invalid")
        _positive("runner_id", self.runner_id)
        _runner_group(self.runner_group_id)
        for name in ("runner_name", "runner_version"):
            _text(name, getattr(self, name))
        _digest("runner_archive_sha256", self.runner_archive_sha256)
        _digest(
            "repository_runner_inventory_sha256",
            self.repository_runner_inventory_sha256,
        )
        if (self.ephemeral, self.disable_update, self.unattended) != (True, True, True):
            raise ExecutionClaimError("production runner must be ephemeral and update-disabled")
        _timestamp("registered_at_utc", self.registered_at_utc)
        if self.schema_version != PROVIDER_RUNNER_BOOTSTRAP_SCHEMA:
            raise ExecutionClaimError("runner bootstrap schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProviderRunnerBootstrapReceipt:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="provider runner bootstrap receipt",
            )
        )


@dataclass(frozen=True)
class ProviderPhasePlan:
    """C1 template after resolving only its enclosing manifest and C1 commit."""

    phase: ProviderPhase
    manifest_sha256: str
    c1_commit: str
    suite_attempt_id_binding: str
    claim_predecessor_binding: str
    claim_receipt_path_template: str
    phase_input_binding: str
    phase_output_binding: str
    provider_plan_path: str
    phase_evidence_root_template: str
    repository: str
    approval_environment: str
    runner_identity: str
    workflow_path: str
    workflow_ref: str
    workflow_sha: str
    run_head_branch: str
    claim_job_name: str
    execute_job_name: str
    execution_claim_inputs: ExecutionClaimInputs | None
    claim_nonce: str
    runner_id: int
    runner_name: str
    runner_registration_bundle_path: str
    runner_registration_bundle_sha256: str
    runner_registration_evidence_file_sha256: str
    runner_group_id: int | None
    runner_version: str
    runner_archive_sha256: str
    runner_bootstrap_receipt: ProviderRunnerBootstrapReceipt
    runner_bootstrap_receipt_path: str
    runner_bootstrap_receipt_file_sha256: str
    provider_operating_system: str
    provider_architecture: str
    host_tools: PhaseHostToolContract
    runtime_probe_receipt_sha256: str
    runtime_image: str
    runtime_platform: str
    runtime_image_role: str
    runtime_index_role: str
    oci_index_digest: str
    oci_platform_manifest_digest: str
    tle_binary_sha256: str | None
    tle_build_provenance_sha256: str | None
    tle_vulnerability_scan_sha256: str | None
    tle_interoperability_receipt_sha256: str | None
    maximum_runtime_seconds: int
    activation_command_id: str
    activation_argv_template: tuple[str, ...]
    schema_version: str = PROVIDER_PHASE_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in PHASE_JOB_NAMES:
            raise ExecutionClaimError("provider phase plan has an unknown phase")
        _digest("manifest_sha256", self.manifest_sha256)
        _git_commit("c1_commit", self.c1_commit)
        if self.suite_attempt_id_binding != PROVIDER_PLAN_SUITE_BINDING:
            raise ExecutionClaimError("provider plan suite-attempt binding differs")
        if self.claim_predecessor_binding != PROVIDER_PLAN_PREDECESSOR_BINDING:
            raise ExecutionClaimError("provider plan predecessor binding differs")
        if self.phase_input_binding != PROVIDER_PLAN_PHASE_INPUT_BINDING:
            raise ExecutionClaimError("provider plan input binding differs")
        if self.phase_output_binding != PROVIDER_PLAN_PHASE_OUTPUT_BINDING:
            raise ExecutionClaimError("provider plan output binding differs")
        for name in (
            "provider_plan_path",
            "phase_evidence_root_template",
            "claim_receipt_path_template",
        ):
            value = _text(name, getattr(self, name))
            check = value.replace("{suite_attempt_id}", "0" * 64)
            _absolute_path(name, check)
            if "{suite_attempt_id}" in value and value.count("{suite_attempt_id}") != 1:
                raise ExecutionClaimError(f"{name} repeats the suite-attempt binding")
        if "{suite_attempt_id}" in self.provider_plan_path:
            raise ExecutionClaimError("fixed self-hosted provider plan path cannot be templated")
        if self.phase_evidence_root_template.count("{suite_attempt_id}") != 1:
            raise ExecutionClaimError("phase evidence root lacks its suite-attempt binding")
        if self.claim_receipt_path_template.count(
            "{suite_attempt_id}"
        ) != 1 or not self.claim_receipt_path_template.endswith("/claim-receipt.json"):
            raise ExecutionClaimError("claim receipt path template differs")
        if self.repository != "mhdk1602/fractal-ann-diagnostics":
            raise ExecutionClaimError("provider plan repository differs")
        if (
            self.approval_environment != PROVIDER_APPROVAL_ENVIRONMENT
            or self.runner_identity != PROVIDER_RUNNER_IDENTITY
            or self.runner_identity != f"github-actions:environment:{self.approval_environment}"
        ):
            raise ExecutionClaimError("provider plan approval environment differs")
        workflow = PROVIDER_PHASE_WORKFLOWS[self.phase]
        claim_job, execute_job = PHASE_JOB_NAMES[self.phase]
        if (
            self.workflow_path != workflow
            or self.workflow_ref
            != f"{self.repository}/{workflow}@refs/tags/confirmatory-apparatus-c0"
            or self.run_head_branch != "confirmatory-apparatus-c0"
            or self.claim_job_name != claim_job
            or self.execute_job_name != execute_job
        ):
            raise ExecutionClaimError("provider plan workflow or job identity differs")
        if self.phase == ONLINE_PHASE:
            if not isinstance(self.execution_claim_inputs, ExecutionClaimInputs):
                raise ExecutionClaimError("online provider plan lacks execution claim inputs")
            if (
                self.execution_claim_inputs.registered_online_runtime_budget_seconds
                > self.maximum_runtime_seconds
            ):
                raise ExecutionClaimError(
                    "registered online runtime budget exceeds the phase ceiling"
                )
        elif self.execution_claim_inputs is not None:
            raise ExecutionClaimError(
                "non-online provider plan cannot carry execution claim inputs"
            )
        _git_commit("workflow_sha", self.workflow_sha)
        _digest("claim_nonce", self.claim_nonce)
        _positive("runner_id", self.runner_id)
        _runner_group(self.runner_group_id)
        for name in (
            "runner_name",
            "runner_version",
            "provider_operating_system",
            "provider_architecture",
        ):
            _text(name, getattr(self, name))
        _digest("runner_archive_sha256", self.runner_archive_sha256)
        _digest("runner_registration_bundle_sha256", self.runner_registration_bundle_sha256)
        _digest(
            "runner_registration_evidence_file_sha256",
            self.runner_registration_evidence_file_sha256,
        )
        if not isinstance(self.runner_bootstrap_receipt, ProviderRunnerBootstrapReceipt):
            raise ExecutionClaimError("provider plan lacks a typed runner bootstrap receipt")
        _digest(
            "runner_bootstrap_receipt_file_sha256",
            self.runner_bootstrap_receipt_file_sha256,
        )
        bootstrap_path = Path(
            _absolute_path(
                "runner_bootstrap_receipt_path",
                self.runner_bootstrap_receipt_path,
            )
        )
        expected_bootstrap_path = (
            Path(self.host_tools.controlled_root)
            / "production"
            / "runners"
            / self.phase
            / derive_phase_runner_label(self.claim_nonce, self.phase)
            / "bootstrap-receipt.json"
        )
        if bootstrap_path != expected_bootstrap_path:
            raise ExecutionClaimError("runner bootstrap receipt path differs from C1")
        registration_root = Path(
            _absolute_path(
                "runner_registration_bundle_path",
                self.runner_registration_bundle_path,
            )
        )
        expected_registration_root = (
            Path(self.host_tools.controlled_root)
            / "production"
            / "runner-registrations"
            / self.phase
            / derive_phase_runner_label(self.claim_nonce, self.phase)
        )
        if registration_root != expected_registration_root:
            raise ExecutionClaimError("runner registration bundle path differs from C1")
        bootstrap_exact = {
            "approval_environment": self.approval_environment,
            "disable_update": self.host_tools.runner_disable_update,
            "ephemeral": self.host_tools.runner_ephemeral,
            "phase": self.phase,
            "repository": self.repository,
            "runner_archive_sha256": self.runner_archive_sha256,
            "runner_group_id": self.runner_group_id,
            "runner_id": self.runner_id,
            "runner_identity": self.runner_identity,
            "runner_label": derive_phase_runner_label(self.claim_nonce, self.phase),
            "runner_name": self.runner_name,
            "runner_version": self.runner_version,
            "unattended": self.host_tools.runner_unattended,
            "workflow_sha": self.workflow_sha,
        }
        for name, expected in bootstrap_exact.items():
            if getattr(self.runner_bootstrap_receipt, name) != expected:
                raise ExecutionClaimError(f"embedded runner bootstrap {name} differs from C1")
        if self.runner_bootstrap_receipt.file_sha256 != self.runner_bootstrap_receipt_file_sha256:
            raise ExecutionClaimError("embedded runner bootstrap bytes differ from C1 digest")
        if (
            self.runner_version != self.host_tools.runner_version
            or self.runner_archive_sha256 != self.host_tools.runner_archive_sha256
            or self.provider_operating_system != "macOS"
            or self.provider_architecture != "ARM64"
        ):
            raise ExecutionClaimError("provider plan runner differs from its host-tool closure")
        _digest("runtime_probe_receipt_sha256", self.runtime_probe_receipt_sha256)
        if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", self.runtime_image) is None:
            raise ExecutionClaimError("provider plan runtime image is not digest-pinned")
        platform, image_role, index_role = PHASE_RUNTIME_BINDINGS[self.phase]
        if (self.runtime_platform, self.runtime_image_role, self.runtime_index_role) != (
            platform,
            image_role,
            index_role,
        ) or self.maximum_runtime_seconds != PROVIDER_PHASE_RUNTIME_CEILINGS[self.phase]:
            raise ExecutionClaimError("provider plan runtime binding or ceiling differs")
        for name in ("oci_index_digest", "oci_platform_manifest_digest"):
            value = getattr(self, name)
            if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ExecutionClaimError(f"{name} must be one OCI SHA-256 digest")
        if self.oci_index_digest != self.runtime_image.rsplit("@", 1)[1]:
            raise ExecutionClaimError("provider plan OCI index differs from runtime image")
        tle_fields = (
            self.tle_binary_sha256,
            self.tle_build_provenance_sha256,
            self.tle_vulnerability_scan_sha256,
            self.tle_interoperability_receipt_sha256,
        )
        if self.phase == LABEL_RELEASE_PHASE:
            if any(value is None for value in tle_fields):
                raise ExecutionClaimError("label-release plan lacks its TLE closure")
            for value in tle_fields:
                _digest("label-release TLE binding", value)
            if self.tle_binary_sha256 != SOURCE_BUILT_LINUX_ARM64_TLE_SHA256:
                raise ExecutionClaimError("label-release plan changes the C0 TLE binary")
        elif any(value is not None for value in tle_fields):
            raise ExecutionClaimError("non-release provider plan introduces TLE")
        if self.activation_command_id != PROVIDER_PHASE_COMMAND_IDS[self.phase]:
            raise ExecutionClaimError("provider phase activation command differs")
        expected_argv = (
            self.host_tools.python_executable,
            "-m",
            "fractal_ann_diagnostics.provider_phase_runtime",
            self.activation_command_id,
            "--provider-plan",
            self.provider_plan_path,
            "--suite-attempt-id",
            PROVIDER_PLAN_SUITE_BINDING,
            "--claim-receipt",
            PROVIDER_PLAN_CLAIM_RECEIPT_BINDING,
            "--phase-input-root",
            PROVIDER_PLAN_PHASE_INPUT_BINDING,
            "--phase-output-root",
            PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
        )
        if self.activation_argv_template != expected_argv:
            raise ExecutionClaimError("provider phase activation argv differs")
        if self.schema_version != PROVIDER_PHASE_PLAN_SCHEMA:
            raise ExecutionClaimError("resolved provider phase plan schema differs")

    @property
    def suite_attempt_id(self) -> str:
        return _sha256(b"fractal-suite-attempt-v1\0" + self.manifest_sha256.encode("ascii"))

    def phase_evidence_root(self, suite_attempt_id: str) -> str:
        if _digest("suite_attempt_id", suite_attempt_id) != self.suite_attempt_id:
            raise ExecutionClaimError("suite-attempt ID differs from the admitted manifest")
        return self.phase_evidence_root_template.replace("{suite_attempt_id}", suite_attempt_id)

    def claim_receipt_path(self, suite_attempt_id: str) -> str:
        if _digest("suite_attempt_id", suite_attempt_id) != self.suite_attempt_id:
            raise ExecutionClaimError("suite-attempt ID differs from the admitted manifest")
        return self.claim_receipt_path_template.replace("{suite_attempt_id}", suite_attempt_id)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "activation_argv_template",
                    "execution_claim_inputs",
                    "host_tools",
                    "runner_bootstrap_receipt",
                }
            },
            "activation_argv_template": list(self.activation_argv_template),
            "c1_commit_source": PROVIDER_PLAN_C1_COMMIT_BINDING,
            "execution_claim_inputs": (
                None
                if self.execution_claim_inputs is None
                else self.execution_claim_inputs.to_dict()
            ),
            "host_tools": self.host_tools.to_dict(),
            "runner_bootstrap_receipt": self.runner_bootstrap_receipt.to_dict(),
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def plan_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @property
    def file_sha256(self) -> str:
        return _sha256(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProviderPhasePlan:
        fields = frozenset(cls.__dataclass_fields__) | {"c1_commit_source"}
        row = _closed(value, fields, label="resolved provider phase plan")
        if row["c1_commit_source"] != PROVIDER_PLAN_C1_COMMIT_BINDING:
            raise ExecutionClaimError("resolved provider-plan C1 source binding differs")
        argv = row["activation_argv_template"]
        if not isinstance(argv, list) or not all(type(item) is str for item in argv):
            raise ExecutionClaimError("resolved provider-plan argv is malformed")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "activation_argv_template",
                    "c1_commit_source",
                    "execution_claim_inputs",
                    "host_tools",
                    "runner_bootstrap_receipt",
                }
            },
            activation_argv_template=tuple(argv),
            execution_claim_inputs=(
                None
                if row["execution_claim_inputs"] is None
                else ExecutionClaimInputs.from_dict(row["execution_claim_inputs"])
            ),
            host_tools=PhaseHostToolContract.from_dict(row["host_tools"]),
            runner_bootstrap_receipt=ProviderRunnerBootstrapReceipt.from_dict(
                row["runner_bootstrap_receipt"]
            ),
        )


def _admit_provider_plan_manifest(
    manifest: Mapping[str, Any],
    *,
    validation_mode: Literal["frozen", "candidate-rehearsal"],
    c0_commit: str | None,
) -> Mapping[str, Any]:
    if validation_mode == "frozen":
        if c0_commit is not None:
            raise ExecutionClaimError("frozen provider-plan validation rejects a C0 override")
        try:
            validate_study_manifest(manifest, require_frozen=True)
        except ValueError as exc:
            raise ExecutionClaimError(f"invalid frozen provider-plan manifest: {exc}") from exc
        return manifest
    if validation_mode != "candidate-rehearsal":
        raise ExecutionClaimError("provider-plan validation mode differs")
    if c0_commit is None:
        raise ExecutionClaimError("candidate rehearsal requires the current C0 commit")
    commit = _git_commit("c0_commit", c0_commit)
    try:
        validate_candidate_rehearsal_manifest(manifest, c0_commit=commit)
        return resolve_candidate_provider_plan_commit_bindings(
            manifest,
            c0_commit=commit,
        )
    except ValueError as exc:
        raise ExecutionClaimError(f"invalid candidate provider-plan manifest: {exc}") from exc


def provider_phase_plan_templates_sha256(
    manifest: Mapping[str, Any],
    *,
    validation_mode: Literal["frozen", "candidate-rehearsal"] = "frozen",
    c0_commit: str | None = None,
) -> str:
    """Hash the three templates after the sole candidate C0 resolution."""

    admitted = _admit_provider_plan_manifest(
        manifest,
        validation_mode=validation_mode,
        c0_commit=c0_commit,
    )
    sealed = admitted.get("sealed_execution")
    if not isinstance(sealed, Mapping):
        raise ExecutionClaimError("provider-plan manifest lacks sealed_execution")
    raw = sealed.get("provider_phase_plans")
    if not isinstance(raw, Mapping) or set(raw) != set(PHASE_JOB_NAMES):
        raise ExecutionClaimError("provider-plan manifest lacks exactly three provider plans")
    return _sha256(_canonical_bytes(raw))


def assert_normalized_provider_phase_plan_closure(
    candidate_manifest: Mapping[str, Any],
    frozen_manifest: Mapping[str, Any],
    *,
    c0_commit: str,
) -> str:
    """Require candidate and C1 provider plans to differ only at the C0 sentinel."""

    candidate_sha256 = provider_phase_plan_templates_sha256(
        candidate_manifest,
        validation_mode="candidate-rehearsal",
        c0_commit=c0_commit,
    )
    frozen_sha256 = provider_phase_plan_templates_sha256(frozen_manifest)
    if candidate_sha256 != frozen_sha256:
        raise ExecutionClaimError("normalized provider-plan closure differs between C0 and C1")
    return frozen_sha256


def _bounded_canonical_provider_plan(path: Path) -> tuple[ProviderPhasePlan, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ExecutionClaimError("cannot open the materialized provider plan") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 4 * 1024**2:
            raise ExecutionClaimError(
                "materialized provider plan must be one bounded singly linked regular file"
            )
        chunks: list[bytes] = []
        observed = 0
        while observed <= 4 * 1024**2:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, 4 * 1024**2 + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ExecutionClaimError("cannot read the materialized provider plan") from exc
    finally:
        os.close(descriptor)
    if observed > 4 * 1024**2 or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ExecutionClaimError("materialized provider plan changed while read")
    encoded = b"".join(chunks)
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ExecutionClaimError("materialized provider plan lacks one terminal newline")
    plan = ProviderPhasePlan.from_dict(
        _strict_object(encoded[:-1], label="materialized provider phase plan")
    )
    if encoded != plan.canonical_file_bytes():
        raise ExecutionClaimError("materialized provider plan bytes are not canonical")
    return plan, encoded


def load_materialized_provider_phase_plan(path: str | Path) -> ProviderPhasePlan:
    """Load the fixed self-hosted resolved C1 plan from exact canonical bytes."""

    candidate = Path(path)
    if not candidate.is_absolute():
        raise ExecutionClaimError("materialized provider plan path must be absolute")
    plan, encoded = _bounded_canonical_provider_plan(candidate)
    if candidate != Path(plan.provider_plan_path):
        raise ExecutionClaimError("provider plan was not loaded from its fixed self-hosted path")
    if _sha256(encoded) != plan.file_sha256:
        raise ExecutionClaimError("materialized provider plan file digest differs")
    return plan


def load_provider_runner_bootstrap(
    plan: ProviderPhasePlan,
) -> ProviderRunnerBootstrapReceipt:
    """Load and rehash the phase-specific C1 runner bootstrap receipt."""

    if not isinstance(plan, ProviderPhasePlan):
        raise ExecutionClaimError("runner bootstrap admission requires a typed provider plan")
    path = Path(plan.runner_bootstrap_receipt_path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ExecutionClaimError("cannot open the C1 runner bootstrap receipt") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > 1024 * 1024
        ):
            raise ExecutionClaimError(
                "runner bootstrap receipt must be one bounded singly linked regular file"
            )
        encoded = b""
        while len(encoded) <= 1024 * 1024:
            chunk = os.read(descriptor, min(1024 * 1024 + 1 - len(encoded), 64 * 1024))
            if not chunk:
                break
            encoded += chunk
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ExecutionClaimError("cannot read the C1 runner bootstrap receipt") from exc
    finally:
        os.close(descriptor)
    if (
        len(encoded) > 1024 * 1024
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or not encoded.endswith(b"\n")
        or encoded.endswith(b"\n\n")
    ):
        raise ExecutionClaimError("runner bootstrap receipt changed or is not canonical")
    receipt = ProviderRunnerBootstrapReceipt.from_dict(
        _strict_object(encoded[:-1], label="provider runner bootstrap receipt")
    )
    if (
        receipt.canonical_file_bytes() != encoded
        or receipt != plan.runner_bootstrap_receipt
        or receipt.file_sha256 != plan.runner_bootstrap_receipt_file_sha256
    ):
        raise ExecutionClaimError("runner bootstrap receipt bytes differ from C1")
    exact = {
        "approval_environment": plan.approval_environment,
        "disable_update": plan.host_tools.runner_disable_update,
        "ephemeral": plan.host_tools.runner_ephemeral,
        "phase": plan.phase,
        "repository": plan.repository,
        "runner_archive_sha256": plan.runner_archive_sha256,
        "runner_group_id": plan.runner_group_id,
        "runner_id": plan.runner_id,
        "runner_identity": plan.runner_identity,
        "runner_label": derive_phase_runner_label(plan.claim_nonce, plan.phase),
        "runner_name": plan.runner_name,
        "runner_version": plan.runner_version,
        "unattended": plan.host_tools.runner_unattended,
        "workflow_sha": plan.workflow_sha,
    }
    for name, expected in exact.items():
        if getattr(receipt, name) != expected:
            raise ExecutionClaimError(f"runner bootstrap {name} differs from C1")
    return receipt


def load_provider_phase_plans(
    manifest_path: str | Path,
    *,
    c1_commit: str,
    validation_mode: Literal["frozen", "candidate-rehearsal"] = "frozen",
    c0_commit: str | None = None,
) -> Mapping[ProviderPhase, ProviderPhasePlan]:
    """Load three templates after the exact lifecycle-specific admission."""

    path = Path(manifest_path)
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise ExecutionClaimError("cannot read the C1 provider-plan manifest") from exc
    if len(encoded) > 16 * 1024 * 1024 or not encoded.endswith(b"\n"):
        raise ExecutionClaimError("C1 provider-plan manifest bytes are not bounded canonical JSON")
    manifest = load_study_manifest(path)
    try:
        admitted = _admit_provider_plan_manifest(
            manifest,
            validation_mode=validation_mode,
            c0_commit=c0_commit,
        )
    except ExecutionClaimError as exc:
        raise ExecutionClaimError(f"provider-plan manifest is invalid: {exc}") from exc
    if encoded != _canonical_bytes(manifest) + b"\n":
        raise ExecutionClaimError("provider-plan manifest bytes are not canonical")
    digest = manifest_sha256(manifest)
    commit = _git_commit("c1_commit", c1_commit)
    sealed = admitted.get("sealed_execution")
    if not isinstance(sealed, Mapping):
        raise ExecutionClaimError("provider-plan manifest lacks sealed_execution")
    raw_plans = sealed.get("provider_phase_plans")
    if not isinstance(raw_plans, Mapping) or set(raw_plans) != set(PHASE_JOB_NAMES):
        raise ExecutionClaimError("provider-plan manifest lacks exactly three provider plans")
    plans: dict[ProviderPhase, ProviderPhasePlan] = {}
    for phase in (ONLINE_PHASE, LABEL_RELEASE_PHASE, ANALYSIS_PHASE):
        row = _closed(
            raw_plans[phase],
            PROVIDER_PHASE_PLAN_TEMPLATE_FIELDS,
            label=f"{phase} provider phase plan template",
        )
        if (
            row["schema_version"] != PROVIDER_PHASE_PLAN_TEMPLATE_SCHEMA
            or row["manifest_sha256_binding"] != PROVIDER_PLAN_MANIFEST_BINDING
            or row["c1_commit_binding"] != PROVIDER_PLAN_C1_COMMIT_BINDING
            or row["workflow_sha"] != sealed.get("code_commit")
        ):
            raise ExecutionClaimError(f"{phase} provider plan resolution binding differs")
        argv = row["activation_argv_template"]
        if not isinstance(argv, list) or not all(type(item) is str for item in argv):
            raise ExecutionClaimError(f"{phase} provider activation argv is malformed")
        plans[phase] = ProviderPhasePlan(
            **{
                key: value
                for key, value in row.items()
                if key
                not in {
                    "activation_argv_template",
                    "c1_commit_binding",
                    "execution_claim_inputs",
                    "host_tools",
                    "runner_bootstrap_receipt",
                    "manifest_sha256_binding",
                    "schema_version",
                }
            },
            activation_argv_template=tuple(argv),
            c1_commit=commit,
            execution_claim_inputs=(
                None
                if row["execution_claim_inputs"] is None
                else ExecutionClaimInputs.from_dict(row["execution_claim_inputs"])
            ),
            host_tools=PhaseHostToolContract.from_dict(row["host_tools"]),
            runner_bootstrap_receipt=ProviderRunnerBootstrapReceipt.from_dict(
                row["runner_bootstrap_receipt"]
            ),
            manifest_sha256=digest,
        )
    if len({plan.provider_plan_path for plan in plans.values()}) != 3:
        raise ExecutionClaimError("C1 provider plans reuse a self-hosted path")
    if len({plan.runner_registration_bundle_path for plan in plans.values()}) != 3:
        raise ExecutionClaimError("C1 provider plans reuse a runner registration bundle")
    return plans


def materialize_provider_phase_plan(
    plan: ProviderPhasePlan,
    output_dir: str | Path,
) -> Path:
    """Write the hosted evidence copy; never replace the fixed self-hosted path."""

    if not isinstance(plan, ProviderPhasePlan):
        raise ExecutionClaimError("provider-plan materialization requires a typed plan")
    root = Path(output_dir)
    if not root.is_absolute():
        raise ExecutionClaimError("provider-plan materialization root must be absolute")
    fixed = Path(plan.provider_plan_path)
    if root == fixed or root in fixed.parents or fixed in root.parents:
        raise ExecutionClaimError(
            "hosted provider-plan materialization overlaps the self-hosted path"
        )
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise ExecutionClaimError("cannot create provider-plan materialization root") from exc
    path = root / "provider-plan.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            encoded = plan.canonical_file_bytes()
            view = memoryview(encoded)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("short provider-plan write")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ExecutionClaimError("cannot materialize the resolved provider plan") from exc
    if (
        written != len(encoded)
        or _hash_file(path, label="materialized provider plan") != plan.file_sha256
    ):
        raise ExecutionClaimError("materialized provider plan failed exact readback")
    return path


def _hash_file(path: Path, *, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ExecutionClaimError(f"cannot read {label}") from exc
    return digest.hexdigest()


def _contained_realpath(path: Path, root: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ExecutionClaimError(f"{label} resolves outside controlled_root") from exc
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise ExecutionClaimError(f"cannot stat {label}") from exc
    if not stat.S_ISREG(mode):
        raise ExecutionClaimError(f"{label} does not resolve to a regular file")
    return resolved


def _venv_tree_digests(venv_root: Path, controlled_root: Path) -> tuple[str, str]:
    entries: list[dict[str, object]] = []
    symlinks: list[dict[str, str]] = []
    try:
        root_stat = venv_root.lstat()
    except OSError as exc:
        raise ExecutionClaimError("cannot lstat registered venv_root") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise ExecutionClaimError("venv_root must be a real directory, not a symlink")
    for directory, directory_names, file_names in os.walk(venv_root, followlinks=False):
        directory_names.sort(key=lambda value: value.encode("utf-8"))
        file_names.sort(key=lambda value: value.encode("utf-8"))
        current = Path(directory)
        names = [*directory_names, *file_names]
        for name in names:
            candidate = current / name
            relative = candidate.relative_to(venv_root).as_posix()
            try:
                mode = candidate.lstat().st_mode
            except OSError as exc:
                raise ExecutionClaimError(f"cannot lstat venv entry {relative}") from exc
            if stat.S_ISLNK(mode):
                try:
                    raw_target = os.readlink(candidate)
                    resolved = candidate.resolve(strict=True)
                    resolved.relative_to(controlled_root)
                except (OSError, ValueError) as exc:
                    raise ExecutionClaimError(
                        f"venv symlink {relative} escapes controlled_root or is broken"
                    ) from exc
                row = {
                    "path": relative,
                    "resolved_path": str(resolved),
                    "target": raw_target,
                }
                symlinks.append(row)
                entries.append({"kind": "symlink", **row})
            elif stat.S_ISDIR(mode):
                entries.append({"kind": "directory", "path": relative})
            elif stat.S_ISREG(mode):
                size = candidate.stat().st_size
                entries.append(
                    {
                        "byte_count": size,
                        "file_sha256": _hash_file(candidate, label=f"venv file {relative}"),
                        "kind": "file",
                        "path": relative,
                    }
                )
            else:
                raise ExecutionClaimError(f"venv entry {relative} has a forbidden file type")
    entries.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    symlinks.sort(key=lambda row: row["path"].encode("utf-8"))
    tree = _sha256(_canonical_bytes({"derivation": HOST_TOOL_TREE_DERIVATION, "entries": entries}))
    inventory = _sha256(_canonical_bytes({"symlinks": symlinks}))
    return tree, inventory


def capture_phase_host_probe() -> PhaseHostProbe:
    """Observe the current macOS host through a fixed, argument-free probe."""

    logical = os.cpu_count()
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError) as exc:
        raise ExecutionClaimError("cannot read physical host memory") from exc
    machine = platform.machine().lower()
    return PhaseHostProbe(
        operating_system="macOS" if platform.system() == "Darwin" else platform.system(),
        operating_system_version=platform.mac_ver()[0],
        kernel_release=platform.release(),
        architecture="ARM64" if machine in {"arm64", "aarch64"} else machine,
        logical_cpu_count=logical if logical is not None else 0,
        physical_memory_bytes=page_size * page_count,
    )


def _docker_json(
    docker_executable: Path,
    arguments: tuple[str, ...],
    *,
    label: str,
) -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            (str(docker_executable), *arguments),
            check=False,
            capture_output=True,
            env={"HOME": "/var/empty", "PATH": "/usr/bin:/bin"},
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExecutionClaimError(f"cannot execute fixed {label}") from exc
    if completed.returncode != 0 or completed.stderr or len(completed.stdout) > 1024 * 1024:
        raise ExecutionClaimError(f"fixed {label} failed or emitted unexpected bytes")
    encoded = completed.stdout.strip()
    if not encoded or b"\n" in encoded or b"\r" in encoded:
        raise ExecutionClaimError(f"fixed {label} did not return one JSON object")
    return _strict_object(encoded, label=label)


def capture_docker_server_probe(docker_executable: str | Path) -> DockerServerProbe:
    """Observe the Docker server with fixed format strings and no caller data."""

    executable = Path(docker_executable)
    _absolute_path("docker_executable", str(executable))
    version = _docker_json(
        executable,
        ("version", "--format", "{{json .Server}}"),
        label="Docker server version probe",
    )
    info = _docker_json(
        executable,
        ("info", "--format", "{{json .}}"),
        label="Docker server info probe",
    )
    return DockerServerProbe(
        engine_version=version.get("Version"),
        engine_build=version.get("GitCommit"),
        kernel_version=info.get("KernelVersion"),
        operating_system=version.get("Os"),
        architecture=version.get("Arch"),
        cpu_count=info.get("NCPU"),
        memory_bytes=info.get("MemTotal"),
    )


def _write_fresh_probe(path: Path, encoded: bytes, *, label: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(encoded)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("short probe write")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ExecutionClaimError(f"cannot write {label} exactly once") from exc


def generate_phase_host_probes(
    contract: PhaseHostToolContract,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Generate canonical fresh probes at fixed evidence filenames."""

    if not isinstance(contract, PhaseHostToolContract):
        raise ExecutionClaimError("probe generation requires a typed host contract")
    root = Path(output_dir)
    if not root.is_absolute():
        raise ExecutionClaimError("fresh probe output directory must be absolute")
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as exc:
        raise ExecutionClaimError("cannot create the fresh probe evidence directory") from exc
    host = capture_phase_host_probe()
    docker = capture_docker_server_probe(contract.docker_executable)
    host_path = root / PHASE_HOST_PROBE_FILENAME
    docker_path = root / DOCKER_SERVER_PROBE_FILENAME
    _write_fresh_probe(host_path, host.canonical_file_bytes(), label="fresh host probe")
    _write_fresh_probe(
        docker_path,
        docker.canonical_file_bytes(),
        label="fresh Docker server probe",
    )
    return host_path, docker_path


@dataclass(frozen=True)
class PhaseHostToolReceipt:
    contract_sha256: str
    controlled_root_realpath: str
    python_executable_sha256: str
    venv_tree_sha256: str
    venv_symlink_inventory_sha256: str
    gh_executable_sha256: str
    runner_listener_sha256: str
    runner_listener_dll_sha256: str
    runner_config_sha256: str
    runner_run_sha256: str
    docker_resolved_executable: str
    docker_executable_sha256: str
    host_probe_receipt_file_sha256: str
    docker_server_probe_receipt_file_sha256: str
    verified_at_utc: str
    schema_version: str = PHASE_HOST_TOOL_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "contract_sha256",
            "python_executable_sha256",
            "venv_tree_sha256",
            "venv_symlink_inventory_sha256",
            "gh_executable_sha256",
            "runner_listener_sha256",
            "runner_listener_dll_sha256",
            "runner_config_sha256",
            "runner_run_sha256",
            "docker_executable_sha256",
            "host_probe_receipt_file_sha256",
            "docker_server_probe_receipt_file_sha256",
        ):
            _digest(name, getattr(self, name))
        _absolute_path("controlled_root_realpath", self.controlled_root_realpath)
        _absolute_path("docker_resolved_executable", self.docker_resolved_executable)
        _timestamp("verified_at_utc", self.verified_at_utc)
        if self.schema_version != PHASE_HOST_TOOL_RECEIPT_SCHEMA:
            raise ExecutionClaimError("phase host-tool receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> PhaseHostToolReceipt:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="phase host-tool receipt",
            )
        )


def verify_phase_host_tools(
    contract: PhaseHostToolContract,
    *,
    probe_output_dir: str | Path,
    verified_at_utc: str,
) -> PhaseHostToolReceipt:
    """Rehash the C1 host closure and reject every symlink escape."""

    if not isinstance(contract, PhaseHostToolContract):
        raise ExecutionClaimError("host-tool verification requires a typed contract")
    root = Path(contract.controlled_root)
    try:
        if stat.S_ISLNK(root.lstat().st_mode):
            raise ExecutionClaimError("controlled_root cannot be a symlink")
        root_real = root.resolve(strict=True)
    except OSError as exc:
        raise ExecutionClaimError("cannot resolve controlled_root") from exc
    if root_real != root or not root_real.is_dir():
        raise ExecutionClaimError("controlled_root must be an existing canonical directory")

    controlled_files = {
        "python_executable": contract.python_executable,
        "gh_executable": contract.gh_executable,
        "runner_listener": contract.runner_listener_executable,
        "runner_listener_dll": contract.runner_listener_dll,
        "runner_config": contract.runner_config_executable,
        "runner_run": contract.runner_run_executable,
    }
    observed: dict[str, str] = {}
    for label, value in controlled_files.items():
        resolved = _contained_realpath(Path(value), root_real, label=label)
        observed[label] = _hash_file(resolved, label=label)
    expected = {
        "python_executable": contract.python_executable_sha256,
        "gh_executable": contract.gh_executable_sha256,
        "runner_listener": contract.runner_listener_sha256,
        "runner_listener_dll": contract.runner_listener_dll_sha256,
        "runner_config": contract.runner_config_sha256,
        "runner_run": contract.runner_run_sha256,
    }
    for label, digest in observed.items():
        if digest != expected[label]:
            raise ExecutionClaimError(f"{label} bytes differ from the C1 contract")

    docker_link = Path(contract.docker_executable)
    try:
        if not stat.S_ISLNK(docker_link.lstat().st_mode):
            raise ExecutionClaimError("Docker invocation path must be the registered symlink")
        docker_real = docker_link.resolve(strict=True)
    except OSError as exc:
        raise ExecutionClaimError("cannot resolve Docker invocation path") from exc
    if str(docker_real) != contract.docker_resolved_executable:
        raise ExecutionClaimError("Docker symlink target differs from the C1 contract")
    docker_digest = _hash_file(docker_real, label="Docker client")
    if docker_digest != contract.docker_executable_sha256:
        raise ExecutionClaimError("Docker client bytes differ from the C1 contract")

    venv_root = Path(contract.venv_root)
    try:
        venv_real = venv_root.resolve(strict=True)
        venv_real.relative_to(root_real)
    except (OSError, ValueError) as exc:
        raise ExecutionClaimError("venv_root resolves outside controlled_root") from exc
    if venv_real != venv_root:
        raise ExecutionClaimError("venv_root or one of its parents is a symlink")
    tree, symlink_inventory = _venv_tree_digests(venv_real, root_real)
    if tree != contract.venv_tree_sha256:
        raise ExecutionClaimError("controlled venv tree differs from the C1 contract")
    if symlink_inventory != contract.venv_symlink_inventory_sha256:
        raise ExecutionClaimError("controlled venv symlink inventory differs from C1")
    host_probe_receipt_path, docker_server_probe_receipt_path = generate_phase_host_probes(
        contract, probe_output_dir
    )
    try:
        host_probe_bytes = host_probe_receipt_path.read_bytes()
        docker_probe_bytes = docker_server_probe_receipt_path.read_bytes()
    except OSError as exc:
        raise ExecutionClaimError("cannot read fresh host or Docker server probe") from exc
    if (
        not host_probe_bytes.endswith(b"\n")
        or host_probe_bytes.endswith(b"\n\n")
        or not docker_probe_bytes.endswith(b"\n")
        or docker_probe_bytes.endswith(b"\n\n")
        or len(host_probe_bytes) > 64 * 1024
        or len(docker_probe_bytes) > 64 * 1024
    ):
        raise ExecutionClaimError("fresh host or Docker probe bytes are not bounded canonical JSON")
    live_host_probe = PhaseHostProbe.from_dict(
        _strict_object(host_probe_bytes[:-1], label="fresh host probe")
    )
    live_docker_probe = DockerServerProbe.from_dict(
        _strict_object(docker_probe_bytes[:-1], label="fresh Docker server probe")
    )
    if (
        host_probe_bytes != live_host_probe.canonical_file_bytes()
        or docker_probe_bytes != live_docker_probe.canonical_file_bytes()
        or live_host_probe != contract.host_probe
        or live_docker_probe != contract.docker_server_probe
    ):
        raise ExecutionClaimError("fresh host or Docker probe differs from C1")
    host_probe_digest = _sha256(host_probe_bytes)
    docker_probe_digest = _sha256(docker_probe_bytes)
    if (
        host_probe_digest != contract.host_probe_receipt_sha256
        or docker_probe_digest != contract.docker_server_probe_receipt_sha256
    ):
        raise ExecutionClaimError("host or Docker server probe differs from the C1 contract")
    return PhaseHostToolReceipt(
        contract_sha256=contract.contract_sha256,
        controlled_root_realpath=str(root_real),
        python_executable_sha256=observed["python_executable"],
        venv_tree_sha256=tree,
        venv_symlink_inventory_sha256=symlink_inventory,
        gh_executable_sha256=observed["gh_executable"],
        runner_listener_sha256=observed["runner_listener"],
        runner_listener_dll_sha256=observed["runner_listener_dll"],
        runner_config_sha256=observed["runner_config"],
        runner_run_sha256=observed["runner_run"],
        docker_resolved_executable=str(docker_real),
        docker_executable_sha256=docker_digest,
        host_probe_receipt_file_sha256=host_probe_digest,
        docker_server_probe_receipt_file_sha256=docker_probe_digest,
        verified_at_utc=_timestamp("verified_at_utc", verified_at_utc).isoformat(),
    )


@dataclass(frozen=True)
class ExecutionBeaconContract:
    """Frozen future randomness and strictly later label-release boundary."""

    drand_network: str
    chain_hash: str
    chain_scheme_id: str
    chain_public_key: str
    chain_genesis_unix_seconds: int
    chain_period_seconds: int
    execution_round: int
    label_release_round: int
    minimum_label_release_safety_rounds: int
    verification_identity: str
    seed_derivation: str = EXECUTION_SEED_DERIVATION
    schema_version: str = EXECUTION_BEACON_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        _https("drand_network", self.drand_network)
        _digest("chain_hash", self.chain_hash)
        _text("chain_scheme_id", self.chain_scheme_id)
        if type(self.chain_public_key) is not str or _HEX.fullmatch(self.chain_public_key) is None:
            raise ExecutionClaimError("chain_public_key must be lowercase hexadecimal bytes")
        _positive("chain_genesis_unix_seconds", self.chain_genesis_unix_seconds)
        _positive("chain_period_seconds", self.chain_period_seconds)
        execution = _positive("execution_round", self.execution_round)
        release = _positive("label_release_round", self.label_release_round)
        safety = _positive(
            "minimum_label_release_safety_rounds",
            self.minimum_label_release_safety_rounds,
        )
        if release < execution + safety:
            raise ExecutionClaimError(
                "label-release round must be strictly later by the registered safety interval"
            )
        _digest("verification_identity", self.verification_identity)
        if self.seed_derivation != EXECUTION_SEED_DERIVATION:
            raise ExecutionClaimError("execution seed derivation differs")
        if self.schema_version != EXECUTION_BEACON_CONTRACT_SCHEMA:
            raise ExecutionClaimError("execution beacon contract schema differs")

    @property
    def execution_publication_time(self) -> datetime:
        seconds = (
            self.chain_genesis_unix_seconds + (self.execution_round - 1) * self.chain_period_seconds
        )
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    @property
    def label_release_publication_time(self) -> datetime:
        seconds = (
            self.chain_genesis_unix_seconds
            + (self.label_release_round - 1) * self.chain_period_seconds
        )
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def contract_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> ExecutionBeaconContract:
        return cls(
            **_closed(value, frozenset(cls.__dataclass_fields__), label="execution beacon contract")
        )


@dataclass(frozen=True)
class AnonymousZenodoAdmission:
    """Anonymous readback of the already-public exact C1 package."""

    record_id: int
    doi: str
    record_uri: str
    published_at_utc: str
    file_count: int
    package_tree_sha256: str
    package_aggregate_sha256: str
    receipt_file_sha256: str
    verified_at_utc: str
    schema_version: str = ZENODO_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        if _positive("record_id", self.record_id) != 21361837:
            raise ExecutionClaimError("Zenodo admission must bind record 21361837")
        if self.doi != "10.5281/zenodo.21361837":
            raise ExecutionClaimError("Zenodo admission DOI differs")
        _https("record_uri", self.record_uri)
        published = _timestamp("published_at_utc", self.published_at_utc)
        verified = _timestamp("verified_at_utc", self.verified_at_utc)
        if verified < published:
            raise ExecutionClaimError("Zenodo readback predates publication")
        if self.file_count != C1_REGISTRATION_PACKAGE_FILE_COUNT:
            raise ExecutionClaimError(
                f"Zenodo admission must verify exactly {C1_REGISTRATION_PACKAGE_FILE_COUNT} files"
            )
        for name in (
            "package_tree_sha256",
            "package_aggregate_sha256",
            "receipt_file_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.schema_version != ZENODO_ADMISSION_SCHEMA:
            raise ExecutionClaimError("Zenodo admission schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> AnonymousZenodoAdmission:
        return cls(**_closed(value, frozenset(cls.__dataclass_fields__), label="Zenodo admission"))


@dataclass(frozen=True)
class ExecutionClaimContract:
    """All C1-fixed data that an online provider claim must equal."""

    repository: str
    claim_workflow_path: str
    claim_workflow_ref: str
    claim_workflow_sha: str
    run_head_branch: str | None
    claim_job_name: str
    execute_job_name: str
    unique_runner_label: str
    claim_nonce: str
    runner_id: int
    runner_name: str
    runner_group_id: int | None
    runner_version: str
    runner_archive_sha256: str
    provider_operating_system: str
    provider_architecture: str
    host_tools: PhaseHostToolContract
    runtime_probe_receipt_sha256: str
    design_seed_sha256: str
    registered_online_runtime_budget_seconds: int
    maximum_online_runtime_seconds: int
    c1_commit: str
    manifest_sha256: str
    label_release_provider_plan_uri: str
    label_release_provider_plan_sha256: str
    analysis_provider_plan_uri: str
    analysis_provider_plan_sha256: str
    run_receipt_sha256: str
    run_receipt_file_sha256: str
    oci_index_digest: str
    oci_platform_manifest_digest: str
    analysis_oci_platform_manifest_digest: str
    release_oci_index_digest: str
    release_oci_platform_manifest_digest: str
    release_tle_binary_sha256: str
    release_tle_build_provenance_sha256: str
    release_tle_vulnerability_scan_sha256: str
    release_tle_interoperability_receipt_sha256: str
    hardware_contract_sha256: str
    corpora: tuple[ClaimCorpusBinding, ...]
    output_aggregate_identity: str
    beacon: ExecutionBeaconContract
    schema_version: str = EXECUTION_CLAIM_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if _REPOSITORY.fullmatch(self.repository) is None:
            raise ExecutionClaimError("claim repository must use owner/repository syntax")
        if _WORKFLOW.fullmatch(self.claim_workflow_path) is None:
            raise ExecutionClaimError("claim workflow path is invalid")
        expected_ref = (
            f"{self.repository}/{self.claim_workflow_path}@refs/tags/confirmatory-apparatus-c0"
        )
        if self.claim_workflow_ref != expected_ref:
            raise ExecutionClaimError("claim workflow ref is not the exact C0 workflow identity")
        _git_commit("claim_workflow_sha", self.claim_workflow_sha)
        _api_head_branch("run_head_branch", self.run_head_branch)
        if (self.claim_job_name, self.execute_job_name) != PHASE_JOB_NAMES[ONLINE_PHASE]:
            raise ExecutionClaimError("online provider job names differ")
        if _RUNNER_LABEL.fullmatch(self.unique_runner_label) is None:
            raise ExecutionClaimError("unique runner label is outside the closed namespace")
        _digest("claim_nonce", self.claim_nonce)
        if self.unique_runner_label != derive_phase_runner_label(self.claim_nonce, ONLINE_PHASE):
            raise ExecutionClaimError("online runner label is not claim-nonce-derived")
        _positive("runner_id", self.runner_id)
        _runner_group(self.runner_group_id)
        for name in (
            "runner_name",
            "runner_version",
            "provider_operating_system",
            "provider_architecture",
        ):
            _text(name, getattr(self, name))
        for name in ("runner_archive_sha256", "runtime_probe_receipt_sha256"):
            _digest(name, getattr(self, name))
        if not isinstance(self.host_tools, PhaseHostToolContract):
            raise ExecutionClaimError("claim host_tools must be a typed C1 contract")
        if (
            self.runner_version != self.host_tools.runner_version
            or self.runner_archive_sha256 != self.host_tools.runner_archive_sha256
            or self.provider_operating_system != self.host_tools.host_operating_system
            or self.provider_architecture != self.host_tools.host_architecture
        ):
            raise ExecutionClaimError("claim runner fields differ from host_tools")
        _digest("design_seed_sha256", self.design_seed_sha256)
        budget = _positive(
            "registered_online_runtime_budget_seconds",
            self.registered_online_runtime_budget_seconds,
        )
        ceiling = _positive(
            "maximum_online_runtime_seconds",
            self.maximum_online_runtime_seconds,
        )
        if ceiling > MAXIMUM_REGISTERED_ONLINE_RUNTIME_SECONDS or budget > ceiling:
            raise ExecutionClaimError(
                "registered online runtime budget does not fit the 20-hour provider-token ceiling"
            )
        _git_commit("c1_commit", self.c1_commit)
        for name in (
            "manifest_sha256",
            "label_release_provider_plan_sha256",
            "analysis_provider_plan_sha256",
            "run_receipt_sha256",
            "run_receipt_file_sha256",
            "hardware_contract_sha256",
            "output_aggregate_identity",
            "release_tle_binary_sha256",
            "release_tle_build_provenance_sha256",
            "release_tle_vulnerability_scan_sha256",
            "release_tle_interoperability_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.release_tle_binary_sha256 != SOURCE_BUILT_LINUX_ARM64_TLE_SHA256:
            raise ExecutionClaimError("release tle binary differs from the source-built C0 pin")
        for name in ("label_release_provider_plan_uri", "analysis_provider_plan_uri"):
            parsed = urlsplit(_text(name, getattr(self, name)))
            if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
                raise ExecutionClaimError(f"{name} must be a canonical file URI")
        if self.label_release_provider_plan_uri == self.analysis_provider_plan_uri:
            raise ExecutionClaimError("phase provider plans must be distinct")
        for name in (
            "oci_index_digest",
            "oci_platform_manifest_digest",
            "analysis_oci_platform_manifest_digest",
            "release_oci_index_digest",
            "release_oci_platform_manifest_digest",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.startswith("sha256:"):
                raise ExecutionClaimError(f"{name} must be an OCI SHA-256 digest")
            _digest(name, value.removeprefix("sha256:"))
        if (
            self.release_oci_index_digest == self.oci_index_digest
            or self.release_oci_platform_manifest_digest
            in {self.oci_platform_manifest_digest, self.analysis_oci_platform_manifest_digest}
        ):
            raise ExecutionClaimError("release and scientific OCI identities must be distinct")
        rows = _fixed_rows("claim corpora", self.corpora, ClaimCorpusBinding)
        expected_output = _sha256(
            _canonical_bytes(
                {
                    "corpora": [
                        {
                            "corpus_id": row.corpus_id,
                            "canonical_namespace_uri": row.canonical_namespace_uri,
                            "staging_namespace_uri": row.staging_namespace_uri,
                        }
                        for row in rows
                    ],
                    "derivation": OUTPUT_AGGREGATE_DERIVATION,
                    "manifest_sha256": self.manifest_sha256,
                }
            )
        )
        if self.output_aggregate_identity != expected_output:
            raise ExecutionClaimError("output aggregate identity is not C1 namespace-derived")
        if not isinstance(self.beacon, ExecutionBeaconContract):
            raise ExecutionClaimError("execution beacon must be typed")
        if self.schema_version != EXECUTION_CLAIM_CONTRACT_SCHEMA:
            raise ExecutionClaimError("execution claim contract schema differs")
        object.__setattr__(self, "corpora", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"beacon", "corpora", "host_tools"}
            },
            "beacon": self.beacon.to_dict(),
            "corpora": [row.to_dict() for row in self.corpora],
            "host_tools": self.host_tools.to_dict(),
        }

    @property
    def contract_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> ExecutionClaimContract:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="execution claim contract")
        raw_corpora = row["corpora"]
        if type(raw_corpora) is not list:
            raise ExecutionClaimError("claim corpora must be an array")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key not in {"beacon", "corpora", "host_tools"}
            },
            beacon=ExecutionBeaconContract.from_dict(row["beacon"]),
            host_tools=PhaseHostToolContract.from_dict(row["host_tools"]),
            corpora=tuple(ClaimCorpusBinding.from_dict(item) for item in raw_corpora),
        )


@dataclass(frozen=True)
class PhaseCorpusBinding:
    """One C1-fixed input and output namespace for a post-online phase."""

    corpus_id: str
    input_uri: str
    input_sha256: str
    supporting_input_uri: str
    supporting_input_sha256: str
    output_uri: str

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ExecutionClaimError("phase corpus is outside the fixed suite")
        for name in ("input_uri", "supporting_input_uri", "output_uri"):
            parsed = urlsplit(_text(name, getattr(self, name)))
            if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
                raise ExecutionClaimError(f"{name} must be a canonical file URI")
        if self.input_uri == self.output_uri:
            raise ExecutionClaimError("phase input and output namespaces must differ")
        _digest("input_sha256", self.input_sha256)
        _digest("supporting_input_sha256", self.supporting_input_sha256)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> PhaseCorpusBinding:
        return cls(
            **_closed(value, frozenset(cls.__dataclass_fields__), label="phase corpus binding")
        )


@dataclass(frozen=True)
class PhaseClaimContract:
    """Closed provider claim for label release or analysis."""

    phase: Literal["label-release", "analysis"]
    repository: str
    claim_workflow_path: str
    claim_workflow_ref: str
    claim_workflow_sha: str
    run_head_branch: str | None
    claim_job_name: str
    execute_job_name: str
    claim_nonce: str
    unique_runner_label: str
    runner_id: int
    runner_name: str
    runner_group_id: int | None
    runner_version: str
    runner_archive_sha256: str
    provider_operating_system: str
    provider_architecture: str
    host_tool_contract_sha256: str
    runtime_probe_receipt_sha256: str
    c1_commit: str
    manifest_sha256: str
    c1_provider_plan_uri: str
    c1_provider_plan_sha256: str
    run_receipt_sha256: str
    oci_index_digest: str
    oci_platform_manifest_digest: str
    tle_binary_sha256: str | None
    online_execution_claim_contract_sha256: str
    predecessor_state_sha256: str
    predecessor_ledger_commit: str
    corpora: tuple[PhaseCorpusBinding, ...]
    phase_input_aggregate_sha256: str
    phase_output_identity: str
    maximum_runtime_seconds: int
    label_release_beacon: ExecutionBeaconContract | None
    schema_version: str = PHASE_CLAIM_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in {LABEL_RELEASE_PHASE, ANALYSIS_PHASE}:
            raise ExecutionClaimError("post-online claim phase is not registered")
        if _REPOSITORY.fullmatch(self.repository) is None:
            raise ExecutionClaimError("phase claim repository is invalid")
        expected_workflow = {
            LABEL_RELEASE_PHASE: ".github/workflows/confirmatory-label-release.yml",
            ANALYSIS_PHASE: ".github/workflows/confirmatory-analysis.yml",
        }[self.phase]
        if self.claim_workflow_path != expected_workflow:
            raise ExecutionClaimError("phase claim workflow path differs")
        expected_ref = f"{self.repository}/{expected_workflow}@refs/tags/confirmatory-apparatus-c0"
        if self.claim_workflow_ref != expected_ref:
            raise ExecutionClaimError("phase claim workflow ref differs from C0")
        _git_commit("claim_workflow_sha", self.claim_workflow_sha)
        _api_head_branch("run_head_branch", self.run_head_branch)
        expected_jobs = PHASE_JOB_NAMES[self.phase]
        if (self.claim_job_name, self.execute_job_name) != expected_jobs:
            raise ExecutionClaimError("phase provider job names differ")
        _digest("claim_nonce", self.claim_nonce)
        if self.unique_runner_label != derive_phase_runner_label(self.claim_nonce, self.phase):
            raise ExecutionClaimError("phase runner label is not claim-nonce-derived")
        _positive("runner_id", self.runner_id)
        _runner_group(self.runner_group_id)
        for name in (
            "runner_name",
            "runner_version",
            "provider_operating_system",
            "provider_architecture",
        ):
            _text(name, getattr(self, name))
        if self.runner_version != OFFICIAL_ACTIONS_RUNNER_VERSION:
            raise ExecutionClaimError("phase runner version differs")
        if self.runner_archive_sha256 != OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256:
            raise ExecutionClaimError("phase runner archive differs")
        if self.provider_operating_system != "macOS" or self.provider_architecture != "ARM64":
            raise ExecutionClaimError("phase provider host differs")
        for name in (
            "runner_archive_sha256",
            "host_tool_contract_sha256",
            "runtime_probe_receipt_sha256",
            "manifest_sha256",
            "c1_provider_plan_sha256",
            "run_receipt_sha256",
            "online_execution_claim_contract_sha256",
            "predecessor_state_sha256",
            "phase_input_aggregate_sha256",
            "phase_output_identity",
        ):
            _digest(name, getattr(self, name))
        for name in ("oci_index_digest", "oci_platform_manifest_digest"):
            value = getattr(self, name)
            if type(value) is not str or not value.startswith("sha256:"):
                raise ExecutionClaimError(f"phase {name} must be an OCI SHA-256 digest")
            _digest(name, value.removeprefix("sha256:"))
        parsed_plan = urlsplit(_text("c1_provider_plan_uri", self.c1_provider_plan_uri))
        if (
            parsed_plan.scheme != "file"
            or parsed_plan.netloc
            or parsed_plan.query
            or parsed_plan.fragment
        ):
            raise ExecutionClaimError("c1_provider_plan_uri must be a canonical file URI")
        for name in ("c1_commit", "predecessor_ledger_commit"):
            _git_commit(name, getattr(self, name))
        rows = _fixed_rows("phase claim corpora", self.corpora, PhaseCorpusBinding)
        expected_input = _sha256(
            _canonical_bytes(
                {
                    "corpora": [
                        {
                            "corpus_id": row.corpus_id,
                            "input_sha256": row.input_sha256,
                            "input_uri": row.input_uri,
                            "supporting_input_sha256": row.supporting_input_sha256,
                            "supporting_input_uri": row.supporting_input_uri,
                        }
                        for row in rows
                    ],
                    "manifest_sha256": self.manifest_sha256,
                    "phase": self.phase,
                    "predecessor_state_sha256": self.predecessor_state_sha256,
                }
            )
        )
        expected_output = _sha256(
            _canonical_bytes(
                {
                    "corpora": [
                        {"corpus_id": row.corpus_id, "output_uri": row.output_uri} for row in rows
                    ],
                    "manifest_sha256": self.manifest_sha256,
                    "phase": self.phase,
                }
            )
        )
        if self.phase_input_aggregate_sha256 != expected_input:
            raise ExecutionClaimError("phase input aggregate is not canonical")
        if self.phase_output_identity != expected_output:
            raise ExecutionClaimError("phase output identity is not canonical")
        maximum = _positive("maximum_runtime_seconds", self.maximum_runtime_seconds)
        phase_ceiling = {
            LABEL_RELEASE_PHASE: 6 * 60 * 60,
            ANALYSIS_PHASE: 12 * 60 * 60,
        }[self.phase]
        if maximum > phase_ceiling:
            raise ExecutionClaimError("phase runtime exceeds its fixed provider job timeout")
        if self.phase == LABEL_RELEASE_PHASE:
            if not isinstance(self.label_release_beacon, ExecutionBeaconContract):
                raise ExecutionClaimError("label release claim lacks its frozen beacon")
            if self.tle_binary_sha256 is None:
                raise ExecutionClaimError("label release claim lacks its C1 tle byte pin")
            _digest("tle_binary_sha256", self.tle_binary_sha256)
        elif self.label_release_beacon is not None:
            raise ExecutionClaimError("analysis claim cannot introduce a beacon")
        elif self.tle_binary_sha256 is not None:
            raise ExecutionClaimError("analysis claim cannot authorize tle")
        if self.schema_version != PHASE_CLAIM_CONTRACT_SCHEMA:
            raise ExecutionClaimError("phase claim contract schema differs")
        object.__setattr__(self, "corpora", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"corpora", "label_release_beacon"}
            },
            "corpora": [row.to_dict() for row in self.corpora],
            "label_release_beacon": (
                None if self.label_release_beacon is None else self.label_release_beacon.to_dict()
            ),
        }

    @property
    def contract_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> PhaseClaimContract:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="phase claim contract")
        corpora = row["corpora"]
        if type(corpora) is not list:
            raise ExecutionClaimError("phase claim corpora must be an array")
        raw_beacon = row["label_release_beacon"]
        if raw_beacon is not None and not isinstance(raw_beacon, Mapping):
            raise ExecutionClaimError("phase label-release beacon is malformed")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key not in {"corpora", "label_release_beacon"}
            },
            corpora=tuple(PhaseCorpusBinding.from_dict(item) for item in corpora),
            label_release_beacon=(
                None if raw_beacon is None else ExecutionBeaconContract.from_dict(raw_beacon)
            ),
        )


@dataclass(frozen=True)
class ProviderExecutionIdentity:
    """Actual GitHub run and exact job selected by provider evidence."""

    repository: str
    workflow_path: str
    workflow_ref: str
    workflow_sha: str
    run_head_branch: str | None
    run_id: int
    run_attempt: int
    claim_job_id: int
    claim_job_name: str
    execute_job_name: str
    runner_id: int
    runner_name: str
    runner_group_id: int | None
    runner_label: str
    runner_version: str
    runner_archive_sha256: str
    provider_operating_system: str
    provider_architecture: str
    host_tool_contract_sha256: str
    runtime_probe_receipt_sha256: str
    self_hosted: bool
    schema_version: str = PROVIDER_EXECUTION_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if _REPOSITORY.fullmatch(self.repository) is None:
            raise ExecutionClaimError("provider repository is invalid")
        if _WORKFLOW.fullmatch(self.workflow_path) is None:
            raise ExecutionClaimError("provider workflow path is invalid")
        _text("workflow_ref", self.workflow_ref)
        _git_commit("workflow_sha", self.workflow_sha)
        _api_head_branch("run_head_branch", self.run_head_branch)
        _positive("run_id", self.run_id)
        if self.run_attempt != 1:
            raise ExecutionClaimError("provider run_attempt must equal 1")
        for name in ("claim_job_id", "runner_id"):
            _positive(name, getattr(self, name))
        _runner_group(self.runner_group_id)
        for name in (
            "claim_job_name",
            "execute_job_name",
            "runner_name",
            "runner_version",
            "provider_operating_system",
            "provider_architecture",
        ):
            _text(name, getattr(self, name))
        for name in (
            "runner_archive_sha256",
            "host_tool_contract_sha256",
            "runtime_probe_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if _RUNNER_LABEL.fullmatch(self.runner_label) is None:
            raise ExecutionClaimError("provider runner label is invalid")
        if self.self_hosted is not True:
            raise ExecutionClaimError("online execute job must be self-hosted")
        if self.schema_version != PROVIDER_EXECUTION_IDENTITY_SCHEMA:
            raise ExecutionClaimError("provider execution identity schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def identity_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    def matches_contract(self, contract: ExecutionClaimContract) -> None:
        exact = {
            "repository": contract.repository,
            "workflow_path": contract.claim_workflow_path,
            "workflow_ref": contract.claim_workflow_ref,
            "workflow_sha": contract.claim_workflow_sha,
            "run_head_branch": contract.run_head_branch,
            "claim_job_name": contract.claim_job_name,
            "execute_job_name": contract.execute_job_name,
            "runner_label": contract.unique_runner_label,
            "runner_id": contract.runner_id,
            "runner_name": contract.runner_name,
            "runner_group_id": contract.runner_group_id,
            "runner_version": contract.runner_version,
            "runner_archive_sha256": contract.runner_archive_sha256,
            "provider_operating_system": contract.provider_operating_system,
            "provider_architecture": contract.provider_architecture,
            "host_tool_contract_sha256": contract.host_tools.contract_sha256,
            "runtime_probe_receipt_sha256": contract.runtime_probe_receipt_sha256,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ExecutionClaimError(f"provider {name} differs from the C1 contract")

    def matches_phase_contract(self, contract: PhaseClaimContract) -> None:
        if not isinstance(contract, PhaseClaimContract):
            raise ExecutionClaimError("provider phase contract must be typed")
        exact = {
            "repository": contract.repository,
            "workflow_path": contract.claim_workflow_path,
            "workflow_ref": contract.claim_workflow_ref,
            "workflow_sha": contract.claim_workflow_sha,
            "run_head_branch": contract.run_head_branch,
            "claim_job_name": contract.claim_job_name,
            "execute_job_name": contract.execute_job_name,
            "runner_label": contract.unique_runner_label,
            "runner_id": contract.runner_id,
            "runner_name": contract.runner_name,
            "runner_group_id": contract.runner_group_id,
            "runner_version": contract.runner_version,
            "runner_archive_sha256": contract.runner_archive_sha256,
            "provider_operating_system": contract.provider_operating_system,
            "provider_architecture": contract.provider_architecture,
            "host_tool_contract_sha256": contract.host_tool_contract_sha256,
            "runtime_probe_receipt_sha256": contract.runtime_probe_receipt_sha256,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ExecutionClaimError(f"provider {name} differs from the phase contract")

    @classmethod
    def from_dict(cls, value: object) -> ProviderExecutionIdentity:
        return cls(
            **_closed(
                value, frozenset(cls.__dataclass_fields__), label="provider execution identity"
            )
        )


class GitHubReadApi(Protocol):
    def get(self, endpoint: str) -> object: ...


@dataclass(frozen=True)
class ProviderRunnerReadinessReceipt:
    provider_plan_sha256: str
    bootstrap_receipt_file_sha256: str
    runner_id: int
    runner_name: str
    runner_group_id: int | None
    status: str
    busy: bool
    labels: tuple[str, ...]
    verified_at_utc: str
    schema_version: str = PROVIDER_RUNNER_READINESS_SCHEMA

    def __post_init__(self) -> None:
        for name in ("provider_plan_sha256", "bootstrap_receipt_file_sha256"):
            _digest(name, getattr(self, name))
        _positive("runner_id", self.runner_id)
        _text("runner_name", self.runner_name)
        _runner_group(self.runner_group_id)
        if self.status != "offline" or self.busy is not False:
            raise ExecutionClaimError("provider runner was not offline and idle before CAS")
        labels = tuple(self.labels)
        if labels != tuple(sorted(labels, key=lambda value: value.encode("utf-8"))) or len(
            labels
        ) != len(set(labels)):
            raise ExecutionClaimError("provider runner labels are not unique byte-sorted strings")
        _timestamp("verified_at_utc", self.verified_at_utc)
        if self.schema_version != PROVIDER_RUNNER_READINESS_SCHEMA:
            raise ExecutionClaimError("provider runner readiness schema differs")
        object.__setattr__(self, "labels", labels)

    def to_dict(self) -> dict[str, object]:
        return {
            **{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "labels"},
            "labels": list(self.labels),
        }

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))


def verify_provider_runner_ready(
    *,
    plan: ProviderPhasePlan,
    api: GitHubReadApi,
    verified_at_utc: str,
) -> ProviderRunnerReadinessReceipt:
    """Recheck the one C1 runner immediately before provider claim CAS."""

    if not isinstance(plan, ProviderPhasePlan):
        raise ExecutionClaimError("runner readiness requires a typed provider plan")
    try:
        response = api.get(f"repos/{plan.repository}/actions/runners?per_page=100")
    except Exception as exc:
        raise ExecutionClaimError("cannot read the live repository runner inventory") from exc
    if not isinstance(response, Mapping) or not isinstance(response.get("runners"), list):
        raise ExecutionClaimError("repository runner API response is malformed")
    inventory = response["runners"]
    total_count = response.get("total_count")
    if (
        type(total_count) is not int
        or total_count != len(inventory)
        or total_count > 100
        or any(not isinstance(row, Mapping) for row in inventory)
    ):
        raise ExecutionClaimError("repository runner inventory is incomplete or malformed")
    matches = [
        row for row in inventory if isinstance(row, Mapping) and row.get("id") == plan.runner_id
    ]
    if len(matches) != 1:
        raise ExecutionClaimError("C1 runner is not a singleton in the complete inventory")
    runner = matches[0]
    raw_labels = runner.get("labels")
    if not isinstance(raw_labels, list):
        raise ExecutionClaimError("C1 runner labels are malformed")
    label_names = tuple(
        sorted(
            (
                row.get("name")
                for row in raw_labels
                if isinstance(row, Mapping) and type(row.get("name")) is str
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    expected_labels = tuple(
        sorted(
            required_execute_runner_labels(derive_phase_runner_label(plan.claim_nonce, plan.phase)),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if (
        len(label_names) != len(raw_labels)
        or label_names != expected_labels
        or runner.get("name") != plan.runner_name
        or runner.get("os") != "macOS"
        or runner.get("status") != "offline"
        or runner.get("busy") is not False
        or plan.runner_group_id is not None
    ):
        raise ExecutionClaimError("live repository runner differs from the C1 idle singleton")
    phase_label = derive_phase_runner_label(plan.claim_nonce, plan.phase)
    phase_label_rows = 0
    for inventory_row in inventory:
        inventory_labels = inventory_row.get("labels")
        if not isinstance(inventory_labels, list) or any(
            not isinstance(label, Mapping) or type(label.get("name")) is not str
            for label in inventory_labels
        ):
            raise ExecutionClaimError("repository runner inventory labels are malformed")
        if phase_label in {str(label["name"]) for label in inventory_labels}:
            phase_label_rows += 1
    if phase_label_rows != 1:
        raise ExecutionClaimError("C1 phase runner label is not unique in the complete inventory")
    return ProviderRunnerReadinessReceipt(
        provider_plan_sha256=plan.plan_sha256,
        bootstrap_receipt_file_sha256=plan.runner_bootstrap_receipt_file_sha256,
        runner_id=plan.runner_id,
        runner_name=plan.runner_name,
        runner_group_id=plan.runner_group_id,
        status="offline",
        busy=False,
        labels=label_names,
        verified_at_utc=_timestamp("verified_at_utc", verified_at_utc).isoformat(),
    )


@dataclass(frozen=True)
class LiveExecuteJobReceipt:
    provider_identity_sha256: str
    repository: str
    workflow_path: str
    workflow_sha: str
    run_head_branch: str | None
    run_id: int
    run_attempt: int
    execute_job_id: int
    execute_job_name: str
    runner_id: int
    runner_name: str
    runner_group_id: int | None
    runner_labels: tuple[str, ...]
    verified_at_utc: str
    schema_version: str = LIVE_EXECUTE_JOB_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _digest("provider_identity_sha256", self.provider_identity_sha256)
        if _REPOSITORY.fullmatch(self.repository) is None:
            raise ExecutionClaimError("live job repository is invalid")
        if _WORKFLOW.fullmatch(self.workflow_path) is None:
            raise ExecutionClaimError("live job workflow path is invalid")
        _git_commit("workflow_sha", self.workflow_sha)
        _api_head_branch("run_head_branch", self.run_head_branch)
        for name in (
            "run_id",
            "run_attempt",
            "execute_job_id",
            "runner_id",
        ):
            _positive(name, getattr(self, name))
        _runner_group(self.runner_group_id)
        for name in ("execute_job_name", "runner_name"):
            _text(name, getattr(self, name))
        labels = tuple(self.runner_labels)
        if (
            not labels
            or labels != tuple(sorted(labels, key=lambda item: item.encode("utf-8")))
            or len(labels) != len(set(labels))
            or not all(type(label) is str and label for label in labels)
        ):
            raise ExecutionClaimError("live runner labels must be unique byte-sorted strings")
        _timestamp("verified_at_utc", self.verified_at_utc)
        if self.schema_version != LIVE_EXECUTE_JOB_RECEIPT_SCHEMA:
            raise ExecutionClaimError("live execute-job receipt schema differs")
        object.__setattr__(self, "runner_labels", labels)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "runner_labels"
            },
            "runner_labels": list(self.runner_labels),
        }

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))


def _api_object(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ExecutionClaimError(f"{label} must be a GitHub API object")
    return value


def _api_positive(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ExecutionClaimError(f"{label} must be a positive GitHub integer")
    return value


def verify_live_execute_job(
    *,
    api: GitHubReadApi,
    contract: ExecutionClaimContract | PhaseClaimContract,
    provider_identity: ProviderExecutionIdentity,
    verified_at_utc: str,
) -> LiveExecuteJobReceipt:
    """Read back the actual in-progress execute job before any phase input opens."""

    if isinstance(contract, ExecutionClaimContract):
        provider_identity.matches_contract(contract)
        workflow_path = contract.claim_workflow_path
        workflow_sha = contract.claim_workflow_sha
        run_head_branch = contract.run_head_branch
        runner_label = contract.unique_runner_label
        execute_job_name = contract.execute_job_name
    elif isinstance(contract, PhaseClaimContract):
        provider_identity.matches_phase_contract(contract)
        workflow_path = contract.claim_workflow_path
        workflow_sha = contract.claim_workflow_sha
        run_head_branch = contract.run_head_branch
        runner_label = contract.unique_runner_label
        execute_job_name = contract.execute_job_name
    else:
        raise ExecutionClaimError("live job verification requires a typed claim contract")
    run = _api_object(
        api.get(
            f"repos/{provider_identity.repository}/actions/runs/"
            f"{provider_identity.run_id}/attempts/{provider_identity.run_attempt}"
        ),
        label="GitHub workflow run",
    )
    repository = _api_object(run.get("repository"), label="GitHub workflow repository")
    run_exact = {
        "id": provider_identity.run_id,
        "run_attempt": provider_identity.run_attempt,
        "event": "workflow_dispatch",
        "status": "in_progress",
        "conclusion": None,
        "head_sha": workflow_sha,
        "head_branch": run_head_branch,
        "path": workflow_path,
    }
    for name, expected in run_exact.items():
        if run.get(name) != expected:
            raise ExecutionClaimError(f"live GitHub run {name} differs from the claim")
    if repository.get("full_name") != provider_identity.repository:
        raise ExecutionClaimError("live GitHub run repository differs from the claim")
    jobs_response = _api_object(
        api.get(
            f"repos/{provider_identity.repository}/actions/runs/"
            f"{provider_identity.run_id}/attempts/{provider_identity.run_attempt}/jobs?per_page=100"
        ),
        label="GitHub workflow jobs",
    )
    jobs = jobs_response.get("jobs")
    if not isinstance(jobs, list):
        raise ExecutionClaimError("GitHub workflow jobs response lacks an array")
    matches = [
        _api_object(row, label="GitHub workflow job")
        for row in jobs
        if isinstance(row, Mapping) and row.get("name") == execute_job_name
    ]
    if len(matches) != 1:
        raise ExecutionClaimError("live GitHub execute job is not a singleton")
    job = matches[0]
    labels_raw = job.get("labels")
    if not isinstance(labels_raw, list) or not all(type(item) is str for item in labels_raw):
        raise ExecutionClaimError("live GitHub execute job labels are malformed")
    labels = tuple(sorted(labels_raw, key=lambda item: item.encode("utf-8")))
    required_labels = {"self-hosted", "macOS", "ARM64", runner_label}
    if not required_labels.issubset(labels):
        raise ExecutionClaimError("live GitHub execute job lacks the registered runner labels")
    claim_labels = [label for label in labels if label.startswith("fractal-ann-confirmatory-")]
    if claim_labels != [runner_label]:
        raise ExecutionClaimError("live GitHub execute job has another confirmatory label")
    job_exact = {
        "name": execute_job_name,
        "status": "in_progress",
        "conclusion": None,
        "run_id": provider_identity.run_id,
        "run_attempt": provider_identity.run_attempt,
        "runner_id": provider_identity.runner_id,
        "runner_name": provider_identity.runner_name,
        "runner_group_id": provider_identity.runner_group_id,
    }
    for name, expected in job_exact.items():
        if job.get(name) != expected:
            raise ExecutionClaimError(f"live GitHub execute job {name} differs from the claim")
    return LiveExecuteJobReceipt(
        provider_identity_sha256=provider_identity.identity_sha256,
        repository=provider_identity.repository,
        workflow_path=workflow_path,
        workflow_sha=workflow_sha,
        run_head_branch=run_head_branch,
        run_id=provider_identity.run_id,
        run_attempt=provider_identity.run_attempt,
        execute_job_id=_api_positive(job.get("id"), label="execute job ID"),
        execute_job_name=execute_job_name,
        runner_id=provider_identity.runner_id,
        runner_name=provider_identity.runner_name,
        runner_group_id=provider_identity.runner_group_id,
        runner_labels=labels,
        verified_at_utc=_timestamp("verified_at_utc", verified_at_utc).isoformat(),
    )


@dataclass(frozen=True)
class FailedExecuteJobReceipt:
    provider_identity_sha256: str
    repository: str
    workflow_path: str
    workflow_sha: str
    run_head_branch: str | None
    run_id: int
    run_attempt: int
    execute_job_id: int
    execute_job_name: str
    conclusion: str
    runner_assigned: bool
    runner_id: int | None
    runner_name: str | None
    runner_group_id: int | None
    runner_labels: tuple[str, ...]
    verified_at_utc: str
    schema_version: str = FAILED_EXECUTE_JOB_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _digest("provider_identity_sha256", self.provider_identity_sha256)
        if _REPOSITORY.fullmatch(self.repository) is None:
            raise ExecutionClaimError("failed job repository is invalid")
        if _WORKFLOW.fullmatch(self.workflow_path) is None:
            raise ExecutionClaimError("failed job workflow path is invalid")
        _git_commit("workflow_sha", self.workflow_sha)
        _api_head_branch("run_head_branch", self.run_head_branch)
        for name in ("run_id", "run_attempt", "execute_job_id"):
            _positive(name, getattr(self, name))
        _text("execute_job_name", self.execute_job_name)
        if self.conclusion not in {
            "action_required",
            "cancelled",
            "failure",
            "stale",
            "startup_failure",
            "timed_out",
        }:
            raise ExecutionClaimError("execute job conclusion is not a terminal failure")
        if type(self.runner_assigned) is not bool:
            raise ExecutionClaimError("runner_assigned must be boolean")
        if self.runner_assigned:
            _positive("runner_id", self.runner_id)
            _text("runner_name", self.runner_name)
            _runner_group(self.runner_group_id)
        elif (
            self.runner_id not in {None, 0}
            or self.runner_name not in {None, ""}
            or self.runner_group_id not in {None, 0}
        ):
            raise ExecutionClaimError("unassigned failed job cannot claim a runner identity")
        labels = tuple(self.runner_labels)
        if (
            not labels
            or not all(type(label) is str and label for label in labels)
            or labels != tuple(sorted(labels, key=lambda item: item.encode("utf-8")))
            or len(labels) != len(set(labels))
        ):
            raise ExecutionClaimError("failed job labels must be unique byte-sorted strings")
        _timestamp("verified_at_utc", self.verified_at_utc)
        if self.schema_version != FAILED_EXECUTE_JOB_RECEIPT_SCHEMA:
            raise ExecutionClaimError("failed execute-job receipt schema differs")
        object.__setattr__(self, "runner_labels", labels)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "runner_labels"
            },
            "runner_labels": list(self.runner_labels),
        }

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))


def verify_failed_execute_job(
    *,
    api: GitHubReadApi,
    contract: ExecutionClaimContract | PhaseClaimContract,
    provider_identity: ProviderExecutionIdentity,
    verified_at_utc: str,
) -> FailedExecuteJobReceipt:
    """Prove a non-success execute job before publishing terminal FAILED."""

    if isinstance(contract, ExecutionClaimContract):
        provider_identity.matches_contract(contract)
        runner_label = contract.unique_runner_label
    elif isinstance(contract, PhaseClaimContract):
        provider_identity.matches_phase_contract(contract)
        runner_label = contract.unique_runner_label
    else:
        raise ExecutionClaimError("failed job verification requires a typed claim contract")
    run = _api_object(
        api.get(
            f"repos/{provider_identity.repository}/actions/runs/"
            f"{provider_identity.run_id}/attempts/{provider_identity.run_attempt}"
        ),
        label="failed GitHub workflow run",
    )
    repository = _api_object(run.get("repository"), label="failed workflow repository")
    exact_run = {
        "id": provider_identity.run_id,
        "run_attempt": provider_identity.run_attempt,
        "event": "workflow_dispatch",
        "head_sha": provider_identity.workflow_sha,
        "head_branch": provider_identity.run_head_branch,
        "path": provider_identity.workflow_path,
    }
    for name, expected in exact_run.items():
        if run.get(name) != expected:
            raise ExecutionClaimError(f"failed GitHub run {name} differs from the claim")
    if repository.get("full_name") != provider_identity.repository:
        raise ExecutionClaimError("failed GitHub run repository differs from the claim")
    if run.get("status") not in {"in_progress", "completed"}:
        raise ExecutionClaimError("failed GitHub run has an inadmissible status")
    if run.get("status") == "completed" and run.get("conclusion") in {
        None,
        "neutral",
        "skipped",
        "success",
    }:
        raise ExecutionClaimError("completed GitHub run does not prove failure")
    response = _api_object(
        api.get(
            f"repos/{provider_identity.repository}/actions/runs/"
            f"{provider_identity.run_id}/attempts/{provider_identity.run_attempt}/jobs?per_page=100"
        ),
        label="failed GitHub workflow jobs",
    )
    jobs = response.get("jobs")
    if not isinstance(jobs, list):
        raise ExecutionClaimError("failed workflow jobs response lacks an array")
    matches = [
        _api_object(row, label="failed GitHub workflow job")
        for row in jobs
        if isinstance(row, Mapping) and row.get("name") == provider_identity.execute_job_name
    ]
    if len(matches) != 1:
        raise ExecutionClaimError("failed GitHub execute job is not a singleton")
    job = matches[0]
    if job.get("status") != "completed":
        raise ExecutionClaimError("execute job is not terminal")
    conclusion = _text("execute job conclusion", job.get("conclusion"))
    labels_raw = job.get("labels")
    if not isinstance(labels_raw, list) or not all(type(item) is str for item in labels_raw):
        raise ExecutionClaimError("failed execute job labels are malformed")
    labels = tuple(sorted(labels_raw, key=lambda item: item.encode("utf-8")))
    if not {"self-hosted", "macOS", "ARM64", runner_label}.issubset(labels):
        raise ExecutionClaimError("failed execute job lacks registered runner labels")
    claim_labels = [label for label in labels if label.startswith("fractal-ann-confirmatory-")]
    if claim_labels != [runner_label]:
        raise ExecutionClaimError("failed execute job has another confirmatory label")
    if (
        job.get("run_id") != provider_identity.run_id
        or job.get("run_attempt") != provider_identity.run_attempt
    ):
        raise ExecutionClaimError("failed execute job belongs to another run attempt")
    job_id = _api_positive(job.get("id"), label="failed execute job ID")
    raw_runner_id = job.get("runner_id")
    assigned = type(raw_runner_id) is int and raw_runner_id > 0
    if assigned and (
        raw_runner_id != provider_identity.runner_id
        or job.get("runner_name") != provider_identity.runner_name
        or job.get("runner_group_id") != provider_identity.runner_group_id
    ):
        raise ExecutionClaimError("failed execute job runner identity differs from the claim")
    return FailedExecuteJobReceipt(
        provider_identity_sha256=provider_identity.identity_sha256,
        repository=provider_identity.repository,
        workflow_path=provider_identity.workflow_path,
        workflow_sha=provider_identity.workflow_sha,
        run_head_branch=provider_identity.run_head_branch,
        run_id=provider_identity.run_id,
        run_attempt=provider_identity.run_attempt,
        execute_job_id=job_id,
        execute_job_name=provider_identity.execute_job_name,
        conclusion=conclusion,
        runner_assigned=assigned,
        runner_id=raw_runner_id if type(raw_runner_id) is int else None,
        runner_name=job.get("runner_name") if type(job.get("runner_name")) is str else None,
        runner_group_id=(
            job.get("runner_group_id") if type(job.get("runner_group_id")) is int else None
        ),
        runner_labels=labels,
        verified_at_utc=_timestamp("verified_at_utc", verified_at_utc).isoformat(),
    )


@dataclass(frozen=True)
class VerifiedBeaconClaims:
    chain_hash: str
    round: int
    beacon_bytes_sha256: str
    randomness: str
    signature: str
    scheme_id: str
    public_key: str
    signature_verified: bool

    def __post_init__(self) -> None:
        _digest("chain_hash", self.chain_hash)
        _positive("round", self.round)
        _digest("beacon_bytes_sha256", self.beacon_bytes_sha256)
        for name in ("randomness", "signature", "public_key"):
            value = getattr(self, name)
            if type(value) is not str or _HEX.fullmatch(value) is None:
                raise ExecutionClaimError(f"{name} must be lowercase hexadecimal bytes")
        _text("scheme_id", self.scheme_id)
        if self.signature_verified is not True:
            raise ExecutionClaimError("drand signature was not verified")


class ExecutionBeaconVerifier(Protocol):
    def verify(
        self,
        *,
        contract: ExecutionBeaconContract,
        beacon_bytes: bytes,
    ) -> VerifiedBeaconClaims: ...


@dataclass(frozen=True)
class ExecutionBeaconReceipt:
    claim_state_sha256: str
    claim_ledger_commit: str
    provider_identity_sha256: str
    beacon_contract_sha256: str
    design_seed_sha256: str
    beacon_bytes_sha256: str
    chain_hash: str
    round: int
    randomness: str
    signature: str
    published_at_utc: str
    verified_at_utc: str
    derived_seed_sha256: str
    permutation_seed: int
    schema_version: str = EXECUTION_BEACON_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "claim_state_sha256",
            "provider_identity_sha256",
            "beacon_contract_sha256",
            "design_seed_sha256",
            "beacon_bytes_sha256",
            "chain_hash",
            "derived_seed_sha256",
        ):
            _digest(name, getattr(self, name))
        _git_commit("claim_ledger_commit", self.claim_ledger_commit)
        _positive("round", self.round)
        for name in ("randomness", "signature"):
            value = getattr(self, name)
            if type(value) is not str or _HEX.fullmatch(value) is None:
                raise ExecutionClaimError(f"{name} must be lowercase hexadecimal bytes")
        published = _timestamp("published_at_utc", self.published_at_utc)
        verified = _timestamp("verified_at_utc", self.verified_at_utc)
        if verified < published:
            raise ExecutionClaimError("execution beacon was allegedly verified before publication")
        if type(self.permutation_seed) is not int or not 0 <= self.permutation_seed < 2**64:
            raise ExecutionClaimError("permutation_seed must be an unsigned 64-bit integer")
        if self.permutation_seed != int.from_bytes(
            bytes.fromhex(self.derived_seed_sha256)[:8], "big"
        ):
            raise ExecutionClaimError("permutation seed differs from its derived seed")
        if self.schema_version != EXECUTION_BEACON_RECEIPT_SCHEMA:
            raise ExecutionClaimError("execution beacon receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> ExecutionBeaconReceipt:
        return cls(
            **_closed(value, frozenset(cls.__dataclass_fields__), label="execution beacon receipt")
        )


def verify_execution_beacon(
    contract: ExecutionBeaconContract,
    *,
    beacon_bytes: bytes,
    claim_state_sha256: str,
    claim_ledger_commit: str,
    provider_identity: ProviderExecutionIdentity,
    claim_attested_at_utc: str,
    verifier: ExecutionBeaconVerifier,
    verified_at_utc: str,
    design_seed_sha256: str,
) -> ExecutionBeaconReceipt:
    """Verify exact drand bytes after the claim and derive the sole runtime seed."""

    if not isinstance(contract, ExecutionBeaconContract):
        raise ExecutionClaimError("execution beacon contract must be typed")
    _digest("claim_state_sha256", claim_state_sha256)
    _digest("design_seed_sha256", design_seed_sha256)
    _git_commit("claim_ledger_commit", claim_ledger_commit)
    if not isinstance(provider_identity, ProviderExecutionIdentity):
        raise ExecutionClaimError("provider identity must be typed")
    claimed = _timestamp("claim_attested_at_utc", claim_attested_at_utc)
    if claimed >= contract.execution_publication_time:
        raise ExecutionClaimError("RUN_CLAIMED attestation must precede beacon publication")
    if not beacon_bytes or len(beacon_bytes) > 1024 * 1024:
        raise ExecutionClaimError("execution beacon bytes are empty or exceed the bound")
    try:
        claims = verifier.verify(contract=contract, beacon_bytes=beacon_bytes)
    except ExecutionClaimError:
        raise
    except Exception as exc:
        raise ExecutionClaimError("execution beacon verifier rejected the bytes") from exc
    if not isinstance(claims, VerifiedBeaconClaims):
        raise ExecutionClaimError("execution beacon verifier returned untyped claims")
    exact = {
        "chain_hash": contract.chain_hash,
        "round": contract.execution_round,
        "beacon_bytes_sha256": _sha256(beacon_bytes),
        "scheme_id": contract.chain_scheme_id,
        "public_key": contract.chain_public_key,
    }
    for name, expected in exact.items():
        if getattr(claims, name) != expected:
            raise ExecutionClaimError(f"verified beacon {name} differs from the contract")
    seed = _sha256(
        b"fractal-execution-seed-v1\0"
        + bytes.fromhex(claim_state_sha256)
        + claim_ledger_commit.encode("ascii")
        + bytes.fromhex(provider_identity.identity_sha256)
        + bytes.fromhex(contract.contract_sha256)
        + bytes.fromhex(design_seed_sha256)
        + beacon_bytes
        + bytes.fromhex(claims.randomness)
    )
    return ExecutionBeaconReceipt(
        claim_state_sha256=claim_state_sha256,
        claim_ledger_commit=claim_ledger_commit,
        provider_identity_sha256=provider_identity.identity_sha256,
        beacon_contract_sha256=contract.contract_sha256,
        design_seed_sha256=design_seed_sha256,
        beacon_bytes_sha256=claims.beacon_bytes_sha256,
        chain_hash=claims.chain_hash,
        round=claims.round,
        randomness=claims.randomness,
        signature=claims.signature,
        published_at_utc=contract.execution_publication_time.isoformat(),
        verified_at_utc=_timestamp("verified_at_utc", verified_at_utc).isoformat(),
        derived_seed_sha256=seed,
        permutation_seed=int.from_bytes(bytes.fromhex(seed)[:8], "big"),
    )


@dataclass(frozen=True)
class PhaseBeaconReceipt:
    phase: Literal["label-release"]
    phase_claim_state_sha256: str
    phase_claim_ledger_commit: str
    provider_identity_sha256: str
    phase_claim_contract_sha256: str
    beacon_contract_sha256: str
    beacon_bytes_sha256: str
    chain_hash: str
    round: int
    randomness: str
    signature: str
    published_at_utc: str
    verified_at_utc: str
    schema_version: str = PHASE_BEACON_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.phase != LABEL_RELEASE_PHASE:
            raise ExecutionClaimError("phase beacon receipt is not label release")
        for name in (
            "phase_claim_state_sha256",
            "provider_identity_sha256",
            "phase_claim_contract_sha256",
            "beacon_contract_sha256",
            "beacon_bytes_sha256",
            "chain_hash",
        ):
            _digest(name, getattr(self, name))
        _git_commit("phase_claim_ledger_commit", self.phase_claim_ledger_commit)
        _positive("round", self.round)
        for name in ("randomness", "signature"):
            value = getattr(self, name)
            if type(value) is not str or _HEX.fullmatch(value) is None:
                raise ExecutionClaimError(f"{name} must be lowercase hexadecimal bytes")
        published = _timestamp("published_at_utc", self.published_at_utc)
        verified = _timestamp("verified_at_utc", self.verified_at_utc)
        if verified < published:
            raise ExecutionClaimError("label-release beacon verification predates publication")
        if self.schema_version != PHASE_BEACON_RECEIPT_SCHEMA:
            raise ExecutionClaimError("phase beacon receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> PhaseBeaconReceipt:
        return cls(
            **_closed(value, frozenset(cls.__dataclass_fields__), label="phase beacon receipt")
        )


def verify_label_release_beacon(
    contract: PhaseClaimContract,
    *,
    beacon_bytes: bytes,
    phase_claim_state_sha256: str,
    phase_claim_ledger_commit: str,
    provider_identity: ProviderExecutionIdentity,
    claim_attested_at_utc: str,
    live_execute_job_receipt: LiveExecuteJobReceipt,
    verifier: ExecutionBeaconVerifier,
    verified_at_utc: str,
) -> PhaseBeaconReceipt:
    """Verify the registered later beacon after LABEL_RELEASE_CLAIMED wins."""

    if contract.phase != LABEL_RELEASE_PHASE or contract.label_release_beacon is None:
        raise ExecutionClaimError("label release beacon requires a label-release contract")
    provider_identity.matches_phase_contract(contract)
    if not isinstance(live_execute_job_receipt, LiveExecuteJobReceipt):
        raise ExecutionClaimError("label release beacon lacks a typed live execute job")
    if (
        live_execute_job_receipt.provider_identity_sha256 != provider_identity.identity_sha256
        or live_execute_job_receipt.execute_job_name != contract.execute_job_name
    ):
        raise ExecutionClaimError("label release live execute job differs from the claim")
    _digest("phase_claim_state_sha256", phase_claim_state_sha256)
    _git_commit("phase_claim_ledger_commit", phase_claim_ledger_commit)
    claimed = _timestamp("claim_attested_at_utc", claim_attested_at_utc)
    beacon_contract = contract.label_release_beacon
    if claimed >= beacon_contract.label_release_publication_time:
        raise ExecutionClaimError("LABEL_RELEASE_CLAIMED must precede label beacon publication")
    if not beacon_bytes or len(beacon_bytes) > 1024 * 1024:
        raise ExecutionClaimError("label-release beacon bytes are empty or exceed the bound")
    release_contract = replace(
        beacon_contract,
        execution_round=beacon_contract.label_release_round,
        label_release_round=(
            beacon_contract.label_release_round
            + beacon_contract.minimum_label_release_safety_rounds
        ),
    )
    try:
        claims = verifier.verify(contract=release_contract, beacon_bytes=beacon_bytes)
    except ExecutionClaimError:
        raise
    except Exception as exc:
        raise ExecutionClaimError("label-release beacon verifier rejected the bytes") from exc
    if not isinstance(claims, VerifiedBeaconClaims):
        raise ExecutionClaimError("label-release beacon verifier returned untyped claims")
    exact = {
        "chain_hash": beacon_contract.chain_hash,
        "round": beacon_contract.label_release_round,
        "beacon_bytes_sha256": _sha256(beacon_bytes),
        "scheme_id": beacon_contract.chain_scheme_id,
        "public_key": beacon_contract.chain_public_key,
    }
    for name, expected in exact.items():
        if getattr(claims, name) != expected:
            raise ExecutionClaimError(f"verified label beacon {name} differs from contract")
    return PhaseBeaconReceipt(
        phase=LABEL_RELEASE_PHASE,
        phase_claim_state_sha256=phase_claim_state_sha256,
        phase_claim_ledger_commit=phase_claim_ledger_commit,
        provider_identity_sha256=provider_identity.identity_sha256,
        phase_claim_contract_sha256=contract.contract_sha256,
        beacon_contract_sha256=beacon_contract.contract_sha256,
        beacon_bytes_sha256=claims.beacon_bytes_sha256,
        chain_hash=claims.chain_hash,
        round=claims.round,
        randomness=claims.randomness,
        signature=claims.signature,
        published_at_utc=beacon_contract.label_release_publication_time.isoformat(),
        verified_at_utc=_timestamp("verified_at_utc", verified_at_utc).isoformat(),
    )


@dataclass(frozen=True)
class CorpusOutputTree:
    corpus_id: str
    output_namespace_uri: str
    tree_sha256: str

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ExecutionClaimError("output tree names an unregistered corpus")
        parsed = urlsplit(_text("output_namespace_uri", self.output_namespace_uri))
        if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
            raise ExecutionClaimError("output tree namespace must be a file URI")
        _digest("tree_sha256", self.tree_sha256)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> CorpusOutputTree:
        return cls(**_closed(value, frozenset(cls.__dataclass_fields__), label="output tree"))


@dataclass(frozen=True)
class RunOutputAggregate:
    claim_state_sha256: str
    claim_ledger_commit: str
    provider_identity_sha256: str
    execute_job_id: int
    output_aggregate_identity: str
    corpus_trees: tuple[CorpusOutputTree, ...]
    aggregate_sha256: str
    schema_version: str = RUN_OUTPUT_AGGREGATE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "claim_state_sha256",
            "provider_identity_sha256",
            "output_aggregate_identity",
            "aggregate_sha256",
        ):
            _digest(name, getattr(self, name))
        _git_commit("claim_ledger_commit", self.claim_ledger_commit)
        _positive("execute_job_id", self.execute_job_id)
        rows = _fixed_rows("run output aggregate", self.corpus_trees, CorpusOutputTree)
        expected = _sha256(
            _canonical_bytes(
                {
                    "claim_ledger_commit": self.claim_ledger_commit,
                    "claim_state_sha256": self.claim_state_sha256,
                    "corpus_trees": [row.to_dict() for row in rows],
                    "derivation": OUTPUT_AGGREGATE_DERIVATION,
                    "output_aggregate_identity": self.output_aggregate_identity,
                    "provider_identity_sha256": self.provider_identity_sha256,
                    "execute_job_id": self.execute_job_id,
                }
            )
        )
        if self.aggregate_sha256 != expected:
            raise ExecutionClaimError("five-tree output aggregate is not canonical")
        if self.schema_version != RUN_OUTPUT_AGGREGATE_SCHEMA:
            raise ExecutionClaimError("run output aggregate schema differs")
        object.__setattr__(self, "corpus_trees", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "corpus_trees"
            },
            "corpus_trees": [row.to_dict() for row in self.corpus_trees],
        }

    @classmethod
    def from_dict(cls, value: object) -> RunOutputAggregate:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="run output aggregate")
        raw = row["corpus_trees"]
        if type(raw) is not list:
            raise ExecutionClaimError("run output corpus_trees must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "corpus_trees"},
            corpus_trees=tuple(CorpusOutputTree.from_dict(item) for item in raw),
        )


@dataclass(frozen=True)
class PartialEvidenceBinding:
    relative_path: str
    byte_count: int
    file_sha256: str

    def __post_init__(self) -> None:
        path = Path(_text("partial relative_path", self.relative_path))
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ExecutionClaimError("partial evidence path must be canonical and relative")
        _nonnegative("partial byte_count", self.byte_count)
        _digest("partial file_sha256", self.file_sha256)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> PartialEvidenceBinding:
        return cls(**_closed(value, frozenset(cls.__dataclass_fields__), label="partial evidence"))


@dataclass(frozen=True)
class ProviderPhaseFailure:
    phase: ProviderPhase
    claim_state_sha256: str
    claim_ledger_commit: str
    provider_identity_sha256: str
    failed_execute_job_receipt_sha256: str
    execute_job_id: int
    phase_input_sha256: str
    exit_status: int | None
    termination_signal: int | None
    incident_uri: str
    incident_byte_count: int
    incident_file_sha256: str
    partial_evidence: tuple[PartialEvidenceBinding, ...]
    schema_version: str = PHASE_FAILURE_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in {ONLINE_PHASE, LABEL_RELEASE_PHASE, ANALYSIS_PHASE}:
            raise ExecutionClaimError("failure phase is not registered")
        for name in (
            "claim_state_sha256",
            "provider_identity_sha256",
            "failed_execute_job_receipt_sha256",
            "phase_input_sha256",
        ):
            _digest(name, getattr(self, name))
        _git_commit("claim_ledger_commit", self.claim_ledger_commit)
        _positive("execute_job_id", self.execute_job_id)
        if (self.exit_status is None) == (self.termination_signal is None):
            raise ExecutionClaimError("failure must bind exactly one exit status or signal")
        if self.exit_status is not None:
            _nonnegative("exit_status", self.exit_status)
        if self.termination_signal is not None:
            _positive("termination_signal", self.termination_signal)
        parsed = urlsplit(_text("incident_uri", self.incident_uri))
        if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
            raise ExecutionClaimError("incident_uri must be a canonical immutable file URI")
        _positive("incident_byte_count", self.incident_byte_count)
        _digest("incident_file_sha256", self.incident_file_sha256)
        rows = tuple(self.partial_evidence)
        if (
            not all(isinstance(row, PartialEvidenceBinding) for row in rows)
            or rows != tuple(sorted(rows, key=lambda row: row.relative_path.encode("utf-8")))
            or len({row.relative_path for row in rows}) != len(rows)
        ):
            raise ExecutionClaimError("partial evidence must be a unique byte-sorted inventory")
        if self.schema_version != PHASE_FAILURE_SCHEMA:
            raise ExecutionClaimError("provider phase failure schema differs")
        object.__setattr__(self, "partial_evidence", rows)

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "partial_evidence"
            },
            "partial_evidence": [row.to_dict() for row in self.partial_evidence],
        }

    @classmethod
    def from_dict(cls, value: object) -> ProviderPhaseFailure:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="provider phase failure")
        raw = row["partial_evidence"]
        if type(raw) is not list:
            raise ExecutionClaimError("partial_evidence must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "partial_evidence"},
            partial_evidence=tuple(PartialEvidenceBinding.from_dict(item) for item in raw),
        )


@dataclass(frozen=True)
class RuntimeClaimReceipt:
    """Portable receipt passed to the isolated runtime; it is not host authority."""

    manifest_sha256: str
    run_receipt_sha256: str
    c1_commit: str
    claim_contract_sha256: str
    claim_state_sha256: str
    claim_ledger_commit: str
    provider_identity_sha256: str
    live_execute_job_receipt_sha256: str
    execute_job_id: int
    beacon_receipt_sha256: str
    beacon_bytes_sha256: str
    design_seed_sha256: str
    derived_seed_sha256: str
    permutation_seed: int
    output_aggregate_identity: str
    schema_version: str = RUNTIME_CLAIM_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "claim_contract_sha256",
            "claim_state_sha256",
            "provider_identity_sha256",
            "live_execute_job_receipt_sha256",
            "beacon_receipt_sha256",
            "beacon_bytes_sha256",
            "design_seed_sha256",
            "derived_seed_sha256",
            "output_aggregate_identity",
        ):
            _digest(name, getattr(self, name))
        _git_commit("c1_commit", self.c1_commit)
        _git_commit("claim_ledger_commit", self.claim_ledger_commit)
        _positive("execute_job_id", self.execute_job_id)
        if type(self.permutation_seed) is not int or not 0 <= self.permutation_seed < 2**64:
            raise ExecutionClaimError("runtime permutation seed must be unsigned 64-bit")
        if self.permutation_seed != int.from_bytes(
            bytes.fromhex(self.derived_seed_sha256)[:8], "big"
        ):
            raise ExecutionClaimError("runtime permutation seed differs from derived seed")
        if self.schema_version != RUNTIME_CLAIM_RECEIPT_SCHEMA:
            raise ExecutionClaimError("runtime claim receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> RuntimeClaimReceipt:
        return cls(
            **_closed(value, frozenset(cls.__dataclass_fields__), label="runtime claim receipt")
        )


def loads_runtime_claim_receipt(encoded: bytes) -> RuntimeClaimReceipt:
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ExecutionClaimError("runtime claim receipt must end with one newline")
    value = _strict_object(encoded[:-1], label="runtime claim receipt")
    receipt = RuntimeClaimReceipt.from_dict(value)
    if receipt.canonical_file_bytes() != encoded:
        raise ExecutionClaimError("runtime claim receipt bytes are not canonical")
    return receipt


@dataclass(frozen=True)
class PhaseRuntimeClaimReceipt:
    """Portable per-corpus receipt; only the in-memory capability is authority."""

    phase: Literal["label-release", "analysis"]
    manifest_sha256: str
    run_receipt_sha256: str
    c1_commit: str
    phase_claim_contract_sha256: str
    phase_claim_state_sha256: str
    phase_claim_ledger_commit: str
    provider_identity_sha256: str
    live_execute_job_receipt_sha256: str
    execute_job_id: int
    phase_input_aggregate_sha256: str
    phase_output_identity: str
    corpus_id: str
    input_uri: str
    input_sha256: str
    supporting_input_uri: str
    supporting_input_sha256: str
    phase_beacon_receipt_sha256: str | None
    schema_version: str = PHASE_RUNTIME_CLAIM_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.phase not in {LABEL_RELEASE_PHASE, ANALYSIS_PHASE}:
            raise ExecutionClaimError("phase runtime receipt has an unknown phase")
        for name in (
            "manifest_sha256",
            "run_receipt_sha256",
            "phase_claim_contract_sha256",
            "phase_claim_state_sha256",
            "provider_identity_sha256",
            "live_execute_job_receipt_sha256",
            "phase_input_aggregate_sha256",
            "phase_output_identity",
            "input_sha256",
            "supporting_input_sha256",
        ):
            _digest(name, getattr(self, name))
        for name in ("c1_commit", "phase_claim_ledger_commit"):
            _git_commit(name, getattr(self, name))
        _positive("execute_job_id", self.execute_job_id)
        if self.corpus_id not in FIXED_CORPORA:
            raise ExecutionClaimError("phase runtime receipt names another corpus")
        for name in ("input_uri", "supporting_input_uri"):
            parsed = urlsplit(_text(name, getattr(self, name)))
            if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
                raise ExecutionClaimError(f"phase runtime {name} must be a file URI")
        if self.phase == LABEL_RELEASE_PHASE:
            if self.phase_beacon_receipt_sha256 is None:
                raise ExecutionClaimError("label runtime receipt lacks its release beacon")
            _digest("phase_beacon_receipt_sha256", self.phase_beacon_receipt_sha256)
        elif self.phase_beacon_receipt_sha256 is not None:
            raise ExecutionClaimError("analysis runtime receipt cannot introduce a beacon")
        if self.schema_version != PHASE_RUNTIME_CLAIM_RECEIPT_SCHEMA:
            raise ExecutionClaimError("phase runtime claim receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: object) -> PhaseRuntimeClaimReceipt:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="phase runtime claim receipt",
            )
        )


_PHASE_CAPABILITY = object()


@dataclass(frozen=True)
class VerifiedPhaseClaimCapability:
    """Fresh non-serializable authority for label release or analysis."""

    contract: PhaseClaimContract
    provider_identity: ProviderExecutionIdentity
    phase_claim_state_sha256: str
    phase_claim_ledger_commit: str
    claim_attested_at_utc: str
    live_execute_job_receipt: LiveExecuteJobReceipt
    phase_beacon_receipt: PhaseBeaconReceipt | None
    _fresh_revalidator: Callable[[], None] = field(repr=False, compare=False)
    _minted_monotonic_ns: int = field(repr=False, compare=False)
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _PHASE_CAPABILITY:
            raise ExecutionClaimError(
                "phase-claim authority can only come from provider verification"
            )
        if not isinstance(self.contract, PhaseClaimContract):
            raise ExecutionClaimError("phase-claim contract is untyped")
        if not isinstance(self.provider_identity, ProviderExecutionIdentity):
            raise ExecutionClaimError("phase-claim provider identity is untyped")
        if not isinstance(self.live_execute_job_receipt, LiveExecuteJobReceipt):
            raise ExecutionClaimError("phase-claim live execute-job receipt is untyped")
        self.provider_identity.matches_phase_contract(self.contract)
        if (
            self.live_execute_job_receipt.provider_identity_sha256
            != self.provider_identity.identity_sha256
            or self.live_execute_job_receipt.execute_job_name != self.contract.execute_job_name
        ):
            raise ExecutionClaimError("live execute job belongs to another phase claim")
        _digest("phase_claim_state_sha256", self.phase_claim_state_sha256)
        _git_commit("phase_claim_ledger_commit", self.phase_claim_ledger_commit)
        claimed = _timestamp("claim_attested_at_utc", self.claim_attested_at_utc)
        if self.contract.phase == LABEL_RELEASE_PHASE:
            if not isinstance(self.phase_beacon_receipt, PhaseBeaconReceipt):
                raise ExecutionClaimError("label-release authority lacks a verified beacon")
            beacon = self.phase_beacon_receipt
            if (
                beacon.phase_claim_state_sha256 != self.phase_claim_state_sha256
                or beacon.phase_claim_ledger_commit != self.phase_claim_ledger_commit
                or beacon.provider_identity_sha256 != self.provider_identity.identity_sha256
                or beacon.phase_claim_contract_sha256 != self.contract.contract_sha256
            ):
                raise ExecutionClaimError("label-release beacon belongs to another claim")
            assert self.contract.label_release_beacon is not None
            if claimed >= self.contract.label_release_beacon.label_release_publication_time:
                raise ExecutionClaimError("label-release claim is not pre-beacon")
        elif self.phase_beacon_receipt is not None:
            raise ExecutionClaimError("analysis authority cannot introduce a beacon")
        if not callable(self._fresh_revalidator):
            raise ExecutionClaimError("phase-claim revalidator is absent")
        self.assert_current()

    def assert_current(self) -> None:
        age = time.monotonic_ns() - self._minted_monotonic_ns
        if age < 0 or age > _MAX_CAPABILITY_AGE_NS:
            raise ExecutionClaimError("phase-claim capability is stale")
        try:
            result = self._fresh_revalidator()
        except ExecutionClaimError:
            raise
        except Exception as exc:
            raise ExecutionClaimError("fresh phase-claim revalidation failed") from exc
        if result is not None:
            raise ExecutionClaimError("phase-claim revalidator returned unexpected data")

    def require_input(
        self,
        *,
        corpus_id: str,
        input_uri: str,
        input_sha256: str,
        supporting_input_uri: str,
        supporting_input_sha256: str,
    ) -> PhaseRuntimeClaimReceipt:
        self.assert_current()
        matches = [row for row in self.contract.corpora if row.corpus_id == corpus_id]
        if len(matches) != 1:
            raise ExecutionClaimError("phase claim lacks the requested corpus")
        binding = matches[0]
        if (
            binding.input_uri != input_uri
            or binding.input_sha256 != input_sha256
            or binding.supporting_input_uri != supporting_input_uri
            or binding.supporting_input_sha256 != supporting_input_sha256
        ):
            raise ExecutionClaimError("phase input differs from the winning claim")
        return PhaseRuntimeClaimReceipt(
            phase=self.contract.phase,
            manifest_sha256=self.contract.manifest_sha256,
            run_receipt_sha256=self.contract.run_receipt_sha256,
            c1_commit=self.contract.c1_commit,
            phase_claim_contract_sha256=self.contract.contract_sha256,
            phase_claim_state_sha256=self.phase_claim_state_sha256,
            phase_claim_ledger_commit=self.phase_claim_ledger_commit,
            provider_identity_sha256=self.provider_identity.identity_sha256,
            live_execute_job_receipt_sha256=self.live_execute_job_receipt.receipt_sha256,
            execute_job_id=self.live_execute_job_receipt.execute_job_id,
            phase_input_aggregate_sha256=self.contract.phase_input_aggregate_sha256,
            phase_output_identity=self.contract.phase_output_identity,
            corpus_id=binding.corpus_id,
            input_uri=binding.input_uri,
            input_sha256=binding.input_sha256,
            supporting_input_uri=binding.supporting_input_uri,
            supporting_input_sha256=binding.supporting_input_sha256,
            phase_beacon_receipt_sha256=(
                None
                if self.phase_beacon_receipt is None
                else self.phase_beacon_receipt.receipt_sha256
            ),
        )


def _mint_verified_phase_claim(
    *,
    contract: PhaseClaimContract,
    provider_identity: ProviderExecutionIdentity,
    phase_claim_state_sha256: str,
    phase_claim_ledger_commit: str,
    claim_attested_at_utc: str,
    live_execute_job_receipt: LiveExecuteJobReceipt,
    phase_beacon_receipt: PhaseBeaconReceipt | None,
    fresh_revalidator: Callable[[], None],
) -> VerifiedPhaseClaimCapability:
    return VerifiedPhaseClaimCapability(
        contract=contract,
        provider_identity=provider_identity,
        phase_claim_state_sha256=phase_claim_state_sha256,
        phase_claim_ledger_commit=phase_claim_ledger_commit,
        claim_attested_at_utc=claim_attested_at_utc,
        live_execute_job_receipt=live_execute_job_receipt,
        phase_beacon_receipt=phase_beacon_receipt,
        _fresh_revalidator=fresh_revalidator,
        _minted_monotonic_ns=time.monotonic_ns(),
        _capability=_PHASE_CAPABILITY,
    )


@dataclass(frozen=True)
class VerifiedRunClaimCapability:
    """Fresh, non-serializable authority for one provider-claimed online suite."""

    contract: ExecutionClaimContract
    provider_identity: ProviderExecutionIdentity
    claim_state_sha256: str
    claim_ledger_commit: str
    claim_attested_at_utc: str
    beacon_receipt: ExecutionBeaconReceipt
    live_execute_job_receipt: LiveExecuteJobReceipt
    zenodo_admission: AnonymousZenodoAdmission
    _fresh_revalidator: Callable[[], None] = field(repr=False, compare=False)
    _minted_monotonic_ns: int = field(repr=False, compare=False)
    _capability: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _CAPABILITY:
            raise ExecutionClaimError(
                "run-claim authority can only come from provider verification"
            )
        if not isinstance(self.contract, ExecutionClaimContract):
            raise ExecutionClaimError("run-claim contract is untyped")
        if not isinstance(self.provider_identity, ProviderExecutionIdentity):
            raise ExecutionClaimError("run-claim provider identity is untyped")
        if not isinstance(self.beacon_receipt, ExecutionBeaconReceipt):
            raise ExecutionClaimError("run-claim beacon receipt is untyped")
        if not isinstance(self.live_execute_job_receipt, LiveExecuteJobReceipt):
            raise ExecutionClaimError("run-claim live execute-job receipt is untyped")
        if not isinstance(self.zenodo_admission, AnonymousZenodoAdmission):
            raise ExecutionClaimError("run-claim Zenodo admission is untyped")
        _digest("claim_state_sha256", self.claim_state_sha256)
        _git_commit("claim_ledger_commit", self.claim_ledger_commit)
        claimed = _timestamp("claim_attested_at_utc", self.claim_attested_at_utc)
        published = _timestamp("Zenodo published_at_utc", self.zenodo_admission.published_at_utc)
        if claimed < published:
            raise ExecutionClaimError("RUN_CLAIMED attestation predates public Zenodo publication")
        if claimed >= self.contract.beacon.execution_publication_time:
            raise ExecutionClaimError("RUN_CLAIMED attestation is not pre-beacon")
        if (
            self.beacon_receipt.claim_state_sha256 != self.claim_state_sha256
            or self.beacon_receipt.claim_ledger_commit != self.claim_ledger_commit
            or self.beacon_receipt.provider_identity_sha256
            != self.provider_identity.identity_sha256
            or self.beacon_receipt.beacon_contract_sha256 != self.contract.beacon.contract_sha256
        ):
            raise ExecutionClaimError("run-claim beacon receipt belongs to another claim")
        self.provider_identity.matches_contract(self.contract)
        if (
            self.live_execute_job_receipt.provider_identity_sha256
            != self.provider_identity.identity_sha256
            or self.live_execute_job_receipt.execute_job_name != self.contract.execute_job_name
        ):
            raise ExecutionClaimError("live execute job belongs to another run claim")
        if not callable(self._fresh_revalidator):
            raise ExecutionClaimError("run-claim revalidator is absent")
        self.assert_current()

    def assert_current(self) -> None:
        age = time.monotonic_ns() - self._minted_monotonic_ns
        if age < 0 or age > _MAX_CAPABILITY_AGE_NS:
            raise ExecutionClaimError("run-claim capability is stale; verify the ledger again")
        try:
            result = self._fresh_revalidator()
        except ExecutionClaimError:
            raise
        except Exception as exc:
            raise ExecutionClaimError("fresh RUN_CLAIMED revalidation failed") from exc
        if result is not None:
            raise ExecutionClaimError("RUN_CLAIMED revalidator returned unexpected data")

    def require_launch(
        self,
        *,
        manifest_sha256: str,
        corpus_id: str,
        runtime_plan_sha256: str,
        output_namespace_uri: str,
    ) -> RuntimeClaimReceipt:
        self.assert_current()
        if _digest("manifest_sha256", manifest_sha256) != self.contract.manifest_sha256:
            raise ExecutionClaimError("launch manifest differs from RUN_CLAIMED")
        matches = [row for row in self.contract.corpora if row.corpus_id == corpus_id]
        if len(matches) != 1:
            raise ExecutionClaimError("RUN_CLAIMED lacks the requested corpus")
        binding = matches[0]
        if (
            binding.runtime_plan_sha256 != runtime_plan_sha256
            or binding.staging_namespace_uri != output_namespace_uri
        ):
            raise ExecutionClaimError("launch plan or namespace differs from RUN_CLAIMED")
        return RuntimeClaimReceipt(
            manifest_sha256=self.contract.manifest_sha256,
            run_receipt_sha256=self.contract.run_receipt_sha256,
            c1_commit=self.contract.c1_commit,
            claim_contract_sha256=self.contract.contract_sha256,
            claim_state_sha256=self.claim_state_sha256,
            claim_ledger_commit=self.claim_ledger_commit,
            provider_identity_sha256=self.provider_identity.identity_sha256,
            live_execute_job_receipt_sha256=self.live_execute_job_receipt.receipt_sha256,
            execute_job_id=self.live_execute_job_receipt.execute_job_id,
            beacon_receipt_sha256=self.beacon_receipt.receipt_sha256,
            beacon_bytes_sha256=self.beacon_receipt.beacon_bytes_sha256,
            design_seed_sha256=self.beacon_receipt.design_seed_sha256,
            derived_seed_sha256=self.beacon_receipt.derived_seed_sha256,
            permutation_seed=self.beacon_receipt.permutation_seed,
            output_aggregate_identity=self.contract.output_aggregate_identity,
        )


def _mint_verified_run_claim(
    *,
    contract: ExecutionClaimContract,
    provider_identity: ProviderExecutionIdentity,
    claim_state_sha256: str,
    claim_ledger_commit: str,
    claim_attested_at_utc: str,
    beacon_receipt: ExecutionBeaconReceipt,
    live_execute_job_receipt: LiveExecuteJobReceipt,
    zenodo_admission: AnonymousZenodoAdmission,
    fresh_revalidator: Callable[[], None],
) -> VerifiedRunClaimCapability:
    """Private bridge used only after provider state and beacon verification."""

    return VerifiedRunClaimCapability(
        contract=contract,
        provider_identity=provider_identity,
        claim_state_sha256=claim_state_sha256,
        claim_ledger_commit=claim_ledger_commit,
        claim_attested_at_utc=claim_attested_at_utc,
        beacon_receipt=beacon_receipt,
        live_execute_job_receipt=live_execute_job_receipt,
        zenodo_admission=zenodo_admission,
        _fresh_revalidator=fresh_revalidator,
        _minted_monotonic_ns=time.monotonic_ns(),
        _capability=_CAPABILITY,
    )


def _add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--phase",
        required=True,
        choices=(ONLINE_PHASE, LABEL_RELEASE_PHASE, ANALYSIS_PHASE),
    )
    parser.add_argument("--suite-attempt-id", required=True, metavar="SHA256")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--github-output", required=True, type=Path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.execution_claim",
        description="Claim and reconcile the provider-backed confirmatory phase ledger.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prerequisites = commands.add_parser(
        "verify-prerequisites",
        help="verify anonymous host inputs or activate one already-claimed execute job",
    )
    _add_common_cli_arguments(prerequisites)
    prerequisites.add_argument("--claim-receipt", type=Path)
    prerequisites.add_argument("--activate-and-execute", action="store_true")

    claim = commands.add_parser(
        "claim",
        help="prepare and publish the sole provider phase claim",
    )
    _add_common_cli_arguments(claim)
    claim.add_argument("--prerequisite-receipt", required=True, type=Path)

    for command in ("complete", "fail"):
        transition = commands.add_parser(
            command,
            help=f"prepare or publish a terminal {command} transition",
        )
        mode = transition.add_mutually_exclusive_group(required=True)
        mode.add_argument("--prepare", action="store_true")
        mode.add_argument("--publish", action="store_true")
        _add_common_cli_arguments(transition)
        transition.add_argument("--claim-receipt", type=Path)
        transition.add_argument("--evidence-root", required=True, type=Path)
        transition.add_argument("--attestation-bundle", type=Path)
        transition.add_argument("--preparation-receipt", type=Path)
    return parser


def _validate_cli_arguments(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> None:
    try:
        _digest("suite_attempt_id", arguments.suite_attempt_id)
    except ExecutionClaimError as exc:
        parser.error(str(exc))
    if arguments.command == "verify-prerequisites":
        if arguments.activate_and_execute and arguments.claim_receipt is None:
            parser.error("--activate-and-execute requires --claim-receipt")
        if not arguments.activate_and_execute and arguments.claim_receipt is not None:
            parser.error("--claim-receipt is valid only with --activate-and-execute")
        return
    if arguments.command == "complete" and arguments.claim_receipt is None:
        parser.error("complete requires --claim-receipt")
    if arguments.command == "fail" and arguments.claim_receipt is not None:
        parser.error("fail recovers the live ledger claim and rejects --claim-receipt")
    if arguments.command in {"complete", "fail"}:
        if arguments.publish:
            if arguments.attestation_bundle is None:
                parser.error("--publish requires --attestation-bundle")
            if arguments.preparation_receipt is None:
                parser.error("--publish requires --preparation-receipt")
        elif arguments.attestation_bundle is not None or arguments.preparation_receipt is not None:
            parser.error("--attestation-bundle and --preparation-receipt require --publish")


def expected_cli_output_keys(arguments: argparse.Namespace) -> frozenset[str]:
    """Return the exact GitHub output interface for one parsed invocation."""

    phase: ProviderPhase = arguments.phase
    if arguments.command == "verify-prerequisites":
        if not arguments.activate_and_execute:
            return PREREQUISITE_OUTPUT_KEYS
        return ACTIVATION_COMMON_OUTPUT_KEYS | ACTIVATION_PHASE_OUTPUT_KEYS[phase]
    if arguments.command == "claim":
        return CLAIM_OUTPUT_KEYS
    if arguments.command == "complete":
        if arguments.prepare:
            return PREPARE_COMMON_OUTPUT_KEYS | {
                "completion_predicate_path",
                "completion_predicate_sha256",
            }
        return PUBLISH_OUTPUT_KEYS
    if arguments.command == "fail":
        if arguments.prepare:
            return PREPARE_COMMON_OUTPUT_KEYS | {
                "failure_predicate_path",
                "failure_predicate_sha256",
                "no_claim_to_fail",
            }
        return PUBLISH_OUTPUT_KEYS | {"no_claim_to_fail"}
    raise ExecutionClaimError("unregistered execution-claim command")


def _cli_verify_prerequisites(arguments: argparse.Namespace) -> Mapping[str, str]:
    from .provider_workflow_orchestration import (
        ProviderWorkflowOrchestrationError,
        execute_verify_prerequisites_command,
    )

    try:
        return execute_verify_prerequisites_command(
            phase=arguments.phase,
            suite_attempt_id=arguments.suite_attempt_id,
            output_dir=arguments.output_dir,
            claim_receipt_path=arguments.claim_receipt,
            activate_and_execute=arguments.activate_and_execute,
        )
    except ProviderWorkflowOrchestrationError as exc:
        raise ExecutionClaimError(str(exc)) from exc


def _cli_claim(arguments: argparse.Namespace) -> Mapping[str, str]:
    from .provider_workflow_orchestration import (
        ProviderWorkflowOrchestrationError,
        execute_claim_command,
    )

    try:
        return execute_claim_command(
            phase=arguments.phase,
            suite_attempt_id=arguments.suite_attempt_id,
            prerequisite_receipt_path=arguments.prerequisite_receipt,
            output_dir=arguments.output_dir,
        )
    except ProviderWorkflowOrchestrationError as exc:
        raise ExecutionClaimError(str(exc)) from exc


def _cli_complete(arguments: argparse.Namespace) -> Mapping[str, str]:
    from .provider_workflow_orchestration import (
        ProviderWorkflowOrchestrationError,
        execute_complete_command,
    )

    try:
        return execute_complete_command(
            phase=arguments.phase,
            suite_attempt_id=arguments.suite_attempt_id,
            prepare=arguments.prepare,
            publish=arguments.publish,
            claim_receipt_path=arguments.claim_receipt,
            evidence_root=arguments.evidence_root,
            attestation_bundle_path=arguments.attestation_bundle,
            preparation_receipt_path=arguments.preparation_receipt,
            output_dir=arguments.output_dir,
        )
    except ProviderWorkflowOrchestrationError as exc:
        raise ExecutionClaimError(str(exc)) from exc


def _cli_fail(arguments: argparse.Namespace) -> Mapping[str, str]:
    from .provider_workflow_orchestration import (
        ProviderWorkflowOrchestrationError,
        execute_fail_command,
    )

    try:
        return execute_fail_command(
            phase=arguments.phase,
            suite_attempt_id=arguments.suite_attempt_id,
            prepare=arguments.prepare,
            publish=arguments.publish,
            claim_receipt_path=arguments.claim_receipt,
            evidence_root=arguments.evidence_root,
            attestation_bundle_path=arguments.attestation_bundle,
            preparation_receipt_path=arguments.preparation_receipt,
            output_dir=arguments.output_dir,
        )
    except ProviderWorkflowOrchestrationError as exc:
        raise ExecutionClaimError(str(exc)) from exc


def _append_github_outputs(path: Path, outputs: Mapping[str, str]) -> None:
    if not isinstance(outputs, Mapping) or any(
        type(key) is not str or type(value) is not str for key, value in outputs.items()
    ):
        raise ExecutionClaimError("command outputs must be a string mapping")
    try:
        with path.open("a", encoding="utf-8", errors="strict", newline="\n") as stream:
            for key in sorted(outputs, key=lambda item: item.encode("utf-8")):
                value = outputs[key]
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
                    raise ExecutionClaimError("GitHub output key is invalid")
                if "\n" in value or "\r" in value:
                    raise ExecutionClaimError("single-line GitHub output contains a newline")
                stream.write(f"{key}={value}\n")
    except OSError as exc:
        raise ExecutionClaimError(f"cannot append GitHub outputs: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Run the closed command interface used by the three provider workflows."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    _validate_cli_arguments(parser, arguments)
    handlers: Mapping[str, Callable[[argparse.Namespace], Mapping[str, str]]] = {
        "verify-prerequisites": _cli_verify_prerequisites,
        "claim": _cli_claim,
        "complete": _cli_complete,
        "fail": _cli_fail,
    }
    try:
        outputs = dict(handlers[arguments.command](arguments))
        expected = expected_cli_output_keys(arguments)
        if set(outputs) != expected:
            raise ExecutionClaimError(
                "command output interface differs; "
                f"missing={sorted(expected - set(outputs))}, "
                f"unexpected={sorted(set(outputs) - expected)}"
            )
        _append_github_outputs(arguments.github_output, outputs)
    except ExecutionClaimError as exc:
        print(f"execution-claim error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess smoke tests
    raise SystemExit(main())
