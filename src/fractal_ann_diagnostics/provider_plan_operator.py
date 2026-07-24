"""Typed two-stage construction of the three provider phase-plan templates.

The operator never edits its source manifest.  Stage one admits typed source
records and writes a closed blueprint.  Stage two consumes exactly three live
runner-bootstrap receipts, derives the provider-plan object, and writes a new
candidate manifest whose plan closure passes the production loader.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import execution_claim as execution_claim_module
from .artifact_integrity import (
    ArtifactIntegrityError,
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
    DockerServerProbe,
    ExecutionBeaconContract,
    ExecutionClaimError,
    ExecutionClaimInputs,
    PhaseHostProbe,
    PhaseHostToolContract,
    ProviderPhase,
    ProviderRunnerBootstrapReceipt,
    derive_phase_runner_label,
    load_provider_phase_plans,
    provider_phase_plan_templates_sha256,
)
from .production_artifact_factory import (
    ProductionArtifactFactoryError,
    load_production_artifact_factory_config,
)
from .production_controls import (
    ProductionControlError,
    load_production_control_blueprint_receipt,
    load_production_control_config,
    load_production_control_config_write_receipt,
)
from .production_workload_registration import production_workload_file_sha256
from .provider_contract import (
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI,
    OFFICIAL_ACTIONS_RUNNER_VERSION,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI,
    OFFICIAL_GH_VERSION,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI,
    OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION,
    REGISTERED_DOCKER_CLIENT_BUILD,
    REGISTERED_DOCKER_CLIENT_VERSION,
    SOURCE_BUILT_LINUX_ARM64_TLE_SHA256,
)
from .provider_rehearsal import CandidateImageClosure
from .study import (
    C0_COMMIT_SENTINEL,
    FIXED_CORPORA,
    PROVIDER_PHASE_COMMAND_IDS,
    PROVIDER_PHASE_JOB_NAMES,
    PROVIDER_PHASE_PLAN_TEMPLATE_SCHEMA,
    PROVIDER_PHASE_RUNTIME_BINDINGS,
    PROVIDER_PHASE_RUNTIME_CEILINGS,
    PROVIDER_PHASE_WORKFLOWS,
    PROVIDER_PLAN_C1_COMMIT_BINDING,
    PROVIDER_PLAN_CLAIM_RECEIPT_BINDING,
    PROVIDER_PLAN_MANIFEST_BINDING,
    PROVIDER_PLAN_PHASE_INPUT_BINDING,
    PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
    PROVIDER_PLAN_PREDECESSOR_BINDING,
    PROVIDER_PLAN_SUITE_BINDING,
)

PROVIDER_PLAN_BLUEPRINT_SCHEMA = "fractal-provider-plan-blueprint-v2"
PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_SCHEMA = "fractal-provider-plan-blueprint-write-receipt-v1"
PROVIDER_PLAN_FINALIZATION_RECEIPT_SCHEMA = "fractal-provider-plan-finalization-receipt-v2"
PROVIDER_PLAN_CLAIM_NONCE_DERIVATION = "sha256-fractal-provider-plan-claim-nonce-v2"
PROVIDER_PLAN_BLUEPRINT_FILENAME = "provider-plan-blueprint.json"
PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_FILENAME = "provider-plan-blueprint-write-receipt.json"
PROVIDER_PLAN_FRAGMENT_FILENAME = "provider-phase-plans.json"
PROVIDER_PLAN_CANDIDATE_MANIFEST_FILENAME = "candidate-manifest.json"
PROVIDER_PLAN_FINALIZATION_RECEIPT_FILENAME = "provider-plan-finalization-receipt.json"
REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
RUN_HEAD_BRANCH = "confirmatory-apparatus-c0"
PHASES: tuple[ProviderPhase, ...] = (
    ONLINE_PHASE,
    LABEL_RELEASE_PHASE,
    ANALYSIS_PHASE,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OCI_REFERENCE = re.compile(r"^[^\s/@]+(?:/[^\s/@]+)+@sha256:[0-9a-f]{64}$")
_MAX_JSON_BYTES = 16 * 1024 * 1024


class ProviderPlanOperatorError(ValueError):
    """Raised when provider-plan construction lacks one exact typed input."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ProviderPlanOperatorError("provider-plan evidence must be canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProviderPlanOperatorError(f"{name} must be a lowercase SHA-256")
    return value


def _commit(name: str, value: object) -> str:
    if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
        raise ProviderPlanOperatorError(f"{name} must be one full lowercase Git commit")
    return value


def _text(name: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ProviderPlanOperatorError(f"{name} must be canonical non-empty text")
    return value


def _positive(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ProviderPlanOperatorError(f"{name} must be a positive integer")
    return value


def _runner_group(name: str, value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ProviderPlanOperatorError(f"{name} must be null or a non-negative integer")
    return value


def _absolute_path(name: str, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.name in {".", ".."}:
        raise ProviderPlanOperatorError(f"{name} must be an absolute path")
    return path


def _canonical_posix_path(name: str, value: str | Path) -> str:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text.startswith("/")
        or "\\" in text
        or pure.as_posix() != text
        or any(part in {"", ".", ".."} for part in pure.parts[1:])
    ):
        raise ProviderPlanOperatorError(f"{name} must be a canonical absolute POSIX path")
    return text


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProviderPlanOperatorError(f"{label} must be an object with string fields")
    observed = frozenset(value)
    if observed != fields:
        raise ProviderPlanOperatorError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unknown={sorted(observed - fields)}"
        )
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderPlanOperatorError(f"provider-plan JSON repeats key {key!r}")
        result[key] = value
    return result


def _parse_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ProviderPlanOperatorError(f"{label} must have exactly one terminal newline")
    try:
        value = json.loads(
            encoded[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderPlanOperatorError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ProviderPlanOperatorError(f"{label} must be a JSON object")
    if encoded != _canonical_bytes(value) + b"\n":
        raise ProviderPlanOperatorError(f"{label} bytes are not canonical")
    return value


def _read_json(path: str | Path, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    source = _absolute_path(label, path)
    try:
        encoded = read_secure_regular_file(source, max_bytes=_MAX_JSON_BYTES, label=label)
    except ArtifactIntegrityError as exc:
        raise ProviderPlanOperatorError(f"cannot read {label}: {exc}") from exc
    return _parse_object(encoded, label=label), encoded


def _open_private_parent(path: Path, *, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open("/", flags)
        for component in path.parent.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        named = path.parent.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino)
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ProviderPlanOperatorError(
                f"{label} parent must be one owner-controlled real directory"
            )
        return descriptor
    except ProviderPlanOperatorError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ProviderPlanOperatorError(f"cannot open {label} parent") from exc


def _rename_noreplace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    *,
    label: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise ProviderPlanOperatorError("atomic no-replace rename is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source,
            parent_descriptor,
            destination,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise ProviderPlanOperatorError("atomic no-replace rename is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source,
            parent_descriptor,
            destination,
            0x00000001,
        )
    else:
        raise ProviderPlanOperatorError("atomic no-replace rename is unavailable")
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ProviderPlanOperatorError(f"{label} already exists")
        raise ProviderPlanOperatorError(f"cannot atomically publish {label}: {os.strerror(error)}")


def _write_bundle_member(
    directory_descriptor: int,
    name: str,
    encoded: bytes,
    *,
    label: str,
) -> None:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
        try:
            view = memoryview(encoded)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("short write")
                offset += written
            os.fsync(descriptor)
            before = os.fstat(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = bytearray()
            while len(observed) <= len(encoded):
                chunk = os.read(descriptor, min(65536, len(encoded) + 1 - len(observed)))
                if not chunk:
                    break
                observed.extend(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProviderPlanOperatorError(f"cannot stage {label}") from exc
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
        raise ProviderPlanOperatorError(f"staged {label} changed during fd readback")
    if (
        bytes(observed) != encoded
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
    ):
        raise ProviderPlanOperatorError(f"staged {label} identity or bytes differ")


def _read_bundle_member(
    directory_descriptor: int,
    name: str,
    *,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > _MAX_JSON_BYTES
                or stat.S_IMODE(before.st_mode) != 0o600
                or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            ):
                raise OSError("bundle member identity differs")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise OSError("short bundle read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise OSError("bundle member grew during read")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProviderPlanOperatorError(f"cannot read published {label}") from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ProviderPlanOperatorError(f"published {label} changed during fd readback")
    return b"".join(chunks)


def _publish_private_bundle(
    output_directory: Path,
    members: Mapping[str, bytes],
    *,
    label: str,
    pre_publish: Callable[[], None],
    after_member_write: Callable[[int], None] | None = None,
) -> None:
    """Publish fixed 0600 members with one parent-fd directory rename."""

    output = _absolute_path(label, output_directory)
    if not output.name or output.name in {".", ".."}:
        raise ProviderPlanOperatorError(f"{label} must name one output directory")
    if not members or any(
        PurePosixPath(name).name != name or name in {".", ".."} for name in members
    ):
        raise ProviderPlanOperatorError(f"{label} member names are not closed")
    parent_descriptor = _open_private_parent(output, label=label)
    staging_name = f".{output.name}.staging-{secrets.token_hex(16)}"
    staging_descriptor: int | None = None
    published = False
    try:
        os.mkdir(staging_name, 0o700, dir_fd=parent_descriptor)
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        staging_descriptor = os.open(
            staging_name,
            directory_flags,
            dir_fd=parent_descriptor,
        )
        staging_metadata = os.fstat(staging_descriptor)
        if (
            not stat.S_ISDIR(staging_metadata.st_mode)
            or stat.S_IMODE(staging_metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and staging_metadata.st_uid != os.geteuid())
        ):
            raise ProviderPlanOperatorError(f"{label} staging directory is not private")
        for position, (name, encoded) in enumerate(members.items(), start=1):
            _write_bundle_member(
                staging_descriptor,
                name,
                encoded,
                label=f"{label} member {name}",
            )
            if after_member_write is not None:
                after_member_write(position)
        os.fsync(staging_descriptor)
        pre_publish()
        _rename_noreplace_at(
            parent_descriptor,
            staging_name,
            output.name,
            label=label,
        )
        published = True
        os.fsync(parent_descriptor)
    except ProviderPlanOperatorError:
        raise
    except OSError as exc:
        raise ProviderPlanOperatorError(f"cannot publish {label}") from exc
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if not published:
            for name in members:
                try:
                    os.unlink(f"{staging_name}/{name}", dir_fd=parent_descriptor)
                except FileNotFoundError:
                    pass
                except OSError:
                    break
            try:
                os.rmdir(staging_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)

    parent_descriptor = _open_private_parent(output, label=label)
    directory_descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory_descriptor = os.open(output.name, flags, dir_fd=parent_descriptor)
        metadata = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise ProviderPlanOperatorError(f"published {label} is not mode 0700")
        observed_names = set(os.listdir(directory_descriptor))
        if observed_names != set(members):
            raise ProviderPlanOperatorError(f"published {label} member set differs")
        for name, encoded in members.items():
            if (
                _read_bundle_member(
                    directory_descriptor,
                    name,
                    label=f"{label} member {name}",
                )
                != encoded
            ):
                raise ProviderPlanOperatorError(f"published {label} member {name} differs")
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(parent_descriptor)


def _mode(path: Path) -> str:
    try:
        return f"{stat.S_IMODE(path.lstat().st_mode):04o}"
    except OSError as exc:
        raise ProviderPlanOperatorError(f"cannot inspect output mode for {path}") from exc


def _paths_overlap(first: str | Path, second: str | Path) -> bool:
    left = Path(first)
    right = Path(second)
    return left == right or left in right.parents or right in left.parents


@dataclass(frozen=True)
class HostToolSources:
    controlled_root: Path
    python_executable: Path
    venv_root: Path
    gh_executable: Path
    runner_listener_executable: Path
    runner_listener_dll: Path
    runner_config_executable: Path
    runner_run_executable: Path
    docker_executable: Path
    host_probe_path: Path
    docker_server_probe_path: Path

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _absolute_path(name, getattr(self, name)))

    def to_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> HostToolSources:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="host-tool sources")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProviderRunnerExpectation:
    phase: ProviderPhase
    runner_name: str
    runner_group_id: int | None
    claim_nonce: str
    runner_label: str

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ProviderPlanOperatorError("runner expectation phase differs")
        _text("runner_name", self.runner_name)
        _runner_group("runner_group_id", self.runner_group_id)
        _digest("claim_nonce", self.claim_nonce)
        if self.runner_label != derive_phase_runner_label(self.claim_nonce, self.phase):
            raise ProviderPlanOperatorError("runner expectation label differs from its nonce")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> ProviderRunnerExpectation:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="runner expectation",
            )
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProviderPlanBlueprint:
    candidate_manifest_path: Path
    candidate_manifest_file_sha256: str
    production_control_config_path: Path
    production_control_config_file_sha256: str
    production_control_config_write_receipt_path: Path
    production_control_config_write_receipt_file_sha256: str
    production_control_blueprint_receipt_path: Path
    production_control_blueprint_receipt_file_sha256: str
    production_control_blueprint_receipt_sha256: str
    candidate_image_closure_path: Path
    candidate_image_closure_file_sha256: str
    candidate_image_source_commit: str
    build_context_tree_sha256: str
    candidate_bootstrap_closure_sha256: str
    scientific_candidate_reference: str
    scientific_production_reference: str
    release_candidate_reference: str
    release_production_reference: str
    approval_environment: str
    runner_identity: str
    host_tool_sources: HostToolSources
    host_tools: PhaseHostToolContract
    execution_beacon_contract_path: Path
    execution_beacon_contract_file_sha256: str
    execution_claim_inputs: ExecutionClaimInputs
    claim_root: str
    evidence_root: str
    runner_expectations: tuple[ProviderRunnerExpectation, ...]
    schema_version: str = PROVIDER_PLAN_BLUEPRINT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "candidate_manifest_path",
            "production_control_config_path",
            "production_control_config_write_receipt_path",
            "production_control_blueprint_receipt_path",
            "candidate_image_closure_path",
            "execution_beacon_contract_path",
        ):
            object.__setattr__(self, name, _absolute_path(name, getattr(self, name)))
        for name in (
            "candidate_manifest_file_sha256",
            "production_control_config_file_sha256",
            "production_control_config_write_receipt_file_sha256",
            "production_control_blueprint_receipt_file_sha256",
            "production_control_blueprint_receipt_sha256",
            "candidate_image_closure_file_sha256",
            "build_context_tree_sha256",
            "candidate_bootstrap_closure_sha256",
            "execution_beacon_contract_file_sha256",
        ):
            _digest(name, getattr(self, name))
        _commit("candidate_image_source_commit", self.candidate_image_source_commit)
        for name in (
            "scientific_candidate_reference",
            "scientific_production_reference",
            "release_candidate_reference",
            "release_production_reference",
        ):
            value = getattr(self, name)
            if type(value) is not str or _OCI_REFERENCE.fullmatch(value) is None:
                raise ProviderPlanOperatorError(f"{name} must be an immutable OCI reference")
        if (
            self.approval_environment != PROVIDER_APPROVAL_ENVIRONMENT
            or self.runner_identity != PROVIDER_RUNNER_IDENTITY
            or self.runner_identity != f"github-actions:environment:{self.approval_environment}"
        ):
            raise ProviderPlanOperatorError("blueprint approval environment differs")
        if not isinstance(self.host_tool_sources, HostToolSources) or not isinstance(
            self.host_tools, PhaseHostToolContract
        ):
            raise ProviderPlanOperatorError("blueprint host-tool inputs must be typed")
        if str(self.host_tool_sources.controlled_root) != self.host_tools.controlled_root:
            raise ProviderPlanOperatorError(
                "host-tool source root differs from the derived contract"
            )
        if not isinstance(self.execution_claim_inputs, ExecutionClaimInputs):
            raise ProviderPlanOperatorError("blueprint execution-claim inputs must be typed")
        object.__setattr__(self, "claim_root", _canonical_posix_path("claim_root", self.claim_root))
        object.__setattr__(
            self,
            "evidence_root",
            _canonical_posix_path("evidence_root", self.evidence_root),
        )
        if _paths_overlap(self.claim_root, self.evidence_root):
            raise ProviderPlanOperatorError("claim_root and evidence_root overlap")
        controlled = str(self.host_tool_sources.controlled_root)
        if _paths_overlap(controlled, self.claim_root) or _paths_overlap(
            controlled, self.evidence_root
        ):
            raise ProviderPlanOperatorError("mutable phase roots overlap controlled_root")
        expectations = tuple(self.runner_expectations)
        if tuple(item.phase for item in expectations) != PHASES:
            raise ProviderPlanOperatorError(
                "runner expectations must follow the three fixed phases"
            )
        if len({item.runner_name for item in expectations}) != len(PHASES):
            raise ProviderPlanOperatorError("runner expectations reuse one runner name")
        if any(item.runner_group_id is not None for item in expectations):
            raise ProviderPlanOperatorError(
                "personal-repository provider runners must use runner_group_id null"
            )
        object.__setattr__(self, "runner_expectations", expectations)
        if self.schema_version != PROVIDER_PLAN_BLUEPRINT_SCHEMA:
            raise ProviderPlanOperatorError("provider-plan blueprint schema differs")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, Path):
                payload[name] = str(value)
            elif name == "host_tool_sources":
                payload[name] = self.host_tool_sources.to_dict()
            elif name == "host_tools":
                payload[name] = self.host_tools.to_dict()
            elif name == "execution_claim_inputs":
                payload[name] = self.execution_claim_inputs.to_dict()
            elif name == "runner_expectations":
                payload[name] = [item.to_dict() for item in self.runner_expectations]
            else:
                payload[name] = value
        return payload

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProviderPlanBlueprint:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="provider-plan blueprint")
        expectations = row["runner_expectations"]
        if type(expectations) is not list:
            raise ProviderPlanOperatorError("runner_expectations must be an array")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "execution_claim_inputs",
                    "host_tool_sources",
                    "host_tools",
                    "runner_expectations",
                }
            },
            execution_claim_inputs=ExecutionClaimInputs.from_dict(row["execution_claim_inputs"]),
            host_tool_sources=HostToolSources.from_dict(row["host_tool_sources"]),
            host_tools=PhaseHostToolContract.from_dict(row["host_tools"]),
            runner_expectations=tuple(
                ProviderRunnerExpectation.from_dict(item) for item in expectations
            ),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProviderPlanBlueprintWriteReceipt:
    blueprint_path: Path
    blueprint_file_sha256: str
    blueprint_readback_sha256: str
    blueprint_byte_count: int
    blueprint_mode: str
    candidate_manifest_file_sha256: str
    production_control_config_file_sha256: str
    production_control_config_write_receipt_file_sha256: str
    production_control_blueprint_receipt_file_sha256: str
    candidate_image_closure_file_sha256: str
    execution_beacon_contract_file_sha256: str
    host_tool_contract_sha256: str
    readback_verified: bool
    schema_version: str = PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "blueprint_path",
            _absolute_path("blueprint_path", self.blueprint_path),
        )
        for name in (
            "blueprint_file_sha256",
            "blueprint_readback_sha256",
            "candidate_manifest_file_sha256",
            "production_control_config_file_sha256",
            "production_control_config_write_receipt_file_sha256",
            "production_control_blueprint_receipt_file_sha256",
            "candidate_image_closure_file_sha256",
            "execution_beacon_contract_file_sha256",
            "host_tool_contract_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.blueprint_file_sha256 != self.blueprint_readback_sha256:
            raise ProviderPlanOperatorError("blueprint readback digest differs")
        _positive("blueprint_byte_count", self.blueprint_byte_count)
        if self.blueprint_mode != "0600" or self.readback_verified is not True:
            raise ProviderPlanOperatorError("blueprint write receipt lacks a 0600 readback")
        if self.schema_version != PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_SCHEMA:
            raise ProviderPlanOperatorError("provider-plan blueprint write-receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            name: str(getattr(self, name))
            if isinstance(getattr(self, name), Path)
            else getattr(self, name)
            for name in self.__dataclass_fields__
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProviderPlanBlueprintWriteReceipt:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="provider-plan blueprint write receipt",
            )
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProviderPlanFinalizationReceipt:
    blueprint_path: Path
    blueprint_file_sha256: str
    blueprint_write_receipt_path: Path
    blueprint_write_receipt_file_sha256: str
    candidate_manifest_source_path: Path
    candidate_manifest_source_file_sha256: str
    candidate_manifest_output_path: Path
    candidate_manifest_output_file_sha256: str
    candidate_manifest_output_mode: str
    provider_plan_fragment_path: Path
    provider_plan_fragment_file_sha256: str
    provider_plan_fragment_mode: str
    registration_bundle_paths: Mapping[ProviderPhase, str]
    registration_bundle_sha256s: Mapping[ProviderPhase, str]
    registration_evidence_file_sha256s: Mapping[ProviderPhase, str]
    registration_receipt_paths: Mapping[ProviderPhase, str]
    registration_receipt_file_sha256s: Mapping[ProviderPhase, str]
    candidate_loader_witness_commit: str
    raw_provider_plan_templates_sha256: str
    witness_normalized_provider_plan_closure_sha256: str
    typed_candidate_loader_verified: bool
    source_manifest_unchanged: bool
    schema_version: str = PROVIDER_PLAN_FINALIZATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "blueprint_path",
            "blueprint_write_receipt_path",
            "candidate_manifest_source_path",
            "candidate_manifest_output_path",
            "provider_plan_fragment_path",
        ):
            object.__setattr__(self, name, _absolute_path(name, getattr(self, name)))
        for name in (
            "blueprint_file_sha256",
            "blueprint_write_receipt_file_sha256",
            "candidate_manifest_source_file_sha256",
            "candidate_manifest_output_file_sha256",
            "provider_plan_fragment_file_sha256",
            "raw_provider_plan_templates_sha256",
            "witness_normalized_provider_plan_closure_sha256",
        ):
            _digest(name, getattr(self, name))
        _commit("candidate_loader_witness_commit", self.candidate_loader_witness_commit)
        for name in (
            "registration_bundle_paths",
            "registration_bundle_sha256s",
            "registration_evidence_file_sha256s",
            "registration_receipt_paths",
            "registration_receipt_file_sha256s",
        ):
            if set(getattr(self, name)) != set(PHASES):
                raise ProviderPlanOperatorError(
                    "finalization receipt lacks exactly three runner registration bundles"
                )
        for phase in PHASES:
            bundle = _absolute_path(
                f"{phase} registration bundle path",
                self.registration_bundle_paths[phase],
            )
            receipt_path = _absolute_path(
                f"{phase} registration receipt path",
                self.registration_receipt_paths[phase],
            )
            if receipt_path != bundle / "registration-receipt.json":
                raise ProviderPlanOperatorError(f"{phase} registration receipt escapes its bundle")
            for name, values in (
                ("bundle", self.registration_bundle_sha256s),
                ("evidence", self.registration_evidence_file_sha256s),
                ("receipt", self.registration_receipt_file_sha256s),
            ):
                _digest(f"{phase} registration {name} SHA-256", values[phase])
        for name, values in (
            ("bundle path", set(self.registration_bundle_paths.values())),
            ("bundle digest", set(self.registration_bundle_sha256s.values())),
            (
                "evidence digest",
                set(self.registration_evidence_file_sha256s.values()),
            ),
            ("receipt path", set(self.registration_receipt_paths.values())),
            (
                "receipt digest",
                set(self.registration_receipt_file_sha256s.values()),
            ),
        ):
            if len(values) != len(PHASES):
                raise ProviderPlanOperatorError(
                    f"finalization receipt reuses one registration {name}"
                )
        if (
            self.candidate_manifest_output_mode != "0600"
            or self.provider_plan_fragment_mode != "0600"
        ):
            raise ProviderPlanOperatorError("finalized provider-plan outputs must use mode 0600")
        if (
            self.typed_candidate_loader_verified is not True
            or self.source_manifest_unchanged is not True
        ):
            raise ProviderPlanOperatorError("provider-plan finalization lacks typed readback")
        if self.schema_version != PROVIDER_PLAN_FINALIZATION_RECEIPT_SCHEMA:
            raise ProviderPlanOperatorError("provider-plan finalization receipt schema differs")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, Path):
                payload[name] = str(value)
            elif name in {
                "registration_bundle_paths",
                "registration_bundle_sha256s",
                "registration_evidence_file_sha256s",
                "registration_receipt_paths",
                "registration_receipt_file_sha256s",
            }:
                payload[name] = dict(value)
            else:
                payload[name] = value
        return payload

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProviderPlanFinalizationReceipt:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="provider-plan finalization receipt",
            )
        )  # type: ignore[arg-type]


def load_provider_plan_blueprint(path: str | Path) -> ProviderPlanBlueprint:
    row, encoded = _read_json(path, label="provider-plan blueprint")
    blueprint = ProviderPlanBlueprint.from_dict(row)
    if encoded != blueprint.canonical_file_bytes():
        raise ProviderPlanOperatorError("provider-plan blueprint typed readback differs")
    return blueprint


def load_provider_plan_blueprint_write_receipt(
    path: str | Path,
) -> ProviderPlanBlueprintWriteReceipt:
    row, encoded = _read_json(path, label="provider-plan blueprint write receipt")
    receipt = ProviderPlanBlueprintWriteReceipt.from_dict(row)
    if encoded != receipt.canonical_file_bytes():
        raise ProviderPlanOperatorError("provider-plan blueprint write receipt differs")
    return receipt


def load_provider_plan_finalization_receipt(
    path: str | Path,
) -> ProviderPlanFinalizationReceipt:
    row, encoded = _read_json(path, label="provider-plan finalization receipt")
    receipt = ProviderPlanFinalizationReceipt.from_dict(row)
    if encoded != receipt.canonical_file_bytes():
        raise ProviderPlanOperatorError("provider-plan finalization receipt differs")
    return receipt


def _read_closed_bundle(
    directory: str | Path,
    names: frozenset[str],
    *,
    label: str,
) -> tuple[Path, Mapping[str, bytes]]:
    root = _absolute_path(label, directory)
    parent_descriptor = _open_private_parent(root, label=label)
    directory_descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory_descriptor = os.open(root.name, flags, dir_fd=parent_descriptor)
        before = os.fstat(directory_descriptor)
        named_before = os.stat(
            root.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(before.st_mode)
            or (before.st_dev, before.st_ino) != (named_before.st_dev, named_before.st_ino)
            or stat.S_IMODE(before.st_mode) != 0o700
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            or set(os.listdir(directory_descriptor)) != names
        ):
            raise ProviderPlanOperatorError(f"{label} directory identity or member set differs")
        encoded = {
            name: _read_bundle_member(
                directory_descriptor,
                name,
                label=f"{label} member {name}",
            )
            for name in sorted(names)
        }
        after = os.fstat(directory_descriptor)
        named_after = os.stat(
            root.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            after_identity != before_identity
            or (after.st_dev, after.st_ino) != (named_after.st_dev, named_after.st_ino)
            or set(os.listdir(directory_descriptor)) != names
        ):
            raise ProviderPlanOperatorError(f"{label} changed during fd-bound readback")
        return root, encoded
    except ProviderPlanOperatorError:
        raise
    except OSError as exc:
        raise ProviderPlanOperatorError(f"cannot read {label}") from exc
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(parent_descriptor)


def load_provider_plan_blueprint_bundle(
    directory: str | Path,
) -> tuple[ProviderPlanBlueprint, ProviderPlanBlueprintWriteReceipt]:
    """Read the closed two-member blueprint bundle."""

    root, encoded = _read_closed_bundle(
        directory,
        frozenset(
            {
                PROVIDER_PLAN_BLUEPRINT_FILENAME,
                PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_FILENAME,
            }
        ),
        label="provider-plan blueprint bundle",
    )
    blueprint_bytes = encoded[PROVIDER_PLAN_BLUEPRINT_FILENAME]
    receipt_bytes = encoded[PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_FILENAME]
    blueprint = ProviderPlanBlueprint.from_dict(
        _parse_object(blueprint_bytes, label="provider-plan blueprint")
    )
    receipt = ProviderPlanBlueprintWriteReceipt.from_dict(
        _parse_object(
            receipt_bytes,
            label="provider-plan blueprint write receipt",
        )
    )
    if (
        blueprint_bytes != blueprint.canonical_file_bytes()
        or receipt_bytes != receipt.canonical_file_bytes()
        or receipt.blueprint_path != root / PROVIDER_PLAN_BLUEPRINT_FILENAME
        or receipt.blueprint_file_sha256 != blueprint.file_sha256
        or receipt.blueprint_readback_sha256 != blueprint.file_sha256
        or receipt.blueprint_byte_count != len(blueprint.canonical_file_bytes())
    ):
        raise ProviderPlanOperatorError("provider-plan blueprint bundle closure differs")
    return blueprint, receipt


def load_provider_plan_finalization_bundle(
    directory: str | Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], ProviderPlanFinalizationReceipt]:
    """Read the closed three-member pre-A finalization bundle."""

    root, encoded = _read_closed_bundle(
        directory,
        frozenset(
            {
                PROVIDER_PLAN_CANDIDATE_MANIFEST_FILENAME,
                PROVIDER_PLAN_FINALIZATION_RECEIPT_FILENAME,
                PROVIDER_PLAN_FRAGMENT_FILENAME,
            }
        ),
        label="provider-plan finalization bundle",
    )
    candidate_encoded = encoded[PROVIDER_PLAN_CANDIDATE_MANIFEST_FILENAME]
    fragment_encoded = encoded[PROVIDER_PLAN_FRAGMENT_FILENAME]
    receipt_encoded = encoded[PROVIDER_PLAN_FINALIZATION_RECEIPT_FILENAME]
    candidate = _parse_object(candidate_encoded, label="derived candidate manifest")
    templates = _parse_object(
        fragment_encoded,
        label="provider phase-plan fragment",
    )
    receipt = ProviderPlanFinalizationReceipt.from_dict(
        _parse_object(
            receipt_encoded,
            label="provider-plan finalization receipt",
        )
    )
    sealed = candidate.get("sealed_execution")
    registration_bindings_match = all(
        isinstance(templates.get(phase), Mapping)
        and templates[phase].get("runner_registration_bundle_path")
        == receipt.registration_bundle_paths[phase]
        and templates[phase].get("runner_registration_bundle_sha256")
        == receipt.registration_bundle_sha256s[phase]
        and templates[phase].get("runner_registration_evidence_file_sha256")
        == receipt.registration_evidence_file_sha256s[phase]
        for phase in PHASES
    )
    if (
        receipt_encoded != receipt.canonical_file_bytes()
        or receipt.candidate_manifest_output_path
        != root / PROVIDER_PLAN_CANDIDATE_MANIFEST_FILENAME
        or receipt.provider_plan_fragment_path != root / PROVIDER_PLAN_FRAGMENT_FILENAME
        or receipt.candidate_manifest_output_file_sha256 != _sha256_bytes(candidate_encoded)
        or receipt.provider_plan_fragment_file_sha256 != _sha256_bytes(fragment_encoded)
        or receipt.raw_provider_plan_templates_sha256 != _sha256_bytes(_canonical_bytes(templates))
        or not isinstance(sealed, Mapping)
        or sealed.get("provider_phase_plans") != templates
        or not registration_bindings_match
    ):
        raise ProviderPlanOperatorError("provider-plan finalization bundle closure differs")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".provider-plan-readback-",
            dir=root.parent,
        ) as temporary:
            candidate_copy = Path(temporary) / PROVIDER_PLAN_CANDIDATE_MANIFEST_FILENAME
            write_exclusive_receipt_bytes(candidate_encoded, candidate_copy)
            loaded = load_provider_phase_plans(
                candidate_copy,
                c1_commit=receipt.candidate_loader_witness_commit,
                validation_mode="candidate-rehearsal",
                c0_commit=receipt.candidate_loader_witness_commit,
            )
        normalized = provider_phase_plan_templates_sha256(
            candidate,
            validation_mode="candidate-rehearsal",
            c0_commit=receipt.candidate_loader_witness_commit,
        )
    except (ArtifactIntegrityError, ExecutionClaimError, OSError) as exc:
        raise ProviderPlanOperatorError(
            "provider-plan finalization bundle fails typed candidate readback"
        ) from exc
    if (
        tuple(loaded) != PHASES
        or normalized != receipt.witness_normalized_provider_plan_closure_sha256
    ):
        raise ProviderPlanOperatorError("provider-plan normalized readback differs")
    return candidate, templates, receipt


def _load_phase_host_probe(path: Path) -> PhaseHostProbe:
    row, encoded = _read_json(path, label="phase host probe")
    try:
        probe = PhaseHostProbe.from_dict(row)
    except ExecutionClaimError as exc:
        raise ProviderPlanOperatorError(f"phase host probe is invalid: {exc}") from exc
    if encoded != probe.canonical_file_bytes():
        raise ProviderPlanOperatorError("phase host probe typed readback differs")
    return probe


def _load_docker_server_probe(path: Path) -> DockerServerProbe:
    row, encoded = _read_json(path, label="Docker server probe")
    try:
        probe = DockerServerProbe.from_dict(row)
    except ExecutionClaimError as exc:
        raise ProviderPlanOperatorError(f"Docker server probe is invalid: {exc}") from exc
    if encoded != probe.canonical_file_bytes():
        raise ProviderPlanOperatorError("Docker server probe typed readback differs")
    return probe


def _load_execution_beacon_contract(path: Path) -> tuple[ExecutionBeaconContract, str]:
    row, encoded = _read_json(path, label="execution beacon contract")
    try:
        contract = ExecutionBeaconContract.from_dict(row)
    except ExecutionClaimError as exc:
        raise ProviderPlanOperatorError(f"execution beacon contract is invalid: {exc}") from exc
    if encoded != _canonical_bytes(contract.to_dict()) + b"\n":
        raise ProviderPlanOperatorError("execution beacon contract typed readback differs")
    return contract, _sha256_bytes(encoded)


def _canonical_regular_file_digest(path: Path, *, label: str) -> str:
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
        if resolved != path or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProviderPlanOperatorError(f"{label} must be one canonical regular file")
        return digest_regular_file(path, label=label)
    except (ArtifactIntegrityError, OSError) as exc:
        raise ProviderPlanOperatorError(f"cannot hash {label}: {exc}") from exc


def _derive_host_tool_contract(sources: HostToolSources) -> PhaseHostToolContract:
    try:
        controlled = sources.controlled_root.resolve(strict=True)
        controlled_metadata = sources.controlled_root.lstat()
    except OSError as exc:
        raise ProviderPlanOperatorError("controlled_root is unavailable") from exc
    if (
        controlled != sources.controlled_root
        or not stat.S_ISDIR(controlled_metadata.st_mode)
        or stat.S_ISLNK(controlled_metadata.st_mode)
    ):
        raise ProviderPlanOperatorError("controlled_root must be one canonical directory")
    try:
        venv = sources.venv_root.resolve(strict=True)
        venv.relative_to(controlled)
        venv_tree_sha256, venv_symlink_inventory_sha256 = execution_claim_module._venv_tree_digests(
            venv, controlled
        )
    except (ExecutionClaimError, OSError, ValueError) as exc:
        raise ProviderPlanOperatorError(f"cannot derive the controlled venv: {exc}") from exc

    digests = {
        "python": _canonical_regular_file_digest(
            sources.python_executable, label="controlled Python executable"
        ),
        "gh": _canonical_regular_file_digest(
            sources.gh_executable, label="controlled GitHub CLI executable"
        ),
        "runner_listener": _canonical_regular_file_digest(
            sources.runner_listener_executable,
            label="Actions runner listener",
        ),
        "runner_listener_dll": _canonical_regular_file_digest(
            sources.runner_listener_dll,
            label="Actions runner listener DLL",
        ),
        "runner_config": _canonical_regular_file_digest(
            sources.runner_config_executable,
            label="Actions runner config executable",
        ),
        "runner_run": _canonical_regular_file_digest(
            sources.runner_run_executable,
            label="Actions runner run executable",
        ),
    }
    try:
        docker_metadata = sources.docker_executable.lstat()
        if not stat.S_ISLNK(docker_metadata.st_mode):
            raise ProviderPlanOperatorError("Docker invocation path must be a symlink")
        docker_resolved = sources.docker_executable.resolve(strict=True)
    except OSError as exc:
        raise ProviderPlanOperatorError("cannot resolve the Docker invocation path") from exc
    docker_digest = _canonical_regular_file_digest(
        docker_resolved,
        label="resolved Docker executable",
    )
    host_probe = _load_phase_host_probe(sources.host_probe_path)
    docker_probe = _load_docker_server_probe(sources.docker_server_probe_path)
    try:
        return PhaseHostToolContract(
            controlled_root=str(controlled),
            python_archive_uri=OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI,
            python_archive_sha256=OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256,
            python_archive_byte_count=OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT,
            python_executable=str(sources.python_executable),
            python_version=OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION,
            python_executable_sha256=digests["python"],
            venv_root=str(venv),
            venv_tree_sha256=venv_tree_sha256,
            venv_symlink_inventory_sha256=venv_symlink_inventory_sha256,
            gh_archive_uri=OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI,
            gh_archive_sha256=OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256,
            gh_archive_byte_count=OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT,
            gh_executable=str(sources.gh_executable),
            gh_executable_sha256=digests["gh"],
            gh_version=OFFICIAL_GH_VERSION,
            runner_archive_uri=OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI,
            runner_archive_sha256=OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
            runner_archive_byte_count=OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT,
            runner_listener_executable=str(sources.runner_listener_executable),
            runner_listener_sha256=digests["runner_listener"],
            runner_listener_dll=str(sources.runner_listener_dll),
            runner_listener_dll_sha256=digests["runner_listener_dll"],
            runner_config_executable=str(sources.runner_config_executable),
            runner_config_sha256=digests["runner_config"],
            runner_run_executable=str(sources.runner_run_executable),
            runner_run_sha256=digests["runner_run"],
            runner_version=OFFICIAL_ACTIONS_RUNNER_VERSION,
            runner_ephemeral=True,
            runner_disable_update=True,
            runner_unattended=True,
            docker_executable=str(sources.docker_executable),
            docker_resolved_executable=str(docker_resolved),
            docker_executable_sha256=docker_digest,
            docker_client_version=REGISTERED_DOCKER_CLIENT_VERSION,
            docker_client_build=REGISTERED_DOCKER_CLIENT_BUILD,
            host_probe=host_probe,
            docker_server_probe=docker_probe,
            host_probe_receipt_sha256=host_probe.file_sha256,
            docker_server_probe_receipt_sha256=docker_probe.file_sha256,
            host_operating_system="macOS",
            host_architecture="ARM64",
        )
    except ExecutionClaimError as exc:
        raise ProviderPlanOperatorError(f"derived host-tool contract is invalid: {exc}") from exc


def _sentinel_paths(value: object, *, path: str = "") -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            result.update(_sentinel_paths(item, path=child))
    elif isinstance(value, list):
        for position, item in enumerate(value):
            result.update(_sentinel_paths(item, path=f"{path}[{position}]"))
    elif value == C0_COMMIT_SENTINEL:
        result.add(path)
    return result


def _source_candidate_sentinel_paths(payload: Mapping[str, Any]) -> set[str]:
    artifacts = payload.get("artifacts")
    workloads = payload.get("production_workloads")
    if not isinstance(artifacts, list) or not isinstance(workloads, list):
        raise ProviderPlanOperatorError(
            "candidate source requires resolved artifacts and production workloads"
        )
    source_positions = [
        position
        for position, item in enumerate(artifacts)
        if isinstance(item, Mapping) and item.get("role") == "source-code"
    ]
    if len(source_positions) != 1:
        raise ProviderPlanOperatorError(
            "candidate source requires exactly one source-code artifact"
        )
    if len(workloads) != len(FIXED_CORPORA):
        raise ProviderPlanOperatorError("candidate source requires five production workloads")
    expected = {
        "sealed_execution.code_commit",
        f"artifacts[{source_positions[0]}].revision",
    }
    for position, corpus_id in enumerate(FIXED_CORPORA):
        item = workloads[position]
        if not isinstance(item, Mapping) or item.get("corpus_id") != corpus_id:
            raise ProviderPlanOperatorError("candidate workloads differ from fixed corpus order")
        spec = item.get("spec")
        if not isinstance(spec, Mapping) or spec.get("corpus_id") != corpus_id:
            raise ProviderPlanOperatorError(f"candidate workload lacks the {corpus_id} spec")
        expected.add(f"production_workloads[{position}].spec.code_commit")
    return expected


def _artifact_by_role(payload: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ProviderPlanOperatorError("candidate artifacts must be an array")
    rows = [item for item in artifacts if isinstance(item, Mapping) and item.get("role") == role]
    if len(rows) != 1:
        raise ProviderPlanOperatorError(f"candidate manifest requires one {role!r} artifact")
    return rows[0]


def _admit_candidate_source_shell(
    payload: Mapping[str, Any],
    *,
    candidate_image_source_commit: str,
    config: Any,
    control_blueprint: Any,
) -> None:
    if payload.get("status") != "draft" or payload.get("protocol_version") != "0.3.0-draft":
        raise ProviderPlanOperatorError("provider-plan source requires draft/0.3.0-draft lifecycle")
    blockers = payload.get("freeze_blockers")
    if not isinstance(blockers, list) or not blockers:
        raise ProviderPlanOperatorError("provider-plan source requires explicit freeze blockers")
    sealed = payload.get("sealed_execution")
    if not isinstance(sealed, Mapping):
        raise ProviderPlanOperatorError("candidate source lacks sealed_execution")
    if sealed.get("c0_evidence_release") != "tbd":
        raise ProviderPlanOperatorError("candidate source C0 evidence release must remain tbd")
    if sealed.get("provider_phase_plans") != "tbd":
        raise ProviderPlanOperatorError(
            "candidate source provider_phase_plans must be the sole unresolved tbd field"
        )
    expected_sentinels = _source_candidate_sentinel_paths(payload)
    observed_sentinels = _sentinel_paths(payload)
    if observed_sentinels != expected_sentinels:
        raise ProviderPlanOperatorError(
            "candidate source C0 sentinel path set differs; "
            f"missing={sorted(expected_sentinels - observed_sentinels)}, "
            f"unexpected={sorted(observed_sentinels - expected_sentinels)}"
        )
    if (
        config.candidate_image_source_commit != candidate_image_source_commit
        or control_blueprint.candidate_image_source_commit != candidate_image_source_commit
    ):
        raise ProviderPlanOperatorError(
            "production-control candidate image source commit differs from P"
        )
    if (
        sealed.get("approval_environment") != config.approval_environment
        or config.approval_environment != PROVIDER_APPROVAL_ENVIRONMENT
        or sealed.get("runner_identity") != config.runner_identity
        or config.runner_identity != PROVIDER_RUNNER_IDENTITY
        or config.runner_identity != f"github-actions:environment:{config.approval_environment}"
        or sealed.get("runner_image") != config.scientific_production_reference
        or control_blueprint.runner_image != config.scientific_production_reference
    ):
        raise ProviderPlanOperatorError(
            "candidate runner identity or production image differs from production controls"
        )
    controls = sealed.get("production_controls")
    expected_controls = {
        "materialization_config_file_sha256": config.file_sha256,
        "blueprint_receipt_sha256": control_blueprint.semantic_sha256,
        "blueprint_receipt_file_sha256": control_blueprint.file_sha256,
    }
    if controls != expected_controls:
        raise ProviderPlanOperatorError(
            "candidate production_controls differs from typed config and blueprint readback"
        )
    workloads = payload.get("production_workloads")
    assert isinstance(workloads, list)
    for position, corpus_id in enumerate(FIXED_CORPORA):
        row = workloads[position]
        spec = row["spec"]
        if (
            spec.get("runner_identity") != config.runner_identity
            or spec.get("runner_image") != config.scientific_production_reference
            or spec.get("code_commit") != C0_COMMIT_SENTINEL
            or row.get("canonical_file_sha256") != production_workload_file_sha256(spec)
        ):
            raise ProviderPlanOperatorError(
                f"{corpus_id} workload differs from the raw candidate template"
            )
    timelock = _artifact_by_role(payload, "timelock-tool")
    if timelock.get("sha256") != SOURCE_BUILT_LINUX_ARM64_TLE_SHA256:
        raise ProviderPlanOperatorError("candidate timelock-tool artifact differs from C0")


def _release_production_reference(closure: CandidateImageClosure) -> str:
    prefix = "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-release-candidate@"
    if not closure.release_image_reference.startswith(prefix):
        raise ProviderPlanOperatorError("candidate release image uses another repository")
    return (
        "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-release@"
        f"{closure.release_image_index_digest}"
    )


def _claim_nonce(
    *,
    phase: ProviderPhase,
    approval_environment: str,
    candidate_source_shell_sha256: str,
    candidate_image_source_commit: str,
    build_context_tree_sha256: str,
    production_control_config_sha256: str,
    candidate_bootstrap_closure_sha256: str,
    scientific_index_digest: str,
    release_index_digest: str,
    host_tool_contract_sha256: str,
    runner_name: str,
    runner_group_id: int | None,
    runner_identity: str,
    claim_root: str,
    evidence_root: str,
) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "approval_environment": approval_environment,
                "candidate_bootstrap_closure_sha256": candidate_bootstrap_closure_sha256,
                "candidate_image_source_commit": candidate_image_source_commit,
                "candidate_source_shell_sha256": candidate_source_shell_sha256,
                "claim_root": claim_root,
                "derivation": PROVIDER_PLAN_CLAIM_NONCE_DERIVATION,
                "evidence_root": evidence_root,
                "host_tool_contract_sha256": host_tool_contract_sha256,
                "phase": phase,
                "production_control_config_sha256": production_control_config_sha256,
                "release_index_digest": release_index_digest,
                "runner_group_id": runner_group_id,
                "runner_identity": runner_identity,
                "runner_name": runner_name,
                "scientific_index_digest": scientific_index_digest,
                "build_context_tree_sha256": build_context_tree_sha256,
            }
        )
    )


def _input_file_sha256(path: Path, *, label: str) -> str:
    try:
        return digest_regular_file(path, label=label)
    except ArtifactIntegrityError as exc:
        raise ProviderPlanOperatorError(f"cannot hash {label}: {exc}") from exc


def _assert_typed_file_bytes(value: object, encoded: bytes, *, label: str) -> None:
    method = getattr(value, "canonical_file_bytes", None)
    if not callable(method):
        raise ProviderPlanOperatorError(f"typed {label} lacks canonical bytes")
    try:
        observed = method()
    except Exception as exc:  # pragma: no cover - typed implementations own detail
        raise ProviderPlanOperatorError(f"cannot encode typed {label}") from exc
    if type(observed) is not bytes or observed != encoded:
        raise ProviderPlanOperatorError(f"typed {label} differs from admitted bytes")


def _type_exact_file_bytes(
    source_path: Path,
    encoded: bytes,
    *,
    label: str,
    loader: Callable[[Path], object],
) -> object:
    try:
        parent = source_path.parent.resolve(strict=True)
    except OSError as exc:
        raise ProviderPlanOperatorError(f"{label} parent is unavailable") from exc
    if parent != source_path.parent:
        raise ProviderPlanOperatorError(f"{label} parent cannot contain symlinks")
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{source_path.name}.typing-",
            dir=parent,
        ) as temporary:
            copy_path = Path(temporary) / source_path.name
            write_exclusive_receipt_bytes(encoded, copy_path)
            value = loader(copy_path)
    except (ArtifactIntegrityError, OSError, ValueError) as exc:
        raise ProviderPlanOperatorError(f"cannot type exact {label} bytes: {exc}") from exc
    _assert_typed_file_bytes(value, encoded, label=label)
    return value


def _candidate_image_closure_from_exact_bytes(
    row: Mapping[str, Any],
    encoded: bytes,
) -> CandidateImageClosure:
    fields = frozenset(CandidateImageClosure.__dataclass_fields__)
    try:
        closure = CandidateImageClosure(**_closed(row, fields, label="candidate image closure"))
    except (TypeError, ValueError) as exc:
        raise ProviderPlanOperatorError(f"candidate image closure is invalid: {exc}") from exc
    if encoded != _canonical_bytes(closure.to_dict()) + b"\n":
        raise ProviderPlanOperatorError("candidate image closure typed bytes differ")
    return closure


def write_provider_plan_blueprint(
    *,
    candidate_manifest_path: str | Path,
    production_control_config_path: str | Path,
    production_control_config_write_receipt_path: str | Path,
    candidate_image_closure_path: str | Path,
    execution_beacon_contract_path: str | Path,
    registered_online_runtime_budget_seconds: int,
    host_tool_sources: HostToolSources,
    claim_root: str | Path,
    evidence_root: str | Path,
    runner_names: Mapping[ProviderPhase, str],
    output_directory: str | Path,
) -> ProviderPlanBlueprintWriteReceipt:
    """Admit typed pre-C0 inputs and write one immutable provider-plan blueprint."""

    if not isinstance(host_tool_sources, HostToolSources):
        raise ProviderPlanOperatorError("host_tool_sources must be typed")
    if set(runner_names) != set(PHASES):
        raise ProviderPlanOperatorError("runner_names must contain exactly three fixed phases")
    names = {phase: _text(f"{phase} runner name", runner_names[phase]) for phase in PHASES}
    if len(set(names.values())) != len(PHASES):
        raise ProviderPlanOperatorError("runner names must be distinct")
    claim = _canonical_posix_path("claim_root", claim_root)
    evidence = _canonical_posix_path("evidence_root", evidence_root)
    bundle_path = _absolute_path("output_directory", output_directory)
    output_path = bundle_path / PROVIDER_PLAN_BLUEPRINT_FILENAME
    if os.path.lexists(bundle_path):
        raise ProviderPlanOperatorError("provider-plan blueprint output already exists")

    manifest_path = _absolute_path("candidate_manifest_path", candidate_manifest_path)
    config_path = _absolute_path("production_control_config_path", production_control_config_path)
    config_receipt_path = _absolute_path(
        "production_control_config_write_receipt_path",
        production_control_config_write_receipt_path,
    )
    closure_path = _absolute_path("candidate_image_closure_path", candidate_image_closure_path)
    beacon_path = _absolute_path("execution_beacon_contract_path", execution_beacon_contract_path)
    inputs = {
        manifest_path,
        config_path,
        config_receipt_path,
        closure_path,
        beacon_path,
        *(
            getattr(host_tool_sources, name)
            for name in host_tool_sources.__dataclass_fields__
            if name not in {"controlled_root", "venv_root"}
        ),
    }
    if any(_paths_overlap(bundle_path, source) for source in inputs):
        raise ProviderPlanOperatorError("provider-plan bundle overlaps one admitted input")

    try:
        manifest, manifest_bytes = _read_json(
            manifest_path,
            label="candidate provider-plan source manifest",
        )
        manifest_sha256 = _sha256_bytes(manifest_bytes)

        config_bytes = read_secure_regular_file(
            config_path,
            max_bytes=_MAX_JSON_BYTES,
            label="production control config",
        )
        config_sha256 = _sha256_bytes(config_bytes)
        config = _type_exact_file_bytes(
            config_path,
            config_bytes,
            label="production control config",
            loader=lambda path: load_production_control_config(
                path,
                expected_sha256=config_sha256,
            ),
        )
        _config_receipt_row, config_receipt_bytes = _read_json(
            config_receipt_path,
            label="production control config write receipt",
        )
        config_receipt = _type_exact_file_bytes(
            config_receipt_path,
            config_receipt_bytes,
            label="production control config write receipt",
            loader=load_production_control_config_write_receipt,
        )
        config_receipt_sha256 = _sha256_bytes(config_receipt_bytes)
        if (
            config_receipt.config_path != config_path
            or config_receipt.config_file_sha256 != config_sha256
            or config_receipt.config_readback_sha256 != config_sha256
            or config_receipt.candidate_image_source_commit != config.candidate_image_source_commit
            or config_receipt.approval_environment != config.approval_environment
        ):
            raise ProviderPlanOperatorError(
                "production-control config differs from its typed write receipt"
            )
        control_blueprint_path = config.blueprint_receipt_path
        _control_row, control_blueprint_bytes = _read_json(
            control_blueprint_path,
            label="production control blueprint receipt",
        )
        control_blueprint = _type_exact_file_bytes(
            control_blueprint_path,
            control_blueprint_bytes,
            label="production control blueprint receipt",
            loader=load_production_control_blueprint_receipt,
        )
        if (
            control_blueprint.materialization_config_sha256 != config_sha256
            or control_blueprint.approval_environment != config.approval_environment
            or control_blueprint.runner_image != config.scientific_production_reference
            or control_blueprint.file_sha256 != _sha256_bytes(control_blueprint_bytes)
        ):
            raise ProviderPlanOperatorError(
                "production-control blueprint differs from its materialization config"
            )

        _closure_row, closure_bytes = _read_json(
            closure_path,
            label="candidate image closure",
        )
        closure = _candidate_image_closure_from_exact_bytes(_closure_row, closure_bytes)
        closure_sha256 = _sha256_bytes(closure_bytes)
        if (
            config.scientific_candidate_reference != closure.scientific_image_reference
            or config.candidate_image_source_commit != closure.github_sha
            or config.scientific_index_digest != closure.scientific_image_index_digest
            or config.scientific_production_reference.rsplit("@", 1)[1]
            != closure.scientific_image_index_digest
        ):
            raise ProviderPlanOperatorError(
                "scientific candidate, production control, and OCI closure differ"
            )
        release_production_reference = _release_production_reference(closure)

        beacon, beacon_file_sha256 = _load_execution_beacon_contract(beacon_path)
        factory_bytes = read_secure_regular_file(
            config.factory_config_path,
            max_bytes=_MAX_JSON_BYTES,
            label="production artifact factory config",
        )
        if _sha256_bytes(factory_bytes) != config.factory_config_sha256:
            raise ProviderPlanOperatorError("production factory config differs from its pin")
        factory = _type_exact_file_bytes(
            config.factory_config_path,
            factory_bytes,
            label="production artifact factory config",
            loader=lambda path: load_production_artifact_factory_config(
                path,
                expected_sha256=config.factory_config_sha256,
            ),
        )
        execution_inputs = ExecutionClaimInputs(
            design_seed_sha256=factory.design_seed_sha256,
            registered_online_runtime_budget_seconds=(registered_online_runtime_budget_seconds),
            beacon=beacon,
        )
        host_tools = _derive_host_tool_contract(host_tool_sources)
        _admit_candidate_source_shell(
            manifest,
            candidate_image_source_commit=closure.github_sha,
            config=config,
            control_blueprint=control_blueprint,
        )
    except ProviderPlanOperatorError:
        raise
    except (
        ArtifactIntegrityError,
        ExecutionClaimError,
        ProductionArtifactFactoryError,
        ProductionControlError,
        OSError,
        ValueError,
    ) as exc:
        raise ProviderPlanOperatorError(f"cannot admit provider-plan inputs: {exc}") from exc

    expectations: list[ProviderRunnerExpectation] = []
    for phase in PHASES:
        nonce = _claim_nonce(
            phase=phase,
            approval_environment=config.approval_environment,
            candidate_source_shell_sha256=manifest_sha256,
            candidate_image_source_commit=closure.github_sha,
            build_context_tree_sha256=closure.build_context_tree_sha256,
            production_control_config_sha256=config_sha256,
            candidate_bootstrap_closure_sha256=closure.bootstrap_closure_sha256,
            scientific_index_digest=closure.scientific_image_index_digest,
            release_index_digest=closure.release_image_index_digest,
            host_tool_contract_sha256=host_tools.contract_sha256,
            runner_name=names[phase],
            runner_group_id=None,
            runner_identity=config.runner_identity,
            claim_root=claim,
            evidence_root=evidence,
        )
        expectations.append(
            ProviderRunnerExpectation(
                phase=phase,
                runner_name=names[phase],
                runner_group_id=None,
                claim_nonce=nonce,
                runner_label=derive_phase_runner_label(nonce, phase),
            )
        )

    blueprint = ProviderPlanBlueprint(
        candidate_manifest_path=manifest_path,
        candidate_manifest_file_sha256=manifest_sha256,
        production_control_config_path=config_path,
        production_control_config_file_sha256=config_sha256,
        production_control_config_write_receipt_path=config_receipt_path,
        production_control_config_write_receipt_file_sha256=config_receipt_sha256,
        production_control_blueprint_receipt_path=control_blueprint_path,
        production_control_blueprint_receipt_file_sha256=control_blueprint.file_sha256,
        production_control_blueprint_receipt_sha256=control_blueprint.semantic_sha256,
        candidate_image_closure_path=closure_path,
        candidate_image_closure_file_sha256=closure_sha256,
        candidate_image_source_commit=closure.github_sha,
        build_context_tree_sha256=closure.build_context_tree_sha256,
        candidate_bootstrap_closure_sha256=closure.bootstrap_closure_sha256,
        scientific_candidate_reference=closure.scientific_image_reference,
        scientific_production_reference=config.scientific_production_reference,
        release_candidate_reference=closure.release_image_reference,
        release_production_reference=release_production_reference,
        approval_environment=config.approval_environment,
        runner_identity=config.runner_identity,
        host_tool_sources=host_tool_sources,
        host_tools=host_tools,
        execution_beacon_contract_path=beacon_path,
        execution_beacon_contract_file_sha256=beacon_file_sha256,
        execution_claim_inputs=execution_inputs,
        claim_root=claim,
        evidence_root=evidence,
        runner_expectations=tuple(expectations),
    )
    encoded = blueprint.canonical_file_bytes()
    receipt = ProviderPlanBlueprintWriteReceipt(
        blueprint_path=output_path,
        blueprint_file_sha256=blueprint.file_sha256,
        blueprint_readback_sha256=blueprint.file_sha256,
        blueprint_byte_count=len(encoded),
        blueprint_mode="0600",
        candidate_manifest_file_sha256=manifest_sha256,
        production_control_config_file_sha256=config_sha256,
        production_control_config_write_receipt_file_sha256=config_receipt_sha256,
        production_control_blueprint_receipt_file_sha256=control_blueprint.file_sha256,
        candidate_image_closure_file_sha256=closure_sha256,
        execution_beacon_contract_file_sha256=beacon_file_sha256,
        host_tool_contract_sha256=host_tools.contract_sha256,
        readback_verified=True,
    )
    _publish_private_bundle(
        bundle_path,
        {
            PROVIDER_PLAN_BLUEPRINT_FILENAME: encoded,
            PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_FILENAME: receipt.canonical_file_bytes(),
        },
        label="provider-plan blueprint bundle",
        pre_publish=lambda: _revalidate_blueprint_sources(blueprint),
    )
    if load_provider_plan_blueprint_bundle(bundle_path) != (blueprint, receipt):
        raise ProviderPlanOperatorError("provider-plan blueprint receipt readback differs")
    return receipt


def _revalidate_blueprint_sources(
    blueprint: ProviderPlanBlueprint,
) -> tuple[Mapping[str, Any], CandidateImageClosure, Any, PhaseHostToolContract]:
    manifest, manifest_bytes = _read_json(
        blueprint.candidate_manifest_path,
        label="candidate provider-plan source manifest",
    )
    if _sha256_bytes(manifest_bytes) != blueprint.candidate_manifest_file_sha256:
        raise ProviderPlanOperatorError("candidate source manifest changed after blueprint")

    config_bytes = read_secure_regular_file(
        blueprint.production_control_config_path,
        max_bytes=_MAX_JSON_BYTES,
        label="production control config",
    )
    config_digest = _sha256_bytes(config_bytes)
    if config_digest != blueprint.production_control_config_file_sha256:
        raise ProviderPlanOperatorError("production control config changed after blueprint")
    try:
        config = _type_exact_file_bytes(
            blueprint.production_control_config_path,
            config_bytes,
            label="production control config",
            loader=lambda path: load_production_control_config(
                path,
                expected_sha256=config_digest,
            ),
        )
        _config_receipt_row, config_receipt_bytes = _read_json(
            blueprint.production_control_config_write_receipt_path,
            label="production control config write receipt",
        )
        config_receipt_digest = _sha256_bytes(config_receipt_bytes)
        config_receipt = _type_exact_file_bytes(
            blueprint.production_control_config_write_receipt_path,
            config_receipt_bytes,
            label="production control config write receipt",
            loader=load_production_control_config_write_receipt,
        )
        if (
            config_receipt_digest != blueprint.production_control_config_write_receipt_file_sha256
            or config_receipt.config_path != blueprint.production_control_config_path
            or config_receipt.config_file_sha256 != config_digest
            or config_receipt.config_readback_sha256 != config_digest
            or config_receipt.candidate_image_source_commit
            != blueprint.candidate_image_source_commit
            or config_receipt.approval_environment != blueprint.approval_environment
        ):
            raise ProviderPlanOperatorError(
                "production control config receipt changed after blueprint"
            )
        _control_row, control_blueprint_bytes = _read_json(
            blueprint.production_control_blueprint_receipt_path,
            label="production control blueprint receipt",
        )
        control_blueprint = _type_exact_file_bytes(
            blueprint.production_control_blueprint_receipt_path,
            control_blueprint_bytes,
            label="production control blueprint receipt",
            loader=load_production_control_blueprint_receipt,
        )
        if (
            control_blueprint.file_sha256
            != blueprint.production_control_blueprint_receipt_file_sha256
            or control_blueprint.file_sha256 != _sha256_bytes(control_blueprint_bytes)
            or control_blueprint.semantic_sha256
            != blueprint.production_control_blueprint_receipt_sha256
            or control_blueprint.materialization_config_sha256 != config_digest
            or config.candidate_image_source_commit != blueprint.candidate_image_source_commit
            or control_blueprint.candidate_image_source_commit
            != blueprint.candidate_image_source_commit
            or control_blueprint.approval_environment != blueprint.approval_environment
            or control_blueprint.runner_image != blueprint.scientific_production_reference
            or config.approval_environment != blueprint.approval_environment
            or config.runner_identity != blueprint.runner_identity
            or blueprint.approval_environment != PROVIDER_APPROVAL_ENVIRONMENT
            or blueprint.runner_identity != PROVIDER_RUNNER_IDENTITY
            or blueprint.runner_identity
            != f"github-actions:environment:{blueprint.approval_environment}"
        ):
            raise ProviderPlanOperatorError(
                "production control blueprint changed after provider-plan blueprint"
            )
        _closure_row, closure_bytes = _read_json(
            blueprint.candidate_image_closure_path,
            label="candidate image closure",
        )
        closure = _candidate_image_closure_from_exact_bytes(_closure_row, closure_bytes)
        if (
            _sha256_bytes(closure_bytes) != blueprint.candidate_image_closure_file_sha256
            or closure.github_sha != blueprint.candidate_image_source_commit
            or closure.build_context_tree_sha256 != blueprint.build_context_tree_sha256
            or closure.bootstrap_closure_sha256 != blueprint.candidate_bootstrap_closure_sha256
            or closure.scientific_image_reference != blueprint.scientific_candidate_reference
            or closure.release_image_reference != blueprint.release_candidate_reference
            or config.scientific_production_reference != blueprint.scientific_production_reference
            or _release_production_reference(closure) != blueprint.release_production_reference
        ):
            raise ProviderPlanOperatorError("candidate image closure changed after blueprint")
        beacon, beacon_digest = _load_execution_beacon_contract(
            blueprint.execution_beacon_contract_path
        )
        factory_bytes = read_secure_regular_file(
            config.factory_config_path,
            max_bytes=_MAX_JSON_BYTES,
            label="production artifact factory config",
        )
        if _sha256_bytes(factory_bytes) != config.factory_config_sha256:
            raise ProviderPlanOperatorError("production factory config changed after blueprint")
        factory = _type_exact_file_bytes(
            config.factory_config_path,
            factory_bytes,
            label="production artifact factory config",
            loader=lambda path: load_production_artifact_factory_config(
                path,
                expected_sha256=config.factory_config_sha256,
            ),
        )
        if (
            beacon_digest != blueprint.execution_beacon_contract_file_sha256
            or beacon != blueprint.execution_claim_inputs.beacon
            or factory.design_seed_sha256 != blueprint.execution_claim_inputs.design_seed_sha256
        ):
            raise ProviderPlanOperatorError("execution-claim inputs changed after blueprint")
        host_tools = _derive_host_tool_contract(blueprint.host_tool_sources)
        if host_tools != blueprint.host_tools:
            raise ProviderPlanOperatorError("host-tool closure changed after blueprint")
        _admit_candidate_source_shell(
            manifest,
            candidate_image_source_commit=blueprint.candidate_image_source_commit,
            config=config,
            control_blueprint=control_blueprint,
        )
    except ProviderPlanOperatorError:
        raise
    except (
        ArtifactIntegrityError,
        ExecutionClaimError,
        ProductionArtifactFactoryError,
        ProductionControlError,
        OSError,
        ValueError,
    ) as exc:
        raise ProviderPlanOperatorError(f"cannot revalidate provider-plan inputs: {exc}") from exc
    return manifest, closure, config, host_tools


def _admit_registration_bundles(
    blueprint: ProviderPlanBlueprint,
    blueprint_directory: Path,
) -> tuple[
    Mapping[ProviderPhase, ProviderRunnerBootstrapReceipt],
    Mapping[ProviderPhase, Path],
    Mapping[ProviderPhase, Path],
    Mapping[ProviderPhase, str],
    Mapping[ProviderPhase, str],
    Mapping[ProviderPhase, str],
]:
    from .provider_runner_activation import (
        ProviderRunnerActivationError,
        admit_provider_runner_registration,
    )

    expectations = {item.phase: item for item in blueprint.runner_expectations}
    receipts: dict[ProviderPhase, ProviderRunnerBootstrapReceipt] = {}
    bundle_paths: dict[ProviderPhase, Path] = {}
    receipt_paths: dict[ProviderPhase, Path] = {}
    receipt_digests: dict[ProviderPhase, str] = {}
    evidence_digests: dict[ProviderPhase, str] = {}
    bundle_digests: dict[ProviderPhase, str] = {}
    for phase in PHASES:
        expectation = expectations[phase]
        expected_root = (
            blueprint.host_tool_sources.controlled_root
            / "production"
            / "runner-registrations"
            / phase
            / expectation.runner_label
        )
        try:
            receipt, evidence, bundle_digest = admit_provider_runner_registration(
                blueprint_directory=blueprint_directory,
                phase=phase,
            )
        except ProviderRunnerActivationError as exc:
            raise ProviderPlanOperatorError(
                f"{phase} registration bundle fails typed admission: {exc}"
            ) from exc
        receipt_path = expected_root / "registration-receipt.json"
        if (
            Path(evidence.registration_receipt_path) != receipt_path
            or Path(evidence.registration_receipt_path).parent != expected_root
            or Path(evidence.blueprint_directory) != blueprint_directory
            or receipt.phase != phase
            or receipt.repository != REPOSITORY
            or receipt.approval_environment != blueprint.approval_environment
            or receipt.runner_identity != blueprint.runner_identity
            or receipt.runner_identity
            != f"github-actions:environment:{receipt.approval_environment}"
            or receipt.workflow_sha != blueprint.candidate_image_source_commit
            or receipt.runner_label != expectation.runner_label
            or receipt.runner_name != expectation.runner_name
            or receipt.runner_group_id != expectation.runner_group_id
            or receipt.runner_version != OFFICIAL_ACTIONS_RUNNER_VERSION
            or receipt.runner_archive_sha256 != OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256
            or (receipt.ephemeral, receipt.disable_update, receipt.unattended) != (True, True, True)
        ):
            raise ProviderPlanOperatorError(
                f"{phase} registration receipt differs from its runner expectation"
            )
        receipts[phase] = receipt
        bundle_paths[phase] = expected_root
        receipt_paths[phase] = receipt_path
        receipt_digests[phase] = receipt.file_sha256
        evidence_digests[phase] = evidence.file_sha256
        bundle_digests[phase] = bundle_digest
    for name, values in (
        ("runner ID", {receipt.runner_id for receipt in receipts.values()}),
        ("runner name", {receipt.runner_name for receipt in receipts.values()}),
        ("runner label", {receipt.runner_label for receipt in receipts.values()}),
        (
            "runner inventory",
            {receipt.repository_runner_inventory_sha256 for receipt in receipts.values()},
        ),
        ("bundle digest", set(bundle_digests.values())),
        ("evidence digest", set(evidence_digests.values())),
    ):
        if len(values) != len(PHASES):
            raise ProviderPlanOperatorError(f"registration bundles reuse one {name}")
    return (
        receipts,
        bundle_paths,
        receipt_paths,
        receipt_digests,
        evidence_digests,
        bundle_digests,
    )


def _phase_image_binding(
    phase: ProviderPhase,
    *,
    blueprint: ProviderPlanBlueprint,
    closure: CandidateImageClosure,
) -> tuple[str, str, str, str]:
    if phase == ONLINE_PHASE:
        return (
            blueprint.scientific_production_reference,
            closure.scientific_image_index_digest,
            closure.scientific_linux_arm64_manifest_digest,
            closure.scientific_linux_arm64_runtime_extraction_sha256,
        )
    if phase == ANALYSIS_PHASE:
        return (
            blueprint.scientific_production_reference,
            closure.scientific_image_index_digest,
            closure.scientific_linux_amd64_manifest_digest,
            closure.scientific_linux_amd64_runtime_extraction_sha256,
        )
    return (
        blueprint.release_production_reference,
        closure.release_image_index_digest,
        closure.release_linux_arm64_manifest_digest,
        closure.release_reproducibility_receipt_sha256,
    )


def _provider_plan_templates(
    blueprint: ProviderPlanBlueprint,
    closure: CandidateImageClosure,
    registrations: Mapping[ProviderPhase, ProviderRunnerBootstrapReceipt],
    registration_bundle_paths: Mapping[ProviderPhase, Path],
    registration_bundle_sha256s: Mapping[ProviderPhase, str],
    registration_evidence_file_sha256s: Mapping[ProviderPhase, str],
) -> dict[str, object]:
    expectations = {item.phase: item for item in blueprint.runner_expectations}
    result: dict[str, object] = {}
    for phase in PHASES:
        expectation = expectations[phase]
        registration = registrations[phase]
        workflow = PROVIDER_PHASE_WORKFLOWS[phase]
        claim_job, execute_job = PROVIDER_PHASE_JOB_NAMES[phase]
        platform, image_role, index_role = PROVIDER_PHASE_RUNTIME_BINDINGS[phase]
        provider_path = (
            blueprint.host_tool_sources.controlled_root
            / "production"
            / "provider-plans"
            / phase
            / "provider-plan.json"
        )
        runtime_image, index_digest, platform_digest, probe_digest = _phase_image_binding(
            phase,
            blueprint=blueprint,
            closure=closure,
        )
        embedded_bootstrap = registration.to_dict()
        embedded_bootstrap["workflow_sha"] = C0_COMMIT_SENTINEL
        embedded_bootstrap_sha256 = _sha256_bytes(_canonical_bytes(embedded_bootstrap) + b"\n")
        activation_receipt_path = (
            blueprint.host_tool_sources.controlled_root
            / "production"
            / "runners"
            / phase
            / expectation.runner_label
            / "bootstrap-receipt.json"
        )
        result[phase] = {
            "activation_argv_template": [
                blueprint.host_tools.python_executable,
                "-m",
                "fractal_ann_diagnostics.provider_phase_runtime",
                PROVIDER_PHASE_COMMAND_IDS[phase],
                "--provider-plan",
                str(provider_path),
                "--suite-attempt-id",
                PROVIDER_PLAN_SUITE_BINDING,
                "--claim-receipt",
                PROVIDER_PLAN_CLAIM_RECEIPT_BINDING,
                "--phase-input-root",
                PROVIDER_PLAN_PHASE_INPUT_BINDING,
                "--phase-output-root",
                PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
            ],
            "activation_command_id": PROVIDER_PHASE_COMMAND_IDS[phase],
            "approval_environment": blueprint.approval_environment,
            "c1_commit_binding": PROVIDER_PLAN_C1_COMMIT_BINDING,
            "claim_job_name": claim_job,
            "claim_nonce": expectation.claim_nonce,
            "claim_predecessor_binding": PROVIDER_PLAN_PREDECESSOR_BINDING,
            "claim_receipt_path_template": (
                f"{blueprint.claim_root}/{{suite_attempt_id}}/{phase}/claim-receipt.json"
            ),
            "execute_job_name": execute_job,
            "execution_claim_inputs": (
                blueprint.execution_claim_inputs.to_dict() if phase == ONLINE_PHASE else None
            ),
            "host_tools": blueprint.host_tools.to_dict(),
            "manifest_sha256_binding": PROVIDER_PLAN_MANIFEST_BINDING,
            "maximum_runtime_seconds": PROVIDER_PHASE_RUNTIME_CEILINGS[phase],
            "oci_index_digest": index_digest,
            "oci_platform_manifest_digest": platform_digest,
            "phase": phase,
            "phase_evidence_root_template": (
                f"{blueprint.evidence_root}/{{suite_attempt_id}}/{phase}"
            ),
            "phase_input_binding": PROVIDER_PLAN_PHASE_INPUT_BINDING,
            "phase_output_binding": PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
            "provider_architecture": "ARM64",
            "provider_operating_system": "macOS",
            "provider_plan_path": str(provider_path),
            "repository": REPOSITORY,
            "run_head_branch": RUN_HEAD_BRANCH,
            "runner_archive_sha256": OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
            "runner_bootstrap_receipt": embedded_bootstrap,
            "runner_bootstrap_receipt_file_sha256": embedded_bootstrap_sha256,
            "runner_bootstrap_receipt_path": str(activation_receipt_path),
            "runner_group_id": registration.runner_group_id,
            "runner_id": registration.runner_id,
            "runner_identity": blueprint.runner_identity,
            "runner_name": registration.runner_name,
            "runner_registration_bundle_path": str(registration_bundle_paths[phase]),
            "runner_registration_bundle_sha256": registration_bundle_sha256s[phase],
            "runner_registration_evidence_file_sha256": (registration_evidence_file_sha256s[phase]),
            "runner_version": registration.runner_version,
            "runtime_image": runtime_image,
            "runtime_image_role": image_role,
            "runtime_index_role": index_role,
            "runtime_platform": platform,
            "runtime_probe_receipt_sha256": probe_digest,
            "schema_version": PROVIDER_PHASE_PLAN_TEMPLATE_SCHEMA,
            "suite_attempt_id_binding": PROVIDER_PLAN_SUITE_BINDING,
            "tle_binary_sha256": (
                SOURCE_BUILT_LINUX_ARM64_TLE_SHA256 if phase == LABEL_RELEASE_PHASE else None
            ),
            "tle_build_provenance_sha256": (
                closure.release_reproducibility_receipt_sha256
                if phase == LABEL_RELEASE_PHASE
                else None
            ),
            "tle_interoperability_receipt_sha256": (
                closure.release_tle_interoperability_receipt_sha256
                if phase == LABEL_RELEASE_PHASE
                else None
            ),
            "tle_vulnerability_scan_sha256": (
                closure.release_security_adjudication_sha256
                if phase == LABEL_RELEASE_PHASE
                else None
            ),
            "workflow_path": workflow,
            "workflow_ref": f"{REPOSITORY}/{workflow}@refs/tags/{RUN_HEAD_BRANCH}",
            "workflow_sha": C0_COMMIT_SENTINEL,
        }
    return result


def finalize_provider_plans(
    *,
    blueprint_path: str | Path,
    blueprint_write_receipt_path: str | Path,
    output_directory: str | Path,
) -> ProviderPlanFinalizationReceipt:
    """Finalize one pre-A blueprint from three derived closed runner bundles."""

    blueprint_source = _absolute_path("blueprint_path", blueprint_path)
    blueprint_receipt_source = _absolute_path(
        "blueprint_write_receipt_path", blueprint_write_receipt_path
    )
    bundle_path = _absolute_path("output_directory", output_directory)
    fragment_output = bundle_path / PROVIDER_PLAN_FRAGMENT_FILENAME
    manifest_output = bundle_path / PROVIDER_PLAN_CANDIDATE_MANIFEST_FILENAME
    if os.path.lexists(bundle_path):
        raise ProviderPlanOperatorError("provider-plan finalization output already exists")
    if any(
        _paths_overlap(bundle_path, source)
        for source in (blueprint_source, blueprint_receipt_source)
    ):
        raise ProviderPlanOperatorError("provider-plan finalization overlaps its blueprint")

    blueprint_directory = blueprint_source.parent
    if (
        blueprint_source != blueprint_directory / PROVIDER_PLAN_BLUEPRINT_FILENAME
        or blueprint_receipt_source
        != blueprint_directory / PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_FILENAME
    ):
        raise ProviderPlanOperatorError(
            "finalization blueprint paths do not name one closed bundle"
        )
    blueprint, blueprint_write_receipt = load_provider_plan_blueprint_bundle(blueprint_directory)
    blueprint_file_sha256 = blueprint.file_sha256
    blueprint_write_receipt_sha256 = blueprint_write_receipt.file_sha256
    if (
        blueprint_write_receipt.blueprint_path != blueprint_source
        or blueprint_write_receipt.blueprint_file_sha256 != blueprint_file_sha256
        or blueprint_write_receipt.candidate_manifest_file_sha256
        != blueprint.candidate_manifest_file_sha256
        or blueprint_write_receipt.host_tool_contract_sha256 != blueprint.host_tools.contract_sha256
    ):
        raise ProviderPlanOperatorError("provider-plan blueprint differs from its write receipt")

    source_manifest, closure, _config, _host_tools = _revalidate_blueprint_sources(blueprint)
    source_before = read_secure_regular_file(
        blueprint.candidate_manifest_path,
        max_bytes=_MAX_JSON_BYTES,
        label="candidate provider-plan source manifest",
    )
    (
        registrations,
        registration_bundle_paths,
        registration_receipt_paths,
        registration_receipt_digests,
        registration_evidence_digests,
        registration_bundle_digests,
    ) = _admit_registration_bundles(
        blueprint,
        blueprint_directory,
    )
    templates = _provider_plan_templates(
        blueprint,
        closure,
        registrations,
        registration_bundle_paths,
        registration_bundle_digests,
        registration_evidence_digests,
    )
    candidate = copy.deepcopy(dict(source_manifest))
    sealed = candidate.get("sealed_execution")
    if not isinstance(sealed, dict) or sealed.get("provider_phase_plans") != "tbd":
        raise ProviderPlanOperatorError("candidate source provider-plan slot changed")
    sealed["provider_phase_plans"] = templates
    candidate_encoded = _canonical_bytes(candidate) + b"\n"
    fragment_encoded = _canonical_bytes(templates) + b"\n"
    raw_templates_sha256 = _sha256_bytes(_canonical_bytes(templates))
    witness_commit = _sha256_bytes(
        b"fractal-provider-plan-candidate-loader-witness-v1\0" + candidate_encoded
    )[:40]
    second_witness_commit = _sha256_bytes(
        b"fractal-provider-plan-candidate-loader-witness-v2\0" + candidate_encoded
    )[:40]
    if witness_commit == second_witness_commit:
        raise ProviderPlanOperatorError("candidate-loader witness derivation collided")

    try:
        temporary_parent = bundle_path.parent.resolve(strict=True)
    except OSError as exc:
        raise ProviderPlanOperatorError("candidate manifest output parent is unavailable") from exc
    if temporary_parent != bundle_path.parent:
        raise ProviderPlanOperatorError("candidate manifest output parent cannot contain symlinks")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".provider-plan-validation-",
            dir=temporary_parent,
        ) as temporary:
            temporary_manifest = Path(temporary) / "candidate-manifest.json"
            write_exclusive_receipt_bytes(candidate_encoded, temporary_manifest)
            loaded_by_witness = []
            for commit in (witness_commit, second_witness_commit):
                loaded_by_witness.append(
                    load_provider_phase_plans(
                        temporary_manifest,
                        c1_commit=commit,
                        validation_mode="candidate-rehearsal",
                        c0_commit=commit,
                    )
                )
            normalized_closure = provider_phase_plan_templates_sha256(
                candidate,
                validation_mode="candidate-rehearsal",
                c0_commit=witness_commit,
            )
    except (ArtifactIntegrityError, ExecutionClaimError, OSError, ValueError) as exc:
        raise ProviderPlanOperatorError(
            f"derived provider plans fail typed candidate admission: {exc}"
        ) from exc
    if any(tuple(loaded) != PHASES for loaded in loaded_by_witness):
        raise ProviderPlanOperatorError("typed candidate loader returned another phase order")
    if witness_commit in candidate_encoded.decode("ascii") or second_witness_commit in (
        candidate_encoded.decode("ascii")
    ):
        raise ProviderPlanOperatorError("raw candidate bytes consumed a later commit witness")

    source_after_validation = read_secure_regular_file(
        blueprint.candidate_manifest_path,
        max_bytes=_MAX_JSON_BYTES,
        label="candidate provider-plan source manifest",
    )
    if source_after_validation != source_before:
        raise ProviderPlanOperatorError("source manifest changed during provider-plan validation")

    final_receipt = ProviderPlanFinalizationReceipt(
        blueprint_path=blueprint_source,
        blueprint_file_sha256=blueprint_file_sha256,
        blueprint_write_receipt_path=blueprint_receipt_source,
        blueprint_write_receipt_file_sha256=blueprint_write_receipt_sha256,
        candidate_manifest_source_path=blueprint.candidate_manifest_path,
        candidate_manifest_source_file_sha256=blueprint.candidate_manifest_file_sha256,
        candidate_manifest_output_path=manifest_output,
        candidate_manifest_output_file_sha256=_sha256_bytes(candidate_encoded),
        candidate_manifest_output_mode="0600",
        provider_plan_fragment_path=fragment_output,
        provider_plan_fragment_file_sha256=_sha256_bytes(fragment_encoded),
        provider_plan_fragment_mode="0600",
        registration_bundle_paths={
            phase: str(registration_bundle_paths[phase]) for phase in PHASES
        },
        registration_bundle_sha256s={phase: registration_bundle_digests[phase] for phase in PHASES},
        registration_evidence_file_sha256s={
            phase: registration_evidence_digests[phase] for phase in PHASES
        },
        registration_receipt_paths={
            phase: str(registration_receipt_paths[phase]) for phase in PHASES
        },
        registration_receipt_file_sha256s={
            phase: registration_receipt_digests[phase] for phase in PHASES
        },
        candidate_loader_witness_commit=witness_commit,
        raw_provider_plan_templates_sha256=raw_templates_sha256,
        witness_normalized_provider_plan_closure_sha256=normalized_closure,
        typed_candidate_loader_verified=True,
        source_manifest_unchanged=True,
    )

    def revalidate_every_source() -> None:
        current_blueprint, current_write_receipt = load_provider_plan_blueprint_bundle(
            blueprint_directory
        )
        if (
            current_blueprint != blueprint
            or current_write_receipt != blueprint_write_receipt
            or current_blueprint.file_sha256 != blueprint_file_sha256
            or current_write_receipt.file_sha256 != blueprint_write_receipt_sha256
        ):
            raise ProviderPlanOperatorError("blueprint bundle changed before final publish")
        current_manifest, current_closure, _config, current_host_tools = (
            _revalidate_blueprint_sources(blueprint)
        )
        (
            current_registrations,
            current_bundle_paths,
            current_receipt_paths,
            current_receipt_digests,
            current_evidence_digests,
            current_bundle_digests,
        ) = _admit_registration_bundles(blueprint, blueprint_directory)
        if (
            _canonical_bytes(current_manifest) + b"\n" != source_before
            or current_closure != closure
            or current_host_tools != blueprint.host_tools
            or current_registrations != registrations
            or current_bundle_paths != registration_bundle_paths
            or current_receipt_paths != registration_receipt_paths
            or current_receipt_digests != registration_receipt_digests
            or current_evidence_digests != registration_evidence_digests
            or current_bundle_digests != registration_bundle_digests
        ):
            raise ProviderPlanOperatorError("provider-plan source set changed before publish")

    _publish_private_bundle(
        bundle_path,
        {
            PROVIDER_PLAN_FRAGMENT_FILENAME: fragment_encoded,
            PROVIDER_PLAN_CANDIDATE_MANIFEST_FILENAME: candidate_encoded,
            PROVIDER_PLAN_FINALIZATION_RECEIPT_FILENAME: (final_receipt.canonical_file_bytes()),
        },
        label="provider-plan finalization bundle",
        pre_publish=revalidate_every_source,
    )
    _candidate_readback, _template_readback, receipt_readback = (
        load_provider_plan_finalization_bundle(bundle_path)
    )
    if receipt_readback != final_receipt:
        raise ProviderPlanOperatorError("provider-plan finalization readback differs")
    try:
        load_provider_phase_plans(
            manifest_output,
            c1_commit=witness_commit,
            validation_mode="candidate-rehearsal",
            c0_commit=witness_commit,
        )
    except ExecutionClaimError as exc:
        raise ProviderPlanOperatorError(
            "published candidate manifest fails typed readback"
        ) from exc
    source_after_publication = read_secure_regular_file(
        blueprint.candidate_manifest_path,
        max_bytes=_MAX_JSON_BYTES,
        label="candidate provider-plan source manifest",
    )
    if source_after_publication != source_before:
        raise ProviderPlanOperatorError("source manifest changed during final publication")
    return final_receipt


def _add_host_tool_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--controlled-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--gh-executable", type=Path, required=True)
    parser.add_argument("--runner-listener-executable", type=Path, required=True)
    parser.add_argument("--runner-listener-dll", type=Path, required=True)
    parser.add_argument("--runner-config-executable", type=Path, required=True)
    parser.add_argument("--runner-run-executable", type=Path, required=True)
    parser.add_argument("--docker-executable", type=Path, required=True)
    parser.add_argument("--host-probe", type=Path, required=True)
    parser.add_argument("--docker-server-probe", type=Path, required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-provider-plans",
        description="Derive the three provider phase plans without hand-authored JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    blueprint = subparsers.add_parser("write-blueprint")
    blueprint.add_argument("--candidate-manifest", type=Path, required=True)
    blueprint.add_argument("--production-control-config", type=Path, required=True)
    blueprint.add_argument(
        "--production-control-config-write-receipt",
        type=Path,
        required=True,
    )
    blueprint.add_argument("--candidate-image-closure", type=Path, required=True)
    blueprint.add_argument("--execution-beacon-contract", type=Path, required=True)
    blueprint.add_argument(
        "--registered-online-runtime-budget-seconds",
        type=int,
        required=True,
    )
    _add_host_tool_arguments(blueprint)
    blueprint.add_argument("--claim-root", type=Path, required=True)
    blueprint.add_argument("--evidence-root", type=Path, required=True)
    blueprint.add_argument("--online-runner-name", required=True)
    blueprint.add_argument("--label-release-runner-name", required=True)
    blueprint.add_argument("--analysis-runner-name", required=True)
    blueprint.add_argument("--output-directory", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--blueprint", type=Path, required=True)
    finalize.add_argument("--blueprint-write-receipt", type=Path, required=True)
    finalize.add_argument("--output-directory", type=Path, required=True)
    return parser


def _host_sources_from_arguments(arguments: argparse.Namespace) -> HostToolSources:
    return HostToolSources(
        controlled_root=arguments.controlled_root,
        python_executable=arguments.python_executable,
        venv_root=arguments.venv_root,
        gh_executable=arguments.gh_executable,
        runner_listener_executable=arguments.runner_listener_executable,
        runner_listener_dll=arguments.runner_listener_dll,
        runner_config_executable=arguments.runner_config_executable,
        runner_run_executable=arguments.runner_run_executable,
        docker_executable=arguments.docker_executable,
        host_probe_path=arguments.host_probe,
        docker_server_probe_path=arguments.docker_server_probe,
    )


def _run(arguments: argparse.Namespace) -> Mapping[str, object]:
    if arguments.command == "write-blueprint":
        receipt = write_provider_plan_blueprint(
            candidate_manifest_path=arguments.candidate_manifest,
            production_control_config_path=arguments.production_control_config,
            production_control_config_write_receipt_path=(
                arguments.production_control_config_write_receipt
            ),
            candidate_image_closure_path=arguments.candidate_image_closure,
            execution_beacon_contract_path=arguments.execution_beacon_contract,
            registered_online_runtime_budget_seconds=(
                arguments.registered_online_runtime_budget_seconds
            ),
            host_tool_sources=_host_sources_from_arguments(arguments),
            claim_root=arguments.claim_root,
            evidence_root=arguments.evidence_root,
            runner_names={
                ONLINE_PHASE: arguments.online_runner_name,
                LABEL_RELEASE_PHASE: arguments.label_release_runner_name,
                ANALYSIS_PHASE: arguments.analysis_runner_name,
            },
            output_directory=arguments.output_directory,
        )
        receipt_path = arguments.output_directory / PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_FILENAME
        return {
            "blueprint_file_sha256": receipt.blueprint_file_sha256,
            "blueprint_path": str(receipt.blueprint_path),
            "receipt_file_sha256": receipt.file_sha256,
            "receipt_path": str(receipt_path),
        }
    receipt = finalize_provider_plans(
        blueprint_path=arguments.blueprint,
        blueprint_write_receipt_path=arguments.blueprint_write_receipt,
        output_directory=arguments.output_directory,
    )
    receipt_path = arguments.output_directory / PROVIDER_PLAN_FINALIZATION_RECEIPT_FILENAME
    return {
        "candidate_manifest_file_sha256": receipt.candidate_manifest_output_file_sha256,
        "candidate_manifest_path": str(receipt.candidate_manifest_output_path),
        "raw_provider_plan_templates_sha256": (receipt.raw_provider_plan_templates_sha256),
        "witness_normalized_provider_plan_closure_sha256": (
            receipt.witness_normalized_provider_plan_closure_sha256
        ),
        "provider_plan_fragment_file_sha256": (receipt.provider_plan_fragment_file_sha256),
        "provider_plan_fragment_path": str(receipt.provider_plan_fragment_path),
        "receipt_file_sha256": receipt.file_sha256,
        "receipt_path": str(receipt_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        result = _run(arguments)
    except ProviderPlanOperatorError as exc:
        print(f"provider-plan operator failed: {exc}", file=sys.stderr)
        return 2
    print(_canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
