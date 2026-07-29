"""Two-phase construction of the five sealed production control closures.

The pre-C1 phase writes only label-excluded workload specifications. The
post-C1 phase admits a publicly verified registration, binds the shared
sealed-run receipt and custody controls, and derives the five configs, runtime
plan templates, and launcher contracts without accepting scientific overrides.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import sys
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from .artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactVerificationReceipt,
    digest_directory_tree,
    digest_regular_file,
    load_local_artifact_map,
    load_verification_receipt,
    read_secure_regular_file,
    verify_local_artifacts,
    write_exclusive_receipt_bytes,
)
from .artifact_stage_bundles import STAGE_BUNDLE_FILENAME
from .authorized_index_store import load_authorized_index_store_receipt
from .c0_evidence_release import (
    C0_CANDIDATE_ASSEMBLY_RECEIPT_ARCHIVE_MEMBER_PATH,
    C0_CANDIDATE_MANIFEST_ARCHIVE_MEMBER_PATH,
)
from .candidate_manifest_assembler import (
    ASSEMBLY_RECEIPT_FILENAME,
    CANDIDATE_MANIFEST_FILENAME,
    CandidateManifestAssemblyError,
    load_closed_candidate_manifest_package,
)
from .custody import (
    OnlineCustodyAdmissionReceipt,
    admit_online_custody,
    load_online_custody_admission_receipt,
)
from .embedding_store import load_embedding_store_receipt
from .opa_runtime_binary import (
    C0RuntimeExtractionReceipt,
    OpaRuntimeBinaryError,
    load_c0_runtime_extraction_receipt,
    load_runtime_attestation_plan_template,
)
from .policy_intervention import RECEIPT_FILENAME as POLICY_RECEIPT_FILENAME
from .policy_intervention import load_policy_intervention_receipt
from .production_artifact_factory import (
    FACTORY_RUNNER_PLATFORM,
    RUNTIME_BACKEND,
    RUNTIME_DRIFT_FAMILY,
    RUNTIME_QUERY_DIRECTORY,
    RUNTIME_RECEIPT_FILENAME,
    VERSION_LAG,
    FactorySuiteCorpus,
    ProductionArtifactFactoryConfig,
    ProductionArtifactFactoryError,
    ProductionArtifactFactorySuiteReceipt,
    load_production_artifact_factory_config,
    load_production_artifact_factory_suite,
    verify_production_artifact_factory,
)
from .production_corpus_run import (
    ONLINE_CUSTODY_ADMISSION_FILENAME,
    PRODUCTION_CORPUS_CONFIG_FILENAME,
    PRODUCTION_CORPUS_WORKLOAD_ID,
    PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME,
    REQUIRED_ARTIFACT_BINDINGS_FILENAME,
    RUNTIME_INVOCATION_MARKER_FILENAME,
    SHARDED_EXECUTION_PLAN_FILENAME,
    TRIAL_RUNTIME_RECEIPT_FILENAME,
    ProductionCorpusRunConfig,
    ProductionCorpusRunError,
    ProductionCorpusWorkloadSpec,
)
from .production_embedding_build import load_production_embedding_config
from .production_workload_registration import (
    ProductionWorkloadRegistrationError,
    canonical_workload_file_bytes,
    production_workload_file_sha256,
    validate_production_workload_registrations,
)
from .provider_rehearsal import (
    CandidateImageClosure,
    ProviderRehearsalError,
)
from .runtime_attestation import (
    RuntimeAttestationPlan,
    RuntimeFilePin,
    RuntimePreflightReceipt,
    argv_sha256,
    environment_sha256,
    launcher_identity_file_bytes,
    loads_runtime_preflight_receipt,
    runtime_attestation_plan_template_file_bytes,
)
from .scalable_execution import (
    ONLINE_EXECUTION_PLAN_FILENAME,
    load_sharded_online_execution_plan,
    loads_sharded_online_execution_plan,
)
from .sealed_container_launcher import (
    PREFLIGHT_DIGEST_SENTINEL,
    PREFLIGHT_INTEGER_SENTINEL,
    PREFLIGHT_TEXT_SENTINEL,
    ClosureFileBinding,
    LauncherBindMount,
    LauncherEnvironmentVariable,
    LauncherGeometry,
    PreflightLaunchContract,
    ProductionRunClosureBindingReceipt,
    RuntimePlanTransitionReceipt,
    SealedContainerLauncherError,
    VerifiedProductionRunClosure,
    _mint_verified_production_run_closure,
    load_preflight_launch_contract,
    load_runtime_plan_transition,
    verify_launcher_mounts,
    verify_production_run_closure_binding,
    verify_runtime_plan_transition,
)
from .sealed_orchestrator import (
    RequiredArtifactIdBindings,
    SealedOrchestratorError,
    derive_required_artifact_id_bindings,
    load_required_artifact_id_bindings,
)
from .study import (
    C0_COMMIT_SENTINEL,
    FIXED_CORPORA,
    SealedRunReceipt,
    load_sealed_run_receipt,
    load_study_manifest,
    manifest_sha256,
    validate_candidate_rehearsal_manifest,
    validate_study_manifest,
)
from .suite_attempt import suite_attempt_id, suite_namespace
from .trial_runtime import (
    QUERY_TRIAL_RECEIPT_FILENAME,
    RuntimeFeatureBinding,
    TrialRuntimeAdmissionReceipt,
    TrialRuntimeError,
    load_trial_runtime_receipt,
)
from .zenodo_publication import verify_production_protocol_registration

PRODUCTION_CONTROL_CONFIG_SCHEMA = "fractal-production-control-materialization-config-v5"
PRODUCTION_CONTROL_CONFIG_WRITE_RECEIPT_SCHEMA = (
    "fractal-production-control-materialization-config-write-receipt-v3"
)
PRODUCTION_CONTROL_FINALIZATION_REQUEST_SCHEMA = (
    "fractal-production-control-finalization-request-v2"
)
PRODUCTION_CONTROL_BLUEPRINT_SCHEMA = "fractal-production-control-blueprint-receipt-v4"
PRODUCTION_CONTROL_C0_INSTANTIATION_SCHEMA = (
    "fractal-production-control-c0-instantiation-receipt-v4"
)
PRODUCTION_CONTROL_FINALIZATION_SCHEMA = "fractal-production-control-finalization-receipt-v2"

BLUEPRINT_RECEIPT_FILENAME = "production-control-blueprint-receipt.json"
FINALIZATION_RECEIPT_FILENAME = "production-control-finalization-receipt.json"
PLAN_TEMPLATE_FILENAME = "runtime-attestation-plan.template.json"
PREFLIGHT_CONTRACT_FILENAME = "preflight-launch-contract.json"
LAUNCHER_IDENTITY_FILENAME = "launcher-identity.json"
RUNTIME_PREFLIGHT_RECEIPT_FILENAME = "runtime-preflight-receipt.json"
RUNTIME_PLAN_TRANSITION_RECEIPT_FILENAME = "runtime-plan-transition-receipt.json"
PRODUCTION_CLOSURE_BINDING_FILENAME = "production-run-closure-binding.json"
PRODUCTION_WORKLOADS_FRAGMENT_FILENAME = "production-workloads.fragment.json"
PRODUCTION_HARDWARE_FRAGMENT_FILENAME = "production-hardware.fragment.json"
C0_INSTANTIATION_RECEIPT_FILENAME = "c0-control-instantiation-receipt.json"
C0_CANDIDATE_PACKAGE_DIRECTORY = "candidate-manifest-package"
C0_CANDIDATE_MANIFEST_RELATIVE_PATH = (
    f"{C0_CANDIDATE_PACKAGE_DIRECTORY}/{CANDIDATE_MANIFEST_FILENAME}"
)
C0_CANDIDATE_ASSEMBLY_RECEIPT_RELATIVE_PATH = (
    f"{C0_CANDIDATE_PACKAGE_DIRECTORY}/{ASSEMBLY_RECEIPT_FILENAME}"
)
PRODUCTION_APPROVAL_ENVIRONMENT = "confirmatory"
_GIB = 1024 * 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OCI_IMAGE = re.compile(r"^[^\s/@]+(?:/[^\s/@]+)+@sha256:[0-9a-f]{64}$")
_OCI_INDEX_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_CONTROL_BYTES = 256 * 1024 * 1024
_PYTHON_PATH = "/opt/venv/bin/python"
_OPA_PATH = "/usr/local/bin/opa"
_UV_LOCK_PATH = "/opt/app/uv.lock"
_LAUNCHER_IDENTITY_PATH = "/input/launcher-identity.json"
_CONTROL_ROOT = "/input/control"
_OUTPUT_ROOT = "/output"
_TMPFS_ROOT = "/tmp"
_UID = 65532
_GID = 65532
_PLACEHOLDER_RETENTION_PREFIX = ".production-run-closure.retained-"

_CONTAINER_PATHS = {
    "artifact_root": "/input/online",
    "authorized_index_store_root": "/input/index",
    "embedding_store_root": "/input/embedding",
    "index_bundle_receipt_path": "/input/bundles/index-stage-bundle.json",
    "partition_audit_path": "/input/partition-audit.json",
    "policy_intervention_root": "/input/policy",
    "policy_bundle_receipt_path": "/input/bundles/policy-stage-bundle.json",
    "pseudonym_key_path": "/run/secrets/audit-pseudonym.key",
    "query_package_root": "/input/query-package",
    "staged_root": "/input/staged",
}

_FIXED_ENVIRONMENT = {
    "HOME": "/home/runner",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LD_LIBRARY_PATH": "/opt/native-libs",
    "LOGNAME": "runner",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONPATH": "/opt/app/src",
    "TMPDIR": "/tmp",
    "TZ": "UTC",
    "USER": "runner",
    "VECLIB_MAXIMUM_THREADS": "1",
    "XDG_CACHE_HOME": "/tmp/fractal-cache",
}

_CONFIG_FIELDS = frozenset(
    {
        "blueprint_root",
        "approval_environment",
        "c0_runtime_extraction_receipt_path",
        "c0_runtime_extraction_receipt_sha256",
        "candidate_image_source_commit",
        "cpuset_cpus",
        "factory_artifact_tree_sha256",
        "factory_config_path",
        "factory_config_sha256",
        "factory_suite_receipt_path",
        "factory_suite_receipt_sha256",
        "finalized_controls_root",
        "hostname",
        "hardware_accelerator",
        "hardware_cpu_model",
        "hardware_instance_type",
        "hardware_operating_system",
        "hardware_provider",
        "hardware_region",
        "memory_limit_bytes",
        "opa_binary_path",
        "opa_binary_sha256",
        "pseudonym_key_path",
        "pseudonym_key_sha256",
        "runner_identity",
        "scientific_candidate_reference",
        "scientific_index_digest",
        "scientific_production_reference",
        "oci_promotion_required",
        "runner_platform",
        "schema_version",
        "suite_base_root",
        "tmpfs_size_bytes",
        "uv_lock_path",
        "uv_lock_sha256",
    }
)

_FINALIZATION_REQUEST_FIELDS = frozenset(
    {
        "artifact_root",
        "artifact_verification_receipt_path",
        "blueprint_receipt_path",
        "blueprint_receipt_sha256",
        "c0_control_instantiation_receipt_path",
        "c1_package_root",
        "custody_seal_receipt_path",
        "frozen_manifest_path",
        "local_artifact_map_path",
        "manifest_lock_path",
        "materialization_config_path",
        "materialization_config_sha256",
        "online_custody_admission_path",
        "protocol_registration_receipt_path",
        "protocol_registry_record_path",
        "required_artifact_bindings_root",
        "runtime_evidence_root",
        "schema_version",
        "sealed_run_receipt_path",
    }
)

_OUTCOME_BEARING_TOKENS = (
    "sealed-label",
    "plaintext-label",
    "qrel",
    "relevance-judgment",
    "prediction-result",
    "analysis-result",
    "protocol-registration",
    "manifest-sha256",
)

_MANIFEST_PRODUCTION_CONTROL_FIELDS = frozenset(
    {
        "blueprint_receipt_file_sha256",
        "blueprint_receipt_sha256",
        "materialization_config_file_sha256",
    }
)


class ProductionControlError(RuntimeError):
    """Raised when a control boundary differs from its closed derivation."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ProductionControlError("production control data must be canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProductionControlError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(name: str, value: object) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ProductionControlError(f"{name} must be a canonical identifier")
    return value


def _text(name: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ProductionControlError(f"{name} must be canonical text")
    return value


def _absolute_path(name: str, value: object) -> Path:
    if not isinstance(value, (str, Path)):
        raise ProductionControlError(f"{name} must be a path")
    raw = os.fspath(value)
    path = PurePosixPath(raw)
    if (
        type(raw) is not str
        or not path.is_absolute()
        or str(path) != raw
        or "\\" in raw
        or unicodedata.normalize("NFC", raw) != raw
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ProductionControlError(f"{name} must be a canonical absolute POSIX path")
    return Path(raw)


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ProductionControlError(f"{name} must be a positive integer")
    return value


def _closed(value: object, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ProductionControlError(f"{label} must be one object with string keys")
    observed = set(value)
    if observed != fields:
        raise ProductionControlError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _parse_object(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProductionControlError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ProductionControlError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionControlError(f"{label} is not canonical UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ProductionControlError(f"{label} must contain one object")
    return value


def _parse_array(encoded: bytes, *, label: str) -> list[Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProductionControlError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ProductionControlError(f"{label} contains non-finite value {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionControlError(f"{label} is not canonical UTF-8 JSON") from exc
    if type(value) is not list:
        raise ProductionControlError(f"{label} must contain one array")
    return value


def _read(path: str | Path, *, label: str) -> bytes:
    try:
        return read_secure_regular_file(path, max_bytes=_MAX_CONTROL_BYTES, label=label)
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(f"cannot read {label}: {exc}") from exc


def _read_pinned(path: str | Path, expected_sha256: str, *, label: str) -> bytes:
    encoded = _read(path, label=label)
    if _sha256_bytes(encoded) != _sha256(f"{label} SHA-256", expected_sha256):
        raise ProductionControlError(f"{label} differs from its external pin")
    return encoded


def _assert_outcome_blind(value: object, *, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            folded = str(key).casefold().replace("_", "-")
            if any(token in folded for token in _OUTCOME_BEARING_TOKENS):
                raise ProductionControlError(f"outcome-bearing field is forbidden at {path}.{key}")
            _assert_outcome_blind(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for position, nested in enumerate(value):
            _assert_outcome_blind(nested, path=f"{path}[{position}]")
    elif isinstance(value, str):
        folded = value.casefold().replace("_", "-")
        if any(token in folded for token in _OUTCOME_BEARING_TOKENS):
            raise ProductionControlError(f"outcome-bearing value is forbidden at {path}")


def _paths_overlap(first: Path, second: Path) -> bool:
    left = PurePosixPath(str(first)).parts
    right = PurePosixPath(str(second)).parts
    common = min(len(left), len(right))
    return left[:common] == right[:common]


def _canonical_cpuset_cpus(value: object) -> tuple[int, ...]:
    if type(value) is str:
        if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", value) is None:
            raise ProductionControlError(
                "cpuset_cpus must be comma-separated canonical nonnegative integers"
            )
        cpus = tuple(int(item) for item in value.split(","))
    else:
        try:
            cpus = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ProductionControlError("cpuset_cpus must be an ordered CPU list") from exc
    if (
        not cpus
        or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
        or cpus != tuple(sorted(set(cpus)))
    ):
        raise ProductionControlError("cpuset_cpus must be unique sorted CPU indices")
    return cpus


def _scientific_image_transition(
    candidate_reference: object,
    production_reference: object,
) -> str:
    for name, value in (
        ("scientific_candidate_reference", candidate_reference),
        ("scientific_production_reference", production_reference),
    ):
        if type(value) is not str or _OCI_IMAGE.fullmatch(value) is None:
            raise ProductionControlError(f"{name} must be digest-qualified")
    if candidate_reference == production_reference:
        raise ProductionControlError(
            "candidate and production scientific references must be distinct"
        )
    candidate_digest = candidate_reference.rsplit("@", 1)[1]
    production_digest = production_reference.rsplit("@", 1)[1]
    if candidate_digest != production_digest:
        raise ProductionControlError(
            "candidate and production references must share scientific_index_digest"
        )
    return candidate_digest


def _cpuset_argument(value: str) -> tuple[int, ...]:
    try:
        return _canonical_cpuset_cpus(value)
    except ProductionControlError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _assert_unsymlinked_root(path: Path, *, label: str) -> None:
    """Reject a link in any existing component of a future destination root."""

    current = Path("/")
    try:
        for component in path.parts[1:]:
            current /= component
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                return
            if stat.S_ISLNK(metadata.st_mode):
                raise ProductionControlError(f"{label} contains a symlink component")
            if current != path and not stat.S_ISDIR(metadata.st_mode):
                raise ProductionControlError(f"{label} has a non-directory parent component")
        if path.exists() and not path.is_dir():
            raise ProductionControlError(f"{label} must be a directory or an absent path")
    except ProductionControlError:
        raise
    except OSError as exc:
        raise ProductionControlError(f"cannot inspect {label}: {exc}") from exc


def _open_private_publish_parent(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    """Open the canonical parent one component at a time without following links."""

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open("/", directory_flags)
        for component in path.parent.parts[1:]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        named_metadata = path.parent.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (named_metadata.st_dev, named_metadata.st_ino)
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ProductionControlError(
                f"{label} parent must be one runner-controlled real directory"
            )
        return descriptor, metadata
    except ProductionControlError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ProductionControlError(f"cannot open {label} parent: {exc}") from exc


def _fsync_private_directory(path: Path, *, label: str) -> None:
    """Persist one runner-owned directory after all child entries are complete."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or (hasattr(os, "geteuid") and opened.st_uid != os.geteuid())
        ):
            raise ProductionControlError(f"{label} is not one private real directory")
        os.fsync(descriptor)
    except ProductionControlError:
        raise
    except OSError as exc:
        raise ProductionControlError(f"cannot fsync {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _rename_noreplace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    *,
    label: str,
) -> None:
    """Rename two children of one pinned directory without replacing the target."""

    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:
            raise ProductionControlError("exclusive rename is unavailable on macOS")
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
            raise ProductionControlError("exclusive rename is unavailable on Linux")
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
        raise ProductionControlError(f"exclusive rename is unsupported on {sys.platform!r}")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ProductionControlError(f"{label} already exists")
        raise ProductionControlError(f"cannot publish {label}: {os.strerror(error_number)}")


def _atomic_publish_file_noreplace(path: Path, encoded: bytes, *, label: str) -> None:
    """Publish complete bytes with an operating-system no-replace rename."""

    target = _absolute_path(label, path)
    if not target.name or target.name in {".", ".."}:
        raise ProductionControlError(f"{label} must name one file")
    parent_descriptor, parent_metadata = _open_private_publish_parent(target, label=label)
    temporary_name: str | None = None
    published = False
    try:
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        file_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        for _attempt in range(16):
            candidate = f".{target.name}.tmp-{secrets.token_hex(16)}"
            try:
                descriptor = os.open(
                    candidate,
                    file_flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor is None or temporary_name is None:
            raise ProductionControlError(f"cannot allocate private temporary {label}")
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise ProductionControlError(f"cannot complete temporary {label}")
                offset += written
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size != len(encoded)
                or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
            ):
                raise ProductionControlError(f"temporary {label} is not one private exact file")
        finally:
            os.close(descriptor)
        _rename_noreplace_at(
            parent_descriptor,
            temporary_name,
            target.name,
            label=label,
        )
        published = True
        os.fsync(parent_descriptor)
        target_metadata = os.stat(
            target.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        named_parent_metadata = target.parent.lstat()
        if (
            not stat.S_ISREG(target_metadata.st_mode)
            or target_metadata.st_nlink != 1
            or stat.S_IMODE(target_metadata.st_mode) != 0o600
            or target_metadata.st_size != len(encoded)
            or (hasattr(os, "geteuid") and target_metadata.st_uid != os.geteuid())
            or (parent_metadata.st_dev, parent_metadata.st_ino)
            != (named_parent_metadata.st_dev, named_parent_metadata.st_ino)
        ):
            raise ProductionControlError(f"published {label} differs from its admitted file")
        descriptor = os.open(
            target.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        try:
            opened_metadata = os.fstat(descriptor)
            if (
                opened_metadata.st_dev,
                opened_metadata.st_ino,
                opened_metadata.st_mode,
                opened_metadata.st_nlink,
                opened_metadata.st_size,
            ) != (
                target_metadata.st_dev,
                target_metadata.st_ino,
                target_metadata.st_mode,
                target_metadata.st_nlink,
                target_metadata.st_size,
            ):
                raise ProductionControlError(f"published {label} changed before verification")
            observed = bytearray()
            while len(observed) <= len(encoded):
                chunk = os.read(descriptor, min(65536, len(encoded) + 1 - len(observed)))
                if not chunk:
                    break
                observed.extend(chunk)
            after = os.fstat(descriptor)
            if (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (
                opened_metadata.st_dev,
                opened_metadata.st_ino,
                opened_metadata.st_mode,
                opened_metadata.st_nlink,
                opened_metadata.st_size,
                opened_metadata.st_mtime_ns,
                opened_metadata.st_ctime_ns,
            ):
                raise ProductionControlError(f"published {label} changed during verification")
        finally:
            os.close(descriptor)
        if bytes(observed) != encoded:
            raise ProductionControlError(f"published {label} bytes differ")
    except ProductionControlError:
        raise
    except OSError as exc:
        raise ProductionControlError(f"cannot publish {label}: {exc}") from exc
    finally:
        if temporary_name is not None and not published:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _publish_or_recover_exact_file(path: Path, encoded: bytes, *, label: str) -> None:
    """Publish once, or recover only the exact private bytes from a prior crash."""

    if not os.path.lexists(path):
        _atomic_publish_file_noreplace(path, encoded, label=label)
        return
    observed = _read(path, label=f"existing {label}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionControlError(f"cannot inspect existing {label}: {exc}") from exc
    if (
        observed != encoded
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise ProductionControlError(f"existing {label} is not the exact recoverable publication")


@dataclass(frozen=True)
class ProductionControlMaterializationConfig:
    factory_config_path: Path
    factory_config_sha256: str
    factory_suite_receipt_path: Path
    factory_suite_receipt_sha256: str
    factory_artifact_tree_sha256: str
    c0_runtime_extraction_receipt_path: Path
    c0_runtime_extraction_receipt_sha256: str
    candidate_image_source_commit: str
    opa_binary_path: Path
    opa_binary_sha256: str
    uv_lock_path: Path
    uv_lock_sha256: str
    pseudonym_key_path: Path
    pseudonym_key_sha256: str
    scientific_candidate_reference: str
    scientific_production_reference: str
    scientific_index_digest: str
    oci_promotion_required: bool
    approval_environment: str
    runner_platform: str
    runner_identity: str
    hostname: str
    hardware_provider: str
    hardware_instance_type: str
    hardware_cpu_model: str
    hardware_accelerator: str
    hardware_region: str
    hardware_operating_system: str
    memory_limit_bytes: int
    cpuset_cpus: tuple[int, ...]
    tmpfs_size_bytes: int
    blueprint_root: Path
    finalized_controls_root: Path
    suite_base_root: Path
    schema_version: str = PRODUCTION_CONTROL_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCTION_CONTROL_CONFIG_SCHEMA:
            raise ProductionControlError("production control config schema differs")
        for name in (
            "factory_config_path",
            "factory_suite_receipt_path",
            "c0_runtime_extraction_receipt_path",
            "opa_binary_path",
            "uv_lock_path",
            "pseudonym_key_path",
            "blueprint_root",
            "finalized_controls_root",
            "suite_base_root",
        ):
            object.__setattr__(self, name, _absolute_path(name, getattr(self, name)))
        for name in (
            "factory_config_sha256",
            "factory_suite_receipt_sha256",
            "factory_artifact_tree_sha256",
            "c0_runtime_extraction_receipt_sha256",
            "opa_binary_sha256",
            "uv_lock_sha256",
            "pseudonym_key_sha256",
        ):
            _sha256(name, getattr(self, name))
        if (
            type(self.candidate_image_source_commit) is not str
            or _GIT_COMMIT.fullmatch(self.candidate_image_source_commit) is None
        ):
            raise ProductionControlError(
                "candidate_image_source_commit must be one full Git commit"
            )
        shared_digest = _scientific_image_transition(
            self.scientific_candidate_reference,
            self.scientific_production_reference,
        )
        if (
            type(self.scientific_index_digest) is not str
            or _OCI_INDEX_DIGEST.fullmatch(self.scientific_index_digest) is None
            or self.scientific_index_digest != shared_digest
        ):
            raise ProductionControlError(
                "candidate and production references must share scientific_index_digest"
            )
        if self.oci_promotion_required is not True:
            raise ProductionControlError("oci_promotion_required must be literal true before C1")
        if self.approval_environment != PRODUCTION_APPROVAL_ENVIRONMENT:
            raise ProductionControlError("approval_environment must equal confirmatory")
        if self.runner_platform != FACTORY_RUNNER_PLATFORM:
            raise ProductionControlError("runner_platform must equal linux/arm64")
        expected_runner_identity = f"github-actions:environment:{self.approval_environment}"
        if _text("runner_identity", self.runner_identity) != expected_runner_identity:
            raise ProductionControlError("runner_identity must identify the approval_environment")
        _identifier("hostname", self.hostname)
        for name in (
            "hardware_provider",
            "hardware_instance_type",
            "hardware_cpu_model",
            "hardware_accelerator",
            "hardware_region",
            "hardware_operating_system",
        ):
            value = _text(name, getattr(self, name))
            if "<" in value or ">" in value:
                raise ProductionControlError(
                    f"{name} must not contain unresolved placeholder delimiters"
                )
        _positive_integer("memory_limit_bytes", self.memory_limit_bytes)
        if self.memory_limit_bytes % _GIB:
            raise ProductionControlError(
                "memory_limit_bytes must be an integral GiB for public hardware registration"
            )
        _positive_integer("tmpfs_size_bytes", self.tmpfs_size_bytes)
        cpus = _canonical_cpuset_cpus(self.cpuset_cpus)
        object.__setattr__(self, "cpuset_cpus", cpus)
        destinations = (
            self.blueprint_root,
            self.finalized_controls_root,
            self.suite_base_root,
        )
        if any(
            _paths_overlap(first, second)
            for position, first in enumerate(destinations)
            for second in destinations[position + 1 :]
        ):
            raise ProductionControlError("production control destination roots overlap")
        _assert_outcome_blind(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, Path):
                payload[field] = str(value)
            elif field == "cpuset_cpus":
                payload[field] = list(self.cpuset_cpus)
            else:
                payload[field] = value
        return payload

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256_bytes(self.canonical_file_bytes())

    @property
    def launcher_identity_path(self) -> Path:
        return self.blueprint_root / LAUNCHER_IDENTITY_FILENAME

    @property
    def blueprint_receipt_path(self) -> Path:
        return self.blueprint_root / BLUEPRINT_RECEIPT_FILENAME

    @property
    def production_workloads_fragment_path(self) -> Path:
        return self.blueprint_root / PRODUCTION_WORKLOADS_FRAGMENT_FILENAME

    @property
    def production_hardware_fragment_path(self) -> Path:
        return self.blueprint_root / PRODUCTION_HARDWARE_FRAGMENT_FILENAME

    @property
    def finalization_receipt_path(self) -> Path:
        return self.finalized_controls_root.parent / (
            f"{self.finalized_controls_root.name}.{FINALIZATION_RECEIPT_FILENAME}"
        )

    @classmethod
    def from_dict(cls, value: object) -> ProductionControlMaterializationConfig:
        row = _closed(value, _CONFIG_FIELDS, label="production control config")
        cpus = row["cpuset_cpus"]
        if type(cpus) is not list:
            raise ProductionControlError("cpuset_cpus must be an array")
        return cls(
            **{key: item for key, item in row.items() if key != "cpuset_cpus"},
            cpuset_cpus=tuple(cpus),
        )


@dataclass(frozen=True)
class ProductionControlMaterializationConfigWriteReceipt:
    config_path: Path
    config_file_sha256: str
    config_readback_sha256: str
    config_byte_count: int
    config_mode: str
    factory_config_file_sha256: str
    factory_suite_receipt_file_sha256: str
    factory_artifact_tree_sha256: str
    c0_runtime_extraction_receipt_file_sha256: str
    candidate_image_source_commit: str
    opa_binary_sha256: str
    uv_lock_sha256: str
    pseudonym_key_sha256: str
    scientific_candidate_reference: str
    scientific_production_reference: str
    scientific_index_digest: str
    oci_promotion_required: bool
    approval_environment: str
    readback_verified: bool
    schema_version: str = PRODUCTION_CONTROL_CONFIG_WRITE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", _absolute_path("config_path", self.config_path))
        for name in (
            "config_file_sha256",
            "config_readback_sha256",
            "factory_config_file_sha256",
            "factory_suite_receipt_file_sha256",
            "factory_artifact_tree_sha256",
            "c0_runtime_extraction_receipt_file_sha256",
            "opa_binary_sha256",
            "uv_lock_sha256",
            "pseudonym_key_sha256",
        ):
            _sha256(name, getattr(self, name))
        if (
            type(self.candidate_image_source_commit) is not str
            or _GIT_COMMIT.fullmatch(self.candidate_image_source_commit) is None
        ):
            raise ProductionControlError(
                "config write receipt candidate image source commit differs"
            )
        if self.config_readback_sha256 != self.config_file_sha256:
            raise ProductionControlError("materialization config readback digest differs")
        shared_digest = _scientific_image_transition(
            self.scientific_candidate_reference,
            self.scientific_production_reference,
        )
        if (
            type(self.scientific_index_digest) is not str
            or _OCI_INDEX_DIGEST.fullmatch(self.scientific_index_digest) is None
            or self.scientific_index_digest != shared_digest
        ):
            raise ProductionControlError(
                "config write receipt scientific references differ from their shared digest"
            )
        if self.oci_promotion_required is not True:
            raise ProductionControlError(
                "config write receipt must require OCI promotion before C1"
            )
        if self.approval_environment != PRODUCTION_APPROVAL_ENVIRONMENT:
            raise ProductionControlError("config write receipt approval environment differs")
        _positive_integer("config_byte_count", self.config_byte_count)
        if self.config_mode != "0600":
            raise ProductionControlError("materialization config write mode must equal 0600")
        if self.readback_verified is not True:
            raise ProductionControlError("materialization config readback must be verified")
        if self.schema_version != PRODUCTION_CONTROL_CONFIG_WRITE_RECEIPT_SCHEMA:
            raise ProductionControlError("materialization config write receipt schema differs")

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
    def from_dict(
        cls,
        value: object,
    ) -> ProductionControlMaterializationConfigWriteReceipt:
        return cls(
            **_closed(
                value,
                frozenset(cls.__dataclass_fields__),
                label="production control config write receipt",
            )
        )


@dataclass(frozen=True)
class ProductionControlFinalizationRequest:
    materialization_config_path: Path
    materialization_config_sha256: str
    blueprint_receipt_path: Path
    blueprint_receipt_sha256: str
    c0_control_instantiation_receipt_path: Path
    frozen_manifest_path: Path
    manifest_lock_path: Path
    c1_package_root: Path
    protocol_registry_record_path: Path
    protocol_registration_receipt_path: Path
    sealed_run_receipt_path: Path
    online_custody_admission_path: Path
    custody_seal_receipt_path: Path
    artifact_verification_receipt_path: Path
    artifact_root: Path
    local_artifact_map_path: Path
    required_artifact_bindings_root: Path
    runtime_evidence_root: Path
    schema_version: str = PRODUCTION_CONTROL_FINALIZATION_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCTION_CONTROL_FINALIZATION_REQUEST_SCHEMA:
            raise ProductionControlError("production control finalization request schema differs")
        for name in (
            "materialization_config_path",
            "blueprint_receipt_path",
            "c0_control_instantiation_receipt_path",
            "frozen_manifest_path",
            "manifest_lock_path",
            "c1_package_root",
            "protocol_registry_record_path",
            "protocol_registration_receipt_path",
            "sealed_run_receipt_path",
            "online_custody_admission_path",
            "custody_seal_receipt_path",
            "artifact_verification_receipt_path",
            "artifact_root",
            "local_artifact_map_path",
            "required_artifact_bindings_root",
            "runtime_evidence_root",
        ):
            object.__setattr__(self, name, _absolute_path(name, getattr(self, name)))
        _sha256("materialization_config_sha256", self.materialization_config_sha256)
        _sha256("blueprint_receipt_sha256", self.blueprint_receipt_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            field: (
                str(getattr(self, field))
                if isinstance(getattr(self, field), Path)
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionControlFinalizationRequest:
        row = _closed(value, _FINALIZATION_REQUEST_FIELDS, label="control finalization request")
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class WorkloadSpecBinding:
    corpus_id: str
    available_family_count: int
    selected_family_count: int
    relative_path: str
    file_sha256: str
    launcher_control_tree_sha256: str
    plan_template_file_sha256: str
    plan_template_semantic_sha256: str
    preflight_contract_sha256: str
    preflight_contract_file_sha256: str

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionControlError("workload binding names another corpus")
        _positive_integer("available_family_count", self.available_family_count)
        _positive_integer("selected_family_count", self.selected_family_count)
        if self.selected_family_count > self.available_family_count:
            raise ProductionControlError("selected families exceed the denominator")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or str(path) != self.relative_path or ".." in path.parts:
            raise ProductionControlError("workload binding path must be canonical and relative")
        for name in (
            "file_sha256",
            "launcher_control_tree_sha256",
            "plan_template_file_sha256",
            "plan_template_semantic_sha256",
            "preflight_contract_sha256",
            "preflight_contract_file_sha256",
        ):
            _sha256(f"workload binding {name}", getattr(self, name))

    def to_dict(self) -> dict[str, object]:
        return {
            "available_family_count": self.available_family_count,
            "corpus_id": self.corpus_id,
            "file_sha256": self.file_sha256,
            "launcher_control_tree_sha256": self.launcher_control_tree_sha256,
            "plan_template_file_sha256": self.plan_template_file_sha256,
            "plan_template_semantic_sha256": self.plan_template_semantic_sha256,
            "preflight_contract_file_sha256": self.preflight_contract_file_sha256,
            "preflight_contract_sha256": self.preflight_contract_sha256,
            "relative_path": self.relative_path,
            "selected_family_count": self.selected_family_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> WorkloadSpecBinding:
        row = _closed(
            value,
            frozenset(
                {
                    "available_family_count",
                    "corpus_id",
                    "file_sha256",
                    "launcher_control_tree_sha256",
                    "plan_template_file_sha256",
                    "plan_template_semantic_sha256",
                    "preflight_contract_file_sha256",
                    "preflight_contract_sha256",
                    "relative_path",
                    "selected_family_count",
                }
            ),
            label="workload spec binding",
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductionControlBlueprintReceipt:
    materialization_config_sha256: str
    factory_config_sha256: str
    factory_suite_receipt_sha256: str
    factory_artifact_tree_sha256: str
    c0_runtime_extraction_receipt_sha256: str
    approval_environment: str
    runner_image: str
    runner_platform: str
    candidate_image_source_commit: str
    launcher_identity_file_sha256: str
    production_hardware_fragment_file_sha256: str
    production_workloads_fragment_file_sha256: str
    provisional_closure_root: str
    provisional_closure_tree_sha256: str
    provisional_closure_entries: tuple[str, ...]
    payload_tree_sha256: str
    workloads: tuple[WorkloadSpecBinding, ...]
    unresolved_c1_fields: tuple[str, ...] = (
        "manifest_sha256",
        "online_custody_admission_file_sha256",
        "required_artifact_bindings_file_sha256",
        "sealed_run_receipt_file_sha256",
    )
    schema_version: str = PRODUCTION_CONTROL_BLUEPRINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCTION_CONTROL_BLUEPRINT_SCHEMA:
            raise ProductionControlError("production control blueprint schema differs")
        for name in (
            "materialization_config_sha256",
            "factory_config_sha256",
            "factory_suite_receipt_sha256",
            "factory_artifact_tree_sha256",
            "c0_runtime_extraction_receipt_sha256",
            "launcher_identity_file_sha256",
            "production_hardware_fragment_file_sha256",
            "production_workloads_fragment_file_sha256",
            "provisional_closure_tree_sha256",
            "payload_tree_sha256",
        ):
            _sha256(name, getattr(self, name))
        if type(self.runner_image) is not str or _OCI_IMAGE.fullmatch(self.runner_image) is None:
            raise ProductionControlError("blueprint runner image is not immutable")
        if self.approval_environment != PRODUCTION_APPROVAL_ENVIRONMENT:
            raise ProductionControlError("blueprint approval environment differs")
        if self.runner_platform != FACTORY_RUNNER_PLATFORM:
            raise ProductionControlError("blueprint runner platform differs")
        if (
            type(self.candidate_image_source_commit) is not str
            or _GIT_COMMIT.fullmatch(self.candidate_image_source_commit) is None
        ):
            raise ProductionControlError("blueprint candidate image source commit differs")
        object.__setattr__(
            self,
            "provisional_closure_root",
            str(_absolute_path("provisional_closure_root", self.provisional_closure_root)),
        )
        entries = tuple(self.provisional_closure_entries)
        if entries:
            raise ProductionControlError("the pre-C1 production closure must be empty")
        object.__setattr__(self, "provisional_closure_entries", entries)
        rows = tuple(self.workloads)
        if tuple(row.corpus_id for row in rows) != FIXED_CORPORA:
            raise ProductionControlError("blueprint workloads must follow FIXED_CORPORA")
        object.__setattr__(self, "workloads", rows)
        expected_unresolved = (
            "manifest_sha256",
            "online_custody_admission_file_sha256",
            "required_artifact_bindings_file_sha256",
            "sealed_run_receipt_file_sha256",
        )
        if tuple(self.unresolved_c1_fields) != expected_unresolved:
            raise ProductionControlError("blueprint unresolved C1 fields differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "approval_environment": self.approval_environment,
            "c0_runtime_extraction_receipt_sha256": self.c0_runtime_extraction_receipt_sha256,
            "candidate_image_source_commit": self.candidate_image_source_commit,
            "factory_artifact_tree_sha256": self.factory_artifact_tree_sha256,
            "factory_config_sha256": self.factory_config_sha256,
            "factory_suite_receipt_sha256": self.factory_suite_receipt_sha256,
            "launcher_identity_file_sha256": self.launcher_identity_file_sha256,
            "materialization_config_sha256": self.materialization_config_sha256,
            "payload_tree_sha256": self.payload_tree_sha256,
            "production_hardware_fragment_file_sha256": (
                self.production_hardware_fragment_file_sha256
            ),
            "production_workloads_fragment_file_sha256": (
                self.production_workloads_fragment_file_sha256
            ),
            "provisional_closure_entries": list(self.provisional_closure_entries),
            "provisional_closure_root": self.provisional_closure_root,
            "provisional_closure_tree_sha256": self.provisional_closure_tree_sha256,
            "runner_image": self.runner_image,
            "runner_platform": self.runner_platform,
            "schema_version": self.schema_version,
            "unresolved_c1_fields": list(self.unresolved_c1_fields),
            "workloads": [row.to_dict() for row in self.workloads],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def canonical_file_bytes(self) -> bytes:
        return self.canonical_bytes() + b"\n"

    @property
    def semantic_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes())

    @property
    def file_sha256(self) -> str:
        return _sha256_bytes(self.canonical_file_bytes())

    @property
    def receipt_sha256(self) -> str:
        """Compatibility alias for the on-disk receipt digest."""

        return self.file_sha256

    @classmethod
    def from_dict(cls, value: object) -> ProductionControlBlueprintReceipt:
        fields = frozenset(
            {
                "approval_environment",
                "c0_runtime_extraction_receipt_sha256",
                "candidate_image_source_commit",
                "factory_artifact_tree_sha256",
                "factory_config_sha256",
                "factory_suite_receipt_sha256",
                "launcher_identity_file_sha256",
                "materialization_config_sha256",
                "payload_tree_sha256",
                "production_hardware_fragment_file_sha256",
                "production_workloads_fragment_file_sha256",
                "provisional_closure_entries",
                "provisional_closure_root",
                "provisional_closure_tree_sha256",
                "runner_image",
                "runner_platform",
                "schema_version",
                "unresolved_c1_fields",
                "workloads",
            }
        )
        row = _closed(value, fields, label="production control blueprint receipt")
        workloads = row["workloads"]
        unresolved = row["unresolved_c1_fields"]
        closure_entries = row["provisional_closure_entries"]
        if (
            type(workloads) is not list
            or type(unresolved) is not list
            or type(closure_entries) is not list
        ):
            raise ProductionControlError("blueprint receipt arrays are malformed")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "provisional_closure_entries",
                    "unresolved_c1_fields",
                    "workloads",
                }
            },
            provisional_closure_entries=tuple(closure_entries),
            workloads=tuple(WorkloadSpecBinding.from_dict(item) for item in workloads),
            unresolved_c1_fields=tuple(unresolved),
        )


@dataclass(frozen=True)
class ProductionControlC0InstantiationReceipt:
    """Post-A derivation of executable controls from the immutable raw blueprint."""

    apparatus_commit: str
    candidate_image_source_commit: str
    build_context_tree_sha256: str
    candidate_image_closure_file_sha256: str
    candidate_bootstrap_closure_sha256: str
    candidate_manifest_sha256: str
    candidate_manifest_file_sha256: str
    candidate_manifest_relative_path: str
    candidate_manifest_assembly_receipt_file_sha256: str
    candidate_manifest_assembly_receipt_relative_path: str
    materialization_config_file_sha256: str
    blueprint_receipt_sha256: str
    blueprint_receipt_file_sha256: str
    blueprint_payload_tree_sha256: str
    scientific_candidate_reference: str
    scientific_production_reference: str
    scientific_index_digest: str
    release_image_index_digest: str
    approval_environment: str
    runner_platform: str
    launcher_identity_file_sha256: str
    instantiated_root: str
    instantiated_payload_tree_sha256: str
    instantiated_payload_entries: tuple[str, ...]
    workloads: tuple[WorkloadSpecBinding, ...]
    schema_version: str = PRODUCTION_CONTROL_C0_INSTANTIATION_SCHEMA

    def __post_init__(self) -> None:
        for name in ("apparatus_commit", "candidate_image_source_commit"):
            value = getattr(self, name)
            if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
                raise ProductionControlError(f"{name} must be one full Git commit")
        for name in (
            "build_context_tree_sha256",
            "candidate_image_closure_file_sha256",
            "candidate_bootstrap_closure_sha256",
            "candidate_manifest_sha256",
            "candidate_manifest_file_sha256",
            "candidate_manifest_assembly_receipt_file_sha256",
            "materialization_config_file_sha256",
            "blueprint_receipt_sha256",
            "blueprint_receipt_file_sha256",
            "blueprint_payload_tree_sha256",
            "launcher_identity_file_sha256",
            "instantiated_payload_tree_sha256",
        ):
            _sha256(name, getattr(self, name))
        if (
            type(self.release_image_index_digest) is not str
            or _OCI_INDEX_DIGEST.fullmatch(self.release_image_index_digest) is None
        ):
            raise ProductionControlError("release_image_index_digest must be one OCI digest")
        shared = _scientific_image_transition(
            self.scientific_candidate_reference,
            self.scientific_production_reference,
        )
        if self.scientific_index_digest != shared:
            raise ProductionControlError("instantiation scientific image references differ from D")
        if self.runner_platform != FACTORY_RUNNER_PLATFORM:
            raise ProductionControlError("instantiation runner platform differs")
        if self.approval_environment != PRODUCTION_APPROVAL_ENVIRONMENT:
            raise ProductionControlError("instantiation approval environment differs")
        root = _absolute_path("instantiated_root", self.instantiated_root)
        object.__setattr__(self, "instantiated_root", str(root))
        if self.candidate_manifest_relative_path != C0_CANDIDATE_MANIFEST_RELATIVE_PATH:
            raise ProductionControlError("candidate manifest snapshot path differs")
        if (
            self.candidate_manifest_assembly_receipt_relative_path
            != C0_CANDIDATE_ASSEMBLY_RECEIPT_RELATIVE_PATH
        ):
            raise ProductionControlError("candidate assembly receipt snapshot path differs")
        entries = tuple(self.instantiated_payload_entries)
        if entries != _c0_instantiated_payload_entries():
            raise ProductionControlError("instantiated payload membership differs")
        object.__setattr__(self, "instantiated_payload_entries", entries)
        workloads = tuple(self.workloads)
        if tuple(row.corpus_id for row in workloads) != FIXED_CORPORA:
            raise ProductionControlError("instantiated workloads are not in fixed order")
        object.__setattr__(self, "workloads", workloads)
        if self.schema_version != PRODUCTION_CONTROL_C0_INSTANTIATION_SCHEMA:
            raise ProductionControlError("C0 control instantiation schema differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "apparatus_commit": self.apparatus_commit,
            "approval_environment": self.approval_environment,
            "blueprint_payload_tree_sha256": self.blueprint_payload_tree_sha256,
            "blueprint_receipt_file_sha256": self.blueprint_receipt_file_sha256,
            "blueprint_receipt_sha256": self.blueprint_receipt_sha256,
            "build_context_tree_sha256": self.build_context_tree_sha256,
            "candidate_bootstrap_closure_sha256": self.candidate_bootstrap_closure_sha256,
            "candidate_image_closure_file_sha256": self.candidate_image_closure_file_sha256,
            "candidate_image_source_commit": self.candidate_image_source_commit,
            "candidate_manifest_assembly_receipt_file_sha256": (
                self.candidate_manifest_assembly_receipt_file_sha256
            ),
            "candidate_manifest_assembly_receipt_relative_path": (
                self.candidate_manifest_assembly_receipt_relative_path
            ),
            "candidate_manifest_file_sha256": self.candidate_manifest_file_sha256,
            "candidate_manifest_relative_path": self.candidate_manifest_relative_path,
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "instantiated_payload_entries": list(self.instantiated_payload_entries),
            "instantiated_payload_tree_sha256": self.instantiated_payload_tree_sha256,
            "instantiated_root": self.instantiated_root,
            "launcher_identity_file_sha256": self.launcher_identity_file_sha256,
            "materialization_config_file_sha256": self.materialization_config_file_sha256,
            "release_image_index_digest": self.release_image_index_digest,
            "runner_platform": self.runner_platform,
            "schema_version": self.schema_version,
            "scientific_candidate_reference": self.scientific_candidate_reference,
            "scientific_index_digest": self.scientific_index_digest,
            "scientific_production_reference": self.scientific_production_reference,
            "workloads": [row.to_dict() for row in self.workloads],
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def file_sha256(self) -> str:
        return _sha256_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionControlC0InstantiationReceipt:
        row = _closed(
            value,
            frozenset(cls.__dataclass_fields__),
            label="C0 control instantiation receipt",
        )
        entries = row["instantiated_payload_entries"]
        workloads = row["workloads"]
        if type(entries) is not list or type(workloads) is not list:
            raise ProductionControlError("C0 control instantiation arrays are malformed")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key not in {"instantiated_payload_entries", "workloads"}
            },
            instantiated_payload_entries=tuple(entries),
            workloads=tuple(WorkloadSpecBinding.from_dict(item) for item in workloads),
        )


@dataclass(frozen=True)
class FinalizedCorpusBinding:
    corpus_id: str
    workload_spec_file_sha256: str
    config_file_sha256: str
    plan_template_file_sha256: str
    plan_template_semantic_sha256: str
    launcher_control_tree_sha256: str
    preflight_contract_sha256: str
    preflight_contract_file_sha256: str
    preflight_receipt_sha256: str
    preflight_receipt_file_sha256: str
    transition_receipt_sha256: str
    transition_receipt_file_sha256: str
    closure_binding: ProductionRunClosureBindingReceipt

    def __post_init__(self) -> None:
        if self.corpus_id not in FIXED_CORPORA:
            raise ProductionControlError("finalized corpus binding names another corpus")
        for name in (
            "workload_spec_file_sha256",
            "config_file_sha256",
            "plan_template_file_sha256",
            "plan_template_semantic_sha256",
            "launcher_control_tree_sha256",
            "preflight_contract_sha256",
            "preflight_contract_file_sha256",
            "preflight_receipt_sha256",
            "preflight_receipt_file_sha256",
            "transition_receipt_sha256",
            "transition_receipt_file_sha256",
        ):
            _sha256(name, getattr(self, name))
        if not isinstance(self.closure_binding, ProductionRunClosureBindingReceipt):
            raise ProductionControlError("finalized corpus closure binding must be typed")
        binding = self.closure_binding
        if (
            binding.corpus_id != self.corpus_id
            or binding.workload_spec_file_sha256 != self.workload_spec_file_sha256
            or binding.config_file_sha256 != self.config_file_sha256
            or binding.preflight_launcher_contract_sha256 != self.preflight_contract_sha256
            or binding.runtime_plan_transition_receipt_sha256 != self.transition_receipt_sha256
        ):
            raise ProductionControlError("corpus closure binding differs from its launch evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            field: (
                self.closure_binding.to_dict()
                if field == "closure_binding"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> FinalizedCorpusBinding:
        row = _closed(value, frozenset(cls.__dataclass_fields__), label="finalized corpus binding")
        return cls(
            **{key: item for key, item in row.items() if key != "closure_binding"},
            closure_binding=ProductionRunClosureBindingReceipt.from_dict(row["closure_binding"]),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductionControlFinalizationReceipt:
    materialization_config_sha256: str
    finalization_request_sha256: str
    blueprint_receipt_sha256: str
    c0_control_instantiation_receipt_file_sha256: str
    manifest_sha256: str
    c0_commit: str
    c1_commit: str
    sealed_run_receipt_file_sha256: str
    online_custody_admission_file_sha256: str
    launcher_identity_file_sha256: str
    provisional_closure_tree_sha256: str
    intermediate_closure_tree_sha256: str
    intermediate_closure_entries: tuple[str, ...]
    intermediate_sealed_run_receipt_byte_count: int
    instantiated_closure_tree_sha256: str
    instantiated_closure_entries: tuple[str, ...]
    retained_intermediate_closure_path: str
    suite_attempt_id: str
    canonical_suite_namespace: str
    pre_c1_output_staging_root: str
    corpora: tuple[FinalizedCorpusBinding, ...]
    schema_version: str = PRODUCTION_CONTROL_FINALIZATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PRODUCTION_CONTROL_FINALIZATION_SCHEMA:
            raise ProductionControlError("production control finalization schema differs")
        for name in (
            "materialization_config_sha256",
            "finalization_request_sha256",
            "blueprint_receipt_sha256",
            "c0_control_instantiation_receipt_file_sha256",
            "manifest_sha256",
            "sealed_run_receipt_file_sha256",
            "online_custody_admission_file_sha256",
            "launcher_identity_file_sha256",
            "provisional_closure_tree_sha256",
            "intermediate_closure_tree_sha256",
            "instantiated_closure_tree_sha256",
        ):
            _sha256(name, getattr(self, name))
        for name in ("c0_commit", "c1_commit"):
            if (
                type(getattr(self, name)) is not str
                or _GIT_COMMIT.fullmatch(getattr(self, name)) is None
            ):
                raise ProductionControlError(f"{name} must be one full Git commit")
        if self.c0_commit == self.c1_commit:
            raise ProductionControlError("C0 and C1 commits must differ")
        entries = tuple(self.intermediate_closure_entries)
        if entries != (f"{self.manifest_sha256}.json",):
            raise ProductionControlError(
                "the pre-finalization closure must contain only the sealed-run receipt"
            )
        object.__setattr__(self, "intermediate_closure_entries", entries)
        _positive_integer(
            "intermediate_sealed_run_receipt_byte_count",
            self.intermediate_sealed_run_receipt_byte_count,
        )
        final_entries = tuple(self.instantiated_closure_entries)
        if (
            not final_entries
            or final_entries != tuple(sorted(final_entries, key=lambda item: item.encode("utf-8")))
            or len(final_entries) != len(set(final_entries))
        ):
            raise ProductionControlError("final closure entries must be unique and sorted")
        object.__setattr__(self, "instantiated_closure_entries", final_entries)
        object.__setattr__(
            self,
            "retained_intermediate_closure_path",
            str(
                _absolute_path(
                    "retained_intermediate_closure_path",
                    self.retained_intermediate_closure_path,
                )
            ),
        )
        _sha256("suite_attempt_id", self.suite_attempt_id)
        for name in ("canonical_suite_namespace", "pre_c1_output_staging_root"):
            object.__setattr__(
                self,
                name,
                str(_absolute_path(name, getattr(self, name))),
            )
        rows = tuple(self.corpora)
        if tuple(row.corpus_id for row in rows) != FIXED_CORPORA:
            raise ProductionControlError("finalized corpus rows must follow FIXED_CORPORA")
        object.__setattr__(self, "corpora", rows)
        shared = tuple(
            (
                row.closure_binding.manifest_sha256,
                row.closure_binding.closure_source,
                row.closure_binding.closure_target,
                row.closure_binding.provisional_closure_tree_sha256,
                row.closure_binding.instantiated_closure_tree_sha256,
                row.closure_binding.sealed_run_receipt_relative_path,
                row.closure_binding.sealed_run_receipt_file_sha256,
                row.closure_binding.entries,
                row.closure_binding.files,
            )
            for row in rows
        )
        if len(set(shared)) != 1:
            raise ProductionControlError("five corpus bindings do not share one exact closure")
        binding = rows[0].closure_binding
        if (
            binding.manifest_sha256 != self.manifest_sha256
            or binding.provisional_closure_tree_sha256 != self.provisional_closure_tree_sha256
            or binding.instantiated_closure_tree_sha256 != self.instantiated_closure_tree_sha256
            or binding.entries != self.instantiated_closure_entries
            or binding.sealed_run_receipt_file_sha256 != self.sealed_run_receipt_file_sha256
        ):
            raise ProductionControlError("shared closure binding differs from finalization state")

    def to_dict(self) -> dict[str, object]:
        return {
            "blueprint_receipt_sha256": self.blueprint_receipt_sha256,
            "c0_control_instantiation_receipt_file_sha256": (
                self.c0_control_instantiation_receipt_file_sha256
            ),
            "c0_commit": self.c0_commit,
            "c1_commit": self.c1_commit,
            "corpora": [row.to_dict() for row in self.corpora],
            "finalization_request_sha256": self.finalization_request_sha256,
            "launcher_identity_file_sha256": self.launcher_identity_file_sha256,
            "manifest_sha256": self.manifest_sha256,
            "materialization_config_sha256": self.materialization_config_sha256,
            "online_custody_admission_file_sha256": self.online_custody_admission_file_sha256,
            "instantiated_closure_entries": list(self.instantiated_closure_entries),
            "instantiated_closure_tree_sha256": self.instantiated_closure_tree_sha256,
            "intermediate_closure_entries": list(self.intermediate_closure_entries),
            "intermediate_closure_tree_sha256": self.intermediate_closure_tree_sha256,
            "intermediate_sealed_run_receipt_byte_count": (
                self.intermediate_sealed_run_receipt_byte_count
            ),
            "provisional_closure_tree_sha256": self.provisional_closure_tree_sha256,
            "retained_intermediate_closure_path": self.retained_intermediate_closure_path,
            "suite_attempt_id": self.suite_attempt_id,
            "canonical_suite_namespace": self.canonical_suite_namespace,
            "pre_c1_output_staging_root": self.pre_c1_output_staging_root,
            "schema_version": self.schema_version,
            "sealed_run_receipt_file_sha256": self.sealed_run_receipt_file_sha256,
        }

    def canonical_file_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict()) + b"\n"

    @property
    def receipt_sha256(self) -> str:
        return _sha256_bytes(self.canonical_file_bytes())

    @classmethod
    def from_dict(cls, value: object) -> ProductionControlFinalizationReceipt:
        fields = frozenset(
            {
                "blueprint_receipt_sha256",
                "c0_control_instantiation_receipt_file_sha256",
                "c0_commit",
                "c1_commit",
                "corpora",
                "finalization_request_sha256",
                "launcher_identity_file_sha256",
                "manifest_sha256",
                "materialization_config_sha256",
                "online_custody_admission_file_sha256",
                "instantiated_closure_entries",
                "instantiated_closure_tree_sha256",
                "intermediate_closure_entries",
                "intermediate_closure_tree_sha256",
                "intermediate_sealed_run_receipt_byte_count",
                "provisional_closure_tree_sha256",
                "retained_intermediate_closure_path",
                "suite_attempt_id",
                "canonical_suite_namespace",
                "pre_c1_output_staging_root",
                "schema_version",
                "sealed_run_receipt_file_sha256",
            }
        )
        row = _closed(value, fields, label="production control finalization receipt")
        corpora = row["corpora"]
        intermediate_entries = row["intermediate_closure_entries"]
        instantiated_entries = row["instantiated_closure_entries"]
        if not all(
            type(item) is list for item in (corpora, intermediate_entries, instantiated_entries)
        ):
            raise ProductionControlError("finalization arrays are malformed")
        return cls(
            **{
                key: item
                for key, item in row.items()
                if key
                not in {
                    "corpora",
                    "instantiated_closure_entries",
                    "intermediate_closure_entries",
                }
            },
            corpora=tuple(FinalizedCorpusBinding.from_dict(item) for item in corpora),
            instantiated_closure_entries=tuple(instantiated_entries),
            intermediate_closure_entries=tuple(intermediate_entries),
        )


def load_production_control_config(
    path: str | Path,
    *,
    expected_sha256: str,
) -> ProductionControlMaterializationConfig:
    encoded = _read_pinned(path, expected_sha256, label="production control config")
    config = ProductionControlMaterializationConfig.from_dict(
        _parse_object(encoded, label="production control config")
    )
    if encoded != config.canonical_file_bytes():
        raise ProductionControlError("production control config bytes are not canonical")
    return config


def load_production_control_config_write_receipt(
    path: str | Path,
) -> ProductionControlMaterializationConfigWriteReceipt:
    encoded = _read(path, label="production control config write receipt")
    receipt = ProductionControlMaterializationConfigWriteReceipt.from_dict(
        _parse_object(encoded, label="production control config write receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionControlError(
            "production control config write receipt bytes are not canonical"
        )
    return receipt


def load_production_control_finalization_request(
    path: str | Path,
    *,
    expected_sha256: str,
) -> ProductionControlFinalizationRequest:
    encoded = _read_pinned(path, expected_sha256, label="control finalization request")
    request = ProductionControlFinalizationRequest.from_dict(
        _parse_object(encoded, label="control finalization request")
    )
    if encoded != request.canonical_file_bytes():
        raise ProductionControlError("control finalization request bytes are not canonical")
    return request


def load_production_control_blueprint_receipt(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ProductionControlBlueprintReceipt:
    encoded = _read(path, label="production control blueprint receipt")
    if expected_sha256 is not None and _sha256_bytes(encoded) != _sha256(
        "blueprint receipt SHA-256", expected_sha256
    ):
        raise ProductionControlError("blueprint receipt differs from its external pin")
    receipt = ProductionControlBlueprintReceipt.from_dict(
        _parse_object(encoded, label="production control blueprint receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionControlError("blueprint receipt bytes are not canonical")
    return receipt


def load_production_control_finalization_receipt(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ProductionControlFinalizationReceipt:
    encoded = _read(path, label="production control finalization receipt")
    if expected_sha256 is not None and _sha256_bytes(encoded) != _sha256(
        "finalization receipt SHA-256", expected_sha256
    ):
        raise ProductionControlError("finalization receipt differs from its external pin")
    receipt = ProductionControlFinalizationReceipt.from_dict(
        _parse_object(encoded, label="production control finalization receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionControlError("finalization receipt bytes are not canonical")
    return receipt


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionControlError(f"cannot prepare private directory {path}: {exc}") from exc
    if (
        path.is_symlink()
        or resolved != path
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ProductionControlError(f"private directory is unsafe: {path}")


def _publish_exact(path: Path, encoded: bytes, *, resume: bool, label: str) -> None:
    if os.path.lexists(path):
        if not resume:
            raise ProductionControlError(f"{label} already exists; use resume after custody review")
        if _read(path, label=label) != encoded:
            raise ProductionControlError(f"existing {label} differs from the derivation")
        return
    try:
        write_exclusive_receipt_bytes(encoded, path)
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(f"cannot publish {label}: {exc}") from exc


def _scan_exact_tree(root: Path, expected: frozenset[str], *, label: str) -> None:
    try:
        digest = digest_directory_tree(root)
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(f"cannot inspect {label}: {exc}") from exc
    observed = frozenset(digest.entries)
    if observed != expected:
        raise ProductionControlError(
            f"{label} membership differs; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _payload_tree_sha256(root: Path, entries: Sequence[str]) -> str:
    try:
        return digest_directory_tree(root, included_entries=entries).sha256
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(f"cannot hash production control payload: {exc}") from exc


def _blueprint_payload_entries() -> tuple[str, ...]:
    entries = [
        LAUNCHER_IDENTITY_FILENAME,
        PRODUCTION_HARDWARE_FRAGMENT_FILENAME,
        PRODUCTION_WORKLOADS_FRAGMENT_FILENAME,
    ]
    for corpus_id in FIXED_CORPORA:
        entries.extend(
            (
                corpus_id,
                f"{corpus_id}/{PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME}",
                f"{corpus_id}/launcher-control",
                f"{corpus_id}/launcher-control/{PLAN_TEMPLATE_FILENAME}",
                f"{corpus_id}/{PREFLIGHT_CONTRACT_FILENAME}",
            )
        )
    return tuple(entries)


def _blueprint_all_entries() -> frozenset[str]:
    return frozenset((*_blueprint_payload_entries(), BLUEPRINT_RECEIPT_FILENAME))


def _c0_instantiated_payload_entries() -> tuple[str, ...]:
    return (
        *_blueprint_payload_entries(),
        C0_CANDIDATE_PACKAGE_DIRECTORY,
        C0_CANDIDATE_MANIFEST_RELATIVE_PATH,
        C0_CANDIDATE_ASSEMBLY_RECEIPT_RELATIVE_PATH,
    )


def _production_workloads_fragment(
    specs: Sequence[ProductionCorpusWorkloadSpec],
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(
        {
            "canonical_file_sha256": spec.file_sha256,
            "corpus_id": spec.corpus_id,
            "spec": spec.to_dict(),
        }
        for spec in specs
    )
    if tuple(row["corpus_id"] for row in rows) != FIXED_CORPORA:
        raise ProductionControlError("production workload fragment must follow FIXED_CORPORA")
    return rows


def _validate_candidate_workload_templates(
    specs: Sequence[ProductionCorpusWorkloadSpec],
    *,
    materialization: ProductionControlMaterializationConfig,
    selected_family_count: int,
) -> None:
    """Admit the five A-independent workload templates before C0 exists."""

    rows = tuple(specs)
    if tuple(spec.corpus_id for spec in rows) != FIXED_CORPORA:
        raise ProductionControlError("candidate workload templates are not in fixed order")
    expected_runner_identity = f"github-actions:environment:{materialization.approval_environment}"
    for spec in rows:
        if (
            spec.code_commit != C0_COMMIT_SENTINEL
            or spec.runner_identity != materialization.runner_identity
            or spec.runner_identity != expected_runner_identity
            or spec.runner_image != materialization.scientific_production_reference
            or spec.selected_family_count != selected_family_count
        ):
            raise ProductionControlError(
                f"{spec.corpus_id} candidate workload execution identity differs"
            )


def _load_candidate_workload_rows(
    value: object,
    *,
    bindings: Sequence[WorkloadSpecBinding],
) -> tuple[tuple[Mapping[str, Any], ProductionCorpusWorkloadSpec], ...]:
    """Parse the raw pre-C0 rows and prove that every hash covers sentinel bytes."""

    if not isinstance(value, list) or len(value) != len(FIXED_CORPORA):
        raise ProductionControlError("candidate workload fragment must contain five rows")
    rows: list[tuple[Mapping[str, Any], ProductionCorpusWorkloadSpec]] = []
    for position, (item, corpus_id, binding) in enumerate(
        zip(value, FIXED_CORPORA, bindings, strict=True)
    ):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"canonical_file_sha256", "corpus_id", "spec"}
            or item.get("corpus_id") != corpus_id
        ):
            raise ProductionControlError(
                f"candidate workload fragment row {position} is not closed"
            )
        try:
            spec = ProductionCorpusWorkloadSpec.from_dict(item["spec"])
        except (KeyError, TypeError, ProductionCorpusRunError, TrialRuntimeError) as exc:
            raise ProductionControlError(
                f"candidate workload fragment is invalid for {corpus_id}"
            ) from exc
        if (
            spec.corpus_id != corpus_id
            or spec.code_commit != C0_COMMIT_SENTINEL
            or item.get("canonical_file_sha256") != spec.file_sha256
            or binding.corpus_id != corpus_id
            or binding.file_sha256 != spec.file_sha256
        ):
            raise ProductionControlError(f"candidate {corpus_id} workload hash or sentinel differs")
        rows.append((item, spec))
    return tuple(rows)


def _resolve_candidate_workload_rows(
    rows: Sequence[tuple[Mapping[str, Any], ProductionCorpusWorkloadSpec]],
    *,
    apparatus_commit: str,
) -> tuple[Mapping[str, Any], ...]:
    """Derive the frozen A-bound wrappers without changing the raw blueprint."""

    if _GIT_COMMIT.fullmatch(apparatus_commit) is None:
        raise ProductionControlError("apparatus commit must be one full Git commit")
    return tuple(
        {
            "canonical_file_sha256": resolved.file_sha256,
            "corpus_id": resolved.corpus_id,
            "spec": resolved.to_dict(),
        }
        for _, candidate in rows
        for resolved in (replace(candidate, code_commit=apparatus_commit),)
    )


def _production_workloads_fragment_file_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    return _canonical_bytes(list(rows)) + b"\n"


def _production_hardware_fragment(
    materialization: ProductionControlMaterializationConfig,
) -> Mapping[str, object]:
    return {
        "accelerator": materialization.hardware_accelerator,
        "cpu_model": materialization.hardware_cpu_model,
        "instance_type": materialization.hardware_instance_type,
        "logical_cores": len(materialization.cpuset_cpus),
        "memory_gib": materialization.memory_limit_bytes // _GIB,
        "operating_system": materialization.hardware_operating_system,
        "provider": materialization.hardware_provider,
        "region": materialization.hardware_region,
    }


def _production_hardware_fragment_file_bytes(
    hardware: Mapping[str, object],
) -> bytes:
    return _canonical_bytes(hardware) + b"\n"


def _finalized_payload_entries(manifest_digest: str) -> tuple[str, ...]:
    entries: list[str] = []
    control_names = (
        ONLINE_CUSTODY_ADMISSION_FILENAME,
        PRODUCTION_CORPUS_CONFIG_FILENAME,
        PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME,
        REQUIRED_ARTIFACT_BINDINGS_FILENAME,
        SHARDED_EXECUTION_PLAN_FILENAME,
        TRIAL_RUNTIME_RECEIPT_FILENAME,
    )
    entries.append(f"{manifest_digest}.json")
    for corpus_id in FIXED_CORPORA:
        prefix = f"{corpus_id}"
        entries.extend((prefix, f"{prefix}/control"))
        entries.extend(f"{prefix}/control/{name}" for name in control_names)
    return tuple(entries)


def _finalized_all_entries(manifest_digest: str) -> frozenset[str]:
    return frozenset(_finalized_payload_entries(manifest_digest))


@dataclass(frozen=True)
class _AdmittedFactory:
    config: ProductionArtifactFactoryConfig
    suite: ProductionArtifactFactorySuiteReceipt
    extraction: C0RuntimeExtractionReceipt
    staged_root: Path


def _admit_factory(config: ProductionControlMaterializationConfig) -> _AdmittedFactory:
    factory = load_production_artifact_factory_config(
        config.factory_config_path,
        expected_sha256=config.factory_config_sha256,
    )
    verified = verify_production_artifact_factory(factory)
    persisted = load_production_artifact_factory_suite(
        config.factory_suite_receipt_path,
        expected_sha256=config.factory_suite_receipt_sha256,
    )
    if verified != persisted or config.factory_suite_receipt_path != factory.suite_receipt_path:
        raise ProductionControlError("factory terminal receipt differs from reproduced state")
    try:
        factory_tree = digest_directory_tree(factory.artifact_root).sha256
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(f"cannot hash the verified factory tree: {exc}") from exc
    if factory_tree != config.factory_artifact_tree_sha256:
        raise ProductionControlError("factory artifact tree differs from the external pin")
    extraction_bytes = _read_pinned(
        config.c0_runtime_extraction_receipt_path,
        config.c0_runtime_extraction_receipt_sha256,
        label="C0 runtime extraction receipt",
    )
    extraction = load_c0_runtime_extraction_receipt(config.c0_runtime_extraction_receipt_path)
    if _sha256_bytes(extraction_bytes) != config.c0_runtime_extraction_receipt_sha256:
        raise ProductionControlError("C0 extraction receipt changed during admission")
    if (
        factory.runner_image != config.scientific_candidate_reference
        or persisted.runner_image != config.scientific_candidate_reference
        or extraction.image_reference != config.scientific_candidate_reference
        or extraction.c0_sha != config.candidate_image_source_commit
        or factory.runner_platform != config.runner_platform
        or persisted.runner_platform != config.runner_platform
        or extraction.platform != config.runner_platform
    ):
        raise ProductionControlError("factory and candidate-image runner identities differ")
    for path, expected, label in (
        (config.opa_binary_path, extraction.opa_sha256, "retained C0 OPA binary"),
        (config.uv_lock_path, extraction.uv_lock_sha256, "C0 uv lock"),
        (config.pseudonym_key_path, factory.hmac_secret_sha256, "pseudonym key"),
    ):
        try:
            observed = digest_regular_file(path, label=label)
        except ArtifactIntegrityError as exc:
            raise ProductionControlError(f"cannot verify {label}: {exc}") from exc
        if observed != expected:
            raise ProductionControlError(f"{label} differs from its config pin")
    if (
        extraction.opa_sha256 != config.opa_binary_sha256
        or extraction.uv_lock_sha256 != config.uv_lock_sha256
        or config.opa_binary_path.stat().st_size != extraction.opa_byte_count
        or config.uv_lock_path.stat().st_size != extraction.uv_lock_byte_count
        or config.pseudonym_key_sha256 != factory.hmac_secret_sha256
        or persisted.hmac_secret_sha256 != factory.hmac_secret_sha256
        or persisted.hmac_key_id != factory.hmac_key_id
        or factory.hmac_key_id != f"sealed-online-ephemeral-sha256-{factory.hmac_secret_sha256}"
    ):
        raise ProductionControlError(
            "runtime or pseudonym material differs from verified C0/factory custody"
        )
    embedding = load_production_embedding_config(
        factory.embedding_build_config_path,
        expected_sha256=factory.embedding_build_config_sha256,
    )
    sources = (
        factory.artifact_root,
        embedding.online_staging_root,
        config.opa_binary_path,
        config.uv_lock_path,
        config.pseudonym_key_path,
    )
    destinations = (
        config.blueprint_root,
        config.finalized_controls_root,
        config.suite_base_root,
    )
    if any(_paths_overlap(source, target) for source in sources for target in destinations):
        raise ProductionControlError("a production control destination overlaps an input")
    return _AdmittedFactory(
        config=factory,
        suite=persisted,
        extraction=extraction,
        staged_root=embedding.online_staging_root,
    )


def _assert_materialization_config_source_snapshot(
    config: ProductionControlMaterializationConfig,
    admitted: _AdmittedFactory,
) -> None:
    for path, expected, label in (
        (config.factory_config_path, config.factory_config_sha256, "factory config"),
        (
            config.factory_suite_receipt_path,
            config.factory_suite_receipt_sha256,
            "factory suite receipt",
        ),
        (
            config.c0_runtime_extraction_receipt_path,
            config.c0_runtime_extraction_receipt_sha256,
            "C0 runtime extraction receipt",
        ),
    ):
        _read_pinned(path, expected, label=label)
    for path, expected, expected_size, label in (
        (
            config.opa_binary_path,
            config.opa_binary_sha256,
            admitted.extraction.opa_byte_count,
            "retained C0 OPA binary",
        ),
        (
            config.uv_lock_path,
            config.uv_lock_sha256,
            admitted.extraction.uv_lock_byte_count,
            "C0 uv lock",
        ),
        (
            config.pseudonym_key_path,
            config.pseudonym_key_sha256,
            None,
            "pseudonym key",
        ),
    ):
        encoded = _read(path, label=label)
        if _sha256_bytes(encoded) != expected or (
            expected_size is not None and len(encoded) != expected_size
        ):
            raise ProductionControlError(f"{label} changed before config publication")
    try:
        tree_sha256 = digest_directory_tree(admitted.config.artifact_root).sha256
    except ArtifactIntegrityError as exc:
        raise ProductionControlError("cannot rehash the verified factory tree") from exc
    if tree_sha256 != config.factory_artifact_tree_sha256:
        raise ProductionControlError("verified factory tree changed before config publication")


def write_production_control_materialization_config(
    *,
    factory_config_path: str | Path,
    c0_runtime_extraction_receipt_path: str | Path,
    opa_binary_path: str | Path,
    uv_lock_path: str | Path,
    pseudonym_key_path: str | Path,
    scientific_candidate_reference: str,
    scientific_production_reference: str,
    approval_environment: str,
    runner_platform: str,
    runner_identity: str,
    hostname: str,
    hardware_provider: str,
    hardware_instance_type: str,
    hardware_cpu_model: str,
    hardware_accelerator: str,
    hardware_region: str,
    hardware_operating_system: str,
    memory_limit_bytes: int,
    cpuset_cpus: Sequence[int] | str,
    tmpfs_size_bytes: int,
    blueprint_root: str | Path,
    finalized_controls_root: str | Path,
    suite_base_root: str | Path,
    output: str | Path,
    receipt_output: str | Path,
) -> ProductionControlMaterializationConfigWriteReceipt:
    """Derive and publish one closed materialization config without hand-authored JSON."""

    paths = {
        "factory_config_path": _absolute_path("factory_config_path", factory_config_path),
        "c0_runtime_extraction_receipt_path": _absolute_path(
            "c0_runtime_extraction_receipt_path",
            c0_runtime_extraction_receipt_path,
        ),
        "opa_binary_path": _absolute_path("opa_binary_path", opa_binary_path),
        "uv_lock_path": _absolute_path("uv_lock_path", uv_lock_path),
        "pseudonym_key_path": _absolute_path("pseudonym_key_path", pseudonym_key_path),
        "blueprint_root": _absolute_path("blueprint_root", blueprint_root),
        "finalized_controls_root": _absolute_path(
            "finalized_controls_root",
            finalized_controls_root,
        ),
        "suite_base_root": _absolute_path("suite_base_root", suite_base_root),
        "output": _absolute_path("output", output),
        "receipt_output": _absolute_path("receipt_output", receipt_output),
    }
    cpus = _canonical_cpuset_cpus(cpuset_cpus)
    scientific_index_digest = _scientific_image_transition(
        scientific_candidate_reference,
        scientific_production_reference,
    )
    destinations = (
        paths["blueprint_root"],
        paths["finalized_controls_root"],
        paths["suite_base_root"],
    )
    if any(
        _paths_overlap(first, second)
        for position, first in enumerate(destinations)
        for second in destinations[position + 1 :]
    ):
        raise ProductionControlError("production control destination roots overlap")
    for label, path in (
        ("blueprint_root", paths["blueprint_root"]),
        ("finalized_controls_root", paths["finalized_controls_root"]),
        ("suite_base_root", paths["suite_base_root"]),
        ("output parent", paths["output"].parent),
        ("receipt output parent", paths["receipt_output"].parent),
    ):
        _assert_unsymlinked_root(path, label=label)
    if _paths_overlap(paths["output"], paths["receipt_output"]):
        raise ProductionControlError("config and write receipt paths overlap")
    if os.path.lexists(paths["receipt_output"]) and not os.path.lexists(paths["output"]):
        raise ProductionControlError(
            "config write receipt exists without its materialization config"
        )
    if any(
        _paths_overlap(target, root)
        for target in (paths["output"], paths["receipt_output"])
        for root in destinations
    ):
        raise ProductionControlError("config publication path overlaps a destination root")

    try:
        factory_config_bytes = _read(
            paths["factory_config_path"],
            label="production artifact factory config",
        )
        factory_config_sha256 = _sha256_bytes(factory_config_bytes)
        factory = load_production_artifact_factory_config(
            paths["factory_config_path"],
            expected_sha256=factory_config_sha256,
        )
        if (
            scientific_candidate_reference != factory.runner_image
            or runner_platform != factory.runner_platform
        ):
            raise ProductionControlError(
                "candidate runner image or platform differs from the verified factory config"
            )
        factory_suite_receipt_path = factory.suite_receipt_path
        factory_suite_bytes = _read(
            factory_suite_receipt_path,
            label="production artifact factory suite receipt",
        )
        factory_suite_receipt_sha256 = _sha256_bytes(factory_suite_bytes)
        persisted_suite = load_production_artifact_factory_suite(
            factory_suite_receipt_path,
            expected_sha256=factory_suite_receipt_sha256,
        )
        factory_artifact_tree_sha256 = digest_directory_tree(factory.artifact_root).sha256

        extraction_bytes = _read(
            paths["c0_runtime_extraction_receipt_path"],
            label="C0 runtime extraction receipt",
        )
        extraction_sha256 = _sha256_bytes(extraction_bytes)
        extraction = load_c0_runtime_extraction_receipt(paths["c0_runtime_extraction_receipt_path"])
        if (
            extraction.image_reference != scientific_candidate_reference
            or extraction.platform != runner_platform
            or persisted_suite.runner_image != scientific_candidate_reference
            or persisted_suite.runner_platform != runner_platform
        ):
            raise ProductionControlError(
                "candidate runner image or platform differs across factory and C0 receipts"
            )
        opa_bytes = _read(paths["opa_binary_path"], label="retained C0 OPA binary")
        uv_lock_bytes = _read(paths["uv_lock_path"], label="C0 uv lock")
        pseudonym_key_bytes = _read(paths["pseudonym_key_path"], label="pseudonym key")
        opa_sha256 = _sha256_bytes(opa_bytes)
        uv_lock_sha256 = _sha256_bytes(uv_lock_bytes)
        pseudonym_key_sha256 = _sha256_bytes(pseudonym_key_bytes)
        if (
            opa_sha256 != extraction.opa_sha256
            or len(opa_bytes) != extraction.opa_byte_count
            or uv_lock_sha256 != extraction.uv_lock_sha256
            or len(uv_lock_bytes) != extraction.uv_lock_byte_count
            or pseudonym_key_sha256 != factory.hmac_secret_sha256
            or persisted_suite.hmac_secret_sha256 != factory.hmac_secret_sha256
            or persisted_suite.hmac_key_id != factory.hmac_key_id
        ):
            raise ProductionControlError(
                "runtime files or pseudonym key differ from verified factory evidence"
            )
        if any(
            _paths_overlap(target, source)
            for target in (paths["output"], paths["receipt_output"])
            for source in (
                factory.artifact_root,
                factory.embedding_source_root,
                paths["factory_config_path"],
                paths["c0_runtime_extraction_receipt_path"],
                paths["opa_binary_path"],
                paths["uv_lock_path"],
                paths["pseudonym_key_path"],
            )
        ):
            raise ProductionControlError("config publication path overlaps a verified input")

        config = ProductionControlMaterializationConfig(
            factory_config_path=paths["factory_config_path"],
            factory_config_sha256=factory_config_sha256,
            factory_suite_receipt_path=factory_suite_receipt_path,
            factory_suite_receipt_sha256=factory_suite_receipt_sha256,
            factory_artifact_tree_sha256=factory_artifact_tree_sha256,
            c0_runtime_extraction_receipt_path=paths["c0_runtime_extraction_receipt_path"],
            c0_runtime_extraction_receipt_sha256=extraction_sha256,
            candidate_image_source_commit=extraction.c0_sha,
            opa_binary_path=paths["opa_binary_path"],
            opa_binary_sha256=opa_sha256,
            uv_lock_path=paths["uv_lock_path"],
            uv_lock_sha256=uv_lock_sha256,
            pseudonym_key_path=paths["pseudonym_key_path"],
            pseudonym_key_sha256=pseudonym_key_sha256,
            scientific_candidate_reference=scientific_candidate_reference,
            scientific_production_reference=scientific_production_reference,
            scientific_index_digest=scientific_index_digest,
            oci_promotion_required=True,
            approval_environment=approval_environment,
            runner_platform=runner_platform,
            runner_identity=runner_identity,
            hostname=hostname,
            hardware_provider=hardware_provider,
            hardware_instance_type=hardware_instance_type,
            hardware_cpu_model=hardware_cpu_model,
            hardware_accelerator=hardware_accelerator,
            hardware_region=hardware_region,
            hardware_operating_system=hardware_operating_system,
            memory_limit_bytes=memory_limit_bytes,
            cpuset_cpus=cpus,
            tmpfs_size_bytes=tmpfs_size_bytes,
            blueprint_root=paths["blueprint_root"],
            finalized_controls_root=paths["finalized_controls_root"],
            suite_base_root=paths["suite_base_root"],
        )
        admitted = _admit_factory(config)
        if admitted.config != factory or admitted.suite != persisted_suite:
            raise ProductionControlError(
                "factory suite receipt differs from reproduced factory state"
            )
        _assert_materialization_config_source_snapshot(config, admitted)
    except ProductionControlError:
        raise
    except (
        ArtifactIntegrityError,
        OpaRuntimeBinaryError,
        ProductionArtifactFactoryError,
        OSError,
    ) as exc:
        raise ProductionControlError(
            f"cannot derive production control materialization config: {exc}"
        ) from exc

    encoded = config.canonical_file_bytes()
    _publish_or_recover_exact_file(
        paths["output"],
        encoded,
        label="materialization config",
    )
    readback = load_production_control_config(
        paths["output"],
        expected_sha256=config.file_sha256,
    )
    if readback != config:
        raise ProductionControlError("materialization config typed readback differs")
    output_metadata = paths["output"].lstat()
    if stat.S_IMODE(output_metadata.st_mode) != 0o600:
        raise ProductionControlError("materialization config readback mode differs from 0600")
    receipt = ProductionControlMaterializationConfigWriteReceipt(
        config_path=paths["output"],
        config_file_sha256=config.file_sha256,
        config_readback_sha256=_sha256_bytes(
            _read(paths["output"], label="published materialization config")
        ),
        config_byte_count=len(encoded),
        config_mode="0600",
        factory_config_file_sha256=config.factory_config_sha256,
        factory_suite_receipt_file_sha256=config.factory_suite_receipt_sha256,
        factory_artifact_tree_sha256=config.factory_artifact_tree_sha256,
        c0_runtime_extraction_receipt_file_sha256=(config.c0_runtime_extraction_receipt_sha256),
        candidate_image_source_commit=config.candidate_image_source_commit,
        opa_binary_sha256=config.opa_binary_sha256,
        uv_lock_sha256=config.uv_lock_sha256,
        pseudonym_key_sha256=config.pseudonym_key_sha256,
        scientific_candidate_reference=config.scientific_candidate_reference,
        scientific_production_reference=config.scientific_production_reference,
        scientific_index_digest=config.scientific_index_digest,
        oci_promotion_required=config.oci_promotion_required,
        approval_environment=config.approval_environment,
        readback_verified=True,
    )
    _publish_or_recover_exact_file(
        paths["receipt_output"],
        receipt.canonical_file_bytes(),
        label="materialization config write receipt",
    )
    if load_production_control_config_write_receipt(paths["receipt_output"]) != receipt:
        raise ProductionControlError("materialization config write receipt readback differs")
    return receipt


def _suite_corpus(
    suite: ProductionArtifactFactorySuiteReceipt,
    corpus_id: str,
) -> FactorySuiteCorpus:
    try:
        return next(row for row in suite.corpora if row.corpus_id == corpus_id)
    except StopIteration as exc:
        raise ProductionControlError("factory suite omits a fixed corpus") from exc


def _feature_bindings(
    receipt: TrialRuntimeAdmissionReceipt,
) -> tuple[RuntimeFeatureBinding, ...]:
    rows = tuple(
        RuntimeFeatureBinding(
            group_order=group.group_order,
            subject=group.subject,
            repetition=group.repetition,
            policy_state=group.policy_state,
            version_lag=VERSION_LAG,
            backend=RUNTIME_BACKEND,
            drift_family=RUNTIME_DRIFT_FAMILY,
            policy_complexity=group.realized_allow_rate,
        )
        for group in receipt.groups
    )
    if tuple(row.group_order for row in rows) != (0, 1, 2):
        raise ProductionControlError("runtime feature rows do not follow the fixed block order")
    return rows


def _tree(path: Path, *, label: str) -> str:
    try:
        return digest_directory_tree(path).sha256
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(f"cannot hash {label}: {exc}") from exc


def _file(path: Path, *, label: str) -> str:
    try:
        return digest_regular_file(path, label=label)
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(f"cannot hash {label}: {exc}") from exc


def _derive_workload_spec(
    materialization: ProductionControlMaterializationConfig,
    admitted: _AdmittedFactory,
    corpus_id: str,
    *,
    code_commit: str = C0_COMMIT_SENTINEL,
) -> ProductionCorpusWorkloadSpec:
    factory = admitted.config
    suite_row = _suite_corpus(admitted.suite, corpus_id)
    factory_row = factory.corpus(corpus_id)
    online_root = factory.artifact_root / "custody" / "online" / corpus_id
    index_root = factory.artifact_root / "authorized-index-stores" / corpus_id / "sealed"
    embedding_root = factory.artifact_root / "embedding-stores" / corpus_id
    policy_root = factory.artifact_root / "policy-workloads" / corpus_id / "sealed"
    runtime_root = factory.artifact_root / "trial-runtime" / corpus_id
    query_root = runtime_root / RUNTIME_QUERY_DIRECTORY
    execution_path = online_root / ONLINE_EXECUTION_PLAN_FILENAME
    runtime_path = runtime_root / RUNTIME_RECEIPT_FILENAME
    query_receipt_path = query_root / QUERY_TRIAL_RECEIPT_FILENAME

    execution = load_sharded_online_execution_plan(execution_path)
    runtime = load_trial_runtime_receipt(runtime_path)
    index = load_authorized_index_store_receipt(index_root)
    policy = load_policy_intervention_receipt(policy_root / POLICY_RECEIPT_FILENAME)
    embedding = load_embedding_store_receipt(embedding_root)
    online_tree = _tree(online_root, label=f"{corpus_id} online package")
    execution_file = _file(execution_path, label=f"{corpus_id} execution plan")
    runtime_file = _file(runtime_path, label=f"{corpus_id} runtime receipt")
    query_receipt_file = _file(query_receipt_path, label=f"{corpus_id} query receipt")
    if (
        execution.artifact_sha256 != suite_row.online_execution_plan_sha256
        or online_tree != suite_row.online_execution_tree_sha256
        or runtime_file != suite_row.runtime_receipt_sha256
        or query_receipt_file != suite_row.query_receipt_sha256
        or runtime.execution_artifact_sha256 != execution.artifact_sha256
        or runtime.embedding_store_receipt_sha256 != embedding.receipt_sha256
        or index.policy_receipt_sha256 != policy.artifact_sha256
        or index.embedding_receipt_sha256 != embedding.receipt_sha256
    ):
        raise ProductionControlError(f"{corpus_id} subordinate factory pins differ")

    return ProductionCorpusWorkloadSpec(
        corpus_id=corpus_id,
        available_family_count=factory_row.available_family_count,
        selected_family_count=factory.selected_family_count,
        factory_config_sha256=factory.file_sha256,
        factory_suite_receipt_sha256=admitted.suite.receipt_sha256,
        factory_artifact_tree_sha256=materialization.factory_artifact_tree_sha256,
        runner_image=materialization.scientific_production_reference,
        runner_platform=materialization.runner_platform,
        runner_identity=materialization.runner_identity,
        code_commit=code_commit,
        artifact_root=Path(_CONTAINER_PATHS["artifact_root"]),
        artifact_tree_sha256=online_tree,
        authorized_index_store_root=Path(_CONTAINER_PATHS["authorized_index_store_root"]),
        authorized_index_store_tree_sha256=_tree(
            index_root,
            label=f"{corpus_id} authorized index store",
        ),
        embedding_store_root=Path(_CONTAINER_PATHS["embedding_store_root"]),
        embedding_store_tree_sha256=_tree(
            embedding_root,
            label=f"{corpus_id} embedding store",
        ),
        partition_audit_path=Path(_CONTAINER_PATHS["partition_audit_path"]),
        partition_audit_file_sha256=_file(
            factory.partition_audit_path,
            label="query partition audit",
        ),
        partition_audit_sha256=factory.partition_audit_sha256,
        policy_intervention_root=Path(_CONTAINER_PATHS["policy_intervention_root"]),
        policy_intervention_tree_sha256=_tree(
            policy_root,
            label=f"{corpus_id} policy intervention",
        ),
        pseudonym_key_path=Path(_CONTAINER_PATHS["pseudonym_key_path"]),
        expected_pseudonym_key_sha256=materialization.pseudonym_key_sha256,
        query_package_root=Path(_CONTAINER_PATHS["query_package_root"]),
        query_package_tree_sha256=_tree(query_root, label=f"{corpus_id} query package"),
        staged_root=Path(_CONTAINER_PATHS["staged_root"]),
        staged_tree_sha256=_tree(admitted.staged_root, label="online staging tree"),
        expected_authorized_index_store_receipt_sha256=index.artifact_sha256,
        expected_policy_intervention_receipt_sha256=policy.artifact_sha256,
        policy_bundle_receipt_sha256=suite_row.policy_bundle_receipt_sha256,
        index_bundle_receipt_sha256=suite_row.index_bundle_receipt_sha256,
        policy_bundle_receipt_path=Path(_CONTAINER_PATHS["policy_bundle_receipt_path"]),
        index_bundle_receipt_path=Path(_CONTAINER_PATHS["index_bundle_receipt_path"]),
        query_receipt_sha256=suite_row.query_receipt_sha256,
        online_execution_plan_sha256=suite_row.online_execution_plan_sha256,
        online_execution_tree_sha256=suite_row.online_execution_tree_sha256,
        sharded_execution_plan_file_sha256=execution_file,
        trial_runtime_admission_receipt_file_sha256=runtime_file,
        feature_bindings=_feature_bindings(runtime),
    )


@dataclass(frozen=True)
class _CorpusSources:
    online_root: Path
    index_root: Path
    embedding_root: Path
    policy_root: Path
    query_root: Path
    execution_plan_path: Path
    trial_runtime_receipt_path: Path
    policy_bundle_receipt_path: Path
    index_bundle_receipt_path: Path


def _corpus_sources(
    admitted: _AdmittedFactory,
    corpus_id: str,
) -> _CorpusSources:
    artifact_root = admitted.config.artifact_root
    online_root = artifact_root / "custody" / "online" / corpus_id
    index_bundle_root = artifact_root / "authorized-index-stores" / corpus_id
    policy_bundle_root = artifact_root / "policy-workloads" / corpus_id
    runtime_root = artifact_root / "trial-runtime" / corpus_id
    return _CorpusSources(
        online_root=online_root,
        index_root=index_bundle_root / "sealed",
        embedding_root=artifact_root / "embedding-stores" / corpus_id,
        policy_root=policy_bundle_root / "sealed",
        query_root=runtime_root / RUNTIME_QUERY_DIRECTORY,
        execution_plan_path=online_root / ONLINE_EXECUTION_PLAN_FILENAME,
        trial_runtime_receipt_path=runtime_root / RUNTIME_RECEIPT_FILENAME,
        policy_bundle_receipt_path=policy_bundle_root / STAGE_BUNDLE_FILENAME,
        index_bundle_receipt_path=index_bundle_root / STAGE_BUNDLE_FILENAME,
    )


def _launcher_environment(config: ProductionControlMaterializationConfig) -> dict[str, str]:
    environment = dict(_FIXED_ENVIRONMENT)
    environment["HOSTNAME"] = config.hostname
    return environment


def _sealed_argv(config_path: Path) -> tuple[str, ...]:
    return (
        _PYTHON_PATH,
        "-m",
        "fractal_ann_diagnostics.cli",
        "run-sealed-corpus",
        "--config",
        str(config_path),
    )


def _attested_mounts(
    materialization: ProductionControlMaterializationConfig,
    admitted: _AdmittedFactory,
    corpus_id: str,
    spec: ProductionCorpusWorkloadSpec,
    *,
    closure_tree_sha256: str,
    launcher_identity_file_sha256: str,
    launcher_identity_path: Path | None = None,
) -> tuple[LauncherBindMount, ...]:
    sources = _corpus_sources(admitted, corpus_id)
    identity_path = (
        materialization.launcher_identity_path
        if launcher_identity_path is None
        else launcher_identity_path
    )
    rows = (
        LauncherBindMount(
            source=str(materialization.finalized_controls_root),
            target=str(materialization.finalized_controls_root),
            role="production-run-closure",
            kind="directory",
            content_sha256=closure_tree_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(sources.online_root),
            target=str(spec.artifact_root),
            role="sealed-online-artifact",
            kind="directory",
            content_sha256=spec.artifact_tree_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(sources.index_root),
            target=str(spec.authorized_index_store_root),
            role="authorized-index-store",
            kind="directory",
            content_sha256=spec.authorized_index_store_tree_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(sources.embedding_root),
            target=str(spec.embedding_store_root),
            role="embedding-store",
            kind="directory",
            content_sha256=spec.embedding_store_tree_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(admitted.config.partition_audit_path),
            target=str(spec.partition_audit_path),
            role="partition-audit",
            kind="file",
            content_sha256=spec.partition_audit_file_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(sources.policy_root),
            target=str(spec.policy_intervention_root),
            role="policy-intervention",
            kind="directory",
            content_sha256=spec.policy_intervention_tree_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(materialization.pseudonym_key_path),
            target=str(spec.pseudonym_key_path),
            role="pseudonym-key",
            kind="file",
            content_sha256=spec.expected_pseudonym_key_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(sources.query_root),
            target=str(spec.query_package_root),
            role="query-package",
            kind="directory",
            content_sha256=spec.query_package_tree_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(admitted.staged_root),
            target=str(spec.staged_root),
            role="staged-inputs",
            kind="directory",
            content_sha256=spec.staged_tree_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(sources.policy_bundle_receipt_path),
            target=str(spec.policy_bundle_receipt_path),
            role="policy-stage-bundle",
            kind="file",
            content_sha256=spec.policy_bundle_receipt_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(sources.index_bundle_receipt_path),
            target=str(spec.index_bundle_receipt_path),
            role="index-stage-bundle",
            kind="file",
            content_sha256=spec.index_bundle_receipt_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(materialization.opa_binary_path),
            target=_OPA_PATH,
            role="opa-runtime-binary",
            kind="file",
            content_sha256=admitted.extraction.opa_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(materialization.uv_lock_path),
            target=_UV_LOCK_PATH,
            role="uv-lock",
            kind="file",
            content_sha256=admitted.extraction.uv_lock_sha256,
            attested_artifact=True,
        ),
        LauncherBindMount(
            source=str(identity_path),
            target=_LAUNCHER_IDENTITY_PATH,
            role="launcher-identity",
            kind="file",
            content_sha256=launcher_identity_file_sha256,
            attested_artifact=True,
        ),
    )
    return tuple(sorted(rows, key=lambda item: item.target.encode("utf-8")))


def _provisional_runtime_plan(
    materialization: ProductionControlMaterializationConfig,
    admitted: _AdmittedFactory,
    corpus_id: str,
    spec: ProductionCorpusWorkloadSpec,
    mounts: tuple[LauncherBindMount, ...],
    *,
    code_commit: str = C0_COMMIT_SENTINEL,
    launcher_identity_path: Path | None = None,
    launcher_identity_file_sha256: str | None = None,
) -> RuntimeAttestationPlan:
    environment = _launcher_environment(materialization)
    config_path = (
        materialization.finalized_controls_root
        / corpus_id
        / "control"
        / PRODUCTION_CORPUS_CONFIG_FILENAME
    )
    argv = _sealed_argv(config_path)
    return RuntimeAttestationPlan(
        attestation_id=f"{corpus_id}.production",
        manifest_sha256=PREFLIGHT_DIGEST_SENTINEL,
        runner_identity=materialization.runner_identity,
        oci_image_digest=materialization.scientific_production_reference,
        code_commit=code_commit,
        operating_system_id=PREFLIGHT_TEXT_SENTINEL,
        operating_system_version_id=PREFLIGHT_TEXT_SENTINEL,
        kernel_release=PREFLIGHT_TEXT_SENTINEL,
        architecture=PREFLIGHT_TEXT_SENTINEL,
        cpu_model=PREFLIGHT_TEXT_SENTINEL,
        logical_cpu_count=PREFLIGHT_INTEGER_SENTINEL,
        memory_limit_bytes=PREFLIGHT_INTEGER_SENTINEL,
        mount_namespace_sha256=PREFLIGHT_DIGEST_SENTINEL,
        mounts=tuple(item.runtime_mount() for item in mounts),
        argv=argv,
        argv_sha256=argv_sha256(argv),
        environment_allowlist=tuple(sorted(environment)),
        environment_sha256=environment_sha256(environment),
        opa_binary=RuntimeFilePin(
            path=admitted.extraction.opa_image_path,
            sha256=admitted.extraction.opa_sha256,
        ),
        python_binary=RuntimeFilePin(
            path=admitted.extraction.python_binary_image_path,
            sha256=admitted.extraction.python_binary_sha256,
        ),
        python_version=PREFLIGHT_TEXT_SENTINEL,
        uv_lock=RuntimeFilePin(
            path=admitted.extraction.uv_lock_image_path,
            sha256=admitted.extraction.uv_lock_sha256,
        ),
        launcher_identity=RuntimeFilePin(
            path=_LAUNCHER_IDENTITY_PATH,
            sha256=(
                _file(
                    materialization.launcher_identity_path
                    if launcher_identity_path is None
                    else launcher_identity_path,
                    label="launcher identity",
                )
                if launcher_identity_file_sha256 is None
                else _sha256(
                    "launcher_identity_file_sha256",
                    launcher_identity_file_sha256,
                )
            ),
        ),
        workload_id=PRODUCTION_CORPUS_WORKLOAD_ID,
        workload_sha256=spec.file_sha256,
        invocation_marker_path=f"{_OUTPUT_ROOT}/{RUNTIME_INVOCATION_MARKER_FILENAME}",
    )


def _launcher_geometry(
    materialization: ProductionControlMaterializationConfig,
    admitted: _AdmittedFactory,
    corpus_id: str,
    *,
    control_tree_sha256: str,
    attested_mounts: tuple[LauncherBindMount, ...],
    code_commit: str = C0_COMMIT_SENTINEL,
    control_root: Path | None = None,
) -> LauncherGeometry:
    resolved_control_root = (
        materialization.blueprint_root / corpus_id / "launcher-control"
        if control_root is None
        else control_root
    )
    mounts = tuple(
        sorted(
            (
                *attested_mounts,
                LauncherBindMount(
                    source=str(resolved_control_root),
                    target=_CONTROL_ROOT,
                    role="runtime-control-tree",
                    kind="directory",
                    content_sha256=control_tree_sha256,
                    attested_artifact=False,
                ),
            ),
            key=lambda item: item.target.encode("utf-8"),
        )
    )
    environment = _launcher_environment(materialization)
    pre_c1_key = materialization.file_sha256[:20]
    volume_suffix = _sha256_bytes(f"{materialization.file_sha256}\0{corpus_id}".encode("utf-8"))[
        :20
    ]
    staging_namespace = materialization.suite_base_root / f".pre-c1-output-{pre_c1_key}"
    return LauncherGeometry(
        corpus_id=corpus_id,
        oci_image_digest=materialization.scientific_production_reference,
        code_commit=code_commit,
        platform=materialization.runner_platform,
        uid=_UID,
        gid=_GID,
        hostname=materialization.hostname,
        environment=tuple(
            LauncherEnvironmentVariable(name=name, value=value)
            for name, value in sorted(environment.items())
        ),
        memory_limit_bytes=materialization.memory_limit_bytes,
        cpuset_cpus=materialization.cpuset_cpus,
        bind_mounts=mounts,
        control_mount_target=_CONTROL_ROOT,
        runtime_plan_template_relative_path=PLAN_TEMPLATE_FILENAME,
        output_volume=f"fractal-{corpus_id}-{volume_suffix}",
        output_volume_subpath=f"pre-c1-{pre_c1_key}/{corpus_id}",
        output_root=_OUTPUT_ROOT,
        copy_output_root=str(staging_namespace / "online" / corpus_id),
        tmpfs_root=_TMPFS_ROOT,
        tmpfs_size_bytes=materialization.tmpfs_size_bytes,
        tmpfs_mode=0o1777,
        tmpfs_flags=("nodev", "noexec", "nosuid"),
    )


def _rederive_blueprint_launch_contract(
    materialization: ProductionControlMaterializationConfig,
    admitted: _AdmittedFactory,
    corpus_id: str,
    spec: ProductionCorpusWorkloadSpec,
    binding: WorkloadSpecBinding,
    blueprint: ProductionControlBlueprintReceipt,
) -> tuple[PreflightLaunchContract, RuntimeAttestationPlan]:
    """Reproduce every pre-C1 launch field from admitted authorities."""

    corpus_root = materialization.blueprint_root / corpus_id
    control_root = corpus_root / "launcher-control"
    identity_bytes = launcher_identity_file_bytes(
        oci_image_digest=materialization.scientific_production_reference,
        code_commit=C0_COMMIT_SENTINEL,
    )
    if (
        _read(materialization.launcher_identity_path, label="launcher identity") != identity_bytes
        or _sha256_bytes(identity_bytes) != blueprint.launcher_identity_file_sha256
    ):
        raise ProductionControlError("launcher identity differs from its C0/image derivation")
    mounts = _attested_mounts(
        materialization,
        admitted,
        corpus_id,
        spec,
        closure_tree_sha256=blueprint.provisional_closure_tree_sha256,
        launcher_identity_file_sha256=blueprint.launcher_identity_file_sha256,
    )
    expected_plan = _provisional_runtime_plan(
        materialization,
        admitted,
        corpus_id,
        spec,
        mounts,
    )
    plan_bytes = runtime_attestation_plan_template_file_bytes(expected_plan)
    plan_path = control_root / PLAN_TEMPLATE_FILENAME
    if _read(plan_path, label=f"{corpus_id} provisional runtime plan") != plan_bytes:
        raise ProductionControlError(
            f"{corpus_id} provisional runtime plan differs from its full derivation"
        )
    _scan_exact_tree(
        control_root,
        frozenset({PLAN_TEMPLATE_FILENAME}),
        label=f"{corpus_id} launcher control tree",
    )
    try:
        control_tree_sha256 = digest_directory_tree(control_root).sha256
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(
            f"cannot rederive {corpus_id} launcher control tree: {exc}"
        ) from exc
    expected_geometry = _launcher_geometry(
        materialization,
        admitted,
        corpus_id,
        control_tree_sha256=control_tree_sha256,
        attested_mounts=mounts,
    )
    expected_contract = PreflightLaunchContract(
        geometry=expected_geometry,
        argv=(
            _PYTHON_PATH,
            "-m",
            "fractal_ann_diagnostics.sealed_container_launcher",
            "capture-preflight",
        ),
        provisional_control_tree_sha256=control_tree_sha256,
        provisional_plan_template_file_sha256=_sha256_bytes(plan_bytes),
    )
    contract_path = corpus_root / PREFLIGHT_CONTRACT_FILENAME
    try:
        observed_contract = load_preflight_launch_contract(contract_path)
    except SealedContainerLauncherError as exc:
        raise ProductionControlError(f"{corpus_id} preflight contract is invalid: {exc}") from exc
    expected_binding = (
        spec.available_family_count,
        spec.selected_family_count,
        spec.file_sha256,
        control_tree_sha256,
        _sha256_bytes(plan_bytes),
        expected_plan.plan_sha256,
        expected_contract.contract_sha256,
        expected_contract.file_sha256,
    )
    observed_binding = (
        binding.available_family_count,
        binding.selected_family_count,
        binding.file_sha256,
        binding.launcher_control_tree_sha256,
        binding.plan_template_file_sha256,
        binding.plan_template_semantic_sha256,
        binding.preflight_contract_sha256,
        binding.preflight_contract_file_sha256,
    )
    if observed_contract != expected_contract or observed_binding != expected_binding:
        raise ProductionControlError(
            f"{corpus_id} launch geometry or blueprint binding differs from full derivation"
        )
    return expected_contract, expected_plan


def _rederive_instantiated_launch_contract(
    materialization: ProductionControlMaterializationConfig,
    admitted: _AdmittedFactory,
    corpus_id: str,
    spec: ProductionCorpusWorkloadSpec,
    binding: WorkloadSpecBinding,
    blueprint: ProductionControlBlueprintReceipt,
    instantiation: ProductionControlC0InstantiationReceipt,
) -> tuple[PreflightLaunchContract, RuntimeAttestationPlan]:
    """Reproduce every A-bound launch field from the raw authorities."""

    root = Path(instantiation.instantiated_root)
    corpus_root = root / corpus_id
    control_root = corpus_root / "launcher-control"
    identity_path = root / LAUNCHER_IDENTITY_FILENAME
    identity_bytes = launcher_identity_file_bytes(
        oci_image_digest=materialization.scientific_production_reference,
        code_commit=instantiation.apparatus_commit,
    )
    if (
        _read(identity_path, label="A-bound launcher identity") != identity_bytes
        or _sha256_bytes(identity_bytes) != instantiation.launcher_identity_file_sha256
    ):
        raise ProductionControlError("A-bound launcher identity differs from A and D")
    mounts = _attested_mounts(
        materialization,
        admitted,
        corpus_id,
        spec,
        closure_tree_sha256=blueprint.provisional_closure_tree_sha256,
        launcher_identity_file_sha256=instantiation.launcher_identity_file_sha256,
        launcher_identity_path=identity_path,
    )
    expected_plan = _provisional_runtime_plan(
        materialization,
        admitted,
        corpus_id,
        spec,
        mounts,
        code_commit=instantiation.apparatus_commit,
        launcher_identity_path=identity_path,
        launcher_identity_file_sha256=instantiation.launcher_identity_file_sha256,
    )
    plan_bytes = runtime_attestation_plan_template_file_bytes(expected_plan)
    plan_path = control_root / PLAN_TEMPLATE_FILENAME
    if _read(plan_path, label=f"{corpus_id} A-bound runtime plan") != plan_bytes:
        raise ProductionControlError(
            f"{corpus_id} A-bound runtime plan differs from its derivation"
        )
    _scan_exact_tree(
        control_root,
        frozenset({PLAN_TEMPLATE_FILENAME}),
        label=f"{corpus_id} A-bound launcher control tree",
    )
    try:
        control_tree_sha256 = digest_directory_tree(control_root).sha256
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(
            f"cannot rederive {corpus_id} A-bound launcher control tree: {exc}"
        ) from exc
    expected_geometry = _launcher_geometry(
        materialization,
        admitted,
        corpus_id,
        control_tree_sha256=control_tree_sha256,
        attested_mounts=mounts,
        code_commit=instantiation.apparatus_commit,
        control_root=control_root,
    )
    expected_contract = PreflightLaunchContract(
        geometry=expected_geometry,
        argv=(
            _PYTHON_PATH,
            "-m",
            "fractal_ann_diagnostics.sealed_container_launcher",
            "capture-preflight",
        ),
        provisional_control_tree_sha256=control_tree_sha256,
        provisional_plan_template_file_sha256=_sha256_bytes(plan_bytes),
    )
    try:
        observed_contract = load_preflight_launch_contract(
            corpus_root / PREFLIGHT_CONTRACT_FILENAME
        )
        verify_launcher_mounts(observed_contract.geometry)
    except SealedContainerLauncherError as exc:
        raise ProductionControlError(
            f"{corpus_id} A-bound preflight contract is invalid: {exc}"
        ) from exc
    expected_binding = (
        spec.available_family_count,
        spec.selected_family_count,
        spec.file_sha256,
        control_tree_sha256,
        _sha256_bytes(plan_bytes),
        expected_plan.plan_sha256,
        expected_contract.contract_sha256,
        expected_contract.file_sha256,
    )
    observed_binding = (
        binding.available_family_count,
        binding.selected_family_count,
        binding.file_sha256,
        binding.launcher_control_tree_sha256,
        binding.plan_template_file_sha256,
        binding.plan_template_semantic_sha256,
        binding.preflight_contract_sha256,
        binding.preflight_contract_file_sha256,
    )
    if observed_contract != expected_contract or observed_binding != expected_binding:
        raise ProductionControlError(
            f"{corpus_id} A-bound launch geometry or binding differs from derivation"
        )
    return expected_contract, expected_plan


def materialize_production_control_blueprint(
    materialization_config_path: str | Path,
    *,
    expected_config_sha256: str,
    resume: bool = False,
) -> ProductionControlBlueprintReceipt:
    """Write the label-payload-excluded five-corpus C1 blueprint and empty closure."""

    materialization = load_production_control_config(
        materialization_config_path,
        expected_sha256=expected_config_sha256,
    )
    admitted = _admit_factory(materialization)
    _ensure_private_directory(materialization.blueprint_root)
    _ensure_private_directory(materialization.finalized_controls_root)
    placeholder = digest_directory_tree(materialization.finalized_controls_root)
    if placeholder.entries:
        raise ProductionControlError("the pre-C1 production closure is not empty")

    identity_bytes = launcher_identity_file_bytes(
        oci_image_digest=materialization.scientific_production_reference,
        code_commit=C0_COMMIT_SENTINEL,
    )
    _publish_exact(
        materialization.launcher_identity_path,
        identity_bytes,
        resume=resume,
        label="launcher identity",
    )
    identity_file_sha256 = _sha256_bytes(identity_bytes)
    bindings: list[WorkloadSpecBinding] = []
    specs: list[ProductionCorpusWorkloadSpec] = []
    for corpus_id in FIXED_CORPORA:
        corpus_root = materialization.blueprint_root / corpus_id
        control_root = corpus_root / "launcher-control"
        _ensure_private_directory(corpus_root)
        _ensure_private_directory(control_root)
        spec = _derive_workload_spec(materialization, admitted, corpus_id)
        specs.append(spec)
        spec_path = corpus_root / PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME
        _publish_exact(
            spec_path,
            spec.canonical_file_bytes(),
            resume=resume,
            label=f"{corpus_id} workload specification",
        )
        mounts = _attested_mounts(
            materialization,
            admitted,
            corpus_id,
            spec,
            closure_tree_sha256=placeholder.sha256,
            launcher_identity_file_sha256=identity_file_sha256,
        )
        plan = _provisional_runtime_plan(
            materialization,
            admitted,
            corpus_id,
            spec,
            mounts,
        )
        plan_bytes = runtime_attestation_plan_template_file_bytes(plan)
        plan_path = control_root / PLAN_TEMPLATE_FILENAME
        _publish_exact(
            plan_path,
            plan_bytes,
            resume=resume,
            label=f"{corpus_id} provisional runtime plan",
        )
        control_tree = digest_directory_tree(control_root)
        if control_tree.entries != (PLAN_TEMPLATE_FILENAME,):
            raise ProductionControlError("launcher-control contains a non-plan member")
        geometry = _launcher_geometry(
            materialization,
            admitted,
            corpus_id,
            control_tree_sha256=control_tree.sha256,
            attested_mounts=mounts,
        )
        verify_launcher_mounts(geometry)
        contract = PreflightLaunchContract(
            geometry=geometry,
            argv=(
                _PYTHON_PATH,
                "-m",
                "fractal_ann_diagnostics.sealed_container_launcher",
                "capture-preflight",
            ),
            provisional_control_tree_sha256=control_tree.sha256,
            provisional_plan_template_file_sha256=_sha256_bytes(plan_bytes),
        )
        contract_path = corpus_root / PREFLIGHT_CONTRACT_FILENAME
        _publish_exact(
            contract_path,
            contract.canonical_file_bytes(),
            resume=resume,
            label=f"{corpus_id} preflight launch contract",
        )
        bindings.append(
            WorkloadSpecBinding(
                corpus_id=corpus_id,
                available_family_count=spec.available_family_count,
                selected_family_count=spec.selected_family_count,
                relative_path=f"{corpus_id}/{PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME}",
                file_sha256=spec.file_sha256,
                launcher_control_tree_sha256=control_tree.sha256,
                plan_template_file_sha256=_sha256_bytes(plan_bytes),
                plan_template_semantic_sha256=plan.plan_sha256,
                preflight_contract_sha256=contract.contract_sha256,
                preflight_contract_file_sha256=contract.file_sha256,
            )
        )

    workload_rows = _production_workloads_fragment(specs)
    _validate_candidate_workload_templates(
        specs,
        materialization=materialization,
        selected_family_count=admitted.config.selected_family_count,
    )
    workload_fragment_bytes = _production_workloads_fragment_file_bytes(workload_rows)
    _publish_exact(
        materialization.production_workloads_fragment_path,
        workload_fragment_bytes,
        resume=resume,
        label="production workloads manifest fragment",
    )
    hardware_fragment_bytes = _production_hardware_fragment_file_bytes(
        _production_hardware_fragment(materialization)
    )
    _publish_exact(
        materialization.production_hardware_fragment_path,
        hardware_fragment_bytes,
        resume=resume,
        label="production hardware manifest fragment",
    )

    payload_entries = _blueprint_payload_entries()
    receipt = ProductionControlBlueprintReceipt(
        materialization_config_sha256=materialization.file_sha256,
        factory_config_sha256=admitted.config.file_sha256,
        factory_suite_receipt_sha256=admitted.suite.receipt_sha256,
        factory_artifact_tree_sha256=materialization.factory_artifact_tree_sha256,
        c0_runtime_extraction_receipt_sha256=(materialization.c0_runtime_extraction_receipt_sha256),
        approval_environment=materialization.approval_environment,
        runner_image=materialization.scientific_production_reference,
        runner_platform=materialization.runner_platform,
        candidate_image_source_commit=materialization.candidate_image_source_commit,
        launcher_identity_file_sha256=identity_file_sha256,
        production_hardware_fragment_file_sha256=_sha256_bytes(hardware_fragment_bytes),
        production_workloads_fragment_file_sha256=_sha256_bytes(workload_fragment_bytes),
        provisional_closure_root=str(materialization.finalized_controls_root),
        provisional_closure_tree_sha256=placeholder.sha256,
        provisional_closure_entries=placeholder.entries,
        payload_tree_sha256=_payload_tree_sha256(
            materialization.blueprint_root,
            payload_entries,
        ),
        workloads=tuple(bindings),
    )
    _publish_exact(
        materialization.blueprint_receipt_path,
        receipt.canonical_file_bytes(),
        resume=resume,
        label="production control blueprint receipt",
    )
    _scan_exact_tree(
        materialization.blueprint_root,
        _blueprint_all_entries(),
        label="production control blueprint",
    )
    return receipt


def load_production_control_c0_instantiation_receipt(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ProductionControlC0InstantiationReceipt:
    encoded = _read(path, label="C0 control instantiation receipt")
    if expected_sha256 is not None and _sha256_bytes(encoded) != _sha256(
        "C0 control instantiation receipt SHA-256",
        expected_sha256,
    ):
        raise ProductionControlError("C0 control instantiation receipt differs from its pin")
    receipt = ProductionControlC0InstantiationReceipt.from_dict(
        _parse_object(encoded, label="C0 control instantiation receipt")
    )
    if encoded != receipt.canonical_file_bytes():
        raise ProductionControlError("C0 control instantiation receipt is not canonical")
    return receipt


def _cleanup_c0_instantiation_stage(stage: Path) -> None:
    entries = (*_c0_instantiated_payload_entries(), C0_INSTANTIATION_RECEIPT_FILENAME)
    for relative in sorted(
        entries,
        key=lambda item: (len(PurePosixPath(item).parts), item),
        reverse=True,
    ):
        target = stage / relative
        try:
            if target.is_dir() and not target.is_symlink():
                target.rmdir()
            else:
                target.unlink()
        except OSError:
            pass
    try:
        stage.rmdir()
    except OSError:
        pass


def instantiate_c0_production_controls(
    *,
    materialization_config_path: str | Path,
    candidate_manifest_package_path: str | Path,
    candidate_image_closure_path: str | Path,
    apparatus_commit: str,
    output_root: str | Path,
) -> ProductionControlC0InstantiationReceipt:
    """Resolve only A-dependent control fields and publish one executable tree."""

    if type(apparatus_commit) is not str or _GIT_COMMIT.fullmatch(apparatus_commit) is None:
        raise ProductionControlError("apparatus_commit must be one full Git commit")
    config_path = _absolute_path("materialization_config_path", materialization_config_path)
    candidate_package_path = _absolute_path(
        "candidate_manifest_package_path",
        candidate_manifest_package_path,
    )
    closure_path = _absolute_path(
        "candidate_image_closure_path",
        candidate_image_closure_path,
    )
    output = _absolute_path("c0_instantiated_controls_root", output_root)
    if not output.name or output.name in {".", ".."}:
        raise ProductionControlError("C0 instantiated controls must name one directory")

    try:
        candidate_package = load_closed_candidate_manifest_package(candidate_package_path)
    except CandidateManifestAssemblyError as exc:
        raise ProductionControlError("raw candidate manifest package is not closed") from exc
    candidate_bytes = candidate_package.manifest_bytes
    candidate = dict(candidate_package.manifest)
    try:
        validate_candidate_rehearsal_manifest(candidate, c0_commit=apparatus_commit)
    except Exception as exc:
        raise ProductionControlError("raw candidate manifest fails rehearsal admission") from exc
    sealed = candidate.get("sealed_execution")
    controls = sealed.get("production_controls") if isinstance(sealed, Mapping) else None
    if not isinstance(controls, Mapping):
        raise ProductionControlError("raw candidate lacks production-control bindings")

    config = load_production_control_config(
        config_path,
        expected_sha256=str(controls.get("materialization_config_file_sha256")),
    )
    blueprint = load_production_control_blueprint_receipt(
        config.blueprint_receipt_path,
        expected_sha256=str(controls.get("blueprint_receipt_file_sha256")),
    )
    if controls.get("blueprint_receipt_sha256") != blueprint.semantic_sha256:
        raise ProductionControlError("candidate blueprint semantic digest differs")
    _scan_exact_tree(
        config.blueprint_root,
        _blueprint_all_entries(),
        label="raw production control blueprint",
    )
    if _payload_tree_sha256(config.blueprint_root, _blueprint_payload_entries()) != (
        blueprint.payload_tree_sha256
    ):
        raise ProductionControlError("raw production control blueprint payload differs")
    admitted = _admit_factory(config)
    _verify_blueprint_authority_header(config, blueprint, admitted)
    _verify_blueprint_manifest_fragments(config, blueprint, candidate)

    try:
        image_closure = CandidateImageClosure.from_file(closure_path)
    except (ProviderRehearsalError, OSError) as exc:
        raise ProductionControlError("candidate image closure is invalid") from exc
    if (
        image_closure.github_sha != config.candidate_image_source_commit
        or blueprint.candidate_image_source_commit != image_closure.github_sha
        or image_closure.scientific_image_reference != config.scientific_candidate_reference
        or image_closure.scientific_image_index_digest != config.scientific_index_digest
        or config.scientific_production_reference.rsplit("@", 1)[1]
        != image_closure.scientific_image_index_digest
    ):
        raise ProductionControlError("candidate image P/T/D closure differs from controls")
    if any(
        _paths_overlap(output, protected)
        for protected in (
            config.blueprint_root,
            config.finalized_controls_root,
            config.suite_base_root,
            admitted.config.artifact_root,
        )
    ):
        raise ProductionControlError("C0 instantiated controls overlap an admitted immutable tree")

    parent_descriptor, parent_metadata = _open_private_publish_parent(
        output,
        label="C0 instantiated controls",
    )
    stage: Path | None = None
    published = False
    try:
        for _attempt in range(16):
            temporary_name = f".{output.name}.tmp-{secrets.token_hex(16)}"
            try:
                os.mkdir(temporary_name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            stage = output.parent / temporary_name
            break
        if stage is None:
            raise ProductionControlError("cannot allocate C0 control staging directory")

        placeholder = digest_directory_tree(config.finalized_controls_root)
        if placeholder.entries:
            raise ProductionControlError("pre-C1 production closure is not empty")
        identity_bytes = launcher_identity_file_bytes(
            oci_image_digest=config.scientific_production_reference,
            code_commit=apparatus_commit,
        )
        identity_sha256 = _sha256_bytes(identity_bytes)
        _publish_exact(
            stage / LAUNCHER_IDENTITY_FILENAME,
            identity_bytes,
            resume=False,
            label="instantiated launcher identity",
        )

        bindings: list[WorkloadSpecBinding] = []
        specs: list[ProductionCorpusWorkloadSpec] = []
        raw_by_corpus = {row.corpus_id: row for row in blueprint.workloads}
        for corpus_id in FIXED_CORPORA:
            corpus_root = stage / corpus_id
            control_stage = corpus_root / "launcher-control"
            control_destination = output / corpus_id / "launcher-control"
            _ensure_private_directory(corpus_root)
            _ensure_private_directory(control_stage)
            raw_binding = raw_by_corpus[corpus_id]
            raw_spec_bytes = _read(
                config.blueprint_root / raw_binding.relative_path,
                label=f"{corpus_id} raw candidate workload",
            )
            raw_spec = _candidate_blueprint_workload_spec(
                raw_spec_bytes,
                binding=raw_binding,
            )
            spec = _derive_workload_spec(
                config,
                admitted,
                corpus_id,
                code_commit=apparatus_commit,
            )
            if (
                _resolve_candidate_workload_spec(
                    raw_spec,
                    apparatus_commit=apparatus_commit,
                )
                != spec
            ):
                raise ProductionControlError(
                    f"{corpus_id} A-bound workload differs from the raw template"
                )
            specs.append(spec)
            _publish_exact(
                corpus_root / PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME,
                spec.canonical_file_bytes(),
                resume=False,
                label=f"{corpus_id} instantiated workload",
            )
            mounts = _attested_mounts(
                config,
                admitted,
                corpus_id,
                spec,
                closure_tree_sha256=placeholder.sha256,
                launcher_identity_file_sha256=identity_sha256,
                launcher_identity_path=output / LAUNCHER_IDENTITY_FILENAME,
            )
            plan = _provisional_runtime_plan(
                config,
                admitted,
                corpus_id,
                spec,
                mounts,
                code_commit=apparatus_commit,
                launcher_identity_file_sha256=identity_sha256,
            )
            plan_bytes = runtime_attestation_plan_template_file_bytes(plan)
            _publish_exact(
                control_stage / PLAN_TEMPLATE_FILENAME,
                plan_bytes,
                resume=False,
                label=f"{corpus_id} instantiated runtime plan",
            )
            control_tree = digest_directory_tree(control_stage)
            geometry = _launcher_geometry(
                config,
                admitted,
                corpus_id,
                control_tree_sha256=control_tree.sha256,
                attested_mounts=mounts,
                code_commit=apparatus_commit,
                control_root=control_destination,
            )
            contract = PreflightLaunchContract(
                geometry=geometry,
                argv=(
                    _PYTHON_PATH,
                    "-m",
                    "fractal_ann_diagnostics.sealed_container_launcher",
                    "capture-preflight",
                ),
                provisional_control_tree_sha256=control_tree.sha256,
                provisional_plan_template_file_sha256=_sha256_bytes(plan_bytes),
            )
            _publish_exact(
                corpus_root / PREFLIGHT_CONTRACT_FILENAME,
                contract.canonical_file_bytes(),
                resume=False,
                label=f"{corpus_id} instantiated preflight contract",
            )
            bindings.append(
                WorkloadSpecBinding(
                    corpus_id=corpus_id,
                    available_family_count=spec.available_family_count,
                    selected_family_count=spec.selected_family_count,
                    relative_path=(f"{corpus_id}/{PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME}"),
                    file_sha256=spec.file_sha256,
                    launcher_control_tree_sha256=control_tree.sha256,
                    plan_template_file_sha256=_sha256_bytes(plan_bytes),
                    plan_template_semantic_sha256=plan.plan_sha256,
                    preflight_contract_sha256=contract.contract_sha256,
                    preflight_contract_file_sha256=contract.file_sha256,
                )
            )
            _fsync_private_directory(
                control_stage,
                label=f"{corpus_id} instantiated launcher-control directory",
            )
            _fsync_private_directory(
                corpus_root,
                label=f"{corpus_id} instantiated corpus directory",
            )

        workload_rows = _production_workloads_fragment(specs)
        try:
            validate_production_workload_registrations(
                workload_rows,
                frozen=True,
                registered_selected_family_count=admitted.config.selected_family_count,
                sealed_execution={
                    "code_commit": apparatus_commit,
                    "runner_identity": config.runner_identity,
                    "runner_image": config.scientific_production_reference,
                },
            )
        except ProductionWorkloadRegistrationError as exc:
            raise ProductionControlError("instantiated workload fragment is invalid") from exc
        _publish_exact(
            stage / PRODUCTION_WORKLOADS_FRAGMENT_FILENAME,
            _production_workloads_fragment_file_bytes(workload_rows),
            resume=False,
            label="instantiated production workloads fragment",
        )
        _publish_exact(
            stage / PRODUCTION_HARDWARE_FRAGMENT_FILENAME,
            _production_hardware_fragment_file_bytes(_production_hardware_fragment(config)),
            resume=False,
            label="instantiated production hardware fragment",
        )
        candidate_snapshot = stage / C0_CANDIDATE_PACKAGE_DIRECTORY
        _ensure_private_directory(candidate_snapshot)
        _publish_exact(
            stage / C0_CANDIDATE_MANIFEST_RELATIVE_PATH,
            candidate_package.manifest_bytes,
            resume=False,
            label="C0 candidate manifest byte snapshot",
        )
        _publish_exact(
            stage / C0_CANDIDATE_ASSEMBLY_RECEIPT_RELATIVE_PATH,
            candidate_package.receipt_bytes,
            resume=False,
            label="C0 candidate assembly receipt byte snapshot",
        )
        _fsync_private_directory(
            candidate_snapshot,
            label="C0 candidate manifest package snapshot",
        )
        snapshot = load_closed_candidate_manifest_package(candidate_snapshot)
        if (
            snapshot.manifest_bytes != candidate_package.manifest_bytes
            or snapshot.receipt_bytes != candidate_package.receipt_bytes
        ):
            raise ProductionControlError("C0 candidate package snapshot bytes differ")
        payload_entries = _c0_instantiated_payload_entries()
        payload_tree_sha256 = _payload_tree_sha256(stage, payload_entries)
        receipt = ProductionControlC0InstantiationReceipt(
            apparatus_commit=apparatus_commit,
            candidate_image_source_commit=image_closure.github_sha,
            build_context_tree_sha256=image_closure.build_context_tree_sha256,
            candidate_image_closure_file_sha256=image_closure.file_sha256,
            candidate_bootstrap_closure_sha256=image_closure.bootstrap_closure_sha256,
            candidate_manifest_sha256=manifest_sha256(candidate),
            candidate_manifest_file_sha256=_sha256_bytes(candidate_bytes),
            candidate_manifest_relative_path=C0_CANDIDATE_MANIFEST_RELATIVE_PATH,
            candidate_manifest_assembly_receipt_file_sha256=(
                _sha256_bytes(candidate_package.receipt_bytes)
            ),
            candidate_manifest_assembly_receipt_relative_path=(
                C0_CANDIDATE_ASSEMBLY_RECEIPT_RELATIVE_PATH
            ),
            materialization_config_file_sha256=config.file_sha256,
            blueprint_receipt_sha256=blueprint.semantic_sha256,
            blueprint_receipt_file_sha256=blueprint.file_sha256,
            blueprint_payload_tree_sha256=blueprint.payload_tree_sha256,
            scientific_candidate_reference=config.scientific_candidate_reference,
            scientific_production_reference=config.scientific_production_reference,
            scientific_index_digest=config.scientific_index_digest,
            release_image_index_digest=image_closure.release_image_index_digest,
            approval_environment=config.approval_environment,
            runner_platform=config.runner_platform,
            launcher_identity_file_sha256=identity_sha256,
            instantiated_root=str(output),
            instantiated_payload_tree_sha256=payload_tree_sha256,
            instantiated_payload_entries=payload_entries,
            workloads=tuple(bindings),
        )
        _publish_exact(
            stage / C0_INSTANTIATION_RECEIPT_FILENAME,
            receipt.canonical_file_bytes(),
            resume=False,
            label="C0 control instantiation receipt",
        )
        _scan_exact_tree(
            stage,
            frozenset((*payload_entries, C0_INSTANTIATION_RECEIPT_FILENAME)),
            label="C0 instantiated control staging tree",
        )
        if (
            load_closed_candidate_manifest_package(candidate_package_path) != candidate_package
            or CandidateImageClosure.from_file(closure_path) != image_closure
            or load_production_control_config(
                config_path,
                expected_sha256=config.file_sha256,
            )
            != config
            or load_production_control_blueprint_receipt(
                config.blueprint_receipt_path,
                expected_sha256=blueprint.file_sha256,
            )
            != blueprint
            or _payload_tree_sha256(config.blueprint_root, _blueprint_payload_entries())
            != blueprint.payload_tree_sha256
        ):
            raise ProductionControlError("C0 instantiation authority changed before publication")
        _fsync_private_directory(stage, label="C0 instantiated control staging root")
        _rename_noreplace_at(
            parent_descriptor,
            stage.name,
            output.name,
            label="C0 instantiated controls",
        )
        published = True
        os.fsync(parent_descriptor)
        named_parent = output.parent.lstat()
        published_metadata = os.stat(
            output.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(published_metadata.st_mode)
            or stat.S_IMODE(published_metadata.st_mode) != 0o700
            or (published_metadata.st_dev, published_metadata.st_ino)
            == (parent_metadata.st_dev, parent_metadata.st_ino)
            or (parent_metadata.st_dev, parent_metadata.st_ino)
            != (named_parent.st_dev, named_parent.st_ino)
        ):
            raise ProductionControlError("published C0 instantiated control identity differs")
        readback = load_production_control_c0_instantiation_receipt(
            output / C0_INSTANTIATION_RECEIPT_FILENAME,
            expected_sha256=receipt.file_sha256,
        )
        snapshot_readback = load_closed_candidate_manifest_package(
            output / C0_CANDIDATE_PACKAGE_DIRECTORY
        )
        _scan_exact_tree(
            output,
            frozenset((*payload_entries, C0_INSTANTIATION_RECEIPT_FILENAME)),
            label="published C0 instantiated control tree",
        )
        for corpus_id in FIXED_CORPORA:
            contract = load_preflight_launch_contract(
                output / corpus_id / PREFLIGHT_CONTRACT_FILENAME
            )
            try:
                verify_launcher_mounts(contract.geometry)
            except SealedContainerLauncherError as exc:
                raise ProductionControlError(
                    f"published {corpus_id} C0 launch mounts differ"
                ) from exc
        if (
            readback != receipt
            or snapshot_readback.manifest_bytes != candidate_package.manifest_bytes
            or snapshot_readback.receipt_bytes != candidate_package.receipt_bytes
            or _payload_tree_sha256(output, payload_entries)
            != receipt.instantiated_payload_tree_sha256
        ):
            raise ProductionControlError("published C0 instantiated controls differ")
        return readback
    except ProductionControlError:
        raise
    except (ArtifactIntegrityError, OSError, ProviderRehearsalError) as exc:
        raise ProductionControlError(f"cannot instantiate C0 production controls: {exc}") from exc
    finally:
        if stage is not None and not published:
            _cleanup_c0_instantiation_stage(stage)
        os.close(parent_descriptor)


def _manifest_artifact(
    manifest: Mapping[str, Any],
    role: str,
    *,
    corpus_id: str | None = None,
) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not list:
        raise ProductionControlError("frozen manifest artifact table is malformed")
    rows = tuple(
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("role") == role
        and (corpus_id is None or item.get("corpus_id") == corpus_id)
    )
    if len(rows) != 1:
        qualifier = "" if corpus_id is None else f" for {corpus_id}"
        raise ProductionControlError(
            f"frozen manifest must contain one {role!r} artifact{qualifier}"
        )
    row = rows[0]
    if corpus_id is None and "corpus_id" in row:
        raise ProductionControlError(f"suite artifact {role!r} is corpus-bound")
    return row


def _verify_c1_production_control_bindings(
    materialization: ProductionControlMaterializationConfig,
    blueprint: ProductionControlBlueprintReceipt,
    manifest: Mapping[str, Any],
) -> None:
    sealed_execution = manifest.get("sealed_execution")
    controls = (
        sealed_execution.get("production_controls")
        if isinstance(sealed_execution, Mapping)
        else None
    )
    if (
        type(controls) is not dict
        or frozenset(controls) != _MANIFEST_PRODUCTION_CONTROL_FIELDS
        or any(type(value) is not str for value in controls.values())
    ):
        raise ProductionControlError(
            "public C1 production-control bindings differ from the closed schema"
        )
    expected = {
        "blueprint_receipt_file_sha256": blueprint.file_sha256,
        "blueprint_receipt_sha256": blueprint.semantic_sha256,
        "materialization_config_file_sha256": materialization.file_sha256,
    }
    if controls != expected:
        raise ProductionControlError(
            "materialization config or blueprint receipt differs from the public C1 pins"
        )


def _verify_blueprint_authority_header(
    materialization: ProductionControlMaterializationConfig,
    blueprint: ProductionControlBlueprintReceipt,
    admitted: _AdmittedFactory,
) -> None:
    observed = {
        "approval_environment": blueprint.approval_environment,
        "c0_runtime_extraction_receipt_sha256": (blueprint.c0_runtime_extraction_receipt_sha256),
        "candidate_image_source_commit": blueprint.candidate_image_source_commit,
        "factory_artifact_tree_sha256": blueprint.factory_artifact_tree_sha256,
        "factory_config_sha256": blueprint.factory_config_sha256,
        "factory_suite_receipt_sha256": blueprint.factory_suite_receipt_sha256,
        "materialization_config_sha256": blueprint.materialization_config_sha256,
        "provisional_closure_root": blueprint.provisional_closure_root,
        "runner_image": blueprint.runner_image,
        "runner_platform": blueprint.runner_platform,
    }
    expected = {
        "approval_environment": materialization.approval_environment,
        "c0_runtime_extraction_receipt_sha256": (
            materialization.c0_runtime_extraction_receipt_sha256
        ),
        "candidate_image_source_commit": admitted.extraction.c0_sha,
        "factory_artifact_tree_sha256": materialization.factory_artifact_tree_sha256,
        "factory_config_sha256": admitted.config.file_sha256,
        "factory_suite_receipt_sha256": admitted.suite.receipt_sha256,
        "materialization_config_sha256": materialization.file_sha256,
        "provisional_closure_root": str(materialization.finalized_controls_root),
        "runner_image": materialization.scientific_production_reference,
        "runner_platform": materialization.runner_platform,
    }
    if observed != expected:
        raise ProductionControlError(
            "blueprint authority header differs from the config and admitted C0 factory"
        )


def _verify_blueprint_manifest_fragments(
    materialization: ProductionControlMaterializationConfig,
    blueprint: ProductionControlBlueprintReceipt,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    _verify_c1_production_control_bindings(materialization, blueprint, manifest)
    workload_bytes = _read_pinned(
        materialization.production_workloads_fragment_path,
        blueprint.production_workloads_fragment_file_sha256,
        label="production workloads manifest fragment",
    )
    workload_rows = _parse_array(
        workload_bytes,
        label="production workloads manifest fragment",
    )
    if workload_bytes != _canonical_bytes(workload_rows) + b"\n":
        raise ProductionControlError(
            "production workloads manifest fragment bytes are not canonical"
        )
    candidate_rows = _load_candidate_workload_rows(
        workload_rows,
        bindings=blueprint.workloads,
    )
    public_workloads = manifest.get("production_workloads")
    sealed_execution = manifest.get("sealed_execution")
    apparatus_commit = (
        sealed_execution.get("code_commit") if isinstance(sealed_execution, Mapping) else None
    )
    if apparatus_commit == C0_COMMIT_SENTINEL:
        if (
            type(public_workloads) is not list
            or _canonical_bytes(public_workloads) + b"\n" != workload_bytes
        ):
            raise ProductionControlError(
                "candidate production workloads differ from the pre-C0 fragment"
            )
        validated = tuple(row for row, _ in candidate_rows)
    else:
        if not isinstance(apparatus_commit, str):
            raise ProductionControlError("public C1 apparatus commit is absent")
        expected_workloads = _resolve_candidate_workload_rows(
            candidate_rows,
            apparatus_commit=apparatus_commit,
        )
        if type(public_workloads) is not list or _canonical_bytes(
            public_workloads
        ) != _canonical_bytes(expected_workloads):
            raise ProductionControlError(
                "public C1 production workloads differ from the resolved pre-C0 templates"
            )
        try:
            validated = validate_production_workload_registrations(
                public_workloads,
                frozen=True,
                registered_selected_family_count=manifest["analysis"]["power"][
                    "selected_families_per_corpus"
                ],
                sealed_execution=manifest["sealed_execution"],
            )
        except (KeyError, TypeError, ProductionWorkloadRegistrationError) as exc:
            raise ProductionControlError(
                "public C1 production workload fragment is invalid"
            ) from exc
        if tuple(row["corpus_id"] for row in validated) != FIXED_CORPORA:
            raise ProductionControlError("public C1 production workloads are not in fixed order")

    hardware_bytes = _read_pinned(
        materialization.production_hardware_fragment_path,
        blueprint.production_hardware_fragment_file_sha256,
        label="production hardware manifest fragment",
    )
    hardware = _parse_object(
        hardware_bytes,
        label="production hardware manifest fragment",
    )
    if hardware_bytes != _canonical_bytes(hardware) + b"\n":
        raise ProductionControlError(
            "production hardware manifest fragment bytes are not canonical"
        )
    if hardware_bytes != _production_hardware_fragment_file_bytes(
        _production_hardware_fragment(materialization)
    ):
        raise ProductionControlError(
            "production hardware fragment differs from the pinned pre-C1 claims"
        )
    public_hardware = (
        sealed_execution.get("hardware") if isinstance(sealed_execution, Mapping) else None
    )
    if (
        type(public_hardware) is not dict
        or _canonical_bytes(public_hardware) + b"\n" != hardware_bytes
    ):
        raise ProductionControlError("public C1 hardware differs from the pre-C1 fragment")
    return validated


def _c0_apparatus_evidence(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    sealed = manifest.get("sealed_execution")
    release = sealed.get("c0_evidence_release") if isinstance(sealed, Mapping) else None
    apparatus = release.get("apparatus_evidence") if isinstance(release, Mapping) else None
    if not isinstance(apparatus, Mapping):
        raise ProductionControlError("C1 omits the closed C0 apparatus evidence")
    return apparatus


def _admit_c0_control_instantiation(
    *,
    receipt_path: Path,
    materialization: ProductionControlMaterializationConfig,
    blueprint: ProductionControlBlueprintReceipt,
    manifest: Mapping[str, Any],
) -> ProductionControlC0InstantiationReceipt:
    """Bind the post-A executable tree to C0 evidence and the frozen C1 bytes."""

    apparatus = _c0_apparatus_evidence(manifest)
    expected_sha256 = apparatus.get("production_control_instantiation_receipt_file_sha256")
    if type(expected_sha256) is not str:
        raise ProductionControlError("C0 apparatus omits the control instantiation pin")
    receipt = load_production_control_c0_instantiation_receipt(
        receipt_path,
        expected_sha256=expected_sha256,
    )
    root = Path(receipt.instantiated_root)
    if receipt_path != root / C0_INSTANTIATION_RECEIPT_FILENAME:
        raise ProductionControlError(
            "control instantiation receipt path differs from its fixed output member"
        )
    _scan_exact_tree(
        root,
        frozenset((*receipt.instantiated_payload_entries, C0_INSTANTIATION_RECEIPT_FILENAME)),
        label="C0 instantiated production controls",
    )
    if (
        _payload_tree_sha256(root, receipt.instantiated_payload_entries)
        != receipt.instantiated_payload_tree_sha256
    ):
        raise ProductionControlError("C0 instantiated production control payload differs")

    sealed = manifest.get("sealed_execution")
    apparatus_commit = sealed.get("code_commit") if isinstance(sealed, Mapping) else None
    expected = {
        "apparatus_commit": apparatus.get("c0_commit"),
        "build_context_tree_sha256": apparatus.get("build_context_tree_sha256"),
        "candidate_bootstrap_closure_sha256": apparatus.get("candidate_bootstrap_closure_sha256"),
        "candidate_image_closure_file_sha256": apparatus.get("candidate_image_closure_sha256"),
        "candidate_image_source_commit": apparatus.get("candidate_image_source_commit"),
        "candidate_manifest_assembly_receipt_file_sha256": apparatus.get(
            "candidate_manifest_assembly_receipt_file_sha256"
        ),
        "candidate_manifest_assembly_receipt_relative_path": (
            C0_CANDIDATE_ASSEMBLY_RECEIPT_RELATIVE_PATH
        ),
        "candidate_manifest_file_sha256": apparatus.get("candidate_manifest_file_sha256"),
        "candidate_manifest_relative_path": C0_CANDIDATE_MANIFEST_RELATIVE_PATH,
        "candidate_manifest_sha256": apparatus.get("rehearsal_manifest_sha256"),
        "release_image_index_digest": apparatus.get("release_image_index_digest"),
        "scientific_index_digest": apparatus.get("scientific_image_index_digest"),
    }
    observed = {name: getattr(receipt, name) for name in expected}
    archive_members = {
        "candidate_manifest_archive_member_path": (
            f"production-control-instantiation/{receipt.candidate_manifest_relative_path}"
        ),
        "candidate_manifest_assembly_receipt_archive_member_path": (
            "production-control-instantiation/"
            f"{receipt.candidate_manifest_assembly_receipt_relative_path}"
        ),
    }
    fixed_archive_members = {
        "candidate_manifest_archive_member_path": C0_CANDIDATE_MANIFEST_ARCHIVE_MEMBER_PATH,
        "candidate_manifest_assembly_receipt_archive_member_path": (
            C0_CANDIDATE_ASSEMBLY_RECEIPT_ARCHIVE_MEMBER_PATH
        ),
    }
    if (
        apparatus_commit != receipt.apparatus_commit
        or observed != expected
        or {name: apparatus.get(name) for name in archive_members} != archive_members
        or archive_members != fixed_archive_members
    ):
        raise ProductionControlError("C0 control instantiation differs from the apparatus evidence")
    if (
        receipt.materialization_config_file_sha256 != materialization.file_sha256
        or receipt.blueprint_receipt_sha256 != blueprint.semantic_sha256
        or receipt.blueprint_receipt_file_sha256 != blueprint.file_sha256
        or receipt.blueprint_payload_tree_sha256 != blueprint.payload_tree_sha256
        or receipt.candidate_image_source_commit != materialization.candidate_image_source_commit
        or receipt.candidate_image_source_commit != blueprint.candidate_image_source_commit
        or receipt.scientific_candidate_reference != materialization.scientific_candidate_reference
        or receipt.scientific_production_reference
        != materialization.scientific_production_reference
        or receipt.approval_environment != materialization.approval_environment
        or receipt.approval_environment != blueprint.approval_environment
        or receipt.runner_platform != materialization.runner_platform
    ):
        raise ProductionControlError(
            "C0 control instantiation differs from its raw config or blueprint"
        )
    identity_bytes = launcher_identity_file_bytes(
        oci_image_digest=materialization.scientific_production_reference,
        code_commit=receipt.apparatus_commit,
    )
    if (
        _read(root / LAUNCHER_IDENTITY_FILENAME, label="C0 launcher identity") != identity_bytes
        or _sha256_bytes(identity_bytes) != receipt.launcher_identity_file_sha256
    ):
        raise ProductionControlError("C0 launcher identity differs from A and D")
    if (
        _read(
            root / PRODUCTION_WORKLOADS_FRAGMENT_FILENAME,
            label="C0 instantiated workload fragment",
        )
        != _canonical_bytes(manifest.get("production_workloads")) + b"\n"
    ):
        raise ProductionControlError("C0 instantiated workloads differ from C1")
    public_hardware = sealed.get("hardware") if isinstance(sealed, Mapping) else None
    if (
        _read(
            root / PRODUCTION_HARDWARE_FRAGMENT_FILENAME,
            label="C0 instantiated hardware fragment",
        )
        != _canonical_bytes(public_hardware) + b"\n"
    ):
        raise ProductionControlError("C0 instantiated hardware differs from C1")

    for binding in receipt.workloads:
        corpus_root = root / binding.corpus_id
        if binding.relative_path != (
            f"{binding.corpus_id}/{PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME}"
        ):
            raise ProductionControlError(f"{binding.corpus_id} instantiated workload path differs")
        spec_bytes = _read(
            root / binding.relative_path,
            label=f"{binding.corpus_id} instantiated workload",
        )
        try:
            spec = ProductionCorpusWorkloadSpec.from_dict(
                _parse_object(
                    spec_bytes,
                    label=f"{binding.corpus_id} instantiated workload",
                )
            )
        except (ProductionCorpusRunError, TrialRuntimeError) as exc:
            raise ProductionControlError(
                f"{binding.corpus_id} instantiated workload is invalid"
            ) from exc
        disclosed = _manifest_workload_spec(manifest, binding.corpus_id)
        plan = load_runtime_attestation_plan_template(
            corpus_root / "launcher-control" / PLAN_TEMPLATE_FILENAME
        )
        contract = load_preflight_launch_contract(corpus_root / PREFLIGHT_CONTRACT_FILENAME)
        try:
            control_tree_sha256 = digest_directory_tree(corpus_root / "launcher-control").sha256
        except ArtifactIntegrityError as exc:
            raise ProductionControlError(
                f"cannot inspect {binding.corpus_id} instantiated launcher controls"
            ) from exc
        if (
            spec_bytes != spec.canonical_file_bytes()
            or spec != disclosed
            or spec.code_commit != receipt.apparatus_commit
            or spec.file_sha256 != binding.file_sha256
            or plan.code_commit != receipt.apparatus_commit
            or plan.oci_image_digest != receipt.scientific_production_reference
            or plan.workload_sha256 != spec.file_sha256
            or plan.plan_sha256 != binding.plan_template_semantic_sha256
            or _file(
                corpus_root / "launcher-control" / PLAN_TEMPLATE_FILENAME,
                label=f"{binding.corpus_id} instantiated runtime plan",
            )
            != binding.plan_template_file_sha256
            or control_tree_sha256 != binding.launcher_control_tree_sha256
            or contract.geometry.code_commit != receipt.apparatus_commit
            or contract.geometry.oci_image_digest != receipt.scientific_production_reference
            or contract.geometry.platform != receipt.runner_platform
            or contract.contract_sha256 != binding.preflight_contract_sha256
            or contract.file_sha256 != binding.preflight_contract_file_sha256
        ):
            raise ProductionControlError(
                f"{binding.corpus_id} instantiated controls differ from A, D, or C1"
            )
    return receipt


def _manifest_workload_spec(
    manifest: Mapping[str, Any],
    corpus_id: str,
) -> ProductionCorpusWorkloadSpec:
    rows = manifest.get("production_workloads")
    try:
        validate_production_workload_registrations(
            rows,
            frozen=True,
            registered_selected_family_count=manifest["analysis"]["power"][
                "selected_families_per_corpus"
            ],
            sealed_execution=manifest["sealed_execution"],
        )
    except (KeyError, TypeError, ProductionWorkloadRegistrationError) as exc:
        raise ProductionControlError("public C1 production workloads are invalid") from exc
    if type(rows) is not list or len(rows) != len(FIXED_CORPORA):
        raise ProductionControlError(
            "frozen manifest omits the five disclosed production workloads"
        )
    if tuple(item.get("corpus_id") for item in rows if isinstance(item, Mapping)) != FIXED_CORPORA:
        raise ProductionControlError("manifest production workloads are not in fixed order")
    try:
        row = next(item for item in rows if item["corpus_id"] == corpus_id)
    except (KeyError, StopIteration, TypeError) as exc:
        raise ProductionControlError(
            f"manifest production workload is absent for {corpus_id}"
        ) from exc
    if set(row) != {"canonical_file_sha256", "corpus_id", "spec"}:
        raise ProductionControlError("manifest production workload wrapper is not closed")
    try:
        spec = ProductionCorpusWorkloadSpec.from_dict(row["spec"])
        wrapper_sha256 = production_workload_file_sha256(row["spec"])
        wrapper_bytes = canonical_workload_file_bytes(row["spec"])
    except (
        KeyError,
        TypeError,
        ProductionCorpusRunError,
        ProductionWorkloadRegistrationError,
        TrialRuntimeError,
    ) as exc:
        raise ProductionControlError(
            f"manifest production workload is invalid for {corpus_id}"
        ) from exc
    if (
        spec.corpus_id != corpus_id
        or row["canonical_file_sha256"] != wrapper_sha256
        or wrapper_bytes != spec.canonical_file_bytes()
    ):
        raise ProductionControlError("manifest production workload digest differs")
    return spec


def _candidate_blueprint_workload_spec(
    encoded: bytes,
    *,
    binding: WorkloadSpecBinding,
) -> ProductionCorpusWorkloadSpec:
    row = _parse_object(encoded, label=f"{binding.corpus_id} candidate workload")
    try:
        spec = ProductionCorpusWorkloadSpec.from_dict(row)
    except (ProductionCorpusRunError, TrialRuntimeError) as exc:
        raise ProductionControlError(
            f"candidate workload is invalid for {binding.corpus_id}"
        ) from exc
    if (
        encoded != spec.canonical_file_bytes()
        or spec.corpus_id != binding.corpus_id
        or spec.code_commit != C0_COMMIT_SENTINEL
        or spec.file_sha256 != binding.file_sha256
    ):
        raise ProductionControlError(f"candidate workload bytes differ for {binding.corpus_id}")
    return spec


def _resolve_candidate_workload_spec(
    candidate: ProductionCorpusWorkloadSpec,
    *,
    apparatus_commit: str,
) -> ProductionCorpusWorkloadSpec:
    if candidate.code_commit != C0_COMMIT_SENTINEL:
        raise ProductionControlError("candidate workload lacks its commit sentinel")
    if _GIT_COMMIT.fullmatch(apparatus_commit) is None:
        raise ProductionControlError("apparatus commit must be one full Git commit")
    return replace(candidate, code_commit=apparatus_commit)


def _verify_manifest_runtime_bindings(
    manifest: Mapping[str, Any],
    admitted: _AdmittedFactory,
    corpus_id: str,
    spec: ProductionCorpusWorkloadSpec,
    *,
    final_plan_template_file_sha256: str,
) -> None:
    sources = _corpus_sources(admitted, corpus_id)
    online = _manifest_artifact(manifest, "online-execution", corpus_id=corpus_id)
    embedding = _manifest_artifact(manifest, "embedding-store", corpus_id=corpus_id)
    staged = _manifest_artifact(manifest, "online-staging-package")
    audit = _manifest_artifact(manifest, "query-partition-audit")
    policy = _manifest_artifact(manifest, "policy-workload", corpus_id=corpus_id)
    index = _manifest_artifact(manifest, "authorized-index-store", corpus_id=corpus_id)
    runtime = _manifest_artifact(manifest, "trial-runtime-package", corpus_id=corpus_id)
    plan = _manifest_artifact(
        manifest,
        "runtime-attestation-plan-template",
        corpus_id=corpus_id,
    )
    if (
        online.get("sha256") != spec.artifact_tree_sha256
        or online.get("revision") != f"sha256:{spec.online_execution_plan_sha256}"
        or embedding.get("sha256") != spec.embedding_store_tree_sha256
        or staged.get("sha256") != spec.staged_tree_sha256
        or audit.get("sha256") != spec.partition_audit_file_sha256
        or plan.get("sha256") != final_plan_template_file_sha256
    ):
        raise ProductionControlError(f"{corpus_id} direct manifest/runtime commitments differ")
    outer_bindings = (
        (
            policy,
            sources.policy_root.parent,
            sources.policy_bundle_receipt_path,
            spec.policy_bundle_receipt_sha256,
            "policy workload",
        ),
        (
            index,
            sources.index_root.parent,
            sources.index_bundle_receipt_path,
            spec.index_bundle_receipt_sha256,
            "authorized index store",
        ),
        (
            runtime,
            sources.query_root.parent,
            sources.trial_runtime_receipt_path,
            spec.trial_runtime_admission_receipt_file_sha256,
            "trial runtime package",
        ),
    )
    for artifact, outer_root, receipt_path, receipt_sha256, label in outer_bindings:
        if (
            artifact.get("sha256") != _tree(outer_root, label=f"{corpus_id} {label}")
            or _file(receipt_path, label=f"{corpus_id} {label} receipt") != receipt_sha256
            or not receipt_path.is_relative_to(outer_root)
        ):
            raise ProductionControlError(
                f"{corpus_id} {label} outer-tree or descendant receipt differs"
            )


def _verify_c0_manifest_runtime(
    manifest: Mapping[str, Any],
    admitted: _AdmittedFactory,
) -> None:
    extraction = admitted.extraction
    opa = _manifest_artifact(manifest, "opa-runtime-binary")
    hnsw = _manifest_artifact(manifest, "strict-authorized-hnsw")
    if opa.get("sha256") != extraction.opa_sha256:
        raise ProductionControlError("manifest OPA binary differs from C0 extraction")
    if (
        hnsw.get("sha256") != extraction.hnswlib_wheel_sha256
        or hnsw.get("revision") != f"sha256:{extraction.hnswlib_receipt_sha256}"
    ):
        raise ProductionControlError(
            "manifest HNSW wheel or runtime provenance differs from C0 extraction"
        )


def _load_preflight_receipt(path: Path) -> RuntimePreflightReceipt:
    encoded = _read(path, label="runtime preflight receipt")
    try:
        return loads_runtime_preflight_receipt(encoded)
    except Exception as exc:
        raise ProductionControlError("runtime preflight receipt is invalid") from exc


def _verified_hardware_observation(
    materialization: ProductionControlMaterializationConfig,
    preflight: RuntimePreflightReceipt,
    final_plan: RuntimeAttestationPlan,
) -> tuple[str, int, int, str, str]:
    observable_fields = (
        "architecture",
        "cpu_model",
        "logical_cpu_count",
        "memory_limit_bytes",
        "operating_system_id",
        "operating_system_version_id",
    )
    if any(getattr(final_plan, field) != getattr(preflight, field) for field in observable_fields):
        raise ProductionControlError(
            "final runtime plan hardware differs from its preflight observation"
        )
    if preflight.architecture not in {"aarch64", "arm64"}:
        raise ProductionControlError("production preflight architecture does not resolve to arm64")
    operating_system = f"{preflight.operating_system_id}-{preflight.operating_system_version_id}"
    if (
        preflight.cpu_model != materialization.hardware_cpu_model
        or preflight.logical_cpu_count != len(materialization.cpuset_cpus)
        or preflight.memory_limit_bytes != materialization.memory_limit_bytes
        or operating_system != materialization.hardware_operating_system
    ):
        raise ProductionControlError(
            "production preflight hardware differs from the pinned pre-C1 claims"
        )
    return (
        preflight.cpu_model,
        preflight.logical_cpu_count,
        preflight.memory_limit_bytes,
        operating_system,
        "arm64",
    )


@dataclass(frozen=True)
class _FinalizationCorpus:
    corpus_id: str
    spec: ProductionCorpusWorkloadSpec
    sharded_execution_plan_bytes: bytes
    trial_runtime_receipt_bytes: bytes
    instantiated_binding: WorkloadSpecBinding
    preflight: PreflightLaunchContract
    preflight_receipt: RuntimePreflightReceipt
    transition: RuntimePlanTransitionReceipt
    final_plan: RuntimeAttestationPlan
    required_artifacts: RequiredArtifactIdBindings


@dataclass(frozen=True)
class _FinalizationContext:
    request: ProductionControlFinalizationRequest
    materialization: ProductionControlMaterializationConfig
    blueprint: ProductionControlBlueprintReceipt
    instantiation: ProductionControlC0InstantiationReceipt
    admitted: _AdmittedFactory
    manifest: Mapping[str, Any]
    manifest_sha256: str
    c0_commit: str
    c1_commit: str
    sealed_run: SealedRunReceipt
    sealed_run_bytes: bytes
    online_admission: OnlineCustodyAdmissionReceipt
    online_admission_bytes: bytes
    verification_receipt: ArtifactVerificationReceipt
    corpora: tuple[_FinalizationCorpus, ...]


@dataclass(frozen=True)
class _RequiredArtifactBindingSuiteContext:
    materialization: ProductionControlMaterializationConfig
    blueprint_receipt_sha256: str
    instantiation: ProductionControlC0InstantiationReceipt
    factory_artifact_root: Path
    manifest: Mapping[str, Any]
    verification_receipt: ArtifactVerificationReceipt
    bindings: tuple[RequiredArtifactIdBindings, ...]


def _load_finalization_context(
    request: ProductionControlFinalizationRequest,
) -> _FinalizationContext:
    materialization = load_production_control_config(
        request.materialization_config_path,
        expected_sha256=request.materialization_config_sha256,
    )
    if request.blueprint_receipt_path != materialization.blueprint_receipt_path:
        raise ProductionControlError("finalization request names another blueprint receipt")
    blueprint = load_production_control_blueprint_receipt(
        request.blueprint_receipt_path,
        expected_sha256=request.blueprint_receipt_sha256,
    )
    if (
        blueprint.materialization_config_sha256 != materialization.file_sha256
        or blueprint.provisional_closure_root != str(materialization.finalized_controls_root)
    ):
        raise ProductionControlError("blueprint differs from the materialization config")
    _scan_exact_tree(
        materialization.blueprint_root,
        _blueprint_all_entries(),
        label="transitioned production control blueprint",
    )
    if (
        _payload_tree_sha256(
            materialization.blueprint_root,
            _blueprint_payload_entries(),
        )
        != blueprint.payload_tree_sha256
    ):
        raise ProductionControlError("production control blueprint payload differs")
    registration = verify_production_protocol_registration(
        request.c1_package_root,
        registration_record_path=request.protocol_registry_record_path,
        registration_receipt_path=request.protocol_registration_receipt_path,
    )
    registration.assert_current()
    if registration.package_root != request.c1_package_root:
        raise ProductionControlError("verified registration package root differs")
    manifest = load_study_manifest(request.frozen_manifest_path)
    validate_study_manifest(manifest, require_frozen=True)
    _verify_c1_production_control_bindings(materialization, blueprint, manifest)
    admitted = _admit_factory(materialization)
    _verify_blueprint_authority_header(materialization, blueprint, admitted)
    registered_workloads = _verify_blueprint_manifest_fragments(
        materialization,
        blueprint,
        manifest,
    )
    digest = manifest_sha256(manifest)
    sealed = manifest["sealed_execution"]
    apparatus_commit = sealed["code_commit"]
    if digest != registration.manifest_sha256 or registration.c0_commit != apparatus_commit:
        raise ProductionControlError("C1 registration differs from the frozen apparatus")
    instantiation = _admit_c0_control_instantiation(
        receipt_path=request.c0_control_instantiation_receipt_path,
        materialization=materialization,
        blueprint=blueprint,
        manifest=manifest,
    )
    package_manifest = request.c1_package_root / "study-manifest.json"
    if _read(package_manifest, label="public C1 manifest") != _read(
        request.frozen_manifest_path,
        label="local frozen manifest",
    ):
        raise ProductionControlError("local manifest differs from the public C1 bytes")
    lock_bytes = _read(request.manifest_lock_path, label="frozen manifest lock")
    if lock_bytes != f"{digest}\n".encode("ascii"):
        raise ProductionControlError("frozen manifest lock differs")
    if (
        sealed["runner_identity"] != materialization.runner_identity
        or sealed["runner_image"] != materialization.scientific_production_reference
    ):
        raise ProductionControlError("sealed execution identity differs from the blueprint")
    _verify_c0_manifest_runtime(manifest, admitted)
    if (
        admitted.config.selected_family_count
        != manifest["analysis"]["power"]["selected_families_per_corpus"]
    ):
        raise ProductionControlError(
            "factory family count differs from the registered power design"
        )

    expected_sealed_path = materialization.finalized_controls_root / f"{digest}.json"
    if request.sealed_run_receipt_path != expected_sealed_path:
        raise ProductionControlError("sealed run receipt is not the manifest-named closure member")
    sealed_run = load_sealed_run_receipt(request.sealed_run_receipt_path)
    sealed_run_bytes = _read(request.sealed_run_receipt_path, label="sealed run receipt")
    if (
        sealed_run.manifest_sha256 != digest
        or sealed_run.runner_identity != materialization.runner_identity
        or sealed_run.code_commit != apparatus_commit
        or sealed_run.runner_image != materialization.scientific_production_reference
    ):
        raise ProductionControlError("sealed run receipt differs from C1 or C0")
    online_admission = admit_online_custody(
        request.frozen_manifest_path,
        custody_seal_receipt_path=request.custody_seal_receipt_path,
        sealed_run_receipt_path=request.sealed_run_receipt_path,
        artifact_verification_receipt_path=request.artifact_verification_receipt_path,
        artifact_root=request.artifact_root,
        local_artifact_map_path=request.local_artifact_map_path,
        runner_identity=materialization.runner_identity,
    )
    persisted_admission = load_online_custody_admission_receipt(
        request.online_custody_admission_path
    )
    online_admission_bytes = _read(
        request.online_custody_admission_path,
        label="online custody admission",
    )
    if (
        persisted_admission != online_admission
        or online_admission_bytes != online_admission.canonical_bytes() + b"\n"
    ):
        raise ProductionControlError("persisted online custody admission differs")
    verification = load_verification_receipt(request.artifact_verification_receipt_path)
    if verification.manifest_sha256 != digest:
        raise ProductionControlError("artifact verification belongs to another manifest")
    try:
        expected_required_bindings = tuple(
            derive_required_artifact_id_bindings(
                manifest,
                verification,
                corpus_id=corpus_id,
            )
            for corpus_id in FIXED_CORPORA
        )
    except SealedOrchestratorError as exc:
        raise ProductionControlError(
            f"cannot derive the five required-artifact bindings: {exc}"
        ) from exc
    supplied_required_bindings = _load_closed_required_artifact_binding_suite(
        request.required_artifact_bindings_root,
        expected=expected_required_bindings,
    )
    required_by_corpus = dict(zip(FIXED_CORPORA, supplied_required_bindings, strict=True))

    corpora: list[_FinalizationCorpus] = []
    hardware_observations: list[tuple[str, int, int, str, str]] = []
    blueprint_by_corpus = {item.corpus_id: item for item in blueprint.workloads}
    instantiated_by_corpus = {item.corpus_id: item for item in instantiation.workloads}
    registered_by_corpus = {item["corpus_id"]: item for item in registered_workloads}
    for corpus_id in FIXED_CORPORA:
        raw_binding = blueprint_by_corpus[corpus_id]
        binding = instantiated_by_corpus[corpus_id]
        public_wrapper_sha256 = registered_by_corpus[corpus_id]["canonical_file_sha256"]
        spec_path = materialization.blueprint_root / raw_binding.relative_path
        spec_bytes = _read(spec_path, label=f"{corpus_id} blueprint workload")
        candidate_spec = _candidate_blueprint_workload_spec(
            spec_bytes,
            binding=raw_binding,
        )
        instantiated_spec_bytes = _read(
            Path(instantiation.instantiated_root) / binding.relative_path,
            label=f"{corpus_id} A-bound workload",
        )
        rederived = _derive_workload_spec(
            materialization,
            admitted,
            corpus_id,
            code_commit=apparatus_commit,
        )
        disclosed = _manifest_workload_spec(manifest, corpus_id)
        if (
            _resolve_candidate_workload_spec(
                candidate_spec,
                apparatus_commit=apparatus_commit,
            )
            != rederived
            or instantiated_spec_bytes != disclosed.canonical_file_bytes()
            or disclosed != rederived
            or public_wrapper_sha256 != rederived.file_sha256
        ):
            raise ProductionControlError(
                f"{corpus_id} raw template, A-bound control, factory, and C1 workload differ"
            )
        preflight, _provisional_plan = _rederive_instantiated_launch_contract(
            materialization,
            admitted,
            corpus_id,
            rederived,
            binding,
            blueprint,
            instantiation,
        )
        evidence_root = request.runtime_evidence_root / corpus_id
        preflight_receipt = _load_preflight_receipt(
            evidence_root / RUNTIME_PREFLIGHT_RECEIPT_FILENAME
        )
        transition_path = evidence_root / RUNTIME_PLAN_TRANSITION_RECEIPT_FILENAME
        transition = load_runtime_plan_transition(transition_path)
        final_plan = verify_runtime_plan_transition(
            preflight,
            preflight_receipt,
            transition,
        )
        if (
            transition.provisional_control_tree_sha256 != binding.launcher_control_tree_sha256
            or transition.provisional_plan_template_file_sha256 != binding.plan_template_file_sha256
            or transition.provisional_plan_template_semantic_sha256
            != binding.plan_template_semantic_sha256
            or final_plan.workload_sha256 != rederived.file_sha256
            or final_plan.workload_sha256 != public_wrapper_sha256
            or final_plan.python_binary
            != RuntimeFilePin(
                path=admitted.extraction.python_binary_image_path,
                sha256=admitted.extraction.python_binary_sha256,
            )
            or final_plan.uv_lock
            != RuntimeFilePin(
                path=admitted.extraction.uv_lock_image_path,
                sha256=admitted.extraction.uv_lock_sha256,
            )
        ):
            raise ProductionControlError(
                f"{corpus_id} transitioned plan differs from C0/C1 blueprint"
            )
        hardware_observations.append(
            _verified_hardware_observation(
                materialization,
                preflight_receipt,
                final_plan,
            )
        )
        _verify_manifest_runtime_bindings(
            manifest,
            admitted,
            corpus_id,
            rederived,
            final_plan_template_file_sha256=(transition.final_plan_template_file_sha256),
        )
        sources = _corpus_sources(admitted, corpus_id)
        sharded_execution_plan_bytes = _read_pinned(
            sources.execution_plan_path,
            rederived.sharded_execution_plan_file_sha256,
            label=f"{corpus_id} sharded execution plan",
        )
        trial_runtime_receipt_bytes = _read_pinned(
            sources.trial_runtime_receipt_path,
            rederived.trial_runtime_admission_receipt_file_sha256,
            label=f"{corpus_id} trial runtime receipt",
        )
        try:
            frozen_execution = loads_sharded_online_execution_plan(sharded_execution_plan_bytes)
            frozen_runtime = TrialRuntimeAdmissionReceipt.from_dict(
                _parse_object(
                    trial_runtime_receipt_bytes,
                    label=f"{corpus_id} trial runtime receipt",
                )
            )
        except Exception as exc:
            raise ProductionControlError(
                f"{corpus_id} frozen execution/runtime snapshot is invalid"
            ) from exc
        if (
            _sha256_bytes(sharded_execution_plan_bytes)
            != rederived.sharded_execution_plan_file_sha256
            or _sha256_bytes(trial_runtime_receipt_bytes)
            != rederived.trial_runtime_admission_receipt_file_sha256
            or trial_runtime_receipt_bytes != frozen_runtime.canonical_file_bytes()
            or frozen_execution.artifact_sha256 != rederived.online_execution_plan_sha256
            or frozen_runtime.execution_artifact_sha256 != frozen_execution.artifact_sha256
            or _feature_bindings(frozen_runtime) != rederived.feature_bindings
        ):
            raise ProductionControlError(
                f"{corpus_id} frozen execution/runtime snapshot differs from its typed bindings"
            )
        required = required_by_corpus[corpus_id]
        corpora.append(
            _FinalizationCorpus(
                corpus_id=corpus_id,
                spec=rederived,
                sharded_execution_plan_bytes=sharded_execution_plan_bytes,
                trial_runtime_receipt_bytes=trial_runtime_receipt_bytes,
                instantiated_binding=binding,
                preflight=preflight,
                preflight_receipt=preflight_receipt,
                transition=transition,
                final_plan=final_plan,
                required_artifacts=required,
            )
        )
    if len(hardware_observations) != len(FIXED_CORPORA) or len(set(hardware_observations)) != 1:
        raise ProductionControlError(
            "five production preflights do not agree on one hardware observation"
        )
    cpu_model, logical_cores, memory_limit_bytes, operating_system, _architecture = (
        hardware_observations[0]
    )
    registered_hardware = manifest["sealed_execution"]["hardware"]
    if (
        registered_hardware["cpu_model"] != cpu_model
        or registered_hardware["logical_cores"] != logical_cores
        or registered_hardware["memory_gib"] != memory_limit_bytes // _GIB
        or registered_hardware["operating_system"] != operating_system
    ):
        raise ProductionControlError(
            "public C1 hardware differs from the five preflight observations"
        )
    return _FinalizationContext(
        request=request,
        materialization=materialization,
        blueprint=blueprint,
        instantiation=instantiation,
        admitted=admitted,
        manifest=manifest,
        manifest_sha256=digest,
        c0_commit=registration.c0_commit,
        c1_commit=registration.c1_commit,
        sealed_run=sealed_run,
        sealed_run_bytes=sealed_run_bytes,
        online_admission=online_admission,
        online_admission_bytes=online_admission_bytes,
        verification_receipt=verification,
        corpora=tuple(corpora),
    )


def _required_artifact_binding_suite_entries() -> frozenset[str]:
    return frozenset(
        entry
        for corpus_id in FIXED_CORPORA
        for entry in (
            corpus_id,
            f"{corpus_id}/{REQUIRED_ARTIFACT_BINDINGS_FILENAME}",
        )
    )


def _derive_required_artifact_binding_suite_context(
    *,
    materialization_config_path: Path,
    c0_control_instantiation_receipt_path: Path,
    frozen_manifest_path: Path,
    artifact_verification_receipt_path: Path,
    artifact_root: Path,
    local_artifact_map_path: Path,
) -> _RequiredArtifactBindingSuiteContext:
    config_bytes = _read(
        materialization_config_path,
        label="production control config",
    )
    materialization = load_production_control_config(
        materialization_config_path,
        expected_sha256=_sha256_bytes(config_bytes),
    )
    blueprint = load_production_control_blueprint_receipt(materialization.blueprint_receipt_path)
    if (
        blueprint.materialization_config_sha256 != materialization.file_sha256
        or blueprint.provisional_closure_root != str(materialization.finalized_controls_root)
    ):
        raise ProductionControlError("blueprint differs from the materialization config")
    _scan_exact_tree(
        materialization.blueprint_root,
        _blueprint_all_entries(),
        label="production control blueprint",
    )
    if (
        _payload_tree_sha256(
            materialization.blueprint_root,
            _blueprint_payload_entries(),
        )
        != blueprint.payload_tree_sha256
    ):
        raise ProductionControlError("production control blueprint payload differs")

    manifest = load_study_manifest(frozen_manifest_path)
    validate_study_manifest(manifest, require_frozen=True)
    _verify_c1_production_control_bindings(materialization, blueprint, manifest)
    admitted = _admit_factory(materialization)
    _verify_blueprint_authority_header(materialization, blueprint, admitted)
    registered_workloads = _verify_blueprint_manifest_fragments(
        materialization,
        blueprint,
        manifest,
    )
    sealed_execution = manifest["sealed_execution"]
    instantiation = _admit_c0_control_instantiation(
        receipt_path=c0_control_instantiation_receipt_path,
        materialization=materialization,
        blueprint=blueprint,
        manifest=manifest,
    )
    if (
        sealed_execution["runner_identity"] != materialization.runner_identity
        or sealed_execution["runner_image"] != materialization.scientific_production_reference
    ):
        raise ProductionControlError("sealed execution identity differs from the blueprint")
    _verify_c0_manifest_runtime(manifest, admitted)
    if (
        admitted.config.selected_family_count
        != manifest["analysis"]["power"]["selected_families_per_corpus"]
    ):
        raise ProductionControlError(
            "factory family count differs from the registered power design"
        )
    blueprint_by_corpus = {item.corpus_id: item for item in blueprint.workloads}
    instantiated_by_corpus = {item.corpus_id: item for item in instantiation.workloads}
    registered_by_corpus = {item["corpus_id"]: item for item in registered_workloads}
    for corpus_id in FIXED_CORPORA:
        raw_binding = blueprint_by_corpus[corpus_id]
        binding = instantiated_by_corpus[corpus_id]
        spec_path = materialization.blueprint_root / raw_binding.relative_path
        spec_bytes = _read(spec_path, label=f"{corpus_id} blueprint workload")
        candidate_spec = _candidate_blueprint_workload_spec(
            spec_bytes,
            binding=raw_binding,
        )
        instantiated_spec_bytes = _read(
            Path(instantiation.instantiated_root) / binding.relative_path,
            label=f"{corpus_id} A-bound workload",
        )
        rederived = _derive_workload_spec(
            materialization,
            admitted,
            corpus_id,
            code_commit=sealed_execution["code_commit"],
        )
        disclosed = _manifest_workload_spec(manifest, corpus_id)
        if (
            _resolve_candidate_workload_spec(
                candidate_spec,
                apparatus_commit=sealed_execution["code_commit"],
            )
            != rederived
            or instantiated_spec_bytes != disclosed.canonical_file_bytes()
            or disclosed != rederived
            or registered_by_corpus[corpus_id]["canonical_file_sha256"] != rederived.file_sha256
        ):
            raise ProductionControlError(
                f"{corpus_id} raw template, A-bound control, factory, and C1 workload differ"
            )

    try:
        verification = load_verification_receipt(artifact_verification_receipt_path)
        pins = {str(artifact["id"]): str(artifact["sha256"]) for artifact in manifest["artifacts"]}
        local_specs = load_local_artifact_map(
            local_artifact_map_path,
            expected_sha256_by_id=pins,
        )
        fresh_verification = verify_local_artifacts(
            artifact_root,
            manifest_sha256=manifest_sha256(manifest),
            artifacts=local_specs,
        )
    except (ArtifactIntegrityError, KeyError, TypeError) as exc:
        raise ProductionControlError(
            f"cannot revalidate the required-artifact authority: {exc}"
        ) from exc
    if fresh_verification.canonical_bytes() != verification.canonical_bytes():
        raise ProductionControlError(
            "fresh local artifact verification differs from the admitted receipt"
        )
    try:
        bindings = tuple(
            derive_required_artifact_id_bindings(
                manifest,
                verification,
                corpus_id=corpus_id,
            )
            for corpus_id in FIXED_CORPORA
        )
    except SealedOrchestratorError as exc:
        raise ProductionControlError(
            f"cannot derive the five required-artifact bindings: {exc}"
        ) from exc
    return _RequiredArtifactBindingSuiteContext(
        materialization=materialization,
        blueprint_receipt_sha256=blueprint.receipt_sha256,
        instantiation=instantiation,
        factory_artifact_root=admitted.config.artifact_root,
        manifest=manifest,
        verification_receipt=verification,
        bindings=bindings,
    )


def _load_closed_required_artifact_binding_suite(
    root: Path,
    *,
    expected: tuple[RequiredArtifactIdBindings, ...],
) -> tuple[RequiredArtifactIdBindings, ...]:
    suite_root = _absolute_path("required_artifact_bindings_root", root)
    if len(expected) != len(FIXED_CORPORA):
        raise ProductionControlError("required-artifact binding expectation is incomplete")
    try:
        before = digest_directory_tree(suite_root)
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(
            f"cannot inspect required-artifact binding suite: {exc}"
        ) from exc
    expected_entries = _required_artifact_binding_suite_entries()
    if frozenset(before.entries) != expected_entries:
        observed = frozenset(before.entries)
        raise ProductionControlError(
            "required-artifact binding suite membership differs; "
            f"missing={sorted(expected_entries - observed)}, "
            f"extra={sorted(observed - expected_entries)}"
        )
    try:
        loaded = tuple(
            load_required_artifact_id_bindings(
                suite_root / corpus_id / REQUIRED_ARTIFACT_BINDINGS_FILENAME
            )
            for corpus_id in FIXED_CORPORA
        )
        after = digest_directory_tree(suite_root)
    except (ArtifactIntegrityError, SealedOrchestratorError) as exc:
        raise ProductionControlError(f"cannot load required-artifact binding suite: {exc}") from exc
    if loaded != expected or after != before:
        raise ProductionControlError(
            "required-artifact binding suite differs from its fresh derivation"
        )
    return loaded


def _cleanup_required_artifact_binding_stage(stage: Path) -> None:
    for corpus_id in reversed(FIXED_CORPORA):
        corpus_root = stage / corpus_id
        try:
            (corpus_root / REQUIRED_ARTIFACT_BINDINGS_FILENAME).unlink()
        except OSError:
            pass
        try:
            corpus_root.rmdir()
        except OSError:
            pass
    try:
        stage.rmdir()
    except OSError:
        pass


def _atomic_publish_required_artifact_binding_suite(
    output_root: Path,
    bindings: tuple[RequiredArtifactIdBindings, ...],
) -> None:
    output = _absolute_path("required_artifact_bindings_output_root", output_root)
    if not output.name or output.name in {".", ".."}:
        raise ProductionControlError("required-artifact binding output must name one directory")
    parent_descriptor, parent_metadata = _open_private_publish_parent(
        output,
        label="required-artifact binding suite",
    )
    stage: Path | None = None
    published = False
    try:
        for _attempt in range(16):
            temporary_name = f".{output.name}.tmp-{secrets.token_hex(16)}"
            try:
                os.mkdir(temporary_name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            stage = output.parent / temporary_name
            break
        if stage is None:
            raise ProductionControlError(
                "cannot allocate required-artifact binding suite staging directory"
            )
        for corpus_id, binding in zip(FIXED_CORPORA, bindings, strict=True):
            corpus_root = stage / corpus_id
            corpus_root.mkdir(mode=0o700)
            _atomic_publish_file_noreplace(
                corpus_root / REQUIRED_ARTIFACT_BINDINGS_FILENAME,
                binding.canonical_file_bytes(),
                label=f"{corpus_id} required-artifact bindings",
            )
            _fsync_private_directory(
                corpus_root,
                label=f"{corpus_id} required-artifact staging directory",
            )
        _fsync_private_directory(
            stage,
            label="required-artifact suite staging directory",
        )
        _load_closed_required_artifact_binding_suite(stage, expected=bindings)
        _rename_noreplace_at(
            parent_descriptor,
            stage.name,
            output.name,
            label="required-artifact binding suite",
        )
        published = True
        os.fsync(parent_descriptor)
        named_parent_metadata = output.parent.lstat()
        output_metadata = os.stat(
            output.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(output_metadata.st_mode)
            or stat.S_IMODE(output_metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and output_metadata.st_uid != os.geteuid())
            or (parent_metadata.st_dev, parent_metadata.st_ino)
            != (named_parent_metadata.st_dev, named_parent_metadata.st_ino)
        ):
            raise ProductionControlError(
                "published required-artifact binding suite has an unsafe identity"
            )
        _load_closed_required_artifact_binding_suite(output, expected=bindings)
    except ProductionControlError:
        raise
    except OSError as exc:
        raise ProductionControlError(
            f"cannot publish required-artifact binding suite: {exc}"
        ) from exc
    finally:
        if stage is not None and not published:
            _cleanup_required_artifact_binding_stage(stage)
        os.close(parent_descriptor)


def write_required_artifact_binding_suite(
    *,
    materialization_config_path: str | Path,
    c0_control_instantiation_receipt_path: str | Path,
    frozen_manifest_path: str | Path,
    artifact_verification_receipt_path: str | Path,
    artifact_root: str | Path,
    local_artifact_map_path: str | Path,
    output_root: str | Path,
) -> tuple[RequiredArtifactIdBindings, ...]:
    """Derive and atomically publish all five required-artifact bindings."""

    materialization_config = _absolute_path(
        "materialization_config_path", materialization_config_path
    )
    instantiation_receipt = _absolute_path(
        "c0_control_instantiation_receipt_path",
        c0_control_instantiation_receipt_path,
    )
    frozen_manifest = _absolute_path("frozen_manifest_path", frozen_manifest_path)
    verification_receipt = _absolute_path(
        "artifact_verification_receipt_path", artifact_verification_receipt_path
    )
    artifacts = _absolute_path("artifact_root", artifact_root)
    artifact_map = _absolute_path("local_artifact_map_path", local_artifact_map_path)
    output = _absolute_path("required_artifact_bindings_output_root", output_root)
    context = _derive_required_artifact_binding_suite_context(
        materialization_config_path=materialization_config,
        c0_control_instantiation_receipt_path=instantiation_receipt,
        frozen_manifest_path=frozen_manifest,
        artifact_verification_receipt_path=verification_receipt,
        artifact_root=artifacts,
        local_artifact_map_path=artifact_map,
    )
    protected_roots = (
        context.materialization.blueprint_root,
        Path(context.instantiation.instantiated_root),
        context.materialization.finalized_controls_root,
        context.factory_artifact_root,
        artifacts,
    )
    if any(_paths_overlap(output, root) for root in protected_roots):
        raise ProductionControlError(
            "required-artifact binding output must remain outside every admitted immutable tree"
        )
    _atomic_publish_required_artifact_binding_suite(output, context.bindings)
    refreshed = _derive_required_artifact_binding_suite_context(
        materialization_config_path=materialization_config,
        c0_control_instantiation_receipt_path=instantiation_receipt,
        frozen_manifest_path=frozen_manifest,
        artifact_verification_receipt_path=verification_receipt,
        artifact_root=artifacts,
        local_artifact_map_path=artifact_map,
    )
    if refreshed != context:
        raise ProductionControlError(
            "required-artifact binding authorities changed during publication"
        )
    return _load_closed_required_artifact_binding_suite(
        output,
        expected=refreshed.bindings,
    )


def write_production_control_finalization_request(
    *,
    materialization_config_path: str | Path,
    c0_control_instantiation_receipt_path: str | Path,
    frozen_manifest_path: str | Path,
    manifest_lock_path: str | Path,
    c1_package_root: str | Path,
    protocol_registry_record_path: str | Path,
    protocol_registration_receipt_path: str | Path,
    online_custody_admission_path: str | Path,
    custody_seal_receipt_path: str | Path,
    artifact_verification_receipt_path: str | Path,
    artifact_root: str | Path,
    local_artifact_map_path: str | Path,
    required_artifact_bindings_root: str | Path,
    runtime_evidence_root: str | Path,
    output: str | Path,
) -> ProductionControlFinalizationRequest:
    """Derive, validate, and exclusively publish the post-C1 request."""

    paths = {
        "materialization_config_path": _absolute_path(
            "materialization_config_path", materialization_config_path
        ),
        "c0_control_instantiation_receipt_path": _absolute_path(
            "c0_control_instantiation_receipt_path",
            c0_control_instantiation_receipt_path,
        ),
        "frozen_manifest_path": _absolute_path("frozen_manifest_path", frozen_manifest_path),
        "manifest_lock_path": _absolute_path("manifest_lock_path", manifest_lock_path),
        "c1_package_root": _absolute_path("c1_package_root", c1_package_root),
        "protocol_registry_record_path": _absolute_path(
            "protocol_registry_record_path", protocol_registry_record_path
        ),
        "protocol_registration_receipt_path": _absolute_path(
            "protocol_registration_receipt_path", protocol_registration_receipt_path
        ),
        "online_custody_admission_path": _absolute_path(
            "online_custody_admission_path", online_custody_admission_path
        ),
        "custody_seal_receipt_path": _absolute_path(
            "custody_seal_receipt_path", custody_seal_receipt_path
        ),
        "artifact_verification_receipt_path": _absolute_path(
            "artifact_verification_receipt_path", artifact_verification_receipt_path
        ),
        "artifact_root": _absolute_path("artifact_root", artifact_root),
        "local_artifact_map_path": _absolute_path(
            "local_artifact_map_path", local_artifact_map_path
        ),
        "required_artifact_bindings_root": _absolute_path(
            "required_artifact_bindings_root", required_artifact_bindings_root
        ),
        "runtime_evidence_root": _absolute_path("runtime_evidence_root", runtime_evidence_root),
    }
    output_path = _absolute_path("finalization_request_output", output)
    config_bytes = _read(
        paths["materialization_config_path"],
        label="production control config",
    )
    config_sha256 = _sha256_bytes(config_bytes)
    materialization = load_production_control_config(
        paths["materialization_config_path"],
        expected_sha256=config_sha256,
    )
    blueprint = load_production_control_blueprint_receipt(materialization.blueprint_receipt_path)
    manifest = load_study_manifest(paths["frozen_manifest_path"])
    validate_study_manifest(manifest, require_frozen=True)
    _verify_c1_production_control_bindings(materialization, blueprint, manifest)
    manifest_digest = manifest_sha256(manifest)
    request = ProductionControlFinalizationRequest(
        materialization_config_path=paths["materialization_config_path"],
        materialization_config_sha256=config_sha256,
        blueprint_receipt_path=materialization.blueprint_receipt_path,
        blueprint_receipt_sha256=blueprint.receipt_sha256,
        c0_control_instantiation_receipt_path=paths["c0_control_instantiation_receipt_path"],
        frozen_manifest_path=paths["frozen_manifest_path"],
        manifest_lock_path=paths["manifest_lock_path"],
        c1_package_root=paths["c1_package_root"],
        protocol_registry_record_path=paths["protocol_registry_record_path"],
        protocol_registration_receipt_path=paths["protocol_registration_receipt_path"],
        sealed_run_receipt_path=(
            materialization.finalized_controls_root / f"{manifest_digest}.json"
        ),
        online_custody_admission_path=paths["online_custody_admission_path"],
        custody_seal_receipt_path=paths["custody_seal_receipt_path"],
        artifact_verification_receipt_path=paths["artifact_verification_receipt_path"],
        artifact_root=paths["artifact_root"],
        local_artifact_map_path=paths["local_artifact_map_path"],
        required_artifact_bindings_root=paths["required_artifact_bindings_root"],
        runtime_evidence_root=paths["runtime_evidence_root"],
    )

    context = _load_finalization_context(request)
    protected_roots = (
        context.materialization.blueprint_root,
        Path(context.instantiation.instantiated_root),
        context.materialization.finalized_controls_root,
        request.c1_package_root,
        request.artifact_root,
        request.required_artifact_bindings_root,
        request.runtime_evidence_root,
    )
    if any(_paths_overlap(output_path, root) for root in protected_roots):
        raise ProductionControlError(
            "finalization request output must remain outside every admitted immutable tree"
        )
    _atomic_publish_file_noreplace(
        output_path,
        request.canonical_file_bytes(),
        label="production control finalization request",
    )
    persisted = load_production_control_finalization_request(
        output_path,
        expected_sha256=request.file_sha256,
    )
    if persisted != request:
        raise ProductionControlError("published finalization request differs from its derivation")
    _load_finalization_context(persisted)
    return persisted


def _production_config(
    context: _FinalizationContext,
    corpus: _FinalizationCorpus,
) -> ProductionCorpusRunConfig:
    closure = context.materialization.finalized_controls_root
    control_root = closure / corpus.corpus_id / "control"
    return ProductionCorpusRunConfig(
        control_root=control_root,
        output_root=Path(_OUTPUT_ROOT),
        sealed_run_receipt_path=closure / f"{context.manifest_sha256}.json",
        runtime_attestation_plan_path=Path(_CONTROL_ROOT) / "runtime-attestation-plan.json",
        workload_spec_file_sha256=corpus.spec.file_sha256,
        online_custody_admission_file_sha256=_sha256_bytes(context.online_admission_bytes),
        required_artifact_bindings_file_sha256=corpus.required_artifacts.file_sha256,
        sealed_run_receipt_file_sha256=_sha256_bytes(context.sealed_run_bytes),
    )


def _closure_file_bindings(root: Path, entries: Sequence[str]) -> tuple[ClosureFileBinding, ...]:
    files: list[ClosureFileBinding] = []
    for relative_path in entries:
        source = root / relative_path
        try:
            metadata = source.lstat()
        except OSError as exc:
            raise ProductionControlError("cannot inspect staged closure member") from exc
        if stat.S_ISREG(metadata.st_mode):
            files.append(
                ClosureFileBinding(
                    relative_path=relative_path,
                    file_sha256=_file(source, label="staged production closure file"),
                    byte_count=metadata.st_size,
                )
            )
    return tuple(files)


def _finalized_control_payloads(
    context: _FinalizationContext,
    corpus: _FinalizationCorpus,
) -> Mapping[str, bytes]:
    config = _production_config(context, corpus)
    return {
        ONLINE_CUSTODY_ADMISSION_FILENAME: context.online_admission_bytes,
        PRODUCTION_CORPUS_CONFIG_FILENAME: config.canonical_file_bytes(),
        PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME: corpus.spec.canonical_file_bytes(),
        REQUIRED_ARTIFACT_BINDINGS_FILENAME: (corpus.required_artifacts.canonical_file_bytes()),
        SHARDED_EXECUTION_PLAN_FILENAME: corpus.sharded_execution_plan_bytes,
        TRIAL_RUNTIME_RECEIPT_FILENAME: corpus.trial_runtime_receipt_bytes,
    }


def _stage_finalized_closure(
    context: _FinalizationContext,
    staging_root: Path,
) -> tuple[str, tuple[str, ...], tuple[ClosureFileBinding, ...]]:
    _ensure_private_directory(staging_root)
    sealed_name = f"{context.manifest_sha256}.json"
    _publish_exact(
        staging_root / sealed_name,
        context.sealed_run_bytes,
        resume=False,
        label="staged sealed-run receipt",
    )
    for corpus in context.corpora:
        corpus_root = staging_root / corpus.corpus_id
        control_root = corpus_root / "control"
        _ensure_private_directory(corpus_root)
        _ensure_private_directory(control_root)
        for name, encoded in _finalized_control_payloads(context, corpus).items():
            _publish_exact(
                control_root / name,
                encoded,
                resume=False,
                label=f"{corpus.corpus_id} finalized control {name}",
            )
        _fsync_private_directory(
            control_root,
            label=f"{corpus.corpus_id} staged control directory",
        )
        _fsync_private_directory(
            corpus_root,
            label=f"{corpus.corpus_id} staged corpus directory",
        )
    _fsync_private_directory(
        staging_root,
        label="production closure staging directory",
    )
    expected = _finalized_all_entries(context.manifest_sha256)
    _scan_exact_tree(staging_root, expected, label="staged production closure")
    tree = digest_directory_tree(staging_root)
    return tree.sha256, tree.entries, _closure_file_bindings(staging_root, tree.entries)


@dataclass(frozen=True)
class _ObservedClosureState:
    kind: str
    tree_sha256: str
    entries: tuple[str, ...]
    files: tuple[ClosureFileBinding, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"receipt-only", "full-final"}:
            raise ProductionControlError("production closure state is unknown")


def _inspect_closure_state(
    context: _FinalizationContext,
    root: Path,
) -> _ObservedClosureState:
    try:
        tree = digest_directory_tree(root)
    except ArtifactIntegrityError as exc:
        raise ProductionControlError(
            f"cannot inspect production closure state at {root}: {exc}"
        ) from exc
    sealed_name = f"{context.manifest_sha256}.json"
    if tree.entries == (sealed_name,):
        sealed_path = root / sealed_name
        try:
            metadata = sealed_path.lstat()
        except OSError as exc:
            raise ProductionControlError("cannot inspect receipt-only production closure") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(context.sealed_run_bytes)
            or _read(sealed_path, label="receipt-only sealed-run receipt")
            != context.sealed_run_bytes
        ):
            raise ProductionControlError(
                "receipt-only production closure differs from the sealed-run receipt"
            )
        return _ObservedClosureState(
            kind="receipt-only",
            tree_sha256=tree.sha256,
            entries=tree.entries,
        )
    if frozenset(tree.entries) == _finalized_all_entries(context.manifest_sha256):
        if _read(root / sealed_name, label="final sealed-run receipt") != context.sealed_run_bytes:
            raise ProductionControlError(
                "full production closure contains another sealed-run receipt"
            )
        for corpus in context.corpora:
            control_root = root / corpus.corpus_id / "control"
            for name, encoded in _finalized_control_payloads(context, corpus).items():
                if (
                    _read(
                        control_root / name,
                        label=f"{corpus.corpus_id} finalized {name}",
                    )
                    != encoded
                ):
                    raise ProductionControlError(
                        f"{corpus.corpus_id} finalized control {name} differs"
                    )
        return _ObservedClosureState(
            kind="full-final",
            tree_sha256=tree.sha256,
            entries=tree.entries,
            files=_closure_file_bindings(root, tree.entries),
        )
    raise ProductionControlError(
        "production closure is neither exact receipt-only nor exact full-final state"
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _open_exchange_directory(parent_descriptor: int, name: str, *, label: str) -> int:
    """Open and bind one private exchange input to its name in the pinned parent."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ProductionControlError(f"cannot open {label}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        path_metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or not stat.S_ISDIR(path_metadata.st_mode)
            or _directory_identity(metadata) != _directory_identity(path_metadata)
            or metadata.st_nlink < 1
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        ):
            raise ProductionControlError(f"{label} must be one runner-owned private real directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _atomic_exchange_directories(first: Path, second: Path) -> None:
    """Exchange two descriptor-bound private directories and durably retain both names."""

    if first == second or first.parent != second.parent:
        raise ProductionControlError(
            "atomic exchange requires two distinct names in one exact parent"
        )
    if first.name in {"", ".", ".."} or second.name in {"", ".", ".."}:
        raise ProductionControlError("atomic exchange names are not canonical")

    parent_descriptor: int | None = None
    first_descriptor: int | None = None
    second_descriptor: int | None = None
    try:
        parent_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        parent_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(first.parent, parent_flags)
        parent_metadata = os.fstat(parent_descriptor)
        parent_path_metadata = first.parent.lstat()
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or not stat.S_ISDIR(parent_path_metadata.st_mode)
            or _directory_identity(parent_metadata) != _directory_identity(parent_path_metadata)
            or parent_metadata.st_nlink < 1
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
            or (hasattr(os, "geteuid") and parent_metadata.st_uid != os.geteuid())
        ):
            raise ProductionControlError(
                "atomic exchange parent must be one runner-controlled real directory"
            )

        first_descriptor = _open_exchange_directory(
            parent_descriptor,
            first.name,
            label="receipt-only production closure",
        )
        second_descriptor = _open_exchange_directory(
            parent_descriptor,
            second.name,
            label="full-final production closure",
        )
        first_metadata = os.fstat(first_descriptor)
        second_metadata = os.fstat(second_descriptor)
        if first_metadata.st_dev != second_metadata.st_dev:
            raise ProductionControlError(
                "atomic exchange requires two real directories on one filesystem"
            )

        first_before = os.stat(first.name, dir_fd=parent_descriptor, follow_symlinks=False)
        second_before = os.stat(second.name, dir_fd=parent_descriptor, follow_symlinks=False)
        parent_before = first.parent.lstat()
        if (
            _directory_identity(first_before) != _directory_identity(first_metadata)
            or _directory_identity(second_before) != _directory_identity(second_metadata)
            or _directory_identity(parent_before) != _directory_identity(parent_metadata)
        ):
            raise ProductionControlError("an atomic exchange name changed before the swap")

        library = ctypes.CDLL(None, use_errno=True)
        first_encoded = os.fsencode(first.name)
        second_encoded = os.fsencode(second.name)
        ctypes.set_errno(0)
        if sys.platform == "darwin":
            function = getattr(library, "renameatx_np", None)
            if function is None:
                raise ProductionControlError("macOS rename-swap is unavailable")
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
                first_encoded,
                parent_descriptor,
                second_encoded,
                0x00000002,
            )
        elif sys.platform.startswith("linux"):
            function = getattr(library, "renameat2", None)
            if function is None:
                raise ProductionControlError("Linux rename-exchange is unavailable")
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
                first_encoded,
                parent_descriptor,
                second_encoded,
                0x00000002,
            )
        else:
            raise ProductionControlError(
                "this platform lacks an admitted atomic directory-exchange primitive"
            )
        if result != 0:
            raise ProductionControlError(
                f"atomic directory exchange failed with errno {ctypes.get_errno()}"
            )

        first_after = os.stat(first.name, dir_fd=parent_descriptor, follow_symlinks=False)
        second_after = os.stat(second.name, dir_fd=parent_descriptor, follow_symlinks=False)
        parent_after = first.parent.lstat()
        if (
            not stat.S_ISDIR(first_after.st_mode)
            or not stat.S_ISDIR(second_after.st_mode)
            or _directory_identity(first_after) != _directory_identity(second_metadata)
            or _directory_identity(second_after) != _directory_identity(first_metadata)
            or _directory_identity(parent_after) != _directory_identity(parent_metadata)
        ):
            raise ProductionControlError(
                "atomic directory exchange did not retain the two admitted trees"
            )
        os.fsync(parent_descriptor)
    except ProductionControlError:
        raise
    except OSError as exc:
        raise ProductionControlError(
            f"cannot exchange production closure directories: {exc}"
        ) from exc
    finally:
        if second_descriptor is not None:
            os.close(second_descriptor)
        if first_descriptor is not None:
            os.close(first_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _binding_receipts(
    context: _FinalizationContext,
    *,
    final_tree_sha256: str,
    final_entries: tuple[str, ...],
    final_files: tuple[ClosureFileBinding, ...],
) -> tuple[FinalizedCorpusBinding, ...]:
    closure = context.materialization.finalized_controls_root
    sealed_relative = f"{context.manifest_sha256}.json"
    sealed_file_sha256 = _sha256_bytes(context.sealed_run_bytes)
    rows: list[FinalizedCorpusBinding] = []
    for corpus in context.corpora:
        config = _production_config(context, corpus)
        control_relative = f"{corpus.corpus_id}/control"
        config_relative = f"{control_relative}/{PRODUCTION_CORPUS_CONFIG_FILENAME}"
        spec_relative = f"{control_relative}/{PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME}"
        closure_binding = ProductionRunClosureBindingReceipt(
            corpus_id=corpus.corpus_id,
            manifest_sha256=context.manifest_sha256,
            preflight_launcher_contract_sha256=corpus.preflight.contract_sha256,
            runtime_plan_transition_receipt_sha256=corpus.transition.receipt_sha256,
            closure_source=str(closure),
            closure_target=str(closure),
            provisional_closure_tree_sha256=(context.blueprint.provisional_closure_tree_sha256),
            instantiated_closure_tree_sha256=final_tree_sha256,
            config_relative_path=config_relative,
            config_file_sha256=config.file_sha256,
            workload_spec_relative_path=spec_relative,
            workload_spec_file_sha256=corpus.spec.file_sha256,
            sealed_run_receipt_relative_path=sealed_relative,
            sealed_run_receipt_file_sha256=sealed_file_sha256,
            entries=final_entries,
            files=final_files,
        )
        rows.append(
            FinalizedCorpusBinding(
                corpus_id=corpus.corpus_id,
                workload_spec_file_sha256=corpus.spec.file_sha256,
                config_file_sha256=config.file_sha256,
                plan_template_file_sha256=(corpus.transition.final_plan_template_file_sha256),
                plan_template_semantic_sha256=(
                    corpus.transition.final_plan_template_semantic_sha256
                ),
                launcher_control_tree_sha256=(corpus.transition.final_control_tree_sha256),
                preflight_contract_sha256=corpus.preflight.contract_sha256,
                preflight_contract_file_sha256=corpus.preflight.file_sha256,
                preflight_receipt_sha256=corpus.preflight_receipt.receipt_sha256,
                preflight_receipt_file_sha256=corpus.preflight_receipt.file_sha256,
                transition_receipt_sha256=corpus.transition.receipt_sha256,
                transition_receipt_file_sha256=corpus.transition.file_sha256,
                closure_binding=closure_binding,
            )
        )
    return tuple(rows)


def _expected_finalization_receipt(
    context: _FinalizationContext,
    *,
    intermediate_tree_sha256: str,
    intermediate_entries: tuple[str, ...],
    retained_path: Path,
    final_tree_sha256: str,
    final_entries: tuple[str, ...],
    final_files: tuple[ClosureFileBinding, ...],
) -> ProductionControlFinalizationReceipt:
    return ProductionControlFinalizationReceipt(
        materialization_config_sha256=context.materialization.file_sha256,
        finalization_request_sha256=context.request.file_sha256,
        blueprint_receipt_sha256=context.blueprint.receipt_sha256,
        c0_control_instantiation_receipt_file_sha256=context.instantiation.file_sha256,
        manifest_sha256=context.manifest_sha256,
        c0_commit=context.c0_commit,
        c1_commit=context.c1_commit,
        sealed_run_receipt_file_sha256=_sha256_bytes(context.sealed_run_bytes),
        online_custody_admission_file_sha256=_sha256_bytes(context.online_admission_bytes),
        launcher_identity_file_sha256=(context.instantiation.launcher_identity_file_sha256),
        provisional_closure_tree_sha256=(context.blueprint.provisional_closure_tree_sha256),
        intermediate_closure_tree_sha256=intermediate_tree_sha256,
        intermediate_closure_entries=intermediate_entries,
        intermediate_sealed_run_receipt_byte_count=len(context.sealed_run_bytes),
        instantiated_closure_tree_sha256=final_tree_sha256,
        instantiated_closure_entries=final_entries,
        retained_intermediate_closure_path=str(retained_path),
        suite_attempt_id=suite_attempt_id(context.manifest_sha256),
        canonical_suite_namespace=str(
            suite_namespace(
                context.materialization.suite_base_root,
                context.manifest_sha256,
            )
        ),
        pre_c1_output_staging_root=str(
            context.materialization.suite_base_root
            / f".pre-c1-output-{context.materialization.file_sha256[:20]}"
        ),
        corpora=_binding_receipts(
            context,
            final_tree_sha256=final_tree_sha256,
            final_entries=final_entries,
            final_files=final_files,
        ),
    )


def _finalization_lock_path(closure: Path) -> Path:
    return closure.parent / f".{closure.name}.finalization.lock"


def _finalization_file_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@contextmanager
def _production_finalization_lock(closure: Path) -> Iterator[None]:
    """Serialize classification, exchange, and receipt publication for one closure."""

    path = _finalization_lock_path(closure)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory: int | None = None
    descriptor: int | None = None
    acquired = False
    try:
        directory = os.open(path.parent, directory_flags)
        directory_metadata = os.fstat(directory)
        directory_path_metadata = path.parent.lstat()
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or (directory_metadata.st_dev, directory_metadata.st_ino)
            != (directory_path_metadata.st_dev, directory_path_metadata.st_ino)
            or (hasattr(os, "geteuid") and directory_metadata.st_uid != os.geteuid())
            or stat.S_IMODE(directory_metadata.st_mode) & 0o022
        ):
            raise ProductionControlError(
                "production finalization lock parent must be a runner-controlled real directory"
            )
        descriptor = os.open(path.name, file_flags, 0o600, dir_fd=directory)
        before = os.fstat(descriptor)
        path_before = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != 0
            or stat.S_IMODE(before.st_mode) != 0o600
            or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
            or (before.st_dev, before.st_ino) != (path_before.st_dev, path_before.st_ino)
        ):
            raise ProductionControlError(
                "production finalization lock must be one private empty regular file"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProductionControlError(
                "production control finalization already has a live worker"
            ) from exc
        acquired = True
        yield
        after = os.fstat(descriptor)
        path_after = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if _finalization_file_signature(before) != _finalization_file_signature(after) or (
            after.st_dev,
            after.st_ino,
        ) != (path_after.st_dev, path_after.st_ino):
            raise ProductionControlError("production finalization lock changed while held")
    except ProductionControlError:
        raise
    except OSError as exc:
        raise ProductionControlError(f"cannot secure production finalization lock: {exc}") from exc
    finally:
        if descriptor is not None:
            if acquired:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(descriptor)
        if directory is not None:
            os.close(directory)


def _finalize_production_controls_unlocked(
    finalization_request_path: str | Path,
    *,
    expected_request_sha256: str,
    finalization_receipt_path: str | Path,
    resume: bool = False,
) -> ProductionControlFinalizationReceipt:
    """Build the complete closure, then atomically exchange it with the receipt-only root."""

    request = load_production_control_finalization_request(
        finalization_request_path,
        expected_sha256=expected_request_sha256,
    )
    context = _load_finalization_context(request)
    closure = context.materialization.finalized_controls_root
    receipt_path = _absolute_path("finalization_receipt_path", finalization_receipt_path)
    if PurePosixPath(str(receipt_path)).is_relative_to(PurePosixPath(str(closure))):
        raise ProductionControlError(
            "finalization receipt must remain outside the production closure"
        )
    if os.path.lexists(receipt_path):
        if not resume:
            raise ProductionControlError(
                "finalization receipt already exists; use resume after custody review"
            )
        persisted = load_production_control_finalization_receipt(receipt_path)
        _verify_finalized_state(context, persisted)
        return persisted
    retained_path = closure.parent / (f"{_PLACEHOLDER_RETENTION_PREFIX}{request.file_sha256[:20]}")
    if os.path.lexists(retained_path):
        if not resume:
            raise ProductionControlError("retained closure exists; use resume after custody review")
        closure_state = _inspect_closure_state(context, closure)
        retained_state = _inspect_closure_state(context, retained_path)
    else:
        closure_state = _inspect_closure_state(context, closure)
        if closure_state.kind != "receipt-only":
            raise ProductionControlError(
                "a missing retained name is valid only before final closure staging"
            )
        staged_tree, staged_entries, staged_files = _stage_finalized_closure(
            context,
            retained_path,
        )
        retained_state = _inspect_closure_state(context, retained_path)
        if (
            retained_state.kind != "full-final"
            or retained_state.tree_sha256 != staged_tree
            or retained_state.entries != staged_entries
            or retained_state.files != staged_files
        ):
            raise ProductionControlError(
                "staged full-final closure differs from its fresh verification"
            )

    exchange_required = False
    if closure_state.kind == "receipt-only" and retained_state.kind == "full-final":
        intermediate_state = closure_state
        final_state = retained_state
        exchange_required = True
    elif closure_state.kind == "full-final" and retained_state.kind == "receipt-only":
        final_state = closure_state
        intermediate_state = retained_state
    else:
        raise ProductionControlError(
            "production closure names are not in an admitted recovery state"
        )

    expected = _expected_finalization_receipt(
        context,
        intermediate_tree_sha256=intermediate_state.tree_sha256,
        intermediate_entries=intermediate_state.entries,
        retained_path=retained_path,
        final_tree_sha256=final_state.tree_sha256,
        final_entries=final_state.entries,
        final_files=final_state.files,
    )
    if exchange_required:
        _atomic_exchange_directories(closure, retained_path)
        observed_final = _inspect_closure_state(context, closure)
        observed_intermediate = _inspect_closure_state(context, retained_path)
        if (
            observed_final.kind != "full-final"
            or observed_final.tree_sha256 != final_state.tree_sha256
            or observed_final.entries != final_state.entries
            or observed_final.files != final_state.files
            or observed_intermediate.kind != "receipt-only"
            or observed_intermediate.tree_sha256 != intermediate_state.tree_sha256
            or observed_intermediate.entries != intermediate_state.entries
        ):
            raise ProductionControlError("atomic exchange state differs from its derived receipt")
    for corpus, row in zip(context.corpora, expected.corpora, strict=True):
        verify_production_run_closure_binding(
            corpus.preflight,
            corpus.transition,
            row.closure_binding,
        )
    _publish_exact(
        receipt_path,
        expected.canonical_file_bytes(),
        resume=False,
        label="production control finalization receipt",
    )
    return expected


def finalize_production_controls(
    finalization_request_path: str | Path,
    *,
    expected_request_sha256: str,
    finalization_receipt_path: str | Path,
    resume: bool = False,
) -> ProductionControlFinalizationReceipt:
    """Finalize one closure while holding its non-reentrant process lock."""

    request = load_production_control_finalization_request(
        finalization_request_path,
        expected_sha256=expected_request_sha256,
    )
    context = _load_finalization_context(request)
    with _production_finalization_lock(context.materialization.finalized_controls_root):
        # Reload every authority and filesystem state after lock acquisition. A
        # waiting process must never act on a pre-exchange classification.
        return _finalize_production_controls_unlocked(
            finalization_request_path,
            expected_request_sha256=expected_request_sha256,
            finalization_receipt_path=finalization_receipt_path,
            resume=resume,
        )


def _verify_finalized_state(
    context: _FinalizationContext,
    receipt: ProductionControlFinalizationReceipt,
) -> None:
    closure = context.materialization.finalized_controls_root
    expected_retained = closure.parent / (
        f"{_PLACEHOLDER_RETENTION_PREFIX}{context.request.file_sha256[:20]}"
    )
    if Path(receipt.retained_intermediate_closure_path) != expected_retained:
        raise ProductionControlError("retained closure path is not deterministically derived")
    retained = _inspect_closure_state(context, expected_retained)
    if (
        retained.kind != "receipt-only"
        or retained.tree_sha256 != receipt.intermediate_closure_tree_sha256
        or retained.entries != receipt.intermediate_closure_entries
    ):
        raise ProductionControlError("retained intermediate closure differs")
    final = _inspect_closure_state(context, closure)
    if (
        final.kind != "full-final"
        or final.tree_sha256 != receipt.instantiated_closure_tree_sha256
        or final.entries != receipt.instantiated_closure_entries
    ):
        raise ProductionControlError("final production closure differs")
    expected = _expected_finalization_receipt(
        context,
        intermediate_tree_sha256=retained.tree_sha256,
        intermediate_entries=retained.entries,
        retained_path=expected_retained,
        final_tree_sha256=final.tree_sha256,
        final_entries=final.entries,
        final_files=final.files,
    )
    if expected != receipt:
        raise ProductionControlError("finalization receipt differs from fresh derivation")
    for corpus, row in zip(context.corpora, receipt.corpora, strict=True):
        verify_production_run_closure_binding(
            corpus.preflight,
            corpus.transition,
            row.closure_binding,
        )


def _load_and_verify_authority_binding(
    *,
    finalization_request_path: Path,
    finalization_receipt_path: Path,
    preflight: PreflightLaunchContract,
    transition: RuntimePlanTransitionReceipt,
) -> ProductionRunClosureBindingReceipt:
    receipt = load_production_control_finalization_receipt(finalization_receipt_path)
    request = load_production_control_finalization_request(
        finalization_request_path,
        expected_sha256=receipt.finalization_request_sha256,
    )
    context = _load_finalization_context(request)
    closure = context.materialization.finalized_controls_root
    if PurePosixPath(str(finalization_receipt_path)).is_relative_to(PurePosixPath(str(closure))):
        raise ProductionControlError("finalization receipt resides inside its closure")
    _verify_finalized_state(context, receipt)
    matching = tuple(
        (corpus, row)
        for corpus, row in zip(context.corpora, receipt.corpora, strict=True)
        if corpus.corpus_id == preflight.geometry.corpus_id
    )
    if len(matching) != 1:
        raise ProductionControlError("preflight corpus is absent from finalization")
    corpus, row = matching[0]
    if corpus.preflight != preflight or corpus.transition != transition:
        raise ProductionControlError(
            "caller launch evidence differs from the freshly reloaded authority"
        )
    verify_production_run_closure_binding(
        preflight,
        transition,
        row.closure_binding,
    )
    return row.closure_binding


def verify_production_run_closure_authority(
    *,
    finalization_request_path: str | Path,
    finalization_receipt_path: str | Path,
    preflight: PreflightLaunchContract,
    transition: RuntimePlanTransitionReceipt,
) -> VerifiedProductionRunClosure:
    """Freshly reproduce C1, factory, custody, transition, and closure authority."""

    if not isinstance(preflight, PreflightLaunchContract) or not isinstance(
        transition,
        RuntimePlanTransitionReceipt,
    ):
        raise ProductionControlError("closure authority launch evidence must be typed")
    request_path = _absolute_path(
        "finalization_request_path",
        finalization_request_path,
    )
    receipt_path = _absolute_path(
        "finalization_receipt_path",
        finalization_receipt_path,
    )

    def revalidate() -> ProductionRunClosureBindingReceipt:
        return _load_and_verify_authority_binding(
            finalization_request_path=request_path,
            finalization_receipt_path=receipt_path,
            preflight=preflight,
            transition=transition,
        )

    binding = revalidate()
    return _mint_verified_production_run_closure(
        binding,
        fresh_revalidator=revalidate,
    )


def _verified_manifest_fragment_document(
    materialization_config_path: Path,
    *,
    expected_config_sha256: str,
    expected_blueprint_receipt_file_sha256: str,
) -> tuple[bytes, ProductionControlMaterializationConfig]:
    materialization = load_production_control_config(
        materialization_config_path,
        expected_sha256=expected_config_sha256,
    )
    blueprint = load_production_control_blueprint_receipt(
        materialization.blueprint_receipt_path,
        expected_sha256=expected_blueprint_receipt_file_sha256,
    )
    _scan_exact_tree(
        materialization.blueprint_root,
        _blueprint_all_entries(),
        label="production control blueprint",
    )
    if (
        _payload_tree_sha256(
            materialization.blueprint_root,
            _blueprint_payload_entries(),
        )
        != blueprint.payload_tree_sha256
    ):
        raise ProductionControlError("production control blueprint payload differs")
    workload_rows = _parse_array(
        _read_pinned(
            materialization.production_workloads_fragment_path,
            blueprint.production_workloads_fragment_file_sha256,
            label="production workloads manifest fragment",
        ),
        label="production workloads manifest fragment",
    )
    hardware = _parse_object(
        _read_pinned(
            materialization.production_hardware_fragment_path,
            blueprint.production_hardware_fragment_file_sha256,
            label="production hardware manifest fragment",
        ),
        label="production hardware manifest fragment",
    )
    manifest_projection = {
        "analysis": {
            "power": {"selected_families_per_corpus": blueprint.workloads[0].selected_family_count}
        },
        "production_workloads": workload_rows,
        "sealed_execution": {
            "code_commit": C0_COMMIT_SENTINEL,
            "hardware": hardware,
            "production_controls": {
                "blueprint_receipt_file_sha256": blueprint.file_sha256,
                "blueprint_receipt_sha256": blueprint.semantic_sha256,
                "materialization_config_file_sha256": materialization.file_sha256,
            },
            "runner_identity": materialization.runner_identity,
            "runner_image": blueprint.runner_image,
        },
    }
    _verify_blueprint_manifest_fragments(
        materialization,
        blueprint,
        manifest_projection,
    )
    return (
        _canonical_bytes(
            {
                "production_workloads": workload_rows,
                "sealed_execution": {
                    "hardware": hardware,
                    "production_controls": {
                        "blueprint_receipt_file_sha256": blueprint.file_sha256,
                        "blueprint_receipt_sha256": blueprint.semantic_sha256,
                        "materialization_config_file_sha256": materialization.file_sha256,
                    },
                },
            }
        )
        + b"\n",
        materialization,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fractal-production-controls")
    commands = parser.add_subparsers(dest="command", required=True)
    write_config = commands.add_parser("write-config")
    write_config.add_argument("--factory-config", type=Path, required=True)
    write_config.add_argument(
        "--c0-runtime-extraction-receipt",
        type=Path,
        required=True,
    )
    write_config.add_argument("--opa-binary", type=Path, required=True)
    write_config.add_argument("--uv-lock", type=Path, required=True)
    write_config.add_argument("--pseudonym-key", type=Path, required=True)
    write_config.add_argument("--scientific-candidate-reference", required=True)
    write_config.add_argument("--scientific-production-reference", required=True)
    write_config.add_argument("--approval-environment", required=True)
    write_config.add_argument("--runner-platform", required=True)
    write_config.add_argument("--runner-identity", required=True)
    write_config.add_argument("--hostname", required=True)
    write_config.add_argument("--hardware-provider", required=True)
    write_config.add_argument("--hardware-instance-type", required=True)
    write_config.add_argument("--hardware-cpu-model", required=True)
    write_config.add_argument("--hardware-accelerator", required=True)
    write_config.add_argument("--hardware-region", required=True)
    write_config.add_argument("--hardware-operating-system", required=True)
    write_config.add_argument("--memory-limit-bytes", type=int, required=True)
    write_config.add_argument("--cpuset-cpus", type=_cpuset_argument, required=True)
    write_config.add_argument("--tmpfs-size-bytes", type=int, required=True)
    write_config.add_argument("--blueprint-root", type=Path, required=True)
    write_config.add_argument("--finalized-controls-root", type=Path, required=True)
    write_config.add_argument("--suite-base-root", type=Path, required=True)
    write_config.add_argument("--output", type=Path, required=True)
    write_config.add_argument("--receipt", type=Path, required=True)
    blueprint = commands.add_parser("materialize-blueprint")
    blueprint.add_argument("--materialization-config", type=Path, required=True)
    blueprint.add_argument("--materialization-config-sha256", required=True)
    blueprint.add_argument("--resume", action="store_true")
    instantiate_c0 = commands.add_parser("instantiate-c0-controls")
    instantiate_c0.add_argument("--materialization-config", type=Path, required=True)
    instantiate_c0.add_argument("--candidate-package", type=Path, required=True)
    instantiate_c0.add_argument("--candidate-image-closure", type=Path, required=True)
    instantiate_c0.add_argument("--apparatus-commit", required=True)
    instantiate_c0.add_argument("--output-root", type=Path, required=True)
    fragments = commands.add_parser("print-manifest-fragments")
    fragments.add_argument("--materialization-config", type=Path, required=True)
    fragments.add_argument("--materialization-config-sha256", required=True)
    fragments.add_argument("--blueprint-receipt-file-sha256", required=True)
    fragments.add_argument("--print-paths", action="store_true")
    write_bindings = commands.add_parser("write-required-artifact-bindings")
    write_bindings.add_argument("--materialization-config", type=Path, required=True)
    write_bindings.add_argument(
        "--c0-control-instantiation-receipt",
        type=Path,
        required=True,
    )
    write_bindings.add_argument("--frozen-manifest", type=Path, required=True)
    write_bindings.add_argument("--artifact-verification-receipt", type=Path, required=True)
    write_bindings.add_argument("--artifact-root", type=Path, required=True)
    write_bindings.add_argument("--artifact-map", type=Path, required=True)
    write_bindings.add_argument("--output-root", type=Path, required=True)
    write_request = commands.add_parser("write-finalization-request")
    write_request.add_argument("--materialization-config", type=Path, required=True)
    write_request.add_argument(
        "--c0-control-instantiation-receipt",
        type=Path,
        required=True,
    )
    write_request.add_argument("--frozen-manifest", type=Path, required=True)
    write_request.add_argument("--manifest-lock", type=Path, required=True)
    write_request.add_argument("--c1-package-root", type=Path, required=True)
    write_request.add_argument("--protocol-registry-record", type=Path, required=True)
    write_request.add_argument("--protocol-registration-receipt", type=Path, required=True)
    write_request.add_argument("--online-custody-admission", type=Path, required=True)
    write_request.add_argument("--custody-seal-receipt", type=Path, required=True)
    write_request.add_argument("--artifact-verification-receipt", type=Path, required=True)
    write_request.add_argument("--artifact-root", type=Path, required=True)
    write_request.add_argument("--artifact-map", type=Path, required=True)
    write_request.add_argument("--required-artifact-bindings-root", type=Path, required=True)
    write_request.add_argument("--runtime-evidence-root", type=Path, required=True)
    write_request.add_argument("--output", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--request", type=Path, required=True)
    finalize.add_argument("--request-sha256", required=True)
    finalize.add_argument("--receipt", type=Path, required=True)
    finalize.add_argument("--resume", action="store_true")
    verify = commands.add_parser("verify-authority")
    verify.add_argument("--request", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--preflight-contract", type=Path, required=True)
    verify.add_argument("--transition-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write-config":
        receipt = write_production_control_materialization_config(
            factory_config_path=args.factory_config,
            c0_runtime_extraction_receipt_path=args.c0_runtime_extraction_receipt,
            opa_binary_path=args.opa_binary,
            uv_lock_path=args.uv_lock,
            pseudonym_key_path=args.pseudonym_key,
            scientific_candidate_reference=args.scientific_candidate_reference,
            scientific_production_reference=args.scientific_production_reference,
            approval_environment=args.approval_environment,
            runner_platform=args.runner_platform,
            runner_identity=args.runner_identity,
            hostname=args.hostname,
            hardware_provider=args.hardware_provider,
            hardware_instance_type=args.hardware_instance_type,
            hardware_cpu_model=args.hardware_cpu_model,
            hardware_accelerator=args.hardware_accelerator,
            hardware_region=args.hardware_region,
            hardware_operating_system=args.hardware_operating_system,
            memory_limit_bytes=args.memory_limit_bytes,
            cpuset_cpus=args.cpuset_cpus,
            tmpfs_size_bytes=args.tmpfs_size_bytes,
            blueprint_root=args.blueprint_root,
            finalized_controls_root=args.finalized_controls_root,
            suite_base_root=args.suite_base_root,
            output=args.output,
            receipt_output=args.receipt,
        )
        print(
            _canonical_bytes(
                {
                    "approval_environment": receipt.approval_environment,
                    "config_file_sha256": receipt.config_file_sha256,
                    "config_path": str(receipt.config_path),
                    "oci_promotion_required": receipt.oci_promotion_required,
                    "receipt_file_sha256": receipt.file_sha256,
                    "receipt_path": str(_absolute_path("receipt_output", args.receipt)),
                    "scientific_index_digest": receipt.scientific_index_digest,
                    "status": "written",
                }
            ).decode("utf-8")
        )
        return 0
    if args.command == "materialize-blueprint":
        receipt = materialize_production_control_blueprint(
            args.materialization_config,
            expected_config_sha256=args.materialization_config_sha256,
            resume=args.resume,
        )
        print(
            _canonical_bytes(
                {
                    "blueprint_receipt_file_sha256": receipt.file_sha256,
                    "blueprint_receipt_sha256": receipt.semantic_sha256,
                    "materialization_config_file_sha256": receipt.materialization_config_sha256,
                    "status": "materialized",
                }
            ).decode("utf-8")
        )
        return 0
    if args.command == "instantiate-c0-controls":
        receipt = instantiate_c0_production_controls(
            materialization_config_path=args.materialization_config,
            candidate_manifest_package_path=args.candidate_package,
            candidate_image_closure_path=args.candidate_image_closure,
            apparatus_commit=args.apparatus_commit,
            output_root=args.output_root,
        )
        print(
            _canonical_bytes(
                {
                    "apparatus_commit": receipt.apparatus_commit,
                    "candidate_image_source_commit": (receipt.candidate_image_source_commit),
                    "instantiated_payload_tree_sha256": (receipt.instantiated_payload_tree_sha256),
                    "receipt_file_sha256": receipt.file_sha256,
                    "receipt_path": str(
                        _absolute_path("output_root", args.output_root)
                        / C0_INSTANTIATION_RECEIPT_FILENAME
                    ),
                    "status": "instantiated",
                }
            ).decode("utf-8")
        )
        return 0
    if args.command == "print-manifest-fragments":
        encoded, materialization = _verified_manifest_fragment_document(
            args.materialization_config,
            expected_config_sha256=args.materialization_config_sha256,
            expected_blueprint_receipt_file_sha256=(args.blueprint_receipt_file_sha256),
        )
        if args.print_paths:
            print(materialization.production_workloads_fragment_path)
            print(materialization.production_hardware_fragment_path)
        else:
            sys.stdout.buffer.write(encoded)
        return 0
    if args.command == "write-required-artifact-bindings":
        bindings = write_required_artifact_binding_suite(
            materialization_config_path=args.materialization_config,
            c0_control_instantiation_receipt_path=(args.c0_control_instantiation_receipt),
            frozen_manifest_path=args.frozen_manifest,
            artifact_verification_receipt_path=args.artifact_verification_receipt,
            artifact_root=args.artifact_root,
            local_artifact_map_path=args.artifact_map,
            output_root=args.output_root,
        )
        print(
            _canonical_bytes(
                {
                    "bindings": [
                        {"corpus_id": corpus_id, "file_sha256": binding.file_sha256}
                        for corpus_id, binding in zip(FIXED_CORPORA, bindings, strict=True)
                    ],
                    "output_root": str(
                        _absolute_path(
                            "required_artifact_bindings_output_root",
                            args.output_root,
                        )
                    ),
                    "status": "written",
                }
            ).decode("utf-8")
        )
        return 0
    if args.command == "write-finalization-request":
        request = write_production_control_finalization_request(
            materialization_config_path=args.materialization_config,
            c0_control_instantiation_receipt_path=(args.c0_control_instantiation_receipt),
            frozen_manifest_path=args.frozen_manifest,
            manifest_lock_path=args.manifest_lock,
            c1_package_root=args.c1_package_root,
            protocol_registry_record_path=args.protocol_registry_record,
            protocol_registration_receipt_path=args.protocol_registration_receipt,
            online_custody_admission_path=args.online_custody_admission,
            custody_seal_receipt_path=args.custody_seal_receipt,
            artifact_verification_receipt_path=args.artifact_verification_receipt,
            artifact_root=args.artifact_root,
            local_artifact_map_path=args.artifact_map,
            required_artifact_bindings_root=args.required_artifact_bindings_root,
            runtime_evidence_root=args.runtime_evidence_root,
            output=args.output,
        )
        print(
            _canonical_bytes(
                {
                    "output": str(_absolute_path("finalization_request_output", args.output)),
                    "request_sha256": request.file_sha256,
                    "status": "written",
                }
            ).decode("utf-8")
        )
        return 0
    if args.command == "finalize":
        receipt = finalize_production_controls(
            args.request,
            expected_request_sha256=args.request_sha256,
            finalization_receipt_path=args.receipt,
            resume=args.resume,
        )
        print(receipt.receipt_sha256)
        return 0
    verified = verify_production_run_closure_authority(
        finalization_request_path=args.request,
        finalization_receipt_path=args.receipt,
        preflight=load_preflight_launch_contract(args.preflight_contract),
        transition=load_runtime_plan_transition(args.transition_receipt),
    )
    print(verified.binding.receipt_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
