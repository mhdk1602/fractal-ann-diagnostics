"""No-network entry point shipped inside the scientific confirmatory image."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .artifact_integrity import (
    ArtifactIntegrityError,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .confirmatory_analysis import (
    ConfirmatoryAnalysisError,
    ConfirmatoryInputArtifact,
    ConfirmatoryResultArtifact,
    run_confirmatory_analysis,
)
from .confirmatory_execution import (
    ConfirmatoryAnalysisAttemptReceipt,
    ConfirmatoryAnalysisResultReceipt,
)
from .confirmatory_modeling import (
    FrozenModelSuite,
    canonical_h1_model_artifact_bytes,
    canonical_h2_model_suite_artifact_bytes,
)
from .offline_analysis_contract import (
    MAX_PACKAGE_FILE_BYTES,
    NETWORK_MODE,
    PACKAGE_MOUNT_PATH,
    RESULTS_MOUNT_PATH,
    RUNTIME_DYNAMIC_ENVIRONMENT_NAMES,
    RUNTIME_ENVIRONMENT,
    RUNTIME_GID,
    RUNTIME_MACHINE,
    RUNTIME_PLATFORM,
    RUNTIME_UID,
    TMPFS_MOUNT_PATH,
    OfflineAnalysisAdmission,
    OfflineAnalysisContractError,
    OfflineAnalysisFileBinding,
    canonical_bytes,
    decode_canonical_object,
    load_offline_analysis_admission,
    load_offline_input_bundle,
    sha256_bytes,
)

_MAX_PROC_BYTES = 4 * 1024 * 1024
_STANDARD_CONTAINER_MOUNT_PATHS = frozenset(
    {
        "/",
        "/dev",
        "/dev/mqueue",
        "/dev/pts",
        "/dev/shm",
        "/etc/hostname",
        "/etc/hosts",
        "/etc/resolv.conf",
        "/proc",
        "/proc/acpi",
        "/proc/bus",
        "/proc/fs",
        "/proc/interrupts",
        "/proc/irq",
        "/proc/kcore",
        "/proc/keys",
        "/proc/latency_stats",
        "/proc/scsi",
        "/proc/sys",
        "/proc/sysrq-trigger",
        "/proc/timer_list",
        "/sys",
        "/sys/firmware",
        "/sys/fs/cgroup",
    }
)
_CPU_THERMAL_THROTTLE_MOUNT = re.compile(
    r"^/sys/devices/system/cpu/cpu(?:0|[1-9][0-9]*)/thermal_throttle$"
)
_MATERIALIZATION_RECEIPT_FIELDS = frozenset(
    {
        "artifact_byte_count",
        "artifact_file_sha256",
        "artifact_sha256",
        "artifact_uri",
        "manifest_sha256",
        "members",
        "run_receipt_sha256",
        "schema_version",
        "suite_attempt_id",
        "suite_descriptor_sha256",
        "suite_state_record_sha256",
    }
)
_MATERIALIZATION_MEMBER_FIELDS = frozenset(
    {
        "byte_count",
        "corpus_id",
        "file_sha256",
        "role",
        "schema_version",
        "semantic_sha256",
        "uri",
    }
)


class OfflineAnalysisRuntimeError(ValueError):
    """Raised before or during the one admitted container analysis."""


@dataclass(frozen=True)
class RuntimeMountObservation:
    mount_path: str
    filesystem_type: str
    mount_options: tuple[str, ...]
    super_options: tuple[str, ...]

    def __post_init__(self) -> None:
        path = PurePosixPath(self.mount_path)
        if not path.is_absolute() or str(path) != self.mount_path:
            raise OfflineAnalysisRuntimeError("runtime mount path is not canonical")
        if not self.filesystem_type:
            raise OfflineAnalysisRuntimeError("runtime mount filesystem is absent")


@dataclass(frozen=True)
class RuntimeObservation:
    """Kernel-facing facts captured before any scientific input is opened."""

    operating_system: str
    machine: str
    uid: int
    gid: int
    environment: tuple[tuple[str, str], ...]
    network_interfaces: tuple[str, ...]
    mounts: tuple[RuntimeMountObservation, ...]


@dataclass(frozen=True)
class OfflineRuntimeOutcome:
    attempt_path: Path
    result_receipt_path: Path
    result_path: Path
    result_artifact_sha256: str


def _closed(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise OfflineAnalysisRuntimeError(f"{label} must be a JSON object")
    observed = set(value)
    if observed != fields:
        raise OfflineAnalysisRuntimeError(
            f"{label} fields differ; missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _read_proc(path: str, *, label: str) -> bytes:
    try:
        with open(path, "rb", buffering=0) as stream:
            encoded = stream.read(_MAX_PROC_BYTES + 1)
    except OSError as exc:
        raise OfflineAnalysisRuntimeError(f"cannot read {label}: {exc}") from exc
    if len(encoded) > _MAX_PROC_BYTES:
        raise OfflineAnalysisRuntimeError(f"{label} exceeds the fixed read limit")
    return encoded


def _decode_mount_path(value: str) -> str:
    replacements = {
        r"\011": "\t",
        r"\012": "\n",
        r"\040": " ",
        r"\134": "\\",
    }
    for encoded, decoded in replacements.items():
        value = value.replace(encoded, decoded)
    return value


def _mount_observations(encoded: bytes) -> tuple[RuntimeMountObservation, ...]:
    try:
        lines = encoded.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise OfflineAnalysisRuntimeError("mountinfo is not valid UTF-8") from exc
    rows: list[RuntimeMountObservation] = []
    for number, line in enumerate(lines, start=1):
        fields = line.split(" ")
        try:
            separator = fields.index("-")
            mount_path = _decode_mount_path(fields[4])
            mount_options = tuple(sorted(set(fields[5].split(","))))
            filesystem_type = fields[separator + 1]
            super_options = tuple(sorted(set(fields[separator + 3].split(","))))
        except (IndexError, ValueError) as exc:
            raise OfflineAnalysisRuntimeError(f"mountinfo line {number} is malformed") from exc
        rows.append(
            RuntimeMountObservation(
                mount_path=mount_path,
                filesystem_type=filesystem_type,
                mount_options=mount_options,
                super_options=super_options,
            )
        )
    return tuple(rows)


def capture_runtime_observation() -> RuntimeObservation:
    """Capture the isolated container boundary without opening study bytes."""

    try:
        interfaces = tuple(
            sorted(
                (entry.name for entry in os.scandir("/sys/class/net")),
                key=lambda value: value.encode("utf-8"),
            )
        )
    except OSError as exc:
        raise OfflineAnalysisRuntimeError(f"cannot inspect network namespace: {exc}") from exc
    environment = tuple(sorted(os.environ.items(), key=lambda row: row[0].encode("utf-8")))
    return RuntimeObservation(
        operating_system=platform.system(),
        machine=platform.machine(),
        uid=os.geteuid(),
        gid=os.getegid(),
        environment=environment,
        network_interfaces=interfaces,
        mounts=_mount_observations(_read_proc("/proc/self/mountinfo", label="container mountinfo")),
    )


def _one_mount(
    observation: RuntimeObservation,
    path: str,
) -> RuntimeMountObservation:
    matches = [row for row in observation.mounts if row.mount_path == path]
    if len(matches) != 1:
        raise OfflineAnalysisRuntimeError(f"runtime must expose one exact {path} mount")
    return matches[0]


def validate_runtime_observation(
    admission: OfflineAnalysisAdmission,
    observation: RuntimeObservation,
) -> None:
    """Reject host execution and any platform, env, network, or mount drift."""

    if not isinstance(admission, OfflineAnalysisAdmission):
        raise OfflineAnalysisRuntimeError("runtime admission is untyped")
    if not isinstance(observation, RuntimeObservation):
        raise OfflineAnalysisRuntimeError("runtime observation is untyped")
    if (
        observation.operating_system != "Linux"
        or observation.machine != RUNTIME_MACHINE
        or admission.runtime_platform != RUNTIME_PLATFORM
        or observation.uid != RUNTIME_UID
        or observation.gid != RUNTIME_GID
    ):
        raise OfflineAnalysisRuntimeError(
            "analysis must run as the registered nonroot Linux AMD64 image identity"
        )
    expected_environment = dict(RUNTIME_ENVIRONMENT)
    observed_environment = dict(observation.environment)
    if (
        len(observed_environment) != len(observation.environment)
        or set(observed_environment)
        != set(expected_environment) | set(RUNTIME_DYNAMIC_ENVIRONMENT_NAMES)
        or any(
            observed_environment.get(name) != value for name, value in expected_environment.items()
        )
    ):
        raise OfflineAnalysisRuntimeError("scientific container environment drifted")
    for name in RUNTIME_DYNAMIC_ENVIRONMENT_NAMES:
        value = observed_environment.get(name)
        if type(value) is not str or not value:
            raise OfflineAnalysisRuntimeError(f"dynamic runtime environment {name} is absent")
    if admission.network_mode != NETWORK_MODE or observation.network_interfaces != ("lo",):
        raise OfflineAnalysisRuntimeError(
            "scientific container has a non-loopback network interface"
        )
    mount_paths = tuple(row.mount_path for row in observation.mounts)
    allowed_mount_paths = _STANDARD_CONTAINER_MOUNT_PATHS | {
        PACKAGE_MOUNT_PATH,
        RESULTS_MOUNT_PATH,
        TMPFS_MOUNT_PATH,
    }
    unexpected_mounts = tuple(
        sorted(
            (
                path
                for path in mount_paths
                if path not in allowed_mount_paths
                and _CPU_THERMAL_THROTTLE_MOUNT.fullmatch(path) is None
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if len(set(mount_paths)) != len(mount_paths) or unexpected_mounts:
        raise OfflineAnalysisRuntimeError(
            f"scientific container mount closure differs; unexpected={list(unexpected_mounts)}"
        )
    root = _one_mount(observation, "/")
    package = _one_mount(observation, PACKAGE_MOUNT_PATH)
    results = _one_mount(observation, RESULTS_MOUNT_PATH)
    tmpfs = _one_mount(observation, TMPFS_MOUNT_PATH)
    if "ro" not in set(root.mount_options) | set(root.super_options):
        raise OfflineAnalysisRuntimeError("container root filesystem is not read-only")
    if "ro" not in package.mount_options or "rw" in package.mount_options:
        raise OfflineAnalysisRuntimeError("analysis package mount is not read-only")
    if "rw" not in results.mount_options or "ro" in results.mount_options:
        raise OfflineAnalysisRuntimeError("analysis results mount is not read-write")
    tmpfs_flags = set(tmpfs.mount_options) | set(tmpfs.super_options)
    if tmpfs.filesystem_type != "tmpfs" or not {
        "rw",
        "noexec",
        "nosuid",
        "nodev",
    }.issubset(tmpfs_flags):
        raise OfflineAnalysisRuntimeError("analysis tmpfs hardening differs")
    for row in observation.mounts:
        for root_path in (PACKAGE_MOUNT_PATH, RESULTS_MOUNT_PATH):
            if row.mount_path.startswith(f"{root_path}/"):
                raise OfflineAnalysisRuntimeError(
                    "analysis package or results tree contains a nested mount"
                )


def _binding(
    admission: OfflineAnalysisAdmission,
    role: str,
) -> OfflineAnalysisFileBinding:
    matches = [row for row in admission.package_files if row.role == role]
    if len(matches) != 1:
        raise OfflineAnalysisRuntimeError(f"admission lacks one exact {role} file")
    return matches[0]


def _read_package_file(
    admission: OfflineAnalysisAdmission,
    role: str,
) -> tuple[Path, bytes]:
    binding = _binding(admission, role)
    path = Path(PACKAGE_MOUNT_PATH) / binding.relative_path
    try:
        encoded = read_secure_regular_file(
            path,
            max_bytes=min(MAX_PACKAGE_FILE_BYTES, binding.byte_count),
            label=f"offline package {role}",
        )
    except ArtifactIntegrityError as exc:
        raise OfflineAnalysisRuntimeError(f"cannot read offline package {role}: {exc}") from exc
    if len(encoded) != binding.byte_count or sha256_bytes(encoded) != binding.file_sha256:
        raise OfflineAnalysisRuntimeError(f"offline package {role} bytes changed")
    return path, encoded


def _verify_package_membership(
    admission: OfflineAnalysisAdmission,
    admission_path: Path,
) -> None:
    expected = {
        admission_path.name,
        *(row.relative_path for row in admission.package_files),
    }
    try:
        entries = tuple(os.scandir(PACKAGE_MOUNT_PATH))
    except OSError as exc:
        raise OfflineAnalysisRuntimeError(f"cannot enumerate analysis package: {exc}") from exc
    observed: set[str] = set()
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise OfflineAnalysisRuntimeError(
                f"cannot inspect package member {entry.name}: {exc}"
            ) from exc
        if (
            entry.name in {"", ".", ".."}
            or entry.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise OfflineAnalysisRuntimeError(
                f"package member {entry.name!r} is not one regular file"
            )
        observed.add(entry.name)
    if observed != expected:
        raise OfflineAnalysisRuntimeError("offline package membership differs from admission")


def _assert_package_unchanged(
    admission: OfflineAnalysisAdmission,
    admission_path: Path,
) -> None:
    _verify_package_membership(admission, admission_path)
    try:
        observed_admission = load_offline_analysis_admission(admission_path)
    except OfflineAnalysisContractError as exc:
        raise OfflineAnalysisRuntimeError(f"cannot revalidate offline admission: {exc}") from exc
    if observed_admission != admission:
        raise OfflineAnalysisRuntimeError("offline admission changed during scientific execution")
    for binding in admission.package_files:
        _read_package_file(admission, binding.role)


def _verify_materialization_receipt(
    admission: OfflineAnalysisAdmission,
    encoded: bytes,
) -> None:
    payload = _closed(
        decode_canonical_object(encoded, label="input materialization receipt"),
        _MATERIALIZATION_RECEIPT_FIELDS,
        label="input materialization receipt",
    )
    if encoded != canonical_bytes(payload) + b"\n":
        raise OfflineAnalysisRuntimeError("input materialization receipt bytes are not canonical")
    members = payload["members"]
    if not isinstance(members, list):
        raise OfflineAnalysisRuntimeError("materialization receipt members are not an array")
    observed_members: list[dict[str, object]] = []
    for item in members:
        row = _closed(
            item,
            _MATERIALIZATION_MEMBER_FIELDS,
            label="materialization receipt member",
        )
        observed_members.append(
            {
                "byte_count": row["byte_count"],
                "corpus_id": row["corpus_id"],
                "file_sha256": row["file_sha256"],
                "role": row["role"],
                "semantic_sha256": row["semantic_sha256"],
                "uri": row["uri"],
            }
        )
    expected_members = [
        {
            "byte_count": row.byte_count,
            "corpus_id": row.corpus_id,
            "file_sha256": row.file_sha256,
            "role": row.role,
            "semantic_sha256": row.semantic_sha256,
            "uri": row.source_uri,
        }
        for row in admission.evidence
    ]
    exact = {
        "artifact_byte_count": admission.confirmatory_input_artifact_byte_count,
        "artifact_file_sha256": admission.confirmatory_input_artifact_file_sha256,
        "artifact_sha256": admission.confirmatory_input_artifact_sha256,
        "manifest_sha256": admission.manifest_sha256,
        "run_receipt_sha256": admission.run_receipt_sha256,
        "suite_attempt_id": admission.suite_attempt_id,
    }
    if (
        sha256_bytes(canonical_bytes(payload)) != admission.confirmatory_input_receipt_sha256
        or any(payload.get(name) != value for name, value in exact.items())
        or observed_members != expected_members
    ):
        raise OfflineAnalysisRuntimeError("materialization receipt differs from offline admission")


def _load_scientific_inputs(
    admission: OfflineAnalysisAdmission,
) -> tuple[ConfirmatoryInputArtifact, FrozenModelSuite]:
    bundle_path, _ = _read_package_file(admission, "offline-input-bundle")
    bundle = load_offline_input_bundle(bundle_path)
    inputs = bundle.to_confirmatory_input()
    _, summary_bytes = _read_package_file(admission, "confirmatory-input")
    _, receipt_bytes = _read_package_file(admission, "confirmatory-input-receipt")
    _, h1_bytes = _read_package_file(admission, "h1-predictive-model")
    _, h2_bytes = _read_package_file(admission, "h2-model-suite")
    _verify_materialization_receipt(admission, receipt_bytes)
    if (
        bundle.bundle_sha256 != admission.offline_input_bundle_sha256
        or inputs.manifest_sha256 != admission.manifest_sha256
        or inputs.run_receipt_sha256 != admission.run_receipt_sha256
        or inputs.artifact_sha256 != admission.confirmatory_input_artifact_sha256
        or summary_bytes != inputs.canonical_bytes() + b"\n"
    ):
        raise OfflineAnalysisRuntimeError("deserialized confirmatory input differs from admission")
    try:
        suite = FrozenModelSuite.from_json(h2_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ArtifactIntegrityError, ValueError) as exc:
        raise OfflineAnalysisRuntimeError(f"cannot load frozen model suite: {exc}") from exc
    if (
        suite.suite_digest != admission.model_suite_sha256
        or h1_bytes != canonical_h1_model_artifact_bytes(suite)
        or h2_bytes != canonical_h2_model_suite_artifact_bytes(suite)
    ):
        raise OfflineAnalysisRuntimeError("model files differ from offline admission")
    try:
        inputs.assert_model_suite_admitted(suite)
    except ConfirmatoryAnalysisError as exc:
        raise OfflineAnalysisRuntimeError(f"model admission failed: {exc}") from exc
    return inputs, suite


def _output_path(value: str, *, expected_parent: Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.parent != expected_parent:
        raise OfflineAnalysisRuntimeError("container output path differs from admission")
    return path


def _assert_initial_results_store(
    admission: OfflineAnalysisAdmission,
) -> None:
    expected = {
        _binding(admission, "confirmatory-input").relative_path,
        _binding(admission, "confirmatory-input-receipt").relative_path,
    }
    try:
        observed = {row.name for row in Path(RESULTS_MOUNT_PATH).iterdir()}
    except OSError as exc:
        raise OfflineAnalysisRuntimeError(f"cannot enumerate results store: {exc}") from exc
    if observed != expected:
        raise OfflineAnalysisRuntimeError(
            "results store is not the two-file materialized input closure"
        )
    _assert_materialized_input_files(admission)


def _assert_materialized_input_files(
    admission: OfflineAnalysisAdmission,
) -> None:
    for role in ("confirmatory-input", "confirmatory-input-receipt"):
        binding = _binding(admission, role)
        package_bytes = _read_package_file(admission, role)[1]
        output_path = Path(RESULTS_MOUNT_PATH) / binding.relative_path
        try:
            output_bytes = read_secure_regular_file(
                output_path,
                max_bytes=binding.byte_count,
                label=f"results-store {role}",
            )
        except ArtifactIntegrityError as exc:
            raise OfflineAnalysisRuntimeError(f"cannot verify results-store {role}: {exc}") from exc
        if output_bytes != package_bytes:
            raise OfflineAnalysisRuntimeError(
                f"results-store {role} differs from read-only package copy"
            )


def _assert_result_binding(
    result: ConfirmatoryResultArtifact,
    *,
    attempt: ConfirmatoryAnalysisAttemptReceipt,
) -> None:
    if not isinstance(result, ConfirmatoryResultArtifact):
        raise OfflineAnalysisRuntimeError("analysis runner returned an untyped confirmatory result")
    expected = {
        "manifest_sha256": attempt.manifest_sha256,
        "run_receipt_sha256": attempt.run_receipt_sha256,
        "confirmatory_input_artifact_sha256": (attempt.confirmatory_input_artifact_sha256),
        "model_suite_sha256": attempt.model_suite_sha256,
    }
    if any(getattr(result, name) != value for name, value in expected.items()):
        raise OfflineAnalysisRuntimeError("confirmatory result differs from the admitted attempt")


def execute_offline_analysis_once(
    admission_path: str | Path,
) -> OfflineRuntimeOutcome:
    """Consume the admission once, durably reserving attempt before computation."""

    try:
        admission = load_offline_analysis_admission(admission_path)
    except OfflineAnalysisContractError as exc:
        raise OfflineAnalysisRuntimeError(f"cannot load offline admission: {exc}") from exc
    validate_runtime_observation(
        admission,
        capture_runtime_observation(),
    )
    path = Path(admission_path)
    _assert_package_unchanged(admission, path)
    _assert_initial_results_store(admission)
    inputs, suite = _load_scientific_inputs(admission)

    attempt = ConfirmatoryAnalysisAttemptReceipt(
        manifest_sha256=admission.manifest_sha256,
        run_receipt_sha256=admission.run_receipt_sha256,
        confirmatory_input_artifact_sha256=(admission.confirmatory_input_artifact_sha256),
        model_suite_sha256=admission.model_suite_sha256,
        runner_identity=inputs.run_receipt.runner_identity,
        result_uri=admission.registered_result_uri,
    )
    if attempt.receipt_sha256 != admission.expected_attempt_receipt_sha256:
        raise OfflineAnalysisRuntimeError("admission does not commit to the reconstructed attempt")
    output_root = Path(RESULTS_MOUNT_PATH)
    attempt_path = _output_path(
        admission.container_attempt_path,
        expected_parent=output_root,
    )
    result_receipt_path = _output_path(
        admission.container_result_receipt_path,
        expected_parent=output_root,
    )
    result_path = _output_path(
        admission.container_result_path,
        expected_parent=output_root,
    )
    try:
        write_exclusive_receipt_bytes(attempt.canonical_bytes() + b"\n", attempt_path)
    except ArtifactIntegrityError as exc:
        raise OfflineAnalysisRuntimeError(
            f"offline analysis attempt was not admitted exclusively: {exc}"
        ) from exc

    result = run_confirmatory_analysis(inputs, suite=suite)
    _assert_result_binding(result, attempt=attempt)
    _assert_package_unchanged(admission, path)
    result_receipt = ConfirmatoryAnalysisResultReceipt(
        manifest_sha256=attempt.manifest_sha256,
        attempt_receipt_sha256=attempt.receipt_sha256,
        result_artifact_sha256=result.artifact_sha256,
        result_uri=attempt.result_uri,
    )
    try:
        write_exclusive_receipt_bytes(
            result_receipt.canonical_bytes() + b"\n",
            result_receipt_path,
        )
        write_exclusive_receipt_bytes(
            result.canonical_bytes() + b"\n",
            result_path,
        )
    except ArtifactIntegrityError as exc:
        raise OfflineAnalysisRuntimeError(
            f"offline analysis result custody write failed: {exc}"
        ) from exc
    _assert_materialized_input_files(admission)
    expected_store = {
        _binding(admission, "confirmatory-input").relative_path,
        _binding(admission, "confirmatory-input-receipt").relative_path,
        attempt_path.name,
        result_receipt_path.name,
        result_path.name,
    }
    if {row.name for row in output_root.iterdir()} != expected_store:
        raise OfflineAnalysisRuntimeError("results store differs from the exact five-file closure")
    return OfflineRuntimeOutcome(
        attempt_path=attempt_path,
        result_receipt_path=result_receipt_path,
        result_path=result_path,
        result_artifact_sha256=result.artifact_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fractal_ann_diagnostics.offline_analysis_runtime",
        description="Execute one admitted confirmatory analysis without network authority.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--admission", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command != "execute":  # pragma: no cover - argparse closes this path
        raise OfflineAnalysisRuntimeError("unknown offline analysis command")
    outcome = execute_offline_analysis_once(arguments.admission)
    print(
        json.dumps(
            {
                "attempt_path": str(outcome.attempt_path),
                "result_artifact_sha256": outcome.result_artifact_sha256,
                "result_path": str(outcome.result_path),
                "result_receipt_path": str(outcome.result_receipt_path),
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
