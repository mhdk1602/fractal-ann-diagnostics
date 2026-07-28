"""Provider-side preparation and launch of disconnected confirmatory analysis."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_directory_tree,
    digest_regular_file,
    read_secure_regular_file,
    write_exclusive_receipt_bytes,
)
from .confirmatory_execution import (
    ConfirmatoryAnalysisAttemptReceipt,
    confirmatory_attempt_path,
    confirmatory_output_filenames,
    confirmatory_result_path,
    confirmatory_result_receipt_path,
    load_confirmatory_analysis_attempt_receipt,
    load_confirmatory_analysis_result_receipt,
    load_confirmatory_result_artifact_bytes,
)
from .confirmatory_input_operator import (
    ConfirmatoryInputMaterializationReceipt,
    ConfirmatoryInputOperatorConfig,
    ConfirmatoryInputOperatorError,
    MaterializedConfirmatoryInput,
    load_admitted_model_suite,
    load_materialized_confirmatory_input,
    materialize_confirmatory_input,
)
from .confirmatory_modeling import (
    FrozenModelSuite,
    canonical_h1_model_artifact_bytes,
    canonical_h2_model_suite_artifact_bytes,
)
from .execution_claim import (
    ANALYSIS_PHASE,
    ProviderPhasePlan,
    VerifiedPhaseClaimCapability,
)
from .offline_analysis_contract import (
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
    OfflineAnalysisEvidenceBinding,
    OfflineAnalysisExecutionReceipt,
    OfflineAnalysisFileBinding,
    OfflineConfirmatoryInputBundle,
    canonical_bytes,
    canonical_file_uri_path,
    load_offline_analysis_execution_receipt,
    sha256_bytes,
)
from .suite_attempt import (
    PhaseClaimBindings,
    SuiteStateRecord,
    VerifiedProviderPredecessor,
    complete_confirmatory_analysis,
)

_INPUT_BUNDLE_SUFFIX = ".offline-input-bundle.json"
_H1_FILENAME = "h1-predictive-model.json"
_H2_FILENAME = "h2-model-suite.json"
_TMPFS_SPEC = "/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65532,gid=65532,mode=1777"
_CONTAINER_HOSTNAME = "fractal-analysis"
_EXECUTION_RECEIPT_SUFFIX = ".offline-analysis-execution-receipt.json"
_MAX_MODEL_BYTES = 64 * 1024 * 1024
_DOCKER_CONTROL_ENVIRONMENT: Mapping[str, str] = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


class OfflineAnalysisProviderError(ValueError):
    """Raised when live provider authority cannot admit an offline analysis."""


class DockerRun(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        check: bool,
        env: Mapping[str, str],
        stdin: int,
        stdout: int,
        stderr: int,
        shell: bool,
        close_fds: bool,
        start_new_session: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]: ...


@dataclass(frozen=True)
class PreparedOfflineAnalysis:
    """Closed host package plus the only admitted Docker invocation."""

    admission: OfflineAnalysisAdmission
    admission_path: Path
    package_root: Path
    results_root: Path
    docker_config_root: Path
    docker_invocation_executable: Path
    docker_resolved_executable: Path
    docker_executable_sha256: str
    docker_pull_argv: tuple[str, ...]
    docker_create_argv: tuple[str, ...]
    docker_start_argv: tuple[str, ...]
    docker_remove_argv: tuple[str, ...]
    docker_inspect_argv: tuple[str, ...]
    execution_receipt_path: Path
    package_tree_sha256: str
    package_entries: tuple[str, ...]
    maximum_runtime_seconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.admission, OfflineAnalysisAdmission):
            raise OfflineAnalysisProviderError("prepared analysis admission is untyped")
        for name, path in (
            ("admission_path", self.admission_path),
            ("package_root", self.package_root),
            ("results_root", self.results_root),
            ("docker_config_root", self.docker_config_root),
            ("docker_invocation_executable", self.docker_invocation_executable),
            ("docker_resolved_executable", self.docker_resolved_executable),
            ("execution_receipt_path", self.execution_receipt_path),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise OfflineAnalysisProviderError(f"{name} must be an absolute Path")
        if (
            self.admission_path.parent != self.package_root
            or self.admission_path.name != self.admission.admission_filename
            or str(self.results_root) != self.admission.host_results_store_path
            or self.execution_receipt_path
            != self.package_root.parent
            / f"{self.admission.manifest_sha256}{_EXECUTION_RECEIPT_SUFFIX}"
        ):
            raise OfflineAnalysisProviderError("prepared analysis paths differ from admission")
        _validate_docker_create_argv(
            self.docker_create_argv,
            admission=self.admission,
            admission_path=self.admission_path,
            package_root=self.package_root,
            results_root=self.results_root,
            docker_config_root=self.docker_config_root,
            docker_resolved_executable=self.docker_resolved_executable,
        )
        if self.docker_pull_argv != _docker_pull_argv(
            str(self.docker_resolved_executable),
            self.admission,
            docker_config_root=self.docker_config_root,
        ):
            raise OfflineAnalysisProviderError("prepared Docker pull differs")
        if self.docker_start_argv != _docker_start_argv(
            str(self.docker_resolved_executable),
            self.admission,
            docker_config_root=self.docker_config_root,
        ):
            raise OfflineAnalysisProviderError("prepared Docker start differs")
        if self.docker_remove_argv != _docker_remove_argv(
            str(self.docker_resolved_executable),
            self.admission,
            docker_config_root=self.docker_config_root,
        ):
            raise OfflineAnalysisProviderError("prepared Docker removal differs")
        if self.docker_inspect_argv != _docker_inspect_argv(
            str(self.docker_resolved_executable),
            self.admission,
            docker_config_root=self.docker_config_root,
        ):
            raise OfflineAnalysisProviderError("prepared Docker inspection differs")
        if len(self.docker_executable_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.docker_executable_sha256
        ):
            raise OfflineAnalysisProviderError("prepared Docker digest is malformed")
        if type(self.maximum_runtime_seconds) is not int or self.maximum_runtime_seconds <= 0:
            raise OfflineAnalysisProviderError("maximum runtime must be positive")
        entries = tuple(self.package_entries)
        if (
            len(entries) != 6
            or not all(
                type(value) is str and value and Path(value).name == value for value in entries
            )
            or entries != tuple(sorted(entries, key=lambda value: value.encode("utf-8")))
            or len(set(entries)) != len(entries)
            or self.admission_path.name not in entries
            or len(self.package_tree_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.package_tree_sha256)
        ):
            raise OfflineAnalysisProviderError("prepared package closure is malformed")


@dataclass(frozen=True)
class OfflineAnalysisOutcome:
    """Rehashed five-file store closure after the disconnected container exits."""

    attempt_path: Path
    attempt_file_sha256: str
    result_receipt_path: Path
    result_receipt_file_sha256: str
    result_path: Path
    result_file_sha256: str
    result_artifact_sha256: str
    execution_receipt_path: Path
    execution_receipt_file_sha256: str
    execution_receipt_sha256: str


@dataclass(frozen=True)
class ExecutedOfflineAnalysis:
    """Container outcome plus the fresh authority that may close suite state."""

    outcome: OfflineAnalysisOutcome
    completion_claimed: VerifiedProviderPredecessor
    completion_phase_claim: VerifiedPhaseClaimCapability


@dataclass(frozen=True)
class ProviderAnalysisCompletion:
    """Published suite candidate plus the execution evidence that justifies it."""

    candidate: SuiteStateRecord
    outcome: OfflineAnalysisOutcome

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate, SuiteStateRecord)
            or self.candidate.state != "ANALYSIS_COMPLETE"
            or not isinstance(self.outcome, OfflineAnalysisOutcome)
        ):
            raise OfflineAnalysisProviderError(
                "provider analysis completion is not a typed closed outcome"
            )


def _private_directory(path: Path, *, label: str, create: bool = False) -> None:
    if not path.is_absolute():
        raise OfflineAnalysisProviderError(f"{label} must be absolute")
    if create:
        try:
            path.mkdir(mode=0o700)
        except OSError as exc:
            raise OfflineAnalysisProviderError(f"cannot create {label}: {exc}") from exc
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise OfflineAnalysisProviderError(f"cannot inspect {label}: {exc}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise OfflineAnalysisProviderError(f"{label} must be a private runner-owned real directory")


def _directory_names(path: Path, *, label: str) -> set[str]:
    try:
        rows = tuple(path.iterdir())
    except OSError as exc:
        raise OfflineAnalysisProviderError(f"cannot enumerate {label}: {exc}") from exc
    if any(row.name in {"", ".", ".."} for row in rows):
        raise OfflineAnalysisProviderError(f"{label} has a non-canonical member")
    return {row.name for row in rows}


def _assert_path_absent(path: Path, *, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise OfflineAnalysisProviderError(f"cannot inspect {label}: {exc}") from exc
    raise OfflineAnalysisProviderError(f"{label} already exists")


def _read_bound_file(
    path: Path,
    *,
    file_sha256: str,
    byte_count: int,
    label: str,
    max_bytes: int,
) -> bytes:
    try:
        encoded = read_secure_regular_file(path, max_bytes=max_bytes, label=label)
    except ArtifactIntegrityError as exc:
        raise OfflineAnalysisProviderError(f"cannot read {label}: {exc}") from exc
    if len(encoded) != byte_count or sha256_bytes(encoded) != file_sha256:
        raise OfflineAnalysisProviderError(f"{label} bytes changed")
    return encoded


def _validate_authority(
    claimed: VerifiedProviderPredecessor,
    phase_claim: VerifiedPhaseClaimCapability,
    plan: ProviderPhasePlan,
) -> PhaseClaimBindings:
    if (
        not isinstance(claimed, VerifiedProviderPredecessor)
        or claimed.state.state != "ANALYSIS_CLAIMED"
    ):
        raise OfflineAnalysisProviderError("offline analysis requires verified ANALYSIS_CLAIMED")
    if not isinstance(phase_claim, VerifiedPhaseClaimCapability):
        raise OfflineAnalysisProviderError("offline analysis lacks typed phase authority")
    if not isinstance(plan, ProviderPhasePlan) or plan.phase != ANALYSIS_PHASE:
        raise OfflineAnalysisProviderError("offline analysis lacks the C1 analysis plan")
    claimed.assert_current()
    phase_claim.assert_current()
    state = claimed.state
    payload = state.payload
    if not isinstance(payload, PhaseClaimBindings):
        raise OfflineAnalysisProviderError("ANALYSIS_CLAIMED payload is malformed")
    contract = payload.phase_claim
    exact_authority = (
        phase_claim.contract == contract,
        phase_claim.provider_identity == payload.provider_identity,
        phase_claim.phase_claim_state_sha256 == state.record_sha256,
        phase_claim.phase_claim_ledger_commit == claimed.ledger_commit,
        contract.phase == ANALYSIS_PHASE,
        contract.manifest_sha256 == state.manifest_sha256,
        contract.run_receipt_sha256 == state.run_receipt_sha256,
        contract.c1_commit == plan.c1_commit,
        contract.manifest_sha256 == plan.manifest_sha256,
        contract.c1_provider_plan_uri == Path(plan.provider_plan_path).as_uri(),
        contract.c1_provider_plan_sha256 == plan.plan_sha256,
        contract.oci_index_digest == plan.oci_index_digest,
        contract.oci_platform_manifest_digest == plan.oci_platform_manifest_digest,
        contract.host_tool_contract_sha256 == plan.host_tools.contract_sha256,
        contract.runtime_probe_receipt_sha256 == plan.runtime_probe_receipt_sha256,
        plan.runtime_platform == RUNTIME_PLATFORM,
        plan.runtime_image_role == "scientific",
        plan.runtime_index_role == "main",
    )
    if not all(exact_authority):
        raise OfflineAnalysisProviderError(
            "live analysis authority differs from the C1 scientific runtime"
        )
    for row in contract.corpora:
        phase_claim.require_input(
            corpus_id=row.corpus_id,
            input_uri=row.input_uri,
            input_sha256=row.input_sha256,
            supporting_input_uri=row.supporting_input_uri,
            supporting_input_sha256=row.supporting_input_sha256,
        )
    claimed.assert_current()
    phase_claim.assert_current()
    return payload


def _verify_plan_file(plan: ProviderPhasePlan) -> None:
    path = Path(plan.provider_plan_path)
    try:
        observed = digest_regular_file(path, label="C1 analysis provider plan")
    except ArtifactIntegrityError as exc:
        raise OfflineAnalysisProviderError(f"cannot verify the C1 provider plan: {exc}") from exc
    if observed != plan.file_sha256:
        raise OfflineAnalysisProviderError("C1 provider plan file changed")


def _evidence_bindings(
    receipt: ConfirmatoryInputMaterializationReceipt,
) -> tuple[OfflineAnalysisEvidenceBinding, ...]:
    rows: list[OfflineAnalysisEvidenceBinding] = []
    for member in receipt.members:
        path = canonical_file_uri_path(member.uri, label=f"{member.role} source URI")
        try:
            metadata = path.stat(follow_symlinks=False)
            observed = digest_regular_file(path, label=f"{member.role} materialization member")
        except (OSError, ArtifactIntegrityError) as exc:
            raise OfflineAnalysisProviderError(
                f"cannot revalidate materialization member {member.role}: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or path.is_symlink()
            or metadata.st_nlink != 1
            or metadata.st_size != member.byte_count
            or observed != member.file_sha256
        ):
            raise OfflineAnalysisProviderError(
                f"materialization member {member.role} changed before admission"
            )
        rows.append(
            OfflineAnalysisEvidenceBinding(
                role=member.role,
                corpus_id=member.corpus_id,
                source_uri=member.uri,
                semantic_sha256=member.semantic_sha256,
                file_sha256=member.file_sha256,
                byte_count=member.byte_count,
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                (row.corpus_id or "").encode("utf-8"),
                row.role.encode("utf-8"),
                row.source_uri.encode("utf-8"),
            ),
        )
    )


def _package_binding(
    *,
    role: str,
    relative_path: str,
    semantic_sha256: str,
    encoded: bytes,
) -> OfflineAnalysisFileBinding:
    return OfflineAnalysisFileBinding(
        role=role,
        relative_path=relative_path,
        semantic_sha256=semantic_sha256,
        file_sha256=sha256_bytes(encoded),
        byte_count=len(encoded),
    )


def _results_store(inputs: Any) -> Path:
    sealed = inputs.frozen_manifest.get("sealed_execution")
    if not isinstance(sealed, Mapping) or "results_store" not in sealed:
        raise OfflineAnalysisProviderError("frozen manifest lacks its results store")
    return canonical_file_uri_path(
        sealed["results_store"],
        label="sealed_execution.results_store",
    )


def _admission(
    materialized: MaterializedConfirmatoryInput,
    suite: FrozenModelSuite,
    plan: ProviderPhasePlan,
    claimed: VerifiedProviderPredecessor,
    phase_claim: VerifiedPhaseClaimCapability,
    *,
    results_root: Path,
    package_bindings: tuple[OfflineAnalysisFileBinding, ...],
    evidence: tuple[OfflineAnalysisEvidenceBinding, ...],
    input_bundle: OfflineConfirmatoryInputBundle,
) -> OfflineAnalysisAdmission:
    inputs = materialized.inputs
    contract = phase_claim.contract
    result_path = confirmatory_result_path(inputs)
    result_receipt_path = confirmatory_result_receipt_path(inputs)
    attempt_path = confirmatory_attempt_path(inputs)
    expected_attempt = ConfirmatoryAnalysisAttemptReceipt(
        manifest_sha256=inputs.manifest_sha256,
        run_receipt_sha256=inputs.run_receipt_sha256,
        confirmatory_input_artifact_sha256=inputs.artifact_sha256,
        model_suite_sha256=suite.suite_digest,
        runner_identity=inputs.run_receipt.runner_identity,
        result_uri=result_path.as_uri(),
    )
    input_binding = next(row for row in package_bindings if row.role == "confirmatory-input")
    receipt_binding = next(
        row for row in package_bindings if row.role == "confirmatory-input-receipt"
    )
    return OfflineAnalysisAdmission(
        suite_attempt_id=claimed.state.suite_attempt_id,
        provider_state_record_sha256=claimed.state.record_sha256,
        provider_ledger_commit=claimed.ledger_commit,
        provider_control_inventory_sha256=claimed.control_inventory_sha256,
        provider_artifact_receipt_sha256=claimed.artifact_receipt_sha256,
        phase_claim_contract_sha256=contract.contract_sha256,
        phase_claim_state_sha256=phase_claim.phase_claim_state_sha256,
        phase_claim_ledger_commit=phase_claim.phase_claim_ledger_commit,
        provider_identity_sha256=phase_claim.provider_identity.identity_sha256,
        live_execute_job_receipt_sha256=(phase_claim.live_execute_job_receipt.receipt_sha256),
        claim_attested_at_utc=phase_claim.claim_attested_at_utc,
        c1_commit=plan.c1_commit,
        c1_provider_plan_uri=Path(plan.provider_plan_path).as_uri(),
        c1_provider_plan_sha256=plan.plan_sha256,
        c1_provider_plan_file_sha256=plan.file_sha256,
        runtime_image=plan.runtime_image,
        runtime_platform=plan.runtime_platform,
        runtime_image_role=plan.runtime_image_role,
        runtime_index_role=plan.runtime_index_role,
        oci_index_digest=plan.oci_index_digest,
        oci_platform_manifest_digest=plan.oci_platform_manifest_digest,
        host_tool_contract_sha256=plan.host_tools.contract_sha256,
        runtime_probe_receipt_sha256=plan.runtime_probe_receipt_sha256,
        manifest_sha256=inputs.manifest_sha256,
        run_receipt_sha256=inputs.run_receipt_sha256,
        confirmatory_input_artifact_sha256=materialized.receipt.artifact_sha256,
        confirmatory_input_artifact_file_sha256=input_binding.file_sha256,
        confirmatory_input_artifact_byte_count=input_binding.byte_count,
        confirmatory_input_receipt_sha256=materialized.receipt.receipt_sha256,
        confirmatory_input_receipt_file_sha256=receipt_binding.file_sha256,
        confirmatory_input_receipt_byte_count=receipt_binding.byte_count,
        offline_input_bundle_sha256=input_bundle.bundle_sha256,
        model_suite_sha256=suite.suite_digest,
        registered_results_store_uri=results_root.as_uri(),
        host_results_store_path=str(results_root),
        package_mount_path=PACKAGE_MOUNT_PATH,
        results_mount_path=RESULTS_MOUNT_PATH,
        tmpfs_mount_path=TMPFS_MOUNT_PATH,
        network_mode=NETWORK_MODE,
        root_filesystem_read_only=True,
        package_mount_read_only=True,
        results_mount_read_write=True,
        runtime_machine=RUNTIME_MACHINE,
        runtime_uid=RUNTIME_UID,
        runtime_gid=RUNTIME_GID,
        runtime_environment=RUNTIME_ENVIRONMENT,
        runtime_dynamic_environment_names=RUNTIME_DYNAMIC_ENVIRONMENT_NAMES,
        container_name=f"fractal-analysis-{claimed.state.suite_attempt_id}",
        registered_attempt_uri=attempt_path.as_uri(),
        registered_result_receipt_uri=result_receipt_path.as_uri(),
        registered_result_uri=result_path.as_uri(),
        container_attempt_path=f"{RESULTS_MOUNT_PATH}/{attempt_path.name}",
        container_result_receipt_path=(f"{RESULTS_MOUNT_PATH}/{result_receipt_path.name}"),
        container_result_path=f"{RESULTS_MOUNT_PATH}/{result_path.name}",
        expected_attempt_receipt_sha256=expected_attempt.receipt_sha256,
        evidence=evidence,
        package_files=package_bindings,
    )


def _mount_argument(source: Path, destination: str, *, read_only: bool) -> str:
    if "," in str(source):
        raise OfflineAnalysisProviderError("Docker bind source cannot contain a comma")
    options = [
        "type=bind",
        f"src={source}",
        f"dst={destination}",
    ]
    if read_only:
        options.append("readonly")
    return ",".join(options)


def _platform_runtime_image(admission: OfflineAnalysisAdmission) -> str:
    repository = admission.runtime_image.rsplit("@", 1)[0]
    return f"{repository}@{admission.oci_platform_manifest_digest}"


def _docker_pull_argv(
    docker_executable: str,
    admission: OfflineAnalysisAdmission,
    *,
    docker_config_root: Path,
) -> tuple[str, ...]:
    return (
        docker_executable,
        "--config",
        str(docker_config_root),
        "pull",
        "--platform",
        RUNTIME_PLATFORM,
        _platform_runtime_image(admission),
    )


def _docker_create_argv(
    plan: ProviderPhasePlan,
    admission: OfflineAnalysisAdmission,
    *,
    admission_path: Path,
    package_root: Path,
    results_root: Path,
    docker_config_root: Path,
) -> tuple[str, ...]:
    return _expected_docker_create_argv(
        plan.host_tools.docker_resolved_executable,
        admission,
        admission_path=admission_path,
        package_root=package_root,
        results_root=results_root,
        docker_config_root=docker_config_root,
    )


def _expected_docker_create_argv(
    docker_resolved_executable: str | Path,
    admission: OfflineAnalysisAdmission,
    *,
    admission_path: Path,
    package_root: Path,
    results_root: Path,
    docker_config_root: Path,
) -> tuple[str, ...]:
    return (
        str(docker_resolved_executable),
        "--config",
        str(docker_config_root),
        "create",
        "--name",
        admission.container_name,
        "--rm",
        "--log-driver",
        "none",
        "--pull",
        "never",
        "--platform",
        RUNTIME_PLATFORM,
        "--network",
        NETWORK_MODE,
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{RUNTIME_UID}:{RUNTIME_GID}",
        "--pids-limit",
        "256",
        "--hostname",
        _CONTAINER_HOSTNAME,
        "--ipc",
        "none",
        "--tmpfs",
        _TMPFS_SPEC,
        "--mount",
        _mount_argument(package_root, PACKAGE_MOUNT_PATH, read_only=True),
        "--mount",
        _mount_argument(results_root, RESULTS_MOUNT_PATH, read_only=False),
        "--workdir",
        "/workspace",
        "--entrypoint",
        "/opt/venv/bin/python",
        _platform_runtime_image(admission),
        "-P",
        "-m",
        "fractal_ann_diagnostics.offline_analysis_runtime",
        "execute",
        "--admission",
        f"{PACKAGE_MOUNT_PATH}/{admission_path.name}",
    )


def _docker_start_argv(
    docker_executable: str,
    admission: OfflineAnalysisAdmission,
    *,
    docker_config_root: Path,
) -> tuple[str, ...]:
    return (
        docker_executable,
        "--config",
        str(docker_config_root),
        "start",
        "--attach",
        admission.container_name,
    )


def _docker_remove_argv(
    docker_executable: str,
    admission: OfflineAnalysisAdmission,
    *,
    docker_config_root: Path,
) -> tuple[str, ...]:
    return (
        docker_executable,
        "--config",
        str(docker_config_root),
        "container",
        "rm",
        "--force",
        "--volumes",
        admission.container_name,
    )


def _docker_inspect_argv(
    docker_executable: str,
    admission: OfflineAnalysisAdmission,
    *,
    docker_config_root: Path,
) -> tuple[str, ...]:
    return (
        docker_executable,
        "--config",
        str(docker_config_root),
        "container",
        "inspect",
        admission.container_name,
    )


def _docker_version_argv(
    docker_executable: str,
    *,
    docker_config_root: Path,
) -> tuple[str, ...]:
    return (
        docker_executable,
        "--config",
        str(docker_config_root),
        "version",
        "--format={{.Server.ID}}",
    )


def _validate_docker_create_argv(
    argv: Sequence[str],
    *,
    admission: OfflineAnalysisAdmission,
    admission_path: Path,
    package_root: Path,
    results_root: Path,
    docker_config_root: Path,
    docker_resolved_executable: Path,
) -> None:
    values = tuple(argv)
    expected = _expected_docker_create_argv(
        docker_resolved_executable,
        admission,
        admission_path=admission_path,
        package_root=package_root,
        results_root=results_root,
        docker_config_root=docker_config_root,
    )
    if values != expected:
        raise OfflineAnalysisProviderError("Docker analysis create invocation is not closed")


def prepare_offline_analysis(
    materialized: MaterializedConfirmatoryInput,
    suite: FrozenModelSuite,
    plan: ProviderPhasePlan,
    claimed: VerifiedProviderPredecessor,
    phase_claim: VerifiedPhaseClaimCapability,
    *,
    package_root: str | Path,
    results_root: str | Path,
    recover_existing: bool = False,
) -> PreparedOfflineAnalysis:
    """Create the closed package only while both live authorities are fresh."""

    if not isinstance(materialized, MaterializedConfirmatoryInput):
        raise OfflineAnalysisProviderError("analysis input was not materially verified")
    if not isinstance(suite, FrozenModelSuite):
        raise OfflineAnalysisProviderError("analysis model suite is untyped")
    _validate_authority(claimed, phase_claim, plan)
    _verify_plan_file(plan)
    inputs = materialized.inputs
    if (
        inputs.manifest_sha256 != claimed.state.manifest_sha256
        or inputs.run_receipt_sha256 != claimed.state.run_receipt_sha256
        or materialized.receipt.artifact_sha256 != inputs.artifact_sha256
    ):
        raise OfflineAnalysisProviderError(
            "materialized analysis input differs from provider authority"
        )
    inputs.assert_model_suite_admitted(suite)

    package = Path(package_root)
    results = Path(results_root)
    package_exists = recover_existing and _path_present(
        package,
        label="offline analysis package",
    )
    _private_directory(
        package,
        label="offline analysis package",
        create=not package_exists,
    )
    docker_config = package.with_name(f"{package.name}.docker-config")
    docker_config_exists = recover_existing and _path_present(
        docker_config,
        label="anonymous Docker configuration",
    )
    _private_directory(
        docker_config,
        label="anonymous Docker configuration",
        create=not docker_config_exists,
    )
    _private_directory(results, label="registered analysis results store")
    package_names = _directory_names(package, label="offline analysis package")
    if package_names and not recover_existing:
        raise OfflineAnalysisProviderError("offline analysis package is not empty")
    if _directory_names(docker_config, label="anonymous Docker configuration"):
        raise OfflineAnalysisProviderError("anonymous Docker configuration is not empty")
    registered_results = _results_store(inputs)
    if results != registered_results:
        raise OfflineAnalysisProviderError(
            "analysis output root differs from the registered results store"
        )
    expected_store = {
        materialized.artifact_path.name,
        materialized.receipt_path.name,
    }
    completed_store = expected_store | set(confirmatory_output_filenames(inputs.manifest_sha256))
    observed_store = _directory_names(
        results,
        label="registered analysis results store",
    )
    if (
        materialized.artifact_path.parent != results
        or materialized.receipt_path.parent != results
        or (
            observed_store != expected_store
            and (not recover_existing or observed_store != completed_store)
        )
    ):
        raise OfflineAnalysisProviderError(
            "results store differs from a recoverable analysis closure"
        )

    input_bytes = _read_bound_file(
        materialized.artifact_path,
        file_sha256=materialized.receipt.artifact_file_sha256,
        byte_count=materialized.receipt.artifact_byte_count,
        label="materialized confirmatory input",
        max_bytes=materialized.receipt.artifact_byte_count,
    )
    receipt_bytes = materialized.receipt.canonical_bytes() + b"\n"
    observed_receipt_bytes = _read_bound_file(
        materialized.receipt_path,
        file_sha256=sha256_bytes(receipt_bytes),
        byte_count=len(receipt_bytes),
        label="confirmatory input materialization receipt",
        max_bytes=len(receipt_bytes),
    )
    if observed_receipt_bytes != receipt_bytes:
        raise OfflineAnalysisProviderError("materialization receipt bytes differ")

    input_bundle = OfflineConfirmatoryInputBundle.from_confirmatory_input(inputs)
    bundle_bytes = input_bundle.canonical_bytes() + b"\n"
    h1_bytes = canonical_h1_model_artifact_bytes(suite)
    h2_bytes = canonical_h2_model_suite_artifact_bytes(suite)
    payloads = {
        materialized.artifact_path.name: (
            "confirmatory-input",
            inputs.artifact_sha256,
            input_bytes,
        ),
        materialized.receipt_path.name: (
            "confirmatory-input-receipt",
            materialized.receipt.receipt_sha256,
            receipt_bytes,
        ),
        f"{inputs.manifest_sha256}{_INPUT_BUNDLE_SUFFIX}": (
            "offline-input-bundle",
            input_bundle.bundle_sha256,
            bundle_bytes,
        ),
        _H1_FILENAME: (
            "h1-predictive-model",
            sha256_bytes(h1_bytes),
            h1_bytes,
        ),
        _H2_FILENAME: (
            "h2-model-suite",
            suite.suite_digest,
            h2_bytes,
        ),
    }
    bindings = tuple(
        sorted(
            (
                _package_binding(
                    role=role,
                    relative_path=filename,
                    semantic_sha256=semantic_sha256,
                    encoded=encoded,
                )
                for filename, (role, semantic_sha256, encoded) in payloads.items()
            ),
            key=lambda row: row.relative_path.encode("utf-8"),
        )
    )
    evidence = _evidence_bindings(materialized.receipt)
    admission = _admission(
        materialized,
        suite,
        plan,
        claimed,
        phase_claim,
        results_root=results,
        package_bindings=bindings,
        evidence=evidence,
        input_bundle=input_bundle,
    )
    admission_path = package / admission.admission_filename
    expected_package_names = {
        admission.admission_filename,
        *payloads,
    }
    if package_names:
        if package_names != expected_package_names:
            raise OfflineAnalysisProviderError(
                "recoverable offline package differs from the exact six-file closure"
            )
        for filename, (_, _, encoded) in payloads.items():
            observed = _read_bound_file(
                package / filename,
                file_sha256=sha256_bytes(encoded),
                byte_count=len(encoded),
                label=f"recoverable offline package member {filename}",
                max_bytes=len(encoded),
            )
            if observed != encoded:
                raise OfflineAnalysisProviderError(
                    f"recoverable offline package member {filename} differs"
                )
        admission_bytes = admission.canonical_bytes() + b"\n"
        observed_admission = _read_bound_file(
            admission_path,
            file_sha256=sha256_bytes(admission_bytes),
            byte_count=len(admission_bytes),
            label="recoverable offline analysis admission",
            max_bytes=len(admission_bytes),
        )
        if observed_admission != admission_bytes:
            raise OfflineAnalysisProviderError("recoverable offline analysis admission differs")
    else:
        for filename in sorted(payloads, key=lambda value: value.encode("utf-8")):
            try:
                write_exclusive_receipt_bytes(payloads[filename][2], package / filename)
            except ArtifactIntegrityError as exc:
                raise OfflineAnalysisProviderError(
                    f"cannot write offline package member {filename}: {exc}"
                ) from exc
        claimed.assert_current()
        phase_claim.assert_current()
        try:
            write_exclusive_receipt_bytes(
                admission.canonical_bytes() + b"\n",
                admission_path,
            )
        except ArtifactIntegrityError as exc:
            raise OfflineAnalysisProviderError(
                f"cannot persist offline analysis admission: {exc}"
            ) from exc
    claimed.assert_current()
    phase_claim.assert_current()
    execution_receipt_path = (
        package.parent / f"{admission.manifest_sha256}{_EXECUTION_RECEIPT_SUFFIX}"
    )
    if not recover_existing:
        _assert_path_absent(
            execution_receipt_path,
            label="offline analysis execution receipt",
        )
    try:
        package_inventory = digest_directory_tree(package)
    except ArtifactIntegrityError as exc:
        raise OfflineAnalysisProviderError(
            f"cannot close the offline analysis package: {exc}"
        ) from exc
    expected_package_entries = tuple(
        sorted(
            {
                admission.admission_filename,
                *(row.relative_path for row in admission.package_files),
            },
            key=lambda value: value.encode("utf-8"),
        )
    )
    if package_inventory.entries != expected_package_entries:
        raise OfflineAnalysisProviderError(
            "offline analysis package differs from the exact six-file closure"
        )
    docker_create_argv = _docker_create_argv(
        plan,
        admission,
        admission_path=admission_path,
        package_root=package,
        results_root=results,
        docker_config_root=docker_config,
    )
    return PreparedOfflineAnalysis(
        admission=admission,
        admission_path=admission_path,
        package_root=package,
        results_root=results,
        docker_config_root=docker_config,
        docker_invocation_executable=Path(plan.host_tools.docker_executable),
        docker_resolved_executable=Path(plan.host_tools.docker_resolved_executable),
        docker_executable_sha256=plan.host_tools.docker_executable_sha256,
        docker_pull_argv=_docker_pull_argv(
            plan.host_tools.docker_resolved_executable,
            admission,
            docker_config_root=docker_config,
        ),
        docker_create_argv=docker_create_argv,
        docker_start_argv=_docker_start_argv(
            plan.host_tools.docker_resolved_executable,
            admission,
            docker_config_root=docker_config,
        ),
        docker_remove_argv=_docker_remove_argv(
            plan.host_tools.docker_resolved_executable,
            admission,
            docker_config_root=docker_config,
        ),
        docker_inspect_argv=_docker_inspect_argv(
            plan.host_tools.docker_resolved_executable,
            admission,
            docker_config_root=docker_config,
        ),
        execution_receipt_path=execution_receipt_path,
        package_tree_sha256=package_inventory.sha256,
        package_entries=package_inventory.entries,
        maximum_runtime_seconds=plan.maximum_runtime_seconds,
    )


def _hash_followed_regular_file(path: Path, *, label: str) -> str:
    try:
        before_path = path.resolve(strict=True)
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError as exc:
        raise OfflineAnalysisProviderError(f"cannot open {label}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OfflineAnalysisProviderError(f"{label} target is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.resolve(strict=True)
    except OSError as exc:
        raise OfflineAnalysisProviderError(f"cannot re-resolve {label}: {exc}") from exc
    signatures = (
        (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ),
        (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
    )
    if before_path != after_path or signatures[0] != signatures[1]:
        raise OfflineAnalysisProviderError(f"{label} changed during verification")
    return digest.hexdigest()


def _verify_docker_tool(prepared: PreparedOfflineAnalysis) -> None:
    source = prepared.docker_invocation_executable
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise OfflineAnalysisProviderError(f"cannot resolve C1 Docker client: {exc}") from exc
    if resolved != prepared.docker_resolved_executable:
        raise OfflineAnalysisProviderError("Docker client symlink resolves outside C1")
    source_digest = _hash_followed_regular_file(source, label="C1 Docker client")
    resolved_digest = _hash_followed_regular_file(
        prepared.docker_resolved_executable,
        label="C1 resolved Docker client",
    )
    if (
        source_digest != prepared.docker_executable_sha256
        or resolved_digest != prepared.docker_executable_sha256
    ):
        raise OfflineAnalysisProviderError("Docker client bytes changed after C1")


def _run_docker_bounded(
    docker_run: DockerRun,
    argv: tuple[str, ...],
    *,
    timeout: int,
    label: str,
    accepted_returncodes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[bytes]:
    if not accepted_returncodes or not all(
        type(value) is int and value >= 0 for value in accepted_returncodes
    ):
        raise OfflineAnalysisProviderError(f"{label} has an invalid return-code contract")
    try:
        completed = docker_run(
            argv,
            check=False,
            env=dict(_DOCKER_CONTROL_ENVIRONMENT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
            start_new_session=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OfflineAnalysisProviderError(f"{label} failed to start: {exc}") from exc
    if completed.returncode not in accepted_returncodes:
        raise OfflineAnalysisProviderError(f"{label} exited with status {completed.returncode}")
    return completed


def _assert_container_absent(
    prepared: PreparedOfflineAnalysis,
    *,
    docker_run: DockerRun,
) -> None:
    _run_docker_bounded(
        docker_run,
        _docker_version_argv(
            str(prepared.docker_resolved_executable),
            docker_config_root=prepared.docker_config_root,
        ),
        timeout=30,
        label="Docker daemon liveness proof",
    )
    deadline = time.monotonic() + 5
    while True:
        inspected = _run_docker_bounded(
            docker_run,
            prepared.docker_inspect_argv,
            timeout=30,
            label="scientific container absence proof",
            accepted_returncodes=frozenset({0, 1}),
        )
        if inspected.returncode == 1:
            return
        if time.monotonic() >= deadline:
            raise OfflineAnalysisProviderError(
                "scientific container is still present in the Docker daemon"
            )
        time.sleep(0.1)


def _force_remove_container(
    prepared: PreparedOfflineAnalysis,
    *,
    docker_run: DockerRun,
) -> None:
    _run_docker_bounded(
        docker_run,
        prepared.docker_remove_argv,
        timeout=30,
        label="scientific container force-removal",
        accepted_returncodes=frozenset({0, 1}),
    )
    _assert_container_absent(prepared, docker_run=docker_run)


def _unchanged_package_tree(prepared: PreparedOfflineAnalysis) -> str:
    try:
        inventory = digest_directory_tree(prepared.package_root)
    except ArtifactIntegrityError as exc:
        raise OfflineAnalysisProviderError(
            f"cannot rehash the offline analysis package: {exc}"
        ) from exc
    if (
        inventory.sha256 != prepared.package_tree_sha256
        or inventory.entries != prepared.package_entries
    ):
        raise OfflineAnalysisProviderError("offline analysis package changed after admission")
    try:
        encoded = read_secure_regular_file(
            prepared.admission_path,
            max_bytes=len(prepared.admission.canonical_bytes()) + 1,
            label="offline analysis admission",
        )
    except ArtifactIntegrityError as exc:
        raise OfflineAnalysisProviderError(
            f"cannot rehash the offline analysis admission: {exc}"
        ) from exc
    if encoded != prepared.admission.canonical_bytes() + b"\n":
        raise OfflineAnalysisProviderError("offline analysis admission changed after preparation")
    return inventory.sha256


def _argv_sha256(argv: tuple[str, ...]) -> str:
    return sha256_bytes(canonical_bytes({"argv": list(argv)}))


def _path_present(path: Path, *, label: str) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise OfflineAnalysisProviderError(f"cannot inspect {label}: {exc}") from exc
    return True


def _close_completed_offline_analysis(
    prepared: PreparedOfflineAnalysis,
    claimed: VerifiedProviderPredecessor,
    phase_claim: VerifiedPhaseClaimCapability,
    plan: ProviderPhasePlan,
    *,
    fresh_claim_supplier: Callable[
        [], tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]
    ],
    docker_run: DockerRun,
) -> ExecutedOfflineAnalysis:
    """Rehash a completed five-file outcome and mint or recover its host receipt."""

    _assert_container_absent(prepared, docker_run=docker_run)
    completion_claimed, completion_phase_claim = _matching_completion_authority(
        claimed,
        phase_claim,
        fresh_claim_supplier,
        plan,
    )
    package_after_sha256 = _unchanged_package_tree(prepared)
    _verify_docker_tool(prepared)
    if _directory_names(
        prepared.docker_config_root,
        label="anonymous Docker configuration",
    ):
        raise OfflineAnalysisProviderError(
            "scientific container persisted Docker client configuration"
        )

    admission = prepared.admission
    attempt_path = (
        Path(admission.host_results_store_path) / Path(admission.container_attempt_path).name
    )
    receipt_path = (
        Path(admission.host_results_store_path) / Path(admission.container_result_receipt_path).name
    )
    result_path = (
        Path(admission.host_results_store_path) / Path(admission.container_result_path).name
    )
    attempt = load_confirmatory_analysis_attempt_receipt(attempt_path)
    receipt = load_confirmatory_analysis_result_receipt(receipt_path)
    result_bytes = load_confirmatory_result_artifact_bytes(
        result_path,
        result_receipt_path=receipt_path,
        attempt_receipt_path=attempt_path,
    )
    if (
        attempt.receipt_sha256 != admission.expected_attempt_receipt_sha256
        or attempt.confirmatory_input_artifact_sha256
        != admission.confirmatory_input_artifact_sha256
        or attempt.model_suite_sha256 != admission.model_suite_sha256
        or receipt.attempt_receipt_sha256 != attempt.receipt_sha256
        or receipt.result_artifact_sha256 != sha256_bytes(result_bytes)
    ):
        raise OfflineAnalysisProviderError("container outcome differs from the offline admission")
    expected_store = {
        row.relative_path
        for row in admission.package_files
        if row.role in {"confirmatory-input", "confirmatory-input-receipt"}
    } | {attempt_path.name, receipt_path.name, result_path.name}
    if (
        _directory_names(
            prepared.results_root,
            label="analysis results store",
        )
        != expected_store
    ):
        raise OfflineAnalysisProviderError(
            "analysis results store differs from the exact five-file closure"
        )
    try:
        results_inventory = digest_directory_tree(prepared.results_root)
    except ArtifactIntegrityError as exc:
        raise OfflineAnalysisProviderError(
            f"cannot close the analysis results tree: {exc}"
        ) from exc
    if set(results_inventory.entries) != expected_store:
        raise OfflineAnalysisProviderError(
            "analysis result inventory differs from the five-file closure"
        )
    attempt_file_sha256 = digest_regular_file(
        attempt_path,
        label="confirmatory analysis attempt",
    )
    receipt_file_sha256 = digest_regular_file(
        receipt_path,
        label="confirmatory result receipt",
    )
    result_file_sha256 = digest_regular_file(
        result_path,
        label="confirmatory result",
    )
    expected_execution_receipt = OfflineAnalysisExecutionReceipt(
        suite_attempt_id=admission.suite_attempt_id,
        manifest_sha256=admission.manifest_sha256,
        run_receipt_sha256=admission.run_receipt_sha256,
        provider_state_record_sha256=admission.provider_state_record_sha256,
        provider_ledger_commit=admission.provider_ledger_commit,
        phase_claim_contract_sha256=admission.phase_claim_contract_sha256,
        phase_claim_state_sha256=admission.phase_claim_state_sha256,
        phase_claim_ledger_commit=admission.phase_claim_ledger_commit,
        provider_identity_sha256=admission.provider_identity_sha256,
        c1_commit=admission.c1_commit,
        admission_uri=prepared.admission_path.as_uri(),
        admission_sha256=admission.admission_sha256,
        admission_file_sha256=digest_regular_file(
            prepared.admission_path,
            label="offline analysis admission",
        ),
        package_root_uri=prepared.package_root.as_uri(),
        package_tree_before_sha256=prepared.package_tree_sha256,
        package_tree_after_sha256=package_after_sha256,
        package_entries=prepared.package_entries,
        docker_executable_sha256=prepared.docker_executable_sha256,
        docker_pull_argv_sha256=_argv_sha256(prepared.docker_pull_argv),
        docker_create_argv_sha256=_argv_sha256(prepared.docker_create_argv),
        docker_start_argv_sha256=_argv_sha256(prepared.docker_start_argv),
        docker_remove_argv_sha256=_argv_sha256(prepared.docker_remove_argv),
        container_name=admission.container_name,
        runtime_image=admission.runtime_image,
        runtime_platform=admission.runtime_platform,
        oci_index_digest=admission.oci_index_digest,
        oci_platform_manifest_digest=admission.oci_platform_manifest_digest,
        attempt_uri=attempt_path.as_uri(),
        attempt_receipt_sha256=attempt.receipt_sha256,
        attempt_file_sha256=attempt_file_sha256,
        result_receipt_uri=receipt_path.as_uri(),
        result_receipt_sha256=receipt.receipt_sha256,
        result_receipt_file_sha256=receipt_file_sha256,
        result_uri=result_path.as_uri(),
        result_artifact_sha256=receipt.result_artifact_sha256,
        result_file_sha256=result_file_sha256,
        results_tree_sha256=results_inventory.sha256,
        results_entries=results_inventory.entries,
        completion_state_record_sha256=completion_claimed.state.record_sha256,
        completion_ledger_commit=completion_claimed.ledger_commit,
        container_absent_after_execution=True,
    )
    if _path_present(
        prepared.execution_receipt_path,
        label="offline analysis execution receipt",
    ):
        try:
            execution_receipt = load_offline_analysis_execution_receipt(
                prepared.execution_receipt_path,
            )
        except Exception as exc:
            raise OfflineAnalysisProviderError(
                f"cannot recover offline execution receipt: {exc}"
            ) from exc
        if execution_receipt != expected_execution_receipt:
            raise OfflineAnalysisProviderError(
                "retained offline execution receipt differs from recovered closure"
            )
    else:
        execution_receipt = expected_execution_receipt
        try:
            write_exclusive_receipt_bytes(
                execution_receipt.canonical_bytes() + b"\n",
                prepared.execution_receipt_path,
            )
        except ArtifactIntegrityError as exc:
            raise OfflineAnalysisProviderError(
                f"cannot persist offline execution receipt: {exc}"
            ) from exc
    execution_receipt_file_sha256 = digest_regular_file(
        prepared.execution_receipt_path,
        label="offline analysis execution receipt",
    )
    expected_execution_file_sha256 = sha256_bytes(execution_receipt.canonical_bytes() + b"\n")
    if execution_receipt_file_sha256 != expected_execution_file_sha256:
        raise OfflineAnalysisProviderError("offline execution receipt failed exact readback")
    return ExecutedOfflineAnalysis(
        outcome=OfflineAnalysisOutcome(
            attempt_path=attempt_path,
            attempt_file_sha256=attempt_file_sha256,
            result_receipt_path=receipt_path,
            result_receipt_file_sha256=receipt_file_sha256,
            result_path=result_path,
            result_file_sha256=result_file_sha256,
            result_artifact_sha256=receipt.result_artifact_sha256,
            execution_receipt_path=prepared.execution_receipt_path,
            execution_receipt_file_sha256=execution_receipt_file_sha256,
            execution_receipt_sha256=execution_receipt.receipt_sha256,
        ),
        completion_claimed=completion_claimed,
        completion_phase_claim=completion_phase_claim,
    )


def execute_prepared_offline_analysis(
    prepared: PreparedOfflineAnalysis,
    claimed: VerifiedProviderPredecessor,
    phase_claim: VerifiedPhaseClaimCapability,
    plan: ProviderPhasePlan,
    *,
    fresh_claim_supplier: Callable[
        [], tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]
    ],
    docker_run: DockerRun = subprocess.run,
) -> ExecutedOfflineAnalysis:
    """Create and start the named container, then retain its closed execution."""

    if not isinstance(prepared, PreparedOfflineAnalysis):
        raise OfflineAnalysisProviderError("offline analysis launch is untyped")
    _validate_authority(claimed, phase_claim, plan)
    _validate_docker_create_argv(
        prepared.docker_create_argv,
        admission=prepared.admission,
        admission_path=prepared.admission_path,
        package_root=prepared.package_root,
        results_root=prepared.results_root,
        docker_config_root=prepared.docker_config_root,
        docker_resolved_executable=prepared.docker_resolved_executable,
    )
    input_store = {
        row.relative_path
        for row in prepared.admission.package_files
        if row.role in {"confirmatory-input", "confirmatory-input-receipt"}
    }
    completed_store = input_store | set(
        confirmatory_output_filenames(prepared.admission.manifest_sha256)
    )
    observed_store = _directory_names(
        prepared.results_root,
        label="analysis results store",
    )
    receipt_present = _path_present(
        prepared.execution_receipt_path,
        label="offline analysis execution receipt",
    )
    if receipt_present or observed_store == completed_store:
        if observed_store != completed_store:
            raise OfflineAnalysisProviderError(
                "offline execution receipt lacks the exact five-file result closure"
            )
        return _close_completed_offline_analysis(
            prepared,
            claimed,
            phase_claim,
            plan,
            fresh_claim_supplier=fresh_claim_supplier,
            docker_run=docker_run,
        )
    if observed_store != input_store:
        raise OfflineAnalysisProviderError(
            "partial confirmatory outcome is terminal; the registered attempt cannot rerun"
        )

    container_created = False
    try:
        if _directory_names(
            prepared.docker_config_root,
            label="anonymous Docker configuration",
        ):
            raise OfflineAnalysisProviderError("anonymous Docker configuration is not empty")
        _unchanged_package_tree(prepared)
        _verify_docker_tool(prepared)
        _assert_container_absent(prepared, docker_run=docker_run)

        # The networked pull consumes no mounted study bytes and is bracketed by
        # the same typed ANALYSIS_CLAIMED authority.
        _validate_authority(claimed, phase_claim, plan)
        _run_docker_bounded(
            docker_run,
            prepared.docker_pull_argv,
            timeout=min(900, prepared.maximum_runtime_seconds),
            label="anonymous platform-image pull",
        )
        _validate_authority(claimed, phase_claim, plan)
        if _directory_names(
            prepared.docker_config_root,
            label="anonymous Docker configuration",
        ):
            raise OfflineAnalysisProviderError(
                "anonymous Docker pull persisted client configuration"
            )
        _unchanged_package_tree(prepared)
        _verify_docker_tool(prepared)
        _assert_container_absent(prepared, docker_run=docker_run)

        _run_docker_bounded(
            docker_run,
            prepared.docker_create_argv,
            timeout=60,
            label="scientific container creation",
        )
        container_created = True
        _run_docker_bounded(
            docker_run,
            prepared.docker_inspect_argv,
            timeout=30,
            label="scientific container creation readback",
        )
        _validate_authority(claimed, phase_claim, plan)
        _unchanged_package_tree(prepared)
        _verify_docker_tool(prepared)
        _run_docker_bounded(
            docker_run,
            prepared.docker_start_argv,
            timeout=prepared.maximum_runtime_seconds,
            label="scientific container execution",
        )
        return _close_completed_offline_analysis(
            prepared,
            claimed,
            phase_claim,
            plan,
            fresh_claim_supplier=fresh_claim_supplier,
            docker_run=docker_run,
        )
    except BaseException:
        if container_created:
            try:
                _force_remove_container(prepared, docker_run=docker_run)
            except Exception as cleanup_exc:
                raise OfflineAnalysisProviderError(
                    "offline analysis failure left an unverifiable Docker container"
                ) from cleanup_exc
        raise


def _matching_completion_authority(
    initial_claimed: VerifiedProviderPredecessor,
    initial_phase_claim: VerifiedPhaseClaimCapability,
    fresh_claim_supplier: Callable[
        [], tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]
    ],
    plan: ProviderPhasePlan,
) -> tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]:
    try:
        fresh_claimed, fresh_phase_claim = fresh_claim_supplier()
    except Exception as exc:
        raise OfflineAnalysisProviderError(
            "cannot refresh ANALYSIS_CLAIMED completion authority"
        ) from exc
    _validate_authority(fresh_claimed, fresh_phase_claim, plan)
    exact = (
        fresh_claimed.state.record_sha256 == initial_claimed.state.record_sha256,
        fresh_claimed.ledger_commit == initial_claimed.ledger_commit,
        fresh_claimed.control_inventory_sha256 == initial_claimed.control_inventory_sha256,
        fresh_claimed.artifact_receipt_sha256 == initial_claimed.artifact_receipt_sha256,
        fresh_phase_claim.contract == initial_phase_claim.contract,
        fresh_phase_claim.provider_identity == initial_phase_claim.provider_identity,
        fresh_phase_claim.phase_claim_state_sha256 == initial_phase_claim.phase_claim_state_sha256,
        fresh_phase_claim.phase_claim_ledger_commit
        == initial_phase_claim.phase_claim_ledger_commit,
    )
    if not all(exact):
        raise OfflineAnalysisProviderError(
            "refreshed completion authority differs from the admitted analysis"
        )
    return fresh_claimed, fresh_phase_claim


def run_provider_claimed_offline_analysis_once(
    config: ConfirmatoryInputOperatorConfig,
    plan: ProviderPhasePlan,
    claimed: VerifiedProviderPredecessor,
    phase_claim: VerifiedPhaseClaimCapability,
    *,
    package_root: str | Path,
    results_root: str | Path,
    fresh_claim_supplier: Callable[
        [], tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]
    ],
    docker_run: DockerRun = subprocess.run,
) -> ProviderAnalysisCompletion:
    """Materialize, run in the C1 image, then complete under fresh authority."""

    _validate_authority(claimed, phase_claim, plan)
    recovered_input = False
    try:
        materialized = materialize_confirmatory_input(config, claimed)
    except ConfirmatoryInputOperatorError as initial_error:
        try:
            materialized = load_materialized_confirmatory_input(config, claimed)
        except ConfirmatoryInputOperatorError:
            raise initial_error
        recovered_input = True
    suite = load_admitted_model_suite(config, materialized.inputs)
    prepared = prepare_offline_analysis(
        materialized,
        suite,
        plan,
        claimed,
        phase_claim,
        package_root=package_root,
        results_root=results_root,
        recover_existing=recovered_input,
    )
    executed = execute_prepared_offline_analysis(
        prepared,
        claimed,
        phase_claim,
        plan,
        fresh_claim_supplier=fresh_claim_supplier,
        docker_run=docker_run,
    )
    outcome = executed.outcome
    completion_claimed = executed.completion_claimed
    completion_phase_claim = executed.completion_phase_claim
    completion_claimed.assert_current()
    completion_phase_claim.assert_current()
    candidate = complete_confirmatory_analysis(
        completion_claimed,
        phase_claim=completion_phase_claim,
        confirmatory_input_artifact_sha256=(prepared.admission.confirmatory_input_artifact_sha256),
        execution_receipt_path=outcome.execution_receipt_path,
        execution_receipt_sha256=outcome.execution_receipt_sha256,
        execution_receipt_file_sha256=outcome.execution_receipt_file_sha256,
        attempt_receipt_path=outcome.attempt_path,
        result_receipt_path=outcome.result_receipt_path,
        final_result_path=outcome.result_path,
    )
    if candidate.state != "ANALYSIS_COMPLETE":
        raise OfflineAnalysisProviderError(
            "offline analysis did not produce an ANALYSIS_COMPLETE candidate"
        )
    return ProviderAnalysisCompletion(candidate=candidate, outcome=outcome)
