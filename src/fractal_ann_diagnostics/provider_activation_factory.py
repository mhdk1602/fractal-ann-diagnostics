"""Closed self-hosted activation for one provider-claimed phase.

The public entry point accepts GitHub artifact identity, not scientific paths.
Every executable path is recovered from the verified claim artifact, C1 plan,
or provider state chain before the live execute job is allowed to open inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    digest_regular_file,
    write_exclusive_receipt_bytes,
)
from .confirmatory_input_operator import (
    ConfirmatoryInputOperatorConfig,
    CorpusEvidenceLocation,
)
from .drand_beacon import (
    DrandBeaconError,
    DrandReadApi,
    QuicknetExecutionBeaconVerifier,
)
from .execution_claim import (
    ACTIVATION_COMMON_OUTPUT_KEYS,
    ACTIVATION_PHASE_OUTPUT_KEYS,
    ANALYSIS_PHASE,
    LABEL_RELEASE_PHASE,
    ONLINE_PHASE,
    PHASE_HOST_TOOL_RECEIPT_FILENAME,
    ExecutionClaimContract,
    ExecutionClaimError,
    LiveExecuteJobReceipt,
    PhaseClaimContract,
    ProviderPhase,
    ProviderPhasePlan,
    VerifiedPhaseClaimCapability,
    load_materialized_provider_phase_plan,
    verify_live_execute_job,
    verify_phase_host_tools,
)
from .github_artifact_transport import (
    GitHubArtifactReadApi,
    derive_and_verify_fixed_claim_artifact,
)
from .offline_analysis_contract import (
    OfflineAnalysisContractError,
    OfflineAnalysisExecutionReceipt,
    load_offline_analysis_execution_receipt,
)
from .production_controls import (
    ProductionControlFinalizationReceipt,
    ProductionControlFinalizationRequest,
    load_production_control_finalization_receipt,
    load_production_control_finalization_request,
)
from .production_corpus_run import RUNTIME_ATTESTATION_RECEIPT_FILENAME
from .provider_phase_runtime import (
    ANALYSIS_RUNTIME_CLAIM_BUNDLE_SCHEMA,
    PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME,
    AnalysisRuntimeClaimBundle,
    FreshLabelClaimAuthority,
    FreshOnlineClaimAuthority,
    LabelReleaseDriverControl,
    OnlineSealedLaunchDriverControl,
    ProviderDriverRequest,
    ProviderPhaseExecutionReceipt,
    ProviderPhaseRuntimeError,
    ProviderPhaseRuntimeRequest,
    admit_analysis_results_store,
    admit_label_release_phase_root,
    execute_provider_phase_request,
    write_provider_phase_runtime_request,
)
from .provider_state_transport import (
    ProviderStateArtifactReadApi,
    materialize_provider_claim,
)
from .provider_workflow_orchestration import (
    ProviderClaimReceipt,
    ProviderWorkflowContext,
    load_provider_claim_receipt,
)
from .sealed_container_launcher import load_sealed_launch_contract
from .study import FIXED_CORPORA, load_study_manifest, manifest_sha256, validate_study_manifest
from .suite_attempt import (
    PhaseClaimBindings,
    RunClaimBindings,
    SuiteOpenBindings,
    VerifiedProviderPredecessor,
    admit_analysis_claim,
    admit_label_release_claim_beacon,
    admit_run_claim_beacon,
)

_DRIVER_IDS: Mapping[ProviderPhase, str] = {
    ONLINE_PHASE: "sealed-online-corpus-v1",
    LABEL_RELEASE_PHASE: "timelock-label-release-v1",
    ANALYSIS_PHASE: "confirmatory-analysis-v1",
}
_ALL_FIVE = "all-five"
_MANIFEST_NAME = "study-manifest.json"
_PLAN_COPY_NAME = "provider-plan.materialized.json"
_CLAIM_RECEIPT_NAME = "claim-receipt.json"
_LIVE_JOB_NAME = "live-execute-job-receipt.json"
_LABEL_INVENTORY_NAME = "label-release-inventory.json"
_LAUNCH_INVENTORY_NAME = "launch-receipt-inventory.json"
_RUNTIME_RECEIPT_BUNDLE_NAME = "runtime-claim-receipts.json"
_TLE_RECEIPT_NAME = "timelock-decryption-receipt.json"
_ANALYSIS_CONTROL_NAME = "confirmatory-input-operator.json"


class ProviderActivationError(ValueError):
    """The activation request differs from the fixed C1/provider authority."""


class GitHubActivationApi(Protocol):
    def get(self, endpoint: str) -> object: ...


@dataclass(frozen=True)
class ProviderActivationResult:
    phase: ProviderPhase
    outputs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        expected = ACTIVATION_COMMON_OUTPUT_KEYS | ACTIVATION_PHASE_OUTPUT_KEYS[self.phase]
        observed = dict(self.outputs)
        if len(observed) != len(self.outputs) or set(observed) != set(expected):
            raise ProviderActivationError("activation outputs differ from the registered interface")

    def output_fields(self) -> dict[str, str]:
        return dict(self.outputs)


@dataclass(frozen=True)
class _OpenedProductionControl:
    opened: SuiteOpenBindings
    finalization_receipt_path: Path
    finalization_request_path: Path
    finalization_receipt: ProductionControlFinalizationReceipt
    finalization_request: ProductionControlFinalizationRequest
    suite_namespace: Path
    completion_root: Path


def _canonical_file_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _write_once(path: Path, encoded: bytes, *, label: str) -> Path:
    try:
        write_exclusive_receipt_bytes(encoded, path)
    except ArtifactIntegrityError as exc:
        raise ProviderActivationError(f"cannot write {label} once") from exc
    return path


def _new_private_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ProviderActivationError(f"{label} must be one new absolute directory")
    parent = path.parent
    try:
        metadata = parent.stat(follow_symlinks=False)
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise ProviderActivationError(f"{label} parent is unavailable") from exc
    if (
        resolved_parent != parent
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
    ):
        raise ProviderActivationError(f"{label} parent is not controlled")
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise ProviderActivationError(f"cannot create {label}") from exc
    return path


def _admit_empty_private_directory(
    path: Path,
    *,
    label: str,
    allowed_entries: frozenset[str] = frozenset(),
) -> Path:
    if not path.is_absolute():
        raise ProviderActivationError(f"{label} must be one absolute directory")
    if path.is_symlink():
        raise ProviderActivationError(f"{label} must not be a symlink")
    if not path.exists():
        return _new_private_directory(path, label=label)
    try:
        metadata = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
        entries = tuple(path.iterdir())
    except OSError as exc:
        raise ProviderActivationError(f"cannot admit {label}") from exc
    if (
        resolved != path
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or (hasattr(os, "geteuid") and metadata.st_uid != os.geteuid())
        or not {entry.name for entry in entries}.issubset(allowed_entries)
    ):
        raise ProviderActivationError(f"{label} must be one controlled empty directory")
    return path


def _uri_path(value: str, *, label: str) -> Path:
    parsed = urlsplit(value)
    path = Path(unquote(parsed.path))
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not path.is_absolute()
        or path.as_uri() != value
    ):
        raise ProviderActivationError(f"{label} is not one canonical local file URI")
    return path


def _unique_suffix(root: Path, suffix: str, *, label: str) -> Path:
    rows = tuple(path for path in root.rglob(suffix) if path.is_file() and not path.is_symlink())
    if len(rows) != 1:
        raise ProviderActivationError(f"verified claim artifact lacks one exact {label}")
    return rows[0]


def _load_packaged_plan(path: Path) -> ProviderPhasePlan:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ProviderActivationError("packaged provider plan repeats a JSON key")
            result[key] = value
        return result

    try:
        encoded = path.read_bytes()
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProviderActivationError(
                    f"packaged provider plan contains non-finite number {value}"
                )
            ),
        )
        plan = ProviderPhasePlan.from_dict(value)
    except ProviderActivationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProviderActivationError("cannot load packaged provider plan") from exc
    if encoded != plan.canonical_file_bytes():
        raise ProviderActivationError("packaged provider plan bytes are not canonical")
    return plan


def _materialize_fixed_claim_receipt(
    *,
    downloaded_path: Path,
    packaged_path: Path,
    plan: ProviderPhasePlan,
    suite_attempt_id: str,
) -> Path:
    if digest_regular_file(
        downloaded_path, label="downloaded claim receipt"
    ) != digest_regular_file(packaged_path, label="verified claim receipt"):
        raise ProviderActivationError("downloaded claim receipt differs from fixed artifact")
    fixed_path = Path(plan.claim_receipt_path(suite_attempt_id))
    try:
        fixed_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise ProviderActivationError("cannot create the C1-fixed claim receipt root") from exc
    if fixed_path.exists():
        if digest_regular_file(fixed_path, label="fixed claim receipt") != digest_regular_file(
            packaged_path, label="verified claim receipt"
        ):
            raise ProviderActivationError("fixed claim receipt differs from verified artifact")
    else:
        _write_once(fixed_path, packaged_path.read_bytes(), label="fixed claim receipt")
    return fixed_path


def _claim_authority(
    claimed: VerifiedProviderPredecessor,
) -> tuple[ExecutionClaimContract | PhaseClaimContract, Any]:
    payload = claimed.state.payload
    if isinstance(payload, RunClaimBindings):
        return payload.execution_claim, payload.provider_identity
    if isinstance(payload, PhaseClaimBindings):
        return payload.phase_claim, payload.provider_identity
    raise ProviderActivationError("materialized claim has no typed provider authority")


def _hosted_claim_context_sha256(context: ProviderWorkflowContext) -> str:
    if context.job != "execute":
        raise ProviderActivationError("activation requires admitted execute-job context")
    identity = context.identity_dict()
    identity.update(
        job="claim",
        runner_environment="github-hosted",
        runner_os="Linux",
        runner_arch="X64",
    )
    return hashlib.sha256(_canonical_file_bytes(identity)[:-1]).hexdigest()


def _cross_check_claim(
    *,
    phase: ProviderPhase,
    suite_attempt_id: str,
    context: ProviderWorkflowContext,
    receipt: ProviderClaimReceipt,
    plan: ProviderPhasePlan,
    claimed: VerifiedProviderPredecessor,
    contract: ExecutionClaimContract | PhaseClaimContract,
    provider_identity: Any,
) -> None:
    expected_state = {
        ONLINE_PHASE: "RUN_CLAIMED",
        LABEL_RELEASE_PHASE: "LABEL_RELEASE_CLAIMED",
        ANALYSIS_PHASE: "ANALYSIS_CLAIMED",
    }[phase]
    if (
        receipt.phase != phase
        or receipt.suite_attempt_id != suite_attempt_id
        or receipt.run_id != context.run_id
        or receipt.workflow_context_sha256 != _hosted_claim_context_sha256(context)
        or receipt.provider_plan_sha256 != plan.plan_sha256
        or receipt.provider_identity_sha256 != provider_identity.identity_sha256
        or receipt.target_state != expected_state
        or receipt.target_sequence != claimed.state.sequence
        or receipt.target_state_record_sha256 != claimed.state.record_sha256
        or receipt.target_ledger_commit != claimed.ledger_commit
        or receipt.claim_contract_sha256 != contract.contract_sha256
        or receipt.runner_label != contract.unique_runner_label
        or receipt.expected_execute_job_name != contract.execute_job_name
        or receipt.manifest_sha256 != contract.manifest_sha256
        or plan.phase != phase
        or plan.suite_attempt_id != suite_attempt_id
        or plan.manifest_sha256 != contract.manifest_sha256
    ):
        raise ProviderActivationError("claim receipt, C1 plan, and current provider state differ")


def _write_live_job(receipt: LiveExecuteJobReceipt, root: Path) -> Path:
    return _write_once(
        root / _LIVE_JOB_NAME, _canonical_file_bytes(receipt.to_dict()), label="live job receipt"
    )


class _FreshClaimReader:
    def __init__(
        self,
        *,
        phase: ProviderPhase,
        suite_attempt_id: str,
        root: Path,
        github_api: Any,
        artifact_api: Any,
    ) -> None:
        self.phase = phase
        self.suite_attempt_id = suite_attempt_id
        self.root = root
        self.github_api = github_api
        self.artifact_api = artifact_api
        self.counter = 0

    def __call__(self) -> VerifiedProviderPredecessor:
        self.counter += 1
        parent = self.root / f"read-{self.counter:04d}"
        _new_private_directory(parent, label="fresh claim read root")
        return materialize_provider_claim(
            self.phase,
            self.suite_attempt_id,
            parent,
            ledger_api=self.github_api,
            artifact_api=self.artifact_api,
        ).predecessor


def _manifest_artifact(
    manifest: Mapping[str, Any], *, role: str, corpus_id: str | None = None
) -> Path:
    rows = [
        row
        for row in manifest.get("artifacts", [])
        if isinstance(row, Mapping)
        and row.get("role") == role
        and (corpus_id is None or row.get("corpus_id") == corpus_id)
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("uri"), str):
        raise ProviderActivationError(f"frozen manifest lacks one {role} locator")
    return _uri_path(rows[0]["uri"], label=f"{role} artifact")


def _opened_production_control(
    claimed: VerifiedProviderPredecessor,
) -> _OpenedProductionControl:
    opened_record = claimed.records[0]
    opened = opened_record.payload
    if not isinstance(opened, SuiteOpenBindings):
        raise ProviderActivationError("provider claim lacks OPENED production authority")
    finalization_receipt_path = _uri_path(
        opened.production_finalization_receipt_uri,
        label="production finalization receipt",
    )
    finalization_receipt = load_production_control_finalization_receipt(
        finalization_receipt_path,
        expected_sha256=opened.production_finalization_receipt_file_sha256,
    )
    finalization_request_path = finalization_receipt_path.with_name("finalization-request.json")
    finalization_request = load_production_control_finalization_request(
        finalization_request_path,
        expected_sha256=finalization_receipt.finalization_request_sha256,
    )
    suite_namespace = Path(finalization_receipt.canonical_suite_namespace)
    if (
        finalization_receipt.finalization_request_sha256
        != opened.production_finalization_request_sha256
        or finalization_receipt.suite_attempt_id != opened_record.suite_attempt_id
        or finalization_receipt.manifest_sha256 != opened_record.manifest_sha256
        or suite_namespace.as_uri() != opened_record.namespace_uri
    ):
        raise ProviderActivationError(
            "OPENED production finalization differs from the provider state identity"
        )
    return _OpenedProductionControl(
        opened=opened,
        finalization_receipt_path=finalization_receipt_path,
        finalization_request_path=finalization_request_path,
        finalization_receipt=finalization_receipt,
        finalization_request=finalization_request,
        suite_namespace=suite_namespace,
        completion_root=suite_namespace / "completion",
    )


def _online_controls(
    production: _OpenedProductionControl,
) -> dict[str, OnlineSealedLaunchDriverControl]:
    result: dict[str, OnlineSealedLaunchDriverControl] = {}
    for binding in production.opened.runtime_attestation_plans:
        contract_path = _uri_path(
            binding.sealed_launch_contract_uri, label="sealed launch contract"
        )
        contract = load_sealed_launch_contract(contract_path)
        if (
            contract.contract_sha256 != binding.sealed_launch_contract_sha256
            or contract.file_sha256 != binding.sealed_launch_contract_file_sha256
            or contract.geometry.corpus_id != binding.corpus_id
        ):
            raise ProviderActivationError("sealed launch contract differs from OPENED state")
        launcher_root = contract_path.parent
        runtime_root = production.finalization_request.runtime_evidence_root / binding.corpus_id
        result[binding.corpus_id] = OnlineSealedLaunchDriverControl(
            preflight_contract_path=str(launcher_root / "preflight-launch-contract.json"),
            preflight_receipt_path=str(runtime_root / "runtime-preflight-receipt.json"),
            transition_receipt_path=str(runtime_root / "runtime-plan-transition-receipt.json"),
            instantiation_receipt_path=str(launcher_root / "plan-instantiation-receipt.json"),
            finalization_request_path=str(production.finalization_request_path),
            finalization_receipt_path=str(production.finalization_receipt_path),
            sealed_contract_path=str(contract_path),
            volume_receipt_path=str(
                launcher_root
                / "volume-initialization-evidence"
                / "volume-initialization-receipt.json"
            ),
            audit_root=str(launcher_root / "sealed-evidence"),
        )
    if set(result) != set(FIXED_CORPORA):
        raise ProviderActivationError("online controls do not cover the fixed corpora")
    return result


def _label_control(
    *,
    corpus_id: str,
    contract: PhaseClaimContract,
    production: _OpenedProductionControl,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    output_root: Path,
) -> LabelReleaseDriverControl:
    binding = next(row for row in contract.corpora if row.corpus_id == corpus_id)
    plaintext = _uri_path(binding.output_uri, label="label output")
    return LabelReleaseDriverControl(
        manifest_path=str(manifest_path),
        custody_seal_path=str(_manifest_artifact(manifest, role="custody-seal-receipt")),
        encryption_receipt_path=str(
            _uri_path(binding.supporting_input_uri, label="encryption receipt")
        ),
        completion_receipt_path=str(
            production.completion_root / f"{corpus_id}-prediction-completion.json"
        ),
        completion_anchor_record_path=str(
            production.completion_root / f"{corpus_id}-prediction-completion-anchor.json"
        ),
        completion_anchor_receipt_path=str(
            production.completion_root / f"{corpus_id}-prediction-completion-anchor-receipt.json"
        ),
        suite_namespace=str(production.suite_namespace),
        ciphertext_path=str(_uri_path(binding.input_uri, label="label ciphertext")),
        tle_binary_path=str(_manifest_artifact(manifest, role="timelock-tool")),
        plaintext_output_path=str(plaintext),
        decryption_receipt_path=str(output_root / _TLE_RECEIPT_NAME),
    )


def _label_control_bytes(control: LabelReleaseDriverControl) -> bytes:
    return _canonical_file_bytes(
        {name: getattr(control, name) for name in control.__dataclass_fields__}
    )


def _analysis_control(
    *,
    claimed: VerifiedProviderPredecessor,
    production: _OpenedProductionControl,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> ConfirmatoryInputOperatorConfig:
    labels = next(
        (record.payload for record in claimed.records if record.state == "LABELS_RELEASED"),
        None,
    )
    online = next(
        (record.payload for record in claimed.records if record.state == "ONLINE_COMPLETE"),
        None,
    )
    if not isinstance(labels, tuple) or online is None or not hasattr(online, "corpora"):
        raise ProviderActivationError("analysis claim lacks released-label and online closures")
    by_label = {row.corpus_id: row for row in labels}
    by_online = {row.corpus_id: row for row in online.corpora}
    evidence = []
    for corpus_id in sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")):
        online_root = _uri_path(by_online[corpus_id].output_uri, label="online output")
        evidence.append(
            CorpusEvidenceLocation(
                corpus_id=corpus_id,
                prediction_completion_receipt_uri=(
                    production.completion_root / f"{corpus_id}-prediction-completion.json"
                ).as_uri(),
                prediction_completion_anchor_record_uri=(
                    production.completion_root / f"{corpus_id}-prediction-completion-anchor.json"
                ).as_uri(),
                prediction_completion_anchor_receipt_uri=(
                    production.completion_root
                    / f"{corpus_id}-prediction-completion-anchor-receipt.json"
                ).as_uri(),
                timelock_decryption_receipt_uri=_uri_path(
                    by_label[corpus_id].decryption_receipt_uri, label="decryption receipt"
                ).as_uri(),
            )
        )
        del online_root
    return ConfirmatoryInputOperatorConfig(
        suite_namespace_uri=production.suite_namespace.as_uri(),
        manifest_uri=manifest_path.as_uri(),
        sealed_run_receipt_uri=production.finalization_request.sealed_run_receipt_path.as_uri(),
        artifact_verification_receipt_uri=(
            production.finalization_request.artifact_verification_receipt_path.as_uri()
        ),
        artifact_root_uri=production.finalization_request.artifact_root.as_uri(),
        corpus_evidence=tuple(evidence),
    )


def _verify_analysis_execution_authority(
    offline_execution: OfflineAnalysisExecutionReceipt,
    *,
    suite_attempt_id: str,
    contract: PhaseClaimContract,
    claimed: VerifiedProviderPredecessor,
    capability: VerifiedPhaseClaimCapability,
    provider_identity: Any,
    result_path: Path,
) -> str:
    """Cross-check retained container evidence against the live analysis claim."""

    result_file_sha256 = digest_regular_file(result_path, label="analysis result")
    if (
        offline_execution.suite_attempt_id != suite_attempt_id
        or offline_execution.manifest_sha256 != contract.manifest_sha256
        or offline_execution.run_receipt_sha256 != contract.run_receipt_sha256
        or offline_execution.provider_state_record_sha256 != claimed.state.record_sha256
        or offline_execution.provider_ledger_commit != claimed.ledger_commit
        or offline_execution.phase_claim_contract_sha256 != contract.contract_sha256
        or offline_execution.phase_claim_state_sha256 != capability.phase_claim_state_sha256
        or offline_execution.phase_claim_ledger_commit != capability.phase_claim_ledger_commit
        or offline_execution.provider_identity_sha256 != provider_identity.identity_sha256
        or offline_execution.c1_commit != contract.c1_commit
        or offline_execution.result_uri != result_path.as_uri()
        or offline_execution.result_file_sha256 != result_file_sha256
    ):
        raise ProviderActivationError(
            "offline analysis execution differs from activation authority"
        )
    return result_file_sha256


def _inventory_file(root: Path, execution: ProviderPhaseExecutionReceipt, *, name: str) -> Path:
    value = {
        "outputs": [row.to_dict() for row in execution.outputs],
        "phase": execution.phase,
        "schema_version": "fractal-provider-activation-inventory-v1",
        "suite_attempt_id": execution.suite_attempt_id,
    }
    return _write_once(root / name, _canonical_file_bytes(value), label="phase inventory")


def _launch_inventory_file(
    root: Path,
    execution: ProviderPhaseExecutionReceipt,
    controls: Mapping[str, OnlineSealedLaunchDriverControl],
) -> Path:
    rows: list[dict[str, str]] = []
    by_output = {row.corpus_id: Path(row.output_root) for row in execution.outputs}
    for corpus_id in sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")):
        control = controls[corpus_id]
        sealed = load_sealed_launch_contract(control.sealed_contract_path)
        plan_path = (
            Path(sealed.geometry.control_mount.source)
            / sealed.geometry.runtime_plan_template_relative_path
        )
        evidence = {
            "runtime_attestation_plan": plan_path,
            "runtime_attestation_receipt": (
                by_output[corpus_id] / RUNTIME_ATTESTATION_RECEIPT_FILENAME
            ),
            "sealed_launch_receipt": (Path(control.audit_root) / "sealed-launch-receipt.json"),
        }
        for role, path in sorted(evidence.items()):
            rows.append(
                {
                    "corpus_id": corpus_id,
                    "file_sha256": digest_regular_file(path, label=f"{corpus_id} {role}"),
                    "path": str(path),
                    "role": role,
                }
            )
    return _write_once(
        root / _LAUNCH_INVENTORY_NAME,
        _canonical_file_bytes(
            {
                "evidence": rows,
                "phase": ONLINE_PHASE,
                "schema_version": "fractal-provider-launch-inventory-v1",
                "suite_attempt_id": execution.suite_attempt_id,
            }
        ),
        label="launch receipt inventory",
    )


def _publish_then_wait_for_label_beacon(
    claimed: VerifiedProviderPredecessor,
    contract: PhaseClaimContract,
    plan: ProviderPhasePlan,
    *,
    completion_anchor_token_fd: int,
    verifier: QuicknetExecutionBeaconVerifier,
    clock_now: Callable[[], str],
    sleeper: Callable[[float], None],
) -> bytes:
    """Publish the outcome-blind anchors before waiting for the release round."""

    assert contract.label_release_beacon is not None
    release_time = contract.label_release_beacon.label_release_publication_time

    def observed_time() -> datetime:
        try:
            value = datetime.fromisoformat(clock_now())
        except ValueError as exc:
            raise ProviderActivationError("verification clock returned malformed time") from exc
        if value.tzinfo is None or value.utcoffset() is None:
            raise ProviderActivationError("verification clock must be timezone-aware")
        return value

    initial_wait_seconds = max(0.0, (release_time - observed_time()).total_seconds())
    if initial_wait_seconds > plan.maximum_runtime_seconds - 120:
        raise ProviderActivationError(
            "label-release round lies outside the registered activation window"
        )
    try:
        from .post_online_completion import (
            PostOnlineCompletionError,
            publish_post_online_completion_anchors,
        )

        publish_post_online_completion_anchors(
            claimed,
            token_fd=completion_anchor_token_fd,
        )
    except PostOnlineCompletionError as exc:
        raise ProviderActivationError(f"post-online completion publication failed: {exc}") from exc
    wait_seconds = max(0.0, (release_time - observed_time()).total_seconds())
    if wait_seconds:
        sleeper(wait_seconds + 1.0)
    last_beacon_error: DrandBeaconError | None = None
    for attempt in range(12):
        try:
            return verifier.fetch(contract.label_release_beacon)
        except DrandBeaconError as exc:
            last_beacon_error = exc
            if attempt + 1 < 12:
                sleeper(5.0)
    raise ProviderActivationError(
        "registered label-release beacon did not become readable"
    ) from last_beacon_error


def activate_and_execute_provider_phase(
    *,
    context: ProviderWorkflowContext,
    phase: ProviderPhase,
    suite_attempt_id: str,
    artifact_id: int,
    artifact_digest: str,
    expected_inventory_sha256: str,
    claim_receipt_destination: str | Path,
    output_dir: str | Path,
    github_api: GitHubActivationApi,
    artifact_api: GitHubArtifactReadApi | ProviderStateArtifactReadApi,
    verified_at_utc: str | None = None,
    drand_api: DrandReadApi | None = None,
    verification_clock: Callable[[], str] | None = None,
    completion_anchor_token_fd: int | None = None,
    release_sleeper: Callable[[float], None] | None = None,
) -> ProviderActivationResult:
    """Verify, activate, and consume exactly one C1-fixed phase request."""

    if context.phase != phase:
        raise ProviderActivationError("workflow context and activation phase differ")
    if phase == LABEL_RELEASE_PHASE:
        if type(completion_anchor_token_fd) is not int or completion_anchor_token_fd < 0:
            raise ProviderActivationError(
                "label-release activation requires a Zenodo token file descriptor"
            )
    elif completion_anchor_token_fd is not None:
        raise ProviderActivationError(
            "only label-release activation accepts a Zenodo token file descriptor"
        )
    downloaded_claim_path = Path(claim_receipt_destination)
    if (
        not downloaded_claim_path.is_absolute()
        or downloaded_claim_path.name != _CLAIM_RECEIPT_NAME
        or not downloaded_claim_path.is_file()
        or downloaded_claim_path.is_symlink()
    ):
        raise ProviderActivationError("downloaded claim evidence differs from its fixed filename")
    root = _new_private_directory(Path(output_dir), label="activation output root")
    artifact_root = root / "fixed-claim"
    derive_and_verify_fixed_claim_artifact(
        context,
        artifact_api,  # type: ignore[arg-type]
        suite_attempt_id=suite_attempt_id,
        artifact_id=artifact_id,
        artifact_digest=artifact_digest,
        expected_inventory_sha256=expected_inventory_sha256,
        destination=artifact_root,
    )
    packaged_claim = artifact_root / _CLAIM_RECEIPT_NAME
    plan_copy = artifact_root / _PLAN_COPY_NAME
    plan_from_artifact = _load_packaged_plan(plan_copy)
    plan = load_materialized_provider_phase_plan(plan_from_artifact.provider_plan_path)
    if plan != plan_from_artifact or plan.canonical_file_bytes() != plan_copy.read_bytes():
        raise ProviderActivationError("self-hosted C1 plan differs from the claim artifact copy")
    host_verified_at = verified_at_utc or (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    try:
        host_receipt = verify_phase_host_tools(
            plan.host_tools,
            probe_output_dir=root / "phase-host-probes",
            verified_at_utc=host_verified_at,
        )
    except ExecutionClaimError as exc:
        raise ProviderActivationError(f"phase host-tool verification failed: {exc}") from exc
    host_receipt_path = _write_once(
        root / PHASE_HOST_TOOL_RECEIPT_FILENAME,
        _canonical_file_bytes(host_receipt.to_dict()),
        label="phase host-tool receipt",
    )
    host_receipt_file_sha256 = digest_regular_file(
        host_receipt_path,
        label="phase host-tool receipt",
    )
    claim_path = _materialize_fixed_claim_receipt(
        downloaded_path=downloaded_claim_path,
        packaged_path=packaged_claim,
        plan=plan,
        suite_attempt_id=suite_attempt_id,
    )
    receipt = load_provider_claim_receipt(claim_path)
    claimed_root = _new_private_directory(root / "claimed-state", label="claimed state root")
    claimed = materialize_provider_claim(
        phase,
        suite_attempt_id,
        claimed_root,
        ledger_api=github_api,
        artifact_api=artifact_api,  # type: ignore[arg-type]
    ).predecessor
    contract, provider_identity = _claim_authority(claimed)
    _cross_check_claim(
        phase=phase,
        suite_attempt_id=suite_attempt_id,
        context=context,
        receipt=receipt,
        plan=plan,
        claimed=claimed,
        contract=contract,
        provider_identity=provider_identity,
    )

    def clock_now() -> str:
        value = (
            verification_clock()
            if verification_clock is not None
            else datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        if type(value) is not str or not value or value != value.strip():
            raise ProviderActivationError("verification clock returned malformed time")
        return value

    now = verified_at_utc or clock_now()
    fresh_root = _new_private_directory(root / "fresh-claims", label="fresh claim roots")
    fresh = _FreshClaimReader(
        phase=phase,
        suite_attempt_id=suite_attempt_id,
        root=fresh_root,
        github_api=github_api,
        artifact_api=artifact_api,
    )
    verifier = QuicknetExecutionBeaconVerifier(drand_api)
    sleep = time.sleep if release_sleeper is None else release_sleeper
    if not callable(sleep):
        raise ProviderActivationError("release sleeper must be callable")
    if phase == ONLINE_PHASE:
        assert isinstance(contract, ExecutionClaimContract)
        live = verify_live_execute_job(
            api=github_api,
            contract=contract,
            provider_identity=provider_identity,
            verified_at_utc=now,
        )
        beacon_bytes = verifier.fetch(contract.beacon)
        capability = admit_run_claim_beacon(
            claimed,
            beacon_bytes=beacon_bytes,
            beacon_verifier=verifier,
            live_execute_job_receipt=live,
            verified_at_utc=now,
            fresh_state_revalidator=fresh,
        )
    elif phase == LABEL_RELEASE_PHASE:
        assert isinstance(contract, PhaseClaimContract)
        assert contract.label_release_beacon is not None
        assert completion_anchor_token_fd is not None
        beacon_bytes = _publish_then_wait_for_label_beacon(
            claimed,
            contract,
            plan,
            completion_anchor_token_fd=completion_anchor_token_fd,
            verifier=verifier,
            clock_now=clock_now,
            sleeper=sleep,
        )
        now = clock_now()
        live = verify_live_execute_job(
            api=github_api,
            contract=contract,
            provider_identity=provider_identity,
            verified_at_utc=now,
        )
        capability = admit_label_release_claim_beacon(
            claimed,
            beacon_bytes=beacon_bytes,
            beacon_verifier=verifier,
            live_execute_job_receipt=live,
            verified_at_utc=now,
            fresh_state_revalidator=fresh,
        )
    else:
        live = verify_live_execute_job(
            api=github_api,
            contract=contract,
            provider_identity=provider_identity,
            verified_at_utc=now,
        )
        capability = admit_analysis_claim(
            claimed,
            live_execute_job_receipt=live,
            fresh_state_revalidator=fresh,
        )
    live_path = _write_live_job(live, root)

    manifest_path = _unique_suffix(artifact_root, _MANIFEST_NAME, label="C1 study manifest")
    manifest = load_study_manifest(manifest_path)
    validate_study_manifest(manifest, require_frozen=True)
    if manifest_sha256(manifest) != contract.manifest_sha256:
        raise ProviderActivationError("claim artifact manifest differs from provider claim")
    production = _opened_production_control(claimed)
    if phase == ANALYSIS_PHASE:
        assert isinstance(contract, PhaseClaimContract)
        results_store = _uri_path(
            manifest["sealed_execution"]["results_store"],
            label="analysis results store",
        )
        if {row.output_uri for row in contract.corpora} != {results_store.as_uri()}:
            raise ProviderActivationError(
                "analysis claim output differs from the frozen results store"
            )
    phase_output_path = Path(plan.phase_evidence_root(suite_attempt_id))
    if phase == LABEL_RELEASE_PHASE:
        admit_label_release_phase_root(
            phase_output_path,
            create_if_absent=True,
        )
        phase_output = phase_output_path
    elif phase == ANALYSIS_PHASE:
        phase_output = _admit_empty_private_directory(
            phase_output_path,
            label="analysis phase evidence root",
            allowed_entries=frozenset({PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME}),
        )
    else:
        phase_output = _new_private_directory(
            phase_output_path,
            label="phase evidence root",
        )
    driver_rows: list[ProviderDriverRequest] = []
    portable_receipts: list[dict[str, object]] = []
    action_admissions: list[dict[str, object]] = []
    online_supplier: Callable[[ProviderDriverRequest], FreshOnlineClaimAuthority] | None = None
    label_supplier: Callable[[ProviderDriverRequest], FreshLabelClaimAuthority] | None = None
    analysis_supplier: (
        Callable[[], tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]] | None
    ) = None
    action_root = _new_private_directory(root / "action-authority", label="action authority root")
    ordered = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
    if phase == ONLINE_PHASE:
        controls = _online_controls(production)
        phase_input = _new_private_directory(root / "phase-input", label="phase input root")
        claims_root = _new_private_directory(
            phase_input / "runtime-claims", label="runtime claims root"
        )
        controls_root = _new_private_directory(
            phase_input / "controls", label="driver controls root"
        )
        by_binding = {row.corpus_id: row for row in contract.corpora}
        for corpus_id in ordered:
            binding = by_binding[corpus_id]
            runtime = capability.require_launch(
                manifest_sha256=contract.manifest_sha256,
                corpus_id=corpus_id,
                runtime_plan_sha256=binding.runtime_plan_sha256,
                output_namespace_uri=binding.staging_namespace_uri,
            )
            receipt_path = _write_once(
                claims_root / f"{corpus_id}.json",
                runtime.canonical_file_bytes(),
                label="runtime claim",
            )
            control_path = _write_once(
                controls_root / f"{corpus_id}.json",
                controls[corpus_id].canonical_file_bytes(),
                label="online sealed-launch control",
            )
            driver_rows.append(
                ProviderDriverRequest(
                    corpus_id=corpus_id,
                    driver_id=_DRIVER_IDS[phase],
                    control_path=str(control_path),
                    control_file_sha256=digest_regular_file(control_path, label="online control"),
                    runtime_claim_receipt_path=str(receipt_path),
                    runtime_claim_receipt_file_sha256=digest_regular_file(
                        receipt_path, label="runtime claim"
                    ),
                    output_root=str(phase_output / corpus_id),
                )
            )

        def online_supplier(row: ProviderDriverRequest) -> FreshOnlineClaimAuthority:
            current = fresh()
            current_contract, current_identity = _claim_authority(current)
            if not isinstance(current_contract, ExecutionClaimContract):
                raise ProviderActivationError("fresh online claim has another contract type")
            _cross_check_claim(
                phase=phase,
                suite_attempt_id=suite_attempt_id,
                context=context,
                receipt=receipt,
                plan=plan,
                claimed=current,
                contract=current_contract,
                provider_identity=current_identity,
            )
            action_now = clock_now()
            action_live = verify_live_execute_job(
                api=github_api,
                contract=current_contract,
                provider_identity=current_identity,
                verified_at_utc=action_now,
            )
            action_beacon = verifier.fetch(current_contract.beacon)
            action_capability = admit_run_claim_beacon(
                current,
                beacon_bytes=action_beacon,
                beacon_verifier=verifier,
                live_execute_job_receipt=action_live,
                verified_at_utc=action_now,
                fresh_state_revalidator=fresh,
            )
            binding = by_binding[row.corpus_id]
            runtime = action_capability.require_launch(
                manifest_sha256=current_contract.manifest_sha256,
                corpus_id=row.corpus_id,
                runtime_plan_sha256=binding.runtime_plan_sha256,
                output_namespace_uri=binding.staging_namespace_uri,
            )
            runtime_path = _write_once(
                action_root / f"{row.corpus_id}.runtime-claim.json",
                runtime.canonical_file_bytes(),
                label="fresh runtime claim",
            )
            live_evidence = _write_once(
                action_root / f"{row.corpus_id}.live-execute-job.json",
                _canonical_file_bytes(action_live.to_dict()),
                label="fresh live execute-job receipt",
            )
            beacon_evidence = _write_once(
                action_root / f"{row.corpus_id}.beacon.json",
                _canonical_file_bytes(action_capability.beacon_receipt.to_dict()),
                label="fresh execution beacon receipt",
            )
            portable_receipts.append(runtime.to_dict())
            action_admissions.append(
                {
                    "corpus_id": row.corpus_id,
                    "live_execute_job_receipt_path": str(live_evidence),
                    "live_execute_job_receipt_sha256": digest_regular_file(
                        live_evidence, label="fresh live execute-job receipt"
                    ),
                    "beacon_receipt_path": str(beacon_evidence),
                    "beacon_receipt_sha256": digest_regular_file(
                        beacon_evidence, label="fresh execution beacon receipt"
                    ),
                    "runtime_claim_receipt_path": str(runtime_path),
                    "runtime_claim_receipt_sha256": digest_regular_file(
                        runtime_path, label="fresh runtime claim"
                    ),
                }
            )
            return FreshOnlineClaimAuthority(
                capability=action_capability,
                claim_bytes=runtime.canonical_file_bytes(),
            )
    else:
        phase_input = _new_private_directory(root / "phase-input", label="phase input root")
        claims_root = _new_private_directory(
            phase_input / "runtime-claims", label="runtime claims root"
        )
        controls_root = _new_private_directory(
            phase_input / "controls", label="driver controls root"
        )
        assert isinstance(contract, PhaseClaimContract)
        by_binding = {row.corpus_id: row for row in contract.corpora}
        phase_receipts = []
        for corpus_id in ordered:
            binding = by_binding[corpus_id]
            phase_receipts.append(
                capability.require_input(
                    corpus_id=corpus_id,
                    input_uri=binding.input_uri,
                    input_sha256=binding.input_sha256,
                    supporting_input_uri=binding.supporting_input_uri,
                    supporting_input_sha256=binding.supporting_input_sha256,
                )
            )
        if phase == LABEL_RELEASE_PHASE:
            for corpus_id, runtime in zip(ordered, phase_receipts, strict=True):
                output_root = phase_output / corpus_id
                control = _label_control(
                    corpus_id=corpus_id,
                    contract=contract,
                    production=production,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    output_root=output_root,
                )
                control_path = _write_once(
                    controls_root / f"{corpus_id}.json",
                    _label_control_bytes(control),
                    label="label control",
                )
                receipt_path = _write_once(
                    claims_root / f"{corpus_id}.json",
                    runtime.canonical_file_bytes(),
                    label="phase runtime claim",
                )
                driver_rows.append(
                    ProviderDriverRequest(
                        corpus_id=corpus_id,
                        driver_id=_DRIVER_IDS[phase],
                        control_path=str(control_path),
                        control_file_sha256=digest_regular_file(
                            control_path, label="label control"
                        ),
                        runtime_claim_receipt_path=str(receipt_path),
                        runtime_claim_receipt_file_sha256=digest_regular_file(
                            receipt_path, label="phase runtime claim"
                        ),
                        output_root=str(output_root),
                    )
                )

            def label_supplier(row: ProviderDriverRequest) -> FreshLabelClaimAuthority:
                current = fresh()
                current_contract, current_identity = _claim_authority(current)
                if not isinstance(current_contract, PhaseClaimContract):
                    raise ProviderActivationError(
                        "fresh label-release claim has another contract type"
                    )
                _cross_check_claim(
                    phase=phase,
                    suite_attempt_id=suite_attempt_id,
                    context=context,
                    receipt=receipt,
                    plan=plan,
                    claimed=current,
                    contract=current_contract,
                    provider_identity=current_identity,
                )
                assert current_contract.label_release_beacon is not None
                action_now = clock_now()
                action_live = verify_live_execute_job(
                    api=github_api,
                    contract=current_contract,
                    provider_identity=current_identity,
                    verified_at_utc=action_now,
                )
                action_beacon = verifier.fetch(current_contract.label_release_beacon)

                def revalidate_claim_and_live_job() -> VerifiedProviderPredecessor:
                    revalidated = fresh()
                    verify_live_execute_job(
                        api=github_api,
                        contract=current_contract,
                        provider_identity=current_identity,
                        verified_at_utc=clock_now(),
                    )
                    return revalidated

                action_capability = admit_label_release_claim_beacon(
                    current,
                    beacon_bytes=action_beacon,
                    beacon_verifier=verifier,
                    live_execute_job_receipt=action_live,
                    verified_at_utc=action_now,
                    fresh_state_revalidator=revalidate_claim_and_live_job,
                )
                binding = by_binding[row.corpus_id]
                runtime = action_capability.require_input(
                    corpus_id=row.corpus_id,
                    input_uri=binding.input_uri,
                    input_sha256=binding.input_sha256,
                    supporting_input_uri=binding.supporting_input_uri,
                    supporting_input_sha256=binding.supporting_input_sha256,
                )
                runtime_path = _write_once(
                    action_root / f"{row.corpus_id}.phase-runtime-claim.json",
                    runtime.canonical_file_bytes(),
                    label="fresh phase runtime claim",
                )
                live_evidence = _write_once(
                    action_root / f"{row.corpus_id}.live-execute-job.json",
                    _canonical_file_bytes(action_live.to_dict()),
                    label="fresh live execute-job receipt",
                )
                assert action_capability.phase_beacon_receipt is not None
                beacon_evidence = _write_once(
                    action_root / f"{row.corpus_id}.beacon.json",
                    _canonical_file_bytes(action_capability.phase_beacon_receipt.to_dict()),
                    label="fresh phase beacon receipt",
                )
                marker = _write_once(
                    action_root / f"{row.corpus_id}.pre-decryption-admission.json",
                    _canonical_file_bytes(
                        {
                            "admitted_at_utc": action_now,
                            "beacon_receipt_sha256": (
                                action_capability.phase_beacon_receipt.receipt_sha256
                            ),
                            "corpus_id": row.corpus_id,
                            "input_sha256": binding.input_sha256,
                            "input_uri": binding.input_uri,
                            "live_execute_job_receipt_sha256": action_live.receipt_sha256,
                            "output_identity_sha256": (current_contract.phase_output_identity),
                            "output_uri": binding.output_uri,
                            "phase": LABEL_RELEASE_PHASE,
                            "phase_claim_contract_sha256": (current_contract.contract_sha256),
                            "phase_claim_ledger_commit": (
                                action_capability.phase_claim_ledger_commit
                            ),
                            "phase_claim_state_sha256": (
                                action_capability.phase_claim_state_sha256
                            ),
                            "provider_identity_sha256": (current_identity.identity_sha256),
                            "runtime_claim_receipt_sha256": digest_regular_file(
                                runtime_path, label="fresh phase runtime claim"
                            ),
                            "schema_version": "fractal-pre-decryption-admission-v1",
                            "suite_attempt_id": suite_attempt_id,
                            "supporting_input_sha256": (binding.supporting_input_sha256),
                            "supporting_input_uri": binding.supporting_input_uri,
                        }
                    ),
                    label="pre-decryption admission marker",
                )
                portable_receipts.append(runtime.to_dict())
                action_admissions.append(
                    {
                        "corpus_id": row.corpus_id,
                        "live_execute_job_receipt_path": str(live_evidence),
                        "live_execute_job_receipt_sha256": digest_regular_file(
                            live_evidence, label="fresh live execute-job receipt"
                        ),
                        "beacon_receipt_path": str(beacon_evidence),
                        "beacon_receipt_sha256": digest_regular_file(
                            beacon_evidence, label="fresh phase beacon receipt"
                        ),
                        "runtime_claim_receipt_path": str(runtime_path),
                        "runtime_claim_receipt_sha256": digest_regular_file(
                            runtime_path, label="fresh phase runtime claim"
                        ),
                        "pre_decryption_admission_path": str(marker),
                        "pre_decryption_admission_sha256": digest_regular_file(
                            marker, label="pre-decryption admission marker"
                        ),
                    }
                )
                return FreshLabelClaimAuthority(
                    capability=action_capability,
                    predecessor=current,
                    claim_bytes=runtime.canonical_file_bytes(),
                    admission_marker_path=str(marker),
                    admission_marker_sha256=digest_regular_file(
                        marker, label="pre-decryption admission marker"
                    ),
                )
        else:
            results_store = _uri_path(
                manifest["sealed_execution"]["results_store"],
                label="analysis results store",
            )
            if not os.path.lexists(results_store):
                _new_private_directory(
                    results_store,
                    label="analysis results store",
                )
            try:
                restart_entries = admit_analysis_results_store(
                    results_store,
                    manifest_sha256=contract.manifest_sha256,
                )
            except ProviderPhaseRuntimeError as exc:
                raise ProviderActivationError(
                    f"analysis results store restart state is invalid: {exc}"
                ) from exc
            store_admission = _write_once(
                action_root / "analysis-results-store-admission.json",
                _canonical_file_bytes(
                    {
                        "entries": list(restart_entries),
                        "phase": ANALYSIS_PHASE,
                        "results_store": str(results_store),
                        "schema_version": "fractal-analysis-restart-store-admission-v1",
                        "suite_attempt_id": suite_attempt_id,
                        "verified_at_utc": now,
                    }
                ),
                label="analysis empty-store admission",
            )
            action_admissions.append(
                {
                    "corpus_id": _ALL_FIVE,
                    "results_store_admission_path": str(store_admission),
                    "results_store_admission_sha256": digest_regular_file(
                        store_admission, label="analysis results-store admission"
                    ),
                    "stage": "pre-execution",
                }
            )
            control = _analysis_control(
                claimed=claimed,
                production=production,
                manifest_path=manifest_path,
                manifest=manifest,
            )
            control_path = _write_once(
                controls_root / _ANALYSIS_CONTROL_NAME,
                control.canonical_bytes() + b"\n",
                label="analysis control",
            )
            bundle = AnalysisRuntimeClaimBundle(
                phase="analysis",
                receipts=tuple(phase_receipts),
                schema_version=ANALYSIS_RUNTIME_CLAIM_BUNDLE_SCHEMA,
            )
            receipt_path = _write_once(
                claims_root / "all-five.json",
                bundle.canonical_file_bytes(),
                label="analysis runtime bundle",
            )
            driver_rows.append(
                ProviderDriverRequest(
                    corpus_id=_ALL_FIVE,
                    driver_id=_DRIVER_IDS[phase],
                    control_path=str(control_path),
                    control_file_sha256=digest_regular_file(control_path, label="analysis control"),
                    runtime_claim_receipt_path=str(receipt_path),
                    runtime_claim_receipt_file_sha256=digest_regular_file(
                        receipt_path, label="analysis runtime bundle"
                    ),
                    output_root=str(results_store),
                )
            )
            portable_receipts.extend(row.to_dict() for row in phase_receipts)

            analysis_refresh_count = 0

            def analysis_supplier() -> tuple[
                VerifiedProviderPredecessor, VerifiedPhaseClaimCapability
            ]:
                nonlocal analysis_refresh_count
                analysis_refresh_count += 1
                if analysis_refresh_count > 2:
                    raise ProviderActivationError(
                        "analysis requested more than start and completion authority"
                    )
                stage = "start" if analysis_refresh_count == 1 else "completion"
                current = fresh()
                current_contract, current_identity = _claim_authority(current)
                if not isinstance(current_contract, PhaseClaimContract):
                    raise ProviderActivationError("fresh analysis claim has another contract type")
                _cross_check_claim(
                    phase=phase,
                    suite_attempt_id=suite_attempt_id,
                    context=context,
                    receipt=receipt,
                    plan=plan,
                    claimed=current,
                    contract=current_contract,
                    provider_identity=current_identity,
                )
                action_now = clock_now()
                action_live = verify_live_execute_job(
                    api=github_api,
                    contract=current_contract,
                    provider_identity=current_identity,
                    verified_at_utc=action_now,
                )
                action_capability = admit_analysis_claim(
                    current,
                    live_execute_job_receipt=action_live,
                    fresh_state_revalidator=fresh,
                )
                live_evidence = _write_once(
                    action_root / f"analysis-{stage}.live-execute-job.json",
                    _canonical_file_bytes(action_live.to_dict()),
                    label="fresh analysis live execute-job receipt",
                )
                action_admissions.append(
                    {
                        "corpus_id": _ALL_FIVE,
                        "live_execute_job_receipt_path": str(live_evidence),
                        "live_execute_job_receipt_sha256": digest_regular_file(
                            live_evidence, label="fresh analysis live execute-job receipt"
                        ),
                        "stage": stage,
                    }
                )
                return current, action_capability

    claimed.assert_current()
    request = ProviderPhaseRuntimeRequest(
        phase=phase,
        activation_command_id=plan.activation_command_id,
        suite_attempt_id=suite_attempt_id,
        provider_plan_path=plan.provider_plan_path,
        provider_plan_sha256=plan.plan_sha256,
        provider_plan_file_sha256=plan.file_sha256,
        claim_receipt_path=str(claim_path),
        claim_receipt_file_sha256=receipt.file_sha256,
        phase_host_tool_receipt_path=str(host_receipt_path),
        phase_host_tool_receipt_sha256=host_receipt.receipt_sha256,
        phase_host_tool_receipt_file_sha256=host_receipt_file_sha256,
        phase_input_root=str(phase_input),
        phase_output_root=str(phase_output),
        drivers=tuple(driver_rows),
    )
    write_provider_phase_runtime_request(request)
    claimed.assert_current()
    execution = execute_provider_phase_request(
        plan=plan,
        request=request,
        online_run_claim_supplier=online_supplier,
        label_phase_claim_supplier=label_supplier,
        provider_claimed=(claimed if phase == ANALYSIS_PHASE else None),
        analysis_phase_claim=(capability if phase == ANALYSIS_PHASE else None),
        analysis_claim_supplier=analysis_supplier,
    )
    expected_admissions = 5 if phase in {ONLINE_PHASE, LABEL_RELEASE_PHASE} else 3
    if len(portable_receipts) != 5 or len(action_admissions) != expected_admissions:
        raise ProviderActivationError("per-action authority evidence is incomplete")
    portable_path = _write_once(
        root / _RUNTIME_RECEIPT_BUNDLE_NAME,
        _canonical_file_bytes(
            {
                "action_admissions": action_admissions,
                "phase": phase,
                "receipts": portable_receipts,
            }
        ),
        label="runtime receipt inventory",
    )
    execution_path = phase_output / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME
    common = {
        "execute_job_id": str(live.execute_job_id),
        "fixed_corpora_completed": "true",
        "live_execute_job_receipt_path": str(live_path),
        "live_execute_job_receipt_sha256": digest_regular_file(live_path, label="live job receipt"),
        "phase_host_tool_receipt_path": str(host_receipt_path),
        "phase_host_tool_receipt_sha256": host_receipt.receipt_sha256,
        "phase_host_tool_receipt_file_sha256": host_receipt_file_sha256,
        "phase_execution_receipt_path": str(execution_path),
        "phase_execution_receipt_sha256": execution.file_sha256,
        "runtime_claim_receipt_path": str(portable_path),
        "runtime_claim_receipt_sha256": digest_regular_file(
            portable_path, label="runtime receipt inventory"
        ),
    }
    if phase == ONLINE_PHASE:
        inventory = _launch_inventory_file(root, execution, controls)
        common.update(
            five_corpora_executed="true",
            launch_receipt_inventory_path=str(inventory),
            launch_receipt_inventory_sha256=digest_regular_file(
                inventory, label="launch inventory"
            ),
        )
    elif phase == LABEL_RELEASE_PHASE:
        inventory = _inventory_file(root, execution, name=_LABEL_INVENTORY_NAME)
        common.update(
            five_label_payloads_decrypted="true",
            label_release_inventory_path=str(inventory),
            label_release_inventory_sha256=digest_regular_file(inventory, label="label inventory"),
        )
    else:
        results_store = _uri_path(
            manifest["sealed_execution"]["results_store"], label="analysis results store"
        )
        result_path = results_store / f"{contract.manifest_sha256}.confirmatory-result.json"
        if len(execution.outputs) != 1:
            raise ProviderActivationError("analysis phase execution lacks its sole driver output")
        analysis_output = execution.outputs[0]
        if (
            analysis_output.output_root != str(results_store)
            or analysis_output.analysis_execution_receipt_uri is None
            or analysis_output.analysis_execution_receipt_sha256 is None
            or analysis_output.analysis_execution_receipt_file_sha256 is None
        ):
            raise ProviderActivationError(
                "analysis phase execution discarded offline execution evidence"
            )
        execution_receipt_path = _uri_path(
            analysis_output.analysis_execution_receipt_uri,
            label="offline analysis execution receipt",
        )
        try:
            offline_execution = load_offline_analysis_execution_receipt(
                execution_receipt_path,
                expected_receipt_sha256=(analysis_output.analysis_execution_receipt_sha256),
                expected_file_sha256=(analysis_output.analysis_execution_receipt_file_sha256),
            )
        except OfflineAnalysisContractError as exc:
            raise ProviderActivationError(
                f"offline analysis execution receipt is invalid: {exc}"
            ) from exc
        result_file_sha256 = _verify_analysis_execution_authority(
            offline_execution,
            suite_attempt_id=suite_attempt_id,
            contract=contract,
            claimed=claimed,
            capability=capability,
            provider_identity=provider_identity,
            result_path=result_path,
        )
        common.update(
            analysis_execution_receipt_path=str(execution_receipt_path),
            analysis_execution_receipt_sha256=(offline_execution.receipt_sha256),
            analysis_execution_receipt_file_sha256=(
                analysis_output.analysis_execution_receipt_file_sha256
            ),
            analysis_result_path=str(result_path),
            analysis_result_sha256=result_file_sha256,
            five_corpora_analyzed="true",
        )
    return ProviderActivationResult(phase=phase, outputs=tuple(sorted(common.items())))
