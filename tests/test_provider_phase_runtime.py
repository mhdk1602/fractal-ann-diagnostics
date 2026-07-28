from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_execution_claim import _host_tools

import fractal_ann_diagnostics.confirmatory_input_operator as input_operator
import fractal_ann_diagnostics.offline_analysis_provider as offline_provider
import fractal_ann_diagnostics.provider_phase_runtime as runtime_module
from fractal_ann_diagnostics.execution_claim import (
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
    OFFICIAL_ACTIONS_RUNNER_VERSION,
    PHASE_JOB_NAMES,
    PHASE_RUNTIME_BINDINGS,
    PROVIDER_PHASE_PLAN_SCHEMA,
    ExecutionBeaconContract,
    ExecutionClaimError,
    ExecutionClaimInputs,
    PhaseRuntimeClaimReceipt,
    ProviderPhasePlan,
    ProviderRunnerBootstrapReceipt,
    VerifiedPhaseClaimCapability,
    VerifiedRunClaimCapability,
    derive_phase_runner_label,
    load_provider_runner_bootstrap,
    verify_provider_runner_ready,
)
from fractal_ann_diagnostics.provider_phase_runtime import (
    PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME,
    PROVIDER_RUNTIME_REQUEST_FILENAME,
    AnalysisRuntimeClaimBundle,
    DockerTleDecryptRunner,
    FreshLabelClaimAuthority,
    FreshOnlineClaimAuthority,
    LabelReleaseDriverControl,
    LabelReleaseOutputAuthority,
    ProviderDriverRequest,
    ProviderPhaseRuntimeError,
    ProviderPhaseRuntimeRequest,
    execute_provider_phase_request,
    load_provider_phase_runtime_request,
    main,
    write_provider_phase_runtime_request,
)
from fractal_ann_diagnostics.study import (
    FIXED_CORPORA,
    PROVIDER_APPROVAL_ENVIRONMENT,
    PROVIDER_PHASE_COMMAND_IDS,
    PROVIDER_PHASE_RUNTIME_CEILINGS,
    PROVIDER_PHASE_WORKFLOWS,
    PROVIDER_PLAN_CLAIM_RECEIPT_BINDING,
    PROVIDER_PLAN_PHASE_INPUT_BINDING,
    PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
    PROVIDER_PLAN_PREDECESSOR_BINDING,
    PROVIDER_PLAN_SUITE_BINDING,
    PROVIDER_RUNNER_IDENTITY,
)
from fractal_ann_diagnostics.suite_attempt import VerifiedProviderPredecessor


def _digest(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _label_authority_parts(
    tmp_path: Path,
    *,
    minted_ns: int = 0,
) -> tuple[VerifiedPhaseClaimCapability, VerifiedProviderPredecessor]:
    state_sha256 = _digest("label-claim-state")
    ledger_commit = "a" * 40
    namespace = tmp_path / "canonical-suite"
    namespace.mkdir(exist_ok=True)
    capability = object.__new__(VerifiedPhaseClaimCapability)
    object.__setattr__(capability, "_test_minted_ns", minted_ns)
    object.__setattr__(capability, "phase_claim_state_sha256", state_sha256)
    object.__setattr__(capability, "phase_claim_ledger_commit", ledger_commit)
    object.__setattr__(
        capability,
        "contract",
        SimpleNamespace(
            corpora=(
                SimpleNamespace(
                    corpus_id="scifact",
                    output_uri=(
                        tmp_path / "released" / "scifact" / "released-labels.json"
                    ).as_uri(),
                ),
            ),
        ),
    )
    predecessor = object.__new__(VerifiedProviderPredecessor)
    object.__setattr__(
        predecessor,
        "records",
        (
            SimpleNamespace(
                state="LABEL_RELEASE_CLAIMED",
                record_sha256=state_sha256,
                namespace_uri=namespace.as_uri(),
            ),
        ),
    )
    object.__setattr__(
        predecessor,
        "evidences",
        (SimpleNamespace(transition_id=ledger_commit),),
    )
    return capability, predecessor


def _fake_label_output_authority(
    row: ProviderDriverRequest,
) -> LabelReleaseOutputAuthority:
    authority = object.__new__(LabelReleaseOutputAuthority)
    object.__setattr__(authority, "corpus_id", row.corpus_id)
    for name in (
        "post_online_completion_aggregate_file_sha256",
        "label_release_claim_state_sha256",
        "label_release_phase_claim_contract_sha256",
        "label_release_phase_beacon_receipt_sha256",
        "label_release_live_execute_job_receipt_sha256",
        "label_release_provider_identity_sha256",
    ):
        object.__setattr__(authority, name, _digest(f"{name}:{row.corpus_id}"))
    object.__setattr__(
        authority,
        "label_release_claim_ledger_commit",
        "a" * 40,
    )
    evidence = SimpleNamespace(to_dict=lambda: {"evidence": row.corpus_id})
    object.__setattr__(
        authority,
        "label_release_phase_beacon_receipt",
        evidence,
    )
    object.__setattr__(
        authority,
        "label_release_live_execute_job_receipt",
        evidence,
    )
    return authority


def _plan(tmp_path: Path, *, phase: str = "online") -> ProviderPhasePlan:
    host_root = tmp_path / "host-tools"
    host_tools = _host_tools(host_root)
    provider_path = host_root / "provider-plans" / f"{phase}.json"
    workflow = PROVIDER_PHASE_WORKFLOWS[phase]
    claim_job, execute_job = PHASE_JOB_NAMES[phase]
    platform, image_role, index_role = PHASE_RUNTIME_BINDINGS[phase]
    runtime_digest = _digest(f"runtime:{phase}")
    tle = phase == "label-release"
    claim_nonce = _digest(f"nonce:{phase}")
    runner_label = derive_phase_runner_label(claim_nonce, phase)  # type: ignore[arg-type]
    bootstrap_receipt = ProviderRunnerBootstrapReceipt(
        phase=phase,  # type: ignore[arg-type]
        repository="mhdk1602/fractal-ann-diagnostics",
        approval_environment=PROVIDER_APPROVAL_ENVIRONMENT,
        runner_identity=PROVIDER_RUNNER_IDENTITY,
        workflow_sha="1" * 40,
        runner_label=runner_label,
        runner_id=101,
        runner_name=f"fractal-confirmatory-{phase}",
        runner_group_id=None,
        runner_version=OFFICIAL_ACTIONS_RUNNER_VERSION,
        runner_archive_sha256=OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
        repository_runner_inventory_sha256=_digest(f"inventory:{phase}"),
        ephemeral=True,
        disable_update=True,
        unattended=True,
        registered_at_utc="2026-07-17T12:00:00+00:00",
    )
    return ProviderPhasePlan(
        phase=phase,  # type: ignore[arg-type]
        manifest_sha256=_digest("manifest"),
        c1_commit="2" * 40,
        suite_attempt_id_binding=PROVIDER_PLAN_SUITE_BINDING,
        claim_predecessor_binding=PROVIDER_PLAN_PREDECESSOR_BINDING,
        claim_receipt_path_template=str(
            tmp_path / "claims" / "{suite_attempt_id}" / phase / "claim-receipt.json"
        ),
        phase_input_binding=PROVIDER_PLAN_PHASE_INPUT_BINDING,
        phase_output_binding=PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
        provider_plan_path=str(provider_path),
        phase_evidence_root_template=str(tmp_path / "evidence" / "{suite_attempt_id}" / phase),
        repository="mhdk1602/fractal-ann-diagnostics",
        approval_environment=PROVIDER_APPROVAL_ENVIRONMENT,
        runner_identity=PROVIDER_RUNNER_IDENTITY,
        workflow_path=workflow,
        workflow_ref=(
            f"mhdk1602/fractal-ann-diagnostics/{workflow}@refs/tags/confirmatory-apparatus-c0"
        ),
        workflow_sha="1" * 40,
        run_head_branch="confirmatory-apparatus-c0",
        claim_job_name=claim_job,
        execute_job_name=execute_job,
        execution_claim_inputs=(
            ExecutionClaimInputs(
                design_seed_sha256=_digest("design-seed"),
                registered_online_runtime_budget_seconds=68_000,
                beacon=ExecutionBeaconContract(
                    drand_network="https://api.drand.sh",
                    chain_hash="a" * 64,
                    chain_scheme_id="bls-unchained-g1-rfc9380",
                    chain_public_key="ab" * 48,
                    chain_genesis_unix_seconds=1_595_431_050,
                    chain_period_seconds=3,
                    execution_round=100,
                    label_release_round=120,
                    minimum_label_release_safety_rounds=10,
                    verification_identity="b" * 64,
                ),
            )
            if phase == "online"
            else None
        ),
        claim_nonce=claim_nonce,
        runner_id=101,
        runner_name=f"fractal-confirmatory-{phase}",
        runner_registration_bundle_path=str(
            host_root / "production" / "runner-registrations" / phase / runner_label
        ),
        runner_registration_bundle_sha256=_digest(f"registration-bundle:{phase}"),
        runner_registration_evidence_file_sha256=_digest(f"registration-evidence:{phase}"),
        runner_group_id=None,
        runner_version=OFFICIAL_ACTIONS_RUNNER_VERSION,
        runner_archive_sha256=OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
        runner_bootstrap_receipt=bootstrap_receipt,
        runner_bootstrap_receipt_path=str(
            host_root / "production" / "runners" / phase / runner_label / "bootstrap-receipt.json"
        ),
        runner_bootstrap_receipt_file_sha256=bootstrap_receipt.file_sha256,
        provider_operating_system="macOS",
        provider_architecture="ARM64",
        host_tools=host_tools,
        runtime_probe_receipt_sha256=_digest(f"probe:{phase}"),
        runtime_image=f"ghcr.io/mhdk1602/runtime@sha256:{runtime_digest}",
        runtime_platform=platform,
        runtime_image_role=image_role,
        runtime_index_role=index_role,
        oci_index_digest=f"sha256:{runtime_digest}",
        oci_platform_manifest_digest=f"sha256:{_digest(f'platform:{phase}')}",
        tle_binary_sha256=(
            "ca9d498b6a3c1ea8edff9ace7bf00eb0f90ce67166343161f9a53f21900a6ef5" if tle else None
        ),
        tle_build_provenance_sha256=_digest("tle-build") if tle else None,
        tle_vulnerability_scan_sha256=_digest("tle-scan") if tle else None,
        tle_interoperability_receipt_sha256=_digest("tle-interop") if tle else None,
        maximum_runtime_seconds=PROVIDER_PHASE_RUNTIME_CEILINGS[phase],
        activation_command_id=PROVIDER_PHASE_COMMAND_IDS[phase],
        activation_argv_template=(
            host_tools.python_executable,
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
        ),
        schema_version=PROVIDER_PHASE_PLAN_SCHEMA,
    )


def _request_fixture(
    tmp_path: Path,
    *,
    phase: str = "online",
) -> tuple[ProviderPhasePlan, ProviderPhaseRuntimeRequest, Path]:
    plan = _plan(tmp_path, phase=phase)
    _write(Path(plan.provider_plan_path), plan.canonical_file_bytes())
    _write(
        Path(plan.runner_bootstrap_receipt_path),
        plan.runner_bootstrap_receipt.canonical_file_bytes(),
    )
    claim_path = Path(plan.claim_receipt_path(plan.suite_attempt_id))
    claim_bytes = b'{"claim":"fixed"}\n'
    _write(claim_path, claim_bytes)
    input_root = tmp_path / "phase-input"
    output_root = Path(plan.phase_evidence_root(plan.suite_attempt_id))
    rows: list[ProviderDriverRequest] = []
    for corpus_id in sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")):
        control_path = input_root / corpus_id / "corpus-run-config.json"
        claim_receipt = input_root / corpus_id / "runtime-claim-receipt.json"
        control_bytes = (
            json.dumps({"corpus_id": corpus_id}, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
            + b"\n"
        )
        runtime_bytes = (
            json.dumps(
                {"corpus_id": corpus_id, "receipt": "test"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )
        _write(control_path, control_bytes)
        _write(claim_receipt, runtime_bytes)
        rows.append(
            ProviderDriverRequest(
                corpus_id=corpus_id,
                driver_id={
                    "online": "sealed-online-corpus-v1",
                    "label-release": "timelock-label-release-v1",
                }[phase],
                control_path=str(control_path),
                control_file_sha256=hashlib.sha256(control_bytes).hexdigest(),
                runtime_claim_receipt_path=str(claim_receipt),
                runtime_claim_receipt_file_sha256=hashlib.sha256(runtime_bytes).hexdigest(),
                output_root=str(output_root / corpus_id),
            )
        )
    request = ProviderPhaseRuntimeRequest(
        phase=phase,  # type: ignore[arg-type]
        activation_command_id=plan.activation_command_id,
        suite_attempt_id=plan.suite_attempt_id,
        provider_plan_path=plan.provider_plan_path,
        provider_plan_sha256=plan.plan_sha256,
        provider_plan_file_sha256=plan.file_sha256,
        claim_receipt_path=str(claim_path),
        claim_receipt_file_sha256=hashlib.sha256(claim_bytes).hexdigest(),
        phase_input_root=str(input_root),
        phase_output_root=str(output_root),
        drivers=tuple(rows),
    )
    input_root.mkdir(parents=True, exist_ok=True)
    request_path = write_provider_phase_runtime_request(request)
    return plan, request, request_path


def _analysis_runtime_receipt(corpus_id: str) -> PhaseRuntimeClaimReceipt:
    return PhaseRuntimeClaimReceipt(
        phase="analysis",
        manifest_sha256=_digest("manifest"),
        run_receipt_sha256=_digest("run-receipt"),
        c1_commit="2" * 40,
        phase_claim_contract_sha256=_digest("analysis-contract"),
        phase_claim_state_sha256=_digest("analysis-state"),
        phase_claim_ledger_commit="3" * 40,
        provider_identity_sha256=_digest("analysis-provider"),
        live_execute_job_receipt_sha256=_digest("analysis-job"),
        execute_job_id=101,
        phase_input_aggregate_sha256=_digest("analysis-inputs"),
        phase_output_identity=_digest("analysis-output"),
        corpus_id=corpus_id,
        input_uri=f"file:///private/input/{corpus_id}.json",
        input_sha256=_digest(f"label:{corpus_id}"),
        supporting_input_uri=f"file:///private/online/{corpus_id}",
        supporting_input_sha256=_digest(f"online:{corpus_id}"),
        phase_beacon_receipt_sha256=None,
    )


def test_runtime_request_round_trip_is_canonical_and_fixed_path(tmp_path: Path) -> None:
    _, request, path = _request_fixture(tmp_path)
    assert path.name == PROVIDER_RUNTIME_REQUEST_FILENAME
    assert load_provider_phase_runtime_request(path) == request
    assert path.read_bytes() == request.canonical_file_bytes()


def test_runner_bootstrap_receipt_is_rehashed_against_the_phase_plan(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipt = ProviderRunnerBootstrapReceipt(
        phase=plan.phase,
        repository=plan.repository,
        approval_environment=plan.approval_environment,
        runner_identity=plan.runner_identity,
        workflow_sha=plan.workflow_sha,
        runner_label=derive_phase_runner_label(plan.claim_nonce, plan.phase),
        runner_id=plan.runner_id,
        runner_name=plan.runner_name,
        runner_group_id=plan.runner_group_id,
        runner_version=plan.runner_version,
        runner_archive_sha256=plan.runner_archive_sha256,
        repository_runner_inventory_sha256=_digest("runner-inventory"),
        ephemeral=True,
        disable_update=True,
        unattended=True,
        registered_at_utc="2026-07-17T12:00:00+00:00",
    )
    plan = replace(
        plan,
        runner_bootstrap_receipt=receipt,
        runner_bootstrap_receipt_file_sha256=receipt.file_sha256,
    )
    _write(Path(plan.runner_bootstrap_receipt_path), receipt.canonical_file_bytes())
    assert load_provider_runner_bootstrap(plan) == receipt

    hostile = replace(receipt, runner_id=receipt.runner_id + 1)
    Path(plan.runner_bootstrap_receipt_path).write_bytes(hostile.canonical_file_bytes())
    with pytest.raises(ExecutionClaimError, match="bytes differ from C1"):
        load_provider_runner_bootstrap(plan)


def test_runner_readiness_requires_exact_idle_singleton_and_labels(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    expected_labels = (
        "self-hosted",
        "macOS",
        "ARM64",
        derive_phase_runner_label(plan.claim_nonce, plan.phase),
    )

    class Api:
        runner = {
            "busy": False,
            "id": plan.runner_id,
            "labels": [{"name": value} for value in expected_labels],
            "name": plan.runner_name,
            "os": "macOS",
            "status": "offline",
        }

        def get(self, _endpoint: str) -> object:
            return {"runners": [self.runner], "total_count": 1}

    receipt = verify_provider_runner_ready(
        plan=plan,
        api=Api(),
        verified_at_utc="2026-07-17T12:01:00+00:00",
    )
    assert receipt.runner_id == plan.runner_id
    assert receipt.busy is False

    for mutation in (
        {"busy": True},
        {"status": "online"},
        {"name": "substitute-runner"},
        {"labels": [{"name": value} for value in expected_labels[:-1]]},
    ):
        api = Api()
        api.runner = {**api.runner, **mutation}
        with pytest.raises(ExecutionClaimError, match="differs from the C1"):
            verify_provider_runner_ready(
                plan=plan,
                api=api,
                verified_at_utc="2026-07-17T12:01:00+00:00",
            )

    class DuplicateLabelApi(Api):
        def get(self, _endpoint: str) -> object:
            duplicate = {
                **self.runner,
                "id": self.runner["id"] + 1,
                "name": "another-runner",
            }
            return {"runners": [self.runner, duplicate], "total_count": 2}

    with pytest.raises(ExecutionClaimError, match="label is not unique"):
        verify_provider_runner_ready(
            plan=plan,
            api=DuplicateLabelApi(),
            verified_at_utc="2026-07-17T12:01:00+00:00",
        )

    class IncompleteApi(Api):
        def get(self, _endpoint: str) -> object:
            return {"runners": [self.runner], "total_count": 101}

    with pytest.raises(ExecutionClaimError, match="incomplete"):
        verify_provider_runner_ready(
            plan=plan,
            api=IncompleteApi(),
            verified_at_utc="2026-07-17T12:01:00+00:00",
        )


def test_runtime_request_rejects_extra_fields_and_symlink(tmp_path: Path) -> None:
    _, request, path = _request_fixture(tmp_path)
    payload = request.to_dict()
    payload["caller_argv"] = ["docker", "run", "mutable:latest"]
    path.unlink()
    _write(
        path,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n",
    )
    with pytest.raises(ProviderPhaseRuntimeError, match="schema differs"):
        load_provider_phase_runtime_request(path)

    path.unlink()
    target = tmp_path / "substitute.json"
    _write(target, request.canonical_file_bytes())
    path.symlink_to(target)
    with pytest.raises(ProviderPhaseRuntimeError, match="cannot open"):
        load_provider_phase_runtime_request(path)


def test_runtime_request_rejects_overlap_subset_and_mutable_driver_id(
    tmp_path: Path,
) -> None:
    _, request, _ = _request_fixture(tmp_path)
    with pytest.raises(ProviderPhaseRuntimeError, match="do not cover"):
        replace(request, drivers=request.drivers[:-1])
    with pytest.raises(ProviderPhaseRuntimeError, match="driver ID"):
        replace(
            request,
            drivers=(
                replace(request.drivers[0], driver_id="caller-selected-shell-v1"),
                *request.drivers[1:],
            ),
        )
    with pytest.raises(ProviderPhaseRuntimeError, match="overlap"):
        replace(request, phase_output_root=request.phase_input_root)


def test_analysis_runtime_bundle_requires_one_shared_claim_for_all_five() -> None:
    ordered = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
    bundle = AnalysisRuntimeClaimBundle(
        phase="analysis",
        receipts=tuple(_analysis_runtime_receipt(corpus_id) for corpus_id in ordered),
    )
    assert AnalysisRuntimeClaimBundle.from_bytes(bundle.canonical_file_bytes()) == bundle
    with pytest.raises(ProviderPhaseRuntimeError, match="ordered five"):
        replace(bundle, receipts=bundle.receipts[:-1])
    with pytest.raises(ProviderPhaseRuntimeError, match="cross provider claims"):
        replace(
            bundle,
            receipts=(
                bundle.receipts[0],
                replace(bundle.receipts[1], phase_claim_ledger_commit="4" * 40),
                *bundle.receipts[2:],
            ),
        )


def test_analysis_driver_reaches_closed_offline_container_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, phase="analysis")
    receipts = tuple(
        _analysis_runtime_receipt(corpus_id)
        for corpus_id in sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8"))
    )
    bundle = AnalysisRuntimeClaimBundle(phase="analysis", receipts=receipts)
    phase_input = tmp_path / "phase-input"
    control_path = phase_input / "controls" / "confirmatory-input-operator.json"
    results_root = tmp_path / "results"
    row = ProviderDriverRequest(
        corpus_id="all-five",
        driver_id="confirmatory-analysis-v1",
        control_path=str(control_path),
        control_file_sha256=_digest("control"),
        runtime_claim_receipt_path=str(phase_input / "runtime-claims" / "all-five.json"),
        runtime_claim_receipt_file_sha256=_digest(bundle.canonical_file_bytes()),
        output_root=str(results_root),
    )
    claimed = object.__new__(VerifiedProviderPredecessor)
    phase_claim = object.__new__(VerifiedPhaseClaimCapability)
    by_corpus = {receipt.corpus_id: receipt for receipt in receipts}
    monkeypatch.setattr(
        VerifiedPhaseClaimCapability,
        "require_input",
        lambda _self, **kwargs: by_corpus[kwargs["corpus_id"]],
    )
    config = object()
    monkeypatch.setattr(
        input_operator,
        "load_confirmatory_input_operator_config",
        lambda path: config if Path(path) == control_path else None,
    )
    observed: dict[str, object] = {}

    def run_offline(*args: object, **kwargs: object) -> SimpleNamespace:
        observed.update(args=args, kwargs=kwargs)
        return SimpleNamespace(
            candidate=SimpleNamespace(state="ANALYSIS_COMPLETE"),
            outcome=SimpleNamespace(),
        )

    monkeypatch.setattr(
        offline_provider,
        "run_provider_claimed_offline_analysis_once",
        run_offline,
    )

    def fresh() -> tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]:
        return claimed, phase_claim

    runtime_module._run_analysis(
        plan,
        row,
        bundle.canonical_file_bytes(),
        claimed,
        phase_claim,
        fresh,
    )

    assert observed["args"] == (config, plan, claimed, phase_claim)
    assert observed["kwargs"] == {
        "package_root": runtime_module.analysis_offline_package_root(plan),
        "results_root": results_root,
        "fresh_claim_supplier": fresh,
    }


@pytest.mark.parametrize("completed_members", (0, 2, 5))
def test_analysis_restart_admits_only_registered_store_prefixes(
    tmp_path: Path,
    completed_members: int,
) -> None:
    from fractal_ann_diagnostics.confirmatory_input_operator import (
        confirmatory_store_closure_filenames,
    )

    manifest_sha256 = _digest("analysis-restart-manifest")
    root = tmp_path / "analysis-results"
    root.mkdir(mode=0o700)
    registered = confirmatory_store_closure_filenames(manifest_sha256)
    admitted = (
        ()
        if completed_members == 0
        else (
            tuple(name for name in registered if ".confirmatory-input" in name)
            if completed_members == 2
            else registered
        )
    )
    for name in admitted:
        (root / name).write_bytes(b"retained-analysis-byte\n")

    assert (
        runtime_module.admit_analysis_results_store(
            root,
            manifest_sha256=manifest_sha256,
        )
        == admitted
    )

    (root / "foreign-result.json").write_bytes(b"foreign\n")
    with pytest.raises(
        runtime_module.ProviderPhaseRuntimeError,
        match="unregistered restart member",
    ):
        runtime_module.admit_analysis_results_store(
            root,
            manifest_sha256=manifest_sha256,
        )


def test_analysis_retry_reuses_only_the_same_closed_phase_receipt(
    tmp_path: Path,
) -> None:
    output = runtime_module.ProviderDriverOutput(
        corpus_id="all-five",
        driver_id="confirmatory-analysis-v1",
        output_root=str(tmp_path / "results"),
        output_tree_sha256=_digest("analysis-output-tree"),
        output_entries=("closed-result.json",),
        analysis_execution_receipt_uri=(
            tmp_path / "results" / "analysis-execution-receipt.json"
        ).as_uri(),
        analysis_execution_receipt_sha256=_digest("analysis-execution"),
        analysis_execution_receipt_file_sha256=_digest("analysis-execution-file"),
    )
    existing = runtime_module.ProviderPhaseExecutionReceipt(
        phase="analysis",
        suite_attempt_id=_digest("suite"),
        provider_plan_sha256=_digest("plan"),
        provider_plan_file_sha256=_digest("plan-file"),
        claim_receipt_file_sha256=_digest("claim-file"),
        runtime_request_sha256=_digest("attempt-one-request"),
        runtime_request_file_sha256=_digest("attempt-one-request-file"),
        outputs=(output,),
    )
    fresh = replace(
        existing,
        runtime_request_sha256=_digest("attempt-two-request"),
        runtime_request_file_sha256=_digest("attempt-two-request-file"),
    )
    target = tmp_path / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME
    _write(target, existing.canonical_file_bytes())

    observed = runtime_module._admit_existing_phase_execution_receipt(
        target,
        SimpleNamespace(phase="analysis"),
        fresh,
    )

    assert observed == existing
    assert observed.runtime_request_sha256 != fresh.runtime_request_sha256

    changed = replace(
        fresh,
        outputs=(
            replace(
                output,
                output_tree_sha256=_digest("changed-analysis-output-tree"),
            ),
        ),
    )
    with pytest.raises(ProviderPhaseRuntimeError, match="differs from fresh closure"):
        runtime_module._admit_existing_phase_execution_receipt(
            target,
            SimpleNamespace(phase="analysis"),
            changed,
        )


def test_execution_rehashes_every_bound_byte_and_writes_one_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, _ = _request_fixture(tmp_path)
    calls: list[str] = []

    def driver(row: ProviderDriverRequest, _: bytes) -> None:
        calls.append(row.corpus_id)
        output = Path(row.output_root)
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "result.json", b'{"ok":true}\n')

    monkeypatch.setitem(runtime_module._DRIVERS, "online", driver)
    receipt = execute_provider_phase_request(plan=plan, request=request)
    assert calls == sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8"))
    receipt_path = Path(request.phase_output_root) / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME
    assert receipt_path.read_bytes() == receipt.canonical_file_bytes()
    assert len(receipt.outputs) == len(FIXED_CORPORA)


def test_online_supplier_mints_immediately_before_each_delayed_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, _ = _request_fixture(tmp_path)
    clock = 0
    minted: list[tuple[str, int]] = []
    launched: list[tuple[str, int]] = []

    def supplier(row: ProviderDriverRequest) -> FreshOnlineClaimAuthority:
        capability = object.__new__(VerifiedRunClaimCapability)
        object.__setattr__(capability, "_test_minted_ns", clock)
        minted.append((row.corpus_id, clock))
        return FreshOnlineClaimAuthority(capability=capability, claim_bytes=b"fresh\n")

    def launch(
        row: ProviderDriverRequest,
        _: bytes,
        capability: VerifiedRunClaimCapability,
    ) -> None:
        nonlocal clock
        launched.append((row.corpus_id, clock))
        assert clock - capability._test_minted_ns == 0  # type: ignore[attr-defined]
        output = Path(row.output_root)
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "result.json", b'{"ok":true}\n')
        clock += 301 * 1_000_000_000

    monkeypatch.setattr(runtime_module, "_run_online", launch)
    Path(request.phase_output_root).mkdir(parents=True)
    execute_provider_phase_request(
        plan=plan,
        request=request,
        online_run_claim_supplier=supplier,
    )

    assert minted == launched
    assert len(minted) == 5
    assert minted[-1][1] == 4 * 301 * 1_000_000_000


def test_label_supplier_mints_and_marks_each_delayed_decryption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, _ = _request_fixture(tmp_path, phase="label-release")
    clock = 0
    minted: list[tuple[str, int]] = []
    decrypted: list[tuple[str, int]] = []

    def supplier(row: ProviderDriverRequest) -> FreshLabelClaimAuthority:
        capability, predecessor = _label_authority_parts(tmp_path, minted_ns=clock)
        marker = tmp_path / "markers" / f"{row.corpus_id}.json"
        marker.parent.mkdir(exist_ok=True)
        marker.write_bytes(f'{{"corpus_id":"{row.corpus_id}"}}\n'.encode())
        minted.append((row.corpus_id, clock))
        return FreshLabelClaimAuthority(
            capability=capability,
            predecessor=predecessor,
            claim_bytes=b"fresh\n",
            admission_marker_path=str(marker),
            admission_marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
        )

    def decrypt(
        row: ProviderDriverRequest,
        _: bytes,
        capability: VerifiedPhaseClaimCapability,
        predecessor: VerifiedProviderPredecessor,
        tle_runner: DockerTleDecryptRunner,
    ) -> object:
        nonlocal clock
        decrypted.append((row.corpus_id, clock))
        assert clock - capability._test_minted_ns == 0  # type: ignore[attr-defined]
        assert predecessor.state.record_sha256 == capability.phase_claim_state_sha256
        assert isinstance(tle_runner, DockerTleDecryptRunner)
        output = Path(row.output_root)
        _write(output / "released-labels.json", b'{"ok":true}\n')
        clock += 301 * 1_000_000_000
        return SimpleNamespace(receipt=SimpleNamespace())

    monkeypatch.setattr(runtime_module, "_run_label_release", decrypt)
    prepared: list[str] = []
    monkeypatch.setattr(
        DockerTleDecryptRunner,
        "prepare",
        lambda self: prepared.append(self.platform_image_reference),
    )
    monkeypatch.setattr(
        runtime_module,
        "_verify_pre_decryption_marker",
        lambda row, authority, **_: Path(authority.admission_marker_path).read_bytes(),
    )
    monkeypatch.setattr(
        runtime_module,
        "_close_label_release_action_authority",
        lambda **kwargs: _fake_label_output_authority(kwargs["row"]),
    )
    monkeypatch.setattr(
        runtime_module,
        "admit_label_release_phase_root",
        lambda *args, **kwargs: SimpleNamespace(
            completed_corpora=tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8"))),
            staged_corpus=None,
            execution_receipt_present=os.path.lexists(
                Path(args[0]) / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME
            ),
        ),
    )
    Path(request.phase_output_root).mkdir(parents=True)
    execute_provider_phase_request(
        plan=plan,
        request=request,
        label_phase_claim_supplier=supplier,
    )

    assert minted == decrypted
    assert prepared == [DockerTleDecryptRunner.from_plan(plan).platform_image_reference]
    assert len(minted) == 5
    assert minted[-1][1] == 4 * 301 * 1_000_000_000


def test_label_release_resumes_after_second_corpus_failure_without_redecrypting_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, _ = _request_fixture(tmp_path, phase="label-release")
    ordered = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
    decryptions = {corpus_id: 0 for corpus_id in ordered}
    supplier_calls = {corpus_id: 0 for corpus_id in ordered}
    fail_second = True

    def supplier(row: ProviderDriverRequest) -> FreshLabelClaimAuthority:
        supplier_calls[row.corpus_id] += 1
        capability, predecessor = _label_authority_parts(tmp_path)
        marker = tmp_path / "markers" / f"{row.corpus_id}.json"
        _write(marker, f'{{"corpus_id":"{row.corpus_id}"}}\n'.encode())
        return FreshLabelClaimAuthority(
            capability=capability,
            predecessor=predecessor,
            claim_bytes=b"fresh\n",
            admission_marker_path=str(marker),
            admission_marker_sha256=_digest(marker.read_bytes()),
        )

    def release(
        row: ProviderDriverRequest,
        *_: object,
    ) -> object:
        nonlocal fail_second
        root = Path(row.output_root)
        if root.exists():
            assert {path.name for path in root.iterdir()} == {
                "released-labels.json",
                "timelock-decryption-receipt.json",
            }
            return SimpleNamespace(receipt=SimpleNamespace())
        if row.corpus_id == ordered[1] and fail_second:
            fail_second = False
            raise ProviderPhaseRuntimeError("synthetic second-corpus failure")
        decryptions[row.corpus_id] += 1
        _write(root / "released-labels.json", f"{row.corpus_id}\n".encode())
        _write(
            root / "timelock-decryption-receipt.json",
            f'{{"corpus_id":"{row.corpus_id}"}}\n'.encode(),
        )
        return SimpleNamespace(receipt=SimpleNamespace())

    def close_authority(**kwargs: object) -> LabelReleaseOutputAuthority:
        row = kwargs["row"]
        assert isinstance(row, ProviderDriverRequest)
        journal = Path(
            row.output_root
        ).parent / runtime_module.label_release_authority_journal_name(row.corpus_id)
        if not journal.exists():
            _write(journal, b'{"restart-authority":true}\n')
        return _fake_label_output_authority(row)

    monkeypatch.setattr(runtime_module, "_run_label_release", release)
    monkeypatch.setattr(
        runtime_module,
        "_close_label_release_action_authority",
        close_authority,
    )
    monkeypatch.setattr(
        runtime_module,
        "_verify_pre_decryption_marker",
        lambda row, authority, **_: Path(authority.admission_marker_path).read_bytes(),
    )
    monkeypatch.setattr(DockerTleDecryptRunner, "prepare", lambda self: None)
    Path(request.phase_output_root).mkdir(parents=True)

    with pytest.raises(
        ProviderPhaseRuntimeError,
        match="synthetic second-corpus failure",
    ):
        execute_provider_phase_request(
            plan=plan,
            request=request,
            label_phase_claim_supplier=supplier,
        )

    phase_root = Path(request.phase_output_root)
    assert {path.name for path in phase_root.iterdir()} == {
        ordered[0],
        runtime_module.label_release_authority_journal_name(ordered[0]),
    }

    receipt = execute_provider_phase_request(
        plan=plan,
        request=request,
        label_phase_claim_supplier=supplier,
    )

    assert decryptions == {corpus_id: 1 for corpus_id in ordered}
    assert supplier_calls[ordered[0]] == 2
    assert {path.name for path in phase_root.iterdir()} == {
        *ordered,
        PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME,
    }
    assert tuple(output.corpus_id for output in receipt.outputs) == ordered
    for corpus_id in ordered:
        assert {path.name for path in (phase_root / corpus_id).iterdir()} == {
            "released-labels.json",
            "timelock-decryption-receipt.json",
        }


def test_real_pre_decryption_marker_admits_semantic_action_hashes_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, _ = _request_fixture(tmp_path, phase="label-release")
    reached: list[str] = []

    def supplier(row: ProviderDriverRequest) -> FreshLabelClaimAuthority:
        capability, predecessor = _label_authority_parts(tmp_path)
        binding = SimpleNamespace(
            corpus_id=row.corpus_id,
            input_uri=f"file:///ciphertext/{row.corpus_id}.tlock",
            input_sha256=_digest(f"ciphertext:{row.corpus_id}"),
            supporting_input_uri=f"file:///receipts/{row.corpus_id}.json",
            supporting_input_sha256=_digest(f"support:{row.corpus_id}"),
            output_uri=(Path(row.output_root) / "released-labels.json").as_uri(),
        )
        contract = SimpleNamespace(
            corpora=(binding,),
            phase_output_identity=_digest("phase-output"),
        )
        object.__setattr__(capability, "contract", contract)
        runtime = PhaseRuntimeClaimReceipt(
            phase="label-release",
            manifest_sha256=_digest("manifest"),
            run_receipt_sha256=_digest("run"),
            c1_commit="2" * 40,
            phase_claim_contract_sha256=_digest("phase-contract"),
            phase_claim_state_sha256=capability.phase_claim_state_sha256,
            phase_claim_ledger_commit=capability.phase_claim_ledger_commit,
            provider_identity_sha256=_digest("provider"),
            live_execute_job_receipt_sha256=_digest("semantic-live-receipt"),
            execute_job_id=101,
            phase_input_aggregate_sha256=_digest("phase-input"),
            phase_output_identity=contract.phase_output_identity,
            corpus_id=row.corpus_id,
            input_uri=binding.input_uri,
            input_sha256=binding.input_sha256,
            supporting_input_uri=binding.supporting_input_uri,
            supporting_input_sha256=binding.supporting_input_sha256,
            phase_beacon_receipt_sha256=_digest("semantic-beacon-receipt"),
        )
        claim_bytes = runtime.canonical_file_bytes()
        marker_row = {
            "admitted_at_utc": "2026-07-28T12:00:00+00:00",
            "beacon_receipt_sha256": runtime.phase_beacon_receipt_sha256,
            "corpus_id": row.corpus_id,
            "input_sha256": binding.input_sha256,
            "input_uri": binding.input_uri,
            "live_execute_job_receipt_sha256": (runtime.live_execute_job_receipt_sha256),
            "output_identity_sha256": contract.phase_output_identity,
            "output_uri": binding.output_uri,
            "phase": "label-release",
            "phase_claim_contract_sha256": (runtime.phase_claim_contract_sha256),
            "phase_claim_ledger_commit": runtime.phase_claim_ledger_commit,
            "phase_claim_state_sha256": runtime.phase_claim_state_sha256,
            "provider_identity_sha256": runtime.provider_identity_sha256,
            "runtime_claim_receipt_sha256": hashlib.sha256(claim_bytes).hexdigest(),
            "schema_version": "fractal-pre-decryption-admission-v1",
            "suite_attempt_id": request.suite_attempt_id,
            "supporting_input_sha256": binding.supporting_input_sha256,
            "supporting_input_uri": binding.supporting_input_uri,
        }
        marker_bytes = (
            json.dumps(
                marker_row,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        marker = tmp_path / "markers" / f"{row.corpus_id}.json"
        _write(marker, marker_bytes)
        return FreshLabelClaimAuthority(
            capability=capability,
            predecessor=predecessor,
            claim_bytes=claim_bytes,
            admission_marker_path=str(marker),
            admission_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
        )

    def release(row: ProviderDriverRequest, *_: object) -> object:
        reached.append(row.corpus_id)
        _write(
            Path(row.output_root) / "released-labels.json",
            b'{"reached":true}\n',
        )
        return SimpleNamespace(receipt=SimpleNamespace())

    monkeypatch.setattr(runtime_module, "_run_label_release", release)
    monkeypatch.setattr(
        runtime_module,
        "_close_label_release_action_authority",
        lambda **kwargs: _fake_label_output_authority(kwargs["row"]),
    )
    monkeypatch.setattr(
        runtime_module,
        "admit_label_release_phase_root",
        lambda *args, **kwargs: SimpleNamespace(
            completed_corpora=tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8"))),
            staged_corpus=None,
            execution_receipt_present=os.path.lexists(
                Path(args[0]) / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME
            ),
        ),
    )
    monkeypatch.setattr(DockerTleDecryptRunner, "prepare", lambda self: None)
    Path(request.phase_output_root).mkdir(parents=True)

    execute_provider_phase_request(
        plan=plan,
        request=request,
        label_phase_claim_supplier=supplier,
    )

    assert reached == list(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))


def test_docker_tle_runner_anonymously_prepares_exact_c1_platform_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, phase="label-release")
    runner = DockerTleDecryptRunner.from_plan(plan)
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        DockerTleDecryptRunner,
        "_verify_docker_client",
        lambda _self: None,
    )

    def run(
        executable: Path,
        arguments: tuple[str, ...],
        *,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> None:
        config = Path(arguments[1])
        assert arguments[0] == "--config"
        assert config.is_dir()
        assert tuple(config.iterdir()) == ()
        assert config.stat().st_mode & 0o777 == 0o700
        observed.update(
            executable=executable,
            arguments=arguments,
            config=config,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    monkeypatch.setattr(runtime_module, "_run_bounded_docker_control", run)
    runner.prepare()

    arguments = observed["arguments"]
    assert isinstance(arguments, tuple)
    assert arguments[2:] == (
        "pull",
        "--quiet",
        "--platform=linux/arm64",
        runner.platform_image_reference,
    )
    assert observed["executable"] == Path(plan.host_tools.docker_resolved_executable)
    assert observed["timeout_seconds"] == 600
    assert observed["max_output_bytes"] == 64 * 1024
    assert not Path(observed["config"]).exists()  # type: ignore[arg-type]
    joined = "\0".join(arguments)
    for forbidden in (
        "--env",
        "--mount",
        "--volume",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "RUNNER_TOKEN",
    ):
        assert forbidden not in joined


def test_docker_tle_runner_revalidates_invocation_symlink_and_client_bytes(
    tmp_path: Path,
) -> None:
    client = tmp_path / "docker-client"
    client.write_bytes(b"pinned Docker client")
    invocation = tmp_path / "docker"
    invocation.symlink_to(client)
    runner = replace(
        DockerTleDecryptRunner.from_plan(_plan(tmp_path, phase="label-release")),
        docker_executable=str(invocation),
        docker_resolved_executable=str(client),
        docker_executable_sha256=_digest(client.read_bytes()),
    )

    runner._verify_docker_client()
    with pytest.raises(ProviderPhaseRuntimeError, match="outside its C1 binding"):
        replace(
            runner,
            docker_resolved_executable=str(tmp_path / "another-client"),
        )._verify_docker_client()


def test_bounded_docker_control_rejects_excess_output() -> None:
    with pytest.raises(ProviderPhaseRuntimeError, match="output bound"):
        runtime_module._run_bounded_docker_control(
            Path("/usr/bin/printf"),
            ("12345",),
            timeout_seconds=10,
            max_output_bytes=4,
        )


def test_docker_tle_runner_uses_c1_image_and_isolated_bounded_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, phase="label-release")
    runner = DockerTleDecryptRunner.from_plan(plan)
    binary = tmp_path / "tle"
    binary.write_bytes(b"host pin only")
    ciphertext = b"ciphertext-through-stdin-only"
    plaintext = b"plaintext-from-stdout-only"
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        DockerTleDecryptRunner,
        "_verify_docker_client",
        lambda _self: None,
    )
    monkeypatch.setattr(
        runtime_module,
        "digest_regular_file",
        lambda *_args, **_kwargs: runner.tle_binary_sha256,
    )

    def create(
        executable: Path,
        arguments: tuple[str, ...],
        **kwargs: object,
    ) -> bytes:
        observed["create_executable"] = executable
        observed["create_arguments"] = arguments
        observed["create_bounds"] = kwargs
        return b"a" * 64 + b"\n"

    monkeypatch.setattr(runtime_module, "_run_bounded_docker_control", create)
    monkeypatch.setattr(
        runtime_module,
        "_force_remove_docker_container",
        lambda *args, **kwargs: observed.update(cleanup=(args, kwargs)),
    )

    def run(
        executable: Path,
        arguments: tuple[str, ...],
        stdin: bytes,
        timeout_seconds: int,
        max_plaintext_bytes: int,
    ) -> bytes:
        observed.update(
            executable=executable,
            arguments=arguments,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
            max_plaintext_bytes=max_plaintext_bytes,
        )
        return plaintext

    monkeypatch.setattr(runtime_module, "_run_pinned_tle_decrypt", run)
    result = runner(
        binary,
        (
            "--decrypt",
            "--network=https://api.drand.sh/",
            f"--chain={'a' * 64}",
        ),
        ciphertext,
        60,
        64 * 1024 * 1024,
    )

    arguments = observed["arguments"]
    create_arguments = observed["create_arguments"]
    assert isinstance(arguments, tuple)
    assert isinstance(create_arguments, tuple)
    assert result == plaintext
    assert observed["executable"] == Path(plan.host_tools.docker_resolved_executable)
    assert observed["create_executable"] == Path(plan.host_tools.docker_resolved_executable)
    assert observed["stdin"] == ciphertext
    assert observed["timeout_seconds"] == 60
    assert observed["max_plaintext_bytes"] == 64 * 1024 * 1024
    assert runner.index_image_reference == plan.runtime_image
    assert runner.oci_index_digest == plan.oci_index_digest
    assert runner.oci_platform_manifest_digest == plan.oci_platform_manifest_digest
    assert runner.platform_image_reference in create_arguments
    config = Path(create_arguments[1])
    assert create_arguments[0] == "--config"
    assert not config.exists()
    for required in (
        "create",
        "--interactive",
        "--rm",
        "--pull=never",
        "--log-driver=none",
        "--network=bridge",
        "--platform=linux/arm64",
        "--user=65532:65532",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        "--entrypoint=/usr/local/bin/tle",
    ):
        assert required in create_arguments
    container_names = [
        value.removeprefix("--name=") for value in create_arguments if value.startswith("--name=")
    ]
    assert len(container_names) == 1
    assert re.fullmatch(r"fractal-tle-[0-9a-f]{32}", container_names[0])
    assert arguments == (
        "--config",
        str(config),
        "container",
        "start",
        "--attach",
        "--interactive",
        "a" * 64,
    )
    joined = "\0".join(create_arguments + arguments)
    for forbidden in (
        "--env",
        "--env-file",
        "--mount",
        "--volume",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "RUNNER_TOKEN",
        str(binary),
        ciphertext.decode(),
    ):
        assert forbidden not in joined


def test_docker_tle_runner_rejects_unbounded_time_or_plaintext(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = DockerTleDecryptRunner.from_plan(_plan(tmp_path, phase="label-release"))
    binary = tmp_path / "tle"
    binary.write_bytes(b"host pin only")
    monkeypatch.setattr(
        DockerTleDecryptRunner,
        "_verify_docker_client",
        lambda _self: None,
    )
    monkeypatch.setattr(
        runtime_module,
        "digest_regular_file",
        lambda *_args, **_kwargs: runner.tle_binary_sha256,
    )
    calls: list[bool] = []
    monkeypatch.setattr(
        runtime_module,
        "_run_pinned_tle_decrypt",
        lambda *_args, **_kwargs: calls.append(True),
    )
    arguments = (
        "--decrypt",
        "--network=https://api.drand.sh/",
        f"--chain={'a' * 64}",
    )

    with pytest.raises(ProviderPhaseRuntimeError, match="timeout"):
        runner(binary, arguments, b"ciphertext", 61, 1024)
    with pytest.raises(ProviderPhaseRuntimeError, match="plaintext bound"):
        runner(binary, arguments, b"ciphertext", 60, 1024 * 1024 * 1024 + 1)
    assert calls == []


def test_docker_tle_runner_force_removes_container_after_attached_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = DockerTleDecryptRunner.from_plan(_plan(tmp_path, phase="label-release"))
    binary = tmp_path / "tle"
    binary.write_bytes(b"host pin only")
    monkeypatch.setattr(
        DockerTleDecryptRunner,
        "_verify_docker_client",
        lambda _self: None,
    )
    monkeypatch.setattr(
        runtime_module,
        "digest_regular_file",
        lambda *_args, **_kwargs: runner.tle_binary_sha256,
    )
    monkeypatch.setattr(
        runtime_module,
        "_run_bounded_docker_control",
        lambda *_args, **_kwargs: b"b" * 64 + b"\n",
    )
    monkeypatch.setattr(
        runtime_module,
        "_run_pinned_tle_decrypt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("attached client failed")),
    )
    removed: list[tuple[Path, Path, str]] = []

    def remove(
        executable: Path,
        *,
        config: Path,
        container_name: str,
    ) -> None:
        removed.append((executable, config, container_name))

    monkeypatch.setattr(runtime_module, "_force_remove_docker_container", remove)
    with pytest.raises(RuntimeError, match="attached client failed"):
        runner(
            binary,
            (
                "--decrypt",
                "--network=https://api.drand.sh/",
                f"--chain={'a' * 64}",
            ),
            b"ciphertext",
            60,
            1024,
        )
    assert len(removed) == 1
    assert removed[0][0] == Path(runner.docker_resolved_executable)
    assert not removed[0][1].exists()
    assert re.fullmatch(r"fractal-tle-[0-9a-f]{32}", removed[0][2])


def test_force_remove_docker_container_removes_volumes_and_proves_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path("/Applications/Docker.app/Contents/Resources/bin/docker")
    config = Path("/private/tmp/anonymous-docker-config")
    observed: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr(
        runtime_module,
        "_run_quiet_docker_status",
        lambda _executable, arguments: observed.append(("remove", arguments)) or 0,
    )
    monkeypatch.setattr(
        runtime_module,
        "_assert_docker_container_absent",
        lambda _executable, *, config, container_name: observed.append(
            ("absent", (str(config), container_name))
        ),
    )

    runtime_module._force_remove_docker_container(
        executable,
        config=config,
        container_name="fractal-tle-deadbeef",
    )

    assert observed == [
        (
            "remove",
            (
                "--config",
                str(config),
                "container",
                "rm",
                "--force",
                "--volumes",
                "fractal-tle-deadbeef",
            ),
        ),
        ("absent", (str(config), "fractal-tle-deadbeef")),
    ]


def test_label_driver_calls_release_api_with_live_predecessor_and_exact_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, phase="label-release")
    runner = DockerTleDecryptRunner.from_plan(plan)
    phase_claim, predecessor = _label_authority_parts(tmp_path)
    namespace = predecessor.namespace
    output_root = tmp_path / "released" / "scifact"
    output_root.parent.mkdir(parents=True)
    control = LabelReleaseDriverControl(
        manifest_path=str(tmp_path / "study-manifest.json"),
        custody_seal_path=str(tmp_path / "custody-seal.json"),
        encryption_receipt_path=str(tmp_path / "encryption.json"),
        completion_receipt_path=str(
            namespace / "completion" / "scifact-prediction-completion.json"
        ),
        completion_anchor_record_path=str(
            namespace / "completion" / "scifact-prediction-completion-anchor.json"
        ),
        completion_anchor_receipt_path=str(
            namespace / "completion" / "scifact-prediction-completion-anchor-receipt.json"
        ),
        suite_namespace=str(namespace),
        ciphertext_path=str(tmp_path / "labels.tlock"),
        tle_binary_path=str(tmp_path / "tle"),
        plaintext_output_path=str(output_root / "released-labels.json"),
        decryption_receipt_path=str(output_root / "timelock-decryption-receipt.json"),
    )
    control_bytes = (
        runtime_module._canonical_bytes(
            {name: getattr(control, name) for name in control.__dataclass_fields__}
        )
        + b"\n"
    )
    control_path = tmp_path / "label-control.json"
    control_path.write_bytes(control_bytes)
    row = ProviderDriverRequest(
        corpus_id="scifact",
        driver_id="timelock-label-release-v1",
        control_path=str(control_path),
        control_file_sha256=hashlib.sha256(control_bytes).hexdigest(),
        runtime_claim_receipt_path=str(tmp_path / "runtime-claim.json"),
        runtime_claim_receipt_file_sha256=_digest("runtime-claim"),
        output_root=str(output_root),
    )
    claim_bytes = b"fresh-runtime-claim\n"
    runtime_receipt = SimpleNamespace(
        corpus_id="scifact",
        input_uri=(tmp_path / "labels.tlock").as_uri(),
        input_sha256=_digest("ciphertext"),
        supporting_input_uri=(tmp_path / "encryption.json").as_uri(),
        supporting_input_sha256=_digest("encryption"),
    )
    expected_runtime = SimpleNamespace(canonical_file_bytes=lambda: claim_bytes)
    monkeypatch.setattr(
        runtime_module,
        "_load_phase_runtime_claim",
        lambda *_args, **_kwargs: runtime_receipt,
    )
    monkeypatch.setattr(
        VerifiedPhaseClaimCapability,
        "require_input",
        lambda *_args, **_kwargs: expected_runtime,
    )
    manifest = object()
    custody = object()
    encryption = object()
    exact_receipt = object()
    release = SimpleNamespace(receipt=exact_receipt)
    monkeypatch.setattr(runtime_module, "load_study_manifest", lambda _path: manifest)
    monkeypatch.setattr(runtime_module, "load_custody_seal_receipt", lambda _path: custody)
    monkeypatch.setattr(
        runtime_module,
        "load_timelock_encryption_receipt",
        lambda _path: encryption,
    )
    observed: dict[str, object] = {}

    def revalidate_completion(
        claimed: object,
        claim: object,
    ) -> object:
        observed["completion_authority"] = (claimed, claim)
        return SimpleNamespace(
            completion_root=namespace / "completion",
        )

    monkeypatch.setattr(
        runtime_module,
        "revalidate_post_online_completion_authority",
        revalidate_completion,
    )

    def release_label(*args: object, **kwargs: object) -> object:
        observed["release_args"] = args
        observed["release_kwargs"] = kwargs
        return release

    monkeypatch.setattr(runtime_module, "release_timelock_label", release_label)
    monkeypatch.setattr(
        runtime_module,
        "load_timelock_decryption_receipt",
        lambda _path: exact_receipt,
    )

    runtime_module._run_label_release(
        row,
        claim_bytes,
        phase_claim,
        predecessor,
        runner,
    )

    release_args = observed["release_args"]
    release_kwargs = observed["release_kwargs"]
    assert release_args == (manifest,)
    assert isinstance(release_kwargs, dict)
    assert release_kwargs["verified_suite_completion"] is predecessor
    assert release_kwargs["verified_phase_claim"] is phase_claim
    assert release_kwargs["trusted_tle_runner"] is runner
    assert release_kwargs["decryption_receipt_output_path"] == control.decryption_receipt_path
    assert (
        release_kwargs["verified_post_online_completion"].completion_root
        == namespace / "completion"
    )
    assert release_kwargs["custody_seal"] is custody
    assert release_kwargs["encryption_receipt"] is encryption
    assert observed["completion_authority"] == (predecessor, phase_claim)


def test_label_tip_change_stops_before_any_plaintext_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, _ = _request_fixture(tmp_path, phase="label-release")
    decryptions: list[str] = []
    monkeypatch.setattr(
        runtime_module,
        "_run_label_release",
        lambda row, *_: decryptions.append(row.corpus_id),
    )
    monkeypatch.setattr(DockerTleDecryptRunner, "prepare", lambda _self: None)

    def changed_tip(_: ProviderDriverRequest) -> FreshLabelClaimAuthority:
        raise ExecutionClaimError("LABEL_RELEASE_CLAIMED provider tip changed")

    Path(request.phase_output_root).mkdir(parents=True)
    with pytest.raises(ExecutionClaimError, match="tip changed"):
        execute_provider_phase_request(
            plan=plan,
            request=request,
            label_phase_claim_supplier=changed_tip,
        )
    assert decryptions == []
    assert not os.path.lexists(request.drivers[0].output_root)


def test_label_release_rejects_missing_fresh_supplier_before_decryption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, _ = _request_fixture(tmp_path, phase="label-release")
    decryptions: list[str] = []
    monkeypatch.setitem(
        runtime_module._DRIVERS,
        "label-release",
        lambda row, *_: decryptions.append(row.corpus_id),
    )

    with pytest.raises(ProviderPhaseRuntimeError, match="fresh claim supplier"):
        execute_provider_phase_request(plan=plan, request=request)

    assert decryptions == []
    assert not Path(request.phase_output_root).exists()


def test_label_release_driver_requires_in_memory_capability_before_claim_access(
    tmp_path: Path,
) -> None:
    _, request, _ = _request_fixture(tmp_path, phase="label-release")

    with pytest.raises(ProviderPhaseRuntimeError, match="in-memory claim authority"):
        runtime_module._run_label_release(
            request.drivers[0],
            b"not-a-runtime-claim",
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
        )


def test_label_release_entrypoint_cannot_execute_retained_claim_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan, request, _ = _request_fixture(tmp_path, phase="label-release")
    decryptions: list[str] = []
    monkeypatch.setitem(
        runtime_module._DRIVERS,
        "label-release",
        lambda row, *_: decryptions.append(row.corpus_id),
    )
    arguments = [
        plan.activation_command_id,
        "--provider-plan",
        plan.provider_plan_path,
        "--suite-attempt-id",
        plan.suite_attempt_id,
        "--claim-receipt",
        request.claim_receipt_path,
        "--phase-input-root",
        request.phase_input_root,
        "--phase-output-root",
        request.phase_output_root,
    ]

    assert main(arguments) == 1
    assert "label runtime lacks a fresh claim supplier" in capsys.readouterr().err
    assert decryptions == []
    assert not Path(request.phase_output_root).exists()


def test_execution_rejects_changed_control_before_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, _ = _request_fixture(tmp_path)
    Path(request.drivers[0].control_path).write_bytes(b'{"changed":true}\n')
    calls: list[str] = []
    monkeypatch.setitem(
        runtime_module._DRIVERS,
        "online",
        lambda row, _: calls.append(row.corpus_id),
    )
    with pytest.raises(ProviderPhaseRuntimeError, match="differs from the request digest"):
        execute_provider_phase_request(plan=plan, request=request)
    assert calls == []


def test_entrypoint_has_no_argv_or_environment_override_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, _ = _request_fixture(tmp_path)

    def driver(row: ProviderDriverRequest, _: bytes) -> None:
        output = Path(row.output_root)
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "result.json", b'{"ok":true}\n')

    monkeypatch.setitem(runtime_module._DRIVERS, "online", driver)
    arguments = [
        plan.activation_command_id,
        "--provider-plan",
        plan.provider_plan_path,
        "--suite-attempt-id",
        plan.suite_attempt_id,
        "--claim-receipt",
        request.claim_receipt_path,
        "--phase-input-root",
        request.phase_input_root,
        "--phase-output-root",
        request.phase_output_root,
    ]
    assert main(arguments) == 0
    with pytest.raises(SystemExit):
        main([*arguments, "--runtime-image", "caller.example/mutable:latest"])


def test_entrypoint_admits_local_bootstrap_before_phase_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, request, _ = _request_fixture(tmp_path)
    Path(plan.runner_bootstrap_receipt_path).unlink()
    phase_input_opened = False

    def forbidden_phase_input(_: Path) -> ProviderPhaseRuntimeRequest:
        nonlocal phase_input_opened
        phase_input_opened = True
        raise AssertionError("phase input opened before runner bootstrap admission")

    monkeypatch.setattr(
        runtime_module,
        "load_provider_phase_runtime_request",
        forbidden_phase_input,
    )
    assert (
        main(
            [
                plan.activation_command_id,
                "--provider-plan",
                plan.provider_plan_path,
                "--suite-attempt-id",
                plan.suite_attempt_id,
                "--claim-receipt",
                request.claim_receipt_path,
                "--phase-input-root",
                request.phase_input_root,
                "--phase-output-root",
                request.phase_output_root,
            ]
        )
        == 1
    )
    assert phase_input_opened is False
