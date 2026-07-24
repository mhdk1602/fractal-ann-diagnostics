from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_execution_claim import _host_tools

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
    FreshLabelClaimAuthority,
    FreshOnlineClaimAuthority,
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


def _digest(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


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
        capability = object.__new__(VerifiedPhaseClaimCapability)
        object.__setattr__(capability, "_test_minted_ns", clock)
        marker = tmp_path / "markers" / f"{row.corpus_id}.json"
        marker.parent.mkdir(exist_ok=True)
        marker.write_bytes(f'{{"corpus_id":"{row.corpus_id}"}}\n'.encode())
        minted.append((row.corpus_id, clock))
        return FreshLabelClaimAuthority(
            capability=capability,
            claim_bytes=b"fresh\n",
            admission_marker_path=str(marker),
            admission_marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
        )

    def decrypt(
        row: ProviderDriverRequest,
        _: bytes,
        capability: VerifiedPhaseClaimCapability,
    ) -> None:
        nonlocal clock
        decrypted.append((row.corpus_id, clock))
        assert clock - capability._test_minted_ns == 0  # type: ignore[attr-defined]
        output = Path(row.output_root)
        _write(output / "released-labels.json", b'{"ok":true}\n')
        clock += 301 * 1_000_000_000

    monkeypatch.setattr(runtime_module, "_run_label_release", decrypt)
    monkeypatch.setattr(
        runtime_module,
        "_verify_pre_decryption_marker",
        lambda row, authority, **_: Path(authority.admission_marker_path).read_bytes(),
    )
    Path(request.phase_output_root).mkdir(parents=True)
    for row in request.drivers:
        Path(row.output_root).mkdir()
    execute_provider_phase_request(
        plan=plan,
        request=request,
        label_phase_claim_supplier=supplier,
    )

    assert minted == decrypted
    assert len(minted) == 5
    assert minted[-1][1] == 4 * 301 * 1_000_000_000


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

    def changed_tip(_: ProviderDriverRequest) -> FreshLabelClaimAuthority:
        raise ExecutionClaimError("LABEL_RELEASE_CLAIMED provider tip changed")

    Path(request.phase_output_root).mkdir(parents=True)
    for row in request.drivers:
        Path(row.output_root).mkdir()
    with pytest.raises(ExecutionClaimError, match="tip changed"):
        execute_provider_phase_request(
            plan=plan,
            request=request,
            label_phase_claim_supplier=changed_tip,
        )
    assert decryptions == []
    assert tuple(Path(request.drivers[0].output_root).iterdir()) == ()


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
