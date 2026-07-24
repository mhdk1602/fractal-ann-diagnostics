from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.confirmatory_input_operator as operator
import fractal_ann_diagnostics.production_controls as production_controls
import fractal_ann_diagnostics.provider_activation_factory as activation
import fractal_ann_diagnostics.provider_phase_runtime as runtime
import fractal_ann_diagnostics.runtime_attestation as runtime_attestation
import fractal_ann_diagnostics.sealed_container_launcher as sealed_launcher
from fractal_ann_diagnostics.execution_claim import (
    ACTIVATION_COMMON_OUTPUT_KEYS,
    ACTIVATION_PHASE_OUTPUT_KEYS,
    RuntimeClaimReceipt,
    VerifiedPhaseClaimCapability,
    VerifiedRunClaimCapability,
)
from fractal_ann_diagnostics.provider_activation_factory import (
    ProviderActivationError,
    ProviderActivationResult,
)
from fractal_ann_diagnostics.provider_phase_runtime import (
    OnlineSealedLaunchDriverControl,
    ProviderDriverRequest,
)
from fractal_ann_diagnostics.suite_attempt import VerifiedProviderPredecessor


def _digest(character: str) -> str:
    return character * 64


def _commit(character: str) -> str:
    return character * 40


def _output_rows(phase: str) -> tuple[tuple[str, str], ...]:
    keys = ACTIVATION_COMMON_OUTPUT_KEYS | ACTIVATION_PHASE_OUTPUT_KEYS[phase]
    return tuple(sorted((key, "value") for key in keys))


@pytest.mark.parametrize("phase", ["online", "label-release", "analysis"])
def test_activation_result_exposes_only_registered_phase_interface(phase: str) -> None:
    result = ProviderActivationResult(phase=phase, outputs=_output_rows(phase))

    assert set(result.output_fields()) == (
        ACTIVATION_COMMON_OUTPUT_KEYS | ACTIVATION_PHASE_OUTPUT_KEYS[phase]
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_activation_result_rejects_open_output_interfaces(mutation: str) -> None:
    rows = list(_output_rows("online"))
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(("caller_selected_output", "x"))
    else:
        rows.append(rows[0])

    with pytest.raises(ProviderActivationError, match="registered interface"):
        ProviderActivationResult(phase="online", outputs=tuple(rows))


def _claim_cross_check_fixture() -> dict[str, object]:
    state = SimpleNamespace(
        sequence=1,
        record_sha256=_digest("a"),
    )
    claimed = SimpleNamespace(state=state, ledger_commit=_commit("b"))
    contract = SimpleNamespace(
        contract_sha256=_digest("c"),
        unique_runner_label="fractal-ann-confirmatory-online-fixed",
        execute_job_name="execute-online",
        manifest_sha256=_digest("d"),
    )
    identity = SimpleNamespace(identity_sha256=_digest("e"))
    execute_identity = {
        "job": "execute",
        "runner_environment": "self-hosted",
        "runner_os": "macOS",
        "runner_arch": "ARM64",
    }
    context = SimpleNamespace(
        run_id=17,
        job="execute",
        identity_dict=lambda: dict(execute_identity),
    )
    claim_identity = {
        **execute_identity,
        "job": "claim",
        "runner_environment": "github-hosted",
        "runner_os": "Linux",
        "runner_arch": "X64",
    }
    claim_context_sha256 = hashlib.sha256(
        activation._canonical_file_bytes(claim_identity)[:-1]
    ).hexdigest()
    plan = SimpleNamespace(
        plan_sha256=_digest("0"),
        phase="online",
        suite_attempt_id=_digest("1"),
        manifest_sha256=_digest("d"),
    )
    receipt = SimpleNamespace(
        phase="online",
        suite_attempt_id=_digest("1"),
        run_id=17,
        workflow_context_sha256=claim_context_sha256,
        provider_plan_sha256=_digest("0"),
        provider_identity_sha256=_digest("e"),
        target_state="RUN_CLAIMED",
        target_sequence=1,
        target_state_record_sha256=_digest("a"),
        target_ledger_commit=_commit("b"),
        claim_contract_sha256=_digest("c"),
        runner_label="fractal-ann-confirmatory-online-fixed",
        expected_execute_job_name="execute-online",
        manifest_sha256=_digest("d"),
    )
    return {
        "phase": "online",
        "suite_attempt_id": _digest("1"),
        "context": context,
        "receipt": receipt,
        "plan": plan,
        "claimed": claimed,
        "contract": contract,
        "provider_identity": identity,
    }


def test_claim_cross_check_accepts_one_exact_authority_tuple() -> None:
    activation._cross_check_claim(**_claim_cross_check_fixture())


@pytest.mark.parametrize(
    "field, changed",
    [
        ("workflow_context_sha256", _digest("9")),
        ("provider_plan_sha256", _digest("8")),
        ("provider_identity_sha256", _digest("7")),
        ("target_state_record_sha256", _digest("6")),
        ("target_ledger_commit", _commit("5")),
        ("claim_contract_sha256", _digest("4")),
        ("runner_label", "fractal-ann-confirmatory-online-substitute"),
    ],
)
def test_claim_cross_check_rejects_artifact_state_or_runner_substitution(
    field: str,
    changed: str,
) -> None:
    values = _claim_cross_check_fixture()
    receipt = values["receipt"]
    setattr(receipt, field, changed)

    with pytest.raises(ProviderActivationError, match="differ"):
        activation._cross_check_claim(**values)


def test_label_control_derives_registered_completion_and_output_paths(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    controlled = root / "controlled"
    completion = controlled / "completion"
    output = root / "suite" / "label-release" / "scifact"
    ciphertext = root / "custody" / "scifact.tlock"
    encryption = root / "custody" / "scifact-encryption.json"
    plaintext = output / "released-labels.json"
    manifest_path = root / "study-manifest.json"
    custody = root / "custody" / "custody-seal.json"
    tle = root / "custody" / "tle"
    contract = SimpleNamespace(
        corpora=(
            SimpleNamespace(
                corpus_id="scifact",
                input_uri=ciphertext.as_uri(),
                supporting_input_uri=encryption.as_uri(),
                output_uri=plaintext.as_uri(),
            ),
        )
    )
    plan = SimpleNamespace(host_tools=SimpleNamespace(controlled_root=str(controlled)))
    manifest = {
        "artifacts": [
            {"role": "custody-seal-receipt", "uri": custody.as_uri()},
            {"role": "timelock-tool", "uri": tle.as_uri()},
        ]
    }

    control = activation._label_control(
        corpus_id="scifact",
        contract=contract,
        plan=plan,
        manifest_path=manifest_path,
        manifest=manifest,
        output_root=output,
    )

    assert control.completion_receipt_path == str(completion / "scifact-completion.json")
    assert control.completion_anchor_record_path == str(completion / "scifact-anchor-record.json")
    assert control.completion_anchor_receipt_path == str(completion / "scifact-anchor-receipt.json")
    assert control.plaintext_output_path == str(plaintext)
    assert control.decryption_receipt_path == str(output / "timelock-decryption-receipt.json")


def test_label_control_rejects_non_file_claim_input(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    contract = SimpleNamespace(
        corpora=(
            SimpleNamespace(
                corpus_id="scifact",
                input_uri="https://example.invalid/ciphertext",
                supporting_input_uri=(root / "encryption.json").as_uri(),
                output_uri=(
                    root / "suite" / "label-release" / "scifact" / "released-labels.json"
                ).as_uri(),
            ),
        )
    )
    manifest = {
        "artifacts": [
            {"role": "custody-seal-receipt", "uri": (root / "seal.json").as_uri()},
            {"role": "timelock-tool", "uri": (root / "tle").as_uri()},
        ]
    }

    with pytest.raises(ProviderActivationError, match="canonical local file URI"):
        activation._label_control(
            corpus_id="scifact",
            contract=contract,
            plan=SimpleNamespace(host_tools=SimpleNamespace(controlled_root=str(root))),
            manifest_path=root / "manifest.json",
            manifest=manifest,
            output_root=root / "output",
        )


def test_fresh_claim_reader_uses_new_derived_root_on_every_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    token = object()

    def materialize(phase: str, suite: str, parent: Path, **_: object) -> object:
        observed.append(parent)
        return SimpleNamespace(predecessor=token)

    monkeypatch.setattr(activation, "materialize_provider_claim", materialize)
    reader = activation._FreshClaimReader(
        phase="online",
        suite_attempt_id=_digest("1"),
        root=tmp_path.resolve(),
        github_api=object(),
        artifact_api=object(),
    )

    assert reader() is token
    assert reader() is token
    assert [path.name for path in observed] == ["read-0001", "read-0002"]
    assert observed[0] != observed[1]


def test_private_directory_rejects_symlink_or_world_writable_parent(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    controlled = root / "controlled"
    controlled.mkdir(mode=0o700)
    link = root / "linked-parent"
    link.symlink_to(controlled, target_is_directory=True)

    with pytest.raises(ProviderActivationError, match="parent is not controlled"):
        activation._new_private_directory(link / "activation", label="activation")

    writable = root / "world-writable"
    writable.mkdir(mode=0o777)
    writable.chmod(0o777)
    with pytest.raises(ProviderActivationError, match="parent is not controlled"):
        activation._new_private_directory(writable / "activation", label="activation")


def test_phase_or_analysis_output_replay_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    phase_output = root / "phase-output"
    phase_output.mkdir(mode=0o700)
    (phase_output / "foreign.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProviderActivationError, match="new absolute directory"):
        activation._new_private_directory(phase_output, label="phase evidence root")

    results_store = root / "results-store"
    results_store.mkdir(mode=0o700)
    (results_store / "prior-result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProviderActivationError, match="controlled empty directory"):
        activation._admit_empty_private_directory(
            results_store,
            label="analysis results store",
        )


def test_activation_rejects_caller_selected_claim_filename_before_transport(
    tmp_path: Path,
) -> None:
    context = SimpleNamespace(phase="online")

    with pytest.raises(ProviderActivationError, match="fixed filename"):
        activation.activate_and_execute_provider_phase(
            context=context,
            phase="online",
            suite_attempt_id=_digest("1"),
            artifact_id=1,
            artifact_digest="sha256:" + _digest("2"),
            expected_inventory_sha256=_digest("3"),
            claim_receipt_destination=tmp_path.resolve() / "substitute.json",
            output_dir=tmp_path.resolve() / "activation",
            github_api=object(),
            artifact_api=object(),
        )


def test_downloaded_claim_evidence_is_installed_only_at_plan_fixed_path(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    downloaded = root / "runner-temp" / "claim-receipt.json"
    packaged = root / "verified-artifact" / "claim-receipt.json"
    fixed = root / "controlled" / "suite" / "claim-receipt.json"
    downloaded.parent.mkdir()
    packaged.parent.mkdir()
    encoded = b'{"claim":"fixed"}\n'
    downloaded.write_bytes(encoded)
    packaged.write_bytes(encoded)
    plan = SimpleNamespace(claim_receipt_path=lambda _: str(fixed))

    observed = activation._materialize_fixed_claim_receipt(
        downloaded_path=downloaded,
        packaged_path=packaged,
        plan=plan,
        suite_attempt_id=_digest("1"),
    )

    assert observed == fixed
    assert observed.read_bytes() == encoded
    assert downloaded != observed


def test_hosted_claim_context_is_derived_from_execute_identity_without_aliasing() -> None:
    execute = {
        "phase": "online",
        "job": "execute",
        "run_id": 17,
        "runner_environment": "self-hosted",
        "runner_os": "macOS",
        "runner_arch": "ARM64",
    }
    context = SimpleNamespace(
        job="execute",
        identity_dict=lambda: dict(execute),
    )
    hosted = activation._hosted_claim_context_sha256(context)
    execute_sha256 = hashlib.sha256(activation._canonical_file_bytes(execute)[:-1]).hexdigest()

    assert hosted != execute_sha256
    expected = {
        **execute,
        "job": "claim",
        "runner_environment": "github-hosted",
        "runner_os": "Linux",
        "runner_arch": "X64",
    }
    assert hosted == hashlib.sha256(activation._canonical_file_bytes(expected)[:-1]).hexdigest()


def _runtime_receipt() -> RuntimeClaimReceipt:
    derived = _digest("1")
    return RuntimeClaimReceipt(
        manifest_sha256=_digest("2"),
        run_receipt_sha256=_digest("3"),
        c1_commit=_commit("4"),
        claim_contract_sha256=_digest("5"),
        claim_state_sha256=_digest("6"),
        claim_ledger_commit=_commit("7"),
        provider_identity_sha256=_digest("8"),
        live_execute_job_receipt_sha256=_digest("9"),
        execute_job_id=17,
        beacon_receipt_sha256=_digest("a"),
        beacon_bytes_sha256=_digest("b"),
        design_seed_sha256=_digest("c"),
        derived_seed_sha256=derived,
        permutation_seed=int.from_bytes(bytes.fromhex(derived)[:8], "big"),
        output_aggregate_identity=_digest("d"),
    )


def _online_driver_fixture(tmp_path: Path) -> tuple[ProviderDriverRequest, bytes]:
    root = tmp_path.resolve()
    output = root / "output" / "scifact"
    control = OnlineSealedLaunchDriverControl(
        preflight_contract_path=str(root / "preflight.json"),
        preflight_receipt_path=str(root / "preflight-receipt.json"),
        transition_receipt_path=str(root / "transition.json"),
        instantiation_receipt_path=str(root / "instantiation.json"),
        finalization_request_path=str(root / "finalization-request.json"),
        finalization_receipt_path=str(root / "finalization-receipt.json"),
        sealed_contract_path=str(root / "sealed.json"),
        volume_receipt_path=str(root / "volume.json"),
        audit_root=str(root / "sealed-evidence"),
    )
    Path(control.preflight_receipt_path).write_bytes(b"{}\n")
    control_path = root / "control.json"
    control_path.write_bytes(control.canonical_file_bytes())
    claim_bytes = _runtime_receipt().canonical_file_bytes()
    claim_path = root / "claim.json"
    claim_path.write_bytes(claim_bytes)
    return (
        ProviderDriverRequest(
            corpus_id="scifact",
            driver_id="sealed-online-corpus-v1",
            control_path=str(control_path),
            control_file_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),
            runtime_claim_receipt_path=str(claim_path),
            runtime_claim_receipt_file_sha256=hashlib.sha256(claim_bytes).hexdigest(),
            output_root=str(output),
        ),
        claim_bytes,
    )


def test_online_runtime_reaches_registered_sealed_launcher_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, claim_bytes = _online_driver_fixture(tmp_path)
    sentinels = {
        name: object()
        for name in ("preflight", "receipt", "transition", "instantiation", "closure", "volume")
    }
    sealed = SimpleNamespace(
        geometry=SimpleNamespace(corpus_id="scifact", copy_output_root=row.output_root)
    )
    monkeypatch.setattr(
        sealed_launcher, "load_preflight_launch_contract", lambda _: sentinels["preflight"]
    )
    monkeypatch.setattr(
        runtime_attestation, "loads_runtime_preflight_receipt", lambda _: sentinels["receipt"]
    )
    monkeypatch.setattr(
        sealed_launcher, "load_runtime_plan_transition", lambda _: sentinels["transition"]
    )
    monkeypatch.setattr(
        sealed_launcher, "load_registered_plan_instantiation", lambda _: sentinels["instantiation"]
    )
    monkeypatch.setattr(sealed_launcher, "load_sealed_launch_contract", lambda _: sealed)
    monkeypatch.setattr(
        sealed_launcher, "load_volume_initialization_receipt", lambda _: sentinels["volume"]
    )
    monkeypatch.setattr(
        production_controls,
        "verify_production_run_closure_authority",
        lambda **_: sentinels["closure"],
    )
    observed: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        sealed_launcher,
        "launch_sealed_once",
        lambda *args, **kwargs: observed.append((*args, kwargs)),
    )
    capability = object.__new__(VerifiedRunClaimCapability)

    runtime._run_online(row, claim_bytes, capability)

    assert len(observed) == 1
    assert observed[0][0] is sealed
    assert observed[0][7] is capability
    assert observed[0][-1]["secret"] == claim_bytes


def test_online_runtime_rejects_sealed_output_namespace_swap_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, claim_bytes = _online_driver_fixture(tmp_path)
    sentinel = object()
    monkeypatch.setattr(sealed_launcher, "load_preflight_launch_contract", lambda _: sentinel)
    monkeypatch.setattr(runtime_attestation, "loads_runtime_preflight_receipt", lambda _: sentinel)
    monkeypatch.setattr(sealed_launcher, "load_runtime_plan_transition", lambda _: sentinel)
    monkeypatch.setattr(
        production_controls, "verify_production_run_closure_authority", lambda **_: sentinel
    )
    monkeypatch.setattr(
        sealed_launcher,
        "load_sealed_launch_contract",
        lambda _: SimpleNamespace(
            geometry=SimpleNamespace(
                corpus_id="scifact",
                copy_output_root=str(tmp_path.resolve() / "substituted-output"),
            )
        ),
    )
    launched: list[bool] = []
    monkeypatch.setattr(
        sealed_launcher, "launch_sealed_once", lambda *_, **__: launched.append(True)
    )

    with pytest.raises(runtime.ProviderPhaseRuntimeError, match="output root"):
        runtime._run_online(row, claim_bytes, object.__new__(VerifiedRunClaimCapability))
    assert launched == []


def test_provider_analysis_adapter_reaches_candidate_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = SimpleNamespace(
        state="LABELS_RELEASED",
        sequence=4,
        suite_attempt_id=_digest("1"),
        record_sha256=_digest("2"),
    )
    claimed_state = SimpleNamespace(
        state="ANALYSIS_CLAIMED",
        sequence=5,
        manifest_sha256=_digest("3"),
        record_sha256=_digest("6"),
        run_receipt_sha256=_digest("4"),
    )
    claimed = object.__new__(VerifiedProviderPredecessor)
    object.__setattr__(claimed, "records", (labels, claimed_state))
    object.__setattr__(
        claimed,
        "evidences",
        (SimpleNamespace(transition_id=_commit("a")),),
    )
    object.__setattr__(claimed, "control_inventory_sha256", _digest("b"))
    object.__setattr__(claimed, "artifact_receipt_sha256", _digest("c"))
    object.__setattr__(claimed, "_fresh_revalidator", lambda: None)
    phase_claim = object.__new__(VerifiedPhaseClaimCapability)
    clock = 0
    minted = {id(phase_claim): clock}

    def assert_current(self: VerifiedPhaseClaimCapability) -> None:
        if clock - minted[id(self)] > 300 * 1_000_000_000:
            raise RuntimeError("phase capability is stale")

    monkeypatch.setattr(
        VerifiedPhaseClaimCapability,
        "assert_current",
        assert_current,
    )
    inputs = SimpleNamespace()
    materialized = SimpleNamespace(
        inputs=inputs,
        receipt=SimpleNamespace(artifact_sha256=_digest("5")),
    )
    monkeypatch.setattr(operator, "materialize_confirmatory_input", lambda *_, **__: materialized)
    monkeypatch.setattr(operator, "load_admitted_model_suite", lambda *_: object())

    def run_analysis(*_: object, **__: object) -> object:
        nonlocal clock
        clock += 361 * 1_000_000_000
        return SimpleNamespace(confirmatory_input_artifact_sha256=_digest("5"))

    monkeypatch.setattr(operator, "run_confirmatory_analysis_once", run_analysis)
    monkeypatch.setattr(operator, "confirmatory_attempt_path", lambda _: tmp_path / "attempt.json")
    monkeypatch.setattr(
        operator, "confirmatory_result_receipt_path", lambda _: tmp_path / "result-receipt.json"
    )
    monkeypatch.setattr(operator, "confirmatory_result_path", lambda _: tmp_path / "result.json")
    observed: dict[str, object] = {}

    def complete(token: object, **kwargs: object) -> object:
        observed.update(token=token, **kwargs)
        return SimpleNamespace(state="ANALYSIS_COMPLETE")

    monkeypatch.setattr(operator, "complete_confirmatory_analysis", complete)
    fresh_phase_claim = object.__new__(VerifiedPhaseClaimCapability)

    def refresh() -> tuple[VerifiedProviderPredecessor, VerifiedPhaseClaimCapability]:
        minted[id(fresh_phase_claim)] = clock
        return claimed, fresh_phase_claim

    candidate = operator.run_provider_claimed_confirmatory_analysis_once(
        SimpleNamespace(),
        claimed,
        phase_claim,
        fresh_claim_supplier=refresh,
    )

    assert candidate.state == "ANALYSIS_COMPLETE"
    assert observed["token"] is claimed
    assert observed["phase_claim"] is fresh_phase_claim
    assert clock == 361 * 1_000_000_000
