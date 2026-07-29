from __future__ import annotations

import copy
import hashlib
import json
import shutil
import signal
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_execution_claim import _beacon, _host_tools
from test_provider_rehearsal import _closure
from test_study import _candidate_rehearsal_manifest

import fractal_ann_diagnostics.provider_plan_operator as operator
import fractal_ann_diagnostics.provider_runner_activation as runner_operator
from fractal_ann_diagnostics.execution_claim import (
    ANALYSIS_PHASE,
    LABEL_RELEASE_PHASE,
    ONLINE_PHASE,
    derive_phase_runner_label,
    load_provider_phase_plans,
    required_execute_runner_labels,
)
from fractal_ann_diagnostics.production_workload_registration import (
    production_workload_file_sha256,
)
from fractal_ann_diagnostics.study import (
    C0_COMMIT_SENTINEL,
    resolve_candidate_provider_plan_commit_bindings,
)

PHASES = (ONLINE_PHASE, LABEL_RELEASE_PHASE, ANALYSIS_PHASE)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_private(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(encoded)
    path.chmod(0o600)


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    controlled = tmp_path / "controlled"
    controlled.mkdir(mode=0o700)
    host_tools = _host_tools(controlled)
    host_sources = operator.HostToolSources(
        controlled_root=controlled,
        python_executable=controlled / "python/bin/python3.12",
        venv_root=controlled / "venv",
        python_import_root=controlled / "venv/lib/python3.12/site-packages",
        gh_executable=controlled / "gh/bin/gh",
        runner_listener_executable=controlled / "runner/bin/Runner.Listener",
        runner_listener_dll=controlled / "runner/bin/Runner.Listener.dll",
        runner_config_executable=controlled / "runner/config.sh",
        runner_run_executable=controlled / "runner/run.sh",
        docker_executable=tmp_path / "docker",
        host_probe_path=tmp_path / "phase-host-probe.json",
        docker_server_probe_path=tmp_path / "docker-server-probe.json",
    )
    monkeypatch.setattr(
        operator,
        "_derive_host_tool_contract",
        lambda sources, **kwargs: (host_tools, "1" * 40),
    )
    candidate_source_root = tmp_path / "candidate-source"
    candidate_source_root.mkdir(mode=0o700)

    closure = _closure()
    closure_path = tmp_path / "candidate-image-closure.json"
    _write_private(closure_path, _canonical(closure.to_dict()) + b"\n")

    config_path = tmp_path / "production-control-config.json"
    config_encoded = _canonical({"record": "production-control-config"}) + b"\n"
    _write_private(config_path, config_encoded)
    config_sha256 = _digest(config_encoded)

    config_receipt_path = tmp_path / "production-control-config-write-receipt.json"
    _write_private(
        config_receipt_path,
        _canonical({"record": "production-control-config-write-receipt"}) + b"\n",
    )
    control_blueprint_path = tmp_path / "production-control-blueprint.json"
    control_blueprint_encoded = _canonical({"record": "production-control-blueprint"}) + b"\n"
    _write_private(control_blueprint_path, control_blueprint_encoded)
    factory_path = tmp_path / "factory-config.json"
    factory_encoded = _canonical({"record": "factory-config"}) + b"\n"
    _write_private(factory_path, factory_encoded)

    production_reference = (
        "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@"
        f"{closure.scientific_image_index_digest}"
    )
    source = _candidate_rehearsal_manifest()
    sealed = source["sealed_execution"]
    assert isinstance(sealed, dict)
    sealed["provider_phase_plans"] = "tbd"
    sealed["runner_image"] = production_reference
    workloads = source["production_workloads"]
    assert isinstance(workloads, list)
    for row in workloads:
        assert isinstance(row, dict)
        spec = row["spec"]
        assert isinstance(spec, dict)
        spec["runner_image"] = production_reference
        row["canonical_file_sha256"] = production_workload_file_sha256(spec)

    runner_identity = sealed["runner_identity"]
    control_blueprint_sha256 = _digest(control_blueprint_encoded)
    control_blueprint_semantic_sha256 = "7" * 64
    sealed["production_controls"] = {
        "blueprint_receipt_file_sha256": control_blueprint_sha256,
        "blueprint_receipt_sha256": control_blueprint_semantic_sha256,
        "materialization_config_file_sha256": config_sha256,
    }
    source_path = tmp_path / "candidate-source-shell.json"
    _write_private(source_path, _canonical(source) + b"\n")

    config = SimpleNamespace(
        approval_environment="confirmatory",
        blueprint_receipt_path=control_blueprint_path,
        candidate_image_source_commit=closure.github_sha,
        canonical_file_bytes=lambda: config_encoded,
        factory_config_path=factory_path,
        factory_config_sha256=_digest(factory_encoded),
        file_sha256=config_sha256,
        runner_identity=runner_identity,
        scientific_candidate_reference=closure.scientific_image_reference,
        scientific_index_digest=closure.scientific_image_index_digest,
        scientific_production_reference=production_reference,
    )
    config_receipt = SimpleNamespace(
        approval_environment="confirmatory",
        candidate_image_source_commit=closure.github_sha,
        canonical_file_bytes=lambda: (
            _canonical({"record": "production-control-config-write-receipt"}) + b"\n"
        ),
        config_path=config_path,
        config_file_sha256=config_sha256,
        config_readback_sha256=config_sha256,
    )
    control_blueprint = SimpleNamespace(
        approval_environment="confirmatory",
        candidate_image_source_commit=closure.github_sha,
        canonical_file_bytes=lambda: control_blueprint_encoded,
        code_commit=closure.github_sha,
        file_sha256=control_blueprint_sha256,
        materialization_config_sha256=config_sha256,
        runner_image=production_reference,
        semantic_sha256=control_blueprint_semantic_sha256,
    )
    factory = SimpleNamespace(
        canonical_file_bytes=lambda: factory_encoded,
        design_seed_sha256="8" * 64,
    )

    def load_config(path: Path, *, expected_sha256: str) -> SimpleNamespace:
        assert expected_sha256 == config_sha256
        return config

    def load_factory(path: Path, *, expected_sha256: str) -> SimpleNamespace:
        assert expected_sha256 == _digest(factory_encoded)
        return factory

    monkeypatch.setattr(operator, "load_production_control_config", load_config)
    monkeypatch.setattr(
        operator,
        "load_production_control_config_write_receipt",
        lambda path: config_receipt,
    )
    monkeypatch.setattr(
        operator,
        "load_production_control_blueprint_receipt",
        lambda path: control_blueprint,
    )
    monkeypatch.setattr(operator, "load_production_artifact_factory_config", load_factory)

    beacon = _beacon()
    beacon_path = tmp_path / "execution-beacon-contract.json"
    _write_private(beacon_path, _canonical(beacon.to_dict()) + b"\n")

    return SimpleNamespace(
        beacon_path=beacon_path,
        blueprint_bundle=tmp_path / "blueprint-bundle",
        closure=closure,
        closure_path=closure_path,
        candidate_source_root=candidate_source_root,
        config_path=config_path,
        config_receipt_path=config_receipt_path,
        controlled=controlled,
        final_bundle=tmp_path / "final-bundle",
        host_sources=host_sources,
        host_tools=host_tools,
        runner_names={phase: f"fractal-registration-{phase}" for phase in PHASES},
        source=source,
        source_path=source_path,
    )


def _write_blueprint(inputs: SimpleNamespace) -> operator.ProviderPlanBlueprintWriteReceipt:
    return operator.write_provider_plan_blueprint(
        candidate_manifest_path=inputs.source_path,
        candidate_source_root=inputs.candidate_source_root,
        production_control_config_path=inputs.config_path,
        production_control_config_write_receipt_path=inputs.config_receipt_path,
        candidate_image_closure_path=inputs.closure_path,
        execution_beacon_contract_path=inputs.beacon_path,
        registered_online_runtime_budget_seconds=68_000,
        host_tool_sources=inputs.host_sources,
        claim_root=str(inputs.source_path.parent / "claims"),
        evidence_root=str(inputs.source_path.parent / "evidence"),
        runner_names=inputs.runner_names,
        output_directory=inputs.blueprint_bundle,
    )


class _RunnerApi:
    def __init__(self, *responses: bytes) -> None:
        self.responses = responses
        self.calls = 0

    def get_bytes(self, endpoint: str) -> bytes:
        assert endpoint == ("repos/mhdk1602/fractal-ann-diagnostics/actions/runners?per_page=100")
        position = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[position]


def _runner_inventory_bytes(
    expectation: operator.ProviderRunnerExpectation,
    *,
    runner_id: int,
    **changes: object,
) -> bytes:
    row: dict[str, object] = {
        "busy": False,
        "id": runner_id,
        "labels": [
            {"name": item} for item in required_execute_runner_labels(expectation.runner_label)
        ],
        "name": expectation.runner_name,
        "os": "macOS",
        "status": "offline",
    }
    row.update(changes)
    return _canonical({"runners": [row], "total_count": 1})


def _write_typed_registration(
    inputs: SimpleNamespace,
    expectation: operator.ProviderRunnerExpectation,
    *,
    runner_id: int,
    captured_at_utc: str,
    api: _RunnerApi | None = None,
) -> runner_operator.ProviderRunnerRegistrationReceipt:
    output = (
        inputs.controlled
        / "production"
        / "runner-registrations"
        / expectation.phase
        / expectation.runner_label
    )
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return runner_operator.write_provider_runner_registration(
        blueprint_directory=inputs.blueprint_bundle,
        phase=expectation.phase,
        api=api or _RunnerApi(_runner_inventory_bytes(expectation, runner_id=runner_id)),
        captured_at_utc=captured_at_utc,
    )


def _registration_bundles(inputs: SimpleNamespace) -> dict[str, Path]:
    blueprint, _receipt = operator.load_provider_plan_blueprint_bundle(inputs.blueprint_bundle)
    result: dict[str, Path] = {}
    for position, expectation in enumerate(blueprint.runner_expectations, start=1):
        receipt = _write_typed_registration(
            inputs,
            expectation,
            runner_id=position,
            captured_at_utc=f"2026-07-17T12:00:0{position}+00:00",
        )
        result[expectation.phase] = Path(receipt.registration_receipt_path).parent
    return result


def _finalize(
    inputs: SimpleNamespace,
) -> operator.ProviderPlanFinalizationReceipt:
    return operator.finalize_provider_plans(
        blueprint_path=(inputs.blueprint_bundle / operator.PROVIDER_PLAN_BLUEPRINT_FILENAME),
        blueprint_write_receipt_path=(
            inputs.blueprint_bundle / operator.PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_FILENAME
        ),
        output_directory=inputs.final_bundle,
    )


def test_operator_builds_one_pre_a_candidate_and_validates_two_later_commits(
    inputs: SimpleNamespace,
) -> None:
    source_before = inputs.source_path.read_bytes()
    _write_blueprint(inputs)
    _registration_bundles(inputs)
    receipt = _finalize(inputs)

    assert inputs.source_path.read_bytes() == source_before
    assert stat.S_IMODE(inputs.blueprint_bundle.stat().st_mode) == 0o700
    assert stat.S_IMODE(inputs.final_bundle.stat().st_mode) == 0o700
    assert {path.name for path in inputs.blueprint_bundle.iterdir()} == {
        operator.PROVIDER_PLAN_BLUEPRINT_FILENAME,
        operator.PROVIDER_PLAN_BLUEPRINT_WRITE_RECEIPT_FILENAME,
    }
    assert {path.name for path in inputs.final_bundle.iterdir()} == {
        operator.PROVIDER_PLAN_CANDIDATE_MANIFEST_FILENAME,
        operator.PROVIDER_PLAN_FINALIZATION_RECEIPT_FILENAME,
        operator.PROVIDER_PLAN_FRAGMENT_FILENAME,
    }
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for directory in (inputs.blueprint_bundle, inputs.final_bundle)
        for path in directory.iterdir()
    )

    candidate_path = inputs.final_bundle / operator.PROVIDER_PLAN_CANDIDATE_MANIFEST_FILENAME
    candidate_bytes = candidate_path.read_bytes()
    candidate = json.loads(candidate_bytes)
    assert C0_COMMIT_SENTINEL in candidate_bytes.decode("ascii")
    assert receipt.candidate_loader_witness_commit not in candidate_bytes.decode("ascii")
    assert receipt.candidate_manifest_output_file_sha256 == _digest(candidate_bytes)

    for later_commit in ("1" * 40, "2" * 40):
        plans = load_provider_phase_plans(
            candidate_path,
            c1_commit=later_commit,
            validation_mode="candidate-rehearsal",
            c0_commit=later_commit,
        )
        assert tuple(plans) == PHASES
        resolved = resolve_candidate_provider_plan_commit_bindings(
            candidate,
            c0_commit=later_commit,
        )
        for phase in PHASES:
            raw = candidate["sealed_execution"]["provider_phase_plans"][phase]
            normalized = resolved["sealed_execution"]["provider_phase_plans"][phase]
            assert raw["runner_bootstrap_receipt_file_sha256"] == _digest(
                _canonical(raw["runner_bootstrap_receipt"]) + b"\n"
            )
            assert normalized["runner_bootstrap_receipt_file_sha256"] == _digest(
                _canonical(normalized["runner_bootstrap_receipt"]) + b"\n"
            )
            assert (
                raw["runner_bootstrap_receipt_file_sha256"]
                != normalized["runner_bootstrap_receipt_file_sha256"]
            )

    raw_templates = candidate["sealed_execution"]["provider_phase_plans"]
    assert receipt.raw_provider_plan_templates_sha256 == _digest(_canonical(raw_templates))
    closed_candidate, closed_templates, closed_receipt = (
        operator.load_provider_plan_finalization_bundle(inputs.final_bundle)
    )
    assert closed_candidate == candidate
    assert closed_templates == raw_templates
    assert closed_receipt == receipt


def test_typed_registration_writes_all_three_p_bound_inputs_for_finalization(
    inputs: SimpleNamespace,
) -> None:
    source_before = inputs.source_path.read_bytes()
    _write_blueprint(inputs)
    blueprint, _blueprint_receipt = operator.load_provider_plan_blueprint_bundle(
        inputs.blueprint_bundle
    )
    assert blueprint.approval_environment == "confirmatory"
    assert blueprint.runner_identity == "github-actions:environment:confirmatory"
    paths: dict[str, Path] = {}
    for position, expectation in enumerate(blueprint.runner_expectations, start=1):
        receipt = _write_typed_registration(
            inputs,
            expectation,
            runner_id=position,
            captured_at_utc=f"2026-07-17T12:00:0{position}+00:00",
        )
        bundle = Path(receipt.registration_receipt_path).parent
        bootstrap, inventory, raw, readback = (
            runner_operator.load_provider_runner_registration_bundle(bundle)
        )
        assert readback == receipt
        assert bootstrap.workflow_sha == inputs.closure.github_sha
        assert bootstrap.approval_environment == blueprint.approval_environment
        assert bootstrap.runner_identity == blueprint.runner_identity
        assert bootstrap.repository_runner_inventory_sha256 == inventory.file_sha256
        assert inventory.response_sha256 == _digest(raw)
        assert receipt.runner_label == derive_phase_runner_label(
            receipt.claim_nonce,
            receipt.phase,
        )
        assert receipt.approval_environment == blueprint.approval_environment
        assert receipt.runner_identity == blueprint.runner_identity
        assert not hasattr(receipt, "apparatus_commit")
        assert not hasattr(receipt, "c1_commit")
        assert stat.S_IMODE(bundle.lstat().st_mode) == 0o700
        assert {path.name for path in bundle.iterdir()} == {
            runner_operator.INVENTORY_RECEIPT_FILENAME,
            runner_operator.RAW_INVENTORY_FILENAME,
            runner_operator.REGISTRATION_EVIDENCE_FILENAME,
            runner_operator.REGISTRATION_RECEIPT_FILENAME,
        }
        assert all(stat.S_IMODE(path.lstat().st_mode) == 0o600 for path in bundle.iterdir())
        paths[expectation.phase] = Path(receipt.registration_receipt_path)

    final = _finalize(inputs)
    assert final.registration_receipt_paths == {phase: str(paths[phase]) for phase in PHASES}
    assert final.registration_bundle_paths == {phase: str(paths[phase].parent) for phase in PHASES}
    finalized = json.loads(
        (inputs.final_bundle / operator.PROVIDER_PLAN_CANDIDATE_MANIFEST_FILENAME).read_bytes()
    )
    for phase, plan in finalized["sealed_execution"]["provider_phase_plans"].items():
        _bootstrap, evidence, bundle_sha256 = runner_operator.admit_provider_runner_registration(
            blueprint_directory=inputs.blueprint_bundle,
            phase=phase,
        )
        assert plan["approval_environment"] == "confirmatory"
        assert plan["runner_identity"] == "github-actions:environment:confirmatory"
        assert plan["runner_registration_bundle_path"] == str(paths[phase].parent)
        assert plan["runner_registration_bundle_sha256"] == bundle_sha256
        assert plan["runner_registration_evidence_file_sha256"] == evidence.file_sha256
        assert final.registration_bundle_sha256s[phase] == bundle_sha256
        assert final.registration_evidence_file_sha256s[phase] == evidence.file_sha256
        assert plan["runner_bootstrap_receipt"]["approval_environment"] == "confirmatory"
        assert (
            plan["runner_bootstrap_receipt"]["runner_identity"]
            == "github-actions:environment:confirmatory"
        )
    assert inputs.source_path.read_bytes() == source_before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approval_environment", "confirmatory-other"),
        ("runner_identity", "github-actions:environment:confirmatory-other"),
        ("runner_identity", "github-actions:ref:refs/heads/main"),
    ],
)
def test_typed_registration_rejects_governance_identity_substitution(
    inputs: SimpleNamespace,
    field: str,
    value: str,
) -> None:
    _write_blueprint(inputs)
    blueprint, _receipt = operator.load_provider_plan_blueprint_bundle(inputs.blueprint_bundle)
    expectation = blueprint.runner_expectations[0]
    receipt = _write_typed_registration(
        inputs,
        expectation,
        runner_id=17,
        captured_at_utc="2026-07-17T12:00:00+00:00",
    )
    payload = receipt.to_dict()
    payload[field] = value
    with pytest.raises(
        runner_operator.ProviderRunnerActivationError,
        match="approval environment",
    ):
        runner_operator.ProviderRunnerRegistrationReceipt.from_dict(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"status": "online"}, "stopped"),
        ({"busy": True}, "stopped"),
        ({"name": "substituted-runner"}, "singleton"),
        ({"labels": [{"name": "self-hosted"}]}, "exact derived label"),
        ({"os": "Linux"}, "stopped"),
    ],
)
def test_typed_registration_rejects_live_runner_identity_or_state_substitution(
    inputs: SimpleNamespace,
    changes: dict[str, object],
    message: str,
) -> None:
    _write_blueprint(inputs)
    blueprint, _receipt = operator.load_provider_plan_blueprint_bundle(inputs.blueprint_bundle)
    expectation = blueprint.runner_expectations[0]
    output = (
        inputs.controlled
        / "production"
        / "runner-registrations"
        / expectation.phase
        / expectation.runner_label
    )
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = _runner_inventory_bytes(expectation, runner_id=17, **changes)
    with pytest.raises(runner_operator.ProviderRunnerActivationError, match=message):
        runner_operator.write_provider_runner_registration(
            blueprint_directory=inputs.blueprint_bundle,
            phase=expectation.phase,
            api=_RunnerApi(raw),
            captured_at_utc="2026-07-17T12:00:00+00:00",
        )
    assert not output.exists()


def test_typed_registration_rejects_inventory_change_between_bounded_reads(
    inputs: SimpleNamespace,
) -> None:
    _write_blueprint(inputs)
    blueprint, _receipt = operator.load_provider_plan_blueprint_bundle(inputs.blueprint_bundle)
    expectation = blueprint.runner_expectations[0]
    output = (
        inputs.controlled
        / "production"
        / "runner-registrations"
        / expectation.phase
        / expectation.runner_label
    )
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    before = _runner_inventory_bytes(expectation, runner_id=17)
    after = _runner_inventory_bytes(expectation, runner_id=18)
    with pytest.raises(runner_operator.ProviderRunnerActivationError, match="changed"):
        runner_operator.write_provider_runner_registration(
            blueprint_directory=inputs.blueprint_bundle,
            phase=expectation.phase,
            api=_RunnerApi(before, after),
            captured_at_utc="2026-07-17T12:00:00+00:00",
        )
    assert not output.exists()


def test_typed_registration_rejects_blueprint_source_change_before_publication(
    inputs: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_blueprint(inputs)
    blueprint, _receipt = operator.load_provider_plan_blueprint_bundle(inputs.blueprint_bundle)
    expectation = blueprint.runner_expectations[0]
    output = (
        inputs.controlled
        / "production"
        / "runner-registrations"
        / expectation.phase
        / expectation.runner_label
    )
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    publish = runner_operator._publish_private_bundle

    def mutate_then_publish(
        output_directory: Path,
        members: dict[str, bytes],
        *,
        label: str,
        pre_publish: object,
        after_member_write: object = None,
    ) -> None:
        assert callable(pre_publish)

        def mutate() -> None:
            changed = copy.deepcopy(inputs.source)
            changed["freeze_blockers"] = ["source changed after registration staging"]
            _write_private(inputs.source_path, _canonical(changed) + b"\n")
            pre_publish()

        publish(
            output_directory,
            members,
            label=label,
            pre_publish=mutate,
            after_member_write=after_member_write,
        )

    monkeypatch.setattr(runner_operator, "_publish_private_bundle", mutate_then_publish)
    with pytest.raises(runner_operator.ProviderRunnerActivationError, match="authorize"):
        _write_typed_registration(
            inputs,
            expectation,
            runner_id=17,
            captured_at_utc="2026-07-17T12:00:00+00:00",
        )
    assert not output.exists()


def test_registration_bundle_rejects_extra_members_and_is_exactly_once(
    inputs: SimpleNamespace,
) -> None:
    _write_blueprint(inputs)
    blueprint, _receipt = operator.load_provider_plan_blueprint_bundle(inputs.blueprint_bundle)
    expectation = blueprint.runner_expectations[0]
    receipt = _write_typed_registration(
        inputs,
        expectation,
        runner_id=17,
        captured_at_utc="2026-07-17T12:00:00+00:00",
    )
    with pytest.raises(runner_operator.ProviderRunnerActivationError, match="already exists"):
        _write_typed_registration(
            inputs,
            expectation,
            runner_id=17,
            captured_at_utc="2026-07-17T12:00:01+00:00",
        )
    bundle = Path(receipt.registration_receipt_path).parent
    _write_private(bundle / "substitution.json", b"{}\n")
    with pytest.raises(runner_operator.ProviderRunnerActivationError, match="closed private"):
        runner_operator.load_provider_runner_registration_bundle(bundle)


def test_registration_cli_exposes_no_a_c1_digest_identity_or_output_override() -> None:
    parser = runner_operator._build_parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    register = subparsers.choices["register"]
    options = {option for action in register._actions for option in action.option_strings}
    assert {"--blueprint-directory", "--phase"} <= options
    assert (
        not {
            "--apparatus-commit",
            "--c0-commit",
            "--c1-commit",
            "--candidate-image-closure",
            "--digest",
            "--manifest",
            "--output",
            "--runner-id",
            "--runner-label",
            "--runner-name",
            "--sha256",
        }
        & options
    )


@pytest.mark.parametrize("attack", ("missing", "reused", "phase-swap"))
def test_finalization_rejects_missing_reused_and_phase_swapped_registration_bundles(
    inputs: SimpleNamespace,
    attack: str,
) -> None:
    _write_blueprint(inputs)
    bundles = _registration_bundles(inputs)
    if attack == "missing":
        bundles[ANALYSIS_PHASE].rename(bundles[ANALYSIS_PHASE].with_name(".registration-missing"))
    elif attack == "reused":
        bundles[ANALYSIS_PHASE].rename(bundles[ANALYSIS_PHASE].with_name(".registration-original"))
        shutil.copytree(bundles[ONLINE_PHASE], bundles[ANALYSIS_PHASE], copy_function=shutil.copy2)
    else:
        temporary = bundles[ONLINE_PHASE].with_name(".registration-swap")
        bundles[ONLINE_PHASE].rename(temporary)
        bundles[LABEL_RELEASE_PHASE].rename(bundles[ONLINE_PHASE])
        temporary.rename(bundles[LABEL_RELEASE_PHASE])
    with pytest.raises(operator.ProviderPlanOperatorError, match="registration bundle"):
        _finalize(inputs)
    assert not inputs.final_bundle.exists()


def test_finalization_rejects_a_hand_authored_bootstrap_without_the_closed_bundle(
    inputs: SimpleNamespace,
) -> None:
    _write_blueprint(inputs)
    bundles = _registration_bundles(inputs)
    target = bundles[ONLINE_PHASE]
    retained = target.with_name(".typed-registration-retained")
    target.rename(retained)
    target.mkdir(mode=0o700)
    _write_private(
        target / runner_operator.REGISTRATION_RECEIPT_FILENAME,
        (retained / runner_operator.REGISTRATION_RECEIPT_FILENAME).read_bytes(),
    )
    with pytest.raises(operator.ProviderPlanOperatorError, match="closed private"):
        _finalize(inputs)
    assert not inputs.final_bundle.exists()


def test_final_source_rehash_rejects_path_substitution_before_publish(
    inputs: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_blueprint(inputs)
    _registration_bundles(inputs)
    publish = operator._publish_private_bundle

    def substitute_then_publish(
        output_directory: Path,
        members: dict[str, bytes],
        *,
        label: str,
        pre_publish: object,
        after_member_write: object = None,
    ) -> None:
        assert callable(pre_publish)

        def substitute() -> None:
            changed = copy.deepcopy(inputs.source)
            changed["freeze_blockers"] = ["substituted after staging"]
            _write_private(inputs.source_path, _canonical(changed) + b"\n")
            pre_publish()

        publish(
            output_directory,
            members,
            label=label,
            pre_publish=substitute,
            after_member_write=after_member_write,
        )

    monkeypatch.setattr(operator, "_publish_private_bundle", substitute_then_publish)
    with pytest.raises(operator.ProviderPlanOperatorError, match="changed"):
        _finalize(inputs)
    assert not inputs.final_bundle.exists()


def test_finalization_rejects_blueprint_bundle_opened_before_publish(
    inputs: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_blueprint(inputs)
    _registration_bundles(inputs)
    publish = operator._publish_private_bundle

    def open_blueprint_then_publish(
        output_directory: Path,
        members: dict[str, bytes],
        *,
        label: str,
        pre_publish: object,
        after_member_write: object = None,
    ) -> None:
        assert callable(pre_publish)

        def substitute() -> None:
            _write_private(inputs.blueprint_bundle / "unregistered.json", b"{}\n")
            pre_publish()

        publish(
            output_directory,
            members,
            label=label,
            pre_publish=substitute,
            after_member_write=after_member_write,
        )

    monkeypatch.setattr(operator, "_publish_private_bundle", open_blueprint_then_publish)
    with pytest.raises(operator.ProviderPlanOperatorError, match="blueprint bundle"):
        _finalize(inputs)
    assert not inputs.final_bundle.exists()


def test_source_with_hand_authored_provider_plan_is_rejected(inputs: SimpleNamespace) -> None:
    changed = copy.deepcopy(inputs.source)
    changed["sealed_execution"]["provider_phase_plans"] = {}  # type: ignore[index]
    _write_private(inputs.source_path, _canonical(changed) + b"\n")
    with pytest.raises(operator.ProviderPlanOperatorError, match="provider_phase_plans"):
        _write_blueprint(inputs)
    assert not inputs.blueprint_bundle.exists()


def test_finalization_is_exactly_once(inputs: SimpleNamespace) -> None:
    _write_blueprint(inputs)
    _registration_bundles(inputs)
    _finalize(inputs)
    with pytest.raises(operator.ProviderPlanOperatorError, match="already exists"):
        _finalize(inputs)


def test_closed_bundle_read_rejects_root_swap(
    inputs: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_blueprint(inputs)
    _registration_bundles(inputs)
    _finalize(inputs)
    read_member = operator._read_bundle_member
    moved = inputs.final_bundle.with_name("final-bundle-original")
    swapped = False

    def swap_root(
        directory_descriptor: int,
        name: str,
        *,
        label: str,
    ) -> bytes:
        nonlocal swapped
        encoded = read_member(directory_descriptor, name, label=label)
        if not swapped:
            swapped = True
            inputs.final_bundle.rename(moved)
            inputs.final_bundle.mkdir(mode=0o700)
            for source in moved.iterdir():
                _write_private(inputs.final_bundle / source.name, source.read_bytes())
        return encoded

    monkeypatch.setattr(operator, "_read_bundle_member", swap_root)
    with pytest.raises(operator.ProviderPlanOperatorError, match="fd-bound readback"):
        operator.load_provider_plan_finalization_bundle(inputs.final_bundle)


def test_finalization_readback_rejects_registration_digest_cross_binding_change(
    inputs: SimpleNamespace,
) -> None:
    _write_blueprint(inputs)
    _registration_bundles(inputs)
    _finalize(inputs)
    receipt_path = inputs.final_bundle / operator.PROVIDER_PLAN_FINALIZATION_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_bytes())
    receipt["registration_bundle_sha256s"][ONLINE_PHASE] = "f" * 64
    _write_private(receipt_path, _canonical(receipt) + b"\n")
    with pytest.raises(operator.ProviderPlanOperatorError, match="closure differs"):
        operator.load_provider_plan_finalization_bundle(inputs.final_bundle)


def test_alternating_valid_config_bytes_are_not_mixed_with_typed_semantics(
    inputs: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = operator.load_production_control_config(
        inputs.config_path,
        expected_sha256=_digest(inputs.config_path.read_bytes()),
    )
    alternate_encoded = _canonical({"record": "alternate-production-control-config"}) + b"\n"
    alternate = SimpleNamespace(
        **{
            **vars(original),
            "canonical_file_bytes": lambda: alternate_encoded,
        }
    )

    def alternate_after_first_read(
        path: Path,
        *,
        expected_sha256: str,
    ) -> SimpleNamespace:
        assert expected_sha256 == _digest(inputs.config_path.read_bytes())
        _write_private(inputs.config_path, alternate_encoded)
        return alternate

    monkeypatch.setattr(
        operator,
        "load_production_control_config",
        alternate_after_first_read,
    )
    with pytest.raises(operator.ProviderPlanOperatorError, match="admitted bytes"):
        _write_blueprint(inputs)
    assert not inputs.blueprint_bundle.exists()


def test_alternating_valid_config_receipt_bytes_are_rejected(
    inputs: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = operator.load_production_control_config_write_receipt(inputs.config_receipt_path)
    alternate_encoded = (
        _canonical({"record": "alternate-production-control-config-write-receipt"}) + b"\n"
    )
    alternate = SimpleNamespace(
        **{
            **vars(original),
            "canonical_file_bytes": lambda: alternate_encoded,
        }
    )

    def alternate_after_first_read(path: Path) -> SimpleNamespace:
        _write_private(inputs.config_receipt_path, alternate_encoded)
        return alternate

    monkeypatch.setattr(
        operator,
        "load_production_control_config_write_receipt",
        alternate_after_first_read,
    )
    with pytest.raises(operator.ProviderPlanOperatorError, match="admitted bytes"):
        _write_blueprint(inputs)
    assert not inputs.blueprint_bundle.exists()


def test_cli_exposes_paths_and_identities_but_no_digest_or_future_commit_flags() -> None:
    parser = operator._build_parser()
    option_strings: set[str] = set()
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if not choices:
            continue
        for choice in choices.values():
            for child_action in choice._actions:
                option_strings.update(child_action.option_strings)
    assert "--c0-commit" not in option_strings
    assert not any("sha256" in option or "digest" in option for option in option_strings)
    assert "--online-runner-name" in option_strings
    assert not any("registration" in option for option in option_strings)


def test_sigkill_during_staging_never_publishes_a_partial_bundle(tmp_path: Path) -> None:
    output = tmp_path / "atomic-bundle"
    script = """
import os
import signal
import sys
from pathlib import Path
from fractal_ann_diagnostics.provider_plan_operator import _publish_private_bundle

def die_after_first(position):
    if position == 1:
        os.kill(os.getpid(), signal.SIGKILL)

_publish_private_bundle(
    Path(sys.argv[1]),
    {"one.json": b"{}\\n", "two.json": b"{\\\"two\\\":2}\\n"},
    label="death-test bundle",
    pre_publish=lambda: None,
    after_member_write=die_after_first,
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        check=False,
        cwd=Path(__file__).parents[1],
    )
    assert completed.returncode == -signal.SIGKILL
    assert not output.exists()

    members = {"one.json": b"{}\n", "two.json": b'{"two":2}\n'}
    operator._publish_private_bundle(
        output,
        members,
        label="death-test bundle",
        pre_publish=lambda: None,
    )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == members
