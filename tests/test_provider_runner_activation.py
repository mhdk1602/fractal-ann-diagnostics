from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from test_provider_phase_runtime import _plan
from test_provider_rehearsal import _closure

import fractal_ann_diagnostics.provider_runner_activation as activation
from fractal_ann_diagnostics.execution_claim import (
    ANALYSIS_PHASE,
    LABEL_RELEASE_PHASE,
    ONLINE_PHASE,
    ProviderPhasePlan,
    derive_phase_runner_label,
    load_provider_runner_bootstrap,
    required_execute_runner_labels,
)
from fractal_ann_diagnostics.production_controls import (
    C0_CANDIDATE_ASSEMBLY_RECEIPT_RELATIVE_PATH,
    C0_CANDIDATE_MANIFEST_RELATIVE_PATH,
    FACTORY_RUNNER_PLATFORM,
    ProductionControlC0InstantiationReceipt,
    WorkloadSpecBinding,
    _c0_instantiated_payload_entries,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA

A = "1" * 40
C1 = "2" * 40
PHASES = (ONLINE_PHASE, LABEL_RELEASE_PHASE, ANALYSIS_PHASE)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, encoded: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(encoded)
    path.chmod(mode)


class _Api:
    def __init__(self, *responses: bytes) -> None:
        self.responses = list(responses)
        self.calls = 0

    def get_bytes(self, endpoint: str) -> bytes:
        assert endpoint == ("repos/mhdk1602/fractal-ann-diagnostics/actions/runners?per_page=100")
        position = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[position]


class _MutatingApi(_Api):
    def __init__(self, response: bytes, mutation: Callable[[], None]) -> None:
        super().__init__(response)
        self.mutation = mutation

    def get_bytes(self, endpoint: str) -> bytes:
        encoded = super().get_bytes(endpoint)
        if self.calls == 1:
            self.mutation()
        return encoded


def _inventory_bytes(plan: ProviderPhasePlan, **changes: object) -> bytes:
    label = derive_phase_runner_label(plan.claim_nonce, plan.phase)
    row: dict[str, object] = {
        "busy": False,
        "id": plan.runner_id,
        "labels": [{"name": item} for item in (*required_execute_runner_labels(label),)],
        "name": plan.runner_name,
        "os": "macOS",
        "status": "offline",
    }
    row.update(changes)
    return _canonical({"runners": [row], "total_count": 1})


def _workloads() -> tuple[WorkloadSpecBinding, ...]:
    return tuple(
        WorkloadSpecBinding(
            corpus_id=corpus_id,
            available_family_count=100,
            selected_family_count=75,
            relative_path=f"workloads/{corpus_id}.json",
            file_sha256=_digest(f"workload:{corpus_id}"),
            launcher_control_tree_sha256=_digest(f"launcher:{corpus_id}"),
            plan_template_file_sha256=_digest(f"plan-file:{corpus_id}"),
            plan_template_semantic_sha256=_digest(f"plan:{corpus_id}"),
            preflight_contract_sha256=_digest(f"preflight:{corpus_id}"),
            preflight_contract_file_sha256=_digest(f"preflight-file:{corpus_id}"),
        )
        for corpus_id in FIXED_CORPORA
    )


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    closure = _closure()
    controlled = tmp_path / "host-tools"
    selected = _plan(tmp_path, phase=ONLINE_PHASE)
    gh_path = controlled / "gh/bin/gh"
    _write(gh_path, b"controlled-gh\n", mode=0o700)
    host_tools = selected.host_tools
    monkeypatch.setattr(
        activation,
        "digest_regular_file",
        lambda path, *, label: host_tools.gh_executable_sha256,
    )
    provider_path = controlled / "production/provider-plans/online/provider-plan.json"
    runner_label = derive_phase_runner_label(selected.claim_nonce, ONLINE_PHASE)
    bootstrap = replace(
        selected.runner_bootstrap_receipt,
        workflow_sha=A,
        runner_label=runner_label,
    )
    production_reference = (
        "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory@"
        f"{closure.scientific_image_index_digest}"
    )
    manifest_value = {"fixture": "frozen-c1"}
    manifest_bytes = _canonical(manifest_value) + b"\n"
    selected = replace(
        selected,
        manifest_sha256=_digest(_canonical(manifest_value)),
        c1_commit=C1,
        provider_plan_path=str(provider_path),
        workflow_sha=A,
        host_tools=host_tools,
        runner_bootstrap_receipt=bootstrap,
        runner_bootstrap_receipt_path=str(
            controlled
            / "production"
            / "runners"
            / ONLINE_PHASE
            / runner_label
            / activation.BOOTSTRAP_RECEIPT_FILENAME
        ),
        runner_bootstrap_receipt_file_sha256=bootstrap.file_sha256,
        runtime_image=production_reference,
        oci_index_digest=closure.scientific_image_index_digest,
        oci_platform_manifest_digest=closure.scientific_linux_arm64_manifest_digest,
        runtime_probe_receipt_sha256=(closure.scientific_linux_arm64_runtime_extraction_sha256),
        activation_argv_template=tuple(
            str(provider_path) if item == selected.provider_plan_path else item
            for item in selected.activation_argv_template
        ),
    )
    _write(provider_path, selected.canonical_file_bytes())
    manifest_path = tmp_path / "frozen-manifest.json"
    _write(manifest_path, manifest_bytes)
    closure_path = tmp_path / "candidate-closure.json"
    _write(closure_path, _canonical(closure.to_dict()) + b"\n")
    c0 = ProductionControlC0InstantiationReceipt(
        apparatus_commit=A,
        candidate_image_source_commit=closure.github_sha,
        build_context_tree_sha256=closure.build_context_tree_sha256,
        candidate_image_closure_file_sha256=closure.file_sha256,
        candidate_bootstrap_closure_sha256=closure.bootstrap_closure_sha256,
        candidate_manifest_sha256=_digest("candidate-manifest"),
        candidate_manifest_file_sha256=_digest("candidate-manifest-file"),
        candidate_manifest_relative_path=C0_CANDIDATE_MANIFEST_RELATIVE_PATH,
        candidate_manifest_assembly_receipt_file_sha256=_digest("candidate-assembly-receipt-file"),
        candidate_manifest_assembly_receipt_relative_path=(
            C0_CANDIDATE_ASSEMBLY_RECEIPT_RELATIVE_PATH
        ),
        materialization_config_file_sha256=_digest("materialization-config"),
        blueprint_receipt_sha256=_digest("blueprint"),
        blueprint_receipt_file_sha256=_digest("blueprint-file"),
        blueprint_payload_tree_sha256=_digest("blueprint-tree"),
        scientific_candidate_reference=closure.scientific_image_reference,
        scientific_production_reference=production_reference,
        scientific_index_digest=closure.scientific_image_index_digest,
        release_image_index_digest=closure.release_image_index_digest,
        approval_environment="confirmatory",
        runner_platform=FACTORY_RUNNER_PLATFORM,
        launcher_identity_file_sha256=_digest("launcher-identity"),
        instantiated_root=str(tmp_path / "instantiated-controls"),
        instantiated_payload_tree_sha256=_digest("instantiated-tree"),
        instantiated_payload_entries=_c0_instantiated_payload_entries(),
        workloads=_workloads(),
    )
    c0_path = tmp_path / "c0-control-instantiation-receipt.json"
    _write(c0_path, c0.canonical_file_bytes())
    plans = {
        ONLINE_PHASE: selected,
        LABEL_RELEASE_PHASE: _plan(tmp_path, phase=LABEL_RELEASE_PHASE),
        ANALYSIS_PHASE: _plan(tmp_path, phase=ANALYSIS_PHASE),
    }

    def load_plans(path: Path, *, c1_commit: str) -> object:
        assert Path(path).read_bytes() == manifest_bytes
        assert c1_commit == C1
        return plans

    monkeypatch.setattr(activation, "load_provider_phase_plans", load_plans)
    output = Path(selected.runner_bootstrap_receipt_path).parent
    output.parent.mkdir(mode=0o700, parents=True)
    return type(
        "Inputs",
        (),
        {
            "c0": c0,
            "c0_path": c0_path,
            "closure": closure,
            "closure_path": closure_path,
            "manifest_path": manifest_path,
            "output": output,
            "plan": selected,
            "plans": plans,
        },
    )()


def _write_activation(inputs: object, api: _Api) -> activation.ProviderRunnerActivationReceipt:
    return activation.write_provider_runner_activation(
        manifest_path=inputs.manifest_path,
        c1_commit=C1,
        phase=ONLINE_PHASE,
        c0_instantiation_receipt_path=inputs.c0_path,
        candidate_image_closure_path=inputs.closure_path,
        api=api,
        captured_at_utc="2026-07-17T18:00:00+00:00",
    )


def test_writes_and_verifies_exact_post_a_activation_bundle(inputs: object) -> None:
    raw = _inventory_bytes(inputs.plan)
    receipt = _write_activation(inputs, _Api(raw))
    assert receipt.approval_environment == "confirmatory"
    assert receipt.runner_identity == "github-actions:environment:confirmatory"
    assert receipt.apparatus_commit == A
    assert receipt.candidate_image_source_commit == inputs.closure.github_sha
    assert receipt.build_context_tree_sha256 == inputs.closure.build_context_tree_sha256
    assert receipt.runtime_index_digest == inputs.closure.scientific_image_index_digest
    assert set(path.name for path in inputs.output.iterdir()) == {
        activation.ACTIVATION_RECEIPT_FILENAME,
        activation.BOOTSTRAP_RECEIPT_FILENAME,
        activation.INVENTORY_RECEIPT_FILENAME,
        activation.RAW_INVENTORY_FILENAME,
    }
    assert stat.S_IMODE(inputs.output.lstat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.lstat().st_mode) == 0o600 for path in inputs.output.iterdir())
    assert load_provider_runner_bootstrap(inputs.plan) == inputs.plan.runner_bootstrap_receipt
    verified = activation.verify_provider_runner_activation(
        manifest_path=inputs.manifest_path,
        c1_commit=C1,
        phase=ONLINE_PHASE,
        c0_instantiation_receipt_path=inputs.c0_path,
        candidate_image_closure_path=inputs.closure_path,
    )
    assert verified == receipt


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approval_environment", "confirmatory-other"),
        ("runner_identity", "github-actions:environment:confirmatory-other"),
        ("runner_identity", "github-actions:ref:refs/heads/main"),
    ],
)
def test_activation_receipt_rejects_governance_identity_substitution(
    inputs: object,
    field: str,
    value: str,
) -> None:
    receipt = _write_activation(inputs, _Api(_inventory_bytes(inputs.plan)))
    payload = receipt.to_dict()
    payload[field] = value
    with pytest.raises(activation.ProviderRunnerActivationError, match="approval environment"):
        activation.ProviderRunnerActivationReceipt.from_dict(payload)


def test_phase_runtime_bindings_preserve_both_registered_image_indexes(inputs: object) -> None:
    closure = inputs.closure
    c0 = inputs.c0
    assert activation._phase_runtime_binding(ONLINE_PHASE, closure, c0) == (
        c0.scientific_production_reference,
        closure.scientific_image_index_digest,
        closure.scientific_linux_arm64_manifest_digest,
        closure.scientific_linux_arm64_runtime_extraction_sha256,
    )
    assert activation._phase_runtime_binding(ANALYSIS_PHASE, closure, c0) == (
        c0.scientific_production_reference,
        closure.scientific_image_index_digest,
        closure.scientific_linux_amd64_manifest_digest,
        closure.scientific_linux_amd64_runtime_extraction_sha256,
    )
    assert activation._phase_runtime_binding(LABEL_RELEASE_PHASE, closure, c0) == (
        "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-release@"
        f"{closure.release_image_index_digest}",
        closure.release_image_index_digest,
        closure.release_linux_arm64_manifest_digest,
        closure.release_reproducibility_receipt_sha256,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("apparatus_commit", "3" * 40, "A/P/T/D"),
        ("candidate_image_source_commit", "4" * 40, "A/P/T/D"),
        ("build_context_tree_sha256", "5" * 64, "A/P/T/D"),
        ("release_image_index_digest", f"sha256:{'6' * 64}", "A/P/T/D"),
    ],
)
def test_rejects_a_p_t_or_d_substitution(
    inputs: object,
    field: str,
    value: str,
    message: str,
) -> None:
    hostile = replace(inputs.c0, **{field: value})
    _write(inputs.c0_path, hostile.canonical_file_bytes())
    with pytest.raises(activation.ProviderRunnerActivationError, match=message):
        _write_activation(inputs, _Api(_inventory_bytes(inputs.plan)))
    assert not inputs.output.exists()


def test_rejects_materialized_plan_substitution(inputs: object) -> None:
    encoded = inputs.plan.canonical_file_bytes()
    _write(Path(inputs.plan.provider_plan_path), encoded.replace(b"online", b"onl1ne", 1))
    with pytest.raises(
        activation.ProviderRunnerActivationError, match="materialized provider plan"
    ):
        _write_activation(inputs, _Api(_inventory_bytes(inputs.plan)))
    assert not inputs.output.exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "online"},
        {"busy": True},
        {"id": 999},
        {
            "labels": [
                {"name": "self-hosted"},
                {"name": "macOS"},
                {"name": "ARM64"},
                {"name": "fractal-ann-confirmatory-online-wrong"},
            ]
        },
    ],
)
def test_rejects_non_stopped_or_substituted_live_runner(
    inputs: object,
    changes: dict[str, object],
) -> None:
    with pytest.raises(activation.ProviderRunnerActivationError):
        _write_activation(inputs, _Api(_inventory_bytes(inputs.plan, **changes)))
    assert not inputs.output.exists()


def test_rejects_live_inventory_change_before_atomic_publication(inputs: object) -> None:
    first = _inventory_bytes(inputs.plan)
    second = _inventory_bytes(inputs.plan, busy=True)
    with pytest.raises(
        activation.ProviderRunnerActivationError,
        match="stopped|revalidate|changed",
    ):
        _write_activation(inputs, _Api(first, second))
    assert not inputs.output.exists()


def test_rejects_source_substitution_before_atomic_publication(inputs: object) -> None:
    api = _MutatingApi(
        _inventory_bytes(inputs.plan),
        lambda: _write(inputs.closure_path, b"{}\n"),
    )
    with pytest.raises(activation.ProviderRunnerActivationError):
        _write_activation(inputs, api)
    assert not inputs.output.exists()


def test_refuses_preexisting_activation_output(inputs: object) -> None:
    inputs.output.mkdir(mode=0o700)
    with pytest.raises(activation.ProviderRunnerActivationError, match="already exists"):
        _write_activation(inputs, _Api(_inventory_bytes(inputs.plan)))


def test_closed_verifier_rejects_extra_member(inputs: object) -> None:
    _write_activation(inputs, _Api(_inventory_bytes(inputs.plan)))
    _write(inputs.output / "extra.json", b"{}\n")
    with pytest.raises(activation.ProviderRunnerActivationError, match="closed private"):
        activation.load_provider_runner_activation_bundle(inputs.output)


def test_cli_has_no_digest_or_output_override() -> None:
    parser = activation._build_parser()
    options: set[str] = set()
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices:
            for command in choices.values():
                for child in command._actions:
                    options.update(child.option_strings)
    assert "--c1-commit" in options
    assert "--phase" in options
    assert "--output-directory" not in options
    assert "--c0-commit" not in options
    assert not any("sha256" in option or "digest" in option for option in options)
