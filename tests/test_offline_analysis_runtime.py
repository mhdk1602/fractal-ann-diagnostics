from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_confirmatory_analysis import (
    SCHEMA,
    _bound_input,
    _development_batch,
    _frozen_manifest,
)
from test_confirmatory_analysis import (
    config as analysis_config_fixture,
)
from test_confirmatory_input_operator import _members

import fractal_ann_diagnostics.confirmatory_execution as execution_module
import fractal_ann_diagnostics.confirmatory_input_operator as input_operator
import fractal_ann_diagnostics.offline_analysis_contract as contract_module
import fractal_ann_diagnostics.offline_analysis_provider as provider_module
import fractal_ann_diagnostics.offline_analysis_runtime as runtime_module
from fractal_ann_diagnostics.artifact_integrity import digest_directory_tree
from fractal_ann_diagnostics.confirmatory_analysis import (
    ConfirmatoryResultArtifact,
    CorpusGeometryResult,
    DirectionalGate,
    EntitlementResult,
    H1Result,
    H2Result,
    H3Result,
    PositionAdjustedSensitivityResult,
)
from fractal_ann_diagnostics.confirmatory_modeling import (
    FrozenModelSuite,
    canonical_h1_model_artifact_bytes,
    canonical_h2_model_suite_artifact_bytes,
    fit_frozen_model_suite,
)
from fractal_ann_diagnostics.execution_claim import (
    ANALYSIS_PHASE,
    ProviderPhasePlan,
    VerifiedPhaseClaimCapability,
)
from fractal_ann_diagnostics.offline_analysis_contract import (
    NETWORK_MODE,
    RUNTIME_ENVIRONMENT,
    RUNTIME_GID,
    RUNTIME_MACHINE,
    RUNTIME_PLATFORM,
    RUNTIME_UID,
    OfflineAnalysisAdmission,
    OfflineAnalysisEvidenceBinding,
    OfflineAnalysisExecutionReceipt,
    OfflineAnalysisFileBinding,
    OfflineConfirmatoryInputBundle,
    load_offline_analysis_execution_receipt,
)
from fractal_ann_diagnostics.offline_analysis_runtime import (
    RuntimeMountObservation,
    RuntimeObservation,
)
from fractal_ann_diagnostics.suite_attempt import VerifiedProviderPredecessor


def _digest(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture(scope="module")
def frozen_suite() -> FrozenModelSuite:
    return fit_frozen_model_suite(
        _development_batch("development-fit", "offline-fit", 10),
        _development_batch("development-calibration", "offline-cal", 6),
        schema=SCHEMA,
        random_seed=37,
    )


class _Token:
    def __init__(self) -> None:
        self.state = SimpleNamespace(
            suite_attempt_id=_digest("suite-attempt"),
            record_sha256=_digest("labels-released"),
        )
        self.descriptor_sha256 = _digest("descriptor")

    def assert_current(self) -> None:
        return None


@pytest.fixture
def offline_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_suite: FrozenModelSuite,
) -> SimpleNamespace:
    results = tmp_path / "results"
    package = tmp_path / "package"
    tmpfs = tmp_path / "tmpfs"
    evidence_root = tmp_path / "evidence"
    for path in (results, package, tmpfs, evidence_root):
        path.mkdir(mode=0o700)

    real_frozen_manifest = _frozen_manifest

    def manifest_with_local_results(*args: object, **kwargs: object) -> dict[str, object]:
        manifest = real_frozen_manifest(*args, **kwargs)
        sealed = manifest["sealed_execution"]
        assert isinstance(sealed, dict)
        sealed["results_store"] = results.as_uri()
        return manifest

    import test_confirmatory_analysis as analysis_helpers

    monkeypatch.setattr(
        analysis_helpers,
        "_frozen_manifest",
        manifest_with_local_results,
    )
    analysis_config = analysis_config_fixture.__wrapped__()
    inputs = _bound_input(analysis_config, suite=frozen_suite)
    members = _members(evidence_root)
    receipt = input_operator._materialization_receipt(inputs, _Token(), members)
    input_path = input_operator.confirmatory_input_path(inputs)
    receipt_path = input_operator.confirmatory_input_receipt_path(inputs)
    input_bytes = inputs.canonical_bytes() + b"\n"
    receipt_bytes = receipt.canonical_bytes() + b"\n"
    input_path.write_bytes(input_bytes)
    receipt_path.write_bytes(receipt_bytes)

    bundle = OfflineConfirmatoryInputBundle.from_confirmatory_input(inputs)
    bundle_path = package / f"{inputs.manifest_sha256}.offline-input-bundle.json"
    bundle_bytes = bundle.canonical_bytes() + b"\n"
    h1_path = package / "h1-predictive-model.json"
    h2_path = package / "h2-model-suite.json"
    h1_bytes = canonical_h1_model_artifact_bytes(frozen_suite)
    h2_bytes = canonical_h2_model_suite_artifact_bytes(frozen_suite)
    package_input_path = package / input_path.name
    package_receipt_path = package / receipt_path.name
    payloads = {
        package_input_path: ("confirmatory-input", inputs.artifact_sha256, input_bytes),
        package_receipt_path: (
            "confirmatory-input-receipt",
            receipt.receipt_sha256,
            receipt_bytes,
        ),
        bundle_path: ("offline-input-bundle", bundle.bundle_sha256, bundle_bytes),
        h1_path: ("h1-predictive-model", _digest(h1_bytes), h1_bytes),
        h2_path: ("h2-model-suite", frozen_suite.suite_digest, h2_bytes),
    }
    bindings = tuple(
        sorted(
            (
                OfflineAnalysisFileBinding(
                    role=role,
                    relative_path=path.name,
                    semantic_sha256=semantic,
                    file_sha256=_digest(encoded),
                    byte_count=len(encoded),
                )
                for path, (role, semantic, encoded) in payloads.items()
            ),
            key=lambda row: row.relative_path.encode("utf-8"),
        )
    )
    for path, (_, _, encoded) in payloads.items():
        path.write_bytes(encoded)
    evidence = tuple(
        OfflineAnalysisEvidenceBinding(
            role=row.role,
            corpus_id=row.corpus_id,
            source_uri=row.uri,
            semantic_sha256=row.semantic_sha256,
            file_sha256=row.file_sha256,
            byte_count=row.byte_count,
        )
        for row in members
    )
    attempt_path = results / f"{inputs.manifest_sha256}.confirmatory-analysis-attempt.json"
    result_receipt_path = results / f"{inputs.manifest_sha256}.confirmatory-result-receipt.json"
    result_path = results / f"{inputs.manifest_sha256}.confirmatory-result.json"
    attempt = runtime_module.ConfirmatoryAnalysisAttemptReceipt(
        manifest_sha256=inputs.manifest_sha256,
        run_receipt_sha256=inputs.run_receipt_sha256,
        confirmatory_input_artifact_sha256=inputs.artifact_sha256,
        model_suite_sha256=frozen_suite.suite_digest,
        runner_identity=inputs.run_receipt.runner_identity,
        result_uri=result_path.as_uri(),
    )
    plan_path = tmp_path / "analysis-provider-plan.json"
    plan_path.write_bytes(b"{}\n")

    monkeypatch.setattr(contract_module, "PACKAGE_MOUNT_PATH", str(package))
    monkeypatch.setattr(contract_module, "RESULTS_MOUNT_PATH", str(results))
    monkeypatch.setattr(contract_module, "TMPFS_MOUNT_PATH", str(tmpfs))
    monkeypatch.setattr(runtime_module, "PACKAGE_MOUNT_PATH", str(package))
    monkeypatch.setattr(runtime_module, "RESULTS_MOUNT_PATH", str(results))
    monkeypatch.setattr(runtime_module, "TMPFS_MOUNT_PATH", str(tmpfs))
    admission = OfflineAnalysisAdmission(
        suite_attempt_id=receipt.suite_attempt_id,
        provider_state_record_sha256=_digest("analysis-claimed"),
        provider_ledger_commit="a" * 40,
        provider_control_inventory_sha256=_digest("control-inventory"),
        provider_artifact_receipt_sha256=_digest("artifact-receipt"),
        phase_claim_contract_sha256=_digest("phase-claim"),
        phase_claim_state_sha256=_digest("analysis-claimed"),
        phase_claim_ledger_commit="a" * 40,
        provider_identity_sha256=_digest("provider"),
        live_execute_job_receipt_sha256=_digest("live-job"),
        claim_attested_at_utc="2026-07-27T12:00:00+00:00",
        c1_commit="c" * 40,
        c1_provider_plan_uri=plan_path.as_uri(),
        c1_provider_plan_sha256=_digest("provider-plan"),
        c1_provider_plan_file_sha256=_digest(plan_path.read_bytes()),
        runtime_image=f"ghcr.io/example/science@sha256:{_digest('index')}",
        runtime_platform=RUNTIME_PLATFORM,
        runtime_image_role="scientific",
        runtime_index_role="main",
        oci_index_digest=f"sha256:{_digest('index')}",
        oci_platform_manifest_digest=f"sha256:{_digest('amd64-manifest')}",
        host_tool_contract_sha256=_digest("host-tools"),
        runtime_probe_receipt_sha256=_digest("runtime-probe"),
        manifest_sha256=inputs.manifest_sha256,
        run_receipt_sha256=inputs.run_receipt_sha256,
        confirmatory_input_artifact_sha256=inputs.artifact_sha256,
        confirmatory_input_artifact_file_sha256=_digest(input_bytes),
        confirmatory_input_artifact_byte_count=len(input_bytes),
        confirmatory_input_receipt_sha256=receipt.receipt_sha256,
        confirmatory_input_receipt_file_sha256=_digest(receipt_bytes),
        confirmatory_input_receipt_byte_count=len(receipt_bytes),
        offline_input_bundle_sha256=bundle.bundle_sha256,
        model_suite_sha256=frozen_suite.suite_digest,
        registered_results_store_uri=results.as_uri(),
        host_results_store_path=str(results),
        package_mount_path=str(package),
        results_mount_path=str(results),
        tmpfs_mount_path=str(tmpfs),
        network_mode=NETWORK_MODE,
        root_filesystem_read_only=True,
        package_mount_read_only=True,
        results_mount_read_write=True,
        runtime_machine=RUNTIME_MACHINE,
        runtime_uid=RUNTIME_UID,
        runtime_gid=RUNTIME_GID,
        runtime_environment=RUNTIME_ENVIRONMENT,
        runtime_dynamic_environment_names=(),
        container_name=f"fractal-analysis-{receipt.suite_attempt_id}",
        registered_attempt_uri=attempt_path.as_uri(),
        registered_result_receipt_uri=result_receipt_path.as_uri(),
        registered_result_uri=result_path.as_uri(),
        container_attempt_path=str(attempt_path),
        container_result_receipt_path=str(result_receipt_path),
        container_result_path=str(result_path),
        expected_attempt_receipt_sha256=attempt.receipt_sha256,
        evidence=evidence,
        package_files=bindings,
    )
    admission_path = package / admission.admission_filename
    admission_path.write_bytes(admission.canonical_bytes() + b"\n")
    observation = RuntimeObservation(
        operating_system="Linux",
        machine=RUNTIME_MACHINE,
        uid=RUNTIME_UID,
        gid=RUNTIME_GID,
        environment=RUNTIME_ENVIRONMENT,
        network_interfaces=("lo",),
        mounts=(
            RuntimeMountObservation("/", "overlay", ("ro",), ("ro",)),
            RuntimeMountObservation(str(package), "virtiofs", ("ro",), ("ro",)),
            RuntimeMountObservation(str(results), "virtiofs", ("rw",), ("rw",)),
            RuntimeMountObservation(
                str(tmpfs),
                "tmpfs",
                ("nodev", "noexec", "nosuid", "rw"),
                ("nodev", "noexec", "nosuid", "rw"),
            ),
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "capture_runtime_observation",
        lambda: observation,
    )
    return SimpleNamespace(
        admission=admission,
        admission_path=admission_path,
        attempt_path=attempt_path,
        package=package,
        results=results,
        tmpfs=tmpfs,
        observation=observation,
        h2_path=h2_path,
        inputs=inputs,
        materialized=input_operator.MaterializedConfirmatoryInput(
            inputs=inputs,
            receipt=receipt,
            artifact_path=input_path,
            receipt_path=receipt_path,
        ),
        suite=frozen_suite,
    )


def _typed_test_result(offline_package: SimpleNamespace) -> ConfirmatoryResultArtifact:
    inputs = offline_package.inputs
    config = inputs.frozen_config
    fixed_count = len(config.fixed_corpora)

    def gate(
        name: str,
        *,
        threshold: float,
        rule: str,
        seed_offset: int,
        n_corpora: int,
        direction: str,
    ) -> DirectionalGate:
        return DirectionalGate(
            name=name,
            estimate=threshold,
            lower=threshold if direction == "greater" else threshold - 0.1,
            upper=threshold + 0.1 if direction == "greater" else threshold,
            threshold=threshold,
            rule=rule,
            confidence=config.confidence,
            n_corpora=n_corpora,
            n_families=n_corpora * config.selected_families_per_corpus,
            bootstrap_replicates=config.bootstrap_replicates,
            bootstrap_seed=config.bootstrap_seed + seed_offset,
            passed=False,
        )

    h2_gates = (
        gate(
            "h2_log_loss_reduction",
            threshold=config.geometry_gain_thresholds.log_loss_reduction,
            rule="directional-lower-greater-than",
            seed_offset=21,
            n_corpora=fixed_count,
            direction="greater",
        ),
        gate(
            "h2_brier_score_reduction",
            threshold=config.geometry_gain_thresholds.brier_score_reduction,
            rule="directional-lower-greater-than",
            seed_offset=22,
            n_corpora=fixed_count,
            direction="greater",
        ),
        gate(
            "h2_auprc_gain",
            threshold=config.geometry_gain_thresholds.auprc_gain,
            rule="directional-lower-greater-than",
            seed_offset=23,
            n_corpora=fixed_count,
            direction="greater",
        ),
    )
    h3_gates = (
        gate(
            "h3_family_latency_relative_reduction",
            threshold=config.minimum_cost_reduction,
            rule="directional-lower-greater-than",
            seed_offset=31,
            n_corpora=fixed_count,
            direction="greater",
        ),
        gate(
            "h3_retrieval_target_difference",
            threshold=-config.retrieval_target_noninferiority_margin,
            rule="directional-lower-greater-than-negative-margin",
            seed_offset=33,
            n_corpora=fixed_count,
            direction="greater",
        ),
        gate(
            "h3_evidence_sufficiency_difference",
            threshold=-config.evidence_sufficiency_noninferiority_margin,
            rule="directional-lower-greater-than-negative-margin",
            seed_offset=34,
            n_corpora=len(config.evidence_corpora),
            direction="greater",
        ),
        gate(
            "h3_p95_family_latency_ratio",
            threshold=config.maximum_p95_latency_ratio,
            rule="directional-upper-less-than",
            seed_offset=32,
            n_corpora=fixed_count,
            direction="less",
        ),
    )
    n_families = fixed_count * config.selected_families_per_corpus
    entitlement = EntitlementResult(
        observed_events=0,
        families_with_events=0,
        n_families=n_families,
        exact_upper_bound=0.1,
        confidence=config.confidence,
        passed=True,
    )
    action_count = len(config.action_set)
    trial_count = fixed_count * config.selected_families_per_corpus * config.nested_rows_per_family
    input_row_count = action_count * trial_count
    return ConfirmatoryResultArtifact(
        manifest_sha256=inputs.manifest_sha256,
        run_receipt_sha256=inputs.run_receipt_sha256,
        confirmatory_input_artifact_sha256=inputs.artifact_sha256,
        corpus_input_digests=inputs.corpus_input_digests,
        frozen_config=config,
        model_suite_sha256=offline_package.admission.model_suite_sha256,
        input_rows_sha256=_digest("typed-test-result-rows"),
        input_row_count=input_row_count,
        trial_count=trial_count,
        h1=H1Result(
            gate=gate(
                "h1_high_minus_low_predictive_risk",
                threshold=config.h1_minimum_risk_increase,
                rule="directional-lower-greater-than",
                seed_offset=11,
                n_corpora=fixed_count,
                direction="greater",
            ),
            model_digest=_digest("typed-test-model"),
            passed=False,
        ),
        h2=H2Result(
            metric_gates=h2_gates,
            corpus_results=tuple(
                CorpusGeometryResult(
                    corpus_id=corpus_id,
                    log_loss_reduction=0.0,
                    brier_score_reduction=0.0,
                    auprc_gain=0.0,
                    passed=False,
                )
                for corpus_id in config.fixed_corpora
            ),
            passing_corpora=(),
            minimum_corpora=config.minimum_corpora_with_geometry_gain,
            row_identity_digest=_digest("typed-test-row-identity"),
            passed=False,
        ),
        h3=H3Result(
            gates=h3_gates,
            entitlement=entitlement,
            position_adjusted_sensitivity=PositionAdjustedSensitivityResult(
                gate=gate(
                    "h3_position_adjusted_log_latency_ratio_sensitivity",
                    threshold=math.log(1.0 - config.minimum_cost_reduction),
                    rule=(
                        "sensitivity-directional-upper-less-than-log-one-minus-minimum-reduction"
                    ),
                    seed_offset=35,
                    n_corpora=fixed_count,
                    direction="less",
                ),
                position_trend_log_ratio_per_position=0.0,
            ),
            execution_state_counts=(
                ("completed", input_row_count),
                ("failed", 0),
                ("abstained", 0),
            ),
            passed=False,
        ),
        primary_claim_passed=False,
    )


def _install_fake_analysis(
    monkeypatch: pytest.MonkeyPatch,
    offline_package: SimpleNamespace,
    calls: list[str],
) -> None:
    def compute(*args: object, **kwargs: object) -> ConfirmatoryResultArtifact:
        calls.append("compute")
        assert args
        assert kwargs["suite"] is not None
        assert Path(offline_package.admission.container_attempt_path).is_file()
        return _typed_test_result(offline_package)

    monkeypatch.setattr(runtime_module, "run_confirmatory_analysis", compute)


def test_typed_result_reconstruction_rejects_registered_gate_drift(
    offline_package: SimpleNamespace,
) -> None:
    payload = _typed_test_result(offline_package).to_dict()
    payload["h3"]["gates"][0]["rule"] = "substituted-decision-rule"

    with pytest.raises(
        execution_module.ConfirmatoryAnalysisError,
        match="registered gate",
    ):
        execution_module._typed_confirmatory_result(payload)


def test_typed_result_reconstruction_accepts_registered_undefined_auprc_gate(
    offline_package: SimpleNamespace,
) -> None:
    payload = _typed_test_result(offline_package).to_dict()
    gate = payload["h2"]["metric_gates"][2]
    gate.update(
        estimate=None,
        lower=None,
        upper=None,
        bootstrap_replicates=0,
        rule="undefined-one-class-corpus_conservative-fail",
    )

    result = execution_module._typed_confirmatory_result(payload)

    assert result.h2.metric_gates[2].estimate is None
    assert result.h2.metric_gates[2].bootstrap_replicates == 0


def _prepared_analysis(
    offline_package: SimpleNamespace,
    tmp_path: Path,
) -> provider_module.PreparedOfflineAnalysis:
    docker = tmp_path / "docker-client"
    docker.write_bytes(b"fixed-test-docker-client\n")
    docker.chmod(0o755)
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir(mode=0o700)
    admission = offline_package.admission
    package_inventory = digest_directory_tree(offline_package.package)
    execution_receipt_path = (
        offline_package.package.parent
        / f"{admission.manifest_sha256}.offline-analysis-execution-receipt.json"
    )
    create = provider_module._expected_docker_create_argv(
        docker,
        admission,
        admission_path=offline_package.admission_path,
        package_root=offline_package.package,
        results_root=offline_package.results,
        docker_config_root=docker_config,
    )
    return provider_module.PreparedOfflineAnalysis(
        admission=admission,
        admission_path=offline_package.admission_path,
        package_root=offline_package.package,
        results_root=offline_package.results,
        docker_config_root=docker_config,
        docker_invocation_executable=docker,
        docker_resolved_executable=docker,
        docker_executable_sha256=_digest(docker.read_bytes()),
        docker_pull_argv=provider_module._docker_pull_argv(
            str(docker),
            admission,
            docker_config_root=docker_config,
        ),
        docker_create_argv=create,
        docker_start_argv=provider_module._docker_start_argv(
            str(docker),
            admission,
            docker_config_root=docker_config,
        ),
        docker_remove_argv=provider_module._docker_remove_argv(
            str(docker),
            admission,
            docker_config_root=docker_config,
        ),
        docker_inspect_argv=provider_module._docker_inspect_argv(
            str(docker),
            admission,
            docker_config_root=docker_config,
        ),
        execution_receipt_path=execution_receipt_path,
        package_tree_sha256=package_inventory.sha256,
        package_entries=package_inventory.entries,
        maximum_runtime_seconds=60,
    )


class _DockerLifecycle:
    def __init__(self, start: object) -> None:
        self._start = start
        self.present = False
        self.events: list[str] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        check: bool,
        env: dict[str, str],
        stdin: int,
        stdout: int,
        stderr: int,
        shell: bool,
        close_fds: bool,
        start_new_session: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        del (
            check,
            env,
            stdin,
            stdout,
            stderr,
            shell,
            close_fds,
            start_new_session,
            timeout,
        )
        command = argv[3]
        if command == "version":
            self.events.append("version")
            return subprocess.CompletedProcess(argv, 0)
        if command == "pull":
            self.events.append("pull")
            return subprocess.CompletedProcess(argv, 0)
        if command == "create":
            self.events.append("create")
            if self.present:
                return subprocess.CompletedProcess(argv, 1)
            self.present = True
            return subprocess.CompletedProcess(argv, 0)
        if command == "start":
            self.events.append("start")
            if not self.present:
                return subprocess.CompletedProcess(argv, 1)
            if isinstance(self._start, BaseException):
                raise self._start
            start = self._start
            assert callable(start)
            start()
            self.present = False
            return subprocess.CompletedProcess(argv, 0)
        if command == "container" and argv[4] == "inspect":
            self.events.append("inspect-present" if self.present else "inspect-absent")
            return subprocess.CompletedProcess(argv, 0 if self.present else 1)
        if command == "container" and argv[4] == "rm":
            self.events.append("remove")
            existed = self.present
            self.present = False
            return subprocess.CompletedProcess(argv, 0 if existed else 1)
        raise AssertionError(f"unexpected Docker command: {argv!r}")


def _matching_analysis_authorities(
    admission: OfflineAnalysisAdmission,
) -> tuple[object, object, object, object]:
    contract = object()
    provider_identity = object()
    claimed_values = {
        "ledger_commit": admission.provider_ledger_commit,
        "control_inventory_sha256": admission.provider_control_inventory_sha256,
        "artifact_receipt_sha256": admission.provider_artifact_receipt_sha256,
    }
    phase_values = {
        "contract": contract,
        "provider_identity": provider_identity,
        "phase_claim_state_sha256": admission.phase_claim_state_sha256,
        "phase_claim_ledger_commit": admission.phase_claim_ledger_commit,
    }
    initial_claimed = SimpleNamespace(
        state=SimpleNamespace(
            record_sha256=admission.provider_state_record_sha256,
        ),
        **claimed_values,
    )
    fresh_claimed = SimpleNamespace(
        state=SimpleNamespace(
            record_sha256=admission.provider_state_record_sha256,
        ),
        **claimed_values,
    )
    return (
        initial_claimed,
        SimpleNamespace(**phase_values),
        fresh_claimed,
        SimpleNamespace(**phase_values),
    )


def test_offline_input_bundle_round_trips_the_complete_typed_input(
    offline_package: SimpleNamespace,
) -> None:
    bundle = OfflineConfirmatoryInputBundle.from_confirmatory_input(offline_package.inputs)
    restored = OfflineConfirmatoryInputBundle.from_dict(
        json.loads(bundle.canonical_bytes())
    ).to_confirmatory_input()
    assert restored.artifact_sha256 == offline_package.inputs.artifact_sha256
    assert restored.analysis_rows() == offline_package.inputs.analysis_rows()


def test_container_attempt_is_durable_before_compute_and_replay_fails(
    offline_package: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_fake_analysis(monkeypatch, offline_package, calls)

    outcome = runtime_module.execute_offline_analysis_once(
        offline_package.admission_path,
    )
    assert calls == ["compute"]
    assert outcome.attempt_path.is_file()
    assert outcome.result_receipt_path.is_file()
    assert outcome.result_path.is_file()
    assert len(tuple(offline_package.results.iterdir())) == 5

    with pytest.raises(
        runtime_module.OfflineAnalysisRuntimeError,
        match="two-file materialized input closure",
    ):
        runtime_module.execute_offline_analysis_once(
            offline_package.admission_path,
        )
    assert calls == ["compute"]


def test_altered_model_file_fails_before_attempt_or_compute(
    offline_package: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_fake_analysis(monkeypatch, offline_package, calls)
    offline_package.h2_path.write_bytes(offline_package.h2_path.read_bytes() + b"\n")

    with pytest.raises(
        runtime_module.OfflineAnalysisRuntimeError,
        match="package h2-predictive-model|package h2-model-suite|bytes changed",
    ):
        runtime_module.execute_offline_analysis_once(
            offline_package.admission_path,
        )
    assert not offline_package.attempt_path.exists()
    assert calls == []


def test_materialized_input_is_rehashed_after_compute(
    offline_package: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    input_name = next(
        row.relative_path
        for row in offline_package.admission.package_files
        if row.role == "confirmatory-input"
    )

    def compute(*args: object, **kwargs: object) -> ConfirmatoryResultArtifact:
        calls.append("compute")
        assert args and kwargs["suite"] is not None
        (offline_package.results / input_name).write_bytes(b"changed during compute\n")
        return _typed_test_result(offline_package)

    monkeypatch.setattr(runtime_module, "run_confirmatory_analysis", compute)
    with pytest.raises(
        runtime_module.OfflineAnalysisRuntimeError,
        match="results-store confirmatory-input differs",
    ):
        runtime_module.execute_offline_analysis_once(
            offline_package.admission_path,
        )
    assert calls == ["compute"]
    assert offline_package.attempt_path.is_file()


def test_admission_is_revalidated_after_compute(
    offline_package: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def compute(*args: object, **kwargs: object) -> ConfirmatoryResultArtifact:
        calls.append("compute")
        assert args and kwargs["suite"] is not None
        offline_package.admission_path.write_bytes(b"{}\n")
        return _typed_test_result(offline_package)

    monkeypatch.setattr(runtime_module, "run_confirmatory_analysis", compute)
    with pytest.raises(
        runtime_module.OfflineAnalysisRuntimeError,
        match="cannot revalidate offline admission",
    ):
        runtime_module.execute_offline_analysis_once(
            offline_package.admission_path,
        )
    assert calls == ["compute"]
    assert offline_package.attempt_path.is_file()
    assert not Path(offline_package.admission.container_result_receipt_path).exists()
    assert not Path(offline_package.admission.container_result_path).exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: replace(row, operating_system="Darwin", machine="arm64"),
            "Linux AMD64",
        ),
        (
            lambda row: replace(
                row,
                environment=(*row.environment, ("GH_TOKEN", "forbidden")),
            ),
            "environment drifted",
        ),
        (
            lambda row: replace(row, network_interfaces=("eth0", "lo")),
            "non-loopback",
        ),
        (
            lambda row: replace(
                row,
                mounts=tuple(
                    replace(item, mount_options=("rw",))
                    if item.mount_path == str(Path(row.mounts[1].mount_path))
                    else item
                    for item in row.mounts
                ),
            ),
            "package mount is not read-only",
        ),
    ],
)
def test_host_platform_network_environment_and_mount_drift_fail_before_inputs(
    offline_package: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    message: str,
) -> None:
    calls: list[str] = []
    _install_fake_analysis(monkeypatch, offline_package, calls)
    mutate = mutation
    assert callable(mutate)
    observation = mutate(offline_package.observation)
    monkeypatch.setattr(
        runtime_module,
        "capture_runtime_observation",
        lambda: observation,
    )
    with pytest.raises(runtime_module.OfflineAnalysisRuntimeError, match=message):
        runtime_module.execute_offline_analysis_once(
            offline_package.admission_path,
        )
    assert not offline_package.attempt_path.exists()
    assert calls == []


def test_admission_rejects_wrong_platform_or_index_identity(
    offline_package: SimpleNamespace,
) -> None:
    with pytest.raises(
        contract_module.OfflineAnalysisContractError,
        match="runtime image binding",
    ):
        replace(offline_package.admission, runtime_platform="linux/arm64")
    with pytest.raises(
        contract_module.OfflineAnalysisContractError,
        match="OCI index digest",
    ):
        replace(
            offline_package.admission,
            oci_index_digest=f"sha256:{_digest('different-index')}",
        )


def test_runtime_rejects_unregistered_extra_mount(
    offline_package: SimpleNamespace,
) -> None:
    observation = replace(
        offline_package.observation,
        mounts=(
            *offline_package.observation.mounts,
            RuntimeMountObservation(
                "/unregistered-study-bytes",
                "virtiofs",
                ("ro",),
                ("ro",),
            ),
        ),
    )
    with pytest.raises(
        runtime_module.OfflineAnalysisRuntimeError,
        match="mount closure differs",
    ):
        runtime_module.validate_runtime_observation(
            offline_package.admission,
            observation,
        )


def test_admission_rejects_duplicate_package_roles(
    offline_package: SimpleNamespace,
) -> None:
    duplicate = replace(
        next(
            row
            for row in offline_package.admission.package_files
            if row.role == "h1-predictive-model"
        ),
        relative_path="second-h1-predictive-model.json",
    )
    package_files = tuple(
        sorted(
            (*offline_package.admission.package_files, duplicate),
            key=lambda row: row.relative_path.encode("utf-8"),
        )
    )
    with pytest.raises(
        contract_module.OfflineAnalysisContractError,
        match="exact canonical closure",
    ):
        replace(offline_package.admission, package_files=package_files)


def test_root_read_only_flag_may_be_reported_as_an_overlay_super_option(
    offline_package: SimpleNamespace,
) -> None:
    root_super_read_only = replace(
        offline_package.observation,
        mounts=tuple(
            replace(row, mount_options=("rw",), super_options=("ro",))
            if row.mount_path == "/"
            else row
            for row in offline_package.observation.mounts
        ),
    )
    runtime_module.validate_runtime_observation(
        offline_package.admission,
        root_super_read_only,
    )
    root_read_write = replace(
        root_super_read_only,
        mounts=tuple(
            replace(row, super_options=("rw",)) if row.mount_path == "/" else row
            for row in root_super_read_only.mounts
        ),
    )
    with pytest.raises(
        runtime_module.OfflineAnalysisRuntimeError,
        match="root filesystem is not read-only",
    ):
        runtime_module.validate_runtime_observation(
            offline_package.admission,
            root_read_write,
        )


def test_docker_launcher_pulls_platform_manifest_and_closes_runtime_flags(
    offline_package: SimpleNamespace,
    tmp_path: Path,
) -> None:
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir(mode=0o700)
    plan = SimpleNamespace(
        host_tools=SimpleNamespace(
            docker_executable="/registered/docker",
            docker_resolved_executable="/resolved/docker-client",
        ),
    )
    argv = provider_module._docker_create_argv(
        plan,
        offline_package.admission,
        admission_path=offline_package.admission_path,
        package_root=offline_package.package,
        results_root=offline_package.results,
        docker_config_root=docker_config,
    )
    pull = provider_module._docker_pull_argv(
        plan.host_tools.docker_resolved_executable,
        offline_package.admission,
        docker_config_root=docker_config,
    )
    platform_image = (
        f"ghcr.io/example/science@{offline_package.admission.oci_platform_manifest_digest}"
    )
    assert pull[0] == "/resolved/docker-client"
    assert argv[0] == "/resolved/docker-client"
    assert pull[-1] == platform_image
    assert platform_image in argv
    assert argv[argv.index("--entrypoint") + 2] == platform_image
    assert ("--network", "none") == (
        argv[argv.index("--network")],
        argv[argv.index("--network") + 1],
    )
    assert "--read-only" in argv
    assert argv[argv.index("--name") + 1] == offline_package.admission.container_name
    assert argv[argv.index("--log-driver") + 1] == "none"
    assert "--env" not in argv
    assert "-e" not in argv
    assert argv.count("--mount") == 2
    provider_module._validate_docker_create_argv(
        argv,
        admission=offline_package.admission,
        admission_path=offline_package.admission_path,
        package_root=offline_package.package,
        results_root=offline_package.results,
        docker_config_root=docker_config,
        docker_resolved_executable=Path(plan.host_tools.docker_resolved_executable),
    )
    drifted = list(argv)
    drifted[drifted.index("none")] = "bridge"
    with pytest.raises(provider_module.OfflineAnalysisProviderError):
        provider_module._validate_docker_create_argv(
            drifted,
            admission=offline_package.admission,
            admission_path=offline_package.admission_path,
            package_root=offline_package.package,
            results_root=offline_package.results,
            docker_config_root=docker_config,
            docker_resolved_executable=Path(plan.host_tools.docker_resolved_executable),
        )
    duplicate = (*argv[:4], "--network", "none", *argv[4:])
    with pytest.raises(
        provider_module.OfflineAnalysisProviderError,
        match="not closed",
    ):
        provider_module._validate_docker_create_argv(
            duplicate,
            admission=offline_package.admission,
            admission_path=offline_package.admission_path,
            package_root=offline_package.package,
            results_root=offline_package.results,
            docker_config_root=docker_config,
            docker_resolved_executable=Path(plan.host_tools.docker_resolved_executable),
        )


def test_prepare_recovers_exact_existing_package_without_rewriting(
    offline_package: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = tmp_path / "recovery-docker-client"
    docker.write_bytes(b"fixed-recovery-docker-client\n")
    docker.chmod(0o755)
    plan = SimpleNamespace(
        host_tools=SimpleNamespace(
            docker_executable=str(docker),
            docker_resolved_executable=str(docker),
            docker_executable_sha256=_digest(docker.read_bytes()),
        ),
        maximum_runtime_seconds=60,
    )
    claimed = SimpleNamespace(
        state=SimpleNamespace(
            manifest_sha256=offline_package.inputs.manifest_sha256,
            run_receipt_sha256=offline_package.inputs.run_receipt_sha256,
        ),
        assert_current=lambda: None,
    )
    phase_claim = SimpleNamespace(assert_current=lambda: None)
    monkeypatch.setattr(provider_module, "_validate_authority", lambda *args: None)
    monkeypatch.setattr(provider_module, "_verify_plan_file", lambda *args: None)
    monkeypatch.setattr(
        provider_module,
        "_admission",
        lambda *args, **kwargs: offline_package.admission,
    )
    package_before = digest_directory_tree(offline_package.package)

    prepared = provider_module.prepare_offline_analysis(
        offline_package.materialized,
        offline_package.suite,
        plan,
        claimed,
        phase_claim,
        package_root=offline_package.package,
        results_root=offline_package.results,
        recover_existing=True,
    )

    assert prepared.package_tree_sha256 == package_before.sha256
    assert digest_directory_tree(offline_package.package) == package_before
    assert prepared.admission == offline_package.admission
    assert not prepared.execution_receipt_path.exists()
    with pytest.raises(
        provider_module.OfflineAnalysisProviderError,
        match="cannot create offline analysis package",
    ):
        provider_module.prepare_offline_analysis(
            offline_package.materialized,
            offline_package.suite,
            plan,
            claimed,
            phase_claim,
            package_root=offline_package.package,
            results_root=offline_package.results,
        )


def test_docker_control_calls_use_only_the_scrubbed_environment() -> None:
    observed: dict[str, object] = {}

    def run(
        argv: tuple[str, ...],
        *,
        check: bool,
        env: dict[str, str],
        stdin: int,
        stdout: int,
        stderr: int,
        shell: bool,
        close_fds: bool,
        start_new_session: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        observed.update(
            argv=argv,
            check=check,
            env=env,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            shell=shell,
            close_fds=close_fds,
            start_new_session=start_new_session,
            timeout=timeout,
        )
        return subprocess.CompletedProcess(argv, 0)

    argv = ("/resolved/docker-client", "--config", "/empty", "pull")
    provider_module._run_docker_bounded(
        run,
        argv,
        timeout=37,
        label="test Docker control",
    )
    assert observed["argv"] == argv
    assert observed["check"] is False
    assert observed["env"] == {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    assert observed["stdin"] == subprocess.DEVNULL
    assert observed["stdout"] == subprocess.DEVNULL
    assert observed["stderr"] == subprocess.DEVNULL
    assert observed["shell"] is False
    assert observed["close_fds"] is True
    assert observed["start_new_session"] is True
    assert observed["timeout"] == 37


def test_provider_executes_named_container_and_retains_execution_receipt(
    offline_package: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_calls: list[str] = []
    _install_fake_analysis(monkeypatch, offline_package, analysis_calls)
    prepared = _prepared_analysis(offline_package, tmp_path)
    lifecycle = _DockerLifecycle(
        lambda: runtime_module.execute_offline_analysis_once(
            offline_package.admission_path,
        )
    )
    claimed, phase_claim, completion_claimed, completion_phase_claim = (
        _matching_analysis_authorities(offline_package.admission)
    )
    authority_checks: list[str] = []
    monkeypatch.setattr(
        provider_module,
        "_validate_authority",
        lambda *args: authority_checks.append("current"),
    )

    def fresh_claim_supplier() -> tuple[object, object]:
        lifecycle.events.append("completion-authority")
        return completion_claimed, completion_phase_claim

    executed = provider_module.execute_prepared_offline_analysis(
        prepared,
        claimed,
        phase_claim,
        object(),
        fresh_claim_supplier=fresh_claim_supplier,
        docker_run=lifecycle,
    )

    assert analysis_calls == ["compute"]
    assert authority_checks == ["current"] * 5
    assert lifecycle.present is False
    assert lifecycle.events.count("pull") == 1
    assert lifecycle.events.count("create") == 1
    assert lifecycle.events.count("start") == 1
    assert lifecycle.events.index("completion-authority") > lifecycle.events.index("start")
    assert "inspect-present" in lifecycle.events
    assert lifecycle.events[-2:] == ["inspect-absent", "completion-authority"]
    assert "--force" in prepared.docker_remove_argv
    assert "--volumes" in prepared.docker_remove_argv
    assert prepared.admission.container_name in prepared.docker_create_argv
    assert prepared.admission.container_name in prepared.docker_start_argv

    receipt = load_offline_analysis_execution_receipt(
        prepared.execution_receipt_path,
        expected_receipt_sha256=executed.outcome.execution_receipt_sha256,
        expected_file_sha256=executed.outcome.execution_receipt_file_sha256,
    )
    assert receipt.receipt_sha256 == executed.outcome.execution_receipt_sha256
    assert receipt.package_tree_before_sha256 == prepared.package_tree_sha256
    assert receipt.package_tree_after_sha256 == prepared.package_tree_sha256
    assert receipt.container_name == offline_package.admission.container_name
    assert receipt.runtime_image == offline_package.admission.runtime_image
    assert receipt.results_tree_sha256 == digest_directory_tree(offline_package.results).sha256
    assert receipt.result_file_sha256 == executed.outcome.result_file_sha256
    assert receipt.container_absent_after_execution is True

    malformed = receipt.to_dict()
    malformed_entries = list(receipt.package_entries)
    malformed_entries[0] = 7
    malformed["package_entries"] = malformed_entries
    with pytest.raises(contract_module.OfflineAnalysisContractError):
        OfflineAnalysisExecutionReceipt.from_dict(malformed)
    with pytest.raises(
        contract_module.OfflineAnalysisContractError,
        match="semantic digest differs",
    ):
        load_offline_analysis_execution_receipt(
            prepared.execution_receipt_path,
            expected_receipt_sha256=_digest("wrong execution receipt"),
        )


def test_provider_recovers_completed_five_file_closure_without_recomputation(
    offline_package: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_calls: list[str] = []
    _install_fake_analysis(monkeypatch, offline_package, analysis_calls)
    prepared = _prepared_analysis(offline_package, tmp_path)
    first_lifecycle = _DockerLifecycle(
        lambda: runtime_module.execute_offline_analysis_once(
            offline_package.admission_path,
        )
    )
    claimed, phase_claim, completion_claimed, completion_phase_claim = (
        _matching_analysis_authorities(offline_package.admission)
    )
    monkeypatch.setattr(provider_module, "_validate_authority", lambda *args: None)

    def fresh_claim_supplier() -> tuple[object, object]:
        return completion_claimed, completion_phase_claim

    first = provider_module.execute_prepared_offline_analysis(
        prepared,
        claimed,
        phase_claim,
        object(),
        fresh_claim_supplier=fresh_claim_supplier,
        docker_run=first_lifecycle,
    )
    prepared.execution_receipt_path.unlink()
    recovery_lifecycle = _DockerLifecycle(
        lambda: (_ for _ in ()).throw(
            AssertionError("recovery cannot recompute the registered attempt")
        )
    )

    recovered = provider_module.execute_prepared_offline_analysis(
        prepared,
        claimed,
        phase_claim,
        object(),
        fresh_claim_supplier=fresh_claim_supplier,
        docker_run=recovery_lifecycle,
    )

    assert analysis_calls == ["compute"]
    assert recovered.outcome.execution_receipt_sha256 == (first.outcome.execution_receipt_sha256)
    assert recovery_lifecycle.events == ["version", "inspect-absent"]
    assert "pull" not in recovery_lifecycle.events
    assert "create" not in recovery_lifecycle.events
    assert "start" not in recovery_lifecycle.events

    second_recovery_lifecycle = _DockerLifecycle(
        lambda: (_ for _ in ()).throw(AssertionError("retained receipt recovery cannot recompute"))
    )
    resumed = provider_module.execute_prepared_offline_analysis(
        prepared,
        claimed,
        phase_claim,
        object(),
        fresh_claim_supplier=fresh_claim_supplier,
        docker_run=second_recovery_lifecycle,
    )
    assert resumed.outcome == recovered.outcome
    assert second_recovery_lifecycle.events == ["version", "inspect-absent"]


def test_provider_rejects_partial_attempt_without_launching_again(
    offline_package: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_analysis(offline_package, tmp_path)
    Path(offline_package.admission.container_attempt_path).write_bytes(b"partial-attempt\n")
    lifecycle = _DockerLifecycle(
        lambda: (_ for _ in ()).throw(AssertionError("partial attempt cannot be relaunched"))
    )
    monkeypatch.setattr(provider_module, "_validate_authority", lambda *args: None)

    with pytest.raises(
        provider_module.OfflineAnalysisProviderError,
        match="partial confirmatory outcome is terminal",
    ):
        provider_module.execute_prepared_offline_analysis(
            prepared,
            object(),
            object(),
            object(),
            fresh_claim_supplier=lambda: (_ for _ in ()).throw(
                AssertionError("partial attempt cannot reach completion authority")
            ),
            docker_run=lifecycle,
        )

    assert lifecycle.events == []


def test_provider_entry_resumes_existing_materialization_and_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = SimpleNamespace(inputs=object())
    suite = object()
    prepared = SimpleNamespace(
        admission=SimpleNamespace(confirmatory_input_artifact_sha256=_digest("confirmatory-input"))
    )
    completion_claimed = SimpleNamespace(assert_current=lambda: None)
    completion_phase_claim = SimpleNamespace(assert_current=lambda: None)
    outcome = SimpleNamespace(
        execution_receipt_path=tmp_path / "execution.json",
        execution_receipt_sha256=_digest("execution"),
        execution_receipt_file_sha256=_digest("execution-file"),
        attempt_path=tmp_path / "attempt.json",
        result_receipt_path=tmp_path / "result-receipt.json",
        result_path=tmp_path / "result.json",
    )
    candidate = SimpleNamespace(state="ANALYSIS_COMPLETE")
    observed: dict[str, object] = {}
    monkeypatch.setattr(provider_module, "_validate_authority", lambda *args: None)
    monkeypatch.setattr(
        provider_module,
        "materialize_confirmatory_input",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            input_operator.ConfirmatoryInputOperatorError("already materialized")
        ),
    )
    monkeypatch.setattr(
        provider_module,
        "load_materialized_confirmatory_input",
        lambda *args, **kwargs: materialized,
    )
    monkeypatch.setattr(
        provider_module,
        "load_admitted_model_suite",
        lambda *args, **kwargs: suite,
    )

    def prepare(*args: object, **kwargs: object) -> object:
        observed["prepare_args"] = args
        observed["prepare_kwargs"] = kwargs
        return prepared

    monkeypatch.setattr(provider_module, "prepare_offline_analysis", prepare)
    monkeypatch.setattr(
        provider_module,
        "execute_prepared_offline_analysis",
        lambda *args, **kwargs: SimpleNamespace(
            outcome=outcome,
            completion_claimed=completion_claimed,
            completion_phase_claim=completion_phase_claim,
        ),
    )

    def complete(*args: object, **kwargs: object) -> object:
        observed["complete_args"] = args
        observed["complete_kwargs"] = kwargs
        return candidate

    monkeypatch.setattr(provider_module, "complete_confirmatory_analysis", complete)
    monkeypatch.setattr(
        provider_module,
        "ProviderAnalysisCompletion",
        lambda *, candidate, outcome: SimpleNamespace(
            candidate=candidate,
            outcome=outcome,
        ),
    )
    result = provider_module.run_provider_claimed_offline_analysis_once(
        object(),
        object(),
        object(),
        object(),
        package_root=tmp_path / "package",
        results_root=tmp_path / "results",
        fresh_claim_supplier=lambda: (object(), object()),
    )

    assert result.candidate is candidate
    assert result.outcome is outcome
    assert observed["prepare_kwargs"]["recover_existing"] is True
    assert observed["complete_kwargs"]["execution_receipt_path"] == (outcome.execution_receipt_path)
    assert observed["complete_kwargs"]["execution_receipt_sha256"] == (
        outcome.execution_receipt_sha256
    )


def test_provider_rejects_changed_completion_authority_after_container_exit(
    offline_package: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_calls: list[str] = []
    _install_fake_analysis(monkeypatch, offline_package, analysis_calls)
    prepared = _prepared_analysis(offline_package, tmp_path)
    lifecycle = _DockerLifecycle(
        lambda: runtime_module.execute_offline_analysis_once(
            offline_package.admission_path,
        )
    )
    claimed, phase_claim, completion_claimed, completion_phase_claim = (
        _matching_analysis_authorities(offline_package.admission)
    )
    completion_claimed.state.record_sha256 = _digest("provider-state-moved")
    monkeypatch.setattr(provider_module, "_validate_authority", lambda *args: None)

    def fresh_claim_supplier() -> tuple[object, object]:
        lifecycle.events.append("completion-authority")
        return completion_claimed, completion_phase_claim

    with pytest.raises(
        provider_module.OfflineAnalysisProviderError,
        match="refreshed completion authority differs",
    ):
        provider_module.execute_prepared_offline_analysis(
            prepared,
            claimed,
            phase_claim,
            object(),
            fresh_claim_supplier=fresh_claim_supplier,
            docker_run=lifecycle,
        )

    assert analysis_calls == ["compute"]
    assert lifecycle.events.index("completion-authority") > lifecycle.events.index("start")
    assert lifecycle.events.count("remove") == 1
    assert lifecycle.events[-2:] == ["version", "inspect-absent"]
    assert lifecycle.present is False
    assert not prepared.execution_receipt_path.exists()


def test_provider_timeout_force_removes_container_and_proves_absence(
    offline_package: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_analysis(offline_package, tmp_path)
    lifecycle = _DockerLifecycle(subprocess.TimeoutExpired(cmd="docker start", timeout=60))
    monkeypatch.setattr(provider_module, "_validate_authority", lambda *args: None)

    with pytest.raises(
        provider_module.OfflineAnalysisProviderError,
        match="scientific container execution failed to start",
    ):
        provider_module.execute_prepared_offline_analysis(
            prepared,
            object(),
            object(),
            object(),
            fresh_claim_supplier=lambda: (_ for _ in ()).throw(
                AssertionError("timeout cannot reach completion authority")
            ),
            docker_run=lifecycle,
        )

    assert lifecycle.present is False
    assert lifecycle.events.count("create") == 1
    assert lifecycle.events.count("start") == 1
    assert lifecycle.events.count("remove") == 1
    assert lifecycle.events[-2:] == ["version", "inspect-absent"]
    assert not prepared.execution_receipt_path.exists()


def test_provider_rejects_authority_drift_immediately_after_pull(
    offline_package: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_analysis(offline_package, tmp_path)
    lifecycle = _DockerLifecycle(lambda: None)
    checks = 0

    def validate(*args: object) -> None:
        nonlocal checks
        del args
        checks += 1
        if checks == 3:
            raise provider_module.OfflineAnalysisProviderError("authority moved after pull")

    monkeypatch.setattr(provider_module, "_validate_authority", validate)

    with pytest.raises(
        provider_module.OfflineAnalysisProviderError,
        match="authority moved after pull",
    ):
        provider_module.execute_prepared_offline_analysis(
            prepared,
            object(),
            object(),
            object(),
            fresh_claim_supplier=lambda: (_ for _ in ()).throw(
                AssertionError("stale authority cannot reach completion")
            ),
            docker_run=lifecycle,
        )

    assert checks == 3
    assert "pull" in lifecycle.events
    assert "create" not in lifecycle.events
    assert "start" not in lifecycle.events
    assert lifecycle.events.count("remove") == 0
    assert lifecycle.events[-2:] == ["inspect-absent", "pull"]
    assert not prepared.execution_receipt_path.exists()


def test_docker_mount_source_with_comma_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(provider_module.OfflineAnalysisProviderError, match="comma"):
        provider_module._mount_argument(
            tmp_path / "ambiguous,source",
            "/analysis-input",
            read_only=True,
        )


def test_stale_provider_state_cannot_mint_admission() -> None:
    claimed = object.__new__(VerifiedProviderPredecessor)
    object.__setattr__(
        claimed,
        "records",
        (SimpleNamespace(state="ANALYSIS_CLAIMED"),),
    )
    object.__setattr__(claimed, "evidences", (SimpleNamespace(transition_id="a" * 40),))

    def stale() -> None:
        raise RuntimeError("provider state moved")

    object.__setattr__(claimed, "_fresh_revalidator", stale)
    phase_claim = object.__new__(VerifiedPhaseClaimCapability)
    plan = object.__new__(ProviderPhasePlan)
    object.__setattr__(plan, "phase", ANALYSIS_PHASE)
    with pytest.raises(Exception, match="revalidation failed"):
        provider_module._validate_authority(claimed, phase_claim, plan)


def test_runtime_module_has_no_github_or_process_transport_imports() -> None:
    source = Path(runtime_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        name.startswith(("github", "requests", "socket", "subprocess")) for name in imports
    )
