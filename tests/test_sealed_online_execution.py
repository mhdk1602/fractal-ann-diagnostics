from __future__ import annotations

import hashlib
import inspect
import io
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fractal_ann_diagnostics.sealed_online_execution as sealed_online
from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    VerifiedArtifact,
)
from fractal_ann_diagnostics.audit import VerifiedProvenanceRegistry
from fractal_ann_diagnostics.authorized_index_store import AuthorizedIndexStoreError
from fractal_ann_diagnostics.controller import (
    ControllerConfig,
    GovernedRetriever,
    RuleController,
)
from fractal_ann_diagnostics.corpora import CorpusDocument, EvidenceQuery, NormalizedCorpus
from fractal_ann_diagnostics.execution_claim import RuntimeClaimReceipt
from fractal_ann_diagnostics.label_separation import (
    OnlineDocument,
    OnlineExecutionArtifact,
    OnlineTrial,
)
from fractal_ann_diagnostics.online_runner import (
    FrozenFeatureContext,
    OnlineTrialRuntime,
    run_online_action_matrix,
)
from fractal_ann_diagnostics.policy import AuthorizationPolicy
from fractal_ann_diagnostics.policy_intervention import (
    OPACompiledMaskData,
    OPAMaskAssignment,
)
from fractal_ann_diagnostics.sealed_online_execution import (
    SealedOnlineAttemptReceipt,
    SealedOnlineExecutionError,
    _run_sealed_online_once_from_objects,
    load_sealed_online_attempt_receipt,
    load_sealed_online_result_receipt,
    run_sealed_online_once,
    sealed_online_attempt_path,
    sealed_online_result_path,
    verify_sealed_online_outputs,
)
from fractal_ann_diagnostics.study import SealedRunReceipt
from fractal_ann_diagnostics.trial_runtime import TrialRuntimeAdmission

_MANIFEST = "a" * 64
_PARTITION = "b" * 64
_PSEUDONYM_KEY = b"sealed-online-pseudonym-key-material-32-bytes"
_COMPONENTS = ("application", "controller", "corpus", "embedding", "index", "policy")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _opa_data() -> OPACompiledMaskData:
    return OPACompiledMaskData(
        document_count=3,
        document_universe_sha256="b" * 64,
        mask_catalog_sha256="c" * 64,
        policy_revision=f"sha256:{'d' * 64}",
        assignments=(
            OPAMaskAssignment(
                subject="analyst",
                policy_state="low",
                mask_id="mask-low",
                mask_sha256="e" * 64,
                authorized_count=1,
            ),
        ),
    )


def _harness() -> tuple[
    object,
    OnlineExecutionArtifact,
    SealedRunReceipt,
    GovernedRetriever,
    VerifiedProvenanceRegistry,
    dict[str, OnlineTrialRuntime],
]:
    corpus = NormalizedCorpus(
        name="scifact",
        stage="sealed",
        documents=tuple(
            CorpusDocument(
                document_id=index,
                external_id=f"document-{index}",
                title=f"Document {index}",
                text=f"fixed text {index}",
                source_uri=f"fixture://document/{index}",
                content_hash=f"sha256:{_digest(f'content:{index}')}",
            )
            for index in range(12)
        ),
        queries=(
            EvidenceQuery(
                query_id="query-0",
                query_family="family-0",
                text="find fixed evidence",
                corpus="scifact",
                stage="sealed",
                answer=None,
                gold_evidence=None,
            ),
        ),
    )
    execution = OnlineExecutionArtifact(
        key_id="sealed-online-key",
        corpus=corpus.name,
        stage=corpus.stage,
        documents=tuple(
            OnlineDocument(
                document_id=row.document_id,
                external_id=row.external_id,
                title=row.title,
                text=row.text,
                source_uri=row.source_uri,
                content_hash=row.content_hash,
            )
            for row in corpus.documents
        ),
        trials=(
            OnlineTrial(
                trial_key=_digest("trial-0"),
                family_key=_digest("family-0"),
                text="find fixed evidence",
                corpus=corpus.name,
                stage=corpus.stage,
            ),
        ),
    )
    verification = ArtifactVerificationReceipt(
        manifest_sha256=_MANIFEST,
        artifacts=tuple(
            VerifiedArtifact(
                artifact_id=f"{component}-artifact",
                relative_path=f"objects/{component}.bin",
                kind="file",
                exact=True,
                expected_sha256=_digest(component),
                verified_sha256=_digest(component),
                file_count=1,
                directory_count=0,
                byte_count=1,
                observed_file_count=1,
                observed_directory_count=0,
                observed_byte_count=1,
            )
            for component in _COMPONENTS
        ),
    )
    registry = VerifiedProvenanceRegistry(
        corpus=corpus,
        verification_receipt=verification,
        component_artifact_ids={component: f"{component}-artifact" for component in _COMPONENTS},
    )
    policy = AuthorizationPolicy(
        roles=("analyst",),
        visibility=np.ones((1, len(corpus.documents)), dtype=bool),
        version="registered-policy",
        document_universe_sha256=registry.document_universe_sha256,
    )
    retriever = GovernedRetriever(
        np.random.default_rng(4).normal(size=(12, 4)).astype(np.float32),
        policy,
        "analyst",
        expected_document_universe_sha256=registry.document_universe_sha256,
        controller=RuleController(
            ControllerConfig(
                low_ef=8,
                high_ef=12,
                probe_k=8,
                exact_scan_threshold=0,
                high_effort_threshold=0.98,
                exact_threshold=1.0,
            )
        ),
    )
    receipt = SealedRunReceipt(
        manifest_sha256=_MANIFEST,
        protocol_version="0.3.0",
        started_at_utc="2026-07-14T12:00:00+00:00",
        runner_identity="sealed-online-test-runner",
        code_commit="c" * 40,
        runner_image=f"ghcr.io/example/runner@sha256:{'d' * 64}",
        protocol_registration_receipt_uri="file:///controlled/protocol-receipt.json",
        protocol_registration_receipt_sha256="e" * 64,
        protocol_registration_record_uri="file:///controlled/protocol-record.json",
        verification_receipt_uri="file:///controlled/verification.json",
        verification_receipt_sha256="f" * 64,
        receipt_uri="file:///controlled/run.json",
    )
    context = FrozenFeatureContext(
        version_lag=1.0,
        backend="hnswlib-0.8.0",
        drift_family="revision",
        policy_complexity=3.0,
    )
    runtimes = {
        execution.trials[0].trial_key: OnlineTrialRuntime(
            active_query_vector=np.asarray([1.0, 0.0, 0.5, -0.5]),
            current_truth_query_vector=np.asarray([0.9, 0.1, 0.4, -0.4]),
            feature_context=context,
            environment={"policy_state": "steady"},
        )
    }
    artifacts = run_online_action_matrix(
        execution=execution,
        run_receipt=receipt,
        retriever=retriever,
        provenance_registry=registry,
        trial_runtimes=runtimes,
        permutation_seed=71,
        expected_policy_version="registered-policy",
        query_partition_audit_sha256=_PARTITION,
        pseudonym_key=_PSEUDONYM_KEY,
        pseudonym_key_id="sealed-online-pseudonym",
        k=3,
        occurred_at_factory=(
            lambda _trial, _action, sequence: f"2026-07-14T12:00:{sequence:02d}+00:00"
        ),
    )
    return artifacts, execution, receipt, retriever, registry, runtimes


def _attempt(execution: OnlineExecutionArtifact, receipt: SealedRunReceipt, root: Path):
    return SealedOnlineAttemptReceipt(
        manifest_sha256=_MANIFEST,
        run_receipt_sha256=receipt.binding_sha256,
        online_custody_admission_receipt_sha256="1" * 64,
        required_artifact_bindings_sha256="2" * 64,
        runtime_attestation_plan_sha256="7" * 64,
        runtime_attestation_receipt_sha256="8" * 64,
        runtime_claim_receipt_sha256="0" * 64,
        claim_state_sha256="0" * 64,
        claim_ledger_commit="0" * 40,
        provider_identity_sha256="0" * 64,
        beacon_receipt_sha256="0" * 64,
        beacon_bytes_sha256="0" * 64,
        derived_seed_sha256="0" * 64,
        output_aggregate_identity="0" * 64,
        trial_runtime_admission_receipt_sha256="3" * 64,
        authorized_index_store_receipt_sha256="4" * 64,
        execution_artifact_sha256=execution.artifact_sha256,
        query_partition_audit_sha256=_PARTITION,
        controller_config_sha256="5" * 64,
        query_feature_sources_sha256="6" * 64,
        policy_revision="registered-policy",
        permutation_seed=71,
        k=3,
        policy_action="retrieve",
        partition_label="primary",
        pseudonym_key_id="sealed-online-pseudonym",
        runner_identity=receipt.runner_identity,
        result_directory_uri=root.as_uri(),
    )


def _runtime_claim(receipt: SealedRunReceipt) -> RuntimeClaimReceipt:
    derived_seed = f"{71:016x}" + "0" * 48
    return RuntimeClaimReceipt(
        manifest_sha256=receipt.manifest_sha256,
        run_receipt_sha256=receipt.binding_sha256,
        c1_commit="c" * 40,
        claim_contract_sha256="1" * 64,
        claim_state_sha256="2" * 64,
        claim_ledger_commit="3" * 40,
        provider_identity_sha256="4" * 64,
        live_execute_job_receipt_sha256="5" * 64,
        execute_job_id=42,
        beacon_receipt_sha256="6" * 64,
        beacon_bytes_sha256="7" * 64,
        design_seed_sha256="8" * 64,
        derived_seed_sha256=derived_seed,
        permutation_seed=71,
        output_aggregate_identity="9" * 64,
    )


def _run_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifacts, execution, receipt, retriever, registry, runtimes = _harness()
    output = tmp_path / "sealed-output"
    output.mkdir(mode=0o700)
    attempt = _attempt(execution, receipt, output)
    calls = 0

    monkeypatch.setattr(sealed_online, "_attempt_receipt", lambda **_kwargs: attempt)

    def invoke(**_kwargs: object):
        nonlocal calls
        calls += 1
        return artifacts

    monkeypatch.setattr(sealed_online, "run_admitted_online_matrix", invoke)
    result = _run_sealed_online_once_from_objects(
        output_root=output,
        admission_receipt=object(),  # type: ignore[arg-type]
        required_artifacts=object(),  # type: ignore[arg-type]
        execution=execution,
        run_receipt=receipt,
        retriever=retriever,
        provenance_registry=registry,
        trial_runtimes=runtimes,
        runtime_receipt=object(),  # type: ignore[arg-type]
        index_receipt=object(),  # type: ignore[arg-type]
        permutation_seed=71,
        expected_policy_version="registered-policy",
        query_partition_audit_sha256=_PARTITION,
        pseudonym_key=_PSEUDONYM_KEY,
        pseudonym_key_id="sealed-online-pseudonym",
        k=3,
    )
    return result, output, calls


def test_one_attempt_persists_and_reverifies_the_full_prelabel_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted, output, calls = _run_once(tmp_path, monkeypatch)

    assert calls == 1
    assert len(persisted.predictions.predictions) == 1
    assert (
        load_sealed_online_attempt_receipt(sealed_online_attempt_path(output, _MANIFEST))
        == persisted.attempt_receipt
    )
    assert (
        load_sealed_online_result_receipt(sealed_online_result_path(output, _MANIFEST))
        == persisted.result_receipt
    )
    verify_sealed_online_outputs(persisted.result_receipt, output_root=output)
    assert {pin.role for pin in persisted.result_receipt.outputs} == {
        "action-panel",
        "action-panel-admission",
        "audit-chain",
        "cache-preparation",
        "execution-order",
        "predictions",
    }


def test_existing_attempt_blocks_before_a_second_online_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted, output, _ = _run_once(tmp_path, monkeypatch)
    calls = 0

    def forbidden(**_kwargs: object):
        nonlocal calls
        calls += 1
        return persisted.artifacts

    monkeypatch.setattr(sealed_online, "run_admitted_online_matrix", forbidden)
    with pytest.raises(SealedOnlineExecutionError, match="already exists"):
        _run_sealed_online_once_from_objects(
            output_root=output,
            admission_receipt=object(),  # type: ignore[arg-type]
            required_artifacts=object(),  # type: ignore[arg-type]
            execution=object(),
            run_receipt=object(),  # type: ignore[arg-type]
            retriever=object(),  # type: ignore[arg-type]
            provenance_registry=object(),  # type: ignore[arg-type]
            trial_runtimes={},
            runtime_receipt=object(),  # type: ignore[arg-type]
            index_receipt=object(),  # type: ignore[arg-type]
            permutation_seed=71,
            expected_policy_version="registered-policy",
            query_partition_audit_sha256=_PARTITION,
            pseudonym_key=_PSEUDONYM_KEY,
            pseudonym_key_id="sealed-online-pseudonym",
            k=3,
        )
    assert calls == 0


def test_output_mutation_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted, output, _ = _run_once(tmp_path, monkeypatch)
    panel = next(pin for pin in persisted.result_receipt.outputs if pin.role == "action-panel")
    (output / panel.filename).write_bytes(b"substituted\n")

    with pytest.raises(SealedOnlineExecutionError, match="changed bytes"):
        verify_sealed_online_outputs(persisted.result_receipt, output_root=output)


def test_production_signature_has_no_scientific_or_runtime_overrides() -> None:
    parameters = set(inspect.signature(run_sealed_online_once).parameters)
    assert parameters == {
        "admission_receipt",
        "artifact_root",
        "authorized_index_store_root",
        "expected_authorized_index_store_receipt_sha256",
        "expected_policy_intervention_receipt_sha256",
        "expected_pseudonym_key_sha256",
        "expected_runtime_receipt_sha256",
        "output_root",
        "policy_intervention_root",
        "pseudonym_key_path",
        "required_artifacts",
        "run_receipt",
        "runtime_admission",
        "runtime_attestation_plan_path",
        "expected_runtime_attestation_plan_sha256",
        "runtime_attestation_receipt_path",
        "expected_runtime_attestation_receipt_sha256",
        "runtime_claim_receipt",
    }
    assert parameters.isdisjoint(
        {
            "controller_config",
            "environment",
            "exact_truth_vectors",
            "execution",
            "expected_policy_version",
            "feature_context",
            "index_receipt",
            "k",
            "permutation_seed",
            "policy_action",
            "retriever",
            "runtime_receipt",
            "runtime_attestation_plan",
            "runtime_attestation_receipt",
            "runtime_observation",
            "runtime_probe",
            "trial_runtimes",
            "opa_binary",
            "opa_endpoint",
            "opa_process",
            "opa_rego",
        }
    )


def test_production_opa_command_is_closed_and_image_relative() -> None:
    data_path = Path("/tmp/fractal-confirmatory-opa-test/opa-data.json")

    assert sealed_online.PRODUCTION_OPA_ENDPOINT == (
        "http://127.0.0.1:8181/v1/data/fractal_auth/retrieval/mask_decision"
    )
    assert sealed_online._production_opa_command(data_path) == (
        "/usr/local/bin/opa",
        "run",
        "--server",
        "--addr=127.0.0.1:8181",
        "--authentication=off",
        "--authorization=off",
        "--log-format=json",
        "--log-level=error",
        "--max-errors=1",
        "--ready-timeout=5",
        "--set=decision_logs.console=true",
        "--shutdown-grace-period=1",
        "--shutdown-wait-period=0",
        "--skip-version-check",
        "/opt/app/policy/opa_compiled_masks.rego",
        f"fractal:{data_path}",
    )


def test_production_opa_start_uses_no_shell_and_a_minimal_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self) -> None:
            self.stderr = io.BytesIO()
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, *, timeout: float) -> int:
            assert timeout == sealed_online._PRODUCTION_OPA_STOP_TIMEOUT_SECONDS
            assert self.returncode is not None
            return self.returncode

    process = FakeProcess()

    def popen(command: tuple[str, ...], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(sealed_online, "_verify_production_opa_image_artifacts", lambda: None)
    monkeypatch.setattr(sealed_online, "_assert_production_opa_port_vacant", lambda: None)
    monkeypatch.setattr(sealed_online, "_wait_for_production_opa", lambda *_args: None)
    monkeypatch.setattr(sealed_online.subprocess, "Popen", popen)

    handle = sealed_online._start_production_opa(
        Path("/tmp/fractal-confirmatory-opa-test/opa-data.json"),
        _opa_data(),
    )
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert captured["command"] == sealed_online._production_opa_command(
        Path("/tmp/fractal-confirmatory-opa-test/opa-data.json")
    )
    assert kwargs["shell"] is False
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert kwargs["umask"] == 0o077
    assert kwargs["cwd"] == "/"
    assert kwargs["env"] == {
        "HOME": "/home/runner",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    }
    sealed_online._stop_production_opa(handle)


def test_production_opa_stderr_is_drained_but_retained_within_a_fixed_bound() -> None:
    collector = sealed_online._BoundedOPAStderr(
        io.BytesIO(b"x" * (sealed_online._PRODUCTION_OPA_STDERR_BYTES + 4096))
    )
    collector.finish()

    assert len(collector._buffer) == sealed_online._PRODUCTION_OPA_STDERR_BYTES
    assert collector._total_bytes == sealed_online._PRODUCTION_OPA_STDERR_BYTES + 4096
    assert collector.diagnostic().endswith("[truncated]")


def test_production_opa_cannot_open_policy_data_before_the_attempt_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, execution, receipt, _, _, _ = _harness()
    attempt = _attempt(execution, receipt, tmp_path)
    opened: list[str] = []

    def forbidden(**_kwargs: object) -> OPACompiledMaskData:
        opened.append("opa-data")
        return _opa_data()

    monkeypatch.setattr(sealed_online, "_admit_production_opa_data", forbidden)
    missing = tmp_path / "missing-attempt.json"

    with pytest.raises(SealedOnlineExecutionError, match="before the sealed online attempt"):
        with sealed_online._production_opa_sidecar(
            attempt_path=missing,
            attempt=attempt,
            policy_root=tmp_path,
            runtime_admission=object(),  # type: ignore[arg-type]
            mask_store=object(),  # type: ignore[arg-type]
            expected_policy_receipt_sha256="f" * 64,
        ):
            raise AssertionError("sidecar unexpectedly opened")

    assert opened == []


def test_production_opa_sidecar_starts_after_marker_and_closes_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    data = _opa_data()

    class FakeProcess:
        def poll(self) -> None:
            return None

    handle = SimpleNamespace(process=FakeProcess())

    def marker(*_args: object) -> None:
        events.append("attempt")

    def admit(**_kwargs: object) -> OPACompiledMaskData:
        assert events == ["attempt"]
        events.append("data")
        return data

    def start(data_path: Path, admitted: OPACompiledMaskData) -> object:
        assert events == ["attempt", "data"]
        assert admitted is data
        assert data_path.read_bytes() == data.canonical_file_bytes()
        events.append("start")
        return handle

    def stop(observed: object) -> None:
        assert observed is handle
        events.append("stop")

    monkeypatch.setattr(sealed_online, "_PRODUCTION_OPA_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr(sealed_online, "_assert_production_attempt_marker", marker)
    monkeypatch.setattr(sealed_online, "_admit_production_opa_data", admit)
    monkeypatch.setattr(sealed_online, "_start_production_opa", start)
    monkeypatch.setattr(sealed_online, "_stop_production_opa", stop)
    monkeypatch.setattr(
        sealed_online,
        "OpenPolicyAgentMaskDecisionPoint",
        lambda endpoint, _store: (endpoint, "pdp"),
    )

    with sealed_online._production_opa_sidecar(
        attempt_path=tmp_path / "attempt.json",
        attempt=object(),  # type: ignore[arg-type]
        policy_root=tmp_path,
        runtime_admission=object(),  # type: ignore[arg-type]
        mask_store=object(),  # type: ignore[arg-type]
        expected_policy_receipt_sha256="f" * 64,
    ) as policy:
        assert policy == (sealed_online.PRODUCTION_OPA_ENDPOINT, "pdp")
        assert events == ["attempt", "data", "start"]
        events.append("workload")

    assert events == ["attempt", "data", "start", "workload", "stop"]
    assert list(tmp_path.glob("fractal-confirmatory-opa-*")) == []


def test_production_opa_cleanup_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _opa_data()
    handle = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(sealed_online, "_PRODUCTION_OPA_SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr(sealed_online, "_assert_production_attempt_marker", lambda *_args: None)
    monkeypatch.setattr(sealed_online, "_admit_production_opa_data", lambda **_kwargs: data)
    monkeypatch.setattr(sealed_online, "_start_production_opa", lambda *_args: handle)
    monkeypatch.setattr(
        sealed_online,
        "_stop_production_opa",
        lambda _handle: (_ for _ in ()).throw(
            SealedOnlineExecutionError("OPA remained alive after cleanup")
        ),
    )
    monkeypatch.setattr(
        sealed_online,
        "OpenPolicyAgentMaskDecisionPoint",
        lambda *_args: object(),
    )

    with pytest.raises(SealedOnlineExecutionError, match="remained alive"):
        with sealed_online._production_opa_sidecar(
            attempt_path=tmp_path / "attempt.json",
            attempt=object(),  # type: ignore[arg-type]
            policy_root=tmp_path,
            runtime_admission=object(),  # type: ignore[arg-type]
            mask_store=object(),  # type: ignore[arg-type]
            expected_policy_receipt_sha256="f" * 64,
        ):
            pass


def test_production_opa_data_must_match_the_runtime_assignment_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _opa_data()
    policy_receipt_sha256 = "f" * 64
    receipt = SimpleNamespace(
        artifact_sha256=policy_receipt_sha256,
        artifacts=(
            SimpleNamespace(
                role="opa-data",
                path="opa-data.json",
                byte_count=len(data.canonical_file_bytes()),
                sha256=hashlib.sha256(data.canonical_file_bytes()).hexdigest(),
            ),
        ),
    )
    mask_store = SimpleNamespace(
        catalog_sha256=data.mask_catalog_sha256,
        catalog=SimpleNamespace(
            document_count=data.document_count,
            document_universe_sha256=data.document_universe_sha256,
            policy_revision=data.policy_revision,
        ),
    )
    mismatched_group = SimpleNamespace(
        subject="analyst",
        policy_state="low",
        mask_id="mask-other",
        mask_sha256="e" * 64,
        authorized_count=1,
    )
    runtime = SimpleNamespace(receipt=SimpleNamespace(groups=(mismatched_group,)))
    monkeypatch.setattr(sealed_online, "load_opa_compiled_mask_data", lambda _path: data)
    monkeypatch.setattr(
        sealed_online,
        "load_policy_intervention_receipt",
        lambda _path: receipt,
    )

    with pytest.raises(SealedOnlineExecutionError, match="runtime groups"):
        sealed_online._admit_production_opa_data(
            policy_root=tmp_path,
            runtime_admission=runtime,  # type: ignore[arg-type]
            mask_store=mask_store,  # type: ignore[arg-type]
            expected_policy_receipt_sha256=policy_receipt_sha256,
        )


def test_live_opa_cache_must_equal_the_runtime_environment_mask() -> None:
    artifacts, _, _, _, _, _ = _harness()
    prepared = artifacts.cache_preparation_receipt.rows[0]
    group = SimpleNamespace(
        environment_sha256=prepared.environment_sha256,
        mask_id="registered-mask",
        mask_sha256="1" * 64,
        authorized_count=prepared.authorized_count,
    )
    admission = SimpleNamespace(receipt=SimpleNamespace(groups=(group,)))
    admitted_mask = np.ones(prepared.authorized_count, dtype=bool)
    mask_store = SimpleNamespace(mask=lambda *_args, **_kwargs: admitted_mask)

    sealed_online._verify_cache_against_runtime_schedule(
        artifacts=artifacts,
        runtime_admission=admission,  # type: ignore[arg-type]
        mask_store=mask_store,  # type: ignore[arg-type]
    )

    other_mask = admitted_mask.copy()
    other_mask[-1] = False
    with pytest.raises(SealedOnlineExecutionError, match="OPA mask selection"):
        sealed_online._verify_cache_against_runtime_schedule(
            artifacts=artifacts,
            runtime_admission=admission,  # type: ignore[arg-type]
            mask_store=SimpleNamespace(  # type: ignore[arg-type]
                mask=lambda *_args, **_kwargs: other_mask
            ),
        )


def test_runtime_attestation_failure_precedes_attempt_and_every_workload_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, execution, receipt, _, _, _ = _harness()
    output = tmp_path / "attestation-failure"
    output.mkdir(mode=0o700)
    runtime_admission = TrialRuntimeAdmission(
        plan=SimpleNamespace(artifact_sha256=execution.artifact_sha256),
        partition_audit_path=(tmp_path / "partition-audit.json").resolve(),
        query_package_root=(tmp_path / "queries").resolve(),
        staged_root=(tmp_path / "staged").resolve(),
        embedding_store_root=(tmp_path / "embeddings").resolve(),
        schedule_path=(tmp_path / "schedule.jsonl").resolve(),
        feature_bindings=(),
        receipt=SimpleNamespace(receipt_sha256="3" * 64),
    )
    source_calls: list[str] = []

    def reject_runtime(**_kwargs: object) -> tuple[object, object]:
        raise SealedOnlineExecutionError("runtime attestation failed: observed environment differs")

    def forbidden_source(*_args: object, **_kwargs: object) -> object:
        source_calls.append("opened")
        raise AssertionError("a workload source opened before runtime admission")

    monkeypatch.setattr(
        sealed_online,
        "_admit_production_runtime_attestation",
        reject_runtime,
    )
    monkeypatch.setattr(sealed_online, "_production_attempt_receipt", forbidden_source)
    monkeypatch.setattr(sealed_online, "load_authorized_index_store_receipt", forbidden_source)
    monkeypatch.setattr(sealed_online, "load_trial_runtime", forbidden_source)
    monkeypatch.setattr(sealed_online, "CompiledPolicyMaskStore", forbidden_source)
    monkeypatch.setattr(sealed_online, "_load_pseudonym_key", forbidden_source)
    monkeypatch.setattr(sealed_online, "open_verified_document_matrices", forbidden_source)

    with pytest.raises(SealedOnlineExecutionError, match="observed environment differs"):
        run_sealed_online_once(
            output_root=output,
            admission_receipt=object(),  # type: ignore[arg-type]
            required_artifacts=object(),  # type: ignore[arg-type]
            run_receipt=receipt,
            runtime_admission=runtime_admission,
            runtime_attestation_plan_path=(tmp_path / "runtime-plan.json").resolve(),
            expected_runtime_attestation_plan_sha256="7" * 64,
            runtime_attestation_receipt_path=(tmp_path / "runtime-receipt.json").resolve(),
            expected_runtime_attestation_receipt_sha256="8" * 64,
            expected_runtime_receipt_sha256="3" * 64,
            artifact_root=(tmp_path / "artifacts").resolve(),
            authorized_index_store_root=(tmp_path / "indexes").resolve(),
            expected_authorized_index_store_receipt_sha256="4" * 64,
            policy_intervention_root=(tmp_path / "policy").resolve(),
            expected_policy_intervention_receipt_sha256="6" * 64,
            pseudonym_key_path=(tmp_path / "pseudonym.key").resolve(),
            expected_pseudonym_key_sha256="5" * 64,
            runtime_claim_receipt=_runtime_claim(receipt),
        )

    assert source_calls == []
    assert not sealed_online_attempt_path(output, _MANIFEST).exists()


def test_production_consumes_attempt_before_loading_and_blocks_result_on_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, execution, receipt, _, _, _ = _harness()
    output = tmp_path / "sealed-production-output"
    output.mkdir(mode=0o700)
    attempt = _attempt(execution, receipt, output)
    attempt_path = sealed_online_attempt_path(output, _MANIFEST)
    result_path = sealed_online_result_path(output, _MANIFEST)
    runtime_receipt = SimpleNamespace(
        receipt_sha256="3" * 64,
        policy_bundle_revision="registered-policy",
        mask_catalog_sha256="7" * 64,
        embedding_store_receipt_sha256="8" * 64,
        execution_artifact_sha256=execution.artifact_sha256,
        permutation_seed=71,
        query_partition_audit_sha256=_PARTITION,
    )
    plan = SimpleNamespace(
        artifact_sha256=execution.artifact_sha256,
        document_universe_sha256="9" * 64,
        trial_keys=(execution.trials[0].trial_key,),
    )
    runtime_admission = TrialRuntimeAdmission(
        receipt=runtime_receipt,
        plan=plan,
        partition_audit_path=(tmp_path / "partition-audit.json").resolve(),
        query_package_root=(tmp_path / "queries").resolve(),
        staged_root=(tmp_path / "staged").resolve(),
        embedding_store_root=(tmp_path / "embeddings").resolve(),
        schedule_path=(tmp_path / "schedule.jsonl").resolve(),
        feature_bindings=(),
    )

    def admit_runtime(**_kwargs: object) -> tuple[SimpleNamespace, SimpleNamespace]:
        assert not attempt_path.exists()
        return (
            SimpleNamespace(
                plan_sha256="7" * 64,
                opa_binary=SimpleNamespace(path="/usr/local/bin/opa"),
            ),
            SimpleNamespace(receipt_sha256="8" * 64),
        )

    monkeypatch.setattr(
        sealed_online,
        "_admit_production_runtime_attestation",
        admit_runtime,
    )

    monkeypatch.setattr(
        sealed_online,
        "_production_attempt_receipt",
        lambda **_kwargs: attempt,
    )

    def load_index(_root: Path) -> SimpleNamespace:
        assert attempt_path.exists()
        return SimpleNamespace(policy_receipt_sha256="6" * 64)

    monkeypatch.setattr(sealed_online, "load_authorized_index_store_receipt", load_index)
    monkeypatch.setattr(
        sealed_online,
        "_verify_production_source_bindings",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(sealed_online, "HnswlibBackend", lambda: object())
    monkeypatch.setattr(
        sealed_online,
        "VerifiedAuthorizedIndexProvider",
        lambda *_args, **_kwargs: SimpleNamespace(retrieval_metric="euclidean"),
    )
    loaded = SimpleNamespace(
        execution=SimpleNamespace(
            plan=plan,
            artifact_sha256=execution.artifact_sha256,
        ),
        trial_runtimes={execution.trials[0].trial_key: object()},
    )
    monkeypatch.setattr(sealed_online, "load_trial_runtime", lambda _value: loaded)
    catalog = SimpleNamespace(
        policy_revision="registered-policy",
        document_universe_sha256="9" * 64,
    )
    monkeypatch.setattr(
        sealed_online,
        "CompiledPolicyMaskStore",
        lambda _path: SimpleNamespace(
            catalog_sha256="7" * 64,
            catalog=catalog,
            verify_all=lambda: ("mask",),
        ),
    )
    monkeypatch.setattr(
        sealed_online,
        "OpenPolicyAgentMaskDecisionPoint",
        lambda *_args, **_kwargs: object(),
    )

    @contextmanager
    def opa_sidecar_context(**_kwargs: object):
        assert attempt_path.exists()
        yield object()

    monkeypatch.setattr(
        sealed_online,
        "_production_opa_sidecar",
        opa_sidecar_context,
    )
    monkeypatch.setattr(
        sealed_online,
        "_production_policy_transitions",
        lambda **_kwargs: {"7" * 64: object()},
    )
    monkeypatch.setattr(
        sealed_online,
        "_load_pseudonym_key",
        lambda *_args, **_kwargs: _PSEUDONYM_KEY,
    )
    monkeypatch.setattr(sealed_online, "_production_subject", lambda _value: "analyst")
    monkeypatch.setattr(sealed_online, "GovernedRetriever", lambda *_args, **_kwargs: object())

    @contextmanager
    def matrix_context(*_args: object, **_kwargs: object):
        assert attempt_path.exists()
        yield SimpleNamespace(
            old_active=np.zeros((2, 2), dtype=np.float32),
            current_truth=np.ones((2, 2), dtype=np.float32),
        )
        raise AuthorizedIndexStoreError(
            "current truth document vectors were mutated or substituted while mapped"
        )

    monkeypatch.setattr(sealed_online, "open_verified_document_matrices", matrix_context)
    provenance = SimpleNamespace()
    monkeypatch.setattr(sealed_online, "DigestOnlyProvenanceRegistry", type(provenance))

    @contextmanager
    def provenance_context(*_args: object, **_kwargs: object):
        yield provenance

    monkeypatch.setattr(sealed_online, "open_digest_provenance_registry", provenance_context)
    monkeypatch.setattr(
        sealed_online,
        "_execute_online_objects",
        lambda **_kwargs: (artifacts, object()),
    )
    monkeypatch.setattr(
        sealed_online,
        "_verify_cache_against_runtime_schedule",
        lambda **_kwargs: None,
    )

    with pytest.raises(AuthorizedIndexStoreError, match="substituted"):
        run_sealed_online_once(
            output_root=output,
            admission_receipt=object(),  # type: ignore[arg-type]
            required_artifacts=SimpleNamespace(
                verification_receipt=object(),
                provenance_component_artifact_ids=(),
            ),  # type: ignore[arg-type]
            run_receipt=receipt,
            runtime_admission=runtime_admission,  # type: ignore[arg-type]
            runtime_attestation_plan_path=(tmp_path / "runtime-plan.json").resolve(),
            expected_runtime_attestation_plan_sha256="7" * 64,
            runtime_attestation_receipt_path=(tmp_path / "runtime-receipt.json").resolve(),
            expected_runtime_attestation_receipt_sha256="8" * 64,
            expected_runtime_receipt_sha256="3" * 64,
            artifact_root=(tmp_path / "artifacts").resolve(),
            authorized_index_store_root=(tmp_path / "indexes").resolve(),
            expected_authorized_index_store_receipt_sha256="4" * 64,
            policy_intervention_root=(tmp_path / "policy").resolve(),
            expected_policy_intervention_receipt_sha256="6" * 64,
            pseudonym_key_path=(tmp_path / "pseudonym.key").resolve(),
            expected_pseudonym_key_sha256="5" * 64,
            runtime_claim_receipt=_runtime_claim(receipt),
        )

    assert attempt_path.exists()
    assert not result_path.exists()
    persisted_attempt = load_sealed_online_attempt_receipt(attempt_path)
    assert persisted_attempt.runtime_attestation_plan_sha256 == "7" * 64
    assert persisted_attempt.runtime_attestation_receipt_sha256 == "8" * 64
