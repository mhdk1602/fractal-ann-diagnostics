from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fractal_ann_diagnostics.execution_claim import (
    ACTIVATION_COMMON_OUTPUT_KEYS,
    ACTIVATION_PHASE_OUTPUT_KEYS,
    C1_REGISTRATION_PACKAGE_FILE_COUNT,
    ProviderPhasePlan,
    derive_phase_runner_label,
)
from fractal_ann_diagnostics.provider_workflow_orchestration import (
    C0_REF,
    CLAIM_RECEIPT_SCHEMA,
    OWNER_ID,
    OWNER_LOGIN,
    OWNER_NODE_ID,
    PREPARATION_RECEIPT_SCHEMA,
    PREREQUISITE_RECEIPT_SCHEMA,
    REPOSITORY,
    REPOSITORY_ID,
    REPOSITORY_NODE_ID,
    WORKFLOW_CONTEXT_SCHEMA,
    EvidenceInventoryRow,
    ProviderClaimReceipt,
    ProviderPrerequisiteReceipt,
    ProviderTransitionPreparationReceipt,
    ProviderWorkflowContext,
    ProviderWorkflowOrchestrationError,
    execute_claim_command,
    execute_complete_command,
    execute_fail_command,
    execute_verify_prerequisites_command,
    inventory_sha256,
    load_provider_claim_receipt,
    load_provider_prerequisite_receipt,
    load_provider_transition_preparation_receipt,
    verify_provider_execution_identity,
    write_provider_receipt,
)
from fractal_ann_diagnostics.study import (
    FIXED_CORPORA,
    PROVIDER_PHASE_JOB_NAMES,
    PROVIDER_PHASE_WORKFLOWS,
)
from fractal_ann_diagnostics.suite_attempt import suite_attempt_id


def _digest(seed: str) -> str:
    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()


def _environment(*, phase: str = "online", job: str = "claim") -> dict[str, str]:
    workflow = PROVIDER_PHASE_WORKFLOWS[phase]
    execute = job == "execute"
    return {
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITHUB_ACTOR": OWNER_LOGIN,
        "GITHUB_ACTOR_ID": str(OWNER_ID),
        "GITHUB_API_URL": "https://api.github.com",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_GRAPHQL_URL": "https://api.github.com/graphql",
        "GITHUB_JOB": job,
        "GITHUB_REF": C0_REF,
        "GITHUB_REF_NAME": "confirmatory-apparatus-c0",
        "GITHUB_REF_PROTECTED": "true",
        "GITHUB_REF_TYPE": "tag",
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_REPOSITORY_ID": str(REPOSITORY_ID),
        "GITHUB_REPOSITORY_OWNER": OWNER_LOGIN,
        "GITHUB_REPOSITORY_OWNER_ID": str(OWNER_ID),
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "983421",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_TRIGGERING_ACTOR": OWNER_LOGIN,
        "GITHUB_WORKFLOW_REF": f"{REPOSITORY}/{workflow}@{C0_REF}",
        "GITHUB_WORKFLOW_SHA": "a" * 40,
        "RUNNER_ARCH": "ARM64" if execute else "X64",
        "RUNNER_ENVIRONMENT": "self-hosted" if execute else "github-hosted",
        "RUNNER_OS": "macOS" if execute else "Linux",
    }


def _provider_plan() -> ProviderPhasePlan:
    plan = object.__new__(ProviderPhasePlan)
    values = {
        "phase": "online",
        "repository": REPOSITORY,
        "workflow_path": PROVIDER_PHASE_WORKFLOWS["online"],
        "workflow_ref": (f"{REPOSITORY}/{PROVIDER_PHASE_WORKFLOWS['online']}@{C0_REF}"),
        "workflow_sha": "a" * 40,
        "run_head_branch": "confirmatory-apparatus-c0",
        "claim_job_name": PROVIDER_PHASE_JOB_NAMES["online"][0],
        "execute_job_name": PROVIDER_PHASE_JOB_NAMES["online"][1],
        "claim_nonce": _digest("online-claim-nonce"),
        "runner_id": 991,
        "runner_name": "fractal-confirmatory-online",
        "runner_group_id": None,
        "runner_version": "2.335.1",
        "runner_archive_sha256": _digest("runner-archive"),
        "provider_operating_system": "macOS",
        "provider_architecture": "ARM64",
        "runtime_probe_receipt_sha256": _digest("runtime-probe"),
        "host_tools": SimpleNamespace(contract_sha256=_digest("host-tools")),
    }
    for name, value in values.items():
        object.__setattr__(plan, name, value)
    return plan


class _ProviderIdentityApi:
    def __init__(self, plan: ProviderPhasePlan) -> None:
        owner = {"id": OWNER_ID, "login": OWNER_LOGIN, "node_id": OWNER_NODE_ID}
        repository = {
            "full_name": REPOSITORY,
            "id": REPOSITORY_ID,
            "node_id": REPOSITORY_NODE_ID,
            "owner": dict(owner),
        }
        self.run = {
            "actor": dict(owner),
            "conclusion": None,
            "event": "workflow_dispatch",
            "head_branch": "confirmatory-apparatus-c0",
            "head_repository": dict(repository),
            "head_sha": "a" * 40,
            "id": 983421,
            "path": PROVIDER_PHASE_WORKFLOWS["online"],
            "repository": dict(repository),
            "run_attempt": 1,
            "status": "in_progress",
            "triggering_actor": dict(owner),
        }
        self.jobs = {
            "jobs": [
                {
                    "conclusion": None,
                    "id": 7788,
                    "labels": ["ubuntu-24.04"],
                    "name": plan.claim_job_name,
                    "run_attempt": 1,
                    "run_id": 983421,
                    "status": "in_progress",
                }
            ],
            "total_count": 1,
        }
        self.final_run = self.run
        self.run_reads = 0

    def get(self, endpoint: str) -> object:
        if endpoint.endswith("/jobs?per_page=100"):
            return self.jobs
        self.run_reads += 1
        return self.run if self.run_reads == 1 else self.final_run


def _prerequisite(root: Path, *, phase: str = "online") -> ProviderPrerequisiteReceipt:
    manifest = _digest("manifest")
    predecessor = {
        "online": ("OPENED", 0),
        "label-release": ("ONLINE_COMPLETE", 2),
        "analysis": ("LABELS_RELEASED", 4),
    }[phase]
    return ProviderPrerequisiteReceipt(
        phase=phase,
        suite_attempt_id=suite_attempt_id(manifest),
        manifest_sha256=manifest,
        c1_commit="b" * 40,
        c1_package_root=str(root / "c1-package"),
        c1_package_inventory_sha256=_digest("c1-inventory"),
        c1_package_file_count=C1_REGISTRATION_PACKAGE_FILE_COUNT,
        zenodo_admission_sha256=_digest("zenodo-admission"),
        provider_plan_sha256=_digest(f"{phase}-plan"),
        provider_plan_file_sha256=_digest(f"{phase}-plan-file"),
        provider_plan_materialization_path=str(root / "provider-plan.json"),
        provider_plan_templates_sha256=_digest("plan-templates"),
        runner_bootstrap_receipt_path=str(root / "runner-bootstrap.json"),
        runner_bootstrap_receipt_file_sha256=_digest("runner-bootstrap-file"),
        runner_readiness_receipt_sha256=_digest("runner-ready"),
        predecessor_state=predecessor[0],
        predecessor_sequence=predecessor[1],
        predecessor_state_record_sha256=_digest("predecessor-state"),
        predecessor_ledger_commit="c" * 40,
        predecessor_ledger_tree="d" * 40,
        predecessor_control_inventory_sha256=_digest("controls"),
        predecessor_artifact_receipt_sha256=_digest("predecessor-artifact"),
        predecessor_artifact_inventory_sha256=_digest("predecessor-artifact-inventory"),
        predecessor_artifact_materialized_root=str(root / "predecessor-artifact"),
        workflow_context_sha256=_digest("workflow-context"),
        phase_evidence_root=str(root / "phase-evidence"),
        schema_version=PREREQUISITE_RECEIPT_SCHEMA,
    )


def _claim(root: Path, *, phase: str = "online") -> ProviderClaimReceipt:
    prerequisite = _prerequisite(root, phase=phase)
    target = {
        "online": ("RUN_CLAIMED", 1),
        "label-release": ("LABEL_RELEASE_CLAIMED", 3),
        "analysis": ("ANALYSIS_CLAIMED", 5),
    }[phase]
    return ProviderClaimReceipt(
        phase=phase,
        suite_attempt_id=prerequisite.suite_attempt_id,
        manifest_sha256=prerequisite.manifest_sha256,
        run_id=983421,
        workflow_context_sha256=_digest("workflow-context"),
        prerequisite_receipt_path=str(root / "prerequisite.json"),
        prerequisite_receipt_file_sha256=prerequisite.file_sha256,
        provider_plan_sha256=prerequisite.provider_plan_sha256,
        provider_identity_sha256=_digest("provider-identity"),
        predecessor_state=prerequisite.predecessor_state,
        predecessor_sequence=prerequisite.predecessor_sequence,
        predecessor_state_record_sha256=prerequisite.predecessor_state_record_sha256,
        predecessor_ledger_commit=prerequisite.predecessor_ledger_commit,
        target_state=target[0],
        target_sequence=target[1],
        target_state_record_sha256=_digest("target-state"),
        target_ledger_commit="e" * 40,
        claim_contract_sha256=_digest("claim-contract"),
        publication_receipt_path=str(root / "claim-publication.json"),
        publication_receipt_file_sha256=_digest("publication-file"),
        claim_subject_path=str(root / "claim-subject.json"),
        claim_subject_sha256=_digest("claim-subject"),
        claim_predicate_path=str(root / "claim-predicate.json"),
        claim_predicate_sha256=_digest("claim-predicate"),
        runner_label="fractal-ann-confirmatory-0123456789abcdef",
        suite_namespace=str(root / f"suite-attempt-{prerequisite.suite_attempt_id}"),
        expected_execute_job_name=PROVIDER_PHASE_JOB_NAMES[phase][1],
        expected_claim_artifact_name=(
            f"confirmatory-{phase}-claim-{prerequisite.suite_attempt_id}-983421"
        ),
        schema_version=CLAIM_RECEIPT_SCHEMA,
    )


def _preparation(
    root: Path,
    *,
    phase: str = "online",
    mode: str = "completion",
) -> ProviderTransitionPreparationReceipt:
    claim = _claim(root, phase=phase)
    rows = (
        EvidenceInventoryRow(
            role="claim-receipt",
            relative_path="claim-receipt.json",
            file_sha256=_digest("claim-file"),
            byte_count=140,
        ),
        EvidenceInventoryRow(
            role="phase-execution-receipt",
            relative_path="phase/provider-execution.json",
            file_sha256=_digest("phase-execution"),
            byte_count=280,
        ),
    )
    completion = {
        "online": ("ONLINE_COMPLETE", 2),
        "label-release": ("LABELS_RELEASED", 4),
        "analysis": ("ANALYSIS_COMPLETE", 6),
    }[phase]
    target = ("FAILED", claim.target_sequence + 1) if mode == "failure" else completion
    return ProviderTransitionPreparationReceipt(
        mode=mode,
        phase=phase,
        suite_attempt_id=claim.suite_attempt_id,
        manifest_sha256=claim.manifest_sha256,
        workflow_context_sha256=_digest("workflow-context"),
        claim_receipt_path=str(root / "claim-receipt.json"),
        claim_receipt_file_sha256=_digest("claim-file"),
        claim_state_record_sha256=claim.target_state_record_sha256,
        claim_ledger_commit=claim.target_ledger_commit,
        provider_identity_sha256=claim.provider_identity_sha256,
        execute_job_id=778899,
        live_execute_job_receipt_path=str(root / "live-job.json"),
        live_execute_job_receipt_file_sha256=_digest("live-job-file"),
        evidence_root=str(root / "evidence"),
        evidence_inventory=rows,
        evidence_inventory_sha256=inventory_sha256(rows),
        target_state=target[0],
        target_sequence=target[1],
        target_state_record_sha256=_digest("prepared-state"),
        prepared_subject_path=str(root / "prepared-state.json"),
        prepared_subject_sha256=_digest("prepared-subject"),
        predicate_path=str(root / "prepared-predicate.json"),
        predicate_sha256=_digest("prepared-predicate"),
        phase_closure_sha256=_digest("phase-closure"),
        failed_execute_job_receipt_sha256=(_digest("failed-job") if mode == "failure" else None),
        incident_inventory_sha256=(_digest("incident-inventory") if mode == "failure" else None),
        schema_version=PREPARATION_RECEIPT_SCHEMA,
    )


def _analysis_execution_fixture(
    root: Path,
    *,
    receipt_root: Path | None = None,
    add_extraneous_file: bool = False,
) -> tuple[object, Path, str]:
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration
    from fractal_ann_diagnostics.artifact_integrity import digest_directory_tree
    from fractal_ann_diagnostics.execution_claim import PhaseClaimContract
    from fractal_ann_diagnostics.offline_analysis_contract import (
        OfflineAnalysisExecutionReceipt,
    )
    from fractal_ann_diagnostics.provider_phase_runtime import (
        PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME,
        ProviderDriverOutput,
        ProviderPhaseExecutionReceipt,
    )

    manifest_digest = _digest("analysis-manifest")
    suite = suite_attempt_id(manifest_digest)
    evidence_root = root / "analysis-evidence"
    evidence_root.mkdir(mode=0o700)
    results_store = root / "analysis-results"
    results_store.mkdir(mode=0o700)
    for name in orchestration._analysis_store_entries(manifest_digest):
        (results_store / name).write_bytes(f"{name}\n".encode())
    if add_extraneous_file:
        (results_store / "pre-existing.json").write_bytes(b'{"foreign":true}\n')

    bound_root = results_store if receipt_root is None else receipt_root
    if receipt_root is not None:
        receipt_root.mkdir(mode=0o700)
        for name in orchestration._analysis_store_entries(manifest_digest):
            (receipt_root / name).write_bytes(f"{name}\n".encode())
    included_entries = orchestration._analysis_store_entries(manifest_digest)
    tree = digest_directory_tree(bound_root, included_entries=included_entries)
    plan_sha256 = _digest("analysis-provider-plan")
    plan_file_sha256 = _digest("analysis-provider-plan-file")
    claim_file_sha256 = _digest("analysis-claim-file")
    provider_state_sha256 = _digest("analysis-provider-state")
    provider_ledger_commit = "e" * 40
    phase_contract_sha256 = _digest("analysis-phase-contract")
    provider_identity_sha256 = _digest("analysis-provider-identity")
    run_receipt_sha256 = _digest("analysis-run-receipt")
    package_root = root / "analysis-package"
    package_root.mkdir(mode=0o700)
    admission_name = f"{manifest_digest}.offline-analysis-admission.json"
    package_entries = tuple(
        sorted(
            (
                admission_name,
                "confirmatory-input.json",
                "confirmatory-input-receipt.json",
                "h1-model.json",
                "h2-model.json",
                "offline-input-bundle.json",
            ),
            key=lambda value: value.encode("utf-8"),
        )
    )
    oci_index_digest = f"sha256:{_digest('analysis-oci-index')}"
    execution = OfflineAnalysisExecutionReceipt(
        suite_attempt_id=suite,
        manifest_sha256=manifest_digest,
        run_receipt_sha256=run_receipt_sha256,
        provider_state_record_sha256=provider_state_sha256,
        provider_ledger_commit=provider_ledger_commit,
        phase_claim_contract_sha256=phase_contract_sha256,
        phase_claim_state_sha256=provider_state_sha256,
        phase_claim_ledger_commit=provider_ledger_commit,
        provider_identity_sha256=provider_identity_sha256,
        c1_commit="b" * 40,
        admission_uri=(package_root / admission_name).as_uri(),
        admission_sha256=_digest("analysis-admission"),
        admission_file_sha256=_digest("analysis-admission-file"),
        package_root_uri=package_root.as_uri(),
        package_tree_before_sha256=_digest("analysis-package-tree"),
        package_tree_after_sha256=_digest("analysis-package-tree"),
        package_entries=package_entries,
        docker_executable_sha256=_digest("docker-executable"),
        docker_pull_argv_sha256=_digest("docker-pull"),
        docker_create_argv_sha256=_digest("docker-create"),
        docker_start_argv_sha256=_digest("docker-start"),
        docker_remove_argv_sha256=_digest("docker-remove"),
        container_name=f"fractal-analysis-{suite}",
        runtime_image=f"ghcr.io/example/analysis@{oci_index_digest}",
        runtime_platform="linux/amd64",
        oci_index_digest=oci_index_digest,
        oci_platform_manifest_digest=f"sha256:{_digest('analysis-oci-platform')}",
        attempt_uri=(bound_root / included_entries[0]).as_uri(),
        attempt_receipt_sha256=_digest("analysis-attempt"),
        attempt_file_sha256=_digest("analysis-attempt-file"),
        result_receipt_uri=(bound_root / included_entries[3]).as_uri(),
        result_receipt_sha256=_digest("analysis-result-receipt"),
        result_receipt_file_sha256=_digest("analysis-result-receipt-file"),
        result_uri=(bound_root / included_entries[4]).as_uri(),
        result_artifact_sha256=_digest("analysis-result"),
        result_file_sha256=_digest("analysis-result-file"),
        results_tree_sha256=tree.sha256,
        results_entries=included_entries,
        completion_state_record_sha256=provider_state_sha256,
        completion_ledger_commit=provider_ledger_commit,
        container_absent_after_execution=True,
    )
    execution_path = (
        package_root.parent / f"{manifest_digest}.offline-analysis-execution-receipt.json"
    )
    execution_path.write_bytes(execution.canonical_bytes() + b"\n")
    output = ProviderDriverOutput(
        corpus_id="all-five",
        driver_id="confirmatory-analysis-v1",
        output_root=str(bound_root),
        output_tree_sha256=tree.sha256,
        output_entries=tree.entries,
        analysis_execution_receipt_uri=execution_path.as_uri(),
        analysis_execution_receipt_sha256=execution.receipt_sha256,
        analysis_execution_receipt_file_sha256=_digest(
            (execution.canonical_bytes() + b"\n").decode("ascii")
        ),
    )
    receipt = ProviderPhaseExecutionReceipt(
        phase="analysis",
        suite_attempt_id=suite,
        provider_plan_sha256=plan_sha256,
        provider_plan_file_sha256=plan_file_sha256,
        claim_receipt_file_sha256=claim_file_sha256,
        phase_host_tool_receipt_path=str(evidence_root / "phase-host-tool-receipt.json"),
        phase_host_tool_receipt_sha256=_digest("analysis-host-tool-receipt"),
        phase_host_tool_receipt_file_sha256=_digest("analysis-host-tool-receipt-file"),
        runtime_request_sha256=_digest("analysis-runtime-request"),
        runtime_request_file_sha256=_digest("analysis-runtime-request-file"),
        outputs=(output,),
    )
    (evidence_root / PROVIDER_PHASE_EXECUTION_RECEIPT_FILENAME).write_bytes(
        receipt.canonical_file_bytes()
    )

    class _AnalysisPhaseClaimContract(PhaseClaimContract):
        @property
        def contract_sha256(self) -> str:
            return phase_contract_sha256

    contract = object.__new__(_AnalysisPhaseClaimContract)
    object.__setattr__(contract, "phase", "analysis")
    object.__setattr__(contract, "c1_commit", "b" * 40)
    object.__setattr__(
        contract,
        "corpora",
        tuple(
            SimpleNamespace(corpus_id=corpus_id, output_uri=results_store.as_uri())
            for corpus_id in sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8"))
        ),
    )
    recovered = SimpleNamespace(
        contract=contract,
        manifest={"sealed_execution": {"results_store": results_store.as_uri()}},
        claim_receipt=SimpleNamespace(
            file_sha256=claim_file_sha256,
            provider_plan_sha256=plan_sha256,
        ),
        provider_plan=SimpleNamespace(file_sha256=plan_file_sha256),
        predecessor=SimpleNamespace(
            state=SimpleNamespace(
                suite_attempt_id=suite,
                manifest_sha256=manifest_digest,
                run_receipt_sha256=run_receipt_sha256,
                record_sha256=provider_state_sha256,
            ),
            ledger_commit=provider_ledger_commit,
        ),
        provider_identity=SimpleNamespace(identity_sha256=provider_identity_sha256),
    )
    return recovered, evidence_root, suite


def test_context_is_minted_only_from_exact_c0_environment() -> None:
    context = ProviderWorkflowContext.from_environment("online", _environment())
    assert context.repository == REPOSITORY
    assert context.workflow_path == PROVIDER_PHASE_WORKFLOWS["online"]
    assert context.identity_sha256 == _digest(
        json.dumps(
            context.identity_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    with pytest.raises(ProviderWorkflowOrchestrationError, match="live environment"):
        ProviderWorkflowContext(**context.identity_dict())


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("GITHUB_ACTOR", "attacker", "fixed C0"),
        ("GITHUB_REPOSITORY_ID", "99", "fixed C0"),
        ("GITHUB_REF_PROTECTED", "false", "fixed C0"),
        ("GITHUB_RUN_ATTEMPT", "2", "fixed C0"),
        ("GITHUB_WORKFLOW_SHA", "b" * 40, "source commits differ"),
        ("GITHUB_JOB", "foreign", "not a provider"),
        ("RUNNER_ARCH", "ARM64", "runner class"),
    ),
)
def test_context_rejects_substituted_identity(
    name: str,
    value: str,
    message: str,
) -> None:
    environment = _environment()
    environment[name] = value
    with pytest.raises(ProviderWorkflowOrchestrationError, match=message):
        ProviderWorkflowContext.from_environment("online", environment)


def test_execute_context_requires_the_self_hosted_macos_arm64_runner() -> None:
    context = ProviderWorkflowContext.from_environment(
        "analysis",
        _environment(phase="analysis", job="execute"),
    )
    assert (context.runner_environment, context.runner_os, context.runner_arch) == (
        "self-hosted",
        "macOS",
        "ARM64",
    )


def test_provider_identity_binds_live_run_job_and_c1_runner() -> None:
    context = ProviderWorkflowContext.from_environment("online", _environment())
    plan = _provider_plan()
    api = _ProviderIdentityApi(plan)
    identity = verify_provider_execution_identity(context=context, plan=plan, api=api)
    assert identity.run_id == context.run_id
    assert identity.claim_job_id == 7788
    assert identity.runner_id == plan.runner_id
    assert identity.runner_label == derive_phase_runner_label(plan.claim_nonce, "online")
    assert api.run_reads == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda api: api.run["actor"].update({"id": 99}), "actor id differs"),
        (lambda api: api.jobs.update({"total_count": 2}), "incomplete or malformed"),
        (
            lambda api: api.jobs["jobs"][0].update({"labels": ["self-hosted"]}),
            "fixed hosted image",
        ),
    ),
)
def test_provider_identity_rejects_substituted_run_or_job(
    mutation: object,
    message: str,
) -> None:
    context = ProviderWorkflowContext.from_environment("online", _environment())
    plan = _provider_plan()
    api = _ProviderIdentityApi(plan)
    assert callable(mutation)
    mutation(api)
    with pytest.raises(ProviderWorkflowOrchestrationError, match=message):
        verify_provider_execution_identity(context=context, plan=plan, api=api)


def test_provider_identity_rejects_run_race_on_final_read() -> None:
    context = ProviderWorkflowContext.from_environment("online", _environment())
    plan = _provider_plan()
    api = _ProviderIdentityApi(plan)
    api.final_run = {**api.run, "status": "completed", "conclusion": "success"}
    with pytest.raises(ProviderWorkflowOrchestrationError, match="changed during"):
        verify_provider_execution_identity(context=context, plan=plan, api=api)


def test_prerequisite_round_trip_is_canonical_and_one_shot(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    receipt = _prerequisite(root)
    output = root / "receipts" / "prerequisite.json"
    write_provider_receipt(receipt, output)
    assert load_provider_prerequisite_receipt(output) == receipt
    assert output.read_bytes() == receipt.canonical_file_bytes()
    with pytest.raises(ProviderWorkflowOrchestrationError, match="create"):
        write_provider_receipt(receipt, output)


def test_prerequisite_rejects_foreign_suite_and_phase_predecessor(tmp_path: Path) -> None:
    receipt = _prerequisite(tmp_path.resolve())
    with pytest.raises(ProviderWorkflowOrchestrationError, match="manifest-derived"):
        replace(receipt, suite_attempt_id=_digest("foreign-suite"))
    with pytest.raises(ProviderWorkflowOrchestrationError, match="state machine"):
        replace(receipt, predecessor_state="ONLINE_COMPLETE")
    with pytest.raises(
        ProviderWorkflowOrchestrationError,
        match=rf"exactly {C1_REGISTRATION_PACKAGE_FILE_COUNT}",
    ):
        replace(
            receipt,
            c1_package_file_count=C1_REGISTRATION_PACKAGE_FILE_COUNT - 1,
        )


def test_claim_closes_state_pair_namespace_job_and_artifact_name(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    receipt = _claim(root, phase="label-release")
    path = root / "claim.json"
    write_provider_receipt(receipt, path)
    assert load_provider_claim_receipt(path) == receipt
    with pytest.raises(ProviderWorkflowOrchestrationError, match="state machine"):
        replace(receipt, target_sequence=4)
    with pytest.raises(ProviderWorkflowOrchestrationError, match="execute-job"):
        replace(receipt, expected_execute_job_name="execute-online")
    with pytest.raises(ProviderWorkflowOrchestrationError, match="artifact name"):
        replace(receipt, expected_claim_artifact_name="caller-selected")
    with pytest.raises(ProviderWorkflowOrchestrationError, match="namespace name"):
        replace(receipt, suite_namespace=str(root / "another-suite"))


def test_preparation_round_trip_and_mode_specific_state_machine(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    completion = _preparation(root)
    completion_path = root / "completion.json"
    write_provider_receipt(completion, completion_path)
    assert load_provider_transition_preparation_receipt(completion_path) == completion

    failure = _preparation(root, mode="failure")
    failure_path = root / "failure.json"
    write_provider_receipt(failure, failure_path)
    assert load_provider_transition_preparation_receipt(failure_path) == failure
    with pytest.raises(ProviderWorkflowOrchestrationError, match="failure-only"):
        replace(
            completion,
            failed_execute_job_receipt_sha256=_digest("unexpected"),
        )
    with pytest.raises(ProviderWorkflowOrchestrationError, match="requires"):
        replace(failure, incident_inventory_sha256=None)


def test_inventory_rejects_aliases_digest_substitution_and_escape(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    preparation = _preparation(root)
    aliased = (
        EvidenceInventoryRow("first", "A.json", _digest("a"), 1),
        EvidenceInventoryRow("second", "a.json", _digest("b"), 1),
    )
    with pytest.raises(ProviderWorkflowOrchestrationError, match="unique"):
        inventory_sha256(aliased)
    with pytest.raises(ProviderWorkflowOrchestrationError, match="inventory digest"):
        replace(preparation, evidence_inventory_sha256=_digest("substituted"))
    with pytest.raises(ProviderWorkflowOrchestrationError, match="safe relative"):
        EvidenceInventoryRow("escape", "../escape.json", _digest("escape"), 1)


def test_loader_rejects_noncanonical_duplicate_and_linked_receipts(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    receipt = _prerequisite(root)
    noncanonical = root / "noncanonical.json"
    noncanonical.write_text(json.dumps(receipt.to_dict(), indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ProviderWorkflowOrchestrationError, match="not canonical"):
        load_provider_prerequisite_receipt(noncanonical)

    duplicate = root / "duplicate.json"
    duplicate.write_text('{"phase":"online","phase":"analysis"}\n', encoding="utf-8")
    with pytest.raises(ProviderWorkflowOrchestrationError, match="repeats JSON key"):
        load_provider_prerequisite_receipt(duplicate)

    original = root / "original.json"
    original.write_bytes(receipt.canonical_file_bytes())
    symlink = root / "symlink.json"
    symlink.symlink_to(original)
    with pytest.raises(ProviderWorkflowOrchestrationError, match="open"):
        load_provider_prerequisite_receipt(symlink)

    hardlink = root / "hardlink.json"
    os.link(original, hardlink)
    with pytest.raises(ProviderWorkflowOrchestrationError, match="singly linked"):
        load_provider_prerequisite_receipt(original)


def test_schema_values_are_fixed(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    context = ProviderWorkflowContext.from_environment("online", _environment())
    assert context.schema_version == WORKFLOW_CONTEXT_SCHEMA
    assert PREREQUISITE_RECEIPT_SCHEMA == "fractal-provider-prerequisite-receipt-v2"
    assert _prerequisite(root).schema_version == PREREQUISITE_RECEIPT_SCHEMA
    assert _claim(root).schema_version == CLAIM_RECEIPT_SCHEMA
    assert _preparation(root).schema_version == PREPARATION_RECEIPT_SCHEMA


def test_analysis_execution_receipt_rehashes_exact_claim_bound_five_file_store(
    tmp_path: Path,
) -> None:
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    recovered, evidence_root, suite = _analysis_execution_fixture(tmp_path.resolve())
    receipt = orchestration._load_phase_execution_receipt(
        phase="analysis",
        suite_attempt_id=suite,
        evidence_root=evidence_root,
        recovered=recovered,
    )
    assert tuple(receipt.outputs[0].output_entries) == orchestration._analysis_store_entries(
        recovered.predecessor.state.manifest_sha256
    )


def test_analysis_execution_receipt_rejects_foreign_output_root(
    tmp_path: Path,
) -> None:
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    root = tmp_path.resolve()
    recovered, evidence_root, suite = _analysis_execution_fixture(
        root,
        receipt_root=root / "foreign-results",
    )
    with pytest.raises(ProviderWorkflowOrchestrationError, match="claim-bound authority"):
        orchestration._load_phase_execution_receipt(
            phase="analysis",
            suite_attempt_id=suite,
            evidence_root=evidence_root,
            recovered=recovered,
        )


def test_analysis_execution_receipt_rejects_preexisting_or_extraneous_file(
    tmp_path: Path,
) -> None:
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    recovered, evidence_root, suite = _analysis_execution_fixture(
        tmp_path.resolve(),
        add_extraneous_file=True,
    )
    with pytest.raises(
        ProviderWorkflowOrchestrationError,
        match="pre-existing or extraneous",
    ):
        orchestration._load_phase_execution_receipt(
            phase="analysis",
            suite_attempt_id=suite,
            evidence_root=evidence_root,
            recovered=recovered,
        )


def test_claim_command_reverifies_authority_and_emits_closed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fractal_ann_diagnostics.github_artifact_transport as artifact_transport
    import fractal_ann_diagnostics.github_state_attestation as state_attestation
    import fractal_ann_diagnostics.provider_claim_publication as claim_publication
    import fractal_ann_diagnostics.provider_prerequisite_factory as prerequisite_factory
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    root = tmp_path.resolve()
    context = ProviderWorkflowContext.from_environment("online", _environment())
    prerequisite = replace(
        _prerequisite(root),
        workflow_context_sha256=context.identity_sha256,
    )
    prerequisite_path = root / "provider-prerequisite-receipt.json"
    write_provider_receipt(prerequisite, prerequisite_path)

    state = SimpleNamespace(
        state=prerequisite.predecessor_state,
        sequence=prerequisite.predecessor_sequence,
        record_sha256=prerequisite.predecessor_state_record_sha256,
    )
    predecessor = SimpleNamespace(
        state=state,
        ledger_commit=prerequisite.predecessor_ledger_commit,
    )
    plan = SimpleNamespace(
        plan_sha256=prerequisite.provider_plan_sha256,
        execute_job_name=PROVIDER_PHASE_JOB_NAMES["online"][1],
    )
    registration = SimpleNamespace(manifest_sha256=prerequisite.manifest_sha256)
    admitted = SimpleNamespace(
        registration=registration,
        plan=plan,
        predecessor=SimpleNamespace(predecessor=predecessor),
        zenodo_admission=object(),
        manifest_rekor_integrated_at_utc="2026-07-01T00:00:00+00:00",
        registry_record_rekor_integrated_at_utc="2026-07-01T00:00:01+00:00",
        prerequisite_fields=lambda: prerequisite.to_dict(),
        assert_current=lambda: None,
    )
    subject_path = root / "claim-output" / "claim-subject.json"
    predicate_path = root / "claim-output" / "claim-predicate.json"
    publication_path = root / "claim-output" / "claim-publication.json"
    result = SimpleNamespace(
        provider_identity=SimpleNamespace(identity_sha256=_digest("provider-identity")),
        state=SimpleNamespace(
            state="RUN_CLAIMED",
            sequence=1,
            record_sha256=_digest("run-claimed"),
        ),
        publication_receipt=SimpleNamespace(
            commit_oid="e" * 40,
            receipt_sha256=_digest("publication-receipt"),
        ),
        publication_receipt_path=publication_path,
        contract=SimpleNamespace(contract_sha256=_digest("claim-contract")),
        subject_path=subject_path,
        subject_sha256=_digest("claim-subject"),
        predicate_path=predicate_path,
        predicate_sha256=_digest("claim-predicate"),
        runner_label="fractal-ann-confirmatory-0123456789abcdef",
        suite_namespace=root / f"suite-attempt-{prerequisite.suite_attempt_id}",
    )
    observed: dict[str, object] = {}

    def fake_build(*args: object, **kwargs: object) -> object:
        observed["build_args"] = args
        observed["build_kwargs"] = kwargs
        return admitted

    def fake_publish(**kwargs: object) -> object:
        observed["publish_kwargs"] = kwargs
        return result

    monkeypatch.setattr(
        orchestration.ProviderWorkflowContext,
        "from_environment",
        lambda phase: context,
    )
    monkeypatch.setattr(prerequisite_factory, "build_hosted_production_prerequisites", fake_build)
    monkeypatch.setattr(claim_publication, "derive_and_publish_provider_claim", fake_publish)
    monkeypatch.setattr(state_attestation, "GhApiClient", lambda: object())
    monkeypatch.setattr(
        artifact_transport,
        "UrllibGitHubArtifactReadApi",
        lambda token: SimpleNamespace(token=token),
    )
    monkeypatch.setenv("GH_TOKEN", "ephemeral-test-token")

    output_dir = root / "claim-output"
    outputs = execute_claim_command(
        phase="online",
        suite_attempt_id=prerequisite.suite_attempt_id,
        prerequisite_receipt_path=prerequisite_path,
        output_dir=output_dir,
    )

    assert set(outputs) == {
        "claim_ledger_commit",
        "claim_predicate_path",
        "claim_predicate_sha256",
        "claim_receipt_path",
        "claim_receipt_sha256",
        "claim_state_sha256",
        "claim_subject_path",
        "claim_subject_sha256",
        "expected_execute_job_name",
        "provider_identity_sha256",
        "runner_label",
        "suite_namespace",
    }
    receipt = load_provider_claim_receipt(outputs["claim_receipt_path"])
    assert receipt.target_state == "RUN_CLAIMED"
    assert receipt.target_ledger_commit == "e" * 40
    assert receipt.prerequisite_receipt_file_sha256 == prerequisite.file_sha256
    assert Path(observed["build_args"][3]) == output_dir / "fresh-prerequisites"
    publish_kwargs = observed["publish_kwargs"]
    assert publish_kwargs["predecessor"] is predecessor
    assert publish_kwargs["c1_registry_rekor_integrated_at_utc"] == ("2026-07-01T00:00:01+00:00")
    assert "ephemeral-test-token" not in receipt.canonical_file_bytes().decode("ascii")


def test_claim_command_rejects_persisted_prerequisite_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fractal_ann_diagnostics.github_artifact_transport as artifact_transport
    import fractal_ann_diagnostics.github_state_attestation as state_attestation
    import fractal_ann_diagnostics.provider_prerequisite_factory as prerequisite_factory
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    root = tmp_path.resolve()
    context = ProviderWorkflowContext.from_environment("online", _environment())
    prerequisite = replace(
        _prerequisite(root),
        workflow_context_sha256=context.identity_sha256,
    )
    prerequisite_path = root / "provider-prerequisite-receipt.json"
    write_provider_receipt(prerequisite, prerequisite_path)
    drifted = replace(prerequisite, provider_plan_sha256=_digest("changed-provider-plan"))
    admitted = SimpleNamespace(prerequisite_fields=lambda: drifted.to_dict())

    monkeypatch.setattr(
        orchestration.ProviderWorkflowContext,
        "from_environment",
        lambda phase: context,
    )
    monkeypatch.setattr(
        prerequisite_factory,
        "build_hosted_production_prerequisites",
        lambda *args, **kwargs: admitted,
    )
    monkeypatch.setattr(state_attestation, "GhApiClient", lambda: object())
    monkeypatch.setattr(
        artifact_transport,
        "UrllibGitHubArtifactReadApi",
        lambda token: object(),
    )
    monkeypatch.setenv("GH_TOKEN", "ephemeral-test-token")

    with pytest.raises(ProviderWorkflowOrchestrationError, match="differs from fresh authority"):
        execute_claim_command(
            phase="online",
            suite_attempt_id=prerequisite.suite_attempt_id,
            prerequisite_receipt_path=prerequisite_path,
            output_dir=root / "claim-output",
        )


def test_command_boundary_requires_claim_for_activation_and_completion(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    suite = suite_attempt_id(_digest("manifest"))
    with pytest.raises(
        ProviderWorkflowOrchestrationError,
        match="activation and claim-receipt presence must agree",
    ):
        execute_verify_prerequisites_command(
            phase="online",
            suite_attempt_id=suite,
            output_dir=root / "activation",
            activate_and_execute=True,
        )
    with pytest.raises(
        ProviderWorkflowOrchestrationError,
        match="activation and claim-receipt presence must agree",
    ):
        execute_verify_prerequisites_command(
            phase="online",
            suite_attempt_id=suite,
            output_dir=root / "activation",
            claim_receipt_path=root / "claim.json",
        )
    with pytest.raises(
        ProviderWorkflowOrchestrationError,
        match="completion requires its exact claim receipt",
    ):
        execute_complete_command(
            phase="online",
            suite_attempt_id=suite,
            prepare=True,
            publish=False,
            claim_receipt_path=None,
            evidence_root=root / "evidence",
            attestation_bundle_path=None,
            preparation_receipt_path=None,
            output_dir=root / "completion",
        )


def test_activation_command_uses_only_fixed_claim_artifact_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fractal_ann_diagnostics.github_artifact_transport as artifact_transport
    import fractal_ann_diagnostics.github_state_attestation as state_attestation
    import fractal_ann_diagnostics.provider_activation_factory as activation_factory
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    root = tmp_path.resolve()
    context = ProviderWorkflowContext.from_environment(
        "online", _environment(phase="online", job="execute")
    )
    claim = root / "claim-receipt.json"
    claim.write_text("{}\n", encoding="utf-8")
    outputs = {
        key: f"value-{key}"
        for key in ACTIVATION_COMMON_OUTPUT_KEYS | ACTIVATION_PHASE_OUTPUT_KEYS["online"]
    }
    observed: dict[str, object] = {}

    def activate(**kwargs: object) -> object:
        observed.update(kwargs)
        return SimpleNamespace(output_fields=lambda: dict(outputs))

    api = object()
    artifact_api = object()
    monkeypatch.setattr(
        orchestration.ProviderWorkflowContext,
        "from_environment",
        lambda phase: context,
    )
    monkeypatch.setattr(state_attestation, "GhApiClient", lambda: api)
    monkeypatch.setattr(
        artifact_transport,
        "UrllibGitHubArtifactReadApi",
        lambda token: artifact_api if token == "ephemeral-test-token" else None,
    )
    monkeypatch.setattr(
        activation_factory,
        "activate_and_execute_provider_phase",
        activate,
    )
    monkeypatch.setenv("GH_TOKEN", "ephemeral-test-token")
    monkeypatch.setenv("CLAIM_ARTIFACT_ID", "731")
    monkeypatch.setenv("CLAIM_ARTIFACT_DIGEST", f"sha256:{_digest('claim-archive')}")
    monkeypatch.setenv("CLAIM_PACKAGE_INVENTORY_SHA256", _digest("claim-inventory"))

    result = execute_verify_prerequisites_command(
        phase="online",
        suite_attempt_id=suite_attempt_id(_digest("manifest")),
        output_dir=root / "activation",
        claim_receipt_path=claim,
        activate_and_execute=True,
    )

    assert result == outputs
    assert observed == {
        "context": context,
        "phase": "online",
        "suite_attempt_id": suite_attempt_id(_digest("manifest")),
        "artifact_id": 731,
        "artifact_digest": f"sha256:{_digest('claim-archive')}",
        "expected_inventory_sha256": _digest("claim-inventory"),
        "claim_receipt_destination": claim,
        "output_dir": root / "activation",
        "github_api": api,
        "artifact_api": artifact_api,
        "completion_anchor_token_fd": None,
    }


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("GH_TOKEN", "", "ephemeral GitHub token"),
        ("CLAIM_ARTIFACT_ID", "0", "artifact ID"),
        ("CLAIM_ARTIFACT_DIGEST", "sha256:BAD", "artifact digest"),
        ("CLAIM_PACKAGE_INVENTORY_SHA256", "BAD", "inventory digest"),
    ],
)
def test_activation_command_rejects_missing_or_malformed_fixed_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    error: str,
) -> None:
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    root = tmp_path.resolve()
    context = ProviderWorkflowContext.from_environment(
        "online", _environment(phase="online", job="execute")
    )
    claim = root / "claim-receipt.json"
    claim.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        orchestration.ProviderWorkflowContext,
        "from_environment",
        lambda phase: context,
    )
    monkeypatch.setenv("GH_TOKEN", "ephemeral-test-token")
    monkeypatch.setenv("CLAIM_ARTIFACT_ID", "731")
    monkeypatch.setenv("CLAIM_ARTIFACT_DIGEST", f"sha256:{_digest('claim-archive')}")
    monkeypatch.setenv("CLAIM_PACKAGE_INVENTORY_SHA256", _digest("claim-inventory"))
    monkeypatch.setenv(name, value)

    with pytest.raises(ProviderWorkflowOrchestrationError, match=error):
        execute_verify_prerequisites_command(
            phase="online",
            suite_attempt_id=suite_attempt_id(_digest("manifest")),
            output_dir=root / "activation",
            claim_receipt_path=claim,
            activate_and_execute=True,
        )


def test_label_activation_requires_the_inherited_completion_anchor_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    root = tmp_path.resolve()
    context = ProviderWorkflowContext.from_environment(
        "label-release",
        _environment(phase="label-release", job="execute"),
    )
    claim = root / "claim-receipt.json"
    claim.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        orchestration.ProviderWorkflowContext,
        "from_environment",
        lambda phase: context,
    )
    monkeypatch.setenv("GH_TOKEN", "ephemeral-test-token")
    monkeypatch.setenv("CLAIM_ARTIFACT_ID", "731")
    monkeypatch.setenv(
        "CLAIM_ARTIFACT_DIGEST",
        f"sha256:{_digest('claim-archive')}",
    )
    monkeypatch.setenv(
        "CLAIM_PACKAGE_INVENTORY_SHA256",
        _digest("claim-inventory"),
    )
    monkeypatch.delenv("COMPLETION_ANCHOR_TOKEN_FD", raising=False)

    with pytest.raises(
        ProviderWorkflowOrchestrationError,
        match="Zenodo token file descriptor",
    ):
        execute_verify_prerequisites_command(
            phase="label-release",
            suite_attempt_id=suite_attempt_id(_digest("manifest")),
            output_dir=root / "activation",
            claim_receipt_path=claim,
            activate_and_execute=True,
        )


def test_command_boundary_separates_completion_from_failure_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    root = tmp_path.resolve()
    suite = suite_attempt_id(_digest("manifest"))
    observed: list[dict[str, object]] = []

    def fake_transition(**kwargs: object) -> dict[str, str]:
        observed.append(dict(kwargs))
        return {"mode": str(kwargs["mode"])}

    monkeypatch.setattr(orchestration, "_production_transition_command", fake_transition)
    completion = execute_complete_command(
        phase="online",
        suite_attempt_id=suite,
        prepare=True,
        publish=False,
        claim_receipt_path=root / "claim.json",
        evidence_root=root / "evidence",
        attestation_bundle_path=None,
        preparation_receipt_path=None,
        output_dir=root / "completion",
    )
    failure = execute_fail_command(
        phase="online",
        suite_attempt_id=suite,
        prepare=True,
        publish=False,
        claim_receipt_path=None,
        evidence_root=root / "evidence",
        attestation_bundle_path=None,
        preparation_receipt_path=None,
        output_dir=root / "failure",
    )
    assert completion == {"mode": "completion"}
    assert failure == {"mode": "failure"}
    assert observed[0]["claim_receipt_path"] == root / "claim.json"
    assert observed[1]["claim_receipt_path"] is None
    with pytest.raises(
        ProviderWorkflowOrchestrationError,
        match="failure must recover the live claim",
    ):
        execute_fail_command(
            phase="online",
            suite_attempt_id=suite,
            prepare=True,
            publish=False,
            claim_receipt_path=root / "caller-claim.json",
            evidence_root=root / "evidence",
            attestation_bundle_path=None,
            preparation_receipt_path=None,
            output_dir=root / "failure",
        )


def test_copied_label_completion_evidence_remains_bound_to_closures(
    tmp_path: Path,
) -> None:
    import hashlib

    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration
    from fractal_ann_diagnostics.suite_attempt import LabelCorpusClosure

    root = tmp_path.resolve()
    portable_root = root / "portable-evidence"
    closures: list[LabelCorpusClosure] = []
    copied_plaintexts: list[Path] = []
    for corpus_id in sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")):
        source_root = root / "source" / corpus_id
        copied_root = portable_root / "phase" / corpus_id
        copied_root.mkdir(parents=True)
        receipt_name = "timelock-decryption-receipt.json"
        plaintext_name = f"{corpus_id}-released-labels.json"
        receipt_bytes = f'{{"corpus_id":"{corpus_id}","released":true}}\n'.encode()
        plaintext_bytes = f'{{"corpus_id":"{corpus_id}","labels":[1,0,1]}}\n'.encode()
        (copied_root / receipt_name).write_bytes(receipt_bytes)
        copied_plaintext = copied_root / plaintext_name
        copied_plaintext.write_bytes(plaintext_bytes)
        copied_plaintexts.append(copied_plaintext)
        closures.append(
            LabelCorpusClosure(
                corpus_id=corpus_id,
                decryption_receipt_uri=(source_root / receipt_name).as_uri(),
                decryption_receipt_sha256=_digest(f"{corpus_id}-receipt"),
                decryption_receipt_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                decryption_receipt_byte_count=len(receipt_bytes),
                plaintext_uri=(source_root / plaintext_name).as_uri(),
                plaintext_sha256=hashlib.sha256(plaintext_bytes).hexdigest(),
                plaintext_byte_count=len(plaintext_bytes),
            )
        )

    candidate = SimpleNamespace(state="LABELS_RELEASED", payload=tuple(closures))
    orchestration._assert_copied_label_completion_evidence(candidate, portable_root)

    mutated = copied_plaintexts[0].read_bytes()
    copied_plaintexts[0].write_bytes(bytes([mutated[0] ^ 1]) + mutated[1:])
    with pytest.raises(
        ProviderWorkflowOrchestrationError,
        match="label evidence differs from its closure",
    ):
        orchestration._assert_copied_label_completion_evidence(candidate, portable_root)


@pytest.mark.parametrize(
    ("prepare", "publish", "bundle", "preparation", "message"),
    (
        (False, False, None, None, "exactly one transition mode"),
        (True, True, None, None, "exactly one transition mode"),
        (False, True, None, None, "publish requires both"),
        (False, True, "bundle.json", None, "publish requires both"),
        (True, False, "bundle.json", "preparation.json", "publish requires both"),
    ),
)
def test_command_boundary_rejects_ambiguous_or_substituted_transition_inputs(
    tmp_path: Path,
    prepare: bool,
    publish: bool,
    bundle: str | None,
    preparation: str | None,
    message: str,
) -> None:
    root = tmp_path.resolve()
    with pytest.raises(ProviderWorkflowOrchestrationError, match=message):
        execute_complete_command(
            phase="analysis",
            suite_attempt_id=suite_attempt_id(_digest("manifest")),
            prepare=prepare,
            publish=publish,
            claim_receipt_path=root / "claim.json",
            evidence_root=root / "evidence",
            attestation_bundle_path=(None if bundle is None else root / bundle),
            preparation_receipt_path=(None if preparation is None else root / preparation),
            output_dir=root / "completion",
        )
