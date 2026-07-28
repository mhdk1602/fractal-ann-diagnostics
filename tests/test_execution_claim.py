from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import fractal_ann_diagnostics.execution_claim as claim_module
from fractal_ann_diagnostics.execution_claim import (
    C1_REGISTRATION_PACKAGE_FILE_COUNT,
    DOCKER_SERVER_PROBE_FILENAME,
    OFFICIAL_ACTIONS_RUNNER_CONFIG_SHA256,
    OFFICIAL_ACTIONS_RUNNER_LISTENER_DLL_SHA256,
    OFFICIAL_ACTIONS_RUNNER_LISTENER_SHA256,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI,
    OFFICIAL_ACTIONS_RUNNER_RUN_SHA256,
    OFFICIAL_ACTIONS_RUNNER_VERSION,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI,
    OFFICIAL_GH_OSX_ARM64_BINARY_SHA256,
    OFFICIAL_GH_VERSION,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI,
    OFFICIAL_PYTHON_BUILD_STANDALONE_BINARY_SHA256,
    OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION,
    PHASE_HOST_PROBE_FILENAME,
    REGISTERED_DOCKER_CLIENT_BUILD,
    REGISTERED_DOCKER_CLIENT_SHA256,
    REGISTERED_DOCKER_CLIENT_VERSION,
    SOURCE_BUILT_LINUX_ARM64_TLE_SHA256,
    AnonymousZenodoAdmission,
    ClaimCorpusBinding,
    CorpusOutputTree,
    DockerServerProbe,
    ExecutionBeaconContract,
    ExecutionClaimContract,
    ExecutionClaimError,
    ExecutionClaimInputs,
    FailedExecuteJobReceipt,
    LiveExecuteJobReceipt,
    PhaseClaimContract,
    PhaseCorpusBinding,
    PhaseHostProbe,
    PhaseHostToolContract,
    ProviderExecutionIdentity,
    RunOutputAggregate,
    VerifiedBeaconClaims,
    _mint_verified_phase_claim,
    _mint_verified_run_claim,
    capture_docker_server_probe,
    derive_phase_runner_label,
    generate_phase_host_probes,
    verify_execution_beacon,
    verify_failed_execute_job,
    verify_label_release_beacon,
    verify_live_execute_job,
    verify_phase_host_tools,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA


def _digest(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _host_tools(root: Path = Path("/private/tmp/fractal-controlled")) -> PhaseHostToolContract:
    host_probe = PhaseHostProbe(
        operating_system="macOS",
        operating_system_version="15.5",
        kernel_release="24.5.0",
        architecture="ARM64",
        logical_cpu_count=12,
        physical_memory_bytes=64 * 1024**3,
    )
    docker_probe = DockerServerProbe(
        engine_version="28.3.2",
        engine_build="synthetic-server-build",
        kernel_version="6.10.14-linuxkit",
        operating_system="linux",
        architecture="arm64",
        cpu_count=12,
        memory_bytes=48 * 1024**3,
    )
    return PhaseHostToolContract(
        controlled_root=str(root),
        python_archive_uri=OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI,
        python_archive_sha256=OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256,
        python_archive_byte_count=OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT,
        python_executable=str(root / "python/bin/python3.12"),
        python_version=OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION,
        python_executable_sha256=OFFICIAL_PYTHON_BUILD_STANDALONE_BINARY_SHA256,
        venv_root=str(root / "venv"),
        venv_tree_sha256=_digest("venv-tree"),
        venv_symlink_inventory_sha256=_digest("venv-symlinks"),
        gh_archive_uri=OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI,
        gh_archive_sha256=OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256,
        gh_archive_byte_count=OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT,
        gh_executable=str(root / "gh/bin/gh"),
        gh_executable_sha256=OFFICIAL_GH_OSX_ARM64_BINARY_SHA256,
        gh_version=OFFICIAL_GH_VERSION,
        runner_archive_uri=OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI,
        runner_archive_sha256=OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
        runner_archive_byte_count=OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT,
        runner_listener_executable=str(root / "runner/bin/Runner.Listener"),
        runner_listener_sha256=OFFICIAL_ACTIONS_RUNNER_LISTENER_SHA256,
        runner_listener_dll=str(root / "runner/bin/Runner.Listener.dll"),
        runner_listener_dll_sha256=OFFICIAL_ACTIONS_RUNNER_LISTENER_DLL_SHA256,
        runner_config_executable=str(root / "runner/config.sh"),
        runner_config_sha256=OFFICIAL_ACTIONS_RUNNER_CONFIG_SHA256,
        runner_run_executable=str(root / "runner/run.sh"),
        runner_run_sha256=OFFICIAL_ACTIONS_RUNNER_RUN_SHA256,
        runner_version=OFFICIAL_ACTIONS_RUNNER_VERSION,
        runner_ephemeral=True,
        runner_disable_update=True,
        runner_unattended=True,
        docker_executable="/usr/local/bin/docker",
        docker_resolved_executable="/Applications/Docker.app/Contents/Resources/bin/docker",
        docker_executable_sha256=REGISTERED_DOCKER_CLIENT_SHA256,
        docker_client_version=REGISTERED_DOCKER_CLIENT_VERSION,
        docker_client_build=REGISTERED_DOCKER_CLIENT_BUILD,
        host_probe=host_probe,
        docker_server_probe=docker_probe,
        host_probe_receipt_sha256=host_probe.file_sha256,
        docker_server_probe_receipt_sha256=docker_probe.file_sha256,
        host_operating_system="macOS",
        host_architecture="ARM64",
    )


def _beacon() -> ExecutionBeaconContract:
    return ExecutionBeaconContract(
        drand_network="https://api.drand.sh",
        chain_hash=_digest("drand-chain"),
        chain_scheme_id="bls-unchained-g1-rfc9380",
        chain_public_key="ab" * 48,
        chain_genesis_unix_seconds=1_700_000_000,
        chain_period_seconds=3,
        execution_round=1,
        label_release_round=101,
        minimum_label_release_safety_rounds=100,
        verification_identity=_digest("tlock-verifier"),
    )


def _claim(root: Path = Path("/private/tmp/fractal-controlled")) -> ExecutionClaimContract:
    repository = "mhdk1602/fractal-ann-diagnostics"
    workflow = ".github/workflows/confirmatory-online-execution.yml"
    manifest = _digest("manifest")
    nonce = _digest("claim-nonce")
    corpora = tuple(
        ClaimCorpusBinding(
            corpus_id=corpus_id,
            staging_namespace_uri=f"file:///private/tmp/staging/{corpus_id}",
            canonical_namespace_uri=f"file:///private/tmp/canonical/{corpus_id}",
            runtime_plan_sha256=_digest(f"plan:{corpus_id}"),
            runtime_plan_file_sha256=_digest(f"plan-file:{corpus_id}"),
        )
        for corpus_id in sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8"))
    )
    identity = _digest(
        _canonical(
            {
                "corpora": [
                    {
                        "corpus_id": row.corpus_id,
                        "canonical_namespace_uri": row.canonical_namespace_uri,
                        "staging_namespace_uri": row.staging_namespace_uri,
                    }
                    for row in corpora
                ],
                "derivation": "sha256-five-canonical-output-trees-v1",
                "manifest_sha256": manifest,
            }
        )
    )
    return ExecutionClaimContract(
        repository=repository,
        claim_workflow_path=workflow,
        claim_workflow_ref=(f"{repository}/{workflow}@refs/tags/confirmatory-apparatus-c0"),
        claim_workflow_sha="1" * 40,
        run_head_branch="confirmatory-apparatus-c0",
        claim_job_name="claim-online",
        execute_job_name="execute-online",
        unique_runner_label=derive_phase_runner_label(nonce, "online"),
        claim_nonce=nonce,
        runner_id=101,
        runner_name="fractal-confirmatory-runner",
        runner_group_id=7,
        runner_version=OFFICIAL_ACTIONS_RUNNER_VERSION,
        runner_archive_sha256=OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
        provider_operating_system="macOS",
        provider_architecture="ARM64",
        host_tools=_host_tools(root),
        runtime_probe_receipt_sha256=_digest("runtime-probe"),
        design_seed_sha256=_digest("registered-design-seed"),
        registered_online_runtime_budget_seconds=68_000,
        maximum_online_runtime_seconds=72_000,
        c1_commit="2" * 40,
        manifest_sha256=manifest,
        label_release_provider_plan_uri="file:///private/tmp/c1/label-release-plan.json",
        label_release_provider_plan_sha256=_digest("label-release-provider-plan"),
        analysis_provider_plan_uri="file:///private/tmp/c1/analysis-plan.json",
        analysis_provider_plan_sha256=_digest("analysis-provider-plan"),
        run_receipt_sha256=_digest("run-receipt"),
        run_receipt_file_sha256=_digest("run-receipt-file"),
        oci_index_digest=f"sha256:{_digest('oci-index')}",
        oci_platform_manifest_digest=f"sha256:{_digest('oci-platform')}",
        analysis_oci_platform_manifest_digest=f"sha256:{_digest('analysis-oci-platform')}",
        release_oci_index_digest=f"sha256:{_digest('release-oci-index')}",
        release_oci_platform_manifest_digest=f"sha256:{_digest('release-oci-platform')}",
        release_tle_binary_sha256=SOURCE_BUILT_LINUX_ARM64_TLE_SHA256,
        release_tle_build_provenance_sha256=_digest("tle-build-provenance"),
        release_tle_vulnerability_scan_sha256=_digest("tle-zero-scan"),
        release_tle_interoperability_receipt_sha256=_digest("tle-interoperability"),
        hardware_contract_sha256=_digest("hardware"),
        corpora=corpora,
        output_aggregate_identity=identity,
        beacon=_beacon(),
    )


def _provider(contract: ExecutionClaimContract) -> ProviderExecutionIdentity:
    return ProviderExecutionIdentity(
        repository=contract.repository,
        workflow_path=contract.claim_workflow_path,
        workflow_ref=contract.claim_workflow_ref,
        workflow_sha=contract.claim_workflow_sha,
        run_head_branch=contract.run_head_branch,
        run_id=31337,
        run_attempt=1,
        claim_job_id=4001,
        claim_job_name="claim-online",
        execute_job_name=contract.execute_job_name,
        runner_id=contract.runner_id,
        runner_name=contract.runner_name,
        runner_group_id=contract.runner_group_id,
        runner_label=contract.unique_runner_label,
        runner_version=contract.runner_version,
        runner_archive_sha256=contract.runner_archive_sha256,
        provider_operating_system=contract.provider_operating_system,
        provider_architecture=contract.provider_architecture,
        host_tool_contract_sha256=contract.host_tools.contract_sha256,
        runtime_probe_receipt_sha256=contract.runtime_probe_receipt_sha256,
        self_hosted=True,
    )


def _phase_contract(phase: str) -> PhaseClaimContract:
    root = _claim()
    rows = tuple(
        PhaseCorpusBinding(
            corpus_id=corpus_id,
            input_uri=f"file:///private/tmp/{phase}/input/{corpus_id}.json",
            input_sha256=_digest(f"{phase}:input:{corpus_id}"),
            supporting_input_uri=f"file:///private/tmp/{phase}/support/{corpus_id}.json",
            supporting_input_sha256=_digest(f"{phase}:support:{corpus_id}"),
            output_uri=f"file:///private/tmp/{phase}/output/{corpus_id}.json",
        )
        for corpus_id in sorted(FIXED_CORPORA, key=lambda item: item.encode("utf-8"))
    )
    predecessor = _digest(f"{phase}:predecessor")
    input_aggregate = _digest(
        _canonical(
            {
                "corpora": [
                    {
                        "corpus_id": row.corpus_id,
                        "input_sha256": row.input_sha256,
                        "input_uri": row.input_uri,
                        "supporting_input_sha256": row.supporting_input_sha256,
                        "supporting_input_uri": row.supporting_input_uri,
                    }
                    for row in rows
                ],
                "manifest_sha256": root.manifest_sha256,
                "phase": phase,
                "predecessor_state_sha256": predecessor,
            }
        )
    )
    output_identity = _digest(
        _canonical(
            {
                "corpora": [
                    {"corpus_id": row.corpus_id, "output_uri": row.output_uri} for row in rows
                ],
                "manifest_sha256": root.manifest_sha256,
                "phase": phase,
            }
        )
    )
    label = phase == "label-release"
    workflow = (
        ".github/workflows/confirmatory-label-release.yml"
        if label
        else ".github/workflows/confirmatory-analysis.yml"
    )
    nonce = _digest(f"{phase}:claim-nonce")
    return PhaseClaimContract(
        phase=phase,  # type: ignore[arg-type]
        repository=root.repository,
        claim_workflow_path=workflow,
        claim_workflow_ref=(f"{root.repository}/{workflow}@refs/tags/confirmatory-apparatus-c0"),
        claim_workflow_sha=root.claim_workflow_sha,
        run_head_branch="confirmatory-apparatus-c0",
        claim_job_name="claim-label-release" if label else "claim-analysis",
        execute_job_name="release-labels" if label else "run-analysis",
        claim_nonce=nonce,
        unique_runner_label=derive_phase_runner_label(nonce, phase),  # type: ignore[arg-type]
        runner_id=202 if label else 303,
        runner_name=f"fractal-{phase}-runner",
        runner_group_id=7,
        runner_version=root.runner_version,
        runner_archive_sha256=root.runner_archive_sha256,
        provider_operating_system=root.provider_operating_system,
        provider_architecture=root.provider_architecture,
        host_tool_contract_sha256=root.host_tools.contract_sha256,
        runtime_probe_receipt_sha256=_digest(f"{phase}:runtime-probe"),
        c1_commit=root.c1_commit,
        manifest_sha256=root.manifest_sha256,
        c1_provider_plan_uri=(
            root.label_release_provider_plan_uri if label else root.analysis_provider_plan_uri
        ),
        c1_provider_plan_sha256=(
            root.label_release_provider_plan_sha256 if label else root.analysis_provider_plan_sha256
        ),
        run_receipt_sha256=root.run_receipt_sha256,
        oci_index_digest=(root.release_oci_index_digest if label else root.oci_index_digest),
        oci_platform_manifest_digest=(
            root.release_oci_platform_manifest_digest
            if label
            else root.analysis_oci_platform_manifest_digest
        ),
        tle_binary_sha256=root.release_tle_binary_sha256 if label else None,
        online_execution_claim_contract_sha256=root.contract_sha256,
        predecessor_state_sha256=predecessor,
        predecessor_ledger_commit="5" * 40,
        corpora=rows,
        phase_input_aggregate_sha256=input_aggregate,
        phase_output_identity=output_identity,
        maximum_runtime_seconds=21_600 if label else 43_200,
        label_release_beacon=root.beacon if label else None,
    )


def _phase_provider(contract: PhaseClaimContract) -> ProviderExecutionIdentity:
    return ProviderExecutionIdentity(
        repository=contract.repository,
        workflow_path=contract.claim_workflow_path,
        workflow_ref=contract.claim_workflow_ref,
        workflow_sha=contract.claim_workflow_sha,
        run_head_branch=contract.run_head_branch,
        run_id=888,
        run_attempt=1,
        claim_job_id=889,
        claim_job_name=contract.claim_job_name,
        execute_job_name=contract.execute_job_name,
        runner_id=contract.runner_id,
        runner_name=contract.runner_name,
        runner_group_id=contract.runner_group_id,
        runner_label=contract.unique_runner_label,
        runner_version=contract.runner_version,
        runner_archive_sha256=contract.runner_archive_sha256,
        provider_operating_system=contract.provider_operating_system,
        provider_architecture=contract.provider_architecture,
        host_tool_contract_sha256=contract.host_tool_contract_sha256,
        runtime_probe_receipt_sha256=contract.runtime_probe_receipt_sha256,
        self_hosted=True,
    )


def _live_job(
    contract: ExecutionClaimContract | PhaseClaimContract,
    provider: ProviderExecutionIdentity,
) -> LiveExecuteJobReceipt:
    return LiveExecuteJobReceipt(
        provider_identity_sha256=provider.identity_sha256,
        repository=provider.repository,
        workflow_path=provider.workflow_path,
        workflow_sha=provider.workflow_sha,
        run_head_branch=provider.run_head_branch,
        run_id=provider.run_id,
        run_attempt=provider.run_attempt,
        execute_job_id=9901,
        execute_job_name=provider.execute_job_name,
        runner_id=provider.runner_id,
        runner_name=provider.runner_name,
        runner_group_id=provider.runner_group_id,
        runner_labels=tuple(
            sorted(
                ("ARM64", "macOS", provider.runner_label, "self-hosted"),
                key=lambda item: item.encode("utf-8"),
            )
        ),
        verified_at_utc="2023-11-14T22:13:18+00:00",
    )


class _GitHubApi:
    def __init__(
        self,
        contract: ExecutionClaimContract | PhaseClaimContract,
        provider: ProviderExecutionIdentity,
    ) -> None:
        self.run: dict[str, object] = {
            "id": provider.run_id,
            "run_attempt": provider.run_attempt,
            "event": "workflow_dispatch",
            "status": "in_progress",
            "conclusion": None,
            "head_sha": provider.workflow_sha,
            "head_branch": provider.run_head_branch,
            "path": provider.workflow_path,
            "repository": {"full_name": provider.repository},
        }
        self.job: dict[str, object] = {
            "id": 9901,
            "name": provider.execute_job_name,
            "status": "in_progress",
            "conclusion": None,
            "run_id": provider.run_id,
            "run_attempt": provider.run_attempt,
            "runner_id": provider.runner_id,
            "runner_name": provider.runner_name,
            "runner_group_id": provider.runner_group_id,
            "labels": ["self-hosted", "macOS", "ARM64", provider.runner_label],
        }

    def get(self, endpoint: str) -> object:
        return {"jobs": [self.job]} if endpoint.endswith("jobs?per_page=100") else self.run


def _zenodo() -> AnonymousZenodoAdmission:
    return AnonymousZenodoAdmission(
        record_id=21361837,
        doi="10.5281/zenodo.21361837",
        record_uri="https://zenodo.org/records/21361837",
        published_at_utc="2023-11-14T22:13:00+00:00",
        file_count=C1_REGISTRATION_PACKAGE_FILE_COUNT,
        package_tree_sha256=_digest("zenodo-tree"),
        package_aggregate_sha256=_digest("zenodo-aggregate"),
        receipt_file_sha256=_digest("zenodo-receipt"),
        verified_at_utc="2023-11-14T22:13:01+00:00",
    )


class _Verifier:
    def verify(
        self,
        *,
        contract: ExecutionBeaconContract,
        beacon_bytes: bytes,
    ) -> VerifiedBeaconClaims:
        return VerifiedBeaconClaims(
            chain_hash=contract.chain_hash,
            round=contract.execution_round,
            beacon_bytes_sha256=_digest(beacon_bytes),
            randomness="cd" * 32,
            signature="ef" * 48,
            scheme_id=contract.chain_scheme_id,
            public_key=contract.chain_public_key,
            signature_verified=True,
        )


def _beacon_receipt(contract: ExecutionClaimContract, provider: ProviderExecutionIdentity):
    return verify_execution_beacon(
        contract.beacon,
        beacon_bytes=b'{"round":1,"randomness":"synthetic"}',
        claim_state_sha256=_digest("claimed-state"),
        claim_ledger_commit="4" * 40,
        provider_identity=provider,
        claim_attested_at_utc="2023-11-14T22:13:19+00:00",
        verifier=_Verifier(),
        verified_at_utc="2023-11-14T22:13:21+00:00",
        design_seed_sha256=contract.design_seed_sha256,
    )


def test_claim_is_closed_and_binds_exact_provider_identity() -> None:
    contract = _claim()
    provider = _provider(contract)
    provider.matches_contract(contract)
    hostile = {**contract.to_dict(), "unregistered": True}
    with pytest.raises(ExecutionClaimError, match="unexpected"):
        ExecutionClaimContract.from_dict(hostile)
    with pytest.raises(ExecutionClaimError, match="workflow_sha"):
        replace(provider, workflow_sha="9" * 40).matches_contract(contract)
    with pytest.raises(ExecutionClaimError, match="claim_job_name"):
        replace(provider, claim_job_name="claim-something-else").matches_contract(contract)
    with pytest.raises(ExecutionClaimError, match="runner_label"):
        replace(
            provider,
            runner_label=derive_phase_runner_label(_digest("other"), "online"),
        ).matches_contract(contract)


def test_live_execute_job_readback_rejects_provider_field_substitution() -> None:
    contract = _claim()
    provider = _provider(contract)
    api = _GitHubApi(contract, provider)
    receipt = verify_live_execute_job(
        api=api,
        contract=contract,
        provider_identity=provider,
        verified_at_utc="2026-07-17T12:00:00+00:00",
    )
    assert receipt.execute_job_id == 9901
    api.job["runner_group_id"] = provider.runner_group_id + 1
    with pytest.raises(ExecutionClaimError, match="runner_group_id"):
        verify_live_execute_job(
            api=api,
            contract=contract,
            provider_identity=provider,
            verified_at_utc="2026-07-17T12:00:00+00:00",
        )
    api.job["runner_group_id"] = provider.runner_group_id
    api.run["head_branch"] = "main"
    with pytest.raises(ExecutionClaimError, match="head_branch"):
        verify_live_execute_job(
            api=api,
            contract=contract,
            provider_identity=provider,
            verified_at_utc="2026-07-17T12:00:00+00:00",
        )


def test_failed_execute_job_binds_terminal_failure_and_assigned_runner() -> None:
    contract = _claim()
    provider = _provider(contract)
    api = _GitHubApi(contract, provider)
    api.job.update(status="completed", conclusion="timed_out")
    receipt = verify_failed_execute_job(
        api=api,
        contract=contract,
        provider_identity=provider,
        verified_at_utc="2026-07-17T12:00:00+00:00",
    )
    assert isinstance(receipt, FailedExecuteJobReceipt)
    assert receipt.runner_assigned is True
    assert receipt.execute_job_id == 9901
    api.job["runner_name"] = "substituted-runner"
    with pytest.raises(ExecutionClaimError, match="runner identity"):
        verify_failed_execute_job(
            api=api,
            contract=contract,
            provider_identity=provider,
            verified_at_utc="2026-07-17T12:00:00+00:00",
        )


def test_failed_execute_job_accepts_unassigned_startup_failure_only_without_identity() -> None:
    contract = _claim()
    provider = _provider(contract)
    api = _GitHubApi(contract, provider)
    api.job.update(
        status="completed",
        conclusion="startup_failure",
        runner_id=0,
        runner_name=None,
        runner_group_id=None,
    )
    receipt = verify_failed_execute_job(
        api=api,
        contract=contract,
        provider_identity=provider,
        verified_at_utc="2026-07-17T12:00:00+00:00",
    )
    assert receipt.runner_assigned is False
    assert receipt.runner_id == 0
    api.job["conclusion"] = "success"
    with pytest.raises(ExecutionClaimError, match="terminal failure"):
        verify_failed_execute_job(
            api=api,
            contract=contract,
            provider_identity=provider,
            verified_at_utc="2026-07-17T12:00:00+00:00",
        )


def test_failed_execute_job_rejects_foreign_confirmatory_label() -> None:
    contract = _claim()
    provider = _provider(contract)
    api = _GitHubApi(contract, provider)
    api.job.update(status="completed", conclusion="failure")
    labels = api.job["labels"]
    assert isinstance(labels, list)
    api.job["labels"] = [
        *labels,
        derive_phase_runner_label(_digest("foreign-claim"), "online"),
    ]
    with pytest.raises(ExecutionClaimError, match="another confirmatory label"):
        verify_failed_execute_job(
            api=api,
            contract=contract,
            provider_identity=provider,
            verified_at_utc="2026-07-17T12:00:00+00:00",
        )


def test_claim_rejects_syntax_only_label_and_runtime_budget_overrun() -> None:
    contract = _claim()
    with pytest.raises(ExecutionClaimError, match="claim-nonce-derived"):
        replace(contract, unique_runner_label="fractal-ann-confirmatory-online-static-label")
    with pytest.raises(ExecutionClaimError, match="20-hour"):
        replace(contract, maximum_online_runtime_seconds=72_001)
    with pytest.raises(ExecutionClaimError, match="20-hour"):
        replace(contract, registered_online_runtime_budget_seconds=72_001)


def test_execution_claim_inputs_reject_legacy_measured_runtime_field() -> None:
    inputs = ExecutionClaimInputs(
        design_seed_sha256=_digest("registered-design-seed"),
        registered_online_runtime_budget_seconds=68_000,
        beacon=_beacon(),
    ).to_dict()
    inputs["measured_full_suite_runtime_seconds"] = 68_000
    with pytest.raises(ExecutionClaimError, match="unexpected"):
        ExecutionClaimInputs.from_dict(inputs)


def test_public_package_and_beacon_are_ordered_before_runtime_authority() -> None:
    contract = _claim()
    provider = _provider(contract)
    receipt = _beacon_receipt(contract, provider)
    claim = _mint_verified_run_claim(
        contract=contract,
        provider_identity=provider,
        claim_state_sha256=_digest("claimed-state"),
        claim_ledger_commit="4" * 40,
        claim_attested_at_utc="2023-11-14T22:13:19+00:00",
        beacon_receipt=receipt,
        live_execute_job_receipt=_live_job(contract, provider),
        zenodo_admission=_zenodo(),
        fresh_revalidator=lambda: None,
    )
    row = contract.corpora[0]
    portable = claim.require_launch(
        manifest_sha256=contract.manifest_sha256,
        corpus_id=row.corpus_id,
        runtime_plan_sha256=row.runtime_plan_sha256,
        output_namespace_uri=row.staging_namespace_uri,
    )
    assert portable.design_seed_sha256 == contract.design_seed_sha256
    assert portable.permutation_seed == receipt.permutation_seed
    with pytest.raises(ExecutionClaimError, match="precede beacon"):
        verify_execution_beacon(
            contract.beacon,
            beacon_bytes=b"beacon",
            claim_state_sha256=_digest("claimed-state"),
            claim_ledger_commit="4" * 40,
            provider_identity=provider,
            claim_attested_at_utc=contract.beacon.execution_publication_time.isoformat(),
            verifier=_Verifier(),
            verified_at_utc="2023-11-14T22:13:21+00:00",
            design_seed_sha256=contract.design_seed_sha256,
        )


def test_beacon_seed_binds_design_seed_and_exact_bytes() -> None:
    contract = _claim()
    provider = _provider(contract)
    first = _beacon_receipt(contract, provider)
    second = verify_execution_beacon(
        contract.beacon,
        beacon_bytes=b'{"round":1,"randomness":"changed"}',
        claim_state_sha256=_digest("claimed-state"),
        claim_ledger_commit="4" * 40,
        provider_identity=provider,
        claim_attested_at_utc="2023-11-14T22:13:19+00:00",
        verifier=_Verifier(),
        verified_at_utc="2023-11-14T22:13:21+00:00",
        design_seed_sha256=contract.design_seed_sha256,
    )
    third = verify_execution_beacon(
        contract.beacon,
        beacon_bytes=b'{"round":1,"randomness":"synthetic"}',
        claim_state_sha256=_digest("claimed-state"),
        claim_ledger_commit="4" * 40,
        provider_identity=provider,
        claim_attested_at_utc="2023-11-14T22:13:19+00:00",
        verifier=_Verifier(),
        verified_at_utc="2023-11-14T22:13:21+00:00",
        design_seed_sha256=_digest("another-design-seed"),
    )
    seeds = {
        first.derived_seed_sha256,
        second.derived_seed_sha256,
        third.derived_seed_sha256,
    }
    assert len(seeds) == 3


def test_authority_cannot_be_forged_and_revalidates_before_each_launch() -> None:
    contract = _claim()
    provider = _provider(contract)
    receipt = _beacon_receipt(contract, provider)
    with pytest.raises(ExecutionClaimError, match="only come from provider"):
        claim_module.VerifiedRunClaimCapability(
            contract=contract,
            provider_identity=provider,
            claim_state_sha256=_digest("claimed-state"),
            claim_ledger_commit="4" * 40,
            claim_attested_at_utc="2023-11-14T22:13:19+00:00",
            beacon_receipt=receipt,
            live_execute_job_receipt=_live_job(contract, provider),
            zenodo_admission=_zenodo(),
            _fresh_revalidator=lambda: None,
            _minted_monotonic_ns=0,
            _capability=object(),
        )
    calls = 0

    def revalidate() -> None:
        nonlocal calls
        calls += 1
        if calls > 2:
            raise ExecutionClaimError("provider tip changed")

    capability = _mint_verified_run_claim(
        contract=contract,
        provider_identity=provider,
        claim_state_sha256=_digest("claimed-state"),
        claim_ledger_commit="4" * 40,
        claim_attested_at_utc="2023-11-14T22:13:19+00:00",
        beacon_receipt=receipt,
        live_execute_job_receipt=_live_job(contract, provider),
        zenodo_admission=_zenodo(),
        fresh_revalidator=revalidate,
    )
    row = contract.corpora[0]
    capability.require_launch(
        manifest_sha256=contract.manifest_sha256,
        corpus_id=row.corpus_id,
        runtime_plan_sha256=row.runtime_plan_sha256,
        output_namespace_uri=row.staging_namespace_uri,
    )
    with pytest.raises(ExecutionClaimError, match="provider tip changed"):
        capability.require_launch(
            manifest_sha256=contract.manifest_sha256,
            corpus_id=row.corpus_id,
            runtime_plan_sha256=row.runtime_plan_sha256,
            output_namespace_uri=row.staging_namespace_uri,
        )


def test_five_tree_aggregate_rejects_relabeling() -> None:
    contract = _claim()
    provider = _provider(contract)
    trees = tuple(
        CorpusOutputTree(
            corpus_id=row.corpus_id,
            output_namespace_uri=row.canonical_namespace_uri,
            tree_sha256=_digest(f"tree:{row.corpus_id}"),
        )
        for row in contract.corpora
    )
    payload = {
        "claim_ledger_commit": "4" * 40,
        "claim_state_sha256": _digest("claimed-state"),
        "corpus_trees": [row.to_dict() for row in trees],
        "derivation": "sha256-five-canonical-output-trees-v1",
        "output_aggregate_identity": contract.output_aggregate_identity,
        "provider_identity_sha256": provider.identity_sha256,
        "execute_job_id": 5001,
    }
    aggregate = RunOutputAggregate(
        claim_state_sha256=payload["claim_state_sha256"],
        claim_ledger_commit=payload["claim_ledger_commit"],
        provider_identity_sha256=payload["provider_identity_sha256"],
        execute_job_id=payload["execute_job_id"],
        output_aggregate_identity=payload["output_aggregate_identity"],
        corpus_trees=trees,
        aggregate_sha256=_digest(_canonical(payload)),
    )
    assert len(aggregate.corpus_trees) == 5
    with pytest.raises(ExecutionClaimError, match="canonical"):
        replace(aggregate, execute_job_id=5002)


def test_zenodo_admission_is_exactly_the_public_package() -> None:
    admission = _zenodo()
    assert admission.file_count == C1_REGISTRATION_PACKAGE_FILE_COUNT
    assert admission.schema_version == "fractal-zenodo-anonymous-admission-v2"
    with pytest.raises(
        ExecutionClaimError,
        match=rf"exactly {C1_REGISTRATION_PACKAGE_FILE_COUNT}",
    ):
        replace(
            admission,
            file_count=C1_REGISTRATION_PACKAGE_FILE_COUNT - 1,
        )
    with pytest.raises(ExecutionClaimError, match="predates publication"):
        replace(admission, verified_at_utc="2023-11-14T22:12:59+00:00")


def test_label_release_round_has_a_fixed_safety_interval() -> None:
    beacon = _beacon()
    with pytest.raises(ExecutionClaimError, match="strictly later"):
        replace(beacon, label_release_round=100)


def test_phase_claims_bind_distinct_images_inputs_and_timeouts() -> None:
    release = _phase_contract("label-release")
    analysis = _phase_contract("analysis")
    assert release.tle_binary_sha256 == SOURCE_BUILT_LINUX_ARM64_TLE_SHA256
    assert analysis.tle_binary_sha256 is None
    assert release.oci_index_digest != analysis.oci_index_digest
    with pytest.raises(ExecutionClaimError, match="fixed provider job timeout"):
        replace(release, maximum_runtime_seconds=21_601)
    with pytest.raises(ExecutionClaimError, match="fixed provider job timeout"):
        replace(analysis, maximum_runtime_seconds=43_201)
    with pytest.raises(ExecutionClaimError, match="cannot authorize tle"):
        replace(analysis, tle_binary_sha256=SOURCE_BUILT_LINUX_ARM64_TLE_SHA256)


def test_label_release_capability_requires_later_beacon_and_fresh_claim() -> None:
    contract = _phase_contract("label-release")
    provider = _phase_provider(contract)
    receipt = verify_label_release_beacon(
        contract,
        beacon_bytes=b'{"round":101,"randomness":"synthetic"}',
        phase_claim_state_sha256=_digest("label-claim-state"),
        phase_claim_ledger_commit="6" * 40,
        provider_identity=provider,
        claim_attested_at_utc="2023-11-14T22:18:19+00:00",
        live_execute_job_receipt=_live_job(contract, provider),
        verifier=_Verifier(),
        verified_at_utc="2023-11-14T22:18:21+00:00",
    )
    capability = _mint_verified_phase_claim(
        contract=contract,
        provider_identity=provider,
        phase_claim_state_sha256=_digest("label-claim-state"),
        phase_claim_ledger_commit="6" * 40,
        claim_attested_at_utc="2023-11-14T22:18:19+00:00",
        live_execute_job_receipt=_live_job(contract, provider),
        phase_beacon_receipt=receipt,
        fresh_revalidator=lambda: None,
    )
    row = contract.corpora[0]
    portable = capability.require_input(
        corpus_id=row.corpus_id,
        input_uri=row.input_uri,
        input_sha256=row.input_sha256,
        supporting_input_uri=row.supporting_input_uri,
        supporting_input_sha256=row.supporting_input_sha256,
    )
    assert portable.phase_beacon_receipt_sha256 == receipt.receipt_sha256
    with pytest.raises(ExecutionClaimError, match="precede label beacon"):
        verify_label_release_beacon(
            contract,
            beacon_bytes=b"beacon",
            phase_claim_state_sha256=_digest("label-claim-state"),
            phase_claim_ledger_commit="6" * 40,
            provider_identity=provider,
            claim_attested_at_utc=(
                contract.label_release_beacon.label_release_publication_time.isoformat()  # type: ignore[union-attr]
            ),
            live_execute_job_receipt=_live_job(contract, provider),
            verifier=_Verifier(),
            verified_at_utc="2023-11-14T22:18:21+00:00",
        )


def test_live_job_and_phase_beacon_identities_exclude_only_observation_time() -> None:
    contract = _phase_contract("label-release")
    provider = _phase_provider(contract)
    live = _live_job(contract, provider)
    later_live = replace(live, verified_at_utc="2023-11-14T22:18:20+00:00")
    assert later_live.receipt_sha256 != live.receipt_sha256
    assert later_live.job_identity_sha256 == live.job_identity_sha256
    assert (
        replace(later_live, execute_job_id=later_live.execute_job_id + 1).job_identity_sha256
        != live.job_identity_sha256
    )

    beacon = verify_label_release_beacon(
        contract,
        beacon_bytes=b'{"round":101,"randomness":"synthetic"}',
        phase_claim_state_sha256=_digest("label-claim-state"),
        phase_claim_ledger_commit="6" * 40,
        provider_identity=provider,
        claim_attested_at_utc="2023-11-14T22:18:19+00:00",
        live_execute_job_receipt=live,
        verifier=_Verifier(),
        verified_at_utc="2023-11-14T22:18:21+00:00",
    )
    later_beacon = replace(beacon, verified_at_utc="2023-11-14T22:18:22+00:00")
    assert later_beacon.receipt_sha256 != beacon.receipt_sha256
    assert later_beacon.beacon_identity_sha256 == beacon.beacon_identity_sha256
    assert (
        replace(later_beacon, signature=later_beacon.signature + "00").beacon_identity_sha256
        != beacon.beacon_identity_sha256
    )


def test_analysis_capability_has_no_beacon_or_tle_rescue_path() -> None:
    contract = _phase_contract("analysis")
    provider = _phase_provider(contract)
    capability = _mint_verified_phase_claim(
        contract=contract,
        provider_identity=provider,
        phase_claim_state_sha256=_digest("analysis-claim-state"),
        phase_claim_ledger_commit="7" * 40,
        claim_attested_at_utc="2023-11-14T22:19:00+00:00",
        live_execute_job_receipt=_live_job(contract, provider),
        phase_beacon_receipt=None,
        fresh_revalidator=lambda: None,
    )
    row = contract.corpora[0]
    portable = capability.require_input(
        corpus_id=row.corpus_id,
        input_uri=row.input_uri,
        input_sha256=row.input_sha256,
        supporting_input_uri=row.supporting_input_uri,
        supporting_input_sha256=row.supporting_input_sha256,
    )
    assert portable.phase == "analysis"
    assert portable.phase_beacon_receipt_sha256 is None


def test_host_verifier_rejects_venv_escape_through_symlinked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "controlled"
    outside = tmp_path / "outside"
    for path in (
        root / "python/bin/python3.12",
        root / "gh/bin/gh",
        root / "runner/bin/Runner.Listener",
        root / "runner/bin/Runner.Listener.dll",
        root / "runner/config.sh",
        root / "runner/run.sh",
        outside / "venv",
    ):
        if path.suffix or path.name in {"gh", "python3.12", "Runner.Listener"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"tool")
        else:
            path.mkdir(parents=True, exist_ok=True)
    (root / "escape").symlink_to(outside, target_is_directory=True)
    docker_target = tmp_path / "Docker.app/bin/docker"
    docker_target.parent.mkdir(parents=True)
    docker_target.write_bytes(b"docker")
    docker_link = tmp_path / "bin/docker"
    docker_link.parent.mkdir()
    docker_link.symlink_to(docker_target)
    contract = replace(
        _host_tools(root),
        venv_root=str(root / "escape/venv"),
        docker_executable=str(docker_link),
        docker_resolved_executable=str(docker_target),
    )
    expected = {
        "python3.12": OFFICIAL_PYTHON_BUILD_STANDALONE_BINARY_SHA256,
        "gh": OFFICIAL_GH_OSX_ARM64_BINARY_SHA256,
        "Runner.Listener": OFFICIAL_ACTIONS_RUNNER_LISTENER_SHA256,
        "Runner.Listener.dll": OFFICIAL_ACTIONS_RUNNER_LISTENER_DLL_SHA256,
        "config.sh": OFFICIAL_ACTIONS_RUNNER_CONFIG_SHA256,
        "run.sh": OFFICIAL_ACTIONS_RUNNER_RUN_SHA256,
        "docker": REGISTERED_DOCKER_CLIENT_SHA256,
    }
    monkeypatch.setattr(
        claim_module,
        "_hash_file",
        lambda path, *, label: expected[path.name],
    )
    with pytest.raises(ExecutionClaimError, match="venv_root resolves outside"):
        verify_phase_host_tools(
            contract,
            probe_output_dir=tmp_path / "fresh-probes",
            verified_at_utc="2026-07-16T12:00:00+00:00",
        )


def test_fresh_probe_generator_uses_fixed_names_and_is_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _host_tools(tmp_path / "controlled")
    monkeypatch.setattr(claim_module, "capture_phase_host_probe", lambda: contract.host_probe)
    monkeypatch.setattr(
        claim_module,
        "capture_docker_server_probe",
        lambda _docker: contract.docker_server_probe,
    )
    root = tmp_path / "fresh-probes"
    host_path, docker_path = generate_phase_host_probes(contract, root)
    assert host_path == root / PHASE_HOST_PROBE_FILENAME
    assert docker_path == root / DOCKER_SERVER_PROBE_FILENAME
    assert host_path.read_bytes() == contract.host_probe.canonical_file_bytes()
    assert docker_path.read_bytes() == contract.docker_server_probe.canonical_file_bytes()
    with pytest.raises(ExecutionClaimError, match="cannot create"):
        generate_phase_host_probes(contract, root)


def test_docker_probe_has_fixed_commands_and_rejects_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fixed_run(argv, **kwargs):
        calls.append(tuple(argv))
        payload = (
            b'{"Arch":"arm64","GitCommit":"server-build","Os":"linux","Version":"28.3.2"}\n'
            if argv[1] == "version"
            else b'{"KernelVersion":"6.10.14-linuxkit","MemTotal":51539607552,"NCPU":12}\n'
        )
        return subprocess.CompletedProcess(argv, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(claim_module.subprocess, "run", fixed_run)
    probe = capture_docker_server_probe("/usr/local/bin/docker")
    assert probe.engine_build == "server-build"
    assert calls == [
        (
            "/usr/local/bin/docker",
            "version",
            "--format",
            "{{json .Server}}",
        ),
        (
            "/usr/local/bin/docker",
            "info",
            "--format",
            "{{json .}}",
        ),
    ]

    monkeypatch.setattr(
        claim_module.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, stdout=b"{}\n", stderr=b"warning\n"
        ),
    )
    with pytest.raises(ExecutionClaimError, match="unexpected bytes"):
        capture_docker_server_probe("/usr/local/bin/docker")


def _run_execution_claim_module(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fractal_ann_diagnostics.execution_claim", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_module_help_exposes_all_provider_commands() -> None:
    result = _run_execution_claim_module("--help")
    assert result.returncode == 0
    assert "verify-prerequisites" in result.stdout
    assert "claim" in result.stdout
    assert "complete" in result.stdout
    assert "fail" in result.stdout


def test_module_rejects_unknown_flags() -> None:
    result = _run_execution_claim_module(
        "verify-prerequisites",
        "--phase",
        "online",
        "--suite-attempt-id",
        _digest("attempt"),
        "--output-dir",
        "/tmp/execution-claim-output",
        "--github-output",
        "/tmp/execution-claim-github-output",
        "--unknown",
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --unknown" in result.stderr


def test_module_rejects_missing_required_options() -> None:
    result = _run_execution_claim_module("claim", "--phase", "online")
    assert result.returncode == 2
    assert "--suite-attempt-id" in result.stderr
    assert "--output-dir" in result.stderr
    assert "--github-output" in result.stderr
    assert "--prerequisite-receipt" in result.stderr


def test_module_rejects_activation_without_claim_receipt(tmp_path: Path) -> None:
    result = _run_execution_claim_module(
        "verify-prerequisites",
        "--phase",
        "online",
        "--suite-attempt-id",
        _digest("attempt"),
        "--activate-and-execute",
        "--output-dir",
        str(tmp_path / "output"),
        "--github-output",
        str(tmp_path / "github-output"),
    )
    assert result.returncode == 2
    assert "requires --claim-receipt" in result.stderr


def test_private_github_output_is_atomically_published_and_existing_file_appends(
    tmp_path: Path,
) -> None:
    target = tmp_path / "attempt.github-output"

    claim_module._append_github_outputs(target, {"zeta": "2", "alpha": "1"})
    assert target.read_bytes() == b"alpha=1\nzeta=2\n"
    assert not tuple(tmp_path.glob(".attempt.github-output.*.tmp"))

    claim_module._append_github_outputs(target, {"omega": "3"})
    assert target.read_bytes() == b"alpha=1\nzeta=2\nomega=3\n"


def test_private_github_output_write_failure_never_exposes_a_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "attempt.github-output"
    real_write = claim_module.os.write
    writes = 0

    def interrupted_write(descriptor: int, encoded: bytes) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            real_write(descriptor, encoded[:1])
            raise OSError("simulated process interruption")
        return real_write(descriptor, encoded)

    monkeypatch.setattr(claim_module.os, "write", interrupted_write)
    with pytest.raises(ExecutionClaimError, match="cannot append GitHub outputs"):
        claim_module._append_github_outputs(target, {"alpha": "1", "omega": "3"})

    assert not target.exists()
    assert not tuple(tmp_path.glob(".attempt.github-output.*.tmp"))


@pytest.mark.parametrize(
    ("arguments", "handler_name", "seam_name", "expected"),
    (
        (
            (
                "verify-prerequisites",
                "--phase",
                "online",
                "--suite-attempt-id",
                _digest("attempt"),
                "--claim-receipt",
                "claim.json",
                "--activate-and-execute",
                "--output-dir",
                "activation",
                "--github-output",
                "github-output",
            ),
            "_cli_verify_prerequisites",
            "execute_verify_prerequisites_command",
            {
                "phase": "online",
                "suite_attempt_id": _digest("attempt"),
                "output_dir": Path("activation"),
                "claim_receipt_path": Path("claim.json"),
                "activate_and_execute": True,
            },
        ),
        (
            (
                "claim",
                "--phase",
                "label-release",
                "--suite-attempt-id",
                _digest("attempt"),
                "--prerequisite-receipt",
                "prerequisite.json",
                "--output-dir",
                "claim-output",
                "--github-output",
                "github-output",
            ),
            "_cli_claim",
            "execute_claim_command",
            {
                "phase": "label-release",
                "suite_attempt_id": _digest("attempt"),
                "prerequisite_receipt_path": Path("prerequisite.json"),
                "output_dir": Path("claim-output"),
            },
        ),
        (
            (
                "complete",
                "--phase",
                "analysis",
                "--suite-attempt-id",
                _digest("attempt"),
                "--publish",
                "--claim-receipt",
                "claim.json",
                "--evidence-root",
                "evidence",
                "--attestation-bundle",
                "attestation.json",
                "--preparation-receipt",
                "preparation.json",
                "--output-dir",
                "publication",
                "--github-output",
                "github-output",
            ),
            "_cli_complete",
            "execute_complete_command",
            {
                "phase": "analysis",
                "suite_attempt_id": _digest("attempt"),
                "prepare": False,
                "publish": True,
                "claim_receipt_path": Path("claim.json"),
                "evidence_root": Path("evidence"),
                "attestation_bundle_path": Path("attestation.json"),
                "preparation_receipt_path": Path("preparation.json"),
                "output_dir": Path("publication"),
            },
        ),
        (
            (
                "fail",
                "--phase",
                "online",
                "--suite-attempt-id",
                _digest("attempt"),
                "--prepare",
                "--evidence-root",
                "evidence",
                "--output-dir",
                "failure",
                "--github-output",
                "github-output",
            ),
            "_cli_fail",
            "execute_fail_command",
            {
                "phase": "online",
                "suite_attempt_id": _digest("attempt"),
                "prepare": True,
                "publish": False,
                "claim_receipt_path": None,
                "evidence_root": Path("evidence"),
                "attestation_bundle_path": None,
                "preparation_receipt_path": None,
                "output_dir": Path("failure"),
            },
        ),
    ),
)
def test_cli_handlers_are_thin_closed_orchestration_seams(
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
    handler_name: str,
    seam_name: str,
    expected: dict[str, object],
) -> None:
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    parsed = claim_module._build_parser().parse_args(arguments)
    calls: list[dict[str, object]] = []

    def seam(**kwargs: object) -> dict[str, str]:
        calls.append(kwargs)
        return {"closed": "yes"}

    monkeypatch.setattr(orchestration, seam_name, seam)
    result = getattr(claim_module, handler_name)(parsed)

    assert result == {"closed": "yes"}
    assert calls == [expected]


def test_cli_handlers_translate_orchestration_refusals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fractal_ann_diagnostics.provider_workflow_orchestration as orchestration

    def refuse(**kwargs: object) -> dict[str, str]:
        raise orchestration.ProviderWorkflowOrchestrationError("provider evidence is stale")

    monkeypatch.setattr(orchestration, "execute_claim_command", refuse)
    parsed = claim_module._build_parser().parse_args(
        (
            "claim",
            "--phase",
            "online",
            "--suite-attempt-id",
            _digest("attempt"),
            "--prerequisite-receipt",
            "prerequisite.json",
            "--output-dir",
            "claim-output",
            "--github-output",
            "github-output",
        )
    )

    with pytest.raises(ExecutionClaimError, match="provider evidence is stale"):
        claim_module._cli_claim(parsed)
