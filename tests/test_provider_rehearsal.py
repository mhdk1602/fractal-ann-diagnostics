from __future__ import annotations

import json
import re
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import fractal_ann_diagnostics.provider_rehearsal as rehearsal
from fractal_ann_diagnostics.execution_claim import PhaseHostToolReceipt

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "confirmatory-provider-rehearsal.yml"
PROVIDER_WORKFLOW_GUIDE = ROOT / "research" / "provider-phase-workflows.md"
SHA = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
COMMIT = "d" * 40
SOURCE_COMMIT = "e" * 40
BUILD_CONTEXT_TREE_SHA256 = "d" * 64
UTC = "2026-07-17T00:00:00+00:00"


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    *,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _closure(**changes: object) -> rehearsal.CandidateImageClosure:
    values: dict[str, object] = {
        "build_context_tree_sha256": BUILD_CONTEXT_TREE_SHA256,
        "candidate_branch": "c0-candidate/rehearsal",
        "candidate_package_checksums_sha256": SHA,
        "github_ref": "refs/heads/c0-candidate/rehearsal",
        "github_run_attempt": 1,
        "github_run_id": 120,
        "github_sha": SOURCE_COMMIT,
        "github_workflow_ref": (
            "mhdk1602/fractal-ann-diagnostics/"
            ".github/workflows/confirmatory-image.yml@"
            "refs/heads/c0-candidate/rehearsal"
        ),
        "github_workflow_sha": SOURCE_COMMIT,
        "mode": "candidate",
        "release_govulncheck_adjudication_sha256": SHA,
        "release_image_index_digest": f"sha256:{SHA_B}",
        "release_image_reference": (
            "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-release-candidate@"
            f"sha256:{SHA_B}"
        ),
        "release_linux_arm64_manifest_digest": f"sha256:{SHA}",
        "release_oci_attestation_bundle_sha256": SHA_B,
        "release_oci_attestation_verification_sha256": SHA_C,
        "release_reproducibility_receipt_sha256": SHA,
        "release_security_adjudication_sha256": SHA_B,
        "release_tle_interoperability_receipt_sha256": SHA_C,
        "repository": rehearsal.REPOSITORY,
        "schema_version": rehearsal.CANDIDATE_IMAGE_CLOSURE_SCHEMA,
        "scientific_image_index_digest": f"sha256:{SHA}",
        "scientific_image_reference": (
            f"ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-candidate@sha256:{SHA}"
        ),
        "scientific_linux_amd64_manifest_digest": f"sha256:{SHA_B}",
        "scientific_linux_amd64_runtime_extraction_sha256": SHA_C,
        "scientific_linux_arm64_manifest_digest": f"sha256:{SHA_C}",
        "scientific_linux_arm64_runtime_extraction_sha256": SHA_B,
        "scientific_oci_attestation_bundle_sha256": SHA,
        "scientific_oci_attestation_verification_sha256": SHA_B,
    }
    values.update(changes)
    return rehearsal.CandidateImageClosure(**values)


def _admission(
    phase: rehearsal.ProviderPhase = rehearsal.ONLINE_PHASE,
) -> rehearsal.RehearsalPhaseAdmission:
    platform, image_role, index_role = rehearsal.PHASE_RUNTIME_BINDINGS[phase]
    plan_sha = {
        rehearsal.ONLINE_PHASE: SHA,
        rehearsal.LABEL_RELEASE_PHASE: SHA_B,
        rehearsal.ANALYSIS_PHASE: SHA_C,
    }[phase]
    candidate, index, platform_digest, probe_digest = rehearsal._candidate_binding(
        phase, _closure()
    )
    return rehearsal.RehearsalPhaseAdmission(
        phase=phase,
        build_context_tree_sha256=BUILD_CONTEXT_TREE_SHA256,
        candidate_bootstrap_closure_sha256=_closure().bootstrap_closure_sha256,
        candidate_image_closure_file_sha256=_closure().file_sha256,
        candidate_image_reference=candidate,
        candidate_image_index_digest=index,
        candidate_platform_manifest_digest=platform_digest,
        candidate_runtime_probe_receipt_sha256=probe_digest,
        candidate_image_source_commit=SOURCE_COMMIT,
        c0_commit=COMMIT,
        manifest_sha256=SHA_C,
        plan_closure_sha256=SHA_B,
        provider_plan_path=f"/controlled/provider-plans/{phase}/provider-plan.json",
        provider_plan_sha256=plan_sha,
        provider_plan_file_sha256=SHA_C,
        host_python_path="/controlled/python/bin/python3",
        host_python_file_sha256=SHA,
        host_gh_path="/controlled/gh/bin/gh",
        host_gh_file_sha256=SHA_B,
        host_docker_path="/controlled/docker/bin/docker",
        host_docker_file_sha256=SHA_C,
        host_tools_contract_sha256=SHA,
        runtime_platform=platform,
        runtime_image_role=image_role,
        runtime_index_role=index_role,
        runner_archive_sha256=("e1a9bc7a3661e06fa0b129d15c2064fe65dc81a431001d8958a9db1409b73769"),
        runner_group_id=None,
        runner_label=rehearsal.derive_rehearsal_runner_label(
            phase=phase,
            plan_sha256=plan_sha,
            workflow_sha=COMMIT,
            run_id=200,
            run_attempt=1,
        ),
        runner_version="2.335.1",
        workflow_sha=COMMIT,
        run_id=200,
        run_attempt=1,
    )


def _bootstrap(
    admission: rehearsal.RehearsalPhaseAdmission,
) -> rehearsal.RehearsalRunnerBootstrapReceipt:
    return rehearsal.RehearsalRunnerBootstrapReceipt(
        phase=admission.phase,
        repository=rehearsal.REPOSITORY,
        workflow_sha=admission.workflow_sha,
        runner_label=admission.runner_label,
        runner_id=901,
        runner_name=f"rehearsal-{admission.phase}",
        runner_group_id=None,
        runner_version="2.335.1",
        runner_archive_sha256=("e1a9bc7a3661e06fa0b129d15c2064fe65dc81a431001d8958a9db1409b73769"),
        repository_runner_inventory_sha256=SHA,
        ephemeral=True,
        disable_update=True,
        unattended=True,
        registered_at_utc=UTC,
    )


def _api_responses(
    admission: rehearsal.RehearsalPhaseAdmission,
    bootstrap: rehearsal.RehearsalRunnerBootstrapReceipt,
    *,
    head_branch: str = "c0-candidate/rehearsal",
) -> dict[str, bytes]:
    run_endpoint = (
        f"repos/{rehearsal.REPOSITORY}/actions/runs/{admission.run_id}/attempts/"
        f"{admission.run_attempt}"
    )
    run = {
        "actor": {"login": "mhdk1602"},
        "conclusion": None,
        "event": "workflow_dispatch",
        "head_branch": head_branch,
        "head_sha": admission.workflow_sha,
        "id": admission.run_id,
        "path": rehearsal.REHEARSAL_WORKFLOW_PATH,
        "repository": {"full_name": rehearsal.REPOSITORY},
        "run_attempt": admission.run_attempt,
        "status": "in_progress",
        "triggering_actor": {"login": "mhdk1602"},
    }
    jobs = {
        "jobs": [
            {
                "conclusion": None,
                "id": 333,
                "labels": [
                    "self-hosted",
                    "macOS",
                    "ARM64",
                    admission.runner_label,
                ],
                "name": rehearsal.REHEARSAL_JOB_NAMES[admission.phase],
                "run_attempt": admission.run_attempt,
                "run_id": admission.run_id,
                "runner_group_id": bootstrap.runner_group_id,
                "runner_id": bootstrap.runner_id,
                "runner_name": bootstrap.runner_name,
                "status": "in_progress",
            }
        ]
    }
    return {
        run_endpoint: json.dumps(run, separators=(",", ":"), sort_keys=True).encode(),
        f"{run_endpoint}/jobs?per_page=100": json.dumps(
            jobs, separators=(",", ":"), sort_keys=True
        ).encode(),
    }


class _FakeApi:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get_bytes(self, endpoint: str) -> bytes:
        self.calls.append(endpoint)
        return self.responses[endpoint]


def test_repository_runner_inventory_records_zero_baseline_without_inventing_a_group(
    tmp_path: Path,
) -> None:
    endpoint = f"repos/{rehearsal.REPOSITORY}/actions/runners?per_page=100"
    raw = b'{"runners":[],"total_count":0}'
    receipt, observed = rehearsal.capture_repository_runner_inventory(
        api=_FakeApi({endpoint: raw}),
        captured_at_utc=UTC,
    )
    assert observed == raw
    assert receipt.total_count == 0
    assert receipt.runners == ()
    path = tmp_path / "repository-runner-inventory.json"
    path.write_bytes(rehearsal._canonical_bytes(receipt.to_dict()) + b"\n")
    assert rehearsal.RepositoryRunnerInventoryReceipt.from_file(path) == receipt


def test_repository_runner_inventory_is_complete_typed_and_page_bounded() -> None:
    endpoint = f"repos/{rehearsal.REPOSITORY}/actions/runners?per_page=100"
    response = {
        "total_count": 1,
        "runners": [
            {
                "id": 77,
                "name": "candidate-online",
                "os": "macOS",
                "status": "offline",
                "busy": False,
                "labels": [
                    {"id": 3, "name": "ARM64", "type": "read-only"},
                    {"id": 2, "name": "macOS", "type": "read-only"},
                    {"id": 1, "name": "self-hosted", "type": "read-only"},
                    {
                        "id": 4,
                        "name": "fractal-ann-rehearsal-online-" + "1" * 24,
                        "type": "custom",
                    },
                ],
            }
        ],
    }
    raw = json.dumps(response, separators=(",", ":"), sort_keys=True).encode()
    receipt, _ = rehearsal.capture_repository_runner_inventory(
        api=_FakeApi({endpoint: raw}),
        captured_at_utc=UTC,
    )
    assert receipt.runners[0].runner_id == 77
    assert receipt.runners[0].labels == tuple(
        sorted(receipt.runners[0].labels, key=lambda value: value.encode("utf-8"))
    )

    response["total_count"] = 2
    truncated = json.dumps(response).encode()
    with pytest.raises(rehearsal.ProviderRehearsalError, match="another page"):
        rehearsal.capture_repository_runner_inventory(
            api=_FakeApi({endpoint: truncated}),
            captured_at_utc=UTC,
        )


def test_bootstrap_is_written_only_for_one_stopped_exactly_labeled_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission()
    endpoint = f"repos/{rehearsal.REPOSITORY}/actions/runners?per_page=100"
    labels = [*rehearsal.BASE_EXECUTE_RUNNER_LABELS, admission.runner_label]
    response = {
        "total_count": 1,
        "runners": [
            {
                "id": 901,
                "name": "candidate-online",
                "os": "macOS",
                "status": "offline",
                "busy": False,
                "labels": [
                    {"id": index, "name": label, "type": "custom"}
                    for index, label in enumerate(labels, start=1)
                ],
            }
        ],
    }
    raw = json.dumps(response, separators=(",", ":"), sort_keys=True).encode()
    host_tools = SimpleNamespace(controlled_root=str(tmp_path / "controlled"))
    monkeypatch.setattr(
        rehearsal,
        "_load_fixed_plan_components",
        lambda value: ({}, b"{}\n", host_tools),
    )
    receipt, inventory, observed, output = rehearsal.prepare_rehearsal_runner_bootstrap(
        admission=admission,
        runner_name="candidate-online",
        api=_FakeApi({endpoint: raw}),
        captured_at_utc=UTC,
    )
    assert observed == raw
    assert receipt.runner_group_id is None
    assert receipt.repository_runner_inventory_sha256 == inventory.file_sha256
    assert (output / "bootstrap-receipt.json").is_file()
    assert (output / "repository-runner-inventory.json").is_file()
    assert (output / "repository-runners-api.raw.json").read_bytes() == raw

    response["runners"][0]["status"] = "online"
    with pytest.raises(rehearsal.ProviderRehearsalError, match="not stopped"):
        rehearsal.prepare_rehearsal_runner_bootstrap(
            admission=admission,
            runner_name="candidate-online",
            api=_FakeApi({endpoint: json.dumps(response).encode()}),
            captured_at_utc=UTC,
        )


def _live_job(
    admission: rehearsal.RehearsalPhaseAdmission,
) -> rehearsal.LiveRehearsalJobReceipt:
    bootstrap = _bootstrap(admission)
    api = _FakeApi(_api_responses(admission, bootstrap))
    receipt, _, _ = rehearsal.verify_live_rehearsal_job(
        api=api,
        admission=admission,
        bootstrap=bootstrap,
        run_head_branch="c0-candidate/rehearsal",
        verified_at_utc=UTC,
    )
    return receipt


def _host_tool_receipt() -> PhaseHostToolReceipt:
    return PhaseHostToolReceipt(
        contract_sha256=SHA,
        controlled_root_realpath="/controlled",
        python_executable_sha256=SHA,
        venv_tree_sha256=SHA,
        venv_symlink_inventory_sha256=SHA,
        gh_executable_sha256=SHA,
        runner_listener_sha256=SHA,
        runner_listener_dll_sha256=SHA,
        runner_config_sha256=SHA,
        runner_run_sha256=SHA,
        docker_resolved_executable="/controlled/docker/bin/docker-real",
        docker_executable_sha256=SHA,
        host_probe_receipt_file_sha256=SHA,
        docker_server_probe_receipt_file_sha256=SHA,
        verified_at_utc=UTC,
    )


def _phase_receipt(
    admission: rehearsal.RehearsalPhaseAdmission,
) -> rehearsal.RehearsalPhaseReceipt:
    host = _host_tool_receipt()
    live_job = _live_job(admission)
    docker = host.docker_resolved_executable
    pull = (
        docker,
        "pull",
        "--platform",
        admission.runtime_platform,
        admission.candidate_image_reference,
    )
    inspect = (docker, "image", "inspect", admission.candidate_image_reference)
    run = (
        docker,
        "run",
        "--rm",
        "--pull",
        "never",
        "--platform",
        admission.runtime_platform,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65532:65532",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",
        admission.candidate_image_reference,
        "--help",
    )
    return rehearsal.RehearsalPhaseReceipt(
        admission=admission,
        live_job=live_job,
        runner_bootstrap_receipt_sha256=live_job.runner_bootstrap_receipt_sha256,
        host_tool_receipt=host,
        host_tool_receipt_sha256=host.receipt_sha256,
        candidate_image_reference=admission.candidate_image_reference,
        candidate_image_index_digest=admission.candidate_image_index_digest,
        candidate_platform_manifest_digest=admission.candidate_platform_manifest_digest,
        runtime_platform=admission.runtime_platform,
        runtime_image_role=admission.runtime_image_role,
        runtime_index_role=admission.runtime_index_role,
        pull_argv=pull,
        inspect_argv=inspect,
        run_argv=run,
        pull_stdout_sha256=SHA,
        pull_stderr_sha256=SHA,
        inspect_stdout_sha256=SHA,
        inspect_stderr_sha256=SHA,
        run_stdout_sha256=SHA,
        run_stderr_sha256=SHA,
        exit_status=0,
        network_mode="none",
        read_only_root=True,
        capabilities_dropped=True,
        no_new_privileges=True,
        study_mount_count=0,
        token_names_scrubbed=("GH_TOKEN",),
        scientific_inputs_opened=False,
        provider_state_mutated=False,
        suite_attempt_id=None,
        completed_at_utc=UTC,
    )


def test_candidate_closure_is_canonical_and_closed(tmp_path: Path) -> None:
    closure = _closure()
    path = tmp_path / "closure.json"
    path.write_bytes(rehearsal._canonical_bytes(closure.to_dict()) + b"\n")
    assert rehearsal.CandidateImageClosure.from_file(path) == closure
    assert closure.file_sha256 == rehearsal._sha256(path.read_bytes())
    assert closure.schema_version == "fractal-c0-candidate-closure-v2"
    assert closure.bootstrap_closure_dict() == {
        "build_context_tree_sha256": BUILD_CONTEXT_TREE_SHA256,
        "release_image_index_digest": f"sha256:{SHA_B}",
        "release_linux_arm64_manifest_digest": f"sha256:{SHA}",
        "schema_version": "fractal-c0-candidate-bootstrap-closure-v1",
        "scientific_image_index_digest": f"sha256:{SHA}",
        "scientific_linux_amd64_manifest_digest": f"sha256:{SHA_B}",
        "scientific_linux_arm64_manifest_digest": f"sha256:{SHA_C}",
    }

    duplicate = path.read_text().replace(
        '"mode":"candidate",', '"mode":"candidate","mode":"candidate",'
    )
    path.write_text(duplicate)
    with pytest.raises(rehearsal.ProviderRehearsalError, match="repeats key"):
        rehearsal.CandidateImageClosure.from_file(path)


def test_label_binds_phase_plan_workflow_run_and_attempt() -> None:
    admission = _admission()
    assert admission.runner_label.startswith("fractal-ann-rehearsal-online-")
    assert admission.runner_label == rehearsal.derive_rehearsal_runner_label(
        phase=admission.phase,
        plan_sha256=admission.provider_plan_sha256,
        workflow_sha=admission.workflow_sha,
        run_id=admission.run_id,
        run_attempt=admission.run_attempt,
    )
    assert (
        rehearsal.derive_rehearsal_runner_label(
            phase=admission.phase,
            plan_sha256=admission.provider_plan_sha256,
            workflow_sha=admission.workflow_sha,
            run_id=201,
            run_attempt=admission.run_attempt,
        )
        != admission.runner_label
    )
    with pytest.raises(rehearsal.ProviderRehearsalError, match="label derivation"):
        replace(admission, run_id=201)


def test_live_job_verifier_reads_real_run_and_jobs_and_accepts_repo_group_null() -> None:
    admission = _admission()
    bootstrap = _bootstrap(admission)
    api = _FakeApi(_api_responses(admission, bootstrap))
    receipt, run_bytes, jobs_bytes = rehearsal.verify_live_rehearsal_job(
        api=api,
        admission=admission,
        bootstrap=bootstrap,
        run_head_branch="c0-candidate/rehearsal",
        verified_at_utc=UTC,
    )
    assert receipt.runner_group_id is None
    assert receipt.execute_job_id == 333
    assert receipt.run_api_sha256 == rehearsal._sha256(run_bytes)
    assert receipt.jobs_api_sha256 == rehearsal._sha256(jobs_bytes)
    assert len(api.calls) == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda run, jobs: run.update(head_branch="master"), "head_branch"),
        (
            lambda run, jobs: jobs["jobs"][0]["labels"].append(
                "fractal-ann-confirmatory-online-" + "1" * 24
            ),
            "another phase label",
        ),
        (lambda run, jobs: jobs["jobs"][0].update(runner_group_id=9), "runner_group_id"),
    ),
)
def test_live_job_verifier_rejects_provider_drift(mutation: Any, message: str) -> None:
    admission = _admission()
    bootstrap = _bootstrap(admission)
    responses = _api_responses(admission, bootstrap)
    endpoints = list(responses)
    run = json.loads(responses[endpoints[0]])
    jobs = json.loads(responses[endpoints[1]])
    mutation(run, jobs)
    responses[endpoints[0]] = json.dumps(run).encode()
    responses[endpoints[1]] = json.dumps(jobs).encode()
    with pytest.raises(rehearsal.ProviderRehearsalError, match=message):
        rehearsal.verify_live_rehearsal_job(
            api=_FakeApi(responses),
            admission=admission,
            bootstrap=bootstrap,
            run_head_branch="c0-candidate/rehearsal",
            verified_at_utc=UTC,
        )


def test_phase_receipt_closes_network_mount_token_and_state_boundaries(tmp_path: Path) -> None:
    receipt = _phase_receipt(_admission())
    path = tmp_path / "phase.json"
    path.write_bytes(rehearsal._canonical_bytes(receipt.to_dict()) + b"\n")
    assert rehearsal.RehearsalPhaseReceipt.from_file(path) == receipt
    assert receipt.run_argv[-1] == "--help"
    assert "--network" in receipt.run_argv
    assert "none" in receipt.run_argv
    assert "--mount" not in receipt.run_argv
    assert receipt.suite_attempt_id is None
    assert receipt.provider_state_mutated is False
    assert receipt.scientific_inputs_opened is False

    with pytest.raises(rehearsal.ProviderRehearsalError, match="production boundary"):
        replace(receipt, provider_state_mutated=True)
    with pytest.raises(rehearsal.ProviderRehearsalError, match="self-check command"):
        replace(receipt, run_argv=(*receipt.run_argv[:-1], "run"))
    with pytest.raises(rehearsal.ProviderRehearsalError, match="bootstrap digest"):
        replace(receipt, runner_bootstrap_receipt_sha256=SHA)
    changed_host_receipt = replace(receipt.host_tool_receipt, contract_sha256=SHA_B)
    with pytest.raises(rehearsal.ProviderRehearsalError, match="receipt contract"):
        replace(
            receipt,
            host_tool_receipt=changed_host_receipt,
            host_tool_receipt_sha256=changed_host_receipt.receipt_sha256,
        )
    with pytest.raises(rehearsal.ProviderRehearsalError, match="job token"):
        replace(receipt, token_names_scrubbed=())


def test_personal_repository_rehearsal_rejects_non_null_runner_groups() -> None:
    admission = _admission()
    with pytest.raises(rehearsal.ProviderRehearsalError, match="must be null"):
        replace(admission, runner_group_id=7)
    with pytest.raises(rehearsal.ProviderRehearsalError, match="must be null"):
        replace(_bootstrap(admission), runner_group_id=7)
    with pytest.raises(rehearsal.ProviderRehearsalError, match="must be null"):
        replace(_live_job(admission), runner_group_id=7)


def test_aggregate_rehashes_three_receipts_against_the_candidate_closure(
    tmp_path: Path,
) -> None:
    paths: dict[str, Path] = {}
    for phase in rehearsal.PHASES:
        receipt = _phase_receipt(_admission(phase))
        path = tmp_path / f"{phase}.json"
        path.write_bytes(rehearsal._canonical_bytes(receipt.to_dict()) + b"\n")
        paths[phase] = path
    closure = _closure()
    aggregate = rehearsal.aggregate_rehearsal_receipts(
        phase_receipt_paths=paths,
        candidate_closure=closure,
        completed_at_utc=UTC,
    )
    assert set(aggregate.phase_receipt_file_sha256) == set(rehearsal.PHASES)
    assert aggregate.candidate_image_closure_file_sha256 == closure.file_sha256
    assert aggregate.candidate_image_source_commit == SOURCE_COMMIT
    assert aggregate.c0_commit == COMMIT
    assert aggregate.build_context_tree_sha256 == BUILD_CONTEXT_TREE_SHA256
    assert aggregate.candidate_bootstrap_closure_sha256 == (closure.bootstrap_closure_sha256)
    assert "c1_commit" not in aggregate.to_dict()
    legacy = aggregate.to_dict()
    legacy["c1_commit"] = legacy.pop("c0_commit")
    with pytest.raises(TypeError):
        rehearsal.RehearsalAggregateReceipt(**legacy)  # type: ignore[arg-type]

    with pytest.raises(rehearsal.ProviderRehearsalError, match="image closure"):
        rehearsal.aggregate_rehearsal_receipts(
            phase_receipt_paths=paths,
            candidate_closure=replace(
                closure,
                scientific_linux_arm64_manifest_digest=f"sha256:{'e' * 64}",
            ),
            completed_at_utc=UTC,
        )

    with pytest.raises(rehearsal.ProviderRehearsalError, match="source, bootstrap, or branch"):
        rehearsal.aggregate_rehearsal_receipts(
            phase_receipt_paths=paths,
            candidate_closure=replace(
                closure,
                build_context_tree_sha256="e" * 64,
            ),
            completed_at_utc=UTC,
        )

    other_branch = "c0-candidate/other"
    with pytest.raises(rehearsal.ProviderRehearsalError, match="source, bootstrap, or branch"):
        rehearsal.aggregate_rehearsal_receipts(
            phase_receipt_paths=paths,
            candidate_closure=replace(
                closure,
                candidate_branch=other_branch,
                github_ref=f"refs/heads/{other_branch}",
                github_workflow_ref=(
                    "mhdk1602/fractal-ann-diagnostics/"
                    ".github/workflows/confirmatory-image.yml@"
                    f"refs/heads/{other_branch}"
                ),
            ),
            completed_at_utc=UTC,
        )


def test_tag_probe_records_null_without_guessing() -> None:
    run_id = 404
    run_attempt = 1
    endpoint = f"repos/{rehearsal.REPOSITORY}/actions/runs/{run_id}/attempts/{run_attempt}"
    run = {
        "conclusion": None,
        "event": "workflow_dispatch",
        "head_branch": None,
        "head_sha": COMMIT,
        "id": run_id,
        "path": rehearsal.REHEARSAL_WORKFLOW_PATH,
        "run_attempt": run_attempt,
        "status": "in_progress",
    }
    jobs = {
        "jobs": [
            {
                "conclusion": None,
                "id": 405,
                "name": "probe-tag-head-branch",
                "status": "in_progress",
            }
        ]
    }
    api = _FakeApi(
        {
            endpoint: json.dumps(run).encode(),
            f"{endpoint}/jobs?per_page=100": json.dumps(jobs).encode(),
        }
    )
    receipt, run_bytes, jobs_bytes = rehearsal.probe_tag_head_branch(
        api=api,
        workflow_sha=COMMIT,
        github_ref="refs/tags/c0-head-branch-probe/one",
        run_id=run_id,
        run_attempt=run_attempt,
        observed_at_utc=UTC,
    )
    assert receipt.observed_head_branch is None
    assert receipt.run_api_sha256 == rehearsal._sha256(run_bytes)
    assert receipt.jobs_api_sha256 == rehearsal._sha256(jobs_bytes)


def test_incident_is_artifact_only_and_cannot_describe_success() -> None:
    receipt = rehearsal.RehearsalIncidentReceipt(
        repository=rehearsal.REPOSITORY,
        workflow_path=rehearsal.REHEARSAL_WORKFLOW_PATH,
        workflow_sha=COMMIT,
        run_head_branch="c0-candidate/rehearsal",
        run_id=200,
        run_attempt=1,
        plan_result="success",
        phase_results={
            rehearsal.ONLINE_PHASE: "failure",
            rehearsal.LABEL_RELEASE_PHASE: "skipped",
            rehearsal.ANALYSIS_PHASE: "skipped",
        },
        production_transition_published=False,
        provider_state_mutated=False,
        suite_attempt_id=None,
        recorded_at_utc=UTC,
    )
    assert receipt.production_transition_published is False
    with pytest.raises(rehearsal.ProviderRehearsalError, match="cannot emit"):
        replace(
            receipt,
            phase_results={phase: "success" for phase in rehearsal.PHASES},
        )


def test_plan_uses_production_loader_and_never_emits_hosted_materialization_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = _closure()
    plans: dict[str, SimpleNamespace] = {}
    materialized: list[Path] = []
    for phase in rehearsal.PHASES:
        platform_name, image_role, index_role = rehearsal.PHASE_RUNTIME_BINDINGS[phase]
        candidate_ref, index_digest, platform_digest, probe = rehearsal._candidate_binding(
            phase, closure
        )
        host_tools = SimpleNamespace(
            python_executable="/controlled/python/bin/python3",
            python_executable_sha256=SHA,
            gh_executable="/controlled/gh/bin/gh",
            gh_executable_sha256=SHA_B,
            docker_executable="/controlled/docker/bin/docker",
            docker_executable_sha256=SHA_C,
            contract_sha256=SHA,
        )
        plans[phase] = SimpleNamespace(
            c1_commit=COMMIT,
            file_sha256=SHA_C,
            host_tools=host_tools,
            manifest_sha256=SHA,
            oci_index_digest=index_digest,
            oci_platform_manifest_digest=platform_digest,
            phase=phase,
            plan_sha256={
                rehearsal.ONLINE_PHASE: SHA,
                rehearsal.LABEL_RELEASE_PHASE: SHA_B,
                rehearsal.ANALYSIS_PHASE: SHA_C,
            }[phase],
            provider_plan_path=f"/controlled/provider-plans/{phase}/provider-plan.json",
            runtime_image=f"ghcr.io/mhdk1602/production@{index_digest}",
            runtime_image_role=image_role,
            runtime_index_role=index_role,
            runtime_platform=platform_name,
            runtime_probe_receipt_sha256=probe,
            runner_archive_sha256=(
                "e1a9bc7a3661e06fa0b129d15c2064fe65dc81a431001d8958a9db1409b73769"
            ),
            runner_group_id=None,
            runner_version="2.335.1",
        )

    def fake_loader(
        path: object,
        *,
        c1_commit: str,
        validation_mode: str,
        c0_commit: str,
    ) -> dict[str, SimpleNamespace]:
        assert Path(path) == tmp_path / "manifest.json"
        assert c1_commit == COMMIT
        assert validation_mode == "candidate-rehearsal"
        assert c0_commit == COMMIT
        return plans

    def fake_materialize(plan: SimpleNamespace, path: Path) -> Path:
        output = Path(path) / "provider-plan.json"
        output.parent.mkdir(parents=True)
        output.write_text("{}\n")
        materialized.append(output)
        return output

    monkeypatch.setattr(rehearsal, "load_provider_phase_plans", fake_loader)
    monkeypatch.setattr(rehearsal, "materialize_provider_phase_plan", fake_materialize)
    monkeypatch.setattr(
        rehearsal,
        "provider_plan_template_closure_sha256",
        lambda path, *, c0_commit: SHA_B,
    )
    admissions, copies = rehearsal.build_rehearsal_admissions(
        manifest_path=tmp_path / "manifest.json",
        c0_commit=COMMIT,
        candidate_closure=closure,
        workflow_sha=COMMIT,
        run_id=200,
        run_attempt=1,
        materialization_root=tmp_path / "hosted",
    )
    assert tuple(materialized) == copies
    assert len(admissions) == 3
    for admission in admissions.values():
        assert admission.c0_commit == COMMIT
        assert admission.candidate_image_source_commit == SOURCE_COMMIT
        assert admission.candidate_image_source_commit != admission.c0_commit
        assert admission.build_context_tree_sha256 == BUILD_CONTEXT_TREE_SHA256
        assert admission.candidate_bootstrap_closure_sha256 == (closure.bootstrap_closure_sha256)
        assert str(tmp_path) not in admission.provider_plan_path
        assert admission.provider_plan_path.startswith("/controlled/provider-plans/")


def _workflow_document() -> dict[str, Any]:
    document = yaml.load(WORKFLOW.read_text(), Loader=_UniqueKeyLoader)
    assert isinstance(document, dict)
    return document


def _module_invocations() -> list[tuple[str, str, tuple[str, ...]]]:
    document = _workflow_document()
    result: list[tuple[str, str, tuple[str, ...]]] = []
    for job_id, job in document["jobs"].items():
        for step in job.get("steps", []):
            run = step.get("run")
            if not isinstance(run, str) or "fractal_ann_diagnostics.provider_rehearsal" not in run:
                continue
            words = shlex.split(run)
            index = words.index("fractal_ann_diagnostics.provider_rehearsal")
            result.append((job_id, step["id"], tuple(words[index + 1 :])))
    return result


def test_workflow_has_no_production_dispatch_or_write_permission() -> None:
    text = WORKFLOW.read_text()
    document = _workflow_document()
    assert document["permissions"] == {}
    assert "contents: write" not in text
    assert "packages: write" not in text
    assert "artifact-metadata: write" not in text
    assert "suite_attempt_id" not in text
    assert "runner_label:" not in text.split("permissions:", maxsplit=1)[0]
    for forbidden in (
        "execution_claim claim",
        "execution_claim complete",
        "execution_claim fail",
        "refs/heads/confirmatory-state",
        "confirmatory-provider-ledger",
        "zenodo",
        "gh release",
        "git tag",
        "git push",
    ):
        assert forbidden not in text.lower()


def test_workflow_keeps_bootstrap_source_p_distinct_from_c0_a() -> None:
    text = WORKFLOW.read_text()
    assert ".head_sha == $sha" not in text
    assert '(.head_sha | test("^[0-9a-f]{40}$"))' in text
    assert (
        "confirmatory-image-candidate-closure-"
        "${{ steps.candidate_run.outputs.source_commit }}-"
        "${{ inputs.candidate_image_run_id }}"
    ) in text
    assert ".github_sha == $source_commit" in text
    assert ".github_workflow_sha == $source_commit" in text


def test_workflow_job_permissions_are_minimal() -> None:
    jobs = _workflow_document()["jobs"]
    assert jobs["plan"]["permissions"] == {"actions": "read", "contents": "read"}
    for job_id in ("execute_online", "execute_label_release", "execute_analysis"):
        assert jobs[job_id]["permissions"] == {"actions": "read"}
    assert jobs["complete"]["permissions"] == {
        "actions": "read",
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    assert jobs["incident"]["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }


def test_self_hosted_jobs_use_nonce_labels_and_no_hosted_artifacts() -> None:
    jobs = _workflow_document()["jobs"]
    for job_id, phase in (
        ("execute_online", "online"),
        ("execute_label_release", "label_release"),
        ("execute_analysis", "analysis"),
    ):
        job = jobs[job_id]
        assert job["runs-on"] == [
            "self-hosted",
            "macOS",
            "ARM64",
            f"${{{{ needs.plan.outputs.{phase}_runner_label }}}}",
        ]
        rendered = json.dumps(job, sort_keys=True)
        assert "actions/checkout" not in rendered
        assert "actions/download-artifact" not in rendered
        assert "hosted-plan-materializations" not in rendered
        assert "candidate-image-source" not in rendered
        assert "docker run" not in rendered


def test_workflow_invocations_parse_and_match_exact_output_interfaces() -> None:
    invocations = _module_invocations()
    assert len(invocations) == 7
    assert [arguments[0] for _, _, arguments in invocations] == [
        "plan",
        "probe-tag-head-branch",
        "execute",
        "execute",
        "execute",
        "complete",
        "incident",
    ]
    parser = rehearsal._build_parser()
    for _, _, invocation in invocations:
        normalized_rows: list[str] = []
        previous = ""
        for item in invocation:
            if previous in {"--run-id", "--run-attempt"}:
                normalized = "1"
            elif previous in {"--c0-commit", "--workflow-sha"}:
                normalized = COMMIT
            elif previous == "--admission-json":
                normalized = json.dumps(_admission().to_dict(), separators=(",", ":"))
            elif "$" in item or item.startswith("$("):
                normalized = "/tmp/rehearsal"
            else:
                normalized = item
            normalized_rows.append(normalized)
            previous = item
        normalized_invocation = tuple(normalized_rows)
        parsed = parser.parse_args(normalized_invocation)
        assert rehearsal.expected_cli_output_keys(parsed)


def test_workflow_step_output_references_stay_inside_module_contracts() -> None:
    text = WORKFLOW.read_text()
    contracts = {
        "plan": rehearsal.PLAN_OUTPUT_KEYS,
        "execute": rehearsal.EXECUTE_OUTPUT_KEYS,
        "complete": rehearsal.COMPLETE_OUTPUT_KEYS,
        "probe": rehearsal.PROBE_OUTPUT_KEYS,
        "incident": rehearsal.INCIDENT_OUTPUT_KEYS,
    }
    for step_id, expected in contracts.items():
        references = set(re.findall(rf"steps\.{step_id}\.outputs\.([A-Za-z_][A-Za-z0-9_]*)", text))
        assert references
        assert references.issubset(expected)


def test_workflow_pins_every_external_action_and_retains_evidence_for_90_days() -> None:
    text = WORKFLOW.read_text()
    uses = re.findall(r"^\s*uses:\s*([^\s#]+)", text, flags=re.MULTILINE)
    assert uses
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", value) for value in uses)
    assert text.count("retention-days: 90") == 7
    assert text.count("create-storage-record: false") == 3


def test_operator_guide_records_zero_baseline_and_reproducible_local_inventory() -> None:
    guide = PROVIDER_WORKFLOW_GUIDE.read_text(encoding="utf-8")
    assert "2026-07-17" in guide
    assert "`total_count = 0`" in guide
    assert "provider_rehearsal inventory" in guide
    assert "prepare-runner-bootstrap" in guide
    assert "runner_group_id: null" in guide
    assert "phase-host-probe.json" in guide
    assert "docker-server-probe.json" in guide
    assert "`execution_claim_inputs`" in guide
    assert "label release and analysis must carry\nJSON null" in guide
