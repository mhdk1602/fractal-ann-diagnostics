from __future__ import annotations

import ast
import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

import fractal_ann_diagnostics.execution_claim as claim_module

ROOT = Path(__file__).parents[1]
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
EXECUTION_CLAIM_SOURCE = ROOT / "src" / "fractal_ann_diagnostics" / "execution_claim.py"
WORKFLOWS = {
    "online": WORKFLOW_ROOT / "confirmatory-online-execution.yml",
    "label-release": WORKFLOW_ROOT / "confirmatory-label-release.yml",
    "analysis": WORKFLOW_ROOT / "confirmatory-analysis.yml",
}
REHEARSAL_WORKFLOW = WORKFLOW_ROOT / "confirmatory-provider-rehearsal.yml"
VERIFIED_LAUNCHER_ENV = "HOST_PYTHON_VERIFIED_LAUNCHER"
VERIFIED_LAUNCHER_MARKER = "fractal-host-python-verified-launcher-v1"
VERIFIED_LAUNCHER_PREFIX = (
    '"$HOST_PYTHON" -I -S -P -s -c "$HOST_PYTHON_VERIFIED_LAUNCHER" '
    "fractal-host-python-verified-launcher-v1"
)
HOST_IMPORT_ENV = frozenset(
    {
        "HOST_CONTROLLED_ROOT",
        "HOST_PYTHON_IMPORT_ROOT",
        "HOST_PYTHON_IMPORT_TREE_SHA256",
        "HOST_PYTHON_PACKAGE_CONTENT_SHA256",
        "HOST_PYTHON_PACKAGE_TREE_SHA256",
        "HOST_PYTHON_VENV_ROOT",
        "HOST_PYTHON_VENV_SYMLINK_INVENTORY_SHA256",
        "HOST_PYTHON_VENV_TREE_SHA256",
        "HOST_PYTHON_VERIFIED_LAUNCHER_SHA256",
    }
)
JOB_NAMES = dict(claim_module.PHASE_JOB_NAMES)
C0_REF = "refs/tags/confirmatory-apparatus-c0"
PINNED_ACTIONS = {
    "actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6",
    "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
    "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
    "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
    "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990",
}
COMMON_CLI_OPTIONS = frozenset({"--github-output", "--output-dir", "--phase", "--suite-attempt-id"})
INVOCATION_CONTRACT = {
    ("claim", "prerequisites"): (
        "verify-prerequisites",
        COMMON_CLI_OPTIONS,
    ),
    ("claim", "claim"): (
        "claim",
        COMMON_CLI_OPTIONS | {"--prerequisite-receipt"},
    ),
    ("execute", "activate"): (
        "verify-prerequisites",
        COMMON_CLI_OPTIONS | {"--activate-and-execute", "--claim-receipt"},
    ),
    ("execute", "prepare"): (
        "complete",
        COMMON_CLI_OPTIONS | {"--claim-receipt", "--evidence-root", "--prepare"},
    ),
    ("complete", "publish"): (
        "complete",
        COMMON_CLI_OPTIONS
        | {
            "--attestation-bundle",
            "--claim-receipt",
            "--evidence-root",
            "--preparation-receipt",
            "--publish",
        },
    ),
    ("fail", "prepare_failure"): (
        "fail",
        COMMON_CLI_OPTIONS | {"--evidence-root", "--prepare"},
    ),
    ("fail", "publish_failure"): (
        "fail",
        COMMON_CLI_OPTIONS
        | {
            "--attestation-bundle",
            "--evidence-root",
            "--preparation-receipt",
            "--publish",
        },
    ),
}


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


def _text(phase: str) -> str:
    return WORKFLOWS[phase].read_text(encoding="utf-8")


def _workflow_document(phase: str, *, text: str | None = None) -> dict[str, Any]:
    document = yaml.load(_text(phase) if text is None else text, Loader=_UniqueKeyLoader)
    assert isinstance(document, dict)
    return document


def _load_workflow(path: Path) -> dict[str, Any]:
    document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    assert isinstance(document, dict)
    return document


def _self_hosted_jobs() -> tuple[tuple[Path, str, dict[str, Any]], ...]:
    rows: list[tuple[Path, str, dict[str, Any]]] = []
    paths = (*WORKFLOWS.values(), REHEARSAL_WORKFLOW)
    for path in paths:
        document = _load_workflow(path)
        jobs = document["jobs"]
        for job_id, job in jobs.items():
            assert isinstance(job_id, str) and isinstance(job, dict)
            runs_on = job.get("runs-on")
            if isinstance(runs_on, list) and "self-hosted" in runs_on:
                rows.append((path, job_id, job))
    return tuple(rows)


def _cli_invocations(phase: str) -> dict[tuple[str, str], tuple[str, ...]]:
    document = _workflow_document(phase)
    jobs = document.get("jobs")
    assert isinstance(jobs, dict)
    invocations: dict[tuple[str, str], tuple[str, ...]] = {}
    module = "fractal_ann_diagnostics.execution_claim"
    for job_id, raw_job in jobs.items():
        assert isinstance(job_id, str) and isinstance(raw_job, dict)
        steps = raw_job.get("steps")
        assert isinstance(steps, list)
        for step in steps:
            assert isinstance(step, dict)
            run = step.get("run")
            if not isinstance(run, str) or module not in run:
                continue
            step_id = step.get("id")
            assert isinstance(step_id, str)
            command_lines: list[str] = []
            run_lines = run.splitlines()
            module_line = next(index for index, line in enumerate(run_lines) if module in line)
            command_lines.append(run_lines[module_line])
            while command_lines[-1].rstrip().endswith("\\"):
                module_line += 1
                command_lines.append(run_lines[module_line])
            command = "\n".join(command_lines).replace("\\\n", " ")
            words = shlex.split(command)
            module_index = words.index(module)
            key = (job_id, step_id)
            assert key not in invocations
            invocations[key] = tuple(words[module_index + 1 :])
    return invocations


def _parse_cli_invocation(arguments: tuple[str, ...]) -> Any:
    normalized = tuple(
        (
            "a" * 64
            if item == "$SUITE_ATTEMPT_ID"
            else "/tmp/provider-workflow-contract"
            if "$" in item
            else item
        )
        for item in arguments
    )
    parser = claim_module._build_parser()
    parsed = parser.parse_args(normalized)
    claim_module._validate_cli_arguments(parser, parsed)
    return parsed


def _dispatch_block(text: str) -> str:
    return text.split("permissions:", maxsplit=1)[0]


def _multiline_run_blocks(text: str) -> tuple[str, ...]:
    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        match = re.match(r"^(\s*)run:\s*[>|]-?\s*$", lines[index])
        if match is None:
            index += 1
            continue
        indentation = len(match.group(1))
        index += 1
        body: list[str] = []
        while index < len(lines):
            current = lines[index]
            if current.strip() and len(current) - len(current.lstrip()) <= indentation:
                break
            body.append(current)
            index += 1
        blocks.append("\n".join(body))
    return tuple(blocks)


def _normalized_shell(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\\\n", " ")).strip()


def _core_output_contracts() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    tree = ast.parse(EXECUTION_CLAIM_SOURCE.read_text(encoding="utf-8"))
    common: dict[str, set[str]] = {}
    phases: dict[str, set[str]] = {}
    phase_names = {
        "ONLINE_PHASE": "online",
        "LABEL_RELEASE_PHASE": "label-release",
        "ANALYSIS_PHASE": "analysis",
    }
    names = {
        "PREREQUISITE_OUTPUT_KEYS",
        "ACTIVATION_COMMON_OUTPUT_KEYS",
        "CLAIM_OUTPUT_KEYS",
        "PREPARE_COMMON_OUTPUT_KEYS",
        "PUBLISH_OUTPUT_KEYS",
    }
    for node in tree.body:
        target: str | None = None
        value: ast.expr | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
            value = node.value
        if target in names and isinstance(value, ast.Call):
            common[target] = set(ast.literal_eval(value.args[0]))
        elif target == "ACTIVATION_PHASE_OUTPUT_KEYS" and isinstance(value, ast.Dict):
            for key, item in zip(value.keys, value.values, strict=True):
                assert isinstance(key, ast.Name) and isinstance(item, ast.Call)
                phases[phase_names[key.id]] = set(ast.literal_eval(item.args[0]))
    assert set(common) == names
    assert set(phases) == set(WORKFLOWS)
    return common, phases


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_dispatch_is_one_manifest_derived_identifier(phase: str) -> None:
    workflow = _text(phase)
    dispatch = _dispatch_block(workflow)
    assert "workflow_dispatch:" in dispatch
    assert dispatch.count("suite_attempt_id:") == 1
    assert re.search(r"suite_attempt_id:\n(?:\s{8,}.*\n)*?\s{8,}required: true", dispatch)
    for forbidden in (
        "c0_sha:",
        "c1_sha:",
        "runner_label:",
        "timestamp:",
        "output_path:",
        "artifact_path:",
        "beacon_round:",
        "zenodo",
    ):
        assert forbidden not in dispatch.lower()


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_workflow_is_fixed_to_c0_and_rejects_reruns(phase: str) -> None:
    workflow = _text(phase)
    expected_path = WORKFLOWS[phase].relative_to(ROOT).as_posix()
    assert f"C0_REF: {C0_REF}" in workflow
    assert "test \"$GITHUB_RUN_ATTEMPT\" = '1'" in workflow
    assert 'test "$GITHUB_SHA" = "$GITHUB_WORKFLOW_SHA"' in workflow
    expected_ref = (
        f'test "$GITHUB_WORKFLOW_REF" = "${{GITHUB_REPOSITORY}}/{expected_path}@${{C0_REF}}"'
    )
    assert re.sub(r"\s*\\\n\s*", " ", workflow).find(expected_ref) >= 0


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_provider_job_names_equal_the_core_contract(phase: str) -> None:
    workflow = _text(phase)
    claim_name, execute_name = JOB_NAMES[phase]
    assert f"\n    name: {claim_name}\n" in workflow
    assert f"\n    name: {execute_name}\n" in workflow


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_requested_execute_labels_equal_the_live_core_contract(phase: str) -> None:
    document = _workflow_document(phase)
    execute = document["jobs"]["execute"]
    requested = execute["runs-on"]
    assert isinstance(requested, list)
    derived_label = "fractal-ann-confirmatory-0123456789abcdef"
    assert tuple([*requested[:-1], derived_label]) == (
        claim_module.required_execute_runner_labels(derived_label)
    )
    assert requested[-1] == "${{ needs.claim.outputs.runner_label }}"


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_terminal_transition_names_equal_the_live_core_contract(phase: str) -> None:
    document = _workflow_document(phase)
    claim_state, success_state, failure_state = claim_module.PHASE_STATE_TRANSITIONS[phase]
    steps = {
        (job_id, step["id"]): step["name"]
        for job_id in ("claim", "complete", "fail")
        for step in document["jobs"][job_id]["steps"]
        if "id" in step
    }
    assert claim_state in steps[("claim", "claim")]
    assert success_state in steps[("complete", "publish")]
    assert failure_state in steps[("fail", "publish_failure")]


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_first_security_command_verifies_public_c1_before_claim_or_data(phase: str) -> None:
    workflow = _text(phase)
    verify = workflow.index("verify-prerequisites")
    claim = workflow.index("\n          claim\n")
    claim_step = workflow.index("- name: Publish", verify)
    assert verify < claim
    expected = (
        "Anonymously verify all "
        f"{claim_module.C1_REGISTRATION_PACKAGE_FILE_COUNT} C1 files and both attestations"
    )
    assert expected in workflow[:claim]
    assert "GH_TOKEN" not in workflow[verify:claim_step]
    forbidden_before_claim = (
        "release-timelock-label",
        "fractal-confirmatory-input",
        "fractal-sealed-container-launcher launch",
        "/run/secrets",
        "docker pull",
        "docker run",
    )
    assert not any(token in workflow[:claim] for token in forbidden_before_claim)


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_actions_are_exactly_pinned_and_permissions_are_closed(phase: str) -> None:
    workflow = _text(phase)
    observed = set(re.findall(r"uses:\s*([^\s#]+)", workflow))
    assert observed
    assert observed <= PINNED_ACTIONS
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in observed)
    assert re.search(r"^permissions:\s*\{\}\s*$", workflow, flags=re.MULTILINE)
    assert "pull-requests: write" not in workflow
    assert "issues: write" not in workflow
    assert "security-events: write" not in workflow
    assert "artifact-metadata: write" not in workflow
    assert "subject-digest:" not in workflow
    assert "subject-path:" in workflow


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_each_provider_job_has_only_its_required_token_scopes(phase: str) -> None:
    jobs = _workflow_document(phase)["jobs"]
    state_writer = {
        "actions": "read",
        "attestations": "write",
        "contents": "write",
        "id-token": "write",
    }
    assert jobs["claim"]["permissions"] == state_writer
    assert jobs["execute"]["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert jobs["complete"]["permissions"] == state_writer
    assert jobs["fail"]["permissions"] == state_writer


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_hosted_verifier_python_is_patch_pinned(phase: str) -> None:
    workflow = _text(phase)
    assert workflow.count('python-version: "3.14.6"') == 3
    assert re.search(r'python-version: "3\.14"(?:\s|$)', workflow) is None


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_personal_repository_attestations_retain_their_sigstore_bundles(
    phase: str,
) -> None:
    workflow = _text(phase)
    assert workflow.count("create-storage-record: false") == 4
    assert "claim.sigstore.bundle.json" in workflow
    assert "completion-attestation-missing.txt" in workflow
    assert "provider-failure.sigstore.bundle.json" in workflow


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_claim_package_has_one_source_for_each_recovery_basename(phase: str) -> None:
    steps = _workflow_document(phase)["jobs"]["claim"]["steps"]
    retain = next(step for step in steps if step["name"] == "Retain immutable claim evidence")
    run = retain["run"]
    assert isinstance(run, str)
    claim_root = f"${{RUNNER_TEMP}}/{phase}-claim/."
    original_prerequisites = f"${{RUNNER_TEMP}}/{phase}-prerequisites/."
    assert run.count(claim_root) == 1
    assert original_prerequisites not in run
    assert run.count('"$evidence/claim-receipt.json"') == 1
    assert run.count('"$evidence/provider-plan.materialized.json"') == 1


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_claim_transport_is_id_bound_digest_bound_and_internally_inventoried(
    phase: str,
) -> None:
    workflow = _text(phase)
    jobs = _workflow_document(phase)["jobs"]
    claim = jobs["claim"]
    assert claim["outputs"]["claim_artifact_id"] == (
        "${{ steps.upload_claim.outputs.artifact-id }}"
    )
    assert claim["outputs"]["claim_artifact_digest"] == (
        "${{ steps.upload_claim.outputs.artifact-digest }}"
    )
    assert claim["outputs"]["claim_package_inventory_sha256"] == (
        "${{ steps.package_inventory.outputs.inventory_sha256 }}"
    )
    assert "id: upload_claim" in workflow
    assert "id: package_inventory" in workflow
    assert "claim-package.SHA256SUMS" in workflow
    assert "-printf '%P\\0'" in workflow
    assert "artifact-ids: ${{ needs.claim.outputs.claim_artifact_id }}" in workflow
    assert "name: confirmatory-" in workflow
    upload_blocks = [
        step
        for step in jobs["claim"]["steps"]
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert len(upload_blocks) == 1
    assert upload_blocks[0]["with"]["include-hidden-files"] is True
    download_blocks = [
        step
        for step in jobs["execute"]["steps"]
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    ]
    assert len(download_blocks) == 1
    assert set(download_blocks[0]["with"]) == {"artifact-ids", "path"}
    assert download_blocks[0]["with"]["artifact-ids"] == (
        "${{ needs.claim.outputs.claim_artifact_id }}"
    )


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_downloaded_claim_bytes_are_not_consumed_before_closed_activation(
    phase: str,
) -> None:
    jobs = _workflow_document(phase)["jobs"]
    steps = jobs["execute"]["steps"]
    download_index = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    )
    activation_index = next(
        index for index, step in enumerate(steps) if step.get("id") == "activate"
    )
    assert download_index < activation_index
    for step in steps[download_index + 1 : activation_index]:
        assert "claim-evidence" not in str(step)
        assert "claim-receipt.json" not in str(step)

    activation = steps[activation_index]
    assert set(
        (
            "CLAIM_ARTIFACT_DIGEST",
            "CLAIM_ARTIFACT_ID",
            "CLAIM_PACKAGE_INVENTORY_SHA256",
            "CLAIM_RECEIPT",
            "GH_TOKEN",
        )
    ).issubset(activation["env"])
    assert '--claim-receipt "$CLAIM_RECEIPT"' in activation["run"]
    for forbidden in (
        "--artifact-digest",
        "--artifact-id",
        "--artifact-inventory",
        "--repository",
        "--run-id",
        "--workflow-id",
    ):
        assert forbidden not in activation["run"]


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_shell_blocks_never_expand_untrusted_actions_expressions(phase: str) -> None:
    workflow = _text(phase)
    for block in _multiline_run_blocks(workflow):
        assert "${{" not in block
    lowered = workflow.lower()
    for forbidden in (
        "github_pat",
        "personal_access_token",
        "pull_request_target",
        "repository_dispatch",
    ):
        assert forbidden not in lowered
    if phase == "label-release":
        assert workflow.count("${{ secrets.ZENODO_TOKEN }}") == 1
        assert workflow.count("secrets.") == 1
    else:
        assert "secrets." not in workflow


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_workflow_uses_only_the_closed_execution_claim_cli_flags(phase: str) -> None:
    allowed = {
        "--activate-and-execute",
        "--attestation-bundle",
        "--claim-receipt",
        "--evidence-root",
        "--github-output",
        "--output-dir",
        "--phase",
        "--preparation-receipt",
        "--prepare",
        "--prerequisite-receipt",
        "--publish",
        "--suite-attempt-id",
    }
    for block in _multiline_run_blocks(_text(phase)):
        if "fractal_ann_diagnostics.execution_claim" not in block:
            continue
        cli_arguments = block.split("fractal_ann_diagnostics.execution_claim", maxsplit=1)[1]
        assert set(re.findall(r"--[a-z][a-z-]+", cli_arguments)) <= allowed


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_all_seven_invocations_equal_the_live_core_cli_and_output_contract(phase: str) -> None:
    invocations = _cli_invocations(phase)
    assert set(invocations) == set(INVOCATION_CONTRACT)
    workflow = _text(phase)
    for key, arguments in invocations.items():
        expected_command, expected_options = INVOCATION_CONTRACT[key]
        assert arguments[0] == expected_command
        assert {item for item in arguments if item.startswith("--")} == expected_options

        parsed = _parse_cli_invocation(arguments)
        assert parsed.phase == phase
        emitted = claim_module.expected_cli_output_keys(parsed)
        step_id = key[1]
        consumed = set(re.findall(rf"steps\.{re.escape(step_id)}\.outputs\.([a-z0-9_]+)", workflow))
        assert consumed <= emitted

        intentionally_internal: set[str] = set()
        if step_id == "prerequisites" and phase != "label-release":
            intentionally_internal.add("tle_binary_sha256")
        elif step_id == "claim":
            intentionally_internal.update({"expected_execute_job_name", "provider_identity_sha256"})
        elif step_id == "publish_failure":
            intentionally_internal.add("no_claim_to_fail")
        assert emitted - consumed == intentionally_internal


@pytest.mark.parametrize(
    ("phase", "output_root", "github_output"),
    (
        ("online", "${RUNNER_TEMP}/online-activation", "$GITHUB_OUTPUT"),
        ("label-release", "$attempt_root", "$attempt_output"),
        ("analysis", "$attempt_root", "$attempt_output"),
    ),
)
def test_activation_invocation_exactly_materializes_the_registered_inner_argv(
    phase: str,
    output_root: str,
    github_output: str,
) -> None:
    assert _cli_invocations(phase)[("execute", "activate")] == (
        "verify-prerequisites",
        "--phase",
        phase,
        "--suite-attempt-id",
        "$SUITE_ATTEMPT_ID",
        "--claim-receipt",
        "$CLAIM_RECEIPT",
        "--activate-and-execute",
        "--output-dir",
        output_root,
        "--github-output",
        github_output,
    )


def test_workflow_parser_rejects_duplicate_keys_and_unregistered_cli_options() -> None:
    duplicate = _text("online").replace(
        "permissions: {}\n",
        "permissions: {}\npermissions: {}\n",
        1,
    )
    with pytest.raises(ValueError, match="duplicate YAML key"):
        _workflow_document("online", text=duplicate)

    activation = _cli_invocations("online")[("execute", "activate")]
    with pytest.raises(SystemExit):
        _parse_cli_invocation((*activation, "--runtime-image", "caller/image:mutable"))

    missing_claim = list(activation)
    claim_index = missing_claim.index("--claim-receipt")
    del missing_claim[claim_index : claim_index + 2]
    with pytest.raises(SystemExit):
        _parse_cli_invocation(tuple(missing_claim))


def test_online_claim_precedes_beacon_and_exactly_five_frozen_launches() -> None:
    workflow = _text("online")
    assert (
        'runs-on: [self-hosted, macOS, ARM64, "${{ needs.claim.outputs.runner_label }}"]'
        in workflow
    )
    assert "inputs.runner_label" not in workflow
    assert "RUN_CLAIMED" in workflow
    claim = workflow.index("Publish RUN_CLAIMED")
    beacon = workflow.index("Verify the future beacon, derive the seed, and execute five")
    guarded_execution = workflow.index("--activate-and-execute")
    aggregate = workflow.index("Build the exact five-tree aggregate")
    assert claim < beacon < guarded_execution < aggregate
    assert workflow.count("--activate-and-execute") == 1
    assert "fractal-sealed-container-launcher launch" not in workflow
    assert "FIVE_CORPORA_EXECUTED" in workflow
    assert "launch_receipt_inventory_path" in workflow
    assert "launch_receipt_inventory_sha256" in workflow
    assert "--execution-seed" not in workflow
    assert "--permutation-seed" not in workflow


def test_online_completion_attests_aggregate_and_cannot_skip_run_claim() -> None:
    workflow = _text("online")
    aggregate = workflow.index("Build the exact five-tree aggregate")
    attest = workflow.index("Attest the exact five-tree aggregate")
    complete = workflow.index("Publish ONLINE_COMPLETE")
    assert aggregate < attest < complete
    completion = workflow[complete:]
    assert "--publish" in completion
    assert '--claim-receipt "$CLAIM_RECEIPT"' in completion
    assert '--attestation-bundle "$ATTESTATION_BUNDLE"' in completion
    assert '--preparation-receipt "$PREPARATION_RECEIPT"' in completion
    assert "OPENED" not in completion


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_guarded_execution_proves_the_fixed_suite_and_phase_receipt(phase: str) -> None:
    workflow = _text(phase)
    execute = workflow.split("\n  execute:\n", maxsplit=1)[1].split("\n  complete:\n", maxsplit=1)[
        0
    ]
    assert "FIXED_CORPORA_COMPLETED" in execute
    assert "live_execute_job_receipt_path" in execute
    assert "live_execute_job_receipt_sha256" in execute
    assert "runtime_claim_receipt_path" in execute
    assert "runtime_claim_receipt_sha256" in execute
    assert "execute_job_id" in execute
    assert "phase_host_tool_receipt_path" in execute
    assert "phase_host_tool_receipt_sha256" in execute
    assert "phase_host_tool_receipt_file_sha256" in execute
    assert "phase_execution_receipt_path" in execute
    assert "phase_execution_receipt_sha256" in execute
    assert 'shasum -a 256 "$PHASE_HOST_TOOL_RECEIPT"' in execute
    assert 'shasum -a 256 "$PHASE_EXECUTION_RECEIPT"' in execute


def test_label_release_claim_and_provider_time_gate_precede_every_decrypt() -> None:
    workflow = _text("label-release")
    assert (
        'runs-on: [self-hosted, macOS, ARM64, "${{ needs.claim.outputs.runner_label }}"]'
        in workflow
    )
    assert "LABEL_RELEASE_CLAIMED" in workflow
    claim = workflow.index("Publish LABEL_RELEASE_CLAIMED")
    time_gate = workflow.index("Prove ONLINE_COMPLETE predates the label beacon")
    guarded_release = workflow.index("--activate-and-execute")
    assert claim < time_gate < guarded_release
    assert workflow.count("--activate-and-execute") == 1
    assert "release-timelock-label" not in workflow
    assert "FIVE_LABEL_PAYLOADS_DECRYPTED" in workflow
    assert "label_release_inventory_path" in workflow
    assert "label_release_inventory_sha256" in workflow
    assert 'shasum -a 256 "$LABEL_RELEASE_INVENTORY"' in workflow
    assert "fractal_ann_diagnostics.provider_workflow_validation" in workflow
    assert "label-release-inventory" in workflow
    assert '--claim-receipt "$CLAIM_RECEIPT"' in workflow
    assert "Publish LABELS_RELEASED" in workflow


@pytest.mark.parametrize(
    ("phase", "activation_prefix"),
    (("label-release", "label-release"), ("analysis", "analysis")),
)
def test_restartable_phase_has_one_bounded_same_job_process_retry(
    phase: str,
    activation_prefix: str,
) -> None:
    steps = _workflow_document(phase)["jobs"]["execute"]["steps"]
    activation = next(step for step in steps if step.get("id") == "activate")
    run = activation["run"]
    assert isinstance(run, str)
    assert run.count("run_activation 1") == 1
    assert run.count("run_activation 2") == 1
    assert run.count("--activate-and-execute") == 1
    assert run.count('--github-output "$attempt_output"') == 1
    assert 'cat -- "$ACTIVATION_OUTPUT" >> "$GITHUB_OUTPUT"' in run
    assert '|| test -s "$attempt_output"; then' in run
    assert 'test ! -f "$attempt_output"' in run
    assert f"{activation_prefix}-activation-${{attempt}}" in run
    assert "GITHUB_RUN_ATTEMPT" not in run

    retain = next(step for step in steps if step.get("name") == "Retain immutable phase evidence")
    assert "for attempt in 1 2" in retain["run"]
    assert "activation-attempt-${attempt}" in retain["run"]


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_attempts", "expected_output"),
    (
        ("succeed", 0, "1", "fake_attempt=1\n"),
        ("fail-first", 0, "2", "fake_attempt=2\n"),
        ("partial-output", 1, "1", ""),
    ),
)
@pytest.mark.parametrize("phase", ("label-release", "analysis"))
def test_phase_retry_publishes_only_one_complete_attempt_output(
    tmp_path: Path,
    mode: str,
    expected_status: int,
    expected_attempts: str,
    expected_output: str,
    phase: str,
) -> None:
    steps = _workflow_document(phase)["jobs"]["execute"]["steps"]
    activation = next(step for step in steps if step.get("id") == "activate")
    fake_python = tmp_path / "host-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if test "${EXPECT_TOKEN_FD:-false}" = true; then
  test -z "${ZENODO_TOKEN+x}"
  [[ "${COMPLETION_ANCHOR_TOKEN_FD:-}" =~ ^[0-9]+$ ]]
  IFS= read -r -u "$COMPLETION_ANCHOR_TOKEN_FD" observed_token
  test "$observed_token" = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
fi
attempts="${RUNNER_TEMP}/fake-attempts"
count=0
if test -f "$attempts"; then
  count="$(<"$attempts")"
fi
count="$((count + 1))"
printf '%s\\n' "$count" > "$attempts"
output=''
root=''
while test "$#" -gt 0; do
  case "$1" in
    --github-output)
      output="$2"
      shift 2
      ;;
    --output-dir)
      root="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -m 0700 "$root"
if test "$FAKE_MODE" = fail-first && test "$count" -eq 1; then
  exit 1
fi
if test "$FAKE_MODE" = partial-output; then
  printf '%s\\n' 'partial=true' > "$output"
  exit 1
fi
printf 'fake_attempt=%s\\n' "$count" > "$output"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)
    github_output = tmp_path / "github-output"
    environment = {
        **os.environ,
        "CLAIM_ARTIFACT_DIGEST": "sha256:" + ("a" * 64),
        "CLAIM_ARTIFACT_ID": "1",
        "CLAIM_PACKAGE_INVENTORY_SHA256": "b" * 64,
        "CLAIM_RECEIPT": str(tmp_path / "claim-receipt.json"),
        "FAKE_MODE": mode,
        "EXPECT_TOKEN_FD": "true" if phase == "label-release" else "false",
        "GH_TOKEN": "test-token",
        "GITHUB_OUTPUT": str(github_output),
        "HOST_CONTROLLED_ROOT": str(tmp_path / "controlled"),
        "HOST_PYTHON": str(fake_python),
        "HOST_PYTHON_IMPORT_ROOT": str(tmp_path / "controlled/venv/lib/python3.12/site-packages"),
        "HOST_PYTHON_IMPORT_TREE_SHA256": "d" * 64,
        "HOST_PYTHON_PACKAGE_CONTENT_SHA256": "e" * 64,
        "HOST_PYTHON_PACKAGE_TREE_SHA256": "f" * 64,
        "HOST_PYTHON_VERIFIED_LAUNCHER": "pass",
        "HOST_PYTHON_VERIFIED_LAUNCHER_SHA256": "3" * 64,
        "HOST_PYTHON_VENV_ROOT": str(tmp_path / "controlled/venv"),
        "HOST_PYTHON_VENV_SYMLINK_INVENTORY_SHA256": "1" * 64,
        "HOST_PYTHON_VENV_TREE_SHA256": "2" * 64,
        "RUNNER_TEMP": str(tmp_path),
        "SUITE_ATTEMPT_ID": "c" * 64,
        "ZENODO_TOKEN": "A" * 32,
    }
    completed = subprocess.run(
        ("bash", "-c", activation["run"]),
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode == expected_status, completed.stderr
    assert (tmp_path / "fake-attempts").read_text(encoding="utf-8").strip() == (expected_attempts)
    assert (
        github_output.read_text(encoding="utf-8") if github_output.exists() else ""
    ) == expected_output


def test_analysis_claim_precedes_input_and_result_attestation() -> None:
    workflow = _text("analysis")
    assert (
        'runs-on: [self-hosted, macOS, ARM64, "${{ needs.claim.outputs.runner_label }}"]'
        in workflow
    )
    assert "ANALYSIS_CLAIMED" in workflow
    claim = workflow.index("Publish ANALYSIS_CLAIMED")
    bind = workflow.index("Bind the exact claimed analysis input closure")
    guarded_analysis = workflow.index("--activate-and-execute")
    prepare = workflow.index("Build the exact confirmatory result subject")
    attest = workflow.index("Attest the exact confirmatory result")
    complete = workflow.index("Publish ANALYSIS_COMPLETE")
    assert claim < bind < guarded_analysis < prepare < attest < complete
    assert workflow.count("--activate-and-execute") == 1
    assert "fractal-confirmatory-input" not in workflow
    assert "FIVE_CORPORA_ANALYZED" in workflow
    assert "analysis_result_path" in workflow
    assert "analysis_result_sha256" in workflow
    assert "analysis_execution_receipt_path" in workflow
    assert "analysis_execution_receipt_sha256" in workflow
    assert "analysis_execution_receipt_file_sha256" in workflow
    assert 'shasum -a 256 "$ANALYSIS_EXECUTION_RECEIPT"' in workflow
    assert "fractal_ann_diagnostics.provider_workflow_validation" in workflow
    assert "analysis-execution-receipt" in workflow
    assert "verify-prerequisites" in workflow[bind:guarded_analysis]
    assert '--claim-receipt "$CLAIM_RECEIPT"' in workflow[bind:prepare]
    assert "Publish ANALYSIS_COMPLETE" in workflow


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_claimed_phase_has_success_and_evidence_backed_failure_closure(phase: str) -> None:
    workflow = _text(phase)
    assert "claimed_state_sha256" not in workflow
    assert "claimed_ledger_commit" not in workflow
    assert "steps.claim.outputs.claim_state_sha256" in workflow
    assert "if: always()" in workflow
    assert "Retain immutable phase evidence" in workflow
    assert "Publish evidence-backed FAILED" in workflow
    assert "overwrite: false" in workflow
    assert "if-no-files-found: error" in workflow
    assert "retention-days: 90" in workflow
    assert "fail\n          --prepare" in workflow
    assert "fail\n          --publish" in workflow
    assert '--evidence-root "$EVIDENCE_ROOT"' in workflow
    for output in (
        "publication_receipt_path",
        "publication_receipt_sha256",
        "state_record_sha256",
        "ledger_commit",
    ):
        assert f"steps.publish.outputs.{output}" in workflow
        assert f"steps.publish_failure.outputs.{output}" in workflow


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_provider_paths_are_derived_and_never_dispatch_controlled(phase: str) -> None:
    workflow = _text(phase)
    assert set(re.findall(r"\binputs\.([a-z_]+)", workflow)) == {"suite_attempt_id"}
    assert "inputs.output" not in workflow
    assert "inputs.path" not in workflow
    assert "inputs.namespace" not in workflow
    assert "inputs.timestamp" not in workflow
    for forbidden in (
        "--gh-path",
        "--host-python",
        "--provider-plan",
        "--runner-listener",
        "--runtime-image",
        "--runtime-platform",
    ):
        assert forbidden not in workflow
    assert 'test -n "$SUITE_NAMESPACE"' in workflow


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_workflow_output_symbols_are_closed_by_the_core_cli_contract(phase: str) -> None:
    workflow = _text(phase)
    common, phases = _core_output_contracts()
    allowed = {
        "prerequisites": common["PREREQUISITE_OUTPUT_KEYS"],
        "claim": common["CLAIM_OUTPUT_KEYS"],
        "activate": common["ACTIVATION_COMMON_OUTPUT_KEYS"] | phases[phase],
        "prepare": common["PREPARE_COMMON_OUTPUT_KEYS"]
        | {"completion_predicate_path", "completion_predicate_sha256"},
        "prepare_failure": common["PREPARE_COMMON_OUTPUT_KEYS"]
        | {
            "failure_predicate_path",
            "failure_predicate_sha256",
            "no_claim_to_fail",
        },
        "publish": common["PUBLISH_OUTPUT_KEYS"],
        "publish_failure": common["PUBLISH_OUTPUT_KEYS"] | {"no_claim_to_fail"},
    }
    for step, keys in allowed.items():
        referenced = set(re.findall(rf"steps\.{step}\.outputs\.([a-z0-9_]+)", workflow))
        assert referenced <= keys


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_provider_identity_is_live_context_not_caller_cli(phase: str) -> None:
    workflow = _text(phase)
    dispatch = _dispatch_block(workflow)
    for forbidden in (
        "--run-id",
        "--run-attempt",
        "--workflow-ref",
        "--workflow-sha",
        "--job-id",
        "--runner-label",
        "--timestamp",
    ):
        assert forbidden not in dispatch
        assert all(
            forbidden not in argument
            for invocation in _cli_invocations(phase).values()
            for argument in invocation
        )
    assert "execute_job_id" not in dispatch
    assert "inputs.execute_job_id" not in workflow
    assert "steps.activate.outputs.execute_job_id" in workflow


def test_phase_images_are_role_and_platform_closed() -> None:
    workflows = {phase: _text(phase) for phase in WORKFLOWS}
    for phase, workflow in workflows.items():
        assert "oci_index_digest" in workflow
        assert "oci_platform_manifest_digest" in workflow
        assert "inputs.runtime_image" not in workflow
        document = _workflow_document(phase)
        platform, image_role, index_role = claim_module.PHASE_RUNTIME_BINDINGS[phase]
        assert document["env"]["REGISTERED_RUNTIME_PLATFORM"] == platform
        assert document["env"]["REGISTERED_RUNTIME_IMAGE_ROLE"] == image_role
        assert document["env"]["REGISTERED_RUNTIME_INDEX_ROLE"] == index_role
        assert 'test "$RUNTIME_PLATFORM" = "$REGISTERED_RUNTIME_PLATFORM"' in workflow
        assert 'test "$RUNTIME_IMAGE_ROLE" = "$REGISTERED_RUNTIME_IMAGE_ROLE"' in workflow
        assert 'test "$RUNTIME_INDEX_ROLE" = "$REGISTERED_RUNTIME_INDEX_ROLE"' in workflow
        assert "runtime_image_role" in workflow
        assert "runtime_index_role" in workflow

    online = workflows["online"]
    release = workflows["label-release"]
    analysis = workflows["analysis"]
    assert "tle_binary_sha256" in release
    assert "TLE_BINARY_SHA256" in release
    assert _workflow_document("label-release")["env"]["REGISTERED_RELEASE_TLE_SHA256"] == (
        claim_module.SOURCE_BUILT_LINUX_ARM64_TLE_SHA256
    )
    assert 'test "$TLE_BINARY_SHA256" = "$REGISTERED_RELEASE_TLE_SHA256"' in release
    assert "tle_binary_sha256" not in online
    assert "tle_binary_sha256" not in analysis


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_hosted_materialization_paths_never_cross_to_the_self_hosted_job(phase: str) -> None:
    workflow = _text(phase)
    document = _workflow_document(phase)
    outputs = document["jobs"]["claim"]["outputs"]
    assert isinstance(outputs, dict)
    assert "provider_plan_materialization_path" in claim_module.PREREQUISITE_OUTPUT_KEYS
    assert outputs["provider_plan_path"] == (
        "${{ steps.prerequisites.outputs.provider_plan_path }}"
    )
    assert "provider_plan_materialization_path" not in outputs
    assert "prerequisite_receipt_path" not in outputs
    assert "claim_receipt_path" not in outputs
    assert all("runner.temp" not in str(value) for value in outputs.values())
    assert workflow.count("steps.prerequisites.outputs.provider_plan_materialization_path") == 1
    assert "needs.claim.outputs.provider_plan_materialization_path" not in workflow
    assert "provider-plan.materialized.json" in workflow

    registered_self_host_paths = {
        "docker_path",
        "docker_resolved_path",
        "gh_path",
        "host_controlled_root",
        "host_python_import_root",
        "host_python_path",
        "host_python_venv_root",
        "phase_evidence_root",
        "provider_plan_path",
        "runner_bootstrap_receipt_path",
        "runner_listener_path",
    }
    for value in outputs.values():
        match = re.fullmatch(r"\$\{\{ steps\.prerequisites\.outputs\.([a-z0-9_]+) }}", str(value))
        if match is not None and (
            match.group(1).endswith(("_path", "_root")) or match.group(1) == "phase_evidence_root"
        ):
            assert match.group(1) in registered_self_host_paths


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_self_hosted_phase_probe_pins_ephemeral_bootstrap(phase: str) -> None:
    workflow = _text(phase)
    execute = workflow.split("\n  execute:\n", maxsplit=1)[1].split("\n  complete:\n", maxsplit=1)[
        0
    ]
    assert "ACTIONS_RUNNER_VERSION: 2.335.1" in workflow
    assert (
        "ACTIONS_RUNNER_ARCHIVE_SHA256: "
        "e1a9bc7a3661e06fa0b129d15c2064fe65dc81a431001d8958a9db1409b73769"
    ) in workflow
    assert 'ACTIONS_RUNNER_ARCHIVE_BYTE_COUNT: "127138003"' in workflow
    assert 'ACTIONS_RUNNER_BOOTSTRAP_FLAGS: "--ephemeral --disableupdate --unattended"' in workflow
    assert "Assert the immutable ephemeral runner bootstrap before phase input" in workflow
    assert "RUNNER_LISTENER" in workflow
    assert "EXPECTED_RUNNER_LABEL" in execute
    assert "^fractal-ann-confirmatory-[a-z0-9-]{16,96}$" in execute
    assert "/opt/fractal-actions-runner" not in workflow
    assert "/controlled" not in workflow
    assert "latest" not in workflow.lower()


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_self_hosted_execution_uses_only_c1_pinned_host_tools(phase: str) -> None:
    workflow = _text(phase)
    execute = workflow.split("\n  execute:\n", maxsplit=1)[1].split("\n  complete:\n", maxsplit=1)[
        0
    ]
    assert "actions/checkout" not in execute
    assert "actions/setup-python" not in execute
    assert "astral-sh/setup-uv" not in execute
    assert "uv sync" not in execute
    # Activation and completion preparation are separate trust decisions.  Both
    # reopen the live GitHub claim authority with the job-scoped token; neither
    # process accepts serialized authority from the other.
    assert execute.count("GH_TOKEN: ${{ github.token }}") == 2
    activation = execute.split("--activate-and-execute", maxsplit=1)[0].rsplit(
        "\n      - name:", maxsplit=1
    )[1]
    assert "GH_TOKEN: ${{ github.token }}" in activation
    preparation = execute.split("\n          --prepare", maxsplit=1)[0].rsplit(
        "\n      - name:", maxsplit=1
    )[1]
    assert "GH_TOKEN: ${{ github.token }}" in preparation
    assert VERIFIED_LAUNCHER_PREFIX in _normalized_shell(execute)
    assert '"$HOST_PYTHON" -m ' not in execute
    assert '"$HOST_PYTHON" - ' not in execute
    assert '"$DOCKER_PATH" version' in execute
    assert "docker version" not in execute
    for token in (
        "DOCKER_CLIENT_BUILD",
        "DOCKER_CLIENT_VERSION",
        "DOCKER_FILE_SHA256",
        "DOCKER_RESOLVED_PATH",
        "GH_VERSION",
        "HOST_PYTHON_SHA256",
        "GH_FILE_SHA256",
        "RUNNER_LISTENER_SHA256",
        "RUNNER_BOOTSTRAP_RECEIPT_SHA256",
        *sorted(HOST_IMPORT_ENV),
    ):
        assert token in execute
    semantic_gh_version_check = (
        'test "$(awk \'NR == 1 {print $3}\' "${RUNNER_TEMP}/gh-version.txt")" = "$GH_VERSION"'
    )
    assert semantic_gh_version_check in execute
    assert 'test "$(head -n 1 "${RUNNER_TEMP}/gh-version.txt")" = "$GH_VERSION"' not in execute


def test_all_eleven_self_hosted_python_calls_use_one_verified_launcher() -> None:
    jobs = _self_hosted_jobs()
    assert len(jobs) == 6
    observed_calls: list[str] = []
    launcher_sources: set[str] = set()
    expected_by_file = {
        "confirmatory-online-execution.yml": 2,
        "confirmatory-label-release.yml": 3,
        "confirmatory-analysis.yml": 3,
        "confirmatory-provider-rehearsal.yml": 3,
    }
    observed_by_file: dict[str, int] = {}

    for path, _job_id, job in jobs:
        document = _load_workflow(path)
        workflow_env = document.get("env", {})
        assert isinstance(workflow_env, dict)
        launcher = workflow_env.get(VERIFIED_LAUNCHER_ENV)
        assert isinstance(launcher, str) and launcher.strip()
        assert "${{" not in launcher
        assert "vars." not in launcher
        launcher_sources.add(launcher)

        job_env = job.get("env", {})
        assert isinstance(job_env, dict)
        assert VERIFIED_LAUNCHER_ENV not in job_env
        merged_env = {**workflow_env, **job_env}
        assert "PYTHONPATH" not in merged_env
        for step in job["steps"]:
            run = step.get("run")
            if not isinstance(run, str):
                continue
            executable_calls = re.findall(
                r'(?m)^\s*"\$HOST_PYTHON"\s+-I\s+-S\s+-P\s+-s\s+-c\s+'
                r'"\$HOST_PYTHON_VERIFIED_LAUNCHER"',
                run,
            )
            if not executable_calls:
                continue
            normalized = _normalized_shell(run)
            call_count = len(executable_calls)
            for _ in range(call_count):
                observed_calls.append(normalized)
                observed_by_file[path.name] = observed_by_file.get(path.name, 0) + 1
            assert call_count == 1
            assert VERIFIED_LAUNCHER_PREFIX in normalized
            assert '"$HOST_PYTHON" -m ' not in normalized
            assert re.search(r'"\$HOST_PYTHON"\s+-\s', normalized) is None
            step_env = step.get("env", {})
            assert isinstance(step_env, dict)
            assert VERIFIED_LAUNCHER_ENV not in step_env
            effective_env = {**merged_env, **step_env}
            assert HOST_IMPORT_ENV <= set(effective_env)

    assert len(launcher_sources) == 1
    assert len(observed_calls) == 11
    assert observed_by_file == expected_by_file


def test_verified_launcher_closes_startup_and_allows_only_registered_modules() -> None:
    documents = [_load_workflow(path) for path in (*WORKFLOWS.values(), REHEARSAL_WORKFLOW)]
    launchers = {document["env"][VERIFIED_LAUNCHER_ENV] for document in documents}
    assert len(launchers) == 1
    launcher = launchers.pop()
    assert VERIFIED_LAUNCHER_MARKER in launcher
    assert hashlib.sha256(launcher.encode("utf-8")).hexdigest() == (
        claim_module.REGISTERED_HOST_PYTHON_LAUNCHER_SHA256
    )
    assert "runpy.run_module" in launcher
    assert launcher.index("runpy.run_module") > launcher.index("HOST_PYTHON_IMPORT_TREE_SHA256")
    for module in (
        "fractal_ann_diagnostics.execution_claim",
        "fractal_ann_diagnostics.provider_rehearsal",
        "fractal_ann_diagnostics.provider_workflow_validation",
    ):
        assert module in launcher
    for forbidden in (
        "sitecustomize",
        "usercustomize",
        "exec(",
        "eval(",
        "subprocess",
    ):
        assert forbidden not in launcher

    for path, _job_id, job in _self_hosted_jobs():
        for step in job["steps"]:
            run = step.get("run")
            if not isinstance(run, str):
                continue
            assert '"$HOST_PYTHON" -m ' not in run, path
            assert re.search(r'"\$HOST_PYTHON"\s+-\s', run) is None, path


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_failure_recovery_does_not_depend_on_claim_job_outputs(phase: str) -> None:
    workflow = _text(phase)
    fail_job = workflow.split("\n  fail:\n", maxsplit=1)[1]
    assert "if: always() && needs.complete.result != 'success'" in fail_job
    assert "Re-admit the fixed C0 failure identity" in fail_job
    assert "test \"$GITHUB_ACTOR\" = 'mhdk1602'" in fail_job
    assert "test \"$GITHUB_TRIGGERING_ACTOR\" = 'mhdk1602'" in fail_job
    assert "test \"$GITHUB_RUN_ATTEMPT\" = '1'" in fail_job
    assert "--claim-receipt" not in fail_job
    assert "Download claimed state evidence" not in fail_job
    assert "claim_state_sha256 != ''" not in fail_job
    assert "failure_receipt_path" not in fail_job
    assert "preparation_receipt_path" in fail_job
    prepare = fail_job.split("Prepare evidence-backed FAILED", maxsplit=1)[1]
    preparation = prepare.split("Attest the immutable failure", maxsplit=1)[0]
    assert "--claim-receipt" not in preparation
    assert '--suite-attempt-id "$SUITE_ATTEMPT_ID"' in preparation
    assert "no_claim_to_fail" in fail_job
    assert fail_job.count("steps.prepare_failure.outputs.no_claim_to_fail != 'true'") == 3
    assert "failure-publication-missing.txt" in fail_job


@pytest.mark.parametrize(
    ("phase", "artifact_directory"),
    (
        ("online", "online-execution-evidence"),
        ("label-release", "label-release-execution-evidence"),
        ("analysis", "analysis-execution-evidence"),
    ),
)
def test_hosted_completion_attests_the_downloaded_subject_path(
    phase: str, artifact_directory: str
) -> None:
    workflow = _text(phase)
    complete = workflow.split("\n  complete:\n", maxsplit=1)[1].split("\n  fail:\n", maxsplit=1)[0]
    expected = f"subject-path: ${{{{ runner.temp }}}}/{artifact_directory}/prepared-subject.json"
    assert expected in complete
    assert "subject-digest:" not in complete
    verify = complete.index("Verify downloaded completion subject and predicate bytes")
    attest = complete.index(expected)
    publish = complete.index("\n          --publish\n")
    assert verify < attest < publish
    assert "sha256sum -c prepared-subject.sha256" in complete
    assert "sha256sum -c completion-predicate.sha256" in complete
    assert "sha256sum -c completion-preparation-receipt.sha256" in complete


@pytest.mark.parametrize("phase", tuple(WORKFLOWS))
def test_claim_and_failure_attest_only_verified_subject_bytes(phase: str) -> None:
    workflow = _text(phase)
    claim_verify = workflow.index("Verify the exact claimed subject and predicate bytes")
    claim_attest = workflow.index("Attest the claimed state")
    assert claim_verify < claim_attest
    assert "claim_subject_sha256" in workflow[claim_verify:claim_attest]
    assert "claim_predicate_sha256" in workflow[claim_verify:claim_attest]

    failure_verify = workflow.index("Verify the exact failure subject and predicate bytes")
    failure_attest = workflow.index("Attest the immutable failure incident")
    failure_publish = workflow.index("Publish evidence-backed FAILED")
    assert failure_verify < failure_attest < failure_publish
    failure = workflow[failure_verify:failure_attest]
    assert "prepared_subject_sha256" in failure
    assert "failure_predicate_sha256" in failure
    assert "steps.prepare_failure.outputs.no_claim_to_fail != 'true'" in failure


@pytest.mark.parametrize(
    ("phase", "timeout_minutes"),
    (("online", 1380), ("label-release", 360), ("analysis", 720)),
)
def test_phase_timeout_is_provider_bounded(phase: str, timeout_minutes: int) -> None:
    workflow = _text(phase)
    execute = workflow.split("\n  execute:\n", maxsplit=1)[1].split("\n  complete:\n", maxsplit=1)[
        0
    ]
    assert f"timeout-minutes: {timeout_minutes}" in execute
