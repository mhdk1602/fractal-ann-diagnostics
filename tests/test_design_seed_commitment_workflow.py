from __future__ import annotations

from pathlib import Path

import yaml

from operators import design_seed_commitment as operator

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_RELATIVE = ".github/workflows/design-seed-commitment-v2.yml"
WORKFLOW = ROOT / WORKFLOW_RELATIVE
SUPERSEDED_WORKFLOW = ROOT / ".github" / "workflows" / "design-seed-commitment.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
RECOVERY_AMENDMENT = ROOT / "research" / "design-seed-v1-failure-receipt.json"
RECOVERY_AMENDMENT_SHA256 = "afb4abf60c2e53318908f0dd1ad81d14f4e845d0dbd3f4b1c6c77cf769e9296d"


def _workflow() -> tuple[str, dict[str, object]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return text, parsed


def test_workflow_is_a_fixed_hosted_one_attempt_apparatus() -> None:
    text, parsed = _workflow()

    assert WORKFLOW.is_file()
    assert not SUPERSEDED_WORKFLOW.exists()
    assert operator.ATTESTATION_WORKFLOW == WORKFLOW_RELATIVE
    assert parsed["name"] == "Burn and attest one design-seed scope v2"
    assert parsed["permissions"] == {
        "actions": "read",
        "attestations": "write",
        "contents": "write",
        "id-token": "write",
    }
    assert parsed["concurrency"] == {
        "group": "design-seed-scope-v2-${{ inputs.scope_sha256 }}",
        "cancel-in-progress": False,
    }
    for check in (
        'test "$GITHUB_REPOSITORY" = "$REPOSITORY"',
        "test \"$GITHUB_ACTOR\" = 'mhdk1602'",
        "test \"$GITHUB_TRIGGERING_ACTOR\" = 'mhdk1602'",
        "test \"$GITHUB_EVENT_NAME\" = 'workflow_dispatch'",
        "test \"$GITHUB_RUN_ATTEMPT\" = '1'",
        "test \"$GITHUB_RUN_NUMBER\" = '1'",
        'test "$GITHUB_REF" = "$APPARATUS_REF"',
        'test "$GITHUB_SHA" = "$GITHUB_WORKFLOW_SHA"',
        "test \"$RUNNER_ENVIRONMENT\" = 'github-hosted'",
        f"APPARATUS_REF: {operator.ATTESTATION_GIT_REF}",
        f"SOURCE_P: {operator.SOURCE_P}",
        f"SOURCE_TREE: {operator.SOURCE_TREE}",
    ):
        assert check in text


def test_fresh_dispatch_replay_stops_before_checkout_api_or_burn() -> None:
    _, parsed = _workflow()
    steps = parsed["jobs"]["burn-and-attest"]["steps"]
    identity = steps[0]
    identity_lines = identity["run"].splitlines()

    assert identity["name"] == "Admit the fixed hosted-runner identity and immutable apparatus tag"
    assert identity_lines[:2] == [
        "set -euo pipefail",
        "test \"$GITHUB_RUN_NUMBER\" = '1'",
    ]
    assert steps[1]["name"] == "Check out the immutable apparatus tag"
    assert "actions/checkout" in steps[1]["uses"]

    commitment = next(
        step
        for step in steps
        if step["name"] == "Verify the exact commitment and exact-P package closure"
    )["run"]
    recovery = next(
        step
        for step in steps
        if step["name"] == "Admit the retained v1 amendment and public no-burn state"
    )["run"]
    predicate = next(
        step for step in steps if step["name"] == "Build the closed API-verifiable predicate"
    )["run"]
    assert '--argjson run_number "$GITHUB_RUN_NUMBER"' in commitment
    assert "and .run_number == $run_number" in commitment
    assert 'test "$(jq -r .run_number "$SUBJECT")" = "$GITHUB_RUN_NUMBER"' in recovery
    assert "and .response.run_number == 1" in recovery
    assert "run_number," in recovery
    assert '--argjson run_number "$GITHUB_RUN_NUMBER"' in predicate
    assert "run_number: $run_number" in predicate


def test_commitment_validation_precedes_irreversible_release_and_attestation() -> None:
    text, parsed = _workflow()
    steps = parsed["jobs"]["burn-and-attest"]["steps"]
    names = [step["name"] for step in steps]
    assert names.index("Verify the exact commitment and exact-P package closure") < names.index(
        "Admit the retained v1 amendment and public no-burn state"
    )
    assert names.index("Admit the retained v1 amendment and public no-burn state") < names.index(
        "Publish the one-shot assetless scope release"
    )
    assert names.index("Publish the one-shot assetless scope release") < names.index(
        "Attest only after the immutable burn is visible"
    )
    assert "environment" not in parsed["jobs"]["burn-and-attest"]
    release = next(
        step for step in steps if step["name"] == "Publish the one-shot assetless scope release"
    )["run"]
    assert "/immutable-releases" not in release
    assert "draft: false" in release
    assert '"/repos/${REPOSITORY}/releases" --input "$request"' in release
    assert "for _attempt in $(seq 1 60)" in release
    assert ".immutable == true" in release
    assert '.assets | type == "array" and length == 0' in release
    assert ".author.login == $author" in release
    assert '.object.type == "commit"' in release


def test_v2_admits_the_tracked_amendment_and_public_v1_terminal_state() -> None:
    text, parsed = _workflow()
    steps = parsed["jobs"]["burn-and-attest"]["steps"]
    recovery = next(
        step
        for step in steps
        if step["name"] == "Admit the retained v1 amendment and public no-burn state"
    )["run"]

    assert RECOVERY_AMENDMENT.is_file()
    assert f"RECOVERY_AMENDMENT_SHA256: {RECOVERY_AMENDMENT_SHA256}" in text
    assert "design-seed-commitment-v2-${SCOPE_SHA256}.json" in text
    assert "fractal-design-seed-recovery-amendment-v1" in recovery
    assert 'test "$(jq -r .recovery_amendment_path "$SUBJECT")"' in recovery
    assert 'test "$(jq -r .recovery_amendment_sha256 "$SUBJECT")"' in recovery
    assert "env -i \\" in recovery
    assert "Authorization" not in recovery
    assert "/actions/runs/${SUPERSEDED_RUN_ID}/attempts/1" in recovery
    assert ".response.run_attempt == 1" in recovery
    assert '.response.conclusion == "failure"' in recovery
    assert ".response.head_sha == $sha" in recovery
    assert '.response.actor.login == "mhdk1602"' in recovery
    assert "design-seed-v1-failure-job-api" in recovery
    assert "/actions/jobs/${SUPERSEDED_JOB_ID}" in recovery
    assert recovery.count('conclusion: "skipped"') == 5
    assert "design-seed-v1-release-absence-api" in recovery
    assert "design-seed-v1-tag-absence-api" in recovery
    assert recovery.count("and .http_status == 404") == 1
    assert 'public_projection design-seed-v1-release-absence-api "$release_path" 404' in recovery
    assert 'public_projection design-seed-v1-tag-absence-api "$tag_path" 404' in recovery
    assert "public_projection design-seed-v2-release-preburn-absence-api" in recovery
    assert 'public_projection design-seed-v2-tag-preburn-absence-api "$v2_tag_path" 404' in recovery
    assert '.response.message == "Not Found"' in recovery
    assert 'chmod 0444 "$evidence"/*.json' in recovery


def test_checkout_fetches_source_p_for_exact_tree_verification() -> None:
    _, parsed = _workflow()
    steps = parsed["jobs"]["burn-and-attest"]["steps"]
    checkout = next(
        step for step in steps if step["name"] == "Check out the immutable apparatus tag"
    )

    assert checkout["with"]["ref"] == "design-seed-apparatus-v2"
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["fetch-tags"] is True


def test_burn_pins_the_ci_python_before_installing_uv() -> None:
    _, parsed = _workflow()
    steps = parsed["jobs"]["burn-and-attest"]["steps"]
    ci = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    ci_steps = ci["jobs"]["design-seed-runtime-closure"]["steps"]
    python = next(step for step in steps if step["name"] == "Set up pinned Python")
    ci_python = next(step for step in ci_steps if step["name"] == "Set up Python")
    names = [step["name"] for step in steps]

    assert (
        python["uses"]
        == ci_python["uses"]
        == ("actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1")
    )
    assert python["with"] == ci_python["with"] == {"python-version": "3.14"}
    assert (
        names.index("Check out the immutable apparatus tag")
        < names.index("Set up pinned Python")
        < names.index("Install the pinned verifier environment")
    )


def test_ci_test_matrix_fetches_source_p_for_exact_tree_verification() -> None:
    parsed = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    steps = parsed["jobs"]["test"]["steps"]
    checkout = next(step for step in steps if step["name"] == "Check out repository")

    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False


def test_public_readback_retries_only_transport_and_named_transient_failures() -> None:
    _, parsed = _workflow()
    recovery = next(
        step
        for step in parsed["jobs"]["burn-and-attest"]["steps"]
        if step["name"] == "Admit the retained v1 amendment and public no-burn state"
    )["run"]
    projection = recovery.split("public_projection() {", 1)[1].split("\n          run_path=", 1)[0]

    assert "local max_attempts=4" in projection
    assert "for ((attempt = 1; attempt <= max_attempts; attempt++)); do" in projection
    assert ': > "$body"' in projection
    assert 'if test "$curl_status" -eq 0 && test "$status" = "$expected_status"; then' in projection
    assert 'if test "$curl_status" -eq 0; then' in projection
    retry_case = projection.split('case "$status" in', 1)[1].split("esac", 1)[0]
    assert [line.strip() for line in retry_case.splitlines() if line.strip()] == [
        "429|500|502|503|504) ;;",
        "*) return 1 ;;",
    ]
    assert 'if test "$attempt" -eq "$max_attempts"; then' in projection
    assert 'sleep "$((1 << (attempt - 1)))"' in projection


def test_ci_exercises_the_operator_in_only_the_locked_core_runtime_closure() -> None:
    parsed = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    job = parsed["jobs"]["design-seed-runtime-closure"]
    assert job["name"] == "Design-seed clean runtime closure"
    steps = {step["name"]: step for step in job["steps"]}

    checkout = steps["Check out repository with exact-P history"]
    assert checkout["with"] == {"fetch-depth": 0, "persist-credentials": False}
    export = steps["Export the exact locked core runtime closure"]["run"]
    assert "uv export --frozen --no-dev --no-emit-project" in export
    assert "--format requirements.txt" in export
    install = steps["Install only the exported runtime closure"]["run"]
    assert "python -m venv" in install
    assert "--require-hashes" in install
    exercise = steps["Exercise the host operator in the clean runtime closure"]["run"]
    assert exercise.count("env -i") == 2
    assert "PYTHONNOUSERSITE=1" in exercise
    assert "PYTHONSAFEPATH=1" in exercise
    assert 'operators/design_seed_commitment.py" --help' in exercise
    assert 'import yaml; assert yaml.__version__ == "6.0.3"' in exercise
    assert "--extra" not in export + install + exercise


def test_every_dispatch_must_create_the_unique_scope_release() -> None:
    text, parsed = _workflow()
    release_step = next(
        step
        for step in parsed["jobs"]["burn-and-attest"]["steps"]
        if step["name"] == "Publish the one-shot assetless scope release"
    )
    assert "if" not in release_step
    release = release_step["run"]
    assert 'tag="design-seed-scope-v2-${SCOPE_SHA256}"' in release
    assert 'tag="design-seed-scope-${SCOPE_SHA256}"' not in release
    assert 'if gh api "/repos/${REPOSITORY}/releases/tags/' not in release
    assert 'if gh api "/repos/${REPOSITORY}/git/ref/tags/' not in release
    assert text.count('"/repos/${REPOSITORY}/releases" --input "$request"') == 1
    assert "cleanup" not in text.lower()
    assert "mode=recover" not in text
    assert "mode=resume" not in text


def test_signed_predicate_contains_only_closed_context_and_release_bindings() -> None:
    text, _ = _workflow()
    for field in (
        "actor",
        "commitment_sha256",
        "event",
        "git_ref",
        "release_id",
        "release_name",
        "release_published_at_utc",
        "release_tag",
        "recovery_amendment_path",
        "recovery_amendment_sha256",
        "repository",
        "run_attempt",
        "run_id",
        "run_number",
        "scope_sha256",
        "source_p",
        "source_tree",
        "triggering_actor",
        "workflow",
        "workflow_ref",
        "workflow_sha",
    ):
        assert f"{field}:" in text
    assert "target_round:" not in text
    assert "design_seed_sha256:" not in text
    assert "actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6" in text
    assert 'schema_version: "fractal-design-seed-attestation-predicate-v2"' in text
    assert f"predicate-type: {operator.ATTESTATION_PREDICATE_TYPE}" in text


def test_workflow_retains_bundle_without_a_credentialed_host_dependency() -> None:
    text, _ = _workflow()
    assert "design-seed-attestation-v2-${SCOPE_SHA256}.sigstore.bundle.json" in text
    assert "scope-release-v2-api.json" in text
    assert "scope-release-v2-tag-api.json" in text
    assert "name: design-seed-evidence-v2-${{ inputs.scope_sha256 }}" in text
    for projection in (
        "design-seed-v1-failure-receipt.json",
        "design-seed-v1-failure-run-api.json",
        "design-seed-v1-failure-job-api.json",
        "design-seed-v1-release-absence-api.json",
        "design-seed-v1-tag-absence-api.json",
        "design-seed-v2-release-preburn-absence-api.json",
        "design-seed-v2-tag-preburn-absence-api.json",
        "recovery-actions-run-api-projection.json",
        "recovery-actions-job-api-projection.json",
        "recovery-release-absence-api-projection.json",
        "recovery-release-tag-absence-api-projection.json",
    ):
        assert projection in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text
    assert "device" not in text.lower()
    assert "personal access token" not in text.lower()
