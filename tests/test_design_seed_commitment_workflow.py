from __future__ import annotations

from pathlib import Path

import yaml

from operators import design_seed_commitment as operator

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / operator.ATTESTATION_WORKFLOW


def _workflow() -> tuple[str, dict[str, object]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return text, parsed


def test_workflow_is_a_fixed_hosted_one_attempt_apparatus() -> None:
    text, parsed = _workflow()

    assert parsed["name"] == "Burn and attest one design-seed scope"
    assert parsed["permissions"] == {
        "actions": "read",
        "attestations": "write",
        "contents": "write",
        "id-token": "write",
    }
    assert parsed["concurrency"] == {
        "group": "design-seed-scope-${{ inputs.scope_sha256 }}",
        "cancel-in-progress": False,
    }
    for check in (
        'test "$GITHUB_REPOSITORY" = "$REPOSITORY"',
        "test \"$GITHUB_ACTOR\" = 'mhdk1602'",
        "test \"$GITHUB_TRIGGERING_ACTOR\" = 'mhdk1602'",
        "test \"$GITHUB_EVENT_NAME\" = 'workflow_dispatch'",
        "test \"$GITHUB_RUN_ATTEMPT\" = '1'",
        'test "$GITHUB_REF" = "$APPARATUS_REF"',
        'test "$GITHUB_SHA" = "$GITHUB_WORKFLOW_SHA"',
        "test \"$RUNNER_ENVIRONMENT\" = 'github-hosted'",
        f"APPARATUS_REF: {operator.ATTESTATION_GIT_REF}",
        f"SOURCE_P: {operator.SOURCE_P}",
        f"SOURCE_TREE: {operator.SOURCE_TREE}",
    ):
        assert check in text


def test_commitment_validation_precedes_irreversible_release_and_attestation() -> None:
    text, parsed = _workflow()
    steps = parsed["jobs"]["burn-and-attest"]["steps"]
    names = [step["name"] for step in steps]
    assert names.index("Verify the exact commitment and exact-P package closure") < names.index(
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


def test_checkout_fetches_source_p_for_exact_tree_verification() -> None:
    _, parsed = _workflow()
    steps = parsed["jobs"]["burn-and-attest"]["steps"]
    checkout = next(
        step for step in steps if step["name"] == "Check out the immutable apparatus tag"
    )

    assert checkout["with"]["ref"] == "design-seed-apparatus-v1"
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["fetch-tags"] is True


def test_every_dispatch_must_create_the_unique_scope_release() -> None:
    text, parsed = _workflow()
    release_step = next(
        step
        for step in parsed["jobs"]["burn-and-attest"]["steps"]
        if step["name"] == "Publish the one-shot assetless scope release"
    )
    assert "if" not in release_step
    release = release_step["run"]
    assert 'tag="design-seed-scope-${SCOPE_SHA256}"' in release
    assert "scope release already exists; this scope is permanently burned" in release
    assert "scope tag already exists; this scope is permanently burned" in release
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
        "repository",
        "run_attempt",
        "run_id",
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
    assert f"predicate-type: {operator.ATTESTATION_PREDICATE_TYPE}" in text


def test_workflow_retains_bundle_without_a_credentialed_host_dependency() -> None:
    text, _ = _workflow()
    assert "design-seed-attestation-${SCOPE_SHA256}.sigstore.bundle.json" in text
    assert "scope-release-api.json" in text
    assert "scope-release-tag-api.json" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text
    assert "device" not in text.lower()
    assert "personal access token" not in text.lower()
