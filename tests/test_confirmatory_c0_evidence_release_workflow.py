from __future__ import annotations

import re
import stat
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest
import yaml

from fractal_ann_diagnostics.zenodo_publication import PACKAGE_FILE_NAMES

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "confirmatory-c0-evidence-release.yml"
IMAGE_WORKFLOW = ROOT / ".github" / "workflows" / "confirmatory-image.yml"


def _workflow() -> tuple[str, dict[str, object]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return text, parsed


def _image_workflow() -> tuple[str, dict[str, object]]:
    text = IMAGE_WORKFLOW.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return text, parsed


def _image_evidence_extraction_script() -> str:
    text, _ = _workflow()
    match = re.search(
        r'/usr/bin/python3 - "\$SOURCE_ROOT/image-evidence\.zip" "\$evidence" <<\'PY\'\n'
        r"(?P<script>.*?)\n          PY",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    return textwrap.dedent(match.group("script"))


def _zip_member(name: str, payload: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    member = zipfile.ZipInfo(name)
    member.create_system = 3
    member.external_attr = (stat.S_IFREG | 0o600) << 16
    return member, payload


def test_workflow_has_one_operator_only_c0_admission_boundary() -> None:
    text, parsed = _workflow()
    assert parsed["name"] == "Publish immutable C0 evidence release"
    assert parsed["permissions"] == {
        "actions": "read",
        "attestations": "read",
        "contents": "write",
    }
    assert parsed["concurrency"] == {
        "group": "confirmatory-c0-evidence-release",
        "cancel-in-progress": False,
    }
    for check in (
        'test "$GITHUB_REPOSITORY" = "$REPOSITORY"',
        "test \"$GITHUB_ACTOR\" = 'mhdk1602'",
        "test \"$GITHUB_TRIGGERING_ACTOR\" = 'mhdk1602'",
        "test \"$GITHUB_RUN_ATTEMPT\" = '1'",
        'test "$GITHUB_REF" = "refs/tags/${C0_TAG}"',
        'test "$GITHUB_SHA" = "$C0_SHA"',
        'test "$GITHUB_WORKFLOW_SHA" = "$C0_SHA"',
        'test "$(gh api "repos/${GITHUB_REPOSITORY}" --jq .full_name)" = "$GITHUB_REPOSITORY"',
    ):
        assert check in text


@pytest.mark.parametrize("alias", ["safe//record.json", "safe/./record.json"])
def test_image_evidence_zip_rejects_canonical_path_aliases(
    tmp_path: Path,
    alias: str,
) -> None:
    archive = tmp_path / "evidence.zip"
    output = tmp_path / "output"
    output.mkdir()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for member, payload in (
            _zip_member("safe/record.json", b"first"),
            _zip_member(alias, b"second"),
        ):
            bundle.writestr(member, payload)

    result = subprocess.run(
        [sys.executable, "-", str(archive), str(output)],
        input=_image_evidence_extraction_script(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "image evidence ZIP contains an unsafe member" in result.stderr


def test_workflow_authenticates_the_exact_production_image_run_and_artifact() -> None:
    text, _ = _workflow()
    for check in (
        ".run_attempt == 1",
        '.event == "workflow_dispatch"',
        '.conclusion == "success"',
        ".head_sha == $sha",
        '.head_branch == "confirmatory-apparatus-c0"',
        '.actor.login == "mhdk1602"',
        '.triggering_actor.login == "mhdk1602"',
        '.path == ".github/workflows/confirmatory-image.yml"',
        'artifact_name="confirmatory-image-production-${C0_SHA}"',
        '($matches[0].digest | test("^sha256:[0-9a-f]{64}$"))',
        'printf \'%s  %s\\n\' "${artifact_digest#sha256:}" "$archive" | sha256sum -c -',
    ):
        assert check in text


def test_workflow_closes_zip_members_and_rechecks_both_checksum_layers() -> None:
    text, _ = _workflow()
    assert "stat.S_ISLNK(mode)" in text
    assert "stat.S_ISREG(mode)" in text
    assert "total > 2 * 1024 * 1024 * 1024" in text
    assert 'required = {"C0-ARTIFACT-SHA256SUMS", "PACKAGE-SHA256SUMS"}' in text
    assert "sha256sum -c PACKAGE-SHA256SUMS" in text
    assert "sha256sum -c C0-ARTIFACT-SHA256SUMS" in text
    assert "candidate-production-byte-identity.json" in text
    assert "oci-promotion-receipt.json" in text
    assert "fractal-c0-oci-promotion-v1" in text
    assert ".bootstrap_candidate_source_commit" in text
    assert ".build_context_tree_sha256" in text
    assert ".scientific_raw_index_equal == true" in text
    assert ".scientific_anonymous_observed_digest == .scientific_index_digest" in text
    assert ".release_anonymous_readback_equal == true" in text
    assert "provider-rehearsal-production-gate.json" in text
    assert "fractal-c0-provider-rehearsal-gate-v2" in text
    assert "fractal-provider-rehearsal-aggregate-v2" in text
    assert ".candidate_bootstrap_closure_sha256" in text
    assert ".candidate_image_closure_file_sha256" in text
    assert ".candidate_image_source_commit" in text
    assert ".candidate_python_package_source_tree" in text
    assert ".workflow_python_package_source_tree" in text
    assert ".host_python_launcher_sha256" in text
    assert ".workflow_python_launcher_sha256" in text
    assert (
        ".candidate_python_package_source_tree\n"
        "                  == .workflow_python_package_source_tree"
    ) in text
    assert (
        ".host_python_launcher_sha256\n                  == .workflow_python_launcher_sha256"
    ) in text
    assert "provider-rehearsal-receipt.json" in text
    assert "rehearsal-attestation-verification.json" in text
    assert ".provider_rehearsal_receipt_sha256" in text


def test_archive_is_deterministic_and_bound_to_c0() -> None:
    text, _ = _workflow()
    for token in (
        "--sort=name",
        "--format=pax",
        "--pax-option=delete=atime,delete=ctime",
        '--mtime="@${source_epoch}"',
        "--owner=0 --group=0 --numeric-owner",
        "gzip -n -9",
        'asset_name="fractal-ann-diagnostics-c0-evidence-${C0_SHA}.tar.gz"',
    ):
        assert token in text


def test_release_transition_is_draft_then_assets_then_publish_then_verify() -> None:
    text, _ = _workflow()
    ordered = (
        "gh api --method POST",
        'printf \'release_id=%s\\n\' "$release_id" >> "$GITHUB_OUTPUT"',
        'gh release upload "$C0_TAG" "$ASSET"',
        'gh release upload "$C0_TAG" "$CHECKSUM"',
        'gh release edit "$C0_TAG" --repo "$REPOSITORY" --draft=false',
        'gh release verify "$C0_TAG"',
        'gh release verify-asset "$C0_TAG" "$ASSET"',
    )
    offsets: list[int] = []
    cursor = 0
    for token in ordered:
        cursor = text.index(token, cursor)
        offsets.append(cursor)
    assert offsets == sorted(offsets)
    assert '"/repos/${REPOSITORY}/immutable-releases"' in text
    assert "jq -e '.enabled == true'" in text
    assert ".draft == true" in text
    assert ".immutable == false" in text
    assert ".draft == false" in text
    assert ".immutable == true" in text
    assert text.count("([.assets[].name] | sort) == ([$asset, $checksum] | sort)") == 3
    assert "Re-read the now-immutable release tag target" in text
    assert 'peeled_ref="${base_ref}^{}"' in text
    assert 'test "$target" = "$C0_SHA"' in text
    assert "Delete an unpublished draft after failure" in text
    assert "needs.publish.result == 'failure'" in text


def test_exact_immutable_release_is_a_read_only_recovery_path() -> None:
    text, parsed = _workflow()
    steps = parsed["jobs"]["publish"]["steps"]
    by_name = {step["name"]: step for step in steps}

    state = by_name["Admit an absent release or recover the exact immutable release"]
    assert state["id"] == "release_state"
    assert "https://api.github.com/repos/${REPOSITORY}/releases/tags/${C0_TAG}" in state["run"]
    assert ".target_commitish == $target" in state["run"]
    assert ".immutable == true" in state["run"]
    assert '[[ "$http_status" == 200 ]]' in state["run"]
    assert '[[ "$http_status" == 404 ]]' in state["run"]
    assert "mode=recover" in state["run"]
    assert "mode=resume" in state["run"]
    assert "mode=create" in state["run"]

    draft = by_name["Create the run-owned draft through one typed API response"]
    attach = by_name["Attach the exact missing assets to the mutable draft"]
    publish = by_name["Publish the newly completed immutable release"]
    readback = by_name["Read and verify the exact immutable release"]
    binding = by_name["Materialize the C1-ready immutable-release binding"]
    retain = by_name["Retain the C1-ready release binding"]
    assert draft["if"] == "steps.release_state.outputs.mode == 'create'"
    assert attach["if"] == "steps.release_state.outputs.mode != 'recover'"
    assert publish["if"] == "steps.release_state.outputs.mode != 'recover'"
    assert "if" not in readback
    assert "if" not in binding
    assert "if" not in retain
    assert "the one C0 evidence release already exists" not in text


def test_lost_create_or_upload_response_has_a_closed_draft_resume_path() -> None:
    text, parsed = _workflow()
    steps = parsed["jobs"]["publish"]["steps"]
    by_name = {step["name"]: step for step in steps}
    state_run = by_name["Admit an absent release or recover the exact immutable release"]["run"]
    create_run = by_name["Create the run-owned draft through one typed API response"]["run"]
    attach_run = by_name["Attach the exact missing assets to the mutable draft"]["run"]

    assert ".draft == true" in state_run
    assert ".immutable == false" in state_run
    assert "fractal-c0-release-owner:[1-9][0-9]*:1:" in state_run
    assert "all(.assets[];" in state_run
    assert "length <= 2" in state_run
    assert '"/repos/${REPOSITORY}/releases" --input "$request"' in create_run
    assert create_run.index("release_id=") < create_run.index("GITHUB_OUTPUT")
    assert "all(.assets[];" in attach_run
    assert 'gh release upload "$C0_TAG" "$ASSET"' in attach_run
    assert 'gh release upload "$C0_TAG" "$CHECKSUM"' in attach_run


def test_failed_release_cleanup_is_protected_and_run_owned() -> None:
    text, parsed = _workflow()
    jobs = parsed["jobs"]
    assert jobs["publish"]["outputs"] == {"release_id": "${{ steps.draft.outputs.release_id }}"}
    cleanup = jobs["cleanup-draft"]
    assert cleanup["environment"] == "confirmatory"
    assert cleanup["permissions"] == {"contents": "write"}
    for token in (
        "needs.publish.outputs.release_id != ''",
        "fractal-c0-release-owner:${GITHUB_RUN_ID}:${GITHUB_RUN_ATTEMPT}:${C0_SHA}",
        "RELEASE_ID: ${{ needs.publish.outputs.release_id }}",
        "test \"$GITHUB_RUN_ATTEMPT\" = '1'",
        '[[ "$RELEASE_ID" =~ ^[1-9][0-9]*$ ]]',
        '"/repos/mhdk1602/fractal-ann-diagnostics/releases/${RELEASE_ID}"',
        ".id == $release_id",
        '.tag_name == "confirmatory-apparatus-c0"',
        ".target_commitish == $c0",
        "contains($marker)",
    ):
        assert token in text
    assert (
        "'/repos/mhdk1602/fractal-ann-diagnostics/releases/tags/confirmatory-apparatus-c0'"
    ) not in text


def test_release_requires_anonymous_asset_and_checksum_readback() -> None:
    text, _ = _workflow()
    assert text.count('env -i HOME="$readback"') == 2
    assert text.count("--proto '=https' --proto-redir '=https' --tlsv1.2") == 2
    assert 'test "$(stat -c %s "$readback/$ASSET_NAME")" = "$ASSET_SIZE"' in text
    assert 'sha256sum -c "$ASSET_NAME.sha256"' in text


def test_c1_binding_contains_release_asset_and_verification_closure() -> None:
    text, _ = _workflow()
    for field in (
        "asset_name",
        "asset_sha256",
        "asset_size",
        "asset_url",
        "checksum_asset_name",
        "checksum_asset_sha256",
        "checksum_asset_size",
        "checksum_asset_url",
        "immutable_release",
        "release_tag",
        "release_url",
        "repository",
        "target_commit",
        "verification_receipt",
        "verification_receipt_sha256",
        "release_attestation_output_sha256",
        "asset_attestation_output_sha256",
        "release_api_output_sha256",
        "release_tag_readback_sha256",
        "release_tag_target_commit",
        "release_tag_target_verified",
        "apparatus_evidence",
        "apparatus_evidence_sha256",
        "build_context_tree_sha256",
        "c0_commit",
        "candidate_bootstrap_closure_sha256",
        "candidate_image_closure_sha256",
        "candidate_image_run_id",
        "candidate_image_source_commit",
        "candidate_manifest_archive_member_path",
        "candidate_manifest_assembly_receipt_archive_member_path",
        "candidate_manifest_assembly_receipt_file_sha256",
        "candidate_manifest_file_sha256",
        "github_environment_control_receipt_file_sha256",
        "oci_promotion_receipt_sha256",
        "production_image_run_id",
        "provider_phase_plan_closure_sha256",
        "provider_rehearsal_gate_sha256",
        "provider_rehearsal_receipt_sha256",
        "provider_rehearsal_run_id",
        "rehearsal_attestation_verification_sha256",
        "rehearsal_manifest_sha256",
        "release_image_index_digest",
        "scientific_image_index_digest",
    ):
        assert re.search(rf"\b{field}\b", text)
    assert "fractal-c0-apparatus-evidence-closure-v5" in text
    assert "production_control_instantiation_receipt_file_sha256" in text
    assert "github-environment-control-receipt.json" in text
    assert "fractal-github-environment-control-v1" in text
    assert "fractal-c0-evidence-release-binding-v2" in text
    assert 'test "$(jq -r .current_c0_commit "$promotion")" = "$C0_SHA"' in text
    assert '"$(jq -r .github_sha "$candidate")"' in text
    assert '"$(jq -r .plan_closure_sha256 "$rehearsal")"' in text
    assert "validate_c0_evidence_release_binding" in text


def test_c0_evidence_is_not_an_extra_zenodo_member() -> None:
    assert len(PACKAGE_FILE_NAMES) == 27
    assert "c0-evidence-release-binding.json" not in PACKAGE_FILE_NAMES
    assert not any("c0-evidence" in name for name in PACKAGE_FILE_NAMES)
    text, _ = _workflow()
    assert "exact 27-file C1" in text
    assert "zenodo_publication" not in text


def test_all_actions_are_full_commit_pins() -> None:
    for workflow in (_workflow, _image_workflow):
        text, _ = workflow()
        pins = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", text)
        assert pins
        assert all(re.fullmatch(r"[0-9a-f]{40}", pin) for pin in pins)


def test_production_image_instantiates_and_retains_one_exact_post_a_control_tree() -> None:
    text, parsed = _image_workflow()
    jobs = parsed["jobs"]
    assert isinstance(jobs, dict)
    control = jobs["instantiate_production_controls"]
    assert control["environment"] == "confirmatory"
    assert control["runs-on"] == ["self-hosted", "macOS", "ARM64", "confirmatory-control"]
    publish = jobs["publish"]
    assert publish["needs"] == "instantiate_production_controls"
    for token in (
        "production_control_config_path",
        "production_control_output_root",
        "candidate_manifest_package_path",
        "candidate_manifest_relative_path",
        "candidate_manifest_assembly_receipt_relative_path",
        "instantiate-c0-controls",
        "--candidate-package",
        "production-control-instantiation-${{ inputs.c0_sha }}",
        "production-control-instantiation-readback",
        "c0-control-instantiation-receipt.json",
        '"approval_environment": "confirmatory"',
        "fractal-production-control-c0-instantiation-receipt-v4",
    ):
        assert token in text
    assert text.index("name: Upload the exact post-A control tree") < text.index(
        "name: Download the exact post-A control tree"
    )
    assert text.index("name: Authenticate and retain the exact post-A control tree") < text.index(
        "name: Close the retained C0 artifact checksum set"
    )


def test_production_image_captures_authenticated_environment_controls_before_closure() -> None:
    text, _ = _image_workflow()
    for token in (
        'test "$(gh api "repos/${GITHUB_REPOSITORY}" --jq .full_name)" = "$GITHUB_REPOSITORY"',
        "/environments?per_page=100",
        "/environments/confirmatory/deployment-branch-policies?per_page=100",
        "/environments/confirmatory-rehearsal/deployment-branch-policies?per_page=100",
        "github_environment_control verify",
        "github_environment_control readback",
        "github-environment-control-receipt.json",
        "fractal-github-environment-control-v1",
    ):
        assert token in text
    assert text.index("name: Authenticate and retain the GitHub approval controls") < text.index(
        "name: Close the retained C0 artifact checksum set"
    )
