from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "confirmatory-registration-attestation.yml"


def _workflow() -> tuple[str, dict[str, Any]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return text, parsed


def _steps() -> tuple[str, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    text, parsed = _workflow()
    steps = parsed["jobs"]["attest"]["steps"]
    assert isinstance(steps, list)
    by_name = {step["name"]: step for step in steps}
    assert len(by_name) == len(steps)
    return text, steps, by_name


def test_workflow_installs_exact_official_gh_release_without_ambient_auth() -> None:
    _, _, by_name = _steps()
    _, parsed = _workflow()
    assert parsed["env"] == {
        "C0_EVIDENCE_REPOSITORY": "mhdk1602/fractal-ann-diagnostics",
        "C0_EVIDENCE_TAG": "confirmatory-apparatus-c0",
        "PINNED_GH_ARCHIVE_SHA256": (
            "83d5c2ccad5498f58bf6368acb1ab32588cf43ab3a4b1c301bf36328b1c8bd60"
        ),
        "PINNED_GH_BINARY_SHA256": (
            "56b8bbbb27b066ecb33dbef9a256dc9d1314adaeff0908a752feba6c34053b40"
        ),
        "PINNED_GH_CHECKSUMS_SHA256": (
            "fc046371efa250e2875208341a786a35a01717d5eebec6903e199a9b8a3f3565"
        ),
        "PINNED_GH_VERSION": "2.96.0",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONPATH": "src",
    }
    assert parsed["jobs"]["attest"]["timeout-minutes"] == 60

    step = by_name["Install and verify the official pinned GitHub CLI"]
    assert step["id"] == "pinned_gh"
    run = step["run"]
    for token in (
        "https://github.com/cli/cli/releases/download/v${PINNED_GH_VERSION}",
        'archive_name="gh_${PINNED_GH_VERSION}_linux_amd64.tar.gz"',
        'checksums_name="gh_${PINNED_GH_VERSION}_checksums.txt"',
        "env -i HOME=\"$anonymous_home\" PATH='/usr/bin:/bin'",
        "--proto '=https'",
        "--proto-redir '=https'",
        "--tlsv1.2",
        "--retry 5",
        "--retry-all-errors",
        '--retry-max-time "$budget"',
        "--connect-timeout 15",
        '--max-time "$budget"',
        '--max-filesize "$maximum"',
        'printf \'%s  %s\\n\' "$PINNED_GH_CHECKSUMS_SHA256" "$checksums"',
        '"${PINNED_GH_ARCHIVE_SHA256}  ${archive_name}"',
        'printf \'%s  %s\\n\' "$PINNED_GH_ARCHIVE_SHA256" "$archive"',
        'member="gh_${PINNED_GH_VERSION}_linux_amd64/bin/gh"',
        "--no-same-owner",
        "--no-same-permissions",
        'printf \'%s  %s\\n\' "$PINNED_GH_BINARY_SHA256" "$gh_path"',
        "printf 'gh version %s (2026-07-02)\\n' \"$PINNED_GH_VERSION\"",
        'cmp -- "${tool_root}/expected-gh-version.txt" "$gh_version"',
        'chmod 0400 "$archive" "$checksums" "$gh_version"',
        'printf \'%s\\n\' "$tool_root" >> "$GITHUB_PATH"',
        '"$checksums" \\\n  1048576 \\\n  180',
        '"$archive" \\\n  67108864 \\\n  180',
    ):
        assert token in run
    assert "GH_TOKEN" not in step.get("env", {})
    assert "gh auth" not in run
    assert "sudo " not in run


def test_workflow_acquires_canonical_and_anonymous_fresh_c0_evidence() -> None:
    _, _, by_name = _steps()
    step = by_name["Acquire the fresh public C0 release evidence"]
    assert step["env"] == {
        "EVIDENCE_DIR": "${{ steps.pinned_gh.outputs.evidence_dir }}",
        "GH_HOST": "github.com",
        "GH_PATH": "${{ steps.pinned_gh.outputs.gh_path }}",
        "GH_PROMPT_DISABLED": "1",
        "GH_TOKEN": "${{ github.token }}",
    }
    run = step["run"]
    for field in (
        "repository",
        "release_tag",
        "asset_name",
        "asset_sha256",
        "asset_size",
        "asset_url",
        "checksum_asset_name",
        "checksum_asset_sha256",
        "checksum_asset_size",
        "checksum_asset_url",
    ):
        assert f".sealed_execution.c0_evidence_release.{field}" in run
    for token in (
        'test "$repository" = "$C0_EVIDENCE_REPOSITORY"',
        'test "$release_tag" = "$C0_EVIDENCE_TAG"',
        'test "$checksum_name" = "${asset_name}.sha256"',
        'release_base="https://github.com/${repository}/releases/download/${release_tag}"',
        'test "$asset_url" = "${release_base}/${asset_name}"',
        'test "$checksum_url" = "${release_base}/${checksum_name}"',
        "timeout --signal=TERM --kill-after=10s 60s",
        "-H 'X-GitHub-Api-Version: 2026-03-10'",
        '"/repos/${repository}/releases/tags/${release_tag}"',
        '| jq -S -c . > "$release_api"',
        "env -u GH_TOKEN -u GITHUB_TOKEN",
        "GIT_CONFIG_NOSYSTEM=1",
        "GIT_CONFIG_GLOBAL=/dev/null",
        "GIT_TERMINAL_PROMPT=0",
        "-c credential.helper=",
        "-c http.followRedirects=false",
        "ls-remote --exit-code --tags",
        '"$base_ref" "$peeled_ref"',
        '| LC_ALL=C sort > "$tag_readback"',
        "env -i HOME=\"$anonymous_home\" PATH='/usr/bin:/bin'",
        "--proto '=https'",
        "--proto-redir '=https'",
        "--tlsv1.2",
        "--retry 5",
        "--retry-all-errors",
        '--retry-max-time "$budget"',
        '--max-time "$budget"',
        '--max-filesize "$maximum"',
        'download_anonymous "$asset_url" "$archive" "$asset_size" 900',
        'download_anonymous "$checksum_url" "$checksum" "$checksum_size" 180',
        'test "$(stat -c %s "$archive")" = "$asset_size"',
        'test "$(sha256sum "$archive" | cut -d \' \' -f 1)" = "$asset_sha256"',
        'printf \'%s  %s\\n\' "$asset_sha256" "$asset_name"',
        '"$GH_PATH" release verify "$release_tag"',
        '"$GH_PATH" release verify-asset "$release_tag" "$archive"',
        "--format json",
        'jq -e \'keys == ["attestation", "verificationResult"]\'',
        'chmod 0400 \\\n  "$release_api"',
    ):
        assert token in run
    assert run.count("for attempt in 1 2 3 4; do") == 3
    assert "${{ github.token }}" not in run
    assert "--with-token" not in run
    assert "Authorization:" not in run


def test_offline_c0_receipt_precedes_and_enters_c1_admission() -> None:
    text, steps, by_name = _steps()
    names = [step["name"] for step in steps]
    close_name = "Close the retained public C0 evidence"
    prepare_name = "Admit only the witnessed freeze child of C0 at the fixed C1 tag"
    assert names.index("Install and verify the official pinned GitHub CLI") < names.index(
        "Acquire the fresh public C0 release evidence"
    )
    assert names.index("Acquire the fresh public C0 release evidence") < names.index(close_name)
    assert names.index(close_name) < names.index(prepare_name)

    close_run = by_name[close_name]["run"]
    for token in (
        'output_dir="${RUNNER_TEMP}/c0-public-verification-output"',
        'receipt="${RUNNER_TEMP}/c0-public-verification.json"',
        'staged_receipt="${output_dir}/c0-public-verification.json"',
        'test ! -e "$output_dir"',
        'install -d -m 0700 "$output_dir"',
        "python -m fractal_ann_diagnostics.c0_public_verification",
        '--manifest "$GITHUB_WORKSPACE/research/study-manifest.json"',
        '--gh-version "${EVIDENCE_DIR}/gh-version.txt"',
        '--release-api "${EVIDENCE_DIR}/published-release-api.json"',
        '--tag-ls-remote "${EVIDENCE_DIR}/published-tag-ls-remote.txt"',
        '--release-verification "${EVIDENCE_DIR}/release-verification.json"',
        '--asset-verification "${EVIDENCE_DIR}/asset-verification.json"',
        '--archive "${EVIDENCE_DIR}/$(jq -er',
        '--checksum "${EVIDENCE_DIR}/$(jq -er',
        '--output "$staged_receipt"',
        'test "$(stat -c %a "$staged_receipt")" = 600',
        'test "$(stat -c %h "$staged_receipt")" = 1',
        'mv --no-clobber -- "$staged_receipt" "$receipt"',
        'test ! -e "$staged_receipt"',
        'test "$(stat -c %a "$receipt")" = 600',
        'test "$(stat -c %h "$receipt")" = 1',
    ):
        assert token in close_run

    prepare_run = by_name[prepare_name]["run"]
    assert '--c0-public-verification "${RUNNER_TEMP}/c0-public-verification.json"' in prepare_run
    assert prepare_run.index("--c0-public-verification") < prepare_run.index("--output-dir")
    assert (
        "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/"
        "prospective-c1-registration/v2"
    ) in text
    assert (
        "https://mhdk1602.github.io/fractal-ann-diagnostics/attestations/"
        "prospective-c1-registry-record/v2"
    ) in text
    assert "prospective-c1-registration/v1" not in text
    assert "prospective-c1-registry-record/v1" not in text


def test_package_retains_receipt_and_pinned_version_before_closed_checksums() -> None:
    _, _, by_name = _steps()
    run = by_name["Assemble the retained registration package"]["run"]
    receipt_copy = (
        'install -m 0600 -- "${RUNNER_TEMP}/c0-public-verification.json" \\\n'
        '  "$package/c0-public-verification.json"'
    )
    version_copy = (
        "install -m 0600 -- \\\n"
        '  "${RUNNER_TEMP}/c0-public-verification-inputs/gh-version.txt" \\\n'
        '  "$package/gh-version.txt"'
    )
    assert receipt_copy in run
    assert version_copy in run
    assert "gh --version" not in run
    checksum = "find . -maxdepth 1 -type f ! -name SHA256SUMS -print0"
    assert checksum in run
    assert run.index(receipt_copy) < run.index(checksum)
    assert run.index(version_copy) < run.index(checksum)
    assert 'sha256sum --check "$checksum_tmp"' in run
    assert 'install -m 0600 -- "$checksum_tmp" "$package/SHA256SUMS"' in run


def test_registration_workflow_actions_are_full_commit_pins() -> None:
    text, _ = _workflow()
    pins = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", text)
    assert pins
    assert all(re.fullmatch(r"[0-9a-f]{40}", pin) for pin in pins)
