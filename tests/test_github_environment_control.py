from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from fractal_ann_diagnostics.github_environment_control import (
    ADMIN_BYPASS_REST_ATTESTATION,
    API_RESPONSE_ROLES,
    APPROVAL_CLASSIFICATION,
    REPOSITORY,
    REPOSITORY_API_URL,
    REVIEWER_LOGIN,
    REVIEWER_USER_ID,
    GitHubEnvironmentApiSnapshots,
    GitHubEnvironmentControlError,
    load_github_environment_control_receipt,
    main,
    verify_github_environment_controls,
    write_github_environment_control_receipt,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _environment(
    name: str,
    environment_id: int,
    *,
    protection_rules: list[object],
) -> dict[str, object]:
    marker_id = environment_id + 10_000
    return {
        "id": environment_id,
        "name": name,
        "url": f"{REPOSITORY_API_URL}/environments/{name}",
        "html_url": f"https://github.com/{REPOSITORY}/deployments/activity_log?environments_filter={name}",
        "protection_rules": [
            *protection_rules,
            {
                "id": marker_id,
                "node_id": f"branch-policy-node-{marker_id}",
                "type": "branch_policy",
            },
        ],
        "deployment_branch_policy": {
            "protected_branches": False,
            "custom_branch_policies": True,
        },
        # This field is deliberately not treated as REST attestation authority.
        "can_admins_bypass": False,
    }


def _policy(name: str, policy_id: int, policy_type: str) -> dict[str, object]:
    return {
        "id": policy_id,
        "node_id": f"deployment-policy-node-{policy_id}",
        "name": name,
        "type": policy_type,
    }


def _reviewer_rule(rule_id: int) -> dict[str, object]:
    return {
        "id": rule_id,
        "node_id": f"required-reviewers-node-{rule_id}",
        "type": "required_reviewers",
        "prevent_self_review": False,
        "reviewers": [
            {
                "type": "User",
                "reviewer": {
                    "id": REVIEWER_USER_ID,
                    "login": REVIEWER_LOGIN,
                    "node_id": "MDQ6VXNlcjE0MjA4NTQx",
                },
            }
        ],
    }


def _payloads() -> dict[str, dict[str, object]]:
    confirmatory = _environment(
        "confirmatory",
        101,
        protection_rules=[_reviewer_rule(701)],
    )
    rehearsal = _environment("confirmatory-rehearsal", 102, protection_rules=[])
    return {
        "environments-list": {
            "total_count": 2,
            "environments": [
                {
                    "id": confirmatory["id"],
                    "name": confirmatory["name"],
                    "url": confirmatory["url"],
                },
                {
                    "id": rehearsal["id"],
                    "name": rehearsal["name"],
                    "url": rehearsal["url"],
                },
            ],
        },
        "environment-confirmatory": confirmatory,
        "deployment-policies-confirmatory": {
            "total_count": 2,
            "branch_policies": [
                _policy("confirmatory-freeze-c1", 202, "tag"),
                _policy("confirmatory-apparatus-c0", 201, "tag"),
            ],
        },
        "environment-confirmatory-rehearsal": rehearsal,
        "deployment-policies-confirmatory-rehearsal": {
            "total_count": 1,
            "branch_policies": [_policy("c0-candidate/*", 203, "branch")],
        },
    }


def _write_snapshots(
    root: Path,
    payloads: dict[str, dict[str, object]],
    *,
    pretty: bool = False,
) -> GitHubEnvironmentApiSnapshots:
    paths: dict[str, Path] = {}
    for role in API_RESPONSE_ROLES:
        path = root / f"{role}.json"
        if pretty:
            encoded = json.dumps(payloads[role], indent=2, sort_keys=False).encode("utf-8") + b"\n"
        else:
            encoded = _canonical(payloads[role]) + b"\n"
        path.write_bytes(encoded)
        paths[role] = path
    return GitHubEnvironmentApiSnapshots(
        environments_list=paths["environments-list"],
        confirmatory_environment=paths["environment-confirmatory"],
        confirmatory_deployment_policies=paths["deployment-policies-confirmatory"],
        rehearsal_environment=paths["environment-confirmatory-rehearsal"],
        rehearsal_deployment_policies=paths["deployment-policies-confirmatory-rehearsal"],
    )


def _verify(root: Path, payloads: dict[str, dict[str, object]] | None = None):
    values = _payloads() if payloads is None else payloads
    return verify_github_environment_controls(_write_snapshots(root, values))


def _set_path(value: object, path: tuple[str | int, ...], replacement: object) -> None:
    cursor: Any = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


def test_verifies_exact_two_environment_contract_and_canonical_response_hashes(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    receipt = _verify(tmp_path, payloads)

    assert receipt.repository == REPOSITORY
    assert receipt.repository_api_url == REPOSITORY_API_URL
    assert receipt.reviewer_login == REVIEWER_LOGIN
    assert receipt.reviewer_user_id == REVIEWER_USER_ID
    assert receipt.approval_classification == APPROVAL_CLASSIFICATION
    assert receipt.independent_custody is False
    assert receipt.admin_bypass_rest_attestation == ADMIN_BYPASS_REST_ATTESTATION
    assert tuple(row.name for row in receipt.environments) == (
        "confirmatory",
        "confirmatory-rehearsal",
    )
    assert [(row.policy_type, row.name) for row in receipt.environments[0].deployment_policies] == [
        ("tag", "confirmatory-apparatus-c0"),
        ("tag", "confirmatory-freeze-c1"),
    ]
    assert [(row.policy_type, row.name) for row in receipt.environments[1].deployment_policies] == [
        ("branch", "c0-candidate/*")
    ]
    for digest in receipt.api_responses:
        canonical = _canonical(payloads[digest.role])
        assert digest.canonical_byte_count == len(canonical)
        assert digest.canonical_sha256 == hashlib.sha256(canonical).hexdigest()


def test_pretty_api_responses_have_the_same_canonical_receipt(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    pretty_root = tmp_path / "pretty"
    canonical_root.mkdir()
    pretty_root.mkdir()
    payloads = _payloads()
    canonical = verify_github_environment_controls(_write_snapshots(canonical_root, payloads))
    pretty = verify_github_environment_controls(
        _write_snapshots(pretty_root, payloads, pretty=True)
    )
    assert pretty == canonical
    assert pretty.canonical_bytes() == canonical.canonical_bytes()


@pytest.mark.parametrize(
    ("role", "path", "replacement", "message"),
    (
        ("environments-list", ("total_count",), 3, "exactly two"),
        (
            "environments-list",
            ("environments", 1, "name"),
            "unexpected",
            "names differ",
        ),
        (
            "environment-confirmatory",
            ("url",),
            "https://api.github.com/repos/other/repo/environments/confirmatory",
            "fixed repository",
        ),
        ("environment-confirmatory", ("id",), 999, "identity differs"),
    ),
)
def test_rejects_repository_and_environment_identity_mutations(
    tmp_path: Path,
    role: str,
    path: tuple[str | int, ...],
    replacement: object,
    message: str,
) -> None:
    payloads = _payloads()
    _set_path(payloads[role], path, replacement)
    with pytest.raises(GitHubEnvironmentControlError, match=message):
        _verify(tmp_path, payloads)


@pytest.mark.parametrize(
    ("role", "path", "replacement", "message"),
    (
        (
            "environment-confirmatory",
            ("protection_rules", 0, "prevent_self_review"),
            True,
            "permit sole-operator self-review",
        ),
        (
            "environment-confirmatory",
            ("protection_rules", 0, "reviewers", 0, "type"),
            "Team",
            "required reviewer differs",
        ),
        (
            "environment-confirmatory",
            ("protection_rules", 0, "reviewers", 0, "reviewer", "id"),
            7,
            "required reviewer differs",
        ),
        (
            "environment-confirmatory",
            ("protection_rules", 0, "reviewers", 0, "reviewer", "login"),
            "other",
            "required reviewer differs",
        ),
        (
            "environment-confirmatory-rehearsal",
            ("protection_rules",),
            [{"type": "wait_timer", "wait_timer": 1}],
            "unknown type",
        ),
    ),
)
def test_rejects_protection_rule_mutations(
    tmp_path: Path,
    role: str,
    path: tuple[str | int, ...],
    replacement: object,
    message: str,
) -> None:
    payloads = _payloads()
    _set_path(payloads[role], path, replacement)
    with pytest.raises(GitHubEnvironmentControlError, match=message):
        _verify(tmp_path, payloads)


def test_rejects_missing_or_multiple_confirmatory_reviewers(tmp_path: Path) -> None:
    for reviewers in ([], [{"type": "User", "reviewer": {"id": 1, "login": "x"}}] * 2):
        payloads = _payloads()
        _set_path(
            payloads["environment-confirmatory"],
            ("protection_rules", 0, "reviewers"),
            reviewers,
        )
        root = tmp_path / str(len(reviewers))
        root.mkdir()
        with pytest.raises(GitHubEnvironmentControlError, match="exactly one reviewer"):
            _verify(root, payloads)


@pytest.mark.parametrize(
    ("role", "path", "replacement", "message"),
    (
        (
            "environment-confirmatory",
            ("deployment_branch_policy", "protected_branches"),
            True,
            "explicit custom",
        ),
        (
            "environment-confirmatory-rehearsal",
            ("deployment_branch_policy", "custom_branch_policies"),
            False,
            "contradicts custom_branch_policies",
        ),
        (
            "deployment-policies-confirmatory",
            ("branch_policies", 0, "name"),
            "confirmatory-*",
            "deployment policies differ",
        ),
        (
            "deployment-policies-confirmatory",
            ("branch_policies", 0, "type"),
            "branch",
            "deployment policies differ",
        ),
        (
            "deployment-policies-confirmatory-rehearsal",
            ("branch_policies", 0, "node_id"),
            203,
            "canonical non-empty string",
        ),
        (
            "deployment-policies-confirmatory-rehearsal",
            ("total_count",),
            2,
            "count is incomplete",
        ),
    ),
)
def test_rejects_deployment_policy_mutations(
    tmp_path: Path,
    role: str,
    path: tuple[str | int, ...],
    replacement: object,
    message: str,
) -> None:
    payloads = _payloads()
    _set_path(payloads[role], path, replacement)
    with pytest.raises(GitHubEnvironmentControlError, match=message):
        _verify(tmp_path, payloads)


def test_rejects_extra_or_missing_deployment_policy(tmp_path: Path) -> None:
    for suffix, policies in (
        ("missing", [_policy("confirmatory-apparatus-c0", 201, "tag")]),
        (
            "extra",
            [
                _policy("confirmatory-apparatus-c0", 201, "tag"),
                _policy("confirmatory-freeze-c1", 202, "tag"),
                _policy("confirmatory-extra", 204, "tag"),
            ],
        ),
    ):
        payloads = _payloads()
        payloads["deployment-policies-confirmatory"] = {
            "total_count": len(policies),
            "branch_policies": policies,
        }
        root = tmp_path / suffix
        root.mkdir()
        with pytest.raises(GitHubEnvironmentControlError, match="deployment policies differ"):
            _verify(root, payloads)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "exactly one branch_policy marker"),
        ("duplicate", "exactly one branch_policy marker"),
        ("unknown-field", "closed schema"),
        ("noninteger-id", "positive integer"),
        ("empty-node", "canonical non-empty string"),
        ("unknown-type", "unknown type"),
        ("duplicate-id", "repeats a protection rule ID"),
        ("duplicate-node", "repeats a protection rule node_id"),
    ),
)
def test_branch_policy_marker_is_exact_and_unique(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    payloads = _payloads()
    rules = payloads["environment-confirmatory"]["protection_rules"]
    assert isinstance(rules, list)
    marker = rules[1]
    assert isinstance(marker, dict)
    reviewer = rules[0]
    assert isinstance(reviewer, dict)

    if mutation == "missing":
        rules.pop()
    elif mutation == "duplicate":
        rules.append(
            {
                "id": 10_102,
                "node_id": "branch-policy-node-10102",
                "type": "branch_policy",
            }
        )
    elif mutation == "unknown-field":
        marker["unexpected"] = True
    elif mutation == "noninteger-id":
        marker["id"] = True
    elif mutation == "empty-node":
        marker["node_id"] = ""
    elif mutation == "unknown-type":
        marker["type"] = "wait_timer"
    elif mutation == "duplicate-id":
        marker["id"] = reviewer["id"]
    else:
        marker["node_id"] = reviewer["node_id"]

    with pytest.raises(GitHubEnvironmentControlError, match=message):
        _verify(tmp_path, payloads)


@pytest.mark.parametrize(
    ("role", "rule_id", "message"),
    (
        (
            "environment-confirmatory",
            702,
            "exactly one required-reviewers rule",
        ),
        (
            "environment-confirmatory-rehearsal",
            703,
            "cannot contain a required-reviewers rule",
        ),
    ),
)
def test_required_reviewer_rule_partition_is_exact(
    tmp_path: Path,
    role: str,
    rule_id: int,
    message: str,
) -> None:
    payloads = _payloads()
    rules = payloads[role]["protection_rules"]
    assert isinstance(rules, list)
    rules.append(_reviewer_rule(rule_id))

    with pytest.raises(GitHubEnvironmentControlError, match=message):
        _verify(tmp_path, payloads)


@pytest.mark.parametrize(
    ("path", "message"),
    (
        (("protection_rules", 0, "unexpected"), "closed schema"),
        (("protection_rules", 0, "reviewers", 0, "unexpected"), "closed schema"),
    ),
)
def test_required_reviewer_transport_objects_reject_unknown_fields(
    tmp_path: Path,
    path: tuple[str | int, ...],
    message: str,
) -> None:
    payloads = _payloads()
    _set_path(payloads["environment-confirmatory"], path, True)
    with pytest.raises(GitHubEnvironmentControlError, match=message):
        _verify(tmp_path, payloads)


@pytest.mark.parametrize(
    ("role", "path", "replacement", "message"),
    (
        (
            "environment-confirmatory",
            ("deployment_branch_policy", "unexpected"),
            True,
            "closed schema",
        ),
        (
            "deployment-policies-confirmatory",
            ("unexpected",),
            True,
            "closed schema",
        ),
        (
            "deployment-policies-confirmatory",
            ("branch_policies", 0, "url"),
            "https://api.github.com/unregistered",
            "closed schema",
        ),
        (
            "deployment-policies-confirmatory",
            ("branch_policies", 1, "id"),
            202,
            "repeats a deployment policy ID",
        ),
        (
            "deployment-policies-confirmatory",
            ("branch_policies", 1, "node_id"),
            "deployment-policy-node-202",
            "repeats a deployment policy node_id",
        ),
    ),
)
def test_deployment_policy_transport_is_closed_and_unambiguous(
    tmp_path: Path,
    role: str,
    path: tuple[str | int, ...],
    replacement: object,
    message: str,
) -> None:
    payloads = _payloads()
    _set_path(payloads[role], path, replacement)
    with pytest.raises(GitHubEnvironmentControlError, match=message):
        _verify(tmp_path, payloads)


def test_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    snapshots = _write_snapshots(tmp_path, _payloads())
    snapshots.confirmatory_environment.write_bytes(b'{"id":101,"id":102}\n')
    with pytest.raises(GitHubEnvironmentControlError, match="repeats key"):
        verify_github_environment_controls(snapshots)

    snapshots.confirmatory_environment.write_bytes(b'{"id":NaN}\n')
    with pytest.raises(GitHubEnvironmentControlError, match="non-finite"):
        verify_github_environment_controls(snapshots)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="filesystem links unavailable")
def test_rejects_linked_api_snapshots(tmp_path: Path) -> None:
    payloads = _payloads()
    snapshots = _write_snapshots(tmp_path, payloads)
    original = snapshots.environments_list
    linked = tmp_path / "linked.json"
    original.rename(linked)
    original.symlink_to(linked)
    with pytest.raises(GitHubEnvironmentControlError, match="without following links"):
        verify_github_environment_controls(snapshots)

    original.unlink()
    os.link(linked, original)
    with pytest.raises(GitHubEnvironmentControlError, match="singly linked"):
        verify_github_environment_controls(snapshots)


def test_atomic_receipt_publication_and_canonical_readback(tmp_path: Path) -> None:
    snapshots_root = tmp_path / "snapshots"
    receipt_root = tmp_path / "receipts"
    snapshots_root.mkdir(mode=0o700)
    receipt_root.mkdir(mode=0o700)
    receipt = verify_github_environment_controls(_write_snapshots(snapshots_root, _payloads()))
    target = receipt_root / "github-environment-control.json"
    write_github_environment_control_receipt(receipt, target)

    assert target.read_bytes() == receipt.canonical_file_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert receipt.file_sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert load_github_environment_control_receipt(target) == receipt
    assert not list(receipt_root.glob(".*.tmp"))

    before = target.read_bytes()
    with pytest.raises(GitHubEnvironmentControlError, match="already exists"):
        write_github_environment_control_receipt(receipt, target)
    assert target.read_bytes() == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("pretty", "not canonical"),
        ("extra", "closed schema"),
        ("independence", "cannot claim independent custody"),
        ("admin-bypass", "overstates REST admin-bypass"),
        ("second-lf", "not canonical"),
    ),
)
def test_receipt_loader_rejects_noncanonical_or_overstated_receipts(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    source_root = tmp_path / "snapshots"
    source_root.mkdir()
    receipt = verify_github_environment_controls(_write_snapshots(source_root, _payloads()))
    payload = receipt.to_dict()
    if mutation == "pretty":
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    elif mutation == "extra":
        payload["unexpected"] = True
        encoded = _canonical(payload) + b"\n"
    elif mutation == "independence":
        payload["independent_custody"] = True
        encoded = _canonical(payload) + b"\n"
    elif mutation == "admin-bypass":
        payload["admin_bypass_rest_attestation"] = "admins-cannot-bypass"
        encoded = _canonical(payload) + b"\n"
    else:
        encoded = receipt.canonical_file_bytes() + b"\n"
    path = tmp_path / f"{mutation}.json"
    path.write_bytes(encoded)
    with pytest.raises(GitHubEnvironmentControlError, match=message):
        load_github_environment_control_receipt(path)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="filesystem links unavailable")
def test_receipt_loader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    snapshots_root = tmp_path / "snapshots"
    snapshots_root.mkdir()
    receipt = verify_github_environment_controls(_write_snapshots(snapshots_root, _payloads()))
    source = tmp_path / "source.json"
    source.write_bytes(receipt.canonical_file_bytes())
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(source)
    with pytest.raises(GitHubEnvironmentControlError, match="without following links"):
        load_github_environment_control_receipt(symlink)

    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(GitHubEnvironmentControlError, match="singly linked"):
        load_github_environment_control_receipt(hardlink)


def test_cli_has_only_offline_verify_and_readback_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshots_root = tmp_path / "snapshots"
    receipt_root = tmp_path / "receipts"
    snapshots_root.mkdir()
    receipt_root.mkdir()
    snapshots = _write_snapshots(snapshots_root, _payloads())
    target = receipt_root / "receipt.json"
    assert (
        main(
            [
                "verify",
                "--environments-list",
                str(snapshots.environments_list),
                "--confirmatory-environment",
                str(snapshots.confirmatory_environment),
                "--confirmatory-deployment-policies",
                str(snapshots.confirmatory_deployment_policies),
                "--rehearsal-environment",
                str(snapshots.rehearsal_environment),
                "--rehearsal-deployment-policies",
                str(snapshots.rehearsal_deployment_policies),
                "--receipt",
                str(target),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "receipt file sha256:" in output

    assert main(["readback", "--receipt", str(target)]) == 0
    readback = capsys.readouterr().out
    assert json.loads(readback) == load_github_environment_control_receipt(target).to_dict()

    with pytest.raises(SystemExit):
        main(["fetch"])
