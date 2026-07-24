from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fractal_ann_diagnostics.github_artifact_transport import (
    MAX_ARCHIVE_BYTES,
    ArchiveMember,
    ExecutionArtifactClaim,
    GitHubArtifactTransportError,
    GitHubHttpResponse,
    UrllibGitHubArtifactReadApi,
    derive_and_verify_fixed_claim_artifact,
    verify_execution_claim,
)

REPOSITORY = "mhdk1602/fractal-ann-diagnostics"
SHA = "a" * 40
WORKFLOW = ".github/workflows/confirmatory-image.yml"
REF = "refs/tags/confirmatory-apparatus-c0"
SUITE_ATTEMPT_ID = "c" * 64
PROVIDER_WORKFLOW = ".github/workflows/confirmatory-online-execution.yml"


class FakeApi:
    def __init__(self, responses: dict[str, GitHubHttpResponse]) -> None:
        self.responses = responses
        self.locations: list[str] = []

    def get(self, location: str, *, accept: str) -> GitHubHttpResponse:
        self.locations.append(location)
        return self.responses[location]


def _zip(rows: dict[str, bytes], *, encrypted: bool = False, symlink: bool = False) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in rows.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            if encrypted:
                info.flag_bits |= 0x1
            if symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, body)
    return output.getvalue()


def _claim(archive: bytes, rows: dict[str, bytes]) -> ExecutionArtifactClaim:
    inventory = tuple(
        ArchiveMember(path=name, sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body))
        for name, body in sorted(rows.items(), key=lambda item: item[0].encode("utf-8"))
    )
    return ExecutionArtifactClaim(
        repository=REPOSITORY,
        repository_id=1_239_189_910,
        repository_node_id="R_kgDOSdyJlg",
        owner_id=9_646_005,
        owner_login="mhdk1602",
        workflow_path=WORKFLOW,
        workflow_id=303,
        workflow_ref=REF,
        run_id=404,
        head_sha=SHA,
        head_branch="confirmatory-apparatus-c0",
        actor_id=9646005,
        actor_login="mhdk1602",
        conclusion="success",
        artifact_id=505,
        artifact_node_id="MDg6QXJ0aWZhY3Q1MDU=",
        artifact_name="confirmatory-image-production-c0",
        artifact_digest=f"sha256:{hashlib.sha256(archive).hexdigest()}",
        artifact_size_bytes=len(archive),
        artifact_created_at="2026-07-17T12:00:00Z",
        artifact_expires_at="2026-10-15T12:00:00Z",
        inventory=inventory,
    )


def _json(value: object, status: int = 200) -> GitHubHttpResponse:
    return GitHubHttpResponse(status, {}, json.dumps(value).encode("utf-8"))


def _responses(claim: ExecutionArtifactClaim, archive: bytes) -> dict[str, GitHubHttpResponse]:
    artifact = {
        "id": claim.artifact_id,
        "node_id": claim.artifact_node_id,
        "name": claim.artifact_name,
        "digest": claim.artifact_digest,
        "size_in_bytes": claim.artifact_size_bytes,
        "created_at": claim.artifact_created_at,
        "expires_at": claim.artifact_expires_at,
        "expired": False,
        "archive_download_url": (
            f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{claim.artifact_id}/zip"
        ),
        "workflow_run": {
            "id": claim.run_id,
            "head_sha": claim.head_sha,
            "head_branch": claim.head_branch,
            "repository_id": claim.repository_id,
            "head_repository_id": claim.repository_id,
        },
    }
    run_endpoint = f"repos/{REPOSITORY}/actions/runs/{claim.run_id}/attempts/1"
    return {
        f"repos/{REPOSITORY}": _json(
            {
                "full_name": REPOSITORY,
                "id": claim.repository_id,
                "node_id": claim.repository_node_id,
                "private": False,
                "fork": False,
                "owner": {
                    "id": claim.owner_id,
                    "login": claim.owner_login,
                    "node_id": "MDQ6VXNlcjk2NDYwMDU=",
                },
            }
        ),
        f"repos/{REPOSITORY}/actions/workflows/{claim.workflow_id}": _json(
            {"id": claim.workflow_id, "path": claim.workflow_path}
        ),
        run_endpoint: _json(
            {
                "id": claim.run_id,
                "run_attempt": 1,
                "event": "workflow_dispatch",
                "workflow_id": claim.workflow_id,
                "head_sha": claim.head_sha,
                "head_branch": claim.head_branch,
                "status": "completed",
                "conclusion": claim.conclusion,
                "path": claim.workflow_path,
                "actor": {"id": claim.actor_id, "login": claim.actor_login},
                "triggering_actor": {"id": claim.actor_id, "login": claim.actor_login},
                "repository": {
                    "id": claim.repository_id,
                    "node_id": claim.repository_node_id,
                    "full_name": claim.repository,
                },
                "head_repository": {
                    "id": claim.repository_id,
                    "node_id": claim.repository_node_id,
                    "full_name": claim.repository,
                },
            }
        ),
        f"repos/{REPOSITORY}/git/ref/tags/confirmatory-apparatus-c0": _json(
            {"ref": REF, "object": {"type": "tag", "sha": "b" * 40}}
        ),
        f"repos/{REPOSITORY}/git/tags/{'b' * 40}": _json(
            {
                "tag": "confirmatory-apparatus-c0",
                "object": {"type": "commit", "sha": claim.head_sha},
                "tagger": {
                    "name": "mhdk1602",
                    "email": "mhdk1602@users.noreply.github.com",
                },
            }
        ),
        f"repos/{REPOSITORY}/actions/runs/{claim.run_id}/artifacts?per_page=100": _json(
            {"total_count": 1, "artifacts": [artifact]}
        ),
        f"repos/{REPOSITORY}/actions/artifacts/{claim.artifact_id}": _json(artifact),
        f"repos/{REPOSITORY}/actions/artifacts/{claim.artifact_id}/zip": GitHubHttpResponse(
            302, {"Location": "https://pipelines.actions.githubusercontent.com/signed"}, b""
        ),
        "https://pipelines.actions.githubusercontent.com/signed": GitHubHttpResponse(
            200, {}, archive
        ),
    }


def _provider_context(claim: ExecutionArtifactClaim) -> SimpleNamespace:
    return SimpleNamespace(
        phase="online",
        job="execute",
        repository=REPOSITORY,
        repository_id=1_239_189_910,
        repository_owner="mhdk1602",
        repository_owner_id=9_646_005,
        actor="mhdk1602",
        actor_id=9_646_005,
        triggering_actor="mhdk1602",
        workflow_path=PROVIDER_WORKFLOW,
        workflow_ref=f"{REPOSITORY}/{PROVIDER_WORKFLOW}@{REF}",
        workflow_sha=claim.head_sha,
        github_ref=REF,
        github_ref_name="confirmatory-apparatus-c0",
        github_ref_type="tag",
        github_ref_protected=True,
        github_sha=claim.head_sha,
        run_id=claim.run_id,
        run_attempt=1,
        event_name="workflow_dispatch",
        runner_environment="self-hosted",
        runner_os="macOS",
        runner_arch="ARM64",
    )


def _sha256sums(rows: list[tuple[str, str]]) -> bytes:
    return b"".join(f"{digest}  {name}\n".encode("utf-8") for name, digest in rows)


def _fixed_claim_fixture(
    archive_rows: dict[str, bytes], inventory: bytes
) -> tuple[bytes, ExecutionArtifactClaim, FakeApi, SimpleNamespace]:
    complete_rows = {**archive_rows, "claim-package.SHA256SUMS": inventory}
    archive = _zip(complete_rows)
    artifact_name = f"confirmatory-online-claim-{SUITE_ATTEMPT_ID}-404"
    claim = replace(
        _claim(archive, complete_rows),
        workflow_path=PROVIDER_WORKFLOW,
        artifact_name=artifact_name,
    )
    api = FakeApi(_responses(claim, archive))
    return archive, claim, api, _provider_context(claim)


def test_verifies_cross_bound_github_evidence_digest_and_closed_archive(tmp_path: Path) -> None:
    rows = {"evidence/receipt.json": b"{}", "manifest.sha256": b"a" * 64 + b"\n"}
    archive = _zip(rows)
    claim = _claim(archive, rows)
    api = FakeApi(_responses(claim, archive))

    receipt = verify_execution_claim(claim, api, destination=tmp_path / "artifact")

    assert receipt.artifact_id == claim.artifact_id
    assert receipt.run_attempt == 1
    assert receipt.archive_sha256 == claim.artifact_digest[7:]
    assert (tmp_path / "artifact" / "evidence" / "receipt.json").read_bytes() == b"{}"
    assert api.locations[-3:] == [
        f"repos/{REPOSITORY}/actions/artifacts/{claim.artifact_id}/zip",
        "https://pipelines.actions.githubusercontent.com/signed",
        f"repos/{REPOSITORY}/actions/artifacts/{claim.artifact_id}",
    ]


def test_rejects_paginated_or_duplicate_matching_run_artifact_inventory(tmp_path: Path) -> None:
    rows = {"receipt.json": b"{}"}
    archive = _zip(rows)
    claim = _claim(archive, rows)
    responses = _responses(claim, archive)
    endpoint = f"repos/{REPOSITORY}/actions/runs/{claim.run_id}/artifacts?per_page=100"
    listing = json.loads(responses[endpoint].body)
    listing["total_count"] = 101
    responses[endpoint] = _json(listing)
    with pytest.raises(GitHubArtifactTransportError, match="incomplete"):
        verify_execution_claim(claim, FakeApi(responses), destination=tmp_path / "page")

    responses = _responses(claim, archive)
    listing = json.loads(responses[endpoint].body)
    listing["total_count"] = 2
    listing["artifacts"].append(dict(listing["artifacts"][0]))
    responses[endpoint] = _json(listing)
    with pytest.raises(GitHubArtifactTransportError, match="singleton"):
        verify_execution_claim(claim, FakeApi(responses), destination=tmp_path / "duplicate")


@pytest.mark.parametrize(
    "field, value", [("expired", True), ("id", 777), ("digest", "sha256:" + "0" * 64)]
)
def test_rejects_expired_deleted_or_reuploaded_artifact_id(
    tmp_path: Path, field: str, value: object
) -> None:
    rows = {"receipt.json": b"{}"}
    archive = _zip(rows)
    claim = _claim(archive, rows)
    responses = _responses(claim, archive)
    artifact_endpoint = f"repos/{REPOSITORY}/actions/artifacts/{claim.artifact_id}"
    artifact = json.loads(responses[artifact_endpoint].body)
    artifact[field] = value
    responses[artifact_endpoint] = _json(artifact)
    if field == "expired":
        responses[f"repos/{REPOSITORY}/actions/artifacts/{claim.artifact_id}/zip"] = (
            GitHubHttpResponse(410, {}, b"")
        )
    with pytest.raises(GitHubArtifactTransportError, match="differs|deleted|expired"):
        verify_execution_claim(claim, FakeApi(responses), destination=tmp_path / field)


def test_rejects_digest_mismatch_and_workflow_or_run_substitution(tmp_path: Path) -> None:
    rows = {"receipt.json": b"{}"}
    archive = _zip(rows)
    claim = _claim(archive, rows)
    responses = _responses(claim, archive)
    responses["https://pipelines.actions.githubusercontent.com/signed"] = GitHubHttpResponse(
        200, {}, archive + b"x"
    )
    with pytest.raises(GitHubArtifactTransportError, match="digest"):
        verify_execution_claim(claim, FakeApi(responses), destination=tmp_path / "digest")

    responses = _responses(claim, archive)
    run_endpoint = f"repos/{REPOSITORY}/actions/runs/{claim.run_id}/attempts/1"
    run = json.loads(responses[run_endpoint].body)
    run["workflow_id"] = claim.workflow_id + 1
    responses[run_endpoint] = _json(run)
    with pytest.raises(GitHubArtifactTransportError, match="workflow_id"):
        verify_execution_claim(claim, FakeApi(responses), destination=tmp_path / "run")

    responses = _responses(claim, archive)
    workflow_endpoint = f"repos/{REPOSITORY}/actions/workflows/{claim.workflow_id}"
    workflow = json.loads(responses[workflow_endpoint].body)
    workflow["path"] = ".github/workflows/other.yml"
    responses[workflow_endpoint] = _json(workflow)
    with pytest.raises(GitHubArtifactTransportError, match="workflow path"):
        verify_execution_claim(claim, FakeApi(responses), destination=tmp_path / "workflow")


@pytest.mark.parametrize(
    "archive_rows, inventory_rows, message",
    [
        ({"../receipt.json": b"{}"}, {"receipt.json": b"{}"}, "archive root"),
        ({"receipt.json": b"{}", "extra.json": b"x"}, {"receipt.json": b"{}"}, "extra"),
    ],
)
def test_rejects_zip_ambiguity_and_substitution(
    tmp_path: Path, archive_rows: dict[str, bytes], inventory_rows: dict[str, bytes], message: str
) -> None:
    archive = _zip(archive_rows)
    # The upload-artifact digest must stay genuine so the ZIP checks are exercised.
    claim = _claim(archive, inventory_rows)
    with pytest.raises(GitHubArtifactTransportError, match=message):
        verify_execution_claim(
            claim, FakeApi(_responses(claim, archive)), destination=tmp_path / "zip"
        )


def test_rejects_case_or_unicode_aliases_in_the_closed_inventory() -> None:
    rows = {"Receipt.json": b"{}", "receipt.json": b"{}"}
    archive = _zip(rows)
    with pytest.raises(GitHubArtifactTransportError, match="case or Unicode"):
        _claim(archive, rows)


def test_rejects_zip_symlinks_and_claim_incompatible_attempts(tmp_path: Path) -> None:
    rows = {"receipt.json": b"target"}
    archive = _zip(rows, symlink=True)
    claim = _claim(archive, rows)
    with pytest.raises(GitHubArtifactTransportError, match="regular file"):
        verify_execution_claim(
            claim, FakeApi(_responses(claim, archive)), destination=tmp_path / "symlink"
        )
    with pytest.raises(GitHubArtifactTransportError, match="attempt 1"):
        replace(claim, run_attempt=2)


def test_rejects_compression_bomb_before_materialization(tmp_path: Path) -> None:
    body = b"x" * 500_000
    rows = {"receipt.json": body}
    archive = _zip(rows)
    claim = _claim(archive, rows)
    with pytest.raises(GitHubArtifactTransportError, match="compression ratio"):
        verify_execution_claim(
            claim, FakeApi(_responses(claim, archive)), destination=tmp_path / "bomb"
        )


def test_receipt_cannot_be_constructed_without_verifier_capability(tmp_path: Path) -> None:
    rows = {"receipt.json": b"{}"}
    archive = _zip(rows)
    claim = _claim(archive, rows)
    receipt = verify_execution_claim(
        claim, FakeApi(_responses(claim, archive)), destination=tmp_path / "ok"
    )
    with pytest.raises(GitHubArtifactTransportError, match="not minted"):
        replace(receipt, _capability=None)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("repository", "attacker/evidence", "fixed production repository"),
        ("workflow_path", ".github/workflows/attacker.yml", "not admitted"),
        ("conclusion", "failure", "successful completed"),
    ],
)
def test_claim_rejects_caller_selected_production_policy(
    field: str, value: object, message: str
) -> None:
    rows = {"receipt.json": b"{}"}
    archive = _zip(rows)
    with pytest.raises(GitHubArtifactTransportError, match=message):
        replace(_claim(archive, rows), **{field: value})


def test_rejects_missing_workflow_run_and_unsafe_destination(tmp_path: Path) -> None:
    rows = {"receipt.json": b"{}"}
    archive = _zip(rows)
    claim = _claim(archive, rows)
    responses = _responses(claim, archive)
    endpoint = f"repos/{REPOSITORY}/actions/runs/{claim.run_id}/artifacts?per_page=100"
    listing = json.loads(responses[endpoint].body)
    listing["artifacts"][0].pop("workflow_run")
    responses[endpoint] = _json(listing)
    with pytest.raises(GitHubArtifactTransportError, match="workflow_run is required"):
        verify_execution_claim(claim, FakeApi(responses), destination=tmp_path / "missing")

    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir()
    os.chmod(unsafe_parent, 0o777)
    with pytest.raises(GitHubArtifactTransportError, match="controlled directory"):
        verify_execution_claim(
            claim, FakeApi(_responses(claim, archive)), destination=unsafe_parent / "x"
        )


def test_rejects_oversize_content_length_before_reading() -> None:
    with pytest.raises(GitHubArtifactTransportError, match="byte limit"):
        UrllibGitHubArtifactReadApi._bounded_response(
            200, {"Content-Length": str(MAX_ARCHIVE_BYTES + 1)}, io.BytesIO(b"")
        )


def test_derives_fixed_claim_from_one_download_and_internal_inventory(tmp_path: Path) -> None:
    rows = {
        "claim-receipt.json": b'{"closed":true}\n',
        "state/1.state.json": b'{"state":"RUN_CLAIMED"}\n',
    }
    inventory = _sha256sums(
        [(name, hashlib.sha256(body).hexdigest()) for name, body in sorted(rows.items())]
    )
    _, claim, api, context = _fixed_claim_fixture(rows, inventory)

    receipt = derive_and_verify_fixed_claim_artifact(
        context,
        api,
        suite_attempt_id=SUITE_ATTEMPT_ID,
        artifact_id=claim.artifact_id,
        artifact_digest=claim.artifact_digest,
        expected_inventory_sha256=hashlib.sha256(inventory).hexdigest(),
        destination=tmp_path / "fixed-claim",
    )

    assert receipt.artifact_name == (f"confirmatory-online-claim-{SUITE_ATTEMPT_ID}-{claim.run_id}")
    assert [row.path for row in receipt.inventory] == [
        "claim-package.SHA256SUMS",
        "claim-receipt.json",
        "state/1.state.json",
    ]
    zip_endpoint = f"repos/{REPOSITORY}/actions/artifacts/{claim.artifact_id}/zip"
    assert api.locations.count(zip_endpoint) == 1
    assert api.locations.count("https://pipelines.actions.githubusercontent.com/signed") == 1
    assert (tmp_path / "fixed-claim" / "claim-package.SHA256SUMS").read_bytes() == inventory


@pytest.mark.parametrize(
    "rows, inventory, message",
    [
        (
            {"receipt.json": b"{}", "unlisted.json": b"x"},
            _sha256sums([("receipt.json", hashlib.sha256(b"{}").hexdigest())]),
            "unexpected",
        ),
        (
            {"receipt.json": b"{}"},
            _sha256sums(
                [
                    ("missing.json", hashlib.sha256(b"missing").hexdigest()),
                    ("receipt.json", hashlib.sha256(b"{}").hexdigest()),
                ]
            ),
            "missing",
        ),
        (
            {"receipt.json": b"{}"},
            _sha256sums([("receipt.json", "0" * 64)]),
            "differs from claim-package",
        ),
        (
            {"receipt.json": b"{}"},
            _sha256sums(
                [
                    ("../escape.json", hashlib.sha256(b"x").hexdigest()),
                    ("receipt.json", hashlib.sha256(b"{}").hexdigest()),
                ]
            ),
            "archive root",
        ),
        (
            {"receipt.json": b"{}"},
            _sha256sums(
                [
                    ("receipt.json", hashlib.sha256(b"{}").hexdigest()),
                    ("receipt.json", hashlib.sha256(b"{}").hexdigest()),
                ]
            ),
            "duplicate",
        ),
        (
            {"receipt.json": b"{}"},
            _sha256sums(
                [
                    (
                        "claim-package.SHA256SUMS",
                        hashlib.sha256(b"self").hexdigest(),
                    ),
                    ("receipt.json", hashlib.sha256(b"{}").hexdigest()),
                ]
            ),
            "self-referential",
        ),
    ],
)
def test_fixed_claim_rejects_open_or_ambiguous_internal_inventory(
    tmp_path: Path,
    rows: dict[str, bytes],
    inventory: bytes,
    message: str,
) -> None:
    _, claim, api, context = _fixed_claim_fixture(rows, inventory)
    with pytest.raises(GitHubArtifactTransportError, match=message):
        derive_and_verify_fixed_claim_artifact(
            context,
            api,
            suite_attempt_id=SUITE_ATTEMPT_ID,
            artifact_id=claim.artifact_id,
            artifact_digest=claim.artifact_digest,
            expected_inventory_sha256=hashlib.sha256(inventory).hexdigest(),
            destination=tmp_path / message.replace(" ", "-"),
        )


def test_fixed_claim_rejects_trusted_inventory_or_artifact_metadata_substitution(
    tmp_path: Path,
) -> None:
    rows = {"receipt.json": b"{}"}
    inventory = _sha256sums([("receipt.json", hashlib.sha256(rows["receipt.json"]).hexdigest())])
    _, claim, api, context = _fixed_claim_fixture(rows, inventory)
    with pytest.raises(GitHubArtifactTransportError, match="trusted inventory digest"):
        derive_and_verify_fixed_claim_artifact(
            context,
            api,
            suite_attempt_id=SUITE_ATTEMPT_ID,
            artifact_id=claim.artifact_id,
            artifact_digest=claim.artifact_digest,
            expected_inventory_sha256="0" * 64,
            destination=tmp_path / "inventory-substitution",
        )

    _, claim, api, context = _fixed_claim_fixture(rows, inventory)
    artifact_endpoint = f"repos/{REPOSITORY}/actions/artifacts/{claim.artifact_id}"
    artifact = json.loads(api.responses[artifact_endpoint].body)
    artifact["name"] = "attacker-selected-claim"
    api.responses[artifact_endpoint] = _json(artifact)
    with pytest.raises(GitHubArtifactTransportError, match="name differs"):
        derive_and_verify_fixed_claim_artifact(
            context,
            api,
            suite_attempt_id=SUITE_ATTEMPT_ID,
            artifact_id=claim.artifact_id,
            artifact_digest=claim.artifact_digest,
            expected_inventory_sha256=hashlib.sha256(inventory).hexdigest(),
            destination=tmp_path / "metadata-substitution",
        )


def test_fixed_claim_rejects_duplicate_zip_members_and_context_substitution(
    tmp_path: Path,
) -> None:
    body = b"{}"
    inventory = _sha256sums([("receipt.json", hashlib.sha256(body).hexdigest())])
    stream = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive_file:
            archive_file.writestr("receipt.json", body)
            archive_file.writestr("receipt.json", body)
            archive_file.writestr("claim-package.SHA256SUMS", inventory)
    archive = stream.getvalue()
    unique_rows = {"receipt.json": body, "claim-package.SHA256SUMS": inventory}
    claim = replace(
        _claim(archive, unique_rows),
        workflow_path=PROVIDER_WORKFLOW,
        artifact_name=f"confirmatory-online-claim-{SUITE_ATTEMPT_ID}-404",
    )
    context = _provider_context(claim)
    with pytest.raises(GitHubArtifactTransportError, match="duplicate"):
        derive_and_verify_fixed_claim_artifact(
            context,
            FakeApi(_responses(claim, archive)),
            suite_attempt_id=SUITE_ATTEMPT_ID,
            artifact_id=claim.artifact_id,
            artifact_digest=claim.artifact_digest,
            expected_inventory_sha256=hashlib.sha256(inventory).hexdigest(),
            destination=tmp_path / "duplicate-members",
        )

    context.phase = "analysis"
    with pytest.raises(GitHubArtifactTransportError, match="workflow_path differs"):
        derive_and_verify_fixed_claim_artifact(
            context,
            FakeApi(_responses(claim, archive)),
            suite_attempt_id=SUITE_ATTEMPT_ID,
            artifact_id=claim.artifact_id,
            artifact_digest=claim.artifact_digest,
            expected_inventory_sha256=hashlib.sha256(inventory).hexdigest(),
            destination=tmp_path / "context-substitution",
        )
