from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from operators import design_seed_commitment as operator

ROUND_1_SIGNATURE = (
    "b55e7cb2d5c613ee0b2e28d6750aabbb78c39dcc96bd9d38c2c2e12198df955"
    "71de8e8e402a0cc48871c7089a2b3af4b"
)
ROUND_1_RANDOMNESS = "1466a6cd24e327188770752f6134001c64d6efcc590ccc26b721611ad96f165a"
ROUND_1_BYTES = (
    b'{"round":1,"randomness":"1466a6cd24e327188770752f6134001c64d6efcc590ccc26'
    b'b721611ad96f165a","signature":"b55e7cb2d5c613ee0b2e28d6750aabbb78c39dcc96bd9d38'
    b'c2c2e12198df95571de8e8e402a0cc48871c7089a2b3af4b"}'
)


def _digest(value: bytes | str) -> str:
    encoded = value.encode("ascii") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _write(path: Path, encoded: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_bytes(encoded)
    os.chmod(path, mode)


def _rewrite_json(path: Path, mutate: Any) -> None:
    value = json.loads(path.read_text(encoding="ascii"))
    mutate(value)
    os.chmod(path, 0o600)
    path.write_bytes(_canonical(value))
    os.chmod(path, 0o400)


def _fake_verifier(**kwargs: object) -> None:
    subject_path = kwargs["subject_path"]
    assert isinstance(subject_path, Path)
    assert subject_path.name.startswith("design-seed-commitment-")
    assert kwargs["repository"] == operator.REPOSITORY
    assert kwargs["workflow_sha"] == "a" * 40
    assert kwargs["git_ref"] == operator.ATTESTATION_GIT_REF


def _github_timestamp(seconds: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _fake_remote_verifier(**kwargs: object) -> dict[str, bytes]:
    commitment = kwargs["commitment"]
    predicate = kwargs["predicate"]
    integrated_at = kwargs["rekor_integrated_at_utc"]
    assert isinstance(commitment, operator.DesignSeedCommitment)
    assert isinstance(predicate, dict)
    assert isinstance(integrated_at, str)
    published = predicate["release_published_at_utc"]
    assert isinstance(published, str)
    run = {
        "actor": operator.OWNER_LOGIN,
        "conclusion": "success",
        "event": operator.EVENT,
        "head_branch": operator._ref_name(operator.ATTESTATION_GIT_REF),
        "head_repository": operator.REPOSITORY,
        "head_sha": commitment.attestation_workflow_sha,
        "id": predicate["run_id"],
        "path": operator.ATTESTATION_WORKFLOW,
        "repository": operator.REPOSITORY,
        "run_attempt": 1,
        "run_started_at": published,
        "status": "completed",
        "triggering_actor": operator.OWNER_LOGIN,
    }
    release = {
        "assets_count": 0,
        "author": operator.RELEASE_AUTHOR,
        "created_at": published,
        "draft": False,
        "id": predicate["release_id"],
        "immutable": True,
        "name": predicate["release_name"],
        "prerelease": False,
        "published_at": published,
        "tag_name": predicate["release_tag"],
        "target_commitish": commitment.attestation_workflow_sha,
    }
    tag = {
        "object_sha": commitment.attestation_workflow_sha,
        "object_type": "commit",
        "ref": f"refs/tags/{predicate['release_tag']}",
    }
    return {
        "actions_run": operator._projection_bytes(run),
        "release": operator._projection_bytes(release),
        "release_tag": operator._projection_bytes(tag),
    }


def _predicate(
    commitment: operator.DesignSeedCommitment,
    *,
    integrated_time: int,
    **changes: object,
) -> dict[str, object]:
    release_tag = operator._release_tag(commitment.scope_sha256)
    values: dict[str, object] = {
        "actor": operator.OWNER_LOGIN,
        "commitment_sha256": commitment.commitment_sha256,
        "event": operator.EVENT,
        "git_ref": operator.ATTESTATION_GIT_REF,
        "release_id": 876543,
        "release_name": release_tag,
        "release_published_at_utc": _github_timestamp(integrated_time - 60),
        "release_tag": release_tag,
        "repository": operator.REPOSITORY,
        "run_attempt": 1,
        "run_id": 987654,
        "schema_version": operator.ATTESTATION_PREDICATE_SCHEMA,
        "scope_sha256": commitment.scope_sha256,
        "source_p": operator.SOURCE_P,
        "source_tree": operator.SOURCE_TREE,
        "triggering_actor": operator.OWNER_LOGIN,
        "workflow": operator.ATTESTATION_WORKFLOW,
        "workflow_ref": (
            f"{operator.REPOSITORY}/{operator.ATTESTATION_WORKFLOW}@{operator.ATTESTATION_GIT_REF}"
        ),
        "workflow_sha": "a" * 40,
    }
    values.update(changes)
    return values


def _bundle(
    commitment: operator.DesignSeedCommitment,
    *,
    integrated_time: int,
    predicate_changes: dict[str, object] | None = None,
    subject_digest: str | None = None,
    signature_token: bytes = b"signed-entry-timestamp",
) -> bytes:
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicate": _predicate(
            commitment,
            integrated_time=integrated_time,
            **(predicate_changes or {}),
        ),
        "predicateType": operator.ATTESTATION_PREDICATE_TYPE,
        "subject": [
            {
                "digest": {
                    "sha256": commitment.commitment_sha256
                    if subject_digest is None
                    else subject_digest
                },
                "name": commitment.attestation_subject_name,
            }
        ],
    }
    bundle = {
        "dsseEnvelope": {
            "payload": base64.b64encode(_canonical(statement)[:-1]).decode("ascii"),
            "payloadType": "application/vnd.in-toto+json",
            "signatures": [{"keyid": "", "sig": base64.b64encode(b"signature").decode()}],
        },
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "verificationMaterial": {
            "tlogEntries": [
                {
                    "canonicalizedBody": base64.b64encode(b"rekor-body").decode("ascii"),
                    "inclusionPromise": {
                        "signedEntryTimestamp": base64.b64encode(signature_token).decode("ascii")
                    },
                    "integratedTime": integrated_time,
                    "logId": {"keyId": base64.b64encode(b"k" * 32).decode("ascii")},
                    "logIndex": 42,
                }
            ]
        },
    }
    return _canonical(bundle)[:-1]


def _base(tmp_path: Path, *, suffix: str = "") -> tuple[Path, operator.DesignSeedCommitment]:
    output = tmp_path / f"controls{suffix}"
    output.mkdir(mode=0o700)
    request_path, _ = operator.build_design_seed_request(
        staged_inventory_sha256=_digest(f"inventory{suffix}"),
        partition_audit_file_sha256=_digest(f"audit{suffix}"),
        phase1_view_receipt_sha256=_digest(f"phase1{suffix}"),
        selection_receipt_sha256=_digest(f"selection{suffix}"),
        attestation_workflow=operator.ATTESTATION_WORKFLOW,
        attestation_workflow_sha="a" * 40,
        attestation_git_ref=operator.ATTESTATION_GIT_REF,
        output_directory=output,
    )
    commitment_path, commitment = operator.build_design_seed_commitment(
        request_path, output_directory=output
    )
    assert commitment_path.exists()
    return output, commitment


def _admit(
    output: Path,
    commitment: operator.DesignSeedCommitment,
    *,
    bundle: bytes | None = None,
) -> tuple[Path, operator.DesignSeedAttestationAdmission]:
    commitment_path = output / commitment.attestation_subject_name
    bundle_path = output / "bundle.json"
    encoded = (
        _bundle(
            commitment,
            integrated_time=operator.QUICKNET_GENESIS_UNIX_SECONDS
            - operator.MINIMUM_PRE_ROUND_LEAD_SECONDS,
        )
        if bundle is None
        else bundle
    )
    _write(bundle_path, encoded)
    return operator.admit_design_seed_attestation(
        commitment_path,
        bundle_path,
        output_directory=output,
        verifier=_fake_verifier,
        remote_verifier=_fake_remote_verifier,
    )


def test_end_to_end_derives_first_future_round_and_bls_verified_seed(tmp_path: Path) -> None:
    output, commitment = _base(tmp_path)
    admission_path, admission = _admit(output, commitment)
    beacon_path = output / "quicknet-round-1.json"
    _write(beacon_path, ROUND_1_BYTES)

    reveal_path, reveal = operator.build_design_seed_reveal(
        output / commitment.attestation_subject_name,
        admission_path,
        beacon_path,
        output_directory=output,
        attestation_verifier=_fake_verifier,
        remote_verifier=_fake_remote_verifier,
    )
    observed_commitment = operator.verify_design_seed_commitment(
        output / commitment.attestation_subject_name,
        expected_sha256=commitment.commitment_sha256,
    )
    observed = operator.verify_design_seed_reveal(
        reveal_path,
        expected_sha256=reveal.reveal_sha256,
        commitment=observed_commitment,
        attestation_verifier=_fake_verifier,
        remote_verifier=_fake_remote_verifier,
    )

    assert admission.target_round == 1
    assert admission.pre_round_lead_seconds == 900
    assert observed.quicknet_signature == ROUND_1_SIGNATURE
    assert observed.quicknet_randomness == ROUND_1_RANDOMNESS
    assert observed.design_seed_sha256 == (
        "1c2576f9fed42ba68252d95a90fe858cccc3bbcbf4f9dee02dbd7024d8362dc6"
    )
    assert observed.design_seed_sha256 == reveal.design_seed_sha256
    assert observed.attestation_admission_path == str(admission_path.resolve())
    assert observed.scope_sha256 == commitment.scope_sha256
    assert observed.staged_inventory_sha256 == commitment.staged_inventory_sha256
    assert observed.partition_audit_file_sha256 == commitment.partition_audit_file_sha256
    assert observed.phase1_view_receipt_sha256 == commitment.phase1_view_receipt_sha256
    assert observed.selection_receipt_sha256 == commitment.selection_receipt_sha256


def test_target_round_is_first_round_at_least_900_seconds_after_rekor_time() -> None:
    integrated = operator.QUICKNET_GENESIS_UNIX_SECONDS + 17
    target, publication, lead = operator._derive_target_round(integrated)

    assert publication == (
        operator.QUICKNET_GENESIS_UNIX_SECONDS + (target - 1) * operator.QUICKNET_PERIOD_SECONDS
    )
    assert lead in {900, 901, 902}
    assert publication - operator.QUICKNET_PERIOD_SECONDS - integrated < 900


def test_commitment_publication_is_no_replace_with_non_authoritative_local_marker(
    tmp_path: Path,
) -> None:
    output = tmp_path / "controls"
    output.mkdir(mode=0o700)
    request_path, request = operator.build_design_seed_request(
        staged_inventory_sha256="1" * 64,
        partition_audit_file_sha256="2" * 64,
        phase1_view_receipt_sha256="3" * 64,
        selection_receipt_sha256="4" * 64,
        attestation_workflow=operator.ATTESTATION_WORKFLOW,
        attestation_workflow_sha="a" * 40,
        attestation_git_ref=operator.ATTESTATION_GIT_REF,
        output_directory=output,
    )
    operator.build_design_seed_commitment(request_path, output_directory=output)

    with pytest.raises(operator.DesignSeedCommitmentError, match="existing output"):
        operator.build_design_seed_commitment(request_path, output_directory=output)

    marker = output / f".design-seed-scope-{request.scope_sha256}.local-attempt.json"
    assert json.loads(marker.read_text(encoding="ascii")) == {
        "authority": "LOCAL_DEFENSE_ONLY",
        "request_sha256": request.request_sha256,
        "schema_version": operator.LOCAL_ATTEMPT_SCHEMA,
        "scope_sha256": request.scope_sha256,
        "state": "ATTEMPTED",
    }
    assert (output / f"design-seed-commitment-{request.scope_sha256}.json").read_bytes() == (
        operator.verify_design_seed_commitment(
            output / f"design-seed-commitment-{request.scope_sha256}.json"
        ).canonical_file_bytes()
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"run_attempt": 2}, "run_attempt"),
        ({"actor": "someone-else"}, "actor"),
        ({"triggering_actor": "someone-else"}, "triggering_actor"),
        ({"event": "push"}, "event"),
        ({"commitment_sha256": "f" * 64}, "commitment_sha256"),
        ({"scope_sha256": "e" * 64}, "scope_sha256"),
    ),
)
def test_attestation_rejects_mutated_signed_predicate(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    output, commitment = _base(tmp_path)
    bundle = _bundle(
        commitment,
        integrated_time=operator.QUICKNET_GENESIS_UNIX_SECONDS - 900,
        predicate_changes=changes,
    )
    bundle_path = output / "bundle.json"
    _write(bundle_path, bundle)

    with pytest.raises(operator.DesignSeedCommitmentError, match=message):
        operator.admit_design_seed_attestation(
            output / commitment.attestation_subject_name,
            bundle_path,
            output_directory=output,
            verifier=_fake_verifier,
            remote_verifier=_fake_remote_verifier,
        )


def test_attestation_rejects_wrong_subject_before_external_verifier(tmp_path: Path) -> None:
    output, commitment = _base(tmp_path)
    bundle_path = output / "bundle.json"
    _write(
        bundle_path,
        _bundle(
            commitment,
            integrated_time=operator.QUICKNET_GENESIS_UNIX_SECONDS - 900,
            subject_digest="0" * 64,
        ),
    )
    called = False

    def verifier(**_kwargs: object) -> None:
        nonlocal called
        called = True

    with pytest.raises(operator.DesignSeedCommitmentError, match="subject digest"):
        operator.admit_design_seed_attestation(
            output / commitment.attestation_subject_name,
            bundle_path,
            output_directory=output,
            verifier=verifier,
            remote_verifier=_fake_remote_verifier,
        )
    assert called is False


def test_attestation_rejects_failed_cryptographic_verifier(tmp_path: Path) -> None:
    output, commitment = _base(tmp_path)
    bundle_path = output / "bundle.json"
    _write(
        bundle_path,
        _bundle(
            commitment,
            integrated_time=operator.QUICKNET_GENESIS_UNIX_SECONDS - 900,
        ),
    )

    def reject(**_kwargs: object) -> None:
        raise RuntimeError("signature rejected")

    with pytest.raises(operator.DesignSeedCommitmentError, match="verification failed"):
        operator.admit_design_seed_attestation(
            output / commitment.attestation_subject_name,
            bundle_path,
            output_directory=output,
            verifier=reject,
            remote_verifier=_fake_remote_verifier,
        )


def test_mutated_derived_round_is_rejected_structurally(tmp_path: Path) -> None:
    output, commitment = _base(tmp_path)
    admission_path, _ = _admit(output, commitment)
    _rewrite_json(admission_path, lambda row: row.__setitem__("target_round", 2))

    with pytest.raises(operator.DesignSeedCommitmentError, match="mechanically derived"):
        operator.verify_design_seed_attestation(
            admission_path,
            commitment=commitment,
            verifier=_fake_verifier,
            remote_verifier=_fake_remote_verifier,
        )


def test_reveal_rejects_valid_signature_for_another_round(tmp_path: Path) -> None:
    output, commitment = _base(tmp_path)
    admission_path, _ = _admit(output, commitment)
    beacon_path = output / "wrong-round.json"
    wrong = json.loads(ROUND_1_BYTES)
    wrong["round"] = 2
    _write(beacon_path, _canonical(wrong)[:-1])

    with pytest.raises(operator.DesignSeedCommitmentError, match="BLS verification"):
        operator.build_design_seed_reveal(
            output / commitment.attestation_subject_name,
            admission_path,
            beacon_path,
            output_directory=output,
            attestation_verifier=_fake_verifier,
            remote_verifier=_fake_remote_verifier,
        )


def test_reveal_rejects_seed_mutation_and_admission_substitution(tmp_path: Path) -> None:
    output, commitment = _base(tmp_path)
    admission_path, _ = _admit(output, commitment)
    beacon_path = output / "quicknet-round-1.json"
    _write(beacon_path, ROUND_1_BYTES)
    reveal_path, reveal = operator.build_design_seed_reveal(
        output / commitment.attestation_subject_name,
        admission_path,
        beacon_path,
        output_directory=output,
        attestation_verifier=_fake_verifier,
        remote_verifier=_fake_remote_verifier,
    )
    _rewrite_json(reveal_path, lambda row: row.__setitem__("design_seed_sha256", "f" * 64))

    with pytest.raises(operator.DesignSeedCommitmentError, match="filename differs"):
        operator.verify_design_seed_reveal(
            reveal_path,
            commitment=commitment,
            attestation_verifier=_fake_verifier,
            remote_verifier=_fake_remote_verifier,
        )

    os.chmod(reveal_path, 0o600)
    reveal_path.write_bytes(reveal.canonical_file_bytes())
    os.chmod(reveal_path, 0o400)
    _rewrite_json(admission_path, lambda row: row.__setitem__("rekor_log_index", 43))
    with pytest.raises(operator.DesignSeedCommitmentError, match="digest differs"):
        operator.verify_design_seed_reveal(
            reveal_path,
            commitment=commitment,
            attestation_verifier=_fake_verifier,
            remote_verifier=_fake_remote_verifier,
        )


def test_closed_schemas_and_canonical_bytes_reject_extension_and_duplicate_key(
    tmp_path: Path,
) -> None:
    output, commitment = _base(tmp_path)
    path = output / commitment.attestation_subject_name
    _rewrite_json(path, lambda row: row.__setitem__("design_seed_sha256", "0" * 64))
    with pytest.raises(operator.DesignSeedCommitmentError, match="unknown"):
        operator.verify_design_seed_commitment(path)

    duplicate_path = output / f"design-seed-request-{'0' * 64}.json"
    _write(
        duplicate_path,
        b'{"schema_version":"x","schema_version":"x"}\n',
        mode=0o400,
    )
    with pytest.raises(operator.DesignSeedCommitmentError, match="repeats JSON key"):
        operator.verify_design_seed_request(duplicate_path)


def test_cli_has_no_round_or_design_seed_override() -> None:
    parser = operator._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "build-request",
                "--staged-inventory-sha256",
                "1" * 64,
                "--partition-audit-file-sha256",
                "2" * 64,
                "--phase1-view-receipt-sha256",
                "3" * 64,
                "--selection-receipt-sha256",
                "4" * 64,
                "--attestation-workflow",
                operator.ATTESTATION_WORKFLOW,
                "--attestation-workflow-sha",
                "a" * 40,
                "--attestation-git-ref",
                operator.ATTESTATION_GIT_REF,
                "--output-directory",
                "/tmp/control",
                "--target-round",
                "99",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "build-reveal",
                "--commitment",
                "/tmp/commitment",
                "--admission",
                "/tmp/admission",
                "--beacon",
                "/tmp/beacon",
                "--output-directory",
                "/tmp/control",
                "--design-seed-sha256",
                "f" * 64,
            ]
        )


def test_quicknet_constants_equal_the_exact_p_verifier() -> None:
    assert operator.QUICKNET_CHAIN_HASH == (
        "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
    )
    assert operator.QUICKNET_PERIOD_SECONDS == 3
    assert operator.QUICKNET_GENESIS_UNIX_SECONDS == 1_692_803_367
    assert operator.QUICKNET_SCHEME_ID == "bls-unchained-g1-rfc9380"
    assert len(bytes.fromhex(operator.QUICKNET_PUBLIC_KEY)) == 96


def _raw_remote_api(
    commitment: operator.DesignSeedCommitment,
    predicate: dict[str, object],
    integrated_time: int,
) -> dict[str, dict[str, object]]:
    published = _github_timestamp(integrated_time - 60)
    run_started = _github_timestamp(integrated_time - 120)
    commit_created = _github_timestamp(integrated_time - 3_600)
    return {
        "run": {
            "actor": {"login": operator.OWNER_LOGIN},
            "conclusion": "success",
            "event": operator.EVENT,
            "head_branch": operator._ref_name(operator.ATTESTATION_GIT_REF),
            "head_repository": {"full_name": operator.REPOSITORY},
            "head_sha": commitment.attestation_workflow_sha,
            "id": predicate["run_id"],
            "path": operator.ATTESTATION_WORKFLOW,
            "repository": {"full_name": operator.REPOSITORY},
            "run_attempt": 1,
            "run_started_at": run_started,
            "status": "completed",
            "triggering_actor": {"login": operator.OWNER_LOGIN},
        },
        "release": {
            "assets": [],
            "author": {"login": operator.RELEASE_AUTHOR},
            # GitHub defines release.created_at as the release commit's date,
            # which can precede the workflow that publishes the release.
            "created_at": commit_created,
            "draft": False,
            "id": predicate["release_id"],
            "immutable": True,
            "name": predicate["release_name"],
            "prerelease": False,
            "published_at": published,
            "tag_name": predicate["release_tag"],
            "target_commitish": commitment.attestation_workflow_sha,
        },
        "tag": {
            "object": {"sha": commitment.attestation_workflow_sha, "type": "commit"},
            "ref": f"refs/tags/{predicate['release_tag']}",
        },
    }


def test_public_api_verifier_closes_run_release_and_tag_when_commit_predates_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, commitment = _base(tmp_path)
    integrated = operator.QUICKNET_GENESIS_UNIX_SECONDS - 900
    predicate = _predicate(commitment, integrated_time=integrated)
    responses = _raw_remote_api(commitment, predicate, integrated)
    paths: list[str] = []

    def read(path: str) -> dict[str, object]:
        paths.append(path)
        if "/actions/runs/" in path:
            return responses["run"]
        if "/releases/" in path:
            return responses["release"]
        return responses["tag"]

    monkeypatch.setattr(operator, "_read_github_api", read)
    evidence = operator._default_remote_admission_verifier(
        commitment=commitment,
        predicate=predicate,
        rekor_integrated_at_utc=_github_timestamp(integrated).replace("Z", "+00:00"),
    )

    assert set(evidence) == operator._REMOTE_EVIDENCE_NAMES
    assert any(path.endswith("/attempts/1") for path in paths)
    assert any(f"/releases/{predicate['release_id']}" in path for path in paths)
    assert any("/git/ref/tags/design-seed-scope-" in path for path in paths)
    release_projection = json.loads(evidence["release"])
    assert "created_at" not in release_projection
    assert release_projection["published_at"] == predicate["release_published_at_utc"]


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    (
        ("run", lambda row: row.__setitem__("id", 123456), "run id"),
        (
            "run",
            lambda row: row.__setitem__("actor", {"login": "someone-else"}),
            "run actor",
        ),
        ("release", lambda row: row.__setitem__("immutable", False), "immutable"),
    ),
)
def test_public_api_verifier_rejects_self_asserted_or_reusable_remote_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    mutation: Any,
    message: str,
) -> None:
    _, commitment = _base(tmp_path)
    integrated = operator.QUICKNET_GENESIS_UNIX_SECONDS - 900
    predicate = _predicate(commitment, integrated_time=integrated)
    responses = _raw_remote_api(commitment, predicate, integrated)
    mutation(responses[target])

    def read(path: str) -> dict[str, object]:
        if "/actions/runs/" in path:
            return responses["run"]
        if "/releases/" in path:
            return responses["release"]
        return responses["tag"]

    monkeypatch.setattr(operator, "_read_github_api", read)
    with pytest.raises(operator.DesignSeedCommitmentError, match=message):
        operator._default_remote_admission_verifier(
            commitment=commitment,
            predicate=predicate,
            rekor_integrated_at_utc=_github_timestamp(integrated).replace("Z", "+00:00"),
        )


def test_atomic_publication_never_exposes_a_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "controls"
    output.mkdir(mode=0o700)
    target = output / "control.json"

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(operator.os, "link", fail_link)
    with pytest.raises(operator.DesignSeedCommitmentError, match="cannot publish"):
        operator._write_exclusive(target, b'{"closed":true}\n')

    assert not target.exists()
    assert not list(output.iterdir())


def test_exact_p_verifier_rejects_another_registered_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = operator._git_output

    def git_output(root: Path, arguments: list[str]) -> bytes:
        if arguments[:1] == ["rev-parse"]:
            return b"0" * 40 + b"\n"
        return original(root, arguments)

    monkeypatch.setattr(operator, "_git_output", git_output)
    with pytest.raises(operator.DesignSeedCommitmentError, match="source P tree differs"):
        operator._verify_exact_p_source()


def test_production_attestation_verifier_closes_github_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = tmp_path / "commitment.json"
    _write(subject, b"commitment\n", mode=0o400)
    observed: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=b'[{"verificationResult":{}}]',
            stderr=b"",
        )

    monkeypatch.setattr(operator.subprocess, "run", run)
    workflow_ref = (
        f"{operator.REPOSITORY}/{operator.ATTESTATION_WORKFLOW}@{operator.ATTESTATION_GIT_REF}"
    )
    operator._default_attestation_verifier(
        subject_path=subject,
        bundle_bytes=b"bundle",
        repository=operator.REPOSITORY,
        workflow_ref=workflow_ref,
        workflow_sha="a" * 40,
        git_ref=operator.ATTESTATION_GIT_REF,
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[0:3] == ["gh", "attestation", "verify"]
    assert command[command.index("--cert-identity") + 1] == f"https://github.com/{workflow_ref}"
    assert command[command.index("--signer-digest") + 1] == "a" * 40
    assert command[command.index("--source-digest") + 1] == "a" * 40
    assert command[command.index("--source-ref") + 1] == operator.ATTESTATION_GIT_REF
    assert command[command.index("--predicate-type") + 1] == (operator.ATTESTATION_PREDICATE_TYPE)
    assert "--deny-self-hosted-runners" in command
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["env"]["GH_PROMPT_DISABLED"] == "1"
    assert kwargs["env"]["GH_CONFIG_DIR"]
    assert "GH_TOKEN" not in kwargs["env"]
    assert "GITHUB_TOKEN" not in kwargs["env"]
