from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fractal_ann_diagnostics.github_state_attestation as state_attestation_module
import fractal_ann_diagnostics.provider_state_transport as transport_module
from fractal_ann_diagnostics.github_artifact_transport import (
    C0_HEAD_BRANCH,
    C0_REF,
    OWNER_ID,
    OWNER_LOGIN,
    REPOSITORY,
    REPOSITORY_ID,
    REPOSITORY_NODE_ID,
    GitHubHttpResponse,
)
from fractal_ann_diagnostics.github_state_attestation import (
    LEDGER_CONTROL_PREFIX,
    LEDGER_REF_PREFIX,
    OIDC_ISSUER,
    PREDICATE_TYPE,
    REKOR_IDENTITY,
    REKOR_URI,
    STATE_SERVICE_IDENTITY,
    STATE_SERVICE_URI,
    WORKFLOW_PATH,
    LedgerControlFile,
    LedgerProtection,
    LedgerSnapshot,
    LedgerTransition,
    ledger_predicate,
    parse_sigstore_bundle,
)
from fractal_ann_diagnostics.provider_state_transport import (
    ProviderStateTransportError,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA
from fractal_ann_diagnostics.suite_attempt import (
    CorpusDigest,
    CorpusNamespace,
    CorpusRuntimePlanBinding,
    SuiteAttestationDescriptor,
    SuiteAttestationEvidence,
    SuiteOpenBindings,
    SuiteStateRecord,
    suite_attempt_id,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _blob_oid(encoded: bytes) -> str:
    header = f"blob {len(encoded)}\0".encode()
    return hashlib.sha1(header + encoded, usedforsecurity=False).hexdigest()


def _descriptor() -> SuiteAttestationDescriptor:
    return SuiteAttestationDescriptor(
        expected_signer_identity=f"https://github.com/{REPOSITORY}/{WORKFLOW_PATH}@{C0_REF}",
        expected_oidc_issuer=OIDC_ISSUER,
        expected_repository=REPOSITORY,
        expected_workflow=WORKFLOW_PATH,
        expected_git_ref=C0_REF,
        expected_signer_digest="1" * 40,
        transparency_log_identity=REKOR_IDENTITY,
        transparency_log_uri=REKOR_URI,
        transparency_log_public_key_sha256=(b"k" * 32).hex(),
        timestamp_authority_identity=REKOR_IDENTITY,
        timestamp_authority_uri=REKOR_URI,
        timestamp_authority_public_key_sha256=(b"k" * 32).hex(),
        state_service_identity=STATE_SERVICE_IDENTITY,
        state_service_uri=STATE_SERVICE_URI,
        state_key_prefix=LEDGER_REF_PREFIX,
    )


def _opened(tmp_path: Path) -> tuple[SuiteAttestationDescriptor, SuiteStateRecord]:
    manifest = _digest("manifest")
    attempt = suite_attempt_id(manifest)
    namespace = tmp_path / f"suite-attempt-{attempt}"
    namespace.mkdir(mode=0o700)
    descriptor = _descriptor()
    finalization = namespace / "production-finalization-receipt.json"
    finalization_bytes = b'{"fixture":"finalization"}\n'
    finalization.write_bytes(finalization_bytes)
    ordered = tuple(sorted(FIXED_CORPORA, key=lambda item: item.encode()))
    contracts: dict[str, tuple[Path, bytes]] = {}
    for corpus_id in ordered:
        path = namespace / "contracts" / f"{corpus_id}.json"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        encoded = (
            json.dumps({"corpus_id": corpus_id}, separators=(",", ":"), sort_keys=True).encode()
            + b"\n"
        )
        path.write_bytes(encoded)
        contracts[corpus_id] = (path, encoded)
    payload = SuiteOpenBindings(
        protocol_registration_receipt_sha256=_digest("registration"),
        protocol_registration_receipt_file_sha256=_digest("registration-file"),
        protocol_registry_record_sha256=_digest("registry"),
        registered_at_utc="2026-07-14T12:00:00+00:00",
        run_receipt_file_sha256=_digest("run-file"),
        run_started_at_utc="2026-07-14T12:01:00+00:00",
        code_commit="1" * 40,
        runner_image=f"ghcr.io/example/runtime@sha256:{'2' * 64}",
        attestation_descriptor_sha256=descriptor.descriptor_sha256,
        production_finalization_receipt_uri=finalization.as_uri(),
        production_finalization_receipt_file_sha256=hashlib.sha256(finalization_bytes).hexdigest(),
        production_finalization_request_sha256=_digest("finalization-request"),
        provisional_closure_tree_sha256=_digest("provisional"),
        instantiated_closure_tree_sha256=_digest("instantiated"),
        runtime_attestation_plans=tuple(
            CorpusRuntimePlanBinding(
                corpus_id=corpus_id,
                plan_sha256=_digest(f"plan:{corpus_id}"),
                file_sha256=_digest(f"plan-file:{corpus_id}"),
                production_run_closure_binding_receipt_sha256=_digest(f"closure:{corpus_id}"),
                registered_plan_instantiation_receipt_sha256=_digest(f"instantiation:{corpus_id}"),
                registered_plan_instantiation_file_sha256=_digest(
                    f"instantiation-file:{corpus_id}"
                ),
                sealed_launch_contract_uri=contracts[corpus_id][0].as_uri(),
                sealed_launch_contract_sha256=_digest(f"contract:{corpus_id}"),
                sealed_launch_contract_file_sha256=hashlib.sha256(
                    contracts[corpus_id][1]
                ).hexdigest(),
            )
            for corpus_id in ordered
        ),
        execution_artifacts=tuple(
            CorpusDigest(corpus_id, _digest(f"execution:{corpus_id}")) for corpus_id in ordered
        ),
        staging_namespaces=tuple(
            CorpusNamespace(corpus_id, (namespace / "staging" / corpus_id).as_uri())
            for corpus_id in ordered
        ),
        output_namespaces=tuple(
            CorpusNamespace(corpus_id, (namespace / "online" / corpus_id).as_uri())
            for corpus_id in ordered
        ),
    )
    return descriptor, SuiteStateRecord(
        suite_attempt_id=attempt,
        manifest_sha256=manifest,
        run_receipt_sha256=_digest("run"),
        namespace_uri=namespace.as_uri(),
        sequence=0,
        state="OPENED",
        previous_state_record_sha256=None,
        payload=payload,
    )


def _bundle(snapshot: LedgerSnapshot, transition: LedgerTransition) -> bytes:
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "predicate": ledger_predicate(snapshot, transition),
        "predicateType": PREDICATE_TYPE,
        "subject": [
            {
                "digest": {"sha256": transition.state.record_sha256},
                "name": transition.state_path,
            }
        ],
    }
    value = {
        "dsseEnvelope": {
            "payload": base64.b64encode(
                json.dumps(statement, separators=(",", ":"), sort_keys=True).encode()
            ).decode(),
            "payloadType": "application/vnd.in-toto+json",
            "signatures": [{"sig": "signature"}],
        },
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "verificationMaterial": {
            "tlogEntries": [
                {
                    "canonicalizedBody": base64.b64encode(b"rekor-body").decode(),
                    "inclusionPromise": {
                        "signedEntryTimestamp": base64.b64encode(b"rekor-set").decode()
                    },
                    "integratedTime": "1784030520",
                    "logId": {"keyId": base64.b64encode(b"k" * 32).decode()},
                    "logIndex": "42",
                }
            ]
        },
    }
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _evidence(
    descriptor: SuiteAttestationDescriptor,
    snapshot: LedgerSnapshot,
    bundle: bytes,
) -> SuiteAttestationEvidence:
    transition = snapshot.tip
    observation = parse_sigstore_bundle(bundle)
    return SuiteAttestationEvidence(
        suite_attempt_id=transition.state.suite_attempt_id,
        state_sequence=transition.state.sequence,
        state_name=transition.state.state,
        state_record_sha256=transition.state.record_sha256,
        descriptor_sha256=descriptor.descriptor_sha256,
        bundle_sha256=hashlib.sha256(bundle).hexdigest(),
        bundle_byte_count=len(bundle),
        signer_identity=descriptor.expected_signer_identity,
        oidc_issuer=descriptor.expected_oidc_issuer,
        repository=descriptor.expected_repository,
        workflow=descriptor.expected_workflow,
        git_ref=descriptor.expected_git_ref,
        signer_digest=descriptor.expected_signer_digest,
        github_hosted_runner=True,
        transparency_log_identity=REKOR_IDENTITY,
        transparency_entry_id=observation.entry_id,
        transparency_log_index=observation.log_index,
        integrated_at_utc=observation.integrated_at_utc,
        timestamp_authority_identity=REKOR_IDENTITY,
        timestamp_token_sha256=observation.timestamp_token_sha256,
        signed_at_utc=observation.integrated_at_utc,
        state_service_identity=STATE_SERVICE_IDENTITY,
        state_key=snapshot.state_key,
        transition_id=transition.commit_oid,
        previous_transition_id=None,
    )


def _snapshot(
    tmp_path: Path,
) -> tuple[SuiteAttestationDescriptor, LedgerSnapshot, dict[str, bytes]]:
    descriptor, state = _opened(tmp_path)
    state_bytes = state.canonical_bytes() + b"\n"
    transition = LedgerTransition(
        commit_oid="a" * 40,
        previous_commit_oid=None,
        tree_oid="b" * 40,
        state_path=f"confirmatory-state/{state.suite_attempt_id}/000.state.json",
        state_bytes=state_bytes,
        state=state,
    )
    descriptor_bytes = descriptor.canonical_bytes() + b"\n"
    ledger_path = f"{LEDGER_CONTROL_PREFIX}/{state.suite_attempt_id}/attestation-descriptor.json"
    control = LedgerControlFile(
        role="attestation-descriptor",
        ledger_path=ledger_path,
        materialization_uri=(tmp_path / "descriptor.json").as_uri(),
        file_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
        byte_count=len(descriptor_bytes),
        blob_oid=_blob_oid(descriptor_bytes),
        encoded=descriptor_bytes,
    )
    inventory = (
        json.dumps([control.to_inventory_dict()], separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )
    snapshot = LedgerSnapshot(
        repository=REPOSITORY,
        state_key=f"{LEDGER_REF_PREFIX}/{state.suite_attempt_id}",
        protection=LedgerProtection(True, True, True, True, (1,)),
        transitions=(transition,),
        controls=(control,),
        control_inventory_bytes=inventory,
    )
    bundle = _bundle(snapshot, transition)
    evidence = _evidence(descriptor, snapshot, bundle)
    rows = {
        "000.state.json": state_bytes,
        "000.attestation.json": evidence.canonical_bytes() + b"\n",
        "000.sigstore.bundle.json": bundle,
        "ledger-controls/inventory.json": inventory,
        "ledger-controls/attestation-descriptor.json": descriptor_bytes,
    }
    return descriptor, snapshot, rows


def _archive(
    rows: dict[str, bytes],
    *,
    extra_unlisted: bool = False,
    inventory_name: str = "SHA256SUMS",
) -> bytes:
    sums = b"".join(
        f"{hashlib.sha256(value).hexdigest()}  ./{name}\n".encode()
        for name, value in sorted(rows.items(), key=lambda item: item[0].encode())
    )
    with io.BytesIO() as output:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, value in rows.items():
                archive.writestr(name, value)
            archive.writestr(inventory_name, sums)
            if extra_unlisted:
                archive.writestr("unlisted.txt", b"untrusted")
        return output.getvalue()


def _json(value: object) -> GitHubHttpResponse:
    return GitHubHttpResponse(200, {}, json.dumps(value).encode())


def _verify_claim_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: dict[str, bytes],
    snapshot: LedgerSnapshot,
) -> Any:
    encoded = _archive(rows, inventory_name="claim-package.SHA256SUMS")
    expectation = transport_module._ArtifactExpectation(
        transition=snapshot.tip,
        workflow_path=".github/workflows/confirmatory-online-execution.yml",
        workflow_sha="1" * 40,
        run_id=71,
        artifact_name="claim-artifact",
        inventory_name="claim-package.SHA256SUMS",
    )
    remote = transport_module._RemoteArtifact(
        expectation=expectation,
        run_id=71,
        workflow_id=72,
        artifact_id=73,
        artifact_node_id="artifact-node",
        artifact_name=expectation.artifact_name,
        artifact_digest=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        artifact_size_bytes=len(encoded),
        artifact_created_at="2026-07-14T12:05:00Z",
        artifact_expires_at="2099-07-14T12:05:00Z",
    )
    monkeypatch.setattr(
        transport_module,
        "_download_archive_exact",
        lambda *args, **kwargs: encoded,
    )
    return transport_module._verify_archive(
        object(),
        remote,
        destination=tmp_path / "retained-claim",
        retain_claim_recovery=True,
    )


class _ArtifactApi:
    def __init__(self, snapshot: LedgerSnapshot, archive: bytes) -> None:
        self.snapshot = snapshot
        self.archive = archive
        self.run_id = 71
        self.workflow_id = 72
        self.artifact_id = 73
        self.calls: list[str] = []
        self.duplicate = False
        self.mutated_digest = False

    def _artifact(self) -> dict[str, Any]:
        transition = self.snapshot.tip
        name = f"confirmatory-state-{transition.state.suite_attempt_id}-0-{transition.commit_oid}"
        digest = hashlib.sha256(self.archive).hexdigest()
        if self.mutated_digest:
            digest = "f" * 64
        return {
            "id": self.artifact_id,
            "node_id": "artifact-node",
            "name": name,
            "digest": f"sha256:{digest}",
            "size_in_bytes": len(self.archive),
            "created_at": "2026-07-14T12:05:00Z",
            "expires_at": "2099-07-14T12:05:00Z",
            "expired": False,
            "archive_download_url": (
                f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/"
                f"{self.artifact_id}/zip"
            ),
            "workflow_run": {
                "id": self.run_id,
                "head_sha": "1" * 40,
                "head_branch": C0_HEAD_BRANCH,
                "repository_id": REPOSITORY_ID,
                "head_repository_id": REPOSITORY_ID,
            },
        }

    def get(self, location: str, *, accept: str) -> GitHubHttpResponse:
        if location == "https://pipelines.actions.githubusercontent.com/signed":
            assert accept == "application/octet-stream"
            return GitHubHttpResponse(200, {}, self.archive)
        assert accept == "application/vnd.github+json"
        self.calls.append(location)
        artifact = self._artifact()
        name = artifact["name"]
        named = f"repos/{REPOSITORY}/actions/artifacts?name={name}&per_page=100"
        if location == named:
            rows = [artifact, dict(artifact)] if self.duplicate else [artifact]
            return _json({"total_count": len(rows), "artifacts": rows})
        if location == f"repos/{REPOSITORY}/actions/runs/{self.run_id}/attempts/1":
            return _json(
                {
                    "id": self.run_id,
                    "run_attempt": 1,
                    "event": "workflow_dispatch",
                    "workflow_id": self.workflow_id,
                    "head_sha": "1" * 40,
                    "head_branch": C0_HEAD_BRANCH,
                    "status": "completed",
                    "conclusion": "success",
                    "path": transport_module.STATE_ATTESTATION_WORKFLOW,
                    "actor": {"id": OWNER_ID, "login": OWNER_LOGIN},
                    "triggering_actor": {"id": OWNER_ID, "login": OWNER_LOGIN},
                    "repository": {
                        "id": REPOSITORY_ID,
                        "node_id": REPOSITORY_NODE_ID,
                        "full_name": REPOSITORY,
                    },
                    "head_repository": {
                        "id": REPOSITORY_ID,
                        "node_id": REPOSITORY_NODE_ID,
                        "full_name": REPOSITORY,
                    },
                }
            )
        if location == f"repos/{REPOSITORY}/actions/workflows/{self.workflow_id}":
            return _json(
                {"id": self.workflow_id, "path": transport_module.STATE_ATTESTATION_WORKFLOW}
            )
        if location == f"repos/{REPOSITORY}/actions/artifacts/{self.artifact_id}":
            return _json(artifact)
        if location == f"repos/{REPOSITORY}":
            return _json(
                {
                    "full_name": REPOSITORY,
                    "id": REPOSITORY_ID,
                    "node_id": REPOSITORY_NODE_ID,
                    "private": False,
                    "fork": False,
                    "owner": {"id": OWNER_ID, "login": OWNER_LOGIN},
                }
            )
        if location == f"repos/{REPOSITORY}/git/ref/tags/{C0_HEAD_BRANCH}":
            return _json({"ref": C0_REF, "object": {"type": "tag", "sha": "2" * 40}})
        if location == f"repos/{REPOSITORY}/git/tags/{'2' * 40}":
            return _json(
                {
                    "tag": C0_HEAD_BRANCH,
                    "object": {"type": "commit", "sha": "1" * 40},
                    "tagger": {
                        "name": OWNER_LOGIN,
                        "email": f"{OWNER_LOGIN}@users.noreply.github.com",
                    },
                }
            )
        if location == f"repos/{REPOSITORY}/actions/artifacts/{self.artifact_id}/zip":
            return GitHubHttpResponse(
                302,
                {"Location": "https://pipelines.actions.githubusercontent.com/signed"},
                b"",
            )
        raise AssertionError(f"unexpected artifact API request: {location}")


class _AttestationVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, **kwargs: object) -> bytes:
        self.calls += 1
        assert Path(kwargs["state_path"]).is_file()
        assert Path(kwargs["bundle_path"]).is_file()
        return b'[{"verificationResult":{}}]'


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows_mutator: Any = None,
    extra_unlisted: bool = False,
) -> tuple[Any, _ArtifactApi, int]:
    _, snapshot, rows = _snapshot(tmp_path)
    if rows_mutator is not None:
        rows_mutator(rows)
    api = _ArtifactApi(snapshot, _archive(rows, extra_unlisted=extra_unlisted))
    ledger_calls = 0

    def load(**kwargs: object) -> LedgerSnapshot:
        nonlocal ledger_calls
        ledger_calls += 1
        assert kwargs["suite_attempt_id"] == snapshot.tip.state.suite_attempt_id
        return snapshot

    monkeypatch.setattr(transport_module, "load_ledger_snapshot", load)
    monkeypatch.setattr(state_attestation_module, "load_ledger_snapshot", load)
    verifier = _AttestationVerifier()
    result = transport_module._materialize_provider_predecessor(
        "online",
        snapshot.tip.state.suite_attempt_id,
        tmp_path,
        ledger_api=object(),
        artifact_api=api,
        attestation_verifier=verifier,
    )
    assert verifier.calls == 1
    return result, api, ledger_calls


def test_materializes_exact_state_and_mints_fresh_provider_predecessor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, api, ledger_calls = _run(tmp_path, monkeypatch)

    assert result.predecessor.state.state == "OPENED"
    assert result.predecessor.ledger_commit == "a" * 40
    assert result.predecessor.artifact_receipt_sha256 == result.receipt.receipt_sha256
    assert result.receipt.artifacts[0].artifact_id == api.artifact_id
    assert result.receipt.predecessor_state == "OPENED"
    assert ledger_calls >= 3
    assert Path(result.receipt.materialized_root, "000.state.json").is_file()


def test_rejects_self_consistent_artifact_state_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(rows: dict[str, bytes]) -> None:
        rows["000.state.json"] = b'{"substituted":true}\n'

    with pytest.raises(ProviderStateTransportError, match="state bytes differ"):
        _run(tmp_path, monkeypatch, rows_mutator=mutate)


def test_rejects_control_substitution_even_when_sha256sums_is_reclosed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(rows: dict[str, bytes]) -> None:
        rows["ledger-controls/attestation-descriptor.json"] = b'{"substituted":true}\n'

    with pytest.raises(ProviderStateTransportError, match="controls differ"):
        _run(tmp_path, monkeypatch, rows_mutator=mutate)


def test_rejects_duplicate_exact_artifact_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, snapshot, rows = _snapshot(tmp_path)
    api = _ArtifactApi(snapshot, _archive(rows))
    api.duplicate = True
    monkeypatch.setattr(transport_module, "load_ledger_snapshot", lambda **_: snapshot)
    monkeypatch.setattr(
        state_attestation_module,
        "load_ledger_snapshot",
        lambda **_: snapshot,
    )

    with pytest.raises(ProviderStateTransportError, match="not a singleton"):
        transport_module._materialize_provider_predecessor(
            "online",
            snapshot.tip.state.suite_attempt_id,
            tmp_path,
            ledger_api=object(),
            artifact_api=api,
            attestation_verifier=_AttestationVerifier(),
        )


def test_rejects_unlisted_zip_member_and_fresh_artifact_authority_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ProviderStateTransportError, match="members differ"):
        _run(tmp_path, monkeypatch, extra_unlisted=True)

    other = tmp_path / "second"
    other.mkdir(mode=0o700)
    result, api, _ = _run(other, monkeypatch)
    api.mutated_digest = True
    with pytest.raises(ProviderStateTransportError, match="artifact authority changed"):
        result.predecessor.assert_current()


def test_claim_archive_retains_only_closed_recovery_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, snapshot, rows = _snapshot(tmp_path)
    base_members = set(rows)
    rows.update(
        {
            "claim-receipt.json": b'{"fixture":"claim"}\n',
            "c1-package/study-manifest.json": b'{"fixture":"manifest"}\n',
            "provider-plan.materialized.json": b'{"fixture":"plan"}\n',
            "listed-but-unneeded.txt": b"not retained\n",
        }
    )
    verified = _verify_claim_archive(tmp_path, monkeypatch, rows, snapshot)
    expected = base_members | {
        "claim-package.SHA256SUMS",
        "claim-receipt.json",
        "c1-package/study-manifest.json",
        "provider-plan.materialized.json",
    }
    assert set(verified.retained) == expected
    assert "listed-but-unneeded.txt" not in verified.retained
    assert not (tmp_path / "retained-claim" / "listed-but-unneeded.txt").exists()


@pytest.mark.parametrize(
    ("mutation", "member"),
    (
        (
            lambda rows: rows.pop("provider-plan.materialized.json"),
            "provider-plan.materialized.json",
        ),
        (
            lambda rows: rows.__setitem__(
                "duplicate/study-manifest.json",
                b'{"fixture":"duplicate"}\n',
            ),
            "study-manifest.json",
        ),
        (
            lambda rows: rows.__setitem__(
                "alias/Study-Manifest.json",
                b'{"fixture":"alias"}\n',
            ),
            "study-manifest.json",
        ),
    ),
)
def test_claim_archive_rejects_missing_duplicate_or_basename_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    member: str,
) -> None:
    _, snapshot, rows = _snapshot(tmp_path)
    rows.update(
        {
            "claim-receipt.json": b'{"fixture":"claim"}\n',
            "c1-package/study-manifest.json": b'{"fixture":"manifest"}\n',
            "provider-plan.materialized.json": b'{"fixture":"plan"}\n',
        }
    )
    mutation(rows)
    with pytest.raises(
        ProviderStateTransportError,
        match=f"requires one canonical {re.escape(member)} member",
    ):
        _verify_claim_archive(tmp_path, monkeypatch, rows, snapshot)


def test_predecessor_transport_does_not_expand_claim_recovery_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def add_claim_members(rows: dict[str, bytes]) -> None:
        rows.update(
            {
                "claim-receipt.json": b'{"fixture":"claim"}\n',
                "c1-package/study-manifest.json": b'{"fixture":"manifest"}\n',
                "provider-plan.materialized.json": b'{"fixture":"plan"}\n',
            }
        )

    result, _, _ = _run(tmp_path, monkeypatch, rows_mutator=add_claim_members)
    artifact_root = Path(result.receipt.materialized_root) / "artifacts"
    retained_names = {path.name for path in artifact_root.rglob("*") if path.is_file()}
    assert not set(transport_module._CLAIM_RECOVERY_BASENAMES) & retained_names


@pytest.mark.parametrize(
    ("phase", "state", "sequence"),
    [
        ("online", "RUN_CLAIMED", 1),
        ("label-release", "LABEL_RELEASE_CLAIMED", 3),
        ("analysis", "ANALYSIS_CLAIMED", 5),
    ],
)
def test_phase_to_claim_target_is_closed(
    phase: str,
    state: str,
    sequence: int,
) -> None:
    assert transport_module._PHASE_CLAIM[phase] == state
    assert transport_module._PHASE_CLAIM_SEQUENCE[phase] == sequence


def test_claim_transport_rejects_preclaim_and_postclaim_tip_substitutions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, snapshot, rows = _snapshot(tmp_path)
    api = _ArtifactApi(snapshot, _archive(rows))
    monkeypatch.setattr(transport_module, "load_ledger_snapshot", lambda **_: snapshot)
    with pytest.raises(ProviderStateTransportError, match="exact phase claim"):
        transport_module.materialize_provider_claim(
            "online",
            snapshot.tip.state.suite_attempt_id,
            tmp_path,
            ledger_api=object(),
            artifact_api=api,
        )
    assert api.calls == []

    postclaim = SimpleNamespace(
        state_key=snapshot.state_key,
        tip=SimpleNamespace(
            state=SimpleNamespace(
                suite_attempt_id=snapshot.tip.state.suite_attempt_id,
                state="ONLINE_COMPLETE",
                sequence=2,
            )
        ),
        transitions=(object(), object(), object()),
    )
    monkeypatch.setattr(transport_module, "load_ledger_snapshot", lambda **_: postclaim)
    with pytest.raises(ProviderStateTransportError, match="exact phase claim"):
        transport_module.materialize_provider_claim(
            "online",
            snapshot.tip.state.suite_attempt_id,
            tmp_path,
            ledger_api=object(),
            artifact_api=api,
        )


def test_materializes_exact_online_claim_from_fixed_claim_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_suite_attempt import (
        _attest,
        _state_chain,
        _transition_id,
    )

    fixture_root = tmp_path / "claim-fixture"
    fixture_root.mkdir(mode=0o700)
    namespace, records = _state_chain(fixture_root, through="RUN_CLAIMED")
    _attest(namespace, records)
    descriptor_bytes = (namespace / "attestation-descriptor.json").read_bytes()
    descriptor = SuiteAttestationDescriptor.from_dict(json.loads(descriptor_bytes))
    ledger_path = (
        f"{LEDGER_CONTROL_PREFIX}/{records[0].suite_attempt_id}/attestation-descriptor.json"
    )
    control = LedgerControlFile(
        role="attestation-descriptor",
        ledger_path=ledger_path,
        materialization_uri=(tmp_path / "claim-descriptor.json").as_uri(),
        file_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
        byte_count=len(descriptor_bytes),
        blob_oid=_blob_oid(descriptor_bytes),
        encoded=descriptor_bytes,
    )
    inventory = (
        json.dumps([control.to_inventory_dict()], separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )
    transitions = tuple(
        LedgerTransition(
            commit_oid=_transition_id(record.sequence),
            previous_commit_oid=(
                None if record.sequence == 0 else _transition_id(record.sequence - 1)
            ),
            tree_oid=f"{record.sequence + 1:x}" * 40,
            state_path=(
                f"confirmatory-state/{record.suite_attempt_id}/{record.sequence:03d}.state.json"
            ),
            state_bytes=record.canonical_bytes() + b"\n",
            state=record,
        )
        for record in records
    )
    snapshot = LedgerSnapshot(
        repository=REPOSITORY,
        state_key=f"{LEDGER_REF_PREFIX}/{records[0].suite_attempt_id}",
        protection=LedgerProtection(True, True, True, True, (1,)),
        transitions=transitions,
        controls=(control,),
        control_inventory_bytes=inventory,
    )
    evidences = {
        record.sequence: SuiteAttestationEvidence.from_dict(
            json.loads((namespace / f"{record.sequence:03d}.attestation.json").read_bytes())
        )
        for record in records
    }
    retained = {
        record.sequence: {
            f"{record.sequence:03d}.state.json": record.canonical_bytes() + b"\n",
            f"{record.sequence:03d}.attestation.json": (
                namespace / f"{record.sequence:03d}.attestation.json"
            ).read_bytes(),
            f"{record.sequence:03d}.sigstore.bundle.json": (
                namespace / f"{record.sequence:03d}.sigstore.bundle.json"
            ).read_bytes(),
        }
        for record in records
    }
    retained[0].update(
        {
            "ledger-controls/inventory.json": inventory,
            "ledger-controls/attestation-descriptor.json": descriptor_bytes,
        }
    )

    def load(**_: object) -> LedgerSnapshot:
        return snapshot

    def remote(_api: object, expectation: Any) -> Any:
        sequence = expectation.transition.state.sequence
        return transport_module._RemoteArtifact(
            expectation=expectation,
            run_id=expectation.run_id or 700 + sequence,
            workflow_id=800 + sequence,
            artifact_id=900 + sequence,
            artifact_node_id=f"artifact-{sequence}",
            artifact_name=expectation.artifact_name,
            artifact_digest=f"sha256:{_digest(f'artifact:{sequence}')}",
            artifact_size_bytes=1000 + sequence,
            artifact_created_at="2026-07-14T12:05:00Z",
            artifact_expires_at="2099-07-14T12:05:00Z",
        )

    retained[1].update(
        {
            "claim-receipt.json": b'{"fixture":"claim"}\n',
            "c1-package/study-manifest.json": b'{"fixture":"manifest"}\n',
            "provider-plan.materialized.json": b'{"fixture":"plan"}\n',
        }
    )

    def verify_archive(
        api: object,
        remote_row: Any,
        *,
        destination: Path,
        retain_claim_recovery: bool = False,
    ) -> Any:
        sequence = remote_row.expectation.transition.state.sequence
        assert retain_claim_recovery is (sequence == 1)
        inventory_name = remote_row.expectation.inventory_name
        rows = dict(retained[sequence])
        rows[inventory_name] = f"inventory:{sequence}\n".encode()
        transport_module._write_retained(destination, rows)
        authority = transport_module.ProviderStateArtifactAuthority(
            sequence=sequence,
            state=remote_row.expectation.transition.state.state,
            ledger_commit=remote_row.expectation.transition.commit_oid,
            workflow_path=remote_row.expectation.workflow_path,
            workflow_sha=remote_row.expectation.workflow_sha,
            run_id=remote_row.run_id,
            workflow_id=remote_row.workflow_id,
            artifact_id=remote_row.artifact_id,
            artifact_node_id=remote_row.artifact_node_id,
            artifact_name=remote_row.artifact_name,
            artifact_digest=remote_row.artifact_digest,
            artifact_size_bytes=remote_row.artifact_size_bytes,
            artifact_created_at=remote_row.artifact_created_at,
            artifact_expires_at=remote_row.artifact_expires_at,
            inventory_name=inventory_name,
            inventory_sha256=hashlib.sha256(rows[inventory_name]).hexdigest(),
            archive_sha256=_digest(f"archive:{sequence}"),
        )
        return transport_module._VerifiedArchive(authority=authority, retained=rows)

    monkeypatch.setattr(transport_module, "load_ledger_snapshot", load)
    monkeypatch.setattr(transport_module, "_remote_artifact", remote)
    monkeypatch.setattr(transport_module, "_verify_archive", verify_archive)
    monkeypatch.setattr(
        transport_module,
        "_verify_one_attestation",
        lambda **kwargs: evidences[kwargs["state"].sequence],
    )
    result = transport_module.materialize_provider_claim(
        "online",
        records[0].suite_attempt_id,
        tmp_path,
        ledger_api=object(),
        artifact_api=object(),
    )

    assert descriptor.descriptor_sha256 == records[0].payload.attestation_descriptor_sha256
    assert result.predecessor.state.state == "RUN_CLAIMED"
    assert result.predecessor.state.sequence == 1
    assert result.receipt.authority_kind == "claim"
    assert result.receipt.artifacts[1].artifact_name == (
        f"confirmatory-online-claim-{records[0].suite_attempt_id}-31337"
    )
    assert result.receipt.artifacts[1].inventory_name == "claim-package.SHA256SUMS"
    claim_root = Path(result.receipt.materialized_root) / "artifacts" / "001-901"
    assert (claim_root / "claim-receipt.json").is_file()
    assert (claim_root / "c1-package" / "study-manifest.json").is_file()
    assert (claim_root / "provider-plan.materialized.json").is_file()

    (claim_root / "claim-receipt.json").write_bytes(b'{"mutated":true}\n')
    with pytest.raises(ProviderStateTransportError, match="materialized state evidence changed"):
        result.predecessor.assert_current()
