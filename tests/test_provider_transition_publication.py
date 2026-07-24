from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.provider_transition_publication as publication_module
from fractal_ann_diagnostics.github_state_attestation import (
    C0_REF,
    REPOSITORY,
    LedgerPublicationReceipt,
)
from fractal_ann_diagnostics.provider_transition_publication import (
    ONLINE_COMPLETION_PREDICATE_TYPE,
    GhProviderTransitionAttestationVerifier,
    ProviderTransitionPublicationError,
    verify_and_publish_provider_transition,
)
from fractal_ann_diagnostics.provider_workflow_orchestration import (
    C0_TAG,
    OWNER_ID,
    OWNER_LOGIN,
    REPOSITORY_ID,
    ProviderWorkflowContext,
)
from fractal_ann_diagnostics.study import PROVIDER_PHASE_WORKFLOWS
from fractal_ann_diagnostics.suite_attempt import (
    SUITE_STATE_SCHEMA,
    SuiteStateRecord,
    VerifiedProviderPredecessor,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(seed: str) -> str:
    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()


class _Payload:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_dict(self) -> dict[str, str]:
        return {"fixture": self.name}


def _state(
    *,
    suite: str,
    state: str,
    sequence: int,
    previous: str | None,
) -> SuiteStateRecord:
    value = object.__new__(SuiteStateRecord)
    for name, item in {
        "suite_attempt_id": suite,
        "manifest_sha256": _digest("manifest"),
        "run_receipt_sha256": _digest("run-receipt"),
        "namespace_uri": f"file:///tmp/suite-attempt-{suite}",
        "sequence": sequence,
        "state": state,
        "previous_state_record_sha256": previous,
        "payload": _Payload(state),
        "schema_version": SUITE_STATE_SCHEMA,
    }.items():
        object.__setattr__(value, name, item)
    return value


def _predecessor(
    *,
    phase: str = "online",
    revalidator: object | None = None,
) -> tuple[VerifiedProviderPredecessor, SuiteStateRecord]:
    suite = _digest("suite")
    claimed_state, sequence = {
        "online": ("RUN_CLAIMED", 1),
        "label-release": ("LABEL_RELEASE_CLAIMED", 3),
        "analysis": ("ANALYSIS_CLAIMED", 5),
    }[phase]
    claimed = _state(
        suite=suite,
        state=claimed_state,
        sequence=sequence,
        previous=_digest("previous"),
    )
    authority = object.__new__(VerifiedProviderPredecessor)
    for name, value in {
        "records": (claimed,),
        "evidences": (SimpleNamespace(transition_id="c" * 40),),
        "control_inventory_sha256": _digest("controls"),
        "artifact_receipt_sha256": _digest("artifact"),
        "_fresh_revalidator": revalidator if revalidator is not None else (lambda: None),
        "_capability": object(),
    }.items():
        object.__setattr__(authority, name, value)
    return authority, claimed


def _target(predecessor: SuiteStateRecord, *, state: str = "ONLINE_COMPLETE") -> SuiteStateRecord:
    return _state(
        suite=predecessor.suite_attempt_id,
        state=state,
        sequence=predecessor.sequence + 1,
        previous=predecessor.record_sha256,
    )


def _environment(*, phase: str = "online", job: str = "complete") -> dict[str, str]:
    workflow = PROVIDER_PHASE_WORKFLOWS[phase]
    return {
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITHUB_ACTOR": OWNER_LOGIN,
        "GITHUB_ACTOR_ID": str(OWNER_ID),
        "GITHUB_API_URL": "https://api.github.com",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_GRAPHQL_URL": "https://api.github.com/graphql",
        "GITHUB_JOB": job,
        "GITHUB_REF": C0_REF,
        "GITHUB_REF_NAME": C0_TAG,
        "GITHUB_REF_PROTECTED": "true",
        "GITHUB_REF_TYPE": "tag",
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_REPOSITORY_ID": str(REPOSITORY_ID),
        "GITHUB_REPOSITORY_OWNER": OWNER_LOGIN,
        "GITHUB_REPOSITORY_OWNER_ID": str(OWNER_ID),
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "983421",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_TRIGGERING_ACTOR": OWNER_LOGIN,
        "GITHUB_WORKFLOW_REF": f"{REPOSITORY}/{workflow}@{C0_REF}",
        "GITHUB_WORKFLOW_SHA": "a" * 40,
        "RUNNER_ARCH": "X64",
        "RUNNER_ENVIRONMENT": "github-hosted",
        "RUNNER_OS": "Linux",
    }


def _bundle(statement: object) -> bytes:
    return _canonical(
        {
            "dsseEnvelope": {
                "payload": base64.b64encode(_canonical(statement)).decode("ascii"),
                "payloadType": "application/vnd.in-toto+json",
                "signatures": [{"keyid": "", "sig": "c2ln"}],
            },
            "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
            "verificationMaterial": {
                "certificate": {"rawBytes": "Y2VydA=="},
                "tlogEntries": [
                    {
                        "canonicalizedBody": base64.b64encode(b"rekor-body").decode("ascii"),
                        "inclusionPromise": {
                            "signedEntryTimestamp": base64.b64encode(b"set").decode("ascii")
                        },
                        "integratedTime": 1_750_000_000,
                        "kindVersion": {"kind": "dsse", "version": "0.0.1"},
                        "logId": {"keyId": base64.b64encode(b"k" * 32).decode("ascii")},
                        "logIndex": 42,
                    }
                ],
            },
        }
    )


def _statement(target: SuiteStateRecord, predicate: object) -> dict[str, object]:
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "predicate": predicate,
        "predicateType": ONLINE_COMPLETION_PREDICATE_TYPE,
        "subject": [
            {
                "digest": {"sha256": target.record_sha256},
                "name": "prepared-subject.json",
            }
        ],
    }


class _Verifier:
    def __init__(self, mutation: object | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.mutation = mutation

    def verify(
        self,
        *,
        subject_path: Path,
        bundle_path: Path,
        context: ProviderWorkflowContext,
        predicate_type: str,
    ) -> bytes:
        self.calls.append((subject_path.name, bundle_path.name, predicate_type))
        if callable(self.mutation):
            self.mutation()
        return b'[{"verificationResult":{"verified":true}}]'


def _materialize_evidence(
    root: Path,
    target: SuiteStateRecord,
    *,
    statement_mutator: object | None = None,
) -> tuple[Path, Path, Path, dict[str, object]]:
    predicate = {
        "schema_version": "fractal-provider-transition-test-v1",
        "state_record_sha256": target.record_sha256,
    }
    statement = _statement(target, predicate)
    if callable(statement_mutator):
        statement_mutator(statement)
    subject_path = root / "prepared-subject.json"
    predicate_path = root / "completion-predicate.json"
    bundle_path = root / "attestation.sigstore.bundle.json"
    subject_path.write_bytes(target.canonical_bytes() + b"\n")
    predicate_path.write_bytes(_canonical(predicate) + b"\n")
    bundle_path.write_bytes(_bundle(statement))
    return subject_path, predicate_path, bundle_path, predicate


def _publication_receipt(target: SuiteStateRecord) -> LedgerPublicationReceipt:
    return LedgerPublicationReceipt(
        repository=REPOSITORY,
        state_key=f"refs/heads/confirmatory-ledger/{target.suite_attempt_id}",
        ruleset_id=9001,
        commit_oid="d" * 40,
        previous_commit_oid="c" * 40,
        tree_oid="e" * 40,
        blob_oid="f" * 40,
        state_path=(
            f"confirmatory-ledger/{target.suite_attempt_id}/{target.sequence:03d}.state.json"
        ),
        state_record_sha256=target.record_sha256,
        state_sequence=target.sequence,
        suite_attempt_id=target.suite_attempt_id,
    )


def test_verified_hosted_attestation_is_cas_published_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    authority, claimed = _predecessor()
    target = _target(claimed)
    subject, predicate, bundle, _ = _materialize_evidence(root, target)
    output = root / "publication"
    output.mkdir(mode=0o700)
    verifier = _Verifier()
    calls: list[tuple[str, str]] = []

    def fake_publish(**kwargs: object) -> tuple[LedgerPublicationReceipt, bool]:
        assert kwargs["target"] is target
        assert kwargs["expected_predecessor_commit"] == "c" * 40
        receipt = _publication_receipt(target)
        receipt_path = Path(kwargs["receipt_path"])
        receipt_path.write_bytes(receipt.canonical_bytes())
        calls.append((receipt_path.name, str(kwargs["expected_predecessor_commit"])))
        return receipt, True

    monkeypatch.setattr(
        publication_module,
        "publish_candidate_ledger_transition",
        fake_publish,
    )
    context = ProviderWorkflowContext.from_environment("online", _environment())
    result = verify_and_publish_provider_transition(
        context=context,
        phase="online",
        mode="completion",
        suite_attempt_id=target.suite_attempt_id,
        predecessor=authority,
        target=target,
        subject_path=subject,
        predicate_path=predicate,
        bundle_path=bundle,
        output_dir=output,
        github_api=object(),
        verifier=verifier,
    )

    assert result.state is target
    assert result.publication_receipt.commit_oid == "d" * 40
    assert result.attestation.subject_sha256 == target.record_sha256
    assert result.attestation.rekor_log_index == 42
    assert verifier.calls == [
        (
            "prepared-subject.json",
            "transition.sigstore.bundle.json",
            ONLINE_COMPLETION_PREDICATE_TYPE,
        )
    ]
    assert calls == [("ledger-publication-receipt.json", "c" * 40)]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda row: row.update({"predicateType": "https://attacker.invalid/type"}),
            "statement differs",
        ),
        (
            lambda row: row["subject"][0].update({"name": "caller-selected.json"}),
            "statement differs",
        ),
        (
            lambda row: row["subject"][0]["digest"].update({"sha256": _digest("other")}),
            "statement differs",
        ),
        (
            lambda row: row.update({"predicate": {"substituted": True}}),
            "statement differs",
        ),
    ),
)
def test_verified_bundle_cannot_substitute_statement_fields(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    authority, claimed = _predecessor()
    target = _target(claimed)
    subject, predicate, bundle, _ = _materialize_evidence(
        tmp_path.resolve(),
        target,
        statement_mutator=mutation,
    )
    output = tmp_path / "publication"
    output.mkdir(mode=0o700)
    context = ProviderWorkflowContext.from_environment("online", _environment())
    with pytest.raises(ProviderTransitionPublicationError, match=message):
        verify_and_publish_provider_transition(
            context=context,
            phase="online",
            mode="completion",
            suite_attempt_id=target.suite_attempt_id,
            predecessor=authority,
            target=target,
            subject_path=subject,
            predicate_path=predicate,
            bundle_path=bundle,
            output_dir=output,
            github_api=object(),
            verifier=_Verifier(),
        )


def test_context_state_machine_and_fresh_authority_are_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = 0

    def revalidate() -> None:
        nonlocal reads
        reads += 1
        if reads == 2:
            raise ValueError("claim lost")

    authority, claimed = _predecessor(revalidator=revalidate)
    target = _target(claimed)
    subject, predicate, bundle, _ = _materialize_evidence(tmp_path.resolve(), target)
    output = tmp_path / "publication"
    output.mkdir(mode=0o700)
    monkeypatch.setattr(
        publication_module,
        "publish_candidate_ledger_transition",
        lambda **kwargs: pytest.fail("CAS must not run after claim loss"),
    )
    context = ProviderWorkflowContext.from_environment("online", _environment())
    with pytest.raises(ValueError, match="fresh provider predecessor revalidation failed"):
        verify_and_publish_provider_transition(
            context=context,
            phase="online",
            mode="completion",
            suite_attempt_id=target.suite_attempt_id,
            predecessor=authority,
            target=target,
            subject_path=subject,
            predicate_path=predicate,
            bundle_path=bundle,
            output_dir=output,
            github_api=object(),
            verifier=_Verifier(),
        )
    assert reads == 2

    failure_context = ProviderWorkflowContext.from_environment(
        "online",
        _environment(job="fail"),
    )
    stable_authority, _ = _predecessor()
    with pytest.raises(ProviderTransitionPublicationError, match="context differs"):
        verify_and_publish_provider_transition(
            context=failure_context,
            phase="online",
            mode="completion",
            suite_attempt_id=target.suite_attempt_id,
            predecessor=stable_authority,
            target=target,
            subject_path=subject,
            predicate_path=predicate,
            bundle_path=bundle,
            output_dir=output,
            github_api=object(),
            verifier=_Verifier(),
        )


def test_original_evidence_mutation_after_verification_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, claimed = _predecessor()
    target = _target(claimed)
    subject, predicate, bundle, predicate_row = _materialize_evidence(tmp_path.resolve(), target)
    output = tmp_path / "publication"
    output.mkdir(mode=0o700)

    def mutate() -> None:
        predicate.write_bytes(_canonical({**predicate_row, "late": True}) + b"\n")

    monkeypatch.setattr(
        publication_module,
        "publish_candidate_ledger_transition",
        lambda **kwargs: pytest.fail("CAS must not run after evidence mutation"),
    )
    context = ProviderWorkflowContext.from_environment("online", _environment())
    with pytest.raises(ProviderTransitionPublicationError, match="changed before publication"):
        verify_and_publish_provider_transition(
            context=context,
            phase="online",
            mode="completion",
            suite_attempt_id=target.suite_attempt_id,
            predecessor=authority,
            target=target,
            subject_path=subject,
            predicate_path=predicate,
            bundle_path=bundle,
            output_dir=output,
            github_api=object(),
            verifier=_Verifier(mutate),
        )


def test_linked_evidence_and_noncanonical_predicate_are_rejected(tmp_path: Path) -> None:
    authority, claimed = _predecessor()
    target = _target(claimed)
    subject, predicate, bundle, _ = _materialize_evidence(tmp_path.resolve(), target)
    output = tmp_path / "publication"
    output.mkdir(mode=0o700)
    linked = tmp_path / "linked-subject.json"
    os.link(subject, linked)
    context = ProviderWorkflowContext.from_environment("online", _environment())
    with pytest.raises(ProviderTransitionPublicationError, match="singly linked"):
        verify_and_publish_provider_transition(
            context=context,
            phase="online",
            mode="completion",
            suite_attempt_id=target.suite_attempt_id,
            predecessor=authority,
            target=target,
            subject_path=subject,
            predicate_path=predicate,
            bundle_path=bundle,
            output_dir=output,
            github_api=object(),
            verifier=_Verifier(),
        )

    linked.unlink()
    predicate.write_text(json.dumps({"not": "canonical"}, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ProviderTransitionPublicationError, match="not canonical"):
        verify_and_publish_provider_transition(
            context=context,
            phase="online",
            mode="completion",
            suite_attempt_id=target.suite_attempt_id,
            predecessor=authority,
            target=target,
            subject_path=subject,
            predicate_path=predicate,
            bundle_path=bundle,
            output_dir=output,
            github_api=object(),
            verifier=_Verifier(),
        )


def test_gh_verifier_pins_c0_hosted_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'[{"verificationResult":{"verified":true}}]',
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    context = ProviderWorkflowContext.from_environment("online", _environment())
    verifier = GhProviderTransitionAttestationVerifier(executable="/usr/local/bin/gh")
    verifier.verify(
        subject_path=tmp_path / "prepared-subject.json",
        bundle_path=tmp_path / "bundle.json",
        context=context,
        predicate_type=ONLINE_COMPLETION_PREDICATE_TYPE,
    )
    command = observed["command"]
    assert command[:3] == ["/usr/local/bin/gh", "attestation", "verify"]
    assert command[command.index("--repo") + 1] == REPOSITORY
    assert command[command.index("--source-ref") + 1] == C0_REF
    assert command[command.index("--source-digest") + 1] == "a" * 40
    assert command[command.index("--signer-digest") + 1] == "a" * 40
    assert "--deny-self-hosted-runners" in command
    assert command[command.index("--cert-identity") + 1] == (
        f"https://github.com/{context.workflow_ref}"
    )

    claim_context = ProviderWorkflowContext.from_environment(
        "online",
        _environment(job="claim"),
    )
    with pytest.raises(ProviderTransitionPublicationError, match="terminal-state job"):
        verifier.verify(
            subject_path=tmp_path / "prepared-subject.json",
            bundle_path=tmp_path / "bundle.json",
            context=claim_context,
            predicate_type=ONLINE_COMPLETION_PREDICATE_TYPE,
        )
