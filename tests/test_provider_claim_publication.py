from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.provider_claim_publication as publication_module
from fractal_ann_diagnostics.execution_claim import (
    C1_REGISTRATION_PACKAGE_FILE_COUNT,
    AnonymousZenodoAdmission,
    ExecutionClaimContract,
    PhaseClaimContract,
    ProviderExecutionIdentity,
    ProviderPhasePlan,
    ProviderRunnerReadinessReceipt,
)
from fractal_ann_diagnostics.github_state_attestation import LedgerPublicationReceipt
from fractal_ann_diagnostics.provider_claim_publication import (
    PROVIDER_CLAIM_PREDICATE_SCHEMA,
    PROVIDER_CLAIM_PREDICATE_TYPE,
    ProviderClaimPublicationError,
    derive_and_publish_provider_claim,
)
from fractal_ann_diagnostics.provider_workflow_orchestration import (
    C0_REF,
    OWNER_ID,
    OWNER_LOGIN,
    REPOSITORY,
    REPOSITORY_ID,
    DerivedPhaseClaim,
    ProviderWorkflowContext,
)
from fractal_ann_diagnostics.study import (
    FIXED_CORPORA,
    PROVIDER_PHASE_JOB_NAMES,
    PROVIDER_PHASE_WORKFLOWS,
    VerifiedC1ProtocolRegistration,
)
from fractal_ann_diagnostics.suite_attempt import SuiteStateRecord, VerifiedProviderPredecessor


def _digest(seed: str | bytes) -> str:
    encoded = seed if isinstance(seed, bytes) else seed.encode()
    return hashlib.sha256(encoded).hexdigest()


def _context(phase: str = "online") -> ProviderWorkflowContext:
    workflow = PROVIDER_PHASE_WORKFLOWS[phase]
    return ProviderWorkflowContext.from_environment(
        phase,  # type: ignore[arg-type]
        {
            "CI": "true",
            "GITHUB_ACTIONS": "true",
            "GITHUB_ACTOR": OWNER_LOGIN,
            "GITHUB_ACTOR_ID": str(OWNER_ID),
            "GITHUB_API_URL": "https://api.github.com",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_GRAPHQL_URL": "https://api.github.com/graphql",
            "GITHUB_JOB": "claim",
            "GITHUB_REF": C0_REF,
            "GITHUB_REF_NAME": "confirmatory-apparatus-c0",
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
        },
    )


def _plan() -> ProviderPhasePlan:
    plan = object.__new__(ProviderPhasePlan)
    for name, value in {
        "phase": "online",
        "c1_commit": "b" * 40,
        "manifest_sha256": _digest("manifest"),
        "claim_nonce": _digest("nonce"),
    }.items():
        object.__setattr__(plan, name, value)
    return plan


def _registration(tmp_path: Path) -> VerifiedC1ProtocolRegistration:
    registration = object.__new__(VerifiedC1ProtocolRegistration)
    object.__setattr__(registration, "package_root", tmp_path / "c1")
    object.__setattr__(registration, "c1_commit", "b" * 40)
    return registration


def _state(
    *,
    namespace: Path,
    manifest_sha256: str,
    sequence: int,
    state: str,
    previous: str | None,
) -> SuiteStateRecord:
    record = object.__new__(SuiteStateRecord)
    attempt = _digest(b"fractal-suite-attempt-v1\0" + manifest_sha256.encode("ascii"))
    for name, value in {
        "manifest_sha256": manifest_sha256,
        "namespace_uri": namespace.as_uri(),
        "payload": SimpleNamespace(to_dict=lambda: {"fixture": state}),
        "previous_state_record_sha256": previous,
        "run_receipt_sha256": _digest("run-receipt"),
        "schema_version": "fractal-suite-state-v1",
        "sequence": sequence,
        "state": state,
        "suite_attempt_id": attempt,
    }.items():
        object.__setattr__(record, name, value)
    return record


def _predecessor(tmp_path: Path, manifest_sha256: str) -> VerifiedProviderPredecessor:
    attempt = _digest(b"fractal-suite-attempt-v1\0" + manifest_sha256.encode("ascii"))
    namespace = tmp_path / f"suite-attempt-{attempt}"
    opened = _state(
        namespace=namespace,
        manifest_sha256=manifest_sha256,
        sequence=0,
        state="OPENED",
        previous=None,
    )
    predecessor = object.__new__(VerifiedProviderPredecessor)
    object.__setattr__(predecessor, "records", (opened,))
    object.__setattr__(predecessor, "evidences", (SimpleNamespace(transition_id="c" * 40),))
    return predecessor


def _contract(tmp_path: Path) -> ExecutionClaimContract:
    contract = object.__new__(ExecutionClaimContract)
    object.__setattr__(
        contract,
        "corpora",
        tuple(
            SimpleNamespace(
                corpus_id=corpus_id,
                staging_namespace_uri=(tmp_path / "staging" / corpus_id).as_uri(),
            )
            for corpus_id in sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8"))
        ),
    )
    return contract


def _identity(*, claim_job_id: int = 7788) -> ProviderExecutionIdentity:
    return ProviderExecutionIdentity(
        repository=REPOSITORY,
        workflow_path=PROVIDER_PHASE_WORKFLOWS["online"],
        workflow_ref=(f"{REPOSITORY}/{PROVIDER_PHASE_WORKFLOWS['online']}@{C0_REF}"),
        workflow_sha="a" * 40,
        run_head_branch="confirmatory-apparatus-c0",
        run_id=983421,
        run_attempt=1,
        claim_job_id=claim_job_id,
        claim_job_name=PROVIDER_PHASE_JOB_NAMES["online"][0],
        execute_job_name=PROVIDER_PHASE_JOB_NAMES["online"][1],
        runner_id=991,
        runner_name="fractal-confirmatory-online",
        runner_group_id=None,
        runner_label="fractal-ann-confirmatory-0123456789abcdef",
        runner_version="2.335.1",
        runner_archive_sha256=_digest("runner-archive"),
        provider_operating_system="macOS",
        provider_architecture="ARM64",
        host_tool_contract_sha256=_digest("host-tools"),
        runtime_probe_receipt_sha256=_digest("runtime-probe"),
        self_hosted=True,
    )


def _readiness() -> ProviderRunnerReadinessReceipt:
    return ProviderRunnerReadinessReceipt(
        provider_plan_sha256=_digest("plan"),
        bootstrap_receipt_file_sha256=_digest("bootstrap"),
        runner_id=991,
        runner_name="fractal-confirmatory-online",
        runner_group_id=None,
        status="offline",
        busy=False,
        labels=("ARM64", "macOS", "self-hosted"),
        verified_at_utc="2026-07-17T12:00:00+00:00",
    )


def _zenodo() -> AnonymousZenodoAdmission:
    return AnonymousZenodoAdmission(
        record_id=21361837,
        doi="10.5281/zenodo.21361837",
        record_uri="https://zenodo.org/records/21361837",
        published_at_utc="2026-07-14T12:01:00+00:00",
        file_count=C1_REGISTRATION_PACKAGE_FILE_COUNT,
        package_tree_sha256=_digest("zenodo-tree"),
        package_aggregate_sha256=_digest("zenodo-aggregate"),
        receipt_file_sha256=_digest("zenodo-receipt"),
        verified_at_utc="2026-07-14T12:01:01+00:00",
    )


def _publication(target: SuiteStateRecord) -> LedgerPublicationReceipt:
    return LedgerPublicationReceipt(
        repository=REPOSITORY,
        state_key=f"refs/heads/confirmatory-ledger/{target.suite_attempt_id}",
        ruleset_id=771,
        commit_oid="d" * 40,
        previous_commit_oid="c" * 40,
        tree_oid="e" * 40,
        blob_oid="f" * 40,
        state_path=f"suite-attempts/{target.suite_attempt_id}/001.state.json",
        state_record_sha256=target.record_sha256,
        state_sequence=target.sequence,
        suite_attempt_id=target.suite_attempt_id,
    )


class _Api:
    pass


def _install_online_stubs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    plan = _plan()
    registration = _registration(tmp_path)
    predecessor = _predecessor(tmp_path, plan.manifest_sha256)
    contract = _contract(tmp_path)
    identity = _identity()
    readiness = _readiness()
    target = _state(
        namespace=Path(predecessor.state.namespace_uri.removeprefix("file://")),
        manifest_sha256=plan.manifest_sha256,
        sequence=1,
        state="RUN_CLAIMED",
        previous=predecessor.state.record_sha256,
    )
    calls = {"registration": 0, "predecessor": 0, "identity": 0, "runner": 0, "publish": 0}

    def registration_current(_self: object) -> None:
        calls["registration"] += 1

    def predecessor_current(_self: object) -> None:
        calls["predecessor"] += 1

    def identity_verify(**_kwargs: object) -> ProviderExecutionIdentity:
        calls["identity"] += 1
        return identity

    def runner_verify(**_kwargs: object) -> ProviderRunnerReadinessReceipt:
        calls["runner"] += 1
        return readiness

    def publish(**kwargs: object) -> tuple[LedgerPublicationReceipt, bool]:
        calls["publish"] += 1
        assert kwargs["target"] is target
        assert kwargs["expected_predecessor_commit"] == "c" * 40
        receipt_path = kwargs["receipt_path"]
        assert isinstance(receipt_path, Path)
        receipt = _publication(target)
        receipt_path.write_bytes(receipt.canonical_bytes())
        return receipt, True

    monkeypatch.setattr(VerifiedC1ProtocolRegistration, "assert_current", registration_current)
    monkeypatch.setattr(VerifiedProviderPredecessor, "assert_current", predecessor_current)
    monkeypatch.setattr(publication_module, "_current_c1_plan", lambda *_args: plan)
    monkeypatch.setattr(
        publication_module,
        "derive_execution_claim_contract_from_provider_opened",
        lambda **_kwargs: contract,
    )
    monkeypatch.setattr(publication_module, "verify_provider_execution_identity", identity_verify)
    monkeypatch.setattr(publication_module, "verify_provider_runner_ready", runner_verify)
    monkeypatch.setattr(
        publication_module,
        "claim_online_provider_candidate",
        lambda *_args, **_kwargs: target,
    )
    monkeypatch.setattr(publication_module, "publish_candidate_ledger_transition", publish)
    monkeypatch.setattr(
        ExecutionClaimContract,
        "contract_sha256",
        property(lambda _self: _digest("contract")),
    )
    monkeypatch.setattr(ProviderExecutionIdentity, "matches_contract", lambda *_args: None)
    monkeypatch.setattr(
        ProviderPhasePlan,
        "plan_sha256",
        property(lambda _self: _digest("plan")),
    )
    return plan, registration, predecessor, identity, readiness, target, calls


def _run(
    *,
    tmp_path: Path,
    plan: ProviderPhasePlan,
    registration: VerifiedC1ProtocolRegistration,
    predecessor: VerifiedProviderPredecessor,
):
    output = tmp_path / "claim"
    output.mkdir(mode=0o700)
    return derive_and_publish_provider_claim(
        registration=registration,
        plan=plan,
        predecessor=predecessor,
        zenodo_admission=_zenodo(),
        context=_context(),
        c1_manifest_rekor_integrated_at_utc="2026-07-14T12:00:00+00:00",
        c1_registry_rekor_integrated_at_utc="2026-07-14T12:00:01+00:00",
        output_dir=output,
        github_api=_Api(),  # type: ignore[arg-type]
    )


def test_online_claim_reverifies_before_cas_and_writes_exact_evidence_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, registration, predecessor, identity, readiness, target, calls = _install_online_stubs(
        monkeypatch, tmp_path
    )
    result = _run(
        tmp_path=tmp_path,
        plan=plan,
        registration=registration,
        predecessor=predecessor,
    )

    assert result.state is target
    assert result.provider_identity == identity
    assert result.runner_readiness == readiness
    assert result.predicate_type == PROVIDER_CLAIM_PREDICATE_TYPE
    assert result.subject_path.read_bytes() == target.canonical_bytes() + b"\n"
    assert result.subject_sha256 == target.record_sha256
    predicate_bytes = result.predicate_path.read_bytes()
    assert result.predicate_sha256 == _digest(predicate_bytes)
    assert json.loads(predicate_bytes)["schema_version"] == PROVIDER_CLAIM_PREDICATE_SCHEMA
    assert set(result.output_paths) == set(FIXED_CORPORA)
    assert not result.input_paths and not result.supporting_input_paths
    assert calls == {
        "registration": 2,
        "predecessor": 2,
        "identity": 2,
        "runner": 2,
        "publish": 1,
    }


def test_c1_plan_swap_after_evidence_write_blocks_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, registration, predecessor, _identity_value, _readiness_value, _target, calls = (
        _install_online_stubs(monkeypatch, tmp_path)
    )
    reads = 0

    def swap(_registration: object, _plan: object) -> ProviderPhasePlan:
        nonlocal reads
        reads += 1
        if reads == 2:
            raise ProviderClaimPublicationError(
                "provider plan differs from the current verified C1"
            )
        return plan

    monkeypatch.setattr(publication_module, "_current_c1_plan", swap)
    with pytest.raises(ProviderClaimPublicationError, match="plan differs"):
        _run(
            tmp_path=tmp_path,
            plan=plan,
            registration=registration,
            predecessor=predecessor,
        )
    assert calls["publish"] == 0


def test_live_identity_change_after_evidence_write_blocks_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, registration, predecessor, identity, _readiness_value, _target, calls = (
        _install_online_stubs(monkeypatch, tmp_path)
    )
    identities = iter((identity, _identity(claim_job_id=8899)))
    monkeypatch.setattr(
        publication_module,
        "verify_provider_execution_identity",
        lambda **_kwargs: next(identities),
    )
    with pytest.raises(ProviderClaimPublicationError, match="identity or runner changed"):
        _run(
            tmp_path=tmp_path,
            plan=plan,
            registration=registration,
            predecessor=predecessor,
        )
    assert calls["publish"] == 0


def test_publication_readback_for_another_predecessor_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, registration, predecessor, _identity_value, _readiness_value, target, _calls = (
        _install_online_stubs(monkeypatch, tmp_path)
    )

    def wrong_readback(**kwargs: object) -> tuple[LedgerPublicationReceipt, bool]:
        receipt_path = kwargs["receipt_path"]
        assert isinstance(receipt_path, Path)
        receipt = _publication(target)
        wrong = LedgerPublicationReceipt(**{**receipt.__dict__, "previous_commit_oid": "9" * 40})
        receipt_path.write_bytes(wrong.canonical_bytes())
        return wrong, True

    monkeypatch.setattr(publication_module, "publish_candidate_ledger_transition", wrong_readback)
    with pytest.raises(ProviderClaimPublicationError, match="readback differs"):
        _run(
            tmp_path=tmp_path,
            plan=plan,
            registration=registration,
            predecessor=predecessor,
        )


def test_uncontrolled_or_reused_output_path_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, registration, predecessor, *_rest = _install_online_stubs(monkeypatch, tmp_path)
    output = tmp_path / "claim"
    output.mkdir(mode=0o700)
    (output / "claim-subject.json").write_text("attacker-controlled", encoding="utf-8")
    with pytest.raises(ProviderClaimPublicationError, match="create provider claim subject once"):
        derive_and_publish_provider_claim(
            registration=registration,
            plan=plan,
            predecessor=predecessor,
            zenodo_admission=_zenodo(),
            context=_context(),
            c1_manifest_rekor_integrated_at_utc="2026-07-14T12:00:00+00:00",
            c1_registry_rekor_integrated_at_utc="2026-07-14T12:00:01+00:00",
            output_dir=output,
            github_api=_Api(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("phase", "predecessor_state", "predecessor_sequence", "target_state", "target_sequence"),
    (
        ("label-release", "ONLINE_COMPLETE", 2, "LABEL_RELEASE_CLAIMED", 3),
        ("analysis", "LABELS_RELEASED", 4, "ANALYSIS_CLAIMED", 5),
    ),
)
def test_post_online_phases_publish_their_derived_path_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    predecessor_state: str,
    predecessor_sequence: int,
    target_state: str,
    target_sequence: int,
) -> None:
    plan = _plan()
    object.__setattr__(plan, "phase", phase)
    registration = _registration(tmp_path)
    namespace = tmp_path / f"suite-attempt-{_digest('post-attempt')}"
    predecessor_record = _state(
        namespace=namespace,
        manifest_sha256=plan.manifest_sha256,
        sequence=predecessor_sequence,
        state=predecessor_state,
        previous=_digest("earlier-state"),
    )
    predecessor = object.__new__(VerifiedProviderPredecessor)
    object.__setattr__(predecessor, "records", (predecessor_record,))
    object.__setattr__(predecessor, "evidences", (SimpleNamespace(transition_id="c" * 40),))
    contract = object.__new__(PhaseClaimContract)
    ordered = tuple(sorted(FIXED_CORPORA, key=lambda value: value.encode("utf-8")))
    derived = DerivedPhaseClaim(
        contract=contract,
        input_paths=tuple((name, str(tmp_path / "input" / name)) for name in ordered),
        supporting_input_paths=tuple((name, str(tmp_path / "support" / name)) for name in ordered),
        output_paths=tuple((name, str(tmp_path / "output" / name)) for name in ordered),
    )
    target = _state(
        namespace=namespace,
        manifest_sha256=plan.manifest_sha256,
        sequence=target_sequence,
        state=target_state,
        previous=predecessor_record.record_sha256,
    )
    identity = _identity()
    captured: dict[str, object] = {}

    monkeypatch.setattr(VerifiedC1ProtocolRegistration, "assert_current", lambda _self: None)
    monkeypatch.setattr(VerifiedProviderPredecessor, "assert_current", lambda _self: None)
    monkeypatch.setattr(publication_module, "_current_c1_plan", lambda *_args: plan)
    monkeypatch.setattr(publication_module, "_assert_registration_evidence", lambda **_kwargs: None)
    monkeypatch.setattr(
        publication_module,
        "derive_post_online_phase_claim",
        lambda **_kwargs: derived,
    )
    monkeypatch.setattr(
        publication_module,
        "verify_provider_execution_identity",
        lambda **_kwargs: identity,
    )
    monkeypatch.setattr(
        publication_module,
        "verify_provider_runner_ready",
        lambda **_kwargs: _readiness(),
    )
    monkeypatch.setattr(ProviderExecutionIdentity, "matches_phase_contract", lambda *_args: None)
    monkeypatch.setattr(
        PhaseClaimContract,
        "contract_sha256",
        property(lambda _self: _digest("phase-contract")),
    )
    monkeypatch.setattr(
        ProviderPhasePlan,
        "plan_sha256",
        property(lambda _self: _digest("plan")),
    )
    monkeypatch.setattr(publication_module, "load_study_manifest", lambda _path: {})

    def candidate(*_args: object, **kwargs: object) -> SuiteStateRecord:
        captured.update(kwargs)
        return target

    if phase == "label-release":
        monkeypatch.setattr(publication_module, "claim_label_release_provider_candidate", candidate)
    else:
        monkeypatch.setattr(publication_module, "claim_analysis_provider_candidate", candidate)

    def publish(**kwargs: object) -> tuple[LedgerPublicationReceipt, bool]:
        receipt = _publication(target)
        path = kwargs["receipt_path"]
        assert isinstance(path, Path)
        path.write_bytes(receipt.canonical_bytes())
        return receipt, True

    monkeypatch.setattr(publication_module, "publish_candidate_ledger_transition", publish)
    output = tmp_path / "claim"
    output.mkdir(mode=0o700)
    result = derive_and_publish_provider_claim(
        registration=registration,
        plan=plan,
        predecessor=predecessor,
        zenodo_admission=_zenodo(),
        context=_context(phase),
        c1_manifest_rekor_integrated_at_utc="2026-07-14T12:00:00+00:00",
        c1_registry_rekor_integrated_at_utc="2026-07-14T12:00:01+00:00",
        output_dir=output,
        github_api=_Api(),  # type: ignore[arg-type]
    )

    assert set(result.input_paths) == set(FIXED_CORPORA)
    assert set(result.supporting_input_paths) == set(FIXED_CORPORA)
    assert set(result.output_paths) == set(FIXED_CORPORA)
    assert captured["phase_contract"] is contract
    if phase == "label-release":
        assert captured["ciphertext_paths"] == dict(result.input_paths)
        assert captured["encryption_receipt_paths"] == dict(result.supporting_input_paths)
