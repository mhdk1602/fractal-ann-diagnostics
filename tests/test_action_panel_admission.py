from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from fractal_ann_diagnostics.action_panel_admission import (
    AdmittedActionPanel,
    FailedActionExecution,
    GovernedActionExecution,
    action_panel_from_governed_executions,
)
from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    VerifiedArtifact,
)
from fractal_ann_diagnostics.audit import (
    AuditRecord,
    EvidenceAudit,
    VerifiedProvenanceRegistry,
    audit_record_from_governed_result,
)
from fractal_ann_diagnostics.confirmatory_analysis import (
    ActionPanelAdmissionReceipt,
    ConfirmatoryAnalysisError,
    load_action_panel_admission_receipt,
    loads_action_panel_admission_receipt,
    write_action_panel_admission_receipt,
)
from fractal_ann_diagnostics.controller import ControllerDecision, GovernedResult
from fractal_ann_diagnostics.corpora import (
    CorpusDocument,
    EvidenceQuery,
    NormalizedCorpus,
)
from fractal_ann_diagnostics.label_separation import (
    OnlineDocument,
    OnlineExecutionArtifact,
    OnlineTrial,
)
from fractal_ann_diagnostics.policy import PolicyDecision
from fractal_ann_diagnostics.retrieval import SearchResult, SearchWork
from fractal_ann_diagnostics.study import SealedRunReceipt

_MANIFEST = "a" * 64
_TRIAL_KEY = sha256(b"panel-trial").hexdigest()
_FAMILY_KEY = sha256(b"panel-family").hexdigest()
_ACTIONS = ("hnsw-low", "hnsw-high", "exact-authorized", "abstain")
_PSEUDONYM_KEY = b"panel-admission-test-key-material-32-bytes"
_COMPONENTS = (
    "application",
    "controller",
    "corpus",
    "embedding",
    "index",
    "policy",
)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _corpus() -> NormalizedCorpus:
    documents = tuple(
        CorpusDocument(
            document_id=document_id,
            external_id=f"document-{document_id}",
            title=f"Document {document_id}",
            text=f"text-{document_id}",
            source_uri=f"fixture://document/{document_id}",
            content_hash=f"sha256:{_digest(f'text-{document_id}')}",
        )
        for document_id in range(3)
    )
    return NormalizedCorpus(
        name="scifact",
        stage="sealed",
        documents=documents,
        queries=(
            EvidenceQuery(
                query_id="query-1",
                query_family="family-1",
                text="Which documents match?",
                corpus="scifact",
                stage="sealed",
                answer=None,
                gold_evidence=None,
            ),
        ),
    )


def _verified_artifact(component: str) -> VerifiedArtifact:
    digest = _digest(component)
    return VerifiedArtifact(
        artifact_id=f"{component}-artifact",
        relative_path=f"artifacts/{component}.bin",
        kind="file",
        exact=True,
        expected_sha256=digest,
        verified_sha256=digest,
        file_count=1,
        directory_count=0,
        byte_count=1,
        observed_file_count=1,
        observed_directory_count=0,
        observed_byte_count=1,
    )


def _registry() -> VerifiedProvenanceRegistry:
    receipt = ArtifactVerificationReceipt(
        manifest_sha256=_MANIFEST,
        artifacts=tuple(_verified_artifact(component) for component in _COMPONENTS),
    )
    return VerifiedProvenanceRegistry(
        corpus=_corpus(),
        verification_receipt=receipt,
        component_artifact_ids={
            component: f"{component}-artifact" for component in _COMPONENTS
        },
    )


def _policy_decision(
    registry: VerifiedProvenanceRegistry,
    *,
    suffix: str,
    mask: np.ndarray | None = None,
) -> PolicyDecision:
    return PolicyDecision(
        subject="analyst",
        action="retrieve",
        policy_version="policy-1",
        authorized_mask=(
            np.array([True, False, True]) if mask is None else np.asarray(mask)
        ),
        decision_id=f"decision-{suffix}",
        document_universe_sha256=registry.document_universe_sha256,
        request_nonce=f"nonce-{suffix}",
    )


def _result(
    action: str,
    registry: VerifiedProvenanceRegistry,
    *,
    mask: np.ndarray | None = None,
) -> GovernedResult:
    initial = _policy_decision(registry, suffix=f"{action}-initial", mask=mask)
    decision = ControllerDecision(
        action=action,  # type: ignore[arg-type]
        risk_score=0.2,
        reasons=(f"execute {action}",),
        policy_version="policy-1",
    )
    if action == "abstain":
        return GovernedResult(
            decision=decision,
            geometry=None,
            search=None,
            initial_authorization=initial,
            request_latency_ms=1.0,
        )
    final = _policy_decision(registry, suffix=f"{action}-final", mask=mask)
    search = SearchResult(
        ids=np.array([0, 2]),
        distances=np.array([0.1, 0.2]),
        strategy=action,
        requested_k=2,
        candidates_examined=16,
        unauthorized_candidates=0,
        unauthorized_context=0,
        latency_ms=0.5,
        work=SearchWork(returned_candidates=2, configured_ef_search=16),
    )
    return GovernedResult(
        decision=decision,
        geometry=None,
        search=search,
        initial_authorization=initial,
        final_authorization=final,
        request_latency_ms=2.0,
    )


def _trial() -> OnlineTrial:
    return OnlineTrial(
        trial_key=_TRIAL_KEY,
        family_key=_FAMILY_KEY,
        text="Which documents match?",
        corpus="scifact",
        stage="sealed",
    )


def _execution() -> OnlineExecutionArtifact:
    corpus = _corpus()
    return OnlineExecutionArtifact(
        key_id="custody-key",
        corpus="scifact",
        stage="sealed",
        documents=tuple(
            OnlineDocument(
                document_id=document.document_id,
                external_id=document.external_id,
                title=document.title,
                text=document.text,
                source_uri=document.source_uri,
                content_hash=document.content_hash,
            )
            for document in corpus.documents
        ),
        trials=(_trial(),),
    )


def _run_receipt() -> SealedRunReceipt:
    return SealedRunReceipt(
        manifest_sha256=_MANIFEST,
        protocol_version="0.3.0",
        started_at_utc="2026-07-13T22:00:00+00:00",
        runner_identity="test-runner",
        code_commit="c" * 40,
        runner_image=f"ghcr.io/example/runner@sha256:{'d' * 64}",
        protocol_registration_receipt_uri="file:///protocol-receipt.json",
        protocol_registration_receipt_sha256="e" * 64,
        protocol_registration_record_uri="file:///protocol-record.json",
        verification_receipt_uri="file:///verification-receipt.json",
        verification_receipt_sha256="f" * 64,
        receipt_uri="file:///run-receipt.json",
    )


def _audit(
    result: GovernedResult,
    registry: VerifiedProvenanceRegistry,
    *,
    previous_record: AuditRecord | None = None,
) -> AuditRecord:
    return audit_record_from_governed_result(
        result,
        request_id=f"request-{result.decision.action}",
        trace_id=f"trace-{result.decision.action}",
        trial_sha256=_TRIAL_KEY,
        subject="analyst",
        pseudonym_key=_PSEUDONYM_KEY,
        pseudonym_key_id="pseudonym-key",
        provenance_registry=registry,
        occurred_at="2026-07-13T22:00:01+00:00",
        previous_record=previous_record,
    )


def _admitted_actions(
    registry: VerifiedProvenanceRegistry,
    actions: tuple[str, ...] = _ACTIONS,
    *,
    masks: dict[str, np.ndarray] | None = None,
) -> tuple[GovernedActionExecution, ...]:
    admitted: list[GovernedActionExecution] = []
    previous: AuditRecord | None = None
    for action in actions:
        result = _result(
            action,
            registry,
            mask=None if masks is None else masks.get(action),
        )
        record = _audit(result, registry, previous_record=previous)
        admitted.append(
            GovernedActionExecution(
                trial=_trial(),
                result=result,
                audit_record=record,
                feature_values=(1.0, "scifact") if action == "hnsw-low" else None,
            )
        )
        previous = record
    return tuple(admitted)


def _failed_action(
    action: str,
    registry: VerifiedProvenanceRegistry,
    *,
    latency_ms: float,
    feature_values: tuple[object, ...] | None = None,
    mask: np.ndarray | None = None,
) -> FailedActionExecution:
    result = _result(action, registry, mask=mask)
    authorization = result.final_authorization or result.initial_authorization
    started = 1_000_000_000
    return FailedActionExecution(
        trial=_trial(),
        decision=result.decision,
        authorization=authorization,
        failure_code="backend-timeout",
        started_monotonic_ns=started,
        finished_monotonic_ns=started + int(latency_ms * 1_000_000),
        runner_identity=_run_receipt().runner_identity,
        feature_values=feature_values,
    )


def _factory_anchors(
    admitted: tuple[GovernedActionExecution, ...],
) -> dict[str, object]:
    head = max(admitted, key=lambda item: item.audit_record.sequence).audit_record
    return {
        "expected_audit_head_sha256": head.record_sha256,
        "query_partition_audit_sha256": _digest("partition-audit"),
        "partition_label": "primary",
    }


def _rehash(record: AuditRecord, **updates: object) -> AuditRecord:
    changed = replace(record, record_sha256="0" * 64, **updates)
    return replace(changed, record_sha256=changed.computed_record_sha256())


def test_panel_factory_derives_governed_outputs_and_binds_each_audit_record() -> None:
    registry = _registry()
    admitted = _admitted_actions(registry)
    selection = admitted[0].result.decision
    admitted_panel = action_panel_from_governed_executions(
        execution=_execution(),
        run_receipt=_run_receipt(),
        governed_executions=admitted,
        selected_decisions={_TRIAL_KEY: selection},
        action_set=_ACTIONS,
        **_factory_anchors(admitted),
    )
    panel = admitted_panel.panel

    assert panel.execution_artifact_sha256 == _execution().artifact_sha256
    assert admitted_panel.admission_receipt.action_panel_artifact_sha256 == (
        panel.artifact_sha256
    )
    assert admitted_panel.admission_receipt.partition_label == "primary"
    assert [row.action for row in panel.rows] == list(_ACTIONS)
    for row, source in zip(panel.rows, admitted, strict=True):
        assert row.audit_record_sha256 == source.audit_record.record_sha256
        assert row.request_latency_ms == source.result.total_online_latency_ms
        assert row.returned_document_ids == source.returned_document_ids
        assert row.entitlement_violations == source.entitlement_violations == 0
    assert [row.action for row in panel.rows if row.controller_selected] == ["hnsw-low"]


def test_failed_action_is_retained_without_ids_entitlements_or_audit_claim() -> None:
    registry = _registry()
    admitted = _admitted_actions(
        registry,
        tuple(action for action in _ACTIONS if action != "hnsw-high"),
    )
    selection = admitted[0].result.decision
    failed = _failed_action("hnsw-high", registry, latency_ms=13.5)
    admitted_panel = action_panel_from_governed_executions(
        execution=_execution(),
        run_receipt=_run_receipt(),
        governed_executions=admitted,
        failed_executions=(failed,),
        selected_decisions={_TRIAL_KEY: selection},
        action_set=_ACTIONS,
        **_factory_anchors(admitted),
    )
    panel = admitted_panel.panel

    row = next(item for item in panel.rows if item.action == "hnsw-high")
    assert row.execution_state == "failed"
    assert row.failure_state == "backend-timeout"
    assert row.request_latency_ms == 13.5
    assert row.audit_record_sha256 is None
    assert row.returned_document_ids == ()
    assert row.entitlement_violations == 0


def test_failed_selected_action_retains_frozen_selection_and_feature_vector() -> None:
    registry = _registry()
    admitted = _admitted_actions(
        registry,
        tuple(action for action in _ACTIONS if action != "hnsw-low"),
    )
    failed = _failed_action(
        "hnsw-low",
        registry,
        latency_ms=8.0,
        feature_values=(1.0, "scifact"),
    )
    failed = replace(failed, failure_code="resource-exhausted")
    admitted_panel = action_panel_from_governed_executions(
        execution=_execution(),
        run_receipt=_run_receipt(),
        governed_executions=admitted,
        failed_executions=(failed,),
        selected_decisions={_TRIAL_KEY: failed.decision},
        action_set=_ACTIONS,
        **_factory_anchors(admitted),
    )
    panel = admitted_panel.panel

    selected = [row for row in panel.rows if row.controller_selected]
    assert len(selected) == 1
    assert selected[0].action == "hnsw-low"
    assert selected[0].execution_state == "failed"
    assert selected[0].feature_values == (1.0, "scifact")


@pytest.mark.parametrize(
    ("started", "finished"),
    [(1, 1), (2, 1), (-1, 1), (True, 2), (1.0, 2)],
)
def test_failed_action_rejects_invalid_monotonic_timing(
    started: object,
    finished: object,
) -> None:
    registry = _registry()
    result = _result("hnsw-high", registry)
    authorization = result.final_authorization or result.initial_authorization
    with pytest.raises(ConfirmatoryAnalysisError, match="monotonic"):
        FailedActionExecution(
            trial=_trial(),
            decision=result.decision,
            authorization=authorization,
            failure_code="backend-error",
            started_monotonic_ns=started,  # type: ignore[arg-type]
            finished_monotonic_ns=finished,  # type: ignore[arg-type]
            runner_identity="test-runner",
        )


def test_failed_action_rejects_unregistered_code_and_duplicate_outcome() -> None:
    registry = _registry()
    result = _result("hnsw-high", registry)
    authorization = result.final_authorization or result.initial_authorization
    with pytest.raises(ConfirmatoryAnalysisError, match="not registered"):
        FailedActionExecution(
            trial=_trial(),
            decision=result.decision,
            authorization=authorization,
            failure_code="mystery-failure",  # type: ignore[arg-type]
            started_monotonic_ns=1,
            finished_monotonic_ns=2,
            runner_identity="test-runner",
        )

    admitted = _admitted_actions(registry)
    with pytest.raises(ConfirmatoryAnalysisError, match="duplicate trial-action"):
        action_panel_from_governed_executions(
            execution=_execution(),
            run_receipt=_run_receipt(),
            governed_executions=admitted,
            failed_executions=(
                _failed_action("hnsw-high", registry, latency_ms=2.0),
            ),
            selected_decisions={_TRIAL_KEY: admitted[0].result.decision},
            action_set=_ACTIONS,
            **_factory_anchors(admitted),
        )


def test_rows_cannot_omit_or_falsely_claim_governed_audit_provenance() -> None:
    registry = _registry()
    admitted = _admitted_actions(registry)[0]
    row = admitted.to_prelabel_row(action_order=0, controller_selected=True)

    with pytest.raises(ConfirmatoryAnalysisError, match="require an audit_record"):
        replace(row, audit_record_sha256=None)
    with pytest.raises(ConfirmatoryAnalysisError, match="cannot claim"):
        replace(
            row,
            execution_state="failed",
            failure_state="backend-timeout",
            returned_document_ids=(),
        )


def test_admission_rejects_tampered_audit_hash_and_rehashed_output_mismatch() -> None:
    registry = _registry()
    result = _result("hnsw-low", registry)
    audit = _audit(result, registry)
    with pytest.raises(ConfirmatoryAnalysisError, match="self-hash"):
        GovernedActionExecution(
            trial=_trial(),
            result=result,
            audit_record=replace(audit, total_online_latency_ms=99.0),
            feature_values=(1.0,),
        )

    shortened = _rehash(audit, returned_evidence=audit.returned_evidence[:1])
    with pytest.raises(ConfirmatoryAnalysisError, match="returned document IDs"):
        GovernedActionExecution(
            trial=_trial(),
            result=result,
            audit_record=shortened,
            feature_values=(1.0,),
        )


def test_entitlement_count_is_derived_from_final_policy_mask() -> None:
    registry = _registry()
    valid = _result("hnsw-low", registry)
    assert valid.search is not None
    unauthorized_search = replace(
        valid.search,
        ids=np.array([1]),
        distances=np.array([0.1]),
        unauthorized_context=1,
        work=SearchWork(returned_candidates=1, configured_ef_search=16),
    )
    unauthorized = replace(valid, search=unauthorized_search)
    audit = _audit(valid, registry)
    forged_observation = _rehash(
        audit,
        returned_evidence=(EvidenceAudit(1, registry.content_sha256(1)),),
    )
    admitted = GovernedActionExecution(
        trial=_trial(),
        result=unauthorized,
        audit_record=forged_observation,
        feature_values=(1.0,),
    )

    assert admitted.entitlement_violations == 1
    assert admitted.to_prelabel_row(
        action_order=0,
        controller_selected=True,
    ).entitlement_violations == 1


def test_panel_rejects_selection_and_counterfactual_policy_drift() -> None:
    registry = _registry()
    admitted = _admitted_actions(registry)
    selection = admitted[0].result.decision
    with pytest.raises(ConfirmatoryAnalysisError, match="frozen controller decision"):
        action_panel_from_governed_executions(
            execution=_execution(),
            run_receipt=_run_receipt(),
            governed_executions=admitted,
            selected_decisions={
                _TRIAL_KEY: replace(selection, risk_score=selection.risk_score + 0.1)
            },
            action_set=_ACTIONS,
            **_factory_anchors(admitted),
        )

    changed_mask = np.array([True, True, True])
    drifted = _admitted_actions(
        registry,
        masks={"hnsw-high": changed_mask},
    )
    with pytest.raises(ConfirmatoryAnalysisError, match="authorization universe"):
        action_panel_from_governed_executions(
            execution=_execution(),
            run_receipt=_run_receipt(),
            governed_executions=drifted,
            selected_decisions={_TRIAL_KEY: selection},
            action_set=_ACTIONS,
            **_factory_anchors(drifted),
        )


def test_optional_anchor_verifies_the_full_governed_audit_chain() -> None:
    registry = _registry()
    admitted: list[GovernedActionExecution] = []
    previous: AuditRecord | None = None
    for action in _ACTIONS:
        result = _result(action, registry)
        record = _audit(result, registry, previous_record=previous)
        admitted.append(
            GovernedActionExecution(
                trial=_trial(),
                result=result,
                audit_record=record,
                feature_values=(1.0, "scifact") if action == "hnsw-low" else None,
            )
        )
        previous = record
    assert previous is not None
    selection = admitted[0].result.decision

    admitted_panel = action_panel_from_governed_executions(
        execution=_execution(),
        run_receipt=_run_receipt(),
        governed_executions=reversed(admitted),
        selected_decisions={_TRIAL_KEY: selection},
        action_set=_ACTIONS,
        expected_audit_head_sha256=previous.record_sha256,
        query_partition_audit_sha256=_digest("partition-audit"),
        partition_label="primary",
    )
    panel = admitted_panel.panel
    assert len(panel.rows) == len(_ACTIONS)

    broken_record = _rehash(
        admitted[1].audit_record,
        previous_record_sha256="9" * 64,
    )
    broken = (
        admitted[0],
        replace(admitted[1], audit_record=broken_record),
        *admitted[2:],
    )
    with pytest.raises(ConfirmatoryAnalysisError, match="audit chain is invalid"):
        action_panel_from_governed_executions(
            execution=_execution(),
            run_receipt=_run_receipt(),
            governed_executions=broken,
            selected_decisions={_TRIAL_KEY: selection},
            action_set=_ACTIONS,
            expected_audit_head_sha256=previous.record_sha256,
            query_partition_audit_sha256=_digest("partition-audit"),
            partition_label="primary",
        )


def test_failed_action_rejects_policy_runner_and_selected_decision_drift() -> None:
    registry = _registry()
    admitted = _admitted_actions(
        registry,
        tuple(action for action in _ACTIONS if action != "hnsw-high"),
    )
    selection = admitted[0].result.decision

    policy_drift = _failed_action(
        "hnsw-high",
        registry,
        latency_ms=4.0,
        mask=np.array([True, True, True]),
    )
    with pytest.raises(ConfirmatoryAnalysisError, match="authorization universe"):
        action_panel_from_governed_executions(
            execution=_execution(),
            run_receipt=_run_receipt(),
            governed_executions=admitted,
            failed_executions=(policy_drift,),
            selected_decisions={_TRIAL_KEY: selection},
            action_set=_ACTIONS,
            **_factory_anchors(admitted),
        )

    failed = _failed_action("hnsw-high", registry, latency_ms=4.0)
    with pytest.raises(ConfirmatoryAnalysisError, match="another runner"):
        action_panel_from_governed_executions(
            execution=_execution(),
            run_receipt=_run_receipt(),
            governed_executions=admitted,
            failed_executions=(replace(failed, runner_identity="other-runner"),),
            selected_decisions={_TRIAL_KEY: selection},
            action_set=_ACTIONS,
            **_factory_anchors(admitted),
        )

    selected_failure_admitted = _admitted_actions(
        registry,
        tuple(action for action in _ACTIONS if action != "hnsw-low"),
    )
    selected_failure = _failed_action(
        "hnsw-low",
        registry,
        latency_ms=4.0,
        feature_values=(1.0, "scifact"),
    )
    with pytest.raises(ConfirmatoryAnalysisError, match="frozen controller decision"):
        action_panel_from_governed_executions(
            execution=_execution(),
            run_receipt=_run_receipt(),
            governed_executions=selected_failure_admitted,
            failed_executions=(selected_failure,),
            selected_decisions={
                _TRIAL_KEY: replace(
                    selected_failure.decision,
                    risk_score=selected_failure.decision.risk_score + 0.1,
                )
            },
            action_set=_ACTIONS,
            **_factory_anchors(selected_failure_admitted),
        )


def test_admission_receipt_rejects_timing_and_panel_digest_tampering() -> None:
    registry = _registry()
    admitted = _admitted_actions(
        registry,
        tuple(action for action in _ACTIONS if action != "hnsw-high"),
    )
    failed = _failed_action("hnsw-high", registry, latency_ms=4.0)
    admitted_panel = action_panel_from_governed_executions(
        execution=_execution(),
        run_receipt=_run_receipt(),
        governed_executions=admitted,
        failed_executions=(failed,),
        selected_decisions={_TRIAL_KEY: admitted[0].result.decision},
        action_set=_ACTIONS,
        **_factory_anchors(admitted),
    )
    failure_record = next(
        record
        for record in admitted_panel.admission_receipt.records
        if record.execution_state == "failed"
    )
    with pytest.raises(ConfirmatoryAnalysisError, match="timing window"):
        replace(
            failure_record,
            failure_finished_monotonic_ns=(
                failure_record.failure_finished_monotonic_ns + 1
            ),
        )

    mismatched = replace(
        admitted_panel.admission_receipt,
        action_panel_artifact_sha256="9" * 64,
    )
    assert isinstance(mismatched, ActionPanelAdmissionReceipt)
    with pytest.raises(ConfirmatoryAnalysisError, match="panel digest"):
        AdmittedActionPanel(
            panel=admitted_panel.panel,
            admission_receipt=mismatched,
        )


def test_admission_receipt_round_trips_as_exclusive_canonical_evidence(
    tmp_path: Path,
) -> None:
    registry = _registry()
    admitted = _admitted_actions(registry)
    bundle = action_panel_from_governed_executions(
        execution=_execution(),
        run_receipt=_run_receipt(),
        governed_executions=admitted,
        selected_decisions={_TRIAL_KEY: admitted[0].result.decision},
        action_set=_ACTIONS,
        **_factory_anchors(admitted),
    )
    receipt = bundle.admission_receipt
    assert loads_action_panel_admission_receipt(
        receipt.canonical_bytes() + b"\n"
    ) == receipt

    target = (tmp_path / "panel-admission.json").resolve()
    write_action_panel_admission_receipt(receipt, target)
    assert target.read_bytes() == receipt.canonical_bytes() + b"\n"
    assert load_action_panel_admission_receipt(target) == receipt
    with pytest.raises(ConfirmatoryAnalysisError, match="already exists"):
        write_action_panel_admission_receipt(receipt, target)
