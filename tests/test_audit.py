from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from inspect import signature

import numpy as np
import pytest

from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    VerifiedArtifact,
)
from fractal_ann_diagnostics.audit import (
    GENESIS_RECORD_SHA256,
    VerifiedProvenanceRegistry,
    audit_record_from_governed_result,
    pseudonymize_subject,
    verify_audit_chain,
)
from fractal_ann_diagnostics.controller import (
    ControllerDecision,
    GovernedResult,
    IndexRefreshWork,
)
from fractal_ann_diagnostics.corpora import (
    CorpusDocument,
    EvidenceQuery,
    NormalizedCorpus,
)
from fractal_ann_diagnostics.geometry import QueryGeometry
from fractal_ann_diagnostics.policy import PolicyDecision
from fractal_ann_diagnostics.retrieval import ProbeTelemetry, SearchResult, SearchWork

PSEUDONYM_KEY = b"audit-test-key-material-is-at-least-32-bytes"
OCCURRED_AT = datetime(2026, 7, 13, 17, 30, tzinfo=timezone.utc)
COMPONENT_ARTIFACT_IDS = {
    "application": "application-artifact",
    "controller": "controller-artifact",
    "corpus": "corpus-artifact",
    "embedding": "embedding-artifact",
    "index": "index-artifact",
    "policy": "policy-artifact",
}


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _corpus(*, mutation: str = "") -> NormalizedCorpus:
    documents = tuple(
        CorpusDocument(
            document_id=document_id,
            external_id=f"external-{document_id}",
            title=f"Document {document_id}",
            text=f"document-{document_id}{mutation}",
            source_uri=f"fixture://document/{document_id}",
            content_hash=f"sha256:{_digest(f'document-{document_id}{mutation}')}",
        )
        for document_id in range(10)
    )
    return NormalizedCorpus(
        name="audit-fixture",
        stage="sealed",
        documents=documents,
        queries=(
            EvidenceQuery(
                query_id="audit-fixture:query-1",
                query_family="audit-fixture:family-1",
                text="Which records are relevant?",
                corpus="audit-fixture",
                stage="sealed",
                answer=None,
                gold_evidence=None,
            ),
        ),
    )


def _verified_artifact(artifact_id: str) -> VerifiedArtifact:
    digest = _digest(f"artifact:{artifact_id}")
    return VerifiedArtifact(
        artifact_id=artifact_id,
        relative_path=f"artifacts/{artifact_id}.bin",
        kind="file",
        exact=True,
        expected_sha256=digest,
        verified_sha256=digest,
        file_count=1,
        directory_count=0,
        byte_count=128,
        observed_file_count=1,
        observed_directory_count=0,
        observed_byte_count=128,
    )


def _receipt(
    component_artifact_ids=COMPONENT_ARTIFACT_IDS,
) -> ArtifactVerificationReceipt:
    return ArtifactVerificationReceipt(
        manifest_sha256=_digest("study-manifest"),
        artifacts=tuple(
            _verified_artifact(artifact_id) for artifact_id in component_artifact_ids.values()
        ),
    )


def _registry(
    *,
    component_artifact_ids=COMPONENT_ARTIFACT_IDS,
    corpus: NormalizedCorpus | None = None,
    receipt: ArtifactVerificationReceipt | None = None,
) -> VerifiedProvenanceRegistry:
    return VerifiedProvenanceRegistry(
        corpus=corpus or _corpus(),
        verification_receipt=receipt or _receipt(component_artifact_ids),
        component_artifact_ids=component_artifact_ids,
    )


def _governed_result(
    provenance_registry: VerifiedProvenanceRegistry | None = None,
) -> GovernedResult:
    provenance_registry = provenance_registry or _registry()
    mask = np.array([False, False, False, True, False, False, False, True, True, False])
    initial = PolicyDecision(
        subject="analyst-42",
        action="retrieve",
        policy_version="policy-7",
        authorized_mask=mask,
        decision_id="pdp-initial-1",
        document_universe_sha256=provenance_registry.document_universe_sha256,
        request_nonce="request-1-initial-policy-nonce",
    )
    final = PolicyDecision(
        subject="analyst-42",
        action="retrieve",
        policy_version="policy-7",
        authorized_mask=mask,
        decision_id="pdp-final-1",
        document_universe_sha256=provenance_registry.document_universe_sha256,
        request_nonce="request-1-final-policy-nonce",
    )
    probe_work = SearchWork(returned_candidates=3, configured_ef_search=128)
    probe = ProbeTelemetry(
        ids=np.array([3, 7, 8]),
        distances=np.array([0.1, 0.2, 0.3]),
        metric="euclidean",
        authorized_count=3,
        corpus_count=10,
        max_neighbors=3,
        search_latency_ms=1.25,
        work=probe_work,
    )
    geometry = QueryGeometry(
        lid=4.0,
        lid_scale_instability=0.05,
        authorized_selectivity=0.3,
        relative_contrast=1.7,
        radius_expansion=1.2,
        policy_churn=0.0,
        embedding_drift=0.0,
        source="bounded-probe",
        probe_neighbors=3,
        search_latency_ms=1.25,
        feature_latency_ms=0.15,
        configured_ef_search=128,
    )
    search = SearchResult(
        ids=np.array([3, 7]),
        distances=np.array([0.1, 0.2]),
        strategy="hnsw-low",
        requested_k=2,
        candidates_examined=128,
        unauthorized_candidates=0,
        unauthorized_context=0,
        latency_ms=1.25,
        work=probe_work,
    )
    mask_digest = sha256(mask.tobytes(order="C")).hexdigest()
    return GovernedResult(
        decision=ControllerDecision(
            action="hnsw-low",
            risk_score=0.17,
            reasons=("geometry score permits the low-effort authorized path",),
            policy_version="policy-7",
        ),
        geometry=geometry,
        search=search,
        initial_authorization=initial,
        final_authorization=final,
        probe=probe,
        index_refresh=IndexRefreshWork(
            policy_version="policy-7",
            mask_sha256=mask_digest,
            rebuilt=True,
            latency_ms=0.5,
            authorized_count=3,
        ),
    )


def _record(
    request_id: str = "request-1",
    *,
    previous_record=None,
    component_artifact_ids=COMPONENT_ARTIFACT_IDS,
):
    registry = _registry(component_artifact_ids=component_artifact_ids)
    result = _governed_result(registry)
    return audit_record_from_governed_result(
        result,
        request_id=request_id,
        trace_id="trace-9",
        trial_sha256=_digest("trial-1"),
        subject="analyst-42",
        pseudonym_key=PSEUDONYM_KEY,
        pseudonym_key_id="audit-key-2026-07",
        provenance_registry=registry,
        output="private generated answer",
        occurred_at=OCCURRED_AT,
        previous_record=previous_record,
    )


def test_factory_binds_required_fields_without_persisting_sensitive_payloads() -> None:
    registry = _registry()
    result = _governed_result(registry)
    record = _record()
    serialized = record.canonical_bytes().decode("utf-8")

    assert record.request_id == "request-1"
    assert record.trace_id == "trace-9"
    assert record.trial_sha256 == _digest("trial-1")
    assert record.subject_pseudonym == pseudonymize_subject(
        "analyst-42", key=PSEUDONYM_KEY, key_id="audit-key-2026-07"
    )
    assert record.initial_authorization.decision_id == "pdp-initial-1"
    assert record.final_authorization.decision_id == "pdp-final-1"
    assert record.initial_authorization.request_nonce == "request-1-initial-policy-nonce"
    assert (
        record.initial_authorization.request_sha256 == result.initial_authorization.request_sha256
    )
    assert record.initial_authorization.mask_sha256 == result.index_refresh.mask_sha256
    assert record.provenance_receipt_sha256 == registry.verification_receipt_sha256
    assert dict(record.component_revisions) == {
        component: _digest(f"artifact:{artifact_id}")
        for component, artifact_id in COMPONENT_ARTIFACT_IDS.items()
    }
    assert [item.document_id for item in record.returned_evidence] == [3, 7]
    assert [item.content_sha256 for item in record.returned_evidence] == [
        _digest("document-3"),
        _digest("document-7"),
    ]
    assert record.probe_work.configured_ef_search == 128
    assert record.search_reused_probe
    assert record.index_refresh.rebuilt
    assert record.authorization_latency_ms == 0.0
    assert record.controller_latency_ms == 0.0
    assert record.output_sha256 == _digest("private generated answer")
    assert record.previous_record_sha256 == GENESIS_RECORD_SHA256
    assert verify_audit_chain([record]).valid

    for sensitive_value in (
        "analyst-42",
        "private generated answer",
        "authorized_mask",
        "distances",
        "source_uri",
        "fixture://document/3",
        "external-3",
        "Which records are relevant?",
        "locator",
    ):
        assert sensitive_value not in serialized


def test_canonical_record_is_stable_across_component_mapping_order() -> None:
    reversed_components = dict(reversed(tuple(COMPONENT_ARTIFACT_IDS.items())))

    first = _record(component_artifact_ids=COMPONENT_ARTIFACT_IDS)
    second = _record(component_artifact_ids=reversed_components)

    assert first.component_revisions == second.component_revisions
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.record_sha256 == second.record_sha256


def test_registry_requires_closed_exact_verified_component_bindings() -> None:
    receipt = _receipt()
    missing = dict(COMPONENT_ARTIFACT_IDS)
    del missing["policy"]
    with pytest.raises(ValueError, match="are missing.*policy"):
        _registry(component_artifact_ids=missing, receipt=receipt)

    unknown_component = {**COMPONENT_ARTIFACT_IDS, "database": "database-artifact"}
    with pytest.raises(ValueError, match="unknown names.*database"):
        _registry(component_artifact_ids=unknown_component, receipt=receipt)

    unknown_artifact = {**COMPONENT_ARTIFACT_IDS, "policy": "unverified-policy"}
    with pytest.raises(ValueError, match="unverified artifacts.*unverified-policy"):
        _registry(component_artifact_ids=unknown_artifact, receipt=receipt)

    duplicate_artifact = {
        **COMPONENT_ARTIFACT_IDS,
        "policy": COMPONENT_ARTIFACT_IDS["controller"],
    }
    with pytest.raises(ValueError, match="distinct artifact ID"):
        _registry(component_artifact_ids=duplicate_artifact, receipt=receipt)

    artifacts = list(receipt.artifacts)
    policy_position = next(
        position
        for position, artifact in enumerate(artifacts)
        if artifact.artifact_id == "policy-artifact"
    )
    artifacts[policy_position] = replace(
        artifacts[policy_position],
        kind="directory",
        exact=False,
        observed_file_count=2,
        observed_byte_count=256,
    )
    non_exact_receipt = ArtifactVerificationReceipt(
        manifest_sha256=receipt.manifest_sha256,
        artifacts=tuple(artifacts),
    )
    with pytest.raises(ValueError, match="require exact verified artifacts"):
        _registry(receipt=non_exact_receipt)


def test_factory_rejects_corpus_substitution_and_mask_size_mismatch() -> None:
    registry = _registry()
    result = _governed_result(registry)
    common = dict(
        request_id="request-provenance",
        trace_id="trace-provenance",
        trial_sha256=_digest("trial-provenance"),
        subject="analyst-42",
        pseudonym_key=PSEUDONYM_KEY,
        pseudonym_key_id="audit-key-2026-07",
        occurred_at=OCCURRED_AT,
    )

    substituted_registry = _registry(corpus=_corpus(mutation="-substituted"))
    with pytest.raises(ValueError, match="does not match verified provenance"):
        audit_record_from_governed_result(
            result,
            provenance_registry=substituted_registry,
            **common,
        )

    short_mask = result.initial_authorization.authorized_mask[:-1]
    malformed = replace(
        result,
        initial_authorization=replace(
            result.initial_authorization,
            authorized_mask=short_mask,
        ),
        final_authorization=replace(
            result.final_authorization,
            authorized_mask=short_mask,
        ),
    )
    with pytest.raises(ValueError, match="mask size does not match"):
        audit_record_from_governed_result(
            malformed,
            provenance_registry=registry,
            **common,
        )


def test_chain_detects_modification_reordering_and_middle_deletion() -> None:
    first = _record("request-1")
    second = _record("request-2", previous_record=first)
    third = _record("request-3", previous_record=second)
    chain = [first, second, third]

    assert verify_audit_chain(chain).valid

    modified = [
        first,
        replace(second, controller_reasons=("modified after emission",)),
        third,
    ]
    assert not verify_audit_chain(modified).valid
    assert any("record hash mismatch" in error for error in verify_audit_chain(modified).errors)

    assert not verify_audit_chain([second, first, third]).valid
    assert not verify_audit_chain([first, third]).valid


def test_trusted_head_or_length_detects_tail_truncation() -> None:
    first = _record("request-1")
    second = _record("request-2", previous_record=first)
    third = _record("request-3", previous_record=second)

    truncated = [first, second]
    assert verify_audit_chain(truncated).valid
    anchored = verify_audit_chain(
        truncated,
        expected_head_sha256=third.record_sha256,
        expected_length=3,
    )
    assert not anchored.valid
    assert "chain head does not match the trusted head" in anchored.errors
    assert "chain length is 2, expected 3" in anchored.errors


def test_sequence_and_trusted_length_reject_boolean_integers() -> None:
    record = _record()
    with pytest.raises(ValueError, match="sequence must be non-negative"):
        replace(record, sequence=True)
    with pytest.raises(ValueError, match="non-negative integer"):
        verify_audit_chain([record], expected_length=True)


def test_factory_removes_unsafe_digest_inputs_and_rejects_subject_mismatch() -> None:
    registry = _registry()
    result = _governed_result(registry)
    common = dict(
        request_id="request-1",
        trace_id="trace-9",
        trial_sha256=_digest("trial-1"),
        pseudonym_key=PSEUDONYM_KEY,
        pseudonym_key_id="audit-key-2026-07",
        provenance_registry=registry,
        occurred_at=OCCURRED_AT,
    )

    parameters = signature(audit_record_from_governed_result).parameters
    assert "content_sha256_by_document_id" not in parameters
    assert "component_revisions" not in parameters
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        audit_record_from_governed_result(
            result,
            subject="analyst-42",
            content_sha256_by_document_id={3: _digest("forged")},  # type: ignore[call-arg]
            **common,
        )
    with pytest.raises(ValueError, match="does not match the audited subject"):
        audit_record_from_governed_result(
            result,
            subject="different-user",
            **common,
        )


def test_abstention_binds_absence_and_rejects_emitted_output() -> None:
    registry = _registry()
    initial = _governed_result(registry).initial_authorization
    result = GovernedResult(
        decision=ControllerDecision(
            action="abstain",
            risk_score=1.0,
            reasons=("policy version mismatch; fail closed",),
            policy_version="policy-7",
        ),
        geometry=None,
        search=None,
        initial_authorization=initial,
    )
    common = dict(
        result=result,
        request_id="request-abstain",
        trace_id="trace-10",
        trial_sha256=_digest("trial-abstain"),
        subject="analyst-42",
        pseudonym_key=PSEUDONYM_KEY,
        pseudonym_key_id="audit-key-2026-07",
        provenance_registry=registry,
        occurred_at=OCCURRED_AT,
    )

    record = audit_record_from_governed_result(**common)
    assert record.abstained
    assert record.returned_evidence == ()
    assert not record.output_emitted
    assert record.output_sha256 is None

    with pytest.raises(ValueError, match="abstention cannot emit output"):
        audit_record_from_governed_result(output="should not exist", **common)


def test_pseudonym_requires_a_secret_key_not_a_plain_digest() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        pseudonymize_subject("analyst-42", key=b"too-short", key_id="key-1")

    first = pseudonymize_subject("analyst-42", key=PSEUDONYM_KEY, key_id="key-1")
    second = pseudonymize_subject("analyst-42", key=PSEUDONYM_KEY, key_id="key-2")
    assert first != second


def test_factory_revalidates_final_permission_instead_of_trusting_result() -> None:
    registry = _registry()
    result = _governed_result(registry)
    final_mask = result.final_authorization.authorized_mask.copy()
    final_mask[7] = False
    forged = replace(
        result,
        final_authorization=replace(
            result.final_authorization,
            authorized_mask=final_mask,
        ),
    )
    with pytest.raises(ValueError, match="authorization snapshots must match|does not permit"):
        audit_record_from_governed_result(
            forged,
            request_id="request-forged",
            trace_id="trace-forged",
            trial_sha256=_digest("trial-forged"),
            subject="analyst-42",
            pseudonym_key=PSEUDONYM_KEY,
            pseudonym_key_id="audit-key-2026-07",
            provenance_registry=registry,
            occurred_at=OCCURRED_AT,
        )
