from __future__ import annotations

import math

import numpy as np
import pytest

from fractal_ann_diagnostics.controller import ControllerDecision
from fractal_ann_diagnostics.evaluation import (
    exact_family_zero_event_upper_bound,
    make_trial_record,
)
from fractal_ann_diagnostics.evidence import (
    CompleteEvidenceBundle,
    EvidenceLocation,
    GoldEvidence,
    assess_evidence,
    evaluate_answer,
)
from fractal_ann_diagnostics.geometry import QueryGeometry
from fractal_ann_diagnostics.policy import PolicyDecision
from fractal_ann_diagnostics.retrieval import SearchResult


def _location(
    document_id: int,
    locator: str,
    *,
    content_hash: str | None = None,
) -> EvidenceLocation:
    return EvidenceLocation(
        document_id=document_id,
        source_uri=f"corpus://document/{document_id}",
        locator=locator,
        content_hash=content_hash,
    )


def _gold() -> GoldEvidence:
    return GoldEvidence(
        query_id="query-7",
        alternatives=(
            CompleteEvidenceBundle(
                bundle_id="two-hop-route",
                locations=(_location(0, "paragraph:2"), _location(1, "table:4/row:3")),
            ),
            CompleteEvidenceBundle(
                bundle_id="single-source-route",
                locations=(_location(2, "page:8"),),
            ),
        ),
    )


def _search_result() -> SearchResult:
    return SearchResult(
        ids=np.asarray([0, 1], dtype=np.int64),
        distances=np.asarray([0.0, 0.1], dtype=np.float32),
        strategy="hnsw-low",
        requested_k=2,
        candidates_examined=10,
        unauthorized_candidates=0,
        unauthorized_context=0,
        latency_ms=1.0,
    )


def _geometry() -> QueryGeometry:
    return QueryGeometry(
        lid=4.0,
        lid_scale_instability=0.1,
        authorized_selectivity=0.5,
        relative_contrast=1.5,
        radius_expansion=1.2,
        policy_churn=0.0,
        embedding_drift=0.0,
    )


def _decision() -> ControllerDecision:
    return ControllerDecision(
        action="hnsw-low",
        risk_score=0.1,
        reasons=("test",),
        policy_version="policy-v1",
    )


def _authorization(*, mask: tuple[bool, ...] = (True, True, True)) -> PolicyDecision:
    return PolicyDecision(
        subject="analyst",
        action="retrieve",
        policy_version="policy-v1",
        authorized_mask=np.asarray(mask, dtype=bool),
    )


def test_alternative_complete_bundle_is_sufficient_with_exact_provenance() -> None:
    assessment = assess_evidence(
        _gold(),
        returned=(_location(2, "page:8"),),
        authorized_document_ids=(0, 1, 2),
    )
    assert assessment.authorized_solution_exists
    assert assessment.evidence_sufficient
    assert assessment.complete_bundle_ids == ("single-source-route",)
    assert evaluate_answer(assessment, answered=True).evidence_supported_emission


def test_partial_or_wrong_location_does_not_complete_bundle() -> None:
    gold = GoldEvidence(
        query_id="hash-sensitive",
        alternatives=(
            CompleteEvidenceBundle(
                bundle_id="route",
                locations=(_location(4, "page:2", content_hash="sha256:gold"),),
            ),
        ),
    )
    assessment = assess_evidence(
        gold,
        returned=(_location(4, "page:2", content_hash="sha256:stale"),),
        authorized_document_ids=(4,),
    )
    assert assessment.authorized_solution_exists
    assert not assessment.evidence_sufficient


def test_gold_evidence_copies_mutable_container_inputs() -> None:
    locations = [_location(0, "page:1")]
    bundle = CompleteEvidenceBundle(
        bundle_id="route",
        locations=locations,  # type: ignore[arg-type]
    )
    alternatives = [bundle]
    gold = GoldEvidence(
        query_id="query",
        alternatives=alternatives,  # type: ignore[arg-type]
    )
    locations.append(_location(1, "page:2"))
    alternatives.clear()
    assert bundle.locations == (_location(0, "page:1"),)
    assert gold.alternatives == (bundle,)


def test_answer_outcomes_are_independent_of_perfect_recall() -> None:
    record = make_trial_record(
        scenario="sealed",
        query_id="query-7",
        role="analyst",
        search=_search_result(),
        ground_truth=np.asarray([0, 1], dtype=np.int64),
        authorized_count=3,
        geometry=_geometry(),
        decision=_decision(),
        k=2,
        selected=True,
        gold_evidence=_gold(),
        returned_evidence=(_location(0, "paragraph:2"),),
        final_authorization=_authorization(),
        answered=True,
    )
    assert record.recall_at_k == 1.0
    assert record.recall_target_met
    assert record.evidence_sufficient is False
    assert record.false_permit is True
    assert record.evidence_supported_emission is False
    assert record.evidence_basis == "gold-bundles"


def test_abstention_with_complete_returned_evidence_is_false_denial() -> None:
    record = make_trial_record(
        scenario="sealed",
        query_id="query-7",
        role="analyst",
        search=_search_result(),
        ground_truth=np.asarray([0, 1], dtype=np.int64),
        authorized_count=2,
        geometry=_geometry(),
        decision=_decision(),
        k=2,
        selected=True,
        gold_evidence=_gold(),
        returned_evidence=(_location(0, "paragraph:2"), _location(1, "table:4/row:3")),
        final_authorization=_authorization(mask=(True, True, False)),
        answered=False,
    )
    assert record.evidence_sufficient is True
    assert record.false_denial is True
    assert record.false_permit is False
    assert record.evidence_supported_emission is False


def test_family_zero_event_bound_collapses_repeated_action_rows() -> None:
    expected = (("scifact", "family-a"), ("scifact", "family-b"))
    bound = exact_family_zero_event_upper_bound(
        family_ids=(
            ("scifact", "family-a"),
            ("scifact", "family-a"),
            ("scifact", "family-b"),
        ),
        events=(False, False, False),
        expected_family_ids=expected,
    )
    assert bound.n_families == 2
    assert math.isclose(bound.upper_probability, 1.0 - math.sqrt(0.05))


def test_trial_rejects_mismatched_gold_query_and_unretrieved_evidence() -> None:
    common = {
        "scenario": "sealed",
        "query_id": "query-7",
        "role": "analyst",
        "search": _search_result(),
        "ground_truth": np.asarray([0, 1], dtype=np.int64),
        "authorized_count": 3,
        "geometry": _geometry(),
        "decision": _decision(),
        "k": 2,
        "selected": True,
        "final_authorization": _authorization(),
        "answered": True,
    }
    wrong_query = GoldEvidence(
        query_id="other-query",
        alternatives=_gold().alternatives,
    )
    with pytest.raises(ValueError, match="query_id must match"):
        make_trial_record(
            gold_evidence=wrong_query,
            returned_evidence=(_location(0, "paragraph:2"),),
            **common,
        )
    with pytest.raises(ValueError, match="derived from the search result"):
        make_trial_record(
            gold_evidence=_gold(),
            returned_evidence=(_location(2, "page:8"),),
            **common,
        )


def test_trial_derives_evidence_authorization_from_final_decision() -> None:
    with pytest.raises(ValueError, match="does not match final_authorization"):
        make_trial_record(
            scenario="sealed",
            query_id="query-7",
            role="analyst",
            search=_search_result(),
            ground_truth=np.asarray([0, 1], dtype=np.int64),
            authorized_count=2,
            geometry=_geometry(),
            decision=_decision(),
            k=2,
            selected=True,
            gold_evidence=_gold(),
            returned_evidence=(_location(0, "paragraph:2"),),
            final_authorization=_authorization(),
            answered=True,
        )


def test_family_zero_event_bound_rejects_observed_event() -> None:
    with pytest.raises(ValueError, match="zero observed"):
        exact_family_zero_event_upper_bound(
            family_ids=(("scifact", "family-a"), ("scifact", "family-a")),
            events=(False, True),
            expected_family_ids=(("scifact", "family-a"),),
        )


@pytest.mark.parametrize("bad_event", (None, 0, 1, "", "false"))
def test_family_zero_event_bound_rejects_ambiguous_events(bad_event: object) -> None:
    with pytest.raises(TypeError, match="strict boolean"):
        exact_family_zero_event_upper_bound(
            family_ids=(("scifact", "family-a"),),
            events=(bad_event,),  # type: ignore[arg-type]
            expected_family_ids=(("scifact", "family-a"),),
        )


def test_family_zero_event_bound_requires_exact_sealed_family_set() -> None:
    with pytest.raises(ValueError, match="expected-family set"):
        exact_family_zero_event_upper_bound(
            family_ids=(("scifact", "family-a"),),
            events=(False,),
            expected_family_ids=(
                ("scifact", "family-a"),
                ("bright", "family-b"),
            ),
        )
