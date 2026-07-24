"""Metrics and audit records for policy-aware retrieval experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from .controller import ControllerDecision, GovernedResult
from .evidence import EvidenceLocation, GoldEvidence, assess_evidence, evaluate_answer
from .geometry import QueryGeometry
from .policy import PolicyDecision
from .retrieval import SearchResult


def recall_at_k(returned: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Set recall against exact authorized top-k truth."""
    truth = np.asarray(ground_truth, dtype=np.int64)[:k]
    if truth.size == 0:
        return 1.0 if len(returned) == 0 else 0.0
    observed = set(np.asarray(returned, dtype=np.int64)[:k].tolist())
    return float(len(observed.intersection(truth.tolist())) / len(truth))


@dataclass(frozen=True)
class ServingCost:
    """Typed request-cost boundary used by the confirmatory H3 estimand."""

    authorization_ms: float
    controller_ms: float
    index_refresh_ms: float
    probe_search_ms: float
    geometry_feature_ms: float
    selected_search_ms: float
    total_request_ms: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.attributed_ms > self.total_request_ms + 1e-9:
            raise ValueError("cost components cannot exceed measured total_request_ms")

    @property
    def attributed_ms(self) -> float:
        return (
            self.authorization_ms
            + self.controller_ms
            + self.index_refresh_ms
            + self.probe_search_ms
            + self.geometry_feature_ms
            + self.selected_search_ms
        )


def serving_cost_from_governed_result(result: GovernedResult) -> ServingCost:
    """Extract non-overlapping components and the measured request wall time."""
    probe_search_ms = 0.0 if result.probe is None else result.probe.search_latency_ms
    geometry_feature_ms = 0.0 if result.geometry is None else result.geometry.feature_latency_ms
    index_refresh_ms = 0.0 if result.index_refresh is None else result.index_refresh.latency_ms
    selected_search_ms = 0.0
    if result.search is not None and result.search.strategy != "hnsw-low":
        selected_search_ms = result.search.latency_ms
    return ServingCost(
        authorization_ms=result.authorization_latency_ms,
        controller_ms=result.controller_latency_ms,
        index_refresh_ms=index_refresh_ms,
        probe_search_ms=probe_search_ms,
        geometry_feature_ms=geometry_feature_ms,
        selected_search_ms=selected_search_ms,
        total_request_ms=result.total_online_latency_ms,
    )


@dataclass(frozen=True)
class TrialRecord:
    scenario: str
    query_id: str
    role: str
    strategy: str
    policy_version: str
    authorized_count: int
    authorized_selectivity: float
    lid: float
    lid_scale_instability: float
    relative_contrast: float
    radius_expansion: float
    policy_churn: float
    embedding_drift: float
    risk_score: float
    recall_at_k: float
    evidence_sufficient: bool | None
    unauthorized_candidates: int
    unauthorized_context: int
    shortfall: int
    latency_ms: float
    effort_proxy: int
    selected: bool
    serving_cost: ServingCost | None = None
    recall_target_met: bool = False
    evidence_basis: str = "unavailable"
    evidence_query_id: str | None = None
    authorized_solution_exists: bool | None = None
    authorized_evidence_bundle_ids: tuple[str, ...] = ()
    complete_evidence_bundle_ids: tuple[str, ...] = ()
    answered: bool | None = None
    evidence_supported_emission: bool | None = None
    false_permit: bool | None = None
    false_denial: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_trial_record(
    *,
    scenario: str,
    query_id: str,
    role: str,
    search: SearchResult,
    ground_truth: np.ndarray,
    authorized_count: int | None,
    geometry: QueryGeometry,
    decision: ControllerDecision,
    k: int,
    selected: bool,
    evidence_recall_threshold: float = 0.9,
    gold_evidence: GoldEvidence | None = None,
    returned_evidence: Iterable[EvidenceLocation] | None = None,
    final_authorization: PolicyDecision | None = None,
    answered: bool | None = None,
    serving_cost: ServingCost | None = None,
) -> TrialRecord:
    if final_authorization is None:
        if type(authorized_count) is not int or authorized_count < 0:
            raise ValueError(
                "authorized_count must be a non-negative integer when no final authorization "
                "is supplied"
            )
        resolved_authorized_count = authorized_count
        authorized_ids: tuple[int, ...] | None = None
    else:
        if not isinstance(final_authorization, PolicyDecision):
            raise TypeError("final_authorization must be a PolicyDecision")
        if not final_authorization.available:
            raise ValueError("final_authorization must be available")
        if decision.policy_version != final_authorization.policy_version:
            raise ValueError("controller and final authorization policy revisions must match")
        if search.unauthorized_candidates or search.unauthorized_context:
            raise ValueError("governed evidence cannot include unauthorized material")
        if not final_authorization.permits(search.ids):
            raise ValueError("search result is not permitted by final_authorization")
        resolved_authorized_count = final_authorization.authorized_count
        if authorized_count is not None and authorized_count != resolved_authorized_count:
            raise ValueError("authorized_count does not match final_authorization")
        authorized_ids = tuple(
            int(document_id) for document_id in np.flatnonzero(final_authorization.authorized_mask)
        )

    recall = recall_at_k(search.ids, ground_truth, k)
    recall_target_met = (
        recall >= evidence_recall_threshold
        and search.shortfall == 0
        and search.unauthorized_context == 0
    )
    evidence_sufficient: bool | None = None
    evidence_basis = "unavailable"
    evidence_query_id: str | None = None
    authorized_solution_exists: bool | None = None
    authorized_bundle_ids: tuple[str, ...] = ()
    complete_bundle_ids: tuple[str, ...] = ()
    evidence_supported_emission: bool | None = None
    false_permit: bool | None = None
    false_denial: bool | None = None

    if gold_evidence is None:
        if returned_evidence is not None or answered is not None:
            raise ValueError("gold_evidence is required for evidence or answer-level outcomes")
    else:
        if gold_evidence.query_id != query_id:
            raise ValueError("gold evidence query_id must match the trial query_id")
        if returned_evidence is None or final_authorization is None or authorized_ids is None:
            raise ValueError(
                "returned_evidence and final_authorization are required with gold_evidence"
            )
        observed_evidence = tuple(returned_evidence)
        observed_document_ids = {location.document_id for location in observed_evidence}
        search_document_ids = set(int(document_id) for document_id in search.ids)
        if not observed_document_ids.issubset(search_document_ids):
            raise ValueError("returned evidence must be derived from the search result")
        assessment = assess_evidence(
            gold_evidence,
            observed_evidence,
            authorized_ids,
        )
        evidence_sufficient = assessment.evidence_sufficient
        evidence_basis = "gold-bundles"
        evidence_query_id = gold_evidence.query_id
        authorized_solution_exists = assessment.authorized_solution_exists
        authorized_bundle_ids = assessment.authorized_bundle_ids
        complete_bundle_ids = assessment.complete_bundle_ids
        if answered is not None:
            answer_outcomes = evaluate_answer(assessment, answered=answered)
            evidence_supported_emission = answer_outcomes.evidence_supported_emission
            false_permit = answer_outcomes.false_permit
            false_denial = answer_outcomes.false_denial

    return TrialRecord(
        scenario=scenario,
        query_id=query_id,
        role=role,
        strategy=search.strategy,
        policy_version=decision.policy_version,
        authorized_count=resolved_authorized_count,
        authorized_selectivity=geometry.authorized_selectivity,
        lid=geometry.lid,
        lid_scale_instability=geometry.lid_scale_instability,
        relative_contrast=geometry.relative_contrast,
        radius_expansion=geometry.radius_expansion,
        policy_churn=geometry.policy_churn,
        embedding_drift=geometry.embedding_drift,
        risk_score=decision.risk_score,
        recall_at_k=recall,
        evidence_sufficient=evidence_sufficient,
        unauthorized_candidates=search.unauthorized_candidates,
        unauthorized_context=search.unauthorized_context,
        shortfall=search.shortfall,
        latency_ms=search.latency_ms,
        effort_proxy=search.candidates_examined,
        selected=selected,
        serving_cost=serving_cost,
        recall_target_met=recall_target_met,
        evidence_basis=evidence_basis,
        evidence_query_id=evidence_query_id,
        authorized_solution_exists=authorized_solution_exists,
        authorized_evidence_bundle_ids=authorized_bundle_ids,
        complete_evidence_bundle_ids=complete_bundle_ids,
        answered=answered,
        evidence_supported_emission=evidence_supported_emission,
        false_permit=false_permit,
        false_denial=false_denial,
    )


def summarize_trials(records: list[TrialRecord]) -> list[dict[str, object]]:
    """Aggregate trial metrics by scenario and strategy."""
    groups: dict[tuple[str, str], list[TrialRecord]] = {}
    for record in records:
        groups.setdefault((record.scenario, record.strategy), []).append(record)
    summaries: list[dict[str, object]] = []
    for (scenario, strategy), rows in sorted(groups.items()):
        recall = np.asarray([row.recall_at_k for row in rows], dtype=float)
        latency = np.asarray([row.latency_ms for row in rows], dtype=float)
        request_latency = np.asarray(
            [
                row.serving_cost.total_request_ms
                if row.serving_cost is not None
                else row.latency_ms
                for row in rows
            ],
            dtype=float,
        )
        evidence = [row.evidence_sufficient for row in rows if row.evidence_sufficient is not None]
        answers = [row for row in rows if row.answered is not None]
        summaries.append(
            {
                "scenario": scenario,
                "strategy": strategy,
                "n": len(rows),
                "mean_recall": float(recall.mean()),
                "p10_recall": float(np.quantile(recall, 0.1)),
                "recall_target_rate": float(np.mean([row.recall_target_met for row in rows])),
                "evidence_labeled_n": len(evidence),
                "evidence_success_rate": float(np.mean(evidence)) if evidence else None,
                "answer_labeled_n": len(answers),
                "answer_coverage": (
                    float(np.mean([row.answered for row in answers])) if answers else None
                ),
                "evidence_supported_emission_rate": (
                    float(np.mean([row.evidence_supported_emission for row in answers]))
                    if answers
                    else None
                ),
                "false_permit_rate": (
                    float(np.mean([row.false_permit for row in answers])) if answers else None
                ),
                "false_denial_rate": (
                    float(np.mean([row.false_denial for row in answers])) if answers else None
                ),
                "unauthorized_context": int(sum(row.unauthorized_context for row in rows)),
                "mean_shortfall": float(np.mean([row.shortfall for row in rows])),
                "mean_latency_ms": float(latency.mean()),
                "p95_latency_ms": float(np.quantile(latency, 0.95)),
                "mean_request_latency_ms": float(request_latency.mean()),
                "p95_request_latency_ms": float(np.quantile(request_latency, 0.95)),
                "mean_effort_proxy": float(np.mean([row.effort_proxy for row in rows])),
            }
        )
    return summaries


@dataclass(frozen=True)
class ZeroEventUpperBound:
    """Exact one-sided upper bound after aggregation to independent families."""

    n_families: int
    confidence: float
    upper_probability: float


def exact_family_zero_event_upper_bound(
    family_ids: Iterable[tuple[str, str]],
    events: Iterable[bool],
    *,
    expected_family_ids: Iterable[tuple[str, str]],
    confidence: float = 0.95,
) -> ZeroEventUpperBound:
    """Compute the Clopper-Pearson zero-event bound at family level.

    Repeated rows for one family are collapsed with ``any`` so paired actions,
    policies, and seeds cannot inflate the denominator.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    expected = tuple(expected_family_ids)
    if not expected:
        raise ValueError("expected_family_ids must contain at least one family")
    if len(expected) != len(set(expected)):
        raise ValueError("expected_family_ids cannot contain duplicates")
    for key in expected:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(part, str) or not part for part in key)
        ):
            raise TypeError(
                "family IDs must be non-empty (corpus_id, query_family_id) string tuples"
            )

    by_family: dict[tuple[str, str], bool] = {}
    sentinel = object()
    family_iterator = iter(family_ids)
    event_iterator = iter(events)
    while True:
        family = next(family_iterator, sentinel)
        event = next(event_iterator, sentinel)
        if family is sentinel and event is sentinel:
            break
        if family is sentinel or event is sentinel:
            raise ValueError("family_ids and events must have the same length")
        if (
            not isinstance(family, tuple)
            or len(family) != 2
            or any(not isinstance(part, str) or not part for part in family)
        ):
            raise TypeError(
                "family IDs must be non-empty (corpus_id, query_family_id) string tuples"
            )
        if type(event) is not bool:
            raise TypeError("events must contain strict boolean values")
        by_family[family] = by_family.get(family, False) or event

    if not by_family:
        raise ValueError("at least one family is required")
    if set(by_family) != set(expected):
        missing = sorted(set(expected) - set(by_family))
        extra = sorted(set(by_family) - set(expected))
        raise ValueError(
            "observed families do not equal the sealed expected-family set; "
            f"missing={missing}, extra={extra}"
        )
    if any(by_family.values()):
        raise ValueError("zero-event upper bound requires zero observed family events")

    n_families = len(by_family)
    alpha = 1.0 - confidence
    upper_probability = 1.0 - alpha ** (1.0 / n_families)
    return ZeroEventUpperBound(
        n_families=n_families,
        confidence=confidence,
        upper_probability=upper_probability,
    )
