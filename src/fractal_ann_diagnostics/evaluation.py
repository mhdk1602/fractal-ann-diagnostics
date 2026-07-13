"""Metrics and audit records for policy-aware retrieval experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .controller import ControllerDecision
from .geometry import QueryGeometry
from .retrieval import SearchResult


def recall_at_k(returned: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Set recall against exact authorized top-k truth."""
    truth = np.asarray(ground_truth, dtype=np.int64)[:k]
    if truth.size == 0:
        return 1.0 if len(returned) == 0 else 0.0
    observed = set(np.asarray(returned, dtype=np.int64)[:k].tolist())
    return float(len(observed.intersection(truth.tolist())) / len(truth))


@dataclass(frozen=True)
class TrialRecord:
    scenario: str
    query_id: int
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
    evidence_sufficient: bool
    unauthorized_candidates: int
    unauthorized_context: int
    shortfall: int
    latency_ms: float
    effort_proxy: int
    selected: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_trial_record(
    *,
    scenario: str,
    query_id: int,
    role: str,
    search: SearchResult,
    ground_truth: np.ndarray,
    authorized_count: int,
    geometry: QueryGeometry,
    decision: ControllerDecision,
    k: int,
    selected: bool,
    evidence_recall_threshold: float = 0.9,
) -> TrialRecord:
    recall = recall_at_k(search.ids, ground_truth, k)
    sufficient = (
        recall >= evidence_recall_threshold
        and search.shortfall == 0
        and search.unauthorized_context == 0
    )
    return TrialRecord(
        scenario=scenario,
        query_id=query_id,
        role=role,
        strategy=search.strategy,
        policy_version=decision.policy_version,
        authorized_count=authorized_count,
        authorized_selectivity=geometry.authorized_selectivity,
        lid=geometry.lid,
        lid_scale_instability=geometry.lid_scale_instability,
        relative_contrast=geometry.relative_contrast,
        radius_expansion=geometry.radius_expansion,
        policy_churn=geometry.policy_churn,
        embedding_drift=geometry.embedding_drift,
        risk_score=decision.risk_score,
        recall_at_k=recall,
        evidence_sufficient=sufficient,
        unauthorized_candidates=search.unauthorized_candidates,
        unauthorized_context=search.unauthorized_context,
        shortfall=search.shortfall,
        latency_ms=search.latency_ms,
        effort_proxy=search.candidates_examined,
        selected=selected,
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
        summaries.append(
            {
                "scenario": scenario,
                "strategy": strategy,
                "n": len(rows),
                "mean_recall": float(recall.mean()),
                "p10_recall": float(np.quantile(recall, 0.1)),
                "evidence_success_rate": float(
                    np.mean([row.evidence_sufficient for row in rows])
                ),
                "unauthorized_context": int(sum(row.unauthorized_context for row in rows)),
                "mean_shortfall": float(np.mean([row.shortfall for row in rows])),
                "mean_latency_ms": float(latency.mean()),
                "p95_latency_ms": float(np.quantile(latency, 0.95)),
                "mean_effort_proxy": float(np.mean([row.effort_proxy for row in rows])),
            }
        )
    return summaries
