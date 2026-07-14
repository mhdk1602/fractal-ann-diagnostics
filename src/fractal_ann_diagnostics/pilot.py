"""Reproducible development pilot for the authorization-first controller."""
from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .controller import ControllerConfig, ControllerDecision, RuleController
from .evaluation import TrialRecord, make_trial_record, summarize_trials
from .geometry import query_geometry_from_probe
from .policy import policy_churn
from .retrieval import (
    AuthorizedHNSWIndex,
    ExactSearchIndex,
    HNSWSearchIndex,
    authorized_hnsw_probe,
    authorized_hnsw_search,
    exact_authorized_search,
    search_result_from_probe,
    unsafe_unfiltered_search,
)
from .synthetic import make_governed_scenarios


@dataclass(frozen=True)
class PilotConfig:
    seed: int = 20260713
    n_documents: int = 2400
    dimension: int = 24
    n_roles: int = 4
    n_queries_per_role: int = 20
    k: int = 10
    low_ef: int = 128
    high_ef: int = 512
    probe_k: int = 101
    hnsw_m: int = 3
    high_effort_threshold: float = 0.24
    exact_threshold: float = 0.36
    evidence_recall_threshold: float = 0.9


def _action_decision(
    selected: ControllerDecision,
    action: str,
) -> ControllerDecision:
    return ControllerDecision(
        action=action,  # type: ignore[arg-type]
        risk_score=selected.risk_score,
        reasons=("counterfactual action replay",),
        policy_version=selected.policy_version,
    )


def run_pilot(config: PilotConfig | None = None) -> tuple[list[TrialRecord], list[dict], dict]:
    """Replay every reference action for every frozen synthetic trial."""
    cfg = config or PilotConfig()
    scenarios = make_governed_scenarios(
        n_documents=cfg.n_documents,
        dimension=cfg.dimension,
        n_roles=cfg.n_roles,
        n_queries_per_role=cfg.n_queries_per_role,
        seed=cfg.seed,
    )
    controller = RuleController(
        ControllerConfig(
            low_ef=cfg.low_ef,
            high_ef=cfg.high_ef,
            probe_k=cfg.probe_k,
            exact_scan_threshold=128,
            high_effort_threshold=cfg.high_effort_threshold,
            exact_threshold=cfg.exact_threshold,
        )
    )
    records: list[TrialRecord] = []

    for scenario in scenarios:
        exact = ExactSearchIndex(scenario.vectors)
        global_hnsw = HNSWSearchIndex(
            scenario.vectors,
            m=cfg.hnsw_m,
            ef_search=cfg.low_ef,
            seed=cfg.seed,
        )
        role_indexes: dict[str, AuthorizedHNSWIndex] = {}
        for role_id, role in enumerate(scenario.policy.roles):
            role_indexes[role] = AuthorizedHNSWIndex(
                scenario.vectors,
                scenario.policy.authorized_mask(role),
                m=cfg.hnsw_m,
                ef_search=cfg.low_ef,
                seed=cfg.seed + role_id + 1,
            )

        for query_id, (query, role) in enumerate(
            zip(scenario.queries, scenario.query_roles, strict=True)
        ):
            mask = scenario.policy.authorized_mask(role)
            churn = policy_churn(scenario.baseline_policy, scenario.policy, role)
            probe = authorized_hnsw_probe(
                role_indexes[role],
                query,
                mask,
                probe_k=cfg.probe_k,
                ef_search=cfg.low_ef,
                max_neighbors=cfg.probe_k,
            )
            geometry = query_geometry_from_probe(
                probe,
                policy_churn=churn,
                embedding_drift=scenario.embedding_drift,
            )
            selected = controller.decide(
                geometry,
                n_authorized=int(mask.sum()),
                policy_version=scenario.policy.version,
                expected_policy_version=scenario.policy.version,
            )
            truth = exact_authorized_search(exact, query, mask, cfg.k)
            actions = [
                search_result_from_probe(
                    probe,
                    cfg.k,
                    strategy="hnsw-low",
                ),
                authorized_hnsw_search(
                    role_indexes[role],
                    query,
                    mask,
                    cfg.k,
                    ef_search=cfg.high_ef,
                    strategy="hnsw-high",
                ),
                truth,
                unsafe_unfiltered_search(global_hnsw, query, mask, cfg.k),
            ]
            for search in actions:
                is_selected = search.strategy == selected.action
                decision = selected if is_selected else _action_decision(selected, search.strategy)
                records.append(
                    make_trial_record(
                        scenario=scenario.name,
                        query_id=str(query_id),
                        role=role,
                        search=search,
                        ground_truth=truth.ids,
                        authorized_count=int(mask.sum()),
                        geometry=geometry,
                        decision=decision,
                        k=cfg.k,
                        selected=is_selected,
                        evidence_recall_threshold=cfg.evidence_recall_threshold,
                    )
                )

    summaries = summarize_trials(records)
    selected_records = [record for record in records if record.selected]
    selected_summary = summarize_trials(selected_records)
    metadata = {
        "status": "development-pilot-not-confirmatory",
        "threshold_status": (
            "fixed on the synthetic engineering tier before any confirmatory corpus"
        ),
        "config": asdict(cfg),
        "controller": asdict(controller.config),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "n_trials": len({(r.scenario, r.query_id) for r in records}),
        "n_action_outcomes": len(records),
        "selected_summary": selected_summary,
        "security_invariant": "unauthorized_context == 0 for every governed action",
        "effort_proxy_note": (
            "hnswlib exposes configured efSearch but not visited-node or distance counters; "
            "the bounded probe is reused as the low action"
        ),
        "graph_stress_note": (
            "M=3 is an intentionally sparse development graph used to exercise recall-failure "
            "paths; it is not a production recommendation"
        ),
    }
    return records, summaries, metadata


def write_pilot_artifacts(
    output_dir: Path,
    config: PilotConfig | None = None,
) -> tuple[list[TrialRecord], list[dict], dict]:
    """Run the pilot and write compact, reviewable artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, summaries, metadata = run_pilot(config)

    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0].to_dict()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "summaries": summaries},
            handle,
            allow_nan=False,
            indent=2,
        )
        handle.write("\n")

    lines = [
        "# Authorization-first retrieval development pilot",
        "",
        "> Status: synthetic development evidence only. These results do not confirm the "
        "paper hypotheses.",
        "",
        "Every pilot search strategy was replayed for every query against exact authorized "
        "top-k ground truth. The unsafe global baseline tests the security accounting. It is "
        "not a deployable action.",
        "",
        "The pilot fixes `M=3` as an intentionally sparse development graph so recall failures "
        "remain observable despite the 101-neighbor geometry probe. It is not a production "
        "recommendation.",
        "",
        "| scenario | strategy | n | recall@10 | recall target | unauthorized context "
        "| p95 ms | effort proxy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['scenario']} | {row['strategy']} | {row['n']} | "
            f"{row['mean_recall']:.3f} | {row['recall_target_rate']:.3f} | "
            f"{row['unauthorized_context']} | {row['p95_latency_ms']:.3f} | "
            f"{row['mean_effort_proxy']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Controller-selected outcomes",
            "",
            "The fixed development rule selected these actions. Thresholds were set on the "
            "synthetic engineering tier and cannot be carried into a confirmatory claim without "
            "the sealed calibration procedure.",
            "",
            "| scenario | selected strategy | n | recall@10 | recall target "
            "| unauthorized context | effort proxy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metadata["selected_summary"]:
        lines.append(
            f"| {row['scenario']} | {row['strategy']} | {row['n']} | "
            f"{row['mean_recall']:.3f} | {row['recall_target_rate']:.3f} | "
            f"{row['unauthorized_context']} | {row['mean_effort_proxy']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The pilot tests code paths, exact policy-conditioned truth, counterfactual "
            "action replay, and fail-closed accounting. It does not establish external "
            "validity, answer faithfulness, production latency, or incremental value from "
            "fractal features. Those claims remain behind "
            "the gates in `research/preregistration.md`.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return records, summaries, metadata
