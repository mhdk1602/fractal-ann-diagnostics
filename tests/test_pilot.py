from __future__ import annotations

import json
from pathlib import Path

from fractal_ann_diagnostics.pilot import (
    PilotConfig,
    run_pilot,
    write_pilot_artifacts,
)


def test_small_pilot_replays_actions_and_enforces_governed_boundary() -> None:
    config = PilotConfig(
        n_documents=480,
        dimension=12,
        n_queries_per_role=2,
        low_ef=128,
        high_ef=256,
        probe_k=101,
    )
    records, summaries, metadata = run_pilot(config)
    assert metadata["n_trials"] == 8 * 3
    assert metadata["n_action_outcomes"] == 8 * 3 * 4
    governed = [row for row in records if row.strategy != "unsafe-unfiltered"]
    unsafe = [row for row in records if row.strategy == "unsafe-unfiltered"]
    assert all(row.unauthorized_context == 0 for row in governed)
    assert any(row.unauthorized_context > 0 for row in unsafe)
    assert summaries


def test_pilot_summary_is_strict_json_and_marks_unlabeled_rates_null(
    tmp_path: Path,
) -> None:
    config = PilotConfig(
        n_documents=480,
        dimension=12,
        n_queries_per_role=1,
        low_ef=128,
        high_ef=256,
        probe_k=101,
    )
    write_pilot_artifacts(tmp_path, config)

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    payload = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite,
    )
    assert all(row["evidence_labeled_n"] == 0 for row in payload["summaries"])
    assert all(row["evidence_success_rate"] is None for row in payload["summaries"])
    assert payload["metadata"]["config"]["hnsw_m"] == 3
    assert "not a production recommendation" in payload["metadata"]["graph_stress_note"]


def test_committed_pilot_exercises_low_effort_recall_failure() -> None:
    artifact = Path(__file__).parents[1] / "artifacts" / "pilot" / "summary.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert any(
        row["strategy"] == "hnsw-low" and row["mean_recall"] < 1.0
        for row in payload["summaries"]
    )
