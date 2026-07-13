from __future__ import annotations

from fractal_ann_diagnostics.pilot import PilotConfig, run_pilot


def test_small_pilot_replays_actions_and_enforces_governed_boundary() -> None:
    config = PilotConfig(
        n_documents=480,
        dimension=12,
        n_queries_per_role=2,
        low_ef=10,
        high_ef=40,
    )
    records, summaries, metadata = run_pilot(config)
    assert metadata["n_trials"] == 8 * 3
    assert metadata["n_action_outcomes"] == 8 * 3 * 4
    governed = [row for row in records if row.strategy != "unsafe-unfiltered"]
    unsafe = [row for row in records if row.strategy == "unsafe-unfiltered"]
    assert all(row.unauthorized_context == 0 for row in governed)
    assert any(row.unauthorized_context > 0 for row in unsafe)
    assert summaries
