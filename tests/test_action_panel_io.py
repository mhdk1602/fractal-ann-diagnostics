from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

import fractal_ann_diagnostics.artifact_integrity as artifact_integrity
from fractal_ann_diagnostics.confirmatory_analysis import (
    ActionPanelArtifact,
    ConfirmatoryAnalysisError,
    PreLabelActionRow,
    load_action_panel_artifact,
    load_prelabel_action_row,
    loads_action_panel_artifact,
    loads_prelabel_action_row,
    write_action_panel_artifact,
    write_prelabel_action_row,
)


def _row(
    action: str = "hnsw-low",
    action_order: int = 0,
    *,
    execution_position: int | None = None,
    controller_selected: bool = True,
) -> PreLabelActionRow:
    abstained = action == "abstain"
    return PreLabelActionRow(
        trial_key="a" * 64,
        family_key="b" * 64,
        action=action,
        action_order=action_order,
        execution_position=(action_order if execution_position is None else execution_position),
        audit_record_sha256="f" * 64,
        execution_state="abstained" if abstained else "completed",
        failure_state="registered-abstention" if abstained else None,
        controller_selected=controller_selected,
        request_latency_ms=1.0 + action_order,
        entitlement_violations=0,
        returned_document_ids=() if abstained else (0,),
        feature_values=(1.0, "scifact") if action == "hnsw-low" else None,
    )


def _panel() -> ActionPanelArtifact:
    actions = ("hnsw-low", "hnsw-high", "exact-authorized", "abstain")
    return ActionPanelArtifact(
        manifest_sha256="c" * 64,
        run_receipt_sha256="d" * 64,
        execution_artifact_sha256="e" * 64,
        corpus="scifact",
        stage="sealed",
        document_count=3,
        action_set=actions,
        rows=tuple(
            _row(
                action,
                action_order,
                controller_selected=action == "hnsw-low",
            )
            for action_order, action in enumerate(actions)
        ),
    )


def test_prelabel_row_round_trips_through_closed_memory_and_file_apis(
    tmp_path: Path,
) -> None:
    row = _row()
    assert PreLabelActionRow.from_dict(row.to_dict()) == row
    assert loads_prelabel_action_row(row.canonical_bytes() + b"\n") == row
    assert loads_prelabel_action_row((row.canonical_bytes() + b"\n").decode("utf-8")) == row

    target = (tmp_path / "row.json").resolve()
    write_prelabel_action_row(row, target)
    assert target.read_bytes() == row.canonical_bytes() + b"\n"
    assert load_prelabel_action_row(target) == row

    with pytest.raises(ConfirmatoryAnalysisError, match="already exists"):
        write_prelabel_action_row(row, target)


def test_action_panel_round_trips_through_closed_memory_and_file_apis(
    tmp_path: Path,
) -> None:
    panel = _panel()
    assert ActionPanelArtifact.from_dict(panel.to_dict()) == panel
    assert loads_action_panel_artifact(panel.canonical_bytes() + b"\n") == panel
    assert loads_action_panel_artifact((panel.canonical_bytes() + b"\n").decode("utf-8")) == panel

    target = (tmp_path / "panel.json").resolve()
    write_action_panel_artifact(panel, target)
    assert target.read_bytes() == panel.canonical_bytes() + b"\n"
    assert load_action_panel_artifact(target) == panel

    with pytest.raises(ConfirmatoryAnalysisError, match="already exists"):
        write_action_panel_artifact(panel, target)


@pytest.mark.parametrize("missing", ["trial_key", "returned_document_ids"])
def test_prelabel_row_from_dict_rejects_missing_fields(missing: str) -> None:
    payload = _row().to_dict()
    del payload[missing]
    with pytest.raises(ConfirmatoryAnalysisError, match="closed schema"):
        PreLabelActionRow.from_dict(payload)


def test_prelabel_row_from_dict_rejects_unknown_fields_and_nonarrays() -> None:
    payload = _row().to_dict()
    payload["unexpected"] = None
    with pytest.raises(ConfirmatoryAnalysisError, match="closed schema"):
        PreLabelActionRow.from_dict(payload)

    payload = _row().to_dict()
    payload["returned_document_ids"] = (0,)
    with pytest.raises(ConfirmatoryAnalysisError, match="must be an array"):
        PreLabelActionRow.from_dict(payload)

    payload = _row().to_dict()
    payload["feature_values"] = (1.0,)
    with pytest.raises(ConfirmatoryAnalysisError, match="array or null"):
        PreLabelActionRow.from_dict(payload)

    payload = _row().to_dict()
    payload["feature_values"] = [float("nan")]
    with pytest.raises(ConfirmatoryAnalysisError, match="non-finite"):
        PreLabelActionRow.from_dict(payload)


@pytest.mark.parametrize("missing", ["action_set", "rows", "manifest_sha256"])
def test_action_panel_from_dict_rejects_missing_fields(missing: str) -> None:
    payload = _panel().to_dict()
    del payload[missing]
    with pytest.raises(ConfirmatoryAnalysisError, match="closed schema"):
        ActionPanelArtifact.from_dict(payload)


def test_action_panel_from_dict_rejects_unknown_and_nonarray_fields() -> None:
    payload = _panel().to_dict()
    payload["unexpected"] = None
    with pytest.raises(ConfirmatoryAnalysisError, match="closed schema"):
        ActionPanelArtifact.from_dict(payload)

    payload = _panel().to_dict()
    payload["action_set"] = tuple(payload["action_set"])
    with pytest.raises(ConfirmatoryAnalysisError, match="action_set must be an array"):
        ActionPanelArtifact.from_dict(payload)

    payload = _panel().to_dict()
    payload["rows"] = tuple(payload["rows"])
    with pytest.raises(ConfirmatoryAnalysisError, match="rows must be an array"):
        ActionPanelArtifact.from_dict(payload)


@pytest.mark.parametrize(
    "loader",
    [loads_prelabel_action_row, loads_action_panel_artifact],
)
def test_loads_rejects_duplicate_keys_and_nonfinite_numbers(loader) -> None:
    with pytest.raises(ConfirmatoryAnalysisError, match="duplicate key"):
        loader(b'{"schema_version":"first","schema_version":"second"}\n')
    with pytest.raises(ConfirmatoryAnalysisError, match="non-finite"):
        loader(b'{"request_latency_ms":NaN}\n')


@pytest.mark.parametrize(
    ("loader", "canonical"),
    [
        (loads_prelabel_action_row, lambda: _row().canonical_bytes()),
        (loads_action_panel_artifact, lambda: _panel().canonical_bytes()),
    ],
)
def test_loads_enforces_exact_canonical_bytes_and_one_newline(
    loader,
    canonical,
) -> None:
    encoded = canonical()
    for noncanonical in (
        encoded,
        encoded + b"\n\n",
        json.dumps(json.loads(encoded), indent=2, sort_keys=True).encode("utf-8") + b"\n",
    ):
        with pytest.raises(ConfirmatoryAnalysisError, match="not canonical"):
            loader(noncanonical)

    with pytest.raises(ConfirmatoryAnalysisError, match="valid UTF-8"):
        loader(b"\xff")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_panel_io_does_not_follow_file_or_parent_symlinks(tmp_path: Path) -> None:
    panel = _panel()
    real = (tmp_path / "real.json").resolve()
    write_action_panel_artifact(panel, real)
    linked_file = (tmp_path / "linked.json").resolve()
    linked_file.symlink_to(real)
    with pytest.raises(ConfirmatoryAnalysisError, match="symlink"):
        load_action_panel_artifact(linked_file)

    real_parent = (tmp_path / "real-parent").resolve()
    real_parent.mkdir()
    linked_parent = (tmp_path / "linked-parent").resolve()
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ConfirmatoryAnalysisError, match="symlink"):
        write_action_panel_artifact(panel, linked_parent / "panel.json")
    assert not (real_parent / "panel.json").exists()


@pytest.mark.skipif(not hasattr(os, "link"), reason="hard links unavailable")
def test_panel_loader_rejects_hard_links(tmp_path: Path) -> None:
    panel = _panel()
    original = (tmp_path / "original.json").resolve()
    write_action_panel_artifact(panel, original)
    alias = (tmp_path / "alias.json").resolve()
    os.link(original, alias)
    with pytest.raises(ConfirmatoryAnalysisError, match="hard-linked"):
        load_action_panel_artifact(alias)


def test_panel_loader_rejects_a_file_that_changes_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    target = (tmp_path / "panel.json").resolve()
    write_action_panel_artifact(panel, target)
    stable_signature = artifact_integrity._stable_stat_signature
    regular_file_observations = 0

    def unstable_signature(metadata: os.stat_result) -> tuple[int, ...]:
        nonlocal regular_file_observations
        signature = stable_signature(metadata)
        if stat.S_ISREG(metadata.st_mode):
            regular_file_observations += 1
            return signature + (regular_file_observations,)
        return signature

    monkeypatch.setattr(artifact_integrity, "_stable_stat_signature", unstable_signature)
    with pytest.raises(ConfirmatoryAnalysisError, match="changed during read"):
        load_action_panel_artifact(target)


def test_writers_require_typed_values_and_preserve_existing_content(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "existing.json").resolve()
    target.write_text("custodian-owned", encoding="utf-8")

    with pytest.raises(ConfirmatoryAnalysisError, match="PreLabelActionRow"):
        write_prelabel_action_row(_panel(), (tmp_path / "wrong-row.json").resolve())
    with pytest.raises(ConfirmatoryAnalysisError, match="ActionPanelArtifact"):
        write_action_panel_artifact(_row(), (tmp_path / "wrong-panel.json").resolve())
    with pytest.raises(ConfirmatoryAnalysisError, match="already exists"):
        write_action_panel_artifact(_panel(), target)
    assert target.read_text(encoding="utf-8") == "custodian-owned"
