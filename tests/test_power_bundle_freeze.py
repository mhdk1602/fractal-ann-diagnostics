from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import fractal_ann_diagnostics.freeze_package as freeze_package
import fractal_ann_diagnostics.joint_power_design as joint_power_design
from fractal_ann_diagnostics.freeze_package import (
    FreezeArtifactLayout,
    FreezePackageError,
    _inspect_target,
    verify_joint_power_bundle,
)
from fractal_ann_diagnostics.joint_power_design import (
    FIXED_CORPORA,
    JOINT_ENDPOINT,
    DependenceSource,
    DevelopmentFamilyRow,
    DevelopmentScenarioPanel,
    EffectScenario,
    GeometryGainThresholds,
    JointPowerDesignConfig,
    OperatingProbability,
    canonical_development_panel_bytes,
    canonical_joint_power_config_bytes,
    canonical_joint_power_report_bytes,
    canonical_joint_power_selection_audit_bytes,
    run_joint_power_design,
    run_joint_power_selection_audit,
)


@dataclass(frozen=True)
class _PowerFixture:
    config: JointPowerDesignConfig
    panels: tuple[DevelopmentScenarioPanel, ...]
    selection_audit: object
    report: object
    manifest: dict[str, object]


def _panel(scenario_id: str = "expected") -> DevelopmentScenarioPanel:
    rows: list[DevelopmentFamilyRow] = []
    for corpus in FIXED_CORPORA:
        for family in range(3):
            for nested, label in enumerate((0, 1)):
                evidence = True if corpus in FIXED_CORPORA[:3] else None
                rows.append(
                    DevelopmentFamilyRow(
                        corpus_id=corpus,
                        family_id=f"{corpus}-family-{family}",
                        row_id=f"{scenario_id}-{corpus}-{family}-{nested}",
                        label=label,
                        reference_probability=0.5,
                        full_probability=0.95 if label else 0.05,
                        proposed_latency_ms=5.0,
                        comparator_latency_ms=10.0,
                        proposed_execution_position=(family * 2 + nested) % 4,
                        comparator_execution_position=(family * 2 + nested + 1) % 4,
                        proposed_retrieval_attained=True,
                        comparator_retrieval_attained=True,
                        proposed_evidence_sufficient=evidence,
                        comparator_evidence_sufficient=evidence,
                        denied_emissions=0,
                    )
                )
    return DevelopmentScenarioPanel(
        scenario_id=scenario_id,
        partition="development-calibration",
        rows=tuple(rows),
    )


def _config(panels: tuple[DevelopmentScenarioPanel, ...]) -> JointPowerDesignConfig:
    return JointPowerDesignConfig(
        dependence_source=DependenceSource(
            artifact_uri="s3://immutable-development/power-source.json",
            artifact_sha256="a" * 64,
            partition="development-calibration",
            description="Pinned development query-family endpoint vectors.",
        ),
        effect_scenarios=tuple(
            EffectScenario(
                scenario_id=panel.scenario_id,
                panel_sha256=panel.sha256,
                description=f"Pinned development effect scenario {panel.scenario_id}.",
                selection_required=True,
            )
            for panel in panels
        ),
        candidate_families_per_corpus=(25, 50, 75, 100, 150, 200),
        nested_rows_per_family=2,
        geometry_gain_thresholds=GeometryGainThresholds(
            log_loss_reduction=0.01,
            brier_score_reduction=0.01,
            auprc_gain=0.01,
        ),
        n_simulations=5_000,
        bound_calibration_simulations=5_000,
        simulation_seed=20260713,
        target_power=0.90,
        test_mode=False,
    )


def _selected_lower_bound(report: object) -> float:
    selected = report.selected_families_per_corpus
    return min(
        estimate.joint_probability.lower_probability_bound
        for estimate in report.estimates
        if estimate.families_per_corpus == selected
    )


def _manifest(config: JointPowerDesignConfig, report: object) -> dict[str, object]:
    return {
        "analysis": {
            "alpha": config.alpha,
            "evidence_corpora": list(config.evidence_corpora),
            "evidence_sufficiency_noninferiority_margin": (
                config.evidence_sufficiency_noninferiority_margin
            ),
            "fixed_corpora": list(config.fixed_corpora),
            "geometry_gain_thresholds": config.geometry_gain_thresholds.to_dict(),
            "maximum_entitlement_violations": config.maximum_denied_emissions,
            "maximum_p95_latency_ratio": config.maximum_p95_latency_ratio,
            "minimum_corpora_with_geometry_gain": (config.minimum_corpora_with_geometry_gain),
            "minimum_cost_reduction": config.minimum_latency_reduction,
            "nested_rows_per_family": config.nested_rows_per_family,
            "power_target": config.target_power,
            "retrieval_target_noninferiority_margin": (
                config.retrieval_target_noninferiority_margin
            ),
            "power": {
                "candidate_families_per_corpus": list(config.candidate_families_per_corpus),
                "dependence_source": config.dependence_source.artifact_uri,
                "effect_scenarios": [scenario.scenario_id for scenario in config.effect_scenarios],
                "registered_endpoints": list(config.endpoint_order),
                "selection_cell_alpha": config.selection_cell_alpha,
                "selection_exact_blocking_failures": 445,
                "selection_exact_qualifying_passes": 4_556,
                "selection_family_size": config.selection_family_size,
                "selection_familywise_confidence": config.monte_carlo_confidence,
                "selection_multiplicity_method": config.selection_multiplicity_method,
                "selected_families_per_corpus": (report.selected_families_per_corpus),
                "selected_joint_power_lower_bound": _selected_lower_bound(report),
                "simulation_count": config.n_simulations,
                "simulation_seed": config.simulation_seed,
            },
        }
    }


def _write_bundle(
    root: Path,
    *,
    config: JointPowerDesignConfig,
    panels: tuple[DevelopmentScenarioPanel, ...],
    selection_audit: object,
    report: object,
) -> Path:
    bundle = root / "joint-power-design"
    panel_root = bundle / "panels"
    panel_root.mkdir(parents=True)
    (bundle / "config.json").write_bytes(canonical_joint_power_config_bytes(config))
    (bundle / "report.json").write_bytes(canonical_joint_power_report_bytes(report))
    (bundle / "selection-audit.json").write_bytes(
        canonical_joint_power_selection_audit_bytes(selection_audit)
    )
    for panel in panels:
        (panel_root / f"{panel.sha256}.json").write_bytes(canonical_development_panel_bytes(panel))
    return bundle.resolve()


@pytest.fixture(scope="module")
def power_fixture() -> _PowerFixture:
    panel = _panel()
    conservative = _panel("conservative")
    panels = (panel, conservative)
    config = _config(panels)
    exact_bounds = {
        "h2-log-loss-reduction": 0.50,
        "h2-brier-score-reduction": 0.20,
        "h2-auprc-gain": 0.40,
        "h3-family-relative-latency-reduction": 0.50,
        "h3-retrieval-target-noninferiority": 0.0,
        "h3-complete-evidence-noninferiority": 0.0,
        "h3-family-mean-p95-latency-ratio": 0.50,
        "h3-position-adjusted-log-latency-ratio-sensitivity": math.log(0.50),
    }
    with mock.patch.object(
        joint_power_design,
        "_exact_registered_percentile_bounds",
        return_value=exact_bounds,
    ):
        selection_audit = run_joint_power_selection_audit(config, panels)
        report = run_joint_power_design(config, panels, selection_audit=selection_audit)
        assert report.freeze_ready
        yield _PowerFixture(
            config=config,
            panels=panels,
            selection_audit=selection_audit,
            report=report,
            manifest=_manifest(config, report),
        )


def test_freeze_inspection_accepts_only_a_recomputed_power_bundle(
    tmp_path: Path,
    power_fixture: _PowerFixture,
) -> None:
    artifact_root = tmp_path / "artifacts"
    bundle = _write_bundle(
        artifact_root / "analysis",
        config=power_fixture.config,
        panels=power_fixture.panels,
        selection_audit=power_fixture.selection_audit,
        report=power_fixture.report,
    )
    expected = artifact_root / "analysis" / "joint-power-design"
    assert bundle == expected.resolve()
    layout = FreezeArtifactLayout(
        artifact_id="registered-power-analysis-report",
        role="power-analysis-report",
        relative_path="analysis/joint-power-design",
        kind="directory",
    )

    row = _inspect_target(
        layout,
        artifact_root,
        Path(__file__).resolve().parents[1],
        power_fixture.manifest,
    )

    assert row["state"] == "present"
    assert row["revision"] == f"sha256:{row['sha256']}"


def test_self_consistent_forged_counts_fail_fresh_rerun(
    tmp_path: Path,
    power_fixture: _PowerFixture,
) -> None:
    report = power_fixture.report
    estimate = report.estimates[0]
    passes = np.ones(estimate.n_simulations, dtype=np.bool_)
    passes[0] = False
    endpoint = estimate.endpoint_probabilities[0]
    forged_endpoint = OperatingProbability.from_passes(
        endpoint.endpoint,
        passes,
        confidence=endpoint.confidence,
    )
    forged_joint = OperatingProbability.from_passes(
        JOINT_ENDPOINT,
        passes,
        confidence=estimate.joint_probability.confidence,
    )
    forged_estimate = replace(
        estimate,
        endpoint_probabilities=(
            forged_endpoint,
            *estimate.endpoint_probabilities[1:],
        ),
        joint_probability=forged_joint,
    )
    forged_report = replace(
        report,
        estimates=(forged_estimate, *report.estimates[1:]),
    )
    bundle = _write_bundle(
        tmp_path,
        config=power_fixture.config,
        panels=power_fixture.panels,
        selection_audit=power_fixture.selection_audit,
        report=forged_report,
    )

    with pytest.raises(FreezePackageError, match="fresh run"):
        verify_joint_power_bundle(bundle, _manifest(power_fixture.config, forged_report))


def test_manifest_scalar_drift_is_rejected_after_rerun(
    tmp_path: Path,
    power_fixture: _PowerFixture,
) -> None:
    bundle = _write_bundle(
        tmp_path,
        config=power_fixture.config,
        panels=power_fixture.panels,
        selection_audit=power_fixture.selection_audit,
        report=power_fixture.report,
    )
    manifest = copy.deepcopy(power_fixture.manifest)
    manifest["analysis"]["minimum_cost_reduction"] = 0.11

    with pytest.raises(FreezePackageError, match="minimum_cost_reduction"):
        verify_joint_power_bundle(bundle, manifest)


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("selection_multiplicity_method", "pointwise"),
        ("selection_familywise_confidence", 0.90),
        ("selection_family_size", 6),
        ("selection_cell_alpha", 0.05),
        ("selection_exact_qualifying_passes", 4_555),
        ("selection_exact_blocking_failures", 446),
    ],
)
def test_manifest_selection_multiplicity_drift_is_rejected(
    tmp_path: Path,
    power_fixture: _PowerFixture,
    field: str,
    forged: object,
) -> None:
    bundle = _write_bundle(
        tmp_path,
        config=power_fixture.config,
        panels=power_fixture.panels,
        selection_audit=power_fixture.selection_audit,
        report=power_fixture.report,
    )
    manifest = copy.deepcopy(power_fixture.manifest)
    manifest["analysis"]["power"][field] = forged

    with pytest.raises(FreezePackageError, match=field):
        verify_joint_power_bundle(bundle, manifest)


def test_freeze_inspection_performs_one_exact_replay_and_one_final_tree_readback(
    tmp_path: Path,
    power_fixture: _PowerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    bundle = _write_bundle(
        artifact_root / "analysis",
        config=power_fixture.config,
        panels=power_fixture.panels,
        selection_audit=power_fixture.selection_audit,
        report=power_fixture.report,
    )
    layout = FreezeArtifactLayout(
        artifact_id="registered-power-analysis-report",
        role="power-analysis-report",
        relative_path="analysis/joint-power-design",
        kind="directory",
    )
    exact_replays = 0
    bundle_verifications = 0
    original_exact = freeze_package.verify_joint_power_selection_audit
    original_bundle = freeze_package.verify_joint_power_bundle

    def counted_exact(*args: object, **kwargs: object):
        nonlocal exact_replays
        exact_replays += 1
        return original_exact(*args, **kwargs)

    def counted_bundle(*args: object, **kwargs: object):
        nonlocal bundle_verifications
        bundle_verifications += 1
        return original_bundle(*args, **kwargs)

    monkeypatch.setattr(freeze_package, "verify_joint_power_selection_audit", counted_exact)
    monkeypatch.setattr(freeze_package, "verify_joint_power_bundle", counted_bundle)

    row = _inspect_target(
        layout,
        artifact_root,
        Path(__file__).resolve().parents[1],
        power_fixture.manifest,
    )

    assert row["revision"] == f"sha256:{row['sha256']}"
    assert bundle == artifact_root / "analysis" / "joint-power-design"
    assert bundle_verifications == 1
    assert exact_replays == 1


def test_swapped_panel_payload_is_rejected_before_execution(
    tmp_path: Path,
    power_fixture: _PowerFixture,
) -> None:
    swapped_panel = _panel("swapped")
    swapped_config = replace(
        power_fixture.config,
        effect_scenarios=(
            EffectScenario(
                scenario_id=swapped_panel.scenario_id,
                panel_sha256=swapped_panel.sha256,
                description="Swapped development effect scenario.",
                selection_required=True,
            ),
            next(
                scenario
                for scenario in power_fixture.config.effect_scenarios
                if scenario.scenario_id == power_fixture.panels[1].scenario_id
            ),
        ),
    )
    bundle = _write_bundle(
        tmp_path,
        config=swapped_config,
        panels=(swapped_panel, power_fixture.panels[1]),
        selection_audit=power_fixture.selection_audit,
        report=power_fixture.report,
    )
    panel_path = bundle / "panels" / f"{swapped_panel.sha256}.json"
    panel_path.write_bytes(canonical_development_panel_bytes(power_fixture.panels[0]))

    with pytest.raises(FreezePackageError, match="differs from its config pin"):
        verify_joint_power_bundle(bundle, power_fixture.manifest)


def test_test_mode_bundle_is_never_freeze_admissible(
    tmp_path: Path,
    power_fixture: _PowerFixture,
) -> None:
    config = replace(power_fixture.config, test_mode=True)
    report = replace(power_fixture.report, test_mode=True, freeze_ready=False)
    bundle = _write_bundle(
        tmp_path,
        config=config,
        panels=power_fixture.panels,
        selection_audit=power_fixture.selection_audit,
        report=report,
    )

    with pytest.raises(FreezePackageError, match="test mode"):
        verify_joint_power_bundle(bundle, _manifest(config, report))


def test_missing_selection_audit_cannot_freeze(
    tmp_path: Path,
    power_fixture: _PowerFixture,
) -> None:
    bundle = _write_bundle(
        tmp_path,
        config=power_fixture.config,
        panels=power_fixture.panels,
        selection_audit=power_fixture.selection_audit,
        report=power_fixture.report,
    )
    (bundle / "selection-audit.json").unlink()
    with pytest.raises(FreezePackageError, match="selection audit.*missing"):
        verify_joint_power_bundle(bundle, power_fixture.manifest)
