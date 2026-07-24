from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import fractal_ann_diagnostics.joint_power_design as joint_power_design
from fractal_ann_diagnostics.joint_power_design import (
    CONTINUOUS_ENDPOINTS,
    ENDPOINT_ORDER,
    FIXED_CORPORA,
    POSITION_SENSITIVITY_ENDPOINT,
    PRIMARY_ENDPOINT_ORDER,
    REGISTERED_CANDIDATE_FAMILY_COUNTS,
    DependenceSource,
    DevelopmentFamilyRow,
    DevelopmentScenarioPanel,
    EffectScenario,
    GeometryGainThresholds,
    JointPowerDesignConfig,
    JointPowerDesignError,
    OperatingProbability,
    audit_percentile_approximation,
    canonical_development_panel_bytes,
    canonical_joint_power_config_bytes,
    canonical_joint_power_report_bytes,
    canonical_joint_power_selection_audit_bytes,
    load_development_panel,
    load_joint_power_config,
    load_joint_power_report,
    load_joint_power_selection_audit,
    run_joint_power_design,
    run_joint_power_selection_audit,
    verify_joint_power_selection_audit,
)


def _panel(
    scenario_id: str = "expected",
    *,
    families_by_corpus: dict[str, int] | None = None,
    proposed_latency_by_corpus: dict[str, float] | None = None,
    denied_emissions: int = 0,
) -> DevelopmentScenarioPanel:
    family_counts = families_by_corpus or {corpus: 3 for corpus in FIXED_CORPORA}
    proposed_latency = proposed_latency_by_corpus or {corpus: 5.0 for corpus in FIXED_CORPORA}
    rows: list[DevelopmentFamilyRow] = []
    for corpus in FIXED_CORPORA:
        for family in range(family_counts[corpus]):
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
                        proposed_latency_ms=proposed_latency[corpus],
                        comparator_latency_ms=10.0,
                        proposed_execution_position=(family * 2 + nested) % 4,
                        comparator_execution_position=(family * 2 + nested + 1) % 4,
                        proposed_retrieval_attained=True,
                        comparator_retrieval_attained=True,
                        proposed_evidence_sufficient=evidence,
                        comparator_evidence_sufficient=evidence,
                        denied_emissions=denied_emissions,
                    )
                )
    return DevelopmentScenarioPanel(
        scenario_id=scenario_id,
        partition="development-calibration",
        rows=tuple(reversed(rows)),
    )


def _config(
    panels: tuple[DevelopmentScenarioPanel, ...],
    *,
    candidates: tuple[int, ...] = (2, 4),
    n_simulations: int = 40,
    target_power: float = 0.80,
    selection_required: dict[str, bool] | None = None,
) -> JointPowerDesignConfig:
    required = selection_required or {panel.scenario_id: True for panel in panels}
    return JointPowerDesignConfig(
        dependence_source=DependenceSource(
            artifact_uri="https://example.org/development-power-source.json",
            artifact_sha256="a" * 64,
            partition="development-calibration",
            description=(
                "Paired calibration families with all endpoint rows retained under each "
                "declared effect scenario."
            ),
        ),
        effect_scenarios=tuple(
            EffectScenario(
                scenario_id=panel.scenario_id,
                panel_sha256=panel.sha256,
                description=f"Pinned development effect scenario {panel.scenario_id}.",
                selection_required=required[panel.scenario_id],
            )
            for panel in panels
        ),
        candidate_families_per_corpus=candidates,
        nested_rows_per_family=2,
        geometry_gain_thresholds=GeometryGainThresholds(
            log_loss_reduction=0.01,
            brier_score_reduction=0.01,
            auprc_gain=0.01,
        ),
        n_simulations=n_simulations,
        bound_calibration_simulations=n_simulations,
        target_power=target_power,
        simulation_seed=817263,
        test_mode=True,
    )


def test_canonical_config_panel_and_report_round_trip() -> None:
    panel = _panel()
    config = _config((panel,))
    report = run_joint_power_design(config, (panel,))

    assert load_development_panel(canonical_development_panel_bytes(panel)) == panel
    assert load_joint_power_config(canonical_joint_power_config_bytes(config)) == config
    loaded_report = load_joint_power_report(canonical_joint_power_report_bytes(report))
    assert loaded_report == report
    assert loaded_report.sha256 == report.sha256
    assert canonical_joint_power_report_bytes(report).endswith(b"\n")


def test_canonical_loaders_reject_duplicate_unknown_nonfinite_and_noncanonical_data() -> None:
    panel = _panel()
    config = _config((panel,))
    canonical = canonical_joint_power_config_bytes(config)
    with pytest.raises(JointPowerDesignError, match="duplicate key"):
        load_joint_power_config(b'{"alpha":0.05,' + canonical[1:])

    payload = config.to_dict()
    payload["unregistered"] = True
    with pytest.raises(JointPowerDesignError, match="closed schema"):
        load_joint_power_config(
            json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        )

    with pytest.raises(JointPowerDesignError, match="non-finite"):
        load_joint_power_config(canonical.replace(b'"target_power":0.8', b'"target_power":NaN'))
    with pytest.raises(JointPowerDesignError, match="not canonical"):
        load_joint_power_config(canonical.replace(b'{"alpha"', b'{ "alpha"'))


def test_power_design_is_deterministic_and_uses_pcg64_hash_streams() -> None:
    panel = _panel()
    config = _config((panel,))
    first = run_joint_power_design(config, (panel,))
    second = run_joint_power_design(config, (panel,))

    assert canonical_joint_power_report_bytes(first) == canonical_joint_power_report_bytes(second)
    assert first.rng_engine == "numpy-pcg64-with-sha256-derived-streams"
    assert first.bound_construction == "registered-percentile-family-bootstrap-plug-in-calibration"
    assert first.selected_families_per_corpus == 2
    assert first.selection_satisfied
    assert not first.freeze_ready
    assert first.test_mode


def test_joint_gate_includes_all_h2_h3_endpoints_and_selects_smallest_candidate() -> None:
    panel = _panel()
    report = run_joint_power_design(_config((panel,)), (panel,))
    first = next(item for item in report.estimates if item.families_per_corpus == 2)

    assert tuple(item.endpoint for item in first.endpoint_probabilities) == ENDPOINT_ORDER
    assert all(item.estimated_probability == 1.0 for item in first.endpoint_probabilities)
    assert first.joint_probability.estimated_probability == 1.0
    assert first.joint_probability.lower_probability_bound >= report.target_power
    assert report.selected_families_per_corpus == 2


def test_selection_uses_lower_probability_bound_not_point_power() -> None:
    panel = _panel()
    config = _config(
        (panel,),
        candidates=(2,),
        n_simulations=20,
        target_power=0.90,
    )
    report = run_joint_power_design(config, (panel,))
    estimate = report.estimates[0]

    assert estimate.joint_probability.estimated_probability == 1.0
    assert estimate.joint_probability.lower_probability_bound < config.target_power
    assert report.selected_families_per_corpus is None
    assert not report.selection_satisfied


def test_required_conservative_scenario_can_block_selection() -> None:
    expected = _panel("expected")
    conservative = _panel(
        "conservative",
        proposed_latency_by_corpus={corpus: 15.0 for corpus in FIXED_CORPORA},
    )
    config = _config(
        (expected, conservative),
        selection_required={
            "expected": True,
            "conservative": True,
        },
    )
    report = run_joint_power_design(config, (conservative, expected))
    conservative_estimate = next(
        item
        for item in report.estimates
        if item.scenario_id == "conservative" and item.families_per_corpus == 2
    )

    assert conservative_estimate.joint_probability.estimated_probability == 0.0
    assert report.selected_families_per_corpus is None


def test_sensitivity_only_scenario_does_not_change_closed_selection_rule() -> None:
    expected = _panel("expected")
    sensitivity = _panel(
        "sensitivity",
        proposed_latency_by_corpus={corpus: 15.0 for corpus in FIXED_CORPORA},
    )
    config = _config(
        (expected, sensitivity),
        selection_required={"expected": True, "sensitivity": False},
    )
    report = run_joint_power_design(config, (expected, sensitivity))

    assert report.selected_families_per_corpus == 2
    assert report.selection_satisfied


def test_position_sensitivity_probability_is_reported_but_not_selection_governing() -> None:
    panel = _panel()
    config = _config((panel,))
    report = run_joint_power_design(config, (panel,))
    estimate = report.estimates[0]
    assert estimate.qualifies(config.target_power)
    failed_sensitivity = OperatingProbability.from_passes(
        POSITION_SENSITIVITY_ENDPOINT,
        np.zeros(estimate.n_simulations, dtype=bool),
        confidence=config.monte_carlo_confidence,
    )
    changed = replace(
        estimate,
        endpoint_probabilities=tuple(
            failed_sensitivity if row.endpoint == POSITION_SENSITIVITY_ENDPOINT else row
            for row in estimate.endpoint_probabilities
        ),
    )
    assert not failed_sensitivity.lower_probability_bound
    assert changed.qualifies(config.target_power)


def test_position_sensitivity_uses_pinned_observed_execution_positions() -> None:
    base = _panel()
    observed_rows = []
    for row in base.rows:
        nested = int(row.row_id.rsplit("-", 1)[1])
        observed_rows.append(
            replace(
                row,
                proposed_execution_position=1 + nested,
                comparator_execution_position=1,
                proposed_latency_ms=float(10.0 * np.exp(-0.3 + 0.2 * nested)),
            )
        )
    observed = replace(base, rows=tuple(observed_rows))
    position_erased = replace(
        observed,
        scenario_id="position-erased",
        rows=tuple(
            replace(
                row,
                row_id=f"position-erased:{row.row_id}",
                proposed_execution_position=1,
                comparator_execution_position=1,
            )
            for row in observed.rows
        ),
    )

    observed_report = run_joint_power_design(
        _config((observed,), candidates=(2,), target_power=0.20),
        (observed,),
    )
    erased_report = run_joint_power_design(
        _config((position_erased,), candidates=(2,), target_power=0.20),
        (position_erased,),
    )
    observed_estimands = dict(observed_report.estimates[0].scenario_estimands)
    erased_estimands = dict(erased_report.estimates[0].scenario_estimands)

    assert np.isclose(observed_estimands[POSITION_SENSITIVITY_ENDPOINT], -0.3)
    assert np.isclose(erased_estimands[POSITION_SENSITIVITY_ENDPOINT], -0.2)


def test_equal_corpus_estimand_does_not_weight_large_development_corpus_more() -> None:
    family_counts = {corpus: 2 for corpus in FIXED_CORPORA}
    family_counts["scifact"] = 20
    proposed_latency = {corpus: 5.0 for corpus in FIXED_CORPORA}
    proposed_latency["scifact"] = 9.0
    panel = _panel(
        families_by_corpus=family_counts,
        proposed_latency_by_corpus=proposed_latency,
    )
    report = run_joint_power_design(
        _config((panel,), candidates=(2,), target_power=0.20),
        (panel,),
    )
    estimands = dict(report.estimates[0].scenario_estimands)

    assert np.isclose(estimands["h3-family-relative-latency-reduction"], 0.42)


def test_auprc_is_recomputed_from_raw_rows_with_ties() -> None:
    panel = _panel()
    report = run_joint_power_design(
        _config((panel,), candidates=(2,), target_power=0.20),
        (panel,),
    )
    estimands = dict(report.estimates[0].scenario_estimands)

    assert np.isclose(estimands["h2-auprc-gain"], 0.5)
    assert set(dict(report.estimates[0].percentile_calibration_offsets)) == set(
        CONTINUOUS_ENDPOINTS
    )


@pytest.mark.parametrize("proposed_latency_ms", [5.0, 15.0])
def test_percentile_approximation_agrees_with_registered_exact_bootstrap_gate(
    proposed_latency_ms: float,
) -> None:
    panel = _panel(
        proposed_latency_by_corpus={corpus: proposed_latency_ms for corpus in FIXED_CORPORA}
    )
    config = _config((panel,), candidates=(2,), n_simulations=40, target_power=0.20)

    audit = audit_percentile_approximation(
        config,
        panel,
        families_per_corpus=2,
        exact_bootstrap_replicates=10_000,
    )

    assert audit.exact_bootstrap_replicates == 10_000
    assert audit.decisions_agree
    assert dict(audit.approximate_passes) == dict(audit.exact_passes)
    if proposed_latency_ms > 10.0:
        assert not dict(audit.exact_passes)["h3-family-relative-latency-reduction"]
        assert not dict(audit.exact_passes)["h3-family-mean-p95-latency-ratio"]


def test_percentile_approximation_agrees_for_heterogeneous_nonzero_offsets() -> None:
    panel = _panel(families_by_corpus={corpus: 6 for corpus in FIXED_CORPORA})
    latency_ratios = (0.55, 0.70, 0.85, 0.95, 1.05, 0.75)
    rows = []
    for row in panel.rows:
        family_number = int(row.family_id.rsplit("-", 1)[1])
        full_probability = 0.72 + 0.04 * family_number if row.label else 0.28 - 0.04 * family_number
        rows.append(
            replace(
                row,
                full_probability=full_probability,
                proposed_latency_ms=10.0 * latency_ratios[family_number],
            )
        )
    panel = replace(panel, rows=tuple(rows))
    config = replace(
        _config((panel,), candidates=(4,), n_simulations=40, target_power=0.20),
        bound_calibration_simulations=5_000,
    )

    audits = tuple(
        audit_percentile_approximation(
            config,
            panel,
            families_per_corpus=4,
            study_index=study_index,
            exact_bootstrap_replicates=10_000,
        )
        for study_index in (0, 1)
    )

    assert all(audit.decisions_agree for audit in audits)
    assert [
        dict(audit.exact_passes)["h3-family-relative-latency-reduction"] for audit in audits
    ] == [True, False]
    assert any(
        not np.isclose(
            dict(audit.approximate_bounds)["h3-family-relative-latency-reduction"],
            dict(audit.exact_bounds)["h3-family-relative-latency-reduction"],
        )
        for audit in audits
    )


def test_zero_denied_emission_gate_is_exact_and_reports_finite_zero_event_bound() -> None:
    panel = _panel(denied_emissions=1)
    report = run_joint_power_design(
        _config((panel,), candidates=(2,), target_power=0.20),
        (panel,),
    )
    estimate = report.estimates[0]
    safety = next(
        item
        for item in estimate.endpoint_probabilities
        if item.endpoint == "h3-zero-entitlement-violations"
    )

    assert safety.passing_simulations == 0
    assert estimate.mean_denied_emissions_per_study > 0.0
    assert 0.0 < estimate.zero_event_family_rate_upper_bound_if_no_events < 1.0
    assert estimate.joint_probability.passing_simulations == 0


def test_sealed_partition_and_nonfinite_or_boundary_values_fail_closed() -> None:
    panel = _panel()
    with pytest.raises(JointPowerDesignError, match="sealed outcomes are inadmissible"):
        replace(panel, partition="sealed")
    with pytest.raises(JointPowerDesignError, match="strictly between"):
        replace(panel.rows[0], full_probability=1.0)
    with pytest.raises(JointPowerDesignError, match="finite"):
        replace(panel.rows[0], proposed_latency_ms=float("nan"))
    with pytest.raises(JointPowerDesignError, match="zero to three"):
        replace(panel.rows[0], proposed_execution_position=4)


def test_panel_digest_and_nested_family_cardinality_are_admission_gates() -> None:
    panel = _panel()
    wrong_pin = replace(
        _config((panel,)),
        effect_scenarios=(
            EffectScenario(
                scenario_id=panel.scenario_id,
                panel_sha256="b" * 64,
                description="Deliberately wrong panel pin for admission testing.",
                selection_required=True,
            ),
        ),
    )
    with pytest.raises(JointPowerDesignError, match="digest mismatch"):
        run_joint_power_design(wrong_pin, (panel,))

    wrong_cardinality = replace(_config((panel,)), nested_rows_per_family=3)
    wrong_cardinality = replace(
        wrong_cardinality,
        effect_scenarios=(
            replace(wrong_cardinality.effect_scenarios[0], panel_sha256=panel.sha256),
        ),
    )
    with pytest.raises(JointPowerDesignError, match="expected 3"):
        run_joint_power_design(wrong_cardinality, (panel,))


def test_closed_report_rejects_tampered_selection_and_unknown_fields() -> None:
    panel = _panel()
    report = run_joint_power_design(_config((panel,)), (panel,))
    payload = report.to_dict()
    payload["selected_families_per_corpus"] = 4
    tampered = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    with pytest.raises(JointPowerDesignError, match="closed rule"):
        load_joint_power_report(tampered)

    payload = report.to_dict()
    payload["unregistered"] = "field"
    unknown = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    with pytest.raises(JointPowerDesignError, match="closed schema"):
        load_joint_power_report(unknown)

    payload = report.to_dict()
    payload["estimates"][0]["joint_probability"]["lower_probability_bound"] = 1.0
    false_bound = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    with pytest.raises(JointPowerDesignError, match="exact one-sided limit"):
        load_joint_power_report(false_bound)


def test_freeze_mode_rejects_fewer_than_5000_independent_simulations() -> None:
    panel = _panel()
    conservative = _panel("conservative")
    config = _config(
        (panel, conservative),
        candidates=REGISTERED_CANDIDATE_FAMILY_COUNTS,
    )
    with pytest.raises(JointPowerDesignError, match="at least 5000"):
        replace(config, test_mode=False, n_simulations=4_999)
    with pytest.raises(JointPowerDesignError, match="at least 5000"):
        replace(config, test_mode=False, bound_calibration_simulations=4_999)


@pytest.mark.parametrize(
    "candidate_grid",
    [
        (25, 50, 75, 100, 150),
        (25, 50, 75, 100, 200, 150),
        (25, 50, 75, 100, 150, 150),
        (25, 50, 75, 100, 150, 201),
    ],
)
def test_production_candidate_grid_cannot_be_truncated_reordered_or_substituted(
    candidate_grid: tuple[int, ...],
) -> None:
    panel = _panel()
    conservative = _panel("conservative")
    config = _config(
        (panel, conservative),
        candidates=REGISTERED_CANDIDATE_FAMILY_COUNTS,
        n_simulations=5_000,
        target_power=0.90,
    )
    with pytest.raises(JointPowerDesignError, match="candidate family counts"):
        replace(config, test_mode=False, candidate_families_per_corpus=candidate_grid)


def test_production_probability_contract_cannot_be_weakened() -> None:
    panel = _panel()
    conservative = _panel("conservative")
    config = _config(
        (panel, conservative),
        candidates=REGISTERED_CANDIDATE_FAMILY_COUNTS,
        n_simulations=5_000,
        target_power=0.90,
    )
    with pytest.raises(JointPowerDesignError, match="target_power must equal 0.90"):
        replace(config, test_mode=False, target_power=0.89)
    with pytest.raises(JointPowerDesignError, match="test_mode must be boolean"):
        replace(config, test_mode="false")


def test_production_selection_uses_a_simultaneous_12_cell_probability_family() -> None:
    expected = _panel()
    conservative = _panel("conservative")
    config = _config(
        (expected, conservative),
        candidates=REGISTERED_CANDIDATE_FAMILY_COUNTS,
        n_simulations=5_000,
        target_power=0.90,
    )
    production = replace(config, test_mode=False)

    assert production.selection_family_size == 12
    assert production.selection_cell_alpha == pytest.approx(0.05 / 12)
    required_successes = joint_power_design._minimum_successes_for_probability_target(
        5_000,
        target_power=0.90,
        alpha=production.selection_cell_alpha,
    )
    assert required_successes == 4_556
    assert 5_000 - required_successes + 1 == 445
    assert (
        joint_power_design._one_sided_probability_lower_bound(
            4_556,
            5_000,
            alpha=production.selection_cell_alpha,
        )
        >= 0.90
    )
    assert (
        joint_power_design._one_sided_probability_lower_bound(
            4_555,
            5_000,
            alpha=production.selection_cell_alpha,
        )
        < 0.90
    )

    with pytest.raises(JointPowerDesignError, match="exactly two"):
        replace(
            _config((expected,), candidates=REGISTERED_CANDIDATE_FAMILY_COUNTS), test_mode=False
        )
    with pytest.raises(JointPowerDesignError, match="multiplicity method"):
        replace(production, selection_multiplicity_method="pointwise")


def test_exact_selection_audit_round_trip_and_closed_coverage() -> None:
    panel = _panel()
    config = _config((panel,))
    audit = run_joint_power_selection_audit(
        config,
        (panel,),
        exact_bootstrap_replicates=100,
    )
    encoded = canonical_joint_power_selection_audit_bytes(audit)
    assert load_joint_power_selection_audit(encoded) == audit
    assert verify_joint_power_selection_audit(config, (panel,), audit) == audit

    certificate = audit.certificates[0]
    assert certificate.disposition == "qualified"
    assert certificate.required_successes == 38
    assert certificate.required_failures == 3
    assert certificate.exact_joint_passes == 38
    assert certificate.exact_joint_failures == 0
    assert certificate.audited_study_indices == tuple(range(38))
    assert len(audit.records) == 38
    assert audit.selection_family_size == 2
    assert audit.selection_cell_alpha == pytest.approx(0.025)
    report = run_joint_power_design(config, (panel,), selection_audit=audit)
    assert report.selection_audit_sha256 == audit.sha256
    assert report.selection_audit_basis_sha256 == audit.selection_basis_sha256
    assert not report.freeze_ready


def test_selection_audit_rejects_missing_duplicate_and_primary_disagreement() -> None:
    panel = _panel()
    config = _config((panel,))
    audit = run_joint_power_selection_audit(
        config,
        (panel,),
        exact_bootstrap_replicates=50,
    )
    with pytest.raises(JointPowerDesignError, match="missing study record"):
        replace(audit, records=audit.records[:-1])
    with pytest.raises(JointPowerDesignError, match="repeats an exact study record"):
        replace(audit, records=(*audit.records, audit.records[0]))

    record = audit.records[0]
    exact_passes = dict(record.exact_passes)
    exact_passes[PRIMARY_ENDPOINT_ORDER[0]] = False
    with pytest.raises(JointPowerDesignError, match="primary decision disagreement"):
        replace(
            record,
            exact_passes=tuple(exact_passes.items()),
            exact_joint_passed=False,
        )


def test_position_sensitivity_disagreement_is_non_gating_but_freshly_detected() -> None:
    panel = _panel()
    config = _config((panel,))
    audit = run_joint_power_selection_audit(
        config,
        (panel,),
        exact_bootstrap_replicates=50,
    )
    record = audit.records[0]
    exact_passes = dict(record.exact_passes)
    exact_passes[POSITION_SENSITIVITY_ENDPOINT] = not exact_passes[POSITION_SENSITIVITY_ENDPOINT]
    sensitivity_record = replace(record, exact_passes=tuple(exact_passes.items()))
    assert not sensitivity_record.sensitivity_decisions_agree
    sensitivity_audit = replace(
        audit,
        records=(sensitivity_record, *audit.records[1:]),
    )
    report = run_joint_power_design(
        config,
        (panel,),
        selection_audit=sensitivity_audit,
    )
    assert report.selected_families_per_corpus == audit.selected_families_per_corpus
    with pytest.raises(JointPowerDesignError, match="fresh exact reproduction"):
        verify_joint_power_selection_audit(config, (panel,), sensitivity_audit)


def test_selection_audit_rejects_stale_inputs_and_altered_exact_values() -> None:
    panel = _panel()
    config = _config((panel,))
    audit = run_joint_power_selection_audit(
        config,
        (panel,),
        exact_bootstrap_replicates=50,
    )
    stale_config = replace(config, simulation_seed=config.simulation_seed + 1)
    with pytest.raises(JointPowerDesignError, match="stale, substituted, or mismatched"):
        run_joint_power_design(stale_config, (panel,), selection_audit=audit)

    reordered_audit = replace(
        audit,
        records=(audit.records[1], audit.records[0], *audit.records[2:]),
    )
    with pytest.raises(JointPowerDesignError, match="partial, reordered, or substituted"):
        run_joint_power_design(config, (panel,), selection_audit=reordered_audit)

    substituted_record = replace(audit.records[0], family_draws_sha256="f" * 64)
    substituted_audit = replace(
        audit,
        records=(substituted_record, *audit.records[1:]),
    )
    with pytest.raises(JointPowerDesignError, match="study provenance"):
        run_joint_power_design(config, (panel,), selection_audit=substituted_audit)

    record = audit.records[0]
    exact_bounds = dict(record.exact_bounds)
    endpoint = CONTINUOUS_ENDPOINTS[0]
    exact_bounds[endpoint] = float(exact_bounds[endpoint]) + 0.001
    altered_record = replace(record, exact_bounds=tuple(exact_bounds.items()))
    altered_audit = replace(audit, records=(altered_record, *audit.records[1:]))
    with pytest.raises(JointPowerDesignError, match="fresh exact reproduction"):
        verify_joint_power_selection_audit(config, (panel,), altered_audit)


def test_exact_bootstrap_batching_is_bit_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _panel()
    heterogeneous_rows = tuple(
        replace(
            row,
            reference_probability=0.42 + 0.02 * (index % 5),
            full_probability=(
                0.70 + 0.02 * (index % 4) if row.label else 0.30 - 0.02 * (index % 4)
            ),
            proposed_latency_ms=4.0 + (index % 7),
            comparator_latency_ms=8.0 + (index % 5),
            proposed_retrieval_attained=index % 5 != 0,
            comparator_retrieval_attained=index % 7 != 0,
            proposed_evidence_sufficient=(
                index % 4 != 0 if row.proposed_evidence_sufficient is not None else None
            ),
            comparator_evidence_sufficient=(
                index % 6 != 0 if row.comparator_evidence_sufficient is not None else None
            ),
        )
        for index, row in enumerate(base.rows)
    )
    panel = DevelopmentScenarioPanel(
        scenario_id=base.scenario_id,
        partition=base.partition,
        rows=heterogeneous_rows,
    )
    config = _config((panel,), candidates=(4,))
    prepared = joint_power_design._prepare_panel(panel, config)
    family_draws = joint_power_design._study_family_draws(
        prepared,
        families_per_corpus=4,
        config=config,
        scenario_id=panel.scenario_id,
        phase="power-evaluation",
        study_index=3,
    )

    monkeypatch.setattr(joint_power_design, "EXACT_BOOTSTRAP_BATCH_SIZE", 1_000)
    unbatched = joint_power_design._exact_registered_percentile_bounds(
        prepared,
        family_draws,
        config,
        n_resamples=1_000,
    )
    monkeypatch.setattr(joint_power_design, "EXACT_BOOTSTRAP_BATCH_SIZE", 17)
    batched = joint_power_design._exact_registered_percentile_bounds(
        prepared,
        family_draws,
        config,
        n_resamples=1_000,
    )
    assert batched == unbatched

    audit_panel = _panel("canonical-audit")
    audit_config = _config((audit_panel,))
    monkeypatch.setattr(joint_power_design, "EXACT_BOOTSTRAP_BATCH_SIZE", 50)
    unbatched_audit = run_joint_power_selection_audit(
        audit_config,
        (audit_panel,),
        exact_bootstrap_replicates=50,
    )
    monkeypatch.setattr(joint_power_design, "EXACT_BOOTSTRAP_BATCH_SIZE", 7)
    batched_audit = run_joint_power_selection_audit(
        audit_config,
        (audit_panel,),
        exact_bootstrap_replicates=50,
    )
    assert canonical_joint_power_selection_audit_bytes(
        batched_audit
    ) == canonical_joint_power_selection_audit_bytes(unbatched_audit)
