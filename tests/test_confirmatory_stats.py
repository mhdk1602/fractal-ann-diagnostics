from __future__ import annotations

import numpy as np
import pytest

from fractal_ann_diagnostics.confirmatory_stats import (
    ClusterBootstrapConfig,
    ClusteredBinaryDesign,
    ConfidenceInterval,
    PowerSimulationConfig,
    equal_corpus_weighted_mean,
    equal_corpus_weighted_paired_difference,
    noninferiority_decision,
    paired_stratified_family_bootstrap,
    paired_stratified_metric_bootstrap,
    search_clustered_binary_sample_size,
    simulate_clustered_binary_power,
    superiority_decision,
    upper_limit_decision,
)


def test_equal_corpus_estimand_does_not_weight_rows_or_large_corpora_more() -> None:
    values = (1.0, 1.0, 3.0, 10.0, 10.0, 10.0, 10.0)
    corpus_ids = ("small", "small", "small", "large", "large", "large", "large")
    family_ids = ("a", "a", "b", "c", "c", "c", "c")
    assert equal_corpus_weighted_mean(values, corpus_ids, family_ids) == 6.0


def test_paired_estimand_is_proposed_minus_comparator_with_family_weighting() -> None:
    proposed = (2.0, 2.0, 5.0)
    comparator = (1.0, 1.0, 1.0)
    corpus_ids = ("corpus", "corpus", "corpus")
    family_ids = ("repeated", "repeated", "single")
    assert (
        equal_corpus_weighted_paired_difference(
            proposed,
            comparator,
            corpus_ids,
            family_ids,
            proposed_pair_ids=("r1", "r2", "r3"),
            comparator_pair_ids=("r1", "r2", "r3"),
        )
        == 2.5
    )


def test_bootstrap_never_resamples_corpora() -> None:
    result = paired_stratified_family_bootstrap(
        proposed=(0.0, 0.0, 2.0, 2.0),
        comparator=(0.0, 0.0, 0.0, 0.0),
        corpus_ids=("a", "a", "b", "b"),
        family_ids=("a1", "a2", "b1", "b2"),
        proposed_pair_ids=("a1", "a2", "b1", "b2"),
        comparator_pair_ids=("a1", "a2", "b1", "b2"),
        config=ClusterBootstrapConfig(n_resamples=100, seed=17),
    )
    assert result.interval.estimate == 1.0
    assert np.all(result.replicates == 1.0)
    assert result.resampling_unit == "query_family_within_corpus"
    assert not result.corpora_resampled
    assert not result.nested_rows_resampled
    assert result.interval.construction == "directional-one-sided"


def test_bootstrap_resamples_whole_families_not_nested_rows() -> None:
    result = paired_stratified_family_bootstrap(
        proposed=(0.0, 0.0, 0.0, 0.0, 4.0),
        comparator=(0.0, 0.0, 0.0, 0.0, 0.0),
        corpus_ids=("corpus",) * 5,
        family_ids=("large-family",) * 4 + ("small-family",),
        proposed_pair_ids=("l1", "l2", "l3", "l4", "s1"),
        comparator_pair_ids=("l1", "l2", "l3", "l4", "s1"),
        config=ClusterBootstrapConfig(n_resamples=200, seed=9),
    )
    assert result.interval.estimate == 2.0
    assert set(result.replicates).issubset({0.0, 2.0, 4.0})


def test_bootstrap_is_deterministic_for_a_registered_seed() -> None:
    kwargs = {
        "proposed": (0.0, 1.0, 1.0, 0.0),
        "comparator": (0.0, 0.0, 0.0, 0.0),
        "corpus_ids": ("a", "a", "b", "b"),
        "family_ids": ("a1", "a2", "b1", "b2"),
        "proposed_pair_ids": ("a1", "a2", "b1", "b2"),
        "comparator_pair_ids": ("a1", "a2", "b1", "b2"),
        "config": ClusterBootstrapConfig(n_resamples=100, seed=1234),
    }
    first = paired_stratified_family_bootstrap(**kwargs)
    second = paired_stratified_family_bootstrap(**kwargs)
    np.testing.assert_array_equal(first.replicates, second.replicates)


def test_bootstrap_replicates_are_copied_and_read_only() -> None:
    result = paired_stratified_family_bootstrap(
        proposed=(0.0, 1.0),
        comparator=(0.0, 0.0),
        corpus_ids=("a", "a"),
        family_ids=("a1", "a2"),
        proposed_pair_ids=("a1", "a2"),
        comparator_pair_ids=("a1", "a2"),
        config=ClusterBootstrapConfig(n_resamples=20, seed=3),
    )
    assert not result.replicates.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.replicates[0] = 99.0


def test_h3_metric_bootstrap_uses_registered_estimand_scales() -> None:
    common = {
        "corpus_ids": ("a", "a", "b", "b"),
        "family_ids": ("a1", "a2", "b1", "b2"),
        "proposed_pair_ids": ("a1", "a2", "b1", "b2"),
        "comparator_pair_ids": ("a1", "a2", "b1", "b2"),
        "config": ClusterBootstrapConfig(n_resamples=100, seed=29),
    }
    cost = paired_stratified_metric_bootstrap(
        proposed=(8.0, 8.0, 16.0, 16.0),
        comparator=(10.0, 10.0, 20.0, 20.0),
        metric="relative-reduction",
        **common,
    )
    latency = paired_stratified_metric_bootstrap(
        proposed=(11.0, 11.0, 22.0, 22.0),
        comparator=(10.0, 10.0, 20.0, 20.0),
        metric="p95-ratio",
        **common,
    )
    assert np.isclose(cost.interval.estimate, 0.2)
    assert np.allclose(cost.replicates, 0.2)
    assert np.isclose(latency.interval.estimate, 1.1)
    assert upper_limit_decision(latency.interval, maximum=1.25).passed


def test_ratio_metrics_reject_nonpositive_comparator_values() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        paired_stratified_metric_bootstrap(
            proposed=(1.0, 2.0),
            comparator=(0.0, 2.0),
            corpus_ids=("a", "a"),
            family_ids=("one", "two"),
            proposed_pair_ids=("one", "two"),
            comparator_pair_ids=("one", "two"),
            metric="relative-reduction",
            config=ClusterBootstrapConfig(n_resamples=20),
        )


def test_relative_reduction_is_equal_weighted_across_families() -> None:
    result = paired_stratified_metric_bootstrap(
        proposed=(1.0, 50.0),
        comparator=(2.0, 200.0),
        corpus_ids=("a", "a"),
        family_ids=("one", "two"),
        proposed_pair_ids=("one", "two"),
        comparator_pair_ids=("one", "two"),
        metric="relative-reduction",
        config=ClusterBootstrapConfig(n_resamples=100, seed=7),
    )
    assert np.isclose(result.interval.estimate, 0.625)
    assert not np.isclose(result.interval.estimate, 1.0 - 51.0 / 202.0)


def test_interval_helpers_apply_superiority_and_noninferiority_margins() -> None:
    beneficial = ConfidenceInterval(estimate=0.12, lower=0.11, upper=0.13, confidence=0.95)
    almost_equal = ConfidenceInterval(
        estimate=0.0,
        lower=-0.005,
        upper=0.006,
        confidence=0.95,
    )
    lower_is_better = ConfidenceInterval(
        estimate=-0.15,
        lower=-0.20,
        upper=-0.10,
        confidence=0.95,
    )
    assert superiority_decision(beneficial, minimum_effect=0.10).passed
    assert noninferiority_decision(almost_equal, margin=0.01).passed
    assert superiority_decision(
        lower_is_better,
        minimum_effect=0.05,
        direction="less",
    ).passed


def test_freeze_ready_power_config_enforces_5000_simulations() -> None:
    with pytest.raises(ValueError, match="at least 5000"):
        PowerSimulationConfig(n_simulations=4_999)
    assert PowerSimulationConfig(n_simulations=50, test_mode=True).n_simulations == 50


def test_clustered_binary_power_is_deterministic() -> None:
    design = ClusteredBinaryDesign(
        comparator_probabilities=(0.40, 0.60),
        proposed_probability_difference=0.15,
        families_per_corpus=20,
        nested_rows_per_family=5,
        intraclass_correlation=0.10,
        cross_action_residual_coupling=0.5,
    )
    config = PowerSimulationConfig(
        n_simulations=200,
        seed=41,
        batch_size=40,
        test_mode=True,
    )
    first = simulate_clustered_binary_power(design, config=config)
    second = simulate_clustered_binary_power(design, config=config)
    assert first == second
    assert 0.0 <= first.estimated_power <= 1.0
    assert first.total_families == 40


def test_sample_size_search_uses_lower_monte_carlo_power_bound() -> None:
    design = ClusteredBinaryDesign(
        comparator_probabilities=(0.40, 0.60),
        proposed_probability_difference=0.12,
        families_per_corpus=5,
        nested_rows_per_family=8,
        intraclass_correlation=0.05,
        cross_action_residual_coupling=0.5,
    )
    config = PowerSimulationConfig(
        n_simulations=500,
        target_power=0.80,
        seed=73,
        batch_size=100,
        test_mode=True,
    )
    result = search_clustered_binary_sample_size(
        design,
        candidate_families_per_corpus=(5, 20, 40),
        config=config,
    )
    assert [estimate.families_per_corpus for estimate in result.estimates] == [5, 20, 40]
    assert result.selected_families_per_corpus in {20, 40}
    selected = next(
        estimate
        for estimate in result.estimates
        if estimate.families_per_corpus == result.selected_families_per_corpus
    )
    assert selected.lower_power_bound >= config.target_power


def test_power_simulation_models_cross_action_residual_dependence() -> None:
    common = {
        "comparator_probabilities": (0.45, 0.55),
        "proposed_probability_difference": 0.0,
        "families_per_corpus": 30,
        "nested_rows_per_family": 6,
        "intraclass_correlation": 0.10,
    }
    config = PowerSimulationConfig(
        n_simulations=1_000,
        seed=109,
        batch_size=100,
        test_mode=True,
    )
    estimates = [
        simulate_clustered_binary_power(
            ClusteredBinaryDesign(
                **common,
                cross_action_residual_coupling=coupling,
            ),
            config=config,
        )
        for coupling in (0.0, 0.5, 1.0)
    ]
    assert all(estimate.estimated_power <= 0.08 for estimate in estimates)
    assert estimates[0].passing_simulations > estimates[-1].passing_simulations


@pytest.mark.parametrize("coupling", [-0.01, 1.01, float("nan")])
def test_power_design_rejects_invalid_residual_coupling(coupling: float) -> None:
    with pytest.raises(ValueError, match="residual_coupling"):
        ClusteredBinaryDesign(
            comparator_probabilities=(0.5,),
            proposed_probability_difference=0.1,
            families_per_corpus=10,
            nested_rows_per_family=2,
            intraclass_correlation=0.1,
            cross_action_residual_coupling=coupling,
        )
