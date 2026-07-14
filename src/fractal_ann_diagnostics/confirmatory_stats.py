"""Frozen estimands, clustered inference, and power planning utilities.

The corpus suite is treated as fixed. Query families are the independent
resampling unit; every nested policy, seed, drift condition, and paired action
row stays attached to its family.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.stats import beta, norm

Direction = Literal["greater", "less"]
DecisionKind = Literal["superiority", "noninferiority"]
IntervalConstruction = Literal["two-sided", "directional-one-sided"]
PairedMetric = Literal["mean-difference", "relative-reduction", "p95-ratio"]


def _as_finite_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _as_ids(values: Iterable[Hashable], *, name: str) -> tuple[Hashable, ...]:
    identifiers = tuple(values)
    if not identifiers:
        raise ValueError(f"{name} must be non-empty")
    for identifier in identifiers:
        try:
            hash(identifier)
        except TypeError as exc:
            raise TypeError(f"{name} must contain hashable values") from exc
    return identifiers


def _stable_identifier_key(identifier: Hashable) -> tuple[str, str]:
    return type(identifier).__qualname__, repr(identifier)


def _validate_panel(
    values: Iterable[float],
    corpus_ids: Iterable[Hashable],
    family_ids: Iterable[Hashable],
) -> tuple[np.ndarray, tuple[Hashable, ...], tuple[Hashable, ...]]:
    vector = _as_finite_vector(values, name="values")
    corpora = _as_ids(corpus_ids, name="corpus_ids")
    families = _as_ids(family_ids, name="family_ids")
    if len(vector) != len(corpora) or len(vector) != len(families):
        raise ValueError("values, corpus_ids, and family_ids must have equal length")
    return vector, corpora, families


def _validate_pair_ids(
    proposed_pair_ids: Iterable[Hashable],
    comparator_pair_ids: Iterable[Hashable],
    *,
    expected_length: int,
) -> tuple[Hashable, ...]:
    proposed_ids = _as_ids(proposed_pair_ids, name="proposed_pair_ids")
    comparator_ids = _as_ids(comparator_pair_ids, name="comparator_pair_ids")
    if len(proposed_ids) != expected_length or len(comparator_ids) != expected_length:
        raise ValueError("pair IDs must match the observation count")
    if len(set(proposed_ids)) != len(proposed_ids):
        raise ValueError("pair IDs must be unique")
    if proposed_ids != comparator_ids:
        raise ValueError("proposed and comparator pair IDs must match in exact order")
    return proposed_ids


def _family_means_by_corpus(
    values: np.ndarray,
    corpus_ids: tuple[Hashable, ...],
    family_ids: tuple[Hashable, ...],
) -> dict[Hashable, np.ndarray]:
    grouped: dict[tuple[Hashable, Hashable], list[float]] = {}
    for value, corpus, family in zip(values, corpus_ids, family_ids, strict=True):
        grouped.setdefault((corpus, family), []).append(float(value))

    by_corpus: dict[Hashable, list[tuple[Hashable, float]]] = {}
    for (corpus, family), nested_values in grouped.items():
        by_corpus.setdefault(corpus, []).append((family, float(np.mean(nested_values))))

    result: dict[Hashable, np.ndarray] = {}
    for corpus in sorted(by_corpus, key=_stable_identifier_key):
        ordered = sorted(by_corpus[corpus], key=lambda item: _stable_identifier_key(item[0]))
        result[corpus] = np.asarray([mean for _, mean in ordered], dtype=np.float64)
    return result


def _paired_family_values_by_corpus(
    proposed: np.ndarray,
    comparator: np.ndarray,
    corpus_ids: tuple[Hashable, ...],
    family_ids: tuple[Hashable, ...],
) -> dict[Hashable, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[Hashable, Hashable], list[tuple[float, float]]] = {}
    for proposed_value, comparator_value, corpus, family in zip(
        proposed,
        comparator,
        corpus_ids,
        family_ids,
        strict=True,
    ):
        grouped.setdefault((corpus, family), []).append(
            (float(proposed_value), float(comparator_value))
        )

    by_corpus: dict[Hashable, list[tuple[Hashable, float, float]]] = {}
    for (corpus, family), nested_pairs in grouped.items():
        nested = np.asarray(nested_pairs, dtype=np.float64)
        by_corpus.setdefault(corpus, []).append(
            (family, float(nested[:, 0].mean()), float(nested[:, 1].mean()))
        )

    result: dict[Hashable, tuple[np.ndarray, np.ndarray]] = {}
    for corpus in sorted(by_corpus, key=_stable_identifier_key):
        ordered = sorted(by_corpus[corpus], key=lambda item: _stable_identifier_key(item[0]))
        result[corpus] = (
            np.asarray([item[1] for item in ordered], dtype=np.float64),
            np.asarray([item[2] for item in ordered], dtype=np.float64),
        )
    return result


def _paired_metric(
    proposed: np.ndarray,
    comparator: np.ndarray,
    metric: PairedMetric,
    *,
    axis: int | None = None,
) -> np.ndarray | float:
    if metric == "mean-difference":
        return np.mean(proposed - comparator, axis=axis)
    if np.any(comparator <= 0.0):
        raise ValueError(f"comparator values must be positive for {metric}")
    if metric == "relative-reduction":
        return np.mean(1.0 - proposed / comparator, axis=axis)
    if metric == "p95-ratio":
        return np.quantile(proposed, 0.95, axis=axis) / np.quantile(
            comparator,
            0.95,
            axis=axis,
        )
    raise ValueError(f"unsupported paired metric: {metric!r}")


def equal_corpus_weighted_mean(
    values: Iterable[float],
    corpus_ids: Iterable[Hashable],
    family_ids: Iterable[Hashable],
) -> float:
    """Average rows within family, families within corpus, then corpora equally."""
    vector, corpora, families = _validate_panel(values, corpus_ids, family_ids)
    family_means = _family_means_by_corpus(vector, corpora, families)
    corpus_means = [float(means.mean()) for means in family_means.values()]
    return float(np.mean(corpus_means))


def equal_corpus_weighted_paired_difference(
    proposed: Iterable[float],
    comparator: Iterable[float],
    corpus_ids: Iterable[Hashable],
    family_ids: Iterable[Hashable],
    *,
    proposed_pair_ids: Iterable[Hashable],
    comparator_pair_ids: Iterable[Hashable],
) -> float:
    """Estimate proposed minus comparator without treating action rows as independent."""
    proposed_vector = _as_finite_vector(proposed, name="proposed")
    comparator_vector = _as_finite_vector(comparator, name="comparator")
    if proposed_vector.shape != comparator_vector.shape:
        raise ValueError("proposed and comparator must have equal length")
    _validate_pair_ids(
        proposed_pair_ids,
        comparator_pair_ids,
        expected_length=len(proposed_vector),
    )
    return equal_corpus_weighted_mean(
        proposed_vector - comparator_vector,
        corpus_ids,
        family_ids,
    )


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    construction: IntervalConstruction = "two-sided"

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        if not all(np.isfinite((self.estimate, self.lower, self.upper))):
            raise ValueError("interval values must be finite")
        if self.lower > self.upper:
            raise ValueError("lower interval bound cannot exceed upper bound")
        if self.construction not in {"two-sided", "directional-one-sided"}:
            raise ValueError("unsupported interval construction")


@dataclass(frozen=True)
class ClusterBootstrapConfig:
    n_resamples: int = 10_000
    confidence: float = 0.95
    seed: int = 20260713
    batch_size: int = 500
    interval_construction: IntervalConstruction = "directional-one-sided"

    def __post_init__(self) -> None:
        if self.n_resamples <= 0:
            raise ValueError("n_resamples must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.interval_construction not in {"two-sided", "directional-one-sided"}:
            raise ValueError("unsupported interval construction")


@dataclass(frozen=True)
class ClusterBootstrapResult:
    interval: ConfidenceInterval
    replicates: np.ndarray
    n_corpora: int
    n_families: int
    seed: int
    resampling_unit: str = "query_family_within_corpus"
    corpora_resampled: bool = False
    nested_rows_resampled: bool = False

    def __post_init__(self) -> None:
        replicates = np.asarray(self.replicates, dtype=np.float64)
        if replicates.ndim != 1 or replicates.size == 0:
            raise ValueError("replicates must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(replicates)):
            raise ValueError("replicates must contain only finite values")
        frozen = replicates.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "replicates", frozen)
        if self.n_corpora <= 0 or self.n_families <= 0:
            raise ValueError("bootstrap result counts must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def paired_stratified_family_bootstrap(
    proposed: Iterable[float],
    comparator: Iterable[float],
    corpus_ids: Iterable[Hashable],
    family_ids: Iterable[Hashable],
    *,
    proposed_pair_ids: Iterable[Hashable],
    comparator_pair_ids: Iterable[Hashable],
    config: ClusterBootstrapConfig | None = None,
) -> ClusterBootstrapResult:
    """Bootstrap paired differences by family, stratified within fixed corpora.

    Corpora are never sampled. Within each corpus, whole query families are
    sampled with replacement; all nested rows have already been attached by
    reducing each family to its paired mean difference.
    """
    cfg = config or ClusterBootstrapConfig()
    proposed_vector = _as_finite_vector(proposed, name="proposed")
    comparator_vector = _as_finite_vector(comparator, name="comparator")
    if proposed_vector.shape != comparator_vector.shape:
        raise ValueError("proposed and comparator must have equal length")
    _validate_pair_ids(
        proposed_pair_ids,
        comparator_pair_ids,
        expected_length=len(proposed_vector),
    )
    differences, corpora, families = _validate_panel(
        proposed_vector - comparator_vector,
        corpus_ids,
        family_ids,
    )
    by_corpus = _family_means_by_corpus(differences, corpora, families)
    if any(len(family_means) < 2 for family_means in by_corpus.values()):
        raise ValueError("each corpus needs at least two independent query families")
    estimate = float(np.mean([means.mean() for means in by_corpus.values()]))
    rng = np.random.default_rng(cfg.seed)
    replicates = np.zeros(cfg.n_resamples, dtype=np.float64)

    for family_means in by_corpus.values():
        n_families = len(family_means)
        for start in range(0, cfg.n_resamples, cfg.batch_size):
            stop = min(start + cfg.batch_size, cfg.n_resamples)
            draws = rng.integers(0, n_families, size=(stop - start, n_families))
            replicates[start:stop] += family_means[draws].mean(axis=1) / len(by_corpus)

    alpha = 1.0 - cfg.confidence
    if cfg.interval_construction == "two-sided":
        quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    else:
        # Each endpoint is a separate directional 1-alpha bound. The pair is
        # not a two-sided 1-alpha interval and must not be interpreted as one.
        quantiles = (alpha, 1.0 - alpha)
    lower, upper = np.quantile(replicates, quantiles)
    interval = ConfidenceInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=cfg.confidence,
        construction=cfg.interval_construction,
    )
    return ClusterBootstrapResult(
        interval=interval,
        replicates=replicates,
        n_corpora=len(by_corpus),
        n_families=sum(len(means) for means in by_corpus.values()),
        seed=cfg.seed,
    )


def paired_stratified_metric_bootstrap(
    proposed: Iterable[float],
    comparator: Iterable[float],
    corpus_ids: Iterable[Hashable],
    family_ids: Iterable[Hashable],
    *,
    metric: PairedMetric,
    proposed_pair_ids: Iterable[Hashable],
    comparator_pair_ids: Iterable[Hashable],
    config: ClusterBootstrapConfig | None = None,
) -> ClusterBootstrapResult:
    """Bootstrap a locked H3 metric by whole family within each fixed corpus.

    Nested rows are averaged within family first. Cost is expressed as relative
    reduction, while tail latency is the proposed/comparator p95 ratio. Corpus
    estimates receive equal weight regardless of corpus size.
    """
    cfg = config or ClusterBootstrapConfig()
    proposed_vector = _as_finite_vector(proposed, name="proposed")
    comparator_vector = _as_finite_vector(comparator, name="comparator")
    if proposed_vector.shape != comparator_vector.shape:
        raise ValueError("proposed and comparator must have equal length")
    _validate_pair_ids(
        proposed_pair_ids,
        comparator_pair_ids,
        expected_length=len(proposed_vector),
    )
    _, corpora, families = _validate_panel(
        proposed_vector,
        corpus_ids,
        family_ids,
    )
    by_corpus = _paired_family_values_by_corpus(
        proposed_vector,
        comparator_vector,
        corpora,
        families,
    )
    if any(len(values[0]) < 2 for values in by_corpus.values()):
        raise ValueError("each corpus needs at least two independent query families")
    corpus_estimates = [
        float(_paired_metric(proposed_values, comparator_values, metric))
        for proposed_values, comparator_values in by_corpus.values()
    ]
    estimate = float(np.mean(corpus_estimates))
    rng = np.random.default_rng(cfg.seed)
    replicates = np.zeros(cfg.n_resamples, dtype=np.float64)

    for proposed_values, comparator_values in by_corpus.values():
        n_families = len(proposed_values)
        for start in range(0, cfg.n_resamples, cfg.batch_size):
            stop = min(start + cfg.batch_size, cfg.n_resamples)
            draws = rng.integers(0, n_families, size=(stop - start, n_families))
            metric_values = _paired_metric(
                proposed_values[draws],
                comparator_values[draws],
                metric,
                axis=1,
            )
            replicates[start:stop] += np.asarray(metric_values) / len(by_corpus)

    alpha = 1.0 - cfg.confidence
    if cfg.interval_construction == "two-sided":
        quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    else:
        quantiles = (alpha, 1.0 - alpha)
    lower, upper = np.quantile(replicates, quantiles)
    return ClusterBootstrapResult(
        interval=ConfidenceInterval(
            estimate=estimate,
            lower=float(lower),
            upper=float(upper),
            confidence=cfg.confidence,
            construction=cfg.interval_construction,
        ),
        replicates=replicates,
        n_corpora=len(by_corpus),
        n_families=sum(len(values[0]) for values in by_corpus.values()),
        seed=cfg.seed,
    )


def upper_limit_decision(
    interval: ConfidenceInterval,
    *,
    maximum: float,
) -> IntervalDecision:
    """Pass a directional constraint only when its upper bound is below a maximum."""
    if not np.isfinite(maximum):
        raise ValueError("maximum must be finite")
    return IntervalDecision(
        kind="superiority",
        direction="less",
        passed=interval.upper < maximum,
        observed_bound=interval.upper,
        threshold=maximum,
        margin=0.0,
    )


@dataclass(frozen=True)
class IntervalDecision:
    kind: DecisionKind
    direction: Direction
    passed: bool
    observed_bound: float
    threshold: float
    margin: float


def superiority_decision(
    interval: ConfidenceInterval,
    *,
    minimum_effect: float = 0.0,
    direction: Direction = "greater",
) -> IntervalDecision:
    """Test whether the confidence interval clears a superiority threshold."""
    if minimum_effect < 0.0 or not np.isfinite(minimum_effect):
        raise ValueError("minimum_effect must be finite and non-negative")
    if direction == "greater":
        bound = interval.lower
        threshold = minimum_effect
        passed = bound > threshold
    elif direction == "less":
        bound = interval.upper
        threshold = -minimum_effect
        passed = bound < threshold
    else:
        raise ValueError("direction must be 'greater' or 'less'")
    return IntervalDecision(
        kind="superiority",
        direction=direction,
        passed=passed,
        observed_bound=bound,
        threshold=threshold,
        margin=minimum_effect,
    )


def noninferiority_decision(
    interval: ConfidenceInterval,
    *,
    margin: float,
    direction: Direction = "greater",
) -> IntervalDecision:
    """Test noninferiority when the estimand is proposed minus comparator."""
    if margin < 0.0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and non-negative")
    if direction == "greater":
        bound = interval.lower
        threshold = -margin
        passed = bound > threshold
    elif direction == "less":
        bound = interval.upper
        threshold = margin
        passed = bound < threshold
    else:
        raise ValueError("direction must be 'greater' or 'less'")
    return IntervalDecision(
        kind="noninferiority",
        direction=direction,
        passed=passed,
        observed_bound=bound,
        threshold=threshold,
        margin=margin,
    )


@dataclass(frozen=True)
class ClusteredBinaryDesign:
    """Paired beta-binomial design for an event coded as favorable."""

    comparator_probabilities: tuple[float, ...]
    proposed_probability_difference: float
    families_per_corpus: int
    nested_rows_per_family: int
    intraclass_correlation: float
    cross_action_residual_coupling: float = 0.5
    decision: DecisionKind = "superiority"
    margin: float = 0.0

    def __post_init__(self) -> None:
        if not self.comparator_probabilities:
            raise ValueError("at least one corpus probability is required")
        if not np.isfinite(self.proposed_probability_difference):
            raise ValueError("proposed_probability_difference must be finite")
        if any(not 0.0 < probability < 1.0 for probability in self.comparator_probabilities):
            raise ValueError("comparator probabilities must be in (0, 1)")
        proposed = [
            probability + self.proposed_probability_difference
            for probability in self.comparator_probabilities
        ]
        if any(not 0.0 < probability < 1.0 for probability in proposed):
            raise ValueError("proposed probabilities must be in (0, 1)")
        if self.families_per_corpus < 2:
            raise ValueError("families_per_corpus must be at least two")
        if self.nested_rows_per_family <= 0:
            raise ValueError("nested_rows_per_family must be positive")
        if not np.isfinite(self.intraclass_correlation) or not (
            0.0 <= self.intraclass_correlation < 1.0
        ):
            raise ValueError("intraclass_correlation must be in [0, 1)")
        if not np.isfinite(self.cross_action_residual_coupling) or not (
            0.0 <= self.cross_action_residual_coupling <= 1.0
        ):
            raise ValueError("cross_action_residual_coupling must be in [0, 1]")
        if not np.isfinite(self.margin) or self.margin < 0.0:
            raise ValueError("margin must be finite and non-negative")
        if self.decision not in {"superiority", "noninferiority"}:
            raise ValueError("decision must be 'superiority' or 'noninferiority'")


@dataclass(frozen=True)
class PowerSimulationConfig:
    n_simulations: int = 5_000
    alpha: float = 0.05
    target_power: float = 0.90
    seed: int = 20260713
    batch_size: int = 250
    test_mode: bool = False

    def __post_init__(self) -> None:
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if not self.test_mode and self.n_simulations < 5_000:
            raise ValueError("freeze-ready power simulation requires at least 5000 studies")
        if not 0.0 < self.alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5)")
        if not 0.0 < self.target_power < 1.0:
            raise ValueError("target_power must be in (0, 1)")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class PowerEstimate:
    families_per_corpus: int
    total_families: int
    n_simulations: int
    passing_simulations: int
    estimated_power: float
    monte_carlo_standard_error: float
    lower_power_bound: float
    upper_power_bound: float
    seed: int


@dataclass(frozen=True)
class SampleSizeSearchResult:
    estimates: tuple[PowerEstimate, ...]
    selected_families_per_corpus: int | None
    target_power: float


def _beta_shapes(probability: float, intraclass_correlation: float) -> tuple[float, float]:
    concentration = 1.0 / intraclass_correlation - 1.0
    return probability * concentration, (1.0 - probability) * concentration


def _paired_binomial_counts(
    rng: np.random.Generator,
    comparator_latent: np.ndarray,
    proposed_latent: np.ndarray,
    *,
    nested_rows_per_family: int,
    residual_coupling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw paired Bernoulli rows with a registered common-shock coupling.

    ``residual_coupling`` is the probability that an action pair uses the same
    uniform shock. It is an interpretable dependence parameter, not a claim
    about a constant Pearson correlation when arm probabilities differ.
    """
    shape = (*comparator_latent.shape, nested_rows_per_family)
    if residual_coupling == 0.0:
        comparator_uniform = rng.random(shape)
        proposed_uniform = rng.random(shape)
    elif residual_coupling == 1.0:
        shared_uniform = rng.random(shape)
        comparator_uniform = shared_uniform
        proposed_uniform = shared_uniform
    else:
        shared_uniform = rng.random(shape)
        coupled = rng.random(shape) < residual_coupling
        comparator_uniform = np.where(coupled, shared_uniform, rng.random(shape))
        proposed_uniform = np.where(coupled, shared_uniform, rng.random(shape))
    comparator_counts = np.sum(
        comparator_uniform < comparator_latent[..., np.newaxis],
        axis=-1,
    )
    proposed_counts = np.sum(
        proposed_uniform < proposed_latent[..., np.newaxis],
        axis=-1,
    )
    return comparator_counts, proposed_counts


def simulate_clustered_binary_power(
    design: ClusteredBinaryDesign,
    *,
    config: PowerSimulationConfig | None = None,
) -> PowerEstimate:
    """Estimate power for a paired, equal-corpus clustered binary design.

    A shared beta quantile aligns hard and easy families across actions. Each
    action then receives a binomial count for all nested rows in that family.
    The test uses a one-sided normal bound on the fixed-suite equal-corpus
    paired difference; confirmatory execution should use the registered family
    bootstrap on observed rows.
    """
    cfg = config or PowerSimulationConfig()
    seed_sequence = np.random.SeedSequence([cfg.seed, design.families_per_corpus])
    rng = np.random.default_rng(seed_sequence)
    critical_value = float(norm.ppf(1.0 - cfg.alpha))
    passing = 0
    remaining = cfg.n_simulations

    while remaining:
        batch = min(cfg.batch_size, remaining)
        estimates = np.zeros(batch, dtype=np.float64)
        variances = np.zeros(batch, dtype=np.float64)
        n_corpora = len(design.comparator_probabilities)

        for comparator_probability in design.comparator_probabilities:
            proposed_probability = (
                comparator_probability + design.proposed_probability_difference
            )
            if design.intraclass_correlation == 0.0:
                comparator_latent = np.full(
                    (batch, design.families_per_corpus),
                    comparator_probability,
                    dtype=np.float64,
                )
                proposed_latent = np.full_like(comparator_latent, proposed_probability)
            else:
                shared_quantiles = rng.random((batch, design.families_per_corpus))
                comparator_shape = _beta_shapes(
                    comparator_probability,
                    design.intraclass_correlation,
                )
                proposed_shape = _beta_shapes(
                    proposed_probability,
                    design.intraclass_correlation,
                )
                comparator_latent = beta.ppf(shared_quantiles, *comparator_shape)
                proposed_latent = beta.ppf(shared_quantiles, *proposed_shape)

            comparator_counts, proposed_counts = _paired_binomial_counts(
                rng,
                comparator_latent,
                proposed_latent,
                nested_rows_per_family=design.nested_rows_per_family,
                residual_coupling=design.cross_action_residual_coupling,
            )
            family_differences = (
                proposed_counts - comparator_counts
            ) / design.nested_rows_per_family
            estimates += family_differences.mean(axis=1) / n_corpora
            variances += (
                family_differences.var(axis=1, ddof=1)
                / design.families_per_corpus
                / n_corpora**2
            )

        standard_errors = np.sqrt(np.clip(variances, 0.0, None))
        if design.decision == "superiority":
            lower_bounds = estimates - critical_value * standard_errors
            passed = lower_bounds > design.margin
        elif design.decision == "noninferiority":
            lower_bounds = estimates - critical_value * standard_errors
            passed = lower_bounds > -design.margin
        else:
            raise ValueError("decision must be 'superiority' or 'noninferiority'")
        passing += int(passed.sum())
        remaining -= batch

    estimated_power = passing / cfg.n_simulations
    monte_carlo_standard_error = float(
        np.sqrt(estimated_power * (1.0 - estimated_power) / cfg.n_simulations)
    )
    if passing == 0:
        lower = 0.0
    else:
        lower = float(beta.ppf(cfg.alpha / 2.0, passing, cfg.n_simulations - passing + 1))
    if passing == cfg.n_simulations:
        upper = 1.0
    else:
        upper = float(
            beta.ppf(
                1.0 - cfg.alpha / 2.0,
                passing + 1,
                cfg.n_simulations - passing,
            )
        )
    return PowerEstimate(
        families_per_corpus=design.families_per_corpus,
        total_families=design.families_per_corpus * len(design.comparator_probabilities),
        n_simulations=cfg.n_simulations,
        passing_simulations=passing,
        estimated_power=estimated_power,
        monte_carlo_standard_error=monte_carlo_standard_error,
        lower_power_bound=lower,
        upper_power_bound=upper,
        seed=cfg.seed,
    )


def search_clustered_binary_sample_size(
    design: ClusteredBinaryDesign,
    candidate_families_per_corpus: Sequence[int],
    *,
    config: PowerSimulationConfig | None = None,
) -> SampleSizeSearchResult:
    """Return the first candidate whose lower Monte Carlo bound reaches target power."""
    cfg = config or PowerSimulationConfig()
    raw_candidates = tuple(candidate_families_per_corpus)
    if any(
        not isinstance(candidate, int) or isinstance(candidate, bool)
        for candidate in raw_candidates
    ):
        raise TypeError("candidate family counts must be integers")
    candidates = sorted(set(raw_candidates))
    if not candidates or candidates[0] < 2:
        raise ValueError("candidate family counts must contain integers of at least two")
    estimates = tuple(
        simulate_clustered_binary_power(
            replace(design, families_per_corpus=families),
            config=cfg,
        )
        for families in candidates
    )
    selected = next(
        (
            estimate.families_per_corpus
            for estimate in estimates
            if estimate.lower_power_bound >= cfg.target_power
        ),
        None,
    )
    return SampleSizeSearchResult(
        estimates=estimates,
        selected_families_per_corpus=selected,
        target_power=cfg.target_power,
    )
