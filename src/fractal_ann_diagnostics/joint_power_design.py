"""Joint H2/H3 design assurance from development family clusters.

This module is deliberately separate from the sealed analysis runner. It
accepts only development partitions and estimates the operating probability
of the full registered gate by resampling whole query families inside each of
five fixed corpora. Raw labels and model probabilities are retained so AUPRC
is recomputed as a corpus metric rather than treated as an additive family
quantity. Continuous gates use a plug-in approximation to the registered
percentile family bootstrap: an independent calibration stream estimates the
centered lower or upper percentile that is added to each simulated study's
point estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import beta

from .confirmatory_stats import (
    ClusterBootstrapConfig,
    paired_stratified_family_bootstrap,
    paired_stratified_metric_bootstrap,
)

CONFIG_SCHEMA = "fractal-joint-power-design-config-v3"
PANEL_SCHEMA = "fractal-joint-power-development-panel-v2"
ROW_SCHEMA = "fractal-joint-power-development-row-v2"
REPORT_SCHEMA = "fractal-joint-power-design-report-v4"
PROBABILITY_SCHEMA = "fractal-operating-probability-v2"
ESTIMATE_SCHEMA = "fractal-joint-power-candidate-estimate-v3"
SELECTION_AUDIT_SCHEMA = "fractal-joint-power-selection-audit-v2"
SELECTION_AUDIT_RECORD_SCHEMA = "fractal-joint-power-selection-audit-record-v1"
SELECTION_AUDIT_CERTIFICATE_SCHEMA = "fractal-joint-power-selection-certificate-v2"
SELECTION_BASIS_SCHEMA = "fractal-joint-power-selection-basis-v2"

FIXED_CORPORA = (
    "scifact",
    "hotpotqa-fullwiki",
    "t2-ragbench",
    "bright",
    "miracl-transfer",
)
EVIDENCE_CORPORA = (
    "scifact",
    "hotpotqa-fullwiki",
    "t2-ragbench",
)
PRIMARY_ENDPOINT_ORDER = (
    "h2-log-loss-reduction",
    "h2-brier-score-reduction",
    "h2-auprc-gain",
    "h2-four-of-five-consistency",
    "h3-family-relative-latency-reduction",
    "h3-retrieval-target-noninferiority",
    "h3-complete-evidence-noninferiority",
    "h3-family-mean-p95-latency-ratio",
    "h3-zero-entitlement-violations",
)
POSITION_SENSITIVITY_ENDPOINT = "h3-position-adjusted-log-latency-ratio-sensitivity"
ENDPOINT_ORDER = PRIMARY_ENDPOINT_ORDER + (POSITION_SENSITIVITY_ENDPOINT,)
CONTINUOUS_ENDPOINTS = (
    "h2-log-loss-reduction",
    "h2-brier-score-reduction",
    "h2-auprc-gain",
    "h3-family-relative-latency-reduction",
    "h3-retrieval-target-noninferiority",
    "h3-complete-evidence-noninferiority",
    "h3-family-mean-p95-latency-ratio",
    POSITION_SENSITIVITY_ENDPOINT,
)
JOINT_ENDPOINT = "h2-and-h3-all-gates-pass"
DESIGN_METHOD = "independent-calibration-and-evaluation-family-cluster-resampling"
BOUND_CONSTRUCTION = "registered-percentile-family-bootstrap-plug-in-calibration"
RNG_ENGINE = "numpy-pcg64-with-sha256-derived-streams"
REGISTERED_BOOTSTRAP_REPLICATES = 10_000
REGISTERED_BOOTSTRAP_SEED = 20260713
EXACT_BOOTSTRAP_BATCH_SIZE = 500
REGISTERED_CANDIDATE_FAMILY_COUNTS = (25, 50, 75, 100, 150, 200)
REGISTERED_REQUIRED_SCENARIO_COUNT = 2
REGISTERED_SELECTION_FAMILY_SIZE = (
    len(REGISTERED_CANDIDATE_FAMILY_COUNTS) * REGISTERED_REQUIRED_SCENARIO_COUNT
)
SELECTION_MULTIPLICITY_METHOD = "bonferroni-fixed-required-scenario-candidate-grid-v1"
SELECTION_RULE = (
    "smallest-candidate-for-which-every-primary-gate-and-the-joint-gate-have-"
    "bonferroni-simultaneous-lower-probability-bounds-at-or-above-target-in-every-"
    "required-scenario"
)
SELECTION_AUDIT_COVERAGE_RULE = (
    "deterministic-sequential-exact-joint-bonferroni-certificate-v2: divide familywise alpha "
    "equally over the fixed required-scenario-by-candidate grid; qualify with the minimum "
    "exact joint-pass count whose one-sided Clopper-Pearson lower bound at the per-cell alpha "
    "reaches target; block with the complementary exact joint-failure count; inspect "
    "approximate-pass-first for a provisionally qualifying cell and approximate-fail-first "
    "otherwise"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 256 * 1024 * 1024
_CONFIG_FIELDS = {
    "alpha",
    "bound_calibration_simulations",
    "bound_construction",
    "candidate_families_per_corpus",
    "dependence_source",
    "effect_scenarios",
    "endpoint_order",
    "evidence_corpora",
    "evidence_sufficiency_noninferiority_margin",
    "fixed_corpora",
    "geometry_gain_thresholds",
    "maximum_denied_emissions",
    "maximum_p95_latency_ratio",
    "minimum_corpora_with_geometry_gain",
    "minimum_latency_reduction",
    "monte_carlo_confidence",
    "nested_rows_per_family",
    "n_simulations",
    "protocol_version",
    "retrieval_target_noninferiority_margin",
    "rng_engine",
    "schema_version",
    "selection_multiplicity_method",
    "selection_rule",
    "simulation_seed",
    "target_power",
    "test_mode",
}
_DEPENDENCE_FIELDS = {
    "artifact_sha256",
    "artifact_uri",
    "construction",
    "description",
    "partition",
}
_SCENARIO_FIELDS = {
    "description",
    "panel_sha256",
    "scenario_id",
    "selection_required",
}
_THRESHOLD_FIELDS = {"auprc_gain", "brier_score_reduction", "log_loss_reduction"}
_PANEL_FIELDS = {"partition", "rows", "scenario_id", "schema_version"}
_ROW_FIELDS = {
    "comparator_evidence_sufficient",
    "comparator_execution_position",
    "comparator_latency_ms",
    "comparator_retrieval_attained",
    "corpus_id",
    "denied_emissions",
    "family_id",
    "full_probability",
    "label",
    "proposed_evidence_sufficient",
    "proposed_execution_position",
    "proposed_latency_ms",
    "proposed_retrieval_attained",
    "reference_probability",
    "row_id",
    "schema_version",
}
_PROBABILITY_FIELDS = {
    "confidence",
    "endpoint",
    "estimated_probability",
    "lower_probability_bound",
    "monte_carlo_standard_error",
    "n_simulations",
    "passing_simulations",
    "schema_version",
}
_ESTIMATE_FIELDS = {
    "bound_calibration_simulations",
    "endpoint_probabilities",
    "families_per_corpus",
    "joint_probability",
    "mean_denied_emissions_per_study",
    "n_simulations",
    "percentile_calibration_offsets",
    "scenario_estimands",
    "scenario_id",
    "schema_version",
    "selection_required",
    "total_families",
    "zero_event_family_rate_upper_bound_if_no_events",
}
_REPORT_FIELDS = {
    "bound_construction",
    "config_sha256",
    "design_method",
    "endpoint_order",
    "estimates",
    "freeze_ready",
    "panel_sha256s",
    "rng_engine",
    "schema_version",
    "selection_audit_basis_sha256",
    "selection_audit_coverage_rule",
    "selection_audit_exact_bootstrap_replicates",
    "selection_audit_sha256",
    "selection_cell_alpha",
    "selection_family_size",
    "selection_familywise_confidence",
    "selection_multiplicity_method",
    "selected_families_per_corpus",
    "selection_satisfied",
    "target_power",
    "test_mode",
}
_SELECTION_AUDIT_RECORD_FIELDS = {
    "approximate_bounds",
    "approximate_joint_passed",
    "approximate_passes",
    "exact_bounds",
    "exact_joint_passed",
    "exact_passes",
    "families_per_corpus",
    "family_draws_sha256",
    "scenario_id",
    "schema_version",
    "study_index",
}
_SELECTION_AUDIT_CERTIFICATE_FIELDS = {
    "audited_study_indices",
    "disposition",
    "exact_joint_failures",
    "exact_joint_passes",
    "families_per_corpus",
    "required_failures",
    "required_successes",
    "scenario_id",
    "schema_version",
}
_SELECTION_AUDIT_FIELDS = {
    "certificates",
    "config_sha256",
    "coverage_rule",
    "exact_bootstrap_replicates",
    "monte_carlo_confidence",
    "n_simulations",
    "panel_sha256s",
    "records",
    "registered_bootstrap_seed",
    "schema_version",
    "selection_cell_alpha",
    "selection_family_size",
    "selection_multiplicity_method",
    "selected_families_per_corpus",
    "selection_basis_sha256",
    "selection_satisfied",
    "target_power",
    "test_mode",
}

Partition = Literal["development-fit", "development-calibration"]


class JointPowerDesignError(ValueError):
    """Raised when a design artifact or simulation contract is invalid."""


def _strict_mapping(payload: object, fields: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise JointPowerDesignError(f"{label} must be an object with string keys")
    observed = set(payload)
    missing = fields - observed
    unexpected = observed - fields
    if missing or unexpected:
        raise JointPowerDesignError(
            f"{label} keys do not match the closed schema; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return payload


def _identifier(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise JointPowerDesignError(f"{name} must be a canonical non-empty string")
    return value


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise JointPowerDesignError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise JointPowerDesignError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise JointPowerDesignError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise JointPowerDesignError(f"{name} must be at least {minimum}")
    return number


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise JointPowerDesignError(f"{name} must be an integer")
    integer = int(value)
    if integer < minimum:
        raise JointPowerDesignError(f"{name} must be at least {minimum}")
    return integer


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise JointPowerDesignError(f"{name} must be boolean")
    return value


def _canonical_payload_bytes(payload: object) -> bytes:
    try:
        body = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JointPowerDesignError("artifact values must be finite canonical JSON") from exc
    return body + b"\n"


def _decode_json_object(payload: str | bytes, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    if type(payload) not in {str, bytes}:
        raise TypeError("payload must be str or bytes")
    supplied = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(supplied) > _MAX_JSON_BYTES:
        raise JointPowerDesignError(f"{label} exceeds the {_MAX_JSON_BYTES}-byte limit")
    try:
        text = supplied.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise JointPowerDesignError(f"{label} must be valid UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise JointPowerDesignError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise JointPowerDesignError(f"{label} contains non-finite value {value!r}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise JointPowerDesignError(f"{label} must be valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, Mapping):
        raise JointPowerDesignError(f"{label} must contain one JSON object")
    return decoded, supplied


@dataclass(frozen=True)
class DependenceSource:
    artifact_uri: str
    artifact_sha256: str
    partition: Partition | str
    description: str
    construction: str = "paired-whole-family-empirical-resampling"

    def __post_init__(self) -> None:
        _identifier("dependence_source.artifact_uri", self.artifact_uri)
        _sha256("dependence_source.artifact_sha256", self.artifact_sha256)
        if self.partition not in {"development-fit", "development-calibration"}:
            raise JointPowerDesignError("dependence source must be a development partition")
        _identifier("dependence_source.description", self.description)
        if self.construction != "paired-whole-family-empirical-resampling":
            raise JointPowerDesignError("unsupported dependence-source construction")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_uri": self.artifact_uri,
            "construction": self.construction,
            "description": self.description,
            "partition": self.partition,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DependenceSource:
        value = _strict_mapping(payload, _DEPENDENCE_FIELDS, label="dependence_source")
        return cls(
            artifact_uri=value["artifact_uri"],
            artifact_sha256=value["artifact_sha256"],
            partition=value["partition"],
            description=value["description"],
            construction=value["construction"],
        )


@dataclass(frozen=True)
class EffectScenario:
    scenario_id: str
    panel_sha256: str
    description: str
    selection_required: bool

    def __post_init__(self) -> None:
        _identifier("effect_scenario.scenario_id", self.scenario_id)
        _sha256("effect_scenario.panel_sha256", self.panel_sha256)
        _identifier("effect_scenario.description", self.description)
        _boolean("effect_scenario.selection_required", self.selection_required)

    def to_dict(self) -> dict[str, object]:
        return {
            "description": self.description,
            "panel_sha256": self.panel_sha256,
            "scenario_id": self.scenario_id,
            "selection_required": self.selection_required,
        }

    @classmethod
    def from_dict(cls, payload: object) -> EffectScenario:
        value = _strict_mapping(payload, _SCENARIO_FIELDS, label="effect_scenario")
        return cls(
            scenario_id=value["scenario_id"],
            panel_sha256=value["panel_sha256"],
            description=value["description"],
            selection_required=value["selection_required"],
        )


@dataclass(frozen=True)
class GeometryGainThresholds:
    log_loss_reduction: float
    brier_score_reduction: float
    auprc_gain: float

    def __post_init__(self) -> None:
        for name in _THRESHOLD_FIELDS:
            value = _finite(f"geometry_gain_thresholds.{name}", getattr(self, name), minimum=0.0)
            if value > 1.0:
                raise JointPowerDesignError(f"geometry_gain_thresholds.{name} cannot exceed one")

    def to_dict(self) -> dict[str, float]:
        return {
            "auprc_gain": self.auprc_gain,
            "brier_score_reduction": self.brier_score_reduction,
            "log_loss_reduction": self.log_loss_reduction,
        }

    @classmethod
    def from_dict(cls, payload: object) -> GeometryGainThresholds:
        value = _strict_mapping(payload, _THRESHOLD_FIELDS, label="geometry_gain_thresholds")
        return cls(
            log_loss_reduction=value["log_loss_reduction"],
            brier_score_reduction=value["brier_score_reduction"],
            auprc_gain=value["auprc_gain"],
        )


@dataclass(frozen=True)
class JointPowerDesignConfig:
    dependence_source: DependenceSource
    effect_scenarios: tuple[EffectScenario, ...]
    candidate_families_per_corpus: tuple[int, ...]
    nested_rows_per_family: int
    geometry_gain_thresholds: GeometryGainThresholds
    n_simulations: int = 5_000
    bound_calibration_simulations: int = 5_000
    simulation_seed: int = 20260713
    target_power: float = 0.90
    alpha: float = 0.05
    monte_carlo_confidence: float = 0.95
    minimum_corpora_with_geometry_gain: int = 4
    minimum_latency_reduction: float = 0.10
    retrieval_target_noninferiority_margin: float = 0.01
    evidence_sufficiency_noninferiority_margin: float = 0.01
    maximum_p95_latency_ratio: float = 1.25
    maximum_denied_emissions: int = 0
    fixed_corpora: tuple[str, ...] = FIXED_CORPORA
    evidence_corpora: tuple[str, ...] = EVIDENCE_CORPORA
    endpoint_order: tuple[str, ...] = ENDPOINT_ORDER
    protocol_version: str = "0.3.0-draft"
    rng_engine: str = RNG_ENGINE
    bound_construction: str = BOUND_CONSTRUCTION
    selection_multiplicity_method: str = SELECTION_MULTIPLICITY_METHOD
    selection_rule: str = SELECTION_RULE
    test_mode: bool = False
    schema_version: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA:
            raise JointPowerDesignError("unsupported joint-power config schema")
        if self.protocol_version != "0.3.0-draft":
            raise JointPowerDesignError("protocol_version must equal 0.3.0-draft")
        test_mode = _boolean("test_mode", self.test_mode)
        object.__setattr__(self, "test_mode", test_mode)
        if not isinstance(self.dependence_source, DependenceSource):
            raise JointPowerDesignError("dependence_source must be typed")
        scenarios = tuple(self.effect_scenarios)
        if not scenarios or not all(isinstance(item, EffectScenario) for item in scenarios):
            raise JointPowerDesignError("effect_scenarios must contain typed scenarios")
        if len({item.scenario_id for item in scenarios}) != len(scenarios):
            raise JointPowerDesignError("effect scenario IDs must be unique")
        if len({item.panel_sha256 for item in scenarios}) != len(scenarios):
            raise JointPowerDesignError("effect scenario panel digests must be unique")
        required_scenario_count = sum(item.selection_required for item in scenarios)
        if required_scenario_count == 0:
            raise JointPowerDesignError("at least one effect scenario must govern selection")
        if not test_mode and required_scenario_count != REGISTERED_REQUIRED_SCENARIO_COUNT:
            raise JointPowerDesignError(
                "production design requires exactly two selection-required effect scenarios"
            )
        object.__setattr__(
            self,
            "effect_scenarios",
            tuple(sorted(scenarios, key=lambda item: item.scenario_id)),
        )

        candidates = tuple(self.candidate_families_per_corpus)
        if not candidates or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 2
            for value in candidates
        ):
            raise JointPowerDesignError("candidate family counts must be integers of at least two")
        normalized_candidates = tuple(int(value) for value in candidates)
        if normalized_candidates != tuple(sorted(set(normalized_candidates))):
            raise JointPowerDesignError("candidate family counts must be strictly increasing")
        if not test_mode and normalized_candidates != REGISTERED_CANDIDATE_FAMILY_COUNTS:
            raise JointPowerDesignError(
                "production candidate family counts must equal (25, 50, 75, 100, 150, 200)"
            )
        object.__setattr__(self, "candidate_families_per_corpus", normalized_candidates)

        _integer("nested_rows_per_family", self.nested_rows_per_family, minimum=1)
        simulations = _integer(
            "n_simulations",
            self.n_simulations,
            minimum=20 if test_mode else 5_000,
        )
        calibration = _integer(
            "bound_calibration_simulations",
            self.bound_calibration_simulations,
            minimum=20 if test_mode else 5_000,
        )
        object.__setattr__(self, "n_simulations", simulations)
        object.__setattr__(self, "bound_calibration_simulations", calibration)
        seed = _integer("simulation_seed", self.simulation_seed, minimum=0)
        if seed >= 2**128:
            raise JointPowerDesignError("simulation_seed must fit in 128 bits")
        object.__setattr__(self, "simulation_seed", seed)
        for name in ("target_power", "alpha", "monte_carlo_confidence"):
            value = _finite(name, getattr(self, name))
            if not 0.0 < value < 1.0:
                raise JointPowerDesignError(f"{name} must be in (0, 1)")
        if self.alpha != 0.05 or self.monte_carlo_confidence != 0.95:
            raise JointPowerDesignError(
                "design and Monte Carlo confidence must be the registered 95%"
            )
        if not test_mode and self.target_power != 0.90:
            raise JointPowerDesignError("production target_power must equal 0.90")
        if self.fixed_corpora != FIXED_CORPORA:
            raise JointPowerDesignError("fixed_corpora must equal the ordered five-corpus suite")
        if self.evidence_corpora != EVIDENCE_CORPORA:
            raise JointPowerDesignError("evidence_corpora must equal the ordered evidence subset")
        if self.endpoint_order != ENDPOINT_ORDER:
            raise JointPowerDesignError("endpoint_order must equal the registered gate order")
        if self.minimum_corpora_with_geometry_gain != 4:
            raise JointPowerDesignError("minimum_corpora_with_geometry_gain must equal four")
        for name in (
            "minimum_latency_reduction",
            "retrieval_target_noninferiority_margin",
            "evidence_sufficiency_noninferiority_margin",
        ):
            value = _finite(name, getattr(self, name), minimum=0.0)
            if value > 1.0:
                raise JointPowerDesignError(f"{name} cannot exceed one")
        if _finite("maximum_p95_latency_ratio", self.maximum_p95_latency_ratio, minimum=0.0) == 0:
            raise JointPowerDesignError("maximum_p95_latency_ratio must be positive")
        if self.maximum_denied_emissions != 0 or isinstance(self.maximum_denied_emissions, bool):
            raise JointPowerDesignError("maximum_denied_emissions must equal zero")
        if not isinstance(self.geometry_gain_thresholds, GeometryGainThresholds):
            raise JointPowerDesignError("geometry_gain_thresholds must be typed")
        if self.rng_engine != RNG_ENGINE:
            raise JointPowerDesignError("unsupported RNG engine")
        if self.bound_construction != BOUND_CONSTRUCTION:
            raise JointPowerDesignError("unsupported directional-bound construction")
        if self.selection_multiplicity_method != SELECTION_MULTIPLICITY_METHOD:
            raise JointPowerDesignError("unsupported selection multiplicity method")
        if self.selection_rule != SELECTION_RULE:
            raise JointPowerDesignError("unsupported family-count selection rule")

    @property
    def selection_family_size(self) -> int:
        required_scenarios = sum(item.selection_required for item in self.effect_scenarios)
        return len(self.candidate_families_per_corpus) * required_scenarios

    @property
    def selection_cell_alpha(self) -> float:
        return (1.0 - self.monte_carlo_confidence) / self.selection_family_size

    @property
    def selection_cell_confidence(self) -> float:
        return 1.0 - self.selection_cell_alpha

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "bound_calibration_simulations": self.bound_calibration_simulations,
            "bound_construction": self.bound_construction,
            "candidate_families_per_corpus": list(self.candidate_families_per_corpus),
            "dependence_source": self.dependence_source.to_dict(),
            "effect_scenarios": [item.to_dict() for item in self.effect_scenarios],
            "endpoint_order": list(self.endpoint_order),
            "evidence_corpora": list(self.evidence_corpora),
            "evidence_sufficiency_noninferiority_margin": (
                self.evidence_sufficiency_noninferiority_margin
            ),
            "fixed_corpora": list(self.fixed_corpora),
            "geometry_gain_thresholds": self.geometry_gain_thresholds.to_dict(),
            "maximum_denied_emissions": self.maximum_denied_emissions,
            "maximum_p95_latency_ratio": self.maximum_p95_latency_ratio,
            "minimum_corpora_with_geometry_gain": self.minimum_corpora_with_geometry_gain,
            "minimum_latency_reduction": self.minimum_latency_reduction,
            "monte_carlo_confidence": self.monte_carlo_confidence,
            "nested_rows_per_family": self.nested_rows_per_family,
            "n_simulations": self.n_simulations,
            "protocol_version": self.protocol_version,
            "retrieval_target_noninferiority_margin": (self.retrieval_target_noninferiority_margin),
            "rng_engine": self.rng_engine,
            "schema_version": self.schema_version,
            "selection_multiplicity_method": self.selection_multiplicity_method,
            "selection_rule": self.selection_rule,
            "simulation_seed": self.simulation_seed,
            "target_power": self.target_power,
            "test_mode": self.test_mode,
        }

    @classmethod
    def from_dict(cls, payload: object) -> JointPowerDesignConfig:
        value = _strict_mapping(payload, _CONFIG_FIELDS, label="joint_power_config")
        try:
            return cls(
                dependence_source=DependenceSource.from_dict(value["dependence_source"]),
                effect_scenarios=tuple(
                    EffectScenario.from_dict(item) for item in value["effect_scenarios"]
                ),
                candidate_families_per_corpus=tuple(value["candidate_families_per_corpus"]),
                nested_rows_per_family=value["nested_rows_per_family"],
                geometry_gain_thresholds=GeometryGainThresholds.from_dict(
                    value["geometry_gain_thresholds"]
                ),
                n_simulations=value["n_simulations"],
                bound_calibration_simulations=value["bound_calibration_simulations"],
                simulation_seed=value["simulation_seed"],
                target_power=value["target_power"],
                alpha=value["alpha"],
                monte_carlo_confidence=value["monte_carlo_confidence"],
                minimum_corpora_with_geometry_gain=value["minimum_corpora_with_geometry_gain"],
                minimum_latency_reduction=value["minimum_latency_reduction"],
                retrieval_target_noninferiority_margin=value[
                    "retrieval_target_noninferiority_margin"
                ],
                evidence_sufficiency_noninferiority_margin=value[
                    "evidence_sufficiency_noninferiority_margin"
                ],
                maximum_p95_latency_ratio=value["maximum_p95_latency_ratio"],
                maximum_denied_emissions=value["maximum_denied_emissions"],
                fixed_corpora=tuple(value["fixed_corpora"]),
                evidence_corpora=tuple(value["evidence_corpora"]),
                endpoint_order=tuple(value["endpoint_order"]),
                protocol_version=value["protocol_version"],
                rng_engine=value["rng_engine"],
                bound_construction=value["bound_construction"],
                selection_multiplicity_method=value["selection_multiplicity_method"],
                selection_rule=value["selection_rule"],
                test_mode=value["test_mode"],
                schema_version=value["schema_version"],
            )
        except TypeError as exc:
            raise JointPowerDesignError("joint_power_config contains invalid sequences") from exc

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_joint_power_config_bytes(self)).hexdigest()


@dataclass(frozen=True)
class DevelopmentFamilyRow:
    corpus_id: str
    family_id: str
    row_id: str
    label: int
    reference_probability: float
    full_probability: float
    proposed_latency_ms: float
    comparator_latency_ms: float
    proposed_execution_position: int
    comparator_execution_position: int
    proposed_retrieval_attained: bool
    comparator_retrieval_attained: bool
    proposed_evidence_sufficient: bool | None
    comparator_evidence_sufficient: bool | None
    denied_emissions: int
    schema_version: str = ROW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ROW_SCHEMA:
            raise JointPowerDesignError("unsupported development row schema")
        if self.corpus_id not in FIXED_CORPORA:
            raise JointPowerDesignError("development row corpus is outside the fixed suite")
        _identifier("family_id", self.family_id)
        _identifier("row_id", self.row_id)
        if isinstance(self.label, bool) or self.label not in {0, 1}:
            raise JointPowerDesignError("label must be the integer zero or one")
        for name in ("reference_probability", "full_probability"):
            probability = _finite(name, getattr(self, name))
            if not 0.0 < probability < 1.0:
                raise JointPowerDesignError(f"{name} must be strictly between zero and one")
        for name in ("proposed_latency_ms", "comparator_latency_ms"):
            if _finite(name, getattr(self, name), minimum=0.0) == 0.0:
                raise JointPowerDesignError(f"{name} must be positive")
        for name in ("proposed_execution_position", "comparator_execution_position"):
            position = _integer(name, getattr(self, name), minimum=0)
            if position >= 4:
                raise JointPowerDesignError(f"{name} must be an action position from zero to three")
            object.__setattr__(self, name, position)
        _boolean("proposed_retrieval_attained", self.proposed_retrieval_attained)
        _boolean("comparator_retrieval_attained", self.comparator_retrieval_attained)
        if self.corpus_id in EVIDENCE_CORPORA:
            _boolean("proposed_evidence_sufficient", self.proposed_evidence_sufficient)
            _boolean("comparator_evidence_sufficient", self.comparator_evidence_sufficient)
        elif (
            self.proposed_evidence_sufficient is not None
            or self.comparator_evidence_sufficient is not None
        ):
            raise JointPowerDesignError(
                "non-evidence corpora must encode evidence outcomes as null"
            )
        _integer("denied_emissions", self.denied_emissions, minimum=0)

    def to_dict(self) -> dict[str, object]:
        return {
            "comparator_evidence_sufficient": self.comparator_evidence_sufficient,
            "comparator_execution_position": self.comparator_execution_position,
            "comparator_latency_ms": self.comparator_latency_ms,
            "comparator_retrieval_attained": self.comparator_retrieval_attained,
            "corpus_id": self.corpus_id,
            "denied_emissions": self.denied_emissions,
            "family_id": self.family_id,
            "full_probability": self.full_probability,
            "label": self.label,
            "proposed_evidence_sufficient": self.proposed_evidence_sufficient,
            "proposed_execution_position": self.proposed_execution_position,
            "proposed_latency_ms": self.proposed_latency_ms,
            "proposed_retrieval_attained": self.proposed_retrieval_attained,
            "reference_probability": self.reference_probability,
            "row_id": self.row_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DevelopmentFamilyRow:
        value = _strict_mapping(payload, _ROW_FIELDS, label="development_row")
        return cls(
            corpus_id=value["corpus_id"],
            family_id=value["family_id"],
            row_id=value["row_id"],
            label=value["label"],
            reference_probability=value["reference_probability"],
            full_probability=value["full_probability"],
            proposed_latency_ms=value["proposed_latency_ms"],
            comparator_latency_ms=value["comparator_latency_ms"],
            proposed_execution_position=value["proposed_execution_position"],
            comparator_execution_position=value["comparator_execution_position"],
            proposed_retrieval_attained=value["proposed_retrieval_attained"],
            comparator_retrieval_attained=value["comparator_retrieval_attained"],
            proposed_evidence_sufficient=value["proposed_evidence_sufficient"],
            comparator_evidence_sufficient=value["comparator_evidence_sufficient"],
            denied_emissions=value["denied_emissions"],
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class DevelopmentScenarioPanel:
    scenario_id: str
    partition: Partition | str
    rows: tuple[DevelopmentFamilyRow, ...]
    schema_version: str = PANEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PANEL_SCHEMA:
            raise JointPowerDesignError("unsupported development panel schema")
        _identifier("scenario_id", self.scenario_id)
        if self.partition not in {"development-fit", "development-calibration"}:
            raise JointPowerDesignError(
                "joint power accepts development-fit or development-calibration only; "
                "sealed outcomes are inadmissible"
            )
        rows = tuple(self.rows)
        if not rows or not all(isinstance(row, DevelopmentFamilyRow) for row in rows):
            raise JointPowerDesignError("development panel rows must be non-empty and typed")
        if len({row.row_id for row in rows}) != len(rows):
            raise JointPowerDesignError("development row IDs must be globally unique")
        observed = {row.corpus_id for row in rows}
        if observed != set(FIXED_CORPORA):
            raise JointPowerDesignError(
                "development panel must contain exactly the fixed five-corpus suite"
            )
        ordered = tuple(sorted(rows, key=lambda row: (row.corpus_id, row.family_id, row.row_id)))
        object.__setattr__(self, "rows", ordered)

    def to_dict(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "rows": [row.to_dict() for row in self.rows],
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DevelopmentScenarioPanel:
        value = _strict_mapping(payload, _PANEL_FIELDS, label="development_panel")
        try:
            rows = tuple(DevelopmentFamilyRow.from_dict(item) for item in value["rows"])
        except TypeError as exc:
            raise JointPowerDesignError("development_panel.rows must be an array") from exc
        return cls(
            scenario_id=value["scenario_id"],
            partition=value["partition"],
            rows=rows,
            schema_version=value["schema_version"],
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_development_panel_bytes(self)).hexdigest()


def canonical_joint_power_config_bytes(config: JointPowerDesignConfig) -> bytes:
    if not isinstance(config, JointPowerDesignConfig):
        raise TypeError("config must be JointPowerDesignConfig")
    return _canonical_payload_bytes(config.to_dict())


def load_joint_power_config(payload: str | bytes) -> JointPowerDesignConfig:
    decoded, supplied = _decode_json_object(payload, label="joint power config")
    config = JointPowerDesignConfig.from_dict(decoded)
    if canonical_joint_power_config_bytes(config) != supplied:
        raise JointPowerDesignError("joint power config bytes are not canonical")
    return config


def canonical_development_panel_bytes(panel: DevelopmentScenarioPanel) -> bytes:
    if not isinstance(panel, DevelopmentScenarioPanel):
        raise TypeError("panel must be DevelopmentScenarioPanel")
    return _canonical_payload_bytes(panel.to_dict())


def load_development_panel(payload: str | bytes) -> DevelopmentScenarioPanel:
    decoded, supplied = _decode_json_object(payload, label="development panel")
    panel = DevelopmentScenarioPanel.from_dict(decoded)
    if canonical_development_panel_bytes(panel) != supplied:
        raise JointPowerDesignError("development panel bytes are not canonical")
    return panel


@dataclass(frozen=True)
class _CorpusFamilies:
    labels: np.ndarray
    reference_probability: np.ndarray
    full_probability: np.ndarray
    proposed_latency: np.ndarray
    comparator_latency: np.ndarray
    proposed_position: np.ndarray
    comparator_position: np.ndarray
    proposed_retrieval: np.ndarray
    comparator_retrieval: np.ndarray
    proposed_evidence: np.ndarray | None
    comparator_evidence: np.ndarray | None
    denied_emissions: np.ndarray

    @property
    def n_families(self) -> int:
        return int(self.labels.shape[0])


def _prepare_panel(
    panel: DevelopmentScenarioPanel,
    config: JointPowerDesignConfig,
) -> dict[str, _CorpusFamilies]:
    if panel.partition != config.dependence_source.partition:
        raise JointPowerDesignError("panel partition does not match the pinned dependence source")
    grouped: dict[str, dict[str, list[DevelopmentFamilyRow]]] = {
        corpus: {} for corpus in FIXED_CORPORA
    }
    for row in panel.rows:
        grouped[row.corpus_id].setdefault(row.family_id, []).append(row)
    prepared: dict[str, _CorpusFamilies] = {}
    for corpus in FIXED_CORPORA:
        families = grouped[corpus]
        if len(families) < 2:
            raise JointPowerDesignError(
                f"corpus {corpus!r} needs at least two development families"
            )
        ordered = []
        for family_id in sorted(families):
            rows = tuple(sorted(families[family_id], key=lambda row: row.row_id))
            if len(rows) != config.nested_rows_per_family:
                raise JointPowerDesignError(
                    f"family {corpus!r}/{family_id!r} has {len(rows)} rows; "
                    f"expected {config.nested_rows_per_family}"
                )
            ordered.append(rows)
        labels = np.asarray([[row.label for row in rows] for rows in ordered], dtype=np.int8)
        if set(np.unique(labels)) != {0, 1}:
            raise JointPowerDesignError(
                f"corpus {corpus!r} needs both outcome classes for the AUPRC design"
            )

        def numbers(name: str) -> np.ndarray:
            return np.asarray(
                [[float(getattr(row, name)) for row in rows] for rows in ordered],
                dtype=np.float64,
            )

        def booleans(name: str) -> np.ndarray:
            return np.asarray(
                [[bool(getattr(row, name)) for row in rows] for rows in ordered],
                dtype=np.float64,
            )

        prepared[corpus] = _CorpusFamilies(
            labels=labels,
            reference_probability=numbers("reference_probability"),
            full_probability=numbers("full_probability"),
            proposed_latency=numbers("proposed_latency_ms"),
            comparator_latency=numbers("comparator_latency_ms"),
            proposed_position=np.asarray(
                [[row.proposed_execution_position for row in rows] for rows in ordered],
                dtype=np.int8,
            ),
            comparator_position=np.asarray(
                [[row.comparator_execution_position for row in rows] for rows in ordered],
                dtype=np.int8,
            ),
            proposed_retrieval=booleans("proposed_retrieval_attained"),
            comparator_retrieval=booleans("comparator_retrieval_attained"),
            proposed_evidence=(
                booleans("proposed_evidence_sufficient") if corpus in EVIDENCE_CORPORA else None
            ),
            comparator_evidence=(
                booleans("comparator_evidence_sufficient") if corpus in EVIDENCE_CORPORA else None
            ),
            denied_emissions=np.asarray(
                [[row.denied_emissions for row in rows] for rows in ordered], dtype=np.int64
            ),
        )
    return prepared


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return -(labels * np.log(probabilities) + (1.0 - labels) * np.log1p(-probabilities))


def _weighted_average_precision_batch(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Compute exact tie-aware AP for equally weighted rows in each matrix row."""
    if labels.ndim != 2 or probabilities.shape != labels.shape:
        raise JointPowerDesignError("AUPRC arrays must be aligned two-dimensional matrices")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    sorted_probability = np.take_along_axis(probabilities, order, axis=1)
    sorted_label = np.take_along_axis(labels, order, axis=1).astype(np.float64)
    true_positive = np.cumsum(sorted_label, axis=1)
    observed = np.arange(1, labels.shape[1] + 1, dtype=np.float64)[np.newaxis, :]
    precision = true_positive / observed
    end_of_tie = np.empty_like(sorted_probability, dtype=bool)
    end_of_tie[:, -1] = True
    end_of_tie[:, :-1] = sorted_probability[:, :-1] != sorted_probability[:, 1:]
    positions = np.arange(labels.shape[1], dtype=np.int64)[np.newaxis, :]
    sentinel = np.full_like(positions, labels.shape[1])
    tie_ends = np.where(end_of_tie, positions, sentinel)
    next_end = np.minimum.accumulate(tie_ends[:, ::-1], axis=1)[:, ::-1]
    precision_at_threshold = np.take_along_axis(precision, next_end, axis=1)
    positives = sorted_label.sum(axis=1)
    negatives = labels.shape[1] - positives
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sum(sorted_label * precision_at_threshold, axis=1) / positives
    result[(positives == 0.0) | (negatives == 0.0)] = np.nan
    return result


def _child_rng(seed: int, *parts: object) -> np.random.Generator:
    material = _canonical_payload_bytes({"parts": list(parts), "seed": seed})
    child_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return np.random.Generator(np.random.PCG64(child_seed))


def _position_adjusted_log_ratio_batch(
    proposed_latency: np.ndarray,
    comparator_latency: np.ndarray,
    proposed_position: np.ndarray,
    comparator_position: np.ndarray,
) -> np.ndarray:
    """Return per-study zero-position-delta intercepts under the frozen linear model."""

    if (
        proposed_latency.shape != comparator_latency.shape
        or proposed_latency.shape != (proposed_position.shape)
        or proposed_latency.shape != comparator_position.shape
    ):
        raise JointPowerDesignError("position sensitivity arrays must have identical shapes")
    if proposed_latency.ndim != 3:
        raise JointPowerDesignError("position sensitivity arrays must be study/family/row cubes")
    y = np.log(proposed_latency / comparator_latency)
    x = proposed_position.astype(np.float64) - comparator_position.astype(np.float64)
    x_mean = x.mean(axis=(1, 2))
    y_mean = y.mean(axis=(1, 2))
    centered_x = x - x_mean[:, np.newaxis, np.newaxis]
    centered_y = y - y_mean[:, np.newaxis, np.newaxis]
    denominator = np.sum(centered_x * centered_x, axis=(1, 2))
    numerator = np.sum(centered_x * centered_y, axis=(1, 2))
    slope = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return y_mean - slope * x_mean


@dataclass(frozen=True)
class _MetricDraws:
    values: dict[str, np.ndarray]
    h2_consistency: np.ndarray
    denied_emissions: np.ndarray
    family_draws: dict[str, np.ndarray] | None = None


def _corpus_draw_metrics(
    families: _CorpusFamilies,
    draws: np.ndarray,
    *,
    evidence: bool,
) -> dict[str, np.ndarray]:
    labels = families.labels[draws].reshape(len(draws), -1)
    reference = families.reference_probability[draws].reshape(len(draws), -1)
    full = families.full_probability[draws].reshape(len(draws), -1)
    reference_log = _binary_log_loss(families.labels, families.reference_probability).mean(axis=1)
    full_log = _binary_log_loss(families.labels, families.full_probability).mean(axis=1)
    reference_brier = ((families.reference_probability - families.labels) ** 2).mean(axis=1)
    full_brier = ((families.full_probability - families.labels) ** 2).mean(axis=1)
    proposed_latency = families.proposed_latency.mean(axis=1)[draws]
    comparator_latency = families.comparator_latency.mean(axis=1)[draws]
    proposed_rows = families.proposed_latency[draws]
    comparator_rows = families.comparator_latency[draws]
    proposed_positions = families.proposed_position[draws]
    comparator_positions = families.comparator_position[draws]
    result = {
        "h2-log-loss-reduction": (reference_log - full_log)[draws].mean(axis=1),
        "h2-brier-score-reduction": (reference_brier - full_brier)[draws].mean(axis=1),
        "h2-auprc-gain": (
            _weighted_average_precision_batch(labels, full)
            - _weighted_average_precision_batch(labels, reference)
        ),
        "h3-family-relative-latency-reduction": (1.0 - proposed_latency / comparator_latency).mean(
            axis=1
        ),
        "h3-retrieval-target-noninferiority": (
            families.proposed_retrieval.mean(axis=1)[draws]
            - families.comparator_retrieval.mean(axis=1)[draws]
        ).mean(axis=1),
        "h3-family-mean-p95-latency-ratio": (
            np.quantile(proposed_latency, 0.95, axis=1)
            / np.quantile(comparator_latency, 0.95, axis=1)
        ),
        POSITION_SENSITIVITY_ENDPOINT: _position_adjusted_log_ratio_batch(
            proposed_rows,
            comparator_rows,
            proposed_positions,
            comparator_positions,
        ),
        "denied-emissions": families.denied_emissions.sum(axis=1)[draws].sum(axis=1),
    }
    if evidence:
        if families.proposed_evidence is None or families.comparator_evidence is None:
            raise JointPowerDesignError(
                "evidence corpus is missing paired development evidence outcomes"
            )
        result["h3-complete-evidence-noninferiority"] = (
            families.proposed_evidence.mean(axis=1)[draws]
            - families.comparator_evidence.mean(axis=1)[draws]
        ).mean(axis=1)
    return result


def _metric_draws(
    prepared: Mapping[str, _CorpusFamilies],
    *,
    families_per_corpus: int,
    n_simulations: int,
    config: JointPowerDesignConfig,
    scenario_id: str,
    phase: str,
    retain_family_draws: bool = False,
) -> _MetricDraws:
    values = {
        endpoint: np.zeros(n_simulations, dtype=np.float64) for endpoint in CONTINUOUS_ENDPOINTS
    }
    passing_corpora = np.zeros(n_simulations, dtype=np.int16)
    denied = np.zeros(n_simulations, dtype=np.int64)
    retained_draws: dict[str, np.ndarray] | None = {} if retain_family_draws else None
    thresholds = config.geometry_gain_thresholds
    for corpus in FIXED_CORPORA:
        families = prepared[corpus]
        rng = _child_rng(
            config.simulation_seed,
            scenario_id,
            families_per_corpus,
            phase,
            corpus,
        )
        draws = rng.integers(
            0,
            families.n_families,
            size=(n_simulations, families_per_corpus),
        )
        if retained_draws is not None:
            integer_dtype = np.min_scalar_type(max(families.n_families - 1, 0))
            retained_draws[corpus] = draws.astype(integer_dtype, copy=False)
        corpus_metrics = _corpus_draw_metrics(
            families,
            draws,
            evidence=corpus in EVIDENCE_CORPORA,
        )
        for endpoint in (
            "h2-log-loss-reduction",
            "h2-brier-score-reduction",
            "h2-auprc-gain",
            "h3-family-relative-latency-reduction",
            "h3-retrieval-target-noninferiority",
            "h3-family-mean-p95-latency-ratio",
            POSITION_SENSITIVITY_ENDPOINT,
        ):
            values[endpoint] += corpus_metrics[endpoint] / len(FIXED_CORPORA)
        if corpus in EVIDENCE_CORPORA:
            values["h3-complete-evidence-noninferiority"] += corpus_metrics[
                "h3-complete-evidence-noninferiority"
            ] / len(EVIDENCE_CORPORA)
        passing_corpora += (
            np.isfinite(corpus_metrics["h2-auprc-gain"])
            & (corpus_metrics["h2-log-loss-reduction"] > thresholds.log_loss_reduction)
            & (corpus_metrics["h2-brier-score-reduction"] > thresholds.brier_score_reduction)
            & (corpus_metrics["h2-auprc-gain"] > thresholds.auprc_gain)
        )
        denied += corpus_metrics["denied-emissions"].astype(np.int64)
    return _MetricDraws(
        values=values,
        h2_consistency=passing_corpora >= config.minimum_corpora_with_geometry_gain,
        denied_emissions=denied,
        family_draws=retained_draws,
    )


def _scenario_estimands(
    prepared: Mapping[str, _CorpusFamilies],
    config: JointPowerDesignConfig,
) -> dict[str, float]:
    values = {endpoint: 0.0 for endpoint in CONTINUOUS_ENDPOINTS}
    passing = 0
    denied_family_events = 0
    total_families = 0
    thresholds = config.geometry_gain_thresholds
    for corpus in FIXED_CORPORA:
        families = prepared[corpus]
        draws = np.arange(families.n_families, dtype=np.int64)[np.newaxis, :]
        metrics = _corpus_draw_metrics(families, draws, evidence=corpus in EVIDENCE_CORPORA)
        for endpoint in (
            "h2-log-loss-reduction",
            "h2-brier-score-reduction",
            "h2-auprc-gain",
            "h3-family-relative-latency-reduction",
            "h3-retrieval-target-noninferiority",
            "h3-family-mean-p95-latency-ratio",
            POSITION_SENSITIVITY_ENDPOINT,
        ):
            value = float(metrics[endpoint][0])
            if not math.isfinite(value):
                raise JointPowerDesignError("development scenario has undefined AUPRC")
            values[endpoint] += value / len(FIXED_CORPORA)
        if corpus in EVIDENCE_CORPORA:
            values["h3-complete-evidence-noninferiority"] += float(
                metrics["h3-complete-evidence-noninferiority"][0]
            ) / len(EVIDENCE_CORPORA)
        passing += int(
            metrics["h2-log-loss-reduction"][0] > thresholds.log_loss_reduction
            and metrics["h2-brier-score-reduction"][0] > thresholds.brier_score_reduction
            and metrics["h2-auprc-gain"][0] > thresholds.auprc_gain
        )
        denied_family_events += int(np.sum(families.denied_emissions.sum(axis=1) > 0))
        total_families += families.n_families
    values["h2-four-of-five-consistency"] = float(
        passing >= config.minimum_corpora_with_geometry_gain
    )
    values["h3-zero-entitlement-violations"] = float(denied_family_events == 0)
    values["development-family-denied-event-rate"] = denied_family_events / total_families
    return values


_LOWER_BOUND_ENDPOINTS = frozenset(CONTINUOUS_ENDPOINTS) - {
    "h3-family-mean-p95-latency-ratio",
    POSITION_SENSITIVITY_ENDPOINT,
}


def _percentile_calibration_offsets(
    calibration: _MetricDraws,
    truth: Mapping[str, float],
    *,
    alpha: float,
) -> dict[str, float]:
    """Estimate centered percentile-bootstrap offsets from an independent stream."""
    offsets: dict[str, float] = {}
    for endpoint in CONTINUOUS_ENDPOINTS:
        errors = calibration.values[endpoint] - truth[endpoint]
        if endpoint == "h2-auprc-gain":
            errors = np.where(np.isfinite(errors), errors, -2.0)
        if not np.all(np.isfinite(errors)):
            raise JointPowerDesignError(f"non-finite calibration errors for {endpoint}")
        quantile = alpha if endpoint in _LOWER_BOUND_ENDPOINTS else 1.0 - alpha
        offsets[endpoint] = float(np.quantile(errors, quantile))
    return offsets


def _continuous_bounds(
    values: Mapping[str, np.ndarray],
    offsets: Mapping[str, float],
) -> dict[str, np.ndarray]:
    return {
        endpoint: np.asarray(values[endpoint], dtype=np.float64) + offsets[endpoint]
        for endpoint in CONTINUOUS_ENDPOINTS
    }


def _continuous_gate_passes(
    bounds: Mapping[str, np.ndarray | float],
    config: JointPowerDesignConfig,
) -> dict[str, np.ndarray]:
    thresholds = {
        "h2-log-loss-reduction": config.geometry_gain_thresholds.log_loss_reduction,
        "h2-brier-score-reduction": config.geometry_gain_thresholds.brier_score_reduction,
        "h2-auprc-gain": config.geometry_gain_thresholds.auprc_gain,
        "h3-family-relative-latency-reduction": config.minimum_latency_reduction,
        "h3-retrieval-target-noninferiority": (-config.retrieval_target_noninferiority_margin),
        "h3-complete-evidence-noninferiority": (-config.evidence_sufficiency_noninferiority_margin),
        "h3-family-mean-p95-latency-ratio": config.maximum_p95_latency_ratio,
        POSITION_SENSITIVITY_ENDPOINT: math.log(1.0 - config.minimum_latency_reduction),
    }
    passes: dict[str, np.ndarray] = {}
    for endpoint in CONTINUOUS_ENDPOINTS:
        bound = np.asarray(bounds[endpoint], dtype=np.float64)
        if endpoint in _LOWER_BOUND_ENDPOINTS:
            passes[endpoint] = np.isfinite(bound) & (bound > thresholds[endpoint])
        else:
            passes[endpoint] = np.isfinite(bound) & (bound < thresholds[endpoint])
    return passes


def _study_family_draws(
    prepared: Mapping[str, _CorpusFamilies],
    *,
    families_per_corpus: int,
    config: JointPowerDesignConfig,
    scenario_id: str,
    phase: str,
    study_index: int,
) -> dict[str, np.ndarray]:
    if study_index < 0:
        raise JointPowerDesignError("study_index must be non-negative")
    draws: dict[str, np.ndarray] = {}
    for corpus in FIXED_CORPORA:
        rng = _child_rng(
            config.simulation_seed,
            scenario_id,
            families_per_corpus,
            phase,
            corpus,
        )
        draws[corpus] = rng.integers(
            0,
            prepared[corpus].n_families,
            size=(study_index + 1, families_per_corpus),
        )[study_index]
    return draws


def _registered_bootstrap_config(
    *,
    seed_offset: int,
    n_resamples: int,
    confidence: float,
) -> ClusterBootstrapConfig:
    return ClusterBootstrapConfig(
        n_resamples=n_resamples,
        confidence=confidence,
        seed=REGISTERED_BOOTSTRAP_SEED + seed_offset,
        interval_construction="directional-one-sided",
    )


def _finite_bootstrap_average_precision(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Match the registered AUPRC bootstrap's one-class replicate convention."""
    result = _weighted_average_precision_batch(labels, probabilities)
    positives = labels.sum(axis=1)
    result[positives == 0] = 0.0
    result[positives == labels.shape[1]] = 1.0
    return result


def _exact_registered_percentile_bounds(
    prepared: Mapping[str, _CorpusFamilies],
    family_draws: Mapping[str, np.ndarray],
    config: JointPowerDesignConfig,
    *,
    n_resamples: int = REGISTERED_BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    """Reproduce the registered inner bootstrap for one simulated study.

    This audit path is intentionally applied to individual studies. Running a
    10,000-replicate inner bootstrap inside every design simulation would make
    the registered 5,000-study candidate grid intractable.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples <= 0:
        raise JointPowerDesignError("n_resamples must be a positive integer")
    expected = set(FIXED_CORPORA)
    if set(family_draws) != expected:
        raise JointPowerDesignError("family_draws must cover the fixed corpus suite")

    corpus_ids: list[str] = []
    family_ids: list[str] = []
    pair_ids: list[str] = []
    reference_log: list[float] = []
    full_log: list[float] = []
    reference_brier: list[float] = []
    full_brier: list[float] = []
    proposed_latency: list[float] = []
    comparator_latency: list[float] = []
    proposed_retrieval: list[float] = []
    comparator_retrieval: list[float] = []
    evidence_corpus_ids: list[str] = []
    evidence_family_ids: list[str] = []
    evidence_pair_ids: list[str] = []
    proposed_evidence: list[float] = []
    comparator_evidence: list[float] = []
    sampled: dict[str, _CorpusFamilies] = {}

    for corpus in FIXED_CORPORA:
        source = prepared[corpus]
        draws = np.asarray(family_draws[corpus], dtype=np.int64)
        if draws.ndim != 1 or len(draws) < 2:
            raise JointPowerDesignError("each simulated corpus needs at least two families")
        if np.any(draws < 0) or np.any(draws >= source.n_families):
            raise JointPowerDesignError("simulated family draw is outside the development panel")
        corpus_sample = _CorpusFamilies(
            labels=source.labels[draws],
            reference_probability=source.reference_probability[draws],
            full_probability=source.full_probability[draws],
            proposed_latency=source.proposed_latency[draws],
            comparator_latency=source.comparator_latency[draws],
            proposed_position=source.proposed_position[draws],
            comparator_position=source.comparator_position[draws],
            proposed_retrieval=source.proposed_retrieval[draws],
            comparator_retrieval=source.comparator_retrieval[draws],
            proposed_evidence=(
                source.proposed_evidence[draws] if source.proposed_evidence is not None else None
            ),
            comparator_evidence=(
                source.comparator_evidence[draws]
                if source.comparator_evidence is not None
                else None
            ),
            denied_emissions=source.denied_emissions[draws],
        )
        sampled[corpus] = corpus_sample
        reference_log_by_family = _binary_log_loss(
            corpus_sample.labels,
            corpus_sample.reference_probability,
        ).mean(axis=1)
        full_log_by_family = _binary_log_loss(
            corpus_sample.labels,
            corpus_sample.full_probability,
        ).mean(axis=1)
        reference_brier_by_family = (
            (corpus_sample.reference_probability - corpus_sample.labels) ** 2
        ).mean(axis=1)
        full_brier_by_family = ((corpus_sample.full_probability - corpus_sample.labels) ** 2).mean(
            axis=1
        )
        for position in range(corpus_sample.n_families):
            family_id = f"simulated-family-{position}"
            pair_id = f"{corpus}\x1f{family_id}"
            corpus_ids.append(corpus)
            family_ids.append(family_id)
            pair_ids.append(pair_id)
            reference_log.append(float(reference_log_by_family[position]))
            full_log.append(float(full_log_by_family[position]))
            reference_brier.append(float(reference_brier_by_family[position]))
            full_brier.append(float(full_brier_by_family[position]))
            proposed_latency.append(float(corpus_sample.proposed_latency[position].mean()))
            comparator_latency.append(float(corpus_sample.comparator_latency[position].mean()))
            proposed_retrieval.append(float(corpus_sample.proposed_retrieval[position].mean()))
            comparator_retrieval.append(float(corpus_sample.comparator_retrieval[position].mean()))
            if corpus in EVIDENCE_CORPORA:
                if (
                    corpus_sample.proposed_evidence is None
                    or corpus_sample.comparator_evidence is None
                ):
                    raise JointPowerDesignError("evidence corpus is missing paired outcomes")
                evidence_corpus_ids.append(corpus)
                evidence_family_ids.append(family_id)
                evidence_pair_ids.append(pair_id)
                proposed_evidence.append(float(corpus_sample.proposed_evidence[position].mean()))
                comparator_evidence.append(
                    float(corpus_sample.comparator_evidence[position].mean())
                )

    confidence = 1.0 - config.alpha
    log_bootstrap = paired_stratified_family_bootstrap(
        reference_log,
        full_log,
        corpus_ids,
        family_ids,
        proposed_pair_ids=pair_ids,
        comparator_pair_ids=pair_ids,
        config=_registered_bootstrap_config(
            seed_offset=21,
            n_resamples=n_resamples,
            confidence=confidence,
        ),
    )
    brier_bootstrap = paired_stratified_family_bootstrap(
        reference_brier,
        full_brier,
        corpus_ids,
        family_ids,
        proposed_pair_ids=pair_ids,
        comparator_pair_ids=pair_ids,
        config=_registered_bootstrap_config(
            seed_offset=22,
            n_resamples=n_resamples,
            confidence=confidence,
        ),
    )
    latency_bootstrap = paired_stratified_metric_bootstrap(
        proposed_latency,
        comparator_latency,
        corpus_ids,
        family_ids,
        metric="relative-reduction",
        proposed_pair_ids=pair_ids,
        comparator_pair_ids=pair_ids,
        config=_registered_bootstrap_config(
            seed_offset=31,
            n_resamples=n_resamples,
            confidence=confidence,
        ),
    )
    tail_bootstrap = paired_stratified_metric_bootstrap(
        proposed_latency,
        comparator_latency,
        corpus_ids,
        family_ids,
        metric="p95-ratio",
        proposed_pair_ids=pair_ids,
        comparator_pair_ids=pair_ids,
        config=_registered_bootstrap_config(
            seed_offset=32,
            n_resamples=n_resamples,
            confidence=confidence,
        ),
    )
    retrieval_bootstrap = paired_stratified_family_bootstrap(
        proposed_retrieval,
        comparator_retrieval,
        corpus_ids,
        family_ids,
        proposed_pair_ids=pair_ids,
        comparator_pair_ids=pair_ids,
        config=_registered_bootstrap_config(
            seed_offset=33,
            n_resamples=n_resamples,
            confidence=confidence,
        ),
    )
    evidence_bootstrap = paired_stratified_family_bootstrap(
        proposed_evidence,
        comparator_evidence,
        evidence_corpus_ids,
        evidence_family_ids,
        proposed_pair_ids=evidence_pair_ids,
        comparator_pair_ids=evidence_pair_ids,
        config=_registered_bootstrap_config(
            seed_offset=34,
            n_resamples=n_resamples,
            confidence=confidence,
        ),
    )

    position_replicates = np.zeros(n_resamples, dtype=np.float64)
    position_rng = np.random.default_rng(REGISTERED_BOOTSTRAP_SEED + 35)
    for corpus in FIXED_CORPORA:
        corpus_sample = sampled[corpus]
        for start in range(0, n_resamples, EXACT_BOOTSTRAP_BATCH_SIZE):
            stop = min(start + EXACT_BOOTSTRAP_BATCH_SIZE, n_resamples)
            draws = position_rng.integers(
                0,
                corpus_sample.n_families,
                size=(stop - start, corpus_sample.n_families),
            )
            position_replicates[start:stop] += _position_adjusted_log_ratio_batch(
                corpus_sample.proposed_latency[draws],
                corpus_sample.comparator_latency[draws],
                corpus_sample.proposed_position[draws],
                corpus_sample.comparator_position[draws],
            ) / len(FIXED_CORPORA)

    exact_bounds = {
        "h2-log-loss-reduction": float(log_bootstrap.interval.lower),
        "h2-brier-score-reduction": float(brier_bootstrap.interval.lower),
        "h2-auprc-gain": float("nan"),
        "h3-family-relative-latency-reduction": float(latency_bootstrap.interval.lower),
        "h3-retrieval-target-noninferiority": float(retrieval_bootstrap.interval.lower),
        "h3-complete-evidence-noninferiority": float(evidence_bootstrap.interval.lower),
        "h3-family-mean-p95-latency-ratio": float(tail_bootstrap.interval.upper),
        POSITION_SENSITIVITY_ENDPOINT: float(np.quantile(position_replicates, 1.0 - config.alpha)),
    }
    auprc_replicates = np.zeros(n_resamples, dtype=np.float64)
    auprc_rng = np.random.default_rng(REGISTERED_BOOTSTRAP_SEED + 23)
    for corpus in FIXED_CORPORA:
        corpus_sample = sampled[corpus]
        labels = corpus_sample.labels.reshape(1, -1)
        if set(np.unique(labels)) != {0, 1}:
            return exact_bounds
        for start in range(0, n_resamples, EXACT_BOOTSTRAP_BATCH_SIZE):
            stop = min(start + EXACT_BOOTSTRAP_BATCH_SIZE, n_resamples)
            draws = auprc_rng.integers(
                0,
                corpus_sample.n_families,
                size=(stop - start, corpus_sample.n_families),
            )
            resampled_labels = corpus_sample.labels[draws].reshape(stop - start, -1)
            resampled_reference = corpus_sample.reference_probability[draws].reshape(
                stop - start,
                -1,
            )
            resampled_full = corpus_sample.full_probability[draws].reshape(stop - start, -1)
            auprc_replicates[start:stop] += (
                _finite_bootstrap_average_precision(resampled_labels, resampled_full)
                - _finite_bootstrap_average_precision(resampled_labels, resampled_reference)
            ) / len(FIXED_CORPORA)

    exact_bounds["h2-auprc-gain"] = float(np.quantile(auprc_replicates, config.alpha))
    return exact_bounds


@dataclass(frozen=True)
class PercentileApproximationAudit:
    """Exact-versus-plug-in gate comparison for one simulated study."""

    scenario_id: str
    families_per_corpus: int
    study_index: int
    exact_bootstrap_replicates: int
    approximate_bounds: tuple[tuple[str, float], ...]
    exact_bounds: tuple[tuple[str, float], ...]
    approximate_passes: tuple[tuple[str, bool], ...]
    exact_passes: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        _identifier("percentile audit scenario_id", self.scenario_id)
        _integer("families_per_corpus", self.families_per_corpus, minimum=2)
        _integer("study_index", self.study_index, minimum=0)
        _integer(
            "exact_bootstrap_replicates",
            self.exact_bootstrap_replicates,
            minimum=1,
        )
        expected = set(CONTINUOUS_ENDPOINTS)
        approximate_bounds = dict(self.approximate_bounds)
        exact_bounds = dict(self.exact_bounds)
        approximate_passes = dict(self.approximate_passes)
        exact_passes = dict(self.exact_passes)
        for name, values in (
            ("approximate_bounds", approximate_bounds),
            ("exact_bounds", exact_bounds),
            ("approximate_passes", approximate_passes),
            ("exact_passes", exact_passes),
        ):
            if set(values) != expected:
                raise JointPowerDesignError(f"{name} does not cover the continuous gate set")
        if not all(
            isinstance(value, (bool, np.bool_))
            for value in (*approximate_passes.values(), *exact_passes.values())
        ):
            raise JointPowerDesignError("percentile audit decisions must be boolean")
        object.__setattr__(
            self,
            "approximate_bounds",
            tuple(
                (endpoint, float(approximate_bounds[endpoint])) for endpoint in CONTINUOUS_ENDPOINTS
            ),
        )
        object.__setattr__(
            self,
            "exact_bounds",
            tuple((endpoint, float(exact_bounds[endpoint])) for endpoint in CONTINUOUS_ENDPOINTS),
        )
        object.__setattr__(
            self,
            "approximate_passes",
            tuple(
                (endpoint, bool(approximate_passes[endpoint])) for endpoint in CONTINUOUS_ENDPOINTS
            ),
        )
        object.__setattr__(
            self,
            "exact_passes",
            tuple((endpoint, bool(exact_passes[endpoint])) for endpoint in CONTINUOUS_ENDPOINTS),
        )

    @property
    def decisions_agree(self) -> bool:
        approximate = dict(self.approximate_passes)
        exact = dict(self.exact_passes)
        return all(
            approximate[endpoint] == exact[endpoint]
            for endpoint in CONTINUOUS_ENDPOINTS
            if endpoint != POSITION_SENSITIVITY_ENDPOINT
        )

    @property
    def sensitivity_decisions_agree(self) -> bool:
        return (
            dict(self.approximate_passes)[POSITION_SENSITIVITY_ENDPOINT]
            == dict(self.exact_passes)[POSITION_SENSITIVITY_ENDPOINT]
        )


def audit_percentile_approximation(
    config: JointPowerDesignConfig,
    panel: DevelopmentScenarioPanel,
    *,
    families_per_corpus: int,
    study_index: int = 0,
    exact_bootstrap_replicates: int = REGISTERED_BOOTSTRAP_REPLICATES,
) -> PercentileApproximationAudit:
    """Compare the design approximation with the registered gate for one study."""
    if not isinstance(config, JointPowerDesignConfig):
        raise TypeError("config must be JointPowerDesignConfig")
    if not isinstance(panel, DevelopmentScenarioPanel):
        raise TypeError("panel must be a DevelopmentScenarioPanel")
    scenarios = {scenario.scenario_id: scenario for scenario in config.effect_scenarios}
    scenario = scenarios.get(panel.scenario_id)
    if scenario is None or scenario.panel_sha256 != panel.sha256:
        raise JointPowerDesignError("audit panel is not pinned by the joint-power config")
    if families_per_corpus not in config.candidate_families_per_corpus:
        raise JointPowerDesignError("audit family count is not a registered candidate")
    if study_index < 0 or study_index >= config.n_simulations:
        raise JointPowerDesignError("audit study index is outside the evaluation stream")

    prepared = _prepare_panel(panel, config)
    truth = _scenario_estimands(prepared, config)
    calibration = _metric_draws(
        prepared,
        families_per_corpus=families_per_corpus,
        n_simulations=config.bound_calibration_simulations,
        config=config,
        scenario_id=scenario.scenario_id,
        phase="bound-calibration",
    )
    offsets = _percentile_calibration_offsets(calibration, truth, alpha=config.alpha)
    evaluation = _metric_draws(
        prepared,
        families_per_corpus=families_per_corpus,
        n_simulations=study_index + 1,
        config=config,
        scenario_id=scenario.scenario_id,
        phase="power-evaluation",
        retain_family_draws=True,
    )
    approximate = {
        endpoint: float(evaluation.values[endpoint][study_index] + offsets[endpoint])
        for endpoint in CONTINUOUS_ENDPOINTS
    }
    family_draws = _study_family_draws(
        prepared,
        families_per_corpus=families_per_corpus,
        config=config,
        scenario_id=scenario.scenario_id,
        phase="power-evaluation",
        study_index=study_index,
    )
    exact = _exact_registered_percentile_bounds(
        prepared,
        family_draws,
        config,
        n_resamples=exact_bootstrap_replicates,
    )
    approximate_passes = {
        endpoint: bool(np.asarray(value).item())
        for endpoint, value in _continuous_gate_passes(approximate, config).items()
    }
    exact_passes = {
        endpoint: bool(np.asarray(value).item())
        for endpoint, value in _continuous_gate_passes(exact, config).items()
    }
    return PercentileApproximationAudit(
        scenario_id=scenario.scenario_id,
        families_per_corpus=families_per_corpus,
        study_index=study_index,
        exact_bootstrap_replicates=exact_bootstrap_replicates,
        approximate_bounds=tuple(approximate.items()),
        exact_bounds=tuple(exact.items()),
        approximate_passes=tuple(approximate_passes.items()),
        exact_passes=tuple(exact_passes.items()),
    )


def _closed_audit_bounds(
    values: Mapping[str, object],
    *,
    label: str,
) -> tuple[tuple[str, float | None], ...]:
    if set(values) != set(CONTINUOUS_ENDPOINTS):
        raise JointPowerDesignError(f"{label} does not cover the continuous endpoint set")
    result: list[tuple[str, float | None]] = []
    for endpoint in CONTINUOUS_ENDPOINTS:
        value = values[endpoint]
        if value is None:
            result.append((endpoint, None))
            continue
        number = _finite(f"{label}.{endpoint}", value)
        result.append((endpoint, number))
    return tuple(result)


def _closed_audit_passes(
    values: Mapping[str, object],
    *,
    label: str,
) -> tuple[tuple[str, bool], ...]:
    if set(values) != set(ENDPOINT_ORDER):
        raise JointPowerDesignError(f"{label} does not cover the registered endpoint set")
    return tuple(
        (endpoint, _boolean(f"{label}.{endpoint}", values[endpoint])) for endpoint in ENDPOINT_ORDER
    )


@dataclass(frozen=True)
class ExactSelectionStudyAudit:
    """One exact inner-bootstrap decision used by the closed selection certificate."""

    scenario_id: str
    families_per_corpus: int
    study_index: int
    family_draws_sha256: str
    approximate_bounds: tuple[tuple[str, float | None], ...]
    exact_bounds: tuple[tuple[str, float | None], ...]
    approximate_passes: tuple[tuple[str, bool], ...]
    exact_passes: tuple[tuple[str, bool], ...]
    approximate_joint_passed: bool
    exact_joint_passed: bool
    schema_version: str = SELECTION_AUDIT_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SELECTION_AUDIT_RECORD_SCHEMA:
            raise JointPowerDesignError("unsupported exact selection-audit record schema")
        _identifier("selection audit scenario_id", self.scenario_id)
        _integer("selection audit families_per_corpus", self.families_per_corpus, minimum=2)
        _integer("selection audit study_index", self.study_index, minimum=0)
        _sha256("selection audit family_draws_sha256", self.family_draws_sha256)
        approximate_bounds = _closed_audit_bounds(
            dict(self.approximate_bounds),
            label="selection audit approximate_bounds",
        )
        exact_bounds = _closed_audit_bounds(
            dict(self.exact_bounds),
            label="selection audit exact_bounds",
        )
        approximate_passes = _closed_audit_passes(
            dict(self.approximate_passes),
            label="selection audit approximate_passes",
        )
        exact_passes = _closed_audit_passes(
            dict(self.exact_passes),
            label="selection audit exact_passes",
        )
        approximate_lookup = dict(approximate_passes)
        exact_lookup = dict(exact_passes)
        approximate_joint = _boolean(
            "selection audit approximate_joint_passed",
            self.approximate_joint_passed,
        )
        exact_joint = _boolean(
            "selection audit exact_joint_passed",
            self.exact_joint_passed,
        )
        if approximate_joint != all(approximate_lookup[name] for name in PRIMARY_ENDPOINT_ORDER):
            raise JointPowerDesignError("approximate joint decision differs from its primary gates")
        if exact_joint != all(exact_lookup[name] for name in PRIMARY_ENDPOINT_ORDER):
            raise JointPowerDesignError("exact joint decision differs from its primary gates")
        if any(approximate_lookup[name] != exact_lookup[name] for name in PRIMARY_ENDPOINT_ORDER):
            raise JointPowerDesignError(
                "selection audit contains an approximate/exact primary decision disagreement"
            )
        object.__setattr__(self, "approximate_bounds", approximate_bounds)
        object.__setattr__(self, "exact_bounds", exact_bounds)
        object.__setattr__(self, "approximate_passes", approximate_passes)
        object.__setattr__(self, "exact_passes", exact_passes)
        object.__setattr__(self, "approximate_joint_passed", approximate_joint)
        object.__setattr__(self, "exact_joint_passed", exact_joint)

    @property
    def sensitivity_decisions_agree(self) -> bool:
        return (
            dict(self.approximate_passes)[POSITION_SENSITIVITY_ENDPOINT]
            == dict(self.exact_passes)[POSITION_SENSITIVITY_ENDPOINT]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "approximate_bounds": dict(self.approximate_bounds),
            "approximate_joint_passed": self.approximate_joint_passed,
            "approximate_passes": dict(self.approximate_passes),
            "exact_bounds": dict(self.exact_bounds),
            "exact_joint_passed": self.exact_joint_passed,
            "exact_passes": dict(self.exact_passes),
            "families_per_corpus": self.families_per_corpus,
            "family_draws_sha256": self.family_draws_sha256,
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "study_index": self.study_index,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ExactSelectionStudyAudit:
        value = _strict_mapping(
            payload,
            _SELECTION_AUDIT_RECORD_FIELDS,
            label="selection_audit_record",
        )
        mappings = (
            value["approximate_bounds"],
            value["exact_bounds"],
            value["approximate_passes"],
            value["exact_passes"],
        )
        if not all(isinstance(item, Mapping) for item in mappings):
            raise JointPowerDesignError("selection audit record collections must be objects")
        return cls(
            scenario_id=value["scenario_id"],
            families_per_corpus=value["families_per_corpus"],
            study_index=value["study_index"],
            family_draws_sha256=value["family_draws_sha256"],
            approximate_bounds=tuple(value["approximate_bounds"].items()),
            exact_bounds=tuple(value["exact_bounds"].items()),
            approximate_passes=tuple(value["approximate_passes"].items()),
            exact_passes=tuple(value["exact_passes"].items()),
            approximate_joint_passed=value["approximate_joint_passed"],
            exact_joint_passed=value["exact_joint_passed"],
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class SelectionCandidateCertificate:
    scenario_id: str
    families_per_corpus: int
    disposition: str
    required_successes: int
    required_failures: int
    exact_joint_passes: int
    exact_joint_failures: int
    audited_study_indices: tuple[int, ...]
    schema_version: str = SELECTION_AUDIT_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SELECTION_AUDIT_CERTIFICATE_SCHEMA:
            raise JointPowerDesignError("unsupported selection certificate schema")
        _identifier("selection certificate scenario_id", self.scenario_id)
        _integer("selection certificate families_per_corpus", self.families_per_corpus, minimum=2)
        if self.disposition not in {"qualified", "blocked", "resolution-blocked"}:
            raise JointPowerDesignError("selection certificate disposition is not registered")
        required_successes = _integer(
            "selection certificate required_successes",
            self.required_successes,
            minimum=0,
        )
        required_failures = _integer(
            "selection certificate required_failures",
            self.required_failures,
            minimum=0,
        )
        passes = _integer(
            "selection certificate exact_joint_passes",
            self.exact_joint_passes,
            minimum=0,
        )
        failures = _integer(
            "selection certificate exact_joint_failures",
            self.exact_joint_failures,
            minimum=0,
        )
        indices = tuple(self.audited_study_indices)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indices
        ):
            raise JointPowerDesignError("selection certificate study indices must be non-negative")
        if len(indices) != len(set(indices)) or passes + failures != len(indices):
            raise JointPowerDesignError(
                "selection certificate contains duplicate or uncounted studies"
            )
        if self.disposition == "qualified" and passes < required_successes:
            raise JointPowerDesignError("qualified selection certificate lacks exact passes")
        if self.disposition == "blocked" and failures < required_failures:
            raise JointPowerDesignError("blocked selection certificate lacks exact failures")
        if self.disposition == "resolution-blocked" and indices:
            raise JointPowerDesignError("resolution-blocked certificate cannot contain studies")
        object.__setattr__(self, "audited_study_indices", indices)

    def to_dict(self) -> dict[str, object]:
        return {
            "audited_study_indices": list(self.audited_study_indices),
            "disposition": self.disposition,
            "exact_joint_failures": self.exact_joint_failures,
            "exact_joint_passes": self.exact_joint_passes,
            "families_per_corpus": self.families_per_corpus,
            "required_failures": self.required_failures,
            "required_successes": self.required_successes,
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: object) -> SelectionCandidateCertificate:
        value = _strict_mapping(
            payload,
            _SELECTION_AUDIT_CERTIFICATE_FIELDS,
            label="selection_candidate_certificate",
        )
        try:
            indices = tuple(value["audited_study_indices"])
        except TypeError as exc:
            raise JointPowerDesignError("audited_study_indices must be an array") from exc
        return cls(
            scenario_id=value["scenario_id"],
            families_per_corpus=value["families_per_corpus"],
            disposition=value["disposition"],
            required_successes=value["required_successes"],
            required_failures=value["required_failures"],
            exact_joint_passes=value["exact_joint_passes"],
            exact_joint_failures=value["exact_joint_failures"],
            audited_study_indices=indices,
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class JointPowerSelectionAudit:
    config_sha256: str
    panel_sha256s: tuple[tuple[str, str], ...]
    selection_basis_sha256: str
    n_simulations: int
    target_power: float
    monte_carlo_confidence: float
    selection_family_size: int
    selection_cell_alpha: float
    selection_multiplicity_method: str
    exact_bootstrap_replicates: int
    registered_bootstrap_seed: int
    coverage_rule: str
    certificates: tuple[SelectionCandidateCertificate, ...]
    records: tuple[ExactSelectionStudyAudit, ...]
    selected_families_per_corpus: int | None
    selection_satisfied: bool
    test_mode: bool
    schema_version: str = SELECTION_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SELECTION_AUDIT_SCHEMA:
            raise JointPowerDesignError("unsupported joint-power selection audit schema")
        test_mode = _boolean("selection audit test_mode", self.test_mode)
        object.__setattr__(self, "test_mode", test_mode)
        _sha256("selection audit config_sha256", self.config_sha256)
        _sha256("selection audit selection_basis_sha256", self.selection_basis_sha256)
        panel_pins = dict(self.panel_sha256s)
        if not panel_pins or len(panel_pins) != len(self.panel_sha256s):
            raise JointPowerDesignError("selection audit panel pins must be unique and non-empty")
        for scenario_id, digest in panel_pins.items():
            _identifier("selection audit panel scenario", scenario_id)
            _sha256("selection audit panel SHA-256", digest)
        n_simulations = _integer(
            "selection audit n_simulations",
            self.n_simulations,
            minimum=1,
        )
        target = _finite("selection audit target_power", self.target_power)
        if not 0.0 < target < 1.0 or self.monte_carlo_confidence != 0.95:
            raise JointPowerDesignError("selection audit probability contract differs")
        if not test_mode and target != 0.90:
            raise JointPowerDesignError("production selection audit target must equal 0.90")
        family_size = _integer(
            "selection audit family size",
            self.selection_family_size,
            minimum=1,
        )
        if not test_mode and family_size != REGISTERED_SELECTION_FAMILY_SIZE:
            raise JointPowerDesignError("production selection audit family size must equal 12")
        cell_alpha = _finite(
            "selection audit cell alpha",
            self.selection_cell_alpha,
        )
        expected_cell_alpha = (1.0 - self.monte_carlo_confidence) / family_size
        if not math.isclose(cell_alpha, expected_cell_alpha, abs_tol=1e-15):
            raise JointPowerDesignError(
                "selection audit cell alpha differs from the Bonferroni allocation"
            )
        if self.selection_multiplicity_method != SELECTION_MULTIPLICITY_METHOD:
            raise JointPowerDesignError("selection audit multiplicity method differs")
        exact_replicates = _integer(
            "selection audit exact_bootstrap_replicates",
            self.exact_bootstrap_replicates,
            minimum=1,
        )
        if not test_mode and exact_replicates != REGISTERED_BOOTSTRAP_REPLICATES:
            raise JointPowerDesignError("production selection audit requires 10000 replicates")
        if self.registered_bootstrap_seed != REGISTERED_BOOTSTRAP_SEED:
            raise JointPowerDesignError("selection audit bootstrap seed differs")
        if self.coverage_rule != SELECTION_AUDIT_COVERAGE_RULE:
            raise JointPowerDesignError("selection audit coverage rule differs")
        certificates = tuple(self.certificates)
        records = tuple(self.records)
        if not certificates or not all(
            isinstance(item, SelectionCandidateCertificate) for item in certificates
        ):
            raise JointPowerDesignError("selection audit certificates must be typed and non-empty")
        if not all(isinstance(item, ExactSelectionStudyAudit) for item in records):
            raise JointPowerDesignError("selection audit records must be typed")
        certificate_keys = tuple(
            (item.scenario_id, item.families_per_corpus) for item in certificates
        )
        if len(certificate_keys) != len(set(certificate_keys)):
            raise JointPowerDesignError("selection audit repeats a scenario-candidate certificate")
        record_keys = tuple(
            (item.scenario_id, item.families_per_corpus, item.study_index) for item in records
        )
        if len(record_keys) != len(set(record_keys)):
            raise JointPowerDesignError("selection audit repeats an exact study record")
        if any(item.study_index >= n_simulations for item in records):
            raise JointPowerDesignError("selection audit study index exceeds n_simulations")
        records_by_key = {
            (item.scenario_id, item.families_per_corpus, item.study_index): item for item in records
        }
        record_lookup = set(records_by_key)
        required_successes = _minimum_successes_for_probability_target(
            n_simulations,
            target_power=target,
            alpha=cell_alpha,
        )
        required_failures = (
            0 if required_successes is None else n_simulations - required_successes + 1
        )
        for certificate in certificates:
            referenced = {
                (certificate.scenario_id, certificate.families_per_corpus, index)
                for index in certificate.audited_study_indices
            }
            if not referenced.issubset(record_lookup):
                raise JointPowerDesignError(
                    "selection certificate references a missing study record"
                )
            expected_successes = 0 if required_successes is None else required_successes
            if (
                certificate.required_successes != expected_successes
                or certificate.required_failures != required_failures
            ):
                raise JointPowerDesignError(
                    "selection certificate probability thresholds differ from the closed rule"
                )
            observed_passes = sum(int(records_by_key[key].exact_joint_passed) for key in referenced)
            observed_failures = len(referenced) - observed_passes
            if (
                certificate.exact_joint_passes != observed_passes
                or certificate.exact_joint_failures != observed_failures
            ):
                raise JointPowerDesignError(
                    "selection certificate counts differ from its exact study records"
                )
            if required_successes is None:
                expected_disposition = "resolution-blocked"
            elif observed_passes >= required_successes:
                expected_disposition = "qualified"
            elif observed_failures >= required_failures:
                expected_disposition = "blocked"
            else:
                raise JointPowerDesignError(
                    "selection certificate stops before its exact result is determined"
                )
            if certificate.disposition != expected_disposition:
                raise JointPowerDesignError(
                    "selection certificate disposition differs from its exact records"
                )
        if record_lookup != {
            (certificate.scenario_id, certificate.families_per_corpus, index)
            for certificate in certificates
            for index in certificate.audited_study_indices
        }:
            raise JointPowerDesignError("selection audit contains an unreferenced study record")
        if self.selection_satisfied != (self.selected_families_per_corpus is not None):
            raise JointPowerDesignError("selection audit selected count and status disagree")
        object.__setattr__(self, "panel_sha256s", tuple(sorted(panel_pins.items())))
        object.__setattr__(self, "certificates", certificates)
        object.__setattr__(self, "records", records)

    def to_dict(self) -> dict[str, object]:
        return {
            "certificates": [item.to_dict() for item in self.certificates],
            "config_sha256": self.config_sha256,
            "coverage_rule": self.coverage_rule,
            "exact_bootstrap_replicates": self.exact_bootstrap_replicates,
            "monte_carlo_confidence": self.monte_carlo_confidence,
            "n_simulations": self.n_simulations,
            "panel_sha256s": dict(self.panel_sha256s),
            "records": [item.to_dict() for item in self.records],
            "registered_bootstrap_seed": self.registered_bootstrap_seed,
            "schema_version": self.schema_version,
            "selection_cell_alpha": self.selection_cell_alpha,
            "selection_family_size": self.selection_family_size,
            "selection_multiplicity_method": self.selection_multiplicity_method,
            "selected_families_per_corpus": self.selected_families_per_corpus,
            "selection_basis_sha256": self.selection_basis_sha256,
            "selection_satisfied": self.selection_satisfied,
            "target_power": self.target_power,
            "test_mode": self.test_mode,
        }

    @classmethod
    def from_dict(cls, payload: object) -> JointPowerSelectionAudit:
        value = _strict_mapping(payload, _SELECTION_AUDIT_FIELDS, label="selection_audit")
        panel_pins = value["panel_sha256s"]
        if not isinstance(panel_pins, Mapping):
            raise JointPowerDesignError("selection audit panel_sha256s must be an object")
        try:
            certificates = tuple(
                SelectionCandidateCertificate.from_dict(item) for item in value["certificates"]
            )
            records = tuple(ExactSelectionStudyAudit.from_dict(item) for item in value["records"])
        except TypeError as exc:
            raise JointPowerDesignError("selection audit records must be arrays") from exc
        return cls(
            config_sha256=value["config_sha256"],
            panel_sha256s=tuple(panel_pins.items()),
            selection_basis_sha256=value["selection_basis_sha256"],
            n_simulations=value["n_simulations"],
            target_power=value["target_power"],
            monte_carlo_confidence=value["monte_carlo_confidence"],
            selection_family_size=value["selection_family_size"],
            selection_cell_alpha=value["selection_cell_alpha"],
            selection_multiplicity_method=value["selection_multiplicity_method"],
            exact_bootstrap_replicates=value["exact_bootstrap_replicates"],
            registered_bootstrap_seed=value["registered_bootstrap_seed"],
            coverage_rule=value["coverage_rule"],
            certificates=certificates,
            records=records,
            selected_families_per_corpus=value["selected_families_per_corpus"],
            selection_satisfied=value["selection_satisfied"],
            test_mode=value["test_mode"],
            schema_version=value["schema_version"],
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_joint_power_selection_audit_bytes(self)).hexdigest()


def _one_sided_probability_lower_bound(
    successes: int,
    total: int,
    *,
    alpha: float,
) -> float:
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


@dataclass(frozen=True)
class OperatingProbability:
    endpoint: str
    n_simulations: int
    passing_simulations: int
    estimated_probability: float
    monte_carlo_standard_error: float
    lower_probability_bound: float
    confidence: float = 0.95
    schema_version: str = PROBABILITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PROBABILITY_SCHEMA:
            raise JointPowerDesignError("unsupported operating-probability schema")
        _identifier("operating_probability.endpoint", self.endpoint)
        total = _integer("n_simulations", self.n_simulations, minimum=1)
        passing = _integer("passing_simulations", self.passing_simulations, minimum=0)
        if passing > total:
            raise JointPowerDesignError("passing_simulations cannot exceed n_simulations")
        for name in (
            "estimated_probability",
            "monte_carlo_standard_error",
            "lower_probability_bound",
        ):
            value = _finite(name, getattr(self, name), minimum=0.0)
            if value > 1.0:
                raise JointPowerDesignError(f"{name} cannot exceed one")
        if not math.isclose(self.estimated_probability, passing / total, abs_tol=1e-15):
            raise JointPowerDesignError("estimated_probability disagrees with pass count")
        confidence = _finite("operating probability confidence", self.confidence)
        if not 0.0 < confidence < 1.0:
            raise JointPowerDesignError("operating probability confidence must be in (0, 1)")
        expected_standard_error = math.sqrt(
            self.estimated_probability * (1.0 - self.estimated_probability) / total
        )
        if not math.isclose(
            self.monte_carlo_standard_error,
            expected_standard_error,
            abs_tol=1e-15,
        ):
            raise JointPowerDesignError("monte_carlo_standard_error disagrees with the pass count")
        expected_lower = _one_sided_probability_lower_bound(
            passing,
            total,
            alpha=1.0 - self.confidence,
        )
        if not math.isclose(self.lower_probability_bound, expected_lower, abs_tol=1e-15):
            raise JointPowerDesignError(
                "lower_probability_bound disagrees with the exact one-sided limit"
            )
        if self.lower_probability_bound > self.estimated_probability:
            raise JointPowerDesignError(
                "lower_probability_bound cannot exceed estimated_probability"
            )

    @classmethod
    def from_passes(
        cls,
        endpoint: str,
        passes: np.ndarray,
        *,
        confidence: float,
    ) -> OperatingProbability:
        total = int(len(passes))
        passing = int(np.sum(passes))
        probability = passing / total
        return cls(
            endpoint=endpoint,
            n_simulations=total,
            passing_simulations=passing,
            estimated_probability=probability,
            monte_carlo_standard_error=float(math.sqrt(probability * (1.0 - probability) / total)),
            lower_probability_bound=_one_sided_probability_lower_bound(
                passing,
                total,
                alpha=1.0 - confidence,
            ),
            confidence=confidence,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence": self.confidence,
            "endpoint": self.endpoint,
            "estimated_probability": self.estimated_probability,
            "lower_probability_bound": self.lower_probability_bound,
            "monte_carlo_standard_error": self.monte_carlo_standard_error,
            "n_simulations": self.n_simulations,
            "passing_simulations": self.passing_simulations,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: object) -> OperatingProbability:
        value = _strict_mapping(payload, _PROBABILITY_FIELDS, label="operating_probability")
        return cls(
            endpoint=value["endpoint"],
            n_simulations=value["n_simulations"],
            passing_simulations=value["passing_simulations"],
            estimated_probability=value["estimated_probability"],
            monte_carlo_standard_error=value["monte_carlo_standard_error"],
            lower_probability_bound=value["lower_probability_bound"],
            confidence=value["confidence"],
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class ScenarioCandidateEstimate:
    scenario_id: str
    selection_required: bool
    families_per_corpus: int
    total_families: int
    n_simulations: int
    bound_calibration_simulations: int
    scenario_estimands: tuple[tuple[str, float], ...]
    percentile_calibration_offsets: tuple[tuple[str, float], ...]
    endpoint_probabilities: tuple[OperatingProbability, ...]
    joint_probability: OperatingProbability
    mean_denied_emissions_per_study: float
    zero_event_family_rate_upper_bound_if_no_events: float
    schema_version: str = ESTIMATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ESTIMATE_SCHEMA:
            raise JointPowerDesignError("unsupported candidate-estimate schema")
        _identifier("candidate_estimate.scenario_id", self.scenario_id)
        _boolean("candidate_estimate.selection_required", self.selection_required)
        families = _integer("families_per_corpus", self.families_per_corpus, minimum=2)
        if self.total_families != families * len(FIXED_CORPORA):
            raise JointPowerDesignError("total_families does not match the fixed suite")
        _integer("n_simulations", self.n_simulations, minimum=1)
        _integer("bound_calibration_simulations", self.bound_calibration_simulations, minimum=1)
        estimands = dict(self.scenario_estimands)
        expected_estimands = set(CONTINUOUS_ENDPOINTS) | {
            "h2-four-of-five-consistency",
            "h3-zero-entitlement-violations",
            "development-family-denied-event-rate",
        }
        if set(estimands) != expected_estimands or len(estimands) != len(self.scenario_estimands):
            raise JointPowerDesignError("scenario_estimands do not match the closed schema")
        if not all(math.isfinite(float(value)) for value in estimands.values()):
            raise JointPowerDesignError("scenario estimands must be finite")
        offsets = dict(self.percentile_calibration_offsets)
        if set(offsets) != set(CONTINUOUS_ENDPOINTS) or len(offsets) != len(
            self.percentile_calibration_offsets
        ):
            raise JointPowerDesignError(
                "percentile_calibration_offsets do not match continuous endpoints"
            )
        if not all(math.isfinite(float(value)) for value in offsets.values()):
            raise JointPowerDesignError("percentile calibration offsets must be finite")
        endpoints = tuple(self.endpoint_probabilities)
        if tuple(item.endpoint for item in endpoints) != ENDPOINT_ORDER:
            raise JointPowerDesignError("endpoint probabilities are outside the registered order")
        if any(item.n_simulations != self.n_simulations for item in endpoints):
            raise JointPowerDesignError("endpoint simulation counts must match the estimate")
        if self.joint_probability.endpoint != JOINT_ENDPOINT:
            raise JointPowerDesignError("joint probability has the wrong endpoint")
        if self.joint_probability.n_simulations != self.n_simulations:
            raise JointPowerDesignError("joint simulation count must match the estimate")
        if any(
            self.joint_probability.passing_simulations > item.passing_simulations
            for item in endpoints
            if item.endpoint in PRIMARY_ENDPOINT_ORDER
        ):
            raise JointPowerDesignError("joint pass count cannot exceed an endpoint pass count")
        _finite(
            "mean_denied_emissions_per_study",
            self.mean_denied_emissions_per_study,
            minimum=0.0,
        )
        upper = _finite(
            "zero_event_family_rate_upper_bound_if_no_events",
            self.zero_event_family_rate_upper_bound_if_no_events,
            minimum=0.0,
        )
        if upper > 1.0:
            raise JointPowerDesignError("zero-event upper bound cannot exceed one")
        expected_upper = 1.0 - 0.05 ** (1.0 / self.total_families)
        if not math.isclose(upper, expected_upper, abs_tol=1e-15):
            raise JointPowerDesignError(
                "zero-event upper bound disagrees with the candidate family count"
            )
        object.__setattr__(self, "scenario_estimands", tuple(sorted(estimands.items())))
        object.__setattr__(
            self,
            "percentile_calibration_offsets",
            tuple(sorted(offsets.items())),
        )

    def qualifies(self, target_power: float) -> bool:
        return self.joint_probability.lower_probability_bound >= target_power and all(
            item.lower_probability_bound >= target_power
            for item in self.endpoint_probabilities
            if item.endpoint in PRIMARY_ENDPOINT_ORDER
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "bound_calibration_simulations": self.bound_calibration_simulations,
            "endpoint_probabilities": [item.to_dict() for item in self.endpoint_probabilities],
            "families_per_corpus": self.families_per_corpus,
            "joint_probability": self.joint_probability.to_dict(),
            "mean_denied_emissions_per_study": self.mean_denied_emissions_per_study,
            "n_simulations": self.n_simulations,
            "percentile_calibration_offsets": dict(self.percentile_calibration_offsets),
            "scenario_estimands": dict(self.scenario_estimands),
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "selection_required": self.selection_required,
            "total_families": self.total_families,
            "zero_event_family_rate_upper_bound_if_no_events": (
                self.zero_event_family_rate_upper_bound_if_no_events
            ),
        }

    @classmethod
    def from_dict(cls, payload: object) -> ScenarioCandidateEstimate:
        value = _strict_mapping(payload, _ESTIMATE_FIELDS, label="candidate_estimate")
        estimands = value["scenario_estimands"]
        offsets = value["percentile_calibration_offsets"]
        if not isinstance(estimands, Mapping) or not isinstance(offsets, Mapping):
            raise JointPowerDesignError("candidate estimate metric collections must be objects")
        try:
            probabilities = tuple(
                OperatingProbability.from_dict(item) for item in value["endpoint_probabilities"]
            )
        except TypeError as exc:
            raise JointPowerDesignError("endpoint_probabilities must be an array") from exc
        return cls(
            scenario_id=value["scenario_id"],
            selection_required=value["selection_required"],
            families_per_corpus=value["families_per_corpus"],
            total_families=value["total_families"],
            n_simulations=value["n_simulations"],
            bound_calibration_simulations=value["bound_calibration_simulations"],
            scenario_estimands=tuple(estimands.items()),
            percentile_calibration_offsets=tuple(offsets.items()),
            endpoint_probabilities=probabilities,
            joint_probability=OperatingProbability.from_dict(value["joint_probability"]),
            mean_denied_emissions_per_study=value["mean_denied_emissions_per_study"],
            zero_event_family_rate_upper_bound_if_no_events=value[
                "zero_event_family_rate_upper_bound_if_no_events"
            ],
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class JointPowerDesignReport:
    config_sha256: str
    panel_sha256s: tuple[tuple[str, str], ...]
    estimates: tuple[ScenarioCandidateEstimate, ...]
    selected_families_per_corpus: int | None
    selection_satisfied: bool
    target_power: float
    selection_family_size: int
    selection_familywise_confidence: float
    selection_cell_alpha: float
    selection_multiplicity_method: str
    test_mode: bool
    freeze_ready: bool
    selection_audit_sha256: str | None = None
    selection_audit_basis_sha256: str | None = None
    selection_audit_exact_bootstrap_replicates: int | None = None
    selection_audit_coverage_rule: str | None = None
    endpoint_order: tuple[str, ...] = ENDPOINT_ORDER
    design_method: str = DESIGN_METHOD
    bound_construction: str = BOUND_CONSTRUCTION
    rng_engine: str = RNG_ENGINE
    schema_version: str = REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_SCHEMA:
            raise JointPowerDesignError("unsupported joint-power report schema")
        test_mode = _boolean("report.test_mode", self.test_mode)
        object.__setattr__(self, "test_mode", test_mode)
        _sha256("report.config_sha256", self.config_sha256)
        panel_pins = dict(self.panel_sha256s)
        if not panel_pins or len(panel_pins) != len(self.panel_sha256s):
            raise JointPowerDesignError("panel_sha256s must contain unique scenario IDs")
        for scenario_id, digest in panel_pins.items():
            _identifier("panel scenario ID", scenario_id)
            _sha256("panel SHA-256", digest)
        estimates = tuple(self.estimates)
        if not estimates or not all(
            isinstance(item, ScenarioCandidateEstimate) for item in estimates
        ):
            raise JointPowerDesignError("report estimates must be non-empty and typed")
        if len({(item.scenario_id, item.families_per_corpus) for item in estimates}) != len(
            estimates
        ):
            raise JointPowerDesignError("report contains duplicate scenario-candidate estimates")
        scenarios = {item.scenario_id for item in estimates}
        if scenarios != set(panel_pins):
            raise JointPowerDesignError("report scenarios do not match panel pins")
        candidate_sets = {
            tuple(
                sorted(
                    item.families_per_corpus for item in estimates if item.scenario_id == scenario
                )
            )
            for scenario in scenarios
        }
        if len(candidate_sets) != 1:
            raise JointPowerDesignError(
                "every scenario must cover the same candidate family counts"
            )
        required_by_scenario: dict[str, bool] = {}
        for item in estimates:
            existing = required_by_scenario.setdefault(item.scenario_id, item.selection_required)
            if existing != item.selection_required:
                raise JointPowerDesignError("selection_required must be stable within a scenario")
        required = {scenario for scenario, required in required_by_scenario.items() if required}
        if not required:
            raise JointPowerDesignError("at least one report scenario must govern selection")
        target = _finite("report.target_power", self.target_power)
        if not 0.0 < target < 1.0:
            raise JointPowerDesignError("report.target_power must be in (0, 1)")
        if not test_mode and target != 0.90:
            raise JointPowerDesignError("production report target must equal 0.90")
        candidates = next(iter(candidate_sets))
        family_size = _integer(
            "report.selection_family_size",
            self.selection_family_size,
            minimum=1,
        )
        expected_family_size = len(candidates) * len(required)
        if family_size != expected_family_size:
            raise JointPowerDesignError(
                "report selection family size differs from the required scenario-candidate grid"
            )
        if not test_mode and family_size != REGISTERED_SELECTION_FAMILY_SIZE:
            raise JointPowerDesignError("production report selection family size must equal 12")
        familywise_confidence = _finite(
            "report.selection_familywise_confidence",
            self.selection_familywise_confidence,
        )
        if familywise_confidence != 0.95:
            raise JointPowerDesignError("report familywise selection confidence must equal 0.95")
        cell_alpha = _finite(
            "report.selection_cell_alpha",
            self.selection_cell_alpha,
        )
        expected_cell_alpha = (1.0 - familywise_confidence) / family_size
        if not math.isclose(cell_alpha, expected_cell_alpha, abs_tol=1e-15):
            raise JointPowerDesignError(
                "report selection cell alpha differs from the Bonferroni allocation"
            )
        if self.selection_multiplicity_method != SELECTION_MULTIPLICITY_METHOD:
            raise JointPowerDesignError("report selection multiplicity method differs")
        for estimate in estimates:
            primary_confidence = (
                1.0 - cell_alpha if estimate.selection_required else familywise_confidence
            )
            for probability in estimate.endpoint_probabilities:
                expected_confidence = (
                    familywise_confidence
                    if probability.endpoint == POSITION_SENSITIVITY_ENDPOINT
                    else primary_confidence
                )
                if not math.isclose(
                    probability.confidence,
                    expected_confidence,
                    abs_tol=1e-15,
                ):
                    raise JointPowerDesignError(
                        "report operating-probability confidence differs from the selection grid"
                    )
            if not math.isclose(
                estimate.joint_probability.confidence,
                primary_confidence,
                abs_tol=1e-15,
            ):
                raise JointPowerDesignError(
                    "report joint-probability confidence differs from the selection grid"
                )
        expected_selection = next(
            (
                candidate
                for candidate in candidates
                if all(
                    next(
                        item
                        for item in estimates
                        if item.scenario_id == scenario and item.families_per_corpus == candidate
                    ).qualifies(target)
                    for scenario in required
                )
            ),
            None,
        )
        if self.selected_families_per_corpus != expected_selection:
            raise JointPowerDesignError("selected family count does not follow the closed rule")
        if self.selection_satisfied != (expected_selection is not None):
            raise JointPowerDesignError("selection_satisfied disagrees with the estimates")
        audit_fields = (
            self.selection_audit_sha256,
            self.selection_audit_basis_sha256,
            self.selection_audit_exact_bootstrap_replicates,
            self.selection_audit_coverage_rule,
        )
        if any(value is None for value in audit_fields) and any(
            value is not None for value in audit_fields
        ):
            raise JointPowerDesignError(
                "report selection-audit metadata must be all present or absent"
            )
        audit_present = all(value is not None for value in audit_fields)
        if audit_present:
            _sha256("report.selection_audit_sha256", self.selection_audit_sha256)
            _sha256(
                "report.selection_audit_basis_sha256",
                self.selection_audit_basis_sha256,
            )
            exact_replicates = _integer(
                "report.selection_audit_exact_bootstrap_replicates",
                self.selection_audit_exact_bootstrap_replicates,
                minimum=1,
            )
            if not test_mode and exact_replicates != REGISTERED_BOOTSTRAP_REPLICATES:
                raise JointPowerDesignError("freeze-ready report requires a 10000-replicate audit")
            if self.selection_audit_coverage_rule != SELECTION_AUDIT_COVERAGE_RULE:
                raise JointPowerDesignError("report selection-audit coverage rule differs")
        expected_freeze_ready = self.selection_satisfied and not test_mode and audit_present
        if self.freeze_ready != expected_freeze_ready:
            raise JointPowerDesignError(
                "freeze_ready requires a powered production report and an exact selection audit"
            )
        if not test_mode and any(
            item.n_simulations < 5_000 or item.bound_calibration_simulations < 5_000
            for item in estimates
        ):
            raise JointPowerDesignError(
                "freeze-ready reports require at least 5000 calibration and evaluation studies"
            )
        if self.endpoint_order != ENDPOINT_ORDER:
            raise JointPowerDesignError("report endpoint order is not registered")
        if self.design_method != DESIGN_METHOD:
            raise JointPowerDesignError("unsupported report design method")
        if self.bound_construction != BOUND_CONSTRUCTION or self.rng_engine != RNG_ENGINE:
            raise JointPowerDesignError("report computation identifiers are invalid")
        object.__setattr__(self, "panel_sha256s", tuple(sorted(panel_pins.items())))
        object.__setattr__(
            self,
            "estimates",
            tuple(sorted(estimates, key=lambda item: (item.scenario_id, item.families_per_corpus))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "bound_construction": self.bound_construction,
            "config_sha256": self.config_sha256,
            "design_method": self.design_method,
            "endpoint_order": list(self.endpoint_order),
            "estimates": [item.to_dict() for item in self.estimates],
            "freeze_ready": self.freeze_ready,
            "panel_sha256s": dict(self.panel_sha256s),
            "rng_engine": self.rng_engine,
            "schema_version": self.schema_version,
            "selection_audit_basis_sha256": self.selection_audit_basis_sha256,
            "selection_audit_coverage_rule": self.selection_audit_coverage_rule,
            "selection_audit_exact_bootstrap_replicates": (
                self.selection_audit_exact_bootstrap_replicates
            ),
            "selection_audit_sha256": self.selection_audit_sha256,
            "selection_cell_alpha": self.selection_cell_alpha,
            "selection_family_size": self.selection_family_size,
            "selection_familywise_confidence": self.selection_familywise_confidence,
            "selection_multiplicity_method": self.selection_multiplicity_method,
            "selected_families_per_corpus": self.selected_families_per_corpus,
            "selection_satisfied": self.selection_satisfied,
            "target_power": self.target_power,
            "test_mode": self.test_mode,
        }

    @classmethod
    def from_dict(cls, payload: object) -> JointPowerDesignReport:
        value = _strict_mapping(payload, _REPORT_FIELDS, label="joint_power_report")
        panel_pins = value["panel_sha256s"]
        if not isinstance(panel_pins, Mapping):
            raise JointPowerDesignError("panel_sha256s must be an object")
        try:
            estimates = tuple(
                ScenarioCandidateEstimate.from_dict(item) for item in value["estimates"]
            )
        except TypeError as exc:
            raise JointPowerDesignError("report estimates must be an array") from exc
        return cls(
            config_sha256=value["config_sha256"],
            panel_sha256s=tuple(panel_pins.items()),
            estimates=estimates,
            selected_families_per_corpus=value["selected_families_per_corpus"],
            selection_satisfied=value["selection_satisfied"],
            target_power=value["target_power"],
            selection_family_size=value["selection_family_size"],
            selection_familywise_confidence=value["selection_familywise_confidence"],
            selection_cell_alpha=value["selection_cell_alpha"],
            selection_multiplicity_method=value["selection_multiplicity_method"],
            test_mode=value["test_mode"],
            freeze_ready=value["freeze_ready"],
            selection_audit_sha256=value["selection_audit_sha256"],
            selection_audit_basis_sha256=value["selection_audit_basis_sha256"],
            selection_audit_exact_bootstrap_replicates=value[
                "selection_audit_exact_bootstrap_replicates"
            ],
            selection_audit_coverage_rule=value["selection_audit_coverage_rule"],
            endpoint_order=tuple(value["endpoint_order"]),
            design_method=value["design_method"],
            bound_construction=value["bound_construction"],
            rng_engine=value["rng_engine"],
            schema_version=value["schema_version"],
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_joint_power_report_bytes(self)).hexdigest()


@dataclass(frozen=True)
class _CandidateComputation:
    estimate: ScenarioCandidateEstimate
    prepared: Mapping[str, _CorpusFamilies]
    evaluation: _MetricDraws
    bounds: Mapping[str, np.ndarray]
    passes: Mapping[str, np.ndarray]
    joint_passes: np.ndarray


def _candidate_computation(
    panel: DevelopmentScenarioPanel,
    scenario: EffectScenario,
    *,
    families_per_corpus: int,
    config: JointPowerDesignConfig,
) -> _CandidateComputation:
    prepared = _prepare_panel(panel, config)
    truth = _scenario_estimands(prepared, config)
    calibration = _metric_draws(
        prepared,
        families_per_corpus=families_per_corpus,
        n_simulations=config.bound_calibration_simulations,
        config=config,
        scenario_id=scenario.scenario_id,
        phase="bound-calibration",
    )
    evaluation = _metric_draws(
        prepared,
        families_per_corpus=families_per_corpus,
        n_simulations=config.n_simulations,
        config=config,
        scenario_id=scenario.scenario_id,
        phase="power-evaluation",
        retain_family_draws=True,
    )
    percentile_offsets = _percentile_calibration_offsets(
        calibration,
        truth,
        alpha=config.alpha,
    )
    bounds = _continuous_bounds(evaluation.values, percentile_offsets)
    passes = _continuous_gate_passes(bounds, config)
    passes["h2-four-of-five-consistency"] = evaluation.h2_consistency
    passes["h3-zero-entitlement-violations"] = (
        evaluation.denied_emissions <= config.maximum_denied_emissions
    )
    ordered_passes = tuple(passes[endpoint] for endpoint in PRIMARY_ENDPOINT_ORDER)
    joint_passes = np.logical_and.reduce(ordered_passes)
    selection_confidence = (
        config.selection_cell_confidence
        if scenario.selection_required
        else config.monte_carlo_confidence
    )
    probabilities = tuple(
        OperatingProbability.from_passes(
            endpoint,
            passes[endpoint],
            confidence=(
                config.monte_carlo_confidence
                if endpoint == POSITION_SENSITIVITY_ENDPOINT
                else selection_confidence
            ),
        )
        for endpoint in ENDPOINT_ORDER
    )
    total_families = families_per_corpus * len(FIXED_CORPORA)
    estimate = ScenarioCandidateEstimate(
        scenario_id=scenario.scenario_id,
        selection_required=scenario.selection_required,
        families_per_corpus=families_per_corpus,
        total_families=total_families,
        n_simulations=config.n_simulations,
        bound_calibration_simulations=config.bound_calibration_simulations,
        scenario_estimands=tuple(truth.items()),
        percentile_calibration_offsets=tuple(percentile_offsets.items()),
        endpoint_probabilities=probabilities,
        joint_probability=OperatingProbability.from_passes(
            JOINT_ENDPOINT,
            joint_passes,
            confidence=selection_confidence,
        ),
        mean_denied_emissions_per_study=float(np.mean(evaluation.denied_emissions)),
        zero_event_family_rate_upper_bound_if_no_events=float(
            1.0 - config.alpha ** (1.0 / total_families)
        ),
    )
    return _CandidateComputation(
        estimate=estimate,
        prepared=prepared,
        evaluation=evaluation,
        bounds=bounds,
        passes=passes,
        joint_passes=joint_passes,
    )


def _candidate_estimate(
    panel: DevelopmentScenarioPanel,
    scenario: EffectScenario,
    *,
    families_per_corpus: int,
    config: JointPowerDesignConfig,
) -> ScenarioCandidateEstimate:
    return _candidate_computation(
        panel,
        scenario,
        families_per_corpus=families_per_corpus,
        config=config,
    ).estimate


def _admit_design_inputs(
    config: JointPowerDesignConfig,
    panels: Sequence[DevelopmentScenarioPanel],
) -> tuple[
    tuple[DevelopmentScenarioPanel, ...],
    dict[str, DevelopmentScenarioPanel],
    dict[str, EffectScenario],
    dict[str, str],
]:
    if not isinstance(config, JointPowerDesignConfig):
        raise TypeError("config must be JointPowerDesignConfig")
    observed_panels = tuple(panels)
    if not observed_panels or not all(
        isinstance(panel, DevelopmentScenarioPanel) for panel in observed_panels
    ):
        raise JointPowerDesignError("panels must contain typed development scenario panels")
    panel_lookup = {panel.scenario_id: panel for panel in observed_panels}
    if len(panel_lookup) != len(observed_panels):
        raise JointPowerDesignError("scenario panel IDs must be unique")
    scenario_lookup = {scenario.scenario_id: scenario for scenario in config.effect_scenarios}
    if set(panel_lookup) != set(scenario_lookup):
        raise JointPowerDesignError("supplied panels do not match the declared effect scenarios")
    panel_pins: dict[str, str] = {}
    for scenario_id, panel in panel_lookup.items():
        scenario = scenario_lookup[scenario_id]
        if panel.sha256 != scenario.panel_sha256:
            raise JointPowerDesignError(f"scenario {scenario_id!r} panel digest mismatch")
        if panel.partition != config.dependence_source.partition:
            raise JointPowerDesignError(f"scenario {scenario_id!r} has the wrong partition")
        panel_pins[scenario_id] = panel.sha256
    return observed_panels, panel_lookup, scenario_lookup, panel_pins


def _selection_basis_sha256(
    config: JointPowerDesignConfig,
    panel_pins: Mapping[str, str],
    estimates: Sequence[ScenarioCandidateEstimate],
) -> str:
    encoded = _canonical_payload_bytes(
        {
            "config_sha256": config.sha256,
            "estimates": [
                item.to_dict()
                for item in sorted(
                    estimates,
                    key=lambda row: (row.scenario_id, row.families_per_corpus),
                )
            ],
            "panel_sha256s": dict(sorted(panel_pins.items())),
            "schema_version": SELECTION_BASIS_SCHEMA,
            "target_power": config.target_power,
            "test_mode": config.test_mode,
        }
    )
    return hashlib.sha256(encoded).hexdigest()


def _minimum_successes_for_probability_target(
    total: int,
    *,
    target_power: float,
    alpha: float,
) -> int | None:
    return next(
        (
            successes
            for successes in range(1, total + 1)
            if _one_sided_probability_lower_bound(successes, total, alpha=alpha) >= target_power
        ),
        None,
    )


def _audit_bound_value(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _exact_selection_study_audit(
    computation: _CandidateComputation,
    *,
    config: JointPowerDesignConfig,
    scenario_id: str,
    families_per_corpus: int,
    study_index: int,
    exact_bootstrap_replicates: int,
) -> ExactSelectionStudyAudit:
    if computation.evaluation.family_draws is None:
        raise JointPowerDesignError("selection audit computation did not retain family draws")
    family_draws = {
        corpus: computation.evaluation.family_draws[corpus][study_index] for corpus in FIXED_CORPORA
    }
    family_draws_sha256 = hashlib.sha256(
        _canonical_payload_bytes(
            {corpus: family_draws[corpus].tolist() for corpus in FIXED_CORPORA}
        )
    ).hexdigest()
    exact_bounds_raw = _exact_registered_percentile_bounds(
        computation.prepared,
        family_draws,
        config,
        n_resamples=exact_bootstrap_replicates,
    )
    approximate_bounds_raw = {
        endpoint: float(computation.bounds[endpoint][study_index])
        for endpoint in CONTINUOUS_ENDPOINTS
    }
    approximate_continuous = {
        endpoint: bool(computation.passes[endpoint][study_index])
        for endpoint in CONTINUOUS_ENDPOINTS
    }
    exact_continuous = {
        endpoint: bool(np.asarray(value).item())
        for endpoint, value in _continuous_gate_passes(exact_bounds_raw, config).items()
    }
    h2_consistency = bool(computation.evaluation.h2_consistency[study_index])
    zero_entitlement = bool(
        computation.evaluation.denied_emissions[study_index] <= config.maximum_denied_emissions
    )
    approximate_passes = {
        **approximate_continuous,
        "h2-four-of-five-consistency": h2_consistency,
        "h3-zero-entitlement-violations": zero_entitlement,
    }
    exact_passes = {
        **exact_continuous,
        "h2-four-of-five-consistency": h2_consistency,
        "h3-zero-entitlement-violations": zero_entitlement,
    }
    approximate_joint = all(approximate_passes[name] for name in PRIMARY_ENDPOINT_ORDER)
    exact_joint = all(exact_passes[name] for name in PRIMARY_ENDPOINT_ORDER)
    try:
        return ExactSelectionStudyAudit(
            scenario_id=scenario_id,
            families_per_corpus=families_per_corpus,
            study_index=study_index,
            family_draws_sha256=family_draws_sha256,
            approximate_bounds=tuple(
                (endpoint, _audit_bound_value(approximate_bounds_raw[endpoint]))
                for endpoint in CONTINUOUS_ENDPOINTS
            ),
            exact_bounds=tuple(
                (endpoint, _audit_bound_value(exact_bounds_raw[endpoint]))
                for endpoint in CONTINUOUS_ENDPOINTS
            ),
            approximate_passes=tuple(
                (endpoint, approximate_passes[endpoint]) for endpoint in ENDPOINT_ORDER
            ),
            exact_passes=tuple((endpoint, exact_passes[endpoint]) for endpoint in ENDPOINT_ORDER),
            approximate_joint_passed=approximate_joint,
            exact_joint_passed=exact_joint,
        )
    except JointPowerDesignError as exc:
        raise JointPowerDesignError(
            "selection audit failed for "
            f"{scenario_id!r}, F={families_per_corpus}, study={study_index}: {exc}"
        ) from exc


def run_joint_power_selection_audit(
    config: JointPowerDesignConfig,
    panels: Sequence[DevelopmentScenarioPanel],
    *,
    exact_bootstrap_replicates: int = REGISTERED_BOOTSTRAP_REPLICATES,
) -> JointPowerSelectionAudit:
    """Build the exact closed certificate needed to admit one provisional selection."""

    _, panel_lookup, _, panel_pins = _admit_design_inputs(config, panels)
    if (
        isinstance(exact_bootstrap_replicates, bool)
        or not isinstance(exact_bootstrap_replicates, int)
        or exact_bootstrap_replicates <= 0
    ):
        raise JointPowerDesignError("exact_bootstrap_replicates must be a positive integer")
    if not config.test_mode and exact_bootstrap_replicates != REGISTERED_BOOTSTRAP_REPLICATES:
        raise JointPowerDesignError("production selection audit requires 10000 replicates")

    computations: dict[tuple[str, int], _CandidateComputation] = {}
    for scenario in config.effect_scenarios:
        for families in config.candidate_families_per_corpus:
            computations[(scenario.scenario_id, families)] = _candidate_computation(
                panel_lookup[scenario.scenario_id],
                scenario,
                families_per_corpus=families,
                config=config,
            )
    estimates = tuple(item.estimate for item in computations.values())
    basis_sha256 = _selection_basis_sha256(config, panel_pins, estimates)
    required_successes = _minimum_successes_for_probability_target(
        config.n_simulations,
        target_power=config.target_power,
        alpha=config.selection_cell_alpha,
    )
    required_failures = (
        0 if required_successes is None else config.n_simulations - required_successes + 1
    )
    required_scenarios = tuple(
        scenario for scenario in config.effect_scenarios if scenario.selection_required
    )
    certificates: list[SelectionCandidateCertificate] = []
    records: list[ExactSelectionStudyAudit] = []
    selected: int | None = None

    for families in config.candidate_families_per_corpus:
        candidate_qualified = required_successes is not None
        for scenario in required_scenarios:
            computation = computations[(scenario.scenario_id, families)]
            if required_successes is None:
                certificates.append(
                    SelectionCandidateCertificate(
                        scenario_id=scenario.scenario_id,
                        families_per_corpus=families,
                        disposition="resolution-blocked",
                        required_successes=0,
                        required_failures=0,
                        exact_joint_passes=0,
                        exact_joint_failures=0,
                        audited_study_indices=(),
                    )
                )
                candidate_qualified = False
                break
            provisional_qualifies = computation.estimate.qualifies(config.target_power)
            predicted_passes = tuple(
                int(value) for value in np.flatnonzero(computation.joint_passes)
            )
            predicted_failures = tuple(
                int(value) for value in np.flatnonzero(~computation.joint_passes)
            )
            study_order = (
                predicted_passes + predicted_failures
                if provisional_qualifies
                else predicted_failures + predicted_passes
            )
            exact_passes = 0
            exact_failures = 0
            audited_indices: list[int] = []
            for study_index in study_order:
                record = _exact_selection_study_audit(
                    computation,
                    config=config,
                    scenario_id=scenario.scenario_id,
                    families_per_corpus=families,
                    study_index=study_index,
                    exact_bootstrap_replicates=exact_bootstrap_replicates,
                )
                records.append(record)
                audited_indices.append(study_index)
                exact_passes += int(record.exact_joint_passed)
                exact_failures += int(not record.exact_joint_passed)
                if exact_passes >= required_successes or exact_failures >= required_failures:
                    break
            disposition = "qualified" if exact_passes >= required_successes else "blocked"
            certificates.append(
                SelectionCandidateCertificate(
                    scenario_id=scenario.scenario_id,
                    families_per_corpus=families,
                    disposition=disposition,
                    required_successes=required_successes,
                    required_failures=required_failures,
                    exact_joint_passes=exact_passes,
                    exact_joint_failures=exact_failures,
                    audited_study_indices=tuple(audited_indices),
                )
            )
            if disposition != "qualified":
                candidate_qualified = False
                break
        if candidate_qualified:
            selected = families
            break

    provisional_selected = next(
        (
            families
            for families in config.candidate_families_per_corpus
            if all(
                computations[(scenario.scenario_id, families)].estimate.qualifies(
                    config.target_power
                )
                for scenario in required_scenarios
            )
        ),
        None,
    )
    if selected != provisional_selected:
        raise JointPowerDesignError(
            "exact selection certificate disagrees with the provisional family-count selection"
        )
    return JointPowerSelectionAudit(
        config_sha256=config.sha256,
        panel_sha256s=tuple(panel_pins.items()),
        selection_basis_sha256=basis_sha256,
        n_simulations=config.n_simulations,
        target_power=config.target_power,
        monte_carlo_confidence=config.monte_carlo_confidence,
        selection_family_size=config.selection_family_size,
        selection_cell_alpha=config.selection_cell_alpha,
        selection_multiplicity_method=config.selection_multiplicity_method,
        exact_bootstrap_replicates=exact_bootstrap_replicates,
        registered_bootstrap_seed=REGISTERED_BOOTSTRAP_SEED,
        coverage_rule=SELECTION_AUDIT_COVERAGE_RULE,
        certificates=tuple(certificates),
        records=tuple(records),
        selected_families_per_corpus=selected,
        selection_satisfied=selected is not None,
        test_mode=config.test_mode,
    )


def _validate_selection_audit_binding(
    config: JointPowerDesignConfig,
    panel_pins: Mapping[str, str],
    estimates: Sequence[ScenarioCandidateEstimate],
    computations: Mapping[tuple[str, int], _CandidateComputation],
    audit: JointPowerSelectionAudit,
) -> None:
    """Validate closed coverage without trusting stored certificate bookkeeping."""

    basis_sha256 = _selection_basis_sha256(config, panel_pins, estimates)
    if (
        audit.config_sha256 != config.sha256
        or dict(audit.panel_sha256s) != dict(panel_pins)
        or audit.selection_basis_sha256 != basis_sha256
        or audit.n_simulations != config.n_simulations
        or audit.target_power != config.target_power
        or audit.monte_carlo_confidence != config.monte_carlo_confidence
        or audit.selection_family_size != config.selection_family_size
        or not math.isclose(
            audit.selection_cell_alpha,
            config.selection_cell_alpha,
            abs_tol=1e-15,
        )
        or audit.selection_multiplicity_method != config.selection_multiplicity_method
        or audit.test_mode != config.test_mode
    ):
        raise JointPowerDesignError("selection audit is stale, substituted, or mismatched")

    required_successes = _minimum_successes_for_probability_target(
        config.n_simulations,
        target_power=config.target_power,
        alpha=config.selection_cell_alpha,
    )
    required_failures = (
        0 if required_successes is None else config.n_simulations - required_successes + 1
    )
    record_lookup = {
        (record.scenario_id, record.families_per_corpus, record.study_index): record
        for record in audit.records
    }
    expected_certificates: list[SelectionCandidateCertificate] = []
    expected_record_keys: list[tuple[str, int, int]] = []
    required_scenarios = tuple(
        scenario for scenario in config.effect_scenarios if scenario.selection_required
    )
    selected: int | None = None

    for families in config.candidate_families_per_corpus:
        candidate_qualified = required_successes is not None
        for scenario in required_scenarios:
            if required_successes is None:
                expected_certificates.append(
                    SelectionCandidateCertificate(
                        scenario_id=scenario.scenario_id,
                        families_per_corpus=families,
                        disposition="resolution-blocked",
                        required_successes=0,
                        required_failures=0,
                        exact_joint_passes=0,
                        exact_joint_failures=0,
                        audited_study_indices=(),
                    )
                )
                candidate_qualified = False
                break

            computation = computations[(scenario.scenario_id, families)]
            provisional_qualifies = computation.estimate.qualifies(config.target_power)
            if provisional_qualifies:
                indices = tuple(
                    int(index)
                    for index in np.flatnonzero(computation.joint_passes)[:required_successes]
                )
                disposition = "qualified"
                exact_passes = required_successes
                exact_failures = 0
            else:
                indices = tuple(
                    int(index)
                    for index in np.flatnonzero(~computation.joint_passes)[:required_failures]
                )
                disposition = "blocked"
                exact_passes = 0
                exact_failures = required_failures
                candidate_qualified = False
            expected_certificates.append(
                SelectionCandidateCertificate(
                    scenario_id=scenario.scenario_id,
                    families_per_corpus=families,
                    disposition=disposition,
                    required_successes=required_successes,
                    required_failures=required_failures,
                    exact_joint_passes=exact_passes,
                    exact_joint_failures=exact_failures,
                    audited_study_indices=indices,
                )
            )
            for study_index in indices:
                key = (scenario.scenario_id, families, study_index)
                record = record_lookup.get(key)
                if record is None:
                    raise JointPowerDesignError(
                        "selection audit omits a study required by the closed certificate"
                    )
                expected_record_keys.append(key)
                expected_bounds = {
                    endpoint: _audit_bound_value(float(computation.bounds[endpoint][study_index]))
                    for endpoint in CONTINUOUS_ENDPOINTS
                }
                expected_passes = {
                    endpoint: bool(computation.passes[endpoint][study_index])
                    for endpoint in ENDPOINT_ORDER
                }
                if computation.evaluation.family_draws is None:
                    raise JointPowerDesignError(
                        "selection audit computation did not retain family draws"
                    )
                family_draws = {
                    corpus: computation.evaluation.family_draws[corpus][study_index]
                    for corpus in FIXED_CORPORA
                }
                expected_draw_sha256 = hashlib.sha256(
                    _canonical_payload_bytes(
                        {corpus: family_draws[corpus].tolist() for corpus in FIXED_CORPORA}
                    )
                ).hexdigest()
                if (
                    dict(record.approximate_bounds) != expected_bounds
                    or dict(record.approximate_passes) != expected_passes
                    or record.approximate_joint_passed
                    != bool(computation.joint_passes[study_index])
                    or record.family_draws_sha256 != expected_draw_sha256
                ):
                    raise JointPowerDesignError(
                        "selection audit study provenance differs from the simulated panel"
                    )
            if disposition == "blocked":
                break
        if candidate_qualified:
            selected = families
            break

    actual_record_keys = tuple(
        (record.scenario_id, record.families_per_corpus, record.study_index)
        for record in audit.records
    )
    if tuple(audit.certificates) != tuple(expected_certificates):
        raise JointPowerDesignError(
            "selection audit certificate coverage differs from the closed rule"
        )
    if actual_record_keys != tuple(expected_record_keys):
        raise JointPowerDesignError(
            "selection audit records are partial, reordered, or substituted"
        )
    if audit.selected_families_per_corpus != selected or audit.selection_satisfied != (
        selected is not None
    ):
        raise JointPowerDesignError("selection audit result differs from the closed rule")


def canonical_joint_power_selection_audit_bytes(
    audit: JointPowerSelectionAudit,
) -> bytes:
    if not isinstance(audit, JointPowerSelectionAudit):
        raise TypeError("audit must be JointPowerSelectionAudit")
    return _canonical_payload_bytes(audit.to_dict())


def load_joint_power_selection_audit(
    payload: str | bytes,
) -> JointPowerSelectionAudit:
    decoded, supplied = _decode_json_object(payload, label="joint power selection audit")
    audit = JointPowerSelectionAudit.from_dict(decoded)
    if canonical_joint_power_selection_audit_bytes(audit) != supplied:
        raise JointPowerDesignError("joint power selection audit bytes are not canonical")
    return audit


def verify_joint_power_selection_audit(
    config: JointPowerDesignConfig,
    panels: Sequence[DevelopmentScenarioPanel],
    audit: JointPowerSelectionAudit,
) -> JointPowerSelectionAudit:
    """Freshly reproduce a selection audit and require exact canonical equality."""

    if not isinstance(audit, JointPowerSelectionAudit):
        raise TypeError("audit must be JointPowerSelectionAudit")
    expected = run_joint_power_selection_audit(
        config,
        panels,
        exact_bootstrap_replicates=audit.exact_bootstrap_replicates,
    )
    expected_bytes = canonical_joint_power_selection_audit_bytes(expected)
    if expected_bytes != canonical_joint_power_selection_audit_bytes(audit):
        raise JointPowerDesignError("selection audit differs from a fresh exact reproduction")
    return expected


def run_joint_power_design(
    config: JointPowerDesignConfig,
    panels: Sequence[DevelopmentScenarioPanel],
    *,
    selection_audit: JointPowerSelectionAudit | None = None,
) -> JointPowerDesignReport:
    """Run the closed H2/H3 design simulator without accepting sealed outcomes."""
    _, panel_lookup, _, panel_pins = _admit_design_inputs(config, panels)

    computations = {
        (scenario.scenario_id, families): _candidate_computation(
            panel_lookup[scenario.scenario_id],
            scenario,
            families_per_corpus=families,
            config=config,
        )
        for scenario in config.effect_scenarios
        for families in config.candidate_families_per_corpus
    }
    estimates = tuple(item.estimate for item in computations.values())
    required = tuple(
        scenario for scenario in config.effect_scenarios if scenario.selection_required
    )
    selected = next(
        (
            families
            for families in config.candidate_families_per_corpus
            if all(
                next(
                    estimate
                    for estimate in estimates
                    if estimate.scenario_id == scenario.scenario_id
                    and estimate.families_per_corpus == families
                ).qualifies(config.target_power)
                for scenario in required
            )
        ),
        None,
    )
    selection_basis_sha256 = _selection_basis_sha256(config, panel_pins, estimates)
    if selection_audit is not None:
        if not isinstance(selection_audit, JointPowerSelectionAudit):
            raise TypeError("selection_audit must be JointPowerSelectionAudit")
        _validate_selection_audit_binding(
            config,
            panel_pins,
            estimates,
            computations,
            selection_audit,
        )
        if (
            selection_audit.selected_families_per_corpus != selected
            or selection_audit.selection_satisfied != (selected is not None)
            or selection_audit.selection_basis_sha256 != selection_basis_sha256
        ):
            raise JointPowerDesignError("selection audit is stale, substituted, or mismatched")
    return JointPowerDesignReport(
        config_sha256=config.sha256,
        panel_sha256s=tuple(panel_pins.items()),
        estimates=estimates,
        selected_families_per_corpus=selected,
        selection_satisfied=selected is not None,
        target_power=config.target_power,
        selection_family_size=config.selection_family_size,
        selection_familywise_confidence=config.monte_carlo_confidence,
        selection_cell_alpha=config.selection_cell_alpha,
        selection_multiplicity_method=config.selection_multiplicity_method,
        test_mode=config.test_mode,
        freeze_ready=(
            selected is not None and not config.test_mode and selection_audit is not None
        ),
        selection_audit_sha256=(None if selection_audit is None else selection_audit.sha256),
        selection_audit_basis_sha256=(
            None if selection_audit is None else selection_audit.selection_basis_sha256
        ),
        selection_audit_exact_bootstrap_replicates=(
            None if selection_audit is None else selection_audit.exact_bootstrap_replicates
        ),
        selection_audit_coverage_rule=(
            None if selection_audit is None else selection_audit.coverage_rule
        ),
    )


def canonical_joint_power_report_bytes(report: JointPowerDesignReport) -> bytes:
    if not isinstance(report, JointPowerDesignReport):
        raise TypeError("report must be JointPowerDesignReport")
    return _canonical_payload_bytes(report.to_dict())


def load_joint_power_report(payload: str | bytes) -> JointPowerDesignReport:
    decoded, supplied = _decode_json_object(payload, label="joint power report")
    report = JointPowerDesignReport.from_dict(decoded)
    if canonical_joint_power_report_bytes(report) != supplied:
        raise JointPowerDesignError("joint power report bytes are not canonical")
    return report
