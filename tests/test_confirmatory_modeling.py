from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from fractal_ann_diagnostics.confirmatory_modeling import (
    REGISTERED_FEATURE_SCHEMA,
    ArtifactIntegrityError,
    DataLeakageError,
    FeatureBatch,
    FeatureSchema,
    FeatureSchemaError,
    FrozenBinaryModel,
    FrozenModelSuite,
    GeometryGainThresholds,
    LabeledFeatureBatch,
    canonical_h1_model_artifact_bytes,
    canonical_h2_model_suite_artifact_bytes,
    evaluate_geometry_gain_gate,
    evaluate_h2_by_corpus,
    fit_frozen_model_suite,
    predictive_geometry_risk_contrast,
)

SCHEMA = FeatureSchema(
    system_numeric=("authorized_count", "dimension", "version_lag"),
    system_categorical=("corpus_stratum", "backend", "drift_family"),
    policy_numeric=("allow_rate", "policy_churn"),
    policy_categorical=("policy_class",),
    geometry_numeric=("lid_k50", "lid_cv", "relative_contrast"),
    geometry_categorical=(),
    lid_feature="lid_k50",
    instability_feature="lid_cv",
)
FIXED_CORPORA = ("corpus-a", "corpus-b")


def _batch(
    partition: str,
    prefix: str,
    *,
    families_per_corpus: int,
) -> LabeledFeatureBatch:
    rows: list[list[object]] = []
    labels: list[int] = []
    corpus_ids: list[str] = []
    family_ids: list[str] = []
    row_ids: list[str] = []
    for corpus_index, corpus_id in enumerate(FIXED_CORPORA):
        for family_index in range(families_per_corpus):
            nested_rows = 1 + family_index % 3
            for replicate in range(nested_rows):
                risk = (family_index + replicate + corpus_index) % 4
                family_id = f"{prefix}-{corpus_id}-family-{family_index}"
                rows.append(
                    [
                        80.0 + 4.0 * family_index,
                        384.0 if family_index % 2 == 0 else 768.0,
                        float((family_index + corpus_index) % 3),
                        corpus_id,
                        "hnsw-a" if family_index % 2 == 0 else "hnsw-b",
                        "corpus" if family_index % 2 == 0 else "policy",
                        0.2 + 0.15 * ((family_index + replicate) % 4),
                        0.01 * ((family_index + corpus_index) % 5),
                        "simple" if family_index % 2 == 0 else "complex",
                        5.0 + 3.0 * risk,
                        0.03 + 0.08 * risk,
                        0.75 - 0.10 * risk,
                    ]
                )
                labels.append(int(risk >= 2))
                corpus_ids.append(corpus_id)
                family_ids.append(family_id)
                row_ids.append(f"{family_id}-row-{replicate}")
    return LabeledFeatureBatch(
        partition=partition,
        feature_names=SCHEMA.input_features,
        features=np.asarray(rows, dtype=object),
        corpus_ids=tuple(corpus_ids),
        family_ids=tuple(family_ids),
        row_ids=tuple(row_ids),
        labels=tuple(labels),
    )


@pytest.fixture
def development_batches() -> tuple[LabeledFeatureBatch, LabeledFeatureBatch]:
    return (
        _batch("development-fit", "fit", families_per_corpus=10),
        _batch("development-calibration", "cal", families_per_corpus=6),
    )


@pytest.fixture
def frozen_suite(
    development_batches: tuple[LabeledFeatureBatch, LabeledFeatureBatch],
) -> FrozenModelSuite:
    training, calibration = development_batches
    return fit_frozen_model_suite(training, calibration, schema=SCHEMA, random_seed=71)


class _LabelsThatMustNotBeRead:
    def __iter__(self):
        raise AssertionError("sealed labels were read")


def test_sealed_labels_cannot_enter_fit_or_calibration(
    development_batches: tuple[LabeledFeatureBatch, LabeledFeatureBatch],
) -> None:
    training, calibration = development_batches
    sealed_training = replace(
        training,
        partition="sealed",
        labels=_LabelsThatMustNotBeRead(),
    )
    with pytest.raises(DataLeakageError, match="development-fit"):
        fit_frozen_model_suite(sealed_training, calibration, schema=SCHEMA)

    sealed_calibration = replace(
        calibration,
        partition="sealed",
        labels=_LabelsThatMustNotBeRead(),
    )
    with pytest.raises(DataLeakageError, match="development-calibration"):
        fit_frozen_model_suite(training, sealed_calibration, schema=SCHEMA)


def test_fit_calibration_and_sealed_families_must_be_disjoint(
    development_batches: tuple[LabeledFeatureBatch, LabeledFeatureBatch],
    frozen_suite: FrozenModelSuite,
) -> None:
    training, calibration = development_batches
    overlapping_families = list(calibration.family_ids)
    overlapping_corpora = list(calibration.corpus_ids)
    overlapping_families[0] = training.family_ids[0]
    overlapping_corpora[0] = training.corpus_ids[0]
    with pytest.raises(DataLeakageError, match="fit and calibration"):
        fit_frozen_model_suite(
            training,
            replace(
                calibration,
                corpus_ids=tuple(overlapping_corpora),
                family_ids=tuple(overlapping_families),
            ),
            schema=SCHEMA,
        )

    sealed = _batch("sealed", "sealed", families_per_corpus=5)
    sealed_families = list(sealed.family_ids)
    sealed_corpora = list(sealed.corpus_ids)
    sealed_families[0] = training.family_ids[0]
    sealed_corpora[0] = training.corpus_ids[0]
    with pytest.raises(DataLeakageError, match="overlap development"):
        evaluate_h2_by_corpus(
            frozen_suite,
            replace(
                sealed,
                corpus_ids=tuple(sealed_corpora),
                family_ids=tuple(sealed_families),
            ),
            fixed_corpora=FIXED_CORPORA,
        )


def test_feature_order_and_schema_drift_fail_closed(frozen_suite: FrozenModelSuite) -> None:
    sealed = _batch("sealed", "sealed", families_per_corpus=5)
    reordered = list(sealed.feature_names)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(FeatureSchemaError, match="feature order"):
        frozen_suite.predict_proba(replace(sealed, feature_names=tuple(reordered)))

    missing = sealed.feature_names[:-1]
    with pytest.raises(FeatureSchemaError, match="schema mismatch"):
        frozen_suite.predict_proba(
            replace(sealed, feature_names=missing, features=sealed.features[:, :-1])
        )

    missing_geometry = np.asarray(sealed.features, dtype=object).copy()
    missing_geometry[:, sealed.feature_names.index("relative_contrast")] = np.nan
    probabilities = frozen_suite.predict_proba(
        replace(sealed, features=missing_geometry),
        model_name="full",
    )
    assert np.all(np.isfinite(probabilities))


def test_fractional_failure_labels_are_rejected_before_fit(
    development_batches: tuple[LabeledFeatureBatch, LabeledFeatureBatch],
) -> None:
    training, calibration = development_batches
    labels = list(training.labels)
    labels[0] = 0.5
    with pytest.raises(
        ValueError,
        match="zero for composite success and one for action failure",
    ):
        fit_frozen_model_suite(
            replace(training, labels=tuple(labels)),
            calibration,
            schema=SCHEMA,
        )


def test_artifact_and_predictions_are_deterministic_and_digest_checked(
    development_batches: tuple[LabeledFeatureBatch, LabeledFeatureBatch],
) -> None:
    training, calibration = development_batches
    first = fit_frozen_model_suite(training, calibration, schema=SCHEMA, random_seed=83)
    second = fit_frozen_model_suite(training, calibration, schema=SCHEMA, random_seed=83)
    assert first.to_json() == second.to_json()

    sealed = _batch("sealed", "sealed", families_per_corpus=5)
    restored = FrozenModelSuite.from_json(first.to_json())
    np.testing.assert_array_equal(first.predict_proba(sealed), restored.predict_proba(sealed))
    assert restored.model("full").model_digest == first.model("full").model_digest
    assert first.model("full").engineered_numeric_features[-1] == "lid_k50__x__lid_cv"
    assert (
        "lid_k50__x__lid_cv"
        not in first.model("system-policy").transformed_feature_names
    )
    assert "standardization" in first.to_json()
    assert "feature_schema_digest" in first.to_json()

    tampered = json.loads(first.to_json())
    tampered["models"][-1]["coefficients"][0] += 0.25
    with pytest.raises(ArtifactIntegrityError, match="model digest mismatch"):
        FrozenModelSuite.from_json(json.dumps(tampered))


def test_registered_probe_telemetry_is_system_not_geometry() -> None:
    for feature in ("probe_latency_ms", "probe_work"):
        assert feature in REGISTERED_FEATURE_SCHEMA.system_numeric
        assert feature not in REGISTERED_FEATURE_SCHEMA.geometry_numeric


def test_model_pin_bytes_have_one_canonical_round_trip_without_trailing_lf(
    frozen_suite: FrozenModelSuite,
) -> None:
    h1_bytes = canonical_h1_model_artifact_bytes(frozen_suite)
    h2_bytes = canonical_h2_model_suite_artifact_bytes(frozen_suite)

    assert not h1_bytes.endswith(b"\n")
    assert not h2_bytes.endswith(b"\n")
    for encoded in (h1_bytes, h2_bytes):
        decoded = json.loads(encoded)
        assert encoded == json.dumps(
            decoded,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    restored_h1 = FrozenBinaryModel.from_dict(json.loads(h1_bytes))
    assert restored_h1 == frozen_suite.model("full")
    restored_h2 = FrozenModelSuite.from_json(h2_bytes.decode("utf-8"))
    assert canonical_h2_model_suite_artifact_bytes(restored_h2) == h2_bytes
    assert hashlib.sha256(
        canonical_h1_model_artifact_bytes(restored_h2)
    ).hexdigest() == hashlib.sha256(h1_bytes).hexdigest()
    assert hashlib.sha256(
        canonical_h2_model_suite_artifact_bytes(restored_h2)
    ).hexdigest() == hashlib.sha256(h2_bytes).hexdigest()


def test_h2_metrics_are_paired_and_equal_family_weighted_by_fixed_corpus(
    frozen_suite: FrozenModelSuite,
) -> None:
    sealed = _batch("sealed", "sealed", families_per_corpus=8)
    result = evaluate_h2_by_corpus(
        frozen_suite,
        sealed,
        fixed_corpora=FIXED_CORPORA,
    )
    assert result.fixed_corpora == FIXED_CORPORA
    assert result.inference == "descriptive-paired-metrics-only"
    assert len(result.row_identity_digest) == 64
    assert tuple(item.corpus_id for item in result.corpus_metrics) == FIXED_CORPORA

    for corpus in result.corpus_metrics:
        assert corpus.n_families == 8
        assert tuple(name for name, _ in corpus.model_metrics) == (
            "system-only",
            "system-policy",
            "geometry-only",
            "full",
        )
        assert np.isclose(
            corpus.for_model("full").log_loss,
            np.mean([family.full_log_loss for family in corpus.family_paired_losses]),
        )
        assert np.isclose(
            corpus.for_model("system-policy").brier_score,
            np.mean(
                [
                    family.system_policy_brier_score
                    for family in corpus.family_paired_losses
                ]
            ),
        )
        for _, metrics in corpus.model_metrics:
            assert np.all(
                np.isfinite((metrics.log_loss, metrics.brier_score, metrics.auprc))
            )

    equal_full = result.equal_corpus_for_model("full")
    assert np.isclose(
        equal_full.auprc,
        np.mean([corpus.for_model("full").auprc for corpus in result.corpus_metrics]),
    )
    with pytest.raises(ValueError, match="fixed suite"):
        evaluate_h2_by_corpus(
            frozen_suite,
            sealed,
            fixed_corpora=("corpus-a", "corpus-c"),
        )


def test_h1_geometry_contrast_is_explicitly_predictive_not_causal(
    frozen_suite: FrozenModelSuite,
) -> None:
    sealed = _batch("sealed", "sealed", families_per_corpus=8)
    feature_only = FeatureBatch(
        partition=sealed.partition,
        feature_names=sealed.feature_names,
        features=sealed.features,
        corpus_ids=sealed.corpus_ids,
        family_ids=sealed.family_ids,
        row_ids=sealed.row_ids,
    )
    contrast = predictive_geometry_risk_contrast(
        frozen_suite,
        feature_only,
        low_geometry={"lid_k50": 5.0, "lid_cv": 0.03},
        high_geometry={"lid_k50": 14.0, "lid_cv": 0.27},
        fixed_corpora=FIXED_CORPORA,
    )
    assert contrast.estimate > 0.0
    assert not contrast.causal
    assert contrast.inference == "point-estimate-only_no-hierarchical-interval"
    assert contrast.model_digest == frozen_suite.model("full").model_digest


def test_geometry_gain_gate_requires_every_metric_inside_each_corpus(
    frozen_suite: FrozenModelSuite,
) -> None:
    sealed = _batch("sealed", "sealed", families_per_corpus=8)
    evaluation = evaluate_h2_by_corpus(
        frozen_suite,
        sealed,
        fixed_corpora=FIXED_CORPORA,
    )
    thresholds = GeometryGainThresholds(
        log_loss_reduction=0.0,
        brier_score_reduction=0.0,
        auprc_gain=0.0,
    )
    decision = evaluate_geometry_gain_gate(
        evaluation,
        thresholds=thresholds,
        minimum_corpora=1,
    )
    expected = tuple(
        corpus.corpus_id
        for corpus in evaluation.corpus_metrics
        if corpus.system_policy_to_full.log_loss_reduction > 0.0
        and corpus.system_policy_to_full.brier_score_reduction > 0.0
        and corpus.system_policy_to_full.auprc_gain is not None
        and corpus.system_policy_to_full.auprc_gain > 0.0
    )
    assert decision.passing_corpora == expected
    assert decision.passed == bool(expected)

    impossible = evaluate_geometry_gain_gate(
        evaluation,
        thresholds=GeometryGainThresholds(1.0, 1.0, 1.0),
        minimum_corpora=1,
    )
    assert not impossible.passed


@pytest.mark.parametrize("single_class", [0, 1])
def test_one_class_corpus_retains_losses_but_marks_auprc_undefined(
    frozen_suite: FrozenModelSuite,
    single_class: int,
) -> None:
    sealed = _batch("sealed", "sealed", families_per_corpus=8)
    labels = np.asarray(sealed.labels, dtype=np.int8)
    corpus_ids = np.asarray(sealed.corpus_ids)
    labels[corpus_ids == FIXED_CORPORA[0]] = single_class

    evaluation = evaluate_h2_by_corpus(
        frozen_suite,
        replace(sealed, labels=labels),
        fixed_corpora=FIXED_CORPORA,
    )
    one_class = evaluation.corpus_metrics[0]
    for _, metrics in one_class.model_metrics:
        assert np.isfinite(metrics.log_loss)
        assert np.isfinite(metrics.brier_score)
        assert metrics.auprc is None
    assert one_class.system_policy_to_full.auprc_gain is None
    assert evaluation.equal_corpus_system_policy_to_full.auprc_gain is None

    gate = evaluate_geometry_gain_gate(
        evaluation,
        thresholds=GeometryGainThresholds(0.0, 0.0, 0.0),
        minimum_corpora=1,
    )
    assert not gate.corpus_decisions[0].passed
