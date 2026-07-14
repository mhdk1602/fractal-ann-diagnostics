"""Frozen predictive models for the registered H1 and H2 analyses.

The module fits only on explicitly marked development partitions.  It stores a
portable numeric artifact rather than a pickle, rejects feature-schema drift,
and evaluates every model on the same identified rows.  H1 output is a
predictive marginal contrast; no causal or hierarchical-inference claim is
made here.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler

Partition = Literal["development-fit", "development-calibration", "sealed"]
FeatureBlockName = Literal["system", "policy", "geometry"]
ModelName = Literal["system-only", "system-policy", "geometry-only", "full"]

ARTIFACT_VERSION = "h1-h2-logistic-v2"
MODEL_SPECS: tuple[tuple[ModelName, tuple[FeatureBlockName, ...]], ...] = (
    ("system-only", ("system",)),
    ("system-policy", ("system", "policy")),
    ("geometry-only", ("system", "geometry")),
    ("full", ("system", "policy", "geometry")),
)


class ModelingContractError(ValueError):
    """Base error for a violated confirmatory-modeling contract."""


class DataLeakageError(ModelingContractError):
    """Raised when a partition or query family crosses a locked boundary."""


class FeatureSchemaError(ModelingContractError):
    """Raised when an input no longer matches the frozen feature schema."""


class ArtifactIntegrityError(ModelingContractError):
    """Raised when a serialized model artifact fails digest validation."""


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(payload: object) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _strict_keys(payload: Mapping[str, object], expected: set[str], *, name: str) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ArtifactIntegrityError(
            f"{name} keys do not match the artifact contract; "
            f"missing={missing}, unexpected={unexpected}"
        )


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered source features divided into registered predictor blocks."""

    system_numeric: tuple[str, ...]
    system_categorical: tuple[str, ...]
    policy_numeric: tuple[str, ...]
    policy_categorical: tuple[str, ...]
    geometry_numeric: tuple[str, ...]
    geometry_categorical: tuple[str, ...]
    lid_feature: str
    instability_feature: str
    label_name: str = "low_effort_failure"

    def __post_init__(self) -> None:
        groups = (
            self.system_numeric,
            self.system_categorical,
            self.policy_numeric,
            self.policy_categorical,
            self.geometry_numeric,
            self.geometry_categorical,
        )
        features = tuple(feature for group in groups for feature in group)
        if not features:
            raise FeatureSchemaError("the feature schema cannot be empty")
        if any(not isinstance(feature, str) or not feature for feature in features):
            raise FeatureSchemaError("feature names must be non-empty strings")
        if len(features) != len(set(features)):
            raise FeatureSchemaError("feature names must be unique across blocks")
        if self.lid_feature == self.instability_feature:
            raise FeatureSchemaError("LID and instability must be distinct features")
        if self.lid_feature not in self.geometry_numeric:
            raise FeatureSchemaError("lid_feature must be a numeric geometry feature")
        if self.instability_feature not in self.geometry_numeric:
            raise FeatureSchemaError("instability_feature must be a numeric geometry feature")
        if not self.label_name or self.label_name in features:
            raise FeatureSchemaError("label_name must be non-empty and separate from features")

    @property
    def input_features(self) -> tuple[str, ...]:
        return (
            self.system_numeric
            + self.system_categorical
            + self.policy_numeric
            + self.policy_categorical
            + self.geometry_numeric
            + self.geometry_categorical
        )

    @property
    def interaction_name(self) -> str:
        return f"{self.lid_feature}__x__{self.instability_feature}"

    def numeric_for(self, blocks: Sequence[FeatureBlockName]) -> tuple[str, ...]:
        features: list[str] = []
        for block in blocks:
            features.extend(getattr(self, f"{block}_numeric"))
        return tuple(features)

    def categorical_for(self, blocks: Sequence[FeatureBlockName]) -> tuple[str, ...]:
        features: list[str] = []
        for block in blocks:
            features.extend(getattr(self, f"{block}_categorical"))
        return tuple(features)

    def as_dict(self) -> dict[str, object]:
        return {
            "geometry_categorical": list(self.geometry_categorical),
            "geometry_numeric": list(self.geometry_numeric),
            "instability_feature": self.instability_feature,
            "label_name": self.label_name,
            "lid_feature": self.lid_feature,
            "policy_categorical": list(self.policy_categorical),
            "policy_numeric": list(self.policy_numeric),
            "system_categorical": list(self.system_categorical),
            "system_numeric": list(self.system_numeric),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FeatureSchema:
        expected = {
            "geometry_categorical",
            "geometry_numeric",
            "instability_feature",
            "label_name",
            "lid_feature",
            "policy_categorical",
            "policy_numeric",
            "system_categorical",
            "system_numeric",
        }
        _strict_keys(payload, expected, name="feature_schema")
        try:
            return cls(
                system_numeric=tuple(str(item) for item in payload["system_numeric"]),
                system_categorical=tuple(
                    str(item) for item in payload["system_categorical"]
                ),
                policy_numeric=tuple(str(item) for item in payload["policy_numeric"]),
                policy_categorical=tuple(
                    str(item) for item in payload["policy_categorical"]
                ),
                geometry_numeric=tuple(str(item) for item in payload["geometry_numeric"]),
                geometry_categorical=tuple(
                    str(item) for item in payload["geometry_categorical"]
                ),
                lid_feature=str(payload["lid_feature"]),
                instability_feature=str(payload["instability_feature"]),
                label_name=str(payload["label_name"]),
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("invalid feature_schema values") from exc


REGISTERED_FEATURE_SCHEMA = FeatureSchema(
    system_numeric=(
        "corpus_size",
        "authorized_universe_size",
        "embedding_dimension",
        "version_lag",
        "drift_severity",
        "probe_latency_ms",
        "probe_work",
    ),
    system_categorical=("corpus_stratum", "backend", "drift_family"),
    policy_numeric=("allow_rate", "policy_complexity", "policy_churn"),
    policy_categorical=(),
    geometry_numeric=(
        "lid_k50",
        "lid_cv",
        "relative_contrast",
        "radius_expansion",
    ),
    geometry_categorical=(),
    lid_feature="lid_k50",
    instability_feature="lid_cv",
)


@dataclass(frozen=True)
class FeatureBatch:
    """Identified feature rows from one locked study partition."""

    partition: Partition | str
    feature_names: Sequence[str]
    features: Sequence[Sequence[object]] | np.ndarray
    corpus_ids: Sequence[str]
    family_ids: Sequence[str]
    row_ids: Sequence[str]


@dataclass(frozen=True)
class LabeledFeatureBatch(FeatureBatch):
    """Rows with the registered intent-to-treat action-failure composite."""

    labels: Sequence[int] | np.ndarray


@dataclass(frozen=True)
class _ValidatedFeatures:
    matrix: np.ndarray
    corpus_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    row_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ValidatedLabeledFeatures(_ValidatedFeatures):
    labels: np.ndarray


def _require_partition(batch: FeatureBatch, expected: Partition) -> None:
    if batch.partition != expected:
        raise DataLeakageError(
            f"expected partition {expected!r}, received {batch.partition!r}; "
            "sealed rows and labels cannot enter development fitting or calibration"
        )


def _validate_identifiers(values: Sequence[str], *, name: str, n_rows: int) -> tuple[str, ...]:
    identifiers = tuple(values)
    if len(identifiers) != n_rows:
        raise ModelingContractError(f"{name} must have one value per row")
    if any(not isinstance(item, str) or not item for item in identifiers):
        raise ModelingContractError(f"{name} must contain non-empty strings")
    return identifiers


def _validate_features(batch: FeatureBatch, schema: FeatureSchema) -> _ValidatedFeatures:
    names = tuple(batch.feature_names)
    expected = schema.input_features
    if names != expected:
        if len(names) == len(expected) and set(names) == set(expected):
            raise FeatureSchemaError("feature order does not match the frozen schema")
        missing = sorted(set(expected) - set(names))
        unexpected = sorted(set(names) - set(expected))
        raise FeatureSchemaError(
            f"feature schema mismatch; missing={missing}, unexpected={unexpected}"
        )

    matrix = np.asarray(batch.features, dtype=object)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ModelingContractError("features must be a non-empty two-dimensional matrix")
    if matrix.shape[1] != len(expected):
        raise FeatureSchemaError("feature matrix width does not match the frozen schema")

    corpus_ids = _validate_identifiers(batch.corpus_ids, name="corpus_ids", n_rows=len(matrix))
    family_ids = _validate_identifiers(batch.family_ids, name="family_ids", n_rows=len(matrix))
    row_ids = _validate_identifiers(batch.row_ids, name="row_ids", n_rows=len(matrix))
    if len(row_ids) != len(set(row_ids)):
        raise ModelingContractError("row_ids must be unique inside a batch")
    return _ValidatedFeatures(matrix, corpus_ids, family_ids, row_ids)


def _validate_labeled_features(
    batch: LabeledFeatureBatch,
    schema: FeatureSchema,
) -> _ValidatedLabeledFeatures:
    features = _validate_features(batch, schema)
    try:
        raw_labels = np.asarray(tuple(batch.labels))
    except (TypeError, ValueError) as exc:
        raise ModelingContractError("labels must contain binary integers") from exc
    if raw_labels.ndim != 1 or len(raw_labels) != len(features.matrix):
        raise ModelingContractError("labels must have one value per row")
    if not np.all(np.isin(raw_labels, (0, 1))):
        raise ModelingContractError(
            "labels must use zero for composite success and one for action failure"
        )
    labels = raw_labels.astype(np.int8)
    return _ValidatedLabeledFeatures(
        features.matrix,
        features.corpus_ids,
        features.family_ids,
        features.row_ids,
        labels,
    )


def _group_hash(corpus_id: str, family_id: str) -> str:
    return _digest({"corpus_id": corpus_id, "family_id": family_id})


def _group_hashes(features: _ValidatedFeatures) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _group_hash(corpus_id, family_id)
                for corpus_id, family_id in zip(
                    features.corpus_ids,
                    features.family_ids,
                    strict=True,
                )
            }
        )
    )


def _column_indices(schema: FeatureSchema, names: Sequence[str]) -> tuple[int, ...]:
    lookup = {name: index for index, name in enumerate(schema.input_features)}
    return tuple(lookup[name] for name in names)


def _numeric_matrix(
    features: _ValidatedFeatures,
    schema: FeatureSchema,
    names: Sequence[str],
) -> np.ndarray:
    try:
        matrix = np.asarray(
            features.matrix[:, _column_indices(schema, names)],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise FeatureSchemaError("numeric features must be numbers or NaN") from exc
    if np.any(np.isinf(matrix)):
        raise FeatureSchemaError("numeric features cannot contain infinity")
    return matrix


def _categorical_matrix(
    features: _ValidatedFeatures,
    schema: FeatureSchema,
    names: Sequence[str],
) -> np.ndarray:
    if not names:
        return np.empty((len(features.matrix), 0), dtype=object)
    matrix = features.matrix[:, _column_indices(schema, names)]
    for value in matrix.ravel():
        if not isinstance(value, str) or not value:
            raise FeatureSchemaError("categorical features must contain non-empty strings")
    return matrix


def _engineer_numeric(
    numeric: np.ndarray,
    numeric_names: Sequence[str],
    schema: FeatureSchema,
    blocks: Sequence[FeatureBlockName],
) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(numeric_names)
    if "geometry" not in blocks:
        return numeric, names
    lookup = {name: index for index, name in enumerate(names)}
    interaction = (
        numeric[:, lookup[schema.lid_feature]]
        * numeric[:, lookup[schema.instability_feature]]
    )
    return np.column_stack((numeric, interaction)), names + (schema.interaction_name,)


def _one_hot_names(
    categorical_names: Sequence[str],
    categorical_levels: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    return tuple(
        f"{name}=={level}"
        for name, levels in zip(categorical_names, categorical_levels, strict=True)
        for level in levels
    )


@dataclass(frozen=True)
class FrozenBinaryModel:
    """Portable fitted parameters for one registered model block combination."""

    name: ModelName
    blocks: tuple[FeatureBlockName, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    categorical_levels: tuple[tuple[str, ...], ...]
    imputation_values: tuple[float, ...]
    engineered_numeric_features: tuple[str, ...]
    standardization_mean: tuple[float, ...]
    standardization_scale: tuple[float, ...]
    transformed_feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    calibration_slope: float
    calibration_intercept: float

    def __post_init__(self) -> None:
        if len(self.numeric_features) != len(self.imputation_values):
            raise ArtifactIntegrityError("numeric imputation vector has the wrong length")
        if len(self.categorical_features) != len(self.categorical_levels):
            raise ArtifactIntegrityError("categorical level vectors have the wrong length")
        if len(self.engineered_numeric_features) != len(self.standardization_mean):
            raise ArtifactIntegrityError("standardization mean has the wrong length")
        if len(self.engineered_numeric_features) != len(self.standardization_scale):
            raise ArtifactIntegrityError("standardization scale has the wrong length")
        expected_transformed = self.engineered_numeric_features + _one_hot_names(
            self.categorical_features,
            self.categorical_levels,
        )
        if self.transformed_feature_names != expected_transformed:
            raise ArtifactIntegrityError("transformed feature names are inconsistent")
        if len(self.transformed_feature_names) != len(self.coefficients):
            raise ArtifactIntegrityError("coefficient vector has the wrong length")
        numeric_values = (
            self.imputation_values
            + self.standardization_mean
            + self.standardization_scale
            + self.coefficients
            + (self.intercept, self.calibration_slope, self.calibration_intercept)
        )
        if not np.all(np.isfinite(numeric_values)):
            raise ArtifactIntegrityError("model parameters must be finite")
        if any(scale <= 0.0 for scale in self.standardization_scale):
            raise ArtifactIntegrityError("standardization scales must be positive")
        for levels in self.categorical_levels:
            if not levels or tuple(sorted(set(levels))) != levels:
                raise ArtifactIntegrityError("categorical levels must be unique and sorted")

    def _payload(self) -> dict[str, object]:
        return {
            "blocks": list(self.blocks),
            "calibration": {
                "intercept": self.calibration_intercept,
                "method": "platt-logistic",
                "slope": self.calibration_slope,
            },
            "categorical_features": list(self.categorical_features),
            "categorical_levels": [list(levels) for levels in self.categorical_levels],
            "coefficients": list(self.coefficients),
            "engineered_numeric_features": list(self.engineered_numeric_features),
            "imputation_values": list(self.imputation_values),
            "intercept": self.intercept,
            "model_kind": "l2-logistic-regression",
            "name": self.name,
            "numeric_features": list(self.numeric_features),
            "standardization": {
                "mean": list(self.standardization_mean),
                "scale": list(self.standardization_scale),
            },
            "transformed_feature_names": list(self.transformed_feature_names),
        }

    @property
    def model_digest(self) -> str:
        return _digest(self._payload())

    def as_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["model_digest"] = self.model_digest
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FrozenBinaryModel:
        expected = {
            "blocks",
            "calibration",
            "categorical_features",
            "categorical_levels",
            "coefficients",
            "engineered_numeric_features",
            "imputation_values",
            "intercept",
            "model_digest",
            "model_kind",
            "name",
            "numeric_features",
            "standardization",
            "transformed_feature_names",
        }
        _strict_keys(payload, expected, name="model")
        if payload["model_kind"] != "l2-logistic-regression":
            raise ArtifactIntegrityError("unsupported model_kind")
        calibration = payload["calibration"]
        standardization = payload["standardization"]
        if not isinstance(calibration, Mapping) or not isinstance(standardization, Mapping):
            raise ArtifactIntegrityError("invalid model parameter blocks")
        _strict_keys(calibration, {"intercept", "method", "slope"}, name="calibration")
        _strict_keys(standardization, {"mean", "scale"}, name="standardization")
        if calibration["method"] != "platt-logistic":
            raise ArtifactIntegrityError("unsupported calibration method")
        try:
            model = cls(
                name=str(payload["name"]),  # type: ignore[arg-type]
                blocks=tuple(str(item) for item in payload["blocks"]),  # type: ignore[arg-type]
                numeric_features=tuple(
                    str(item) for item in payload["numeric_features"]  # type: ignore[union-attr]
                ),
                categorical_features=tuple(
                    str(item)
                    for item in payload["categorical_features"]  # type: ignore[union-attr]
                ),
                categorical_levels=tuple(
                    tuple(str(level) for level in levels)
                    for levels in payload["categorical_levels"]  # type: ignore[union-attr]
                ),
                imputation_values=tuple(
                    float(item) for item in payload["imputation_values"]  # type: ignore[union-attr]
                ),
                engineered_numeric_features=tuple(
                    str(item)
                    for item in payload["engineered_numeric_features"]  # type: ignore[union-attr]
                ),
                standardization_mean=tuple(
                    float(item) for item in standardization["mean"]  # type: ignore[union-attr]
                ),
                standardization_scale=tuple(
                    float(item) for item in standardization["scale"]  # type: ignore[union-attr]
                ),
                transformed_feature_names=tuple(
                    str(item)
                    for item in payload["transformed_feature_names"]  # type: ignore[union-attr]
                ),
                coefficients=tuple(
                    float(item) for item in payload["coefficients"]  # type: ignore[union-attr]
                ),
                intercept=float(payload["intercept"]),
                calibration_slope=float(calibration["slope"]),
                calibration_intercept=float(calibration["intercept"]),
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("invalid model parameter values") from exc
        if str(payload["model_digest"]) != model.model_digest:
            raise ArtifactIntegrityError(f"model digest mismatch for {model.name}")
        return model


@dataclass(frozen=True)
class FrozenModelSuite:
    """Four frozen H2 models sharing one schema and development boundary."""

    schema: FeatureSchema
    models: tuple[FrozenBinaryModel, ...]
    development_group_hashes: tuple[str, ...]
    random_seed: int
    sklearn_version: str

    def __post_init__(self) -> None:
        expected_names = tuple(name for name, _ in MODEL_SPECS)
        if tuple(model.name for model in self.models) != expected_names:
            raise ArtifactIntegrityError("model suite must contain the four registered models")
        for model, (_, blocks) in zip(self.models, MODEL_SPECS, strict=True):
            if model.blocks != blocks:
                raise ArtifactIntegrityError(f"incorrect feature blocks for {model.name}")
            expected_numeric = self.schema.numeric_for(blocks)
            expected_categorical = self.schema.categorical_for(blocks)
            expected_engineered = expected_numeric
            if "geometry" in blocks:
                expected_engineered += (self.schema.interaction_name,)
            if model.numeric_features != expected_numeric:
                raise ArtifactIntegrityError(f"numeric schema mismatch for {model.name}")
            if model.categorical_features != expected_categorical:
                raise ArtifactIntegrityError(f"categorical schema mismatch for {model.name}")
            if model.engineered_numeric_features != expected_engineered:
                raise ArtifactIntegrityError(f"engineered schema mismatch for {model.name}")
        if self.random_seed < 0:
            raise ArtifactIntegrityError("random_seed must be non-negative")
        if not self.sklearn_version:
            raise ArtifactIntegrityError("sklearn_version cannot be empty")
        if self.development_group_hashes != tuple(sorted(set(self.development_group_hashes))):
            raise ArtifactIntegrityError("development group hashes must be unique and sorted")
        if any(len(item) != 64 for item in self.development_group_hashes):
            raise ArtifactIntegrityError("invalid development group hash")

    @property
    def schema_digest(self) -> str:
        return _digest(self.schema.as_dict())

    @property
    def development_group_digest(self) -> str:
        return _digest(list(self.development_group_hashes))

    def model(self, name: ModelName) -> FrozenBinaryModel:
        for model in self.models:
            if model.name == name:
                return model
        raise KeyError(name)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "development_group_digest": self.development_group_digest,
            "development_group_hashes": list(self.development_group_hashes),
            "feature_schema": self.schema.as_dict(),
            "feature_schema_digest": self.schema_digest,
            "models": [model.as_dict() for model in self.models],
            "random_seed": self.random_seed,
            "sklearn_version": self.sklearn_version,
        }

    @property
    def suite_digest(self) -> str:
        return _digest(self._payload())

    def to_json(self) -> str:
        payload = self._payload()
        payload["suite_digest"] = self.suite_digest
        return _canonical_json(payload)

    @classmethod
    def from_json(cls, artifact: str) -> FrozenModelSuite:
        try:
            payload = json.loads(artifact)
        except json.JSONDecodeError as exc:
            raise ArtifactIntegrityError("model artifact is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ArtifactIntegrityError("model artifact must be a JSON object")
        expected = {
            "artifact_version",
            "development_group_digest",
            "development_group_hashes",
            "feature_schema",
            "feature_schema_digest",
            "models",
            "random_seed",
            "sklearn_version",
            "suite_digest",
        }
        _strict_keys(payload, expected, name="model artifact")
        if payload["artifact_version"] != ARTIFACT_VERSION:
            raise ArtifactIntegrityError("unsupported model artifact version")
        schema_payload = payload["feature_schema"]
        if not isinstance(schema_payload, Mapping):
            raise ArtifactIntegrityError("feature_schema must be a JSON object")
        schema = FeatureSchema.from_dict(schema_payload)
        if str(payload["feature_schema_digest"]) != _digest(schema.as_dict()):
            raise ArtifactIntegrityError("feature schema digest mismatch")
        try:
            models = tuple(
                FrozenBinaryModel.from_dict(model_payload)
                for model_payload in payload["models"]  # type: ignore[union-attr]
            )
            group_hashes = tuple(
                str(item)
                for item in payload["development_group_hashes"]  # type: ignore[union-attr]
            )
            suite = cls(
                schema=schema,
                models=models,
                development_group_hashes=group_hashes,
                random_seed=int(payload["random_seed"]),
                sklearn_version=str(payload["sklearn_version"]),
            )
        except ArtifactIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("invalid model artifact values") from exc
        if str(payload["development_group_digest"]) != suite.development_group_digest:
            raise ArtifactIntegrityError("development group digest mismatch")
        if str(payload["suite_digest"]) != suite.suite_digest:
            raise ArtifactIntegrityError("model suite digest mismatch")
        return suite

    def assert_group_disjoint(self, features: _ValidatedFeatures) -> None:
        overlap = set(self.development_group_hashes).intersection(_group_hashes(features))
        if overlap:
            raise DataLeakageError(
                "evaluation query families overlap development fit or calibration families"
            )

    def predict_proba(
        self,
        batch: FeatureBatch,
        *,
        model_name: ModelName = "full",
    ) -> np.ndarray:
        features = _validate_features(batch, self.schema)
        return _predict_model(self.model(model_name), features, self.schema)


def canonical_h1_model_artifact_bytes(suite: FrozenModelSuite) -> bytes:
    """Serialize the frozen H1 full model as canonical UTF-8 JSON without a final LF."""

    if not isinstance(suite, FrozenModelSuite):
        raise TypeError("suite must be a FrozenModelSuite")
    return _canonical_json(suite.model("full").as_dict()).encode("utf-8")


def canonical_h2_model_suite_artifact_bytes(suite: FrozenModelSuite) -> bytes:
    """Serialize the frozen H2 suite as canonical UTF-8 JSON without a final LF."""

    if not isinstance(suite, FrozenModelSuite):
        raise TypeError("suite must be a FrozenModelSuite")
    return suite.to_json().encode("utf-8")


def _fit_preprocessor_and_model(
    model_name: ModelName,
    blocks: tuple[FeatureBlockName, ...],
    training: _ValidatedLabeledFeatures,
    calibration: _ValidatedLabeledFeatures,
    schema: FeatureSchema,
    *,
    random_seed: int,
) -> FrozenBinaryModel:
    numeric_names = schema.numeric_for(blocks)
    categorical_names = schema.categorical_for(blocks)

    training_numeric = _numeric_matrix(training, schema, numeric_names)
    calibration_numeric = _numeric_matrix(calibration, schema, numeric_names)
    if np.any(np.all(np.isnan(training_numeric), axis=0)):
        raise FeatureSchemaError("a development-fit numeric feature is entirely missing")
    imputer = SimpleImputer(strategy="median")
    training_imputed = imputer.fit_transform(training_numeric)
    calibration_imputed = imputer.transform(calibration_numeric)
    if not np.all(np.isfinite(imputer.statistics_)):
        raise FeatureSchemaError("numeric imputation produced a non-finite value")

    training_engineered, engineered_names = _engineer_numeric(
        training_imputed,
        numeric_names,
        schema,
        blocks,
    )
    calibration_engineered, _ = _engineer_numeric(
        calibration_imputed,
        numeric_names,
        schema,
        blocks,
    )
    scaler = StandardScaler()
    training_scaled = scaler.fit_transform(training_engineered)
    calibration_scaled = scaler.transform(calibration_engineered)

    training_categorical = _categorical_matrix(training, schema, categorical_names)
    calibration_categorical = _categorical_matrix(calibration, schema, categorical_names)
    if categorical_names:
        encoder = OneHotEncoder(
            dtype=np.float64,
            handle_unknown="error",
            sparse_output=False,
        )
        training_encoded = encoder.fit_transform(training_categorical)
        try:
            calibration_encoded = encoder.transform(calibration_categorical)
        except ValueError as exc:
            raise FeatureSchemaError(
                "calibration contains a categorical level absent from development fit"
            ) from exc
        categorical_levels = tuple(
            tuple(str(level) for level in levels) for levels in encoder.categories_
        )
    else:
        training_encoded = np.empty((len(training.matrix), 0), dtype=np.float64)
        calibration_encoded = np.empty((len(calibration.matrix), 0), dtype=np.float64)
        categorical_levels = ()

    transformed_names = engineered_names + _one_hot_names(
        categorical_names,
        categorical_levels,
    )
    training_design = np.column_stack((training_scaled, training_encoded))
    calibration_design = np.column_stack((calibration_scaled, calibration_encoded))

    classifier = LogisticRegression(
        C=1.0,
        max_iter=2_000,
        random_state=random_seed,
        solver="lbfgs",
        tol=1e-10,
    )
    classifier.fit(training_design, training.labels)
    calibration_scores = classifier.decision_function(calibration_design)
    calibrator = LogisticRegression(
        C=1.0,
        max_iter=2_000,
        random_state=random_seed,
        solver="lbfgs",
        tol=1e-10,
    )
    calibrator.fit(calibration_scores.reshape(-1, 1), calibration.labels)

    return FrozenBinaryModel(
        name=model_name,
        blocks=blocks,
        numeric_features=numeric_names,
        categorical_features=categorical_names,
        categorical_levels=categorical_levels,
        imputation_values=tuple(float(item) for item in imputer.statistics_),
        engineered_numeric_features=engineered_names,
        standardization_mean=tuple(float(item) for item in scaler.mean_),
        standardization_scale=tuple(float(item) for item in scaler.scale_),
        transformed_feature_names=transformed_names,
        coefficients=tuple(float(item) for item in classifier.coef_[0]),
        intercept=float(classifier.intercept_[0]),
        calibration_slope=float(calibrator.coef_[0, 0]),
        calibration_intercept=float(calibrator.intercept_[0]),
    )


def fit_frozen_model_suite(
    training: LabeledFeatureBatch,
    calibration: LabeledFeatureBatch,
    *,
    schema: FeatureSchema = REGISTERED_FEATURE_SCHEMA,
    random_seed: int = 20260713,
) -> FrozenModelSuite:
    """Fit and calibrate the four registered models using development data only."""
    _require_partition(training, "development-fit")
    _require_partition(calibration, "development-calibration")
    if random_seed < 0:
        raise ModelingContractError("random_seed must be non-negative")

    validated_training = _validate_labeled_features(training, schema)
    validated_calibration = _validate_labeled_features(calibration, schema)
    for name, labels in (
        ("development fit", validated_training.labels),
        ("development calibration", validated_calibration.labels),
    ):
        if len(np.unique(labels)) != 2:
            raise ModelingContractError(f"{name} labels must contain both binary classes")

    training_groups = set(_group_hashes(validated_training))
    calibration_groups = set(_group_hashes(validated_calibration))
    if training_groups.intersection(calibration_groups):
        raise DataLeakageError(
            "development fit and calibration query families must be disjoint"
        )

    models = tuple(
        _fit_preprocessor_and_model(
            name,
            blocks,
            validated_training,
            validated_calibration,
            schema,
            random_seed=random_seed,
        )
        for name, blocks in MODEL_SPECS
    )
    return FrozenModelSuite(
        schema=schema,
        models=models,
        development_group_hashes=tuple(sorted(training_groups | calibration_groups)),
        random_seed=random_seed,
        sklearn_version=sklearn.__version__,
    )


def _predict_model(
    model: FrozenBinaryModel,
    features: _ValidatedFeatures,
    schema: FeatureSchema,
) -> np.ndarray:
    numeric = _numeric_matrix(features, schema, model.numeric_features)
    imputation = np.asarray(model.imputation_values, dtype=np.float64)
    numeric = np.where(np.isnan(numeric), imputation, numeric)
    engineered, engineered_names = _engineer_numeric(
        numeric,
        model.numeric_features,
        schema,
        model.blocks,
    )
    if engineered_names != model.engineered_numeric_features:
        raise FeatureSchemaError("engineered feature order no longer matches the artifact")
    scaled = (
        engineered - np.asarray(model.standardization_mean, dtype=np.float64)
    ) / np.asarray(model.standardization_scale, dtype=np.float64)

    categorical = _categorical_matrix(features, schema, model.categorical_features)
    encoded_columns: list[np.ndarray] = []
    for column_index, levels in enumerate(model.categorical_levels):
        values = categorical[:, column_index]
        unknown = sorted(set(values) - set(levels))
        if unknown:
            raise FeatureSchemaError(
                f"categorical feature {model.categorical_features[column_index]!r} "
                f"contains unknown levels: {unknown}"
            )
        encoded_columns.extend((values == level).astype(np.float64) for level in levels)
    encoded = (
        np.column_stack(encoded_columns)
        if encoded_columns
        else np.empty((len(features.matrix), 0), dtype=np.float64)
    )
    design = np.column_stack((scaled, encoded))
    scores = design @ np.asarray(model.coefficients, dtype=np.float64) + model.intercept
    return expit(model.calibration_slope * scores + model.calibration_intercept)


@dataclass(frozen=True)
class BinaryPredictiveMetrics:
    log_loss: float
    brier_score: float
    auprc: float | None


@dataclass(frozen=True)
class H2MetricImprovement:
    """System-policy minus full losses and full minus system-policy AUPRC."""

    log_loss_reduction: float
    brier_score_reduction: float
    relative_brier_reduction: float
    auprc_gain: float | None


@dataclass(frozen=True)
class FamilyPairedLoss:
    corpus_id: str
    family_id: str
    n_rows: int
    system_policy_log_loss: float
    full_log_loss: float
    system_policy_brier_score: float
    full_brier_score: float


@dataclass(frozen=True)
class CorpusH2Metrics:
    corpus_id: str
    n_rows: int
    n_families: int
    model_metrics: tuple[tuple[ModelName, BinaryPredictiveMetrics], ...]
    system_policy_to_full: H2MetricImprovement
    family_paired_losses: tuple[FamilyPairedLoss, ...]

    def for_model(self, name: ModelName) -> BinaryPredictiveMetrics:
        for model_name, metrics in self.model_metrics:
            if model_name == name:
                return metrics
        raise KeyError(name)


@dataclass(frozen=True)
class H2Evaluation:
    fixed_corpora: tuple[str, ...]
    corpus_metrics: tuple[CorpusH2Metrics, ...]
    equal_corpus_model_metrics: tuple[tuple[ModelName, BinaryPredictiveMetrics], ...]
    equal_corpus_system_policy_to_full: H2MetricImprovement
    row_identity_digest: str
    weighting: str = "equal-family-within-fixed-corpus_then-equal-corpus"
    inference: str = "descriptive-paired-metrics-only"

    def equal_corpus_for_model(self, name: ModelName) -> BinaryPredictiveMetrics:
        for model_name, metrics in self.equal_corpus_model_metrics:
            if model_name == name:
                return metrics
        raise KeyError(name)


@dataclass(frozen=True)
class GeometryGainThresholds:
    """Frozen minimum gains for the full model over the system-policy model."""

    log_loss_reduction: float
    brier_score_reduction: float
    auprc_gain: float

    def __post_init__(self) -> None:
        values = (
            self.log_loss_reduction,
            self.brier_score_reduction,
            self.auprc_gain,
        )
        if not np.all(np.isfinite(values)) or any(value < 0.0 for value in values):
            raise ModelingContractError("geometry gain thresholds must be finite and non-negative")


@dataclass(frozen=True)
class CorpusGeometryGainDecision:
    corpus_id: str
    observed: H2MetricImprovement
    passed: bool


@dataclass(frozen=True)
class GeometryGainGateDecision:
    thresholds: GeometryGainThresholds
    minimum_corpora: int
    corpus_decisions: tuple[CorpusGeometryGainDecision, ...]
    passing_corpora: tuple[str, ...]
    passed: bool


def evaluate_geometry_gain_gate(
    evaluation: H2Evaluation,
    *,
    thresholds: GeometryGainThresholds,
    minimum_corpora: int,
) -> GeometryGainGateDecision:
    """Apply the registered all-metrics gate independently to each fixed corpus."""

    if not isinstance(evaluation, H2Evaluation):
        raise ModelingContractError("evaluation must be an H2Evaluation")
    if not isinstance(thresholds, GeometryGainThresholds):
        raise ModelingContractError("thresholds must be GeometryGainThresholds")
    if (
        isinstance(minimum_corpora, bool)
        or not isinstance(minimum_corpora, int)
        or not 1 <= minimum_corpora <= len(evaluation.fixed_corpora)
    ):
        raise ModelingContractError(
            "minimum_corpora must be an integer inside the fixed suite"
        )
    decisions = tuple(
        CorpusGeometryGainDecision(
            corpus_id=corpus.corpus_id,
            observed=corpus.system_policy_to_full,
            passed=(
                corpus.system_policy_to_full.log_loss_reduction
                > thresholds.log_loss_reduction
                and corpus.system_policy_to_full.brier_score_reduction
                > thresholds.brier_score_reduction
                and corpus.system_policy_to_full.auprc_gain is not None
                and corpus.system_policy_to_full.auprc_gain > thresholds.auprc_gain
            ),
        )
        for corpus in evaluation.corpus_metrics
    )
    passing = tuple(decision.corpus_id for decision in decisions if decision.passed)
    return GeometryGainGateDecision(
        thresholds=thresholds,
        minimum_corpora=minimum_corpora,
        corpus_decisions=decisions,
        passing_corpora=passing,
        passed=len(passing) >= minimum_corpora,
    )


def _family_equal_weights(family_ids: Sequence[str]) -> np.ndarray:
    counts: dict[str, int] = {}
    for family_id in family_ids:
        counts[family_id] = counts.get(family_id, 0) + 1
    return np.asarray([1.0 / counts[family_id] for family_id in family_ids], dtype=np.float64)


def _binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> BinaryPredictiveMetrics:
    auprc = (
        None
        if len(np.unique(labels)) != 2
        else float(
            average_precision_score(
                labels,
                probabilities,
                sample_weight=weights,
            )
        )
    )
    return BinaryPredictiveMetrics(
        log_loss=float(log_loss(labels, probabilities, labels=(0, 1), sample_weight=weights)),
        brier_score=float(brier_score_loss(labels, probabilities, sample_weight=weights)),
        auprc=auprc,
    )


def _improvement(
    reference: BinaryPredictiveMetrics,
    full: BinaryPredictiveMetrics,
) -> H2MetricImprovement:
    if reference.brier_score <= 0.0:
        raise ModelingContractError(
            "relative Brier reduction is undefined when system-policy Brier score is zero"
        )
    return H2MetricImprovement(
        log_loss_reduction=reference.log_loss - full.log_loss,
        brier_score_reduction=reference.brier_score - full.brier_score,
        relative_brier_reduction=1.0 - full.brier_score / reference.brier_score,
        auprc_gain=(
            None
            if reference.auprc is None or full.auprc is None
            else full.auprc - reference.auprc
        ),
    )


def _mean_if_defined(values: Sequence[float | None]) -> float | None:
    observed = tuple(values)
    if any(value is None for value in observed):
        return None
    return float(np.mean(observed))


def _validate_fixed_corpora(
    observed: Sequence[str],
    fixed_corpora: Sequence[str],
) -> tuple[str, ...]:
    fixed = tuple(fixed_corpora)
    if not fixed or any(not isinstance(corpus, str) or not corpus for corpus in fixed):
        raise ModelingContractError("fixed_corpora must contain non-empty strings")
    if len(fixed) != len(set(fixed)):
        raise ModelingContractError("fixed_corpora cannot contain duplicates")
    if set(observed) != set(fixed):
        missing = sorted(set(fixed) - set(observed))
        unexpected = sorted(set(observed) - set(fixed))
        raise ModelingContractError(
            f"sealed corpora do not match the fixed suite; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return fixed


def evaluate_h2_by_corpus(
    suite: FrozenModelSuite,
    sealed: LabeledFeatureBatch,
    *,
    fixed_corpora: Sequence[str],
) -> H2Evaluation:
    """Compute paired descriptive H2 metrics on group-disjoint sealed rows.

    Every model receives the same row matrix.  Row weights sum to one per query
    family inside each corpus, so a family with more nested rows has no extra
    weight.  AUPRC is a corpus metric and is computed with those same family
    weights; it is not decomposed into fictitious per-family contributions.
    """
    _require_partition(sealed, "sealed")
    features = _validate_labeled_features(sealed, suite.schema)
    suite.assert_group_disjoint(features)
    fixed = _validate_fixed_corpora(features.corpus_ids, fixed_corpora)

    probabilities = {
        model.name: _predict_model(model, features, suite.schema) for model in suite.models
    }
    corpus_results: list[CorpusH2Metrics] = []
    for corpus_id in fixed:
        indices = np.asarray(
            [index for index, value in enumerate(features.corpus_ids) if value == corpus_id],
            dtype=np.int64,
        )
        labels = features.labels[indices]
        family_ids = tuple(features.family_ids[index] for index in indices)
        weights = _family_equal_weights(family_ids)
        metrics = tuple(
            (
                model.name,
                _binary_metrics(labels, probabilities[model.name][indices], weights),
            )
            for model in suite.models
        )
        metric_lookup = dict(metrics)

        family_losses: list[FamilyPairedLoss] = []
        for family_id in sorted(set(family_ids)):
            local = np.asarray(
                [index for index, value in enumerate(family_ids) if value == family_id],
                dtype=np.int64,
            )
            family_labels = labels[local]
            reference_probability = probabilities["system-policy"][indices][local]
            full_probability = probabilities["full"][indices][local]
            family_losses.append(
                FamilyPairedLoss(
                    corpus_id=corpus_id,
                    family_id=family_id,
                    n_rows=len(local),
                    system_policy_log_loss=float(
                        log_loss(family_labels, reference_probability, labels=(0, 1))
                    ),
                    full_log_loss=float(
                        log_loss(family_labels, full_probability, labels=(0, 1))
                    ),
                    system_policy_brier_score=float(
                        brier_score_loss(family_labels, reference_probability)
                    ),
                    full_brier_score=float(brier_score_loss(family_labels, full_probability)),
                )
            )
        corpus_results.append(
            CorpusH2Metrics(
                corpus_id=corpus_id,
                n_rows=len(indices),
                n_families=len(set(family_ids)),
                model_metrics=metrics,
                system_policy_to_full=_improvement(
                    metric_lookup["system-policy"],
                    metric_lookup["full"],
                ),
                family_paired_losses=tuple(family_losses),
            )
        )

    equal_corpus_metrics = tuple(
        (
            model_name,
            BinaryPredictiveMetrics(
                log_loss=float(
                    np.mean([result.for_model(model_name).log_loss for result in corpus_results])
                ),
                brier_score=float(
                    np.mean(
                        [result.for_model(model_name).brier_score for result in corpus_results]
                    )
                ),
                auprc=_mean_if_defined(
                    [result.for_model(model_name).auprc for result in corpus_results]
                ),
            ),
        )
        for model_name, _ in MODEL_SPECS
    )
    equal_lookup = dict(equal_corpus_metrics)
    row_identity_digest = _digest(
        [
            {
                "corpus_id": corpus_id,
                "family_id": family_id,
                "row_id": row_id,
            }
            for corpus_id, family_id, row_id in zip(
                features.corpus_ids,
                features.family_ids,
                features.row_ids,
                strict=True,
            )
        ]
    )
    return H2Evaluation(
        fixed_corpora=fixed,
        corpus_metrics=tuple(corpus_results),
        equal_corpus_model_metrics=equal_corpus_metrics,
        equal_corpus_system_policy_to_full=_improvement(
            equal_lookup["system-policy"],
            equal_lookup["full"],
        ),
        row_identity_digest=row_identity_digest,
    )


@dataclass(frozen=True)
class PredictiveGeometryRiskContrast:
    """H1 point contrast with no causal or interval interpretation."""

    estimate: float
    per_row_differences: tuple[float, ...]
    fixed_corpora: tuple[str, ...]
    model_digest: str
    causal: bool = False
    inference: str = "point-estimate-only_no-hierarchical-interval"


def _replace_feature_values(
    batch: FeatureBatch,
    schema: FeatureSchema,
    values: Mapping[str, float],
) -> FeatureBatch:
    matrix = np.asarray(batch.features, dtype=object).copy()
    lookup = {name: index for index, name in enumerate(schema.input_features)}
    for name, value in values.items():
        matrix[:, lookup[name]] = float(value)
    return FeatureBatch(
        partition=batch.partition,
        feature_names=batch.feature_names,
        features=matrix,
        corpus_ids=batch.corpus_ids,
        family_ids=batch.family_ids,
        row_ids=batch.row_ids,
    )


def _equal_family_fixed_corpus_mean(
    values: np.ndarray,
    features: _ValidatedFeatures,
    fixed_corpora: Sequence[str],
) -> float:
    corpus_means: list[float] = []
    for corpus_id in fixed_corpora:
        family_means: list[float] = []
        families = sorted(
            {
                family_id
                for observed_corpus, family_id in zip(
                    features.corpus_ids,
                    features.family_ids,
                    strict=True,
                )
                if observed_corpus == corpus_id
            }
        )
        for family_id in families:
            indices = [
                index
                for index, (observed_corpus, observed_family) in enumerate(
                    zip(features.corpus_ids, features.family_ids, strict=True)
                )
                if observed_corpus == corpus_id and observed_family == family_id
            ]
            family_means.append(float(np.mean(values[indices])))
        corpus_means.append(float(np.mean(family_means)))
    return float(np.mean(corpus_means))


def predictive_geometry_risk_contrast(
    suite: FrozenModelSuite,
    sealed: FeatureBatch,
    *,
    low_geometry: Mapping[str, float],
    high_geometry: Mapping[str, float],
    fixed_corpora: Sequence[str],
) -> PredictiveGeometryRiskContrast:
    """Contrast frozen full-model risk under two registered geometry profiles.

    All non-geometry values retain their observed sealed distribution.  The
    result is a predictive standardization contrast, not an intervention or a
    causal effect.  Clustered interval construction belongs in a separately
    registered analysis runner.
    """
    _require_partition(sealed, "sealed")
    features = _validate_features(sealed, suite.schema)
    suite.assert_group_disjoint(features)
    fixed = _validate_fixed_corpora(features.corpus_ids, fixed_corpora)

    if set(low_geometry) != set(high_geometry):
        raise ModelingContractError("low and high geometry profiles must set identical features")
    required = {suite.schema.lid_feature, suite.schema.instability_feature}
    if not required.issubset(low_geometry):
        raise ModelingContractError("geometry profiles must set LID and instability")
    allowed = set(suite.schema.geometry_numeric)
    unexpected = sorted(set(low_geometry) - allowed)
    if unexpected:
        raise ModelingContractError(
            f"geometry profiles contain non-geometry features: {unexpected}"
        )
    for profile in (low_geometry, high_geometry):
        if not np.all(np.isfinite(tuple(profile.values()))):
            raise ModelingContractError("geometry profile values must be finite")

    low_batch = _replace_feature_values(sealed, suite.schema, low_geometry)
    high_batch = _replace_feature_values(sealed, suite.schema, high_geometry)
    low_probability = suite.predict_proba(low_batch, model_name="full")
    high_probability = suite.predict_proba(high_batch, model_name="full")
    differences = high_probability - low_probability
    return PredictiveGeometryRiskContrast(
        estimate=_equal_family_fixed_corpus_mean(differences, features, fixed),
        per_row_differences=tuple(float(item) for item in differences),
        fixed_corpora=fixed,
        model_digest=suite.model("full").model_digest,
    )
