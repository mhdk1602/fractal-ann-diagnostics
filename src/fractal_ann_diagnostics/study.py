"""Closed manifest schema and one-shot controls for the confirmatory study."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import ssl
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import unquote, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactVerificationReceipt,
    load_local_artifact_map,
    load_verification_receipt,
    read_secure_control_file,
    verify_local_artifacts,
    write_exclusive_receipt_bytes,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OCI_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_PLACEHOLDERS = {"", "tbd", "todo", "latest", "main", "master", "unassigned"}
_RECEIPT_DIGEST_TOKEN = "{manifest_sha256}"
PROTOCOL_REGISTRATION_RECEIPT_SCHEMA = "fractal-ann-protocol-registration-v1"
PROTOCOL_REGISTRY_RECORD_SCHEMA = "fractal-ann-protocol-registry-record-v1"
MAX_PROTOCOL_REGISTRY_RECORD_BYTES = 64 * 1024
_PROTOCOL_REGISTRY_FETCH_TIMEOUT_SECONDS = 10.0
_PROTOCOL_REGISTRY_RECORD_FIELDS = {
    "manifest_sha256",
    "protocol_version",
    "registered_at_utc",
    "registry_identity",
    "registry_uri",
    "schema_version",
}
_PROTOCOL_REGISTRATION_FIELDS = {
    "manifest_sha256",
    "protocol_version",
    "registered_at_utc",
    "registry_identity",
    "registry_record_sha256",
    "registry_uri",
    "schema_version",
}
_SEALED_RUN_RECEIPT_FIELDS = {
    "code_commit",
    "manifest_sha256",
    "protocol_registration_receipt_sha256",
    "protocol_registration_receipt_uri",
    "protocol_registration_record_uri",
    "protocol_version",
    "receipt_uri",
    "runner_identity",
    "runner_image",
    "started_at_utc",
    "verification_receipt_sha256",
    "verification_receipt_uri",
}

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
REGISTERED_ACTION_SET = (
    "hnsw-low",
    "hnsw-high",
    "exact-authorized",
    "abstain",
)
REGISTERED_PRIMARY_CLAIM = (
    "On the fixed five-corpus suite, a frozen full model that adds LID at k=50, "
    "LID-CV, relative contrast, and radius expansion to the frozen system-policy "
    "baseline improves held-out prediction of intent-to-treat low-effort action "
    "failure beyond the frozen H2 thresholds; and a frozen adaptive controller "
    "achieves an equal-corpus mean family-level relative end-to-end request-latency "
    "reduction greater than 10% relative to a frozen static action while authorized "
    "retrieval-target attainment and complete-evidence sufficiency remain noninferior "
    "within one percentage point, the equal-corpus mean of within-corpus "
    "proposed-to-comparator p95 ratios of family-mean end-to-end request latency "
    "remains below 1.25, and no denied item is emitted at the controlled retrieval "
    "boundary."
)
REGISTERED_POWER_ENDPOINTS = (
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
REGISTERED_POWER_FAMILY_CANDIDATES = (25, 50, 75, 100, 150, 200)

_ROOT_FIELDS = {
    "analysis",
    "artifacts",
    "claim_scope",
    "freeze_blockers",
    "primary_claim",
    "protocol_version",
    "schema_version",
    "sealed_execution",
    "status",
}
_ANALYSIS_FIELDS = {
    "action_set",
    "alpha",
    "bootstrap_replicates",
    "bootstrap_seed",
    "cluster_unit",
    "corpus_weighting",
    "cost_estimand",
    "evidence_corpora",
    "evidence_sufficiency_noninferiority_margin",
    "failure_recall_threshold",
    "fixed_corpora",
    "gatekeeping",
    "geometry_candidate_model",
    "geometry_gain_metrics",
    "geometry_gain_thresholds",
    "geometry_reference_model",
    "h1_minimum_risk_increase",
    "high_geometry",
    "interval_construction",
    "k",
    "low_geometry",
    "maximum_entitlement_violations",
    "maximum_p95_latency_ratio",
    "minimum_corpora_with_geometry_gain",
    "minimum_cost_reduction",
    "nested_rows_per_family",
    "power",
    "power_target",
    "retrieval_target_noninferiority_margin",
    "static_comparator_action",
}
_GEOMETRY_GAIN_THRESHOLD_FIELDS = {
    "auprc_gain",
    "brier_score_reduction",
    "log_loss_reduction",
}
_POWER_FIELDS = {
    "candidate_families_per_corpus",
    "dependence_source",
    "effect_scenarios",
    "joint_success_event",
    "model",
    "registered_endpoints",
    "selected_families_per_corpus",
    "selected_joint_power_lower_bound",
    "simulation_count",
    "simulation_seed",
}
_SEALED_EXECUTION_FIELDS = {
    "approval_environment",
    "code_commit",
    "custodian",
    "hardware",
    "interactive_access",
    "label_artifacts_withheld_until_prediction_receipt",
    "public_query_reidentification_risk",
    "receipt_uri_template",
    "reserve_fraction",
    "results_store",
    "runner_identity",
    "runner_image",
    "runner_network_access",
}
_HARDWARE_FIELDS = {
    "accelerator",
    "cpu_model",
    "instance_type",
    "logical_cores",
    "memory_gib",
    "operating_system",
    "provider",
    "region",
}
_ARTIFACT_BASE_FIELDS = {"id", "kind", "license", "revision", "role", "sha256", "uri"}
_CORPUS_BOUND_ROLES = {
    "corpus-normalizer",
    "online-execution",
    "policy-workload",
    "sealed-inputs",
    "sealed-labels",
}
_ARTIFACT_ROLE_SPECS: dict[str, tuple[str, int]] = {
    "sealed-inputs": ("dataset", len(FIXED_CORPORA)),
    "sealed-labels": ("dataset", len(FIXED_CORPORA)),
    "online-execution": ("execution", len(FIXED_CORPORA)),
    "corpus-normalizer": ("normalizer", len(FIXED_CORPORA)),
    "policy-workload": ("policy-data", len(FIXED_CORPORA)),
    "development-fit-data": ("dataset", 1),
    "development-calibration-data": ("dataset", 1),
    "query-partition-audit": ("partition-audit", 1),
    "primary-embedding": ("embedding", 1),
    "exact-authorized-oracle": ("backend", 1),
    "strict-authorized-hnsw": ("backend", 1),
    "opa-pdp": ("policy", 1),
    "frozen-controller": ("controller", 1),
    "static-comparator": ("comparator", 1),
    "h1-predictive-model": ("model", 1),
    "h2-model-suite": ("model", 1),
    "power-analysis-report": ("analysis", 1),
    "analysis-runner": ("analysis", 1),
    "source-code": ("source", 1),
}


class StudyManifestError(ValueError):
    """Raised when a study manifest is incomplete or internally inconsistent."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _decode_json(encoded: bytes, *, label: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StudyManifestError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise StudyManifestError(f"{label} contains non-finite number {value!r}")

    try:
        text = encoded.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StudyManifestError(f"cannot decode {label}: {exc}") from exc


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Return the digest used to bind a protocol to its sealed runner."""
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def load_study_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        payload = _decode_json(manifest_path.read_bytes(), label="study manifest")
    except (OSError, StudyManifestError) as exc:
        raise StudyManifestError(f"cannot load study manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StudyManifestError("study manifest root must be a JSON object")
    return payload


def _closed_object(
    value: object,
    fields: set[str],
    *,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyManifestError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise StudyManifestError(f"{path} field names must be strings")
    observed = set(value)
    if observed != fields:
        missing = sorted(fields - observed)
        unknown = sorted(observed - fields)
        raise StudyManifestError(
            f"{path} schema mismatch; missing={missing}, unknown={unknown}"
        )
    return value


def _is_placeholder(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS


def _registered_number(
    mapping: Mapping[str, Any],
    key: str,
    expected: float,
    *,
    path: str = "analysis",
) -> None:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudyManifestError(f"{path}.{key} must be numeric")
    if not math.isfinite(float(value)) or abs(float(value) - expected) > 1e-12:
        raise StudyManifestError(f"{path}.{key} must equal {expected}")


def _positive_integer(value: object, *, path: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StudyManifestError(f"{path} must be an integer of at least {minimum}")
    return value


def _draftable_number(
    value: object,
    *,
    path: str,
    frozen: bool,
    lower: float,
    upper: float,
    inclusive_lower: bool = True,
) -> float | None:
    if _is_placeholder(value):
        if frozen:
            raise StudyManifestError(f"{path} must be pinned before freeze")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudyManifestError(f"{path} must be numeric or 'tbd' in a draft")
    number = float(value)
    if not math.isfinite(number):
        raise StudyManifestError(f"{path} must be finite")
    lower_ok = number >= lower if inclusive_lower else number > lower
    if not lower_ok or number > upper:
        left = "[" if inclusive_lower else "("
        raise StudyManifestError(f"{path} must be in {left}{lower}, {upper}]")
    return number


def _draftable_integer(
    value: object,
    *,
    path: str,
    frozen: bool,
    minimum: int = 0,
) -> int | None:
    if _is_placeholder(value):
        if frozen:
            raise StudyManifestError(f"{path} must be pinned before freeze")
        return None
    return _positive_integer(value, path=path, minimum=minimum)


def _validate_freeze_blockers(value: object, *, frozen: bool) -> None:
    if not isinstance(value, list):
        raise StudyManifestError("freeze_blockers must be an array")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise StudyManifestError("freeze_blockers must contain non-empty strings")
    if len(value) != len(set(value)):
        raise StudyManifestError("freeze_blockers cannot contain duplicates")
    if frozen and value:
        raise StudyManifestError("a frozen manifest cannot retain freeze_blockers")
    if not frozen and not value:
        raise StudyManifestError("a draft manifest must state its explicit freeze blockers")


def _validate_power(power_value: object, *, frozen: bool, power_target: float) -> None:
    power = _closed_object(power_value, _POWER_FIELDS, path="analysis.power")
    if power["model"] != "development-family-cluster-resampling":
        raise StudyManifestError(
            "analysis.power.model must be 'development-family-cluster-resampling'"
        )
    if power["joint_success_event"] != "h2-and-h3-all-gates-pass":
        raise StudyManifestError(
            "analysis.power.joint_success_event must be 'h2-and-h3-all-gates-pass'"
        )
    endpoints = power["registered_endpoints"]
    if not isinstance(endpoints, list) or tuple(endpoints) != REGISTERED_POWER_ENDPOINTS:
        raise StudyManifestError(
            "analysis.power.registered_endpoints must equal the registered ordered "
            "joint endpoint list"
        )
    _pinned_text(
        power["dependence_source"],
        path="analysis.power.dependence_source",
        frozen=frozen,
    )
    effect_scenarios = power["effect_scenarios"]
    if not isinstance(effect_scenarios, list) or not effect_scenarios:
        raise StudyManifestError(
            "analysis.power.effect_scenarios must be a non-empty array"
        )
    for position, scenario in enumerate(effect_scenarios):
        _pinned_text(
            scenario,
            path=f"analysis.power.effect_scenarios[{position}]",
            frozen=frozen,
        )
    if len(effect_scenarios) != len(set(effect_scenarios)):
        raise StudyManifestError(
            "analysis.power.effect_scenarios cannot contain duplicates"
        )
    _draftable_integer(
        power["simulation_seed"],
        path="analysis.power.simulation_seed",
        frozen=frozen,
        minimum=0,
    )
    _positive_integer(
        power["simulation_count"],
        path="analysis.power.simulation_count",
        minimum=5_000,
    )

    candidates_value = power["candidate_families_per_corpus"]
    if not isinstance(candidates_value, list) or not candidates_value:
        raise StudyManifestError(
            "analysis.power.candidate_families_per_corpus must be a non-empty array"
        )
    candidates = tuple(
        _positive_integer(
            candidate,
            path="analysis.power.candidate_families_per_corpus[]",
        )
        for candidate in candidates_value
    )
    if candidates != REGISTERED_POWER_FAMILY_CANDIDATES:
        raise StudyManifestError(
            "analysis.power.candidate_families_per_corpus must equal the registered "
            "candidate grid [25, 50, 75, 100, 150, 200]"
        )
    selected = _draftable_integer(
        power["selected_families_per_corpus"],
        path="analysis.power.selected_families_per_corpus",
        frozen=frozen,
        minimum=2,
    )
    if selected is not None and selected not in candidates:
        raise StudyManifestError(
            "analysis.power.selected_families_per_corpus must be a registered candidate"
        )
    lower_bound = _draftable_number(
        power["selected_joint_power_lower_bound"],
        path="analysis.power.selected_joint_power_lower_bound",
        frozen=frozen,
        lower=0.0,
        upper=1.0,
    )
    if frozen and lower_bound is not None and lower_bound < power_target:
        raise StudyManifestError(
            "analysis.power.selected_joint_power_lower_bound must reach "
            "analysis.power_target"
        )


def _validate_analysis(value: object, *, frozen: bool) -> None:
    analysis = _closed_object(value, _ANALYSIS_FIELDS, path="analysis")
    if analysis["k"] != 10 or isinstance(analysis["k"], bool):
        raise StudyManifestError("analysis.k must equal the registered K=10")
    for key, expected in (
        ("failure_recall_threshold", 0.90),
        ("alpha", 0.05),
        ("power_target", 0.90),
        ("retrieval_target_noninferiority_margin", 0.01),
        ("evidence_sufficiency_noninferiority_margin", 0.01),
        ("minimum_cost_reduction", 0.10),
        ("maximum_p95_latency_ratio", 1.25),
        ("h1_minimum_risk_increase", 0.0),
    ):
        _registered_number(analysis, key, expected)
    if analysis["bootstrap_seed"] != 20260713 or isinstance(
        analysis["bootstrap_seed"], bool
    ):
        raise StudyManifestError("analysis.bootstrap_seed must equal 20260713")
    if (
        isinstance(analysis["maximum_entitlement_violations"], bool)
        or analysis["maximum_entitlement_violations"] != 0
    ):
        raise StudyManifestError("analysis.maximum_entitlement_violations must equal zero")
    if analysis["minimum_corpora_with_geometry_gain"] != 4:
        raise StudyManifestError(
            "analysis.minimum_corpora_with_geometry_gain must equal four"
        )
    if analysis["geometry_reference_model"] != "system-policy":
        raise StudyManifestError(
            "analysis.geometry_reference_model must equal 'system-policy'"
        )
    if analysis["geometry_candidate_model"] != "full":
        raise StudyManifestError("analysis.geometry_candidate_model must equal 'full'")
    if analysis["geometry_gain_metrics"] != [
        "log_loss_reduction",
        "brier_score_reduction",
        "auprc_gain",
    ]:
        raise StudyManifestError(
            "analysis.geometry_gain_metrics must equal the registered ordered metrics"
        )
    gain_thresholds = _closed_object(
        analysis["geometry_gain_thresholds"],
        _GEOMETRY_GAIN_THRESHOLD_FIELDS,
        path="analysis.geometry_gain_thresholds",
    )
    for metric in sorted(_GEOMETRY_GAIN_THRESHOLD_FIELDS):
        _draftable_number(
            gain_thresholds[metric],
            path=f"analysis.geometry_gain_thresholds.{metric}",
            frozen=frozen,
            lower=0.0,
            upper=1.0,
        )
    geometry_profiles: dict[str, tuple[str, ...] | None] = {}
    for profile_name in ("low_geometry", "high_geometry"):
        profile_value = analysis[profile_name]
        if _is_placeholder(profile_value):
            if frozen:
                raise StudyManifestError(
                    f"analysis.{profile_name} must be pinned before freeze"
                )
            geometry_profiles[profile_name] = None
            continue
        if not isinstance(profile_value, Mapping) or not profile_value:
            raise StudyManifestError(
                f"analysis.{profile_name} must be a non-empty object or 'tbd' in a draft"
            )
        features: list[str] = []
        for feature, feature_value in profile_value.items():
            if (
                not isinstance(feature, str)
                or not feature
                or feature != feature.strip()
            ):
                raise StudyManifestError(
                    f"analysis.{profile_name} feature names must be canonical strings"
                )
            _draftable_number(
                feature_value,
                path=f"analysis.{profile_name}.{feature}",
                frozen=frozen,
                lower=-1_000_000_000.0,
                upper=1_000_000_000.0,
            )
            features.append(feature)
        geometry_profiles[profile_name] = tuple(sorted(features))
    if (
        geometry_profiles["low_geometry"] is not None
        and geometry_profiles["low_geometry"]
        != geometry_profiles["high_geometry"]
    ):
        raise StudyManifestError(
            "analysis.low_geometry and analysis.high_geometry must name identical features"
        )
    if not isinstance(analysis["fixed_corpora"], list):
        raise StudyManifestError("analysis.fixed_corpora must be an array")
    if tuple(analysis["fixed_corpora"]) != FIXED_CORPORA:
        raise StudyManifestError("analysis.fixed_corpora must equal the registered suite")
    if not isinstance(analysis["evidence_corpora"], list):
        raise StudyManifestError("analysis.evidence_corpora must be an array")
    if tuple(analysis["evidence_corpora"]) != EVIDENCE_CORPORA:
        raise StudyManifestError(
            "analysis.evidence_corpora must equal the registered evidence subset"
        )
    if not isinstance(analysis["action_set"], list):
        raise StudyManifestError("analysis.action_set must be an array")
    if tuple(analysis["action_set"]) != REGISTERED_ACTION_SET:
        raise StudyManifestError("analysis.action_set must equal the registered action set")
    comparator = analysis["static_comparator_action"]
    if _is_placeholder(comparator):
        if frozen:
            raise StudyManifestError(
                "analysis.static_comparator_action must be pinned before freeze"
            )
    elif comparator not in REGISTERED_ACTION_SET[:-1]:
        raise StudyManifestError(
            "analysis.static_comparator_action must be a non-abstention registered action"
        )
    for key, expected in (
        ("cluster_unit", "query_family"),
        ("corpus_weighting", "equal"),
        ("interval_construction", "directional-one-sided-95"),
        ("gatekeeping", "intersection-union-primary-gates"),
        ("cost_estimand", "end-to-end-request-latency-family-relative-reduction"),
    ):
        if analysis[key] != expected:
            raise StudyManifestError(f"analysis.{key} must equal {expected!r}")
    _positive_integer(
        analysis["bootstrap_replicates"],
        path="analysis.bootstrap_replicates",
        minimum=10_000,
    )
    _draftable_integer(
        analysis["nested_rows_per_family"],
        path="analysis.nested_rows_per_family",
        frozen=frozen,
        minimum=1,
    )
    _validate_power(
        analysis["power"],
        frozen=frozen,
        power_target=float(analysis["power_target"]),
    )


def _validate_artifact_pin(artifact: Mapping[str, Any], *, identifier: str) -> None:
    for field in ("uri", "revision", "sha256", "license"):
        value = artifact[field]
        if not isinstance(value, str) or _is_placeholder(value):
            raise StudyManifestError(
                f"frozen artifact {identifier!r} needs a pinned {field}"
            )
    if _SHA256.fullmatch(artifact["sha256"]) is None:
        raise StudyManifestError(f"frozen artifact {identifier!r} has an invalid SHA-256")


def _validate_artifacts(
    value: object,
    *,
    frozen: bool,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    if not isinstance(value, list) or not value:
        raise StudyManifestError("artifacts must be a non-empty array")
    identifiers: set[str] = set()
    grouped: dict[str, list[Mapping[str, Any]]] = {
        role: [] for role in _ARTIFACT_ROLE_SPECS
    }
    for position, artifact_value in enumerate(value):
        if not isinstance(artifact_value, Mapping):
            raise StudyManifestError(f"artifacts[{position}] must be an object")
        role = artifact_value.get("role")
        if not isinstance(role, str):
            raise StudyManifestError(f"artifacts[{position}].role must be a string")
        if role not in _ARTIFACT_ROLE_SPECS:
            raise StudyManifestError(f"artifacts[{position}].role is not registered: {role!r}")
        expected_fields = set(_ARTIFACT_BASE_FIELDS)
        if role in _CORPUS_BOUND_ROLES:
            expected_fields.add("corpus_id")
        artifact = _closed_object(
            artifact_value,
            expected_fields,
            path=f"artifacts[{position}]",
        )
        identifier = artifact["id"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise StudyManifestError(f"artifacts[{position}].id must be non-empty")
        if identifier in identifiers:
            raise StudyManifestError(f"duplicate artifact id: {identifier}")
        identifiers.add(identifier)
        expected_kind, _ = _ARTIFACT_ROLE_SPECS[role]
        if artifact["kind"] != expected_kind:
            raise StudyManifestError(
                f"artifact role {role!r} requires kind {expected_kind!r}"
            )
        for field in ("uri", "revision", "sha256", "license"):
            if not isinstance(artifact[field], str) or not artifact[field].strip():
                raise StudyManifestError(
                    f"artifacts[{position}].{field} must be a non-empty string"
                )
        if not _is_placeholder(artifact["sha256"]):
            if _SHA256.fullmatch(artifact["sha256"]) is None:
                raise StudyManifestError(
                    f"artifact {identifier!r} has an invalid SHA-256"
                )
        if role in _CORPUS_BOUND_ROLES:
            corpus_id = artifact["corpus_id"]
            if corpus_id not in FIXED_CORPORA:
                raise StudyManifestError(
                    f"artifact {identifier!r} has an unregistered corpus_id"
                )
        if frozen:
            _validate_artifact_pin(artifact, identifier=identifier)
        grouped[role].append(artifact)

    for role, (_, required_count) in _ARTIFACT_ROLE_SPECS.items():
        artifacts = grouped[role]
        if len(artifacts) != required_count:
            raise StudyManifestError(
                f"artifact role {role!r} requires exactly {required_count} entries; "
                f"observed {len(artifacts)}"
            )
        if role in _CORPUS_BOUND_ROLES:
            observed_corpora = [str(artifact["corpus_id"]) for artifact in artifacts]
            if set(observed_corpora) != set(FIXED_CORPORA) or len(observed_corpora) != len(
                set(observed_corpora)
            ):
                raise StudyManifestError(
                    f"artifact role {role!r} must cover every fixed corpus exactly once"
                )
    if frozen:
        inputs = {
            str(artifact["corpus_id"]): artifact for artifact in grouped["sealed-inputs"]
        }
        labels = {
            str(artifact["corpus_id"]): artifact for artifact in grouped["sealed-labels"]
        }
        executions = {
            str(artifact["corpus_id"]): artifact
            for artifact in grouped["online-execution"]
        }
        policies = {
            str(artifact["corpus_id"]): artifact for artifact in grouped["policy-workload"]
        }
        for corpus_id in FIXED_CORPORA:
            if (
                inputs[corpus_id]["uri"] == labels[corpus_id]["uri"]
                or inputs[corpus_id]["sha256"] == labels[corpus_id]["sha256"]
            ):
                raise StudyManifestError(
                    f"sealed inputs and labels for {corpus_id!r} must be separately pinned"
                )
            if (
                executions[corpus_id]["uri"]
                in {inputs[corpus_id]["uri"], labels[corpus_id]["uri"]}
                or executions[corpus_id]["sha256"]
                in {inputs[corpus_id]["sha256"], labels[corpus_id]["sha256"]}
            ):
                raise StudyManifestError(
                    f"online execution for {corpus_id!r} must be separately pinned"
                )
            if (
                policies[corpus_id]["uri"] in {
                    inputs[corpus_id]["uri"],
                    labels[corpus_id]["uri"],
                }
                or policies[corpus_id]["sha256"] in {
                    inputs[corpus_id]["sha256"],
                    labels[corpus_id]["sha256"],
                }
            ):
                raise StudyManifestError(
                    f"policy workload for {corpus_id!r} must be separately pinned"
                )
        fit_data = grouped["development-fit-data"][0]
        calibration_data = grouped["development-calibration-data"][0]
        if (
            fit_data["uri"] == calibration_data["uri"]
            or fit_data["sha256"] == calibration_data["sha256"]
        ):
            raise StudyManifestError(
                "development fit and calibration data must be separately pinned"
            )
    return {role: tuple(artifacts) for role, artifacts in grouped.items()}


def _pinned_text(value: object, *, path: str, frozen: bool) -> str | None:
    if not isinstance(value, str) or not value.strip():
        raise StudyManifestError(f"{path} must be a non-empty string")
    if _is_placeholder(value):
        if frozen:
            raise StudyManifestError(f"{path} must be pinned before freeze")
        return None
    return value


def _validate_hardware(value: object, *, frozen: bool) -> None:
    hardware = _closed_object(value, _HARDWARE_FIELDS, path="sealed_execution.hardware")
    for key in (
        "accelerator",
        "cpu_model",
        "instance_type",
        "operating_system",
        "provider",
        "region",
    ):
        _pinned_text(hardware[key], path=f"sealed_execution.hardware.{key}", frozen=frozen)
    _draftable_integer(
        hardware["logical_cores"],
        path="sealed_execution.hardware.logical_cores",
        frozen=frozen,
        minimum=1,
    )
    _draftable_number(
        hardware["memory_gib"],
        path="sealed_execution.hardware.memory_gib",
        frozen=frozen,
        lower=0.0,
        upper=1_000_000.0,
        inclusive_lower=False,
    )


def _validate_receipt_template(value: object, *, frozen: bool) -> None:
    template = _pinned_text(
        value,
        path="sealed_execution.receipt_uri_template",
        frozen=frozen,
    )
    if template is None:
        return
    if template.count(_RECEIPT_DIGEST_TOKEN) != 1:
        raise StudyManifestError(
            "sealed_execution.receipt_uri_template must contain one "
            "'{manifest_sha256}' token"
        )
    parsed = urlsplit(template)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise StudyManifestError(
            "sealed_execution.receipt_uri_template must be an absolute file URI"
        )
    if parsed.query or parsed.fragment or not parsed.path.startswith("/"):
        raise StudyManifestError(
            "sealed_execution.receipt_uri_template cannot contain a query or fragment"
        )
    if Path(unquote(parsed.path)).name != f"{_RECEIPT_DIGEST_TOKEN}.json":
        raise StudyManifestError(
            "sealed_execution.receipt_uri_template filename must be "
            "'{manifest_sha256}.json'"
        )


def _validate_sealed_execution(
    value: object,
    *,
    frozen: bool,
    artifacts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    sealed = _closed_object(
        value,
        _SEALED_EXECUTION_FIELDS,
        path="sealed_execution",
    )
    _registered_number(sealed, "reserve_fraction", 0.0, path="sealed_execution")
    for field in ("custodian", "approval_environment", "runner_identity"):
        _pinned_text(sealed[field], path=f"sealed_execution.{field}", frozen=frozen)
    results_store = _pinned_text(
        sealed["results_store"],
        path="sealed_execution.results_store",
        frozen=frozen,
    )
    if results_store is not None:
        parsed_store = urlsplit(results_store)
        if parsed_store.scheme not in {"file", "gs", "s3"} or not (
            parsed_store.netloc or parsed_store.path
        ):
            raise StudyManifestError(
                "sealed_execution.results_store must be a pinned file, gs, or s3 URI"
            )
    _validate_hardware(sealed["hardware"], frozen=frozen)
    _validate_receipt_template(sealed["receipt_uri_template"], frozen=frozen)
    if sealed["label_artifacts_withheld_until_prediction_receipt"] is not True:
        raise StudyManifestError(
            "sealed_execution.label_artifacts_withheld_until_prediction_receipt "
            "must be true"
        )
    if sealed["public_query_reidentification_risk"] != (
        "accepted-public-benchmark-limitation"
    ):
        raise StudyManifestError(
            "sealed_execution.public_query_reidentification_risk must acknowledge "
            "the public-benchmark limitation"
        )
    for field in ("runner_network_access", "interactive_access"):
        if sealed[field] != "disabled":
            raise StudyManifestError(f"sealed_execution.{field} must equal 'disabled'")

    code_commit = _pinned_text(
        sealed["code_commit"],
        path="sealed_execution.code_commit",
        frozen=frozen,
    )
    runner_image = _pinned_text(
        sealed["runner_image"],
        path="sealed_execution.runner_image",
        frozen=frozen,
    )
    if code_commit is not None and _GIT_COMMIT.fullmatch(code_commit) is None:
        raise StudyManifestError(
            "sealed_execution.code_commit must be one full lowercase Git commit"
        )
    if runner_image is not None and _OCI_DIGEST.fullmatch(runner_image) is None:
        raise StudyManifestError(
            "sealed_execution.runner_image must use an OCI SHA-256 digest"
        )
    if frozen and code_commit is not None:
        source_revision = artifacts["source-code"][0]["revision"]
        if source_revision != code_commit:
            raise StudyManifestError(
                "the source-code artifact revision must equal sealed_execution.code_commit"
            )


def validate_study_manifest(
    payload: Mapping[str, Any],
    *,
    require_frozen: bool = False,
) -> None:
    """Validate the closed schema and all prerequisites claimed by its status."""
    root = _closed_object(payload, _ROOT_FIELDS, path="study manifest")
    if root["schema_version"] != "1.0":
        raise StudyManifestError("schema_version must equal '1.0'")
    protocol_version = root["protocol_version"]
    if not isinstance(protocol_version, str) or not protocol_version.strip():
        raise StudyManifestError("protocol_version must be a non-empty string")
    status = root["status"]
    if status not in {"draft", "frozen"}:
        raise StudyManifestError("status must be 'draft' or 'frozen'")
    if require_frozen and status != "frozen":
        raise StudyManifestError("sealed execution requires status='frozen'")
    frozen = status == "frozen"
    expected_protocol = "0.3.0" if frozen else "0.3.0-draft"
    if protocol_version != expected_protocol:
        raise StudyManifestError(
            f"protocol_version must equal {expected_protocol!r} for status {status!r}"
        )
    if root["claim_scope"] != "suite-conditional-retrieval-control":
        raise StudyManifestError(
            "claim_scope must remain 'suite-conditional-retrieval-control'"
        )
    if root["primary_claim"] != REGISTERED_PRIMARY_CLAIM:
        raise StudyManifestError("primary_claim must equal the prespecified v0.3 claim")

    _validate_freeze_blockers(root["freeze_blockers"], frozen=frozen)
    _validate_analysis(root["analysis"], frozen=frozen)
    artifacts = _validate_artifacts(root["artifacts"], frozen=frozen)
    _validate_sealed_execution(
        root["sealed_execution"],
        frozen=frozen,
        artifacts=artifacts,
    )


def sealed_receipt_uri(payload: Mapping[str, Any]) -> str:
    """Derive the sole permitted receipt URI from the pinned template and digest."""
    sealed = payload.get("sealed_execution")
    if not isinstance(sealed, Mapping):
        raise StudyManifestError("sealed_execution must be an object")
    template = sealed.get("receipt_uri_template")
    if not isinstance(template, str) or _RECEIPT_DIGEST_TOKEN not in template:
        raise StudyManifestError("sealed execution has no valid receipt URI template")
    return template.replace(_RECEIPT_DIGEST_TOKEN, manifest_sha256(payload))


def _receipt_path_from_uri(uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise StudyManifestError("only pinned file receipt URIs support atomic opening")
    return Path(unquote(parsed.path))


def _validate_artifact_verification_receipt(
    receipt: ArtifactVerificationReceipt,
    *,
    payload: Mapping[str, Any],
    manifest_digest: str,
) -> None:
    if receipt.manifest_sha256 != manifest_digest:
        raise StudyManifestError(
            "artifact verification receipt is bound to a different manifest digest"
        )
    manifest_artifacts = payload["artifacts"]
    pinned_by_id = {
        str(artifact["id"]): str(artifact["sha256"])
        for artifact in manifest_artifacts
    }
    verified_by_id = {artifact.artifact_id: artifact for artifact in receipt.artifacts}
    if set(verified_by_id) != set(pinned_by_id):
        missing = sorted(set(pinned_by_id) - set(verified_by_id))
        unexpected = sorted(set(verified_by_id) - set(pinned_by_id))
        raise StudyManifestError(
            "artifact verification receipt must cover every manifest artifact exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for artifact_id, expected_sha256 in pinned_by_id.items():
        verified = verified_by_id[artifact_id]
        if not verified.exact:
            raise StudyManifestError(
                f"artifact verification receipt row {artifact_id!r} must be exact"
            )
        if (
            verified.expected_sha256 != expected_sha256
            or verified.verified_sha256 != expected_sha256
        ):
            raise StudyManifestError(
                f"artifact verification receipt digest mismatch for {artifact_id!r}"
            )


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StudyManifestError(f"{field} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StudyManifestError(f"{field} must be ISO 8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StudyManifestError(f"{field} must include a timezone")
    if parsed.utcoffset().total_seconds() != 0:
        raise StudyManifestError(f"{field} must use UTC")
    return parsed


def _validate_external_registry_uri(value: object) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise StudyManifestError("registry_uri must be a canonical HTTPS URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise StudyManifestError(
            "registry_uri must be an absolute HTTPS URL without credentials or fragment"
        )
    return value


class _NoProtocolRegistryRedirects(urllib_request.HTTPRedirectHandler):
    """Reject redirects while revalidating the independently registered record."""

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, msg, headers, newurl
        raise StudyManifestError(
            f"protocol registry revalidation refused HTTP redirect status {code}"
        )


def _fetch_protocol_registry_record(registry_uri: str, max_bytes: int) -> bytes:
    """Fetch one bounded registry record through verified HTTPS without redirects."""

    _validate_external_registry_uri(registry_uri)
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes <= 0
        or max_bytes > MAX_PROTOCOL_REGISTRY_RECORD_BYTES
    ):
        raise StudyManifestError(
            "protocol registry fetch max_bytes exceeds the registered safety limit"
        )
    context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    opener = urllib_request.build_opener(
        _NoProtocolRegistryRedirects(),
        urllib_request.HTTPSHandler(context=context),
    )
    request = urllib_request.Request(
        registry_uri,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "fractal-ann-diagnostics/0.3 registry-revalidation",
        },
        method="GET",
    )
    try:
        with opener.open(
            request,
            timeout=_PROTOCOL_REGISTRY_FETCH_TIMEOUT_SECONDS,
        ) as response:
            status = response.getcode()
            if status != 200:
                if isinstance(status, int) and 300 <= status < 400:
                    raise StudyManifestError(
                        "protocol registry revalidation refused an HTTP redirect"
                    )
                raise StudyManifestError(
                    f"protocol registry revalidation returned HTTP status {status}"
                )
            if response.geturl() != registry_uri:
                raise StudyManifestError(
                    "protocol registry revalidation response URL changed"
                )
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                if not content_length.isdecimal():
                    raise StudyManifestError(
                        "protocol registry response has an invalid Content-Length"
                    )
                if int(content_length) > max_bytes:
                    raise StudyManifestError(
                        "protocol registry record exceeds the maximum byte limit"
                    )
            encoded = response.read(max_bytes + 1)
    except StudyManifestError:
        raise
    except urllib_error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise StudyManifestError(
                "protocol registry revalidation refused an HTTP redirect"
            ) from exc
        raise StudyManifestError(
            f"protocol registry revalidation returned HTTP status {exc.code}"
        ) from exc
    except (OSError, TimeoutError, urllib_error.URLError, ValueError) as exc:
        raise StudyManifestError(
            "protocol registry record could not be fetched over verified HTTPS"
        ) from exc
    if not isinstance(encoded, bytes):
        raise StudyManifestError("protocol registry fetcher must return bytes")
    if len(encoded) > max_bytes:
        raise StudyManifestError(
            "protocol registry record exceeds the maximum byte limit"
        )
    return encoded


@dataclass(frozen=True)
class ProtocolRegistryRecord:
    """Canonical record deposited in the independent protocol registry."""

    manifest_sha256: str
    protocol_version: str
    registered_at_utc: str
    registry_identity: str
    registry_uri: str
    schema_version: str = PROTOCOL_REGISTRY_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.manifest_sha256) is None:
            raise StudyManifestError(
                "registry record manifest_sha256 must be a lowercase SHA-256"
            )
        if self.protocol_version != "0.3.0":
            raise StudyManifestError(
                "registry record protocol_version must equal '0.3.0'"
            )
        _parse_utc(self.registered_at_utc, field="registered_at_utc")
        if (
            not isinstance(self.registry_identity, str)
            or not self.registry_identity.strip()
            or self.registry_identity != self.registry_identity.strip()
        ):
            raise StudyManifestError(
                "registry_identity must be a canonical non-empty string"
            )
        _validate_external_registry_uri(self.registry_uri)
        if self.schema_version != PROTOCOL_REGISTRY_RECORD_SCHEMA:
            raise StudyManifestError(
                "registry record schema_version must equal "
                f"{PROTOCOL_REGISTRY_RECORD_SCHEMA!r}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProtocolRegistryRecord:
        record = _closed_object(
            payload,
            _PROTOCOL_REGISTRY_RECORD_FIELDS,
            path="protocol registry record",
        )
        return cls(
            manifest_sha256=record["manifest_sha256"],
            protocol_version=record["protocol_version"],
            registered_at_utc=record["registered_at_utc"],
            registry_identity=record["registry_identity"],
            registry_uri=record["registry_uri"],
            schema_version=record["schema_version"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "protocol_version": self.protocol_version,
            "registered_at_utc": self.registered_at_utc,
            "registry_identity": self.registry_identity,
            "registry_uri": self.registry_uri,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def record_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes() + b"\n").hexdigest()


@dataclass(frozen=True)
class ProtocolRegistrationReceipt:
    """Local pointer to an independently administered protocol registration.

    The external record must expose the exact frozen-manifest digest. The local
    receipt binds that record's bytes and public URI; it is not itself treated as
    evidence of registration.
    """

    manifest_sha256: str
    protocol_version: str
    registered_at_utc: str
    registry_identity: str
    registry_uri: str
    registry_record_sha256: str
    schema_version: str = PROTOCOL_REGISTRATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.manifest_sha256) is None:
            raise StudyManifestError(
                "registration manifest_sha256 must be a lowercase SHA-256"
            )
        if self.protocol_version != "0.3.0":
            raise StudyManifestError(
                "registration protocol_version must equal '0.3.0'"
            )
        _parse_utc(self.registered_at_utc, field="registered_at_utc")
        if (
            not isinstance(self.registry_identity, str)
            or not self.registry_identity.strip()
            or self.registry_identity != self.registry_identity.strip()
        ):
            raise StudyManifestError(
                "registry_identity must be a canonical non-empty string"
            )
        _validate_external_registry_uri(self.registry_uri)
        if _SHA256.fullmatch(self.registry_record_sha256) is None:
            raise StudyManifestError(
                "registry_record_sha256 must be a lowercase SHA-256"
            )
        if self.schema_version != PROTOCOL_REGISTRATION_RECEIPT_SCHEMA:
            raise StudyManifestError(
                "registration schema_version must equal "
                f"{PROTOCOL_REGISTRATION_RECEIPT_SCHEMA!r}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProtocolRegistrationReceipt:
        registration = _closed_object(
            payload,
            _PROTOCOL_REGISTRATION_FIELDS,
            path="protocol registration receipt",
        )
        return cls(
            manifest_sha256=registration["manifest_sha256"],
            protocol_version=registration["protocol_version"],
            registered_at_utc=registration["registered_at_utc"],
            registry_identity=registration["registry_identity"],
            registry_uri=registration["registry_uri"],
            registry_record_sha256=registration["registry_record_sha256"],
            schema_version=registration["schema_version"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "protocol_version": self.protocol_version,
            "registered_at_utc": self.registered_at_utc,
            "registry_identity": self.registry_identity,
            "registry_record_sha256": self.registry_record_sha256,
            "registry_uri": self.registry_uri,
            "schema_version": self.schema_version,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def load_protocol_registration_receipt(
    path: str | Path,
) -> ProtocolRegistrationReceipt:
    receipt_path = Path(path)
    try:
        encoded = read_secure_control_file(
            receipt_path,
            label="protocol registration receipt",
        )
        payload = _decode_json(encoded, label="protocol registration receipt")
    except (ArtifactIntegrityError, StudyManifestError) as exc:
        raise StudyManifestError(
            f"cannot load protocol registration receipt {receipt_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise StudyManifestError("protocol registration receipt must be an object")
    receipt = ProtocolRegistrationReceipt.from_dict(payload)
    if encoded != receipt.canonical_bytes() + b"\n":
        raise StudyManifestError(
            "protocol registration receipt bytes must equal canonical JSON plus one newline"
        )
    return receipt


def load_protocol_registry_record(path: str | Path) -> ProtocolRegistryRecord:
    record_path = Path(path)
    try:
        encoded = read_secure_control_file(
            record_path,
            label="protocol registry record",
        )
        payload = _decode_json(encoded, label="protocol registry record")
    except (ArtifactIntegrityError, StudyManifestError) as exc:
        raise StudyManifestError(
            f"cannot load protocol registry record {record_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise StudyManifestError("protocol registry record must be an object")
    record = ProtocolRegistryRecord.from_dict(payload)
    if encoded != record.canonical_bytes() + b"\n":
        raise StudyManifestError(
            "protocol registry record bytes must equal canonical JSON plus one newline"
        )
    return record


@dataclass(frozen=True)
class SealedRunReceipt:
    manifest_sha256: str
    protocol_version: str
    started_at_utc: str
    runner_identity: str
    code_commit: str
    runner_image: str
    protocol_registration_receipt_uri: str
    protocol_registration_receipt_sha256: str
    protocol_registration_record_uri: str
    verification_receipt_uri: str
    verification_receipt_sha256: str
    receipt_uri: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.manifest_sha256) is None:
            raise StudyManifestError("receipt manifest_sha256 must be a lowercase SHA-256")
        if self.protocol_version != "0.3.0":
            raise StudyManifestError("receipt protocol_version must equal '0.3.0'")
        _parse_utc(self.started_at_utc, field="receipt started_at_utc")
        if not isinstance(self.runner_identity, str) or not self.runner_identity.strip():
            raise StudyManifestError("receipt runner_identity must be non-empty")
        if _GIT_COMMIT.fullmatch(self.code_commit) is None:
            raise StudyManifestError("receipt code_commit must be a full lowercase Git commit")
        if _OCI_DIGEST.fullmatch(self.runner_image) is None:
            raise StudyManifestError("receipt runner_image must use an OCI SHA-256 digest")
        registration_uri = urlsplit(self.protocol_registration_receipt_uri)
        if (
            registration_uri.scheme != "file"
            or registration_uri.netloc not in {"", "localhost"}
            or not Path(unquote(registration_uri.path)).is_absolute()
            or registration_uri.query
            or registration_uri.fragment
        ):
            raise StudyManifestError(
                "protocol_registration_receipt_uri must be an absolute file URI"
            )
        if _SHA256.fullmatch(self.protocol_registration_receipt_sha256) is None:
            raise StudyManifestError(
                "protocol_registration_receipt_sha256 must be a lowercase SHA-256"
            )
        registration_record_uri = urlsplit(self.protocol_registration_record_uri)
        if (
            registration_record_uri.scheme != "file"
            or registration_record_uri.netloc not in {"", "localhost"}
            or not Path(unquote(registration_record_uri.path)).is_absolute()
            or registration_record_uri.query
            or registration_record_uri.fragment
        ):
            raise StudyManifestError(
                "protocol_registration_record_uri must be an absolute file URI"
            )
        verification_uri = urlsplit(self.verification_receipt_uri)
        if (
            verification_uri.scheme != "file"
            or verification_uri.netloc not in {"", "localhost"}
            or not Path(unquote(verification_uri.path)).is_absolute()
            or verification_uri.query
            or verification_uri.fragment
        ):
            raise StudyManifestError(
                "verification_receipt_uri must be an absolute file URI"
            )
        if _SHA256.fullmatch(self.verification_receipt_sha256) is None:
            raise StudyManifestError(
                "verification_receipt_sha256 must be a lowercase SHA-256"
            )
        parsed = urlsplit(self.receipt_uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise StudyManifestError("receipt_uri must be an absolute file URI")

    def to_dict(self) -> dict[str, str]:
        return {
            "code_commit": self.code_commit,
            "manifest_sha256": self.manifest_sha256,
            "protocol_version": self.protocol_version,
            "protocol_registration_receipt_sha256": (
                self.protocol_registration_receipt_sha256
            ),
            "protocol_registration_receipt_uri": (
                self.protocol_registration_receipt_uri
            ),
            "protocol_registration_record_uri": self.protocol_registration_record_uri,
            "receipt_uri": self.receipt_uri,
            "runner_identity": self.runner_identity,
            "runner_image": self.runner_image,
            "started_at_utc": self.started_at_utc,
            "verification_receipt_sha256": self.verification_receipt_sha256,
            "verification_receipt_uri": self.verification_receipt_uri,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SealedRunReceipt:
        receipt = _closed_object(
            payload,
            _SEALED_RUN_RECEIPT_FIELDS,
            path="sealed run receipt",
        )
        return cls(
            manifest_sha256=receipt["manifest_sha256"],
            protocol_version=receipt["protocol_version"],
            started_at_utc=receipt["started_at_utc"],
            runner_identity=receipt["runner_identity"],
            code_commit=receipt["code_commit"],
            runner_image=receipt["runner_image"],
            protocol_registration_receipt_uri=receipt[
                "protocol_registration_receipt_uri"
            ],
            protocol_registration_receipt_sha256=receipt[
                "protocol_registration_receipt_sha256"
            ],
            protocol_registration_record_uri=receipt[
                "protocol_registration_record_uri"
            ],
            verification_receipt_uri=receipt["verification_receipt_uri"],
            verification_receipt_sha256=receipt[
                "verification_receipt_sha256"
            ],
            receipt_uri=receipt["receipt_uri"],
        )


def load_sealed_run_receipt(path: str | Path) -> SealedRunReceipt:
    """Load one canonical run receipt from its manifest-derived path."""

    target = Path(path)
    try:
        encoded = read_secure_control_file(target, label="sealed run receipt")
        payload = _decode_json(encoded, label="sealed run receipt")
    except (ArtifactIntegrityError, StudyManifestError) as exc:
        raise StudyManifestError(f"cannot load sealed run receipt: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise StudyManifestError("sealed run receipt must contain one object")
    receipt = SealedRunReceipt.from_dict(payload)
    if encoded != receipt.canonical_bytes() + b"\n":
        raise StudyManifestError(
            "sealed run receipt bytes must equal canonical JSON plus one newline"
        )
    if not target.is_absolute() or target.as_uri() != receipt.receipt_uri:
        raise StudyManifestError(
            "sealed run receipt is not at its manifest-derived receipt_uri"
        )
    return receipt


def begin_sealed_run(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    runner_identity: str,
    artifact_verification_receipt_path: str | Path,
    artifact_root: str | Path | None = None,
    local_artifact_map_path: str | Path | None = None,
    protocol_registration_receipt_path: str | Path,
    protocol_registration_record_path: str | Path,
    trusted_registry_record_fetcher: Callable[[str, int], bytes] | None = None,
) -> SealedRunReceipt:
    """Revalidate frozen controls and atomically create the one-shot run receipt.

    The default path performs a fresh, certificate-validated HTTPS fetch. The
    injectable fetcher is a trusted test/integration seam, not evidence of an
    independently administered registration. The production CLI never injects it.
    Returned bytes remain subject to the same size, digest, and byte-equality checks.
    """
    if trusted_registry_record_fetcher is not None and not callable(
        trusted_registry_record_fetcher
    ):
        raise StudyManifestError(
            "trusted_registry_record_fetcher must be callable or None"
        )
    if (artifact_root is None) != (local_artifact_map_path is None):
        raise StudyManifestError(
            "artifact_root and local_artifact_map_path must be supplied together"
        )
    if (
        artifact_root is None
        and local_artifact_map_path is None
        and trusted_registry_record_fetcher is None
    ):
        raise StudyManifestError(
            "production sealed execution requires a fresh local artifact revalidation"
        )
    try:
        manifest_bytes = read_secure_control_file(
            manifest_path,
            label="frozen study manifest",
        )
        payload = _decode_json(manifest_bytes, label="frozen study manifest")
    except (ArtifactIntegrityError, StudyManifestError) as exc:
        raise StudyManifestError(f"cannot load frozen study manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise StudyManifestError("frozen study manifest root must be a JSON object")
    validate_study_manifest(payload, require_frozen=True)
    sealed = payload["sealed_execution"]
    pinned_identity = sealed["runner_identity"]
    if runner_identity != pinned_identity:
        raise StudyManifestError(
            "runner_identity does not equal sealed_execution.runner_identity"
        )

    digest = manifest_sha256(payload)
    try:
        expected = read_secure_control_file(
            lock_path,
            label="manifest lock",
        ).decode("ascii").strip()
    except (ArtifactIntegrityError, UnicodeDecodeError) as exc:
        raise StudyManifestError(f"cannot read manifest lock: {exc}") from exc
    if _SHA256.fullmatch(expected) is None:
        raise StudyManifestError("manifest lock must contain one lowercase SHA-256")
    if digest != expected:
        raise StudyManifestError("manifest digest does not match the frozen lock")

    registration_receipt = load_protocol_registration_receipt(
        protocol_registration_receipt_path
    )
    if registration_receipt.manifest_sha256 != digest:
        raise StudyManifestError(
            "protocol registration receipt is bound to a different manifest digest"
        )
    if registration_receipt.protocol_version != payload["protocol_version"]:
        raise StudyManifestError(
            "protocol registration receipt has a different protocol version"
        )
    registered_at = _parse_utc(
        registration_receipt.registered_at_utc,
        field="registered_at_utc",
    )
    if registered_at > datetime.now(timezone.utc):
        raise StudyManifestError("protocol registration timestamp cannot be in the future")
    registry_record = load_protocol_registry_record(
        protocol_registration_record_path
    )
    if registry_record.record_sha256 != registration_receipt.registry_record_sha256:
        raise StudyManifestError(
            "protocol registration record digest does not match its receipt"
        )
    for field in (
        "manifest_sha256",
        "protocol_version",
        "registered_at_utc",
        "registry_identity",
        "registry_uri",
    ):
        if getattr(registry_record, field) != getattr(registration_receipt, field):
            raise StudyManifestError(
                f"protocol registration record {field} does not match its receipt"
            )

    try:
        verification_receipt = load_verification_receipt(
            artifact_verification_receipt_path
        )
    except ArtifactIntegrityError as exc:
        raise StudyManifestError(
            f"invalid artifact verification receipt: {exc}"
        ) from exc
    _validate_artifact_verification_receipt(
        verification_receipt,
        payload=payload,
        manifest_digest=digest,
    )
    if artifact_root is not None and local_artifact_map_path is not None:
        pins = {
            str(artifact["id"]): str(artifact["sha256"])
            for artifact in payload["artifacts"]
        }
        try:
            local_specs = load_local_artifact_map(
                local_artifact_map_path,
                expected_sha256_by_id=pins,
            )
            fresh_verification_receipt = verify_local_artifacts(
                artifact_root,
                manifest_sha256=digest,
                artifacts=local_specs,
            )
        except ArtifactIntegrityError as exc:
            raise StudyManifestError(
                f"fresh local artifact revalidation failed: {exc}"
            ) from exc
        if not hmac.compare_digest(
            fresh_verification_receipt.canonical_bytes(),
            verification_receipt.canonical_bytes(),
        ):
            raise StudyManifestError(
                "fresh local artifact revalidation differs from the admitted receipt"
            )

    local_registry_record_bytes = registry_record.canonical_bytes() + b"\n"
    registry_fetcher = (
        _fetch_protocol_registry_record
        if trusted_registry_record_fetcher is None
        else trusted_registry_record_fetcher
    )
    try:
        fetched_registry_record_bytes = registry_fetcher(
            registration_receipt.registry_uri,
            MAX_PROTOCOL_REGISTRY_RECORD_BYTES,
        )
    except StudyManifestError:
        raise
    except Exception as exc:
        raise StudyManifestError(
            "trusted protocol registry fetcher failed during revalidation"
        ) from exc
    if not isinstance(fetched_registry_record_bytes, bytes):
        raise StudyManifestError("protocol registry fetcher must return bytes")
    if len(fetched_registry_record_bytes) > MAX_PROTOCOL_REGISTRY_RECORD_BYTES:
        raise StudyManifestError(
            "protocol registry record exceeds the maximum byte limit"
        )
    fetched_registry_record_sha256 = hashlib.sha256(
        fetched_registry_record_bytes
    ).hexdigest()
    if not hmac.compare_digest(
        fetched_registry_record_sha256,
        registration_receipt.registry_record_sha256,
    ):
        raise StudyManifestError(
            "fetched protocol registry record digest does not match its receipt"
        )
    if not hmac.compare_digest(
        fetched_registry_record_bytes,
        local_registry_record_bytes,
    ):
        raise StudyManifestError(
            "fetched protocol registry record bytes do not match the secure local record"
        )

    receipt_uri = sealed_receipt_uri(payload)
    target = _receipt_path_from_uri(receipt_uri)
    verification_receipt_path = Path(artifact_verification_receipt_path)
    registration_receipt_path = Path(protocol_registration_receipt_path)
    registration_record_path = Path(protocol_registration_record_path)
    receipt = SealedRunReceipt(
        manifest_sha256=digest,
        protocol_version=str(payload["protocol_version"]),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        runner_identity=runner_identity,
        code_commit=str(sealed["code_commit"]),
        runner_image=str(sealed["runner_image"]),
        protocol_registration_receipt_uri=(
            registration_receipt_path.as_uri()
        ),
        protocol_registration_receipt_sha256=(
            registration_receipt.receipt_sha256
        ),
        protocol_registration_record_uri=(
            registration_record_path.as_uri()
        ),
        verification_receipt_uri=verification_receipt_path.as_uri(),
        verification_receipt_sha256=verification_receipt.receipt_sha256,
        receipt_uri=receipt_uri,
    )
    try:
        write_exclusive_receipt_bytes(receipt.canonical_bytes() + b"\n", target)
    except ArtifactIntegrityError as exc:
        if "already exists" in str(exc):
            raise StudyManifestError(
                f"sealed run receipt already exists at {receipt_uri}; "
                "one-shot execution has already been consumed and reserve_fraction is 0.0, "
                "so no rerun or rescue is permitted"
            ) from exc
        raise StudyManifestError(f"cannot write sealed run receipt safely: {exc}") from exc
    return receipt
