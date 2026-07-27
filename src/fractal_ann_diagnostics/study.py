"""Closed manifest schema and one-shot controls for the confirmatory study."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
import ssl
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import unquote, urlsplit

from .artifact_integrity import (
    ArtifactIntegrityError,
    ArtifactVerificationReceipt,
    digest_regular_file,
    load_local_artifact_map,
    load_verification_receipt,
    read_secure_control_file,
    verify_local_artifacts,
    write_exclusive_receipt_bytes,
)
from .c0_evidence_release import (
    C0EvidenceReleaseError,
    validate_c0_evidence_release_binding,
)
from .production_workload_registration import (
    FIXED_PRODUCTION_CORPORA,
    ProductionWorkloadRegistrationError,
    production_workload_file_sha256,
    validate_production_workload_registrations,
)
from .provider_contract import (
    DOCKER_SERVER_PROBE_FIELDS,
    DOCKER_SERVER_PROBE_SCHEMA,
    OFFICIAL_ACTIONS_RUNNER_CONFIG_SHA256,
    OFFICIAL_ACTIONS_RUNNER_LISTENER_DLL_SHA256,
    OFFICIAL_ACTIONS_RUNNER_LISTENER_SHA256,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
    OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI,
    OFFICIAL_ACTIONS_RUNNER_RUN_SHA256,
    OFFICIAL_ACTIONS_RUNNER_VERSION,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256,
    OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI,
    OFFICIAL_GH_OSX_ARM64_BINARY_SHA256,
    OFFICIAL_GH_VERSION,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256,
    OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI,
    OFFICIAL_PYTHON_BUILD_STANDALONE_BINARY_SHA256,
    OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION,
    PHASE_HOST_PROBE_FIELDS,
    PHASE_HOST_PROBE_SCHEMA,
    PHASE_HOST_TOOL_CONTRACT_FIELDS,
    PHASE_HOST_TOOL_CONTRACT_SCHEMA,
    REGISTERED_DOCKER_CLIENT_BUILD,
    REGISTERED_DOCKER_CLIENT_SHA256,
    REGISTERED_DOCKER_CLIENT_VERSION,
    SOURCE_BUILT_LINUX_ARM64_TLE_SHA256,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA256_REVISION = re.compile(r"^sha256:([0-9a-f]{64})$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OCI_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_PLACEHOLDERS = {"", "tbd", "todo", "latest", "main", "master", "unassigned"}
_RECEIPT_DIGEST_TOKEN = "{manifest_sha256}"
PROTOCOL_REGISTRATION_RECEIPT_SCHEMA = "fractal-ann-protocol-registration-v1"
PROTOCOL_REGISTRY_RECORD_SCHEMA = "fractal-ann-protocol-registry-record-v1"
SEALED_RUN_RECEIPT_BINDING_SCHEMA = "fractal-sealed-run-receipt-binding-v1"
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
_C1_REGISTRATION_CAPABILITY = object()
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

FIXED_CORPORA = FIXED_PRODUCTION_CORPORA
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
REGISTERED_POWER_REQUIRED_SCENARIO_COUNT = 2
REGISTERED_POWER_SELECTION_FAMILY_SIZE = 12
REGISTERED_POWER_SELECTION_FAMILYWISE_CONFIDENCE = 0.95
REGISTERED_POWER_SELECTION_CELL_ALPHA = 0.05 / REGISTERED_POWER_SELECTION_FAMILY_SIZE
REGISTERED_POWER_SELECTION_EXACT_QUALIFYING_PASSES = 4_556
REGISTERED_POWER_SELECTION_EXACT_BLOCKING_FAILURES = 445
REGISTERED_POWER_SELECTION_MULTIPLICITY_METHOD = (
    "bonferroni-fixed-required-scenario-candidate-grid-v1"
)

_ROOT_FIELDS = {
    "analysis",
    "artifacts",
    "claim_scope",
    "freeze_blockers",
    "primary_claim",
    "protocol_version",
    "production_workloads",
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
    "selection_cell_alpha",
    "selection_exact_blocking_failures",
    "selection_exact_qualifying_passes",
    "selection_family_size",
    "selection_familywise_confidence",
    "selection_multiplicity_method",
    "selected_families_per_corpus",
    "selected_joint_power_lower_bound",
    "simulation_count",
    "simulation_seed",
}
_SEALED_EXECUTION_FIELDS = {
    "approval_environment",
    "c0_evidence_release",
    "code_commit",
    "custodian",
    "hardware",
    "interactive_access",
    "label_artifacts_withheld_until_prediction_receipt",
    "public_query_reidentification_risk",
    "production_controls",
    "provider_phase_plans",
    "receipt_uri_template",
    "reserve_fraction",
    "results_store",
    "runner_identity",
    "runner_image",
    "runner_network_access",
}
PROVIDER_PHASE_PLAN_TEMPLATE_SCHEMA = "fractal-provider-phase-plan-template-v2"
PROVIDER_APPROVAL_ENVIRONMENT = "confirmatory"
PROVIDER_RUNNER_IDENTITY = f"github-actions:environment:{PROVIDER_APPROVAL_ENVIRONMENT}"
PROVIDER_PLAN_MANIFEST_BINDING = "enclosing-canonical-study-manifest-sha256"
PROVIDER_PLAN_C1_COMMIT_BINDING = "containing-confirmatory-freeze-c1-commit"
PROVIDER_PLAN_SUITE_BINDING = "sha256-fractal-suite-attempt-from-enclosing-manifest"
PROVIDER_PLAN_PREDECESSOR_BINDING = "protected-provider-ledger-tip-for-suite-attempt"
PROVIDER_PLAN_CLAIM_RECEIPT_BINDING = "provider-claim-receipt-for-suite-attempt"
PROVIDER_PLAN_PHASE_INPUT_BINDING = "verified-predecessor-phase-inputs"
PROVIDER_PLAN_PHASE_OUTPUT_BINDING = "fixed-phase-output-namespace-for-suite-attempt"
C0_COMMIT_SENTINEL = "containing-confirmatory-apparatus-c0-commit"
PROVIDER_PHASES = ("online", "label-release", "analysis")
PROVIDER_PHASE_JOB_NAMES = {
    "online": ("claim-online", "execute-online"),
    "label-release": ("claim-label-release", "release-labels"),
    "analysis": ("claim-analysis", "run-analysis"),
}
PROVIDER_PHASE_RUNTIME_BINDINGS = {
    "online": ("linux/arm64", "scientific", "main"),
    "label-release": ("linux/arm64", "timelock-release", "release"),
    "analysis": ("linux/amd64", "scientific", "main"),
}
PROVIDER_PHASE_RUNTIME_CEILINGS = {
    "online": 72_000,
    "label-release": 21_600,
    "analysis": 43_200,
}
PROVIDER_PHASE_WORKFLOWS = {
    "online": ".github/workflows/confirmatory-online-execution.yml",
    "label-release": ".github/workflows/confirmatory-label-release.yml",
    "analysis": ".github/workflows/confirmatory-analysis.yml",
}
PROVIDER_PHASE_COMMAND_IDS = {
    "online": "execute-five-frozen-online-corpora-v1",
    "label-release": "release-five-timelocked-label-payloads-v1",
    "analysis": "analyze-five-confirmatory-corpora-v1",
}
C0_COMMIT_SENTINEL_PATHS = frozenset(
    {
        "sealed_execution.code_commit",
        "artifacts[role=source-code].revision",
        *(
            f"production_workloads[corpus_id={corpus_id}].spec.code_commit"
            for corpus_id in FIXED_CORPORA
        ),
        *(
            f"sealed_execution.provider_phase_plans.{phase}.workflow_sha"
            for phase in PROVIDER_PHASES
        ),
        *(
            f"sealed_execution.provider_phase_plans.{phase}.runner_bootstrap_receipt.workflow_sha"
            for phase in PROVIDER_PHASES
        ),
    }
)
_PROVIDER_RUNNER_LABEL_DERIVATION = "sha256-fractal-phase-runner-label-v1"


def _derive_provider_runner_label(claim_nonce: str, phase: str) -> str:
    suffix = hashlib.sha256(
        _PROVIDER_RUNNER_LABEL_DERIVATION.encode("ascii")
        + b"\0"
        + phase.encode("ascii")
        + b"\0"
        + bytes.fromhex(claim_nonce)
    ).hexdigest()[:24]
    return f"fractal-ann-confirmatory-{phase}-{suffix}"


EXECUTION_CLAIM_INPUT_FIELDS = frozenset(
    {
        "beacon",
        "design_seed_sha256",
        "registered_online_runtime_budget_seconds",
    }
)
EXECUTION_BEACON_CONTRACT_FIELDS = frozenset(
    {
        "chain_genesis_unix_seconds",
        "chain_hash",
        "chain_period_seconds",
        "chain_public_key",
        "chain_scheme_id",
        "drand_network",
        "execution_round",
        "label_release_round",
        "minimum_label_release_safety_rounds",
        "schema_version",
        "seed_derivation",
        "verification_identity",
    }
)
PROVIDER_RUNNER_BOOTSTRAP_FIELDS = frozenset(
    {
        "approval_environment",
        "disable_update",
        "ephemeral",
        "phase",
        "registered_at_utc",
        "repository",
        "repository_runner_inventory_sha256",
        "runner_archive_sha256",
        "runner_group_id",
        "runner_id",
        "runner_identity",
        "runner_label",
        "runner_name",
        "runner_version",
        "schema_version",
        "unattended",
        "workflow_sha",
    }
)
PROVIDER_PHASE_PLAN_TEMPLATE_FIELDS = frozenset(
    {
        "activation_argv_template",
        "activation_command_id",
        "approval_environment",
        "c1_commit_binding",
        "claim_job_name",
        "claim_nonce",
        "claim_predecessor_binding",
        "claim_receipt_path_template",
        "execute_job_name",
        "execution_claim_inputs",
        "host_tools",
        "manifest_sha256_binding",
        "maximum_runtime_seconds",
        "oci_index_digest",
        "oci_platform_manifest_digest",
        "phase",
        "phase_evidence_root_template",
        "phase_input_binding",
        "phase_output_binding",
        "provider_architecture",
        "provider_operating_system",
        "provider_plan_path",
        "repository",
        "run_head_branch",
        "runner_archive_sha256",
        "runner_bootstrap_receipt",
        "runner_bootstrap_receipt_file_sha256",
        "runner_bootstrap_receipt_path",
        "runner_group_id",
        "runner_id",
        "runner_identity",
        "runner_name",
        "runner_registration_bundle_path",
        "runner_registration_bundle_sha256",
        "runner_registration_evidence_file_sha256",
        "runner_version",
        "runtime_image",
        "runtime_image_role",
        "runtime_index_role",
        "runtime_platform",
        "runtime_probe_receipt_sha256",
        "schema_version",
        "suite_attempt_id_binding",
        "tle_binary_sha256",
        "tle_build_provenance_sha256",
        "tle_interoperability_receipt_sha256",
        "tle_vulnerability_scan_sha256",
        "workflow_path",
        "workflow_ref",
        "workflow_sha",
    }
)
_PRODUCTION_CONTROL_BINDING_FIELDS = {
    "blueprint_receipt_file_sha256",
    "blueprint_receipt_sha256",
    "materialization_config_file_sha256",
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
    "authorized-index-store",
    "corpus-normalizer",
    "embedding-store",
    "online-execution",
    "policy-workload",
    "runtime-attestation-plan-template",
    "sealed-inputs",
    "sealed-label-ciphertext",
    "sealed-labels",
    "timelock-encryption-receipt",
    "trial-runtime-package",
}
_ARTIFACT_ROLE_SPECS: dict[str, tuple[str, int]] = {
    "sealed-inputs": ("dataset", len(FIXED_CORPORA)),
    "sealed-labels": ("dataset", len(FIXED_CORPORA)),
    "sealed-label-ciphertext": ("encrypted-dataset", len(FIXED_CORPORA)),
    "timelock-encryption-receipt": ("custody", len(FIXED_CORPORA)),
    "online-execution": ("execution", len(FIXED_CORPORA)),
    "corpus-normalizer": ("normalizer", len(FIXED_CORPORA)),
    "policy-workload": ("policy-data", len(FIXED_CORPORA)),
    "embedding-store": ("embedding-store", len(FIXED_CORPORA)),
    "authorized-index-store": ("index-store", len(FIXED_CORPORA)),
    "trial-runtime-package": ("runtime-input", len(FIXED_CORPORA)),
    "study-data-package": ("dataset-package", 1),
    "online-staging-package": ("dataset-package", 1),
    "development-freeze-package": ("development-package", 1),
    "development-fit-data": ("dataset", 1),
    "development-calibration-data": ("dataset", 1),
    "query-partition-audit": ("partition-audit", 1),
    "primary-embedding": ("embedding", 1),
    "stale-embedding": ("embedding", 1),
    "exact-authorized-oracle": ("backend", 1),
    "strict-authorized-hnsw": ("backend", 1),
    "opa-pdp": ("policy", 1),
    "opa-runtime-binary": ("tool", 1),
    "frozen-controller": ("controller", 1),
    "static-comparator": ("comparator", 1),
    "h1-predictive-model": ("model", 1),
    "h2-model-suite": ("model", 1),
    "power-analysis-report": ("analysis", 1),
    "analysis-runner": ("analysis", 1),
    "source-code": ("source", 1),
    "custody-seal-receipt": ("custody", 1),
    "tlock-release-provenance": ("custody", 1),
    "timelock-tool": ("tool", 1),
    "custody-builder": ("execution", 1),
    "runtime-attestation-plan-template": ("execution", len(FIXED_CORPORA)),
    "suite-attestation-descriptor": ("attestation", 1),
}


class StudyManifestError(ValueError):
    """Raised when a study manifest is incomplete or internally inconsistent."""


def revision_sha256(value: object, *, field: str = "artifact revision") -> str:
    """Return the logical digest carried by one immutable SHA-256 revision."""

    if not isinstance(value, str):
        raise StudyManifestError(f"{field} must equal 'sha256:<logical-digest>'")
    matched = _SHA256_REVISION.fullmatch(value)
    if matched is None:
        raise StudyManifestError(f"{field} must equal 'sha256:<logical-digest>'")
    return matched.group(1)


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
        raise StudyManifestError(f"{path} schema mismatch; missing={missing}, unknown={unknown}")
    return value


def _is_placeholder(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS


def _candidate_c0_commit_slots(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, str, dict[str, Any], str], ...]:
    """Locate the only fields allowed to carry the pre-C0 commit sentinel."""

    root = _closed_object(payload, _ROOT_FIELDS, path="study manifest")
    slots: list[tuple[str, str, dict[str, Any], str]] = []

    sealed_value = root["sealed_execution"]
    if not isinstance(sealed_value, dict):
        raise StudyManifestError("sealed_execution must be an object")
    slots.append(
        (
            "sealed_execution.code_commit",
            "sealed_execution.code_commit",
            sealed_value,
            "code_commit",
        )
    )

    artifacts = root["artifacts"]
    if not isinstance(artifacts, list):
        raise StudyManifestError("artifacts must be an array")
    source_rows = [
        (position, row)
        for position, row in enumerate(artifacts)
        if isinstance(row, dict) and row.get("role") == "source-code"
    ]
    if len(source_rows) != 1:
        raise StudyManifestError(
            "candidate C0 sentinel registration requires exactly one source-code artifact"
        )
    source_position, source_row = source_rows[0]
    slots.append(
        (
            "artifacts[role=source-code].revision",
            f"artifacts[{source_position}].revision",
            source_row,
            "revision",
        )
    )

    workloads = root["production_workloads"]
    if not isinstance(workloads, list) or len(workloads) != len(FIXED_CORPORA):
        raise StudyManifestError(
            "candidate C0 sentinel registration requires five production workloads"
        )
    for position, corpus_id in enumerate(FIXED_CORPORA):
        row = workloads[position]
        if not isinstance(row, dict) or row.get("corpus_id") != corpus_id:
            raise StudyManifestError(
                "candidate C0 sentinel registration requires workloads in fixed corpus order"
            )
        spec = row.get("spec")
        if not isinstance(spec, dict) or spec.get("corpus_id") != corpus_id:
            raise StudyManifestError(
                f"candidate C0 sentinel registration lacks the {corpus_id!r} WorkloadSpec"
            )
        slots.append(
            (
                f"production_workloads[corpus_id={corpus_id}].spec.code_commit",
                f"production_workloads[{position}].spec.code_commit",
                spec,
                "code_commit",
            )
        )

    plans = sealed_value.get("provider_phase_plans")
    if not isinstance(plans, dict) or set(plans) != set(PROVIDER_PHASES):
        raise StudyManifestError(
            "candidate C0 sentinel registration requires exactly three provider plans"
        )
    for phase in PROVIDER_PHASES:
        plan = plans[phase]
        if not isinstance(plan, dict):
            raise StudyManifestError(f"candidate {phase} provider plan must be an object")
        slots.append(
            (
                f"sealed_execution.provider_phase_plans.{phase}.workflow_sha",
                f"sealed_execution.provider_phase_plans.{phase}.workflow_sha",
                plan,
                "workflow_sha",
            )
        )
        bootstrap = plan.get("runner_bootstrap_receipt")
        if not isinstance(bootstrap, dict):
            raise StudyManifestError(
                f"candidate {phase} provider plan lacks its runner bootstrap receipt"
            )
        slots.append(
            (
                "sealed_execution.provider_phase_plans."
                f"{phase}.runner_bootstrap_receipt.workflow_sha",
                "sealed_execution.provider_phase_plans."
                f"{phase}.runner_bootstrap_receipt.workflow_sha",
                bootstrap,
                "workflow_sha",
            )
        )

    registered = {label for label, _, _, _ in slots}
    if registered != set(C0_COMMIT_SENTINEL_PATHS):  # pragma: no cover - module invariant
        raise RuntimeError("candidate C0 sentinel path registry drifted")
    return tuple(slots)


def _find_scalar_value_paths(value: object, target: str, *, path: str = "") -> set[str]:
    observed: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            observed.update(_find_scalar_value_paths(item, target, path=child))
    elif isinstance(value, list):
        for position, item in enumerate(value):
            observed.update(_find_scalar_value_paths(item, target, path=f"{path}[{position}]"))
    elif value == target:
        observed.add(path)
    return observed


def _validate_candidate_c0_commit_sentinel_paths(payload: Mapping[str, Any]) -> None:
    slots = _candidate_c0_commit_slots(payload)
    expected = {concrete for _, concrete, _, _ in slots}
    observed = _find_scalar_value_paths(payload, C0_COMMIT_SENTINEL)
    if observed != expected:
        raise StudyManifestError(
            "candidate C0 commit sentinel path set differs; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def resolve_candidate_c0_commit_sentinels(
    payload: Mapping[str, Any],
    *,
    c0_commit: str,
) -> dict[str, Any]:
    """Resolve the exact registered pre-C0 fields and no other value."""

    if not isinstance(c0_commit, str) or _GIT_COMMIT.fullmatch(c0_commit) is None:
        raise StudyManifestError("candidate C0 commit must be one full lowercase Git commit")
    resolved = copy.deepcopy(dict(payload))
    _validate_candidate_c0_commit_sentinel_paths(resolved)
    for _, _, container, key in _candidate_c0_commit_slots(resolved):
        container[key] = c0_commit
    if _find_scalar_value_paths(resolved, C0_COMMIT_SENTINEL):  # pragma: no cover - invariant
        raise RuntimeError("candidate C0 sentinel survived exact resolution")
    return resolved


def _validate_candidate_bootstrap_template_digests(
    payload: Mapping[str, Any],
) -> None:
    sealed = payload.get("sealed_execution")
    plans = sealed.get("provider_phase_plans") if isinstance(sealed, Mapping) else None
    if not isinstance(plans, Mapping) or set(plans) != set(PROVIDER_PHASES):
        raise StudyManifestError(
            "candidate bootstrap template validation requires exactly three provider plans"
        )
    for phase in PROVIDER_PHASES:
        plan = plans[phase]
        bootstrap = plan.get("runner_bootstrap_receipt") if isinstance(plan, Mapping) else None
        if not isinstance(bootstrap, Mapping):
            raise StudyManifestError(
                f"candidate {phase} provider plan lacks its bootstrap template"
            )
        expected = hashlib.sha256(_canonical_bytes(bootstrap) + b"\n").hexdigest()
        if plan.get("runner_bootstrap_receipt_file_sha256") != expected:
            raise StudyManifestError(
                f"candidate {phase} bootstrap template digest differs from raw sentinel bytes"
            )


def _validate_candidate_workload_template_digests(
    payload: Mapping[str, Any],
) -> None:
    workloads = payload.get("production_workloads")
    if not isinstance(workloads, list) or len(workloads) != len(FIXED_CORPORA):
        raise StudyManifestError(
            "candidate workload template validation requires the five fixed corpora"
        )
    for position, corpus_id in enumerate(FIXED_CORPORA):
        row = workloads[position]
        spec = row.get("spec") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or row.get("corpus_id") != corpus_id
            or not isinstance(spec, Mapping)
        ):
            raise StudyManifestError(
                f"candidate workload template {position} differs from {corpus_id}"
            )
        expected = production_workload_file_sha256(spec)
        if row.get("canonical_file_sha256") != expected:
            raise StudyManifestError(
                f"candidate {corpus_id} workload digest differs from raw sentinel bytes"
            )


def resolve_candidate_provider_plan_commit_bindings(
    payload: Mapping[str, Any],
    *,
    c0_commit: str,
) -> dict[str, Any]:
    """Resolve C0 sentinels and all eight consequent file digests."""

    _validate_candidate_bootstrap_template_digests(payload)
    _validate_candidate_workload_template_digests(payload)
    resolved = resolve_candidate_c0_commit_sentinels(payload, c0_commit=c0_commit)
    sealed = resolved["sealed_execution"]
    plans = sealed["provider_phase_plans"]
    for phase in PROVIDER_PHASES:
        plan = plans[phase]
        bootstrap = plan["runner_bootstrap_receipt"]
        plan["runner_bootstrap_receipt_file_sha256"] = hashlib.sha256(
            _canonical_bytes(bootstrap) + b"\n"
        ).hexdigest()
    workloads = resolved["production_workloads"]
    for row in workloads:
        row["canonical_file_sha256"] = production_workload_file_sha256(row["spec"])
    return resolved


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
        raise StudyManifestError("analysis.power.effect_scenarios must be a non-empty array")
    for position, scenario in enumerate(effect_scenarios):
        _pinned_text(
            scenario,
            path=f"analysis.power.effect_scenarios[{position}]",
            frozen=frozen,
        )
    if len(effect_scenarios) != len(set(effect_scenarios)):
        raise StudyManifestError("analysis.power.effect_scenarios cannot contain duplicates")
    if len(effect_scenarios) != REGISTERED_POWER_REQUIRED_SCENARIO_COUNT:
        raise StudyManifestError(
            "analysis.power.effect_scenarios must contain exactly two required scenarios"
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
    if power["selection_multiplicity_method"] != REGISTERED_POWER_SELECTION_MULTIPLICITY_METHOD:
        raise StudyManifestError(
            "analysis.power.selection_multiplicity_method must equal the registered "
            "Bonferroni grid method"
        )
    _registered_number(
        power,
        "selection_familywise_confidence",
        REGISTERED_POWER_SELECTION_FAMILYWISE_CONFIDENCE,
        path="analysis.power",
    )
    _registered_number(
        power,
        "selection_cell_alpha",
        REGISTERED_POWER_SELECTION_CELL_ALPHA,
        path="analysis.power",
    )
    family_size = _positive_integer(
        power["selection_family_size"],
        path="analysis.power.selection_family_size",
    )
    expected_family_size = len(candidates) * len(effect_scenarios)
    if family_size != expected_family_size or family_size != REGISTERED_POWER_SELECTION_FAMILY_SIZE:
        raise StudyManifestError(
            "analysis.power.selection_family_size must equal the fixed 12-cell grid"
        )
    qualifying_passes = _positive_integer(
        power["selection_exact_qualifying_passes"],
        path="analysis.power.selection_exact_qualifying_passes",
    )
    if qualifying_passes != REGISTERED_POWER_SELECTION_EXACT_QUALIFYING_PASSES:
        raise StudyManifestError("analysis.power.selection_exact_qualifying_passes must equal 4556")
    blocking_failures = _positive_integer(
        power["selection_exact_blocking_failures"],
        path="analysis.power.selection_exact_blocking_failures",
    )
    if blocking_failures != REGISTERED_POWER_SELECTION_EXACT_BLOCKING_FAILURES:
        raise StudyManifestError("analysis.power.selection_exact_blocking_failures must equal 445")
    if qualifying_passes + blocking_failures != power["simulation_count"] + 1:
        raise StudyManifestError(
            "analysis.power exact selection thresholds must partition simulation_count"
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
            "analysis.power.selected_joint_power_lower_bound must reach analysis.power_target"
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
    if analysis["bootstrap_seed"] != 20260713 or isinstance(analysis["bootstrap_seed"], bool):
        raise StudyManifestError("analysis.bootstrap_seed must equal 20260713")
    if (
        isinstance(analysis["maximum_entitlement_violations"], bool)
        or analysis["maximum_entitlement_violations"] != 0
    ):
        raise StudyManifestError("analysis.maximum_entitlement_violations must equal zero")
    if analysis["minimum_corpora_with_geometry_gain"] != 4:
        raise StudyManifestError("analysis.minimum_corpora_with_geometry_gain must equal four")
    if analysis["geometry_reference_model"] != "system-policy":
        raise StudyManifestError("analysis.geometry_reference_model must equal 'system-policy'")
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
                raise StudyManifestError(f"analysis.{profile_name} must be pinned before freeze")
            geometry_profiles[profile_name] = None
            continue
        if not isinstance(profile_value, Mapping) or not profile_value:
            raise StudyManifestError(
                f"analysis.{profile_name} must be a non-empty object or 'tbd' in a draft"
            )
        features: list[str] = []
        for feature, feature_value in profile_value.items():
            if not isinstance(feature, str) or not feature or feature != feature.strip():
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
        and geometry_profiles["low_geometry"] != geometry_profiles["high_geometry"]
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
    nested_rows = _draftable_integer(
        analysis["nested_rows_per_family"],
        path="analysis.nested_rows_per_family",
        frozen=frozen,
        minimum=1,
    )
    if nested_rows is not None and nested_rows != 3:
        raise StudyManifestError(
            "analysis.nested_rows_per_family must equal the registered value 3"
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
            raise StudyManifestError(f"frozen artifact {identifier!r} needs a pinned {field}")
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
    grouped: dict[str, list[Mapping[str, Any]]] = {role: [] for role in _ARTIFACT_ROLE_SPECS}
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
            raise StudyManifestError(f"artifact role {role!r} requires kind {expected_kind!r}")
        for field in ("uri", "revision", "sha256", "license"):
            if not isinstance(artifact[field], str) or not artifact[field].strip():
                raise StudyManifestError(
                    f"artifacts[{position}].{field} must be a non-empty string"
                )
        if not _is_placeholder(artifact["sha256"]):
            if _SHA256.fullmatch(artifact["sha256"]) is None:
                raise StudyManifestError(f"artifact {identifier!r} has an invalid SHA-256")
        if role == "online-execution" and not _is_placeholder(artifact["revision"]):
            logical_sha256 = revision_sha256(
                artifact["revision"],
                field=f"online-execution artifact {identifier!r} revision",
            )
            if not _is_placeholder(artifact["sha256"]) and logical_sha256 == artifact["sha256"]:
                raise StudyManifestError(
                    f"online-execution artifact {identifier!r} must distinguish "
                    "its logical plan revision from its outer directory-tree SHA-256"
                )
        if role == "opa-runtime-binary" and not _is_placeholder(artifact["revision"]):
            logical_sha256 = revision_sha256(
                artifact["revision"],
                field="opa-runtime-binary artifact revision",
            )
            if not _is_placeholder(artifact["sha256"]) and logical_sha256 != artifact["sha256"]:
                raise StudyManifestError(
                    "opa-runtime-binary revision must equal its exact file SHA-256"
                )
        if role in _CORPUS_BOUND_ROLES:
            corpus_id = artifact["corpus_id"]
            if corpus_id not in FIXED_CORPORA:
                raise StudyManifestError(f"artifact {identifier!r} has an unregistered corpus_id")
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
        inputs = {str(artifact["corpus_id"]): artifact for artifact in grouped["sealed-inputs"]}
        labels = {str(artifact["corpus_id"]): artifact for artifact in grouped["sealed-labels"]}
        ciphertexts = {
            str(artifact["corpus_id"]): artifact for artifact in grouped["sealed-label-ciphertext"]
        }
        encryption_receipts = {
            str(artifact["corpus_id"]): artifact
            for artifact in grouped["timelock-encryption-receipt"]
        }
        executions = {
            str(artifact["corpus_id"]): artifact for artifact in grouped["online-execution"]
        }
        policies = {str(artifact["corpus_id"]): artifact for artifact in grouped["policy-workload"]}
        embedding_stores = {
            str(artifact["corpus_id"]): artifact for artifact in grouped["embedding-store"]
        }
        index_stores = {
            str(artifact["corpus_id"]): artifact for artifact in grouped["authorized-index-store"]
        }
        runtime_packages = {
            str(artifact["corpus_id"]): artifact for artifact in grouped["trial-runtime-package"]
        }
        for corpus_id in FIXED_CORPORA:
            if (
                inputs[corpus_id]["uri"] == labels[corpus_id]["uri"]
                or inputs[corpus_id]["sha256"] == labels[corpus_id]["sha256"]
            ):
                raise StudyManifestError(
                    f"sealed inputs and labels for {corpus_id!r} must be separately pinned"
                )
            if executions[corpus_id]["uri"] in {
                inputs[corpus_id]["uri"],
                labels[corpus_id]["uri"],
            } or executions[corpus_id]["sha256"] in {
                inputs[corpus_id]["sha256"],
                labels[corpus_id]["sha256"],
            }:
                raise StudyManifestError(
                    f"online execution for {corpus_id!r} must be separately pinned"
                )
            if ciphertexts[corpus_id]["uri"] in {
                inputs[corpus_id]["uri"],
                labels[corpus_id]["uri"],
                executions[corpus_id]["uri"],
            } or ciphertexts[corpus_id]["sha256"] in {
                inputs[corpus_id]["sha256"],
                labels[corpus_id]["sha256"],
                executions[corpus_id]["sha256"],
            }:
                raise StudyManifestError(
                    f"sealed-label ciphertext for {corpus_id!r} must be separately pinned"
                )
            if encryption_receipts[corpus_id]["uri"] in {
                labels[corpus_id]["uri"],
                ciphertexts[corpus_id]["uri"],
                executions[corpus_id]["uri"],
            } or encryption_receipts[corpus_id]["sha256"] in {
                labels[corpus_id]["sha256"],
                ciphertexts[corpus_id]["sha256"],
                executions[corpus_id]["sha256"],
            }:
                raise StudyManifestError(
                    f"timelock encryption receipt for {corpus_id!r} must be separately pinned"
                )
            if policies[corpus_id]["uri"] in {
                inputs[corpus_id]["uri"],
                labels[corpus_id]["uri"],
            } or policies[corpus_id]["sha256"] in {
                inputs[corpus_id]["sha256"],
                labels[corpus_id]["sha256"],
            }:
                raise StudyManifestError(
                    f"policy workload for {corpus_id!r} must be separately pinned"
                )
            runtime_inputs = (
                executions[corpus_id],
                policies[corpus_id],
                embedding_stores[corpus_id],
                index_stores[corpus_id],
                runtime_packages[corpus_id],
            )
            if len({str(row["uri"]) for row in runtime_inputs}) != len(runtime_inputs) or len(
                {str(row["sha256"]) for row in runtime_inputs}
            ) != len(runtime_inputs):
                raise StudyManifestError(
                    f"execution, policy, embedding, index, and trial-runtime packages for "
                    f"{corpus_id!r} must be separately pinned"
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
        staged = grouped["study-data-package"][0]
        online_staging = grouped["online-staging-package"][0]
        if staged["uri"] == online_staging["uri"] or staged["sha256"] == online_staging["sha256"]:
            raise StudyManifestError(
                "complete staged data and its label-free online projection must be "
                "separately pinned"
            )
        current_embedding = grouped["primary-embedding"][0]
        stale_embedding = grouped["stale-embedding"][0]
        if (
            current_embedding["uri"],
            current_embedding["revision"],
        ) == (
            stale_embedding["uri"],
            stale_embedding["revision"],
        ) or current_embedding["sha256"] == stale_embedding["sha256"]:
            raise StudyManifestError(
                "current and stale embedding model trees must be separately pinned"
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


def _validate_production_control_bindings(value: object, *, frozen: bool) -> None:
    bindings = _closed_object(
        value,
        _PRODUCTION_CONTROL_BINDING_FIELDS,
        path="sealed_execution.production_controls",
    )
    for field in sorted(_PRODUCTION_CONTROL_BINDING_FIELDS):
        digest = _pinned_text(
            bindings[field],
            path=f"sealed_execution.production_controls.{field}",
            frozen=frozen,
        )
        if digest is not None and _SHA256.fullmatch(digest) is None:
            raise StudyManifestError(
                f"sealed_execution.production_controls.{field} must be a lowercase SHA-256"
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
            "sealed_execution.receipt_uri_template must contain one '{manifest_sha256}' token"
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
            "sealed_execution.receipt_uri_template filename must be '{manifest_sha256}.json'"
        )


def _reject_unregistered_provider_placeholder(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise StudyManifestError(f"{path} has a non-canonical field name")
            _reject_unregistered_provider_placeholder(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_unregistered_provider_placeholder(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        registered_main_index = path.endswith(".runtime_index_role") and value == "main"
        if _is_placeholder(value) and not registered_main_index:
            raise StudyManifestError(f"{path} retains an unresolved placeholder")
        brace_tokens = re.findall(r"\{[^{}]+\}", value)
        if brace_tokens and brace_tokens != ["{suite_attempt_id}"]:
            raise StudyManifestError(f"{path} contains an unknown path placeholder")


def _provider_absolute_path(value: object, *, path: str, templated: bool = False) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StudyManifestError(f"{path} must be one canonical absolute path")
    candidate = value
    if templated:
        if candidate.count("{suite_attempt_id}") != 1:
            raise StudyManifestError(
                f"{path} must contain exactly one '{{suite_attempt_id}}' binding"
            )
        candidate = candidate.replace("{suite_attempt_id}", "0" * 64)
    if (
        not candidate.startswith("/")
        or "//" in candidate
        or any(part in {"", ".", ".."} for part in Path(candidate).parts[1:])
        or str(Path(candidate)) != candidate
    ):
        raise StudyManifestError(f"{path} must be one canonical absolute POSIX path")
    return value


def _validate_provider_host_tools(value: object, *, path: str) -> Mapping[str, Any]:
    tools = _closed_object(value, set(PHASE_HOST_TOOL_CONTRACT_FIELDS), path=path)
    if tools["schema_version"] != PHASE_HOST_TOOL_CONTRACT_SCHEMA:
        raise StudyManifestError(f"{path}.schema_version differs")
    root = _provider_absolute_path(tools["controlled_root"], path=f"{path}.controlled_root")
    if root == "/":
        raise StudyManifestError(f"{path}.controlled_root cannot be the filesystem root")
    for field in (
        "python_executable",
        "venv_root",
        "gh_executable",
        "runner_listener_executable",
        "runner_listener_dll",
        "runner_config_executable",
        "runner_run_executable",
    ):
        candidate = _provider_absolute_path(tools[field], path=f"{path}.{field}")
        try:
            Path(candidate).relative_to(root)
        except ValueError as exc:
            raise StudyManifestError(f"{path}.{field} escapes controlled_root") from exc
    docker_path = _provider_absolute_path(
        tools["docker_executable"], path=f"{path}.docker_executable"
    )
    docker_resolved = _provider_absolute_path(
        tools["docker_resolved_executable"],
        path=f"{path}.docker_resolved_executable",
    )
    if docker_path == docker_resolved:
        raise StudyManifestError(f"{path} must distinguish Docker link and resolved target")
    exact = {
        "python_archive_uri": OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_URI,
        "python_archive_sha256": OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_SHA256,
        "python_archive_byte_count": OFFICIAL_PYTHON_BUILD_STANDALONE_ARCHIVE_BYTE_COUNT,
        "python_version": OFFICIAL_PYTHON_BUILD_STANDALONE_VERSION,
        "python_executable_sha256": OFFICIAL_PYTHON_BUILD_STANDALONE_BINARY_SHA256,
        "gh_archive_uri": OFFICIAL_GH_OSX_ARM64_ARCHIVE_URI,
        "gh_archive_sha256": OFFICIAL_GH_OSX_ARM64_ARCHIVE_SHA256,
        "gh_archive_byte_count": OFFICIAL_GH_OSX_ARM64_ARCHIVE_BYTE_COUNT,
        "gh_executable_sha256": OFFICIAL_GH_OSX_ARM64_BINARY_SHA256,
        "gh_version": OFFICIAL_GH_VERSION,
        "runner_archive_uri": OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_URI,
        "runner_archive_sha256": OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256,
        "runner_archive_byte_count": OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_BYTE_COUNT,
        "runner_listener_sha256": OFFICIAL_ACTIONS_RUNNER_LISTENER_SHA256,
        "runner_listener_dll_sha256": OFFICIAL_ACTIONS_RUNNER_LISTENER_DLL_SHA256,
        "runner_config_sha256": OFFICIAL_ACTIONS_RUNNER_CONFIG_SHA256,
        "runner_run_sha256": OFFICIAL_ACTIONS_RUNNER_RUN_SHA256,
        "runner_version": OFFICIAL_ACTIONS_RUNNER_VERSION,
        "docker_executable_sha256": REGISTERED_DOCKER_CLIENT_SHA256,
        "docker_client_version": REGISTERED_DOCKER_CLIENT_VERSION,
        "docker_client_build": REGISTERED_DOCKER_CLIENT_BUILD,
        "host_operating_system": "macOS",
        "host_architecture": "ARM64",
    }
    for field, expected in exact.items():
        if tools[field] != expected:
            raise StudyManifestError(f"{path}.{field} differs from the official pin")
    for field in (
        "runner_ephemeral",
        "runner_disable_update",
        "runner_unattended",
    ):
        if tools[field] is not True:
            raise StudyManifestError(f"{path}.{field} must be true")
    for field in (
        "venv_tree_sha256",
        "venv_symlink_inventory_sha256",
        "host_probe_receipt_sha256",
        "docker_server_probe_receipt_sha256",
    ):
        if not isinstance(tools[field], str) or _SHA256.fullmatch(tools[field]) is None:
            raise StudyManifestError(f"{path}.{field} must be one SHA-256 digest")
    host_probe = _closed_object(
        tools["host_probe"], set(PHASE_HOST_PROBE_FIELDS), path=f"{path}.host_probe"
    )
    if (
        host_probe["schema_version"] != PHASE_HOST_PROBE_SCHEMA
        or host_probe["operating_system"] != "macOS"
        or host_probe["architecture"] != "ARM64"
    ):
        raise StudyManifestError(f"{path}.host_probe differs from macOS ARM64")
    for field in ("logical_cpu_count", "physical_memory_bytes"):
        _positive_integer(host_probe[field], path=f"{path}.host_probe.{field}")
    for field in ("operating_system_version", "kernel_release"):
        _pinned_text(host_probe[field], path=f"{path}.host_probe.{field}", frozen=True)
    docker_probe = _closed_object(
        tools["docker_server_probe"],
        set(DOCKER_SERVER_PROBE_FIELDS),
        path=f"{path}.docker_server_probe",
    )
    if (
        docker_probe["schema_version"] != DOCKER_SERVER_PROBE_SCHEMA
        or docker_probe["operating_system"] != "linux"
        or docker_probe["architecture"] != "arm64"
    ):
        raise StudyManifestError(f"{path}.docker_server_probe differs from linux/arm64")
    for field in ("cpu_count", "memory_bytes"):
        _positive_integer(docker_probe[field], path=f"{path}.docker_server_probe.{field}")
    for field in ("engine_version", "engine_build", "kernel_version"):
        _pinned_text(docker_probe[field], path=f"{path}.docker_server_probe.{field}", frozen=True)
    host_probe_file_sha256 = hashlib.sha256(_canonical_bytes(host_probe) + b"\n").hexdigest()
    docker_probe_file_sha256 = hashlib.sha256(_canonical_bytes(docker_probe) + b"\n").hexdigest()
    if (
        tools["host_probe_receipt_sha256"] != host_probe_file_sha256
        or tools["docker_server_probe_receipt_sha256"] != docker_probe_file_sha256
    ):
        raise StudyManifestError(f"{path} probe receipt digest differs from canonical bytes")
    return tools


def _validate_provider_phase_plans(
    value: object,
    *,
    frozen: bool,
    sealed_approval_environment: object,
    sealed_code_commit: object,
    sealed_runner_identity: object,
    sealed_runner_image: object,
    artifacts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    path = "sealed_execution.provider_phase_plans"
    if not frozen:
        if value != "tbd":
            raise StudyManifestError(f"{path} must be the literal 'tbd' before freeze")
        return
    plans = _closed_object(value, set(PROVIDER_PHASES), path=path)
    observed_paths: set[str] = set()
    observed_evidence_roots: set[str] = set()
    observed_claim_paths: set[str] = set()
    observed_bootstrap_paths: set[str] = set()
    observed_registration_roots: set[str] = set()
    observed_nonces: set[str] = set()
    scientific_image: str | None = None
    canonical_host_tools: bytes | None = None
    controlled_root: str | None = None
    for phase in PROVIDER_PHASES:
        plan_path = f"{path}.{phase}"
        plan = _closed_object(
            plans[phase],
            set(PROVIDER_PHASE_PLAN_TEMPLATE_FIELDS),
            path=plan_path,
        )
        if plan["schema_version"] != PROVIDER_PHASE_PLAN_TEMPLATE_SCHEMA:
            raise StudyManifestError(f"{plan_path}.schema_version differs")
        if plan["phase"] != phase:
            raise StudyManifestError(f"{plan_path}.phase differs from its map key")
        claim_inputs = plan["execution_claim_inputs"]
        if phase == "online":
            inputs = _closed_object(
                claim_inputs,
                set(EXECUTION_CLAIM_INPUT_FIELDS),
                path=f"{plan_path}.execution_claim_inputs",
            )
            design_seed = inputs["design_seed_sha256"]
            if not isinstance(design_seed, str) or _SHA256.fullmatch(design_seed) is None:
                raise StudyManifestError(
                    f"{plan_path}.execution_claim_inputs.design_seed_sha256 must be pinned"
                )
            runtime_budget = inputs["registered_online_runtime_budget_seconds"]
            if (
                type(runtime_budget) is not int
                or runtime_budget <= 0
                or runtime_budget > plan["maximum_runtime_seconds"]
            ):
                raise StudyManifestError(
                    f"{plan_path}.execution_claim_inputs registered runtime budget "
                    "exceeds the phase ceiling"
                )
            beacon = _closed_object(
                inputs["beacon"],
                set(EXECUTION_BEACON_CONTRACT_FIELDS),
                path=f"{plan_path}.execution_claim_inputs.beacon",
            )
            if (
                beacon["schema_version"] != "fractal-execution-beacon-contract-v1"
                or beacon["seed_derivation"] != "sha256-fractal-execution-seed-v1-u64be"
            ):
                raise StudyManifestError(f"{plan_path}.execution_claim_inputs.beacon differs")
            for field in ("chain_hash", "verification_identity"):
                value = beacon[field]
                if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                    raise StudyManifestError(
                        f"{plan_path}.execution_claim_inputs.beacon.{field} must be pinned"
                    )
            for field in (
                "chain_genesis_unix_seconds",
                "chain_period_seconds",
                "execution_round",
                "label_release_round",
                "minimum_label_release_safety_rounds",
            ):
                _positive_integer(
                    beacon[field],
                    path=f"{plan_path}.execution_claim_inputs.beacon.{field}",
                )
            if beacon["label_release_round"] < (
                beacon["execution_round"] + beacon["minimum_label_release_safety_rounds"]
            ):
                raise StudyManifestError(
                    f"{plan_path}.execution_claim_inputs.beacon label-release round is too early"
                )
            for field in (
                "chain_public_key",
                "chain_scheme_id",
                "drand_network",
            ):
                _pinned_text(
                    beacon[field],
                    path=f"{plan_path}.execution_claim_inputs.beacon.{field}",
                    frozen=True,
                )
        elif claim_inputs is not None:
            raise StudyManifestError(
                f"{plan_path}.execution_claim_inputs must be null outside the online phase"
            )
        exact_bindings = {
            "manifest_sha256_binding": PROVIDER_PLAN_MANIFEST_BINDING,
            "c1_commit_binding": PROVIDER_PLAN_C1_COMMIT_BINDING,
            "suite_attempt_id_binding": PROVIDER_PLAN_SUITE_BINDING,
            "claim_predecessor_binding": PROVIDER_PLAN_PREDECESSOR_BINDING,
            "phase_input_binding": PROVIDER_PLAN_PHASE_INPUT_BINDING,
            "phase_output_binding": PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
        }
        for field, expected in exact_bindings.items():
            if plan[field] != expected:
                raise StudyManifestError(f"{plan_path}.{field} differs")
        self_hosted_path = _provider_absolute_path(
            plan["provider_plan_path"],
            path=f"{plan_path}.provider_plan_path",
        )
        if self_hosted_path in observed_paths:
            raise StudyManifestError(f"{path} reuses one fixed self-hosted plan path")
        observed_paths.add(self_hosted_path)
        evidence_template = _provider_absolute_path(
            plan["phase_evidence_root_template"],
            path=f"{plan_path}.phase_evidence_root_template",
            templated=True,
        )
        if evidence_template in observed_evidence_roots:
            raise StudyManifestError(f"{path} reuses one phase evidence root")
        observed_evidence_roots.add(evidence_template)
        claim_template = _provider_absolute_path(
            plan["claim_receipt_path_template"],
            path=f"{plan_path}.claim_receipt_path_template",
            templated=True,
        )
        if not claim_template.endswith("/claim-receipt.json"):
            raise StudyManifestError(f"{plan_path}.claim_receipt_path_template filename differs")
        if claim_template in observed_claim_paths:
            raise StudyManifestError(f"{path} reuses one claim receipt path")
        observed_claim_paths.add(claim_template)
        workflow = PROVIDER_PHASE_WORKFLOWS[phase]
        claim_job, execute_job = PROVIDER_PHASE_JOB_NAMES[phase]
        if (
            plan["approval_environment"] != sealed_approval_environment
            or plan["approval_environment"] != PROVIDER_APPROVAL_ENVIRONMENT
            or plan["runner_identity"] != sealed_runner_identity
            or plan["runner_identity"] != PROVIDER_RUNNER_IDENTITY
            or plan["runner_identity"]
            != f"github-actions:environment:{plan['approval_environment']}"
            or plan["repository"] != "mhdk1602/fractal-ann-diagnostics"
            or plan["workflow_path"] != workflow
            or plan["workflow_ref"]
            != f"mhdk1602/fractal-ann-diagnostics/{workflow}@refs/tags/confirmatory-apparatus-c0"
            or plan["run_head_branch"] != "confirmatory-apparatus-c0"
            or plan["workflow_sha"] != sealed_code_commit
            or plan["claim_job_name"] != claim_job
            or plan["execute_job_name"] != execute_job
        ):
            raise StudyManifestError(f"{plan_path} workflow or job identity differs")
        for field in (
            "claim_nonce",
            "runner_archive_sha256",
            "runner_bootstrap_receipt_file_sha256",
            "runner_registration_bundle_sha256",
            "runner_registration_evidence_file_sha256",
            "runtime_probe_receipt_sha256",
            "oci_index_digest",
            "oci_platform_manifest_digest",
        ):
            text = plan[field]
            digest = text.removeprefix("sha256:") if isinstance(text, str) else text
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise StudyManifestError(f"{plan_path}.{field} is not a SHA-256 binding")
        nonce = str(plan["claim_nonce"])
        if nonce in observed_nonces:
            raise StudyManifestError(f"{path} reuses one claim nonce")
        observed_nonces.add(nonce)
        bootstrap_path = _provider_absolute_path(
            plan["runner_bootstrap_receipt_path"],
            path=f"{plan_path}.runner_bootstrap_receipt_path",
        )
        expected_label = _derive_provider_runner_label(nonce, phase)
        expected_bootstrap = (
            Path(str(plan["host_tools"]["controlled_root"]))
            / "production"
            / "runners"
            / phase
            / expected_label
            / "bootstrap-receipt.json"
        )
        if Path(bootstrap_path) != expected_bootstrap:
            raise StudyManifestError(f"{plan_path}.runner_bootstrap_receipt_path differs")
        if bootstrap_path in observed_bootstrap_paths:
            raise StudyManifestError(f"{path} reuses one runner bootstrap receipt")
        observed_bootstrap_paths.add(bootstrap_path)
        registration_root = _provider_absolute_path(
            plan["runner_registration_bundle_path"],
            path=f"{plan_path}.runner_registration_bundle_path",
        )
        expected_registration_root = (
            Path(str(plan["host_tools"]["controlled_root"]))
            / "production"
            / "runner-registrations"
            / phase
            / expected_label
        )
        if Path(registration_root) != expected_registration_root:
            raise StudyManifestError(f"{plan_path}.runner_registration_bundle_path differs")
        if registration_root in observed_registration_roots:
            raise StudyManifestError(f"{path} reuses one runner registration bundle")
        observed_registration_roots.add(registration_root)
        bootstrap = _closed_object(
            plan["runner_bootstrap_receipt"],
            set(PROVIDER_RUNNER_BOOTSTRAP_FIELDS),
            path=f"{plan_path}.runner_bootstrap_receipt",
        )
        if (
            bootstrap["schema_version"] != "fractal-provider-runner-bootstrap-v2"
            or bootstrap["approval_environment"] != plan["approval_environment"]
            or bootstrap["runner_identity"] != plan["runner_identity"]
            or bootstrap["runner_identity"]
            != f"github-actions:environment:{bootstrap['approval_environment']}"
            or bootstrap["phase"] != phase
            or bootstrap["repository"] != plan["repository"]
            or bootstrap["workflow_sha"] != plan["workflow_sha"]
            or bootstrap["runner_label"] != expected_label
            or bootstrap["runner_id"] != plan["runner_id"]
            or bootstrap["runner_name"] != plan["runner_name"]
            or bootstrap["runner_group_id"] != plan["runner_group_id"]
            or bootstrap["runner_version"] != plan["runner_version"]
            or bootstrap["runner_archive_sha256"] != plan["runner_archive_sha256"]
            or (bootstrap["ephemeral"], bootstrap["disable_update"], bootstrap["unattended"])
            != (True, True, True)
        ):
            raise StudyManifestError(f"{plan_path}.runner_bootstrap_receipt differs")
        inventory_digest = bootstrap["repository_runner_inventory_sha256"]
        if not isinstance(inventory_digest, str) or _SHA256.fullmatch(inventory_digest) is None:
            raise StudyManifestError(
                f"{plan_path}.runner_bootstrap_receipt inventory digest differs"
            )
        _pinned_text(
            bootstrap["registered_at_utc"],
            path=f"{plan_path}.runner_bootstrap_receipt.registered_at_utc",
            frozen=True,
        )
        expected_bootstrap_digest = hashlib.sha256(_canonical_bytes(bootstrap) + b"\n").hexdigest()
        if plan["runner_bootstrap_receipt_file_sha256"] != expected_bootstrap_digest:
            raise StudyManifestError(
                f"{plan_path}.runner_bootstrap_receipt_file_sha256 differs from embedded bytes"
            )
        _positive_integer(plan["runner_id"], path=f"{plan_path}.runner_id")
        runner_group_id = plan["runner_group_id"]
        if runner_group_id is not None and (
            type(runner_group_id) is not int or runner_group_id < 0
        ):
            raise StudyManifestError(
                f"{plan_path}.runner_group_id must be null or a non-negative integer"
            )
        for field in (
            "runner_name",
            "runner_version",
            "provider_operating_system",
            "provider_architecture",
        ):
            _pinned_text(plan[field], path=f"{plan_path}.{field}", frozen=True)
        if plan["provider_operating_system"] != "macOS" or plan["provider_architecture"] != "ARM64":
            raise StudyManifestError(f"{plan_path} provider host differs from macOS ARM64")
        if (
            plan["runner_version"] != OFFICIAL_ACTIONS_RUNNER_VERSION
            or plan["runner_archive_sha256"] != OFFICIAL_ACTIONS_RUNNER_OSX_ARM64_ARCHIVE_SHA256
        ):
            raise StudyManifestError(f"{plan_path} runner release differs from C0")
        host_tools = _validate_provider_host_tools(
            plan["host_tools"], path=f"{plan_path}.host_tools"
        )
        encoded_host_tools = _canonical_bytes(host_tools)
        if canonical_host_tools is not None and encoded_host_tools != canonical_host_tools:
            raise StudyManifestError(f"{path} changes the phase host-tool closure")
        canonical_host_tools = encoded_host_tools
        phase_controlled_root = str(host_tools["controlled_root"])
        if controlled_root is not None and phase_controlled_root != controlled_root:
            raise StudyManifestError(f"{path} changes controlled_root between phases")
        controlled_root = phase_controlled_root
        try:
            Path(self_hosted_path).relative_to(phase_controlled_root)
        except ValueError as exc:
            raise StudyManifestError(
                f"{plan_path}.provider_plan_path escapes controlled_root"
            ) from exc
        platform, image_role, index_role = PROVIDER_PHASE_RUNTIME_BINDINGS[phase]
        if (
            plan["runtime_platform"] != platform
            or plan["runtime_image_role"] != image_role
            or plan["runtime_index_role"] != index_role
            or plan["maximum_runtime_seconds"] != PROVIDER_PHASE_RUNTIME_CEILINGS[phase]
        ):
            raise StudyManifestError(f"{plan_path} runtime role or ceiling differs")
        image = plan["runtime_image"]
        if not isinstance(image, str) or _OCI_DIGEST.fullmatch(image) is None:
            raise StudyManifestError(f"{plan_path}.runtime_image must be digest-pinned")
        if plan["oci_index_digest"] != image.rsplit("@", 1)[1]:
            raise StudyManifestError(f"{plan_path}.oci_index_digest differs from runtime_image")
        if phase in {"online", "analysis"}:
            if image != sealed_runner_image:
                raise StudyManifestError(
                    f"{plan_path}.runtime_image differs from sealed_execution.runner_image"
                )
            if scientific_image is not None and image != scientific_image:
                raise StudyManifestError(f"{path} changes the scientific runtime image")
            scientific_image = image
        elif image == sealed_runner_image:
            raise StudyManifestError("label-release must use a distinct timelock image")
        tle_fields = (
            "tle_binary_sha256",
            "tle_build_provenance_sha256",
            "tle_interoperability_receipt_sha256",
            "tle_vulnerability_scan_sha256",
        )
        if phase == "label-release":
            for field in tle_fields:
                if not isinstance(plan[field], str) or _SHA256.fullmatch(plan[field]) is None:
                    raise StudyManifestError(f"{plan_path}.{field} must be pinned")
            timelock_tool = artifacts["timelock-tool"][0]
            if (
                plan["tle_binary_sha256"] != timelock_tool["sha256"]
                or plan["tle_binary_sha256"] != SOURCE_BUILT_LINUX_ARM64_TLE_SHA256
            ):
                raise StudyManifestError(
                    f"{plan_path}.tle_binary_sha256 differs from the timelock-tool artifact"
                )
        elif any(plan[field] is not None for field in tle_fields):
            raise StudyManifestError(f"{plan_path} cannot introduce a timelock binary")
        if plan["activation_command_id"] != PROVIDER_PHASE_COMMAND_IDS[phase]:
            raise StudyManifestError(f"{plan_path}.activation_command_id differs")
        argv = plan["activation_argv_template"]
        python_path = host_tools.get("python_executable")
        expected_argv = [
            python_path,
            "-m",
            "fractal_ann_diagnostics.provider_phase_runtime",
            PROVIDER_PHASE_COMMAND_IDS[phase],
            "--provider-plan",
            self_hosted_path,
            "--suite-attempt-id",
            PROVIDER_PLAN_SUITE_BINDING,
            "--claim-receipt",
            PROVIDER_PLAN_CLAIM_RECEIPT_BINDING,
            "--phase-input-root",
            PROVIDER_PLAN_PHASE_INPUT_BINDING,
            "--phase-output-root",
            PROVIDER_PLAN_PHASE_OUTPUT_BINDING,
        ]
        if argv != expected_argv:
            raise StudyManifestError(f"{plan_path}.activation_argv_template differs")
        _reject_unregistered_provider_placeholder(plan, path=plan_path)
    assert controlled_root is not None
    materialized_paths = [
        Path(value.replace("{suite_attempt_id}", "0" * 64))
        for value in (*observed_evidence_roots, *observed_claim_paths)
    ]
    controlled = Path(controlled_root)
    for candidate in materialized_paths:
        if (
            candidate == controlled
            or controlled in candidate.parents
            or candidate in controlled.parents
        ):
            raise StudyManifestError(f"{path} overlaps mutable phase evidence with controlled_root")
    fixed_paths = [
        Path(value)
        for value in (
            *observed_paths,
            *observed_bootstrap_paths,
            *observed_registration_roots,
        )
    ]
    for position, left in enumerate(fixed_paths):
        for right in fixed_paths[position + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise StudyManifestError(f"{path} plan paths collide or overlap")
    for position, left in enumerate(materialized_paths):
        for right in materialized_paths[position + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise StudyManifestError(f"{path} runtime paths collide or overlap")


def _validate_sealed_execution(
    value: object,
    *,
    frozen: bool,
    artifacts: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_rehearsal: bool = False,
) -> None:
    if candidate_rehearsal and not frozen:  # pragma: no cover - internal invariant
        raise RuntimeError("candidate rehearsal must use frozen content semantics")
    sealed = _closed_object(
        value,
        _SEALED_EXECUTION_FIELDS,
        path="sealed_execution",
    )
    _registered_number(sealed, "reserve_fraction", 0.0, path="sealed_execution")
    _pinned_text(
        sealed["custodian"],
        path="sealed_execution.custodian",
        frozen=frozen,
    )
    approval_environment = _pinned_text(
        sealed["approval_environment"],
        path="sealed_execution.approval_environment",
        frozen=frozen,
    )
    runner_identity = _pinned_text(
        sealed["runner_identity"],
        path="sealed_execution.runner_identity",
        frozen=frozen,
    )
    if approval_environment is not None and runner_identity is not None:
        expected_runner_identity = f"github-actions:environment:{approval_environment}"
        if runner_identity != expected_runner_identity:
            raise StudyManifestError(
                "sealed_execution.runner_identity must equal "
                "github-actions:environment:{approval_environment}"
            )
        if (
            approval_environment != PROVIDER_APPROVAL_ENVIRONMENT
            or runner_identity != PROVIDER_RUNNER_IDENTITY
        ):
            raise StudyManifestError(
                "sealed_execution.approval_environment must equal confirmatory"
            )
    results_store = _pinned_text(
        sealed["results_store"],
        path="sealed_execution.results_store",
        frozen=frozen,
    )
    if results_store is not None:
        parsed_store = urlsplit(results_store)
        results_path = Path(unquote(parsed_store.path))
        if (
            parsed_store.scheme != "file"
            or parsed_store.netloc
            or parsed_store.query
            or parsed_store.fragment
            or not results_path.is_absolute()
            or results_path.as_uri() != results_store
            or any(part in {"", ".", ".."} for part in results_path.parts[1:])
        ):
            raise StudyManifestError(
                "sealed_execution.results_store must be one canonical absolute file URI"
            )
    _validate_hardware(sealed["hardware"], frozen=frozen)
    _validate_production_control_bindings(
        sealed["production_controls"],
        frozen=frozen,
    )
    _validate_receipt_template(sealed["receipt_uri_template"], frozen=frozen)
    if sealed["label_artifacts_withheld_until_prediction_receipt"] is not True:
        raise StudyManifestError(
            "sealed_execution.label_artifacts_withheld_until_prediction_receipt must be true"
        )
    if sealed["public_query_reidentification_risk"] != ("accepted-public-benchmark-limitation"):
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
        raise StudyManifestError("sealed_execution.runner_image must use an OCI SHA-256 digest")
    try:
        validate_c0_evidence_release_binding(
            sealed["c0_evidence_release"],
            frozen=frozen and not candidate_rehearsal,
            code_commit=code_commit,
        )
    except C0EvidenceReleaseError as exc:
        raise StudyManifestError(str(exc)) from exc
    _validate_provider_phase_plans(
        sealed["provider_phase_plans"],
        frozen=frozen,
        sealed_approval_environment=sealed["approval_environment"],
        sealed_code_commit=sealed["code_commit"],
        sealed_runner_identity=sealed["runner_identity"],
        sealed_runner_image=sealed["runner_image"],
        artifacts=artifacts,
    )
    if frozen and code_commit is not None:
        source_revision = artifacts["source-code"][0]["revision"]
        if source_revision != code_commit:
            raise StudyManifestError(
                "the source-code artifact revision must equal sealed_execution.code_commit"
            )


def validate_candidate_rehearsal_manifest(
    payload: Mapping[str, Any],
    *,
    c0_commit: str,
) -> None:
    """Admit the sole pre-C0 state while applying frozen content semantics."""

    root = _closed_object(payload, _ROOT_FIELDS, path="study manifest")
    if root["schema_version"] != "1.0":
        raise StudyManifestError("schema_version must equal '1.0'")
    if root["status"] != "draft" or root["protocol_version"] != "0.3.0-draft":
        raise StudyManifestError(
            "candidate rehearsal requires status='draft' and protocol_version='0.3.0-draft'"
        )
    if root["claim_scope"] != "suite-conditional-retrieval-control":
        raise StudyManifestError("claim_scope must remain 'suite-conditional-retrieval-control'")
    if root["primary_claim"] != REGISTERED_PRIMARY_CLAIM:
        raise StudyManifestError("primary_claim must equal the prespecified v0.3 claim")
    _validate_freeze_blockers(root["freeze_blockers"], frozen=False)
    sealed = root["sealed_execution"]
    if not isinstance(sealed, Mapping) or sealed.get("c0_evidence_release") != "tbd":
        raise StudyManifestError(
            "candidate rehearsal requires sealed_execution.c0_evidence_release='tbd'"
        )

    resolved = resolve_candidate_provider_plan_commit_bindings(
        payload,
        c0_commit=c0_commit,
    )
    resolved_root = _closed_object(resolved, _ROOT_FIELDS, path="resolved candidate manifest")
    _validate_analysis(resolved_root["analysis"], frozen=True)
    artifacts = _validate_artifacts(resolved_root["artifacts"], frozen=True)
    _validate_sealed_execution(
        resolved_root["sealed_execution"],
        frozen=True,
        artifacts=artifacts,
        candidate_rehearsal=True,
    )
    try:
        validate_production_workload_registrations(
            resolved_root["production_workloads"],
            frozen=True,
            registered_selected_family_count=resolved_root["analysis"]["power"][  # type: ignore[index]
                "selected_families_per_corpus"
            ],
            sealed_execution=resolved_root["sealed_execution"],
            fixed_corpora=FIXED_CORPORA,
        )
    except ProductionWorkloadRegistrationError as exc:
        raise StudyManifestError(
            f"invalid candidate production workload registration: {exc}"
        ) from exc


def validate_candidate_rehearsal_to_frozen_transition(
    candidate: Mapping[str, Any],
    frozen: Mapping[str, Any],
    *,
    c0_commit: str,
) -> None:
    """Prove that C1 changed only lifecycle fields, C0 sentinels, and the C0 release."""

    validate_candidate_rehearsal_manifest(candidate, c0_commit=c0_commit)
    validate_study_manifest(frozen, require_frozen=True)
    frozen_sealed = frozen.get("sealed_execution")
    if not isinstance(frozen_sealed, Mapping):  # pragma: no cover - frozen validation owns this
        raise StudyManifestError("frozen manifest lacks sealed_execution")
    evidence_release = frozen_sealed.get("c0_evidence_release")
    if not isinstance(evidence_release, Mapping):  # pragma: no cover - binding validation owns this
        raise StudyManifestError("frozen manifest lacks the C0 evidence release binding")
    apparatus = evidence_release.get("apparatus_evidence")
    if not isinstance(apparatus, Mapping):  # pragma: no cover - binding validation owns this
        raise StudyManifestError("C0 evidence release lacks apparatus evidence")
    candidate_sha256 = manifest_sha256(candidate)
    if apparatus.get("rehearsal_manifest_sha256") != candidate_sha256:
        raise StudyManifestError(
            "C0 evidence rehearsal manifest digest differs from the raw candidate manifest"
        )
    # Imported here because execution_claim imports this module for manifest admission.
    from .execution_claim import provider_phase_plan_templates_sha256

    candidate_plan_sha256 = provider_phase_plan_templates_sha256(
        candidate,
        validation_mode="candidate-rehearsal",
        c0_commit=c0_commit,
    )
    if apparatus.get("provider_phase_plan_closure_sha256") != candidate_plan_sha256:
        raise StudyManifestError(
            "C0 evidence provider-plan closure differs from the normalized candidate plans"
        )
    expected = resolve_candidate_provider_plan_commit_bindings(
        candidate,
        c0_commit=c0_commit,
    )
    expected["status"] = "frozen"
    expected["protocol_version"] = "0.3.0"
    expected["freeze_blockers"] = []
    expected_sealed = expected.get("sealed_execution")
    if not isinstance(expected_sealed, dict) or not isinstance(frozen_sealed, Mapping):
        raise StudyManifestError("candidate or frozen manifest lacks sealed_execution")
    expected_sealed["c0_evidence_release"] = copy.deepcopy(frozen_sealed["c0_evidence_release"])
    if _canonical_bytes(expected) != _canonical_bytes(frozen):
        raise StudyManifestError(
            "frozen C1 changes fields outside the registered candidate transition"
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
        raise StudyManifestError("claim_scope must remain 'suite-conditional-retrieval-control'")
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
    try:
        validate_production_workload_registrations(
            root["production_workloads"],
            frozen=frozen,
            registered_selected_family_count=root["analysis"]["power"][  # type: ignore[index]
                "selected_families_per_corpus"
            ],
            sealed_execution=root["sealed_execution"],
            fixed_corpora=FIXED_CORPORA,
        )
    except ProductionWorkloadRegistrationError as exc:
        raise StudyManifestError(f"invalid production workload registration: {exc}") from exc


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
    pinned_by_id = {str(artifact["id"]): str(artifact["sha256"]) for artifact in manifest_artifacts}
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
    """Reject redirects while revalidating the publicly registered record."""

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
                raise StudyManifestError("protocol registry revalidation response URL changed")
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
        raise StudyManifestError("protocol registry record exceeds the maximum byte limit")
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
            raise StudyManifestError("registry record manifest_sha256 must be a lowercase SHA-256")
        if self.protocol_version != "0.3.0":
            raise StudyManifestError("registry record protocol_version must equal '0.3.0'")
        _parse_utc(self.registered_at_utc, field="registered_at_utc")
        if (
            not isinstance(self.registry_identity, str)
            or not self.registry_identity.strip()
            or self.registry_identity != self.registry_identity.strip()
        ):
            raise StudyManifestError("registry_identity must be a canonical non-empty string")
        _validate_external_registry_uri(self.registry_uri)
        if self.schema_version != PROTOCOL_REGISTRY_RECORD_SCHEMA:
            raise StudyManifestError(
                f"registry record schema_version must equal {PROTOCOL_REGISTRY_RECORD_SCHEMA!r}"
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
            raise StudyManifestError("registration manifest_sha256 must be a lowercase SHA-256")
        if self.protocol_version != "0.3.0":
            raise StudyManifestError("registration protocol_version must equal '0.3.0'")
        _parse_utc(self.registered_at_utc, field="registered_at_utc")
        if (
            not isinstance(self.registry_identity, str)
            or not self.registry_identity.strip()
            or self.registry_identity != self.registry_identity.strip()
        ):
            raise StudyManifestError("registry_identity must be a canonical non-empty string")
        _validate_external_registry_uri(self.registry_uri)
        if _SHA256.fullmatch(self.registry_record_sha256) is None:
            raise StudyManifestError("registry_record_sha256 must be a lowercase SHA-256")
        if self.schema_version != PROTOCOL_REGISTRATION_RECEIPT_SCHEMA:
            raise StudyManifestError(
                f"registration schema_version must equal {PROTOCOL_REGISTRATION_RECEIPT_SCHEMA!r}"
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
class VerifiedC1ProtocolRegistration:
    """File-backed capability for the fixed, publicly verified C1 deposit.

    The generic registry record and local receipt are data objects.  They do not
    authorize production.  This capability is minted only after the closed C1
    package, both GitHub attestations, and every public Zenodo byte have been
    verified.  The run opener rechecks the local files and invokes the retained
    fresh verifier before it admits the capability.
    """

    record: ProtocolRegistryRecord
    receipt: ProtocolRegistrationReceipt
    package_root: Path
    registration_record_path: Path
    registration_receipt_path: Path
    c0_commit: str
    c1_commit: str
    package_file_sha256s: tuple[tuple[str, str], ...]
    _fresh_revalidator: Callable[[], None] = dataclass_field(repr=False, compare=False)
    _capability: object = dataclass_field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _C1_REGISTRATION_CAPABILITY:
            raise StudyManifestError(
                "verified C1 registration can only come from the production verifier"
            )
        if not isinstance(self.record, ProtocolRegistryRecord) or not isinstance(
            self.receipt, ProtocolRegistrationReceipt
        ):
            raise StudyManifestError("verified C1 registration has untyped record evidence")
        for name in ("package_root", "registration_record_path", "registration_receipt_path"):
            path = Path(getattr(self, name))
            if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
                raise StudyManifestError(f"verified C1 {name} must be a canonical absolute path")
            object.__setattr__(self, name, path)
        for name in ("c0_commit", "c1_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
                raise StudyManifestError(f"verified C1 {name} must be one full Git commit")
        if self.c0_commit == self.c1_commit:
            raise StudyManifestError("verified C1 commit must differ from C0")
        rows = tuple(self.package_file_sha256s)
        if not rows or any(
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not row[0]
            or "/" in row[0]
            or "\\" in row[0]
            or row[0] in {".", ".."}
            or not isinstance(row[1], str)
            or _SHA256.fullmatch(row[1]) is None
            for row in rows
        ):
            raise StudyManifestError("verified C1 package inventory is malformed")
        expected_rows = tuple(sorted(rows, key=lambda row: row[0].encode("utf-8")))
        if rows != expected_rows or len({row[0] for row in rows}) != len(rows):
            raise StudyManifestError("verified C1 package inventory must be unique and sorted")
        if "protocol-registry-record.json" not in {row[0] for row in rows}:
            raise StudyManifestError("verified C1 package omits the protocol registry record")
        object.__setattr__(self, "package_file_sha256s", rows)
        if not callable(self._fresh_revalidator):
            raise StudyManifestError("verified C1 registration lacks a fresh revalidator")
        self._assert_local_current()

    @property
    def manifest_sha256(self) -> str:
        return self.record.manifest_sha256

    def _assert_local_current(self) -> None:
        root = self.package_root
        try:
            resolved_root = root.resolve(strict=True)
            observed_names = {path.name for path in root.iterdir()}
        except OSError as exc:
            raise StudyManifestError("verified C1 package is no longer readable") from exc
        if root.is_symlink() or resolved_root != root:
            raise StudyManifestError("verified C1 package root changed or became a link")
        expected = dict(self.package_file_sha256s)
        if observed_names != set(expected):
            raise StudyManifestError("verified C1 package file set changed after verification")
        try:
            for name, digest in self.package_file_sha256s:
                observed_digest = digest_regular_file(
                    root / name,
                    label=f"verified C1 package file {name}",
                )
                if observed_digest != digest:
                    raise StudyManifestError(
                        f"verified C1 package file {name!r} changed after verification"
                    )
            record = load_protocol_registry_record(self.registration_record_path)
            receipt = load_protocol_registration_receipt(self.registration_receipt_path)
            record_file_sha256 = digest_regular_file(
                self.registration_record_path,
                label="verified local C1 protocol registry record",
            )
        except ArtifactIntegrityError as exc:
            raise StudyManifestError(f"verified C1 local evidence changed: {exc}") from exc
        if record != self.record or receipt != self.receipt:
            raise StudyManifestError("verified C1 local record or receipt changed")
        if record_file_sha256 != self.record.record_sha256:
            raise StudyManifestError("verified local C1 registry-record bytes changed")
        if expected["protocol-registry-record.json"] != self.record.record_sha256:
            raise StudyManifestError("verified C1 package and local registry record differ")
        shared = (
            "manifest_sha256",
            "protocol_version",
            "registered_at_utc",
            "registry_identity",
            "registry_uri",
        )
        if any(getattr(record, name) != getattr(receipt, name) for name in shared):
            raise StudyManifestError("verified C1 registry record and receipt differ")
        if receipt.registry_record_sha256 != record.record_sha256:
            raise StudyManifestError("verified C1 receipt binds another registry record")

    def assert_current(self) -> None:
        """Recheck retained files, both attestations, and anonymous Zenodo bytes."""

        self._assert_local_current()
        try:
            result = self._fresh_revalidator()
        except StudyManifestError:
            raise
        except Exception as exc:
            raise StudyManifestError("fresh C1 registration revalidation failed") from exc
        if result is not None:
            raise StudyManifestError("fresh C1 registration revalidator returned data")
        self._assert_local_current()


def _mint_verified_c1_protocol_registration(
    *,
    record: ProtocolRegistryRecord,
    receipt: ProtocolRegistrationReceipt,
    package_root: str | Path,
    registration_record_path: str | Path,
    registration_receipt_path: str | Path,
    c0_commit: str,
    c1_commit: str,
    package_file_sha256s: Sequence[tuple[str, str]],
    fresh_revalidator: Callable[[], None],
) -> VerifiedC1ProtocolRegistration:
    """Private bridge used by the fixed Zenodo verifier to mint admission."""

    return VerifiedC1ProtocolRegistration(
        record=record,
        receipt=receipt,
        package_root=Path(package_root),
        registration_record_path=Path(registration_record_path),
        registration_receipt_path=Path(registration_receipt_path),
        c0_commit=c0_commit,
        c1_commit=c1_commit,
        package_file_sha256s=tuple(package_file_sha256s),
        _fresh_revalidator=fresh_revalidator,
        _capability=_C1_REGISTRATION_CAPABILITY,
    )


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
            raise StudyManifestError("verification_receipt_uri must be an absolute file URI")
        if _SHA256.fullmatch(self.verification_receipt_sha256) is None:
            raise StudyManifestError("verification_receipt_sha256 must be a lowercase SHA-256")
        parsed = urlsplit(self.receipt_uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise StudyManifestError("receipt_uri must be an absolute file URI")

    def to_dict(self) -> dict[str, str]:
        return {
            "code_commit": self.code_commit,
            "manifest_sha256": self.manifest_sha256,
            "protocol_version": self.protocol_version,
            "protocol_registration_receipt_sha256": (self.protocol_registration_receipt_sha256),
            "protocol_registration_receipt_uri": (self.protocol_registration_receipt_uri),
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

    @property
    def binding_sha256(self) -> str:
        """Digest the semantic run-receipt binding used across custody stages."""

        payload: dict[str, str] = self.to_dict()
        payload["schema_version"] = SEALED_RUN_RECEIPT_BINDING_SCHEMA
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

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
            protocol_registration_receipt_uri=receipt["protocol_registration_receipt_uri"],
            protocol_registration_receipt_sha256=receipt["protocol_registration_receipt_sha256"],
            protocol_registration_record_uri=receipt["protocol_registration_record_uri"],
            verification_receipt_uri=receipt["verification_receipt_uri"],
            verification_receipt_sha256=receipt["verification_receipt_sha256"],
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
        raise StudyManifestError("sealed run receipt is not at its manifest-derived receipt_uri")
    return receipt


def begin_sealed_run(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    runner_identity: str,
    artifact_verification_receipt_path: str | Path,
    artifact_root: str | Path,
    local_artifact_map_path: str | Path,
    verified_protocol_registration: VerifiedC1ProtocolRegistration,
) -> SealedRunReceipt:
    """Revalidate frozen controls and atomically create the one-shot run receipt.

    A generic HTTPS registry record is insufficient. Production admission
    requires the guarded C1 capability, which freshly rechecks both GitHub
    attestations and all public bytes in the fixed Zenodo deposit.
    """
    if not isinstance(verified_protocol_registration, VerifiedC1ProtocolRegistration):
        raise StudyManifestError(
            "production sealed execution requires verified fixed C1 registration"
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
        raise StudyManifestError("runner_identity does not equal sealed_execution.runner_identity")

    digest = manifest_sha256(payload)
    try:
        expected = (
            read_secure_control_file(
                lock_path,
                label="manifest lock",
            )
            .decode("ascii")
            .strip()
        )
    except (ArtifactIntegrityError, UnicodeDecodeError) as exc:
        raise StudyManifestError(f"cannot read manifest lock: {exc}") from exc
    if _SHA256.fullmatch(expected) is None:
        raise StudyManifestError("manifest lock must contain one lowercase SHA-256")
    if digest != expected:
        raise StudyManifestError("manifest digest does not match the frozen lock")

    verified_protocol_registration.assert_current()
    registration_receipt = verified_protocol_registration.receipt
    if registration_receipt.manifest_sha256 != digest:
        raise StudyManifestError(
            "protocol registration receipt is bound to a different manifest digest"
        )
    if registration_receipt.protocol_version != payload["protocol_version"]:
        raise StudyManifestError("protocol registration receipt has a different protocol version")
    registered_at = _parse_utc(
        registration_receipt.registered_at_utc,
        field="registered_at_utc",
    )
    if registered_at > datetime.now(timezone.utc):
        raise StudyManifestError("protocol registration timestamp cannot be in the future")
    registry_record = verified_protocol_registration.record
    if registry_record.record_sha256 != registration_receipt.registry_record_sha256:
        raise StudyManifestError("protocol registration record digest does not match its receipt")
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
        verification_receipt = load_verification_receipt(artifact_verification_receipt_path)
    except ArtifactIntegrityError as exc:
        raise StudyManifestError(f"invalid artifact verification receipt: {exc}") from exc
    _validate_artifact_verification_receipt(
        verification_receipt,
        payload=payload,
        manifest_digest=digest,
    )
    pins = {str(artifact["id"]): str(artifact["sha256"]) for artifact in payload["artifacts"]}
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
        raise StudyManifestError(f"fresh local artifact revalidation failed: {exc}") from exc
    if not hmac.compare_digest(
        fresh_verification_receipt.canonical_bytes(),
        verification_receipt.canonical_bytes(),
    ):
        raise StudyManifestError(
            "fresh local artifact revalidation differs from the admitted receipt"
        )

    receipt_uri = sealed_receipt_uri(payload)
    target = _receipt_path_from_uri(receipt_uri)
    verification_receipt_path = Path(artifact_verification_receipt_path)
    registration_receipt_path = verified_protocol_registration.registration_receipt_path
    registration_record_path = verified_protocol_registration.registration_record_path
    receipt = SealedRunReceipt(
        manifest_sha256=digest,
        protocol_version=str(payload["protocol_version"]),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        runner_identity=runner_identity,
        code_commit=str(sealed["code_commit"]),
        runner_image=str(sealed["runner_image"]),
        protocol_registration_receipt_uri=(registration_receipt_path.as_uri()),
        protocol_registration_receipt_sha256=(registration_receipt.receipt_sha256),
        protocol_registration_record_uri=(registration_record_path.as_uri()),
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
