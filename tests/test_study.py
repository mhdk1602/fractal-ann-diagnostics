from __future__ import annotations

import hashlib
import json
import ssl
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request

import pytest

import fractal_ann_diagnostics.study as study_module
from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    VerifiedArtifact,
    load_verification_receipt,
    write_verification_receipt,
)
from fractal_ann_diagnostics.study import (
    EVIDENCE_CORPORA,
    FIXED_CORPORA,
    MAX_PROTOCOL_REGISTRY_RECORD_BYTES,
    REGISTERED_ACTION_SET,
    REGISTERED_POWER_ENDPOINTS,
    REGISTERED_POWER_FAMILY_CANDIDATES,
    REGISTERED_PRIMARY_CLAIM,
    ProtocolRegistrationReceipt,
    ProtocolRegistryRecord,
    SealedRunReceipt,
    StudyManifestError,
    begin_sealed_run,
    load_protocol_registration_receipt,
    load_sealed_run_receipt,
    load_study_manifest,
    manifest_sha256,
    sealed_receipt_uri,
    validate_study_manifest,
)

_ROLE_KINDS = (
    ("development-fit-data", "dataset"),
    ("development-calibration-data", "dataset"),
    ("query-partition-audit", "partition-audit"),
    ("primary-embedding", "embedding"),
    ("exact-authorized-oracle", "backend"),
    ("strict-authorized-hnsw", "backend"),
    ("opa-pdp", "policy"),
    ("frozen-controller", "controller"),
    ("static-comparator", "comparator"),
    ("h1-predictive-model", "model"),
    ("h2-model-suite", "model"),
    ("power-analysis-report", "analysis"),
    ("analysis-runner", "analysis"),
    ("source-code", "source"),
)
_COMMIT = "1" * 40
_RUNNER_IDENTITY = "github-actions:environment:confirmatory"


def _artifact(
    role: str,
    kind: str,
    *,
    frozen: bool,
    corpus_id: str | None = None,
) -> dict[str, object]:
    identifier = f"{corpus_id}-{role}" if corpus_id is not None else role
    revision = _COMMIT if frozen and role == "source-code" else "v1.0.0"
    artifact: dict[str, object] = {
        "kind": kind,
        "id": identifier,
        "uri": f"https://example.test/{identifier}",
        "revision": revision if frozen else "tbd",
        "sha256": hashlib.sha256(identifier.encode("utf-8")).hexdigest()
        if frozen
        else "tbd",
        "license": "MIT",
        "role": role,
    }
    if corpus_id is not None:
        artifact["corpus_id"] = corpus_id
    return artifact


def _artifacts(*, frozen: bool) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for role, kind in (
        ("sealed-inputs", "dataset"),
        ("sealed-labels", "dataset"),
        ("online-execution", "execution"),
        ("corpus-normalizer", "normalizer"),
        ("policy-workload", "policy-data"),
    ):
        artifacts.extend(
            _artifact(role, kind, frozen=frozen, corpus_id=corpus_id)
            for corpus_id in FIXED_CORPORA
        )
    artifacts.extend(
        _artifact(role, kind, frozen=frozen) for role, kind in _ROLE_KINDS
    )
    return artifacts


def _manifest(
    *,
    frozen: bool = False,
    receipt_root: Path | None = None,
) -> dict[str, object]:
    receipt_directory = receipt_root or Path("/tmp/fractal-ann-confirmatory-receipts")
    return {
        "schema_version": "1.0",
        "protocol_version": "0.3.0" if frozen else "0.3.0-draft",
        "status": "frozen" if frozen else "draft",
        "claim_scope": "suite-conditional-retrieval-control",
        "primary_claim": REGISTERED_PRIMARY_CLAIM,
        "freeze_blockers": [] if frozen else ["artifact hashes and power design remain tbd"],
        "analysis": {
            "k": 10,
            "failure_recall_threshold": 0.9,
            "alpha": 0.05,
            "bootstrap_seed": 20260713,
            "h1_minimum_risk_increase": 0.0,
            "power_target": 0.9,
            "retrieval_target_noninferiority_margin": 0.01,
            "evidence_sufficiency_noninferiority_margin": 0.01,
            "minimum_cost_reduction": 0.1,
            "maximum_p95_latency_ratio": 1.25,
            "maximum_entitlement_violations": 0,
            "minimum_corpora_with_geometry_gain": 4,
            "geometry_reference_model": "system-policy",
            "geometry_candidate_model": "full",
            "geometry_gain_metrics": [
                "log_loss_reduction",
                "brier_score_reduction",
                "auprc_gain",
            ],
            "geometry_gain_thresholds": {
                "log_loss_reduction": 0.0 if frozen else "tbd",
                "brier_score_reduction": 0.0 if frozen else "tbd",
                "auprc_gain": 0.0 if frozen else "tbd",
            },
            "low_geometry": (
                {"instability": 0.1, "lid": 1.0} if frozen else "tbd"
            ),
            "high_geometry": (
                {"instability": 0.9, "lid": 9.0} if frozen else "tbd"
            ),
            "cluster_unit": "query_family",
            "corpus_weighting": "equal",
            "interval_construction": "directional-one-sided-95",
            "gatekeeping": "intersection-union-primary-gates",
            "cost_estimand": "end-to-end-request-latency-family-relative-reduction",
            "bootstrap_replicates": 10_000,
            "nested_rows_per_family": 4 if frozen else "tbd",
            "fixed_corpora": list(FIXED_CORPORA),
            "evidence_corpora": list(EVIDENCE_CORPORA),
            "action_set": list(REGISTERED_ACTION_SET),
            "static_comparator_action": "hnsw-high" if frozen else "tbd",
            "power": {
                "model": "development-family-cluster-resampling",
                "joint_success_event": "h2-and-h3-all-gates-pass",
                "registered_endpoints": list(REGISTERED_POWER_ENDPOINTS),
                "dependence_source": (
                    "development query-family endpoint vectors" if frozen else "tbd"
                ),
                "effect_scenarios": (
                    ["registered-minimum-effects", "development-observed-effects"]
                    if frozen
                    else ["tbd"]
                ),
                "candidate_families_per_corpus": list(
                    REGISTERED_POWER_FAMILY_CANDIDATES
                ),
                "selected_families_per_corpus": 100 if frozen else "tbd",
                "simulation_seed": 71 if frozen else "tbd",
                "simulation_count": 5_000,
                "selected_joint_power_lower_bound": 0.91 if frozen else "tbd",
            },
        },
        "artifacts": _artifacts(frozen=frozen),
        "sealed_execution": {
            "reserve_fraction": 0.0,
            "custodian": "custodian@example.test" if frozen else "unassigned",
            "approval_environment": "confirmatory" if frozen else "tbd",
            "results_store": "s3://immutable-results" if frozen else "tbd",
            "runner_identity": _RUNNER_IDENTITY if frozen else "tbd",
            "code_commit": _COMMIT if frozen else "tbd",
            "runner_image": (
                f"ghcr.io/example/study@sha256:{'2' * 64}" if frozen else "tbd"
            ),
            "hardware": {
                "provider": "aws" if frozen else "tbd",
                "instance_type": "c7i.4xlarge" if frozen else "tbd",
                "cpu_model": "Intel Xeon Platinum 8488C" if frozen else "tbd",
                "logical_cores": 16 if frozen else "tbd",
                "memory_gib": 32 if frozen else "tbd",
                "accelerator": "none" if frozen else "tbd",
                "region": "us-east-1" if frozen else "tbd",
                "operating_system": "ubuntu-24.04" if frozen else "tbd",
            },
            "receipt_uri_template": (
                receipt_directory.resolve().as_uri() + "/{manifest_sha256}.json"
                if frozen
                else "tbd"
            ),
            "label_artifacts_withheld_until_prediction_receipt": True,
            "public_query_reidentification_risk": (
                "accepted-public-benchmark-limitation"
            ),
            "runner_network_access": "disabled",
            "interactive_access": "disabled",
        },
    }


def _artifact_for(payload: dict[str, object], role: str) -> dict[str, object]:
    return next(
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] == role
    )


def _verification_receipt_path(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    name: str = "artifact-verification.json",
    manifest_digest: str | None = None,
    omit_id: str | None = None,
    digest_override: tuple[str, str] | None = None,
    exact_override: tuple[str, bool] | None = None,
    add_unexpected: bool = False,
) -> Path:
    rows: list[VerifiedArtifact] = []
    for position, artifact in enumerate(payload["artifacts"]):  # type: ignore[union-attr]
        artifact_id = str(artifact["id"])
        if artifact_id == omit_id:
            continue
        digest = str(artifact["sha256"])
        if digest_override is not None and artifact_id == digest_override[0]:
            digest = digest_override[1]
        exact = not (
            exact_override is not None
            and artifact_id == exact_override[0]
            and exact_override[1] is False
        )
        rows.append(
            VerifiedArtifact(
                artifact_id=artifact_id,
                relative_path=f"objects/{position}.bin",
                kind="file" if exact else "directory",
                exact=exact,
                expected_sha256=digest,
                verified_sha256=digest,
                file_count=1,
                directory_count=0,
                byte_count=1,
                observed_file_count=1,
                observed_directory_count=0,
                observed_byte_count=1,
            )
        )
    if add_unexpected:
        rows.append(
            VerifiedArtifact(
                artifact_id="unexpected-artifact",
                relative_path="objects/unexpected.bin",
                kind="file",
                exact=True,
                expected_sha256="e" * 64,
                verified_sha256="e" * 64,
                file_count=1,
                directory_count=0,
                byte_count=1,
                observed_file_count=1,
                observed_directory_count=0,
                observed_byte_count=1,
            )
        )
    receipt = ArtifactVerificationReceipt(
        manifest_sha256=manifest_digest or manifest_sha256(payload),
        artifacts=tuple(rows),
    )
    receipt_root = tmp_path / "artifact-receipts"
    receipt_root.mkdir(exist_ok=True)
    target = receipt_root / name
    write_verification_receipt(receipt, target)
    return target


def _registration_receipt_path(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    manifest_digest: str | None = None,
    registered_at_utc: str = "2026-07-13T12:00:00+00:00",
    extra_field: bool = False,
) -> Path:
    manifest_value = manifest_digest or manifest_sha256(payload)
    registry_identity = "osf-registration:test-2026-07-13"
    registry_uri = "https://osf.io/registries/test-registration"
    record = tmp_path / "protocol-registration-record.json"
    registry_record = ProtocolRegistryRecord(
        manifest_sha256=manifest_value,
        protocol_version="0.3.0",
        registered_at_utc=registered_at_utc,
        registry_identity=registry_identity,
        registry_uri=registry_uri,
    )
    record.write_bytes(registry_record.canonical_bytes() + b"\n")
    receipt = ProtocolRegistrationReceipt(
        manifest_sha256=manifest_value,
        protocol_version="0.3.0",
        registered_at_utc=registered_at_utc,
        registry_identity=registry_identity,
        registry_uri=registry_uri,
        registry_record_sha256=registry_record.record_sha256,
    ).to_dict()
    if extra_field:
        receipt["unregistered_field"] = "forbidden"
    target = tmp_path / "protocol-registration.json"
    target.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return target


def _trusted_registry_record_fetcher(
    record_path: Path,
) -> Callable[[str, int], bytes]:
    """Return the explicit trusted transport seam used by deterministic tests."""

    encoded = record_path.read_bytes()

    def fetch(registry_uri: str, max_bytes: int) -> bytes:
        assert registry_uri == "https://osf.io/registries/test-registration"
        assert max_bytes == MAX_PROTOCOL_REGISTRY_RECORD_BYTES
        return encoded

    return fetch


def test_repository_draft_manifest_validates_with_explicit_blockers() -> None:
    payload = load_study_manifest("research/study-manifest.json")
    validate_study_manifest(payload)
    assert payload["status"] == "draft"
    assert payload["freeze_blockers"]


def test_draft_manifest_is_valid_but_cannot_open_sealed_run() -> None:
    payload = _manifest()
    validate_study_manifest(payload)
    with pytest.raises(StudyManifestError, match="status='frozen'"):
        validate_study_manifest(payload, require_frozen=True)
    payload["freeze_blockers"] = []
    with pytest.raises(StudyManifestError, match="must state its explicit freeze blockers"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("location", "field"),
    (
        ("root", "surprise"),
        ("analysis", "unregistered_gate"),
        ("power", "unregistered_assumption"),
        ("artifact", "mutable_tag"),
        ("hardware", "benchmark_mode"),
    ),
)
def test_closed_schema_rejects_unknown_fields(location: str, field: str) -> None:
    payload = _manifest()
    if location == "root":
        payload[field] = True
    elif location == "analysis":
        payload["analysis"][field] = True  # type: ignore[index]
    elif location == "power":
        payload["analysis"]["power"][field] = True  # type: ignore[index]
    elif location == "artifact":
        payload["artifacts"][0][field] = True  # type: ignore[index]
    else:
        payload["sealed_execution"]["hardware"][field] = True  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="unknown"):
        validate_study_manifest(payload)


def test_exact_artifact_roles_and_corpus_coverage_are_required() -> None:
    payload = _manifest(frozen=True)
    payload["artifacts"] = [  # type: ignore[assignment]
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] != "h1-predictive-model"
    ]
    with pytest.raises(StudyManifestError, match="h1-predictive-model.*exactly 1"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    normalizers = [
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] == "corpus-normalizer"
    ]
    normalizers[0]["corpus_id"] = normalizers[1]["corpus_id"]
    with pytest.raises(StudyManifestError, match="cover every fixed corpus exactly once"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    _artifact_for(payload, "sealed-labels")["role"] = "sealed-inputs"
    with pytest.raises(StudyManifestError, match="sealed-inputs.*exactly 5"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["artifacts"] = [  # type: ignore[assignment]
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] != "online-execution"
    ]
    with pytest.raises(StudyManifestError, match="online-execution.*exactly 5"):
        validate_study_manifest(payload)


def test_frozen_analysis_requires_registered_seed_and_geometry_profiles() -> None:
    payload = _manifest(frozen=True)
    payload["analysis"]["bootstrap_seed"] = 20260714  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="bootstrap_seed"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["analysis"]["low_geometry"] = "tbd"  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="low_geometry.*pinned"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["analysis"]["nested_rows_per_family"] = "tbd"  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="nested_rows_per_family.*pinned"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["analysis"]["nested_rows_per_family"] = 0  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="nested_rows_per_family.*at least 1"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["analysis"]["high_geometry"] = {  # type: ignore[index]
        "different-feature": 0.9
    }
    with pytest.raises(StudyManifestError, match="name identical features"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    input_artifact = _artifact_for(payload, "sealed-inputs")
    label_artifact = next(
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] == "sealed-labels"
        and artifact["corpus_id"] == input_artifact["corpus_id"]
    )
    label_artifact["uri"] = input_artifact["uri"]
    label_artifact["sha256"] = input_artifact["sha256"]
    with pytest.raises(StudyManifestError, match="must be separately pinned"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    input_artifact = _artifact_for(payload, "sealed-inputs")
    execution_artifact = next(
        artifact
        for artifact in payload["artifacts"]  # type: ignore[union-attr]
        if artifact["role"] == "online-execution"
        and artifact["corpus_id"] == input_artifact["corpus_id"]
    )
    execution_artifact["uri"] = input_artifact["uri"]
    execution_artifact["sha256"] = input_artifact["sha256"]
    with pytest.raises(StudyManifestError, match="online execution.*separately pinned"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    fit_data = _artifact_for(payload, "development-fit-data")
    calibration_data = _artifact_for(payload, "development-calibration-data")
    calibration_data["uri"] = fit_data["uri"]
    with pytest.raises(StudyManifestError, match="fit and calibration"):
        validate_study_manifest(payload)


def test_frozen_status_always_enforces_pins_and_rejects_minimal_artifacts() -> None:
    payload = _manifest(frozen=True)
    _artifact_for(payload, "primary-embedding")["sha256"] = "tbd"
    with pytest.raises(StudyManifestError, match="pinned sha256"):
        validate_study_manifest(payload)

    payload = _manifest(frozen=True)
    payload["artifacts"] = payload["artifacts"][:4]  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="requires exactly"):
        validate_study_manifest(payload)


def test_action_set_and_noninferiority_gates_are_exact() -> None:
    payload = _manifest()
    payload["analysis"]["action_set"].append("rerank")  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="registered action set"):
        validate_study_manifest(payload)

    for field in (
        "retrieval_target_noninferiority_margin",
        "evidence_sufficiency_noninferiority_margin",
    ):
        payload = _manifest()
        payload["analysis"][field] = 0.02  # type: ignore[index]
        with pytest.raises(StudyManifestError, match=field):
            validate_study_manifest(payload)


def test_registered_claim_and_geometry_baseline_are_exact() -> None:
    assert REGISTERED_PRIMARY_CLAIM == (
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

    payload = _manifest()
    payload["analysis"]["geometry_reference_model"] = "policy-only"  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="system-policy"):
        validate_study_manifest(payload)


@pytest.mark.parametrize("mutation", ("missing", "legacy"))
def test_joint_power_contract_is_closed(mutation: str) -> None:
    payload = _manifest()
    power = payload["analysis"]["power"]  # type: ignore[index]
    if mutation == "missing":
        del power["dependence_source"]
    else:
        power["favorable_event"] = "low-effort-retrieval-success"
    with pytest.raises(StudyManifestError, match="schema mismatch"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model", "beta-binomial", "development-family-cluster-resampling"),
        ("joint_success_event", "h2-only", "h2-and-h3-all-gates-pass"),
    ),
)
def test_joint_power_design_literals_are_exact(
    field: str,
    value: str,
    message: str,
) -> None:
    payload = _manifest()
    payload["analysis"]["power"][field] = value  # type: ignore[index]
    with pytest.raises(StudyManifestError, match=message):
        validate_study_manifest(payload)


@pytest.mark.parametrize("mutation", ("removed", "reordered", "appended"))
def test_joint_power_registered_endpoint_order_is_exact(mutation: str) -> None:
    payload = _manifest()
    endpoints = payload["analysis"]["power"]["registered_endpoints"]  # type: ignore[index]
    if mutation == "removed":
        endpoints.pop()
    elif mutation == "reordered":
        endpoints[0], endpoints[1] = endpoints[1], endpoints[0]
    else:
        endpoints.append("unregistered-endpoint")
    with pytest.raises(StudyManifestError, match="registered ordered joint endpoint"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    "candidate_grid",
    (
        [1],
        [2],
        [50, 25, 75, 100, 150, 200],
        [25, 50, 75, 100, 150],
        [25, 50, 75, 100, 150, 200, 250],
    ),
)
def test_joint_power_candidate_grid_is_exact(candidate_grid: list[int]) -> None:
    payload = _manifest()
    payload["analysis"]["power"][  # type: ignore[index]
        "candidate_families_per_corpus"
    ] = candidate_grid
    with pytest.raises(StudyManifestError, match="registered candidate grid"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ([], "non-empty array"),
        (["scenario", "scenario"], "duplicates"),
        ([""], "non-empty string"),
    ),
)
def test_joint_power_effect_scenarios_are_nonempty_unique_draftable_text(
    value: list[str],
    message: str,
) -> None:
    payload = _manifest()
    payload["analysis"]["power"]["effect_scenarios"] = value  # type: ignore[index]
    with pytest.raises(StudyManifestError, match=message):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dependence_source", "tbd", "must be pinned"),
        ("effect_scenarios", ["tbd"], "must be pinned"),
        ("simulation_seed", "tbd", "must be pinned"),
        ("selected_families_per_corpus", "tbd", "must be pinned"),
        ("selected_families_per_corpus", 1, "at least 2"),
        ("selected_families_per_corpus", 60, "registered candidate"),
        ("selected_joint_power_lower_bound", "tbd", "must be pinned"),
        ("selected_joint_power_lower_bound", 0.89, "power_target"),
        ("simulation_count", 4_999, "at least 5000"),
    ),
)
def test_frozen_power_assumptions_and_selected_family_count_are_enforced(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _manifest(frozen=True)
    payload["analysis"]["power"][field] = value  # type: ignore[index]
    with pytest.raises(StudyManifestError, match=message):
        validate_study_manifest(payload)


@pytest.mark.parametrize("reserve_fraction", (0.2, -0.01, 0.01))
def test_one_shot_sealed_execution_has_no_reserve_rescue(
    reserve_fraction: object,
) -> None:
    payload = _manifest()
    payload["sealed_execution"]["reserve_fraction"] = reserve_fraction  # type: ignore[index]
    with pytest.raises(StudyManifestError, match="reserve_fraction.*0.0"):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("runner_identity", "must be pinned"),
        ("code_commit", "full lowercase Git commit"),
        ("runner_image", "OCI SHA-256 digest"),
        ("hardware", "logical_cores.*pinned"),
        ("source_revision", "source-code artifact revision"),
        ("receipt", "manifest_sha256"),
    ),
)
def test_frozen_runner_and_receipt_contract_is_fully_pinned(
    mutation: str,
    message: str,
) -> None:
    payload = _manifest(frozen=True)
    sealed = payload["sealed_execution"]  # type: ignore[assignment]
    if mutation == "runner_identity":
        sealed["runner_identity"] = "tbd"
    elif mutation == "code_commit":
        sealed["code_commit"] = "short"
    elif mutation == "runner_image":
        sealed["runner_image"] = "ghcr.io/example/study:latest"
    elif mutation == "hardware":
        sealed["hardware"]["logical_cores"] = "tbd"
    elif mutation == "source_revision":
        _artifact_for(payload, "source-code")["revision"] = "v1.0.0"
    else:
        sealed["receipt_uri_template"] = "file:///tmp/receipt.json"
    with pytest.raises(StudyManifestError, match=message):
        validate_study_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "protocol_registration_receipt_uri",
            "https://example.test/registration",
            "absolute file URI",
        ),
        (
            "protocol_registration_receipt_sha256",
            "not-a-digest",
            "lowercase SHA-256",
        ),
        ("verification_receipt_uri", "https://example.test/receipt", "absolute file URI"),
        ("verification_receipt_uri", "file:relative.json", "absolute file URI"),
        ("verification_receipt_sha256", "not-a-digest", "lowercase SHA-256"),
    ),
)
def test_sealed_run_receipt_requires_a_canonical_verification_pointer(
    field: str,
    value: str,
    message: str,
) -> None:
    arguments = {
        "manifest_sha256": "a" * 64,
        "protocol_version": "0.3.0",
        "started_at_utc": "2026-07-13T12:00:00+00:00",
        "runner_identity": _RUNNER_IDENTITY,
        "code_commit": _COMMIT,
        "runner_image": f"ghcr.io/example/study@sha256:{'2' * 64}",
        "protocol_registration_receipt_uri": "file:///tmp/registration.json",
        "protocol_registration_receipt_sha256": "c" * 64,
        "protocol_registration_record_uri": "file:///tmp/registration-record.json",
        "verification_receipt_uri": "file:///tmp/verification.json",
        "verification_receipt_sha256": "b" * 64,
        "receipt_uri": "file:///tmp/run.json",
    }
    arguments[field] = value
    with pytest.raises(StudyManifestError, match=message):
        SealedRunReceipt(**arguments)


def test_protocol_registration_receipt_is_closed_and_externally_addressed(
    tmp_path: Path,
) -> None:
    payload = _manifest(frozen=True)
    path = _registration_receipt_path(tmp_path, payload)
    receipt = load_protocol_registration_receipt(path)
    assert receipt.manifest_sha256 == manifest_sha256(payload)
    assert receipt.registry_uri.startswith("https://")
    assert len(receipt.receipt_sha256) == 64

    _registration_receipt_path(tmp_path, payload, extra_field=True)
    with pytest.raises(StudyManifestError, match="unknown"):
        load_protocol_registration_receipt(path)


def test_builtin_registry_fetch_uses_verified_https_and_one_bounded_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_uri = "https://registry.example.test/records/protocol.json"
    expected = b'{"record":"exact"}\n'
    observed: dict[str, object] = {}

    class Response:
        headers = {"Content-Length": str(len(expected))}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getcode(self) -> int:
            return 200

        def geturl(self) -> str:
            return registry_uri

        def read(self, limit: int) -> bytes:
            observed["read_limit"] = limit
            return expected

    class Opener:
        def open(self, request: Request, *, timeout: float) -> Response:
            observed["request"] = request
            observed["timeout"] = timeout
            return Response()

    def build_opener(*handlers: object) -> Opener:
        observed["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(study_module.urllib_request, "build_opener", build_opener)
    fetched = study_module._fetch_protocol_registry_record(
        registry_uri,
        MAX_PROTOCOL_REGISTRY_RECORD_BYTES,
    )

    assert fetched == expected
    assert observed["read_limit"] == MAX_PROTOCOL_REGISTRY_RECORD_BYTES + 1
    request = observed["request"]
    assert isinstance(request, Request)
    assert request.full_url == registry_uri
    assert request.get_method() == "GET"
    assert request.get_header("Accept-encoding") == "identity"
    handlers = observed["handlers"]
    assert isinstance(handlers, tuple)
    assert any(
        isinstance(handler, study_module._NoProtocolRegistryRedirects)
        for handler in handlers
    )
    https_handler = next(
        handler
        for handler in handlers
        if isinstance(handler, study_module.urllib_request.HTTPSHandler)
    )
    assert https_handler._context.check_hostname is True
    assert https_handler._context.verify_mode == ssl.CERT_REQUIRED


def test_builtin_registry_fetch_refuses_redirects() -> None:
    handler = study_module._NoProtocolRegistryRedirects()
    with pytest.raises(StudyManifestError, match="refused HTTP redirect status 302"):
        handler.redirect_request(
            Request("https://registry.example.test/original"),
            None,
            302,
            "Found",
            {},
            "https://registry.example.test/substitute",
        )


@pytest.mark.parametrize("declared_oversize", (True, False))
def test_builtin_registry_fetch_rejects_oversize_headers_and_bodies(
    monkeypatch: pytest.MonkeyPatch,
    declared_oversize: bool,
) -> None:
    registry_uri = "https://registry.example.test/record.json"

    class Response:
        headers = {
            "Content-Length": str(
                MAX_PROTOCOL_REGISTRY_RECORD_BYTES + 1
                if declared_oversize
                else MAX_PROTOCOL_REGISTRY_RECORD_BYTES
            )
        }

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getcode(self) -> int:
            return 200

        def geturl(self) -> str:
            return registry_uri

        def read(self, limit: int) -> bytes:
            assert not declared_oversize
            return b"x" * limit

    class Opener:
        def open(self, request: Request, *, timeout: float) -> Response:
            del request, timeout
            return Response()

    monkeypatch.setattr(
        study_module.urllib_request,
        "build_opener",
        lambda *handlers: Opener(),
    )
    with pytest.raises(StudyManifestError, match="maximum byte limit"):
        study_module._fetch_protocol_registry_record(
            registry_uri,
            MAX_PROTOCOL_REGISTRY_RECORD_BYTES,
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        (URLError("offline"), "verified HTTPS"),
        (TimeoutError("timed out"), "verified HTTPS"),
    ),
)
def test_builtin_registry_fetch_fails_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    message: str,
) -> None:
    class Opener:
        def open(self, request: Request, *, timeout: float) -> None:
            del request, timeout
            raise failure

    monkeypatch.setattr(
        study_module.urllib_request,
        "build_opener",
        lambda *handlers: Opener(),
    )
    with pytest.raises(StudyManifestError, match=message):
        study_module._fetch_protocol_registry_record(
            "https://registry.example.test/record.json",
            MAX_PROTOCOL_REGISTRY_RECORD_BYTES,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("one-byte-mismatch", "digest does not match"),
        ("missing-newline", "digest does not match"),
        ("oversize", "maximum byte limit"),
        ("non-bytes", "must return bytes"),
        ("unavailable", "fetcher failed"),
    ),
)
def test_sealed_run_rejects_adversarial_remote_registry_records(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    receipt_root = tmp_path / "remote-revalidation-receipts"
    receipt_root.mkdir()
    payload = _manifest(frozen=True, receipt_root=receipt_root)
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    artifact_receipt = _verification_receipt_path(tmp_path, payload)
    registration_receipt = _registration_receipt_path(tmp_path, payload)
    registration_record = tmp_path / "protocol-registration-record.json"
    local_bytes = registration_record.read_bytes()

    def fetch(registry_uri: str, max_bytes: int) -> bytes:
        assert registry_uri == "https://osf.io/registries/test-registration"
        assert max_bytes == MAX_PROTOCOL_REGISTRY_RECORD_BYTES
        if mutation == "one-byte-mismatch":
            return local_bytes.replace(b"osf-registration", b"OSF-registration", 1)
        if mutation == "missing-newline":
            return local_bytes[:-1]
        if mutation == "oversize":
            return b"x" * (max_bytes + 1)
        if mutation == "non-bytes":
            return "not bytes"  # type: ignore[return-value]
        raise TimeoutError("registry unavailable")

    with pytest.raises(StudyManifestError, match=message):
        begin_sealed_run(
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt,
            protocol_registration_receipt_path=registration_receipt,
            protocol_registration_record_path=registration_record,
            trusted_registry_record_fetcher=fetch,
        )
    assert not (receipt_root / f"{digest}.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("manifest", "different manifest digest"),
        ("future", "cannot be in the future"),
        ("record", "record digest does not match"),
    ),
)
def test_sealed_run_requires_a_prior_external_protocol_registration(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    receipt_root = tmp_path / "registration-gated-receipts"
    receipt_root.mkdir()
    payload = _manifest(frozen=True, receipt_root=receipt_root)
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    artifact_receipt = _verification_receipt_path(tmp_path, payload)
    registration_receipt = _registration_receipt_path(
        tmp_path,
        payload,
        manifest_digest="f" * 64 if mutation == "manifest" else None,
        registered_at_utc=(
            "2999-01-01T00:00:00+00:00"
            if mutation == "future"
            else "2026-07-13T12:00:00+00:00"
        ),
    )
    registration_record = tmp_path / "protocol-registration-record.json"
    if mutation == "record":
        changed_record = ProtocolRegistryRecord(
            manifest_sha256=digest,
            protocol_version="0.3.0",
            registered_at_utc="2026-07-13T12:00:00+00:00",
            registry_identity="osf-registration:substituted",
            registry_uri="https://osf.io/registries/test-registration",
        )
        registration_record.write_bytes(changed_record.canonical_bytes() + b"\n")

    with pytest.raises(StudyManifestError, match=message):
        begin_sealed_run(
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt,
            protocol_registration_receipt_path=registration_receipt,
            protocol_registration_record_path=registration_record,
            trusted_registry_record_fetcher=_trusted_registry_record_fetcher(
                registration_record
            ),
        )
    assert not tuple(receipt_root.iterdir())


def test_sealed_run_uses_digest_derived_manifest_receipt_and_pinned_identity(
    tmp_path: Path,
) -> None:
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    payload = _manifest(frozen=True, receipt_root=receipt_root)
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    receipt_path = receipt_root / f"{digest}.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    artifact_receipt = _verification_receipt_path(tmp_path, payload)
    registration_receipt = _registration_receipt_path(tmp_path, payload)
    registration_record = tmp_path / "protocol-registration-record.json"
    fetch_calls: list[tuple[str, int]] = []

    def trusted_fetcher(registry_uri: str, max_bytes: int) -> bytes:
        fetch_calls.append((registry_uri, max_bytes))
        return registration_record.read_bytes()

    assert sealed_receipt_uri(payload) == receipt_path.resolve().as_uri()
    with pytest.raises(StudyManifestError, match="does not equal"):
        begin_sealed_run(
            manifest,
            lock,
            runner_identity="different-runner",
            artifact_verification_receipt_path=artifact_receipt,
            protocol_registration_receipt_path=registration_receipt,
            protocol_registration_record_path=(
                tmp_path / "protocol-registration-record.json"
            ),
            trusted_registry_record_fetcher=_trusted_registry_record_fetcher(
                tmp_path / "protocol-registration-record.json"
            ),
        )
    assert not receipt_path.exists()

    observed = begin_sealed_run(
        manifest,
        lock,
        runner_identity=_RUNNER_IDENTITY,
        artifact_verification_receipt_path=artifact_receipt,
        protocol_registration_receipt_path=registration_receipt,
        protocol_registration_record_path=registration_record,
        trusted_registry_record_fetcher=trusted_fetcher,
    )
    assert fetch_calls == [
        (
            "https://osf.io/registries/test-registration",
            MAX_PROTOCOL_REGISTRY_RECORD_BYTES,
        )
    ]
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert observed.manifest_sha256 == digest
    assert observed.receipt_uri == receipt_path.resolve().as_uri()
    assert observed.verification_receipt_uri == artifact_receipt.as_uri()
    assert (
        observed.verification_receipt_sha256
        == load_verification_receipt(artifact_receipt).receipt_sha256
    )
    assert stored["runner_identity"] == _RUNNER_IDENTITY
    assert stored["code_commit"] == _COMMIT
    assert stored["verification_receipt_uri"] == artifact_receipt.as_uri()
    assert stored["protocol_registration_receipt_uri"] == registration_receipt.as_uri()
    assert stored["protocol_registration_record_uri"] == (
        tmp_path / "protocol-registration-record.json"
    ).as_uri()
    assert (
        stored["protocol_registration_receipt_sha256"]
        == load_protocol_registration_receipt(registration_receipt).receipt_sha256
    )
    assert (
        stored["verification_receipt_sha256"]
        == observed.verification_receipt_sha256
    )
    assert load_sealed_run_receipt(receipt_path.resolve()) == observed
    relocated = (tmp_path / "relocated-run.json").resolve()
    relocated.write_bytes(receipt_path.read_bytes())
    with pytest.raises(StudyManifestError, match="manifest-derived receipt_uri"):
        load_sealed_run_receipt(relocated)
    with pytest.raises(
        StudyManifestError,
        match="one-shot execution has already been consumed",
    ) as error:
        begin_sealed_run(
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt,
            protocol_registration_receipt_path=registration_receipt,
            protocol_registration_record_path=registration_record,
            trusted_registry_record_fetcher=trusted_fetcher,
        )
    assert "reserve_fraction is 0.0" in str(error.value)
    assert "no rerun or rescue is permitted" in str(error.value)
    assert "use the registered reserve set" not in str(error.value)


def test_production_run_requires_fresh_local_artifact_revalidation(
    tmp_path: Path,
) -> None:
    with pytest.raises(StudyManifestError, match="fresh local artifact revalidation"):
        begin_sealed_run(
            tmp_path / "missing-manifest.json",
            tmp_path / "missing-lock.txt",
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=tmp_path / "missing-receipt.json",
            protocol_registration_receipt_path=tmp_path / "missing-registration.json",
            protocol_registration_record_path=tmp_path / "missing-record.json",
        )


def test_sealed_run_reopens_every_local_artifact_before_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_root = tmp_path / "fresh-verification-receipts"
    receipt_root.mkdir()
    payload = _manifest(frozen=True, receipt_root=receipt_root)
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    artifact_receipt_path = _verification_receipt_path(tmp_path, payload)
    admitted_receipt = load_verification_receipt(artifact_receipt_path)
    registration_receipt = _registration_receipt_path(tmp_path, payload)
    registration_record = tmp_path / "protocol-registration-record.json"
    artifact_root = tmp_path / "artifact-root"
    artifact_root.mkdir()
    artifact_map = tmp_path / "artifact-map.json"
    artifact_map.write_text("{}", encoding="utf-8")
    expected_pins = {
        str(artifact["id"]): str(artifact["sha256"])
        for artifact in payload["artifacts"]
    }
    specs = (object(),)
    calls: list[tuple[object, ...]] = []

    def load_map(path: Path, *, expected_sha256_by_id: object) -> object:
        calls.append(("map", path, expected_sha256_by_id))
        return specs

    def verify(
        root: Path,
        *,
        manifest_sha256: str,
        artifacts: object,
    ) -> ArtifactVerificationReceipt:
        calls.append(("verify", root, manifest_sha256, artifacts))
        return admitted_receipt

    monkeypatch.setattr(study_module, "load_local_artifact_map", load_map)
    monkeypatch.setattr(study_module, "verify_local_artifacts", verify)
    observed = begin_sealed_run(
        manifest,
        lock,
        runner_identity=_RUNNER_IDENTITY,
        artifact_verification_receipt_path=artifact_receipt_path,
        artifact_root=artifact_root,
        local_artifact_map_path=artifact_map,
        protocol_registration_receipt_path=registration_receipt,
        protocol_registration_record_path=registration_record,
        trusted_registry_record_fetcher=_trusted_registry_record_fetcher(
            registration_record
        ),
    )
    assert observed.verification_receipt_sha256 == admitted_receipt.receipt_sha256
    assert calls == [
        ("map", artifact_map, expected_pins),
        ("verify", artifact_root, digest, specs),
    ]

    mismatched = ArtifactVerificationReceipt(
        manifest_sha256=digest,
        artifacts=admitted_receipt.artifacts[:-1],
    )
    monkeypatch.setattr(
        study_module,
        "verify_local_artifacts",
        lambda *args, **kwargs: mismatched,
    )
    with pytest.raises(StudyManifestError, match="differs from the admitted receipt"):
        begin_sealed_run(
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt_path,
            artifact_root=artifact_root,
            local_artifact_map_path=artifact_map,
            protocol_registration_receipt_path=registration_receipt,
            protocol_registration_record_path=registration_record,
            trusted_registry_record_fetcher=_trusted_registry_record_fetcher(
                registration_record
            ),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("manifest", "different manifest digest"),
        ("missing", "cover every manifest artifact exactly"),
        ("extra", "cover every manifest artifact exactly"),
        ("digest", "digest mismatch"),
        ("inexact", "must be exact"),
    ),
)
def test_sealed_run_rejects_unbound_or_incomplete_artifact_receipts(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    receipt_root = tmp_path / "run-receipts"
    receipt_root.mkdir()
    payload = _manifest(frozen=True, receipt_root=receipt_root)
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    first_id = str(payload["artifacts"][0]["id"])  # type: ignore[index]
    options: dict[str, object] = {"name": f"{mutation}.json"}
    if mutation == "manifest":
        options["manifest_digest"] = "f" * 64
    elif mutation == "missing":
        options["omit_id"] = first_id
    elif mutation == "extra":
        options["add_unexpected"] = True
    elif mutation == "digest":
        options["digest_override"] = (first_id, "d" * 64)
    else:
        options["exact_override"] = (first_id, False)
    artifact_receipt = _verification_receipt_path(tmp_path, payload, **options)
    registration_receipt = _registration_receipt_path(tmp_path, payload)

    with pytest.raises(StudyManifestError, match=message):
        begin_sealed_run(
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt,
            protocol_registration_receipt_path=registration_receipt,
            protocol_registration_record_path=(
                tmp_path / "protocol-registration-record.json"
            ),
            trusted_registry_record_fetcher=_trusted_registry_record_fetcher(
                tmp_path / "protocol-registration-record.json"
            ),
        )
    assert not (receipt_root / f"{digest}.json").exists()


def test_sealed_run_refuses_a_symlinked_receipt_parent(tmp_path: Path) -> None:
    real_receipts = tmp_path / "real-receipts"
    real_receipts.mkdir()
    linked_receipts = tmp_path / "linked-receipts"
    linked_receipts.symlink_to(real_receipts, target_is_directory=True)
    payload = _manifest(frozen=True, receipt_root=linked_receipts)
    payload["sealed_execution"]["receipt_uri_template"] = (  # type: ignore[index]
        linked_receipts.as_uri() + "/{manifest_sha256}.json"
    )
    manifest = tmp_path / "study.json"
    lock = tmp_path / "study.sha256"
    digest = manifest_sha256(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    lock.write_text(digest + "\n", encoding="utf-8")
    artifact_receipt = _verification_receipt_path(tmp_path, payload)
    registration_receipt = _registration_receipt_path(tmp_path, payload)

    with pytest.raises(StudyManifestError, match="cannot write sealed run receipt safely"):
        begin_sealed_run(
            manifest,
            lock,
            runner_identity=_RUNNER_IDENTITY,
            artifact_verification_receipt_path=artifact_receipt,
            protocol_registration_receipt_path=registration_receipt,
            protocol_registration_record_path=(
                tmp_path / "protocol-registration-record.json"
            ),
            trusted_registry_record_fetcher=_trusted_registry_record_fetcher(
                tmp_path / "protocol-registration-record.json"
            ),
        )
    assert not tuple(real_receipts.iterdir())
