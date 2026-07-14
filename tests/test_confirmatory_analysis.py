from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

import fractal_ann_diagnostics.confirmatory_analysis as confirmatory_analysis
from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    VerifiedArtifact,
)
from fractal_ann_diagnostics.audit import GENESIS_RECORD_SHA256
from fractal_ann_diagnostics.confirmatory_analysis import (
    ACTION_PANEL_ROW_SCHEMA,
    ActionPanelAdmissionReceipt,
    ActionPanelAdmissionRecord,
    ActionPanelArtifact,
    ConfirmatoryAnalysisConfig,
    ConfirmatoryAnalysisError,
    ConfirmatoryInputArtifact,
    ConfirmatoryTrialRow,
    PreLabelActionRow,
    run_confirmatory_analysis,
)
from fractal_ann_diagnostics.confirmatory_modeling import (
    FeatureSchema,
    GeometryGainThresholds,
    LabeledFeatureBatch,
    canonical_h1_model_artifact_bytes,
    canonical_h2_model_suite_artifact_bytes,
    fit_frozen_model_suite,
)
from fractal_ann_diagnostics.label_separation import (
    JoinedEvaluationTrial,
    LabelSeparationError,
    OfflineEvaluationArtifact,
    OnlinePrediction,
    PredictionCompletionReceipt,
    SealedEvidenceBundle,
    SealedEvidenceLocation,
    SealedLabelArtifact,
    SealedTrialLabels,
    sealed_run_receipt_sha256,
)
from fractal_ann_diagnostics.study import (
    EVIDENCE_CORPORA,
    FIXED_CORPORA,
    REGISTERED_ACTION_SET,
    REGISTERED_POWER_FAMILY_CANDIDATES,
    REGISTERED_PRIMARY_CLAIM,
    SealedRunReceipt,
    manifest_sha256,
)

SCHEMA = FeatureSchema(
    system_numeric=("load",),
    system_categorical=("corpus",),
    policy_numeric=("allow_rate",),
    policy_categorical=(),
    geometry_numeric=("lid", "instability"),
    geometry_categorical=(),
    lid_feature="lid",
    instability_feature="instability",
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _panel_admission_receipt(
    panel: ActionPanelArtifact,
    *,
    query_partition_audit_sha256: str,
    runner_identity: str,
) -> ActionPanelAdmissionReceipt:
    records: list[ActionPanelAdmissionRecord] = []
    previous = GENESIS_RECORD_SHA256
    audit_hashes: list[str] = []
    mask_sha256 = _digest(f"{panel.corpus}-authorization-mask")
    environment_sha256 = _digest("test-policy-environment")
    document_universe_sha256 = _digest(f"{panel.corpus}-documents")
    audit_sequence = 0
    for row_index, row in enumerate(panel.rows):
        controller_policy_version = "test-policy-v1"
        controller_reasons = (f"execute {row.action}",)
        controller_risk_score = 0.2
        controller_sha256 = _canonical_digest(
            {
                "action": row.action,
                "policy_version": controller_policy_version,
                "reasons": list(controller_reasons),
                "risk_score": controller_risk_score,
            }
        )
        decision_id = f"decision-{row_index}"
        request_sha256 = _digest(f"{panel.corpus}-request-{row_index}")
        authorization_sha256 = _canonical_digest(
            {
                "available": True,
                "decision_id": decision_id,
                "document_universe_sha256": document_universe_sha256,
                "environment_sha256": environment_sha256,
                "mask_sha256": mask_sha256,
                "mask_size": panel.document_count,
                "policy_version": controller_policy_version,
                "request_sha256": request_sha256,
            }
        )
        if row.execution_state == "failed":
            record_audit_sequence = None
            record_previous = None
            record_sha256 = None
            failure_started = 1_000_000_000 + row_index * 1_000_000_000
            failure_finished = failure_started + int(
                row.request_latency_ms * 1_000_000
            )
            failure_runner = runner_identity
            failure_timing_sha256 = _canonical_digest(
                {
                    "action": row.action,
                    "authorization_decision_sha256": authorization_sha256,
                    "controller_decision_sha256": controller_sha256,
                    "failure_code": row.failure_state,
                    "family_key": row.family_key,
                    "finished_monotonic_ns": failure_finished,
                    "runner_identity": failure_runner,
                    "schema_version": "fractal-runner-failure-timing-v1",
                    "started_monotonic_ns": failure_started,
                    "trial_key": row.trial_key,
                }
            )
        else:
            assert row.audit_record_sha256 is not None
            record_audit_sequence = audit_sequence
            record_previous = previous
            record_sha256 = row.audit_record_sha256
            failure_started = None
            failure_finished = None
            failure_runner = None
            failure_timing_sha256 = None
            audit_hashes.append(row.audit_record_sha256)
            previous = row.audit_record_sha256
            audit_sequence += 1
        records.append(
            ActionPanelAdmissionRecord(
                trial_key=row.trial_key,
                family_key=row.family_key,
                action=row.action,
                action_order=row.action_order,
                controller_selected=row.controller_selected,
                execution_state=row.execution_state,
                controller_risk_score=controller_risk_score,
                controller_reasons=controller_reasons,
                controller_policy_version=controller_policy_version,
                controller_decision_sha256=controller_sha256,
                authorization_decision_id=decision_id,
                authorization_request_sha256=request_sha256,
                authorization_mask_sha256=mask_sha256,
                authorization_mask_size=panel.document_count,
                authorization_decision_sha256=authorization_sha256,
                policy_available=True,
                environment_sha256=environment_sha256,
                document_universe_sha256=document_universe_sha256,
                audit_sequence=record_audit_sequence,
                audit_previous_record_sha256=record_previous,
                audit_record_sha256=record_sha256,
                failure_code=(
                    row.failure_state if row.execution_state == "failed" else None
                ),
                failure_started_monotonic_ns=failure_started,
                failure_finished_monotonic_ns=failure_finished,
                failure_runner_identity=failure_runner,
                failure_timing_receipt_sha256=failure_timing_sha256,
            )
        )
    return ActionPanelAdmissionReceipt(
        manifest_sha256=panel.manifest_sha256,
        run_receipt_sha256=panel.run_receipt_sha256,
        execution_artifact_sha256=panel.execution_artifact_sha256,
        action_panel_artifact_sha256=panel.artifact_sha256,
        corpus=panel.corpus,
        query_partition_audit_sha256=query_partition_audit_sha256,
        partition_label="primary",
        audit_head_sha256=audit_hashes[-1],
        audit_chain_length=len(audit_hashes),
        audit_record_sha256s=tuple(audit_hashes),
        records=tuple(records),
    )


def _development_batch(partition: str, prefix: str, families: int) -> LabeledFeatureBatch:
    features: list[list[object]] = []
    labels: list[int] = []
    corpus_ids: list[str] = []
    family_ids: list[str] = []
    row_ids: list[str] = []
    for corpus_id in FIXED_CORPORA:
        for family_index in range(families):
            high_risk = family_index % 2 == 1
            family_id = f"{prefix}-{corpus_id}-{family_index}"
            features.append(
                [
                    1.0,
                    corpus_id,
                    0.5,
                    9.0 if high_risk else 1.0,
                    0.9 if high_risk else 0.1,
                ]
            )
            labels.append(int(high_risk))
            corpus_ids.append(corpus_id)
            family_ids.append(family_id)
            row_ids.append(f"{family_id}-row")
    return LabeledFeatureBatch(
        partition=partition,
        feature_names=SCHEMA.input_features,
        features=features,
        corpus_ids=corpus_ids,
        family_ids=family_ids,
        row_ids=row_ids,
        labels=labels,
    )


@pytest.fixture(scope="module")
def suite():
    return fit_frozen_model_suite(
        _development_batch("development-fit", "fit", 10),
        _development_batch("development-calibration", "cal", 6),
        schema=SCHEMA,
        random_seed=37,
    )


@pytest.fixture(scope="module")
def config() -> ConfirmatoryAnalysisConfig:
    return ConfirmatoryAnalysisConfig(
        fixed_corpora=FIXED_CORPORA,
        evidence_corpora=EVIDENCE_CORPORA,
        action_set=REGISTERED_ACTION_SET,
        static_comparator_action="hnsw-high",
        low_geometry=(("lid", 1.0), ("instability", 0.1)),
        high_geometry=(("lid", 9.0), ("instability", 0.9)),
        geometry_gain_thresholds=GeometryGainThresholds(
            log_loss_reduction=0.001,
            brier_score_reduction=0.001,
            auprc_gain=0.001,
        ),
        selected_families_per_corpus=25,
        nested_rows_per_family=1,
        bootstrap_replicates=10_000,
        bootstrap_seed=20260713,
    )


def test_direct_config_requires_two_families_for_bootstrap(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    with pytest.raises(ConfirmatoryAnalysisError, match="at least two"):
        replace(config, selected_families_per_corpus=1)


_NON_CORPUS_ARTIFACTS = (
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


def _frozen_manifest(
    config: ConfirmatoryAnalysisConfig,
    *,
    suite=None,
    selected_families_per_corpus: int | None = None,
    nested_rows_per_family: int | None = None,
    artifact_digest_overrides: dict[tuple[str, str], str] | None = None,
) -> dict[str, object]:
    selected_families = (
        config.selected_families_per_corpus
        if selected_families_per_corpus is None
        else selected_families_per_corpus
    )
    nested_rows = (
        config.nested_rows_per_family
        if nested_rows_per_family is None
        else nested_rows_per_family
    )
    artifacts: list[dict[str, object]] = []
    digest_overrides = artifact_digest_overrides or {}
    for role, kind in (
        ("sealed-inputs", "dataset"),
        ("sealed-labels", "dataset"),
        ("online-execution", "execution"),
        ("corpus-normalizer", "normalizer"),
        ("policy-workload", "policy-data"),
    ):
        for corpus_id in FIXED_CORPORA:
            artifact_id = f"{corpus_id}-{role}"
            digest = digest_overrides.get((role, corpus_id), _digest(artifact_id))
            artifacts.append(
                {
                    "kind": kind,
                    "id": artifact_id,
                    "uri": f"https://example.test/{artifact_id}",
                    "revision": "v1.0.0",
                    "sha256": digest,
                    "license": "MIT",
                    "role": role,
                    "corpus_id": corpus_id,
                }
            )
    for role, kind in _NON_CORPUS_ARTIFACTS:
        if role == "h1-predictive-model" and suite is not None:
            digest = hashlib.sha256(
                canonical_h1_model_artifact_bytes(suite)
            ).hexdigest()
        elif role == "h2-model-suite" and suite is not None:
            digest = hashlib.sha256(
                canonical_h2_model_suite_artifact_bytes(suite)
            ).hexdigest()
        else:
            digest = _digest(role)
        artifacts.append(
            {
                "kind": kind,
                "id": role,
                "uri": f"https://example.test/{role}",
                "revision": "c" * 40 if role == "source-code" else "v1.0.0",
                "sha256": digest,
                "license": "MIT",
                "role": role,
            }
        )
    return {
        "schema_version": "1.0",
        "protocol_version": "0.3.0",
        "status": "frozen",
        "claim_scope": "suite-conditional-retrieval-control",
        "primary_claim": REGISTERED_PRIMARY_CLAIM,
        "freeze_blockers": [],
        "analysis": {
            "k": config.k,
            "failure_recall_threshold": config.failure_recall_threshold,
            "alpha": config.alpha,
            "bootstrap_seed": config.bootstrap_seed,
            "h1_minimum_risk_increase": config.h1_minimum_risk_increase,
            "power_target": 0.9,
            "retrieval_target_noninferiority_margin": (
                config.retrieval_target_noninferiority_margin
            ),
            "evidence_sufficiency_noninferiority_margin": (
                config.evidence_sufficiency_noninferiority_margin
            ),
            "minimum_cost_reduction": config.minimum_cost_reduction,
            "maximum_p95_latency_ratio": config.maximum_p95_latency_ratio,
            "maximum_entitlement_violations": config.maximum_entitlement_violations,
            "minimum_corpora_with_geometry_gain": (
                config.minimum_corpora_with_geometry_gain
            ),
            "nested_rows_per_family": nested_rows,
            "geometry_reference_model": "system-policy",
            "geometry_candidate_model": "full",
            "geometry_gain_metrics": [
                "log_loss_reduction",
                "brier_score_reduction",
                "auprc_gain",
            ],
            "geometry_gain_thresholds": {
                "log_loss_reduction": (
                    config.geometry_gain_thresholds.log_loss_reduction
                ),
                "brier_score_reduction": (
                    config.geometry_gain_thresholds.brier_score_reduction
                ),
                "auprc_gain": config.geometry_gain_thresholds.auprc_gain,
            },
            "low_geometry": dict(config.low_geometry),
            "high_geometry": dict(config.high_geometry),
            "cluster_unit": "query_family",
            "corpus_weighting": "equal",
            "interval_construction": "directional-one-sided-95",
            "gatekeeping": "intersection-union-primary-gates",
            "cost_estimand": "end-to-end-request-latency-family-relative-reduction",
            "bootstrap_replicates": config.bootstrap_replicates,
            "fixed_corpora": list(config.fixed_corpora),
            "evidence_corpora": list(config.evidence_corpora),
            "action_set": list(config.action_set),
            "static_comparator_action": config.static_comparator_action,
            "power": {
                "model": "development-family-cluster-resampling",
                "joint_success_event": "h2-and-h3-all-gates-pass",
                "registered_endpoints": [
                    "h2-log-loss-reduction",
                    "h2-brier-score-reduction",
                    "h2-auprc-gain",
                    "h2-four-of-five-consistency",
                    "h3-family-relative-latency-reduction",
                    "h3-retrieval-target-noninferiority",
                    "h3-complete-evidence-noninferiority",
                    "h3-family-mean-p95-latency-ratio",
                    "h3-zero-entitlement-violations",
                ],
                "dependence_source": "test development query-family endpoint vectors",
                "effect_scenarios": [
                    "registered-minimum-effects",
                    "development-observed-effects",
                ],
                "candidate_families_per_corpus": list(
                    REGISTERED_POWER_FAMILY_CANDIDATES
                ),
                "selected_families_per_corpus": selected_families,
                "simulation_seed": 71,
                "simulation_count": 5_000,
                "selected_joint_power_lower_bound": 0.91,
            },
        },
        "artifacts": artifacts,
        "sealed_execution": {
            "reserve_fraction": 0.0,
            "custodian": "custodian@example.test",
            "approval_environment": "confirmatory",
            "results_store": "s3://immutable-results",
            "runner_identity": "confirmatory-test-runner",
            "code_commit": "c" * 40,
            "runner_image": f"ghcr.io/example/runner@sha256:{'d' * 64}",
            "hardware": {
                "provider": "aws",
                "instance_type": "c7i.4xlarge",
                "cpu_model": "Intel Xeon Platinum 8488C",
                "logical_cores": 16,
                "memory_gib": 32,
                "accelerator": "none",
                "region": "us-east-1",
                "operating_system": "ubuntu-24.04",
            },
            "receipt_uri_template": "file:///receipts/{manifest_sha256}.json",
            "label_artifacts_withheld_until_prediction_receipt": True,
            "public_query_reidentification_risk": (
                "accepted-public-benchmark-limitation"
            ),
            "runner_network_access": "disabled",
            "interactive_access": "disabled",
        },
    }


def _verification_receipt(
    manifest: dict[str, object],
) -> ArtifactVerificationReceipt:
    rows = tuple(
        VerifiedArtifact(
            artifact_id=str(artifact["id"]),
            relative_path=f"objects/{index}.bin",
            kind="file",
            exact=True,
            expected_sha256=str(artifact["sha256"]),
            verified_sha256=str(artifact["sha256"]),
            file_count=1,
            directory_count=0,
            byte_count=1,
            observed_file_count=1,
            observed_directory_count=0,
            observed_byte_count=1,
        )
        for index, artifact in enumerate(manifest["artifacts"])
    )
    return ArtifactVerificationReceipt(
        manifest_sha256=manifest_sha256(manifest),
        artifacts=rows,
    )


def _run_receipt(
    manifest: dict[str, object],
    receipt: ArtifactVerificationReceipt,
) -> SealedRunReceipt:
    manifest_digest = manifest_sha256(manifest)
    return SealedRunReceipt(
        manifest_sha256=manifest_digest,
        protocol_version="0.3.0",
        started_at_utc="2026-07-13T22:00:00+00:00",
        runner_identity="confirmatory-test-runner",
        code_commit="c" * 40,
        runner_image=f"ghcr.io/example/runner@sha256:{'d' * 64}",
        protocol_registration_receipt_uri="file:///receipts/protocol.json",
        protocol_registration_receipt_sha256="e" * 64,
        protocol_registration_record_uri="file:///receipts/external-record.json",
        verification_receipt_uri="file:///receipts/artifacts.json",
        verification_receipt_sha256=receipt.receipt_sha256,
        receipt_uri=f"file:///receipts/{manifest_digest}.json",
    )


def _prelabel_rows(
    corpus_id: str,
    *,
    entitlement_event: bool,
    families_per_corpus: int = 12,
    nested_rows_per_family: int = 1,
    force_low_effort_success: bool = False,
    empty_authorized_truth: bool = False,
) -> tuple[PreLabelActionRow, ...]:
    rows: list[PreLabelActionRow] = []
    for family_index in range(families_per_corpus):
        high_risk = family_index % 2 == 1
        family_key = _digest(f"family-{corpus_id}-{family_index}")
        for nested_index in range(nested_rows_per_family):
            trial_key = _digest(
                f"trial-{corpus_id}-{family_index}-{nested_index}"
            )
            for action_order, action in enumerate(REGISTERED_ACTION_SET):
                selected = action == (
                    "exact-authorized" if high_risk else "hnsw-low"
                )
                if action == "abstain":
                    state = "abstained"
                    failure_state = "registered-abstention"
                    latency = 1.0
                    returned = ()
                elif (
                    action == "hnsw-low"
                    and high_risk
                    and family_index == 1
                    and nested_index == 0
                    and not force_low_effort_success
                ):
                    state = "failed"
                    failure_state = "backend-timeout"
                    latency = 7.0
                    returned = ()
                else:
                    state = "completed"
                    failure_state = None
                    latency = {
                        "hnsw-low": 5.0,
                        "hnsw-high": 10.0,
                        "exact-authorized": 6.0,
                    }[action]
                    if action == "exact-authorized":
                        returned = () if empty_authorized_truth else (0,)
                    elif action == "hnsw-low":
                        if force_low_effort_success:
                            returned = () if empty_authorized_truth else (0,)
                        elif empty_authorized_truth and not high_risk:
                            returned = ()
                        else:
                            returned = (1,) if high_risk else (0,)
                    else:
                        returned = (0,)
                rows.append(
                    PreLabelActionRow(
                        trial_key=trial_key,
                        family_key=family_key,
                        action=action,
                        action_order=action_order,
                        audit_record_sha256=(
                            None
                            if state == "failed"
                            else _digest(
                                "audit-"
                                f"{corpus_id}-{family_index}-{nested_index}-{action}"
                            )
                        ),
                        execution_state=state,
                        failure_state=failure_state,
                        controller_selected=selected,
                        request_latency_ms=latency,
                        entitlement_violations=int(
                            entitlement_event
                            and corpus_id == FIXED_CORPORA[-1]
                            and family_index == families_per_corpus - 1
                            and nested_index == nested_rows_per_family - 1
                            and action == "exact-authorized"
                        ),
                        returned_document_ids=returned,
                        feature_values=(
                            (
                                1.0,
                                corpus_id,
                                0.5,
                                9.0 if high_risk else 1.0,
                                0.9 if high_risk else 0.1,
                            )
                            if action == "hnsw-low"
                            else None
                        ),
                    )
                )
    return tuple(rows)


def _sealed_labels_for_panel_rows(
    corpus_id: str,
    *,
    execution_artifact_sha256: str,
    rows: tuple[PreLabelActionRow, ...],
    relevant_document_ids: tuple[int, ...] = (0,),
) -> SealedLabelArtifact:
    rows_by_trial: dict[str, list[PreLabelActionRow]] = {}
    for row in rows:
        rows_by_trial.setdefault(row.trial_key, []).append(row)
    labels: list[SealedTrialLabels] = []
    for trial_key, action_rows in rows_by_trial.items():
        selected = next(row for row in action_rows if row.controller_selected)
        bundles = (
            (
                SealedEvidenceBundle(
                    bundle_id=f"bundle-{trial_key}",
                    locations=(
                        SealedEvidenceLocation(
                            document_id=0,
                            source_uri=f"https://example.test/{corpus_id}/0",
                            locator="document",
                            content_hash=None,
                        ),
                    ),
                ),
            )
            if corpus_id in EVIDENCE_CORPORA
            else ()
        )
        labels.append(
            SealedTrialLabels(
                trial_key=trial_key,
                family_key=selected.family_key,
                answer=None,
                relevant_document_ids=relevant_document_ids,
                evidence_bundles=bundles,
                label_metadata=(),
            )
        )
    return SealedLabelArtifact(
        execution_artifact_sha256=execution_artifact_sha256,
        key_id="test-custodian-key",
        corpus=corpus_id,
        stage="sealed",
        document_count=3,
        labels=tuple(labels),
    )


def _bound_input(
    config: ConfirmatoryAnalysisConfig,
    *,
    entitlement_event: bool = False,
    suite=None,
    registered_families_per_corpus: int | None = None,
    registered_nested_rows_per_family: int | None = None,
    observed_families_per_corpus: int = 25,
    observed_nested_rows_per_family: int = 1,
    label_relevant_document_ids: tuple[int, ...] = (0,),
    one_class_success_corpus: str | None = None,
    empty_authorized_truth: bool = False,
) -> ConfirmatoryInputArtifact:
    rows_by_corpus = {
        corpus_id: _prelabel_rows(
            corpus_id,
            entitlement_event=entitlement_event,
            families_per_corpus=observed_families_per_corpus,
            nested_rows_per_family=observed_nested_rows_per_family,
            force_low_effort_success=corpus_id == one_class_success_corpus,
            empty_authorized_truth=empty_authorized_truth,
        )
        for corpus_id in FIXED_CORPORA
    }
    execution_sha_by_corpus = {
        corpus_id: _digest(f"{corpus_id}-online-execution")
        for corpus_id in FIXED_CORPORA
    }
    labels_by_corpus = {
        corpus_id: _sealed_labels_for_panel_rows(
            corpus_id,
            execution_artifact_sha256=execution_sha_by_corpus[corpus_id],
            rows=rows_by_corpus[corpus_id],
            relevant_document_ids=label_relevant_document_ids,
        )
        for corpus_id in FIXED_CORPORA
    }
    manifest = _frozen_manifest(
        config,
        suite=suite,
        selected_families_per_corpus=registered_families_per_corpus,
        nested_rows_per_family=registered_nested_rows_per_family,
        artifact_digest_overrides={
            ("sealed-labels", corpus_id): labels_by_corpus[corpus_id].artifact_sha256
            for corpus_id in FIXED_CORPORA
        },
    )
    verification_receipt = _verification_receipt(manifest)
    run_receipt = _run_receipt(manifest, verification_receipt)
    run_digest = sealed_run_receipt_sha256(run_receipt)
    manifest_digest = manifest_sha256(manifest)
    pinned_execution_sha_by_corpus = {
        str(artifact["corpus_id"]): str(artifact["sha256"])
        for artifact in manifest["artifacts"]
        if artifact["role"] == "online-execution"
    }
    query_partition_audit_sha256 = next(
        str(artifact["sha256"])
        for artifact in manifest["artifacts"]
        if artifact["role"] == "query-partition-audit"
    )
    panels: list[ActionPanelArtifact] = []
    admissions: list[ActionPanelAdmissionReceipt] = []
    completions: list[PredictionCompletionReceipt] = []
    evaluations: list[OfflineEvaluationArtifact] = []
    for corpus_id in FIXED_CORPORA:
        execution_digest = pinned_execution_sha_by_corpus[corpus_id]
        prediction_digest = _digest(f"prediction-{corpus_id}")
        panel = ActionPanelArtifact(
            manifest_sha256=manifest_digest,
            run_receipt_sha256=run_digest,
            execution_artifact_sha256=execution_digest,
            corpus=corpus_id,
            stage="sealed",
            document_count=3,
            action_set=REGISTERED_ACTION_SET,
            rows=rows_by_corpus[corpus_id],
        )
        panel_trials: dict[str, list[PreLabelActionRow]] = {}
        for row in panel.rows:
            panel_trials.setdefault(row.trial_key, []).append(row)
        joined: list[JoinedEvaluationTrial] = []
        for trial_key, action_rows in panel_trials.items():
            selected = next(row for row in action_rows if row.controller_selected)
            prediction = OnlinePrediction(
                trial_key=trial_key,
                family_key=selected.family_key,
                returned_document_ids=selected.returned_document_ids,
            )
            labels = next(
                row
                for row in labels_by_corpus[corpus_id].labels
                if row.trial_key == trial_key
            )
            joined.append(JoinedEvaluationTrial(prediction=prediction, labels=labels))
        completion = PredictionCompletionReceipt(
            manifest_sha256=manifest_digest,
            run_receipt_sha256=run_digest,
            execution_artifact_sha256=execution_digest,
            prediction_artifact_sha256=prediction_digest,
            action_panel_binding=panel.completion_binding(),
            prediction_count=len(joined),
            corpus=corpus_id,
            stage="sealed",
            external_anchor_identity="test-transparency-log",
            external_anchor_uri=f"https://rekor.example.test/entries/{corpus_id}",
            anchored_at_utc="2026-07-13T22:01:00+00:00",
        )
        evaluation = OfflineEvaluationArtifact(
            manifest_sha256=manifest_digest,
            run_receipt_sha256=run_digest,
            execution_artifact_sha256=execution_digest,
            prediction_artifact_sha256=prediction_digest,
            prediction_completion_receipt_sha256=completion.receipt_sha256,
            sealed_label_artifact_sha256=labels_by_corpus[corpus_id].artifact_sha256,
            corpus=corpus_id,
            stage="sealed",
            trials=tuple(joined),
        )
        panels.append(panel)
        admissions.append(
            _panel_admission_receipt(
                panel,
                query_partition_audit_sha256=query_partition_audit_sha256,
                runner_identity=run_receipt.runner_identity,
            )
        )
        completions.append(completion)
        evaluations.append(evaluation)
    return ConfirmatoryInputArtifact(
        run_receipt=run_receipt,
        frozen_manifest=manifest,
        artifact_verification_receipt=verification_receipt,
        completion_receipts=tuple(completions),
        offline_evaluations=tuple(evaluations),
        sealed_label_artifacts=tuple(
            labels_by_corpus[corpus_id] for corpus_id in FIXED_CORPORA
        ),
        action_panels=tuple(panels),
        action_panel_admission_receipts=tuple(admissions),
    )


def test_prelabel_panel_contains_no_post_label_outcome_fields(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config)
    payload = inputs.action_panels[0].rows[0].to_dict()
    assert payload["schema_version"] == ACTION_PANEL_ROW_SCHEMA
    assert not {
        "answer",
        "evidence_sufficient",
        "gold",
        "labels",
        "recall_at_k",
        "relevance",
    }.intersection(payload)

    derived = inputs.analysis_rows()[0]
    restored = ConfirmatoryTrialRow.from_dict(derived.to_dict())
    assert restored.to_dict() == derived.to_dict()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"request_latency_ms": float("inf")}, "must be finite"),
        (
            {"execution_state": "failed", "failure_state": None},
            "explicit failure_state",
        ),
    ],
)
def test_nonfinite_outcomes_and_implicit_failures_are_rejected(
    updates: dict[str, object],
    message: str,
) -> None:
    row = _prelabel_rows(FIXED_CORPORA[0], entitlement_event=False)[0]
    with pytest.raises(ConfirmatoryAnalysisError, match=message):
        replace(row, **updates)


def test_panel_rejects_duplicate_missing_pairs_and_action_order() -> None:
    rows = _prelabel_rows(FIXED_CORPORA[0], entitlement_event=False)
    common = {
        "manifest_sha256": "a" * 64,
        "run_receipt_sha256": "b" * 64,
        "execution_artifact_sha256": "c" * 64,
        "corpus": FIXED_CORPORA[0],
        "stage": "sealed",
        "document_count": 3,
        "action_set": REGISTERED_ACTION_SET,
    }
    with pytest.raises(ConfirmatoryAnalysisError, match="duplicate"):
        ActionPanelArtifact(rows=rows + (rows[0],), **common)
    with pytest.raises(ConfirmatoryAnalysisError, match="complete action set"):
        ActionPanelArtifact(rows=rows[1:], **common)
    wrong_order = (replace(rows[0], action_order=1),) + rows[1:]
    with pytest.raises(ConfirmatoryAnalysisError, match="action-set order"):
        ActionPanelArtifact(rows=wrong_order, **common)

    exact_index = next(
        index for index, row in enumerate(rows) if row.action == "exact-authorized"
    )
    failed_exact = list(rows)
    failed_exact[exact_index] = replace(
        failed_exact[exact_index],
        audit_record_sha256=None,
        execution_state="failed",
        failure_state="oracle-failure",
        returned_document_ids=(),
    )
    with pytest.raises(ConfirmatoryAnalysisError, match="completed exact-authorized"):
        ActionPanelArtifact(rows=tuple(failed_exact), **common)


def test_typed_input_rejects_missing_corpus_and_receipt_panel_mismatch(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config)
    with pytest.raises(ConfirmatoryAnalysisError, match="frozen corpus suite"):
        replace(
            inputs,
            completion_receipts=inputs.completion_receipts[:-1],
        )
    with pytest.raises(ConfirmatoryAnalysisError, match="frozen corpus suite"):
        replace(
            inputs,
            action_panel_admission_receipts=(
                inputs.action_panel_admission_receipts[:-1]
            ),
        )

    first_completion = inputs.completion_receipts[0]
    mismatched = (
        replace(
            first_completion,
            action_panel_binding=replace(
                first_completion.action_panel_binding,
                action_panel_artifact_sha256="9" * 64,
            ),
        ),
    ) + inputs.completion_receipts[1:]
    with pytest.raises(ConfirmatoryAnalysisError, match="does not bind the action panel"):
        replace(
            inputs,
            completion_receipts=mismatched,
        )


def test_typed_input_requires_primary_partition_receipt_and_run_runner(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config)
    first_receipt = inputs.action_panel_admission_receipts[0]

    with pytest.raises(ConfirmatoryAnalysisError, match="primary partition"):
        replace(
            inputs,
            action_panel_admission_receipts=(
                replace(first_receipt, partition_label="reserve"),
            )
            + inputs.action_panel_admission_receipts[1:],
        )

    with pytest.raises(ConfirmatoryAnalysisError, match="query-partition audit"):
        replace(
            inputs,
            action_panel_admission_receipts=(
                replace(first_receipt, query_partition_audit_sha256="9" * 64),
            )
            + inputs.action_panel_admission_receipts[1:],
        )

    failed = next(
        record
        for record in first_receipt.records
        if record.execution_state == "failed"
    )
    other_runner = "other-runner"
    changed_timing_sha256 = _canonical_digest(
        {
            "action": failed.action,
            "authorization_decision_sha256": failed.authorization_decision_sha256,
            "controller_decision_sha256": failed.controller_decision_sha256,
            "failure_code": failed.failure_code,
            "family_key": failed.family_key,
            "finished_monotonic_ns": failed.failure_finished_monotonic_ns,
            "runner_identity": other_runner,
            "schema_version": "fractal-runner-failure-timing-v1",
            "started_monotonic_ns": failed.failure_started_monotonic_ns,
            "trial_key": failed.trial_key,
        }
    )
    changed_failed = replace(
        failed,
        failure_runner_identity=other_runner,
        failure_timing_receipt_sha256=changed_timing_sha256,
    )
    changed_records = tuple(
        changed_failed if record == failed else record
        for record in first_receipt.records
    )
    changed_receipt = replace(first_receipt, records=changed_records)
    with pytest.raises(ConfirmatoryAnalysisError, match="another runner"):
        replace(
            inputs,
            action_panel_admission_receipts=(changed_receipt,)
            + inputs.action_panel_admission_receipts[1:],
        )


def test_typed_input_derives_config_from_manifest_and_binds_receipt(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config)

    assert inputs.frozen_config == config
    assert not ConfirmatoryInputArtifact.__dataclass_fields__["frozen_config"].init
    assert inputs.to_dict()["artifact_verification_receipt_sha256"] == (
        inputs.artifact_verification_receipt.receipt_sha256
    )
    panels_by_corpus = {panel.corpus: panel for panel in inputs.action_panels}
    for corpus_digest in inputs.corpus_input_digests:
        assert (
            panels_by_corpus[corpus_digest.corpus_id].execution_artifact_sha256
            == corpus_digest.online_execution_artifact_sha256
        )


def test_typed_input_rejects_execution_digest_not_pinned_as_online_execution(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config)
    wrong_digest = "8" * 64
    changed_panel = replace(
        inputs.action_panels[0],
        execution_artifact_sha256=wrong_digest,
    )
    changed_completion = replace(
        inputs.completion_receipts[0],
        execution_artifact_sha256=wrong_digest,
        action_panel_binding=changed_panel.completion_binding(),
    )
    changed_evaluation = replace(
        inputs.offline_evaluations[0],
        execution_artifact_sha256=wrong_digest,
    )
    changed_labels = replace(
        inputs.sealed_label_artifacts[0],
        execution_artifact_sha256=wrong_digest,
    )
    changed_admission = replace(
        inputs.action_panel_admission_receipts[0],
        execution_artifact_sha256=wrong_digest,
        action_panel_artifact_sha256=changed_panel.artifact_sha256,
    )

    with pytest.raises(
        ConfirmatoryAnalysisError,
        match="online execution digest does not match manifest artifact",
    ):
        replace(
            inputs,
            action_panels=(changed_panel,) + inputs.action_panels[1:],
            action_panel_admission_receipts=(changed_admission,)
            + inputs.action_panel_admission_receipts[1:],
            completion_receipts=(changed_completion,)
            + inputs.completion_receipts[1:],
            offline_evaluations=(changed_evaluation,)
            + inputs.offline_evaluations[1:],
            sealed_label_artifacts=(changed_labels,)
            + inputs.sealed_label_artifacts[1:],
        )


def test_typed_input_enforces_registered_family_and_nesting_counts(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    with pytest.raises(
        ConfirmatoryAnalysisError,
        match="exactly 50 registered query families",
    ):
        _bound_input(config, registered_families_per_corpus=50)

    with pytest.raises(
        ConfirmatoryAnalysisError,
        match="exactly 2 nested trials per family",
    ):
        _bound_input(config, registered_nested_rows_per_family=2)


def test_typed_input_accepts_exact_registered_nested_design(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(
        config,
        registered_nested_rows_per_family=2,
        observed_nested_rows_per_family=2,
    )

    assert inputs.frozen_config.nested_rows_per_family == 2
    for panel in inputs.action_panels:
        trials_by_family: dict[str, set[str]] = {}
        for row in panel.rows:
            trials_by_family.setdefault(row.family_key, set()).add(row.trial_key)
        assert set(map(len, trials_by_family.values())) == {2}


def test_typed_input_requires_completion_anchor_to_postdate_run(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config)
    equal_time = replace(
        inputs.completion_receipts[0],
        anchored_at_utc=inputs.run_receipt.started_at_utc,
    )

    with pytest.raises(ConfirmatoryAnalysisError, match="must postdate"):
        replace(
            inputs,
            completion_receipts=(equal_time,) + inputs.completion_receipts[1:],
        )


def test_typed_input_rejects_forged_verification_row_and_run_drift(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config)
    first = inputs.artifact_verification_receipt.artifacts[0]
    forged_row = replace(
        first,
        expected_sha256="9" * 64,
        verified_sha256="9" * 64,
    )
    forged_receipt = ArtifactVerificationReceipt(
        manifest_sha256=inputs.manifest_sha256,
        artifacts=(forged_row,) + inputs.artifact_verification_receipt.artifacts[1:],
    )
    forged_run = replace(
        inputs.run_receipt,
        verification_receipt_sha256=forged_receipt.receipt_sha256,
    )
    with pytest.raises(ConfirmatoryAnalysisError, match="verification mismatch"):
        replace(
            inputs,
            run_receipt=forged_run,
            artifact_verification_receipt=forged_receipt,
        )

    with pytest.raises(ConfirmatoryAnalysisError, match="code_commit"):
        replace(
            inputs,
            run_receipt=replace(inputs.run_receipt, code_commit="4" * 40),
        )


def test_typed_input_rejects_unpinned_sealed_label_digest(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config)
    changed_evaluation = replace(
        inputs.offline_evaluations[0],
        sealed_label_artifact_sha256="7" * 64,
    )
    with pytest.raises(
        ConfirmatoryAnalysisError,
        match="sealed-label digest does not match admitted artifact",
    ):
        replace(
            inputs,
            offline_evaluations=(changed_evaluation,)
            + inputs.offline_evaluations[1:],
        )


def test_typed_input_rejects_mutated_joined_labels_with_retained_digest(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config)
    evaluation = inputs.offline_evaluations[0]
    first = evaluation.trials[0]
    mutated = replace(
        first,
        labels=replace(first.labels, answer="post-release mutation"),
    )
    changed_evaluation = replace(
        evaluation,
        trials=(mutated,) + evaluation.trials[1:],
    )

    with pytest.raises(
        ConfirmatoryAnalysisError,
        match="joined label payload differs from the sealed artifact",
    ):
        replace(
            inputs,
            offline_evaluations=(changed_evaluation,)
            + inputs.offline_evaluations[1:],
        )


def test_typed_input_rejects_family_rebinding_after_label_release(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config)
    evaluation = inputs.offline_evaluations[0]
    first = evaluation.trials[0]
    rebound_family = "8" * 64
    with pytest.raises(LabelSeparationError, match="joined family keys"):
        JoinedEvaluationTrial(
            prediction=replace(first.prediction, family_key=rebound_family),
            labels=first.labels,
        )


def test_authorized_recall_is_derived_from_exact_oracle_not_relevance_labels(
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config, label_relevant_document_ids=(1,))
    panel = inputs.action_panels[0]
    target = next(
        row
        for row in panel.rows
        if row.action == "hnsw-low" and row.returned_document_ids == (1,)
    )
    row = next(
        row
        for row in inputs.analysis_rows()
        if row.trial_id == target.trial_key and row.action == "hnsw-low"
    )
    assert row.recall_at_k == 0.0


def test_low_effort_label_is_intent_to_treat_action_failure_composite(
    suite,
    config: ConfirmatoryAnalysisConfig,
) -> None:
    rows = tuple(
        row
        for row in _bound_input(config, suite=suite).analysis_rows()
        if row.action == "hnsw-low"
    )
    batch = confirmatory_analysis._feature_batch(
        rows,
        suite,
        include_action_failure_labels=True,
        failure_recall_threshold=config.failure_recall_threshold,
    )
    assert isinstance(batch, LabeledFeatureBatch)
    observed = tuple(zip(rows, batch.labels, strict=True))
    assert any(row.execution_state == "failed" and label == 1 for row, label in observed)
    assert any(
        row.execution_state == "completed"
        and row.recall_at_k is not None
        and row.recall_at_k < config.failure_recall_threshold
        and label == 1
        for row, label in observed
    )
    assert any(
        row.execution_state == "completed"
        and row.recall_at_k is not None
        and row.recall_at_k >= config.failure_recall_threshold
        and label == 0
        for row, label in observed
    )


def test_empty_authorized_truth_requires_an_empty_completed_return(
    suite,
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(
        config,
        suite=suite,
        empty_authorized_truth=True,
    )
    panel = inputs.action_panels[0]
    empty_trial = next(
        row.trial_key
        for row in panel.rows
        if row.action == "hnsw-low"
        and row.execution_state == "completed"
        and not row.returned_document_ids
    )
    extraneous_trial = next(
        row.trial_key
        for row in panel.rows
        if row.action == "hnsw-low"
        and row.execution_state == "completed"
        and row.returned_document_ids
    )
    analysis_rows = inputs.analysis_rows()
    empty_row = next(
        row
        for row in analysis_rows
        if row.action == "hnsw-low" and row.trial_id == empty_trial
    )
    extraneous_row = next(
        row
        for row in analysis_rows
        if row.action == "hnsw-low" and row.trial_id == extraneous_trial
    )
    assert empty_row.recall_at_k == 1.0
    assert extraneous_row.recall_at_k == 0.0
    batch = confirmatory_analysis._feature_batch(
        (empty_row, extraneous_row),
        suite,
        include_action_failure_labels=True,
        failure_recall_threshold=config.failure_recall_threshold,
    )
    assert isinstance(batch, LabeledFeatureBatch)
    assert tuple(batch.labels) == (0, 1)


def test_one_class_corpus_emits_complete_conservative_h2_result(
    suite,
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(
        config,
        suite=suite,
        one_class_success_corpus=FIXED_CORPORA[0],
    )
    result = run_confirmatory_analysis(inputs, suite=suite)

    one_class = next(
        corpus
        for corpus in result.h2.corpus_results
        if corpus.corpus_id == FIXED_CORPORA[0]
    )
    auprc_gate = next(
        gate for gate in result.h2.metric_gates if gate.name == "h2_auprc_gain"
    )
    assert np.isfinite(one_class.log_loss_reduction)
    assert np.isfinite(one_class.brier_score_reduction)
    assert one_class.auprc_gain is None
    assert not one_class.passed
    assert (auprc_gate.estimate, auprc_gate.lower, auprc_gate.upper) == (
        None,
        None,
        None,
    )
    assert auprc_gate.rule == "undefined-one-class-corpus_conservative-fail"
    assert auprc_gate.bootstrap_replicates == 0
    assert not auprc_gate.passed
    assert not result.h2.passed
    assert not result.primary_claim_passed
    assert json.loads(result.canonical_bytes())["h2"]["metric_gates"][2][
        "estimate"
    ] is None


def test_h1_orientation_does_not_gate_primary_success(
    suite,
    config: ConfirmatoryAnalysisConfig,
) -> None:
    reversed_orientation = replace(
        config,
        low_geometry=config.high_geometry,
        high_geometry=config.low_geometry,
    )
    result = run_confirmatory_analysis(
        _bound_input(reversed_orientation, suite=suite),
        suite=suite,
    )

    assert not result.h1.passed
    assert result.h2.passed
    assert result.h3.passed
    assert result.primary_claim_passed


def test_runner_emits_receipt_derived_canonical_intersection_union_result(
    suite,
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config, suite=suite)
    result = run_confirmatory_analysis(inputs, suite=suite)

    assert result.manifest_sha256 == inputs.run_receipt.manifest_sha256
    assert result.run_receipt_sha256 == sealed_run_receipt_sha256(inputs.run_receipt)
    assert result.confirmatory_input_artifact_sha256 == inputs.artifact_sha256
    assert result.corpus_input_digests == inputs.corpus_input_digests
    assert result.model_suite_sha256 == suite.suite_digest
    assert result.to_dict()["frozen_config_sha256"] == config.config_sha256
    assert result.artifact_sha256 == result.artifact_sha256
    assert result.h1.passed
    assert result.h2.passed
    assert result.h3.passed
    assert result.primary_claim_passed
    assert len(result.h2.metric_gates) == 3
    assert len(result.h2.passing_corpora) == 5
    assert len(result.h3.gates) == 4
    assert dict(result.h3.execution_state_counts)["failed"] == len(FIXED_CORPORA)
    assert result.h3.entitlement.observed_events == 0
    expected_upper = 1.0 - config.alpha ** (1.0 / result.h3.entitlement.n_families)
    assert np.isclose(result.h3.entitlement.exact_upper_bound, expected_upper)

    with pytest.raises(ConfirmatoryAnalysisError, match="ConfirmatoryInputArtifact"):
        run_confirmatory_analysis(inputs.analysis_rows(), suite=suite)


def test_runner_rejects_model_suite_not_pinned_by_manifest(
    suite,
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config, suite=suite)
    changed_full_model = replace(
        suite.model("full"),
        intercept=suite.model("full").intercept + 0.1,
    )
    h1_model_drift = replace(
        suite,
        models=tuple(
            changed_full_model if model.name == "full" else model
            for model in suite.models
        ),
    )

    with pytest.raises(
        ConfirmatoryAnalysisError,
        match="h1-predictive-model artifact",
    ):
        run_confirmatory_analysis(inputs, suite=h1_model_drift)

    suite_metadata_drift = replace(
        suite,
        sklearn_version=f"{suite.sklearn_version}-drift",
    )
    with pytest.raises(
        ConfirmatoryAnalysisError,
        match="h2-model-suite artifact",
    ):
        run_confirmatory_analysis(inputs, suite=suite_metadata_drift)


def test_any_anchored_entitlement_event_fails_the_primary_intersection(
    suite,
    config: ConfirmatoryAnalysisConfig,
) -> None:
    inputs = _bound_input(config, entitlement_event=True, suite=suite)
    result = run_confirmatory_analysis(inputs, suite=suite)

    assert result.h3.entitlement.observed_events == 1
    assert not result.h3.entitlement.passed
    assert not result.h3.passed
    assert not result.primary_claim_passed
