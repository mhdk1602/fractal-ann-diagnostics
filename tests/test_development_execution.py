from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fractal_ann_diagnostics.confirmatory_modeling import REGISTERED_FEATURE_SCHEMA
from fractal_ann_diagnostics.development_cohort import (
    CALIBRATION_FAMILY_COUNT,
    FIT_FAMILY_COUNT,
    DevelopmentEmbeddingBinding,
)
from fractal_ann_diagnostics.development_execution import (
    DEVELOPMENT_ACTION_FILENAME,
    DEVELOPMENT_ACTION_PERMUTATION_SEED,
    DEVELOPMENT_CONFIG_FILENAME,
    DEVELOPMENT_ORDER_FILENAME,
    DevelopmentExecutionError,
    DevelopmentExecutionInput,
    DevelopmentExecutionOrder,
    DevelopmentExecutionOrderRow,
    DevelopmentOutputArtifact,
    DevelopmentPairedActionRow,
    DevelopmentPairedExecutionConfig,
    DevelopmentPairedExecutionReceipt,
    DevelopmentStratumExecution,
    DevelopmentStratumReceipt,
    _default_stratum_executor,
    _development_feature_values,
    _PinnedOPADataTransport,
    _validate_source_bindings,
    _validate_stratum_execution,
    _verify_output_tree,
    load_development_paired_execution_config,
    load_development_paired_execution_receipt,
    run_development_paired_execution,
)
from fractal_ann_diagnostics.development_freeze import REGISTERED_ACTIONS
from fractal_ann_diagnostics.joint_power_design import FIXED_CORPORA
from fractal_ann_diagnostics.online_runner import portable_balanced_action_orders
from fractal_ann_diagnostics.policy_intervention import (
    OPACompiledMaskData,
    OPAMaskAssignment,
)
from fractal_ann_diagnostics.retrieval import packed_policy_mask_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[DevelopmentExecutionInput, ...]:
    rows = []
    for stage in ("development-fit", "development-calibration"):
        for corpus in FIXED_CORPORA:
            rows.append(
                DevelopmentExecutionInput(
                    corpus=corpus,
                    stage=stage,
                    policy_intervention_root=tmp_path / f"policy-{stage}-{corpus}",
                    policy_intervention_receipt_sha256=_digest(f"policy-{stage}-{corpus}"),
                    authorized_index_root=tmp_path / f"index-{stage}-{corpus}",
                    authorized_index_receipt_sha256=_digest(f"index-{stage}-{corpus}"),
                )
            )
    return tuple(rows)


def _config(tmp_path: Path) -> DevelopmentPairedExecutionConfig:
    return DevelopmentPairedExecutionConfig(
        materialization_root=tmp_path / "materialized-development",
        materialization_receipt_sha256=_digest("materialization"),
        inputs=_inputs(tmp_path),
        output_root=tmp_path / "paired-development-output",
    )


def _feature_values() -> dict[str, object]:
    values: dict[str, object] = {
        "corpus_size": 100.0,
        "authorized_universe_size": 25.0,
        "embedding_dimension": 8.0,
        "version_lag": 1.0,
        "drift_severity": 0.1,
        "probe_latency_ms": 0.25,
        "probe_work": None,
        "corpus_stratum": FIXED_CORPORA[0],
        "backend": "hnswlib",
        "drift_family": "qwen-terminal-token-revision-lag-one",
        "allow_rate": 0.25,
        "policy_complexity": 1.0,
        "policy_churn": 0.05,
        "lid_k50": 2.0,
        "lid_cv": 0.1,
        "relative_contrast": 1.5,
        "radius_expansion": 1.1,
    }
    assert tuple(values) == REGISTERED_FEATURE_SCHEMA.input_features
    return values


def _action_row(**overrides: object) -> DevelopmentPairedActionRow:
    values: dict[str, object] = {
        "schedule_order": 0,
        "trial_key": _digest("trial"),
        "family_id": _digest("family"),
        "query_id": "query-1",
        "action": "hnsw-low",
        "execution_position": 0,
        "execution_state": "completed",
        "failure_state": None,
        "request_latency_ms": 1.0,
        "entitlement_violations": 0,
        "returned_document_rows": (0,),
        "feature_values": _feature_values(),
    }
    values.update(overrides)
    return DevelopmentPairedActionRow(**values)  # type: ignore[arg-type]


def _plan(corpus: str = FIXED_CORPORA[0], stage: str = "development-fit") -> object:
    family = _digest(f"family-{stage}-{corpus}")
    trials = tuple(
        SimpleNamespace(
            trial_key=_digest(f"trial-{stage}-{corpus}-{nested}"),
            family_key=family,
            query_id="query-1",
        )
        for nested in range(3)
    )
    return SimpleNamespace(
        artifact_sha256=_digest(f"plan-{stage}-{corpus}"),
        corpus=corpus,
        stage=stage,
        selected_family_count=1,
        trials=trials,
    )


def _registered_plan(corpus: str, stage: str) -> object:
    family_count = FIT_FAMILY_COUNT if stage == "development-fit" else CALIBRATION_FAMILY_COUNT
    trials = tuple(
        SimpleNamespace(
            trial_key=_digest(f"trial-{stage}-{corpus}-{family}-{nested}"),
            family_key=_digest(f"family-{stage}-{corpus}-{family}"),
            query_id=f"query-{family:03d}",
        )
        for family in range(family_count)
        for nested in range(3)
    )
    return SimpleNamespace(
        artifact_sha256=_digest(f"plan-{stage}-{corpus}"),
        corpus=corpus,
        stage=stage,
        selected_family_count=family_count,
        trials=trials,
    )


def _stratum_execution(
    plan: object,
    source: DevelopmentExecutionInput,
    embedding: DevelopmentEmbeddingBinding,
) -> DevelopmentStratumExecution:
    orders = portable_balanced_action_orders(
        permutation_seed=DEVELOPMENT_ACTION_PERMUTATION_SEED,
        execution_artifact_sha256=plan.artifact_sha256,
        trial_families=tuple((trial.trial_key, trial.family_key) for trial in plan.trials),
    )
    order_rows = tuple(
        DevelopmentExecutionOrderRow(
            schedule_order=position,
            trial_key=trial.trial_key,
            family_id=trial.family_key,
            query_id=trial.query_id,
            actions=orders[trial.trial_key],
        )
        for position, trial in enumerate(plan.trials)
    )
    action_rows = tuple(
        DevelopmentPairedActionRow(
            schedule_order=position,
            trial_key=trial.trial_key,
            family_id=trial.family_key,
            query_id=trial.query_id,
            action=action,
            execution_position=orders[trial.trial_key].index(action),
            execution_state=("abstained" if action == "abstain" else "completed"),
            failure_state=("registered-abstention" if action == "abstain" else None),
            request_latency_ms=1.0 + position,
            entitlement_violations=0,
            returned_document_rows=(),
            feature_values=_feature_values() if action == "hnsw-low" else None,
        )
        for position, trial in enumerate(plan.trials)
        for action in REGISTERED_ACTIONS
    )
    return DevelopmentStratumExecution(
        action_rows=action_rows,
        execution_order=DevelopmentExecutionOrder(
            execution_plan_sha256=plan.artifact_sha256,
            permutation_seed=DEVELOPMENT_ACTION_PERMUTATION_SEED,
            rows=order_rows,
        ),
        embedding_receipt_sha256=embedding.receipt_sha256,
        policy_config_sha256=_digest(f"config-{source.stage}-{source.corpus}"),
        policy_catalog_sha256=_digest(f"catalog-{source.stage}-{source.corpus}"),
        policy_schedule_sha256=_digest(f"schedule-{source.stage}-{source.corpus}"),
        policy_intervention_receipt_sha256=(source.policy_intervention_receipt_sha256),
        authorized_index_receipt_sha256=source.authorized_index_receipt_sha256,
    )


def test_config_is_closed_canonical_and_covers_exact_ten_strata(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = tmp_path / "execution-input.json"
    path.write_bytes(config.canonical_file_bytes())

    assert load_development_paired_execution_config(path) == config

    payload = json.loads(config.canonical_file_bytes())
    payload["undeclared"] = True
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(DevelopmentExecutionError, match="fields differ"):
        load_development_paired_execution_config(path)

    with pytest.raises(DevelopmentExecutionError, match="fixed ten strata"):
        DevelopmentPairedExecutionConfig(
            materialization_root=tmp_path / "another-materialization",
            materialization_receipt_sha256=_digest("another-materialization"),
            inputs=config.inputs[:-1],
            output_root=tmp_path / "another-output",
        )


def test_config_rejects_forbidden_or_overlapping_boundaries(tmp_path: Path) -> None:
    with pytest.raises(DevelopmentExecutionError, match="non-development boundary"):
        DevelopmentExecutionInput(
            corpus=FIXED_CORPORA[0],
            stage="development-fit",
            policy_intervention_root=tmp_path / "sealed-policy",
            policy_intervention_receipt_sha256=_digest("policy"),
            authorized_index_root=tmp_path / "index",
            authorized_index_receipt_sha256=_digest("index"),
        )

    inputs = list(_inputs(tmp_path))
    inputs[0] = DevelopmentExecutionInput(
        corpus=inputs[0].corpus,
        stage=inputs[0].stage,
        policy_intervention_root=tmp_path / "materialized-development" / "policy",
        policy_intervention_receipt_sha256=inputs[0].policy_intervention_receipt_sha256,
        authorized_index_root=inputs[0].authorized_index_root,
        authorized_index_receipt_sha256=inputs[0].authorized_index_receipt_sha256,
    )
    with pytest.raises(DevelopmentExecutionError, match="roots overlap"):
        DevelopmentPairedExecutionConfig(
            materialization_root=tmp_path / "materialized-development",
            materialization_receipt_sha256=_digest("materialization"),
            inputs=tuple(inputs),
            output_root=tmp_path / "output",
        )


def test_stratum_validator_requires_complete_canonical_action_panel(
    tmp_path: Path,
) -> None:
    source = _inputs(tmp_path)[0]
    plan = _plan(source.corpus, str(source.stage))
    embedding = DevelopmentEmbeddingBinding(
        corpus=source.corpus,
        development_stage=str(source.stage),
        root=tmp_path / "embedding",
        receipt_sha256=_digest("embedding"),
    )
    result = _stratum_execution(plan, source, embedding)

    _validate_stratum_execution(
        result,
        plan,
        {"query-1": "text"},
        source,
        embedding,
        permutation_seed=DEVELOPMENT_ACTION_PERMUTATION_SEED,
    )

    missing = DevelopmentStratumExecution(
        action_rows=result.action_rows[:-1],
        execution_order=result.execution_order,
        embedding_receipt_sha256=result.embedding_receipt_sha256,
        policy_config_sha256=result.policy_config_sha256,
        policy_catalog_sha256=result.policy_catalog_sha256,
        policy_schedule_sha256=result.policy_schedule_sha256,
        policy_intervention_receipt_sha256=(result.policy_intervention_receipt_sha256),
        authorized_index_receipt_sha256=result.authorized_index_receipt_sha256,
    )
    with pytest.raises(DevelopmentExecutionError, match="complete action set"):
        _validate_stratum_execution(
            missing,
            plan,
            {"query-1": "text"},
            source,
            embedding,
            permutation_seed=DEVELOPMENT_ACTION_PERMUTATION_SEED,
        )

    wrong_position = replace(
        result,
        action_rows=(
            replace(
                result.action_rows[0],
                execution_position=(result.action_rows[0].execution_position + 1) % 4,
            ),
            *result.action_rows[1:],
        ),
    )
    with pytest.raises(DevelopmentExecutionError, match="positions differ"):
        _validate_stratum_execution(
            wrong_position,
            plan,
            {"query-1": "text"},
            source,
            embedding,
            permutation_seed=DEVELOPMENT_ACTION_PERMUTATION_SEED,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"request_latency_ms": 0.0}, "positive"),
        ({"entitlement_violations": 1}, "entitlement violation"),
        ({"returned_document_rows": tuple(range(11))}, "registered k"),
        ({"returned_document_rows": (2, 2)}, "unique integers"),
        ({"execution_position": 4}, "zero to three"),
        ({"action": "hnsw-high"}, "only hnsw-low"),
    ),
)
def test_action_row_rejects_invalid_or_unauthorized_telemetry(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(DevelopmentExecutionError, match=message):
        _action_row(**overrides)


def test_source_binding_rejects_reordered_policy_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _inputs(tmp_path)[0]
    base = _plan(source.corpus, str(source.stage))
    plan = SimpleNamespace(
        artifact_sha256=base.artifact_sha256,
        corpus=base.corpus,
        stage=base.stage,
        selected_family_count=base.selected_family_count,
        trials=base.trials,
        document_count=10,
        document_universe_sha256=_digest("document-universe"),
    )
    embedding = DevelopmentEmbeddingBinding(
        corpus=source.corpus,
        development_stage=str(source.stage),
        root=tmp_path / "embedding",
        receipt_sha256=_digest("embedding"),
    )
    result = _stratum_execution(plan, source, embedding)
    policy_config = SimpleNamespace(config_sha256=result.policy_config_sha256)
    schedule = SimpleNamespace(
        artifact_sha256=result.policy_schedule_sha256,
        config_sha256=result.policy_config_sha256,
        mask_catalog_sha256=result.policy_catalog_sha256,
        execution_artifact_sha256=plan.artifact_sha256,
        corpus=plan.corpus,
        stage=plan.stage,
        document_count=plan.document_count,
        document_universe_sha256=plan.document_universe_sha256,
        rows=tuple(
            SimpleNamespace(
                schedule_order=row.schedule_order,
                trial_key=row.trial_key,
                family_key=row.family_id,
            )
            for row in result.execution_order.rows
        ),
    )
    policy_receipt = SimpleNamespace(
        artifact_sha256=result.policy_intervention_receipt_sha256,
        config_sha256=result.policy_config_sha256,
        execution_artifact_sha256=plan.artifact_sha256,
    )
    index_receipt = SimpleNamespace(
        artifact_sha256=result.authorized_index_receipt_sha256,
        policy_catalog_sha256=result.policy_catalog_sha256,
        policy_receipt_sha256=result.policy_intervention_receipt_sha256,
        embedding_receipt_sha256=embedding.receipt_sha256,
        policy_execution_artifact_sha256=plan.artifact_sha256,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.load_policy_intervention_config",
        lambda path: policy_config,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.load_canonical_trial_schedule",
        lambda path: schedule,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.load_policy_intervention_receipt",
        lambda path: policy_receipt,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.CompiledPolicyMaskStore",
        lambda path: SimpleNamespace(catalog_sha256=result.policy_catalog_sha256),
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.load_authorized_index_store_receipt",
        lambda root: index_receipt,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.load_embedding_store_receipt",
        lambda root: SimpleNamespace(receipt_sha256=embedding.receipt_sha256),
    )

    _validate_source_bindings(result, plan, source, embedding)

    schedule.rows = tuple(reversed(schedule.rows))
    with pytest.raises(DevelopmentExecutionError, match="canonical policy schedule"):
        _validate_source_bindings(result, plan, source, embedding)


def _receipt_tree(tmp_path: Path) -> tuple[Path, DevelopmentPairedExecutionReceipt]:
    root = tmp_path / "execution-tree"
    root.mkdir()
    (root / DEVELOPMENT_CONFIG_FILENAME).write_bytes(b"{}\n")
    strata = []
    for stage in ("development-fit", "development-calibration"):
        for corpus in FIXED_CORPORA:
            family_count = (
                FIT_FAMILY_COUNT if stage == "development-fit" else CALIBRATION_FAMILY_COUNT
            )
            trial_count = family_count * 3
            prefix = f"{stage}/{corpus}"
            action_path = f"{prefix}/{DEVELOPMENT_ACTION_FILENAME}"
            order_path = f"{prefix}/{DEVELOPMENT_ORDER_FILENAME}"
            action_bytes = b'{"row":1}\n'
            order_bytes = b"{}\n"
            (root / prefix).mkdir(parents=True)
            (root / action_path).write_bytes(action_bytes)
            (root / order_path).write_bytes(order_bytes)
            strata.append(
                DevelopmentStratumReceipt(
                    corpus=corpus,
                    stage=stage,
                    execution_plan_sha256=_digest(f"plan-{stage}-{corpus}"),
                    selected_family_count=family_count,
                    trial_count=trial_count,
                    embedding_receipt_sha256=_digest(f"embedding-{stage}-{corpus}"),
                    policy_config_sha256=_digest(f"config-{stage}-{corpus}"),
                    policy_catalog_sha256=_digest(f"catalog-{stage}-{corpus}"),
                    policy_schedule_sha256=_digest(f"schedule-{stage}-{corpus}"),
                    policy_intervention_receipt_sha256=_digest(f"policy-{stage}-{corpus}"),
                    authorized_index_receipt_sha256=_digest(f"index-{stage}-{corpus}"),
                    outputs=(
                        DevelopmentOutputArtifact(
                            path=action_path,
                            role="paired-actions",
                            sha256=hashlib.sha256(action_bytes).hexdigest(),
                            byte_count=len(action_bytes),
                            record_count=trial_count * len(REGISTERED_ACTIONS),
                        ),
                        DevelopmentOutputArtifact(
                            path=order_path,
                            role="execution-order",
                            sha256=hashlib.sha256(order_bytes).hexdigest(),
                            byte_count=len(order_bytes),
                            record_count=trial_count,
                        ),
                    ),
                )
            )
    receipt = DevelopmentPairedExecutionReceipt(
        config_sha256=_digest("config"),
        materialization_receipt_sha256=_digest("materialization"),
        selection_receipt_sha256=_digest("selection"),
        strata=tuple(strata),
    )
    (root / "execution-receipt.json").write_bytes(receipt.canonical_file_bytes())
    return root, receipt


def test_stratum_receipt_closes_registered_counts_paths_and_roles(tmp_path: Path) -> None:
    _, receipt = _receipt_tree(tmp_path)
    stratum = receipt.strata[0]

    with pytest.raises(DevelopmentExecutionError, match="family count"):
        replace(stratum, selected_family_count=1, trial_count=3)

    paired = next(row for row in stratum.outputs if row.role == "paired-actions")
    with pytest.raises(DevelopmentExecutionError, match="path or record count"):
        replace(
            stratum,
            outputs=(
                replace(paired, path="development-fit/scifact/substitute.jsonl"),
                *(row for row in stratum.outputs if row.role != "paired-actions"),
            ),
        )
    with pytest.raises(DevelopmentExecutionError, match="path or record count"):
        replace(
            stratum,
            outputs=(
                replace(paired, record_count=paired.record_count - 1),
                *(row for row in stratum.outputs if row.role != "paired-actions"),
            ),
        )
    with pytest.raises(DevelopmentExecutionError, match="output roles"):
        replace(stratum, outputs=stratum.outputs + (replace(paired, path="duplicate.jsonl"),))


def test_output_tree_rejects_extra_symlink_and_hardlink(tmp_path: Path) -> None:
    root, receipt = _receipt_tree(tmp_path)
    _verify_output_tree(root, receipt)

    extra = root / "undeclared.json"
    extra.write_text("{}\n")
    with pytest.raises(DevelopmentExecutionError, match="extra"):
        _verify_output_tree(root, receipt)
    extra.unlink()

    link = root / "linked.json"
    link.symlink_to(root / DEVELOPMENT_CONFIG_FILENAME)
    with pytest.raises(DevelopmentExecutionError, match="cannot verify"):
        _verify_output_tree(root, receipt)
    link.unlink()

    hardlink = root / "hardlinked.json"
    os.link(root / DEVELOPMENT_CONFIG_FILENAME, hardlink)
    with pytest.raises(DevelopmentExecutionError, match="cannot verify"):
        _verify_output_tree(root, receipt)


def test_pinned_opa_transport_is_an_exact_subject_state_lookup() -> None:
    assignment = OPAMaskAssignment(
        subject="reader",
        policy_state="allow-25",
        mask_id="reader-allow-25",
        mask_sha256=_digest("mask"),
        authorized_count=2,
    )
    data = OPACompiledMaskData(
        document_count=10,
        document_universe_sha256=_digest("universe"),
        mask_catalog_sha256=_digest("catalog"),
        policy_revision="sha256:" + _digest("policy"),
        assignments=(assignment,),
    )
    transport = _PinnedOPADataTransport(data)
    policy_input = {
        "action": "retrieve",
        "catalog_request_sha256": _digest("request-catalog"),
        "document_count": 10,
        "document_universe_sha256": data.document_universe_sha256,
        "environment": {
            "assignment_repetition": 0,
            "policy_state": "allow-25",
        },
        "environment_sha256": _digest("environment"),
        "mask_catalog_sha256": data.mask_catalog_sha256,
        "policy_revision": data.policy_revision,
        "request_nonce": "nonce",
        "request_sha256": _digest("request"),
        "subject": "reader",
    }
    response = transport(
        "http://127.0.0.1:8181/v1/data/fractal/retrieval/mask",
        json.dumps({"input": policy_input}, sort_keys=True, separators=(",", ":")).encode(),
        1.0,
    )
    result = json.loads(response.body)["result"]
    assert result["mask_id"] == assignment.mask_id
    assert result["mask_sha256"] == assignment.mask_sha256
    assert "document_ids" not in result

    changed = dict(policy_input)
    changed["environment"] = {
        "assignment_repetition": 0,
        "policy_state": "undeclared",
    }
    denied = transport(
        "http://127.0.0.1:8181/v1/data/fractal/retrieval/mask",
        json.dumps({"input": changed}, sort_keys=True, separators=(",", ":")).encode(),
        1.0,
    )
    assert denied.status == 404


def test_feature_projection_uses_only_registered_development_covariates() -> None:
    geometry = SimpleNamespace(
        distance_evaluations=None,
        visited_candidates=None,
        embedding_drift=0.1,
        search_latency_ms=0.5,
        policy_churn=0.02,
        lid=2.0,
        lid_scale_instability=0.1,
        relative_contrast=1.3,
        radius_expansion=1.2,
    )
    values = _development_feature_values(
        geometry=geometry,
        authorized_count=25,
        document_count=100,
        dimension=8,
        corpus=FIXED_CORPORA[0],
        backend="hnswlib",
        group_order=0,
    )
    assert tuple(values) == REGISTERED_FEATURE_SCHEMA.input_features
    assert values["probe_work"] is None
    assert values["allow_rate"] == 0.25


def test_default_executor_uses_both_epochs_and_emits_complete_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _inputs(tmp_path)[0]
    embedding = DevelopmentEmbeddingBinding(
        corpus=source.corpus,
        development_stage=str(source.stage),
        root=tmp_path / "paired-embedding",
        receipt_sha256=_digest("paired-embedding"),
    )
    base = _plan(source.corpus, str(source.stage))
    trials = tuple(
        SimpleNamespace(
            trial_key=row.trial_key,
            family_key=row.family_key,
            query_id=row.query_id,
            query_row=0,
        )
        for row in base.trials
    )
    plan = SimpleNamespace(
        artifact_sha256=base.artifact_sha256,
        corpus=base.corpus,
        stage=base.stage,
        selected_family_count=base.selected_family_count,
        trials=trials,
        document_count=4,
        document_universe_sha256=_digest("universe"),
        document_row_order_sha256=_digest("universe"),
        query_row_order_sha256=_digest("query-order"),
        embedding_receipt_sha256=embedding.receipt_sha256,
    )
    masks = []
    schedule_rows = []
    for position, trial in enumerate(trials):
        mask = np.zeros(4, dtype=bool)
        mask[: position + 1] = True
        mask.setflags(write=False)
        masks.append(mask)
        schedule_rows.append(
            SimpleNamespace(
                schedule_order=position,
                group_order=position,
                trial_key=trial.trial_key,
                family_key=trial.family_key,
                environment=(
                    ("assignment_repetition", 0),
                    ("policy_state", f"allow-{position}"),
                ),
                environment_sha256=_digest(f"environment-{position}"),
                subject="reader",
                authorized_count=position + 1,
                mask_sha256=packed_policy_mask_sha256(mask),
            )
        )
    schedule = SimpleNamespace(
        rows=tuple(schedule_rows),
        policy_bundle_revision="sha256:" + _digest("policy"),
        artifact_sha256=_digest("schedule"),
    )
    policy_config = SimpleNamespace(config_sha256=_digest("policy-config"))
    policy_receipt = SimpleNamespace(
        artifact_sha256=source.policy_intervention_receipt_sha256,
        policy_bundle_revision=schedule.policy_bundle_revision,
    )
    mask_store = SimpleNamespace(catalog_sha256=_digest("catalog"))
    embedding_receipt = SimpleNamespace(
        receipt_sha256=embedding.receipt_sha256,
        old_model=object(),
        document_count=4,
        vectors={
            name: SimpleNamespace(
                relative_path=f"{name}.npy",
                byte_count=100,
                shape=(1, 2) if "queries" in name else (4, 2),
            )
            for name in (
                "old_documents",
                "current_documents",
                "old_queries",
                "current_queries",
            )
        },
        row_orders={
            "documents": SimpleNamespace(file_sha256=plan.document_row_order_sha256),
            "queries": SimpleNamespace(file_sha256=plan.query_row_order_sha256),
        },
    )
    index_receipt = SimpleNamespace(
        artifact_sha256=source.authorized_index_receipt_sha256,
        embedding_receipt_sha256=embedding.receipt_sha256,
        policy_receipt_sha256=source.policy_intervention_receipt_sha256,
        policy_catalog_sha256=mask_store.catalog_sha256,
        policy_execution_artifact_sha256=plan.artifact_sha256,
        policy_revision=schedule.policy_bundle_revision,
        document_count=plan.document_count,
        document_universe_sha256=plan.document_universe_sha256,
        document_row_order_sha256=plan.document_row_order_sha256,
        backend_id="hnswlib",
    )
    old_documents = np.zeros((4, 2), dtype=np.float32)
    current_documents = np.ones((4, 2), dtype=np.float32)
    old_queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    current_queries = np.asarray([[0.0, 1.0]], dtype=np.float32)
    for value in (old_documents, current_documents, old_queries, current_queries):
        value.setflags(write=False)

    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.verify_embedding_store",
        lambda root: embedding_receipt,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution._load_policy_objects",
        lambda source, plan: (
            policy_config,
            schedule,
            policy_receipt,
            SimpleNamespace(assignments=()),
            mask_store,
        ),
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.load_authorized_index_store_receipt",
        lambda root: index_receipt,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.HnswlibBackend",
        lambda: object(),
    )

    class FakeProvider:
        retrieval_metric = "cosine"

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.VerifiedAuthorizedIndexProvider",
        FakeProvider,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution._policy_transitions",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.OpenPolicyAgentMaskDecisionPoint",
        lambda *args, **kwargs: object(),
    )

    @contextmanager
    def documents(*args: object, **kwargs: object):
        yield SimpleNamespace(old_active=old_documents, current_truth=current_documents)

    @contextmanager
    def queries(*args: object, **kwargs: object):
        yield old_queries, current_queries

    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.open_verified_document_matrices",
        documents,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution._open_query_epochs",
        queries,
    )
    observed_epoch_pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    class FakeRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.controller = kwargs["controller"]

        def prepare_authorization(
            self,
            *,
            environment: dict[str, object],
            **kwargs: object,
        ) -> object:
            position = int(str(environment["policy_state"]).split("-")[1])
            return SimpleNamespace(
                authorized_count=position + 1,
                authorized_mask=masks[position],
            )

        def seal_authorized_index_cache(self) -> None:
            pass

        def query(
            self,
            old: np.ndarray,
            *,
            current_truth_query: np.ndarray,
            environment: dict[str, object],
            **kwargs: object,
        ) -> object:
            observed_epoch_pairs.append((tuple(old), tuple(current_truth_query)))
            position = int(str(environment["policy_state"]).split("-")[1])
            action = self.controller.action
            authorization = SimpleNamespace(authorized_mask=masks[position])
            geometry = SimpleNamespace(
                distance_evaluations=None,
                visited_candidates=None,
                embedding_drift=0.2,
                search_latency_ms=0.5,
                policy_churn=0.1,
                lid=2.0,
                lid_scale_instability=0.1,
                relative_contrast=1.2,
                radius_expansion=1.1,
            )
            search = (
                None
                if action == "abstain"
                else SimpleNamespace(
                    ids=np.asarray([0], dtype=np.int64),
                    unauthorized_candidates=0,
                    unauthorized_context=0,
                )
            )
            return SimpleNamespace(
                geometry=geometry,
                decision=SimpleNamespace(action=action),
                search=search,
                initial_authorization=authorization,
                final_authorization=authorization,
                total_online_latency_ms=1.0,
            )

    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.GovernedRetriever",
        FakeRetriever,
    )

    result = _default_stratum_executor(
        source,
        plan,
        {"query-1": "text"},
        embedding,
        controller=DevelopmentPairedExecutionConfig(
            materialization_root=tmp_path / "materialized",
            materialization_receipt_sha256=_digest("materialized"),
            inputs=_inputs(tmp_path / "config-roots"),
            output_root=tmp_path / "output",
        ).controller,
        k=10,
        permutation_seed=DEVELOPMENT_ACTION_PERMUTATION_SEED,
    )

    assert len(result.action_rows) == 12
    assert len(result.execution_order.rows) == 3
    assert set(row.action for row in result.action_rows) == set(REGISTERED_ACTIONS)
    assert set(observed_epoch_pairs) == {((1.0, 0.0), (0.0, 1.0))}


def test_injected_runner_publishes_only_predeclared_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "paired-execution-config.json"
    config_path.write_bytes(config.canonical_file_bytes())
    embeddings = tuple(
        DevelopmentEmbeddingBinding(
            corpus=source.corpus,
            development_stage=str(source.stage),
            root=tmp_path / f"embedding-{source.stage}-{source.corpus}",
            receipt_sha256=_digest(f"embedding-{source.stage}-{source.corpus}"),
        )
        for source in config.inputs
    )
    materialization = SimpleNamespace(
        artifact_sha256=config.materialization_receipt_sha256,
        selection_receipt_sha256=_digest("selection"),
        embedding_bindings=embeddings,
    )
    verify_calls: list[bool] = []

    def fake_verify(*args: object, **kwargs: object) -> object:
        verify_calls.append(kwargs["verify_label_payloads"])
        return materialization

    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.verify_materialized_development_cohort",
        fake_verify,
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution._load_materialized_plan",
        lambda root, receipt, *, corpus, stage: _registered_plan(corpus, stage),
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution._load_materialized_queries",
        lambda root, receipt, *, corpus, stage: {
            f"query-{family:03d}": "text"
            for family in range(
                FIT_FAMILY_COUNT if stage == "development-fit" else CALIBRATION_FAMILY_COUNT
            )
        },
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution._embedding_binding",
        lambda receipt, *, corpus, stage: next(
            row for row in embeddings if row.corpus == corpus and row.development_stage == stage
        ),
    )
    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution._validate_source_bindings",
        lambda *args, **kwargs: None,
    )

    def executor(
        source: DevelopmentExecutionInput,
        plan: object,
        queries: object,
        embedding: DevelopmentEmbeddingBinding,
        **kwargs: object,
    ) -> DevelopmentStratumExecution:
        return _stratum_execution(plan, source, embedding)

    monkeypatch.setattr(
        "fractal_ann_diagnostics.development_execution.verify_development_paired_execution",
        lambda root, *, expected_receipt_sha256=None: load_development_paired_execution_receipt(
            Path(root) / "execution-receipt.json",
            expected_artifact_sha256=expected_receipt_sha256,
        ),
    )

    receipt = run_development_paired_execution(config_path, executor=executor)

    assert len(receipt.strata) == 10
    assert verify_calls == [False]
    expected = {
        DEVELOPMENT_CONFIG_FILENAME,
        "execution-receipt.json",
        *(
            f"{stage}/{corpus}/{name}"
            for stage in ("development-fit", "development-calibration")
            for corpus in FIXED_CORPORA
            for name in (DEVELOPMENT_ACTION_FILENAME, DEVELOPMENT_ORDER_FILENAME)
        ),
    }
    observed = {
        path.relative_to(config.output_root).as_posix()
        for path in config.output_root.rglob("*")
        if path.is_file()
    }
    assert observed == expected
