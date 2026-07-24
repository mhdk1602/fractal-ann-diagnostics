from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import fractal_ann_diagnostics.development_freeze as freeze
import fractal_ann_diagnostics.policy_intervention as intervention
from fractal_ann_diagnostics.development_freeze import (
    REGISTERED_ACTIONS,
    ActionOutcome,
    DevelopmentCorpusSources,
    DevelopmentFreezeConfig,
    DevelopmentFreezeError,
    DevelopmentPartitionData,
    DevelopmentTrial,
    PinnedDevelopmentFile,
    PinnedDevelopmentSelectionReceipt,
    PinnedEmbeddingStore,
    compile_development_freeze,
    verify_development_freeze,
)
from fractal_ann_diagnostics.embedding_store import (
    EmbeddingStoreReceipt,
    RowOrderDescriptor,
    VectorDescriptor,
)
from fractal_ann_diagnostics.joint_power_design import (
    EVIDENCE_CORPORA,
    FIXED_CORPORA,
    load_development_panel,
    load_joint_power_config,
)
from fractal_ann_diagnostics.policy import policy_environment_sha256
from fractal_ann_diagnostics.policy_intervention import (
    CanonicalTrialSchedule,
    TrialScheduleRow,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write(path: Path, payload: bytes) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _pinned_path(
    path: Path,
    payload: bytes,
    *,
    corpus: str,
    stage: str,
    role: str,
) -> PinnedDevelopmentFile:
    return PinnedDevelopmentFile(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        corpus_id=corpus,
        stage=stage,
        role=role,
    )


def _pin(root: Path, corpus: str, stage: str, role: str) -> PinnedDevelopmentFile:
    return PinnedDevelopmentFile(
        path=root / stage / corpus / f"{role}.jsonl",
        sha256=_digest(f"{stage}:{corpus}:{role}"),
        byte_count=1,
        corpus_id=corpus,
        stage=stage,
        role=role,
    )


def _config(tmp_path: Path) -> DevelopmentFreezeConfig:
    source_root = Path("/tmp/development-freeze-inputs")
    sources: list[DevelopmentCorpusSources] = []
    for stage in ("development-fit", "development-calibration"):
        for corpus in FIXED_CORPORA:
            evidence = (
                _pin(source_root, corpus, stage, "evidence-bundles")
                if corpus in EVIDENCE_CORPORA
                else None
            )
            sources.append(
                DevelopmentCorpusSources(
                    corpus_id=corpus,
                    stage=stage,
                    queries=_pin(source_root, corpus, stage, "queries"),
                    qrels=_pin(source_root, corpus, stage, "qrels"),
                    evidence_bundles=evidence,
                    policy_schedule=_pin(source_root, corpus, stage, "policy-schedule"),
                    paired_actions=_pin(source_root, corpus, stage, "paired-actions"),
                    embedding_store=PinnedEmbeddingStore(
                        root=source_root / stage / corpus / "embedding-store",
                        receipt_sha256=_digest(f"{stage}:{corpus}:embedding"),
                        corpus_id=corpus,
                        stage=stage,
                    ),
                )
            )
    return DevelopmentFreezeConfig(
        sources=tuple(sources),
        selection_receipt=PinnedDevelopmentSelectionReceipt(
            path=source_root / "selection-receipt.json",
            sha256=_digest("development-selection"),
            byte_count=1,
        ),
        output_root=tmp_path / "development-freeze",
    )


def _outcomes(*, evidence: bool, failed_low: bool) -> tuple[ActionOutcome, ...]:
    values: list[ActionOutcome] = []
    for execution_position, (action, latency) in enumerate(
        zip(REGISTERED_ACTIONS, (5.0, 10.0, 20.0, 1.0), strict=True)
    ):
        abstained = action == "abstain"
        retrieval = not abstained and not (action == "hnsw-low" and failed_low)
        evidence_sufficient = None
        if evidence:
            evidence_sufficient = retrieval
        values.append(
            ActionOutcome(
                action=action,
                execution_position=execution_position,
                execution_state="abstained" if abstained else "completed",
                failure_state="registered-abstention" if abstained else None,
                request_latency_ms=latency,
                retrieval_attained=retrieval,
                qrel_recall_at_k=float(retrieval),
                evidence_sufficient=evidence_sufficient,
                entitlement_violations=0,
                returned_document_rows=() if not retrieval else (0,),
                returned_document_ids=() if not retrieval else ("document-0",),
            )
        )
    return tuple(values)


def _partition(stage: str) -> DevelopmentPartitionData:
    prefix = "fit" if stage == "development-fit" else "calibration"
    trials: list[DevelopmentTrial] = []
    for corpus_index, corpus in enumerate(FIXED_CORPORA):
        for family_index in range(2):
            failed = family_index == 1
            for rate_index, rate in enumerate((0.25, 0.50, 0.75)):
                if failed:
                    geometry = (100.0, 0.50, 1.0, 3.0)
                    drift = 1.0
                    churn = 0.10
                else:
                    geometry = (2.0, 0.01, 3.0, 1.1)
                    drift = 0.01
                    churn = 0.01
                trial_key = _digest(f"{prefix}:{corpus}:{family_index}:{rate_index}")
                feature_values = (
                    1_000.0,
                    1_000.0 * rate,
                    16.0,
                    1.0,
                    drift,
                    1.0 + 0.1 * corpus_index,
                    101.0,
                    corpus,
                    "hnswlib-0.8.0",
                    "revision-lag-one",
                    rate,
                    1.0 + rate_index,
                    churn,
                    *geometry,
                )
                trials.append(
                    DevelopmentTrial(
                        partition=stage,
                        corpus_id=corpus,
                        family_id=f"{prefix}-{corpus}-family-{family_index}",
                        query_id=f"{prefix}-{corpus}-query-{family_index}",
                        trial_key=trial_key,
                        subject="subject-1",
                        repetition=0,
                        target_allow_rate=rate,
                        realized_allow_rate=rate,
                        authorized_count=int(1_000 * rate),
                        feature_values=feature_values,
                        label=int(failed),
                        outcomes=_outcomes(
                            evidence=corpus in EVIDENCE_CORPORA,
                            failed_low=failed,
                        ),
                    )
                )
    return DevelopmentPartitionData(stage, tuple(trials))


def test_canonical_config_round_trip_has_closed_exact_schema(tmp_path: Path) -> None:
    config = _config(tmp_path)
    encoded = freeze.canonical_development_freeze_config_bytes(config)
    config_path = tmp_path / "development-freeze-config.json"
    _write(config_path, encoded)

    restored = freeze.load_development_freeze_config(config_path)

    assert restored == config
    assert freeze.canonical_development_freeze_config_bytes(restored) == encoded
    decoded = json.loads(encoded)
    assert set(decoded) == {
        "failure_recall_threshold",
        "k",
        "model_seed",
        "output_root",
        "schema_version",
        "selection_receipt",
        "sources",
    }
    assert decoded["schema_version"] == freeze.DEVELOPMENT_FREEZE_CONFIG_SCHEMA
    assert len(decoded["sources"]) == 10


def test_config_loader_rejects_unknown_duplicate_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    canonical = freeze.canonical_development_freeze_config_bytes(config)
    decoded = json.loads(canonical)
    decoded["unexpected"] = True
    unknown_path = tmp_path / "unknown-config.json"
    _write(unknown_path, _canonical(decoded))
    with pytest.raises(DevelopmentFreezeError, match="keys differ"):
        freeze.load_development_freeze_config(unknown_path)

    nested = json.loads(canonical)
    nested["sources"][0]["queries"]["unexpected"] = True
    nested_unknown_path = tmp_path / "nested-unknown-config.json"
    _write(nested_unknown_path, _canonical(nested))
    with pytest.raises(DevelopmentFreezeError, match="keys differ"):
        freeze.load_development_freeze_config(nested_unknown_path)

    duplicate_path = tmp_path / "duplicate-config.json"
    _write(
        duplicate_path,
        (
            b'{"schema_version":"'
            + freeze.DEVELOPMENT_FREEZE_CONFIG_SCHEMA.encode("ascii")
            + b'",'
            + canonical[1:]
        ),
    )
    with pytest.raises(DevelopmentFreezeError, match="repeats key"):
        freeze.load_development_freeze_config(duplicate_path)

    noncanonical_path = tmp_path / "noncanonical-config.json"
    _write(noncanonical_path, canonical.replace(b"{", b"{ ", 1))
    with pytest.raises(DevelopmentFreezeError, match="not canonical"):
        freeze.load_development_freeze_config(noncanonical_path)


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ("relative/queries.jsonl", "absolute POSIX"),
        ("/tmp/development/tbd/queries.jsonl", "placeholder"),
        ("/tmp/development/<queries>.jsonl", "absolute POSIX"),
        ("/tmp/development/sealed/queries.jsonl", "forbidden sealed-role"),
        ("/tmp/development/custody/queries.jsonl", "forbidden sealed-role"),
    ),
)
def test_config_loader_rejects_relative_placeholder_and_outcome_paths(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    decoded = json.loads(freeze.canonical_development_freeze_config_bytes(_config(tmp_path)))
    decoded["sources"][0]["queries"]["path"] = replacement
    config_path = tmp_path / f"rejected-{hashlib.sha256(replacement.encode()).hexdigest()}.json"
    _write(config_path, _canonical(decoded))

    with pytest.raises(DevelopmentFreezeError, match=message):
        freeze.load_development_freeze_config(config_path)


def test_config_loader_rejects_relative_config_path() -> None:
    with pytest.raises(DevelopmentFreezeError, match="absolute POSIX"):
        freeze.load_development_freeze_config("development-freeze-config.json")


def test_cli_rejects_custody_path_before_loading_any_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    safe_root = tmp_path.parent / f"config-preflight-{_digest(str(tmp_path))[:12]}"
    safe_root.mkdir()
    config = _config(safe_root)
    decoded = json.loads(freeze.canonical_development_freeze_config_bytes(config))
    decoded["sources"][-1]["paired_actions"]["path"] = (
        "/controlled/development/custody/paired-actions.jsonl"
    )
    config_path = safe_root / "rejected-config.json"
    _write(config_path, _canonical(decoded))
    loaded = False

    def fail_if_loaded(_config: DevelopmentFreezeConfig) -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("development source opened before config rejection")

    monkeypatch.setattr(freeze, "_load_development_sources", fail_if_loaded)

    with pytest.raises(SystemExit) as raised:
        freeze.main(["compile", "--config", str(config_path)])

    assert raised.value.code == 2
    assert "forbidden sealed-role" in capsys.readouterr().err
    assert loaded is False
    assert not config.output_root.exists()


def _file_backed_source(
    root: Path,
) -> tuple[DevelopmentCorpusSources, EmbeddingStoreReceipt]:
    corpus = "scifact"
    stage = "development-fit"
    source_root = root / stage / corpus
    query_ids = ("query-0", "query-1")
    queries_bytes = b"".join(
        _canonical({"id": query_id, "text": f"Question {position}"})
        for position, query_id in enumerate(query_ids)
    )
    qrels_bytes = b"".join(
        _canonical(
            {
                "document_id": f"document-{position}",
                "query_id": query_id,
                "relevance": 1,
            }
        )
        for position, query_id in enumerate(query_ids)
    )
    evidence_bytes = b"".join(
        _canonical(
            {
                "answer": None,
                "evidence_bundles": [
                    {
                        "bundle_id": f"bundle-{position}",
                        "locations": [
                            {
                                "document_id": f"document-{position}",
                                "locator": f"line-{position}",
                            }
                        ],
                    }
                ],
                "label_metadata": [],
                "query_id": query_id,
            }
        )
        for position, query_id in enumerate(query_ids)
    )
    queries_path = source_root / "queries.jsonl"
    qrels_path = source_root / "qrels.jsonl"
    evidence_path = source_root / "evidence-bundles.jsonl"
    _write(queries_path, queries_bytes)
    _write(qrels_path, qrels_bytes)
    _write(evidence_path, evidence_bytes)

    policy_revision = f"sha256:{_digest('policy-bundle')}"
    baseline_policy_revision = f"sha256:{_digest('baseline-policy-bundle')}"
    states = ("low", "medium", "high")
    rates = (0.25, 0.50, 0.75)
    assignment_seed = _digest("development-assignment-seed")
    baseline_seed = _digest("baseline-development-assignment-seed")
    family_keys = {query_id: _digest(f"family:{query_id}") for query_id in query_ids}
    trial_assignments: dict[tuple[str, str], str] = {}
    for query_id in query_ids:
        family_key = family_keys[query_id]
        candidates = tuple(_digest(f"trial:{query_id}:{position}") for position in range(3))
        ranked = sorted(
            candidates,
            key=lambda trial_key: (
                intervention._trial_state_rank_sha256(
                    seed_sha256=assignment_seed,
                    family_key=family_key,
                    trial_key=trial_key,
                ),
                trial_key,
            ),
        )
        trial_assignments.update(
            {(query_id, state): trial_key for state, trial_key in zip(states, ranked, strict=True)}
        )
    schedule_rows: list[TrialScheduleRow] = []
    paired_rows: list[dict[str, object]] = []
    schedule_order = 0
    for group_order, (state, rate) in enumerate(zip(states, rates, strict=True)):
        for query_position, query_id in enumerate(query_ids):
            trial_key = trial_assignments[(query_id, state)]
            environment = {
                "assignment_repetition": 0,
                "policy_state": state,
            }
            schedule_rows.append(
                TrialScheduleRow(
                    schedule_order=schedule_order,
                    group_order=group_order,
                    trial_key=trial_key,
                    family_key=family_keys[query_id],
                    repetition=0,
                    subject="subject-1",
                    environment=tuple(sorted(environment.items())),
                    environment_sha256=policy_environment_sha256(environment),
                    baseline_policy_revision=baseline_policy_revision,
                    baseline_mask_id=f"baseline-mask-{state}",
                    baseline_mask_path=f"baseline-masks/baseline-mask-{state}.bin",
                    baseline_mask_sha256=_digest(f"baseline-mask:{state}"),
                    baseline_mask_byte_count=1,
                    baseline_authorized_count=group_order + 1,
                    mask_id=f"mask-{state}",
                    mask_sha256=_digest(f"mask:{state}"),
                    authorized_count=group_order + 1,
                    realized_allow_rate=rate,
                    policy_churn=0.01,
                    expected_policy_revision=policy_revision,
                )
            )
            drift = float(query_position)
            features = {
                "allow_rate": rate,
                "authorized_universe_size": float(group_order + 1),
                "backend": "hnswlib-0.8.0",
                "corpus_size": 4.0,
                "corpus_stratum": corpus,
                "drift_family": "revision-lag-one",
                "drift_severity": drift,
                "embedding_dimension": 2.0,
                "lid_cv": 0.05 + 0.20 * query_position,
                "lid_k50": 5.0 + 40.0 * query_position,
                "policy_churn": 0.01,
                "policy_complexity": 1.0,
                "probe_latency_ms": 0.5,
                "probe_work": 101.0,
                "radius_expansion": 1.1 + query_position,
                "relative_contrast": 2.0 - 0.8 * query_position,
                "version_lag": 1.0,
            }
            for execution_position, (action, latency) in enumerate(
                zip(
                    REGISTERED_ACTIONS,
                    (5.0, 10.0, 20.0, 1.0),
                    strict=True,
                )
            ):
                abstained = action == "abstain"
                low_failure = action == "hnsw-low" and query_position == 1
                returned = [] if abstained or low_failure else [query_position]
                paired_rows.append(
                    {
                        "action": action,
                        "entitlement_violations": 0,
                        "execution_position": execution_position,
                        "execution_state": "abstained" if abstained else "completed",
                        "failure_state": "registered-abstention" if abstained else None,
                        "family_id": f"family-{query_position}",
                        "feature_values": features if action == "hnsw-low" else None,
                        "query_id": query_id,
                        "request_latency_ms": latency,
                        "returned_document_rows": returned,
                        "schedule_order": schedule_order,
                        "schema_version": freeze.PAIRED_ACTION_ROW_SCHEMA,
                        "trial_key": trial_key,
                    }
                )
            schedule_order += 1
    frozen_schedule_rows = tuple(schedule_rows)
    schedule = CanonicalTrialSchedule(
        execution_artifact_sha256=_digest("development-execution"),
        corpus=corpus,
        stage=stage,
        document_count=4,
        document_universe_sha256=_digest("document-universe"),
        config_sha256=_digest("policy-config"),
        policy_bundle_revision=policy_revision,
        mask_catalog_sha256=_digest("mask-catalog"),
        grouped_execution_order=states,
        assignment_seed_sha256=assignment_seed,
        baseline_seed_sha256=baseline_seed,
        baseline_policy_revision=baseline_policy_revision,
        assignment_algorithm=intervention.TRIAL_STATE_ASSIGNMENT_ALGORITHM,
        assignment_map_sha256=intervention._trial_assignment_map_sha256(frozen_schedule_rows),
        rows=frozen_schedule_rows,
    )
    schedule_path = source_root / "trial-schedule.json"
    schedule_bytes = _write(schedule_path, schedule.canonical_file_bytes())
    paired_path = source_root / "paired-actions.jsonl"
    paired_bytes = _write(paired_path, b"".join(_canonical(row) for row in paired_rows))

    embedding_root = source_root / "embedding-store"
    embedding_root.mkdir()
    document_order_bytes = b"".join(
        _canonical(
            {
                "dataset": corpus,
                "id": f"document-{position}",
                "kind": "document",
                "source_path": "corpus.jsonl",
                "source_row": position + 1,
                "stage": None,
            }
        )
        for position in range(4)
    )
    query_order_bytes = b"".join(
        _canonical(
            {
                "dataset": corpus,
                "id": query_id,
                "kind": "query",
                "source_path": "queries.jsonl",
                "source_row": position + 1,
                "stage": stage,
            }
        )
        for position, query_id in enumerate(query_ids)
    )
    document_order_path = embedding_root / "document-order.jsonl"
    query_order_path = embedding_root / "query-order.jsonl"
    _write(document_order_path, document_order_bytes)
    _write(query_order_path, query_order_bytes)

    vectors = {
        "old_documents": np.asarray([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32),
        "current_documents": np.asarray([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32),
        "old_queries": np.asarray([[1, 0], [1, 0]], dtype=np.float32),
        "current_queries": np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    }
    vector_descriptors: dict[str, VectorDescriptor] = {}
    old_model = {
        "encoder_id": "local-test-encoder",
        "revision": "old-revision",
        "tree_sha256": _digest("old-model-tree"),
    }
    current_model = {
        "encoder_id": "local-test-encoder",
        "revision": "current-revision",
        "tree_sha256": _digest("current-model-tree"),
    }
    for name, matrix in vectors.items():
        vector_path = embedding_root / f"{name}.npy"
        np.save(vector_path, matrix, allow_pickle=False)
        encoded = vector_path.read_bytes()
        is_query = name.endswith("queries")
        model = old_model if name.startswith("old_") else current_model
        vector_descriptors[name] = VectorDescriptor(
            relative_path=vector_path.name,
            dtype="float32",
            shape=matrix.shape,
            row_order_sha256=hashlib.sha256(
                query_order_bytes if is_query else document_order_bytes
            ).hexdigest(),
            byte_count=len(encoded),
            file_sha256=hashlib.sha256(encoded).hexdigest(),
            model_tree_sha256=model["tree_sha256"],
            model_revision=model["revision"],
            prompt_sha256=_digest("query-prompt" if is_query else "document-prompt"),
        )
    receipt = EmbeddingStoreReceipt(
        staged_inventory_sha256=_digest("staged-inventory"),
        source_inventory_sha256=_digest("source-inventory"),
        config_sha256=_digest("embedding-config"),
        document_count=4,
        query_count=2,
        current_model=current_model,
        old_model=old_model,
        row_orders={
            "documents": RowOrderDescriptor(
                relative_path=document_order_path.name,
                row_count=4,
                byte_count=len(document_order_bytes),
                row_order_sha256=hashlib.sha256(document_order_bytes).hexdigest(),
                file_sha256=hashlib.sha256(document_order_bytes).hexdigest(),
            ),
            "queries": RowOrderDescriptor(
                relative_path=query_order_path.name,
                row_count=2,
                byte_count=len(query_order_bytes),
                row_order_sha256=hashlib.sha256(query_order_bytes).hexdigest(),
                file_sha256=hashlib.sha256(query_order_bytes).hexdigest(),
            ),
        },
        vectors=vector_descriptors,
    )
    source = DevelopmentCorpusSources(
        corpus_id=corpus,
        stage=stage,
        queries=_pinned_path(
            queries_path,
            queries_bytes,
            corpus=corpus,
            stage=stage,
            role="queries",
        ),
        qrels=_pinned_path(
            qrels_path,
            qrels_bytes,
            corpus=corpus,
            stage=stage,
            role="qrels",
        ),
        evidence_bundles=_pinned_path(
            evidence_path,
            evidence_bytes,
            corpus=corpus,
            stage=stage,
            role="evidence-bundles",
        ),
        policy_schedule=_pinned_path(
            schedule_path,
            schedule_bytes,
            corpus=corpus,
            stage=stage,
            role="policy-schedule",
        ),
        paired_actions=_pinned_path(
            paired_path,
            paired_bytes,
            corpus=corpus,
            stage=stage,
            role="paired-actions",
        ),
        embedding_store=PinnedEmbeddingStore(
            root=embedding_root,
            receipt_sha256=receipt.receipt_sha256,
            corpus_id=corpus,
            stage=stage,
        ),
    )
    return source, receipt


def test_sealed_role_path_is_rejected_before_any_source_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    first = config.sources[0]
    sealed_query = replace(
        first.queries,
        path=Path("/tmp/development/sealed-labels/queries.jsonl"),
    )
    sealed_source = replace(first, queries=sealed_query)
    config = replace(config, sources=(sealed_source, *config.sources[1:]))
    opened = False

    def fail_if_loaded(_config: DevelopmentFreezeConfig):
        nonlocal opened
        opened = True
        raise AssertionError("source loader ran before sealed-role rejection")

    monkeypatch.setattr(freeze, "_load_development_sources", fail_if_loaded)
    with pytest.raises(DevelopmentFreezeError, match="forbidden sealed-role"):
        compile_development_freeze(config)
    assert opened is False
    assert not config.output_root.exists()


def test_file_backed_join_uses_dual_epoch_drift_schedule_qrels_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, embedding_receipt = _file_backed_source(tmp_path / "development-inputs")
    monkeypatch.setattr(
        freeze,
        "verify_embedding_store",
        lambda root: embedding_receipt,
    )

    trials = freeze._load_one_source(source)

    assert len(trials) == 6
    assert {trial.subject for trial in trials} == {"subject-1"}
    assert {trial.repetition for trial in trials} == {0}
    assert {trial.target_allow_rate for trial in trials} == {0.25, 0.50, 0.75}
    by_query = {query_id: [] for query_id in ("query-0", "query-1")}
    for trial in trials:
        by_query[trial.query_id].append(trial)
    drift_index = freeze.REGISTERED_FEATURE_SCHEMA.input_features.index("drift_severity")
    assert {trial.feature_values[drift_index] for trial in by_query["query-0"]} == {0.0}
    assert {trial.feature_values[drift_index] for trial in by_query["query-1"]} == {1.0}
    assert {trial.label for trial in by_query["query-0"]} == {0}
    assert {trial.label for trial in by_query["query-1"]} == {1}
    assert all(trial.outcome("hnsw-high").evidence_sufficient is True for trial in trials)


def test_compiler_freezes_models_controller_profiles_and_power_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    fit = _partition("development-fit")
    calibration = _partition("development-calibration")
    monkeypatch.setattr(
        freeze,
        "_load_development_sources",
        lambda _config: (fit, calibration),
    )

    receipt = compile_development_freeze(config)
    verified = verify_development_freeze(config.output_root)
    second_config = replace(config, output_root=tmp_path / "development-freeze-repeat")
    second_receipt = compile_development_freeze(second_config)

    assert verified == receipt
    assert second_receipt == receipt
    assert {path.name: path.read_bytes() for path in config.output_root.iterdir()} == {
        path.name: path.read_bytes() for path in second_config.output_root.iterdir()
    }
    assert receipt["static_comparator"] == "hnsw-high"
    assert len(receipt["source_bindings"]) == 57
    assert any(row["role"] == "development-cohort-selection" for row in receipt["source_bindings"])
    controller = json.loads((config.output_root / "controller.json").read_text())
    assert controller["selected_metrics"]["passes_constraints"] is True
    assert controller["selected_metrics"]["equal_corpus_retrieval_loss"] == 0.0
    assert controller["constraints"]["retrieval_loss_maximum"] == 0.005
    assert controller["constraints"]["p95_latency_ratio_strict_upper_bound"] == 1.2

    profiles = json.loads((config.output_root / "geometry-profiles.json").read_text())
    assert profiles["fit_partition_only"] is True
    assert profiles["low_geometry"]["lid_k50"] < profiles["high_geometry"]["lid_k50"]
    assert (
        profiles["low_geometry"]["relative_contrast"]
        > profiles["high_geometry"]["relative_contrast"]
    )
    assert profiles["geometry_gain_thresholds"] == {
        "auprc_gain": 0.005,
        "brier_score_reduction": 0.001,
        "log_loss_reduction": 0.002,
    }

    expected = load_development_panel(
        (config.output_root / "joint-power-expected-panel.json").read_bytes()
    )
    conservative = load_development_panel(
        (config.output_root / "joint-power-conservative-panel.json").read_bytes()
    )
    power = load_joint_power_config((config.output_root / "joint-power-config.json").read_bytes())
    assert power.nested_rows_per_family == 3
    assert power.geometry_gain_thresholds.log_loss_reduction == 0.002
    assert {scenario.panel_sha256 for scenario in power.effect_scenarios} == {
        expected.sha256,
        conservative.sha256,
    }
    expected_by_id = {row.row_id: row for row in expected.rows}
    for row in conservative.rows:
        source = expected_by_id[row.row_id.removeprefix("conservative:")]
        expected_logit_gap = abs(
            math.log(source.full_probability / (1.0 - source.full_probability))
            - math.log(source.reference_probability / (1.0 - source.reference_probability))
        )
        conservative_logit_gap = abs(
            math.log(row.full_probability / (1.0 - row.full_probability))
            - math.log(row.reference_probability / (1.0 - row.reference_probability))
        )
        assert conservative_logit_gap == pytest.approx(0.75 * expected_logit_gap)
        assert row.proposed_retrieval_attained <= source.proposed_retrieval_attained


def test_module_cli_compiles_and_verifies_from_canonical_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "development-freeze-config.json"
    _write(config_path, freeze.canonical_development_freeze_config_bytes(config))
    partitions = (_partition("development-fit"), _partition("development-calibration"))
    monkeypatch.setattr(
        freeze,
        "_load_development_sources",
        lambda _config: partitions,
    )

    assert freeze.main(["compile", "--config", str(config_path)]) == 0
    compile_result = json.loads(capfd.readouterr().out)
    assert compile_result == {
        "command": "compile",
        "receipt_sha256": compile_result["receipt_sha256"],
        "root": str(config.output_root),
        "schema_version": freeze.DEVELOPMENT_FREEZE_CLI_RESULT_SCHEMA,
    }
    assert len(compile_result["receipt_sha256"]) == 64

    assert freeze.main(["verify", "--root", str(config.output_root)]) == 0
    verify_result = json.loads(capfd.readouterr().out)
    assert verify_result["command"] == "verify"
    assert verify_result["root"] == str(config.output_root)
    assert verify_result["receipt_sha256"] == compile_result["receipt_sha256"]


def test_existing_output_blocks_second_compile_before_loading_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    partitions = (_partition("development-fit"), _partition("development-calibration"))
    calls = 0

    def load(_config: DevelopmentFreezeConfig):
        nonlocal calls
        calls += 1
        return partitions

    monkeypatch.setattr(freeze, "_load_development_sources", load)
    compile_development_freeze(config)
    with pytest.raises(DevelopmentFreezeError, match="already exists"):
        compile_development_freeze(config)
    assert calls == 1


def test_verifier_rejects_rehashed_derived_artifact_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    partitions = (_partition("development-fit"), _partition("development-calibration"))
    monkeypatch.setattr(
        freeze,
        "_load_development_sources",
        lambda _config: partitions,
    )
    compile_development_freeze(config)

    controller_path = config.output_root / "controller.json"
    controller = json.loads(controller_path.read_text())
    controller["selected_metrics"]["equal_corpus_family_latency_reduction"] += 0.01
    controller_bytes = _canonical(controller)
    controller_path.write_bytes(controller_bytes)
    receipt_path = config.output_root / "freeze-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    controller_binding = next(
        row for row in receipt["artifacts"] if row["path"] == "controller.json"
    )
    controller_binding["byte_count"] = len(controller_bytes)
    controller_binding["sha256"] = hashlib.sha256(controller_bytes).hexdigest()
    receipt_path.write_bytes(_canonical(receipt))

    with pytest.raises(DevelopmentFreezeError, match="does not reproduce"):
        verify_development_freeze(config.output_root)


def test_dual_epoch_drift_is_one_minus_cosine_and_rejects_zero_norm() -> None:
    assert freeze._cosine_drift(
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
    ) == pytest.approx(1.0)
    assert freeze._cosine_drift(
        np.asarray([1.0, 0.0]),
        np.asarray([-1.0, 0.0]),
    ) == pytest.approx(2.0)
    with pytest.raises(DevelopmentFreezeError, match="zero norm"):
        freeze._cosine_drift(np.zeros(2), np.ones(2))


def test_source_derived_feature_fields_cannot_be_spoofed() -> None:
    values = dict(
        zip(
            freeze.REGISTERED_FEATURE_SCHEMA.input_features,
            _partition("development-fit").trials[0].feature_values,
            strict=True,
        )
    )
    values["version_lag"] = 2.0
    with pytest.raises(DevelopmentFreezeError, match="version_lag"):
        freeze._feature_tuple(
            values,
            document_count=1_000,
            authorized_count=250,
            dimension=16,
            drift=0.01,
            target_allow_rate=0.25,
        )
