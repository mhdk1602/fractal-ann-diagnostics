from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.artifact_pipeline as pipeline
from fractal_ann_diagnostics.artifact_pipeline import (
    ARTIFACT_PIPELINE_ORDER,
    ArtifactPipelineError,
    RuntimePackageVerification,
    build_artifact_pipeline,
    verify_artifact_pipeline,
    verify_runtime_package,
)
from fractal_ann_diagnostics.joint_power_design import FIXED_CORPORA


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _empty_artifact_tree(root: Path) -> None:
    for role in (
        "embedding-stores",
        "policy-workloads",
        "authorized-index-stores",
        "trial-runtime",
    ):
        for corpus_id in FIXED_CORPORA:
            (root / role / corpus_id).mkdir(parents=True)


def _install_pipeline_spies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    inventory_sha256: str,
) -> list[str]:
    calls: list[str] = []

    def verify_projection(
        root: Path,
        *,
        expected_inventory_sha256: str,
    ) -> SimpleNamespace:
        assert root.is_dir()
        assert expected_inventory_sha256 == inventory_sha256
        calls.append("projection")
        return SimpleNamespace(
            inventory_sha256=inventory_sha256,
            projected_artifact_set_sha256=_digest("projected-artifact-set"),
        )

    def verify_embedding(root: Path) -> SimpleNamespace:
        corpus_id = root.name
        calls.append(f"{corpus_id}:embedding")
        return SimpleNamespace(
            receipt_sha256=_digest(f"{corpus_id}:embedding"),
            staged_inventory_sha256=inventory_sha256,
        )

    def policy_receipt(corpus_id: str) -> SimpleNamespace:
        return SimpleNamespace(receipt_sha256=_digest(f"{corpus_id}:policy"))

    def seal_policy(root: Path, *, corpus_id: str) -> SimpleNamespace:
        assert root.name == corpus_id
        calls.append(f"{corpus_id}:policy")
        return policy_receipt(corpus_id)

    def verify_policy(root: Path, *, expected_corpus_id: str) -> SimpleNamespace:
        return seal_policy(root, corpus_id=expected_corpus_id)

    def index_receipt(corpus_id: str) -> SimpleNamespace:
        return SimpleNamespace(receipt_sha256=_digest(f"{corpus_id}:index"))

    def seal_index(
        root: Path,
        *,
        corpus_id: str,
        embedding_store_root: Path,
        policy_bundle_root: Path,
    ) -> SimpleNamespace:
        assert root.name == embedding_store_root.name == policy_bundle_root.name == corpus_id
        calls.append(f"{corpus_id}:index")
        return index_receipt(corpus_id)

    def verify_index(
        root: Path,
        *,
        embedding_store_root: Path,
        policy_bundle_root: Path,
        expected_corpus_id: str,
    ) -> SimpleNamespace:
        return seal_index(
            root,
            corpus_id=expected_corpus_id,
            embedding_store_root=embedding_store_root,
            policy_bundle_root=policy_bundle_root,
        )

    def verify_runtime(root: Path, **kwargs: object) -> RuntimePackageVerification:
        corpus_id = str(kwargs["corpus_id"])
        assert root.name == corpus_id
        assert kwargs["online_inventory_sha256"] == inventory_sha256
        calls.append(f"{corpus_id}:runtime")
        return RuntimePackageVerification(
            tree_sha256=_digest(f"{corpus_id}:runtime-tree"),
            execution_plan_sha256=_digest(f"{corpus_id}:plan"),
            query_receipt_sha256=_digest(f"{corpus_id}:query"),
            runtime_receipt_sha256=_digest(f"{corpus_id}:runtime"),
            query_count=225,
        )

    monkeypatch.setattr(pipeline, "verify_online_staging_projection", verify_projection)
    monkeypatch.setattr(pipeline, "verify_embedding_store", verify_embedding)
    monkeypatch.setattr(pipeline, "seal_policy_stage_bundle", seal_policy)
    monkeypatch.setattr(pipeline, "verify_policy_stage_bundle", verify_policy)
    monkeypatch.setattr(pipeline, "seal_index_stage_bundle", seal_index)
    monkeypatch.setattr(pipeline, "verify_index_stage_bundle", verify_index)
    monkeypatch.setattr(pipeline, "verify_runtime_package", verify_runtime)
    return calls


def test_five_corpus_pipeline_is_fixed_order_reproducible_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = (tmp_path / "artifacts").resolve()
    staging = (tmp_path / "online-staging").resolve()
    staging.mkdir()
    _empty_artifact_tree(artifacts)
    inventory_sha256 = _digest("online-inventory")
    calls = _install_pipeline_spies(
        monkeypatch,
        inventory_sha256=inventory_sha256,
    )
    receipt_path = tmp_path / "artifact-pipeline.json"

    built = build_artifact_pipeline(
        artifacts,
        staging,
        receipt_path,
        expected_online_inventory_sha256=inventory_sha256,
    )
    expected_calls = ["projection"]
    for corpus_id in FIXED_CORPORA:
        expected_calls.extend(
            (
                f"{corpus_id}:embedding",
                f"{corpus_id}:policy",
                f"{corpus_id}:index",
                f"{corpus_id}:runtime",
            )
        )
    assert calls == expected_calls
    assert tuple(row.corpus_id for row in built.corpora) == FIXED_CORPORA
    assert built.artifact_order == ARTIFACT_PIPELINE_ORDER

    calls.clear()
    assert verify_artifact_pipeline(artifacts, staging, receipt_path) == built
    assert calls == expected_calls
    assert receipt_path.read_bytes() == built.canonical_file_bytes()


def test_runtime_package_rejects_label_or_undeclared_membership(tmp_path: Path) -> None:
    runtime_root = (tmp_path / "runtime").resolve()
    runtime_root.mkdir()
    for relative_path in pipeline._RUNTIME_EXPECTED_ENTRIES:
        target = runtime_root / relative_path
        if relative_path == pipeline.QUERY_PACKAGE_DIRECTORY:
            target.mkdir(exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"{}\n")
    (runtime_root / "sealed-labels.json").write_bytes(b"forbidden\n")

    with pytest.raises(ArtifactPipelineError, match="membership differs"):
        verify_runtime_package(
            runtime_root,
            corpus_id="scifact",
            online_inventory_sha256=_digest("online-inventory"),
            embedding_receipt=SimpleNamespace(),  # type: ignore[arg-type]
            policy_bundle=SimpleNamespace(),  # type: ignore[arg-type]
            index_bundle=SimpleNamespace(),  # type: ignore[arg-type]
        )


def test_runtime_package_reproduces_the_full_label_free_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = (tmp_path / "runtime").resolve()
    query_root = runtime_root / pipeline.QUERY_PACKAGE_DIRECTORY
    query_root.mkdir(parents=True)
    for relative_path in (
        pipeline.QUERY_TRIAL_FILENAME,
        pipeline.QUERY_TRIAL_RECEIPT_FILENAME,
    ):
        (query_root / relative_path).write_bytes(b"{}\n")
    for relative_path in (
        pipeline.SHARDED_PLAN_FILENAME,
        pipeline.TRIAL_RUNTIME_ADMISSION_FILENAME,
    ):
        (runtime_root / relative_path).write_bytes(b"{}\n")

    inventory = _digest("online-inventory")
    source_inventory = _digest("source-inventory")
    universe = _digest("document-universe")
    embedding_receipt_sha256 = _digest("embedding-receipt")

    def vector(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            file_sha256=_digest(f"{name}:file"),
            row_order_sha256=(universe if name.endswith("documents") else _digest("query-order")),
            model_tree_sha256=_digest(f"{name}:model-tree"),
            model_revision=f"{name}-revision",
            prompt_sha256=_digest(f"{name}:prompt"),
            dtype="float32",
            shape=(19, 4) if name.endswith("documents") else (3, 4),
        )

    old_documents = vector("old_documents")
    current_documents = vector("current_documents")
    old_queries = vector("old_queries")
    current_queries = vector("current_queries")
    embedding = SimpleNamespace(
        receipt_sha256=embedding_receipt_sha256,
        staged_inventory_sha256=inventory,
        source_inventory_sha256=source_inventory,
        document_count=19,
        row_orders={"documents": SimpleNamespace(row_order_sha256=universe)},
        vectors={
            "old_documents": old_documents,
            "current_documents": current_documents,
            "old_queries": old_queries,
            "current_queries": current_queries,
        },
    )

    def epoch(role: str, source: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            role=role,
            file_sha256=source.file_sha256,
            row_order_sha256=source.row_order_sha256,
            model_tree_sha256=source.model_tree_sha256,
            model_revision=source.model_revision,
            prompt_sha256=source.prompt_sha256,
            dtype=source.dtype,
            shape=source.shape,
        )

    active_epoch = epoch("active-migration", old_queries)
    truth_epoch = epoch("current-exact-truth", current_queries)
    trials = tuple(
        sorted(
            (SimpleNamespace(trial_key=_digest(f"trial-{position}")) for position in range(3)),
            key=lambda row: row.trial_key,
        )
    )
    query_store_sha256 = _digest("query-store")
    query_receipt_sha256 = _digest("query-receipt")
    partition_audit_sha256 = _digest("partition-audit")
    policy_receipt_sha256 = _digest("policy-receipt")
    policy_config_sha256 = _digest("policy-config")
    policy_catalog_sha256 = _digest("policy-catalog")
    policy_schedule_sha256 = _digest("policy-schedule")
    policy_revision = f"sha256:{_digest('policy-revision')}"
    assignment_seed_sha256 = _digest("assignment-seed")
    assignment_map_sha256 = _digest("assignment-map")
    plan = SimpleNamespace(
        corpus="scifact",
        stage="sealed",
        artifact_sha256=_digest("execution-plan"),
        document_count=19,
        ordered_document_universe_sha256=universe,
        query_partition_audit_sha256=partition_audit_sha256,
        permutation_seed=20260714,
        trials=trials,
        query_trial_store=SimpleNamespace(
            artifact=SimpleNamespace(
                relative_path=(
                    f"{pipeline.QUERY_PACKAGE_DIRECTORY}/{pipeline.QUERY_TRIAL_FILENAME}"
                ),
                sha256=query_store_sha256,
                byte_count=5,
            ),
            receipt=SimpleNamespace(
                relative_path=(
                    f"{pipeline.QUERY_PACKAGE_DIRECTORY}/{pipeline.QUERY_TRIAL_RECEIPT_FILENAME}"
                ),
                sha256=query_receipt_sha256,
                byte_count=7,
            ),
        ),
        active_vector_store=SimpleNamespace(
            artifact=SimpleNamespace(sha256=old_documents.file_sha256)
        ),
        current_truth_vector_store=SimpleNamespace(
            artifact=SimpleNamespace(sha256=current_documents.file_sha256)
        ),
    )
    query = SimpleNamespace(
        corpus="scifact",
        stage="sealed",
        query_trial_store_sha256=query_store_sha256,
        query_trial_store_byte_count=5,
        receipt_sha256=query_receipt_sha256,
        receipt_byte_count=7,
        opaque_trials=trials,
        record_count=3,
        query_partition_audit_sha256=partition_audit_sha256,
        embedding_store_receipt_sha256=embedding_receipt_sha256,
        staged_inventory_sha256=inventory,
        source_inventory_sha256=source_inventory,
        active_query_epoch=active_epoch,
        current_truth_query_epoch=truth_epoch,
    )
    runtime = SimpleNamespace(
        receipt_sha256=_digest("runtime-receipt"),
        execution_artifact_sha256=plan.artifact_sha256,
        query_trial_store_sha256=query_store_sha256,
        query_count=3,
        query_partition_audit_sha256=partition_audit_sha256,
        permutation_seed=plan.permutation_seed,
        embedding_store_receipt_sha256=embedding_receipt_sha256,
        staged_inventory_sha256=inventory,
        source_inventory_sha256=source_inventory,
        active_query_epoch=active_epoch,
        current_truth_query_epoch=truth_epoch,
        policy_bundle_revision=policy_revision,
        policy_config_sha256=policy_config_sha256,
        mask_catalog_sha256=policy_catalog_sha256,
        schedule_sha256=policy_schedule_sha256,
        assignment_seed_sha256=assignment_seed_sha256,
        assignment_map_sha256=assignment_map_sha256,
    )
    sealed_policy = SimpleNamespace(
        trial_count=3,
        receipt_sha256=policy_receipt_sha256,
        policy_revision=policy_revision,
        config_sha256=policy_config_sha256,
        catalog_sha256=policy_catalog_sha256,
        schedule_sha256=policy_schedule_sha256,
        assignment_seed_sha256=assignment_seed_sha256,
        assignment_map_sha256=assignment_map_sha256,
    )
    sealed_index = SimpleNamespace(
        embedding_receipt_sha256=embedding_receipt_sha256,
        policy_receipt_sha256=policy_receipt_sha256,
        old_active_vector_sha256=old_documents.file_sha256,
        current_truth_vector_sha256=current_documents.file_sha256,
        document_universe_sha256=universe,
    )
    monkeypatch.setattr(pipeline, "load_sharded_online_execution_plan", lambda _: plan)
    monkeypatch.setattr(pipeline, "load_trial_runtime_receipt", lambda _: runtime)
    monkeypatch.setattr(pipeline, "_load_query_receipt", lambda _: query)
    monkeypatch.setattr(pipeline, "_load_query_rows", lambda *_: ())

    verification = verify_runtime_package(
        runtime_root,
        corpus_id="scifact",
        online_inventory_sha256=inventory,
        embedding_receipt=embedding,  # type: ignore[arg-type]
        policy_bundle=SimpleNamespace(stages=(sealed_policy,)),  # type: ignore[arg-type]
        index_bundle=SimpleNamespace(stages=(sealed_index,)),  # type: ignore[arg-type]
    )
    assert verification.query_count == 3

    runtime.assignment_map_sha256 = _digest("forged-assignment-map")
    with pytest.raises(ArtifactPipelineError, match="staged artifact chain"):
        verify_runtime_package(
            runtime_root,
            corpus_id="scifact",
            online_inventory_sha256=inventory,
            embedding_receipt=embedding,  # type: ignore[arg-type]
            policy_bundle=SimpleNamespace(stages=(sealed_policy,)),  # type: ignore[arg-type]
            index_bundle=SimpleNamespace(stages=(sealed_index,)),  # type: ignore[arg-type]
        )
