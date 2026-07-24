from __future__ import annotations

import fcntl
import hashlib
import os
import weakref
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import fractal_ann_diagnostics.production_artifact_factory as factory
from fractal_ann_diagnostics.artifact_integrity import digest_directory_tree
from fractal_ann_diagnostics.authorized_index_store import AuthorizedIndexConfig
from fractal_ann_diagnostics.joint_power_design import FIXED_CORPORA
from fractal_ann_diagnostics.production_artifact_factory import (
    INDEX_REPLICATE_COUNT,
    INDEX_REPLICATE_DIRECTORIES,
    SELECTED_INDEX_REPLICATE,
    FactorySuiteCorpus,
    FullHnswReplicateEvidence,
    FullHnswReproducibilityReceipt,
    IndexReplicateEvidence,
    IndexReproducibilityStageReceipt,
    IndexReproducibilitySuiteReceipt,
    ProductionArtifactFactoryConfig,
    ProductionArtifactFactoryError,
    ProductionArtifactFactoryShardArtifact,
    ProductionArtifactFactoryShardReceipt,
    ProductionArtifactFactoryShardRequest,
    ProductionArtifactFactorySuiteReceipt,
    ProductionCorpusFactoryConfig,
    ProductionCorpusFactoryEvidence,
    ReproducibleIndexPayload,
)

_HMAC_SECRET = b"fixed-production-factory-secret-material-v1"
_HMAC_SECRET_SHA256 = hashlib.sha256(_HMAC_SECRET).hexdigest()
_HMAC_KEY_ID = f"sealed-online-ephemeral-sha256-{_HMAC_SECRET_SHA256}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _index_config() -> AuthorizedIndexConfig:
    return AuthorizedIndexConfig(
        backend_version="0.8.0",
        backend_build_sha256=_digest("hnsw-build"),
        metric="cosine",
        m=16,
        ef_construction=128,
        random_seed=20260714,
        batch_size=512,
        verification_ef=64,
    )


def _config(tmp_path: Path) -> ProductionArtifactFactoryConfig:
    design_seed_sha256 = _digest("design-seed")
    design = factory._derive_factory_design_bindings(design_seed_sha256)
    return ProductionArtifactFactoryConfig(
        artifact_root=(tmp_path / "output").resolve(),
        artifact_stage_order=factory.ARTIFACT_STAGE_ORDER,
        embedding_build_config_path=(tmp_path / "embedding-config.json").resolve(),
        embedding_build_config_sha256=_digest("embedding-config"),
        embedding_source_root=(tmp_path / "embedding-source").resolve(),
        embedding_source_tree_sha256=_digest("embedding-source-tree"),
        embedding_suite_receipt_sha256=_digest("embedding-suite"),
        development_materialization_root=(
            tmp_path / "development-operator" / "materialized-development"
        ).resolve(),
        development_materialization_receipt_sha256=_digest("development"),
        development_operator_root=(tmp_path / "development-operator").resolve(),
        development_operator_receipt_sha256=_digest("development-operator-receipt"),
        development_operator_joint_power_report_tree_sha256=_digest(
            "development-operator-power-tree"
        ),
        design_seed_sha256=design_seed_sha256,
        partition_audit_path=(tmp_path / "partition-audit.json").resolve(),
        partition_audit_sha256=_digest("partition-audit"),
        joint_power_report_path=(
            tmp_path / "development-operator" / "analysis" / "joint-power-design" / "report.json"
        ).resolve(),
        joint_power_report_sha256=_digest("joint-power"),
        runner_image=f"ghcr.io/mhdk1602/fractal-c0@sha256:{_digest('runner')}",
        runner_platform="linux/arm64",
        policy_seed_sha256=design.policy_seed_sha256,
        baseline_policy_seed_sha256=design.baseline_policy_seed_sha256,
        policy_bundle_revision=design.policy_bundle_revision,
        baseline_policy_bundle_revision=design.baseline_policy_bundle_revision,
        selection_seed_sha256=design.selection_seed_sha256,
        permutation_seed=design.permutation_seed,
        hmac_key_id=_HMAC_KEY_ID,
        hmac_secret_sha256=_HMAC_SECRET_SHA256,
        index_config=_index_config(),
        index_replicate_count=INDEX_REPLICATE_COUNT,
        index_replicate_directories=INDEX_REPLICATE_DIRECTORIES,
        selected_family_count=5,
        selected_index_replicate=SELECTED_INDEX_REPLICATE,
        corpora=tuple(
            ProductionCorpusFactoryConfig(corpus_id=corpus_id, available_family_count=9)
            for corpus_id in FIXED_CORPORA
        ),
    )


def _payload(
    *, index: str = "index", row_map: str = "row-map"
) -> tuple[ReproducibleIndexPayload, ...]:
    return (
        ReproducibleIndexPayload(
            mask_id="mask-a",
            index_sha256=_digest(index),
            row_map_sha256=_digest(row_map),
            build_binding_sha256=_digest("build-binding"),
        ),
    )


def _replicates() -> tuple[IndexReplicateEvidence, ...]:
    return tuple(
        IndexReplicateEvidence(
            replicate=replicate,
            receipt_sha256=_digest("receipt"),
            tree_sha256=_digest("tree"),
            index_payloads=_payload(),
            elapsed_monotonic_ns=replicate,
            process_peak_rss_bytes=1024,
        )
        for replicate in range(1, INDEX_REPLICATE_COUNT + 1)
    )


def _stage(corpus_id: str, stage: str) -> IndexReproducibilityStageReceipt:
    return IndexReproducibilityStageReceipt(
        factory_config_sha256=_digest("factory"),
        runner_image=f"ghcr.io/mhdk1602/fractal-c0@sha256:{_digest('runner')}",
        runner_platform="linux/arm64",
        corpus_id=corpus_id,
        stage=stage,
        backend_id="hnswlib",
        backend_version="0.8.0",
        backend_build_sha256=_digest("hnsw-build"),
        replicates=_replicates(),
        selected_replicate=SELECTED_INDEX_REPLICATE,
        selected_final_receipt_sha256=_digest("receipt"),
    )


def _full_hnsw_replicates() -> tuple[FullHnswReplicateEvidence, ...]:
    return tuple(
        FullHnswReplicateEvidence(
            replicate=replicate,
            relative_path=(
                f"{INDEX_REPLICATE_DIRECTORIES[replicate - 1]}/"
                f"{factory.FULL_HNSW_REPLICATE_FILENAME}"
            ),
            byte_count=4096,
            sha256=_digest("full-hnsw"),
            elapsed_monotonic_ns=replicate,
            process_peak_rss_bytes=1024,
        )
        for replicate in range(1, INDEX_REPLICATE_COUNT + 1)
    )


def _full_hnsw(corpus_id: str) -> FullHnswReproducibilityReceipt:
    return FullHnswReproducibilityReceipt(
        factory_config_sha256=_digest("factory"),
        runner_image=f"ghcr.io/mhdk1602/fractal-c0@sha256:{_digest('runner')}",
        runner_platform="linux/arm64",
        corpus_id=corpus_id,
        backend_id="hnswlib-python-v1",
        backend_version="0.8.0",
        backend_build_sha256=_digest("hnsw-build"),
        source_vector_sha256=_digest(f"{corpus_id}-old-documents"),
        document_count=10,
        dimension=2,
        format_revision="hnswlib-full-v1",
        replicates=_full_hnsw_replicates(),
        selected_replicate=SELECTED_INDEX_REPLICATE,
        selected_final_sha256=_digest("full-hnsw"),
    )


def _suite(config: ProductionArtifactFactoryConfig) -> ProductionArtifactFactorySuiteReceipt:
    corpus_rows = tuple(
        FactorySuiteCorpus(
            corpus_id=corpus_id,
            evidence_sha256=_digest(f"{corpus_id}-evidence"),
            evidence_file_sha256=_digest(f"{corpus_id}-evidence"),
            policy_bundle_receipt_sha256=_digest(f"{corpus_id}-policy"),
            index_bundle_receipt_sha256=_digest(f"{corpus_id}-index"),
            query_receipt_sha256=_digest(f"{corpus_id}-query"),
            online_execution_plan_sha256=_digest(f"{corpus_id}-plan"),
            online_execution_tree_sha256=_digest(f"{corpus_id}-tree"),
            runtime_receipt_sha256=_digest(f"{corpus_id}-runtime"),
        )
        for corpus_id in FIXED_CORPORA
    )
    return ProductionArtifactFactorySuiteReceipt(
        factory_config_sha256=config.file_sha256,
        runner_image=config.runner_image,
        runner_platform=config.runner_platform,
        embedding_source_tree_sha256=config.embedding_source_tree_sha256,
        embedding_destination_tree_sha256=config.embedding_source_tree_sha256,
        embedding_suite_receipt_sha256=config.embedding_suite_receipt_sha256,
        hmac_key_id=config.hmac_key_id,
        hmac_secret_sha256=config.hmac_secret_sha256,
        online_inventory_sha256=_digest("online-inventory"),
        index_reproducibility_receipt_sha256=_digest("reproducibility"),
        artifact_pipeline_receipt_sha256=_digest("pipeline"),
        corpora=corpus_rows,
    )


def _materialize_shard_fixture(
    config: ProductionArtifactFactoryConfig,
    corpus_id: str,
) -> tuple[ProductionCorpusFactoryEvidence, ProductionArtifactFactoryShardReceipt]:
    request = next(
        row
        for row in factory.derive_production_artifact_factory_shard_requests(config)
        if row.corpus_id == corpus_id
    )
    config.artifact_root.mkdir(mode=0o700, exist_ok=True)
    for relative_path in request.owned_relative_paths[:-1]:
        root = config.artifact_root / relative_path
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        (root / "payload.bin").write_bytes(f"{corpus_id}:{relative_path}".encode())
    online_tree = digest_directory_tree(
        config.artifact_root / request.owned_relative_paths[3]
    ).sha256
    evidence = ProductionCorpusFactoryEvidence(
        corpus_id=corpus_id,
        factory_config_sha256=config.file_sha256,
        embedding_receipt_sha256=_digest(f"{corpus_id}-embedding"),
        policy_bundle_receipt_sha256=_digest(f"{corpus_id}-policy"),
        index_bundle_receipt_sha256=_digest(f"{corpus_id}-index"),
        query_receipt_sha256=_digest(f"{corpus_id}-query"),
        online_execution_plan_sha256=_digest(f"{corpus_id}-plan"),
        online_execution_tree_sha256=online_tree,
        runtime_receipt_sha256=_digest(f"{corpus_id}-runtime"),
        started_at_utc="2026-07-16T12:00:00Z",
        completed_at_utc="2026-07-16T12:00:01Z",
        elapsed_monotonic_ns=1,
        process_peak_rss_bytes=1024,
        status="built",
    )
    evidence_path = config.artifact_root / request.owned_relative_paths[-1]
    evidence_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    evidence_path.write_bytes(evidence.canonical_file_bytes())
    receipt = factory._derive_factory_shard_receipt(config, request, evidence)
    return evidence, receipt


def test_factory_config_is_closed_canonical_and_secret_free(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert ProductionArtifactFactoryConfig.from_dict(config.to_dict()) == config
    encoded = config.canonical_file_bytes()
    assert b"linux/arm64" in encoded
    assert _HMAC_SECRET not in encoded
    assert _HMAC_SECRET_SHA256.encode() in encoded
    assert _HMAC_KEY_ID.encode() in encoded
    assert b"plugin" not in encoded

    unknown = config.to_dict()
    unknown["plugin"] = "forbidden"
    with pytest.raises(ProductionArtifactFactoryError, match="unknown"):
        ProductionArtifactFactoryConfig.from_dict(unknown)


def test_factory_config_rejects_reordered_corpora_and_overlapping_source(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    reordered = config.to_dict()
    reordered["corpora"] = list(reversed(cast(list[object], reordered["corpora"])))
    with pytest.raises(ProductionArtifactFactoryError, match="FIXED_CORPORA"):
        ProductionArtifactFactoryConfig.from_dict(reordered)

    overlap = config.to_dict()
    overlap["embedding_source_root"] = str(config.artifact_root / "source")
    with pytest.raises(ProductionArtifactFactoryError, match="cannot overlap"):
        ProductionArtifactFactoryConfig.from_dict(overlap)


def test_factory_config_binds_stage_replicate_and_hmac_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)

    wrong_key = config.to_dict()
    wrong_key["hmac_key_id"] = "sealed-online-ephemeral-sha256-" + _digest("other")
    with pytest.raises(ProductionArtifactFactoryError, match="derived"):
        ProductionArtifactFactoryConfig.from_dict(wrong_key)

    wrong_stages = config.to_dict()
    wrong_stages["artifact_stage_order"] = ["sealed", "fit", "calibration"]
    with pytest.raises(ProductionArtifactFactoryError, match="ARTIFACT_STAGE_ORDER"):
        ProductionArtifactFactoryConfig.from_dict(wrong_stages)


def test_shard_requests_are_exact_closed_and_derived_from_full_config(tmp_path: Path) -> None:
    config = _config(tmp_path)

    requests = factory.derive_production_artifact_factory_shard_requests(config)

    assert tuple(request.corpus_id for request in requests) == FIXED_CORPORA
    assert len({request.request_sha256 for request in requests}) == len(FIXED_CORPORA)
    assert all(request.factory_config_sha256 == config.file_sha256 for request in requests)
    assert all(request.hmac_secret_sha256 == config.hmac_secret_sha256 for request in requests)
    assert all(request.hmac_key_id == config.hmac_key_id for request in requests)
    owned = [path for request in requests for path in request.owned_relative_paths]
    assert len(owned) == len(set(owned))
    assert all(
        ProductionArtifactFactoryShardRequest.from_dict(request.to_dict()) == request
        for request in requests
    )

    redirected = replace(config, artifact_root=(tmp_path / "different-output").resolve())
    with pytest.raises(ProductionArtifactFactoryError, match="full pinned factory config"):
        factory._verify_shard_request_binding(redirected, requests[0])

    missing_path = requests[0].to_dict()
    missing_path["owned_relative_paths"] = list(requests[0].owned_relative_paths[:-1])
    with pytest.raises(ProductionArtifactFactoryError, match="owned paths"):
        ProductionArtifactFactoryShardRequest.from_dict(missing_path)

    unknown = requests[0].to_dict()
    unknown["scientific_override"] = "forbidden"
    with pytest.raises(ProductionArtifactFactoryError, match="unknown"):
        ProductionArtifactFactoryShardRequest.from_dict(unknown)


def test_shard_hmac_bytes_commitment_and_key_id_fail_before_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    request = factory.derive_production_artifact_factory_shard_requests(config)[0]
    admission_called = False

    def admit(_config: ProductionArtifactFactoryConfig) -> Any:
        nonlocal admission_called
        admission_called = True
        raise AssertionError("admission must not run")

    monkeypatch.setattr(factory, "_admit_factory_inputs", admit)
    with pytest.raises(ProductionArtifactFactoryError, match="identify one secret"):
        factory.build_production_artifact_factory_shard(
            config,
            request,
            hmac_secret=b"wrong-secret-material" * 2,
            resume=False,
        )
    assert admission_called is False
    assert not config.artifact_root.exists()

    wrong_key_id = request.to_dict()
    wrong_key_id["hmac_key_id"] = "sealed-online-ephemeral-sha256-" + _digest("other")
    with pytest.raises(ProductionArtifactFactoryError, match="key ID"):
        ProductionArtifactFactoryShardRequest.from_dict(wrong_key_id)


def test_prepare_shards_publishes_only_the_closed_five_request_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.artifact_root.mkdir(mode=0o700)
    request_root = (tmp_path / "shard-control" / "requests").resolve()
    request_root.parent.mkdir(mode=0o700)

    admitted = SimpleNamespace(
        embedding_config=SimpleNamespace(
            online_staging_root=(tmp_path / "online-staging").resolve(),
            current_model_root=(tmp_path / "current-model").resolve(),
            stale_model_root=(tmp_path / "stale-model").resolve(),
        )
    )
    monkeypatch.setattr(factory, "_admit_factory_inputs", lambda _config: admitted)
    monkeypatch.setattr(factory, "_backend", lambda _config: cast(Any, object()))

    def copy_embeddings(
        observed_config: ProductionArtifactFactoryConfig,
        _inputs: Any,
    ) -> str:
        (observed_config.artifact_root / "embedding-stores").mkdir(mode=0o700)
        return observed_config.embedding_source_tree_sha256

    monkeypatch.setattr(factory, "_ensure_embedding_copy", copy_embeddings)

    with pytest.raises(ProductionArtifactFactoryError, match="cannot overlap"):
        factory.prepare_production_artifact_factory_shards(
            config,
            request_directory=admitted.embedding_config.online_staging_root / "requests",
        )
    assert tuple(config.artifact_root.iterdir()) == ()

    requests = factory.prepare_production_artifact_factory_shards(
        config,
        request_directory=request_root,
    )

    expected_names = tuple(
        f"{index:02d}-{corpus_id}.json" for index, corpus_id in enumerate(FIXED_CORPORA, start=1)
    )
    assert tuple(path.name for path in sorted(request_root.iterdir())) == expected_names
    for request, name in zip(requests, expected_names):
        assert (
            factory.load_production_artifact_factory_shard_request(
                request_root / name,
                expected_sha256=request.request_sha256,
            )
            == request
        )
    assert not any(path.name.startswith(".") for path in request_root.parent.iterdir())


def test_shard_receipt_covers_exact_owned_tree_and_rejects_partial_or_replay(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    evidence, receipt = _materialize_shard_fixture(config, "scifact")

    assert ProductionArtifactFactoryShardReceipt.from_dict(receipt.to_dict()) == receipt
    factory._verify_factory_shard_receipt_current(config, receipt, evidence)

    policy_payload = config.artifact_root / receipt.artifacts[0].relative_path / "payload.bin"
    policy_payload.write_bytes(b"partial replacement")
    with pytest.raises(ProductionArtifactFactoryError, match="current corpus tree"):
        factory._verify_factory_shard_receipt_current(config, receipt, evidence)

    missing = receipt.to_dict()
    missing["artifacts"] = list(receipt.to_dict()["artifacts"][:-1])
    with pytest.raises(ProductionArtifactFactoryError, match="exact owned set"):
        ProductionArtifactFactoryShardReceipt.from_dict(missing)

    extra = receipt.to_dict()
    extra["artifacts"] = [
        *receipt.to_dict()["artifacts"],
        ProductionArtifactFactoryShardArtifact(
            relative_path="forbidden/extra",
            artifact_kind="tree",
            sha256=_digest("extra"),
        ).to_dict(),
    ]
    with pytest.raises(ProductionArtifactFactoryError, match="exact owned set"):
        ProductionArtifactFactoryShardReceipt.from_dict(extra)

    rogue = receipt.to_dict()
    rogue["corpus_id"] = "undeclared-corpus"
    with pytest.raises(ProductionArtifactFactoryError, match="corpus differs"):
        ProductionArtifactFactoryShardReceipt.from_dict(rogue)

    replay_config = replace(config, artifact_root=(tmp_path / "replayed-output").resolve())
    with pytest.raises(ProductionArtifactFactoryError, match="pinned factory config"):
        factory._verify_shard_request_binding(
            replay_config,
            factory.derive_production_artifact_factory_shard_requests(config)[0],
        )


def test_shard_aggregate_rejects_missing_duplicate_wrong_config_and_wrong_secret(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    receipts = tuple(
        _materialize_shard_fixture(config, corpus_id)[1] for corpus_id in FIXED_CORPORA
    )

    ordered = factory._order_and_validate_factory_shard_receipts(config, tuple(reversed(receipts)))
    assert tuple(receipt.corpus_id for receipt in ordered) == FIXED_CORPORA

    with pytest.raises(ProductionArtifactFactoryError, match="missing="):
        factory._order_and_validate_factory_shard_receipts(config, receipts[:-1])
    with pytest.raises(ProductionArtifactFactoryError, match="duplicate="):
        factory._order_and_validate_factory_shard_receipts(config, (*receipts, receipts[0]))

    wrong_config_receipt = replace(
        receipts[0], factory_config_sha256=_digest("wrong-factory-config")
    )
    with pytest.raises(ProductionArtifactFactoryError, match="factory_config_sha256"):
        factory._order_and_validate_factory_shard_receipts(
            config, (wrong_config_receipt, *receipts[1:])
        )

    wrong_secret_receipt = replace(
        receipts[0],
        hmac_secret_sha256=_digest("wrong-secret"),
        hmac_key_id="sealed-online-ephemeral-sha256-" + _digest("wrong-secret"),
    )
    with pytest.raises(ProductionArtifactFactoryError, match="hmac_key_id"):
        factory._order_and_validate_factory_shard_receipts(
            config, (wrong_secret_receipt, *receipts[1:])
        )


def test_shard_aggregate_is_completion_order_independent_and_reference_byte_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    materialized = {
        corpus_id: _materialize_shard_fixture(config, corpus_id) for corpus_id in FIXED_CORPORA
    }
    evidence = {corpus_id: row[0] for corpus_id, row in materialized.items()}
    receipts = tuple(row[1] for row in materialized.values())
    terminal_inputs = SimpleNamespace(
        embedding_source_tree_sha256=config.embedding_source_tree_sha256,
        embedding_suite=SimpleNamespace(receipt_sha256=config.embedding_suite_receipt_sha256),
        embedding_config=SimpleNamespace(online_inventory_sha256=_digest("inventory")),
    )
    reproducibility = SimpleNamespace(receipt_sha256=_digest("reproducibility"))
    pipeline = SimpleNamespace(receipt_sha256=_digest("pipeline"))
    ordered_receipts = factory._order_and_validate_factory_shard_receipts(
        config, tuple(reversed(receipts))
    )
    sequential_terminal = factory._suite_from_evidence(
        config,
        terminal_inputs,
        tuple(evidence[corpus_id] for corpus_id in FIXED_CORPORA),
        reproducibility,
        pipeline,
        embedding_destination_tree_sha256=config.embedding_source_tree_sha256,
    )
    sharded_terminal = factory._suite_from_evidence(
        config,
        terminal_inputs,
        tuple(evidence[receipt.corpus_id] for receipt in ordered_receipts),
        reproducibility,
        pipeline,
        embedding_destination_tree_sha256=config.embedding_source_tree_sha256,
    )
    assert sequential_terminal.canonical_file_bytes() == sharded_terminal.canonical_file_bytes()

    inputs = SimpleNamespace()
    terminal = _suite(config)
    publication_orders: list[tuple[str, ...]] = []

    monkeypatch.setattr(factory, "_admit_factory_inputs", lambda _config: inputs)
    monkeypatch.setattr(factory, "_validate_factory_root_membership", lambda *_a, **_k: None)
    monkeypatch.setattr(factory, "_require_prepared_shard_root", lambda *_a, **_k: None)
    monkeypatch.setattr(factory, "_ensure_embedding_copy", lambda *_a, **_k: _digest("copy"))
    monkeypatch.setattr(factory, "_verify_embedding_copy", lambda *_a, **_k: _digest("copy"))
    monkeypatch.setattr(factory, "_prepare_factory_roots", lambda *_a, **_k: None)
    monkeypatch.setattr(factory, "_backend", lambda _config: cast(Any, object()))
    monkeypatch.setattr(
        factory,
        "_build_one_corpus",
        lambda _config, _inputs, *, corpus_id, **_kwargs: evidence[corpus_id],
    )
    monkeypatch.setattr(
        factory,
        "_verify_one_corpus",
        lambda _config, _inputs, *, corpus_id, **_kwargs: evidence[corpus_id],
    )
    monkeypatch.setattr(
        factory,
        "_factory_corpus_locks",
        lambda *_a, **_k: nullcontext(),
    )

    def publish(
        _config: ProductionArtifactFactoryConfig,
        _inputs: Any,
        rows: Any,
        **_kwargs: Any,
    ) -> ProductionArtifactFactorySuiteReceipt:
        publication_orders.append(tuple(row.corpus_id for row in rows))
        return terminal

    monkeypatch.setattr(factory, "_publish_production_artifact_factory_terminal", publish)

    sequential = factory.build_production_artifact_factory(
        config,
        hmac_secret=_HMAC_SECRET,
        resume=True,
    )
    sharded = factory.aggregate_production_artifact_factory_shards(
        config,
        tuple(reversed(receipts)),
    )

    assert sequential.canonical_file_bytes() == sharded.canonical_file_bytes()
    assert publication_orders == [FIXED_CORPORA, FIXED_CORPORA]


def test_corpus_lane_lock_rejects_duplicate_worker(tmp_path: Path) -> None:
    config = _config(tmp_path)
    lock_root = config.reproducibility_root / "scifact"
    lock_root.mkdir(parents=True, mode=0o700)

    with factory._factory_corpus_locks(config, ("scifact",)):
        with pytest.raises(ProductionArtifactFactoryError, match="already active"):
            with factory._factory_corpus_locks(config, ("scifact",)):
                raise AssertionError("duplicate worker acquired the same lane")

    wrong_roots = config.to_dict()
    wrong_roots["index_replicate_directories"] = [
        "replicate-01",
        "replicate-01",
        "replicate-03",
    ]
    with pytest.raises(ProductionArtifactFactoryError, match="distinct fixed roots"):
        ProductionArtifactFactoryConfig.from_dict(wrong_roots)

    wrong_count = config.to_dict()
    wrong_count["index_replicate_count"] = 2
    with pytest.raises(ProductionArtifactFactoryError, match="equal three"):
        ProductionArtifactFactoryConfig.from_dict(wrong_count)


def test_factory_suite_echoes_derived_hmac_identity(tmp_path: Path) -> None:
    receipt = _suite(_config(tmp_path))

    assert receipt.hmac_key_id == _HMAC_KEY_ID
    assert receipt.hmac_secret_sha256 == _HMAC_SECRET_SHA256
    assert ProductionArtifactFactorySuiteReceipt.from_dict(receipt.to_dict()) == receipt
    with pytest.raises(ProductionArtifactFactoryError, match="secret commitment"):
        replace(
            receipt,
            hmac_key_id="sealed-online-ephemeral-sha256-" + _digest("wrong"),
        )
    with pytest.raises(ProductionArtifactFactoryError, match="admitted source tree"):
        replace(receipt, embedding_destination_tree_sha256=_digest("wrong-copy"))


def test_public_suite_loader_enforces_canonical_bytes_and_optional_pin(tmp_path: Path) -> None:
    receipt = _suite(_config(tmp_path))
    path = tmp_path / factory.FACTORY_SUITE_FILENAME
    path.write_bytes(receipt.canonical_file_bytes())

    assert (
        factory.load_production_artifact_factory_suite(
            path,
            expected_sha256=receipt.receipt_sha256,
        )
        == receipt
    )
    with pytest.raises(ProductionArtifactFactoryError, match="caller pin"):
        factory.load_production_artifact_factory_suite(
            path,
            expected_sha256=_digest("wrong-suite"),
        )


def test_policy_seed_derivation_is_fixed_and_domain_separated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = config.policy_config("scifact", "fit")

    assert first == factory.derive_production_policy_config(
        config.design_seed_sha256,
        "scifact",
        "fit",
    )
    assert first == config.policy_config("scifact", "fit")
    assert first.seed_sha256 != config.policy_config("scifact", "sealed").seed_sha256
    assert first.seed_sha256 != config.policy_config("bright", "fit").seed_sha256
    assert first.seed_sha256 != first.baseline_seed_sha256

    with pytest.raises(ProductionArtifactFactoryError, match="FIXED_CORPORA"):
        factory.derive_production_policy_config(config.design_seed_sha256, "other", "fit")
    with pytest.raises(ProductionArtifactFactoryError, match="ARTIFACT_STAGE_ORDER"):
        factory.derive_production_policy_config(config.design_seed_sha256, "scifact", "other")


def test_config_rejects_mutated_design_derivation_and_index_constants(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(ProductionArtifactFactoryError, match="design-seed derivation"):
        replace(config, policy_seed_sha256=_digest("operator-policy-seed"))
    with pytest.raises(ProductionArtifactFactoryError, match="C0 production constants"):
        replace(config, index_config=replace(config.index_config, m=17))


def test_available_family_counts_reproduce_audited_assignment_bytes(tmp_path: Path) -> None:
    staging = (tmp_path / "staging").resolve()
    staging.mkdir()
    rows: list[dict[str, object]] = []
    query_counts: list[SimpleNamespace] = []
    for corpus_id in FIXED_CORPORA:
        for stage in factory.ARTIFACT_STAGE_ORDER:
            query_counts.append(SimpleNamespace(dataset=corpus_id, stage=stage, query_count=2))
            for family in range(2):
                query_id = f"{corpus_id}-{stage}-{family}"
                rows.append(
                    {
                        "assignment_key_sha256": _digest(f"assignment-{query_id}"),
                        "dataset": corpus_id,
                        "domain": None,
                        "partition_component_sha256": _digest(
                            f"component-{corpus_id}-{stage}-{family}"
                        ),
                        "query_id": query_id,
                        "query_text_sha256": _digest(f"text-{query_id}"),
                        "schema_version": factory.ASSIGNMENT_SCHEMA,
                        "source_split": "test",
                        "stage": stage,
                    }
                )
    encoded = b"".join(factory._canonical_bytes(row) + b"\n" for row in rows)
    assignment_path = staging / "assignments.jsonl"
    assignment_path.write_bytes(encoded)
    source = SimpleNamespace(
        role="assignments",
        path="assignments.jsonl",
        visibility="online",
        dataset=None,
        stage=None,
        sha256=hashlib.sha256(encoded).hexdigest(),
        byte_count=len(encoded),
        record_count=len(rows),
    )
    audit = SimpleNamespace(
        source_artifacts=(source,),
        assignment_artifact_sha256=source.sha256,
        query_counts=tuple(query_counts),
        assignment_count=len(rows),
    )

    observed = factory._derive_available_family_counts(staging, cast(Any, audit))

    assert [row.available_family_count for row in observed] == [2] * len(FIXED_CORPORA)

    missing_coverage = SimpleNamespace(
        source_artifacts=(source,),
        assignment_artifact_sha256=source.sha256,
        query_counts=tuple(query_counts[:-1]),
        assignment_count=len(rows),
    )
    with pytest.raises(ProductionArtifactFactoryError, match="do not cover"):
        factory._derive_available_family_counts(staging, cast(Any, missing_coverage))


def test_full_hnsw_releases_builder_before_readback(tmp_path: Path) -> None:
    class BuildIndex:
        def init_index(self, **_kwargs: Any) -> None:
            pass

        def set_num_threads(self, _threads: int) -> None:
            pass

        def add_items(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def save_index(self, path: str) -> None:
            Path(path).write_bytes(b"deterministic-hnsw")

    class ReadIndex:
        def load_index(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def set_num_threads(self, _threads: int) -> None:
            pass

        def set_ef(self, _ef: int) -> None:
            pass

        def knn_query(self, *_args: Any, **_kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
            return np.asarray([[0]], dtype=np.int64), np.asarray([[0.0]], dtype=np.float32)

    class Backend:
        def __init__(self) -> None:
            self.calls = 0
            self.builder: weakref.ReferenceType[BuildIndex] | None = None

        def create_index(self, **_kwargs: Any) -> BuildIndex | ReadIndex:
            self.calls += 1
            if self.calls == 1:
                index = BuildIndex()
                self.builder = weakref.ref(index)
                return index
            assert self.builder is not None
            assert self.builder() is None
            return ReadIndex()

    backend = Backend()
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    pin = factory._build_full_hnsw(
        cast(Any, matrix),
        tmp_path / "full-active.hnsw",
        corpus_id="scifact",
        config=_index_config(),
        backend=cast(Any, backend),
    )

    assert backend.calls == 2
    assert pin.byte_count == len(b"deterministic-hnsw")


def test_full_hnsw_three_build_receipt_retains_and_reverifies_exact_replicas(
    tmp_path: Path,
) -> None:
    class Index:
        def __init__(self, backend: Backend) -> None:
            self.backend = backend

        def init_index(self, **_kwargs: Any) -> None:
            pass

        def set_num_threads(self, _threads: int) -> None:
            pass

        def add_items(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def save_index(self, path: str) -> None:
            self.backend.save_calls += 1
            Path(path).write_bytes(b"deterministic-full-hnsw")

        def load_index(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def set_ef(self, _ef: int) -> None:
            pass

        def knn_query(self, *_args: Any, **_kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
            return np.asarray([[0]], dtype=np.int64), np.asarray([[0.0]], dtype=np.float32)

    class Backend:
        def __init__(self) -> None:
            self.save_calls = 0

        def create_index(self, **_kwargs: Any) -> Index:
            return Index(self)

    config = _config(tmp_path)
    config.artifact_root.mkdir(mode=0o700)
    corpus_root = config.reproducibility_root / "scifact"
    corpus_root.mkdir(parents=True, mode=0o700)
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    embedding = SimpleNamespace(
        vectors={
            "old_documents": SimpleNamespace(
                shape=(2, 2),
                file_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
            )
        }
    )
    backend = Backend()

    receipt = factory._admit_full_hnsw_reproducibility(
        config,
        corpus_id="scifact",
        embedding=cast(Any, embedding),
        matrix=cast(Any, matrix),
        source_vector_sha256=embedding.vectors["old_documents"].file_sha256,
        backend=cast(Any, backend),
        build_missing=True,
    )

    assert backend.save_calls == 3
    assert len(receipt.replicates) == 3
    assert len({row.sha256 for row in receipt.replicates}) == 1
    assert receipt.selected_final_sha256 == receipt.replicates[0].sha256
    assert (
        factory._admit_full_hnsw_reproducibility(
            config,
            corpus_id="scifact",
            embedding=cast(Any, embedding),
            matrix=cast(Any, matrix),
            source_vector_sha256=embedding.vectors["old_documents"].file_sha256,
            backend=cast(Any, backend),
            build_missing=False,
        )
        == receipt
    )
    assert backend.save_calls == 3

    selected = (
        factory._full_hnsw_reproducibility_root(config, "scifact")
        / receipt.replicates[0].relative_path
    )
    selected.write_bytes(b"drift")
    with pytest.raises(ProductionArtifactFactoryError, match="byte pin"):
        factory._admit_full_hnsw_reproducibility(
            config,
            corpus_id="scifact",
            embedding=cast(Any, embedding),
            matrix=cast(Any, matrix),
            source_vector_sha256=embedding.vectors["old_documents"].file_sha256,
            backend=cast(Any, backend),
            build_missing=False,
        )


def test_full_hnsw_receipt_rejects_replica_drift_or_another_selected_copy() -> None:
    accepted = _full_hnsw("scifact")
    assert FullHnswReproducibilityReceipt.from_dict(accepted.to_dict()) == accepted
    unknown = accepted.to_dict()
    unknown["unregistered_field"] = True
    with pytest.raises(ProductionArtifactFactoryError, match="unknown"):
        FullHnswReproducibilityReceipt.from_dict(unknown)
    changed = list(accepted.replicates)
    changed[2] = replace(changed[2], sha256=_digest("different-full-hnsw"))
    with pytest.raises(ProductionArtifactFactoryError, match="bytes differ"):
        replace(accepted, replicates=tuple(changed))
    with pytest.raises(ProductionArtifactFactoryError, match="selected"):
        replace(accepted, selected_replicate=2)


def test_three_build_receipt_accepts_exact_stores_and_rejects_byte_drift() -> None:
    accepted = _stage("scifact", "fit")
    assert accepted.selected_replicate == 1

    changed_payload = list(_replicates())
    changed_payload[2] = replace(changed_payload[2], index_payloads=_payload(index="changed"))
    with pytest.raises(ProductionArtifactFactoryError, match="HNSW or row-map"):
        replace(accepted, replicates=tuple(changed_payload))

    changed_tree = list(_replicates())
    changed_tree[1] = replace(changed_tree[1], tree_sha256=_digest("changed-tree"))
    with pytest.raises(ProductionArtifactFactoryError, match="store bytes"):
        replace(accepted, replicates=tuple(changed_tree))


def test_reproducibility_suite_requires_fixed_corpus_stage_order() -> None:
    stages = tuple(
        _stage(corpus_id, stage)
        for corpus_id in FIXED_CORPORA
        for stage in factory.ARTIFACT_STAGE_ORDER
    )
    receipt = IndexReproducibilitySuiteReceipt(
        factory_config_sha256=_digest("factory"),
        runner_image=f"ghcr.io/mhdk1602/fractal-c0@sha256:{_digest('runner')}",
        runner_platform="linux/arm64",
        replicate_count=3,
        selected_replicate=1,
        stages=stages,
        full_hnsw_indexes=tuple(_full_hnsw(corpus_id) for corpus_id in FIXED_CORPORA),
    )
    assert len(receipt.stages) == len(FIXED_CORPORA) * 3
    assert tuple(row.corpus_id for row in receipt.full_hnsw_indexes) == FIXED_CORPORA
    assert receipt.schema_version == "fractal-index-reproducibility-suite-v2"
    assert IndexReproducibilitySuiteReceipt.from_dict(receipt.to_dict()) == receipt

    old_schema = receipt.to_dict()
    old_schema["schema_version"] = "fractal-authorized-index-reproducibility-suite-v1"
    with pytest.raises(ProductionArtifactFactoryError, match="schema differs"):
        IndexReproducibilitySuiteReceipt.from_dict(old_schema)

    missing_full_hnsw = receipt.to_dict()
    del missing_full_hnsw["full_hnsw_indexes"]
    with pytest.raises(ProductionArtifactFactoryError, match="missing"):
        IndexReproducibilitySuiteReceipt.from_dict(missing_full_hnsw)

    with pytest.raises(ProductionArtifactFactoryError, match="fixed corpus/stage order"):
        replace(receipt, stages=tuple(reversed(stages)))
    with pytest.raises(ProductionArtifactFactoryError, match="fixed corpus order"):
        replace(receipt, full_hnsw_indexes=tuple(reversed(receipt.full_hnsw_indexes)))
    rebound = list(receipt.full_hnsw_indexes)
    rebound[0] = replace(
        rebound[0],
        runner_image=f"ghcr.io/mhdk1602/other@sha256:{_digest('other-runner')}",
    )
    with pytest.raises(ProductionArtifactFactoryError, match="binding differs"):
        replace(receipt, full_hnsw_indexes=tuple(rebound))


def test_online_plan_binds_selected_full_hnsw_and_raw_active_vector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    raw_vector_sha256 = _digest("raw-active-vector")
    full_hnsw = FullHnswReproducibilityReceipt(
        factory_config_sha256=config.file_sha256,
        runner_image=config.runner_image,
        runner_platform=config.runner_platform,
        corpus_id="scifact",
        backend_id=config.index_config.backend_id,
        backend_version=config.index_config.backend_version,
        backend_build_sha256=config.index_config.backend_build_sha256,
        source_vector_sha256=raw_vector_sha256,
        document_count=10,
        dimension=2,
        format_revision=factory._full_hnsw_format_revision(config.index_config),
        replicates=_full_hnsw_replicates(),
        selected_replicate=1,
        selected_final_sha256=_digest("full-hnsw"),
    )
    query = SimpleNamespace(
        hmac_key_id=config.hmac_key_id,
        query_partition_audit_sha256=_digest("partition"),
        opaque_trials=(),
    )
    universe = _digest("document-universe")
    plan = SimpleNamespace(
        artifact_sha256=_digest("plan"),
        corpus="scifact",
        stage="sealed",
        key_id=config.hmac_key_id,
        permutation_seed=config.permutation_seed,
        query_partition_audit_sha256=query.query_partition_audit_sha256,
        trials=(),
        document_count=10,
        ordered_document_universe_sha256=universe,
        active_vector_store=SimpleNamespace(
            shape=(10, 2),
            artifact=SimpleNamespace(sha256=raw_vector_sha256),
        ),
        current_truth_vector_store=SimpleNamespace(shape=(10, 2)),
        hnsw_index=SimpleNamespace(
            artifact=SimpleNamespace(
                byte_count=full_hnsw.replicates[0].byte_count,
                sha256=full_hnsw.replicates[0].sha256,
            ),
            source_vector_sha256=raw_vector_sha256,
            format_revision=full_hnsw.format_revision,
        ),
    )
    embedding = SimpleNamespace(
        document_count=10,
        vectors={
            "old_documents": SimpleNamespace(
                shape=(10, 2),
                file_sha256=_digest("npy-container-is-not-the-raw-vector"),
            ),
            "current_documents": SimpleNamespace(shape=(10, 2)),
        },
        row_orders={"documents": SimpleNamespace(row_order_sha256=universe)},
    )
    monkeypatch.setattr(
        factory,
        "digest_directory_tree",
        lambda _root: SimpleNamespace(sha256=_digest("online-tree")),
    )
    monkeypatch.setattr(factory, "load_sharded_online_execution_plan", lambda _path: plan)
    monkeypatch.setattr(
        factory,
        "verify_online_execution_package",
        lambda *_args, **_kwargs: SimpleNamespace(plan=plan),
    )

    observed, tree_sha256 = factory._verify_online_source_binding(
        tmp_path,
        corpus_id="scifact",
        embedding=cast(Any, embedding),
        query_receipt=query,
        config=config,
        full_hnsw=full_hnsw,
    )

    assert observed is plan
    assert tree_sha256 == _digest("online-tree")
    plan.hnsw_index.artifact.sha256 = _digest("another-hnsw")
    with pytest.raises(ProductionArtifactFactoryError, match="frozen sources"):
        factory._verify_online_source_binding(
            tmp_path,
            corpus_id="scifact",
            embedding=cast(Any, embedding),
            query_receipt=query,
            config=config,
            full_hnsw=full_hnsw,
        )


def test_online_install_copies_only_the_selected_full_hnsw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"selected-full-active-hnsw"
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    replicates = tuple(
        replace(row, byte_count=len(payload), sha256=payload_sha256)
        for row in _full_hnsw_replicates()
    )
    receipt = replace(
        _full_hnsw("scifact"),
        replicates=replicates,
        selected_final_sha256=payload_sha256,
    )
    reproducibility_root = tmp_path / "reproducibility"
    selected_source = reproducibility_root / replicates[0].relative_path
    selected_source.parent.mkdir(parents=True, mode=0o700)
    selected_source.write_bytes(payload)
    selected_source.chmod(0o600)
    work = tmp_path / "online"
    (work / "indexes").mkdir(parents=True, mode=0o700)
    verified: list[Path] = []

    def verify(_matrix: Any, path: Path, **kwargs: Any) -> None:
        verified.append(path)
        assert kwargs["expected_byte_count"] == len(payload)
        assert kwargs["expected_sha256"] == payload_sha256

    monkeypatch.setattr(factory, "_verify_full_hnsw_file", verify)

    pin = factory._install_selected_full_hnsw(
        cast(Any, np.asarray([[1.0]], dtype=np.float32)),
        work,
        reproducibility_root,
        receipt,
        config=_index_config(),
        backend=cast(Any, object()),
    )

    target = work / factory.ONLINE_HNSW_PATH
    assert target.read_bytes() == payload
    assert target.stat().st_ino != selected_source.stat().st_ino
    assert verified == [target]
    assert pin.byte_count == len(payload)
    assert pin.sha256 == payload_sha256
    assert tuple(path.relative_to(work).as_posix() for path in work.rglob("*.hnsw")) == (
        factory.ONLINE_HNSW_PATH,
    )

    poisoned_work = tmp_path / "poisoned-online"
    (poisoned_work / "indexes").mkdir(parents=True, mode=0o700)
    (poisoned_work / "vectors").mkdir(mode=0o700)
    decoy = poisoned_work / "vectors" / "replicate-02.hnsw"
    decoy.write_bytes(payload)
    decoy.chmod(0o600)
    with pytest.raises(ProductionArtifactFactoryError, match="another HNSW copy"):
        factory._install_selected_full_hnsw(
            cast(Any, np.asarray([[1.0]], dtype=np.float32)),
            poisoned_work,
            reproducibility_root,
            receipt,
            config=_index_config(),
            backend=cast(Any, object()),
        )
    assert decoy.read_bytes() == payload


def test_resume_removes_only_private_unpublished_authorized_index_residue(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "reproducibility" / "scifact" / "fit"
    parent.mkdir(parents=True, mode=0o700)
    output = parent / "replicate-01"
    output.mkdir(mode=0o700)
    sentinel = output / "published.bin"
    sentinel.write_bytes(b"published")
    lock = parent / ".replicate-01.authorized-index.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    staging = parent / ".replicate-01.staging-0123456789abcdef01234567"
    staging.mkdir(mode=0o700)
    partial = staging / "partial.hnsw"
    partial.write_bytes(b"partial")
    partial.chmod(0o600)

    factory._recover_authorized_index_builder_residue(output)

    assert sentinel.read_bytes() == b"published"
    assert not lock.exists()
    assert not staging.exists()

    lock.write_bytes(b"")
    lock.chmod(0o600)
    quarantined = parent / (
        ".replicate-01.staging-0123456789abcdef01234567.recovery-fedcba9876543210fedcba98"
    )
    quarantined.mkdir(mode=0o700)
    quarantined_payload = quarantined / "partial.hnsw"
    quarantined_payload.write_bytes(b"partial")
    quarantined_payload.chmod(0o600)

    factory._recover_authorized_index_builder_residue(output)

    assert sentinel.read_bytes() == b"published"
    assert not lock.exists()
    assert not quarantined.exists()


def test_exclusive_publish_pins_source_and_target_inode_and_fsyncs_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "publish"
    parent.mkdir(mode=0o700)
    work = parent / ".artifact.staging-0123456789abcdef01234567"
    work.mkdir(mode=0o700)
    payload = work / "payload.bin"
    payload.write_bytes(b"immutable")
    payload.chmod(0o600)
    source = work.stat()
    observed_fsync_modes: list[int] = []
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        observed_fsync_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(factory.os, "fsync", fsync)
    target = parent / "artifact"

    factory._exclusive_publish_directory(work, target, label="test artifact")

    published = target.stat()
    assert (published.st_dev, published.st_ino) == (source.st_dev, source.st_ino)
    assert not os.path.lexists(work)
    assert (target / "payload.bin").read_bytes() == b"immutable"
    assert len(observed_fsync_modes) == 2
    assert all(stat_mode & 0o170000 == 0o040000 for stat_mode in observed_fsync_modes)


def test_factory_staging_cleanup_never_removes_published_or_malformed_names(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "online"
    parent.mkdir(mode=0o700)
    published = parent / "scifact"
    published.mkdir(mode=0o700)
    sentinel = published / "receipt.json"
    sentinel.write_bytes(b"published")
    sentinel.chmod(0o600)
    published_inode = published.stat().st_ino
    partial = parent / ".scifact.staging-0123456789abcdef01234567"
    partial.mkdir(mode=0o700)
    (partial / "partial.bin").write_bytes(b"partial")

    factory._recover_factory_staging_directories(
        parent,
        prefix=".scifact.staging-",
        label="scifact online staging",
    )

    assert not partial.exists()
    assert published.stat().st_ino == published_inode
    assert sentinel.read_bytes() == b"published"

    quarantined = parent / (
        ".scifact.staging-0123456789abcdef01234567.recovery-fedcba9876543210fedcba98"
    )
    quarantined.mkdir(mode=0o700)
    quarantined_payload = quarantined / "partial.bin"
    quarantined_payload.write_bytes(b"partial")
    quarantined_payload.chmod(0o600)
    factory._recover_factory_staging_directories(
        parent,
        prefix=".scifact.staging-",
        label="scifact online staging",
    )
    assert not quarantined.exists()
    assert sentinel.read_bytes() == b"published"

    malformed = parent / ".scifact.staging-not-a-token"
    malformed.mkdir(mode=0o700)
    with pytest.raises(ProductionArtifactFactoryError, match="undeclared temporary name"):
        factory._recover_factory_staging_directories(
            parent,
            prefix=".scifact.staging-",
            label="scifact online staging",
        )
    assert malformed.exists()
    assert published.stat().st_ino == published_inode


def test_temporary_cleanup_does_not_delete_a_between_check_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "online"
    parent.mkdir(mode=0o700)
    partial = parent / ".scifact.staging-0123456789abcdef01234567"
    partial.mkdir(mode=0o700)
    original_payload = partial / "original.bin"
    original_payload.write_bytes(b"original")
    original_payload.chmod(0o600)
    original_inode = partial.stat().st_ino
    hidden_original = parent / ".pinned-original"
    real_rename = factory._rename_directory_noreplace_at

    def substitute_then_rename(
        parent_descriptor: int,
        source_name: str,
        destination_name: str,
        *,
        label: str,
    ) -> None:
        os.rename(
            source_name,
            hidden_original.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        replacement = parent / source_name
        replacement.mkdir(mode=0o700)
        live_payload = replacement / "live-worker.bin"
        live_payload.write_bytes(b"live")
        live_payload.chmod(0o600)
        real_rename(
            parent_descriptor,
            source_name,
            destination_name,
            label=label,
        )

    monkeypatch.setattr(factory, "_rename_directory_noreplace_at", substitute_then_rename)

    with pytest.raises(ProductionArtifactFactoryError, match="not the pinned temporary inode"):
        factory._validate_and_remove_temporary_tree(partial, label="scifact online staging")

    assert hidden_original.stat().st_ino == original_inode
    assert (hidden_original / "original.bin").read_bytes() == b"original"
    quarantines = tuple(parent.glob(f"{partial.name}.recovery-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "live-worker.bin").read_bytes() == b"live"


def test_resume_rejects_live_or_malformed_authorized_index_residue(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "reproducibility" / "scifact" / "fit"
    parent.mkdir(parents=True, mode=0o700)
    output = parent / "replicate-01"
    lock = parent / ".replicate-01.authorized-index.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    staging = parent / ".replicate-01.staging-0123456789abcdef01234567"
    staging.mkdir(mode=0o700)

    live_descriptor = os.open(lock, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    fcntl.flock(live_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(ProductionArtifactFactoryError, match="still active"):
            factory._recover_authorized_index_builder_residue(output)
    finally:
        fcntl.flock(live_descriptor, fcntl.LOCK_UN)
        os.close(live_descriptor)
    assert lock.exists()
    assert staging.exists()

    lock.unlink()
    with pytest.raises(ProductionArtifactFactoryError, match="without its builder lock"):
        factory._recover_authorized_index_builder_residue(output)

    lock.write_bytes(b"")
    lock.chmod(0o600)
    (staging / "outside").symlink_to(tmp_path / "unrelated")
    with pytest.raises(ProductionArtifactFactoryError, match="unsafe temporary member"):
        factory._recover_authorized_index_builder_residue(output)
    assert lock.exists()
    assert staging.exists()


def test_hmac_secret_is_read_only_from_bounded_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = iter((_HMAC_SECRET, b""))
    observed_fds: list[int] = []

    def read(fd: int, _size: int) -> bytes:
        observed_fds.append(fd)
        return next(chunks)

    monkeypatch.setattr(factory.os, "read", read)
    assert factory._read_hmac_secret(0) == _HMAC_SECRET
    assert observed_fds == [0, 0]

    with pytest.raises(ProductionArtifactFactoryError, match="descriptor 0"):
        factory._read_hmac_secret(7)

    short = iter((b"short", b""))
    monkeypatch.setattr(factory.os, "read", lambda _fd, _size: next(short))
    with pytest.raises(ProductionArtifactFactoryError, match="fewer than 32"):
        factory._read_hmac_secret(0)

    oversized = iter((b"x" * 4096, b"x"))
    monkeypatch.setattr(factory.os, "read", lambda _fd, _size: next(oversized))
    with pytest.raises(ProductionArtifactFactoryError, match="exceeds 4096"):
        factory._read_hmac_secret(0)


def test_cli_requires_stdin_for_build_and_has_no_secret_or_label_path() -> None:
    parser = factory._parser()
    common = ["--config", "/input/factory.json", "--config-sha256", _digest("config")]

    build = parser.parse_args(["build", *common, "--hmac-secret-fd", "0"])
    assert build.hmac_secret_fd == 0
    resume = parser.parse_args(["resume", *common])
    assert resume.hmac_secret_fd is None

    with pytest.raises(SystemExit):
        parser.parse_args(["build", *common])
    with pytest.raises(SystemExit):
        parser.parse_args(["build", *common, "--hmac-secret-fd", "1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["build", *common, "--hmac-secret-fd", "0", "--label-path", "x"])

    shard = [
        "build-shard",
        *common,
        "--request",
        "/control/01-scifact.json",
        "--request-sha256",
        _digest("request"),
        "--hmac-secret-fd",
        "0",
        "--receipt-output",
        "/control/receipts/01-scifact.json",
    ]
    assert parser.parse_args(shard).hmac_secret_fd == 0
    with pytest.raises(SystemExit):
        parser.parse_args([*shard, "--corpus-id", "scifact"])
    with pytest.raises(SystemExit):
        parser.parse_args([*shard, "--index-m", "32"])
    wrong_shard_fd = list(shard)
    wrong_shard_fd[wrong_shard_fd.index("--hmac-secret-fd") + 1] = "9"
    with pytest.raises(SystemExit):
        parser.parse_args(wrong_shard_fd)

    aggregate = [
        "aggregate-shards",
        *common,
        "--shard-receipt",
        "/control/receipts/01-scifact.json",
    ]
    assert parser.parse_args(aggregate).shard_receipt == [Path("/control/receipts/01-scifact.json")]
    with pytest.raises(SystemExit):
        parser.parse_args([*aggregate, "--selected-replicate", "2"])

    write_config = [
        "write-config",
        "--artifact-root",
        "/output/artifacts",
        "--embedding-config",
        "/input/embedding.json",
        "--embedding-config-sha256",
        _digest("embedding"),
        "--development-operator-root",
        "/input/development-operator",
        "--development-operator-receipt-sha256",
        _digest("development-operator"),
        "--partition-audit",
        "/input/partition.json",
        "--partition-audit-sha256",
        _digest("partition"),
        "--runner-image",
        f"ghcr.io/mhdk1602/fractal-c0@sha256:{_digest('runner')}",
        "--hmac-secret-fd",
        "0",
        "--output",
        "/output/factory.json",
    ]
    assert parser.parse_args(write_config).hmac_secret_fd == 0
    wrong_write_fd = list(write_config)
    wrong_write_fd[wrong_write_fd.index("--hmac-secret-fd") + 1] = "1"
    with pytest.raises(SystemExit):
        parser.parse_args(wrong_write_fd)
    with pytest.raises(SystemExit):
        parser.parse_args([*write_config, "--label-path", "/labels"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [*write_config, "--design-seed-sha256", _digest("second-source-of-truth")]
        )


def test_write_config_derives_closed_fields_and_writes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_tree_sha256 = _digest("embedding-source-tree")
    online_inventory_sha256 = _digest("online-inventory")
    projected_sha256 = _digest("projected-set")
    embedding_suite_sha256 = _digest("embedding-suite")
    power_bytes = b'{"joint":"power"}\n'
    embedding_config_path = (tmp_path / "embedding-config.json").resolve()
    audit_path = (tmp_path / "partition-audit.json").resolve()
    embedding_root = (tmp_path / "embedding-source").resolve()
    staging_root = (tmp_path / "staging").resolve()
    current_root = (tmp_path / "current-model").resolve()
    stale_root = (tmp_path / "stale-model").resolve()
    development_operator_root = (tmp_path / "development-operator").resolve()
    power_path = development_operator_root / "analysis" / "joint-power-design" / "report.json"
    power_path.parent.mkdir(parents=True)
    power_path.write_bytes(power_bytes)
    artifact_root = (tmp_path / "artifacts").resolve()
    output = (tmp_path / "factory-config.json").resolve()
    embedding_config = SimpleNamespace(
        output_root=embedding_root,
        online_staging_root=staging_root,
        online_inventory_sha256=online_inventory_sha256,
        projected_artifact_set_sha256=projected_sha256,
        current_model_root=current_root,
        stale_model_root=stale_root,
    )
    embedding_suite = SimpleNamespace(
        receipt_sha256=embedding_suite_sha256,
        online_inventory_sha256=online_inventory_sha256,
        projected_artifact_set_sha256=projected_sha256,
    )
    projection = SimpleNamespace(
        inventory_sha256=online_inventory_sha256,
        projected_artifact_set_sha256=projected_sha256,
    )
    audit = SimpleNamespace(staged_inventory_sha256=online_inventory_sha256)
    corpora = tuple(
        ProductionCorpusFactoryConfig(corpus_id=corpus_id, available_family_count=8)
        for corpus_id in FIXED_CORPORA
    )
    observed_secret_fds: list[int] = []

    def read_secret(fd: int | None) -> bytes:
        assert fd is not None
        observed_secret_fds.append(fd)
        return _HMAC_SECRET

    monkeypatch.setattr(factory, "_read_hmac_secret", read_secret)
    monkeypatch.setattr(
        factory, "load_production_embedding_config", lambda *_a, **_k: embedding_config
    )
    monkeypatch.setattr(factory, "_require_real_directory", lambda *_a, **_k: None)
    monkeypatch.setattr(factory, "_require_read_only_filesystem", lambda *_a, **_k: None)
    monkeypatch.setattr(
        factory,
        "digest_directory_tree",
        lambda *_a, **_k: SimpleNamespace(sha256=source_tree_sha256),
    )
    monkeypatch.setattr(
        factory,
        "admit_frozen_production_embedding_suite",
        lambda *_a, **_k: embedding_suite,
    )
    monkeypatch.setattr(factory, "verify_online_staging_projection", lambda *_a, **_k: projection)
    monkeypatch.setattr(factory, "load_scalable_partition_audit", lambda *_a, **_k: audit)
    monkeypatch.setattr(
        factory,
        "load_joint_power_report",
        lambda *_a, **_k: SimpleNamespace(
            freeze_ready=True,
            selected_families_per_corpus=6,
        ),
    )
    monkeypatch.setattr(factory, "_derive_available_family_counts", lambda *_a, **_k: corpora)
    monkeypatch.setattr(
        factory,
        "production_authorized_index_components",
        lambda: (_index_config(), cast(Any, object())),
    )
    monkeypatch.setattr(
        factory,
        "_verify_development_operator_binding",
        lambda **_kwargs: SimpleNamespace(
            development_materialization_receipt_sha256=_digest("development-receipt"),
            design_seed_sha256=_digest("operator-design-seed"),
            joint_power_report_sha256=hashlib.sha256(power_bytes).hexdigest(),
            joint_power_report_tree_sha256=_digest("joint-power-tree"),
            selected_families_per_corpus=6,
        ),
    )
    monkeypatch.setattr(factory, "_admit_factory_inputs", lambda *_a, **_k: None)
    monkeypatch.setattr(factory, "_validate_factory_root_membership", lambda *_a, **_k: None)

    design_seed_sha256 = _digest("operator-design-seed")
    config = factory.write_production_artifact_factory_config(
        artifact_root=artifact_root,
        embedding_build_config_path=embedding_config_path,
        embedding_build_config_sha256=_digest("embedding-config"),
        development_operator_root=development_operator_root,
        development_operator_receipt_sha256=_digest("development-operator-receipt"),
        partition_audit_path=audit_path,
        partition_audit_sha256=_digest("partition-audit"),
        runner_image=f"ghcr.io/mhdk1602/fractal-c0@sha256:{_digest('runner')}",
        hmac_secret_fd=0,
        output=output,
    )

    assert observed_secret_fds == [0]
    assert config.design_seed_sha256 == design_seed_sha256
    assert config.development_materialization_root == (
        development_operator_root / "materialized-development"
    )
    assert config.joint_power_report_path == power_path
    assert config.selected_family_count == 6
    assert config.index_config == _index_config()
    assert config.hmac_secret_sha256 == _HMAC_SECRET_SHA256
    assert config.policy_config("scifact", "fit") == factory.derive_production_policy_config(
        design_seed_sha256,
        "scifact",
        "fit",
    )
    assert _HMAC_SECRET not in output.read_bytes()
    assert (
        factory.load_production_artifact_factory_config(
            output,
            expected_sha256=config.file_sha256,
        )
        == config
    )

    rejected_output = (tmp_path / "rejected-config.json").resolve()
    monkeypatch.setattr(
        factory,
        "_admit_factory_inputs",
        lambda *_a, **_k: (_ for _ in ()).throw(
            ProductionArtifactFactoryError("hostile upstream admission")
        ),
    )
    with pytest.raises(ProductionArtifactFactoryError, match="hostile upstream"):
        factory.write_production_artifact_factory_config(
            artifact_root=artifact_root,
            embedding_build_config_path=embedding_config_path,
            embedding_build_config_sha256=_digest("embedding-config"),
            development_operator_root=development_operator_root,
            development_operator_receipt_sha256=_digest("development-operator-receipt"),
            partition_audit_path=audit_path,
            partition_audit_sha256=_digest("partition-audit"),
            runner_image=f"ghcr.io/mhdk1602/fractal-c0@sha256:{_digest('runner')}",
            hmac_secret_fd=0,
            output=rejected_output,
        )
    assert not rejected_output.exists()


def test_development_operator_binding_checks_every_factory_join(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fractal_ann_diagnostics.post_embedding_development as development

    values: dict[str, object] = {
        "artifact_sha256": _digest("operator-receipt"),
        "development_materialization_receipt_sha256": _digest("materialization"),
        "design_seed_sha256": _digest("design"),
        "embedding_suite_receipt_sha256": _digest("embedding-suite"),
        "index_config_sha256": _digest("index-config"),
        "joint_power_report_sha256": _digest("power-report"),
        "joint_power_report_tree_sha256": _digest("power-tree"),
        "partition_audit_file_sha256": _digest("partition-audit"),
        "partition_audit_sha256": _digest("partition-audit"),
        "selected_families_per_corpus": 7,
    }
    observed: list[tuple[Path, str, object, object, Path, object]] = []
    embedding_path = (tmp_path / "embedding-config.json").resolve()
    partition_audit_path = (tmp_path / "partition-audit.json").resolve()
    embedding_config = object()
    embedding_suite = object()
    partition_audit = object()

    def verify(
        root: Path,
        *,
        expected_receipt_sha256: str,
        production_embedding_config_path: Path,
        embedding_config: object,
        embedding_suite: object,
        partition_audit_path: Path,
        partition_audit: object,
    ) -> SimpleNamespace:
        observed.append(
            (
                root,
                expected_receipt_sha256,
                embedding_config,
                embedding_suite,
                partition_audit_path,
                partition_audit,
            )
        )
        assert production_embedding_config_path == embedding_path
        return SimpleNamespace(**values)

    monkeypatch.setattr(development, "admit_frozen_post_embedding_development", verify)
    operator_root = (tmp_path / "development-operator").resolve()
    common = {
        "root": operator_root,
        "receipt_sha256": values["artifact_sha256"],
        "embedding_config_path": embedding_path,
        "embedding_config": embedding_config,
        "embedding_suite": embedding_suite,
        "partition_audit_path": partition_audit_path,
        "partition_audit": partition_audit,
        "embedding_suite_receipt_sha256": values["embedding_suite_receipt_sha256"],
        "development_materialization_receipt_sha256": values[
            "development_materialization_receipt_sha256"
        ],
        "design_seed_sha256": values["design_seed_sha256"],
        "partition_audit_sha256": values["partition_audit_sha256"],
        "joint_power_report_sha256": values["joint_power_report_sha256"],
        "index_config_sha256": values["index_config_sha256"],
        "selected_family_count": values["selected_families_per_corpus"],
        "joint_power_report_tree_sha256": values["joint_power_report_tree_sha256"],
    }

    assert (
        factory._verify_development_operator_binding(**cast(Any, common)).artifact_sha256
        == values["artifact_sha256"]
    )
    assert observed == [
        (
            operator_root,
            values["artifact_sha256"],
            embedding_config,
            embedding_suite,
            partition_audit_path,
            partition_audit,
        )
    ]

    values["design_seed_sha256"] = _digest("wrong-design")
    with pytest.raises(ProductionArtifactFactoryError, match="design_seed_sha256"):
        factory._verify_development_operator_binding(**cast(Any, common))


@pytest.mark.parametrize(
    "field,redirected",
    (
        ("development_materialization_root", "redirected-development"),
        ("joint_power_report_path", "redirected-power.json"),
    ),
)
def test_factory_admission_rejects_redirected_development_operator_artifacts(
    tmp_path: Path,
    field: str,
    redirected: str,
) -> None:
    base = _config(tmp_path)
    operator_root = base.development_operator_root
    closed = replace(
        base,
        development_materialization_root=operator_root / "materialized-development",
        joint_power_report_path=(operator_root / "analysis" / "joint-power-design" / "report.json"),
    )
    with pytest.raises(ProductionArtifactFactoryError, match=field):
        replace(closed, **{field: (tmp_path / redirected).resolve()})


def test_factory_development_stage_must_match_the_artifacts_used_for_power(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    embedding_sha = _digest("scifact-embedding")
    policy_sha = _digest("scifact-fit-policy")
    index_sha = _digest("scifact-fit-index")
    stratum = SimpleNamespace(
        corpus="scifact",
        source_stage="fit",
        embedding_receipt_sha256=embedding_sha,
        policy_config_sha256=config.policy_config("scifact", "fit").config_sha256,
        policy_intervention_receipt_sha256=policy_sha,
        authorized_index_config_sha256=config.index_config.config_sha256,
        authorized_index_receipt_sha256=index_sha,
    )
    inputs = SimpleNamespace(
        development_operator=SimpleNamespace(strata=(stratum,)),
        embeddings={"scifact": SimpleNamespace(receipt_sha256=embedding_sha)},
    )
    observed = {
        "policy": SimpleNamespace(artifact_sha256=policy_sha),
        "index": SimpleNamespace(artifact_sha256=index_sha),
    }
    monkeypatch.setattr(
        factory,
        "load_policy_intervention_receipt",
        lambda *_args, **_kwargs: observed["policy"],
    )
    monkeypatch.setattr(
        factory,
        "load_authorized_index_store_receipt",
        lambda *_args, **_kwargs: observed["index"],
    )

    factory._verify_development_stage_parity(
        config,
        cast(Any, inputs),
        corpus_id="scifact",
        stage="fit",
    )

    observed["policy"] = SimpleNamespace(artifact_sha256=_digest("different-factory-policy"))
    with pytest.raises(
        ProductionArtifactFactoryError,
        match="policy_intervention_receipt_sha256",
    ):
        factory._verify_development_stage_parity(
            config,
            cast(Any, inputs),
            corpus_id="scifact",
            stage="fit",
        )

    observed["policy"] = SimpleNamespace(artifact_sha256=policy_sha)
    observed["index"] = SimpleNamespace(artifact_sha256=_digest("different-factory-index"))
    with pytest.raises(
        ProductionArtifactFactoryError,
        match="authorized_index_receipt_sha256",
    ):
        factory._verify_development_stage_parity(
            config,
            cast(Any, inputs),
            corpus_id="scifact",
            stage="fit",
        )


def test_embedding_copy_uses_new_tree_and_preserves_exact_digest(tmp_path: Path) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    (source / "nested").mkdir()
    (source / "root.json").write_bytes(b'{"fixed":true}\n')
    (source / "nested" / "vectors.bin").write_bytes(bytes(range(64)))
    target = (tmp_path / "target").resolve()

    factory._copy_regular_tree_exclusive(source, target)

    assert digest_directory_tree(target).sha256 == digest_directory_tree(source).sha256
    with pytest.raises(ProductionArtifactFactoryError, match="copy boundary"):
        factory._copy_regular_tree_exclusive(source, target)


def test_embedding_copy_rejects_multiply_linked_source_file(tmp_path: Path) -> None:
    source = (tmp_path / "source").resolve()
    source.mkdir()
    original = source / "vectors.bin"
    original.write_bytes(b"fixed vectors")
    (source / "second-name.bin").hardlink_to(original)

    with pytest.raises(ProductionArtifactFactoryError, match="single-link"):
        factory._copy_regular_tree_exclusive(source, (tmp_path / "target").resolve())


def test_embedding_source_requires_read_only_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory.os, "statvfs", lambda _path: SimpleNamespace(f_flag=0))
    with pytest.raises(ProductionArtifactFactoryError, match="mounted read-only"):
        factory._require_read_only_filesystem(tmp_path, label="embedding_source_root")

    read_only = getattr(factory.os, "ST_RDONLY", 1)
    monkeypatch.setattr(
        factory.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_flag=read_only),
    )
    factory._require_read_only_filesystem(tmp_path, label="embedding_source_root")


def test_interrupted_embedding_copy_is_terminal(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.artifact_root.mkdir(mode=0o700)
    (config.artifact_root / ".embedding-stores.partial").mkdir(mode=0o700)

    with pytest.raises(ProductionArtifactFactoryError, match="terminal"):
        factory._ensure_embedding_copy(config, cast(Any, None))


def test_index_receipt_never_repairs_a_missing_registered_replicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    stage_root = config.reproducibility_root / "scifact" / "fit"
    stage_root.mkdir(parents=True, mode=0o700)
    receipt_path = stage_root / "reproducibility-receipt.json"
    receipt_path.write_bytes(_stage("scifact", "fit").canonical_file_bytes())
    monkeypatch.setattr(
        factory,
        "load_policy_intervention_receipt",
        lambda _path: SimpleNamespace(artifact_sha256=_digest("policy")),
    )
    build_called = False

    def build(*_args: Any, **_kwargs: Any) -> None:
        nonlocal build_called
        build_called = True

    monkeypatch.setattr(factory, "build_authorized_index_store", build)

    with pytest.raises(ProductionArtifactFactoryError, match="replicate 1 is missing"):
        factory._ensure_reproducible_index_stage(
            config,
            corpus_id="scifact",
            stage="fit",
            embedding=cast(Any, SimpleNamespace(receipt_sha256=_digest("embedding"))),
            backend=cast(Any, object()),
        )

    assert build_called is False


def test_status_preserves_fixed_order_and_does_not_write(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.artifact_root.mkdir(mode=0o700)
    before = tuple(config.artifact_root.iterdir())

    status = factory.production_artifact_factory_status(config)

    assert [row["corpus_id"] for row in status["corpora"]] == list(FIXED_CORPORA)
    assert status["embedding_copy"] is False
    assert tuple(config.artifact_root.iterdir()) == before


def test_query_receipt_must_echo_committed_key_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.artifact_root.mkdir(mode=0o700)
    (config.artifact_root / "trial-runtime").mkdir(mode=0o700)
    query_root = (
        config.artifact_root / "trial-runtime" / "scifact" / factory.RUNTIME_QUERY_DIRECTORY
    )
    query_root.mkdir(parents=True, mode=0o700)
    inputs = SimpleNamespace(
        embedding_config=SimpleNamespace(online_staging_root=tmp_path / "staging"),
        selected_family_count=5,
    )
    monkeypatch.setattr(
        factory,
        "verify_query_trial_store",
        lambda *_args, **_kwargs: SimpleNamespace(hmac_key_id="wrong-key-id"),
    )

    with pytest.raises(ProductionArtifactFactoryError, match="frozen commitment"):
        factory._ensure_query_package(
            config,
            cast(Any, inputs),
            corpus_id="scifact",
            hmac_secret=None,
        )


def test_hmac_commitment_fails_before_admission_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    admission_called = False

    def admit(_config: ProductionArtifactFactoryConfig) -> Any:
        nonlocal admission_called
        admission_called = True
        raise AssertionError("admission must not run")

    monkeypatch.setattr(factory, "_admit_factory_inputs", admit)
    with pytest.raises(ProductionArtifactFactoryError, match="frozen commitment"):
        factory.build_production_artifact_factory(
            config,
            hmac_secret=b"x" * 32,
            resume=False,
        )

    assert admission_called is False
    assert not config.artifact_root.exists()


def test_runtime_features_are_derived_without_labels() -> None:
    schedule = SimpleNamespace(
        rows=(
            SimpleNamespace(
                group_order=0,
                subject="confirmatory-reader",
                repetition=0,
                policy_state="baseline",
                realized_allow_rate=0.25,
            ),
            SimpleNamespace(
                group_order=1,
                subject="confirmatory-reader",
                repetition=0,
                policy_state="current",
                realized_allow_rate=0.75,
            ),
        )
    )

    rows = factory._runtime_feature_bindings(schedule)

    assert [row.policy_complexity for row in rows] == [0.25, 0.75]
    assert {row.backend for row in rows} == {"hnsw"}
    assert {row.drift_family for row in rows} == {"qwen-revision-lag"}


def test_backend_fails_closed_outside_c0_linux_arm64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory.platform, "machine", lambda: "arm64")

    with pytest.raises(ProductionArtifactFactoryError, match="Linux arm64"):
        factory._backend(_config(tmp_path))


def test_factory_manual_uses_the_c0_python_module_contract() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "research" / "production-artifact-factory.md"
    ).read_text(encoding="utf-8")

    assert "--entrypoint /opt/venv/bin/python" in text
    assert '"$IMAGE" \\\n  -m fractal_ann_diagnostics.production_artifact_factory' in text
    assert "write-config" in text
    assert "fractal-production-artifacts build" not in text
    assert "fractal-production-artifacts resume" not in text
    assert "fractal-production-artifacts verify" not in text
    assert "fractal-production-artifacts status" not in text
    assert "--development-materialization-root" not in text
    assert "--joint-power-report" not in text
    assert "--design-seed-sha256" not in text
