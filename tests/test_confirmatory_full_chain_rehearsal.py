from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_production_artifact_factory import (
    _HMAC_SECRET,
)
from test_production_artifact_factory import (
    _config as base_factory_config,
)
from test_production_embedding_build import (
    _install_fake_store_builder,
)
from test_production_embedding_build import (
    _write_config as write_embedding_config,
)
from test_suite_attempt import _opened_transfer_fixture, _run_output_aggregate

import fractal_ann_diagnostics.production_artifact_factory as factory
import fractal_ann_diagnostics.production_controls as controls
import fractal_ann_diagnostics.production_embedding_build as embedding_build
import fractal_ann_diagnostics.suite_attempt as suite_attempt
from fractal_ann_diagnostics.artifact_integrity import (
    digest_directory_tree,
    digest_regular_file,
)
from fractal_ann_diagnostics.artifact_stage_bundles import STAGE_BUNDLE_FILENAME
from fractal_ann_diagnostics.authorized_index_store import (
    RECEIPT_FILENAME as AUTHORIZED_INDEX_RECEIPT_FILENAME,
)
from fractal_ann_diagnostics.policy_intervention import (
    RECEIPT_FILENAME as POLICY_RECEIPT_FILENAME,
)
from fractal_ann_diagnostics.production_artifact_factory import (
    FULL_HNSW_REPLICATE_FILENAME,
    FULL_HNSW_REPLICATE_RECEIPT_FILENAME,
    FULL_HNSW_REPRODUCIBILITY_RECEIPT_FILENAME,
    INDEX_REPLICATE_DIRECTORIES,
    SELECTED_INDEX_REPLICATE,
    FullHnswReplicateEvidence,
    FullHnswReproducibilityReceipt,
    ProductionArtifactFactoryConfig,
    ProductionArtifactFactoryError,
    ProductionCorpusFactoryEvidence,
)
from fractal_ann_diagnostics.production_controls import (
    PREFLIGHT_CONTRACT_FILENAME,
    ProductionControlError,
    ProductionControlMaterializationConfig,
    _AdmittedFactory,
    materialize_production_control_blueprint,
)
from fractal_ann_diagnostics.production_corpus_run import (
    PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME,
)
from fractal_ann_diagnostics.qwen_revision_encoder import (
    QwenPairedRevisionEmbeddingAdapter,
)
from fractal_ann_diagnostics.sealed_container_launcher import (
    load_preflight_launch_contract,
)
from fractal_ann_diagnostics.study import FIXED_CORPORA
from fractal_ann_diagnostics.suite_attempt import (
    OnlineSuiteClosure,
    RunClaimBindings,
    SuiteOpenBindings,
)


def _digest(value: str | bytes) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, encoded: bytes) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(encoded)
    path.chmod(0o600)
    return _digest(encoded)


def _install_three_equal_hnsw_replicas(
    config: ProductionArtifactFactoryConfig,
    corpus_id: str,
) -> tuple[FullHnswReproducibilityReceipt, Path]:
    root = factory._full_hnsw_reproducibility_root(config, corpus_id)
    root.mkdir(mode=0o700, parents=True)
    payload = f"synthetic-full-hnsw:{corpus_id}\n".encode()
    payload_sha256 = _digest(payload)
    replicas: list[FullHnswReplicateEvidence] = []
    second_payload: Path | None = None
    for replicate, directory in enumerate(INDEX_REPLICATE_DIRECTORIES, start=1):
        replicate_root = root / directory
        replicate_root.mkdir(mode=0o700)
        payload_path = replicate_root / FULL_HNSW_REPLICATE_FILENAME
        _write(payload_path, payload)
        if replicate == 2:
            second_payload = payload_path
        evidence = FullHnswReplicateEvidence(
            replicate=replicate,
            relative_path=f"{directory}/{FULL_HNSW_REPLICATE_FILENAME}",
            byte_count=len(payload),
            sha256=payload_sha256,
            elapsed_monotonic_ns=replicate,
            process_peak_rss_bytes=4096,
        )
        _write(
            replicate_root / FULL_HNSW_REPLICATE_RECEIPT_FILENAME,
            evidence.canonical_file_bytes(),
        )
        replicas.append(evidence)
    receipt = FullHnswReproducibilityReceipt(
        factory_config_sha256=config.file_sha256,
        runner_image=config.runner_image,
        runner_platform=config.runner_platform,
        corpus_id=corpus_id,
        backend_id="hnswlib-python-v1",
        backend_version=config.index_config.backend_version,
        backend_build_sha256=config.index_config.backend_build_sha256,
        source_vector_sha256=_digest(f"{corpus_id}:old-document-vectors"),
        document_count=1,
        dimension=256,
        format_revision="hnswlib-full-v1",
        replicates=tuple(replicas),
        selected_replicate=SELECTED_INDEX_REPLICATE,
        selected_final_sha256=payload_sha256,
    )
    _write(root / FULL_HNSW_REPRODUCIBILITY_RECEIPT_FILENAME, receipt.canonical_file_bytes())
    assert second_payload is not None
    return receipt, second_payload


def test_label_free_five_corpus_chain_reaches_atomic_online_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rehearse the closed chain without labels, networks, or production compute."""

    embedding, embedding_config_path = write_embedding_config(tmp_path, monkeypatch)
    embedding_receipts = _install_fake_store_builder(monkeypatch)
    adapter = QwenPairedRevisionEmbeddingAdapter(
        embedding.current_encoder_config,
        embedding.stale_encoder_config,
    )
    shard_evidence = tuple(
        embedding_build.build_production_embedding_shard(
            embedding,
            corpus_id,
            paired_encoder=adapter,
        )
        for corpus_id in reversed(FIXED_CORPORA)
    )
    assert tuple(row.corpus_id for row in shard_evidence) == tuple(reversed(FIXED_CORPORA))
    embedding_suite = embedding_build.aggregate_production_embedding_shards(embedding)
    assert embedding_build.verify_production_embedding_suite(embedding) == embedding_suite
    assert all(receipt.old_model is not None for receipt in embedding_receipts.values())
    assert all(
        receipt.current_model["tree_sha256"] != receipt.old_model["tree_sha256"]
        for receipt in embedding_receipts.values()
        if receipt.old_model is not None
    )

    embedding_tree_sha256 = digest_directory_tree(embedding.output_root).sha256
    base = base_factory_config(tmp_path)
    config = replace(
        base,
        embedding_build_config_path=embedding_config_path.resolve(),
        embedding_build_config_sha256=embedding.file_sha256,
        embedding_source_root=embedding.output_root,
        embedding_source_tree_sha256=embedding_tree_sha256,
        embedding_suite_receipt_sha256=embedding_suite.receipt_sha256,
    )
    _write(config.partition_audit_path, b"synthetic-partition-audit\n")
    config.artifact_root.mkdir(mode=0o700)
    (config.artifact_root / "embedding-stores").mkdir(mode=0o700)

    execution_by_corpus: dict[str, Any] = {}
    runtime_by_corpus: dict[str, Any] = {}
    index_by_corpus: dict[str, Any] = {}
    policy_by_corpus: dict[str, Any] = {}
    evidence_by_corpus: dict[str, ProductionCorpusFactoryEvidence] = {}
    hnsw_second_payload: dict[str, Path] = {}
    shard_receipts = []
    for position, corpus_id in enumerate(FIXED_CORPORA):
        request = factory.derive_production_artifact_factory_shard_requests(config)[position]
        embedding_receipt = embedding_receipts[corpus_id]
        embedding_root = config.artifact_root / "embedding-stores" / corpus_id
        embedding_root.mkdir(mode=0o700)
        _write(embedding_root / "receipt.json", embedding_receipt.canonical_bytes() + b"\n")
        _write(embedding_root / "paired-vectors.synthetic", f"paired:{corpus_id}\n".encode())

        online_root = config.artifact_root / "custody" / "online" / corpus_id
        index_root = config.artifact_root / "authorized-index-stores" / corpus_id
        sealed_index_root = index_root / "sealed"
        policy_root = config.artifact_root / "policy-workloads" / corpus_id
        sealed_policy_root = policy_root / "sealed"
        runtime_root = config.artifact_root / "trial-runtime" / corpus_id
        query_root = runtime_root / factory.RUNTIME_QUERY_DIRECTORY
        for root in (online_root, sealed_index_root, sealed_policy_root, query_root):
            root.mkdir(mode=0o700, parents=True)

        execution_digest = _digest(f"execution:{corpus_id}")
        execution_path = online_root / factory.ONLINE_EXECUTION_PLAN_FILENAME
        _write(execution_path, f"execution:{corpus_id}\n".encode())
        query_digest = _write(
            query_root / factory.QUERY_TRIAL_RECEIPT_FILENAME,
            f"query-receipt:{corpus_id}\n".encode(),
        )
        runtime_digest = _write(
            runtime_root / factory.RUNTIME_RECEIPT_FILENAME,
            f"runtime-receipt:{corpus_id}\n".encode(),
        )
        policy_bundle_digest = _write(
            policy_root / STAGE_BUNDLE_FILENAME,
            f"policy-bundle:{corpus_id}\n".encode(),
        )
        index_bundle_digest = _write(
            index_root / STAGE_BUNDLE_FILENAME,
            f"index-bundle:{corpus_id}\n".encode(),
        )
        _write(sealed_policy_root / POLICY_RECEIPT_FILENAME, b"{}\n")
        _write(sealed_index_root / AUTHORIZED_INDEX_RECEIPT_FILENAME, b"{}\n")
        _write(query_root / "query-trials.jsonl", f"query:{corpus_id}\n".encode())
        policy_semantic = _digest(f"policy:{corpus_id}")
        index_semantic = _digest(f"index:{corpus_id}")
        execution_by_corpus[corpus_id] = SimpleNamespace(artifact_sha256=execution_digest)
        policy_by_corpus[corpus_id] = SimpleNamespace(artifact_sha256=policy_semantic)
        index_by_corpus[corpus_id] = SimpleNamespace(
            artifact_sha256=index_semantic,
            embedding_receipt_sha256=embedding_receipt.receipt_sha256,
            policy_receipt_sha256=policy_semantic,
        )
        runtime_by_corpus[corpus_id] = SimpleNamespace(
            execution_artifact_sha256=execution_digest,
            embedding_store_receipt_sha256=embedding_receipt.receipt_sha256,
            groups=tuple(
                SimpleNamespace(
                    group_order=order,
                    subject=f"synthetic-reader-{order}",
                    repetition=0,
                    policy_state=("baseline", "current", "baseline")[order],
                    realized_allow_rate=(0.2, 0.8, 0.5)[order],
                )
                for order in range(3)
            ),
        )
        _, second = _install_three_equal_hnsw_replicas(config, corpus_id)
        hnsw_second_payload[corpus_id] = second
        online_tree_sha256 = digest_directory_tree(online_root).sha256
        evidence = ProductionCorpusFactoryEvidence(
            corpus_id=corpus_id,
            factory_config_sha256=config.file_sha256,
            embedding_receipt_sha256=embedding_receipt.receipt_sha256,
            policy_bundle_receipt_sha256=policy_bundle_digest,
            index_bundle_receipt_sha256=index_bundle_digest,
            query_receipt_sha256=query_digest,
            online_execution_plan_sha256=execution_digest,
            online_execution_tree_sha256=online_tree_sha256,
            runtime_receipt_sha256=runtime_digest,
            started_at_utc="2026-07-16T12:00:00Z",
            completed_at_utc="2026-07-16T12:00:01Z",
            elapsed_monotonic_ns=1,
            process_peak_rss_bytes=4096,
            status="built",
        )
        evidence_path = config.evidence_root / f"{corpus_id}.json"
        _write(evidence_path, evidence.canonical_file_bytes())
        evidence_by_corpus[corpus_id] = evidence
        shard_receipts.append(factory._derive_factory_shard_receipt(config, request, evidence))

    inputs = SimpleNamespace(
        embedding_source_tree_sha256=embedding_tree_sha256,
        embedding_suite=embedding_suite,
        embedding_config=embedding,
    )
    monkeypatch.setattr(factory, "_admit_factory_inputs", lambda _config: inputs)
    monkeypatch.setattr(factory, "_require_prepared_shard_root", lambda *_a, **_k: None)
    monkeypatch.setattr(
        factory,
        "_verify_embedding_copy",
        lambda *_a, **_k: embedding_tree_sha256,
    )
    monkeypatch.setattr(factory, "_backend", lambda _config: cast(Any, object()))
    monkeypatch.setattr(factory, "_factory_corpus_locks", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(
        factory,
        "_verify_one_corpus",
        lambda _config, _inputs, *, corpus_id, **_kwargs: evidence_by_corpus[corpus_id],
    )
    monkeypatch.setattr(
        factory,
        "_assemble_reproducibility_suite",
        lambda *_a, **_k: SimpleNamespace(receipt_sha256=_digest("reproducibility-suite")),
    )
    monkeypatch.setattr(
        factory,
        "build_artifact_pipeline",
        lambda *_a, **_k: SimpleNamespace(receipt_sha256=_digest("artifact-pipeline")),
    )
    monkeypatch.setattr(factory, "_validate_factory_root_membership", lambda *_a, **_k: None)
    factory_suite = factory.aggregate_production_artifact_factory_shards(
        config,
        tuple(reversed(shard_receipts)),
    )
    assert tuple(row.corpus_id for row in factory_suite.corpora) == FIXED_CORPORA
    assert factory_suite.embedding_suite_receipt_sha256 == embedding_suite.receipt_sha256

    hostile_hnsw = hnsw_second_payload[FIXED_CORPORA[0]]
    original_hnsw = hostile_hnsw.read_bytes()
    hostile_hnsw.write_bytes(b"substituted-replica\n")
    with pytest.raises(ProductionArtifactFactoryError, match="current corpus tree"):
        factory.aggregate_production_artifact_factory_shards(config, shard_receipts)
    hostile_hnsw.write_bytes(original_hnsw)

    factory_config_path = tmp_path / "factory-config.json"
    _write(factory_config_path, config.canonical_file_bytes())
    pseudonym_key = tmp_path / "pseudonym.key"
    opa = tmp_path / "opa"
    uv_lock = tmp_path / "uv.lock"
    _write(pseudonym_key, _HMAC_SECRET)
    _write(opa, b"synthetic-opa\n")
    _write(uv_lock, b"version = 1\n")
    extraction = SimpleNamespace(
        c0_sha="c" * 40,
        opa_image_path="/usr/local/bin/opa",
        opa_sha256=digest_regular_file(opa),
        python_binary_image_path="/opt/venv/bin/python",
        python_binary_sha256=_digest("python-binary"),
        uv_lock_image_path="/opt/app/uv.lock",
        uv_lock_sha256=digest_regular_file(uv_lock),
    )
    admitted = _AdmittedFactory(
        config=config,
        suite=factory_suite,
        extraction=cast(Any, extraction),
        staged_root=embedding.online_staging_root,
    )
    monkeypatch.setattr(controls, "_admit_factory", lambda _config: admitted)
    monkeypatch.setattr(
        controls,
        "load_sharded_online_execution_plan",
        lambda path: execution_by_corpus[path.parents[0].name],
    )
    monkeypatch.setattr(
        controls,
        "load_trial_runtime_receipt",
        lambda path: runtime_by_corpus[path.parents[0].name],
    )
    monkeypatch.setattr(
        controls,
        "load_authorized_index_store_receipt",
        lambda path: index_by_corpus[path.parents[0].name],
    )
    monkeypatch.setattr(
        controls,
        "load_policy_intervention_receipt",
        lambda path: policy_by_corpus[path.parents[1].name],
    )
    scientific_index_digest = config.runner_image.rsplit("@", 1)[1]
    scientific_production_reference = (
        "ghcr.io/mhdk1602/fractal-ann-diagnostics-confirmatory-production"
        f"@{scientific_index_digest}"
    )

    first_query_receipt = (
        config.artifact_root
        / "trial-runtime"
        / FIXED_CORPORA[0]
        / factory.RUNTIME_QUERY_DIRECTORY
        / factory.QUERY_TRIAL_RECEIPT_FILENAME
    )
    original_query_receipt = first_query_receipt.read_bytes()
    first_query_receipt.write_bytes(b"substituted-query-receipt\n")
    with pytest.raises(ProductionControlError, match="subordinate factory pins differ"):
        controls._derive_workload_spec(
            ProductionControlMaterializationConfig(
                factory_config_path=factory_config_path,
                factory_config_sha256=config.file_sha256,
                factory_suite_receipt_path=config.suite_receipt_path,
                factory_suite_receipt_sha256=factory_suite.receipt_sha256,
                factory_artifact_tree_sha256=digest_directory_tree(config.artifact_root).sha256,
                c0_runtime_extraction_receipt_path=tmp_path / "c0-extraction.json",
                c0_runtime_extraction_receipt_sha256=_digest("c0-extraction"),
                opa_binary_path=opa,
                opa_binary_sha256=digest_regular_file(opa),
                uv_lock_path=uv_lock,
                uv_lock_sha256=digest_regular_file(uv_lock),
                pseudonym_key_path=pseudonym_key,
                pseudonym_key_sha256=digest_regular_file(pseudonym_key),
                scientific_candidate_reference=config.runner_image,
                scientific_production_reference=scientific_production_reference,
                scientific_index_digest=scientific_index_digest,
                candidate_image_source_commit="2" * 40,
                oci_promotion_required=True,
                approval_environment="confirmatory",
                runner_platform=config.runner_platform,
                runner_identity="github-actions:environment:confirmatory",
                hostname="sealed-runner",
                hardware_provider="synthetic",
                hardware_instance_type="arm64-rehearsal",
                hardware_cpu_model="synthetic-arm64",
                hardware_accelerator="none",
                hardware_region="offline",
                hardware_operating_system="ubuntu-24.04",
                memory_limit_bytes=8 * 1024**3,
                cpuset_cpus=(0, 1),
                tmpfs_size_bytes=1024**2,
                blueprint_root=tmp_path / "unused-blueprint",
                finalized_controls_root=tmp_path / "unused-closure",
                suite_base_root=tmp_path / "unused-suite",
            ),
            admitted,
            FIXED_CORPORA[0],
        )
    first_query_receipt.write_bytes(original_query_receipt)

    materialization = ProductionControlMaterializationConfig(
        factory_config_path=factory_config_path,
        factory_config_sha256=config.file_sha256,
        factory_suite_receipt_path=config.suite_receipt_path,
        factory_suite_receipt_sha256=factory_suite.receipt_sha256,
        factory_artifact_tree_sha256=digest_directory_tree(config.artifact_root).sha256,
        c0_runtime_extraction_receipt_path=tmp_path / "c0-extraction.json",
        c0_runtime_extraction_receipt_sha256=_digest("c0-extraction"),
        opa_binary_path=opa,
        opa_binary_sha256=digest_regular_file(opa),
        uv_lock_path=uv_lock,
        uv_lock_sha256=digest_regular_file(uv_lock),
        pseudonym_key_path=pseudonym_key,
        pseudonym_key_sha256=digest_regular_file(pseudonym_key),
        scientific_candidate_reference=config.runner_image,
        scientific_production_reference=scientific_production_reference,
        scientific_index_digest=scientific_index_digest,
        candidate_image_source_commit="2" * 40,
        oci_promotion_required=True,
        approval_environment="confirmatory",
        runner_platform=config.runner_platform,
        runner_identity="github-actions:environment:confirmatory",
        hostname="sealed-runner",
        hardware_provider="synthetic",
        hardware_instance_type="arm64-rehearsal",
        hardware_cpu_model="synthetic-arm64",
        hardware_accelerator="none",
        hardware_region="offline",
        hardware_operating_system="ubuntu-24.04",
        memory_limit_bytes=8 * 1024**3,
        cpuset_cpus=(0, 1),
        tmpfs_size_bytes=1024**2,
        blueprint_root=tmp_path / "blueprint",
        finalized_controls_root=tmp_path / "production-run-closure",
        suite_base_root=tmp_path / "suite-base",
    )
    materialization_path = tmp_path / "production-control-materialization.json"
    _write(materialization_path, materialization.canonical_file_bytes())
    blueprint = materialize_production_control_blueprint(
        materialization_path,
        expected_config_sha256=materialization.file_sha256,
    )
    assert tuple(row.corpus_id for row in blueprint.workloads) == FIXED_CORPORA
    for row in blueprint.workloads:
        contract = load_preflight_launch_contract(
            materialization.blueprint_root / row.corpus_id / PREFLIGHT_CONTRACT_FILENAME
        )
        assert contract.geometry.corpus_id == row.corpus_id
        assert (
            materialization.blueprint_root
            / row.corpus_id
            / PRODUCTION_CORPUS_WORKLOAD_SPEC_FILENAME
        ).is_file()

    suite_root = tmp_path / "synthetic-suite-state"
    suite_root.mkdir(mode=0o700)
    namespace, opened, closures, _files = _opened_transfer_fixture(suite_root)
    open_payload = opened.records[0].payload
    assert isinstance(open_payload, SuiteOpenBindings)
    assert {row.corpus_id: row.sha256 for row in open_payload.execution_artifacts} == {
        row.corpus_id: row.online_execution_plan_sha256 for row in factory_suite.corpora
    }
    transfer = suite_attempt._transfer_staged_online_outputs(opened, closures)
    claim_record = opened.state
    claim_payload = claim_record.payload
    assert isinstance(claim_payload, RunClaimBindings)
    online_complete = suite_attempt._write_transition(
        opened,
        state="ONLINE_COMPLETE",
        payload=OnlineSuiteClosure(
            corpora=closures,
            output_transfer_receipt_uri=(
                namespace.parent / f"{namespace.name}.output-transfer.json"
            ).as_uri(),
            output_transfer_receipt_sha256=transfer.receipt_sha256,
            output_transfer_receipt_file_sha256=transfer.file_sha256,
            source_online_tree_sha256=transfer.source_online_tree_sha256,
            canonical_online_tree_sha256=transfer.canonical_online_tree_sha256,
            run_output_aggregate=_run_output_aggregate(
                closures,
                claim_state_sha256=claim_record.record_sha256,
                claim_ledger_commit="4" * 40,
                provider_identity_sha256=(claim_payload.provider_identity.identity_sha256),
                output_aggregate_identity=(claim_payload.execution_claim.output_aggregate_identity),
            ),
        ),
    )
    assert online_complete.state == "ONLINE_COMPLETE"
    assert len(transfer.corpora) == len(FIXED_CORPORA)
    assert sum(len(row.files) for row in transfer.corpora) == 11 * len(FIXED_CORPORA)
