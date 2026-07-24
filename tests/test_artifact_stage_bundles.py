from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from test_authorized_index_store import (
    _config as _index_config,
)
from test_authorized_index_store import (
    _embedding_store,
    _Execution,
    _FakeBackend,
)

from fractal_ann_diagnostics.artifact_stage_bundles import (
    ARTIFACT_STAGE_ORDER,
    POLICY_SOURCE_STAGE_BY_BUNDLE_STAGE,
    ArtifactStageBundleError,
    inspect_index_stage,
    seal_index_stage_bundle,
    seal_policy_stage_bundle,
    verify_index_stage_bundle,
    verify_policy_stage_bundle,
)
from fractal_ann_diagnostics.authorized_index_store import (
    RECEIPT_FILENAME as INDEX_RECEIPT_FILENAME,
)
from fractal_ann_diagnostics.authorized_index_store import (
    build_authorized_index_store,
    load_authorized_index_store_receipt,
)
from fractal_ann_diagnostics.embedding_store import verify_embedding_store
from fractal_ann_diagnostics.freeze_package import FreezeArtifactLayout, _inspect_target
from fractal_ann_diagnostics.policy_intervention import (
    PolicyInterventionConfig,
    load_policy_intervention_receipt,
    write_policy_intervention_package,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_bundles(tmp_path: Path, *, corpus_id: str = "scifact") -> tuple[Path, Path, Path]:
    artifact_root = (tmp_path / "artifacts").resolve()
    embedding_parent = artifact_root / "embedding-stores"
    embedding_parent.mkdir(parents=True)
    generated_embedding, _ = _embedding_store(embedding_parent, document_count=257)
    embedding_root = embedding_parent / corpus_id
    generated_embedding.rename(embedding_root)
    embedding = verify_embedding_store(embedding_root)

    policy_root = artifact_root / "policy-workloads" / corpus_id
    policy_root.mkdir(parents=True)
    for stage in ARTIFACT_STAGE_ORDER:
        execution = _Execution(
            embedding.document_count,
            embedding.row_orders["documents"].row_order_sha256,
        )
        execution.corpus = corpus_id
        execution.stage = POLICY_SOURCE_STAGE_BY_BUNDLE_STAGE[stage]
        execution.artifact_sha256 = _digest(f"{corpus_id}-{stage}-execution")
        config = PolicyInterventionConfig(
            seed_sha256=_digest(f"{corpus_id}-{stage}-policy-seed"),
            baseline_seed_sha256=_digest(f"{corpus_id}-{stage}-baseline-seed"),
            policy_bundle_revision=f"sha256:{_digest(f'{corpus_id}-{stage}-policy')}",
            baseline_policy_revision=(f"sha256:{_digest(f'{corpus_id}-{stage}-baseline-policy')}"),
            subject_ids=("reader-a",),
            assignment_repetitions=1,
        )
        write_policy_intervention_package(execution, config, policy_root / stage)
    policy_bundle = seal_policy_stage_bundle(policy_root, corpus_id=corpus_id)

    index_root = artifact_root / "authorized-index-stores" / corpus_id
    index_root.mkdir(parents=True)
    for stage in ARTIFACT_STAGE_ORDER:
        policy_stage_root = policy_root / stage
        policy_receipt = load_policy_intervention_receipt(
            policy_stage_root / "intervention-receipt.json"
        )
        build_authorized_index_store(
            embedding_root,
            policy_stage_root,
            index_root / stage,
            expected_embedding_receipt_sha256=embedding.receipt_sha256,
            expected_policy_receipt_sha256=policy_receipt.artifact_sha256,
            config=_index_config(),
            backend=_FakeBackend(),
        )
    seal_index_stage_bundle(
        index_root,
        corpus_id=corpus_id,
        embedding_store_root=embedding_root,
        policy_bundle_root=policy_root,
    )
    assert policy_bundle.corpus_id == corpus_id
    return artifact_root, policy_root, index_root


def test_stage_bundles_close_membership_and_freeze_inspection(tmp_path: Path) -> None:
    artifact_root, policy_root, index_root = _build_bundles(tmp_path)
    embedding_root = artifact_root / "embedding-stores" / "scifact"

    policy = verify_policy_stage_bundle(policy_root, expected_corpus_id="scifact")
    indexes = verify_index_stage_bundle(
        index_root,
        embedding_store_root=embedding_root,
        policy_bundle_root=policy_root,
        expected_corpus_id="scifact",
    )
    assert tuple(row.stage for row in policy.stages) == ARTIFACT_STAGE_ORDER
    assert tuple(row.stage for row in indexes.stages) == ARTIFACT_STAGE_ORDER

    policy_row = _inspect_target(
        FreezeArtifactLayout(
            artifact_id="scifact-policy-workload",
            role="policy-workload",
            relative_path="policy-workloads/scifact",
            kind="directory",
        ),
        artifact_root,
        tmp_path,
    )
    index_row = _inspect_target(
        FreezeArtifactLayout(
            artifact_id="scifact-authorized-index-store",
            role="authorized-index-store",
            relative_path="authorized-index-stores/scifact",
            kind="directory",
        ),
        artifact_root,
        tmp_path,
    )
    assert policy_row["revision"] == f"sha256:{policy.receipt_sha256}"
    assert index_row["revision"] == f"sha256:{indexes.receipt_sha256}"

    extra = policy_root / "sealed" / "undeclared.bin"
    extra.write_bytes(b"undeclared")
    with pytest.raises(ArtifactStageBundleError, match="membership differs"):
        verify_policy_stage_bundle(policy_root, expected_corpus_id="scifact")
    extra.unlink()

    linked = policy_root / "undeclared-link"
    linked.symlink_to(policy_root / "stage-bundle.json")
    with pytest.raises(ArtifactStageBundleError, match="symlink|link"):
        verify_policy_stage_bundle(policy_root, expected_corpus_id="scifact")


def test_index_stage_recomputes_mask_and_build_bindings(tmp_path: Path) -> None:
    artifact_root, policy_root, index_root = _build_bundles(tmp_path)
    embedding = verify_embedding_store(artifact_root / "embedding-stores" / "scifact")
    policy = verify_policy_stage_bundle(policy_root, expected_corpus_id="scifact")
    stage = "sealed"
    stage_root = index_root / stage
    receipt_path = stage_root / INDEX_RECEIPT_FILENAME
    receipt = load_authorized_index_store_receipt(stage_root)
    forged_item = replace(receipt.indexes[0], build_binding_sha256="0" * 64)
    forged = replace(receipt, indexes=(forged_item, *receipt.indexes[1:]))
    receipt_path.write_bytes(forged.canonical_file_bytes())

    with pytest.raises(ArtifactStageBundleError, match="compiled policy mask"):
        inspect_index_stage(
            stage_root,
            expected_stage=stage,
            embedding_receipt=embedding,
            policy_stage=policy.stages[-1],
            policy_stage_root=policy_root / stage,
        )


def test_noncanonical_bundle_metadata_is_rejected(tmp_path: Path) -> None:
    _, policy_root, _ = _build_bundles(tmp_path)
    receipt_path = policy_root / "stage-bundle.json"
    receipt_path.write_bytes(b"{}\n")

    with pytest.raises(ArtifactStageBundleError, match="fields differ"):
        verify_policy_stage_bundle(policy_root, expected_corpus_id="scifact")
