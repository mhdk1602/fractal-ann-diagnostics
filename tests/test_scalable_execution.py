from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import fractal_ann_diagnostics.scalable_execution as scalable
from fractal_ann_diagnostics.artifact_integrity import (
    ArtifactVerificationReceipt,
    LocalArtifactSpec,
    digest_directory_tree,
    verify_local_artifacts,
    write_verification_receipt,
)
from fractal_ann_diagnostics.scalable_execution import (
    EXECUTION_LEAF_RECEIPT_FILENAME,
    ONLINE_EXECUTION_PLAN_FILENAME,
    CorpusShard,
    CorpusShardInventory,
    ExecutionLeafVerificationReceipt,
    ImmutableArtifactPin,
    IndexArtifactDescriptor,
    OnlineExecutionPackage,
    OpaqueTrialRow,
    ProvenanceSidecarDescriptor,
    QueryTrialStoreDescriptor,
    ScalableExecutionError,
    ShardedOnlineExecutionPlan,
    VectorStoreDescriptor,
    execution_compatibility_view,
    finalize_online_execution_package,
    load_corpus_shard_inventory,
    load_execution_leaf_verification_receipt,
    load_pinned_corpus_shard_inventory,
    load_sharded_online_execution_plan,
    loads_sharded_online_execution_plan,
    open_digest_provenance_registry,
    verify_online_execution_package,
    write_corpus_shard_inventory,
    write_sharded_online_execution_plan,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(root: Path, relative_path: str, payload: bytes) -> Path:
    target = root.joinpath(*relative_path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def _pin(
    artifact_id: str,
    relative_path: str,
    payload: bytes,
) -> ImmutableArtifactPin:
    return ImmutableArtifactPin(
        artifact_id=artifact_id,
        relative_path=relative_path,
        kind="file",
        byte_count=len(payload),
        sha256=_digest(payload),
    )


@dataclass(frozen=True)
class _ScaleFixture:
    root: Path
    plan: ShardedOnlineExecutionPlan
    inventory: CorpusShardInventory
    receipt: ArtifactVerificationReceipt
    component_receipt: ArtifactVerificationReceipt
    component_bindings: tuple[tuple[str, str], ...]
    plan_path: Path
    inventory_path: Path
    sidecar_path: Path
    content_digests: tuple[str, ...]


@pytest.fixture
def scale_fixture(tmp_path: Path) -> _ScaleFixture:
    root = (tmp_path / "artifacts").resolve()
    root.mkdir()
    manifest_sha256 = _digest(b"manifest")
    universe_sha256 = _digest(b"ordered-document-universe")
    document_count = 5

    shard_payloads = (
        b'{"document_id":0,"text":"alpha"}\n{"document_id":1,"text":"beta"}\n',
        b'{"document_id":2,"text":"gamma"}\n'
        b'{"document_id":3,"text":"delta"}\n'
        b'{"document_id":4,"text":"epsilon"}\n',
    )
    shard_paths = (
        "corpus/part-00000.jsonl",
        "corpus/part-00001.jsonl",
    )
    shard_pins = tuple(
        _pin(f"fullwiki-shard-{position}", path, payload)
        for position, (path, payload) in enumerate(zip(shard_paths, shard_payloads))
    )
    for path, payload in zip(shard_paths, shard_payloads):
        _write(root, path, payload)
    inventory = CorpusShardInventory(
        corpus="fullwiki",
        stage="sealed",
        document_count=document_count,
        ordered_document_universe_sha256=universe_sha256,
        shards=(
            CorpusShard(
                artifact=shard_pins[0],
                first_document_id=0,
                document_count=2,
            ),
            CorpusShard(
                artifact=shard_pins[1],
                first_document_id=2,
                document_count=3,
            ),
        ),
    )
    inventory_path = root / "control" / "corpus-shards.json"
    inventory_path.parent.mkdir()
    write_corpus_shard_inventory(inventory, inventory_path)
    inventory_payload = inventory.canonical_file_bytes()
    inventory_pin = _pin(
        "fullwiki-shard-inventory",
        "control/corpus-shards.json",
        inventory_payload,
    )

    query_payload = (
        b'{"family_key":"opaque","query_row":0}\n{"family_key":"opaque","query_row":1}\n'
    )
    query_pin = _pin(
        "fullwiki-query-trial-store",
        "queries/sealed-trials.jsonl",
        query_payload,
    )
    _write(root, query_pin.relative_path, query_payload)
    query_receipt_payload = b'{"schema_version":"fixture-query-trial-receipt-v1"}\n'
    query_receipt_pin = _pin(
        "fullwiki-query-trial-receipt",
        "queries/query-trial-receipt.json",
        query_receipt_payload,
    )
    _write(root, query_receipt_pin.relative_path, query_receipt_payload)

    vector_shape = (document_count, 3)
    active_payload = bytes(range(60))
    truth_payload = bytes(range(60, 120))
    active_pin = _pin(
        "fullwiki-active-vectors",
        "vectors/active-f32.raw",
        active_payload,
    )
    truth_pin = _pin(
        "fullwiki-current-truth-vectors",
        "vectors/current-truth-f32.raw",
        truth_payload,
    )
    _write(root, active_pin.relative_path, active_payload)
    _write(root, truth_pin.relative_path, truth_payload)

    content_digests = tuple(
        _digest(f"fullwiki-document-{document_id}".encode("utf-8"))
        for document_id in range(document_count)
    )
    sidecar_payload = b"".join(bytes.fromhex(value) for value in content_digests)
    sidecar_pin = _pin(
        "fullwiki-content-sha256",
        "provenance/content-sha256.bin",
        sidecar_payload,
    )
    sidecar_path = _write(root, sidecar_pin.relative_path, sidecar_payload)

    index_payload = b"hnsw-index-fixture-v1"
    index_pin = _pin(
        "fullwiki-hnsw-index",
        "indexes/active-hnsw.bin",
        index_payload,
    )
    _write(root, index_pin.relative_path, index_payload)

    all_pins = (
        *shard_pins,
        inventory_pin,
        query_pin,
        query_receipt_pin,
        active_pin,
        truth_pin,
        sidecar_pin,
        index_pin,
    )
    receipt = verify_local_artifacts(
        root,
        manifest_sha256=manifest_sha256,
        artifacts=tuple(
            LocalArtifactSpec(
                artifact_id=pin.artifact_id,
                relative_path=pin.relative_path,
                kind="file",
                expected_sha256=pin.sha256,
            )
            for pin in all_pins
        ),
    )
    component_bindings = tuple(
        (component, f"{component}-component")
        for component in (
            "application",
            "controller",
            "corpus",
            "embedding",
            "index",
            "policy",
        )
    )
    component_root = (tmp_path / "registered-components").resolve()
    component_root.mkdir()
    component_specs: list[LocalArtifactSpec] = []
    for component, artifact_id in component_bindings:
        payload = f"registered-{component}-component".encode("utf-8")
        relative_path = f"{component}.bin"
        _write(component_root, relative_path, payload)
        component_specs.append(
            LocalArtifactSpec(
                artifact_id=artifact_id,
                relative_path=relative_path,
                kind="file",
                expected_sha256=_digest(payload),
            )
        )
    component_receipt = verify_local_artifacts(
        component_root,
        manifest_sha256=_digest(b"c1-manifest"),
        artifacts=tuple(component_specs),
    )

    active = VectorStoreDescriptor(
        artifact=active_pin,
        role="active-migration",
        dtype="<f4",
        shape=vector_shape,
        document_universe_sha256=universe_sha256,
    )
    truth = VectorStoreDescriptor(
        artifact=truth_pin,
        role="current-exact-truth",
        dtype="<f4",
        shape=vector_shape,
        document_universe_sha256=universe_sha256,
    )
    plan = ShardedOnlineExecutionPlan(
        key_id="opaque-trial-hmac-2026-07",
        corpus="fullwiki",
        stage="sealed",
        document_count=document_count,
        ordered_document_universe_sha256=universe_sha256,
        permutation_seed=20260714,
        trials=(
            OpaqueTrialRow(
                trial_key=_digest(b"trial-1"),
                family_key=_digest(b"family-1"),
                query_row=1,
                query_record_sha256=_digest(b"query-record-1"),
            ),
            OpaqueTrialRow(
                trial_key=_digest(b"trial-0"),
                family_key=_digest(b"family-0"),
                query_row=0,
                query_record_sha256=_digest(b"query-record-0"),
            ),
        ),
        query_partition_audit_sha256=_digest(b"query-partition-audit"),
        corpus_shard_inventory=inventory_pin,
        query_trial_store=QueryTrialStoreDescriptor(
            artifact=query_pin,
            receipt=query_receipt_pin,
            record_count=2,
        ),
        active_vector_store=active,
        current_truth_vector_store=truth,
        provenance_sha256_sidecar=ProvenanceSidecarDescriptor(
            artifact=sidecar_pin,
            record_count=document_count,
            document_universe_sha256=universe_sha256,
        ),
        hnsw_index=IndexArtifactDescriptor(
            artifact=index_pin,
            document_count=document_count,
            document_universe_sha256=universe_sha256,
            source_vector_sha256=active_pin.sha256,
            format_revision="hnswlib-0.8.0-cosine-v1",
        ),
    )
    plan_path = root / "control" / "online-plan.json"
    write_sharded_online_execution_plan(plan, plan_path)
    return _ScaleFixture(
        root=root,
        plan=plan,
        inventory=inventory,
        receipt=receipt,
        component_receipt=component_receipt,
        component_bindings=component_bindings,
        plan_path=plan_path,
        inventory_path=inventory_path,
        sidecar_path=sidecar_path,
        content_digests=content_digests,
    )


def _finalize_package(fixture: _ScaleFixture) -> OnlineExecutionPackage:
    fixture.plan_path.replace(fixture.root / ONLINE_EXECUTION_PLAN_FILENAME)
    return finalize_online_execution_package(fixture.root)


def test_plan_and_multi_shard_inventory_are_bounded_canonical_controls(
    scale_fixture: _ScaleFixture,
) -> None:
    fixture = scale_fixture
    loaded_plan = load_sharded_online_execution_plan(fixture.plan_path)
    loaded_inventory = load_corpus_shard_inventory(fixture.inventory_path)

    assert loaded_plan == fixture.plan
    assert loaded_inventory == fixture.inventory
    assert loaded_plan.trial_keys == tuple(sorted(loaded_plan.trial_keys))
    assert [row.first_document_id for row in loaded_inventory.shards] == [0, 2]
    assert [row.document_count for row in loaded_inventory.shards] == [2, 3]
    assert "documents" not in loaded_plan.to_dict()
    assert stat.S_IMODE(fixture.plan_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(fixture.inventory_path.stat().st_mode) == 0o600

    with pytest.raises(ScalableExecutionError, match="already exists"):
        write_sharded_online_execution_plan(fixture.plan, fixture.plan_path)
    with pytest.raises(ScalableExecutionError, match="already exists"):
        write_corpus_shard_inventory(fixture.inventory, fixture.inventory_path)


def test_plan_parser_rejects_unknown_duplicate_noncanonical_and_oversize(
    scale_fixture: _ScaleFixture,
) -> None:
    payload = scale_fixture.plan.to_dict()
    payload["documents"] = []
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    with pytest.raises(ScalableExecutionError, match="fields differ"):
        loads_sharded_online_execution_plan(encoded)

    with pytest.raises(ScalableExecutionError, match="duplicate key"):
        loads_sharded_online_execution_plan(b'{"schema_version":"one","schema_version":"two"}\n')

    noncanonical = (
        json.dumps(scale_fixture.plan.to_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    with pytest.raises(ScalableExecutionError, match="not canonical"):
        loads_sharded_online_execution_plan(noncanonical)

    with pytest.raises(ScalableExecutionError, match="exceeds"):
        loads_sharded_online_execution_plan(b"x" * (8 * 1024 * 1024 + 1))


def test_inventory_parser_rejects_unknown_noncanonical_and_noncontiguous_rows(
    scale_fixture: _ScaleFixture,
) -> None:
    payload = scale_fixture.inventory.to_dict()
    payload["inline_documents"] = []
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    with pytest.raises(ScalableExecutionError, match="fields differ"):
        scalable.loads_corpus_shard_inventory(encoded)

    reversed_rows = scale_fixture.inventory.to_dict()
    reversed_rows["shards"].reverse()
    noncanonical = (
        json.dumps(reversed_rows, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    )
    with pytest.raises(ScalableExecutionError, match="not canonical"):
        scalable.loads_corpus_shard_inventory(noncanonical)

    with pytest.raises(ScalableExecutionError, match="contiguous"):
        replace(
            scale_fixture.inventory,
            shards=(
                scale_fixture.inventory.shards[0],
                replace(
                    scale_fixture.inventory.shards[1],
                    first_document_id=3,
                ),
            ),
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_control_loaders_reject_symlink_and_hard_link_aliases(
    scale_fixture: _ScaleFixture,
) -> None:
    fixture = scale_fixture
    plan_link = fixture.root / "control" / "plan-link.json"
    plan_link.symlink_to(fixture.plan_path)
    with pytest.raises(ScalableExecutionError, match="symlink"):
        load_sharded_online_execution_plan(plan_link)

    inventory_alias = fixture.root / "control" / "inventory-hardlink.json"
    os.link(fixture.inventory_path, inventory_alias)
    with pytest.raises(ScalableExecutionError, match="hard-linked"):
        load_corpus_shard_inventory(inventory_alias)


def test_vector_roles_shape_row_order_and_byte_count_are_fail_closed(
    scale_fixture: _ScaleFixture,
) -> None:
    active = scale_fixture.plan.active_vector_store
    with pytest.raises(ScalableExecutionError, match="byte_count"):
        VectorStoreDescriptor(
            artifact=replace(
                active.artifact,
                byte_count=active.artifact.byte_count - 4,
            ),
            role=active.role,
            dtype=active.dtype,
            shape=active.shape,
            document_universe_sha256=active.document_universe_sha256,
        )

    with pytest.raises(ScalableExecutionError, match="explicit portable"):
        replace(active, dtype="float32")
    with pytest.raises(ScalableExecutionError, match="row_order"):
        replace(active, row_order="shard-order")
    with pytest.raises(ScalableExecutionError, match="active_vector_store"):
        replace(
            scale_fixture.plan,
            active_vector_store=replace(active, role="current-exact-truth"),
        )
    with pytest.raises(ScalableExecutionError, match="bind the active"):
        replace(
            scale_fixture.plan,
            hnsw_index=replace(
                scale_fixture.plan.hnsw_index,
                source_vector_sha256="f" * 64,
            ),
        )


def test_registry_uses_fixed_width_pread_and_never_opens_corpus_jsonl(
    scale_fixture: _ScaleFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = scale_fixture
    shard_paths = [
        fixture.root.joinpath(*row.artifact.relative_path.split("/"))
        for row in fixture.inventory.shards
    ]
    assert all(b'"document_id"' in path.read_bytes() for path in shard_paths)
    original_json_loads = scalable.json.loads

    def reject_corpus_json(value, *args, **kwargs):
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        if isinstance(text, str) and '"document_id"' in text:
            raise AssertionError("registry admission parsed corpus JSONL")
        return original_json_loads(value, *args, **kwargs)

    monkeypatch.setattr(scalable.json, "loads", reject_corpus_json)
    for path in shard_paths:
        path.chmod(0)
    registry = None
    try:
        registry = open_digest_provenance_registry(
            fixture.plan,
            artifact_root=fixture.root,
            verification_receipt=fixture.receipt,
            component_verification_receipt=fixture.component_receipt,
            component_artifact_ids=fixture.component_bindings,
        )
        original_pread = scalable.os.pread
        preads: list[tuple[int, int]] = []

        def observed_pread(fd: int, size: int, offset: int) -> bytes:
            preads.append((size, offset))
            return original_pread(fd, size, offset)

        monkeypatch.setattr(scalable.os, "pread", observed_pread)
        assert registry.shard_count == 2
        assert registry.content_sha256(4) == fixture.content_digests[4]
        assert registry.lookup_content_hash(0) == (f"sha256:{fixture.content_digests[0]}")
        assert preads == [(32, 128), (32, 0)]
        assert "_document_content_sha256" not in registry.__dict__
    finally:
        if registry is not None:
            registry.close()
        for path in shard_paths:
            path.chmod(0o600)

    assert registry is not None
    with pytest.raises(ScalableExecutionError, match="closed"):
        registry.content_sha256(0)


def test_registry_accepts_a_secure_receipt_path(
    scale_fixture: _ScaleFixture,
) -> None:
    receipt_path = scale_fixture.root / "control" / "artifact-receipt.json"
    write_verification_receipt(scale_fixture.receipt, receipt_path)
    with open_digest_provenance_registry(
        scale_fixture.plan,
        artifact_root=scale_fixture.root,
        verification_receipt=receipt_path,
        component_verification_receipt=scale_fixture.component_receipt,
        component_artifact_ids=scale_fixture.component_bindings,
    ) as registry:
        assert registry.content_sha256(2) == scale_fixture.content_digests[2]


def test_registry_exposes_only_manifest_bound_audit_components(
    scale_fixture: _ScaleFixture,
) -> None:
    fixture = scale_fixture
    expected_by_id = {
        row.artifact_id: row.verified_sha256 for row in fixture.component_receipt.artifacts
    }
    expected_revisions = tuple(
        (component, expected_by_id[artifact_id])
        for component, artifact_id in fixture.component_bindings
    )

    with open_digest_provenance_registry(
        fixture.plan,
        artifact_root=fixture.root,
        verification_receipt=fixture.receipt,
        component_verification_receipt=fixture.component_receipt,
        component_artifact_ids=fixture.component_bindings,
    ) as registry:
        assert registry.component_revisions == expected_revisions
        assert {name for name, _ in registry.component_revisions} == {
            "application",
            "controller",
            "corpus",
            "embedding",
            "index",
            "policy",
        }
        assert registry.verification_receipt_sha256 == fixture.component_receipt.receipt_sha256
        assert registry.execution_verification_receipt_sha256 == fixture.receipt.receipt_sha256
        assert not {
            "active_vector_store",
            "current_truth_vector_store",
            "hnsw_index",
            "provenance_sha256_sidecar",
            "query_trial_store",
        } & {name for name, _ in registry.component_revisions}


@pytest.mark.parametrize("mutation", ("missing", "extra", "unsorted", "duplicate-id"))
def test_registry_rejects_noncanonical_component_binding_closure(
    scale_fixture: _ScaleFixture,
    mutation: str,
) -> None:
    fixture = scale_fixture
    bindings = fixture.component_bindings
    if mutation == "missing":
        mutated = bindings[:-1]
    elif mutation == "extra":
        mutated = (*bindings, ("runtime", bindings[0][1]))
    elif mutation == "unsorted":
        mutated = (bindings[1], bindings[0], *bindings[2:])
    else:
        mutated = (*bindings[:-1], (bindings[-1][0], bindings[0][1]))

    with pytest.raises(ScalableExecutionError, match="exact six sorted audit components"):
        open_digest_provenance_registry(
            fixture.plan,
            artifact_root=fixture.root,
            verification_receipt=fixture.receipt,
            component_verification_receipt=fixture.component_receipt,
            component_artifact_ids=mutated,
        )


def test_registry_rejects_unverified_or_nonexact_audit_component(
    scale_fixture: _ScaleFixture,
) -> None:
    fixture = scale_fixture
    missing_bindings = tuple(
        (component, "unverified-component" if component == "policy" else artifact_id)
        for component, artifact_id in fixture.component_bindings
    )
    with pytest.raises(ScalableExecutionError, match="unverified IDs"):
        open_digest_provenance_registry(
            fixture.plan,
            artifact_root=fixture.root,
            verification_receipt=fixture.receipt,
            component_verification_receipt=fixture.component_receipt,
            component_artifact_ids=missing_bindings,
        )

    policy_artifact_id = dict(fixture.component_bindings)["policy"]
    nonexact_rows = tuple(
        replace(
            row,
            kind="directory",
            exact=False,
            observed_file_count=row.observed_file_count + 1,
            observed_byte_count=row.observed_byte_count + 1,
        )
        if row.artifact_id == policy_artifact_id
        else row
        for row in fixture.component_receipt.artifacts
    )
    nonexact_receipt = replace(fixture.component_receipt, artifacts=nonexact_rows)
    with pytest.raises(ScalableExecutionError, match="was not verified exactly"):
        open_digest_provenance_registry(
            fixture.plan,
            artifact_root=fixture.root,
            verification_receipt=fixture.receipt,
            component_verification_receipt=nonexact_receipt,
            component_artifact_ids=fixture.component_bindings,
        )


@pytest.mark.parametrize("mutation", ["length", "digest", "hard-link"])
def test_registry_rejects_length_digest_and_hard_link_mutations(
    scale_fixture: _ScaleFixture,
    mutation: str,
) -> None:
    fixture = scale_fixture
    if mutation == "length":
        fixture.sidecar_path.write_bytes(fixture.sidecar_path.read_bytes() + b"x")
        expected = "length differs"
    elif mutation == "digest":
        payload = bytearray(fixture.sidecar_path.read_bytes())
        payload[0] ^= 0xFF
        fixture.sidecar_path.write_bytes(bytes(payload))
        expected = "SHA-256 differs"
    else:
        os.link(fixture.sidecar_path, fixture.root / "provenance" / "alias.bin")
        expected = "hard-linked"

    with pytest.raises(ScalableExecutionError, match=expected):
        open_digest_provenance_registry(
            fixture.plan,
            artifact_root=fixture.root,
            verification_receipt=fixture.receipt,
            component_verification_receipt=fixture.component_receipt,
            component_artifact_ids=fixture.component_bindings,
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_registry_rejects_a_sidecar_symlink(
    scale_fixture: _ScaleFixture,
) -> None:
    fixture = scale_fixture
    payload = fixture.sidecar_path.read_bytes()
    real = fixture.root / "provenance" / "content-real.bin"
    real.write_bytes(payload)
    fixture.sidecar_path.unlink()
    fixture.sidecar_path.symlink_to(real)

    with pytest.raises(ScalableExecutionError, match="symlink"):
        open_digest_provenance_registry(
            fixture.plan,
            artifact_root=fixture.root,
            verification_receipt=fixture.receipt,
            component_verification_receipt=fixture.component_receipt,
            component_artifact_ids=fixture.component_bindings,
        )


def test_registry_detects_path_substitution_after_admission(
    scale_fixture: _ScaleFixture,
) -> None:
    fixture = scale_fixture
    registry = open_digest_provenance_registry(
        fixture.plan,
        artifact_root=fixture.root,
        verification_receipt=fixture.receipt,
        component_verification_receipt=fixture.component_receipt,
        component_artifact_ids=fixture.component_bindings,
    )
    payload = fixture.sidecar_path.read_bytes()
    admitted = fixture.root / "provenance" / "admitted.bin"
    fixture.sidecar_path.rename(admitted)
    fixture.sidecar_path.write_bytes(payload)
    try:
        with pytest.raises(ScalableExecutionError, match="substituted"):
            registry.content_sha256(0)
    finally:
        registry.close()


def test_receipt_and_inventory_binding_reject_substitution(
    scale_fixture: _ScaleFixture,
) -> None:
    fixture = scale_fixture
    substituted = replace(
        fixture.plan,
        corpus_shard_inventory=replace(
            fixture.plan.corpus_shard_inventory,
            sha256="e" * 64,
        ),
    )
    with pytest.raises(ScalableExecutionError, match="does not exactly attest"):
        load_pinned_corpus_shard_inventory(
            substituted,
            artifact_root=fixture.root,
            verification_receipt=fixture.receipt,
        )


def test_plan_has_no_future_digest_backreferences(
    scale_fixture: _ScaleFixture,
) -> None:
    payload = scale_fixture.plan.to_dict()
    assert payload["schema_version"] == "fractal-sharded-online-execution-v4"
    assert "manifest_sha256" not in payload
    assert "artifact_verification_receipt_sha256" not in payload

    for forbidden in ("manifest_sha256", "artifact_verification_receipt_sha256"):
        mutated = dict(payload)
        mutated[forbidden] = "f" * 64
        encoded = json.dumps(mutated, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with pytest.raises(ScalableExecutionError, match="fields differ"):
            loads_sharded_online_execution_plan(encoded)


def test_online_execution_package_separates_tree_pin_from_logical_revision(
    scale_fixture: _ScaleFixture,
) -> None:
    package = _finalize_package(scale_fixture)

    assert package.tree_sha256 == digest_directory_tree(package.root).sha256
    assert package.tree_sha256 != package.plan.artifact_sha256
    assert package.revision == f"sha256:{package.plan.artifact_sha256}"
    with pytest.raises(ScalableExecutionError, match="must differ"):
        replace(package, tree_sha256=package.plan.artifact_sha256)
    assert package.leaf_receipt.plan_sha256 == package.plan.artifact_sha256
    assert package.leaf_receipt.plan_file_sha256 == package.plan.file_sha256
    assert {row.artifact_id for row in package.leaf_receipt.artifacts} == {
        pin.artifact_id
        for pin in (
            *package.plan.direct_artifact_pins,
            *(shard.artifact for shard in package.inventory.shards),
        )
    }
    assert b"manifest" not in package.plan.canonical_file_bytes().lower()
    assert b"manifest" not in package.leaf_receipt.canonical_file_bytes().lower()

    assert (
        verify_online_execution_package(
            package.root,
            expected_tree_sha256=package.tree_sha256,
            expected_plan_revision=package.revision,
        )
        == package
    )
    assert (
        load_execution_leaf_verification_receipt(package.root / EXECUTION_LEAF_RECEIPT_FILENAME)
        == package.leaf_receipt
    )


def test_package_finalizer_rejects_undeclared_members_before_writing_receipt(
    scale_fixture: _ScaleFixture,
) -> None:
    scale_fixture.plan_path.replace(scale_fixture.root / ONLINE_EXECUTION_PLAN_FILENAME)
    (scale_fixture.root / "undeclared.bin").write_bytes(b"not in the plan")

    with pytest.raises(ScalableExecutionError, match="membership differs"):
        finalize_online_execution_package(scale_fixture.root)
    assert not (scale_fixture.root / EXECUTION_LEAF_RECEIPT_FILENAME).exists()


@pytest.mark.parametrize(
    "mutation",
    ("addition", "omission", "substitution", "receipt-substitution"),
)
def test_package_rejects_membership_and_leaf_drift_even_with_recomputed_tree_pin(
    scale_fixture: _ScaleFixture,
    mutation: str,
) -> None:
    package = _finalize_package(scale_fixture)
    if mutation == "addition":
        (package.root / "undeclared.bin").write_bytes(b"not in the plan")
        expected = "membership differs"
    else:
        pin = (
            package.plan.query_trial_store.receipt
            if mutation == "receipt-substitution"
            else package.plan.query_trial_store.artifact
        )
        target = package.root.joinpath(*pin.relative_path.split("/"))
        if mutation == "omission":
            target.unlink()
            expected = "membership differs"
        else:
            target.write_bytes(b"substituted query rows\n")
            expected = "leaf verification failed"
    mutated_tree_sha256 = digest_directory_tree(package.root).sha256

    with pytest.raises(ScalableExecutionError, match=expected):
        verify_online_execution_package(
            package.root,
            expected_tree_sha256=mutated_tree_sha256,
            expected_plan_revision=package.revision,
        )


def test_package_rejects_plan_or_receipt_substitution_under_a_new_tree_pin(
    scale_fixture: _ScaleFixture,
) -> None:
    package = _finalize_package(scale_fixture)
    plan_path = package.root / ONLINE_EXECUTION_PLAN_FILENAME
    changed_plan = replace(package.plan, key_id="different-opaque-key")
    plan_path.unlink()
    plan_path.write_bytes(changed_plan.canonical_file_bytes())
    changed_tree_sha256 = digest_directory_tree(package.root).sha256

    with pytest.raises(ScalableExecutionError, match="binds a different logical plan"):
        verify_online_execution_package(
            package.root,
            expected_tree_sha256=changed_tree_sha256,
            expected_plan_revision=f"sha256:{changed_plan.artifact_sha256}",
        )

    plan_path.unlink()
    plan_path.write_bytes(package.plan.canonical_file_bytes())
    receipt_path = package.root / EXECUTION_LEAF_RECEIPT_FILENAME
    receipt_path.unlink()
    changed_receipt = replace(
        package.leaf_receipt,
        plan_file_sha256="f" * 64,
    )
    receipt_path.write_bytes(changed_receipt.canonical_file_bytes())
    changed_tree_sha256 = digest_directory_tree(package.root).sha256

    with pytest.raises(ScalableExecutionError, match="binds a different logical plan"):
        verify_online_execution_package(
            package.root,
            expected_tree_sha256=changed_tree_sha256,
            expected_plan_revision=package.revision,
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
@pytest.mark.parametrize("mutation", ("symlink", "hard-link"))
def test_package_tree_rejects_link_aliases(
    scale_fixture: _ScaleFixture,
    mutation: str,
) -> None:
    package = _finalize_package(scale_fixture)
    target = package.root.joinpath(
        *package.plan.query_trial_store.artifact.relative_path.split("/")
    )
    alias = package.root.parent / f"{mutation}-alias.bin"
    if mutation == "symlink":
        alias.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(alias)
        expected = "symlink is forbidden"
    else:
        os.link(target, alias)
        expected = "hard-linked file is forbidden"

    with pytest.raises(ScalableExecutionError, match=expected):
        verify_online_execution_package(
            package.root,
            expected_tree_sha256=package.tree_sha256,
            expected_plan_revision=package.revision,
        )


def test_leaf_receipt_rejects_noncanonical_and_future_manifest_fields(
    scale_fixture: _ScaleFixture,
) -> None:
    package = _finalize_package(scale_fixture)
    payload = package.leaf_receipt.to_dict()
    payload["manifest_sha256"] = "a" * 64
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with pytest.raises(ScalableExecutionError, match="fields differ"):
        scalable.loads_execution_leaf_verification_receipt(encoded)

    noncanonical = json.dumps(
        package.leaf_receipt.to_dict(),
        indent=2,
        sort_keys=True,
    )
    with pytest.raises(ScalableExecutionError, match="not canonical"):
        scalable.loads_execution_leaf_verification_receipt(noncanonical)

    with pytest.raises(ScalableExecutionError, match="verified artifact rows"):
        ExecutionLeafVerificationReceipt(
            plan_sha256=package.plan.artifact_sha256,
            plan_file_sha256=package.plan.file_sha256,
            artifacts=(),
        )


def test_compatibility_helpers_cover_sharded_and_inline_shapes(
    scale_fixture: _ScaleFixture,
) -> None:
    sharded = execution_compatibility_view(scale_fixture.plan)
    assert sharded.document_count == 5
    assert sharded.document_universe_sha256 == scale_fixture.plan.ordered_document_universe_sha256
    assert sharded.trial_keys == scale_fixture.plan.trial_keys
    assert sharded.artifact_sha256 == scale_fixture.plan.artifact_sha256

    canonical = b'{"inline":"execution"}'
    inline = SimpleNamespace(
        documents=tuple(
            SimpleNamespace(
                document_id=index,
                external_id=f"doc-{index}",
                source_uri=f"corpus://doc/{index}",
                content_hash=f"sha256:{index + 1:064x}",
            )
            for index in range(3)
        ),
        trials=(
            SimpleNamespace(trial_key="b" * 64),
            SimpleNamespace(trial_key="a" * 64),
        ),
        canonical_bytes=lambda: canonical,
        artifact_sha256=_digest(canonical),
    )
    view = execution_compatibility_view(inline)
    assert view.document_count == 3
    assert len(view.document_universe_sha256) == 64
    assert view.trial_keys == ("a" * 64, "b" * 64)
    assert view.artifact_sha256 == _digest(canonical)

    inline.artifact_sha256 = "c" * 64
    with pytest.raises(ScalableExecutionError, match="differs"):
        execution_compatibility_view(inline)
