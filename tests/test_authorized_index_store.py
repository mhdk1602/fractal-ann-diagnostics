from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fractal_ann_diagnostics.authorized_index_store import (
    AUTHORIZED_INDEX_BUILDER_IDENTITY,
    FAILURE_POLICY,
    AuthorizedIndexConfig,
    AuthorizedIndexStoreError,
    VerifiedAuthorizedIndexProvider,
    build_authorized_index_store,
    load_authorized_index_store_receipt,
    open_verified_document_matrices,
    verify_authorized_index_store,
)
from fractal_ann_diagnostics.compiled_policy import CompiledPolicyMaskStore
from fractal_ann_diagnostics.embedding_store import (
    EMBEDDING_BUILDER_VERSION,
    EmbeddingStoreReceipt,
    RowOrderDescriptor,
    VectorDescriptor,
)
from fractal_ann_diagnostics.policy import PolicyDecision
from fractal_ann_diagnostics.policy_intervention import (
    PolicyInterventionConfig,
    load_policy_intervention_receipt,
    write_policy_intervention_package,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_npy(path: Path, values: np.ndarray) -> tuple[int, str]:
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=values.dtype,
        shape=values.shape,
        fortran_order=False,
        version=(2, 0),
    )
    array[:] = values
    array.flush()
    del array
    encoded = path.read_bytes()
    return len(encoded), _digest(encoded)


def _vector_descriptor(
    path: Path,
    values: np.ndarray,
    *,
    row_order_sha256: str,
    model_revision: str,
    model_tree_sha256: str,
    prompt_sha256: str,
) -> VectorDescriptor:
    byte_count, file_sha256 = _write_npy(path, values)
    return VectorDescriptor(
        relative_path=path.name,
        dtype=values.dtype.name,
        shape=values.shape,
        row_order_sha256=row_order_sha256,
        byte_count=byte_count,
        file_sha256=file_sha256,
        model_tree_sha256=model_tree_sha256,
        model_revision=model_revision,
        prompt_sha256=prompt_sha256,
    )


class _Trial:
    def __init__(self, trial_key: str, family_key: str) -> None:
        self.trial_key = trial_key
        self.family_key = family_key


class _Execution:
    corpus = "fixture"
    stage = "sealed"

    def __init__(self, document_count: int, universe: str) -> None:
        self.document_count = document_count
        self.document_universe_sha256 = universe
        self.artifact_sha256 = _digest("fixture-execution")
        family_key = _digest("family-0")
        self.trials = tuple(
            _Trial(_digest(f"trial-{position}"), family_key) for position in range(3)
        )


class _FakeIndex:
    def __init__(self, backend: _FakeBackend, metric: str, dimension: int) -> None:
        self.backend = backend
        self.metric = metric
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.labels = np.empty(0, dtype=np.int64)

    def init_index(self, **kwargs: Any) -> None:
        self.backend.init_calls.append(dict(kwargs))
        self.max_elements = int(kwargs["max_elements"])

    def set_num_threads(self, count: int) -> None:
        self.backend.thread_calls.append(count)

    def add_items(self, vectors: np.ndarray, labels: np.ndarray, *, num_threads: int) -> None:
        self.backend.add_threads.append(num_threads)
        self.backend.max_batch = max(self.backend.max_batch, len(vectors))
        self.backend.added_values.append(np.asarray(vectors).copy())
        self.backend.add_count += 1
        if self.backend.fail_after_add is not None and (
            self.backend.add_count > self.backend.fail_after_add
        ):
            raise RuntimeError("injected backend failure")
        self.vectors = np.concatenate([self.vectors, np.asarray(vectors, dtype=np.float32)])
        self.labels = np.concatenate([self.labels, np.asarray(labels, dtype=np.int64)])

    def save_index(self, path: str) -> None:
        payload = {
            "dimension": self.dimension,
            "labels": self.labels.tolist(),
            "metric": self.metric,
            "vectors": self.vectors.tolist(),
        }
        Path(path).write_bytes(_canonical(payload) + b"\n")
        self.backend.save_calls += 1

    def load_index(self, path: str, *, max_elements: int) -> None:
        payload = json.loads(Path(path).read_bytes())
        self.dimension = int(payload["dimension"])
        self.metric = str(payload["metric"])
        self.vectors = np.asarray(payload["vectors"], dtype=np.float32)
        self.labels = np.asarray(payload["labels"], dtype=np.int64)
        assert len(self.labels) == max_elements
        self.backend.load_calls += 1

    def set_ef(self, value: int) -> None:
        self.backend.ef_calls.append(value)

    def knn_query(
        self, vectors: np.ndarray, *, k: int, num_threads: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.backend.query_threads.append(num_threads)
        assert k == 1
        distances = ((self.vectors - vectors[0]) ** 2).sum(axis=1)
        position = int(np.argmin(distances))
        return (
            np.asarray([[self.labels[position]]], dtype=np.int64),
            np.asarray([[distances[position]]], dtype=np.float32),
        )


class _FakeBackend:
    backend_id = "hnswlib-python-v1"
    package_version = "0.8.0"
    build_sha256 = _digest("fake-hnswlib-build")

    def __init__(self, *, fail_after_add: int | None = None) -> None:
        self.fail_after_add = fail_after_add
        self.add_count = 0
        self.max_batch = 0
        self.save_calls = 0
        self.load_calls = 0
        self.init_calls: list[dict[str, object]] = []
        self.thread_calls: list[int] = []
        self.add_threads: list[int] = []
        self.query_threads: list[int] = []
        self.ef_calls: list[int] = []
        self.added_values: list[np.ndarray] = []

    def create_index(self, *, metric: str, dimension: int) -> _FakeIndex:
        return _FakeIndex(self, metric, dimension)


def _embedding_store(tmp_path: Path, *, document_count: int = 19) -> tuple[Path, np.ndarray]:
    root = tmp_path / "embedding-store"
    root.mkdir()
    dimension = 4
    old_documents = np.arange(document_count * dimension, dtype=np.float32).reshape(
        document_count, dimension
    )
    old_documents += 1.0
    old_documents /= np.linalg.norm(old_documents, axis=1, keepdims=True)
    current_documents = -np.roll(old_documents, shift=1, axis=1)
    old_queries = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    old_queries /= np.linalg.norm(old_queries, axis=1, keepdims=True)
    current_queries = -np.roll(old_queries, shift=1, axis=1)

    document_rows = b"".join(
        _canonical({"id": f"document-{row}", "row": row}) + b"\n" for row in range(document_count)
    )
    query_rows = _canonical({"id": "query-0", "row": 0}) + b"\n"
    (root / "document-rows.jsonl").write_bytes(document_rows)
    (root / "query-rows.jsonl").write_bytes(query_rows)
    document_order = RowOrderDescriptor(
        relative_path="document-rows.jsonl",
        row_count=document_count,
        byte_count=len(document_rows),
        row_order_sha256=_digest(document_rows),
        file_sha256=_digest(document_rows),
    )
    query_order = RowOrderDescriptor(
        relative_path="query-rows.jsonl",
        row_count=1,
        byte_count=len(query_rows),
        row_order_sha256=_digest(query_rows),
        file_sha256=_digest(query_rows),
    )
    current_tree = _digest("current-model-tree")
    old_tree = _digest("old-model-tree")
    document_prompt = _digest(b"")
    query_prompt = _digest("query prompt")
    vectors = {
        "current_documents": _vector_descriptor(
            root / "current-documents.npy",
            current_documents,
            row_order_sha256=document_order.row_order_sha256,
            model_revision="current-revision",
            model_tree_sha256=current_tree,
            prompt_sha256=document_prompt,
        ),
        "current_queries": _vector_descriptor(
            root / "current-queries.npy",
            current_queries,
            row_order_sha256=query_order.row_order_sha256,
            model_revision="current-revision",
            model_tree_sha256=current_tree,
            prompt_sha256=query_prompt,
        ),
        "old_documents": _vector_descriptor(
            root / "old-documents.npy",
            old_documents,
            row_order_sha256=document_order.row_order_sha256,
            model_revision="old-revision",
            model_tree_sha256=old_tree,
            prompt_sha256=document_prompt,
        ),
        "old_queries": _vector_descriptor(
            root / "old-queries.npy",
            old_queries,
            row_order_sha256=query_order.row_order_sha256,
            model_revision="old-revision",
            model_tree_sha256=old_tree,
            prompt_sha256=query_prompt,
        ),
    }
    config = {"fixture": "embedding-config-v1"}
    source_inventory = {"fixture": "source-inventory-v1"}
    (root / "config.json").write_bytes(_canonical(config) + b"\n")
    (root / "source-inventory.json").write_bytes(_canonical(source_inventory) + b"\n")
    receipt = EmbeddingStoreReceipt(
        staged_inventory_sha256=_digest("staged-inventory"),
        source_inventory_sha256=_digest(_canonical(source_inventory)),
        config_sha256=_digest(_canonical(config)),
        document_count=document_count,
        query_count=1,
        current_model={
            "encoder_id": "fixture-encoder",
            "revision": "current-revision",
            "tree_sha256": current_tree,
        },
        old_model={
            "encoder_id": "fixture-encoder",
            "revision": "old-revision",
            "tree_sha256": old_tree,
        },
        row_orders={"documents": document_order, "queries": query_order},
        vectors=vectors,
        builder_version=EMBEDDING_BUILDER_VERSION,
    )
    (root / "receipt.json").write_bytes(receipt.canonical_bytes() + b"\n")
    return root.resolve(), old_documents


def _policy_store(tmp_path: Path, embedding_root: Path) -> Path:
    embedding_receipt = EmbeddingStoreReceipt.from_dict(
        json.loads((embedding_root / "receipt.json").read_bytes())
    )
    execution = _Execution(
        embedding_receipt.document_count,
        embedding_receipt.row_orders["documents"].row_order_sha256,
    )
    config = PolicyInterventionConfig(
        seed_sha256=_digest("policy-seed"),
        baseline_seed_sha256=_digest("baseline-policy-seed"),
        policy_bundle_revision=f"sha256:{_digest('policy-bundle')}",
        baseline_policy_revision=f"sha256:{_digest('baseline-policy-bundle')}",
        subject_ids=("reader-a",),
        assignment_repetitions=1,
    )
    root = (tmp_path / "policy-store").resolve()
    write_policy_intervention_package(execution, config, root)
    return root


def _config(*, batch_size: int = 4) -> AuthorizedIndexConfig:
    return AuthorizedIndexConfig(
        backend_version=_FakeBackend.package_version,
        backend_build_sha256=_FakeBackend.build_sha256,
        metric="cosine",
        m=4,
        ef_construction=16,
        random_seed=20260714,
        batch_size=batch_size,
        verification_ef=8,
    )


def _source_pins(embedding_root: Path, policy_root: Path) -> tuple[str, str]:
    embedding = EmbeddingStoreReceipt.from_dict(
        json.loads((embedding_root / "receipt.json").read_bytes())
    )
    policy = load_policy_intervention_receipt(policy_root / "intervention-receipt.json")
    return embedding.receipt_sha256, policy.artifact_sha256


def _build(tmp_path: Path, *, backend: _FakeBackend | None = None):
    embedding_root, old_vectors = _embedding_store(tmp_path)
    policy_root = _policy_store(tmp_path, embedding_root)
    embedding_pin, policy_pin = _source_pins(embedding_root, policy_root)
    selected_backend = backend or _FakeBackend()
    output = (tmp_path / "authorized-indexes").resolve()
    result = build_authorized_index_store(
        embedding_root,
        policy_root,
        output,
        expected_embedding_receipt_sha256=embedding_pin,
        expected_policy_receipt_sha256=policy_pin,
        config=_config(),
        backend=selected_backend,
    )
    return (
        result,
        embedding_root,
        policy_root,
        embedding_pin,
        policy_pin,
        selected_backend,
        old_vectors,
    )


def test_builds_one_bounded_old_vector_index_and_exact_row_map_per_mask(
    tmp_path: Path,
) -> None:
    (
        result,
        embedding_root,
        policy_root,
        embedding_pin,
        policy_pin,
        backend,
        old_vectors,
    ) = _build(tmp_path)
    receipt = load_authorized_index_store_receipt(result.root)
    mask_store = CompiledPolicyMaskStore(policy_root / "compiled-policy-catalog.json")

    assert receipt.builder_identity == AUTHORIZED_INDEX_BUILDER_IDENTITY
    assert receipt.failure_policy == FAILURE_POLICY
    assert len(receipt.indexes) == len(mask_store.catalog.masks) == 3
    assert receipt.document_universe_sha256 == receipt.document_row_order_sha256
    assert receipt.old_active_vector.role == "active-old-stale"
    assert receipt.current_truth_vector.role == "current-exact-truth"
    assert receipt.old_active_vector.file_sha256 != receipt.current_truth_vector.file_sha256
    assert backend.max_batch <= _config().batch_size
    assert set(backend.add_threads) == {1}
    assert set(backend.thread_calls) == {1}
    assert set(backend.query_threads) == {1}
    assert min(float(values.min()) for values in backend.added_values) >= 0.0
    assert all(
        call
        == {
            "M": 4,
            "allow_replace_deleted": False,
            "ef_construction": 16,
            "max_elements": call["max_elements"],
            "random_seed": 20260714,
        }
        for call in backend.init_calls
    )

    for artifact in receipt.indexes:
        mask_descriptor = next(
            mask for mask in mask_store.catalog.masks if mask.mask_id == artifact.mask_id
        )
        mask = mask_store.mask(
            artifact.mask_id,
            expected_sha256=artifact.mask_sha256,
            expected_authorized_count=artifact.authorized_count,
        )
        row_map = np.load(result.root / artifact.row_map_path, allow_pickle=False)
        assert row_map.dtype == np.dtype("<i8")
        assert np.array_equal(row_map, np.flatnonzero(mask))
        assert artifact.authorized_count == mask_descriptor.authorized_count
        assert np.all(old_vectors[row_map] >= 0.0)

    assert (
        verify_authorized_index_store(
            result.root,
            embedding_store_root=embedding_root,
            policy_intervention_root=policy_root,
            expected_embedding_receipt_sha256=embedding_pin,
            expected_policy_receipt_sha256=policy_pin,
            backend=backend,
            expected_store_receipt_sha256=result.receipt_sha256,
        )
        == result
    )


def test_verified_provider_loads_only_the_live_frozen_mask_and_reuses_it(
    tmp_path: Path,
) -> None:
    (
        result,
        embedding_root,
        policy_root,
        embedding_pin,
        policy_pin,
        backend,
        old_vectors,
    ) = _build(tmp_path)
    receipt = load_authorized_index_store_receipt(result.root)
    mask_store = CompiledPolicyMaskStore(policy_root / "compiled-policy-catalog.json")
    artifact = receipt.indexes[0]
    mask = mask_store.mask(
        artifact.mask_id,
        expected_sha256=artifact.mask_sha256,
        expected_authorized_count=artifact.authorized_count,
    )
    decision = PolicyDecision(
        subject="reader-a",
        action="retrieve",
        policy_version=receipt.policy_revision,
        authorized_mask=mask,
        document_universe_sha256=receipt.document_universe_sha256,
    )
    provider = VerifiedAuthorizedIndexProvider(
        result.root,
        embedding_store_root=embedding_root,
        policy_intervention_root=policy_root,
        expected_embedding_receipt_sha256=embedding_pin,
        expected_policy_receipt_sha256=policy_pin,
        expected_store_receipt_sha256=result.receipt_sha256,
        backend=backend,
    )
    before = backend.load_calls
    index = provider.index_for(decision)
    ids, distances = index.query_with_ef(
        old_vectors[index.original_ids[0]],
        1,
        ef_search=8,
    )

    assert ids.tolist() == [int(index.original_ids[0])]
    assert distances.tolist() == [pytest.approx(0.0)]
    assert backend.load_calls == before + 1
    assert provider.index_for(decision) is index
    assert backend.load_calls == before + 1


def test_verified_provider_rejects_a_mask_outside_the_frozen_catalog(tmp_path: Path) -> None:
    result, embedding_root, policy_root, embedding_pin, policy_pin, backend, _ = _build(tmp_path)
    receipt = load_authorized_index_store_receipt(result.root)
    mask = np.zeros(receipt.document_count, dtype=bool)
    mask[[0, 1]] = True
    decision = PolicyDecision(
        subject="reader-a",
        action="retrieve",
        policy_version=receipt.policy_revision,
        authorized_mask=mask,
        document_universe_sha256=receipt.document_universe_sha256,
    )
    provider = VerifiedAuthorizedIndexProvider(
        result.root,
        embedding_store_root=embedding_root,
        policy_intervention_root=policy_root,
        expected_embedding_receipt_sha256=embedding_pin,
        expected_policy_receipt_sha256=policy_pin,
        expected_store_receipt_sha256=result.receipt_sha256,
        backend=backend,
    )

    with pytest.raises(AuthorizedIndexStoreError, match="frozen policy mask"):
        provider.index_for(decision)


@pytest.mark.parametrize("payload", ["index", "row-map"])
def test_verifier_rejects_any_frozen_payload_mutation(tmp_path: Path, payload: str) -> None:
    result, embedding_root, policy_root, embedding_pin, policy_pin, backend, _ = _build(tmp_path)
    receipt = load_authorized_index_store_receipt(result.root)
    artifact = receipt.indexes[0]
    path = result.root / (artifact.index_path if payload == "index" else artifact.row_map_path)
    encoded = bytearray(path.read_bytes())
    encoded[-1] ^= 1
    path.write_bytes(encoded)

    with pytest.raises(AuthorizedIndexStoreError, match="digest|pin|tree"):
        verify_authorized_index_store(
            result.root,
            embedding_store_root=embedding_root,
            policy_intervention_root=policy_root,
            expected_embedding_receipt_sha256=embedding_pin,
            expected_policy_receipt_sha256=policy_pin,
            backend=backend,
        )


def test_backend_binding_mismatch_is_rejected_before_output(tmp_path: Path) -> None:
    embedding_root, _ = _embedding_store(tmp_path)
    policy_root = _policy_store(tmp_path, embedding_root)
    embedding_pin, policy_pin = _source_pins(embedding_root, policy_root)
    backend = _FakeBackend()
    backend.package_version = "0.9.0"
    output = (tmp_path / "authorized-indexes").resolve()

    with pytest.raises(AuthorizedIndexStoreError, match="runtime backend"):
        build_authorized_index_store(
            embedding_root,
            policy_root,
            output,
            expected_embedding_receipt_sha256=embedding_pin,
            expected_policy_receipt_sha256=policy_pin,
            config=_config(),
            backend=backend,
        )
    assert not output.exists()


def test_failure_is_fail_clean_and_does_not_leave_lock_or_staging(tmp_path: Path) -> None:
    embedding_root, _ = _embedding_store(tmp_path)
    policy_root = _policy_store(tmp_path, embedding_root)
    embedding_pin, policy_pin = _source_pins(embedding_root, policy_root)
    output = (tmp_path / "authorized-indexes").resolve()

    with pytest.raises(RuntimeError, match="injected backend failure"):
        build_authorized_index_store(
            embedding_root,
            policy_root,
            output,
            expected_embedding_receipt_sha256=embedding_pin,
            expected_policy_receipt_sha256=policy_pin,
            config=_config(batch_size=2),
            backend=_FakeBackend(fail_after_add=1),
        )
    assert not output.exists()
    assert not (tmp_path / ".authorized-indexes.authorized-index.lock").exists()
    assert not list(tmp_path.glob(".authorized-indexes.staging-*"))


def test_existing_target_is_never_overwritten(tmp_path: Path) -> None:
    result, embedding_root, policy_root, embedding_pin, policy_pin, backend, _ = _build(tmp_path)
    before = result.receipt_sha256
    with pytest.raises(AuthorizedIndexStoreError, match="already exists"):
        build_authorized_index_store(
            embedding_root,
            policy_root,
            result.root,
            expected_embedding_receipt_sha256=embedding_pin,
            expected_policy_receipt_sha256=policy_pin,
            config=_config(),
            backend=backend,
        )
    assert load_authorized_index_store_receipt(result.root).artifact_sha256 == before


def test_existing_builder_lock_is_preserved(tmp_path: Path) -> None:
    embedding_root, _ = _embedding_store(tmp_path)
    policy_root = _policy_store(tmp_path, embedding_root)
    embedding_pin, policy_pin = _source_pins(embedding_root, policy_root)
    lock = tmp_path / ".authorized-indexes.authorized-index.lock"
    lock.write_bytes(b"other-builder\n")

    with pytest.raises(AuthorizedIndexStoreError, match="holds the lock"):
        build_authorized_index_store(
            embedding_root,
            policy_root,
            (tmp_path / "authorized-indexes").resolve(),
            expected_embedding_receipt_sha256=embedding_pin,
            expected_policy_receipt_sha256=policy_pin,
            config=_config(),
            backend=_FakeBackend(),
        )
    assert lock.read_bytes() == b"other-builder\n"


def test_policy_universe_must_equal_embedding_document_row_order(tmp_path: Path) -> None:
    embedding_root, _ = _embedding_store(tmp_path)
    embedding = EmbeddingStoreReceipt.from_dict(
        json.loads((embedding_root / "receipt.json").read_bytes())
    )
    execution = _Execution(embedding.document_count, _digest("different-universe"))
    config = PolicyInterventionConfig(
        seed_sha256=_digest("policy-seed"),
        baseline_seed_sha256=_digest("baseline-policy-seed"),
        policy_bundle_revision=f"sha256:{_digest('policy-bundle')}",
        baseline_policy_revision=f"sha256:{_digest('baseline-policy-bundle')}",
        subject_ids=("reader-a",),
        assignment_repetitions=1,
    )
    policy_root = (tmp_path / "policy-store").resolve()
    write_policy_intervention_package(execution, config, policy_root)
    embedding_pin, policy_pin = _source_pins(embedding_root, policy_root)

    with pytest.raises(AuthorizedIndexStoreError, match="row order"):
        build_authorized_index_store(
            embedding_root,
            policy_root,
            (tmp_path / "authorized-indexes").resolve(),
            expected_embedding_receipt_sha256=embedding_pin,
            expected_policy_receipt_sha256=policy_pin,
            config=_config(),
            backend=_FakeBackend(),
        )


def test_source_symlink_is_rejected_without_backend_work(tmp_path: Path) -> None:
    embedding_root, _ = _embedding_store(tmp_path)
    policy_root = _policy_store(tmp_path, embedding_root)
    embedding_pin, policy_pin = _source_pins(embedding_root, policy_root)
    old_path = embedding_root / "old-documents.npy"
    outside = tmp_path / "outside.npy"
    old_path.rename(outside)
    old_path.symlink_to(outside)
    backend = _FakeBackend()

    with pytest.raises(AuthorizedIndexStoreError, match="embedding store admission"):
        build_authorized_index_store(
            embedding_root,
            policy_root,
            (tmp_path / "authorized-indexes").resolve(),
            expected_embedding_receipt_sha256=embedding_pin,
            expected_policy_receipt_sha256=policy_pin,
            config=_config(),
            backend=backend,
        )
    assert backend.add_count == 0


def test_receipt_parser_is_closed_and_canonical(tmp_path: Path) -> None:
    result, *_ = _build(tmp_path)
    receipt_path = result.root / "receipt.json"
    payload = json.loads(receipt_path.read_bytes())
    payload["unexpected"] = True
    receipt_path.write_bytes(_canonical(payload) + b"\n")

    with pytest.raises(AuthorizedIndexStoreError, match="fields differ"):
        load_authorized_index_store_receipt(result.root)


def test_config_enforces_single_thread_and_exact_backend_build() -> None:
    with pytest.raises(AuthorizedIndexStoreError, match="one thread"):
        replace(_config(), num_threads=2)
    with pytest.raises(AuthorizedIndexStoreError, match="SHA-256"):
        replace(_config(), backend_build_sha256="latest")
    with pytest.raises(AuthorizedIndexStoreError, match="at least m"):
        replace(_config(), m=32, ef_construction=16)


def test_document_epochs_are_opened_from_the_exact_index_receipt(
    tmp_path: Path,
) -> None:
    result, embedding_root, _, embedding_pin, _, _, old_vectors = _build(tmp_path)
    receipt = load_authorized_index_store_receipt(result.root)

    with open_verified_document_matrices(
        embedding_root,
        index_receipt=receipt,
        expected_embedding_receipt_sha256=embedding_pin,
    ) as matrices:
        assert np.array_equal(matrices.old_active, old_vectors)
        assert matrices.current_truth.shape == old_vectors.shape
        assert not matrices.old_active.flags.writeable
        assert not matrices.current_truth.flags.writeable


@pytest.mark.parametrize("role", ["old_active_vector", "current_truth_vector"])
def test_same_shape_document_epoch_substitution_is_detected_on_context_exit(
    tmp_path: Path,
    role: str,
) -> None:
    result, embedding_root, _, embedding_pin, _, _, _ = _build(tmp_path)
    receipt = load_authorized_index_store_receipt(result.root)
    binding = getattr(receipt, role)

    with pytest.raises(AuthorizedIndexStoreError, match="substituted"):
        with open_verified_document_matrices(
            embedding_root,
            index_receipt=receipt,
            expected_embedding_receipt_sha256=embedding_pin,
        ) as matrices:
            source = matrices.old_active if role == "old_active_vector" else matrices.current_truth
            replacement = tmp_path / f"replacement-{role}.npy"
            _write_npy(replacement, np.zeros_like(source))
            replacement.replace(embedding_root / binding.relative_path)
