from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pytest

import fractal_ann_diagnostics.embedding_store as store_module
from fractal_ann_diagnostics.artifact_integrity import digest_directory_tree
from fractal_ann_diagnostics.embedding_store import (
    EMBEDDING_BUILDER_VERSION,
    EMBEDDING_CHECKPOINT_SCHEMA,
    EmbeddingStoreConfig,
    EmbeddingStoreError,
    LocalModelSpec,
    SentenceTransformersLocalEncoder,
    StagedEmbeddingSources,
    build_embedding_store,
    load_embedding_store_receipt,
    verify_embedding_store,
)

_DOCUMENT_PATHS = (
    "datasets/demo/corpus/part-00000.jsonl",
    "datasets/demo/corpus/part-00001.jsonl",
)
_QUERY_PATHS = ("datasets/demo/sealed/online/queries.jsonl",)
_OUTCOME_PATH = "datasets/demo/sealed/custody/qrels.jsonl"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(_canonical(row) + b"\n" for row in rows)
    path.write_bytes(encoded)
    return encoded


def _artifact_row(
    path: str,
    encoded: bytes,
    *,
    role: str,
    stage: str | None,
    visibility: str,
) -> dict[str, object]:
    return {
        "byte_count": len(encoded),
        "dataset": "demo",
        "path": path,
        "record_count": encoded.count(b"\n"),
        "role": role,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "stage": stage,
        "visibility": visibility,
    }


@dataclass(frozen=True)
class _StageFixture:
    root: Path
    selection: StagedEmbeddingSources
    outcome_path: Path


def _stage(
    tmp_path: Path,
    *,
    name: str = "stage",
    text_suffix: str = "",
    query_rows: Sequence[dict[str, object]] | None = None,
) -> _StageFixture:
    root = tmp_path / name
    document_groups = (
        [
            {"id": "document-0", "text": f"alpha{text_suffix}", "title": "Alpha"},
            {"id": "document-1", "text": f"beta{text_suffix}", "title": "Beta"},
            {"id": "document-2", "text": f"gamma{text_suffix}", "title": "Gamma"},
        ],
        [
            {"id": "document-3", "text": f"delta{text_suffix}", "title": "Delta"},
            {"id": "document-4", "text": f"epsilon{text_suffix}", "title": "Epsilon"},
        ],
    )
    artifacts: list[dict[str, object]] = []
    for relative_path, rows in zip(_DOCUMENT_PATHS, document_groups, strict=True):
        encoded = _write_jsonl(root / relative_path, rows)
        artifacts.append(
            _artifact_row(
                relative_path,
                encoded,
                role="corpus-shard",
                stage=None,
                visibility="online",
            )
        )
    selected_queries = (
        list(query_rows)
        if query_rows is not None
        else [
            {"id": "query-0", "text": f"find alpha{text_suffix}"},
            {"id": "query-1", "text": f"find epsilon{text_suffix}"},
            {"id": "query-2", "text": f"find gamma{text_suffix}"},
        ]
    )
    query_encoded = _write_jsonl(root / _QUERY_PATHS[0], selected_queries)
    artifacts.append(
        _artifact_row(
            _QUERY_PATHS[0],
            query_encoded,
            role="queries",
            stage="sealed",
            visibility="online",
        )
    )

    outside = tmp_path / f"{name}-outcome.jsonl"
    outcome_encoded = _write_jsonl(
        outside,
        [{"document_id": "document-0", "query_id": "query-0", "relevance": 1}],
    )
    outcome_path = root / _OUTCOME_PATH
    outcome_path.parent.mkdir(parents=True, exist_ok=True)
    outcome_path.symlink_to(outside)
    artifacts.append(
        _artifact_row(
            _OUTCOME_PATH,
            outcome_encoded,
            role="qrels",
            stage="sealed",
            visibility="custody",
        )
    )
    artifacts.sort(key=lambda row: str(row["path"]).encode("utf-8"))
    inventory_bytes = (
        _canonical(
            {
                "artifacts": artifacts,
                "schema_version": "fractal-study-data-inventory-v2",
                "withhold_sealed_labels_from_online_process": True,
            }
        )
        + b"\n"
    )
    (root / "inventory.json").write_bytes(inventory_bytes)
    selection = StagedEmbeddingSources(
        root=root.resolve(),
        inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
        document_paths=_DOCUMENT_PATHS,
        query_paths=_QUERY_PATHS,
    )
    return _StageFixture(root=root, selection=selection, outcome_path=outcome_path)


def _model(tmp_path: Path, name: str, content: bytes) -> LocalModelSpec:
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_bytes(content)
    return LocalModelSpec(
        path=path.resolve(),
        revision=hashlib.sha1(content).hexdigest(),
        tree_sha256=digest_directory_tree(path.resolve()).sha256,
    )


def _config(*, batch_size: int = 2, output_dtype: str = "float32") -> EmbeddingStoreConfig:
    return EmbeddingStoreConfig(
        query_prompt=(
            "Instruct: Given a web search query, retrieve relevant passages that answer "
            "the query\nQuery: "
        ),
        document_prompt="",
        max_sequence_length=512,
        output_dimension=4,
        normalize=True,
        batch_size=batch_size,
        output_dtype=output_dtype,  # type: ignore[arg-type]
        device="cpu",
        deterministic_seed=20260714,
    )


class _DeterministicEncoder:
    implementation_id = "deterministic-test-encoder-v1"

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls = 0
        self.max_batch = 0

    def encode(
        self,
        texts: Sequence[str],
        *,
        model_path: Path,
        prompt: str,
        max_sequence_length: int,
        output_dimension: int,
        normalize: bool,
        device: str,
        seed: int,
    ) -> np.ndarray:
        del max_sequence_length, normalize, device
        self.calls += 1
        self.max_batch = max(self.max_batch, len(texts))
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("injected encoder interruption")
        model_bytes = (model_path / "config.json").read_bytes()
        rows = []
        for text in texts:
            digest = hashlib.sha256(
                model_bytes
                + prompt.encode("utf-8")
                + seed.to_bytes(8, "big")
                + text.encode("utf-8")
            ).digest()
            values = np.frombuffer(digest, dtype=np.uint8)[:output_dimension].astype(np.float32)
            values += 1.0
            values /= np.linalg.norm(values)
            rows.append(values)
        return np.stack(rows)


class _BadEncoder(_DeterministicEncoder):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.implementation_id = f"bad-test-encoder-{mode}"

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        self.calls += 1
        dimension = int(kwargs["output_dimension"])
        if self.mode == "nonfinite":
            return np.full((len(texts), dimension), np.nan, dtype=np.float32)
        return np.ones((len(texts), dimension), dtype=np.float32)


class _PairedDeterministicEncoder:
    current_implementation_id = "paired-test-current-v1"
    old_implementation_id = "paired-test-old-v1"

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls = 0
        self.max_batch = 0
        self.rows: list[tuple[str, ...]] = []

    def encode_pair(
        self,
        texts: Sequence[str],
        *,
        current_model_path: Path,
        old_model_path: Path,
        prompt: str,
        max_sequence_length: int,
        output_dimension: int,
        normalize: bool,
        device: str,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del max_sequence_length, normalize, device
        self.calls += 1
        self.max_batch = max(self.max_batch, len(texts))
        self.rows.append(tuple(texts))
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("injected paired encoder interruption")

        def arm(path: Path) -> np.ndarray:
            model_bytes = (path / "config.json").read_bytes()
            rows = []
            for text in texts:
                digest = hashlib.sha256(
                    model_bytes
                    + prompt.encode("utf-8")
                    + seed.to_bytes(8, "big")
                    + text.encode("utf-8")
                ).digest()
                values = np.frombuffer(digest, dtype=np.uint8)[:output_dimension].astype(np.float32)
                values += 1.0
                values /= np.linalg.norm(values)
                rows.append(values)
            return np.stack(rows)

        return arm(current_model_path), arm(old_model_path)


def _file_digests(root: Path) -> dict[str, str]:
    return {
        child.name: hashlib.sha256(child.read_bytes()).hexdigest()
        for child in root.iterdir()
        if child.is_file()
    }


def test_streams_bounded_batches_and_emits_exact_descriptors(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    model = _model(tmp_path, "current-model", b"current-model")
    encoder = _DeterministicEncoder()
    output = tmp_path / "embedding-store"

    receipt = build_embedding_store(
        stage.selection,
        output,
        current_model=model,
        current_encoder=encoder,
        config=_config(batch_size=2),
    )

    assert encoder.max_batch == 2
    assert receipt.document_count == 5
    assert receipt.query_count == 3
    assert set(receipt.vectors) == {"current_documents", "current_queries"}
    documents = np.load(output / "current-documents.npy", mmap_mode="r")
    queries = np.load(output / "current-queries.npy", mmap_mode="r")
    assert documents.shape == (5, 4)
    assert queries.shape == (3, 4)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(queries, axis=1), 1.0, atol=1e-5)
    assert verify_embedding_store(output) == receipt
    assert load_embedding_store_receipt(output) == receipt
    config_bytes = (output / "config.json").read_bytes()
    source_inventory_bytes = (output / "source-inventory.json").read_bytes()
    assert hashlib.sha256(config_bytes[:-1]).hexdigest() == receipt.config_sha256
    assert (
        hashlib.sha256(source_inventory_bytes[:-1]).hexdigest() == receipt.source_inventory_sha256
    )
    assert not output.with_name(f".{output.name}.partial").exists()
    assert not output.with_name(f".{output.name}.checkpoint.json").exists()

    descriptor = receipt.vectors["current_documents"]
    assert descriptor.dtype == "float32"
    assert descriptor.shape == (5, 4)
    assert descriptor.model_tree_sha256 == model.tree_sha256
    assert descriptor.model_revision == model.revision
    assert descriptor.builder_version == EMBEDDING_BUILDER_VERSION
    assert descriptor.row_order_sha256 == receipt.row_orders["documents"].row_order_sha256
    assert descriptor.byte_count == (output / descriptor.relative_path).stat().st_size
    assert (
        descriptor.file_sha256
        == hashlib.sha256((output / descriptor.relative_path).read_bytes()).hexdigest()
    )


def test_row_order_digest_binds_source_path_line_and_identifier(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    model = _model(tmp_path, "current-model", b"current")
    output = tmp_path / "store"
    receipt = build_embedding_store(
        stage.selection,
        output,
        current_model=model,
        current_encoder=_DeterministicEncoder(),
        config=_config(),
    )
    expected = b""
    identifiers = (
        ("document-0", _DOCUMENT_PATHS[0], 1),
        ("document-1", _DOCUMENT_PATHS[0], 2),
        ("document-2", _DOCUMENT_PATHS[0], 3),
        ("document-3", _DOCUMENT_PATHS[1], 1),
        ("document-4", _DOCUMENT_PATHS[1], 2),
    )
    for identifier, source_path, source_row in identifiers:
        expected += (
            _canonical(
                {
                    "dataset": "demo",
                    "id": identifier,
                    "kind": "documents",
                    "source_path": source_path,
                    "source_row": source_row,
                    "stage": None,
                }
            )
            + b"\n"
        )
    assert (output / "document-rows.jsonl").read_bytes() == expected
    assert receipt.row_orders["documents"].row_order_sha256 == hashlib.sha256(expected).hexdigest()
    assert (
        receipt.vectors["current_documents"].row_order_sha256
        == receipt.row_orders["documents"].row_order_sha256
    )


def test_resume_rejects_config_and_source_substitution_then_completes(
    tmp_path: Path,
) -> None:
    original = _stage(tmp_path, name="original")
    substituted = _stage(tmp_path, name="substituted", text_suffix=" changed")
    model = _model(tmp_path, "model", b"model")
    output = tmp_path / "resumable-store"
    interrupted = _DeterministicEncoder(fail_after=1)

    with pytest.raises(RuntimeError, match="interruption"):
        build_embedding_store(
            original.selection,
            output,
            current_model=model,
            current_encoder=interrupted,
            config=_config(batch_size=2),
        )

    partial = output.with_name(f".{output.name}.partial")
    checkpoint_path = output.with_name(f".{output.name}.checkpoint.json")
    assert partial.is_dir()
    assert checkpoint_path.is_file()
    assert not output.exists()
    assert not (partial / "receipt.json").exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == EMBEDDING_CHECKPOINT_SCHEMA
    assert checkpoint["progress"]["current_documents"] == 2
    with pytest.raises(EmbeddingStoreError, match="cannot load embedding store receipt"):
        load_embedding_store_receipt(partial)

    with pytest.raises(EmbeddingStoreError, match="checkpoint binding changed"):
        build_embedding_store(
            original.selection,
            output,
            current_model=model,
            current_encoder=_DeterministicEncoder(),
            config=_config(batch_size=3),
        )
    with pytest.raises(EmbeddingStoreError, match="checkpoint binding changed"):
        build_embedding_store(
            substituted.selection,
            output,
            current_model=model,
            current_encoder=_DeterministicEncoder(),
            config=_config(batch_size=2),
        )

    resumed = _DeterministicEncoder()
    receipt = build_embedding_store(
        original.selection,
        output,
        current_model=model,
        current_encoder=resumed,
        config=_config(batch_size=2),
    )
    assert receipt.document_count == 5
    assert verify_embedding_store(output) == receipt
    assert not partial.exists()
    assert not checkpoint_path.exists()


def test_old_and_current_models_remain_separate_with_identical_order(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path)
    current = _model(tmp_path, "current", b"current")
    old = _model(tmp_path, "old", b"old")
    output = tmp_path / "store"
    receipt = build_embedding_store(
        stage.selection,
        output,
        current_model=current,
        current_encoder=_DeterministicEncoder(),
        old_model=old,
        old_encoder=_DeterministicEncoder(),
        config=_config(),
    )
    assert set(receipt.vectors) == {
        "current_documents",
        "current_queries",
        "old_documents",
        "old_queries",
    }
    current_documents = np.load(output / "current-documents.npy")
    old_documents = np.load(output / "old-documents.npy")
    current_queries = np.load(output / "current-queries.npy")
    old_queries = np.load(output / "old-queries.npy")
    assert current_documents.shape == old_documents.shape == (5, 4)
    assert current_queries.shape == old_queries.shape == (3, 4)
    assert not np.array_equal(current_documents, old_documents)
    assert not np.array_equal(current_queries, old_queries)
    for kind in ("documents", "queries"):
        assert (
            receipt.vectors[f"current_{kind}"].row_order_sha256
            == receipt.vectors[f"old_{kind}"].row_order_sha256
        )
        assert receipt.vectors[f"current_{kind}"].model_tree_sha256 == current.tree_sha256
        assert receipt.vectors[f"old_{kind}"].model_tree_sha256 == old.tree_sha256


def test_paired_store_writes_both_arms_in_one_ordered_stream(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    current = _model(tmp_path, "paired-current", b"current")
    old = _model(tmp_path, "paired-old", b"old")
    encoder = _PairedDeterministicEncoder()
    output = tmp_path / "paired-store"
    receipt = build_embedding_store(
        stage.selection,
        output,
        current_model=current,
        old_model=old,
        paired_encoder=encoder,
        config=_config(batch_size=2),
    )

    assert encoder.calls == 5
    assert encoder.max_batch == 2
    assert encoder.rows == [
        ("alpha", "beta"),
        ("gamma", "delta"),
        ("epsilon",),
        ("find alpha", "find epsilon"),
        ("find gamma",),
    ]
    assert receipt.current_model["encoder_id"] == encoder.current_implementation_id
    assert receipt.old_model is not None
    assert receipt.old_model["encoder_id"] == encoder.old_implementation_id
    for kind in ("documents", "queries"):
        current_values = np.load(output / f"current-{kind}.npy")
        old_values = np.load(output / f"old-{kind}.npy")
        assert current_values.shape == old_values.shape
        assert not np.array_equal(current_values, old_values)
        assert (
            receipt.vectors[f"current_{kind}"].row_order_sha256
            == receipt.vectors[f"old_{kind}"].row_order_sha256
        )


def test_paired_store_resumes_both_arms_at_one_checkpoint_boundary(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path)
    current = _model(tmp_path, "resume-current", b"current")
    old = _model(tmp_path, "resume-old", b"old")
    output = tmp_path / "paired-resume"
    interrupted = _PairedDeterministicEncoder(fail_after=1)
    with pytest.raises(RuntimeError, match="paired encoder interruption"):
        build_embedding_store(
            stage.selection,
            output,
            current_model=current,
            old_model=old,
            paired_encoder=interrupted,
            config=_config(batch_size=2),
        )
    checkpoint_path = output.with_name(f".{output.name}.checkpoint.json")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["progress"]["current_documents"] == 2
    assert checkpoint["progress"]["old_documents"] == 2

    resumed = _PairedDeterministicEncoder()
    receipt = build_embedding_store(
        stage.selection,
        output,
        current_model=current,
        old_model=old,
        paired_encoder=resumed,
        config=_config(batch_size=2),
    )
    assert resumed.rows[0] == ("gamma", "delta")
    assert verify_embedding_store(output) == receipt
    assert not checkpoint_path.exists()


def test_paired_store_rejects_divergent_progress_and_encoder_substitution(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path)
    current = _model(tmp_path, "reject-current", b"current")
    old = _model(tmp_path, "reject-old", b"old")
    output = tmp_path / "paired-reject"
    with pytest.raises(RuntimeError, match="paired encoder interruption"):
        build_embedding_store(
            stage.selection,
            output,
            current_model=current,
            old_model=old,
            paired_encoder=_PairedDeterministicEncoder(fail_after=1),
            config=_config(batch_size=2),
        )
    checkpoint_path = output.with_name(f".{output.name}.checkpoint.json")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["progress"]["old_documents"] = 0
    checkpoint_path.write_bytes(_canonical(checkpoint) + b"\n")
    unused = _PairedDeterministicEncoder()
    with pytest.raises(EmbeddingStoreError, match="paired checkpoint progress differs"):
        build_embedding_store(
            stage.selection,
            output,
            current_model=current,
            old_model=old,
            paired_encoder=unused,
            config=_config(batch_size=2),
        )
    assert unused.calls == 0

    checkpoint["progress"]["old_documents"] = 2
    checkpoint_path.write_bytes(_canonical(checkpoint) + b"\n")
    replacement = _PairedDeterministicEncoder()
    replacement.old_implementation_id = "substituted-old-encoder"
    with pytest.raises(EmbeddingStoreError, match="checkpoint binding changed"):
        build_embedding_store(
            stage.selection,
            output,
            current_model=current,
            old_model=old,
            paired_encoder=replacement,
            config=_config(batch_size=2),
        )
    assert replacement.calls == 0


def test_forbidden_paths_and_fields_reject_before_encoding(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingStoreError, match="forbidden outcome path"):
        StagedEmbeddingSources(
            root=tmp_path.resolve(),
            inventory_sha256="a" * 64,
            document_paths=_DOCUMENT_PATHS,
            query_paths=(_OUTCOME_PATH,),
        )

    stage = _stage(
        tmp_path,
        name="forbidden-field",
        query_rows=[{"answer": "hidden", "id": "query-0", "text": "query"}],
    )
    model = _model(tmp_path, "model", b"model")
    encoder = _DeterministicEncoder()
    output = tmp_path / "store"
    with pytest.raises(EmbeddingStoreError, match="forbidden fields"):
        build_embedding_store(
            stage.selection,
            output,
            current_model=model,
            current_encoder=encoder,
            config=_config(),
        )
    assert encoder.calls == 0
    assert not output.exists()
    assert not output.with_name(f".{output.name}.partial").exists()


def test_unselected_outcome_file_is_never_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage(tmp_path)
    model = _model(tmp_path, "model", b"model")
    opened: list[str] = []
    original = store_module._open_source_stream

    @contextmanager
    def tracked(root: Path, source: Any) -> Iterator[Any]:
        opened.append(source.relative_path)
        with original(root, source) as handle:
            yield handle

    monkeypatch.setattr(store_module, "_open_source_stream", tracked)
    build_embedding_store(
        stage.selection,
        tmp_path / "store",
        current_model=model,
        current_encoder=_DeterministicEncoder(),
        config=_config(),
    )
    assert stage.outcome_path.is_symlink()
    assert _OUTCOME_PATH not in opened
    assert set(opened) <= set((*_DOCUMENT_PATHS, *_QUERY_PATHS))


@pytest.mark.parametrize(
    "target",
    ("source-symlink", "source-hardlink", "model-symlink", "model-hardlink"),
)
def test_links_are_rejected_before_encoding(tmp_path: Path, target: str) -> None:
    stage = _stage(tmp_path)
    model = _model(tmp_path, "model", b"model")
    if target == "source-symlink":
        selected = stage.root / _DOCUMENT_PATHS[0]
        outside = tmp_path / "outside-documents.jsonl"
        outside.write_bytes(selected.read_bytes())
        selected.unlink()
        selected.symlink_to(outside)
    elif target == "source-hardlink":
        selected = stage.root / _DOCUMENT_PATHS[0]
        (tmp_path / "linked-documents.jsonl").hardlink_to(selected)
    elif target == "model-symlink":
        outside = tmp_path / "outside-model.bin"
        outside.write_bytes(b"outside")
        (model.path / "linked.bin").symlink_to(outside)
    else:
        (tmp_path / "linked-model-config.json").hardlink_to(model.path / "config.json")
    encoder = _DeterministicEncoder()
    with pytest.raises(EmbeddingStoreError, match="link|model tree"):
        build_embedding_store(
            stage.selection,
            tmp_path / "store",
            current_model=model,
            current_encoder=encoder,
            config=_config(),
        )
    assert encoder.calls == 0


def test_mutated_model_tree_is_rejected_before_encoding(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    model = _model(tmp_path, "model", b"model")
    (model.path / "config.json").write_bytes(b"changed")
    encoder = _DeterministicEncoder()
    with pytest.raises(EmbeddingStoreError, match="model tree differs"):
        build_embedding_store(
            stage.selection,
            tmp_path / "store",
            current_model=model,
            current_encoder=encoder,
            config=_config(),
        )
    assert encoder.calls == 0


@pytest.mark.parametrize("mode", ("nonfinite", "nonunit"))
def test_invalid_vector_geometry_never_finalizes(tmp_path: Path, mode: str) -> None:
    stage = _stage(tmp_path, name=f"stage-{mode}")
    model = _model(tmp_path, f"model-{mode}", mode.encode("utf-8"))
    output = tmp_path / f"store-{mode}"
    with pytest.raises(EmbeddingStoreError, match="non-finite|unit normalized"):
        build_embedding_store(
            stage.selection,
            output,
            current_model=model,
            current_encoder=_BadEncoder(mode),
            config=_config(),
        )
    assert not output.exists()
    assert not (output.with_name(f".{output.name}.partial") / "receipt.json").exists()


def test_final_store_is_never_overwritten(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    model = _model(tmp_path, "model", b"model")
    output = tmp_path / "store"
    build_embedding_store(
        stage.selection,
        output,
        current_model=model,
        current_encoder=_DeterministicEncoder(),
        config=_config(),
    )
    before = _file_digests(output)
    unused = _DeterministicEncoder(fail_after=0)
    with pytest.raises(EmbeddingStoreError, match="cannot be overwritten"):
        build_embedding_store(
            stage.selection,
            output,
            current_model=model,
            current_encoder=unused,
            config=_config(),
        )
    assert unused.calls == 0
    assert _file_digests(output) == before


def test_production_adapter_is_lazy_eval_inference_and_local_only() -> None:
    adapter = SentenceTransformersLocalEncoder()
    assert adapter._model is None
    load_source = inspect.getsource(SentenceTransformersLocalEncoder._load)
    encode_source = inspect.getsource(SentenceTransformersLocalEncoder.encode)
    assert "local_files_only=True" in load_source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in load_source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in load_source
    assert "trust_remote_code=False" in load_source
    assert "model.eval()" in load_source
    assert "torch.inference_mode()" in encode_source
    assert "torch.use_deterministic_algorithms(True)" in encode_source
    assert "normalize_embeddings=normalize" in encode_source
    assert "truncate_dim=output_dimension" in encode_source
